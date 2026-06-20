#!/usr/bin/env bash
#
# Run one GitHub-corpus prompt against a temp repo copy and write a live ledger row.
#
# Usage:
#   scripts/github-corpus-run.sh [pr-number]
#
# Environment:
#   PIKU_GITHUB_CORPUS_PROMPT=/path/to/prompt.md   use an existing prompt
#   PIKU_LIVE_PROVIDER=openrouter                  optional provider override
#   PIKU_LIVE_MODEL=openai/gpt-4o-mini             optional model override
#   PIKU_LIVE_KEY_VAR=OPENROUTER_API_KEY           optional key env override
#   PIKU_LIVE_LEDGER=/abs/path/runs.jsonl          optional ledger override
#   PIKU_BIN=/path/to/piku                         optional piku binary override
#   PIKU_GITHUB_CORPUS_VALIDATE=0                  skip validation for harness dry runs
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

need() {
  if ! command -v "$1" >/dev/null 2>&1; then
    printf 'error: required command not found: %s\n' "$1" >&2
    exit 1
  fi
}

need jq
need rsync

has_key() {
  local key_var="$1"
  [[ -n "${!key_var:-}" ]]
}

choose_provider() {
  if [[ -n "${PIKU_LIVE_PROVIDER:-}" && -n "${PIKU_LIVE_MODEL:-}" ]]; then
    provider="$PIKU_LIVE_PROVIDER"
    model="$PIKU_LIVE_MODEL"
    key_var="${PIKU_LIVE_KEY_VAR:-}"
    if [[ -z "$key_var" ]]; then
      case "$provider" in
        anthropic) key_var="ANTHROPIC_API_KEY" ;;
        groq) key_var="GROQ_API_KEY" ;;
        *) key_var="OPENROUTER_API_KEY" ;;
      esac
    fi
    if ! has_key "$key_var"; then
      printf 'error: %s is not set for %s/%s\n' "$key_var" "$provider" "$model" >&2
      exit 1
    fi
    return
  fi

  local rows=(
    "openrouter|openai/gpt-4o-mini|OPENROUTER_API_KEY"
    "openrouter|anthropic/claude-sonnet-4-5|OPENROUTER_API_KEY"
    "openrouter|google/gemini-2.5-flash|OPENROUTER_API_KEY"
    "anthropic|claude-haiku-4-5|ANTHROPIC_API_KEY"
    "groq|moonshotai/kimi-k2-instruct|GROQ_API_KEY"
  )
  local available=()
  local row row_provider row_model row_key_var

  for row in "${rows[@]}"; do
    IFS='|' read -r row_provider row_model row_key_var <<<"$row"
    if has_key "$row_key_var"; then
      available+=("$row")
    fi
  done

  if (( ${#available[@]} == 0 )); then
    printf 'error: set OPENROUTER_API_KEY, ANTHROPIC_API_KEY, or GROQ_API_KEY for corpus dogfood\n' >&2
    exit 1
  fi

  row="${available[$((RANDOM % ${#available[@]}))]}"
  IFS='|' read -r provider model key_var <<<"$row"
}

default_ledger() {
  if [[ -n "${PIKU_LIVE_LEDGER:-}" ]]; then
    ledger="$PIKU_LIVE_LEDGER"
    return
  fi

  local stamp safe
  stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  safe="github_corpus-${provider}-${model}-${stamp}"
  safe="${safe//\//_}"
  safe="${safe//:/_}"
  safe="${safe// /_}"
  ledger="$REPO_ROOT/target/live-ledger/${safe}.jsonl"
}

latest_trace_path() {
  local config_dir="$1"
  find "$config_dir/piku/traces" -maxdepth 1 -type f -name '*.jsonl' 2>/dev/null |
    sort |
    tail -n 1
}

trace_count() {
  local trace="$1"
  local filter="$2"
  jq -s -r "[.[] | $filter] | length" "$trace"
}

trace_last_number() {
  local trace="$1"
  local field="$2"
  jq -s -r "[.[] | select(.event == \"turn_end\")][-1].${field} // 0" "$trace"
}

changed_files_from_prompt() {
  awk '/^- `[^`]+` \(/ { sub(/^- `/, ""); sub(/`.*/, ""); print }' "$prompt_path"
}

changed_files_total() {
  changed_files_from_prompt | wc -l | tr -d ' '
}

changed_files_read_count() {
  local trace="$1"
  local changed_file read_path count=0
  while IFS= read -r changed_file; do
    [[ -n "$changed_file" ]] || continue
    while IFS= read -r read_path; do
      if [[ "$read_path" == "$changed_file" || "$read_path" == "./$changed_file" || "$read_path" == */"$changed_file" ]]; then
        count=$((count + 1))
        break
      fi
    done < <(jq -r 'select(.event == "tool_start" and .tool == "read_file") | .input.path // empty' "$trace")
  done < <(changed_files_from_prompt)
  printf '%s\n' "$count"
}

changed_files_mentioned_count() {
  local output_file="$1"
  local changed_file count=0
  while IFS= read -r changed_file; do
    [[ -n "$changed_file" ]] || continue
    if grep -Fq "$changed_file" "$output_file"; then
      count=$((count + 1))
    fi
  done < <(changed_files_from_prompt)
  printf '%s\n' "$count"
}

validate_run() {
  local trace="$1"
  local output_file="$2"
  local min_reads="${PIKU_GITHUB_CORPUS_MIN_READS:-3}"
  local min_changed_reads="${PIKU_GITHUB_CORPUS_MIN_CHANGED_READS:-2}"
  local min_changed_mentions="${PIKU_GITHUB_CORPUS_MIN_CHANGED_MENTIONS:-2}"
  local read_count changed_file_total changed_read_count changed_mention_count bad_tools failed_tools

  if [[ "${PIKU_GITHUB_CORPUS_VALIDATE:-1}" == "0" ]]; then
    validation_error=""
    validation_changed_files_read=0
    validation_changed_files_mentioned=0
    return 0
  fi

  if [[ -z "$trace" || ! -f "$trace" ]]; then
    validation_error="missing_trace"
    return 1
  fi

  changed_file_total="$(changed_files_total)"
  if (( changed_file_total < min_changed_reads )); then
    min_changed_reads="$changed_file_total"
  fi
  if (( changed_file_total < min_changed_mentions )); then
    min_changed_mentions="$changed_file_total"
  fi

  bad_tools="$(
    jq -s -r '
      [
        .[]
        | select(.event == "tool_start")
        | .tool
        | select(. != "read_file" and . != "grep" and . != "glob" and . != "list_dir")
      ]
      | unique
      | join(",")
    ' "$trace"
  )"
  if [[ -n "$bad_tools" ]]; then
    validation_error="read_only_violation:${bad_tools}"
    return 1
  fi

  failed_tools="$(trace_count "$trace" 'select(.event == "tool_end" and .ok != true)')"
  if (( failed_tools > 0 )); then
    validation_error="tool_failure"
    return 1
  fi

  read_count="$(trace_count "$trace" 'select(.event == "tool_start" and .tool == "read_file")')"
  changed_read_count="$(changed_files_read_count "$trace")"
  validation_changed_files_read="$changed_read_count"
  changed_mention_count="$(changed_files_mentioned_count "$output_file")"
  validation_changed_files_mentioned="$changed_mention_count"

  if (( read_count < min_reads )); then
    validation_error="weak_evidence:read_file_count=${read_count}"
    return 1
  fi

  if (( changed_read_count < min_changed_reads )); then
    validation_error="weak_evidence:changed_files_read=${changed_read_count}"
    return 1
  fi

  if (( changed_mention_count < min_changed_mentions )); then
    validation_error="weak_response:changed_files_mentioned=${changed_mention_count}"
    return 1
  fi

  if ! grep -Eiq 'deterministic|test|check|verification|doc' "$output_file"; then
    validation_error="weak_response:missing_test_or_check"
    return 1
  fi

  validation_error=""
  return 0
}

append_ledger() {
  local result="$1"
  local failure_class="$2"
  local duration_ms="$3"
  local trace_path="${4:-}"
  local input_tokens=0
  local output_tokens=0
  local iterations=0
  local tool_starts=0
  local tool_ends=0
  local failed_tools=0
  local permission_denied=0
  local changed_files_read="${validation_changed_files_read:-0}"
  local changed_files_mentioned="${validation_changed_files_mentioned:-0}"
  local validation="${validation_error:-}"

  if [[ -n "$trace_path" && -f "$trace_path" ]]; then
    input_tokens="$(trace_last_number "$trace_path" input_tokens)"
    output_tokens="$(trace_last_number "$trace_path" output_tokens)"
    iterations="$(trace_last_number "$trace_path" iterations)"
    tool_starts="$(trace_count "$trace_path" 'select(.event == "tool_start")')"
    tool_ends="$(trace_count "$trace_path" 'select(.event == "tool_end")')"
    failed_tools="$(trace_count "$trace_path" 'select(.event == "tool_end" and .ok != true)')"
    permission_denied="$(trace_count "$trace_path" 'select(.event == "permission_denied")')"
  fi

  if [[ "$result" == "failure" && "$failure_class" == "unknown_failure" ]]; then
    if (( permission_denied > 0 )); then
      failure_class="permission_denied"
    elif (( failed_tools > 0 )); then
      failure_class="tool_failure"
    fi
  fi

  mkdir -p "$(dirname "$ledger")"
  jq -cn \
    --arg suite "github_corpus" \
    --arg test "pr-${pr_number}" \
    --arg provider "$provider" \
    --arg model "$model" \
    --arg result "$result" \
    --arg failure_class "$failure_class" \
    --arg trace_path "$trace_path" \
    --arg prompt_path "$prompt_path" \
    --arg output_path "${output_path:-}" \
    --arg run_dir "$run_dir" \
    --arg validation_error "$validation" \
    --argjson input_tokens "$input_tokens" \
    --argjson output_tokens "$output_tokens" \
    --argjson iterations "$iterations" \
    --argjson tool_starts "$tool_starts" \
    --argjson tool_ends "$tool_ends" \
    --argjson failed_tools "$failed_tools" \
    --argjson permission_denied "$permission_denied" \
    --argjson changed_files_read "$changed_files_read" \
    --argjson changed_files_mentioned "$changed_files_mentioned" \
    --argjson duration_ms "$duration_ms" \
    '{
      suite: $suite,
      test: $test,
      provider: $provider,
      model: $model,
      result: $result,
      failure_class: $failure_class,
      trace_path: (if $trace_path == "" then null else $trace_path end),
      prompt_path: $prompt_path,
      output_path: (if $output_path == "" then null else $output_path end),
      run_dir: $run_dir,
      validation_error: (if $validation_error == "" then null else $validation_error end),
      input_tokens: $input_tokens,
      output_tokens: $output_tokens,
      iterations: $iterations,
      tool_starts: $tool_starts,
      tool_ends: $tool_ends,
      failed_tools: $failed_tools,
      permission_denied: $permission_denied,
      changed_files_read: $changed_files_read,
      changed_files_mentioned: $changed_files_mentioned,
      duration_ms: $duration_ms
    }' >>"$ledger"
}

pr_number="${1:-}"

if [[ -n "${PIKU_GITHUB_CORPUS_PROMPT:-}" ]]; then
  prompt_path="$PIKU_GITHUB_CORPUS_PROMPT"
  if [[ ! -f "$prompt_path" ]]; then
    printf 'error: PIKU_GITHUB_CORPUS_PROMPT does not exist: %s\n' "$prompt_path" >&2
    exit 1
  fi
  if [[ -z "$pr_number" ]]; then
    pr_number="$(basename "$prompt_path" .md)"
    pr_number="${pr_number#pr-}"
  fi
else
  if [[ -n "$pr_number" ]]; then
    prompt_line="$("$REPO_ROOT/scripts/github-corpus-prompt.sh" "" "$pr_number")"
  else
    prompt_line="$("$REPO_ROOT/scripts/github-corpus-prompt.sh")"
  fi
  prompt_path="${prompt_line#prompt: }"
  if [[ -z "$pr_number" ]]; then
    pr_number="$(basename "$prompt_path" .md)"
    pr_number="${pr_number#pr-}"
  fi
fi

if [[ -z "$pr_number" ]]; then
  printf 'error: could not infer PR number for ledger row\n' >&2
  exit 1
fi

choose_provider
default_ledger

bin="${PIKU_BIN:-$REPO_ROOT/target/debug/piku}"
if [[ ! -x "$bin" ]]; then
  cargo build -p piku
fi
if [[ ! -x "$bin" ]]; then
  printf 'error: piku binary not found: %s\n' "$bin" >&2
  exit 1
fi

run_dir="$(mktemp -d)"
config_dir="$run_dir/config"
repo_copy="$run_dir/repo"
output_path="$run_dir/output.txt"
validation_error=""
validation_changed_files_read=0
validation_changed_files_mentioned=0
mkdir -p "$config_dir"
rsync -a --delete --exclude target --exclude .git "$REPO_ROOT/" "$repo_copy/"

printf 'provider: %s\n' "$provider"
printf 'model: %s\n' "$model"
printf 'prompt: %s\n' "$prompt_path"
printf 'ledger: %s\n' "$ledger"
printf 'run_dir: %s\n' "$run_dir"

prompt="$(cat "$prompt_path")"
start_seconds="$(date +%s)"
set +e
(
  cd "$repo_copy"
  XDG_CONFIG_HOME="$config_dir" "$bin" --print --provider "$provider" --model "$model" "$prompt"
) | tee "$output_path"
status=$?
set -e
end_seconds="$(date +%s)"
duration_ms=$(( (end_seconds - start_seconds) * 1000 ))

trace_path="$(latest_trace_path "$config_dir" || true)"
if (( status == 0 )) && validate_run "$trace_path" "$output_path"; then
  append_ledger "success" "none" "$duration_ms" "$trace_path"
else
  if (( status == 0 )); then
    status=1
  fi
  if [[ "$validation_error" == read_only_violation:* ]]; then
    append_ledger "failure" "read_only_violation" "$duration_ms" "$trace_path"
  elif [[ "$validation_error" == weak_evidence:* ]]; then
    append_ledger "failure" "weak_evidence" "$duration_ms" "$trace_path"
  elif [[ "$validation_error" == weak_response:* ]]; then
    append_ledger "failure" "weak_response" "$duration_ms" "$trace_path"
  elif [[ "$validation_error" == "tool_failure" ]]; then
    append_ledger "failure" "tool_failure" "$duration_ms" "$trace_path"
  elif [[ "$validation_error" == "missing_trace" ]]; then
    append_ledger "failure" "missing_trace" "$duration_ms" "$trace_path"
  else
    append_ledger "failure" "unknown_failure" "$duration_ms" "$trace_path"
  fi
fi

printf 'trace: %s\n' "${trace_path:-}"
printf 'output: %s\n' "$output_path"
if [[ -n "$validation_error" ]]; then
  printf 'validation: %s\n' "$validation_error"
fi
printf 'exit: %s\n' "$status"
exit "$status"
