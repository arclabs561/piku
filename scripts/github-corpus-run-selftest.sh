#!/usr/bin/env bash
#
# Deterministic tests for github-corpus-run.sh validation.
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

die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

tmp="$(mktemp -d)"
cleanup() {
  local status=$?
  if (( status != 0 )) || [[ -n "${PIKU_CORPUS_SELFTEST_KEEP_TMP:-}" ]]; then
    printf 'selftest tmp: %s\n' "$tmp" >&2
  else
    rm -rf "$tmp"
  fi
}
trap cleanup EXIT

prompt="$tmp/pr-999.md"
cat >"$prompt" <<'EOF'
# Dogfood prompt seed: PR #999

Task:
- Do not edit files. This is a read-only analysis prompt.
- Say what deterministic test or doc check this history suggests.

## Changed files

- `scripts/github-corpus-run.sh` (modified, +1/-0)
- `scripts/github-corpus-prompt.sh` (modified, +1/-0)
EOF

fake_piku="$tmp/fake-piku.sh"
cat >"$fake_piku" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

trace_dir="$XDG_CONFIG_HOME/piku/traces"
trace="$trace_dir/fake.jsonl"
saw_read_only=0
saw_print=0

for arg in "$@"; do
  case "$arg" in
    --read-only) saw_read_only=1 ;;
    --print|-p) saw_print=1 ;;
  esac
done

if (( ! saw_read_only )); then
  printf 'fake piku expected --read-only\n' >&2
  exit 2
fi

if (( saw_print )); then
  printf 'fake piku should not receive --print in corpus mode\n' >&2
  exit 2
fi

write_success_trace() {
  mkdir -p "$trace_dir"
  cat >"$trace" <<'JSONL'
{"event":"tool_start","tool":"read_file","input":{"path":"scripts/github-corpus-run.sh"}}
{"event":"tool_end","tool":"read_file","ok":true}
{"event":"tool_start","tool":"read_file","input":{"path":"scripts/github-corpus-prompt.sh"}}
{"event":"tool_end","tool":"read_file","ok":true}
{"event":"tool_start","tool":"read_file","input":{"path":"docs/repo-artifact-dogfood-design.md"}}
{"event":"tool_end","tool":"read_file","ok":true}
{"event":"turn_end","input_tokens":100,"output_tokens":20,"iterations":1}
JSONL
}

case "${PIKU_FAKE_TRACE_CASE:-success}" in
  success)
    write_success_trace
    printf 'Observed scripts/github-corpus-run.sh and scripts/github-corpus-prompt.sh. Deterministic test: run the corpus runner selftest.\n'
    ;;
  missing_trace)
    printf 'No trace written.\n'
    ;;
  read_only_violation)
    mkdir -p "$trace_dir"
    cat >"$trace" <<'JSONL'
{"event":"tool_start","tool":"write_file","input":{"path":"scripts/github-corpus-run.sh"}}
{"event":"tool_end","tool":"write_file","ok":true}
{"event":"turn_end","input_tokens":100,"output_tokens":20,"iterations":1}
JSONL
    printf 'Tried to edit scripts/github-corpus-run.sh. Deterministic test: should fail.\n'
    ;;
  tool_failure)
    mkdir -p "$trace_dir"
    cat >"$trace" <<'JSONL'
{"event":"tool_start","tool":"read_file","input":{"path":"scripts/github-corpus-run.sh"}}
{"event":"tool_end","tool":"read_file","ok":false}
{"event":"tool_start","tool":"read_file","input":{"path":"scripts/github-corpus-prompt.sh"}}
{"event":"tool_end","tool":"read_file","ok":true}
{"event":"tool_start","tool":"read_file","input":{"path":"docs/repo-artifact-dogfood-design.md"}}
{"event":"tool_end","tool":"read_file","ok":true}
{"event":"turn_end","input_tokens":100,"output_tokens":20,"iterations":1}
JSONL
    printf 'Observed scripts/github-corpus-run.sh and scripts/github-corpus-prompt.sh. Deterministic test: should fail.\n'
    ;;
  weak_evidence)
    mkdir -p "$trace_dir"
    cat >"$trace" <<'JSONL'
{"event":"tool_start","tool":"read_file","input":{"path":"scripts/github-corpus-run.sh"}}
{"event":"tool_end","tool":"read_file","ok":true}
{"event":"turn_end","input_tokens":100,"output_tokens":20,"iterations":1}
JSONL
    printf 'Observed scripts/github-corpus-run.sh. Deterministic test: should fail.\n'
    ;;
  weak_response)
    write_success_trace
    printf 'Looks fine. Deterministic test: run something.\n'
    ;;
  *)
    printf 'unknown PIKU_FAKE_TRACE_CASE\n' >&2
    exit 2
    ;;
esac
EOF
chmod +x "$fake_piku"

assert_eq() {
  local label="$1"
  local expected="$2"
  local actual="$3"
  if [[ "$actual" != "$expected" ]]; then
    die "$label: expected '$expected', got '$actual'"
  fi
}

assert_prefix() {
  local label="$1"
  local expected="$2"
  local actual="$3"
  if [[ -z "$expected" ]]; then
    assert_eq "$label" "" "$actual"
    return
  fi
  case "$actual" in
    "$expected"*) ;;
    *) die "$label: expected prefix '$expected', got '$actual'" ;;
  esac
}

run_case() {
  local case_name="$1"
  local expected_status="$2"
  local expected_result="$3"
  local expected_failure="$4"
  local expected_validation="$5"
  local expected_changed_read="$6"
  local expected_changed_mentioned="$7"
  local ledger="$tmp/${case_name}.jsonl"
  local output="$tmp/${case_name}.out"
  local status result failure validation changed_read changed_mentioned output_path

  set +e
  OPENROUTER_API_KEY=dummy \
    PIKU_LIVE_PROVIDER=openrouter \
    PIKU_LIVE_MODEL=fake \
    PIKU_LIVE_KEY_VAR=OPENROUTER_API_KEY \
    PIKU_LIVE_LEDGER="$ledger" \
    PIKU_BIN="$fake_piku" \
    PIKU_FAKE_TRACE_CASE="$case_name" \
    PIKU_GITHUB_CORPUS_PROMPT="$prompt" \
    "$REPO_ROOT/scripts/github-corpus-run.sh" 999 >"$output" 2>&1
  status=$?
  set -e

  assert_eq "$case_name status" "$expected_status" "$status"
  [[ -s "$ledger" ]] || die "$case_name did not write a ledger row"

  result="$(jq -r '.result' "$ledger")"
  failure="$(jq -r '.failure_class' "$ledger")"
  validation="$(jq -r '.validation_error // ""' "$ledger")"
  changed_read="$(jq -r '.changed_files_read' "$ledger")"
  changed_mentioned="$(jq -r '.changed_files_mentioned' "$ledger")"
  output_path="$(jq -r '.output_path // ""' "$ledger")"

  assert_eq "$case_name result" "$expected_result" "$result"
  assert_eq "$case_name failure_class" "$expected_failure" "$failure"
  assert_prefix "$case_name validation_error" "$expected_validation" "$validation"
  if [[ "$expected_changed_read" != "-" ]]; then
    assert_eq "$case_name changed_files_read" "$expected_changed_read" "$changed_read"
  fi
  if [[ "$expected_changed_mentioned" != "-" ]]; then
    assert_eq "$case_name changed_files_mentioned" "$expected_changed_mentioned" "$changed_mentioned"
  fi
  [[ -n "$output_path" && -f "$output_path" ]] || die "$case_name did not record output_path"
}

run_case success 0 success none "" 2 2
run_case missing_trace 1 failure missing_trace missing_trace 0 0
run_case read_only_violation 1 failure read_only_violation read_only_violation: - -
run_case tool_failure 1 failure tool_failure tool_failure - -
run_case weak_evidence 1 failure weak_evidence weak_evidence: 1 -
run_case weak_response 1 failure weak_response weak_response: 2 0

printf 'github corpus runner selftest passed\n'
