#!/usr/bin/env bash
#
# Build a dogfood prompt seed from an exported GitHub corpus.
#
# Usage:
#   scripts/github-corpus-prompt.sh [bundle-json] [pr-number] [out-file]
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

corpus="${1:-}"
if [[ -z "$corpus" ]]; then
  corpus="$(
    find "$REPO_ROOT/target/github-corpus" \
      -maxdepth 1 \
      -type f \
      -name '*.json' \
      ! -name '*.prs.json' \
      ! -name '*.issues.json' |
      sort |
      tail -n 1
  )"
fi

if [[ -z "$corpus" || ! -f "$corpus" ]]; then
  printf 'error: corpus bundle not found; run `just github-corpus` first\n' >&2
  exit 1
fi

pr_number="${2:-}"
if [[ -z "$pr_number" ]]; then
  pr_number="$(jq -r '.prs[0].number // empty' "$corpus")"
fi

if [[ -z "$pr_number" ]]; then
  printf 'error: corpus has no PR rows\n' >&2
  exit 1
fi

details_file="$(jq -r '.detail_files.prs_jsonl // empty' "$corpus")"
if [[ -z "$details_file" || ! -s "$details_file" ]]; then
  printf 'error: PR detail file missing; rerun `just github-corpus` with PIKU_GITHUB_CORPUS_DETAILS=1\n' >&2
  exit 1
fi

row="$(jq -c --argjson number "$pr_number" 'select(.number == $number)' "$details_file")"
if [[ -z "$row" ]]; then
  printf 'error: PR #%s not found in %s\n' "$pr_number" "$details_file" >&2
  exit 1
fi

out_file="${3:-$REPO_ROOT/target/github-corpus/prompts/pr-${pr_number}.md}"
mkdir -p "$(dirname "$out_file")"

title="$(jq -r '.title' <<<"$row")"
url="$(jq -r '.url' <<<"$row")"
state="$(jq -r '.state' <<<"$row")"
merged_at="$(jq -r '.mergedAt // "not merged"' <<<"$row")"
changed_files="$(jq -r '.changedFiles' <<<"$row")"
additions="$(jq -r '.additions' <<<"$row")"
deletions="$(jq -r '.deletions' <<<"$row")"
body="$(jq -r '.body // ""' <<<"$row")"

{
  printf '# Dogfood prompt seed: PR #%s\n\n' "$pr_number"
  printf 'Use piku on this repository. Answer from the PR artifact and local files.\n\n'
  printf 'Task:\n'
  printf -- '- Do not edit files. This is a read-only analysis prompt.\n'
  printf -- '- Summarize what changed in plain terms.\n'
  printf -- '- Name the files most likely to matter.\n'
  printf -- '- Say what deterministic test or doc check this history suggests.\n'
  printf -- '- Call out any stale claim, missing verification, or follow-up risk.\n\n'
  printf '## PR\n\n'
  printf -- '- title: `%s`\n' "$title"
  printf -- '- url: `%s`\n' "$url"
  printf -- '- state: `%s`\n' "$state"
  printf -- '- merged_at: `%s`\n' "$merged_at"
  printf -- '- changed_files: `%s`\n' "$changed_files"
  printf -- '- additions: `%s`\n' "$additions"
  printf -- '- deletions: `%s`\n\n' "$deletions"
  printf '## Body\n\n'
  printf '%s\n\n' "$body"
  printf '## Changed files\n\n'
  jq -r '.files[] | "- `" + .path + "` (" + (.changeType // "changed") + ", +" + (.additions | tostring) + "/-" + (.deletions | tostring) + ")"' <<<"$row"
  printf '\n## Commits\n\n'
  jq -r '.commits[] | "- `" + (.oid[0:7]) + "` " + .messageHeadline' <<<"$row"
} >"$out_file"

printf 'prompt: %s\n' "$out_file"
