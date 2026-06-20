#!/usr/bin/env bash
#
# Deterministic tests for github-corpus-prompt.sh prompt shape.
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

details="$tmp/prs.jsonl"
cat >"$details" <<'JSONL'
{"number":101,"title":"tighten corpus prompt","url":"https://example.invalid/pull/101","state":"MERGED","mergedAt":"2026-06-20T00:00:00Z","changedFiles":2,"additions":12,"deletions":3,"body":"This PR makes prompt evidence explicit.","files":[{"path":"scripts/github-corpus-prompt.sh","changeType":"modified","additions":8,"deletions":1},{"path":"docs/repo-artifact-dogfood-design.md","changeType":"modified","additions":4,"deletions":2}],"commits":[{"oid":"abcdef1234567890","messageHeadline":"prompt evidence requirements"}]}
{"number":202,"title":"secondary row","url":"https://example.invalid/pull/202","state":"OPEN","changedFiles":1,"additions":1,"deletions":0,"body":"Should not be selected when asking for 101.","files":[{"path":"README.md","changeType":"modified","additions":1,"deletions":0}],"commits":[{"oid":"1234567890abcdef","messageHeadline":"other change"}]}
JSONL

bundle="$tmp/bundle.json"
jq -cn --arg details "$details" '{
  prs: [{number: 101}, {number: 202}],
  detail_files: {prs_jsonl: $details}
}' >"$bundle"

out="$tmp/prompt.md"
line="$("$REPO_ROOT/scripts/github-corpus-prompt.sh" "$bundle" 101 "$out")"

contains() {
  local label="$1"
  local needle="$2"
  if ! grep -Fq -- "$needle" "$out"; then
    die "$label missing: $needle"
  fi
}

not_contains() {
  local label="$1"
  local needle="$2"
  if grep -Fq -- "$needle" "$out"; then
    die "$label should not contain: $needle"
  fi
}

[[ "$line" == "prompt: $out" ]] || die "unexpected prompt line: $line"
[[ -s "$out" ]] || die "prompt file was not written"

contains "title" '# Dogfood prompt seed: PR #101'
contains "read-only task" '- Do not edit files. This is a read-only analysis prompt.'
contains "deterministic check task" '- Say what deterministic test or doc check this history suggests.'
contains "evidence section" 'Evidence requirements:'
contains "local file evidence" '- Read local files before answering. Do not rely on the PR body alone.'
contains "path citation requirement" '- Cite exact repo paths for each concrete claim.'
contains "facts versus recommendations" '- Separate observed facts from recommendations.'
contains "pr title" '- title: `tighten corpus prompt`'
contains "changed file one" '- `scripts/github-corpus-prompt.sh` (modified, +8/-1)'
contains "changed file two" '- `docs/repo-artifact-dogfood-design.md` (modified, +4/-2)'
contains "commit truncation" '- `abcdef1` prompt evidence requirements'
contains "body" 'This PR makes prompt evidence explicit.'
not_contains "wrong row" 'secondary row'
not_contains "wrong changed file" '- `README.md`'

missing_out="$tmp/missing.out"
set +e
"$REPO_ROOT/scripts/github-corpus-prompt.sh" "$bundle" 999 "$tmp/missing.md" >"$missing_out" 2>&1
missing_status=$?
set -e
[[ "$missing_status" == 1 ]] || die "missing PR status: expected 1, got $missing_status"
grep -Fq -- 'error: PR #999 not found' "$missing_out" || die "missing PR error was not specific"

printf 'github corpus prompt selftest passed\n'
