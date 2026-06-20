#!/usr/bin/env bash
#
# Export GitHub PR and issue artifacts for local dogfood.
#
# Usage:
#   scripts/github-corpus.sh [owner/repo] [out-dir]
#
# Environment:
#   PIKU_GITHUB_CORPUS_LIMIT=50       max PRs and issues to fetch
#   PIKU_GITHUB_CORPUS_DETAILS=1      fetch per-PR and per-issue details
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

need gh
need jq

repo="${1:-}"
if [[ -z "$repo" ]]; then
  repo="$(gh repo view --json nameWithOwner --jq '.nameWithOwner')"
fi

out_dir="${2:-$REPO_ROOT/target/github-corpus}"
limit="${PIKU_GITHUB_CORPUS_LIMIT:-50}"
details="${PIKU_GITHUB_CORPUS_DETAILS:-1}"

safe_repo="${repo//\//_}"
mkdir -p "$out_dir"

prs_file="$out_dir/${safe_repo}.prs.json"
issues_file="$out_dir/${safe_repo}.issues.json"
pr_details_file="$out_dir/${safe_repo}.pr-details.jsonl"
issue_details_file="$out_dir/${safe_repo}.issue-details.jsonl"
bundle_file="$out_dir/${safe_repo}.json"
summary_file="$out_dir/${safe_repo}.summary.md"

printf 'repo: %s\n' "$repo"
printf 'limit: %s\n' "$limit"

gh pr list \
  --repo "$repo" \
  --state all \
  --limit "$limit" \
  --json number,title,state,mergedAt,headRefName,body,url,author \
  >"$prs_file"

gh issue list \
  --repo "$repo" \
  --state all \
  --limit "$limit" \
  --json number,title,state,body,url,labels,createdAt,updatedAt,author \
  >"$issues_file"

: >"$pr_details_file"
: >"$issue_details_file"

if [[ "$details" != "0" ]]; then
  while IFS= read -r number; do
    gh pr view "$number" \
      --repo "$repo" \
      --json number,title,state,body,comments,reviews,files,commits,additions,deletions,changedFiles,mergedAt,url \
      >>"$pr_details_file"
    printf '\n' >>"$pr_details_file"
  done < <(jq -r '.[].number' "$prs_file")

  while IFS= read -r number; do
    gh issue view "$number" \
      --repo "$repo" \
      --comments \
      --json number,title,state,body,comments,labels,createdAt,updatedAt,url,author \
      >>"$issue_details_file"
    printf '\n' >>"$issue_details_file"
  done < <(jq -r '.[].number' "$issues_file")
fi

generated_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

jq -n \
  --arg repo "$repo" \
  --arg generated_at "$generated_at" \
  --arg pr_details_path "$pr_details_file" \
  --arg issue_details_path "$issue_details_file" \
  --slurpfile prs "$prs_file" \
  --slurpfile issues "$issues_file" \
  '{
    repo: $repo,
    generated_at: $generated_at,
    prs: $prs[0],
    issues: $issues[0],
    detail_files: {
      prs_jsonl: $pr_details_path,
      issues_jsonl: $issue_details_path
    }
  }' >"$bundle_file"

pr_count="$(jq 'length' "$prs_file")"
merged_pr_count="$(jq '[.[] | select(.state == "MERGED")] | length' "$prs_file")"
issue_count="$(jq 'length' "$issues_file")"
closed_issue_count="$(jq '[.[] | select(.state == "CLOSED")] | length' "$issues_file")"

{
  printf '# GitHub corpus: %s\n\n' "$repo"
  printf -- '- generated: `%s`\n' "$generated_at"
  printf -- '- prs: `%s` (`%s` merged)\n' "$pr_count" "$merged_pr_count"
  printf -- '- issues: `%s` (`%s` closed)\n' "$issue_count" "$closed_issue_count"
  printf -- '- bundle: `%s`\n' "$bundle_file"
  printf -- '- pr details: `%s`\n' "$pr_details_file"
  printf -- '- issue details: `%s`\n' "$issue_details_file"
} >"$summary_file"

printf 'bundle: %s\n' "$bundle_file"
printf 'summary: %s\n' "$summary_file"
