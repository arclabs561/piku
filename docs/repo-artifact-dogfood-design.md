# Design: repo artifact dogfood corpus

## Problem

Synthetic dogfood prompts are useful, but they are too clean. Real project
history has better task material: PR bodies, commit messages, changed files,
review comments, issue reports, labels, and stale follow-ups.

## Chosen approach

Treat GitHub repo artifacts as a local dogfood corpus. A script exports PR and
issue data into `target/github-corpus/`; dogfood and agentic prompts can consume
that data later without adding GitHub auth or network access to piku runtime.

The first slice is `scripts/github-corpus.sh` plus
`scripts/github-corpus-prompt.sh`. The exporter fetches PR and issue lists,
optional PR and issue detail JSONL, a combined bundle, and a short summary. The
prompt script turns one PR detail row into a read-only Markdown seed for local
dogfood. The output is ignored build data, not committed fixture data.

## Non-goals

- Do not make piku a GitHub client yet. The corpus is harness input.
- Do not run GitHub network calls in normal PR CI.
- Do not file issues or comments automatically.
- Do not treat issue labels, PR titles, or model prose as ground truth.
- Do not add a hosted control plane.

## Options considered

### Keep only synthetic dogfood

This stays cheap, but it misses the messy parts: incomplete PR bodies, stale
claims, review drift, and real changed-file sets.

### Commit a static corpus

This would make tests reproducible, but it creates data-retention questions and
goes stale quickly. The current repo is public, but the same harness should also
work on private repos without committing their history.

### Fetch GitHub data inside piku

This is premature. Product behavior would need auth, permissions, rate-limit
handling, and a threat model. The local harness can learn from the data first.

## Decision gates

- If three corpus-backed findings become deterministic tests, add a focused
  dogfood scenario that reads one exported bundle.
- If users ask piku to work directly from issues or PRs, write a separate
  product ADR before adding runtime GitHub integration.
- If the corpus exporter starts needing pagination, caching, or redaction rules,
  split those into a second design.

## Next step

Run `just github-corpus`, then `just github-prompt`. Use the prompt under
`target/github-corpus/prompts/` as a local dogfood seed.

---
Decided: 2026-06-20
