# Design: repo artifact dogfood corpus

## Problem

Synthetic dogfood prompts are useful, but they are too clean. Real project
history has better task material: PR bodies, commit messages, changed files,
review comments, issue reports, labels, and stale follow-ups.

## Chosen approach

Treat GitHub repo artifacts as a local dogfood corpus. A script exports PR and
issue data into `target/github-corpus/`; dogfood and agentic prompts can consume
that data later without adding GitHub auth or network access to piku runtime.

The first slice is `scripts/github-corpus.sh`,
`scripts/github-corpus-prompt.sh`, and `scripts/github-corpus-run.sh`. The
exporter fetches PR and issue lists, optional PR and issue detail JSONL, a
combined bundle, and a short summary. The prompt script turns one PR detail row
into a read-only Markdown seed for local dogfood. The runner executes that seed
against a temp repo copy and writes the same kind of JSONL ledger row as other
live dogfood.

The runner also validates the trace before calling a run successful. It rejects
mutating tools, failed tools, missing traces, weak file-reading evidence, and
responses that do not mention changed files or a test/doc check. The output is
ignored build data, not committed fixture data.

CI runs a deterministic selftest for the runner with fake traces. That covers
success, missing trace, mutating tool, failed tool, weak evidence, and weak
response cases without calling a live provider.

## Non-goals

- Do not make piku a GitHub client yet. The corpus is harness input.
- Do not run GitHub network calls in normal PR CI.
- Do not file issues or comments automatically.
- Do not treat issue labels, PR titles, or model prose as ground truth.
- Do not add a hosted control plane.
- Do not treat corpus prompt success as a product guarantee.

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

Run `just github-corpus`, then `just github-dogfood`. Inspect the new row under
`target/live-ledger/`.

---
Decided: 2026-06-20
