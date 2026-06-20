# ADR 0008: Repo artifact dogfood corpus

---
status: accepted
date: 2026-06-20
governs:
  - scripts/github-corpus.sh
  - scripts/github-corpus-prompt.sh
  - justfile
  - docs/repo-artifact-dogfood-design.md
  - docs/live-dogfood-roadmap.md
why: Real PR and issue artifacts give local dogfood richer task material without making piku runtime depend on GitHub.
rejected:
  - Keep only synthetic dogfood prompts: they miss real PR bodies, changed-file sets, review comments, and stale claims.
  - Commit a static GitHub corpus: it goes stale and creates data-retention friction for private repos.
  - Fetch GitHub artifacts from piku runtime: that adds product auth and network behavior before the harness proves the value.
confidence: medium
review_trigger: Revisit after three corpus-backed findings are promoted into deterministic tests, or before adding runtime GitHub integration.
---

## Context

The live dogfood loop now records provider/model behavior, but its scenarios are
mostly synthetic. The next useful data source is the repo itself: merged PRs,
review comments, changed files, issue reports, labels, and historical fixes.

Those artifacts are valuable for local dogfood and eval prompts, but they should
not become product behavior by accident. Pulling from GitHub at runtime would add
auth, rate limits, permission scope, and prompt-injection concerns.

## Decision

Use GitHub repo artifacts as local dogfood corpus input. `scripts/github-corpus.sh`
exports PR and issue JSON under `target/github-corpus/`; `just github-corpus`
is the local entrypoint. `scripts/github-corpus-prompt.sh` turns one PR detail
row into a local Markdown prompt seed. The output is ignored generated data.

Piku runtime stays GitHub-agnostic. Normal PR CI does not fetch repo artifacts.

## Consequences

Local dogfood can use real project history without automatic spend or product
scope creep. Future scenarios can sample from exported PRs and issues, compare
model behavior against known changed files, and promote repeated failures into
deterministic tests.

The tradeoff is that this corpus is local and time-varying. Reproducible tests
still need distilled fixtures or scripted-provider coverage after a useful
finding is understood.
