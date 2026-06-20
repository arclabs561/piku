# ADR 0006: Live dogfood ledger

---
status: accepted
date: 2026-06-20
governs:
  - crates/piku/tests/llm_e2e.rs
  - crates/piku/tests/dogfood.rs
  - .github/workflows/live-llm.yml
why: Live dogfood needs comparable run records, and the tests are the only place that reliably knows each temp trace path.
rejected:
  - Post-run trace discovery script: live tests write traces under per-test temp config dirs, so discovery is incomplete.
  - Parse model prose: trace events already contain the tool and token facts we need.
  - Store live results in a database: a JSONL artifact is enough until repeated runs prove otherwise.
confidence: medium
review_trigger: Revisit after three manual live matrix runs, or when ledger rows are used for release decisions.
---

## Context

ADR 0005 adds a manual live LLM matrix. That matrix can show pass/fail, but it
does not by itself preserve enough evidence to compare providers over time.

The existing trace writer records tool starts, tool ends, permission denials,
and token counts. The live test harnesses create temp config dirs per scenario,
so they know the trace path at the moment the run finishes.

See also: [live dogfood ledger design](../dogfood-ledger-design.md).

## Decision

When `PIKU_LIVE_LEDGER` is set, live test helpers append one JSON object per
completed scenario to that path. The row records provider, model, suite, test
name, result, failure class, trace path, token counts, tool counts, and duration.

The manual live workflow uploads the ledger file as an artifact. Normal PR CI
does not set `PIKU_LIVE_LEDGER`.

## Consequences

Live runs now leave a small comparable artifact without changing normal tests.
The ledger only includes scenarios that reached the helper's post-run code; a
process crash can still leave no row. That is acceptable for the first version
because the workflow log remains the source for infrastructure failures.
