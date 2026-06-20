# Design: live dogfood ledger

status: accepted
decisions: ADR-0006
decided: 2026-06-20

## Problem

The live matrix tells us whether a model row passed, but it does not leave a
small record that can be compared across runs. The trace files already contain
tool events and token counts, but live tests write traces under per-test temp
config directories, so a standalone post-run script cannot reliably discover
them.

## Chosen approach

The live test helpers append one JSONL row per completed scenario when
`PIKU_LIVE_LEDGER` is set. Each row records the suite, test name, provider,
model, result, failure class, trace path, token counts, tool counts, and
duration.

The manual live workflow sets `PIKU_LIVE_LEDGER` to a file under
`target/live-ledger/` and uploads that directory as an artifact. Local dogfood
can do the same by setting `PIKU_LIVE_LEDGER` before running `scripts/ci.sh
live` or `scripts/ci.sh live-random`.

## Non-goals

- Do not parse model prose for ledger facts.
- Do not add pricing math yet. Token counts are enough for the first version.
- Do not make ledger writing required for normal tests.
- Do not add a database or long-term storage format.

## Decision gates

- If the ledger fields are not enough to explain repeated failures, add fields
  from trace events before adding a new storage layer.
- If model quality comparison becomes a release input, add a checked-in summary
  format with review rules.
- If provider cost becomes a decision point, add a separate pricing ADR.
