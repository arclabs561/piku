# ADR 0007: Live failure promotion

---
status: accepted
date: 2026-06-20
governs:
  - docs/live-failure-promotions.md
  - docs/live-dogfood-roadmap.md
  - crates/piku-runtime/tests/e2e.rs
  - crates/piku/tests/dogfood.rs
why: Live LLM failures should either become deterministic coverage or explicit provider notes, not untracked anecdotes.
rejected:
  - Treat every live failure as release-blocking: provider behavior is not deterministic enough for normal PR gates.
  - Keep findings only in workflow logs: logs expire and are hard to compare.
  - Auto-promote failures from model text: the reliable evidence is trace data, test output, and final files.
confidence: medium
review_trigger: Revisit after the first three promoted failures, or if live failures start blocking releases.
---

## Context

ADR 0005 adds a manual live matrix. ADR 0006 records comparable ledger rows for
manual runs. The next risk is that live failures remain outside the normal PR
gate.

## Decision

Repeated live failures are tracked in `docs/live-failure-promotions.md`. Each
row gets one classification and one next action: deterministic test, parser or
trace test, provider quarantine, product design decision, or infrastructure-only
note.

Normal PR CI stays deterministic. Live results feed the deterministic suite only
after a failure is understood well enough to reproduce without provider secrets.

## Consequences

The live matrix can improve piku without becoming a flaky release gate. The cost
is a small amount of manual triage after live runs.
