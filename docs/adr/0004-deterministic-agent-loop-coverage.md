# ADR 0004: Deterministic agent-loop coverage

---
status: accepted
date: 2026-06-19
governs:
  - crates/piku-runtime/tests/e2e.rs
  - crates/piku/tests/llm_e2e.rs
  - crates/piku/tests/dogfood.rs
  - scripts/ci.sh
  - .github/workflows/ci.yml
why: Live LLM tests are opt-in and secret-gated, so PR-blocking coverage must exercise the agent loop without provider keys.
rejected:
  - Keep agent-loop coverage live-only: normal PR CI would keep reporting the product path as ignored.
  - Add a fake provider mode to the binary: that adds CLI-only plumbing before proving runtime behavior.
  - Run live suites on every PR: forks and local runs cannot rely on provider secrets.
confidence: medium
review_trigger: Revisit when CI has a secret-gated nightly provider lane, or when a live scenario cannot be represented through the runtime scripted-provider harness.
---

## Context

`llm_e2e` and `dogfood` now mark live-provider tests as ignored by default. That
is honest, but it means the normal CI gate does not run those product-path
assertions. The runtime crate already has `crates/piku-runtime/tests/e2e.rs`,
with scripted providers, real tools, real permission logic, and real session
state.

See also: [deterministic agent-loop coverage design](../deterministic-agent-loop-coverage-design.md).

## Decision

PR-blocking agent-loop coverage should use the runtime scripted-provider harness.
Live suites remain opt-in smoke tests for provider behavior and binary wiring.
When a live test checks deterministic runtime behavior, add or move that
property into `crates/piku-runtime/tests/e2e.rs`.

## Consequences

The default gate can cover more of the agent loop without network keys. The live
suites stay useful, but they no longer carry the only assertion for file effects,
tool retries, or session behavior.

Some scenarios will exist in both places for a while. That is acceptable while
the deterministic version proves the runtime property and the live version keeps
checking provider integration.
