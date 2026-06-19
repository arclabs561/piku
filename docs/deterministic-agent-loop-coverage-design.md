# Design: Deterministic agent-loop coverage

status: accepted
decisions: ADR-0004
decided: 2026-06-19

## Problem

The live LLM suites are now honest: default CI reports them as ignored, and an
opt-in run without provider keys fails loudly. That fixes false confidence, but
it leaves the agent loop without enough always-on PR coverage for scenarios that
currently live only in `llm_e2e` and `dogfood`.

## Chosen approach

Move deterministic versions of the live-suite assertions into
`crates/piku-runtime/tests/e2e.rs`, using the existing scripted `Provider`
harness. Keep the live suites as opt-in smoke tests for provider behavior and
binary wiring. The blocking gate should exercise the runtime loop, real tools,
permissions, sessions, and filesystem effects without needing a network key.

## Non-goals

- Do not delete the live suites in this pass.
- Do not add a fake provider mode to the `piku` binary just for tests.
- Do not assert on model prose when filesystem or tool effects are the property.
- Do not move every dogfood scenario at once.
- Do not add provider secrets to normal PR CI.

## Decision gates

- If a live scenario needs binary-only behavior, keep that part in the live suite
  and cover the runtime behavior underneath it separately.
- If a scripted test starts duplicating provider wire-format parsing, move that
  coverage to `piku-api` parser tests instead.
- If CI gets a secret-gated nightly lane, keep it separate from PR-blocking
  deterministic coverage.

## First implementation slice

Start with an audit, not new tests. Map each live assertion to an existing
runtime e2e test and only add coverage for real gaps. The first pass found that
the simple `llm_e2e` file-effect cases are mostly already represented in
`crates/piku-runtime/tests/e2e.rs`.

Likely next candidates are dogfood scenarios that check loop shape rather than
provider quality:

- read, edit, then verify by reading again
- multi-file rename using search before edit
- trace/session artifact checks that do not need a live model
- any live-only scenario whose assertion is about tool order or filesystem state

Each moved scenario should name the property it checks and should call
`run_turn` with scripted events. Leave the live test in place unless the live
version becomes duplicate noise after the deterministic version lands.

## Why not keep the coverage live-only?

Provider keys are not available on normal PRs, so live-only assertions do not
protect the default gate.

## Why not fake the whole CLI?

The behavior we need to cover is the runtime loop. A binary-level fake provider
would add CLI test plumbing before proving the core loop behavior.
