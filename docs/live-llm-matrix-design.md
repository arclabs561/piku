# Design: Live LLM matrix

status: accepted
decisions: ADR-0005
decided: 2026-06-20

## Problem

The deterministic PR gate checks the agent loop without network keys. That is
the right blocking signal, but it does not show whether real providers and
different model families still follow piku's tool-use contract.

## Chosen approach

Add a separate live LLM workflow that runs only when manually dispatched. It uses
a small provider/model matrix for `llm_e2e` by default, with `dogfood`
selectable from `workflow_dispatch`. The live test harnesses also accept
`PIKU_LIVE_PROVIDER`, `PIKU_LIVE_MODEL`, and `PIKU_LIVE_KEY_VAR`, so local
dogfood can pin or randomize model choice without changing code.

The workflow is restricted to the repository owner account and uses the
`live-llm` GitHub environment for provider secrets. That environment requires
approval by `arclabs561` and has admin bypass disabled. Provider keys should be
environment secrets, not general repository secrets.

## Non-goals

- Do not run live LLM suites on normal PR events.
- Do not make random model selection part of normal `piku` behavior yet.
- Do not make missing optional provider secrets fail the whole workflow.
- Do not add scheduled live runs until the spend policy is explicit.
- Do not store live provider keys where pull request workflows can access them.

## Decision gates

- If one provider starts failing because of an API change, fix or quarantine that
  provider row without weakening deterministic PR coverage.
- If model diversity exposes stable product differences, design a user-facing
  model-selection mode from those findings.
- If scheduled live runs are added later, define the budget and cadence before
  enabling the schedule.
- If another maintainer needs to run live dogfood, add them through the
  `live-llm` environment review policy rather than broadening PR CI.

## Why not live LLMs in PR CI?

Provider secrets are unavailable on forks and not guaranteed locally. PR CI must
remain reproducible without network spend.

## Why not random model selection in the product now?

Randomness is useful for dogfood because it exposes overfitting. In normal user
runs it makes cost, latency, and repro harder to explain.
