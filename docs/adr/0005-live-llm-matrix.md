# ADR 0005: Live LLM matrix

---
status: accepted
date: 2026-06-20
governs:
  - .github/workflows/live-llm.yml
  - scripts/ci.sh
  - crates/piku/tests/llm_e2e.rs
  - crates/piku/tests/dogfood.rs
why: Live provider behavior should be tested across model families without making normal PR CI depend on secrets or spend.
rejected:
  - Run live LLM suites on every PR: forks and local runs cannot rely on provider secrets.
  - Add random model selection to normal piku runs now: that makes user-facing cost, latency, and repro harder before dogfood proves the value.
  - Keep a single live model: provider and model differences are exactly what live dogfood should expose.
confidence: medium
review_trigger: Revisit when live matrix failures show a stable provider-specific product gap, or before adding scheduled live-test spend.
---

## Context

ADR 0004 keeps PR-blocking coverage deterministic through scripted providers.
That protects normal CI, but it does not check whether OpenRouter-compatible,
Anthropic, and Groq-backed live models still make valid tool calls against the
compiled `piku` binary.

See also: [live LLM matrix design](../live-llm-matrix-design.md).

## Decision

Live LLM coverage runs in a separate manual-only workflow. The default suite is
`llm_e2e`; heavier dogfood can be selected manually. The workflow uses a small
provider/model matrix and skips rows whose provider secret is not present.

The workflow is gated to the repository owner actor and uses the `live-llm`
GitHub environment. That environment requires approval by `arclabs561` and has
admin bypass disabled. Live provider keys should be stored as environment
secrets for that environment, not as secrets available to normal PR CI.

The live harnesses also accept explicit provider/model/key environment variables
so CI matrix rows and local random dogfood runs use the same path.

## Consequences

Normal PR CI stays deterministic and cheap. Live dogfood now tests more than one
model family, which should expose provider-specific tool-call behavior.

The live matrix is not a release gate yet. A failure means "investigate provider
or model behavior," not "the deterministic runtime is broken."
