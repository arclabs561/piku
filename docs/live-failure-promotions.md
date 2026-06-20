# Live failure promotions

This file tracks repeated live LLM failures that should affect the deterministic
test suite, provider matrix, or product design.

## Classification

| Class | Meaning | Default next action |
|---|---|---|
| `provider_outage` | Credentials, rate limits, upstream downtime, or provider transport failure. | Infrastructure-only note. |
| `provider_behavior` | One model/provider makes invalid or unstable tool calls. | Provider quarantine or prompt follow-up. |
| `piku_parser` | piku rejects a valid provider response or parses it incorrectly. | Parser test. |
| `piku_tool_loop` | piku calls the wrong tool, wrong args, or misses a needed verification step. | Scripted-provider e2e test. |
| `prompt_weakness` | The live prompt allows too much ambiguity. | Test prompt change plus deterministic coverage if possible. |
| `harness_bug` | The dogfood test or workflow is wrong. | Harness test or script fix. |
| `product_decision` | The model exposes behavior that needs a user-facing policy decision. | Design or ADR before code. |

## Promotions

| Date | Suite | Provider | Model | Failure class | Evidence | Next action | Status |
|---|---|---|---|---|---|---|---|
| _none yet_ | | | | | | | |
