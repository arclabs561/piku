# ADR 0002: Explicit embedding backend config

---
status: accepted
date: 2026-06-18
governs:
  - crates/piku-runtime/src/embed_memory.rs
  - crates/piku-api/src/ollama.rs
why: Embedding config currently guesses protocol from URL text and repeats the Ollama default host outside the Ollama provider module.
rejected:
  - Keep URL sniffing: backend selection would keep depending on hostnames and ports instead of protocol intent.
  - Move embedding fallback order into `piku-api`: protocol clients would need runtime memory policy.
confidence: medium
review_trigger: Revisit when embedding config supports multiple named backends or a second runtime consumer needs different fallback order.
---

## Context

`piku-runtime` owns embedding memory. `piku-api` owns provider client defaults,
including the Ollama host used for provider construction.

The embedding config path currently treats `PIKU_EMBED_URL` as both endpoint
and backend selector. It checks substrings such as `openrouter`, `openai`,
`groq`, `11434`, and `ollama` to pick a protocol. That can choose the wrong
protocol for proxies or private endpoints.

See also: [embed backend config design](../embed-backend-config-design.md).

## Decision

Embedding backend selection for custom URLs is explicit. `PIKU_EMBED_BACKEND`
selects the protocol, with `ollama` and `openai-compat` as the supported values.
Custom URLs without an explicit backend default to OpenAI-compatible. Ollama
default host values stay in `piku-api::ollama` and are reused by runtime config.

## Consequences

Custom embedding endpoints no longer depend on URL naming conventions. Adding
more backend protocols now requires an explicit config value instead of another
substring branch.

This is a small behavior change for users who set `PIKU_EMBED_URL` to an Ollama
URL without also setting `PIKU_EMBED_BACKEND=ollama`. The default no-env path
still uses Ollama.
