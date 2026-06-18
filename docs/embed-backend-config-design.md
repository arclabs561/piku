# Design: Explicit embedding backend config

status: accepted
decisions: ADR-0002
decided: 2026-06-18

## Problem

Embedding config currently guesses the backend protocol from URL text. It also
keeps its own copy of the default Ollama host, separate from the Ollama provider
module.

## Chosen approach

Use an explicit embedding backend environment variable for custom embedding
URLs. `PIKU_EMBED_BACKEND=ollama` selects the Ollama native embedding API.
`PIKU_EMBED_BACKEND=openai-compat` selects the OpenAI-compatible embedding API.
If no backend is set for a custom URL, use OpenAI-compatible because most remote
embedding endpoints expose that shape.

Keep Ollama host defaults in `piku-api::ollama` and reuse them from runtime
embedding config. The runtime still owns fallback order; the API crate owns the
provider-specific defaults.

## Non-goals

- Do not add a config file.
- Do not change the default no-env fallback from Ollama.
- Do not change embedding request or response parsing in this pass.
- Do not split `embed_memory.rs` in this pass.

## Decision gates

- If users need more than one embedding backend configured at once, revisit this
  before adding provider tables.
- If another runtime consumer needs different embedding fallback order, revisit
  whether this config belongs behind a separate resolver API.
- If Ollama defaults stop matching provider defaults, keep the shared constant
  in `piku-api` and update consumers.

## Why not keep URL sniffing?

URL sniffing is convenient, but it makes backend choice depend on hostnames and
ports instead of the protocol the endpoint actually speaks. That is brittle for
local proxies, private gateways, and self-hosted OpenAI-compatible servers.

## Why not move all embedding config into `piku-api`?

`piku-api` owns protocol clients and provider defaults. The runtime owns the
embedding memory path and its fallback order, so the resolver stays there for
now.
