# ADR 0001: Runtime-owned provider resolution

---
status: accepted
date: 2026-06-18
governs:
  - crates/piku/src/cli.rs
  - crates/piku/src/main.rs
  - crates/piku/src/tui_repl.rs
  - crates/piku-runtime/src/*.rs
  - crates/piku-api/src/*.rs
why: The binary currently constructs concrete provider clients, while the runtime owns the agent loop that consumes them.
rejected:
  - Keep resolution in `piku`: every provider addition keeps coupling the binary to concrete client types.
  - Move app selection policy into `piku-api`: protocol clients would need application fallback order and default-model policy.
confidence: medium
review_trigger: Revisit when a second binary needs different provider selection semantics, or when provider config becomes data-driven.
---

## Context

`piku-api` defines provider clients and streaming/event types. `piku-runtime`
owns the agent loop and re-exports the provider trait. The `piku` binary
currently parses CLI arguments and also constructs concrete providers from
environment variables.

That shape works, but it puts application provider policy in the same module as
CLI parsing. It also makes `piku` depend directly on provider implementation
types even though it only needs a trait object and default model.

See also: [provider-resolution design](../provider-resolution-design.md).

## Decision

Provider resolution belongs in `piku-runtime`. `piku-api` keeps concrete
protocol clients and provider-specific `from_env` helpers. The `piku` binary
keeps CLI parsing and help text, then asks `piku-runtime` for a resolved provider
and default model.

## Consequences

Adding a provider still requires a protocol client in `piku-api`, but the
application selection order lives in one runtime module. The binary no longer
needs to import concrete provider types for normal execution.

The runtime now owns one more application-policy surface. That is acceptable
because the provider handle is consumed by the runtime loop, and the decision can
be revisited if another binary needs different selection semantics.
