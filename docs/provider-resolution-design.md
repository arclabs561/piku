# Design: Runtime-owned provider resolution

status: accepted
decisions: ADR-0001
decided: 2026-06-18

## Problem

`piku` currently parses CLI arguments and also constructs concrete provider clients.
That makes the binary know about both the runtime-level `Provider` trait and each
provider implementation in `piku-api`.

## Chosen approach

Move provider resolution into `piku-runtime`. The binary will keep CLI parsing and
user-facing help text, then ask the runtime for a resolved provider and default
model. This keeps provider selection next to the agent loop that consumes the
provider, while leaving wire-format clients in `piku-api`.

## Non-goals

- Do not change provider behavior or priority order in this pass.
- Do not move HTTP client implementations out of `piku-api`.
- Do not introduce config-file provider tables yet.
- Do not remove live-provider tests; they stay as opt-in smoke tests.

## Decision gates

- If a second binary needs different provider selection semantics, revisit whether
  provider resolution should be a lower-level API crate helper instead.
- If provider configuration becomes data-driven, revisit the runtime-owned resolver
  shape before adding more hardcoded branches.

## Why not keep it in `piku`?

That preserves today’s behavior, but every provider addition keeps editing the
binary crate. The binary should assemble the application; the runtime should own
the model/provider handle passed into the agent loop.

## Why not move factories into `piku-api`?

`piku-api` owns protocol clients. It should not need to know the application’s
opportunistic selection order, default model policy, or user-facing fallback
message.
