# Design: Split embedding memory modules

status: accepted
decisions: ADR-0003
decided: 2026-06-18

## Problem

`crates/piku-runtime/src/embed_memory.rs` is over 2700 lines. It mixes storage
types, embedding HTTP clients, scoring, eviction, attempt trees, extraction, and
tests in one file.

## Chosen approach

Keep `piku_runtime::embed_memory` as the public facade. Move internal concerns
into submodules under `crates/piku-runtime/src/embed_memory/`, one concern per
commit. Re-export the same public types and functions from the facade so callers
do not change.

Start with backend/config because it is already isolated: `EmbedBackend`,
`EmbedConfig`, `embed_text`, and `embed_text_with_config`. Later slices can move
extraction, eviction, attempt trees, and tests when each boundary is clear.

## Non-goals

- Do not change memory file format.
- Do not change public `piku_runtime::embed_memory::*` names.
- Do not tune retrieval, eviction, or extraction behavior while moving code.
- Do not split unrelated large files in the same commit.

## Decision gates

- If a move needs public API changes, stop and record that as a separate ADR.
- If a submodule needs many cross-module private calls, stop and choose a smaller
  boundary.
- If tests fail without behavior changes, treat the split as suspect and reduce
  scope.

## Why not keep the single file?

The file is already past the repo's split signal. Small changes now require
scrolling across unrelated systems, which raises review cost and makes future
behavior changes harder to isolate.

## Why not split the whole file at once?

A full split would touch too many unrelated concerns in one diff. Smaller moves
make it easier to prove that the work is only a relocation.
