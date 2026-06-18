# ADR 0003: Split embedding memory modules

---
status: accepted
date: 2026-06-18
governs:
  - crates/piku-runtime/src/embed_memory.rs
  - crates/piku-runtime/src/embed_memory/*.rs
why: `embed_memory.rs` mixes multiple responsibilities in one 2700-line file, making review and future behavior changes harder than necessary.
rejected:
  - Keep one file: every embedding memory change keeps crossing unrelated storage, backend, extraction, eviction, and test code.
  - Split everything at once: a large move would be hard to verify as behavior-preserving.
confidence: medium
review_trigger: Revisit if a split requires public API changes or if submodules need broad private cross-calls.
---

## Context

`embed_memory.rs` owns several different concerns: memory storage, vector
scoring, embedding backend config, HTTP embedding requests, session extraction,
memory conflict judging, eviction, attempt-tree helpers, tool view glue, and
tests.

The public API is used through `piku_runtime::embed_memory` and through reexports
from `piku_runtime`. The split should reduce file size without forcing callers
to learn a new module tree.

See also: [embed memory module split design](../embed-memory-module-split-design.md).

## Decision

Keep `embed_memory` as the public facade and move internals into focused
submodules under `crates/piku-runtime/src/embed_memory/`. Split one concern at a
time and re-export existing public names from the facade.

## Consequences

Callers keep the same import paths. Future changes can land near the code they
affect, and reviewers can verify each move separately.

The facade will temporarily contain both old code and submodule exports while
the split progresses. That is acceptable as long as each commit keeps behavior
the same and tests stay green.
