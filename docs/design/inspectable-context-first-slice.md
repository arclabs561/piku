Status: superseded by `durable-investigation-core-and-surfaces.md`

# Design: inspectable context as Piku's first workbench slice

## Problem

Piku persists a chronological `Session` and curates that history before each
provider request, but the user cannot inspect the resulting composition. The
workbench transcript identifies explicit model context as the smallest
falsifiable step toward durable investigation: before introducing threads,
blocks, forks, or a native client, Piku should make its existing context
selection legible.

## Chosen approach

Add a read-only context manifest derived by the same runtime code that builds a
provider request, and expose a compact `/context` view in the existing TUI. The
manifest reports the model window and history budget, system-prompt sections,
available tools, and every session message's role, approximate size, relevance,
and selected/excluded state. The first slice inspects current context; it does
not yet edit or persist context selections. Sharing the curation path is the
load-bearing requirement: a parallel approximation would create the exact
opacity this feature is meant to remove.

## Non-goals

- No new thread or notebook document model; the experiment must first prove
  that context visibility changes how Piku is used.
- No automatic replay, branching, or staleness graph; those require stable run
  identity and side-effect semantics that Piku does not yet have.
- No Swift client or embedded terminal; the existing TUI is sufficient to test
  the core interaction.
- No raw prompt dump by default; the compact view shows composition and bounded
  previews without turning context inspection into another transcript.
- No claim that token estimates equal provider billing counts; Piku's existing
  character heuristic is explicitly approximate.

## Decision gates

- The manifest and provider request must select the same message indices in
  tests; any separate selection implementation stops the change.
- `/context` must be read-only and work before the first turn and after curation
  pressure without changing the session.
- If the view does not help identify at least one surprising included or
  excluded input during dogfooding, do not build editing or forking on top of
  it.
- A second UI is deferred until the runtime exposes a typed event/run contract;
  `OutputSink` callbacks are not that contract.

## Why not start with a completion review?

Completion review is valuable, but it operates after the agent has already
acted. Context inspection tests the transcript's more foundational claim: the
person should be able to know what the model is about to reason from.

## Why not add explicit attachments immediately?

Attachments require durable references, snapshot rules, and prompt-composition
semantics. First exposing current behavior gives those later choices an
observable baseline and keeps this slice reversible.

---
Decided: 2026-08-05 | Session: 019fd257
