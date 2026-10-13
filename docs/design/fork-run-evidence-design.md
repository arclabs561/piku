# Design: durable parent→child run evidence for fork inspection

## Problem

The fork evaluation proved continuity (parent marker reaches the child request,
child result returns, spawn/join recorded, child execution durable). But the
static browser workbench could not follow a fork end-to-end: the typed
parent→child relationship lived only beside the run as a link file, and the
parent timeline carried it only as human-formatted `agent_join` prose. A static
view either parsed prose (brittle) or scanned ambient local directories
(unsafe, scope-leaking).

## Chosen approach

Record a typed `RunEvent::ChildRunRef` in the durable parent run record at the
moment of spawn. Legacy references without `reference_root` remain relative to
the parent run record's directory. For configured sibling run/session stores
with the parent record directly in the run store, the optional typed
`reference_root: "configured_evidence_root"` selects their configured common
owner; references then name `runs/<child>.jsonl` and `sessions/<child>.json`.
The legacy basis is preferred whenever it already contains both references.
Other layouts fail closed, and both modes retain Normal-only relative paths.
The basis is context metadata, not an immutable host identity: relocation or
config rebinding requires current config-aware review. No automatic dereference
or cross-run reader is introduced; the current UI displays metadata only. Any
future resolver must use the current configured basis and validate the bundle.
No absolute paths in the browser; no ambient directory scans.

Scope: the event is turn-scoped (emitted during the spawn turn), not run-scoped,
because the durable record's append path classifies it by audit scope and the
spawn turn is the natural owner.

## Non-goals

- No path scanning of a links directory inferred from local config in the browser.
- No parsing of `agent_join` text to reconstruct relationships.
- No cross-run bundle rendering until a validated readonly reader exists.

## Decision gates

- If a second surface (mutable browser, ACP, Swift) needs live fork comparison,
  build the validated cross-run reader first and gate the surface on it.
- If `agent_join` prose is removed, `ChildRunRef` already carries the contract.

## Why not the link-file-scan option?

It would give a static view ambient read access to unrelated local files and
still require re-implementing relationship resolution per surface. The typed
event is replayable, path-safe, and surfaces once.

---
Decided: 2026-08-05 | Session: piku fork-evidence seam
