# Design: Trace-backed dogfood assertions

status: accepted
decided: 2026-06-19

## Problem

`crates/piku/tests/dogfood.rs` parses human stdout to infer which tools ran. That
keeps the report readable, but it makes assertions depend on display text that is
not the stable record of the run.

## Chosen approach

Keep stdout parsing for the printed dogfood report. Add a trace JSONL reader for
assertions, using the trace file that single-shot runs already write under
`XDG_CONFIG_HOME/piku/traces`. Tool-order and tool-success checks should prefer
`tool_start` and `tool_end` trace events; final file state should still be read
from the workspace.

## Non-goals

- Do not replace the human dogfood report in this pass.
- Do not change trace event schema unless an existing field is missing.
- Do not make live dogfood PR-blocking.
- Do not parse model prose when a trace event or file state can prove the same
  property.

## Decision gates

- If trace events miss a field needed by multiple assertions, add that field to
  the trace schema with a focused test.
- If a dogfood assertion is about rendered output, keep stdout parsing for that
  assertion.
- If trace parsing grows beyond a few helpers, move it out of the dogfood test
  file.

## Why not keep stdout as the assertion source?

Stdout is the UI. Changing a glyph, color, or compact display format should not
silently change what the dogfood harness thinks the agent did.

## Why not assert only on files?

File state proves the final result, but it cannot prove the agent searched before
editing or retried after a failed tool call. Trace events can check that loop
shape without relying on a live model's wording.
