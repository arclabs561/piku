# Design: Trace-backed dogfood assertions

status: accepted
decisions: none (ADR pending)
decided: 2026-06-19

## Implementation status

Implemented in `2bea6d3`. `crates/piku/tests/dogfood.rs` reads trace JSONL for
tool-order and tool-success assertions while retaining stdout for the human
report and workspace state for final-effect assertions. The decision predates
the local ADR ledger and has not yet been recorded as an ADR; the roadmap
should treat that as an archival decision-record task, not as missing code.

## Original problem

At decision time, `crates/piku/tests/dogfood.rs` parsed human stdout to infer
which tools ran. That kept the report readable, but made assertions depend on
display text that was not the stable record of the run.

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
