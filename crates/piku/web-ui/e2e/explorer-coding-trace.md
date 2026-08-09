Evaluate Piku only through Playwright MCP at {{PIKU_WEB_URL}}. Your perspective
is `coding_trace`: determine whether an operator can attribute goal, supplied
context, model intent, actions, files or page mutations, diffs, verification,
and result. Do not inspect repository source or run shell commands.

Use the exact surface `{{SURFACE}}`, request ID `{{REQUEST_ID}}`, and write
screenshots only below `{{RUN_DIR}}`. Create the surface through the UI and use
only four cards: note, chat, change, and page preview. Attach the note explicitly
to chat without submitting it. Perform one page-source change, inspect its source
diff, then use its rerun control once. Record only observed evidence, with IDs
prefixed `coding_trace:`. Do not add a file card or make a separate chat request.
Stay below {{MAX_CALLS}} Playwright calls and {{MAX_SNAPSHOTS}} snapshots.

Set and retain a 1440×1000 CSS-pixel viewport. Before each screenshot, record
window.innerWidth/innerHeight, devicePixelRatio, prefers-color-scheme, and the
canvas scroll offsets. Return the final values in `viewport`. Distinguish saved
canvas geometry from viewport-relative screenshot geometry.

Capture final provenance and geometry predicates in a dedicated,
non-destructive `browser_evaluate` call before cleanup. Do not combine evidence
capture, surface deletion, dialog handling, or browser close in one compound
tool call.

Delete the surface through the UI and close the browser before returning. Return
only JSON matching the supplied schema. Include concrete todo, idea, or retest
followups that would help a later run; do not turn speculation into a finding.
