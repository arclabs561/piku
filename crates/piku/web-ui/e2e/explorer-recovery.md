Evaluate Piku only through Playwright MCP at {{PIKU_WEB_URL}}. Your perspective
is `recovery`: determine whether stale output, errors, cancellation, reload,
resume, selection, geometry, and stacking remain understandable. Do not inspect
repository source or run shell commands.

Use the exact surface `{{SURFACE}}`, request ID `{{REQUEST_ID}}`, and write
screenshots only below `{{RUN_DIR}}`. Create the surface through the UI and use
only the three cards needed for this lens: file, chat, and note. Exercise one
invalid file input, one two-turn chat, edit-and-rerun from the second turn, then
start one additional chat turn and cancel it while it is visibly active. Reload
and verify the note, prompts, completed outputs, cancelled state, and geometry
that persist. Never start a terminal or approve host actions. Capture one
screenshot before reload and one after rerun; inspect
console or network failures once, near the end. Record evidence with IDs prefixed
`recovery:`. Stay below
{{MAX_CALLS}} Playwright calls and {{MAX_SNAPSHOTS}} snapshots.

Set and retain a 1440×1000 CSS-pixel viewport. Before each screenshot, record
window.innerWidth/innerHeight, devicePixelRatio, prefers-color-scheme, and the
canvas scroll offsets. Return the final values in `viewport`. Treat saved canvas
coordinates separately from viewport-relative card bounds across reload or focus.

After reload, capture all persistence and geometry predicates in one dedicated,
non-destructive `browser_evaluate` call and record that evidence before cleanup.
Do not combine verification, surface deletion, dialog handling, or browser close
inside one compound tool call. A cleanup result cannot substitute for the
post-reload evidence result.

Delete the surface through the UI and close the browser before returning. Return
only JSON matching the supplied schema. Include concrete todo, idea, or retest
followups that would help a later run; do not turn speculation into a finding.
