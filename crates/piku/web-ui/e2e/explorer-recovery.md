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

Give every screenshot call a unique filename and never overwrite or recapture a
filename, including after reload or while correcting evidence. Screenshot
provenance is rejected when more than one producer writes the same path.

Before acting, state at least one falsifiable mechanism hypothesis in the
structured report: what persistence or recovery mechanism should preserve
understandable state, what observation it predicts, and what observation would
falsify it. After the journey, record the observed outcome and mark the
hypothesis supported, disconfirmed, mixed, or not tested. Name confounders and
plausible alternative explanations with a distinguishing retest. Cite at least
one evidence ID for every disposition except `not_tested`. A score,
finding severity, or overall verdict is not a mechanism. If provider failure,
timing, incomplete cancellation, or missing evidence compromises the inference,
say so in `causal_assessment.validity` and do not claim the product caused the
outcome.

Set and retain a 1440×1000 CSS-pixel viewport. Before each screenshot, record
window.innerWidth/innerHeight, devicePixelRatio, prefers-color-scheme, and the
canvas scroll offsets. Return the final values in `viewport`. Treat saved canvas
coordinates separately from viewport-relative card bounds across reload or focus.

After reload, capture all persistence and geometry predicates in one dedicated,
non-destructive `browser_evaluate` call and record that evidence before cleanup.
Do not combine verification, surface deletion, dialog handling, or browser close
inside one compound tool call. A cleanup result cannot substitute for the
post-reload evidence result.

For every negative UI claim such as “no error,” “no stale state,” or “no
cancelled marker,” enumerate the exact visible text and status values captured
from the relevant card or control. Inspect the complete visible status/error
region; do not use absence from a fixed keyword regex as evidence of absence.
Reconcile that inventory against the screenshot pixels. If the screenshot and
predicate disagree, report the contradiction, mark validity compromised, and do
not promote the negative claim to a finding or hypothesis outcome.

For console or network evidence, capture representative exact messages, source
URL/origin, timing, and observed product impact. Separate host-page failures from
sandboxed `about:srcdoc` preview errors and intentional rejected requests. A raw
error count or HTTP status without this classification cannot support a product
finding.

Delete the surface through the UI and close the browser before returning. Return
only JSON matching the supplied schema. Include concrete todo, idea, or retest
followups that would help a later run; do not turn speculation into a finding.
Set every `artifact_metadata` to null; the harness attests artifact bytes after
your report is validated.
