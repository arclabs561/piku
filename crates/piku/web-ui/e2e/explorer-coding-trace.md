Evaluate Piku only through Playwright MCP at {{PIKU_WEB_URL}}. Your perspective
is `coding_trace`: determine whether an operator can attribute goal, supplied
context, model intent, actions, files or page mutations, diffs, verification,
and result. Do not inspect repository source or run shell commands.

Use the exact surface `{{SURFACE}}`, request ID `{{REQUEST_ID}}`, and write
screenshots only below `{{RUN_DIR}}`. Create the surface through the UI and use
only four cards: note, chat, change, and page preview. Attach the note explicitly
to chat without submitting it. First use the change card to create a small seeded
page containing a heading, one styled control, and an unrelated layout element.
Then edit the same card's instruction and submit a narrow heading-only change.
Inspect the second run's source diff and rendered result, then use its rerun
control once. Attribute preservation only from the seeded-to-edited diff; the
initial empty-to-document creation is setup, not evidence of broad rewriting.
Record only observed evidence, with IDs
prefixed `coding_trace:`. Do not add a file card or make a separate chat request.
Stay below {{MAX_CALLS}} Playwright calls and {{MAX_SNAPSHOTS}} snapshots.

Before acting, state at least one falsifiable mechanism hypothesis in the
structured report: what UI mechanism should make attribution understandable,
what observation it predicts, and what observation would falsify it. After the
journey, record the observed outcome and mark the hypothesis supported,
disconfirmed, mixed, or not tested. Cite at least one evidence ID for every
disposition except `not_tested`. Name confounders and plausible alternative
explanations with a distinguishing retest. A score, finding severity, or overall
verdict is not a mechanism. If provider failure, missing state, or incomplete
exercise compromises the inference, say so in `causal_assessment.validity` and
do not claim the product caused the outcome.

Set and retain a 1440×1000 CSS-pixel viewport. Before each screenshot, record
window.innerWidth/innerHeight, devicePixelRatio, prefers-color-scheme, and the
canvas scroll offsets. Return the final values in `viewport`. Distinguish saved
canvas geometry from viewport-relative screenshot geometry.

Capture final provenance and geometry predicates in a dedicated,
non-destructive `browser_evaluate` call before cleanup. Do not combine evidence
capture, surface deletion, dialog handling, or browser close in one compound
tool call.

Classify every top-level article by its accessible label, `data-persistence`,
and membership in `.workspace-object`. An `Execution trace` marked `transient`
is runtime provenance, not an authored or persisted workspace card; do not count
it as object proliferation. If console errors exist, record representative exact
messages, source URL/origin, timing, and observed product impact. Separate host
page errors from sandboxed `about:srcdoc` preview errors and intentional rejected
requests. An aggregate console count alone cannot support a finding.

Delete the surface through the UI and close the browser before returning. Return
only JSON matching the supplied schema. Include concrete todo, idea, or retest
followups that would help a later run; do not turn speculation into a finding.
Set every `artifact_metadata` to null; the harness attests artifact bytes after
your report is validated.
