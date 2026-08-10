You are an autonomous developer evaluating Piku as a replacement for a
chat-only coding agent. This is not primarily a component tour or visual smoke
test. Your central question is: after a realistic bounded task, can an operator
understand what the agent knew, intended, changed, observed, and verified?

Target: `{{PIKU_WEB_URL}}`

Use Playwright MCP browser tools for every interaction. Do not inspect source,
run shell commands, or use curl. Do not start or stop the server. Your first
action must be a Playwright call. The repository and host are out of bounds:
do not start the terminal, request repository edits, or approve host actions.
You MAY submit at most two model requests through the visible UI: one chat turn
and one selected-page-source change. Those requests are required unless the UI
or backend visibly blocks them.

This distinction is essential:

- `status` reports whether this evaluation journey completed credibly.
- `product_thesis.verdict` reports whether the observed product actually
  supports the Codex-replacement thesis. A valid run may pass while the product
  verdict is `partial` or `not_supported`.

Aim to complete the journey within 105 browser calls and never exceed 120.
Keep accessibility snapshots to seven or fewer. Recover from a failed locator
with a targeted DOM query; do not restart a completed phase.
Use bounded browser-code actions for clustered interactions and targeted
locators afterward. Take exactly the five named screenshots below. Inspect the
returned image pixels for hierarchy, density, clipping, layering, progress,
and comprehension; accessibility text alone is not visual evidence.

Do not emit a plan, partial report, or assistant prose before all browser work
and cleanup. For every screenshot, pass the full absolute filename
`{{RUN_DIR}}/<filename>` to the screenshot tool; relative filenames are invalid.
Write artifacts only under `{{RUN_DIR}}`. Report screenshot paths
relative to the repository as
`.artifacts/playwright-agent/runs/{{RUN_ID}}/playwright-output/<filename>`.

Create the exact temporary surface `{{SURFACE}}` through the visible UI. The
harness owns and independently deletes this surface after the judge exits.
Use this representative task throughout the journey:

> Build an evidence board that explains the difference between conversation,
> workspace arrangement, page-source editing, file observation, and terminal
> authority. Preserve a note containing the operator goal and make the result
> understandable after reload.

Complete exactly these phases in order:

1. ORIENT — At 1440x1000, arrive without prior product explanation. Capture
   `01-empty-desktop.png`. State what work the surface appears to support, what
   is merely implied, and whether the next action is obvious.

2. DISCOVER — Open the blank-canvas menu and inspect every creation choice.
   Verify a second blank click closes it, plus keyboard focus and Escape.
   Capture `02-create-menu.png`. Form a hypothesis about which object owns each
   kind of authority and name what evidence would falsify it.

3. CONSTRUCT — Create note, file, chat, change, terminal, and page-preview
   objects. Put the representative task in the note. Open `README.md` in the
   file card. Inspect both change targets. Do not start the terminal. Assess
   whether relationships among cards are explicit or require manual inference.

4. MANIPULATE — Put a concise, operator-written summary of the note/file facts
   into the chat card's optional context. Submit one chat turn asking the model
   to explain the authority boundaries. Observe queued/running/completed/error
   transitions, rendered Markdown, elapsed/progress information, and any
   inspectable model/tool provenance. Edit that turn and use `run from here` or
   the closest visible equivalent. Determine whether downstream output is
   visibly invalidated and whether the rerun is attributable to the edited
   input. Drag, resize, overlap, and raise cards; reload and verify content,
   geometry, and stacking. Capture `03-workspace-desktop.png`.

5. TRUST — Through the change card, target selected page source and submit one
   request for a compact evidence board reflecting the five authority types.
   Observe in-progress output, completion, error recovery, and whether the
   exact saved artifact and its source relationship are legible. Confirm that
   the ordinary chat did not mutate the page or workspace. Inspect one invalid
   file path and the terminal warning without starting a shell. Explicitly
   record whether the UI exposes proposed actions, executed actions, files,
   diffs, tool calls, verification, model identity, and context boundaries.
   Do not award credit for capabilities that are merely named in copy.

6. STRESS — Maintain at least five objects. Rapidly switch selection, edit text
   while another result is visible, dismiss a menu, and reload. If a visible
   request can be safely cancelled without another model call, test it;
   otherwise record cancellation as untested. Inspect console errors and failed
   network requests. For every failure, distinguish denied-by-design,
   unavailable capability, infrastructure failure, and product defect.

7. REFLOW — Resize to 390x844. Use the object picker to reach an offscreen
   object, focus an input, and inspect the evidence board. Capture
   `04-narrow.png`. Return directly to 1440x1000 without reloading and capture
   `05-final-desktop.png`. Verify narrow scroll state does not displace the
   desktop canvas.

8. REFLECT — Compare the empty, working, reloaded, and narrow states. Evaluate
   the eight thesis dimensions below using only observed evidence. A dimension
   is `absent` when the workflow requires guessing, transcript memory, or
   manual reconstruction. Visual polish cannot compensate for absent
   provenance or state semantics.

Required thesis dimensions, in this exact order:

1. `task_comprehension` — the operator can restate the goal and current state.
2. `action_provenance` — intent, model actions, mutations, and verification are
   attributable and inspectable.
3. `state_visibility` — proposed, running, completed, failed, cancelled, and
   stale states are distinguishable.
4. `context_control` — included context and cross-card references are explicit.
5. `rerun_semantics` — editing history visibly invalidates and recomputes the
   correct downstream state.
6. `recovery` — errors and interruption preserve understandable resumable state.
7. `authority_clarity` — conversation, page, workspace, file, and terminal
   powers are distinct before activation.
8. `spatial_utility` — the canvas improves task understanding over a linear
   transcript rather than merely arranging windows.

For each dimension return a 1–5 score, `demonstrated`, `partial`, `absent`, or
`blocked`, and concrete evidence. Use `supported` only if all eight dimensions
are demonstrated and none scores below 4. Use `not_supported` if either task
comprehension or action provenance is absent, or four dimensions are absent.
Otherwise use `partial`.

Findings must describe reproduced behavior and operator impact, not inferred
source causes. Include capability absences when they block the product thesis;
do not reduce the report to CSS defects. Prefer one root finding over several
symptoms. A high finding means the representative task cannot be understood,
controlled, recovered, or safely attributed.

Before recording any finding, reconcile its evidence across modalities:

- A failed locator proves only that the locator failed. It does not prove that
  the visible control or capability is absent. Check a current screenshot and
  a targeted DOM predicate before reporting absence.
- Attribute console errors to their actual origin and observed impact. Treat
  evaluator-generated errors as compromised evidence, not product defects.
- If screenshot pixels and a predicate disagree, report the contradiction as
  compromised evaluation evidence and do not promote it to a product finding.
- Inspect every saved screenshot yourself during REFLECT. Record visually
  apparent clipping, overlap, illegible density, or hierarchy failures even
  when accessibility text remains correct.

Return up to eight structured `followups` to preserve momentum across runs.
Use `todo` for a concrete harness or evaluation action, `idea` for an unproven
hypothesis, and `retest` for a prior capability that needs fresh evidence. Give
each a priority, rationale, suggested perspective, and the evidence IDs that
motivated it. Do not disguise an unsupported finding as a todo.
Number findings locally as `f1`, `f2`, and so on, and followups as `o1`, `o2`,
and so on. Every followup must cite at least one concrete evidence ID or local
finding ID. Set `retest_of` only to an exact prior fully scoped obligation ID
when that lineage is known; otherwise use null. Similar prose or overlapping
evidence does not establish identity.

Delete the temporary surface through the visible UI and close the headless
browser. Cleanup failure belongs in REFLECT evidence. Return only the JSON
required by the schema. `coverage` needs one concrete statement per phase;
`journey` must contain exactly the eight uppercase phases in order. `artifacts`
must contain the five screenshots in journey order.
