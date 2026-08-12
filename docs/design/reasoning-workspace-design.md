# Design: prediction-oriented reasoning workspace

Status: proposed; design brainstorm, not yet an implementation plan

## Problem

Piku records an append-only, durable run record (`run_record.rs`) and projects
it to a terminal and a static HTML workbench (`piku inspect <id> --html`). Both
projections are chronological event feeds. They show *what the agent did*, but
they do not let a returning human quickly reconstruct *why* work happened, what
the agent believed, or what it needs next.

Research across Claude Code, Codex, Cursor, and smaller harnesses points to a
consistent cluster of failures that are really one failure: the human is the
attention bottleneck, and the interface treats the human like an unbounded
context window.

- Context rot and compaction degrade long sessions well before the window is
  full; summaries drop detail.
- Agents claim "fixed" or completion without showing proof that it held.
- Walls of text exceed working memory; people skim, anchor on salient claims,
  and stop independently checking.
- When agents work in parallel, each independently compacts into a diverging
  summary of "what happened."
- Trust-reliance studies show confidence signals do not by themselves improve
  humans' judgment; people become *more confident after wrong reliance
  decisions* in some contexts. Showing more or showing calibrated signal does
  not cure mis-attention.
- The agent that does the work also decides what deserves attention, which is a
  trust boundary.

The recurring conclusion is that showing more information is not enough for a
person to *grok* — to build an accurate enough model of the agent and task to
predict, correct, and take responsibility.

## Goal and product objective

Give a returning human an accurate mental model of an agent run in seconds.
"Grokking" is defined operationally, matching the roadmap's
`measure comprehension alongside volume` gate:

> After ~10 seconds of orienting on the workspace, can a returning user answer
> (a) what the current crux is, (b) what the agent believed last time they
> looked that it has since reconsidered, and (c) what it needs from them now —
> with tolerable error, and while also detecting a planted defect or a claim
> whose evidence contradicts it?

This is a Pareto improvement, not a readability score: reduce extraneous
attention while preserving or improving task acceptance, defect detection,
delayed comprehension, modification, and debugging. A compact surface that hides
failures is a regression even if it uses fewer lines.

## Existing assets (audited)

The hard part is more built than a from-scratch design would imply.

- `crates/piku-runtime/src/run_record.rs` — `RunEventEnvelope` is "the semantic
  source shared by terminal, browser, editor, and native projections." `RunEvent`
  already models `TurnStarted`, `ContextBuilt`, `CompactionApplied`,
  `AssistantMessage`, `ToolStarted`, `PermissionDecision`, `ToolCompleted`,
  `TurnCompleted`, `Warning`, `UserDisposition`, `ChildRunRef`.
- It already carries the changed-file set (`ToolEffect`), proof
  (`VerificationRecord`, `VerificationStatus`), content provenance
  (`ContentRef`: inline / artifact / unavailable), and exact context selection
  (`ContextManifest` with selected/excluded messages and reasons).
- `crates/piku/src/run_view.rs` already renders a self-contained, CSP-constrained
  HTML workbench; `piku inspect <id> --html` writes it next to the run record.
  It is deliberate editorial styling, not yet the brutalist reasoning surface.
- A deterministic evaluation path and fixture corpus already exist
  (`docs/repo-artifact-dogfood-corpus.md`, `run_surface_eval`,
  `run_recovery_eval`).

## Gap

The record records *activity*, not *reasoning*. There is no
`decision_made`, `hypothesis_adopted`, `hypothesis_abandoned`,
`claim_concluded`, or `prediction_*`. The arc the human needs to grok — what the
agent believed and then reconsidered — is not captured. A decision spine cannot
be derived by rendering a chronological event feed; it must be recorded as
first-class typed events.

## Surface: a prediction-oriented reasoning workspace

Not a chat terminal. Not a code browser. Not a prettier transcript. The product
is the agent's operational understanding of the task, rendered as a live,
correctable object, with the human forced to test their own model against it.

### Breakthrough: prediction-first

At a load-bearing (model-changing) decision, before the agent reveals its taken
course, the workspace asks the human to commit:

> "The agent believes the constraint is at X. Before it proceeds, where do you
> think it is?"

Then it reveals what happened. This is the move that actually builds the human's
model, and it is the unproven-but-highest-leverage alternative to showing
calibrated confidence, which the reliance literature shows does not move the
needle.

- Rustic `prediction_committed`, `prediction_confirmed`, `prediction_refuted`
  are first-class events.
- Not every step requests a prediction — only model-changing nodes, to avoid
  turning the workspace into a nagging questionnaire. A *prediction budget* is a
  design we should actively tune and measure.

### Orient-on-return

When the human looks again, the workspace leads with the three answers that
drove the goal: what changed since they last looked, what the agent believed
last time that it has since reconsidered, and what needs them now. This is the
single most underbuilt surface across all harnesses.

### Decision spine, not a timeline

The unit of organization is the decision, not the event. Each node carries the
claim, the evidence links, who made it, whether the human predicted it, whether
it has since been contradicted, and the diff/code it produced. The current
flat-timeline `run_view` becomes the "full trace" fallback under each node.

### Editable working model, grok-required

The human can grab any node in the agent's model and correct it directly: "no,
the constraint is here, not there." Correcting requires committing to their own
reading first, so the human is building an actual model rather than reacting.
This preserves the friction that forms the mental model, per the positioning
doc's "some friction is how a user forms and tests the mental model required to
own the result."

### Calibration and tension feed

- Mark prediction-confirmation (builds calibrated trust) and surprise (the
  learning event).
- Flag claims that changed ("earlier said X, now says not-X, here's why").
- Voluntarily surface contradictions the agent itself notices: "you changed X
  but earlier decided Y," or "claim says fixed but test Z still fails." That is
  the independent-check trust boundary, made concrete.

### Sturdy guardrail

The agent doing the work also classifies what deserves attention. The workspace
must assume the agent is sometimes wrong about what is important. Surprise and
contradiction detection must therefore be computed *independently* of the
agent's own summary where possible, and the human's predicted-vs-actual is a
check on the agent's self-classification. High-risk decisions get conservative
escalation.

## Aesthetic: brutalism, deliberately

The visual direction is brutalist web design, not the current editorial
monospace styling and not antidesign.

- Brutalism = expose the structure: flat 1px solid borders, no rounded corners,
  no shadows, harsh color blocks for status and signal, typography as the
  primary layout driver, unpolished on purpose.
- It signals *raw evidence stated as-is*, not a finished artifact — inviting
  scrutiny rather than passive consumption, which is the point of the whole
  design.
- Not antidesign (monospace, blue links, deliberate ugliness or nostalgia) and
  not refined/editorial (the current ink/paper monospace), which reads as
  "designed" and quietly cues passive trust.
- System font stack for actual text; heavy weight for decisions and status;
  readable sizing. No node-weight to impress, only to make structure legible.
- Preserve the existing guarantees from `run_view.rs`: self-contained document,
  CSP constrained, artifact path escape checks, size-bounded inline content.

## Phased build

### Prototype evidence and revised boundary

The local canvas prototype supplied the concrete browser scenario that the
earlier gate required. A user clicked at the intended locus, asked for a small
interface, and then could not tell whether the agent understood the request,
what it was doing, why it was still running, or why the canvas remained
unchanged. The underlying run spent 20 iterations using repository tools and
ended without a canvas artifact. This is evidence for a mutable local
projection, but also evidence that reusing an unrestricted coding-agent loop is
the wrong authority boundary.

The first live slice therefore has a stable host-owned shell around a
customizable artifact:

- the user may compose the center canvas, but cannot customize away request
  state, authority, failures, or provenance;
- a click creates a prompt at that locus, and submission leaves a persistent
  activity card there instead of moving work into a hidden bottom transcript;
- canvas requests receive no filesystem, shell, or repository tools;
- model narration and partial HTML stream as typed events, so the browser can
  explain the current phase and update a sandboxed preview before completion;
- incomplete output and iteration exhaustion are failures with an unchanged
  saved canvas, never successful empty completions; and
- one surface accepts one active request at a time, preventing cloned-session
  races, while session history stores instructions and outcome summaries rather
  than repeated full HTML documents; and
- terminal logs summarize goal, model, outcome, iterations, elapsed time, and
  whether the canvas changed. Provider call IDs and result byte counts are
  diagnostic detail, not the primary operator story.

Arbitrary generated HTML belongs only in the sandboxed artifact frame. The
host shell and future evidence components remain trusted code. This is the
minimum separation needed for a canvas that can eventually be rearranged and
extended without letting generated presentation redefine its own authority.

### Terminal element and execution boundary

The canvas needs both a human shell and agent execution evidence. Calling both
of them a “terminal” obscures their different authority. Neither may be owned
by generated HTML.

Three shapes were considered:

1. A human-owned browser PTY is the chosen direct-manipulation shell. It is an
   interactive login shell rooted at the workspace and inherits the operator's
   ambient process, filesystem, environment, and network authority. It is not a
   sandbox, approval surface, agent tool, or evidence record. Its value is that
   the human can work directly without leaving the spatial workspace.
2. A typed command cell remains the chosen **agent execution** shape. The trusted host renders
   an immutable proposal containing executable and arguments, workspace-relative
   working directory, explicit environment projection, timeout, network posture,
   and requested authority. Approval binds to that exact proposal. Output,
   exit status, elapsed time, and observed effects become separate records.
3. A read-only file/evidence object remains useful for bounded inspection and
   lifecycle projection while command-cell enforcement is absent. It is not a
   substitute for the human shell.

The file object remains below shell authority. The operator may invoke bounded
typed host operations such as listing one workspace directory and reading a
line range from one text file. Paths are relative, canonicalized, and rejected
when they escape through traversal or symlinks. Sensitive path classes are
omitted or denied. Output is bounded and never appended to model context.

The PTY is a separate host capability. The browser connects to a loopback,
same-origin WebSocket only after the human creates or explicitly restarts a
terminal object. The host starts the configured login shell in the workspace,
caps concurrent sessions, forwards only terminal bytes plus bounded resize
messages, and kills and reaps the child when the object or connection closes.
Terminal output and input are not persisted, copied into chat, or exposed to
the model. Restoring layout restores an inert terminal object, never a shell.

The generated artifact may suggest that a terminal component should exist or
where it belongs. It cannot create an executable terminal, synthesize an
approval UI, send terminal events, or call an execution endpoint. Those are
host-owned capabilities.

Security invariants:

- generated HTML remains an opaque-origin sandbox with no network access;
- the web service binds to loopback, rejects non-loopback `Host` values and
  cross-origin mutations, and sends no-store, anti-framing, and content-type
  hardening headers;
- the human PTY is explicitly labeled as ambient user authority, while generated
  HTML, chat, canvas generation, and future agent tools cannot open it, type in
  it, read its output, or treat it as approval;
- PTY upgrades require a same-origin browser request, concurrent sessions are
  capped, and disconnect closes and reaps the child;
- command execution is parse-then-carry, not a string validated and later
  reinterpreted by a shell;
- working directories resolve beneath an operator-approved workspace root;
- environment values use an allowlist and secrets are references, never values
  in browser payloads or logs;
- network access is a separate capability, default denied;
- output is bounded, time-limited, cancellable, and attributed to one command;
- deletion, navigation, or client disconnect cannot silently orphan execution;
- every approval records the exact command identity and scope it authorized;
  and
- repository mutation is not added until the executor can record effects and
  apply the existing permission policy at the same boundary.

The lethal-trifecta rule is decisive for agent authority: an agent executor must
never combine private repository or secret access, untrusted artifact content,
and unrestricted network egress. The human PTY may possess all three because it
is the operator's ordinary shell, so the safety claim is narrower: no model or
generated artifact can observe or control it. If that topology cannot be
enforced below the model and browser layers, the PTY must not ship and command
cells remain read-only proposals.

### Spatial object creation and intent

A blank-canvas click opens a host-owned creation palette rather than assuming
that typing means “change the canvas.” The object types are `chat`, workspace
or page change, `terminal`, `file`, `note`, and page preview. Objects can be
moved by their exposed header, resized by pointer or keyboard, raised by
interaction, and closed explicitly. Identity, geometry, z-order, and bounded
object content persist on the server per surface. Browser storage retains only
viewport position as a local convenience. A terminal card persists, but its
live PTY process and output remain transient.

Intent is enforced at the request boundary. A chat request uses a separate
conversation session and a system prompt with no filesystem, shell, network,
or canvas authority. Only the explicit canvas-change request receives the
existing HTML artifact and may return replacement HTML. Vague canvas-change
requests ask for clarification and leave the saved canvas unchanged instead of
inventing a generic interface. The persistent bottom composer is conversational
chat, not an implicit rendering command.

The generated HTML artifact is not itself a workspace object authority. It
remains inside the opaque-origin iframe and cannot create trusted chat,
terminal, file, approval, or activity objects. Spatial controls, persistence,
and capability labels belong to the host application.

### Tasks, revisions, and artifacts are different objects

The board must not confuse an instruction with its output. A canvas-change
object is a task/control object: it records what the human asked, its authority,
request identity, current phase, and terminal outcome. Generated HTML is an
artifact object: a separately movable and resizable sandboxed page with a stable
identity. A successful task creates a new artifact or revises a named artifact;
it does not replace the board itself. The visible relationship is:

```text
task --targets--> artifact --has--> revisions
  |                   |
  +--status/evidence  +--sandboxed rendered result
```

This distinction prevents several current failure modes: a chat reply cannot
silently become a page; a page cannot visually erase the task that produced it;
in-progress output has somewhere stable to land; and future comparison or undo
can operate on artifact revisions rather than reconstructing chat history.

The current prototype has only one saved HTML artifact per surface. The first
coherent UI step renders that HTML as a host-owned artifact object rather than
as the full-canvas background. Stable server-side object identity, multiple
artifacts, revision history, and task-to-artifact provenance remain follow-up
data-model work. Browser `localStorage` geometry is prototype state, not the
durable workspace source of truth.

Existing artifacts are revised through host-applied source operations, not
whole-document regeneration. Creation accepts one initial self-contained HTML
document. Revision accepts only ordered exact replacements whose search text is
non-empty and unique in the current source. The host applies them to the saved
source, rejects ambiguous, missing, no-op, or oversized mutations, and streams
the resulting source back to the same artifact object. This is the first
request-scoped canvas tool seam. It should become ordinary typed model tools
once the runtime supports a per-turn executor context; exposing the generic
filesystem `edit_file` tool would violate the canvas-only authority boundary.

### Web source layout

Axum routing, persistence, provider calls, PTY lifecycle, and security
enforcement remain in `crates/piku/src/web.rs` and `crates/piku/src/web/`.
Browser behavior and presentation are authored in
`crates/piku/web-ui/{app.js,style.css}`. A pinned esbuild step bundles xterm.js
and the authored source into `crates/piku/src/web/{app.js,app.css}`, which Rust
embeds with `include_str!` so the release remains one binary. Node is a
build-time dependency, not a runtime server or framework commitment.

The design deliberately avoids starting with a browser server or a live tail.

### Phase 1: evidence-first decision spine (static, additive)

The loop emits tool-level activity, not model reasoning. The honest way to form
a decision spine first is to derive nodes from **deterministic evidence**, not
from the model's prose claims:

- Cluster `ToolEffect` (files created/modified), `VerificationRecord`
  (test passed/failed/indeterminate), and `PermissionDecision` into
  decision-shaped groups. These are facts about what changed and whether it
  held, decidable without any model-judgment call.
- Render the spine brutalist: `piku inspect <id> --html --view=decisions`.
  Additive, in the existing run_view idiom, no new surface, no new event types.
- This is the experiment that tests whether a returning human groks the decision
  spine in seconds. Deterministic, additive, reversible.

Prose-derived reasoning events (`DecisionMade { claim, .. }`,
`HypothesisAdopted`, `Prediction*`) are a **later, separate, lower-trust layer**,
explicitly labeled as derived from the model's own claims, never merged into the
evidence spine. They need a tool-boundary or evidence-backed emission point, not
free-text inference.

### Phase 2: live and independent signal

- Tail the run record and re-render (or stream) for a live watch-it-work mode.
- Compute contradiction detection independently where data allows (e.g., "claim
  says fixed but verification is Failed").

### Phase 3: browser reach only if Phase 1/2 earn it

- The positioned doc gates browser surfaces behind a concrete scenario and
  acceptance check. This design is the candidate scenario. A stable external
  event protocol / browser server is deferred until the projection proves it
  reduces attention cost without hiding failures.

## Risks and open questions

- Evidence-first nodes are the authoritative spine; prose-derived reasoning
  events are a lower-trust layer that must be labeled as such and never merged
  into the evidence spine. A fabricated `decision_made` inferred from prose is
  worse than none; its emission point needs a tool-boundary or evidence-backed
  mechanism and tests before real trust is placed in it.
- Prediction-first adds interaction friction exactly where the human is likeliest
  to want to skim. The prediction budget is the untested control; it must be
  tuned and measured, not assumed.
- The event schema is additive but must stay compatible with existing
  `RunEventEnvelope` schema versioning (currently v2, run/turn scope explicit).
  Adding variants should not break v1/v2 readers.
- What exactly count as "model-changing" nodes is a judgment call by the same
  agent whose work we are trying to keep honest; independent signal is a
  requirement, not an optimization.
- The measure of grokking (the three orient-on-return answers + planted-defect
  detection) still needs a concrete, repeatable protocol and baseline from a
  control that shows the flat transcript.

## Sources

- Existing: `docs/design/agent-investigation-workbench.md`,
  `docs/design/durable-investigation-core-and-surfaces.md`,
  `docs/design/agentic-harness-roadmap.md` (decision-view and
  `measure comprehension alongside volume` gates),
  `docs/design/agentic-harness-positioning.md` (browser-surface and
  authority gates), ChatGPT export under `docs/design/sources/`.
- Harness engineering and failure modes: Ady Osmani, *Agent Harness
  Engineering*; Nimbalyst, *What Is an Agent Harness? Eight Pillars*.
- Reliance/trust: arXival 2412.15584 (*To Rely or Not to Rely?*) and
  2603.22634 (*Learning to Trust*).
- Aesthetic: NNGroup *Brutalism and Antidesign*; brutalistwebsites.com.
