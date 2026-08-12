# Operator journey roadmap

Status: implementing

Baseline: `97b37f3` on `main`

Review trigger: revisit after the first held-out repository-change journey passes
end to end, or earlier if the mutation boundary cannot be enforced without
giving the web process ambient write authority.

## Purpose

Make Piku useful as an inspectable coding workbench before expanding its
ontology or evaluation machinery. The next milestone is one coherent operator
journey:

```text
open a real file
  -> attach exact, visible context
  -> reason in a resumable notebook thread
  -> request and approve a repository edit
  -> inspect the proposed and actual effects
  -> run a check
  -> record the evidence and human conclusion
  -> reload and continue without reconstructing the transcript
```

This is the integration roadmap for the current web workbench. It narrows, but
does not replace, the broader harness roadmap. Accepted security and executor
decisions remain authoritative.

## Product invariant

Piku should help an operator understand and control work, not merely watch an
agent produce it. At every consequential step the operator must be able to
answer:

- what was asked;
- which exact context the executor received, including exclusions and stale
  snapshots;
- which executor, thread, turn, tool policy, and working directory acted;
- what was proposed, approved, attempted, changed, and verified;
- which claims are model interpretation versus deterministic evidence;
- what remains unresolved and where to resume.

Ordinary repository files, Git state, tests, commands, and notes remain
canonical. Piku stores identities, hashes, relationships, decisions, and
results rather than replacing those artifacts with a proprietary notebook.

## Current checkpoint

The following pieces are implemented and should be composed rather than
rebuilt:

- a server-owned spatial board with persistent typed objects and geometry;
- notebook chat with editable turns, rerun, rerun-from-here, cancellation,
  persistence, optional context, and Codex thread lineage;
- authoritative server resolution of selected context-object IDs and recorded
  source hashes;
- a read-only Codex app-server executor and a separate provider executor;
- bounded workspace-relative file reads and persistent file-card snapshots;
- a real, human-started PTY rooted at the workspace;
- page-source editing with durable source history and sandboxed preview;
- durable run records with context, tool, effect, verification, and identity
  vocabulary, plus a static run inspector;
- Markdown, KaTeX, Mermaid, light/dark themes, and stable cross-surface
  evaluation identities.

The repository-mutation bridge is partially implemented behind a fail-closed
gate. Piku has a single-use per-turn lease, an explicit browser review surface,
Codex workspace-write wiring, native command and file-change projection,
deadline/cancellation handling, and durable authority/effect records. Ordinary
chat remains read-only, page prompts cannot change repository files, and the
human PTY remains a separate ambient operator shell.

The browser write action stays disabled until the version-pinned app-server
probe exercises thread start, resume, a real turn, writable-root containment,
network denial, and native elevation denial. A held-out repository mutation
journey and before/after effect inventory are also still missing.

## Decisions before implementation

### 1. Repository mutation approval

The accepted direction is Codex workspace-write under Piku-owned approval.
ADR 0011 chose a revocable, single-use, per-turn lease; its implementation is
present but its acceptance gates remain open.

Options:

1. Approve a whole turn for bounded workspace write.
2. Approve each exact proposed effect or command.
3. Keep web chat read-only and require all edits in the human terminal.

Preferred first slice: option 1 with a visible, revocable per-turn lease bounded
to the selected workspace, executor, thread, turn, lifetime, and tool profile.
Before execution, show the requested authority and working directory. During
execution, record permission decisions and tool effects. Afterward, show the
actual changed-file inventory and verification results. This preserves Codex's
native coding loop without pretending every eventual effect can be predicted
in advance.

Option 2 may later be added for especially sensitive commands, but making it
the only first path risks approval fatigue and cannot reliably predeclare every
edit in an iterative coding turn. Option 3 is safe but does not satisfy the
inspectable-agent product thesis.

ADR 0011 records this contract and distinguishes permission from containment.

### 2. File-card freshness

Current file attachments send the saved card snapshot, not a fresh repository
read. The UI identifies the card primarily by path, so this is easy to
misunderstand.

Options:

1. Snapshot semantics: attachments always use the captured bytes.
2. Live semantics: attachments reread the path when a turn starts.
3. Explicit dual semantics: snapshots are reproducible; refresh is an operator
   action that creates a new revision.

Preferred: option 3. Show revision, digest, capture time, and `current` or
`stale` state. Sending uses the visible captured revision. Refresh performs the
same bounded, workspace-relative server read, creates a new revision, and makes
the context change explicit. Reruns retain their original snapshot unless the
operator chooses refresh. This preserves reproducibility without hiding drift.

This is a product contract and should be recorded in the mutation ADR or a
small companion ADR before context provenance becomes public API.

### 3. Terminal and agent execution

Keep the human PTY and agent execution distinct:

- the PTY is a direct-manipulation object started by the human;
- agent commands are typed run events governed by the executor lease;
- workspace operations never type into the PTY;
- PTY output becomes model context or durable evidence only through an explicit
  capture action with source, byte bounds, and operator intent.

No new decision is required unless Piku later attempts shared terminal control.

### 4. Investigation identity

Do not introduce a new top-level `Investigation` object for this milestone.
Use the existing workspace, card, session, thread, turn, request, and run IDs.
Revisit only if the resume view cannot express one journey without ambiguous
joins. This avoids solving identity theory before a consumer fails.

## Delivery sequence

### Phase 0: freeze the acceptance contract

Write one held-out Playwright scenario and a matching deterministic contract
before adding the write path. Its clauses must independently check:

1. create or reopen a workspace at a known repository revision;
2. open a named real file successfully;
3. attach its exact saved revision and one note to a chat turn;
4. inspect included and excluded source IDs and rendered-context digest;
5. receive a read-only response on the existing thread;
6. request a precise repository edit;
7. see and approve the exact workspace-write lease;
8. observe permission, command/file effects, and changed-file inventory;
9. run a real verification command through the agent executor;
10. distinguish command success from the model's interpretation;
11. add a human conclusion and unresolved follow-up;
12. reload, reopen the same objects, append to the same thread, and inspect the
    same run lineage and artifacts.

Each clause is `pass`, `fail`, or `inconclusive`. Harness, dependency, timeout,
and evidence-capture failures cannot become product failures. The test fixture
must be isolated from the Piku checkout and cleaned only after artifact capture.

Gate: the contract fails for the known missing mutation seam and passes all
already implemented clauses. No aggregate score may hide a failed clause.

### Phase 1: settle authority and freshness

Write and accept the mutation/freshness ADR. Define:

- lease identity, scope, lifetime, cancellation, and denial precedence;
- workspace root and working-directory resolution;
- executor and tool-profile identity;
- environment and network policy;
- permission events and operator provenance;
- changed-file and external-effect limits;
- snapshot, refresh, staleness, and rerun semantics;
- what survives reload and what is deliberately ephemeral.

Gate: tests prove denial dominates later allows, a lease cannot escape its
workspace through normal file tools, cancellation revokes it, and neither a
lease nor a path boundary is described as OS containment. If unrestricted shell
remains available, external effects must be labeled unobserved.

### Phase 2: wire one write-enabled Codex journey

Add workspace-write as an explicit mode of the existing Codex executor. Reuse
the runtime run-record vocabulary rather than adding web-only activity types.
The host owns approval; Codex owns its native coding loop; Piku owns context,
authority, persistence, and effect projection.

The first UI should be deliberately small:

- `read only` and `propose changes` are distinct turn intents;
- the notebook shows executor, mode, cwd, selected context, and lease state;
- starting a write turn presents one approval surface;
- active work streams semantic stages and concrete tool effects;
- completion shows changed files, diff access, checks, errors, uncertainty, and
  a link to the durable run record;
- cancel has an honest terminal state and preserves partial effects as such.

Do not let generic bottom chat infer mutation intent. A write-enabled turn must
originate from an explicit notebook action or typed task object.

Gate: the held-out fixture receives exactly the requested edit, records the
permission and effects, runs its check, survives reload, and cannot read or
write an unrelated sibling fixture through Piku file tools. Partial mutation,
timeout, cancellation, and failed verification have distinct outcomes.

### Phase 3: bring evidence into the workspace

Replace the new-tab-only run inspection path with a workspace evidence panel
that projects the existing durable record. It may be a focused/full-window
view of a card; it is not a new runtime or notebook format.

Show, with lossless drill-down:

- task and human intent;
- context manifest, source revisions, staleness, exclusions, and total budget;
- executor, model, thread, turn, request, run, tool profile, cwd, and authority;
- chronological semantic stages;
- proposed versus actual file and shell effects;
- checks and their raw artifacts;
- model claims, deterministic facts, warnings, and unresolved questions;
- the operator's conclusion or disposition.

Routine events may be coalesced, but errors, mutations, permission decisions,
verification failures, divergence, and truncation must remain visible. Every
collapsed item must lead to the complete artifact or explain why it is absent.

Gate: after a cold reload, a person can identify what the agent saw, did,
changed, verified, and failed to establish without reading server logs or the
entire transcript.

### Phase 4: finish resume and direct manipulation

Make the notebook capable of occupying the useful workspace viewport while
remaining the same movable object. Add focus/full-window presentation, restore
prior geometry on exit, and keep keyboard and screen-reader focus coherent.

Add explicit evidence capture from a human terminal selection or command result
as a bounded immutable attachment. Do not record all terminal bytes by default.
Add a human conclusion and next-action field to the journey; keep it distinct
from assistant output and evaluator follow-ups.

Gate: focus mode does not fork identity or lose geometry; explicit terminal
captures retain provenance and size bounds; resume opens on the active question,
latest conclusion, stale inputs, failed checks, and next action.

### Phase 5: evaluate the integrated journey

Keep the current `coding_trace` and `recovery` explorers and fresh synthesis
judge unchanged for the first integrated run. Before starting them, validate
that each required evidence modality is available. Missing screenshots, DOM,
run records, or deterministic predicates make the run inconclusive before
synthesis.

Judge prompts receive the immutable task contract, exact harness/model versions,
viewport and color scheme, bounded artifacts, and their own lens. Explorers do
not receive prior verdicts. Synthesis must cite stable evidence IDs and cannot
override deterministic predicates.

Inspect each finding qualitatively: reproduction, mechanism, alternative
explanation, user impact, and what observation would reverse it. Scores are a
secondary summary, not evidence of progress.

Add `first_use` and `authority` perspectives only after:

- the two-perspective run completes within its time and cost budgets;
- setup, cleanup, artifact, and evidence validation are deterministic;
- planted mutations in progress, provenance, stale context, and authority
  labels are detected;
- the additional perspective has a non-overlapping question.

Gate: one blinded held-out run completes with valid evidence, one deliberately
broken variant is detected for the right reason, and all run artifacts include
revision, dirty state, harness version, model/executor identity, viewport, color
scheme, prompts/schemas/tool-profile hashes, and screenshot provenance.

### Phase 6: promote only repeated evidence

Turn repeated, freshly reproduced findings into deterministic regression tests.
Keep annotations, judge proposals, operator corrections, and promoted focus as
separate append-only records. Only an explicit operator promotion may steer a
future evaluator, and promoted focus remains an advisory question with an
expiry and retest obligation.

Gate: a judge proposal cannot alter a later prompt, rubric, tools, or authority
without a distinct operator event; blinded runs remain possible; resolved and
obsolete findings stop consuming prompt budget.

## Parallel work that is safe

Only work with independent contracts should proceed concurrently:

- Phase 0 acceptance fixtures can be written while the ADR is reviewed.
- File snapshot/staleness UI can proceed beside the executor lease if both use
  the agreed context manifest.
- Full-window presentation can proceed beside write-path work if it does not
  change object identity or persistence.
- Deterministic evaluator preflight can proceed beside product work if prompts
  and scoring are frozen.

Do not run multiple agents against the mutation contract, run-record schema, or
shared evaluation validator simultaneously. Those are convergence points with
one mutation owner each.

## Consolidation and deletion

Before expanding the product, remove or retire drift that obscures the journey:

- mark older chronological claims in brainstorm documents as historical when
  newer implemented designs supersede them;
- keep one shared evaluation envelope and validator; adapters remain
  surface-specific;
- remove compatibility paths for pre-stable finding IDs only after retained
  ledgers no longer require them;
- replace duplicate transient web activity shapes with the durable run-event
  projection as each consumer migrates;
- do not add a second context store, workspace-write event schema, browser
  runtime, or evaluation ledger;
- keep generated assets and build intermediates outside source organization
  where the existing build permits it.

## Deliberately deferred

The following are attractive but are not prerequisites for the operator
journey:

- a general resolver/plugin marketplace;
- arbitrary computed context pipelines beyond the selected-card and host-fact
  seams;
- a universal notebook runtime or reactive visual DAG;
- automatic terminal recording or agent control of the human PTY;
- Codex-driven arbitrary page tools;
- multi-agent spatial orchestration;
- ACP/editor or native macOS projections;
- a database for evaluation ledgers;
- additional judges, meta-judges, or self-modifying evaluator prompts;
- claims of OS sandboxing without an enforced and tested process boundary.

Reconsider a deferred item only when a held-out journey fails because the
current composition cannot express it, not because the abstraction is
available.

## Stop and reversal conditions

- If Piku cannot enforce and explain workspace-write authority independently of
  prompt text, keep web Codex read-only and retain human-terminal editing.
- If typed context manifests do not select the same bytes sent to the executor,
  stop UI work and repair request construction first.
- If the evidence panel becomes a second copy of repository files or raw
  transcripts, return to stable references plus complete artifacts.
- If live judges fail to detect planted deterministic regressions, freeze judge
  expansion and improve the harness rather than tuning scores.
- If focus injection increases unsupported historical claims or merely echoes
  operator wording, disable it and retain annotations as review-only records.
- If a second artifact type cannot fit the current typed object model, write a
  schema decision then; do not preemptively build a plugin system.

## Immediate checkpoint

The next cohesive change is Phase 0 plus the Phase 1 ADR. Do not implement
write-enabled Codex until that ADR defines approval, workspace, environment,
network, cancellation, effect-recording, and file-freshness semantics, and the
held-out scenario demonstrably fails only at the missing mutation boundary.
