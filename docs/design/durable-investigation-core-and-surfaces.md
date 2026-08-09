# Design: durable investigation core and product surfaces

Status: active; durable core and first projection comparison implemented

## Problem

Piku already owns a provider-neutral agent loop, tools, sessions, permissions,
hooks, memory, background agents, and a terminal UI. Its evaluator has begun to
record stronger semantics than the product itself: stable review claims,
evidence references, observer dispositions, executable acceptance results,
spend, and an engineering handoff. The runtime still divides the same work
between chronological session messages, provider streaming events, selective
traces, terminal callbacks, and final repository state.

That split cannot support the workbench described by the recovered transcript.
A compact TUI, a Swift investigation app, an editor client, and a browser
capture tool will drift if each reconstructs context, authority, evidence,
changes, and completion independently. A polished summary also cannot be the
canonical record: the central requirement is less reading without losing the
details needed to understand, challenge, modify, or debug the result.

## Chosen approach

Build one small, append-only, provider-neutral investigation record in
`piku-runtime`, then treat every interface as a projection or adapter. The
record is not a transcript replacement or universal notebook format. It gives
stable identity to the events that affect a person's understanding and
authority:

```text
Investigation -> Turn -> Item -> Evidence
                       -> Decision / Disposition
```

The exact names may change during implementation. The required semantics do
not: model and context attribution, user intent, tool invocation and complete
result reference, permission decision, file/change effect, verification,
compaction, warning or divergence, agent handoff, and human conclusion. Raw
text and full outputs remain ordinary artifacts behind stable references.

The product objective is a Pareto improvement, not a single readability score:
reduce extraneous attention while preserving or improving task acceptance,
defect detection, delayed comprehension, successful modification, and
debugging. A compact surface that hides failures is a regression even when it
uses fewer lines.

## Layer placement

### 0. Canonical work artifacts

Repository files, Git history, tests, commands, datasets, citations, and human
notes remain canonical. Piku records identities, hashes, revisions, results,
and relationships; it does not trap the work in a proprietary document.

### 1. Piku runtime: semantic source of truth

`piku-runtime` owns lifecycle semantics and persistence. It should replace the
ad hoc `OutputSink` contract with typed events, retain a compatibility
projection into the existing `Session`, and make the exact provider-request
context an event-backed manifest rather than a TUI-only approximation.

The existing provider `Event` remains a wire-normalization layer. It is too
narrow to be the product record. The current trace remains useful for
diagnostics but is explicitly selective and lossy. The agentic playground
ledger remains evaluation evidence; the minimum durable concepts it has proven
should move into the runtime rather than making test schema the product API.

### 2. Headless and TUI projections: prove the interaction first

The first client is the existing TUI plus a stable JSON/JSONL projection. It
should expose:

- exact context composition and compaction changes;
- coalesced routine activity with lossless expansion;
- salient errors, mutations, authority decisions, and divergence;
- a completion packet grouped by goal, semantic changes, files, verification,
  risk, and unresolved decisions;
- a distinct place for the user's conclusion or disposition.

This is the cheapest place to test the thesis because it exercises the real
runtime and existing PTY evaluator without committing to another application.

### 3. ACP/editor projection: next when a second client is real

An ACP server adapter can expose Piku inside Zed or another compatible editor.
ACP supplies client-agent session, prompt, progress, permission, file, and tool
interactions. It should translate the Piku record, not become Piku's internal
ontology. Piku continues to own model selection, tools, permissions, and
persistence; the editor owns presentation and editor-native resources.

ACP is not yet the next implementation step. The current ACP v2 migration
guide describes v2 as a draft and recommends v1 and v2 side by side behind
version negotiation or feature flags. It also moves filesystem and terminal
execution toward MCP while adding asynchronous prompt acknowledgement,
message upserts, replay, and background or multiple-client semantics. Those
are real compatibility costs. Until Piku has a concrete second client and a
scenario the existing projections cannot satisfy, an ACP adapter would be a
speculative second API. If that gate changes, the adapter stays thin over the
durable record and supports negotiated protocol versions rather than teaching
the runtime ACP's ontology.

### 4. Local web workbench: a first-class second-client candidate

A browser application may be the strongest custom investigation surface. It can
combine Markdown and math rendering, virtualized history, semantic diffs,
side-by-side forks, evidence graphs, plots, and interactive inspection with a
shorter iteration cycle and broader portability than a native Mac client. This
is the relevant comparison to Jupyter or marimo; it is distinct from automating
the user's foreground browser.

The first browser projection is deliberately static and read-only. It consumes
the same artifact-aware search entries as the terminal projection and embeds
the complete evidence needed for inspection. A controlled retrieval scenario
found that both the composed terminal path and browser found the exact deep
artifact with provenance; the scripted browser path required one navigation
action versus two in the terminal. That is enough to keep the browser as a
first-class surface. It is not evidence that a server or browser-owned runtime
is better.

A mutable workbench should be served by a local Piku process and consume the
same typed event/query contract as every other client. It must not gain ambient
file or shell authority through a generic local HTTP server. Its gate is a
measured need for live updates, actions, executable views, or fork comparison
that static HTML cannot satisfy. Before implementation, the design needs an
explicit loopback binding, random per-launch capability, origin and CSRF
policy, CSP, artifact allowlisting, read/write boundary tests, and lifecycle
cleanup.

After the TUI establishes a baseline, compare a thin local web prototype with an
ACP/editor projection on the same review, recovery, and forking scenarios. Favor
the web path if its custom spatial interactions materially improve decision time,
comprehension, or fork comparison without creating a second semantic model.

Jupyter and marimo strengthen this placement, with an important limit. Jupyter
separates the browser-facing server, document, and kernel, while marimo exposes
a browser IDE, reactive dataflow, app mode, static export, and a WebAssembly
mode that can execute entirely in the browser. They demonstrate that a browser
can be the primary interaction surface for serious work. They do not show that
Piku should adopt mutable notebook cells, a Python kernel, or browser state as
its semantic source of truth. Piku's analogous split is durable investigation
record, query and projection layer, then terminal, browser, and editor clients.

### 5. Swift investigation app: only for demonstrated native advantages

A Mac app remains a possible later surface, but a local web workbench can provide
most of its initially proposed value. Swift should consume the same local runtime
service or event stream and must not duplicate the agent loop. Build it only when
measurements expose a native-only requirement such as OS-wide capture, deep
accessibility integration, robust offline document ownership, or interaction
quality the browser cannot provide at acceptable cost.

### 6. Browser capture and external-harness adapters

A Safari extension or bookmarklet is an import adapter, not the web workbench.
It captures an exact
conversation identity and extracts rendered turns or an export into a bounded,
private artifact. It does not become Piku's automation or control plane.
Controlling a foreground Safari tab is fragile, disrupts unrelated browsing,
and makes target identity and privacy harder to prove.

Codex and Claude Code are peer harnesses as well as model ecosystems. Piku may
later import their supported structured streams or exports for comparison and
review. Their private transcript files must not become Piku's canonical schema.
Wrapping them as Piku's primary execution engine is deferred because it would
collapse Piku's authority and evidence model to vendor-specific behavior.

### 7. MCP and hosted web

MCP is a tool, resource, and prompt integration boundary, not the investigation
or UI model. Add it only for a concrete tool-access scenario. A hosted or
multi-user web product remains a separate decision requiring identity,
isolation, secrets, retention, and operations. Success of a capability-scoped
local web workbench does not authorize turning it into a hosted control plane.

## Evaluation design

The agentic-user playground should evaluate the thesis, not only discover
classical agent bugs. Add controlled scenarios for five user intents:

1. **Delivery:** locate what changed and whether it is safe.
2. **Review:** detect a planted defect or unsupported claim.
3. **Ownership:** explain an invariant, then make a small novel modification.
4. **Recovery:** resume after interruption and identify the next justified step.
5. **Forking:** revise an assumption and distinguish inherited from recomputed
   evidence.

Each scenario freezes revision, model, seed, viewport, task contract, evidence
catalog, and presentation arm. It records pass, fail, harness failure, timeout,
and unavailable evidence separately. Deterministic predicates retain authority
only for their exact properties. Judges may assess open-ended explanations, but
every claim must cite stable evidence and recursive review cannot manufacture a
product verdict.

Use a metric vector rather than optimizing a blind composite:

| Dimension | Measures |
| --- | --- |
| Outcome | executable task acceptance, regression count, planted-defect detection |
| Attention | time to locate result, transcript bytes/lines, scroll distance, expansions, approval interruptions |
| Understanding | delayed explanation accuracy, invariant recall, successful modification, debugging transfer |
| Evidence | supported-claim rate, provenance coverage, time to reach full artifact, false reassurance |
| Control | unapproved mutation count, authority-decision recall, predicted versus actual side effects |
| Continuity | time to resume, correct next step, stale-context detection, fork comparison accuracy |
| Efficiency | input/output tokens, cost, wall time, retries, tool-selection misses |

The north-star phrase remains “verified, understood change per unit of scarce
human attention,” but it should not be collapsed into one score until its
components predict real outcomes. Ship only when acceptance and safety floors
hold and at least one attention or understanding measure improves. Agentic
simulation can prove event completeness, salience rules, recoverability, and
judge grounding; actual claims about human learning or fatigue require repeated
human trials.

## Implementation plan

1. Define versioned runtime `RunEvent`/`RunItem` types and an append-only writer.
   Preserve the existing session format through a compatibility projection.
2. Give every full tool result and pre-compaction state a durable artifact or
   explicit unavailable marker; eliminate silent truncation from evidence.
3. Emit exact context manifests from the same request-building path, including
   selected content identities, model, tools, instructions, revision, and
   compaction decisions.
4. Build TUI and headless completion/context projections from the event record.
5. Add delivery, review, ownership, recovery, and fork scenarios plus the metric
   vector to the playground ledger. Run raw-transcript and compact-view controls.
6. Add a user-authored conclusion/disposition and test that resume and fork keep
   it distinct from model claims.
7. Continue the static browser projection as a first-class inspection surface.
   Add a local mutable server only after a live-action scenario beats static
   export; add ACP only when a real editor client supplies a distinct win.
8. Implement Swift only when a native-only workflow has a measured advantage
   over TUI, local web, and editor projections.

Steps 1-6 are reversible through versioned files and compatibility readers.
External protocol commitments require explicit compatibility policy before
release.

## Implementation checkpoint

The first sequence is now implemented on `main` through `ffc7af9`:

- `piku-runtime` persists append-only run events, complete large-output
  artifacts, context manifests, user dispositions, effect and verification
  evidence, and durable child-agent session/run/link records.
- Piku emits and audits those events, exposes text, JSON, and static browser
  inspection projections, and reports a metric vector without pretending that
  a machine trial measured human comprehension.
- Paired deterministic recovery and fork evaluations distinguish continuity
  from a fresh-session control. A live isolated recovery trial also crossed two
  real Piku processes through OpenRouter and measured two attempted and two
  completed turns, one session identity, contiguous event sequences, and exact
  marker recall.
- The terminal and static browser projections both retrieved a deep artifact
  with provenance. The browser required fewer scripted navigation actions, so
  it remains a first-class candidate; this does not yet satisfy the gate for a
  mutable local server.
- Recovery presentation now has an explicit boundary metric: the compact
  terminal shows aggregate continuity, the static browser locates the exact
  selected `ContextBuilt` record with less filtered evidence, and only a
  composed terminal can join that record to the durable session message that
  contains the prior marker. The browser must not claim that end-to-end proof
  until a validated session/run bundle exists.

The next product gate is therefore not another shell around the agent loop. It
is the remaining fork comparison plus a decision on a typed session/run bundle
for the browser. A mutable browser service is justified
only if live actions or executable views improve the metric vector while the
security boundary above passes. ACP waits for a concrete editor client, and
Swift waits for a measured native-only advantage.

### Child-run browser evidence

The parent run must record a typed, path-safe child-link event before a static
browser workbench can traverse a fork. The event is the authority for which
child evidence belongs to the parent; the browser may then resolve only
validated, run-relative child artifacts included in that bundle.

Scanning an inferred links directory is rejected. It makes the view depend on
ambient machine state, can accidentally include unrelated runs, and cannot be
replayed from the parent record. Parsing the current `agent_join` prose is also
rejected: it makes a user-facing string a schema and leaks absolute local paths.

This does not add a browser-owned runtime, arbitrary filesystem browsing, or a
recursive graph viewer. It only makes one recorded parent-to-child relationship
inspectable. Reverse this direction if a typed event cannot be emitted at the
tool boundary without weakening the recorder's one-event-per-semantic-fact
invariant, or if the bounded bundle cannot reject escapes and cycles.

## Non-goals

- No universal notebook runtime or visual DAG in the first implementation.
- No feature-parity race with Codex, Claude Code, or OpenCode.
- No web or Swift app that owns a second agent loop or parses terminal
  presentation.
- No foreground Safari automation as a capture dependency.
- No single metric that rewards shorter output while hiding defects.
- No claim that agent judges prove human comprehension.

## Decision gates

- A second surface may start only after one typed runtime stream can reproduce
  the current TUI lifecycle without parsing presentation text.
- A local web prototype must bind to loopback, require a per-launch capability,
  constrain artifact access, and pass origin/CSRF/CSP boundary tests before it
  can invoke or approve work.
- Every collapsed item must reach its complete artifact or explicitly say why
  the artifact is unavailable.
- Context manifests and actual provider requests must share construction and
  select the same content identities.
- Compact presentation must not reduce planted-defect detection or executable
  acceptance relative to the transcript control.
- Ownership interventions survive only if delayed explanation, modification,
  or debugging improves enough to justify their extra time.
- If the semantic record grows into a second copy of all repository content,
  stop and return to references plus content-addressed artifacts.

## Open questions

- Whether `Investigation` should initially be a new identity or a projection
  over `Session` plus stable turns.
- JSONL alone versus JSONL plus a derived SQLite index for local querying.
- The minimum portable extension metadata ACP needs for Piku-specific evidence
  and decision projections.
- Which first human repeated-measures task can validate the ownership metrics.
- Which live-action or fork-comparison task, if any, justifies crossing from a
  static browser artifact to a capability-scoped local server.

## Current evidence

The mutable local prototype has now exposed a narrower, concrete use case than
the hosted/browser-owned runtime rejected above: direct manipulation of a
local, sandboxed artifact while a single Piku session remains the continuity
owner. The initial implementation failed this test by giving a canvas prompt
the full coding-tool catalog and returning only a generic `working` state. It
could spend many iterations changing repository files while producing no
canvas result.

The accepted prototype boundary is therefore canvas-only generation with no
repository tools, a host-owned activity overlay, incremental artifact
snapshots, and semantic lifecycle events shared by browser presentation and
operator logs. The canonical artifact is stored once; conversational continuity
keeps compact instructions and outcomes so repeated edits do not multiply the
entire document through model context. A surface serializes mutations rather
than allowing concurrent session clones to race. This does not authorize
arbitrary filesystem browsing, approvals, or a second agent loop. Crossing
those boundaries still requires
the capability, origin, and audit gates above.

- Durable run record, semantic scopes, complete artifacts, context manifests,
  audit, dispositions, terminal/browser projections, and principle metrics:
  commits `46350de` through `42e8954`.
- Browser/terminal controlled retrieval comparison:
  `tests/run_surface_eval.rs` at `42e8954`.
- Codex app-server's typed thread/turn/item API supports the adapter-over-core
  split: <https://github.com/openai/codex/blob/main/codex-rs/app-server/README.md>.
- ACP v2 migration and compatibility costs:
  <https://agentclientprotocol.com/protocol/v2/migration>.
- Jupyter's separated architecture:
  <https://docs.jupyter.org/en/latest/projects/architecture/content-architecture.html>.
- marimo's browser editor, app, static, and WebAssembly modes:
  <https://docs.marimo.io/llms.txt>.

---
Proposed: 2026-08-05 | Session: current Codex session
