# Agentic harness roadmap

Status: proposal

Reviewed: 2026-08-03

## Purpose

Turn the harness survey, human-attention research, current design corpus, and a
fresh repository orientation into a staged plan for Piku. This is a product and
architecture roadmap, not a commitment to feature parity.

The governing objective is verified change per unit of scarce human attention.
That objective has two constraints:

- reducing transcript volume must not hide evidence needed to understand,
  verify, modify, or debug a change;
- preserving learning must not force users through boilerplate or delay that
  does not improve later ownership.

The tracked operational evidence loop remains in
`docs/live-dogfood-roadmap.md`. This document is part of the intentionally
owner-local design corpus and may refer to the local ADR ledger.

## Inputs

- `docs/design.md`, the current implementation guide;
- `docs/agentic-harness-landscape.md`, the external evidence synthesis;
- `docs/live-dogfood-roadmap.md`, the operational evaluation loop;
- focused designs under `docs/` and accepted decisions under `docs/adr/`;
- fresh inspection of the workspace graph, CLI and TUI surfaces, tool registry,
  agent lifecycle, tests, scripts, and CI.

## Orientation reconciliation

| Roadmap premise | Fresh Orient finding | Consequence |
| --- | --- | --- |
| Provider/runtime separation is already narrow and coherent. | `piku-api` owns provider protocols; `piku-runtime` owns resolution and the loop; the binary owns CLI and TUI wiring. | Do not schedule another provider refactor. Finish focused configuration cleanup only when a consumer requires it. |
| Deterministic evaluation is a real strength. | Scripted-provider integration tests, isolated PTY smoke tests, trace-backed dogfood, and one canonical `just check` path exist. | Use these boundaries as gates for authority, lifecycle, and attention changes. |
| Capability behavior differs by surface. | The entered TUI has prompts, hooks, a task registry, and completion injection. Every writable launch turn uses `AllowAll`, omits hooks and the task registry, yet advertises agent tools. | Make surface truth and non-interactive authority the first implementation work. |
| Worktrees are not an enforcement boundary. | A worktree path is placed in the child prompt, but executor and file-tool working directories are unchanged. Bash-only changes can evade the dirty heuristic. | Decide and implement an executor-scoped workspace boundary before calling this isolation. |
| Child notification is partial, not absent. | The TUI injects completions through a bounded channel; launch turns have no equivalent. Delivery is unacknowledged and output is unbounded. | Improve lifecycle semantics instead of adding a duplicate notification feature. |
| Evidence precedence is narrower than the prose implied. | Scenario predicates do not cover every goal clause, and verifier infrastructure failures share the product-failure path. | Add pass, fail, and inconclusive states before relying on acceptance outcomes as roadmap authority. |
| Tool discovery is not lazy. | `tool_search` searches metadata for schemas already included in the request. | Measure and implement actual schema disclosure only after tool-result semantics are dependable. |
| Memory is implemented but weakly attributable. | Automatically extracted semantic entries lack source session, model, evidence, and authorization provenance. | Audit present memory before adding mutable skills or more automatic recall. |
| Reading fatigue is plausible but not yet measured. | Tool cards truncate without lossless expansion; events remain chronological; resume and child completion are not organized around decisions. | Experiment with a lossless decision view and measure comprehension alongside volume. |
| Large files are change-risk indicators, not proof of bad boundaries. | `agentic_user.rs`, `tui_repl.rs`, `embed_memory.rs`, `agent_loop.rs`, and `input_helper.rs` are the largest surfaces. | Extract only along a phase’s tested seam; do not create a size-reduction phase. |
| Several ecosystem integrations are absent. | No MCP, LSP, browser, image input, patch protocol, or `AGENTS.md` discovery exists. | Treat portable instructions as a candidate backed by cross-harness demand, but require a failing Piku scenario before implementation; defer the rest on the same rule. |

## Decision queue

The roadmap does not silently choose architecture where accepted decisions are
missing. These ADRs should be written before their implementation phases:

### ADR 0010: non-interactive authority lease

Options:

1. document full-user authority as the permanent writable-launch contract;
2. add a declarative capability lease covering tools, paths or commands,
   lifetime, denial precedence, turns, and cost;
3. require an OS sandbox for every writable non-interactive turn.

Preferred direction: option 2 as the policy contract, with containment designed
as a separate lower layer. A lease is not a sandbox and must never be described
as one.

### ADR 0011: task workspace execution boundary

Options:

1. retain prompt-routed worktrees and rename the feature accordingly;
2. scope every child file and shell operation to an explicit executor working
   directory and determine dirty state from repository state;
3. require an isolated worktree for every child.

Preferred direction: option 2, with worktree allocation optional. It fixes the
false working-directory claim without requiring Git for read-only or report-only
delegation. It does not contain absolute-path, network, process, or other shell
effects; those require a lower execution boundary or an explicitly incomplete
effect record.

### ADR 0012: trace-backed dogfood evidence boundary

The trace-backed dogfood design is already implemented and relied on by later
decisions, but has no matching ADR. Record the existing distinction among trace,
workspace, viewport, and reviewer evidence without expanding its scope.

No ADR is yet justified for lazy tools, attention modes, a stable external event
API, MCP, LSP, or mutable skills. Those remain experiments or deferred forks.

## Phase 0: restore truthful evidence and surface contracts

Goal: stop current interfaces and evaluators from claiming more than they prove.

Work:

- represent scenario predicate results as pass, fail, or inconclusive;
- treat verifier spawn, unavailable dependency, and timeout failures as harness
  outcomes rather than product failures;
- bind every authoritative goal clause to a direct predicate or mark it
  unverified;
- make writable launch turns omit agent tools until they have a task registry, or wire
  the registry deliberately;
- make `background: false` block or remove the false synchronous claim;
- document hooks as TUI-only and disabled in read-only mode;
- disable automatic semantic-memory injection into child prompts until each
  injected entry has source and policy provenance, or mark that provenance
  explicitly unknown and exclude it from verified child evidence;
- expose a per-surface capability inventory in tests and help output, including
  the implemented `/permissions` and `/hooks` handlers.

Consumers:

- `agentic_user` evaluation and its development-context handoff;
- TUI help and status;
- headless automation callers;
- later authority and lifecycle phases.

Reversibility: high. Most changes refine state classification and advertised
capabilities without changing provider or session formats.

Gate:

- deterministic tests inject predicate failure, verifier spawn failure, timeout,
  unavailable task registry, and each surface’s capability set;
- no inconclusive outcome can select a product-fix next action;
- `just check` remains green.

## Phase 1: decide authority and workspace boundaries

Goal: make non-interactive and delegated execution honest, bounded, and testable.

Prerequisites: ADRs 0010 and 0011.

Work:

- introduce an explicit capability lease for writable launch and child turns;
- guarantee child authority is no broader than the parent lease;
- evaluate configured deny rules before `Safe` classification and per-turn
  allow-all;
- require every state mutation to be promptable or covered by an explicit lease;
- keep permission decisions separate from process containment;
- pass an explicit working directory through tool dispatch rather than prompt
  text;
- determine changed repository state inside the task workspace before cleanup,
  including changes made through shell;
- mark arbitrary shell effects outside the task workspace unobserved unless a
  lower containment or mutation-observation boundary is active;
- preserve prompt guidance as explanation, not enforcement.

Consumers:

- permission policy and TUI prompting;
- launch-turn and headless execution;
- `TaskRegistry`, worktree cleanup, file tools, and shell;
- dogfood scenarios that make unattended changes.

Reversibility: medium. The lease representation and executor context will become
shared runtime contracts, so test them before exposing a stable public API.

Gate:

- deny rules dominate every later allow;
- deterministic tests prove a configured deny blocks new-file, Markdown-memory,
  and attempt writes even after per-turn allow-all;
- a child cannot widen its parent’s tool, path, command, turn, or cost bounds;
- file and shell calls operate in the explicit task directory;
- repository changes made only through shell keep the worktree for review;
- behavior outside the lease fails closed with a typed reason.

This phase does not claim a complete shell-effect inventory. If the lease permits
an unrestricted command, absolute-path writes, process changes, and network
effects remain outside the task-workspace record.

## Phase 2: make tool effects reliable

Goal: give the model, UI, and evaluator one trustworthy result contract before
optimizing how tools are selected.

Work:

- write a focused propagation design before changing persisted result shapes;
  map `ToolResult` through the runtime's current `(String, bool)` projection,
  model-visible tool result, `OutputSink`, session `ContentBlock`, trace, resume,
  and evaluator;
- bound inline output and retain a durable full artifact when truncated;
- report authoritative completion state and a recoverable failure class;
- attach a task-workspace changed-file inventory to mutating results and mark it
  incomplete when unrestricted shell effects are possible;
- detect stale reads or failed edit preconditions;
- cap retries when the failure and workspace state do not change;
- preserve the raw result behind any coalesced presentation.

Consumers:

- agent retry behavior;
- traces and scenario predicates;
- completion review and reading-fatigue experiments;
- background result verification.

Reversibility: medium. Add fields compatibly and avoid declaring a stable
external event protocol until a second client exists.

Gate:

- fault-injection tests cover completion, truncation, stale input, unchanged
  retry, timeout, and partial mutation;
- changed-file inventory agrees with repository state inside the declared task
  workspace and never implies coverage of uncontained external shell effects;
- completion state and artifact references survive session save/resume and trace
  replay without being flattened away;
- no retry ceiling converts an uncertain effect into success.

## Phase 3: finish delegation as a lifecycle

Goal: make a child result arrive once, fit within bounds, and end in an explicit
disposition.

Work:

- provide reliable completion delivery with acknowledgement and bounded payload;
- define cancellation and terminal states;
- attach authority, workspace, ownership, and evidence provenance to a task;
  child context with unknown semantic-memory provenance cannot satisfy this gate;
- add explicit accept, reject, retain, and clean dispositions for changed
  worktrees;
- require parent verification before a child result counts as completion.

Consumers:

- `TaskRegistry` and TUI interjections;
- launch turns if they gain agent tools;
- session persistence and completion review;
- completion hygiene for temporary worktrees and processes.

Reversibility: medium. Keep the lifecycle internal until another client needs it.

Gate:

- deterministic scenarios cover delivered, dropped, cancelled, failed,
  acknowledged, accepted, and rejected tasks;
- no completed child is silently lost or injected twice;
- every retained temporary artifact has an owner and reason.

## Phase 4: reduce context and reading cost without hiding evidence

Goal: replace chronology-heavy review with a lossless decision surface and test
whether it improves both delivery and ownership.

Work:

- add drill-down from truncated tool cards to durable full output;
- group completion by goal, semantic changes, files, checks, risks, and unresolved
  decisions;
- coalesce routine successful events while keeping errors, mutations, invariant
  failures, and divergence visible;
- record why compaction fired, what was masked, and where the prior artifact can
  be inspected;
- run an opt-in learning or ownership condition that asks for prediction, active
  modification, or compact explanation at a high-leverage point.

Measures:

- transcript lines and bytes, scroll distance, approval count, time to locate
  the result, and expansion rate;
- missed failures, defect detection, and correct risk identification;
- delayed explanation, modification, and debugging performance;
- total task time and accepted change, so productive friction is not confused
  with delay for its own sake.

Reversibility: high while the decision view and learning condition remain
optional overlays on durable events.

Gate:

- the compact condition reduces review cost without reducing defect detection or
  later ability to explain, modify, and debug the change;
- every collapsed item has a lossless source;
- remove any learning intervention that adds time without measurable ownership.

## Phase 5: make context discovery selective and attributable

Goal: reduce fixed prompt load without replacing it with misses, stale memory, or
untraceable influence.

Work:

- first record a reproducible Piku workflow that fails because portable
  repository instructions exist only in `AGENTS.md`; add discovery only if that
  scenario and its boundary test justify a durable precedence contract;
- attach source session, model, evidence, author class, and authorization to
  automatically extracted semantic memory;
- record which retrieved entries entered a turn and whether they affected the
  accepted result;
- prototype a small always-hot tool set with on-demand schema disclosure and a
  deterministic full-catalog fallback.

Consumers:

- system prompt construction and child context;
- semantic memory extraction and retrieval;
- provider token budgets and tool selection;
- repository portability across harnesses.

Reversibility: medium. Instruction precedence and memory provenance become
durable contracts; lazy disclosure remains experimental until measurements pass.

Gate:

- if the portable-instructions scenario passes its entry gate, tests cover
  discovery order, nesting, conflict, and size limits;
- every generated memory injection is traceable to source and policy;
- retrieval improves an acceptance measure without increasing stale-instruction
  failures;
- lazy disclosure saves prompt tokens without increasing misses, retries, or
  failed acceptance.

## Deferred forks

- A stable thread, turn, and item event API waits for a second real client.
- MCP waits for a workflow that needs portable remote discovery or service-owned
  authorization and cannot be served by an existing CLI or direct API.
- LSP, browser, image input, and patch protocols wait for concrete failing Piku
  scenarios.
- Hosted and multi-user execution require a separate threat model for identity,
  isolation, secrets, network, tenancy, and operations.
- Parallel writers wait for authority inheritance, ownership, and merge
  disposition to pass deterministic lifecycle tests.
- Mutable self-authored skills wait for versioned provenance, authorization,
  evaluation, and rollback.
- Full workspace rewind waits until arbitrary shell effects participate in a
  mutation ledger.

## Roadmap review triggers

Revisit ordering when:

- a second client needs shared runtime lifecycle state;
- three comparable dogfood pairs contradict a phase premise;
- a concrete workflow cannot be expressed through the current local tool set;
- a compact presentation lowers defect detection or delayed comprehension;
- a repeated incident crosses the permission, task, context, or memory boundary;
- a phase would require changing an accepted ADR rather than appending a new
  decision or superseding it explicitly.

## First checkpoint

Complete phase 0, then write ADRs 0010 and 0011 from observed tests and consumers.
Do not begin lazy tools, more concurrency, or a presentation redesign while the
current surfaces and evaluator still misclassify their own capabilities and
evidence.
