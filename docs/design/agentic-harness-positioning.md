# Design: Piku's agentic harness position

status: proposed

evidence: `docs/agentic-harness-landscape.md`

## Problem

Piku now implements enough of an agentic harness that feature requests from
larger systems can pull it in several incompatible directions: a broad local
assistant, a hosted service, an extensible platform, or a small instrumented
coding loop. Chasing all four would grow the authority, UI, integration, and
operations surfaces faster than the current quality loop can validate them.

The research question is not which competitor has the longest feature list. It
is which boundary gives Piku a coherent reason to exist and an evidence path for
deciding what enters that boundary.

## Context

Piku already owns a provider-neutral stream, tools, sessions, hooks, memory,
background agents, a terminal UI, deterministic runtime tests, PTY tests, and an
opt-in live playground. Recent work makes a deterministic check authoritative
over judge prose only for the exact property the check directly asserts.

Across Codex, Claude Code, OpenCode, Hermes, Pi, Aider, Cline, Goose, Roo Code,
SWE-agent, and newer systems, users consistently value reliable execution,
explicit authority, recoverable context, visible delegation, portable
instructions, and control over attention and cost.
The recurring failures come from weak containment claims, excessive tool/context
load, brittle terminal behavior, opaque compaction, and background work without a
complete lifecycle.

## Non-goals

- Do not build a hosted control plane as an incremental extension of the local
  binary. Identity, isolation, secrets, networking, and operations require a
  separate decision.
- Do not use competitor feature parity as an acceptance criterion.
- Do not add MCP, LSP, browser, image, or desktop surfaces without a concrete
  Piku scenario that existing local tools cannot satisfy.
- Do not enable parallel writers before authority inheritance, ownership,
  cancellation, result disposition, and verification are explicit.
- Do not claim arbitrary workspace rewind while shell side effects are outside a
  complete mutation ledger.
- Do not promote mutable self-authored skills without versioned provenance and a
  rollback path.
- Do not minimize every pause, explanation, or review step. Some friction is how
  a user forms and tests the mental model required to own the result.
- Do not force a learning-oriented workflow on routine delivery work, or an
  autonomy-oriented workflow on users who need to understand and maintain the
  code themselves.

## Options considered

### Broad local feature parity

Add the integrations, modes, plugins, model routing, and UI surfaces that users
request from larger harnesses. This may make comparisons easier, but it imports
their prompt load, configuration sprawl, permission complexity, and maintenance
cost before Piku has users or measurements proving those surfaces matter.

Rejected as the default direction. Individual capabilities remain eligible when
they pass a scenario and boundary gate.

### Hosted or multi-user platform

Turn the runtime into a service with remote execution, GitHub integration,
automations, and multiple clients. Codex and OpenCode demonstrate the value of a
stable core protocol, but the deployment and authorization problem is a distinct
product.

Deferred. A real second client or remote deployment request should trigger a
separate design for the event protocol and trust boundaries.

### Evidence-first local harness

Keep the product local and provider-neutral. Invest in authority boundaries,
tool reliability, context visibility, portable repo instructions, deterministic
tests, and trace-backed dogfood. Add capabilities only after a failing scenario
shows that the current narrow waist is insufficient.

Chosen. It compounds Piku's existing strengths and keeps architectural changes
reversible while usage evidence is limited.

## Chosen approach

Piku will optimize for a small, observable local agent loop with a high-quality
evaluation path. “Small” describes the public and authority surface, not the
absence of safety or diagnostic mechanisms.

The product objective is verified change per unit of scarce human attention.
That requires reducing process narration and repeated scanning while preserving
the prediction, explanation, review, and debugging work that creates ownership.
It is intentionally different from maximizing autonomous task completion.

The runtime's narrow waist is:

```text
provider request/events
  -> session and turn state
  -> permission lease and hooks
  -> tool or child-agent lifecycle
  -> trace and workspace effects
  -> completion and verification
```

The TUI, headless CLI, and any future client should share these semantics. A new
surface is not a reason to fork the loop.

The current implementation does not yet satisfy that narrow waist uniformly.
Writable launch turns advertise background-agent tools without the task registry
used by the TUI. Child worktree selection is conveyed through prompt text rather
than an executor working directory, `background: false` does not block, and
completion delivery is a bounded best-effort TUI channel without
acknowledgement. These are alignment defects, not evidence for adding a broader
platform. Memory, attempt-tree, and `tool_search` operations use shared runtime
paths and remain available during launch turns.

Capability proposals enter through this sequence:

1. Name the user scenario and executable acceptance check.
2. Show why existing tools or composition cannot satisfy it.
3. Define authority, data, failure, and lifecycle boundaries.
4. Implement the smallest reversible version.
5. Run deterministic coverage and one bounded live control/sample comparison.
6. Promote only repeatable gains; remove or defer the rest.

## Immediate design sequence

### Authority before breadth

Define a non-interactive capability lease for writable launch and child-agent
execution. The lease must express tool class, path or command scope, denial
precedence, lifetime, and budget. Process containment remains a separate layer.

### Dependable tool effects

Preserve result metadata through the model-visible result, session, trace, UI,
resume, and evaluator before optimizing schema discovery. A task-workspace file
inventory must not imply observation of absolute-path, process, or network
effects from unrestricted shell commands.

### Context provenance before delegation proof

Add source session, model, evidence, and authorization provenance to semantic
memory injected into child prompts, or disable that injection until it can be
attributed. A child lifecycle cannot be called verified while part of its input
has unknown origin or policy.

### Complete delegation lifecycle

Build on the TUI's existing completion injection by specifying reliable bounded
delivery, acknowledgement, child identity, inherited authority, cancellation,
file ownership, enforced task working directory, worktree disposition, and
parent verification before increasing write concurrency.

### Portable project context

First record a reproducible Piku workflow that fails because portable
instructions exist only in `AGENTS.md`. Add discovery only if that scenario
justifies a durable precedence contract. If it does, treat `AGENTS.md` as the
portable base convention and `PIKU.md` as a documented Piku-specific overlay;
record provenance and nesting order in the prompt.

### Inspectable context changes

Keep deterministic observation masking. Add a bounded record of why reduction
fired, which evidence was transformed, and how to inspect the prior form.

### Real progressive disclosure

After tool-result and delegation contracts are reliable, keep a small always-hot
tool set and disclose other schemas on demand. Preserve a deterministic fallback
so retrieval misses do not make a capability disappear. Measure prompt reduction
and task misses together.

### Lossless attention management

Default output should be a decision view rather than a chronological replay:
coalesce routine successful activity, preserve visible errors and mutations,
stabilize scrollback, and end with a semantic review packet covering the goal,
changed files, tests, invariants, risks, and unresolved decisions. Every collapsed
item must remain expandable or point to a complete artifact.

Add an opt-in learning and ownership mode. Before substantial generation it can
ask for expected behavior or an invariant; after generation it can require active
modification or a compact explanation. The mode succeeds only if later
explanation, modification, or debugging improves. Time spent is not evidence of
learning by itself.

## Tradeoffs

- Piku will lag broad harnesses in integrations and presentation surfaces.
- Some apparently small features will wait for a scenario or boundary design.
- Deterministic fixtures and ledgers add test code that does not directly expand
  the product surface.
- Provider neutrality may prevent provider-specific optimizations from becoming
  default behavior.
- A local-only product cannot offer the convenience of hosted background jobs or
  phone-to-pull-request workflows.
- A decision-focused default may conceal a useful clue unless expansion and
  artifact retention are reliable.
- A learning mode intentionally trades immediate throughput for a measurable
  chance of better comprehension and ownership.

In exchange, the product keeps a comprehensible trust boundary, lower maintenance
cost, clearer failure attribution, and a stronger basis for removing changes that
do not improve real tasks.

## Decision gates

- If a second client is being implemented, design a typed thread/turn/item event
  protocol before adding client-specific callbacks.
- If three comparable dogfood runs show a capability materially improves
  acceptance without unacceptable cost or authority expansion, reconsider its
  deferral.
- If users require remote or multi-user operation, stop and design identity,
  tenancy, isolation, secrets, egress, and audit as a separate system.
- If tool-schema selection misses cause more retries or failed acceptance than
  the prompt reduction saves, keep the full catalog until retrieval improves.
- If isolated sequential handoffs become the measured bottleneck after lifecycle
  work is complete, reconsider concurrent writers.
- If a compact presentation lowers defect detection or delayed comprehension,
  restore the omitted evidence or change the grouping.
- If learning-mode prompts do not improve later explanation, modification, or
  debugging, remove them rather than preserving ceremonial friction.

## Open questions

- What is the smallest non-interactive lease that is useful for `piku -p` without
  making configuration itself a programming language?
- Should child agents inherit a frozen parent lease or receive a strictly smaller
  lease declared at spawn time?
- Which tool definitions are always hot, and what deterministic fallback makes a
  retrieval miss recoverable?
- Does portable instruction discovery use nearest-file wins, hierarchical merge,
  or explicit imports when `AGENTS.md` and `PIKU.md` overlap?
- What evidence artifact is sufficient for inspecting compaction without storing
  costly full snapshots on every turn?
- Which activity can be coalesced without lowering error detection, and which
  mutations or invariant failures must always remain visible?
- Should learning intent be selected per session, per task, or inferred only
  after an explicit user signal?
