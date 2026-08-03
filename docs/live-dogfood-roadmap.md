# Roadmap: agentic quality loop

Status: active operational evidence loop

Created: 2026-06-20

Last reviewed: 2026-08-03

Scope: deterministic agent coverage, live-model dogfood, terminal playground
evidence, and promotion of findings into product changes. Broader product
sequencing remains in the intentionally owner-local design corpus.

Grounded in the tracked deterministic-loop, live-matrix, dogfood-ledger,
failure-promotion, repository-artifact, and trace-backed focused designs under
`docs/`, plus `docs/agentic-harness-landscape.md`. The intentionally owner-local
decision ledger is not required to interpret this tracked roadmap.

## Current state

Implemented:

- Pull-request CI exercises the runtime agent loop with scripted providers.
- Deterministic tests cover multi-iteration reads, edits, verification, retries,
  permissions, traces, sessions, hooks, memory, and background-agent behavior.
- PTY smoke tests run separately from the normal workspace test stage.
- Live LLM and dogfood suites are explicitly ignored and opt-in.
- A manual GitHub Actions live matrix exists behind owner and environment gates.
- Live helpers can append provider, model, result, trace, token, duration, and
  failure-class records to JSONL ledgers.
- Repository pull-request and issue artifacts can seed local read-only dogfood.
- The agentic-user playground drives the real binary through a PTY, records
  viewport and transcript evidence, and verifies the resulting workspace.
- Every regular persona has an executable scenario contract calibrated against
  the pristine fixture.
- Model review failures are named, invalid reviews contribute no product
  observations, and findings expire unless reproduced against the current Piku
  revision.
- Runs record model attribution, elapsed-time components, review spend, and a
  deterministic development-context handoff.

Fresh remote check on 2026-08-03: the `live-llm.yml` workflow has no recorded
runs. Local corpus-backed and terminal-playground ledgers exist, but they do not
replace remote workflow verification.

## Evidence hierarchy

When signals conflict, use this order:

1. Executable workspace acceptance for the exact property its predicate checks.
2. Deterministic trace and process facts.
3. PTY viewport and session transcript evidence.
4. Primary LLM review tied to existing turn identifiers.
5. Recursive review of the primary review.
6. Unreproduced historical allegations.

A lower tier may explain a higher-tier failure but does not overturn it by
opinion. A plausible response with a failing exact-property predicate is a failed
run. A verifier that cannot start, times out, or lacks a predicate for part of the
goal is inconclusive for that property, not proof of product failure. A model
claim with no reproducible evidence is a hypothesis.

## Completed phases

### Deterministic loop coverage

The normal gate no longer relies on provider keys to prove the core loop. Live
scenarios that reveal a deterministic runtime property should still be distilled
into `crates/piku-runtime/tests/e2e.rs` or a focused parser/tool test.

### Trace-backed dogfood

Tool assertions prefer trace events; final-state assertions inspect the workspace;
rendering assertions inspect terminal output. This prevents UI wording from
silently changing what the harness believes happened.

### Local live ledger

Opt-in suites can write comparable JSONL rows. The ledger is evidence input, not
a score table. Provider outages, invalid model responses, harness defects, and
product failures remain distinct classes.

### Repository-artifact corpus

`just github-corpus`, `just github-prompt`, and `just github-dogfood` use repository
history as local evaluation input without making GitHub access a runtime feature.
The original “run the corpus once” next action is complete.

### Grounded terminal playground

The playground now combines real PTY interaction, session transcripts, workspace
checks, a primary review, a bounded recursive review, spend accounting, and an
append-only improvement handoff. Recent work corrected blank viewport evidence,
expired stale findings by revision, separated review latency from Piku latency,
and made task acceptance authoritative.

## Active operational priorities

### 1. Preserve pass, fail, and inconclusive evidence

The evaluator currently maps any scenario-verifier `passed: false`, including
spawn and timeout failures, to the same engineering next action as a failed
product assertion. Some scenario goal clauses also lack a direct predicate.

Next shape:

- Represent product failure, harness failure, unavailable evidence, and timeout
  as distinct outcomes.
- Bind each scenario goal clause to a predicate or mark it explicitly unverified.
- Apply deterministic precedence only to the exact asserted property.

Gate:

- Injected verifier spawn and timeout failures cannot produce a product-failure
  disposition, while an actual predicate failure still does.

### 2. Finish evidence-addressed review

Current review validation requires a nonempty, in-range run-level
`evidence_turns` list. Review bodies and recursive observations are still opaque
strings, so individual allegations are not bound to stable evidence addresses.

Next shape:

- Give each review claim a stable ID and disposition.
- Require every claim to cite one or more turn, trace, transcript, or workspace
  evidence IDs.
- Reject unknown IDs and keep deterministic findings independently authoritative.
- Let the recursive reviewer retract or mark a claim inconclusive only through
  those IDs.
- Build the final handoff from dispositions rather than free-form review text.

Gate:

- A fabricated turn, unsupported claim count, or missing evidence ID produces an
  invalid review record and cannot change the engineering next action.

### 3. Make mutation authority an executable invariant

The current permission path allows `Safe` calls before configuration rules and
lets a prior per-turn allow-all precede deny rules. New unprotected files,
Markdown-memory writes, and attempt writes are state mutations currently
classified `Safe`.

Next shape:

- Evaluate configured deny rules before `Safe` classification and per-turn
  allow-all.
- Require each state mutation to be promptable or covered by an explicit,
  inspectable capability lease.
- Record the effective authority decision as evidence without mistaking policy
  for process containment.

Gate:

- Deterministic scenarios prove a configured deny blocks new-file,
  Markdown-memory, and attempt writes, including after per-turn allow-all.
- Writable launch and child turns cannot silently widen the declared lease.

### 4. Make tool effects dependable evidence

Before optimizing tool discovery, define the result contract that both the agent
and evaluator can trust:

- bounded inline output with a durable full artifact when truncated;
- authoritative completion state and recoverable failure class;
- changed-file inventory for the declared task workspace, including repository
  changes made through shell, with an explicit incomplete marker when
  unrestricted external shell effects are possible;
- stale-read detection or explicit precondition failure;
- a retry ceiling that prevents unchanged failures from consuming turns.

Gate:

- Deterministic fault-injection tests distinguish completed, failed,
  inconclusive, truncated, stale, and retry-exhausted outcomes. The changed-file
  inventory agrees with repository state inside the task workspace and never
  implies coverage of uncontained external effects.

### 5. Prepare a genuinely lazy tool-disclosure experiment

`tool_search` currently searches schemas already sent to the model. Replace this
with a small always-hot safety/orchestration set plus on-demand schema disclosure
and a deterministic full-catalog fallback only after the tool-result and
delegation contracts below are reliable. This priority defines the experiment;
it does not move implementation ahead of those prerequisites.

Gate:

- Measure prompt tokens, tool-selection misses, added round trips, and task
  acceptance against the current full-catalog control.
- Do not ship if context savings merely move cost into retries or missed tools.

### 6. Complete background-agent lifecycle

The TUI already injects completion through a bounded notification channel. Before
adding more concurrent writers, make delivery reliable and bounded, then define
acknowledgement, cancellation, inherited authority, file ownership,
changed-worktree disposition, and parent verification. The launch-turn surface must
either provide a task registry or stop advertising agent tools. `background:
false` must block or cease claiming that it does. Semantic memory injected into
a child must first gain source and policy provenance or be disabled for that
verified path.

Gate:

- A deterministic scenario can spawn work in an executor-scoped directory,
  observe and acknowledge completion, detect file changes made through both file
  tools and shell, verify or reject the result, and leave no unexplained
  temporary branch or process.

### 7. Test the need for portable repository instructions

Piku currently loads `PIKU.md`. Before adding native `AGENTS.md` discovery,
record a reproducible Piku workflow that fails because portable instructions
exist only in `AGENTS.md`. If that scenario passes its entry gate, use
`AGENTS.md` as the portable base convention and document `PIKU.md` as a
Piku-specific overlay or override.

Gate:

- The scenario proves a concrete compatibility failure. If implementation then
  proceeds, tests cover discovery order, nesting, size bounds, conflicts, and
  the exact prompt sections visible to parent and child agents.

### 8. Make context reduction inspectable

Record why compaction fired, what content classes contributed tokens, which
observations were masked, and where the pre-rewrite artifact can be inspected.
Keep automatic masking deterministic.

Gate:

- A resumed run can explain what changed without storing full duplicate snapshots
  in every turn or making the memory footprint dominate the session.

### 9. Measure human attention without deleting useful reasoning

The current TUI truncates successful tool cards but offers no lossless expansion,
adds one row for every tool event, resumes from a fixed recent tail, and has no
first-class semantic diff or completion-review surface. A child completion can
also re-enter the parent as a large block. These choices can produce both hidden
evidence and reading fatigue.

Next shape:

- Add a stable decision view that coalesces routine successful activity while
  keeping errors, mutations, invariant failures, and divergences visible.
- Retain an expandable or durable full artifact behind every collapsed item.
- Summarize completion by goal, semantic change group, files, tests, risks, and
  unresolved decisions rather than transcript chronology.
- Add an opt-in learning and ownership experiment using prediction, active
  modification, or compact explanation. Do not equate added delay with learning.

Gate:

- Compare transcript lines and bytes, scroll distance, time to locate the result,
  expansion rate, and approval prompts against missed failures and defect
  detection.
- For learning-oriented runs, measure delayed explanation, modification, and
  debugging, not only immediate task completion.
- Reject a concise presentation that saves reading by hiding evidence users need
  to understand or verify the result.

### 10. Audit semantic-memory provenance and usefulness

Automatically extracted semantic memory currently has no source session, model,
evidence, or authorization provenance. Before expanding automatic recall:

- attach stable source and extraction provenance to every generated entry;
- distinguish user-authored, project-authored, and model-extracted material;
- record which retrieved entries entered a turn and whether the result used them;
- test stale, conflicting, and unauthorized entries.

Gate:

- A reviewer can trace every injected entry to its source and policy, and a
  dogfood comparison shows retrieval improves an acceptance measure without
  increasing stale-instruction failures.

## Deferred until evidence changes

- Hosted execution, web UI, GitHub application, and multi-user control plane.
- Scheduled provider spend or live-model pull-request gating.
- User-facing random or “top model” selection.
- MCP, LSP, browser, or image capabilities added only for parity.
- Full workspace rewind while arbitrary shell side effects are not recorded.
- Parallel writes without explicit ownership and merge disposition.
- Mutable self-authored skills without versioned provenance and rollback.

## Review triggers

Revisit this roadmap when any of the following occurs:

- the first protected remote live-matrix run completes;
- three comparable control/sample pairs exist for one product hypothesis;
- a second Piku client needs the runtime lifecycle state;
- a real user workflow requires a remote integration rather than a local CLI;
- a repeated failure crosses the permission, context, or subagent boundaries;
- a deferred capability becomes necessary for a concrete acceptance scenario.
- a compact-output experiment changes defect detection or delayed comprehension.

## Next action

Separate product failure from verifier failure and bind every authoritative
scenario claim to a direct predicate. The strategic roadmap then begins with the
non-interactive authority and execution-boundary decisions.
