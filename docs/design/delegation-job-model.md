# Delegation as job control

Status: research and design proposal

Reviewed: 2026-10-23

Scope: align Piku's subagent and future background-shell lifecycle with
classical shell job control, while keeping the places where an agent job is not
a process explicit.

This document is input to the task-lifecycle phase of
[the harness roadmap](agentic-harness-roadmap.md). It does not authorize
implementation before the authority and tool-result phases.

## Why the analogy is worth taking seriously

Operators already know job control: start work in the background, list it,
wait for it, stop it, and know where its output went. Piku currently exposes
`spawn_agent`, `agent_status`, and `agent_join` with an opaque task id, no
cancellation, and a best-effort notice channel. That is a job model without the
job vocabulary, which is why it feels unfinished even when it works.

The analogy is also a trap if taken literally. A shell job is a process with a
trustworthy exit status and implicit authority. An agent job is a model-driven
worker with a claim for a result, an explicit authority lease, and a cost. The
design has to say which parts transfer.

## Classical semantics

Sources: [POSIX shell command language](https://pubs.opengroup.org/onlinepubs/9799919799/utilities/V3_chap02.html),
[POSIX wait](https://pubs.opengroup.org/onlinepubs/9799919799/utilities/wait.html),
[Bash job control](https://www.gnu.org/software/bash/manual/html_node/Job-Control.html).

| Primitive | Classical meaning | Property worth matching | Caveat |
| --- | --- | --- | --- |
| `cmd &` | asynchronous list; returns a job id immediately | start never blocks on completion | |
| `$!` | pid of the last background job | a stable handle exists before completion | pid of the last element of an async list, not a group leader |
| `jobs` | list jobs with state | state is listable at any time | |
| `%n`, `%%`, `%-` | job specs | the operator can address a job without copying an opaque id | |
| `wait` | block until a job terminates; returns its status | completion is observable and typed | POSIX waits for termination; Bash may return on a stop |
| `kill %n` | signal a job | running work can be stopped by the operator | |
| `Ctrl-Z` / `SIGTSTP` | stop a foreground job | a job can be suspended rather than failed | |
| `fg` / `bg` | resume in foreground or background | suspension is recoverable | |
| `SIGTTIN` | a background job reading a controlling terminal stops | a background job needing input stops and becomes visible | unreachable for an agent without a controlling terminal; harnesses engineer it away |
| `( ... )` | subshell: isolated cwd and variables | scoped execution context | |
| `disown`, `nohup` | remove a job from the shell's table, or ignore SIGHUP | survival is explicit, never implicit | neither re-parents nor guarantees survival |
| exit status | encodes exit or signal (128+n); stop state is shell-optional | one typed terminal outcome | POSIX `wait` does not return on a stop |

Two of these matter most for agents. A shell job that needs the terminal does
not fail; it stops, but that mechanism requires a controlling terminal an agent
does not have, so the equivalent has to be built rather than inherited. A shell
job also does not outlive its shell unless the operator asks, while Piku's
in-process children die with the binary without a stated policy. Piku currently
denies a child that needs a new permission instead of stopping it.

## Shipped prior art

| Harness | Model | Evidence |
| --- | --- | --- |
| OpenCode | one `Job` service covering background shell and subagent jobs: `get`, `start`, `wait`, `block`, `background`, `cancel`, `completeBackground`; statuses `running/completed/error/cancelled`; a notification id and a `recovery` record per job | `packages/core/src/job.ts` |
| Codex | agent status derived from turn events: `Running`, `Completed(message)`, `Errored(message)`, `Interrupted`, `Shutdown`; final-state predicate; status subscription; explicit cancellation through runtime shutdown | `codex-rs/core/src/agent/status.rs`, `codex-rs/core/src/agent/control.rs` |
| Claude Code | background tasks with `/tasks`, `TaskStop`, per-task output files, and explicit time limits; a foreground command that times out moves to the background with a stated notice | [tools reference](https://code.claude.com/docs/en/tools-reference) |
| Crush | background shell jobs with `job_output`, automatic backgrounding after 60 seconds, and structured start/end/cwd/shell metadata | `internal/agent/tools/bash.go` |
| Piku | `TaskRegistry` with depth and turn budgets, `Running/Done/Failed`, waiters, bounded TUI notices, and persisted parent-child evidence | `crates/piku-runtime/src/task.rs` |

The convergence is notable: every external implementation surveyed has a
**typed terminal status, a cancellation path, and a durable
notification/artifact**. Piku is the exception in the same table: it has none of
the three beyond two terminal states and an in-memory notice. None of these
harnesses treat a child report as a verified exit status.

## Alignment matrix

| Shell | Piku analogue | Holds | Diverges | Required work |
| --- | --- | --- | --- | --- |
| `cmd &` | `spawn_agent` | start returns immediately; the task id is already in the tool result and in the durable `ChildRunRef` | no short operator-facing handle; ids are opaque | add a display handle, not a new identity |
| `jobs` | `agent_status`, `/agents` | listable | no stopped/cancelled states | state machine below |
| `wait` | `agent_join` | returns the entry status and evidence paths | a `Failed` entry is delivered as a non-error tool result | make failure an error result and return a typed outcome |
| `kill` | none | n/a | no cancellation at all | cooperative cancel plus a terminal `cancelled` state |
| `Ctrl-Z` | none | n/a | a background child that needs approval is denied | surface as `stopped_needs_input` with an operator path |
| `fg` | none | n/a | no way to attend to a stopped child | attach/inspect and resolve the pending decision |
| `( )` | `isolation: worktree` | scoped intent | cwd is prompt-routed, not executor state | Phase 1 executor directory |
| `disown` | none | n/a | children die with the binary, unstated | document process-lifetime semantics; do not promise survival |
| exit status | `Done`/`Failed` string | terminal state exists | state is mutable after terminal; no reason class | monotonic terminal transitions and typed outcome |
| output file | preview plus `agent_join` | output retrievable | no durable artifact | Phase 2 artifact reference |

## What Piku already has

- **Foreground turn cancellation** exists: `CancelFlag`, a `TurnCancelled`
  event, and Ctrl-C handling. Already-running tools settle before cancellation
  takes effect, which is a stated limit rather than a bug.
- **Subagent cancellation was never attempted.** The child path passes `None`
  and says so in a comment. No abandoned or reverted job-control
  implementation appears in history.
- **Dispositions are descriptive, not enforced.** `piku conclude` records a
  run-scoped `UserDisposition`; the review-contract design explicitly chose
  human review over machine-enforced acceptance and asked for no new task bus.
- **Executor cwd was deliberately rejected.** The worktree path changes prompt
  text rather than calling `set_current_dir`, because that mutates global
  process state and races with the parent's tool calls. Any cwd carry-over must
  pass through tool dispatch instead.
- **Shared shell/subagent registry is an open question**, not a settled design.

## Current implementation constraints

These are verified in the current tree and change what the design can promise:

- **Terminal predicate.** `wait_for` treats any status other than `Running` as
  terminal (`task.rs:814`), so adding `stopping` or `stopped_needs_input`
  without an explicit `is_terminal()` predicate would make `agent_join` return
  those states as finished work.
- **No cancel handle.** `TaskEntry` carries no cancellation channel, the child
  task handle is discarded (`agent_loop.rs:2156`), and children are explicitly
  passed `None` for cancellation (`agent_loop.rs:2297`, `:2318`). A forced
  cancel needs a supervisor or a stored abort handle.
- **Failure is not an error result.** A `Failed` entry is delivered by
  `agent_join` with `is_error = false` (`agent_loop.rs:2519-2528`), so the
  parent model can read failure as success.
- **No durability.** Tasks are in memory, ids embed the pid, and a resumed
  session intentionally does not reconstruct live task state. The job table
  cannot be described as surviving restart without new persistence work.
- **Dispositions already exist elsewhere.** `piku conclude` records
  `Accepted`, `NeedsWork`, or `Abandoned` as a run-scoped `UserDisposition`.
  Reuse that vocabulary and bind it to the task, rather than adding a second
  disposition system.
- **Worktree cleanup is Drop-only.** It cannot run after process termination,
  so cancellation alone cannot promise an explained worktree; an orphan reaper
  is a separate requirement.
- **One deny shape.** Child permission checks return a single `Deny` reason for
  both a configured rule and "needs interactive approval", so a stop state
  needs a grantable predicate before it can distinguish the two.

## Proposed state machine

```text
          start
            |
         running --(stop request)--> stopping --> cancelled
            |                            |
            |                            +-------> failed (stop timed out)
            |
            +--(needs new authority)--> stopped_needs_input --(operator)--> running
            |
            +--(turn/budget end)-----> completed | failed | budget_limited

terminal: completed | failed | cancelled | budget_limited
disposition (separate, human or parent): accepted | rejected | retained
```

Rules:

- terminal status is immutable; a late or duplicate completion is recorded as
  evidence and cannot overwrite the first terminal transition. This changes
  shipped behavior and its tests, so it needs an explicit migration;
- `is_terminal()` replaces every `status != Running` check before any new state
  lands;
- every terminal transition emits exactly one notification, with a stable
  notification id, and the notification is not marked consumed until the
  parent admits it;
- cancellation is cooperative with a deadline, then forced through a stored
  abort handle or a supervising task; the reason is recorded;
- `stopped_needs_input` is a real state only when a grantable predicate says the
  request could be approved; a configured deny stays denied and never produces
  a prompt loop;
- disposition is not completion, and it reuses the existing `UserDisposition`
  vocabulary. A completed child whose worktree is unverified is `completed`
  with disposition `retained`.

Piku currently allows `complete` and `fail` to run more than once, and its
existing tests exercise that path. Under a job model that is a defect: the first
terminal transition must win.

## Command surface

| Operator action | Tool or slash command | Notes |
| --- | --- | --- |
| list | `agent_status` / `/agents` | show handle, state, age, turns, workspace |
| wait | `agent_join` | returns typed outcome and evidence refs |
| cancel | `agent_cancel` / `/cancel %1` | cooperative, bounded, records reason |
| attend | `/agents %1` | inspect the child session and its artifact |
| resolve | `/accept %1`, `/reject %1`, `/retain %1` | disposition only, never implies verification |
| logs | `agent_output` or artifact path | full output behind the preview |

Handles should be short and stable within a session (`%1`, `%2`) with the
opaque id retained for durability. Handles are display sugar, not identity.

## Divergences that must stay explicit

- **No byte pipes.** Agent output is a report, not a stream. Do not promise
  `a | b` composition between agents until a typed result contract exists.
- **A report is not an exit status.** A `completed` child means the child
  stopped, not that the workspace changed as requested. Parent verification
  remains the acceptance gate.
- **Authority is leased, not inherited.** A child can narrow but never widen
  its parent's lease; a background child needing more stops rather than
  silently acquiring it.
- **Cost is not free.** Job control needs budget visibility (turns, tokens,
  wall time) because `kill` has to reclaim spend as well as compute.
- **No implicit survival.** In-process children die with the binary. `disown`
  and `nohup` semantics would require a durable worker and a separate decision.
- **No parallel writes.** Job control does not relax the ownership and merge
  disposition rules.

## Gates

- deterministic scenarios cover start, list, wait, cancel at each state,
  cancellation deadline, duplicate terminal transition, and notification
  admission once;
- `wait_for` and `agent_join` honor an explicit terminal predicate, and a
  stopped or stopping job is never delivered as finished;
- a failed child reaches the parent as an error result with a typed status;
- a child that requests authority outside its lease enters `stopped_needs_input`
  and can be resumed or rejected by the operator, while a configured deny stays
  denied;
- a cancelled child leaves no running process, and either the worktree is
  explained or an orphan reaper is shown to recover it;
- a terminal outcome is persisted with one disposition, and a resumed session
  reports honestly that live jobs were not reconstructed;
- `completed` is not reported as accepted without parent verification.

## Open questions

- **Open**: does `stopped_needs_input` suspend the child's turn or end it with a
  resumable checkpoint? Suspension is closer to job control but requires
  serializable model state.
- **Open**: are handles session-scoped or durable across resume? A resumed
  session that renumbers `%1` would break operator muscle memory.
- **Open**: should background shell jobs and subagent jobs share one registry
  and one `/jobs` surface, as OpenCode does, or stay separate until the shell
  result contract lands?
- **Open**: what is the retention bound for completed job output, and does it
  follow the Phase 2 artifact policy?

## Review trigger

Revisit when Phase 3 begins, when a child cancellation request appears in a real
workflow, or when a background child is denied a permission it could have been
granted interactively.
