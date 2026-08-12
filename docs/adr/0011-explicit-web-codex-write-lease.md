# ADR 0011: Require an explicit web Codex write lease

---
status: proposed
date: 2026-08-09
verified: 2026-08-12
extends:
  - 0009
governs:
  - crates/piku/src/web.rs
  - crates/piku/src/web/codex.rs
  - crates/piku/web-ui/app.js
  - crates/piku/web-ui/e2e/**
  - crates/piku-runtime/src/run_record.rs
why: Web Codex is intentionally read-only, while the operator journey now requires repository mutation without making generic chat or a human PTY an implicit authority grant.
rejected:
  - Inferring mutation intent from prose would let ordinary chat silently acquire write authority.
  - Treating every native Codex approval request as a browser prompt would add an asynchronous approval protocol before one bounded turn lease has been proven useful.
  - Routing agent commands through the human PTY would conflate ambient operator authority with model execution and destroy typed effect provenance.
confidence: medium
review_trigger: The Codex workspace-write sandbox cannot enforce the selected root with network disabled, a useful coding turn requires native elevation, or per-turn approval produces unacceptable authority breadth.
---

## Context

The implemented web Codex adapter starts and resumes threads with a read-only
sandbox and `approvalPolicy: never`. It rejects server-initiated interactive
requests. The notebook exposes ordinary chat only. The runtime can already
record tool starts, permission decisions, tool completion, file and shell
effects, verification, cancellation, and failure, but the web Codex adapter
does not yet project native Codex activity into that record.

The browser PTY is intentionally a human-started ambient shell. It is neither a
model tool nor an approval surface. Reusing it for agent execution would make
terminal possession look like authorization and would leave command effects
outside the durable run contract.

The installed Codex app-server schema describes workspace-write policy, a
working directory, network configuration, and per-turn overrides. Schema
availability does not prove enforcement on every supported host or across
start and resume. A version-pinned capability probe must establish those
properties before Piku exposes the write mode. Codex may also request native
approval for operations outside its active policy. The first slice does not
grant those elevations.

## Decision

A write-capable web Codex turn requires an explicit, revocable Piku lease bound
by a single-use nonce and request digest to one canonical workspace root,
executor, thread, turn, immutable prompt, start deadline, lifetime, working
directory, environment hash, network posture, and tool profile. Generic chat
remains read-only, and prose never selects the write-capable mode. Reload,
rerun, edited input, thread fork, expiry, cancellation, or any bound-field
change invalidates the lease.

After operator approval, Piku starts the turn with Codex workspace-write
containment rooted at the canonical workspace, network disabled, and native
`approvalPolicy: never`. The lease authorizes only operations the active Codex
sandbox permits. Any server-initiated request for broader approval is denied
and recorded; the first slice has no automatic or interactive elevation path.
Cancellation first revokes the lease for future activity, then interrupts the
turn and waits for acknowledged interruption or confirmed child-process exit.
Only after that quiescence boundary may Piku record a terminal cancellation
state and inventory partial effects. Cancellation does not erase effects
already produced.

Piku records the lease as a distinct durable authority event rather than
fabricating a tool-scoped permission decision. Native item events then project
through the existing run vocabulary for tool starts, decisions, results,
effects, verification, cancellation, and failure. Native item events are
causally attributed to Codex. A before/after workspace inventory is supporting
evidence only: changes without a matching native item are labeled
`unattributed`, because a human PTY or another process may write concurrently.
Piku labels unrestricted or external effects unobserved. A
permission lease is not described as an OS sandbox; the Codex containment mode
and its tested properties are named separately.

## Consequences

The first write journey needs one approval action and has a small protocol
surface. Operations that require privilege beyond Codex workspace-write fail
rather than opening a hidden elevation channel. A later per-command or
per-file escalation workflow requires a new decision and asynchronous browser
protocol.

Before enabling the UI, a version-pinned integration probe must demonstrate
the claimed writable-root and network behavior for thread start, resume, and a
turn on each supported host. Failure keeps web Codex read-only.

The current deterministic probe establishes only the lower-level
`command/exec` boundary: read-only denies writes, workspace-write permits the
canonical root, sibling writes fail, and network is disabled on Codex CLI
0.146.0. That is necessary but not sufficient. It does not establish identical
policy enforcement across thread start, resume, and turn execution, nor does
it provide native item fixtures for honest effect attribution. The browser
write action therefore remains disabled while ordinary Codex chat stays
read-only.

The adapter must parse and record native command and file-change items, add a
durable turn-lease event, inventory partial effects after cancellation or
failure, and keep the human PTY isolated. Tests must prove generic chat stays
read-only, native elevation is denied, the writable root is canonical,
network remains disabled, cancellation reaches quiescence, sibling paths are
not writable through the enforced boundary, and concurrent unmatched changes
remain explicitly unattributed.

## Update (2026-08-12)

The lease store, explicit browser approval flow, isolated Codex turn policy,
native command and file-change projection, durable authority event, deadline,
and cancellation path now exist. Ordinary chat remains read-only and the human
PTY remains separate.

The decision remains proposed because its acceptance probe is incomplete. The
implemented lower-level `command/exec` probe did not exercise thread start,
resume, a real write turn, network denial, native elevation denial, or a
version-pinned Codex contract. It incorrectly enabled the browser action and
allowed completion evidence to call network denial passed. Piku now fails
closed: the lease implementation stays in place, but the browser does not
advertise or issue workspace-write authority until the complete probe and a
real held-out mutation journey satisfy this ADR's gates. Before/after workspace
inventory and explicit attribution of concurrent or partial effects also
remain required.
