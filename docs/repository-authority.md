# Design: Repository authority

Status: implemented; canonical checks and isolated admission probes verified.

## Overview

Piku keeps provider connections, project tool approval and executable replacement
under separate operator control. Opening a repository supplies task context and
restrictions; granting one capability does not grant the others. This design
defines the admission rules used by the [configuration](design.md) and
[developer self-update](self-update.md) paths.

## Problem

Opening a checkout previously let its dotenv choose provider endpoints, its
settings extend tool grants, and a newer local build replace the running Piku.
These are separate operator powers. Repository content must not acquire them
merely because Piku starts in that directory.

## Chosen approach

Provider connections belong to the operator's launching environment, global
configuration and explicit command options. Piku stops importing provider
variables from repository or ancestor `.env` files. Project settings cannot
override provider/model selection or provider connection blocks. Actor labels
and instructions remain project task data; actor connection choices remain
operator-controlled. Existing parsed provider blocks are not made executable
configuration by this change.

Project deny rules apply immediately. Project allow rules require an explicit
review and approval tied to the canonical repository root and a digest of the
permission policy. `piku permissions review` displays that root, policy and
digest; `piku permissions approve --root ROOT --digest DIGEST` approves only
that reviewed revision. Changing permission rules requires renewed review;
editing ordinary source does not. Global grants and project-hook approval keep
their existing, separate scopes. A project can lower the effective turn budget,
but cannot increase the operator's limit. Scoped shell grants must not silently
authorize additional commands through shell control syntax.

Developer self-update is disabled by default. `--self-update-root` explicitly
names an absolute Cargo target directory; only its `release/piku` candidate is
eligible. Capture a handle to that approved directory before hooks run; open the
release directory and candidate relative to it, rejecting symlinks. Copy the
opened candidate into a private staging file before replacement, so a pathname
swap cannot choose a different source. The same policy governs startup, idle
polling and tool-result updates. Read-only mode keeps updates disabled.

## Tradeoffs and alternatives

Projects that relied on dotenv provider credentials must export them explicitly
in their launch environment. Project routing moves to operator settings or
command options. Tool-rule approval adds a review step after policy changes;
default-disabled self-update needs a launch flag during self-hosting.

Global-only tool rules would be simpler but would discard useful reviewed
project capabilities. A blanket repository grant would combine credentials,
tool execution and binary replacement, and let later checkout changes inherit
unrelated authority. Per-update manual installation remains an alternative to
developer mode.

## Scope boundaries

- Approval is not OS containment: approved builds and tests execute repository
  code with the operator's permissions. Sandbox enforcement remains separate.
- Local-web client identity and PTY access retain their separate acceptance gate.
- Hook approval remains independent; this change does not redesign hook trust.
- Task acceptance, recording policy and verified child completion retain their
  existing contracts.

## Acceptance and decision gates

Synthetic loopback endpoints prove repository settings cannot redirect an
operator credential, while explicit operator routing still works. Real CLI
review/approval tests prove root binding, rule-change invalidation, persistent
denial and benign approved capabilities. Isolated executable copies prove
unapproved candidates cannot replace Piku, while opted-in updates preserve
session recovery. Run the canonical `just check` before delivery.

If scoped shell patterns cannot distinguish their reviewed capability from a
compound command, decline that automatic grant rather than infer approval.
No approval store or build-root flag is evidence of sandbox enforcement.

## Containment proposal

Status: proposed; implementation and negative-effect probes remain unexecuted.
The implemented admission contract above stays accepted. This section proposes
the separate execution boundary for a terminal coding turn that edits a task
checkout and runs its existing checks. It does not select containment as a
requirement for every turn or change the provider/backend choice.

### Required boundary for that workflow

The proposed contained profile permits reads of the explicit task source and
reviewed toolchain/runtime roots, writes within the task source and private
scratch, and execution of the task's checks. It denies unrelated host reads,
sibling writes, direct networking, host service sockets, provider credentials,
host process control, and executable replacement. Repository code, shell
startup files and descendant processes receive the same restrictions. A
worktree protects work organization; it is not this boundary.

Today `piku-tools::execute_tool` dispatches direct-file tools in Piku's process;
`bash::execute` starts `sh -c` without clearing its environment. Narrowing the
shell launch alone would leave file reads and writes under Piku's host
authority. The proposal is a credential-free tool worker, with a bounded
inherited request/result channel, that executes both file and shell tools
inside one OS-enforced profile. The host retains provider connections,
permission decisions and persistence. The worker cannot ask its channel for
arbitrary host file access, provider calls or broader execution. Tool names,
arguments and bounded results cross it; no model/backend is added. This builds
on the separation demonstrated by the [managed page broker](managed-page-executor-boundary.md),
without extending that page-only protocol into a general executor.

| Boundary | Proposed contract |
| --- | --- |
| Credentials | Clear worker environment; allow only reviewed launch/toolchain variables. Keep provider/auth files, application state, SSH agents and credential sockets outside readable roots. Close unrelated inherited descriptors. A fake home alone is insufficient. |
| Reads | Explicit task and runtime roots only; no broad home read. Include `read_file`, `glob`, `grep`, `list_dir`, shell commands and delegated attachments. Results still enter the selected provider context, so readable roots are a disclosure grant. |
| Writes | Task root and worker scratch only. Session/history/memory persistence stays host-owned under existing policy; the worker cannot overwrite it. Cache/output roots require explicit separate writable grants. |
| Network and IPC | Deny external, LAN, loopback, DNS and host Unix sockets by default in this profile. Provider egress remains host-only. A later fetch/install capability requires its own narrow host broker or reviewed network profile; never retry unrestricted. |
| Processes | Restrictions follow descendants, including detached children; deny host signaling/debugging and privilege escalation. Cancellation must reap the worker scope and prevent later effects. Process-group cleanup is useful but does not prove confinement. |
| Hooks and updates | Contained hooks use the same worker. An existing ambient hook is separately visible authority, not contained execution. Contained mode does not opt into self-update or the human browser PTY. |

Linked worktrees need special treatment: their Git administrative paths can
point outside the task root. Do not grant the primary repository's whole Git
directory to make a command work. Test the exact required metadata scope or use
an isolated repository copy for the probe. Reject unreviewed shared writable
inodes/hardlinks to outsider files; a path-only write grant cannot protect an
outside alias of an allowed inode. Integration remains an explicit host/operator
step and preserves unrelated human work.

### Options and platform posture

1. **Existing host tools plus permission rules.** Preserve as the current
   compatibility behavior, with honest ambient-authority labeling. They do not
   satisfy the proposed contained profile: path classification, environment
   filtering and worktrees cannot stop arbitrary repository code.
2. **Credential-free worker under a native OS launcher.** Recommended first
   feasibility slice, still a proposal. On macOS, evaluate the installed
   deterministic `codex sandbox` launcher with explicit restricted reads and
   network disabled. It launches commands without an inference turn; using it
   would be a sandbox dependency, not choosing Codex as the agent executor.
   Validate its exact installed interface and executable identity first.
   For Linux, a reviewed bubblewrap mount/PID/network/IPC profile plus syscall
   hardening is a grounded alternative. Neither is currently integrated with
   the own-loop dispatcher. A native launcher must demonstrate every required
   boundary before being advertised as supported.
3. **Isolated VM/container tool worker.** Retain as the alternative if native
   private-read/process isolation fails. Expose only the task projection and
   toolchain, with no host credentials, runtime sockets or broad home mounts.
   This adds provisioning and artifact integration cost; a container name
   alone does not establish its mount, network or process policy.

Official [Codex documentation](https://learn.chatgpt.com/docs/agent-approvals-security#os-level-sandbox)
describes macOS Seatbelt and Linux bubblewrap/seccomp enforcement. Its generic
workspace-write/read-only labels do not establish this proposal's exact private
read boundary. The installed CLI help must resolve launch syntax; documentation
examples and installed interfaces can differ.
[Bubblewrap's primary documentation](https://github.com/containers/bubblewrap#sandbox-security)
explicitly assigns security policy to the launcher arguments and warns that
mounted host sockets can acquire host authority. [Landlock documentation](https://docs.kernel.org/userspace-api/landlock.html)
describes inherited restrictions and ABI-dependent rights; do not degrade a
required denial silently on an older kernel. No new Windows implementation is
selected. Unsupported platform, unavailable kernel primitive, failed/stale
probe or launch-policy mismatch makes the *contained profile* unavailable;
never substitute ambient execution or another provider. Whether contained
mode becomes a default or required mode remains an owner decision.

### Reuse existing attestation, preserve its scope

The optional native Codex write path already uses the private v5 attestation in
`web/codex_attestation.rs` and `scripts/codex-app-server-probe.mjs`. It binds Piku
and Codex versions, executable digest, OS/architecture, launch policy, effective
allowlisted environment, age and required gates. The probe pairs workspace
write success with sibling write denial and a real loopback connection attempt;
interactive mode checks command/file-change events and start/resume lifecycle.
Piku refuses broader native approval requests independently.

Reuse those identity/freshness checks and independent sentinel/listener
witnesses when implementing the own-loop profile. Do not reuse its successful
gate set as proof of private-read, IPC, hardlink or detached-process denial:
those are additional gates. An own-loop receipt must identify its actual
worker/launcher and effective policy; a native Codex receipt cannot authorize
Piku's host file dispatcher. Local attestation is regression evidence, not
protection against another hostile same-user process forging host state.

### Negative-effect acceptance and next gate

Use only synthetic canaries, task copies and cooperating host witnesses. Each
test must confirm that the operation was attempted; a model refusal, missing
command, provider failure or timeout is inconclusive. Run real own-loop tool
dispatch and repeat through a delegated child; launcher-only success is not
product acceptance.

| Probe | Independent acceptance predicate |
| --- | --- |
| Useful coding turn | Read/edit a task file and run the repository's existing check; assert intended content and check result from the host. |
| Private reads and credentials | Place synthetic outsider/auth/environment canaries; attempt absolute, relative, symlink and file-tool/shell reads. Canary bytes never appear in results; allowed reads still work. |
| Outside mutations | Attempt write, truncate, rename, unlink and symlink replacement against seeded sibling files; host content/identity snapshots remain unchanged. Include shared-inode/hardlink rejection. |
| Network and IPC | First prove cooperating host TCP/UDP/Unix listeners accept an ambient synthetic client. Attempts from worker and descendants deliver no connection/datagram/request; test loopback and available IPv4/IPv6 paths. |
| Process boundary | Attempt signaling a cooperating outsider and a detached child that writes after cancellation. Outsider remains alive; no late sentinel appears after the worker scope is reaped. |
| Fail-closed admission | Change executable/policy/environment digest, expire the receipt, remove required primitive, or inject a broader approval request. Tool operation never begins and the stable reason is visible. |
| All entry paths | Exercise headless, launch-to-TUI, TUI, delegated tools and supported hooks. No direct host-dispatch fallback or inherited broker/provider descriptor is reachable. |

Next implementation gate: resolve the macOS launch-policy syntax and demonstrate
allowed task IO plus denied outsider read/write, network/IPC and detached-child
effects with a deterministic credential-free test process. Record exact host,
launcher and policy identities. If that fails, compare the isolated VM/container
option before wiring a worker. Only then integrate one own-loop tool path,
prove the same predicates through actual dispatch, expand to all entry paths,
and run the canonical `just check`. No probe execution is claimed here.
