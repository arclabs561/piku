# Shell surfaces research

Status: research snapshot

Reviewed: 2026-10-23

Scope: what to take from shipped harness implementations for three requested
shell surfaces:

1. a persistent agent shell (state survives across tool calls);
2. a composable CLI contract (`piku run`, pipes, JSON events);
3. a shell replacement (Piku as `$SHELL` or a shell wrapper).

Ownership decision, 2026-10-23: the shell-replacement surface moves out of
Piku to the `wish` project, which owns the human shell. Piku keeps surface 1
(the executor shell inside the agent loop) and surface 2 (the composable CLI).
The dependency seam is one-way: wish may consume Piku's CLI contract; Piku does
not depend on wish.

This document is input to a decision, not a decision. It does not authorize
implementation ahead of the authority and tool-result phases in
[the harness roadmap](agentic-harness-roadmap.md).

## Question

Piku executes each shell call as `sh -c` in a fresh child with the launching
process working directory and environment, a 256 KiB per-stream cap, no durable
full artifact, no background job, and no state carry-over. Other harnesses solve
parts of this differently. Which mechanisms are worth taking, and which are
traps?

## Coverage and limits

Evidence came from reading shipped source or product documentation on
2026-10-23:

- Claude Code 2.1.285 (closed source; product documentation only);
- Codex CLI 0.164.0 (`openai/codex` source, `codex-rs/core/src/shell*.rs`);
- Crush 0.98.1 (`charmbracelet/crush`, `internal/shell/`, `internal/agent/tools/bash.go`);
- OpenCode 1.19.x/2.1.3 (`sst/opencode`, `packages/core/src/tool/plugin/shell.ts`, `packages/core/src/pty.ts`, `persistent-pty/`);
- pi 0.84.2 (installed package, `dist/core/tools/bash.js`).

Product documentation is a claim about behavior; it is not code. Claude Code's
runtime is closed, so its mechanisms are inferred from documented behavior only.
Version drift is likely: paths and defaults change between releases.

## What each does

| Harness | Process model | State across calls | Output contract | Background | Notes |
| --- | --- | --- | --- | --- | --- |
| Claude Code | one process per command | cwd carries over inside allowed roots; env persists only by explicit opt-in; aliases, functions, and shell options come from a one-time startup snapshot | streamed to a working file; ~30k chars inline, then path + preview; failures get a head/tail excerpt; 5 GB kill | yes, with notification; foreground timeout moves to background; time limits | exit 1 is benign only for a fixed command list (grep, rg, egrep, fgrep, find, diff, test, git diff, git grep) |
| Codex | per-call execution plus a reusable process manager (`unified_exec`) | none by default; a shell snapshot captures the shell environment with rebuild rules; login-shell option | typed events, sandbox policy applied per call | process manager supports open and background terminals | separates policy, sandbox, and shell invocation explicitly |
| Crush | embedded POSIX interpreter (`mvdan/sh`) plus external binary dispatch | cwd and env persist across `Exec` calls; in-process builtins and coreutils | 30k char cap, structured metadata (start/end, cwd, shell id) | native background jobs with `job_output`; auto-background after 60 s | deny rules compiled into the interpreter's exec handler, so they apply inside scripts; shebang/ELF/Mach-O dispatch |
| OpenCode | per-call shell service, separate from its PTY subsystem | shell tool takes `workdir`; PTY sessions are a different surface | full output saved to a file, truncated preview inline, structured `exit/signal/truncated/timeout` | yes, with a "do not poll" contract and completion notification | PTY layer retains 2 MiB per session, supports attach with cursors and replay, caps retained exited sessions |
| pi | one process per command, process-group kill | none; environment snapshot per call, `PI_*` variables exposed | byte and line truncation limits, no default timeout | extensions own background execution | minimal core: shell is a tool, workflow is extension policy |

## Transferable mechanisms

### 1. Working-directory carry-over, not environment carry-over

Claude Code persists the cwd automatically, only inside allowed roots, and
appends an explicit reset notice when a `cd` escapes them. Environment
variables do not persist unless explicitly opted in through `CLAUDE_ENV_FILE`
or a SessionStart hook; aliases, functions, and shell options are captured once
from the user's shell startup and applied to every command.

This is the smallest useful state model. It gives the common `cd foo && cargo
test` workflow without turning a persistent process into an unobserved ambient
session. It also composes with Piku's existing permission checks, because the
cwd remains a checked value rather than hidden process state.

Cost: none to containment. The cwd is already an explicit tool parameter in
Piku's file tools, and the executor working directory is an open Phase 1 item.

### 2. Shell environment snapshot

Codex captures the user's shell environment and applies it to each command;
Claude does the same for aliases, functions, and shell options. Both avoid the
two failure modes of naive `sh -c`: a PATH that differs from the user's
terminal, and aliases that work interactively but not for the agent.

For Piku this is a bounded, read-only capture at session start with a recorded
digest, not a mutation. It removes a recurring class of "works in my terminal,
fails in the agent" reports.

### 3. Stream to a file, return a bounded window plus an artifact reference

Claude Code and OpenCode both stream output to a file and inline a bounded
window. Past the window the model receives a path or artifact reference and can
read or search the rest. Failures get head-and-tail excerpts instead of a
truncated head.

Piku currently caps at 256 KiB per stream with an inline truncation marker and
no artifact, which is the exact gap the harness roadmap's Phase 2 names. The
transferable part is the split: bounded inline content, durable full artifact,
and different shaping for success versus failure.

### 4. Benign-failure classification

Claude Code treats exit 1 as success only for `grep`, `rg`, `diff`, `git diff`,
`test`, and a few others. Every other exit 1 is a failure. This is a small,
cheap correction to the common "nonzero means broken" heuristic, and it reduces
retry loops on search commands.

### 5. Background with an anti-polling contract

OpenCode's shell tool returns immediately, tells the model it will be notified,
and explicitly instructs it not to poll. Crush auto-backgrounds after 60 seconds
and exposes `job_output`. Piku has background agents but not background shell
jobs.

This is only safe after Piku's task lifecycle has acknowledgement and
disposition, so it belongs with Phase 3, not before it.

### 6. In-process interpreter with inherited deny rules

Crush embeds `mvdan/sh` and installs deny rules in the interpreter's exec
handler, so scripts run in-process inherit them. The boundary is the
interpreter: any command that escapes to a real binary is opaque to those
rules. Builtins are in-process, but Go coreutils are opt-in outside Windows, so
most external commands still spawn binaries.

This is the strongest prior art for "deny rules are structural, not a prefix
match", and it directly addresses the class of bug Piku's own landscape review
cites from Crush #2882. It is also the most expensive option: an embedded Rust
POSIX shell is a large dependency and a new compatibility surface, and
mvdan/sh's own documentation notes that a pure-Go interpreter cannot fork, so
it has no real PIDs or process groups. That is also why an embedded interpreter
is not a shell. Treat it as a possible future containment layer, not a first
step.

## What not to take

- **A single long-lived login shell as the default tool.** It hides cwd, env,
  and process state from the permission layer, and it makes one command's effect
  record depend on an earlier command. Crush's interpreter is not this: state is
  in-process and inspectable.
- **PTY output as model context by default.** OpenCode keeps terminals as a
  separate surface with explicit attach and replay. Piku's web PTY already
  follows that separation; keep it.
- **Unbounded background jobs before a lifecycle.** Without acknowledgement and
  disposition, background output produces exactly the lost-result failures the
  dogfood roadmap documents.
- **Polling loops as a completion strategy.** Prefer notification with a
  bounded payload and a durable artifact.
- **Full workspace shell containment claims.** None of these harnesses provides
  it through the shell tool alone; Codex uses an OS sandbox as a separate layer.

## Fit against the requested surfaces

### Persistent agent shell

Cheapest credible version: per-call process plus cwd carry-over inside the
leased workspace, plus a shell environment snapshot, plus a durable output
artifact. That is Claude Code's model and it needs no new containment story.

A truly persistent session (PTY or long-lived `sh`) should wait for the
authority lease and an explicit executor working directory, because the session
process itself becomes state that permission checks cannot see per call.

### Composable CLI

Piku already has `-p`, which prints a result, but not a stable pipe contract.
The transferable pattern is Claude Code's `--output-format stream-json` and
Codex's `exec --json`: newline-delimited typed events, stdout reserved for the
payload, diagnostics on stderr, and a nonzero exit that means the run failed
rather than the model disagreed.

This is low risk and independent of the shell process model. It is the natural
first surface to ship, because it makes Piku scriptable without touching tool
authority.

### Shell replacement

Out of scope for Piku. The `wish` project owns this surface; its design
direction is a full shell, and the surveyed prior art above is handed there
rather than duplicated here.

No surveyed harness ships this. Crush's embedded interpreter is the closest
primitive, and OpenCode's PTY layer is the closest presentation surface. A
`$SHELL` replacement would put the owning product in the login path for every
terminal, which multiplies the authority and failure surface far beyond Piku's
current product boundary.

The one-way seam: if a shell needs an agent, wish invokes the Piku CLI contract
from surface 2. Piku never imports wish, and Piku does not gain a login-shell
mode.

## Recommended sequence

1. **Shell result contract** with the Phase 2 work: bounded inline output,
   durable full artifact, failure shaping, benign-failure classification.
   Reversible, no authority change.
2. **Composable CLI contract**: `piku run` with typed NDJSON events, stdout and
   stderr separation, documented exit codes. Reversible.
3. **Executor cwd carry-over plus environment snapshot** as part of Phase 1's
   explicit executor directory. Needs the authority decision.
4. **Background shell jobs** after Phase 3 lifecycle acknowledgement.
5. **Persistent shell session** only if steps 1-4 leave a measured workflow
   failure that per-call execution cannot satisfy.

The shell-replacement surface is not in this sequence; it belongs to wish.

## Open questions

- **Open**: does cwd carry-over belong to the session or to the turn? Claude
  scopes it to the session with a reset on out-of-root `cd`; Piku's sessions can
  resume across days.
- **Open**: should the shell snapshot be per-session, per-project, or per
  provider/model route? A stale snapshot is a silent environment drift.
- **Open**: what is the artifact retention policy for full shell output, and
  does it interact with the run-record size bounds?

## Review trigger

Revisit when the Phase 2 result contract lands, when a measured workflow fails
because shell state does not persist, or when a shell command produces effects
that the current per-call model cannot attribute.
