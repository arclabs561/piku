# Design: hot dev reload for piku

status: implemented 2026-10-24
date: 2026-10-24

## Problem

Changing piku's implementation or settings today requires a manual rebuild and
relaunch. A watcher that just runs `cargo run` would destroy the live session,
drop the prompt draft, and leave the terminal in raw mode when a build fails.
The operator wants save-to-live behavior: the running TUI executes the latest
code and the latest configuration within seconds, without losing the
conversation.

## Context

- The TUI is a single hand-rolled ANSI loop that owns raw mode, the scroll
  region, and the cursor. It installs panic and signal hooks to restore the
  terminal (`crates/piku/src/tui_repl.rs:1521`).
- Configuration is resolved once at startup from CLI flags, the operator
  settings file, and `.piku/settings.toml` (`crates/piku/src/config.rs:115`).
- Project instructions are already hot: `build_system_prompt` runs per turn
  and re-reads `PIKU.md` (`crates/piku-runtime/src/prompt.rs:44`, called at
  `crates/piku/src/tui_repl.rs:2365`).
- Sessions are saved after every turn (`crates/piku/src/tui_repl.rs:2588`) and
  can be resumed by ID, including from an environment variable
  (`crates/piku/src/main.rs:353`).
- The risky part is already implemented and tested: save a checkpoint, then
  `replace_and_exec` the new binary with `PIKU_SESSION_ID` at the turn boundary
  (`crates/piku/src/tui_repl.rs:2588` to `2612`,
  `crates/piku/src/self_update.rs:74`). It is gated behind an explicit
  `--self-update-root` and refuses to run without a saved checkpoint.
- The prompt draft exists only in memory. History is loaded from and saved to a
  file at TUI start and exit (`crates/piku/src/input_helper.rs:557`,
  `crates/piku/src/tui_repl.rs:1576`).
- The browser surface is a separate process and subcommand (`piku web`,
  `crates/piku/src/cli.rs:134`).
- `just check` (`scripts/ci.sh`) is the completion gate. There is no dev,
  watch, or reload recipe today.
- The runtime already signals for restart through
  `AgentLoopResult::replace_and_exec` (`crates/piku-runtime/src/agent_loop.rs:126`).

## Non-goals

- No in-process hot swap of Rust code through a dylib. The ABI surface and
  panic/unwind behavior make it a worse risk than a process restart.
- No reload while a turn, tool call, or permission prompt is in flight. A
  reload request waits for the turn boundary.
- No weakening of self-update. Dev reload is a separate explicit mode and never
  reuses the self-update approval to replace the installed binary.
- No editor or LSP work. This is lifecycle only.
- No dependency resolution or `Cargo.lock` changes during watch.

## Options considered

**A. Supervisor process with restart at the turn boundary.** One small
supervisor owns the child lifetime, a watcher, and a build queue; the child
saves its session, writes its draft, and exits; the supervisor restarts it with
the session ID. Reuses the checkpoint and resume primitives that already exist.

**B. In-process `hot-lib-reloader` style dylib swap.** Fastest feedback, but
requires a stable ABI across the whole binary, cannot run the TUI state through
a rebuild safely, and makes panics undefined in practice. Rejected.

**C. Thin TUI client plus runtime sidecar over local RPC.** The cleanest
long-term shape: reset the surface without touching the runtime, and change
config live. It is a larger refactor of the loop, providers, and tool
execution, and the web surface already provides a second client that would have
to agree on the same protocol. Deferred, with a gate.

**D. Config-only live reload, manual restart for code.** Cheap, but does not
meet "latest impl always". Rejected as insufficient alone; it becomes the first
slice of A for settings that are safe to re-resolve.

**E. `bacon` or `cargo-watch` as the driver.** Proven terminal restoration, but
neither knows about piku sessions, drafts, or config resolution, and both
restart on every successful build with no turn-boundary awareness. Borrow the
terminal-restore behavior, not the process model.

## Chosen approach

A dedicated supervisor, `piku dev`, plus an explicit reload protocol between
supervisor and child.

### Process topology

1. **Supervisor (P0)** runs the original binary in dev mode. It owns the
   watcher thread, the build queue, the status file, and the child lifetime. It
   stays out of raw mode and never writes to the transcript region.
2. **Builder (P1)** is a `cargo build -p piku` child, one at a time, with a
   pinned profile (debug by default, `--release` opt-in). It writes to the
   repository target directory so incremental artifacts are reused.
3. **TUI child (P2)** is the built binary with an inherited terminal. It owns
   raw mode exactly as today, including the existing panic and signal restore
   hooks.
4. **Web child (P3)**, optional with `--web`, runs `piku web` against the same
   config and session directory and is restarted on the same successful build.

The supervisor uses the shared TTY rather than a PTY. A PTY would isolate
terminal corruption but breaks the child's existing `term_size`, raw-mode, and
signal assumptions and adds input forwarding and resize propagation. PTY
ownership is a fallback only if TTY sharing proves leaky (see gates).

### Reload triggers and the safe point

- The watcher covers `crates/**/*.rs`, `Cargo.toml`, `Cargo.lock`,
  `$XDG_CONFIG_HOME/piku/settings.toml`, and `.piku/settings.toml`. Events are
  debounced (300 ms) and coalesced into one queued build.
- A successful build writes a status record (git revision, dirty flag, build
  duration, binary path) and marks a reload as pending.
- The supervisor sends `SIGUSR1` to the child. The child sets a pending flag
  and handles it at the same turn boundary the self-update path uses.
- At the boundary the child saves the session, writes a private draft record
  (mode 0600, `$XDG_STATE_HOME/piku/dev-reload.json`), and exits with code 75.
- The supervisor restores the terminal, then re-execs the child with
  `PIKU_SESSION_ID` and the draft reference. The child loads the session,
  restores the draft, and replays the session tail as it does on resume.

Exit-and-restart is chosen over `exec`-in-place so dev reload never touches the
installed binary and never enters the self-update path. `exec`-in-place remains
the fallback for running without the supervisor.

### Configuration

For the first version, any settings change follows the same restart path. That
keeps one code path and satisfies "latest config always". Live re-resolution of
a safe subset (permissions, budgets, model) is a later slice behind a gate,
because provider and credential re-resolution at a turn boundary is the only
part that can silently change authority.

Instructions already reload per turn and are excluded from the watch set.

### State that must survive

| State | Mechanism | Status |
|---|---|---|
| Conversation | session JSON, saved at boundary | exists |
| Prompt draft | private dev state file | new |
| Prompt history | per-session history file | new, separate small change |
| Scroll position | not preserved in v1 | gate |
| In-flight turn | never interrupted; reload deferred | policy |

### Failure handling

- Build failure: child keeps running, status file records the error, the child
  footer shows `dev: build failed`. The next successful build replaces it.
- New child exits nonzero within two seconds twice: the supervisor restores the
  previous binary path, marks the revision bad, and waits for a source change.
- Reload storm: at most six reloads per minute, coalesced.
- Terminal: the supervisor always runs a restore sequence after any child exit,
  covering `SIGKILL` and panics that escape the child hook.
- Every reload appends a receipt (revision, reason, build ms, reload ms,
  session id, outcome) so reload quality is evidence rather than impression.

## Tradeoffs

- The child loses its PID, scroll position, and any in-memory UI state on each
  reload. Only file-backed state survives.
- Debug builds are the default, so measured latency is not release latency.
- The supervisor is new surface that must be tested, especially terminal
  restoration. A broken supervisor is worse than no watcher.
- Restarting for config changes is slower than live re-resolve and briefly
  interrupts typing, but it avoids a second authority path.

## Implementation plan

1. Add `piku dev` parsing and a supervisor skeleton that starts one child and
   restarts it on a keypress. Reversible.
2. Add the watcher, debounce, build queue, and exit-75 reload protocol with
   session resume. Reversible.
3. Add the draft state file and restore path. Reversible.
4. Harden terminal restore, add the bad-build and crash-loop guards, and write
   receipts. Reversible.
5. Add config and settings files to the watch set. Reversible.
6. Add the optional web child. Reversible.
7. Add PTY acceptance tests and a `just dev` recipe; document the mode in the
   README. Reversible.

## Decision gates

- If edit-to-live exceeds 8 seconds for a typical incremental build, add a
  compiler cache or split the crate before considering any in-process reload.
- If the PTY restore or draft-restore test fails twice, drop state carry and
  ship code-only restart; do not grow the protocol to compensate.
- If any reload loses a completed turn, stop and make reload an explicit
  keypress rather than automatic.
- If the runtime gains a shared session daemon, revisit option C; reload then
  becomes free and the supervisor shrinks to a client reset.
- If TTY sharing causes double echo, input latency, or resize artifacts, move
  the child to a PTY or fall back to `exec`-in-place.

## Resolved questions

- The build uses the repository `target` directory so incremental artifacts are
  shared with manual builds.
- Receipts, drafts, the build log, and the last good binary live under
  `$XDG_STATE_HOME/piku/dev`, not in the watched tree.
- Debug is the default profile; `--release` is opt-in.
- `--web` restarts the web child on each successful build. The remaining
  question is narrower: whether a web session can hold an in-flight turn that a
  TUI reload would interrupt. Until that is answered, `--web` is explicit.

## Implementation notes

- `crates/piku/src/dev.rs` owns the supervisor, the watch set, the build queue,
  the crash guard, the receipt log, and the child-side signal and draft hooks.
- `crates/piku/src/input_helper.rs` polls the reload flag and the status source
  while the prompt is idle, renders the status token, and shows the history
  picker instead of a single reverse-search line.
- `crates/piku/src/tui_repl.rs` handles the turn-boundary reload, restores the
  draft, and implements `/recap` as a local digest.
- `crates/piku/tests/dev_supervisor.rs` drives the real supervisor with a
  scripted child and build. `crates/piku/tests/tui_smoke.rs` sends `SIGUSR1`
  to a real TUI and checks session save, draft, terminal restore, and exit 75.
