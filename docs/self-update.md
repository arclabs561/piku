# piku Self-Update Design

How piku rebuilds itself while it's running.

---

## The Problem

piku is a coding agent. One of the first things you'll use it for is building piku itself. That creates a specific challenge: the running binary is being asked to edit its own source and recompile itself.

The three hard constraints:

1. **No data loss** — session history and tool results must survive the restart
2. **No process corruption** — the running binary must not be overwritten mid-execution
3. **Clean restart** — the new binary must come up correctly and pick up where it left off

---

## The Flow

```
piku "add streaming progress bar to the TUI"
  │
  ├─ reads src/tui.rs
  ├─ edits src/tui.rs
  ├─ bash: cargo build --release
  │         └─ writes target/release/piku  (new inode, different from running binary)
  │
  ├─ piku detects: target/release/piku is newer than current exe
  │
  ├─ session auto-persisted to ~/.config/piku/sessions/<id>.json  ← already done per-turn
  │
  ├─ self_replace::self_replace("target/release/piku")
  │         └─ tempfile in same dir as current exe → rename()
  │            atomic on same filesystem, new inode at old path
  │
  └─ exec(current_exe, same_args, PIKU_RESTARTED=1)
            └─ process image replaced, same PID, same TTY
               new piku starts, loads session, continues
```

---

## Why This Design

### Atomic rename, not in-place write

Writing directly to the running binary (`open(O_WRONLY)`) triggers `ETXTBSY` on some platforms. More importantly, a partial write would corrupt the binary.

The correct approach: write to a tempfile in the same directory as the current executable, then `rename()` into place. POSIX guarantees `rename()` is atomic within a filesystem — readers see either the old binary or the new one, never a partial state.

### Same directory for the tempfile

`rename()` is only atomic within a single filesystem. `/tmp` is often a separate tmpfs mount. Placing the tempfile in the same directory as the current executable ensures same-filesystem semantics.

### Why exec(2) not spawn+exit

`exec(2)` replaces the process image without creating a new process:
- Same PID — shell job control stays intact
- Same TTY — the terminal doesn't see a disconnect
- Same file descriptors — no fd leaks
- No zombie process

Rust stdlib: `std::os::unix::process::CommandExt::exec()`.

### macOS Gatekeeper note

macOS Gatekeeper caches code signatures by inode number. Overwriting a file in-place retains the old inode; the OS sends `SIGKILL` to any new process trying to exec it. `rename()` into place always gives the new binary a fresh inode at the old path — correct behavior.

### Session persistence enables seamless resume

piku already auto-persists sessions after every turn (`~/.config/piku/sessions/<id>.json`). The bash tool result (cargo build output) is in the session. After exec, the new binary sees `PIKU_RESTARTED=1` and can load the session to continue.

---

## Detection

piku detects a self-rebuild by checking two conditions after any bash tool call:

1. The bash output contains `"Finished"` (cargo's success indicator)
2. `target/release/piku` exists and has a newer mtime than `current_exe()`

This is intentionally conservative. False negatives (rebuild happened but not detected) are safe — the old binary just keeps running. False positives would cause unnecessary restarts but are very unlikely given the mtime check.

---

## What piku Does NOT Do

- **No automatic update from GitHub releases** — piku is your tool, you build it
- **No signature verification** — you built it yourself, on your machine
- **No rollback mechanism** — git is the rollback mechanism (`git checkout HEAD~1`)
- **No background update check** — no network calls without prompting

---

## Implementation

`crates/piku/src/self_update.rs`:

```rust
// Check if a new build is available
pub fn is_newer_than_running(new_binary: &Path) -> bool

// Get the default cargo build output path
pub fn default_build_output() -> PathBuf  // "target/release/piku"

// Detect if a bash command just rebuilt piku
pub fn detect_self_build(bash_output: &str, exit_success: bool) -> Option<PathBuf>

// Atomic replace + exec — does not return on success
pub fn replace_and_exec(new_binary: &Path) -> Result<(), SelfUpdateError>

// Was this process started via self-update?
pub fn was_restarted() -> bool
```

The bash tool's `execute()` return value is checked in the agentic loop. If `detect_self_build()` returns `Some(path)`, the loop:

1. Confirms the session is persisted (already done)
2. Prints a status line: `[piku] rebuilt — restarting...`
3. Calls `replace_and_exec(&path)`

The new process sees `PIKU_RESTARTED=1` and can optionally print a banner like `[piku] restarted after self-rebuild`.

---

## Future: Interactive Confirmation

In the TUI (v1), before exec:

```
╔══════════════════════════════════════════╗
║ piku was rebuilt. Restart with new binary? ║
║ [y] restart now  [n] keep running         ║
╚══════════════════════════════════════════╝
```

In single-shot mode (v0), auto-restart without prompting since there's no interactive session.
