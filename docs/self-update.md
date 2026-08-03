# piku self-update implementation

status: implemented

Piku can replace and restart itself after it builds a newer
`target/release/piku`. This is local self-hosting support, not a release updater:
Piku never downloads a binary or checks a remote service.

## Current flow

1. A writable Piku run executes a successful `bash` tool call.
2. The output must contain Cargo's `Finished` marker and a Piku-specific marker:
   either `Compiling piku v` or `target/release/piku`.
3. `target/release/piku` must exist and have an mtime newer than the running
   binary or the TUI's startup mtime baseline.
4. Piku saves the current session before attempting replacement.
5. `self_replace::self_replace` replaces the running executable. The lower-level
   test path uses a copy to a temporary file beside the target followed by
   `rename`.
6. `exec` starts the replacement with the same arguments and
   `PIKU_RESTARTED=1`.

The TUI also compares the release binary with its startup baseline before it
draws the interface and at each input-loop iteration. This catches a release
build produced by another process while the TUI is running. Read-only TUI mode
does not perform these restart checks.

## Session behavior by surface

The interactive TUI passes `PIKU_SESSION_ID` into the replacement process. The
new process loads that saved session and resumes the TUI. This preserves the
conversation, tool results, terminal ownership, and process identity across the
`exec`.

Every writable launch turn, including `--print` and an ordinary prompt that
would later enter the TUI, saves its session but calls `replace_and_exec` without
`PIKU_SESSION_ID`. Replacement therefore restarts the same argument vector
rather than restoring the saved session. Only updates detected inside an
already-running TUI pass the session ID for seamless resume.

If replacement or `exec` fails, Piku reports the error and continues with the
old process where the caller can recover.

## Why replacement precedes `exec`

Writing into a running executable can fail or leave a partial binary. The
replacement path installs a complete new file and only then calls `exec`. The
testable arbitrary-target path places its temporary file beside the target so
the final rename stays on one filesystem. For the actual executable,
`self_replace` owns platform and symlink handling.

`exec` preserves the PID and terminal association. Piku explicitly forwards its
arguments, inherited environment, the restart marker, and, for TUI restarts,
the session identifier.

## Detection contract

`crates/piku/src/self_update.rs` exposes these relevant operations:

```rust
pub fn default_build_output() -> PathBuf
pub fn detect_self_build(output: &str, exit_success: bool) -> Option<PathBuf>
pub fn running_mtime() -> Option<SystemTime>
pub fn is_newer_than_mtime(path: &Path, baseline: SystemTime) -> bool
pub fn replace_and_exec(new_binary: &Path) -> Result<(), SelfUpdateError>
pub fn replace_and_exec_with_env(
    new_binary: &Path,
    extra_env: &[(&str, &str)],
) -> Result<(), SelfUpdateError>
pub fn was_restarted() -> bool
```

The content markers are deliberately conservative. A successful `cargo check`
or a build of another workspace crate should not trigger replacement. Mtime is
the final guard, so matching output alone is insufficient.

## Boundaries and recovery

- There is no interactive confirmation. A qualifying writable build restarts
  automatically.
- There is no GitHub release updater, background network check, signature
  verifier, or built-in rollback.
- Source control can help an operator reconstruct an older build, but Piku does
  not prescribe a destructive Git command as rollback.
- Detection assumes the default release path relative to Piku's working
  directory.
- A copied executable outside that path is replaced only when the release build
  is detected and is newer.

Unit and binary tests in `crates/piku/src/self_update.rs` and
`crates/piku/tests/self_update_e2e.rs` cover marker discrimination, mtimes,
replacement error paths, restart markers, and integration with the bash result
path.
