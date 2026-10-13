# piku self-update implementation

status: implemented

Self-update is disabled by default. An operator can enable developer mode with
`--self-update-root /absolute/cargo/target`, approving replacement from that
directory's `release/piku`. Piku never downloads a binary or checks a remote
service.

For a default Cargo build root, launch from any working directory with:

```bash
piku --self-update-root /absolute/path/to/piku/target
```

The root must already exist and be an absolute directory. Piku canonicalizes
and pins its directory handle at launch. There is no environment or project
settings opt-in; `PIKU_SELF_UPDATE_ROOT` is ignored. Read-only mode always
disables replacement, even when this flag is present.

## Current flow

1. An operator explicitly selects a build root for a writable run, which then
   executes a successful `bash` tool call.
2. The output must contain Cargo's `Finished` marker and a Piku-specific marker:
   either `Compiling piku v` or `target/release/piku`.
3. `<approved root>/release/piku` must be a nonempty regular executable file
   with an mtime newer than the running binary's launch baseline. Piku opens
   `release` and `piku` relative to the pinned root with `O_NOFOLLOW`, rejecting
   symlinks for either entry, including links within the approved root.
4. Piku saves the current session before attempting replacement.
5. Admission opens and checks the approved source again, then copies that same
   open handle into a unique temporary snapshot beside the running executable.
   `self_replace::self_replace` installs that snapshot. Candidate path or
   ancestor swaps cannot redirect the copy after admission. The lower-level
   test path uses a temporary file beside the target followed by `rename`.
6. `exec` starts the replacement with the same arguments and
   `PIKU_RESTARTED=1`.

With operator opt-in, the TUI also compares the approved release binary with
its launch baseline before it draws the interface and at each input-loop
iteration. This catches a release
build produced by another process while the TUI is running. Read-only TUI mode
does not perform these restart checks.

## Session behavior by surface

The interactive TUI passes `PIKU_SESSION_ID` into the replacement process. The
new process loads that saved session and resumes the TUI. This preserves the
conversation, tool results, terminal ownership, and process identity across the
`exec`.

An opted-in writable launch turn, including `--print` and an ordinary prompt
that would later enter the TUI, saves its session but restarts without
`PIKU_SESSION_ID`. Replacement therefore restarts the same argument vector
rather than restoring the saved session. Only updates detected inside an
already-running TUI pass the session ID for seamless resume.

Invalid writable launch roots fail with an explicit startup error. Missing
release builds are normal and do not restart. Candidate inspection, replacement,
and `exec` failures are reported, and Piku continues with the existing process
where the caller can recover. A failed `exec` can occur after replacement has
already installed new bytes; there is no automatic rollback.

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
SelfUpdatePolicy::from_operator_root(root: Option<&Path>, read_only: bool)
    -> Result<SelfUpdatePolicy, SelfUpdateError>
policy.check_for_update() -> Result<Option<PathBuf>, SelfUpdateError>
policy.detect_self_build(output: &str, exit_success: bool)
    -> Result<Option<PathBuf>, SelfUpdateError>
policy.replace_and_exec(candidate: &Path, extra_env: &[(&str, &str)])
    -> Result<(), SelfUpdateError>
```

All automatic callers use the immutable policy. Existing library TUI convenience
entry points keep the default disabled policy; explicit callers use
`run_tui_repl_with_policy`. Low-level replacement primitives remain testable but
do not authorize automatic replacement.

Cargo text and mtimes are readiness hints, not provenance or approval. Forged
Cargo output or a newer ambient `target/release/piku` cannot enable replacement.
Within an approved root, a successful `cargo check` or another crate's build
without the Piku markers should not trigger tool-output replacement.

## Boundaries and recovery

- The explicit developer flag approves automatic replacement from one build
  directory for that launch. There is no additional per-build prompt.
- There is no GitHub release updater, background network check, signature
  verifier, or built-in rollback.
- Source control can help an operator reconstruct an older build, but Piku does
  not prescribe a destructive Git command as rollback.
- Developer approval controls application behavior; it provides no OS
  containment. Hooks and executed code retain the process's filesystem and
  network authority and can modify operator-owned files. An approved build
  root is not a signature or authenticity guarantee.
- Relative working-directory candidates, project settings, dotenv values, and
  Cargo output cannot select or change the approved root. The root handle stays
  pinned to the selected directory identity across pathname replacement.
- A copied executable outside the build root may be replaced by an approved
  candidate. Test probes use isolated copies, never the installed executable.

Unit and binary tests in `crates/piku/src/self_update.rs` and
`crates/piku/tests/self_update_e2e.rs` cover admission, marker discrimination, mtimes,
source/ancestor swaps, replacement error paths, restart markers, real copied
application startup, and forged Cargo output through the headless bash path.
