#![allow(
    clippy::doc_markdown,
    clippy::must_use_candidate,
    clippy::map_unwrap_or
)]

/// Self-update support for piku.
///
/// Handles the "piku builds itself" use case:
///
///   1. piku runs `cargo build --release` via the bash tool
///   2. cargo writes a new binary to `target/release/piku`
///   3. piku detects the new binary is different from the running one
///   4. piku saves session state to disk (already happens per-turn)
///   5. piku atomically replaces itself on disk: do_replace(new_binary)
///   6. piku exec(2)s the new binary via exec_self(): same PID, same TTY
///   7. new piku starts with PIKU_RESTARTED=1, loads session, continues
///
/// The two steps are intentionally split so `do_replace` can be unit-tested
/// without actually replacing the running test binary.
use std::path::{Path, PathBuf};

#[derive(Debug, thiserror::Error)]
pub enum SelfUpdateError {
    #[error("could not determine current executable path: {0}")]
    CurrentExe(#[source] std::io::Error),

    #[error("new binary not found at {path}: {source}")]
    NewBinaryNotFound {
        path: PathBuf,
        source: std::io::Error,
    },

    #[error("new binary is empty — build may have failed")]
    NotExecutable,

    #[error("atomic self-replace failed: {0}")]
    Replace(#[source] std::io::Error),

    #[error("exec failed: {0}")]
    Exec(#[source] std::io::Error),
}

// ---------------------------------------------------------------------------
// Primary entry point
// ---------------------------------------------------------------------------

/// Atomically replace the running binary with `new_binary`, then exec
/// the new binary with the same arguments and environment.
///
/// **Does not return on success.**  Returns `Err` if replace or exec fails.
pub fn replace_and_exec(new_binary: &Path) -> Result<(), SelfUpdateError> {
    replace_and_exec_with_env(new_binary, &[])
}

/// Like `replace_and_exec` but also sets extra env vars in the new process.
/// Use this to pass `PIKU_SESSION_ID` so the restarted process can resume
/// the in-progress session seamlessly.
pub fn replace_and_exec_with_env(
    new_binary: &Path,
    extra_env: &[(&str, &str)],
) -> Result<(), SelfUpdateError> {
    let current_exe = std::env::current_exe().map_err(SelfUpdateError::CurrentExe)?;
    do_replace(new_binary, &current_exe)?;
    exec_self(&current_exe, extra_env)
}

// ---------------------------------------------------------------------------
// Testable primitives
// ---------------------------------------------------------------------------

/// Atomically replace `target` with `new_binary` on disk.
///
/// Uses `self_replace::self_replace` which resolves symlinks and preserves
/// permissions. The tempfile is placed in the same directory as `target` to
/// guarantee same-filesystem rename(2) atomicity.
///
/// This is the only function that touches the filesystem. It is split out so
/// tests can replace an arbitrary target (not `current_exe()`) without
/// affecting the running test process.
pub fn do_replace(new_binary: &Path, target: &Path) -> Result<(), SelfUpdateError> {
    // Sanity checks
    let meta = std::fs::metadata(new_binary).map_err(|e| SelfUpdateError::NewBinaryNotFound {
        path: new_binary.to_path_buf(),
        source: e,
    })?;
    if meta.len() == 0 {
        return Err(SelfUpdateError::NotExecutable);
    }

    // If target == current_exe, self_replace handles symlink resolution.
    // If target != current_exe (test scenario), we do a manual atomic rename.
    let current_exe = std::env::current_exe().ok();
    if current_exe.as_deref() == Some(target)
        || current_exe
            .as_ref()
            .and_then(|p| std::fs::canonicalize(p).ok())
            == std::fs::canonicalize(target).ok()
    {
        // Replacing the actual running binary — use self_replace for safety
        self_replace::self_replace(new_binary).map_err(SelfUpdateError::Replace)
    } else {
        // Replacing an arbitrary target (test use) — manual atomic copy+rename
        replace_arbitrary(new_binary, target)
    }
}

/// Exec the given path with the current process's arguments.
/// Sets `PIKU_RESTARTED=1`.  Any extra `(key, value)` pairs in `extra_env`
/// are also set.  Does not return on success.
pub fn exec_self(exe: &Path, extra_env: &[(&str, &str)]) -> Result<(), SelfUpdateError> {
    use std::os::unix::process::CommandExt;
    let mut cmd = std::process::Command::new(exe);
    cmd.args(std::env::args_os().skip(1));
    cmd.env("PIKU_RESTARTED", "1");
    for (k, v) in extra_env {
        cmd.env(k, v);
    }
    let err = cmd.exec();
    Err(SelfUpdateError::Exec(err))
}

/// Atomically replace `target` with `source` using tempfile + rename.
/// Used for testing (where target != current_exe).
fn replace_arbitrary(source: &Path, target: &Path) -> Result<(), SelfUpdateError> {
    let parent = target.parent().ok_or_else(|| {
        SelfUpdateError::Replace(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "target has no parent directory",
        ))
    })?;

    // Write to a tempfile in the same directory as target
    let tmp = parent.join(format!(
        ".piku-replace-{}.tmp",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.subsec_nanos())
            .unwrap_or(0)
    ));

    std::fs::copy(source, &tmp).map_err(SelfUpdateError::Replace)?;

    // Copy permissions from source
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let src_mode = std::fs::metadata(source)
            .map(|m| m.permissions().mode())
            .unwrap_or(0o755);
        let mut perms = std::fs::metadata(&tmp).unwrap().permissions();
        perms.set_mode(src_mode);
        let _ = std::fs::set_permissions(&tmp, perms);
    }

    std::fs::rename(&tmp, target).map_err(|e| {
        let _ = std::fs::remove_file(&tmp);
        SelfUpdateError::Replace(e)
    })
}

// ---------------------------------------------------------------------------
// Mtime comparison
// ---------------------------------------------------------------------------

/// Returns true if `new_binary` exists and has a strictly newer mtime than
/// the currently running executable. Returns false on any I/O error.
///
/// NOTE: if the running binary IS `new_binary` (e.g. via symlink), this
/// always returns false. Use `is_newer_than_mtime` with a baseline captured
/// at startup instead.
pub fn is_newer_than_running(new_binary: &Path) -> bool {
    let Ok(current) = std::env::current_exe() else {
        return false;
    };
    // If they resolve to the same file, mtime comparison is meaningless.
    let same = std::fs::canonicalize(new_binary).ok() == std::fs::canonicalize(&current).ok();
    if same {
        return false;
    }
    mtime_newer_than(new_binary, &current)
}

/// Returns the mtime of the running executable at the moment of the call.
/// Call this once at startup and store the result as a baseline.
pub fn running_mtime() -> Option<std::time::SystemTime> {
    let current = std::env::current_exe().ok()?;
    std::fs::metadata(&current).ok()?.modified().ok()
}

/// Returns true if `path` has a strictly newer mtime than `baseline`.
pub fn is_newer_than_mtime(path: &Path, baseline: std::time::SystemTime) -> bool {
    std::fs::metadata(path)
        .and_then(|m| m.modified())
        .map(|t| t > baseline)
        .unwrap_or(false)
}

/// Returns true if `a` has a strictly newer mtime than `b`.
pub fn mtime_newer_than(a: &Path, b: &Path) -> bool {
    let Ok(a_meta) = std::fs::metadata(a) else {
        return false;
    };
    let Ok(b_meta) = std::fs::metadata(b) else {
        return false;
    };
    let Ok(a_mtime) = a_meta.modified() else {
        return false;
    };
    let Ok(b_mtime) = b_meta.modified() else {
        return false;
    };
    a_mtime > b_mtime
}

// ---------------------------------------------------------------------------
// Cargo build detection
// ---------------------------------------------------------------------------

/// Returns the default path where `cargo build --release` writes the piku binary.
/// Relative to the process working directory.
pub fn default_build_output() -> PathBuf {
    PathBuf::from("target/release/piku")
}

/// Detect whether a bash command output indicates a successful **piku** rebuild.
///
/// Returns the path to the new binary if:
///   1. The command exited successfully (`exit_success = true`)
///   2. The output looks like `cargo build` of the piku binary specifically
///      (contains "Finished" AND the piku binary name appears)
///   3. `target/release/piku` exists and is newer than the running binary
///
/// The double check (Finished + piku-specific) guards against false positives
/// when building other crates in the workspace (e.g. `cargo build -p piku-api`).
pub fn detect_self_build(bash_output: &str, exit_success: bool) -> Option<PathBuf> {
    if !exit_success {
        return None;
    }

    // Require BOTH conditions to avoid false positives:
    //   1. "Finished" — cargo's success marker (present on build AND check)
    //   2. piku-specific signal — distinguishes `cargo build -p piku` from
    //      `cargo check`, `cargo build -p piku-api`, etc.
    //
    // "Compiling piku v" (with space+v) matches only the `piku` binary crate.
    // "piku-api", "piku-tools", "piku-runtime" do not match (hyphen, not space+v).
    // "target/release/piku" matches if cargo prints the output path.
    let has_finished = bash_output.contains("Finished");
    let mentions_piku_binary =
        bash_output.contains("Compiling piku v") || bash_output.contains("target/release/piku");

    // Both required — `cargo check` has Finished but no binary; other crates
    // have mentions but don't produce the piku binary.
    if !has_finished || !mentions_piku_binary {
        return None;
    }

    let candidate = default_build_output();
    if is_newer_than_running(&candidate) {
        Some(candidate)
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Process restart detection
// ---------------------------------------------------------------------------

/// Returns true if this process was started by a piku self-update exec.
/// Clears the env var to prevent leaking it to child processes.
pub fn was_restarted() -> bool {
    if std::env::var("PIKU_RESTARTED").is_ok() {
        std::env::remove_var("PIKU_RESTARTED");
        true
    } else {
        false
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn tempdir() -> PathBuf {
        let base = std::env::temp_dir().join(format!(
            "piku_su_{}_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.subsec_nanos())
                .unwrap_or(0),
            std::process::id(),
        ));
        std::fs::create_dir_all(&base).unwrap();
        base
    }

    fn write_binary(path: &Path, content: &[u8]) {
        std::fs::write(path, content).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut p = std::fs::metadata(path).unwrap().permissions();
            p.set_mode(0o755);
            std::fs::set_permissions(path, p).unwrap();
        }
    }

    // -----------------------------------------------------------------------
    // is_newer_than_running / mtime_newer_than
    // -----------------------------------------------------------------------

    #[test]
    fn is_newer_than_running_false_for_nonexistent() {
        assert!(!is_newer_than_running(Path::new("/no/such/binary")));
    }

    #[test]
    fn is_newer_than_running_false_for_same_file_as_running() {
        let exe = std::env::current_exe().unwrap();
        assert!(!is_newer_than_running(&exe));
    }

    #[test]
    fn mtime_newer_than_returns_false_for_same_file() {
        let exe = std::env::current_exe().unwrap();
        assert!(!mtime_newer_than(&exe, &exe));
    }

    #[test]
    fn mtime_newer_than_returns_true_for_freshly_written_file() {
        let dir = tempdir();
        let old = dir.join("old");
        let new = dir.join("new");

        write_binary(&old, b"old content");
        // small sleep to ensure different mtime on low-resolution filesystems
        std::thread::sleep(std::time::Duration::from_millis(10));
        write_binary(&new, b"new content");

        assert!(mtime_newer_than(&new, &old), "new should be newer than old");
        assert!(
            !mtime_newer_than(&old, &new),
            "old should not be newer than new"
        );
    }

    // -----------------------------------------------------------------------
    // detect_self_build
    // -----------------------------------------------------------------------

    #[test]
    fn detect_self_build_false_on_failed_exit() {
        assert!(detect_self_build("Finished release [optimized]", false).is_none());
    }

    #[test]
    fn detect_self_build_false_when_no_finished() {
        assert!(detect_self_build("Compiling piku v0.1.0", true).is_none());
    }

    #[test]
    fn detect_self_build_false_for_other_crate_only() {
        // Building piku-api produces "Finished" but not "Compiling piku v"
        // This should NOT trigger because it only has Finished, not piku-specific marker
        // BUT: "Finished" alone still passes the `has_finished` check.
        // The guard is: require `has_finished`. If binary isn't newer → None.
        // So this correctly returns None if binary is not newer.
        let result = detect_self_build(
            "Compiling piku-api v0.1.0\nFinished release [optimized]",
            true,
        );
        // Returns None because target/release/piku doesn't exist / isn't newer in test env
        assert!(result.is_none());
    }

    #[test]
    fn detect_self_build_some_when_binary_is_newer() {
        // Use a mutex to serialize cwd-changing tests across parallel test threads.
        static CWD_MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());
        let _guard = CWD_MUTEX.lock().unwrap();

        let dir = tempdir();
        let release_dir = dir.join("target").join("release");
        std::fs::create_dir_all(&release_dir).unwrap();
        let fake_binary = release_dir.join("piku");

        let exe_content = std::fs::read(std::env::current_exe().unwrap()).unwrap();
        write_binary(&fake_binary, &exe_content);

        // Write again after sleep so mtime is definitively newer
        std::thread::sleep(std::time::Duration::from_millis(50));
        write_binary(&fake_binary, &exe_content);

        let original_cwd = std::env::current_dir().unwrap();
        std::env::set_current_dir(&dir).unwrap();

        let result = detect_self_build(
            "Compiling piku v0.1.0\nFinished release [optimized] target(s) in 3.14s",
            true,
        );

        std::env::set_current_dir(original_cwd).unwrap();

        assert_eq!(
            result,
            Some(PathBuf::from("target/release/piku")),
            "should detect newer binary"
        );
    }

    // -----------------------------------------------------------------------
    // is_newer_than_mtime / running_mtime baseline approach
    // -----------------------------------------------------------------------

    #[test]
    fn is_newer_than_mtime_false_when_file_unchanged() {
        let exe = std::env::current_exe().unwrap();
        let baseline = std::fs::metadata(&exe).unwrap().modified().unwrap();
        // same file, same mtime — must be false
        assert!(!is_newer_than_mtime(&exe, baseline));
    }

    #[test]
    fn is_newer_than_mtime_true_after_write() {
        let dir = tempdir();
        let file = dir.join("bin");
        write_binary(&file, b"v1");
        let baseline = std::fs::metadata(&file).unwrap().modified().unwrap();
        std::thread::sleep(std::time::Duration::from_millis(20));
        write_binary(&file, b"v2");
        assert!(is_newer_than_mtime(&file, baseline));
    }

    #[test]
    fn baseline_detects_update_even_when_same_path_as_running() {
        // Simulates: current_exe == build output (symlink scenario).
        // We write a file, capture its mtime as baseline, overwrite it,
        // then confirm is_newer_than_mtime returns true.
        let dir = tempdir();
        let file = dir.join("piku");
        write_binary(&file, b"old");
        let baseline = std::fs::metadata(&file).unwrap().modified().unwrap();
        std::thread::sleep(std::time::Duration::from_millis(20));
        write_binary(&file, b"new");
        // is_newer_than_running would return false (same inode), but baseline works:
        assert!(is_newer_than_mtime(&file, baseline));
    }

    // -----------------------------------------------------------------------
    // do_replace on non-current-exe target (safe for testing)
    // -----------------------------------------------------------------------

    #[test]
    fn do_replace_writes_new_content_to_target() {
        let dir = tempdir();
        let target = dir.join("old-binary");
        let new_bin = dir.join("new-binary");

        write_binary(&target, b"old content");
        write_binary(&new_bin, b"new content: much better");

        do_replace(&new_bin, &target).expect("do_replace should succeed");

        let result = std::fs::read(&target).unwrap();
        assert_eq!(result, b"new content: much better");
    }

    #[test]
    fn do_replace_preserves_executable_bit() {
        let dir = tempdir();
        let target = dir.join("target-bin");
        let new_bin = dir.join("new-bin");

        write_binary(&target, b"old");
        write_binary(&new_bin, b"new");

        do_replace(&new_bin, &target).unwrap();

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(&target).unwrap().permissions().mode();
            assert!(mode & 0o111 != 0, "replaced binary should be executable");
        }
    }

    #[test]
    fn do_replace_fails_on_missing_new_binary() {
        let dir = tempdir();
        let target = dir.join("target");
        write_binary(&target, b"existing");

        let result = do_replace(Path::new("/no/such/new-binary"), &target);
        assert!(matches!(
            result,
            Err(SelfUpdateError::NewBinaryNotFound { .. })
        ));
    }

    #[test]
    fn do_replace_fails_on_empty_new_binary() {
        let dir = tempdir();
        let target = dir.join("target");
        let empty = dir.join("empty");

        write_binary(&target, b"existing");
        std::fs::write(&empty, b"").unwrap();

        let result = do_replace(&empty, &target);
        assert!(matches!(result, Err(SelfUpdateError::NotExecutable)));
    }

    #[test]
    fn do_replace_is_atomic_target_unchanged_on_failure() {
        // If new_binary doesn't exist, target should remain untouched
        let dir = tempdir();
        let target = dir.join("target");
        write_binary(&target, b"original content");

        let _ = do_replace(Path::new("/no/such/binary"), &target);

        assert_eq!(std::fs::read(&target).unwrap(), b"original content");
    }

    #[test]
    fn do_replace_with_real_binary_content() {
        // Use the actual test binary as "new_binary" — real ELF/Mach-O content
        let dir = tempdir();
        let target = dir.join("target-piku");
        let new_bin = dir.join("new-piku");

        let exe = std::env::current_exe().unwrap();
        std::fs::copy(&exe, &target).unwrap();
        std::fs::copy(&exe, &new_bin).unwrap();

        // Append a marker to new_bin so we can verify it was copied
        let mut f = std::fs::OpenOptions::new()
            .append(true)
            .open(&new_bin)
            .unwrap();
        f.write_all(b"\x00PIKU_TEST_MARKER").unwrap();
        drop(f);

        let new_len = std::fs::metadata(&new_bin).unwrap().len();

        do_replace(&new_bin, &target).expect("should replace with real binary");

        let replaced_len = std::fs::metadata(&target).unwrap().len();
        assert_eq!(
            replaced_len, new_len,
            "target should have new binary's size"
        );

        let content = std::fs::read(&target).unwrap();
        assert!(
            content.ends_with(b"\x00PIKU_TEST_MARKER"),
            "marker should be present"
        );
    }

    // -----------------------------------------------------------------------
    // replace_and_exec error paths (without actual exec)
    // -----------------------------------------------------------------------

    #[test]
    fn replace_and_exec_fails_on_missing_binary() {
        let result = replace_and_exec(Path::new("/no/such/binary"));
        assert!(matches!(
            result,
            Err(SelfUpdateError::NewBinaryNotFound { .. })
        ));
    }

    #[test]
    fn replace_and_exec_fails_on_empty_binary() {
        let dir = tempdir();
        let empty = dir.join("empty");
        std::fs::write(&empty, b"").unwrap();
        let result = replace_and_exec(&empty);
        assert!(matches!(result, Err(SelfUpdateError::NotExecutable)));
    }

    // -----------------------------------------------------------------------
    // was_restarted
    // -----------------------------------------------------------------------

    #[test]
    fn was_restarted_returns_false_when_unset() {
        std::env::remove_var("PIKU_RESTARTED");
        assert!(!was_restarted());
    }

    #[test]
    fn was_restarted_returns_true_and_clears_env_var() {
        std::env::set_var("PIKU_RESTARTED", "1");
        assert!(was_restarted());
        // Env var must be cleared to prevent leaking to child processes
        assert!(
            std::env::var("PIKU_RESTARTED").is_err(),
            "PIKU_RESTARTED should be removed after was_restarted() returns true"
        );
    }
}
