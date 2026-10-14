#![allow(
    clippy::doc_markdown,
    clippy::must_use_candidate,
    clippy::map_unwrap_or
)]

/// Explicitly approved developer self-update support for piku.
///
/// With an operator-selected build root, handles the "piku builds itself" use case:
///
///   1. piku runs `cargo build --release` via the bash tool
///   2. cargo writes a new binary to `<approved root>/release/piku`
///   3. piku detects the new binary is different from the running one
///   4. piku saves session state to disk (already happens per-turn)
///   5. piku atomically replaces itself on disk: do_replace(new_binary)
///   6. piku exec(2)s the new binary via exec_self(): same PID, same TTY
///   7. on the already-running TUI path, the new piku also receives the session
///      ID and continues it; launch-turn replacement restarts the same argv
///
/// The two steps are intentionally split so `do_replace` can be unit-tested
/// without actually replacing the running test binary.
use std::path::{Path, PathBuf};

#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum SelfUpdateError {
    #[error("developer self-update is disabled; select --self-update-root explicitly")]
    Disabled,

    #[error("invalid self-update build root {path}: {reason}")]
    InvalidRoot { path: PathBuf, reason: String },

    #[error("self-update candidate could not be inspected at {path}: {source}")]
    CandidateIo {
        path: PathBuf,
        source: std::io::Error,
    },

    #[error("self-update candidate is no longer ready or does not match the approved path")]
    CandidateNotReady,

    #[error("could not determine current executable path: {0}")]
    CurrentExe(#[source] std::io::Error),

    #[error("new binary not found at {path}: {source}")]
    NewBinaryNotFound {
        path: PathBuf,
        source: std::io::Error,
    },

    #[error("new binary must be a nonempty regular executable file")]
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
// Developer approval and build detection
// ---------------------------------------------------------------------------

/// Immutable operator approval for local developer replacement.
///
/// Default is disabled. Project configuration, the working directory, Cargo
/// output, and environment variables cannot enable this policy. This is an
/// application admission rule, not an OS sandbox or binary authenticity check.
#[derive(Clone, Debug, Default)]
pub struct SelfUpdatePolicy {
    approved_root: Option<ApprovedRoot>,
    baseline: Option<std::time::SystemTime>,
}

#[derive(Clone, Debug)]
struct ApprovedRoot {
    path: PathBuf,
    directory: std::sync::Arc<std::fs::File>,
}

/// Pin the canonical directory without following a replaced path component.
fn open_root_directory(path: &Path) -> std::io::Result<std::fs::File> {
    use nix::fcntl::{open, openat, OFlag};
    use nix::sys::stat::Mode;
    let flags = OFlag::O_RDONLY | OFlag::O_DIRECTORY | OFlag::O_NOFOLLOW | OFlag::O_CLOEXEC;
    let mut directory = open(Path::new("/"), flags, Mode::empty())?;
    for component in path.components() {
        if let std::path::Component::Normal(name) = component {
            directory = openat(&directory, Path::new(name), flags, Mode::empty())?;
        }
    }
    Ok(directory.into())
}

/// Snapshot the already-open approved source; never reopen its pathname.
fn snapshot_source(
    mut source: std::fs::File,
    target: &Path,
) -> Result<tempfile::NamedTempFile, SelfUpdateError> {
    let parent = target.parent().ok_or_else(|| {
        SelfUpdateError::Replace(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "target has no parent directory",
        ))
    })?;
    let mut snapshot = tempfile::Builder::new()
        .prefix(".piku-update-")
        .tempfile_in(parent)
        .map_err(SelfUpdateError::Replace)?;
    std::io::copy(&mut source, snapshot.as_file_mut()).map_err(SelfUpdateError::Replace)?;
    let permissions = source
        .metadata()
        .map_err(SelfUpdateError::Replace)?
        .permissions();
    snapshot
        .as_file()
        .set_permissions(permissions)
        .map_err(SelfUpdateError::Replace)?;
    snapshot
        .as_file()
        .sync_all()
        .map_err(SelfUpdateError::Replace)?;
    Ok(snapshot)
}

impl SelfUpdatePolicy {
    /// Capture an explicitly selected absolute Cargo target root once at launch.
    /// Read-only launches disable replacement even if a root was supplied.
    pub fn from_operator_root(
        root: Option<&Path>,
        read_only: bool,
    ) -> Result<Self, SelfUpdateError> {
        let Some(root) = root.filter(|_| !read_only) else {
            return Ok(Self::default());
        };
        if !root.is_absolute() {
            return Err(SelfUpdateError::InvalidRoot {
                path: root.to_path_buf(),
                reason: "must be an absolute directory path".to_string(),
            });
        }
        let approved_root = root
            .canonicalize()
            .map_err(|error| SelfUpdateError::InvalidRoot {
                path: root.to_path_buf(),
                reason: error.to_string(),
            })?;
        if !approved_root.is_dir() {
            return Err(SelfUpdateError::InvalidRoot {
                path: root.to_path_buf(),
                reason: "must be a directory".to_string(),
            });
        }
        let directory =
            open_root_directory(&approved_root).map_err(|error| SelfUpdateError::InvalidRoot {
                path: root.to_path_buf(),
                reason: error.to_string(),
            })?;
        let baseline = running_mtime().ok_or_else(|| {
            SelfUpdateError::CurrentExe(std::io::Error::other(
                "could not capture executable modification time",
            ))
        })?;
        Ok(Self {
            approved_root: Some(ApprovedRoot {
                path: approved_root,
                directory: std::sync::Arc::new(directory),
            }),
            baseline: Some(baseline),
        })
    }

    /// Open the candidate relative to the pinned approved root. Both release
    /// and piku must be real entries, never symlinks. Missing builds are normal.
    fn open_candidate(&self) -> Result<Option<std::fs::File>, SelfUpdateError> {
        use nix::fcntl::{openat, OFlag};
        use nix::sys::stat::Mode;
        let Some(root) = &self.approved_root else {
            return Ok(None);
        };
        let path = root.path.join("release/piku");
        let inspect_error = |source: nix::errno::Errno| SelfUpdateError::CandidateIo {
            path: path.clone(),
            source: source.into(),
        };
        let directory = match openat(
            root.directory.as_ref(),
            "release",
            OFlag::O_RDONLY | OFlag::O_DIRECTORY | OFlag::O_NOFOLLOW | OFlag::O_CLOEXEC,
            Mode::empty(),
        ) {
            Ok(directory) => directory,
            Err(nix::errno::Errno::ENOENT) => return Ok(None),
            Err(error) => return Err(inspect_error(error)),
        };
        let source: std::fs::File = match openat(
            &directory,
            "piku",
            OFlag::O_RDONLY | OFlag::O_NONBLOCK | OFlag::O_NOFOLLOW | OFlag::O_CLOEXEC,
            Mode::empty(),
        ) {
            Ok(source) => source.into(),
            Err(nix::errno::Errno::ENOENT) => return Ok(None),
            Err(error) => return Err(inspect_error(error)),
        };
        let meta = source
            .metadata()
            .map_err(|source| SelfUpdateError::CandidateIo {
                path: path.clone(),
                source,
            })?;
        if !meta.is_file() || meta.len() == 0 {
            return Err(SelfUpdateError::NotExecutable);
        }
        use std::os::unix::fs::PermissionsExt;
        if meta.permissions().mode() & 0o111 == 0 {
            return Err(SelfUpdateError::NotExecutable);
        }
        let modified = meta
            .modified()
            .map_err(|source| SelfUpdateError::CandidateIo { path, source })?;
        Ok(self
            .baseline
            .is_some_and(|baseline| modified > baseline)
            .then_some(source))
    }

    /// Inspect only the approved candidate using metadata from its open handle.
    pub fn check_for_update(&self) -> Result<Option<PathBuf>, SelfUpdateError> {
        Ok(self.open_candidate()?.and_then(|_| {
            self.approved_root
                .as_ref()
                .map(|root| root.path.join("release/piku"))
        }))
    }

    /// Cargo text is a readiness hint only within an already approved root.
    pub fn detect_self_build(
        &self,
        output: &str,
        exit_success: bool,
    ) -> Result<Option<PathBuf>, SelfUpdateError> {
        if !exit_success
            || !output.contains("Finished")
            || !(output.contains("Compiling piku v") || output.contains("target/release/piku"))
        {
            return Ok(None);
        }
        self.check_for_update()
    }

    /// Recheck admission immediately before the filesystem replacement.
    /// No automatic caller may use the lower-level replacement primitives.
    pub fn replace_and_exec(
        &self,
        candidate: &Path,
        extra_env: &[(&str, &str)],
    ) -> Result<(), SelfUpdateError> {
        if self.approved_root.is_none() {
            return Err(SelfUpdateError::Disabled);
        }
        let root = self
            .approved_root
            .as_ref()
            .ok_or(SelfUpdateError::Disabled)?;
        if candidate != root.path.join("release/piku") {
            return Err(SelfUpdateError::CandidateNotReady);
        }
        let source = self
            .open_candidate()?
            .ok_or(SelfUpdateError::CandidateNotReady)?;
        let current_exe = std::env::current_exe().map_err(SelfUpdateError::CurrentExe)?;
        let snapshot = snapshot_source(source, &current_exe)?;
        do_replace(snapshot.path(), &current_exe)?;
        // Close/remove the snapshot before exec, which never runs Rust Drop.
        drop(snapshot);
        exec_self(&current_exe, extra_env)
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
    use std::sync::atomic::{AtomicU64, Ordering};

    fn tempdir() -> PathBuf {
        static NEXT_TEMP_ID: AtomicU64 = AtomicU64::new(0);

        let base = std::env::temp_dir().join(format!(
            "piku_su_{}_{}_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.subsec_nanos())
                .unwrap_or(0),
            std::process::id(),
            NEXT_TEMP_ID.fetch_add(1, Ordering::Relaxed),
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
        assert!(SelfUpdatePolicy::default()
            .detect_self_build("Finished release [optimized]", false)
            .unwrap()
            .is_none());
    }

    #[test]
    fn detect_self_build_false_when_no_finished() {
        assert!(SelfUpdatePolicy::default()
            .detect_self_build("Compiling piku v0.1.0", true)
            .unwrap()
            .is_none());
    }

    #[test]
    fn detect_self_build_false_for_other_crate_only() {
        // Building piku-api produces "Finished" but not "Compiling piku v"
        // This should NOT trigger because it only has Finished, not piku-specific marker
        // BUT: "Finished" alone still passes the `has_finished` check.
        // The guard is: require `has_finished`. If binary isn't newer → None.
        // So this correctly returns None if binary is not newer.
        let result = SelfUpdatePolicy::default().detect_self_build(
            "Compiling piku-api v0.1.0\nFinished release [optimized]",
            true,
        );
        // Returns None because target/release/piku doesn't exist / isn't newer in test env
        assert!(result.unwrap().is_none());
    }

    #[test]
    fn detect_self_build_some_only_with_approved_root() {
        let dir = tempdir();
        let policy = SelfUpdatePolicy::from_operator_root(Some(&dir), false).unwrap();
        let release = dir.join("release");
        std::fs::create_dir_all(&release).unwrap();
        let candidate = release.join("piku");
        write_binary(&candidate, b"approved build");
        let output = "Compiling piku v0.1.0\nFinished release [optimized]";
        assert!(SelfUpdatePolicy::default()
            .detect_self_build(output, true)
            .unwrap()
            .is_none());
        assert_eq!(
            policy.detect_self_build(output, true).unwrap(),
            Some(candidate.canonicalize().unwrap())
        );
    }

    #[test]
    fn cargo_hints_are_checked_only_inside_approved_root() {
        let root = tempdir();
        std::fs::create_dir_all(root.join("release")).unwrap();
        write_binary(&root.join("release/piku"), b"approved");
        let mut policy = SelfUpdatePolicy::from_operator_root(Some(&root), false).unwrap();
        policy.baseline = Some(std::time::UNIX_EPOCH);
        for (output, success) in [
            ("Finished release", true),
            ("Compiling piku v0.1.0", true),
            ("Compiling piku-api v0.1.0\nFinished release", true),
            ("Compiling piku-tools v0.1.0\nFinished release", true),
            ("Compiling piku v0.1.0\nFinished release", false),
        ] {
            assert!(
                policy.detect_self_build(output, success).unwrap().is_none(),
                "{output}"
            );
        }
        assert!(policy
            .detect_self_build("Compiling piku v0.1.0\nFinished release", true)
            .unwrap()
            .is_some());
    }

    #[test]
    fn read_only_disables_explicit_root() {
        let dir = tempdir();
        let policy = SelfUpdatePolicy::from_operator_root(Some(&dir), true).unwrap();
        assert!(policy.check_for_update().unwrap().is_none());
        assert!(matches!(
            policy.replace_and_exec(Path::new("unused"), &[]),
            Err(SelfUpdateError::Disabled)
        ));
    }

    #[test]
    fn relative_or_file_root_is_rejected() {
        assert!(matches!(
            SelfUpdatePolicy::from_operator_root(Some(Path::new("target")), false),
            Err(SelfUpdateError::InvalidRoot { .. })
        ));
        let dir = tempdir();
        let file = dir.join("file");
        write_binary(&file, b"content");
        assert!(matches!(
            SelfUpdatePolicy::from_operator_root(Some(&file), false),
            Err(SelfUpdateError::InvalidRoot { .. })
        ));
    }

    #[cfg(unix)]
    #[test]
    fn escaped_candidate_is_rejected_before_detection_and_replacement() {
        let dir = tempdir();
        let outside = tempdir().join("outside-piku");
        write_binary(&outside, b"outside");
        let release = dir.join("release");
        std::fs::create_dir_all(&release).unwrap();
        std::os::unix::fs::symlink(&outside, release.join("piku")).unwrap();
        let policy = SelfUpdatePolicy::from_operator_root(Some(&dir), false).unwrap();
        assert!(matches!(
            policy.check_for_update(),
            Err(SelfUpdateError::CandidateIo { .. })
        ));
        assert!(matches!(
            policy.replace_and_exec(&dir.canonicalize().unwrap().join("release/piku"), &[]),
            Err(SelfUpdateError::CandidateIo { .. })
        ));
    }

    #[test]
    fn snapshot_uses_open_source_after_path_is_swapped_to_outside_symlink() {
        let root = tempdir();
        let release = root.join("release");
        std::fs::create_dir_all(&release).unwrap();
        let candidate = release.join("piku");
        write_binary(&candidate, b"approved source bytes");
        let policy = SelfUpdatePolicy::from_operator_root(Some(&root), false).unwrap();
        // Test this race independently of wall-clock mtime precision.
        let mut policy = policy;
        policy.baseline = Some(std::time::UNIX_EPOCH);
        let source = policy.open_candidate().unwrap().unwrap();
        let outside = tempdir().join("outside");
        write_binary(&outside, b"outside source bytes");
        std::fs::rename(&candidate, release.join("piku-original")).unwrap();
        std::os::unix::fs::symlink(&outside, &candidate).unwrap();
        let target = tempdir().join("copy-target");
        write_binary(&target, b"old target");
        let snapshot = snapshot_source(source, &target).unwrap();
        do_replace(snapshot.path(), &target).unwrap();
        assert_eq!(std::fs::read(&target).unwrap(), b"approved source bytes");
        assert!(matches!(
            policy.check_for_update(),
            Err(SelfUpdateError::CandidateIo { .. })
        ));
    }

    #[test]
    fn release_directory_symlink_is_rejected_and_root_identity_is_pinned() {
        let root = tempdir();
        let policy = SelfUpdatePolicy::from_operator_root(Some(&root), false).unwrap();
        let moved = root.with_extension("moved");
        std::fs::rename(&root, &moved).unwrap();
        std::fs::create_dir_all(&root).unwrap();
        let outside = tempdir();
        std::fs::create_dir_all(outside.join("release")).unwrap();
        write_binary(&outside.join("release/piku"), b"outside");
        std::os::unix::fs::symlink(&outside, root.join("release")).unwrap();
        // The old approved directory has no release entry; the new pathname
        // and its outside symlink cannot supply a candidate to the pinned root.
        assert!(policy.check_for_update().unwrap().is_none());
        std::os::unix::fs::symlink(outside.join("release"), moved.join("release")).unwrap();
        assert!(matches!(
            policy.check_for_update(),
            Err(SelfUpdateError::CandidateIo { .. })
        ));
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
    #[serial_test::serial(piku_restart_env)]
    fn was_restarted_returns_false_when_unset() {
        std::env::remove_var("PIKU_RESTARTED");
        assert!(!was_restarted());
    }

    #[test]
    #[serial_test::serial(piku_restart_env)]
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
