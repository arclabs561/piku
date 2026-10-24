//! `piku dev`: rebuild and restart the TUI on source and settings changes.
//!
//! The supervisor stays out of raw mode. It watches the workspace, runs one
//! `cargo build -p piku` at a time into a log file, then signals the running
//! child with `SIGUSR1`. The child handles that signal at the next safe point,
//! saves its session, writes the prompt draft, and exits with
//! [`RELOAD_EXIT_CODE`]. The supervisor restores the terminal and starts the
//! new binary on the same session.
//!
//! See `docs/design/hot-dev-reload.md` for the process topology, the rejected
//! alternatives, and the gates.

use std::collections::{BTreeMap, VecDeque};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant, SystemTime};

/// Exit code the child uses to ask for a restart after a successful rebuild.
pub(crate) const RELOAD_EXIT_CODE: i32 = 75;

/// Set for the supervised child only.
const DEV_ENV: &str = "PIKU_DEV";

const POLL: Duration = Duration::from_millis(250);
const DEBOUNCE: Duration = Duration::from_millis(300);
const RATE_WINDOW: Duration = Duration::from_secs(60);
const MAX_RELOADS: usize = 6;
const CRASH_WINDOW: Duration = Duration::from_secs(2);
const CRASH_LIMIT: u32 = 2;

static RELOAD_REQUESTED: AtomicBool = AtomicBool::new(false);
static STOP_REQUESTED: AtomicBool = AtomicBool::new(false);

// ---------------------------------------------------------------------------
// Child side: signal, draft, and reload timing
// ---------------------------------------------------------------------------

/// Install the reload signal handler when the supervisor set [`DEV_ENV`].
///
/// Returns true when dev mode is active. The handler only stores a flag; it
/// never writes, locks, or allocates, so it is safe to run from a signal.
pub(crate) fn install_reload_handler() -> bool {
    if std::env::var_os(DEV_ENV).is_none() {
        return false;
    }
    install_signal(libc::SIGUSR1, || {
        RELOAD_REQUESTED.store(true, Ordering::Relaxed);
    });
    true
}

/// Whether a reload was requested. Checked at the turn boundary and once per
/// input poll tick.
pub(crate) fn reload_requested() -> bool {
    RELOAD_REQUESTED.load(Ordering::Relaxed)
}

/// Latest status written by the supervisor, rendered on the idle prompt row.
pub(crate) fn status() -> Option<String> {
    let text = std::fs::read_to_string(state_dir().join("status")).ok()?;
    let text = text.trim();
    (!text.is_empty()).then_some(text.to_string())
}

#[derive(serde::Serialize, serde::Deserialize)]
struct Draft {
    session_id: String,
    text: String,
}

/// Persist the in-progress prompt so the restarted process can restore it.
///
/// The file is private (0600) because a draft can hold pasted secrets, and it
/// is written through a rename so a crash cannot leave half a draft behind.
pub(crate) fn write_draft(session_id: &str, text: &str) -> std::io::Result<()> {
    write_draft_at(&state_dir(), session_id, text)
}

fn write_draft_at(dir: &Path, session_id: &str, text: &str) -> std::io::Result<()> {
    std::fs::create_dir_all(dir)?;
    let path = dir.join("draft.json");
    let temporary = dir.join("draft.json.tmp");
    let body = serde_json::to_vec(&Draft {
        session_id: session_id.to_string(),
        text: text.to_string(),
    })
    .map_err(std::io::Error::other)?;

    let mut options = std::fs::OpenOptions::new();
    options.write(true).create(true).truncate(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut file = options.open(&temporary)?;
    file.write_all(&body)?;
    file.sync_all()?;
    drop(file);
    std::fs::rename(temporary, path)
}

/// Read and remove the draft for `session_id`. A draft written for another
/// session is left alone.
pub(crate) fn take_draft(session_id: &str) -> Option<String> {
    take_draft_at(&state_dir(), session_id)
}

fn take_draft_at(dir: &Path, session_id: &str) -> Option<String> {
    let path = dir.join("draft.json");
    let body = std::fs::read(&path).ok()?;
    let draft: Draft = serde_json::from_slice(&body).ok()?;
    if draft.session_id != session_id {
        return None;
    }
    let _ = std::fs::remove_file(&path);
    Some(draft.text)
}

/// Local state for dev mode: drafts, receipts, the last good binary, and the
/// build log. `PIKU_DEV_STATE_DIR` exists for the supervisor's own tests.
fn state_dir() -> PathBuf {
    if let Some(path) = std::env::var_os("PIKU_DEV_STATE_DIR") {
        return PathBuf::from(path);
    }
    let base = std::env::var_os("XDG_STATE_HOME").map_or_else(
        || {
            std::env::var_os("HOME").map_or_else(
                || PathBuf::from(".local/state"),
                |home| PathBuf::from(home).join(".local/state"),
            )
        },
        PathBuf::from,
    );
    base.join("piku").join("dev")
}

// ---------------------------------------------------------------------------
// Supervisor side
// ---------------------------------------------------------------------------

/// Watch, rebuild, and restart until the child exits on its own or the
/// supervisor receives an interrupt.
///
/// # Errors
///
/// Returns the first failure that leaves nothing runnable: an unresolvable
/// workspace, or an initial build that does not produce a binary. Later build
/// failures keep the previous child running and are recorded instead.
pub fn run(release: bool, web: bool) -> anyhow::Result<()> {
    let root = workspace_root()?;
    let watch_root = watch_root(&root);
    let binary = target_binary(&root, release);
    let state = state_dir();
    std::fs::create_dir_all(&state)?;

    println!(
        "[dev] workspace {} ({} profile)",
        watch_root.display(),
        if release { "release" } else { "debug" }
    );
    println!("[dev] build log {}", state.join("build.log").display());
    write_status(&state, "building");
    build(&root, release, &state)?;
    write_status(&state, "ready");
    let last_good = state.join("last-good-piku");
    copy_binary(&binary, &last_good)?;

    let mut web_child = if web {
        println!("[dev] web log {}", state.join("web.log").display());
        Some(spawn_web(&binary, &state)?)
    } else {
        None
    };

    let saved_termios = save_termios();
    install_stop_handler();

    let mut watch = WatchSet::capture(&watch_root);
    let mut guard = CrashGuard::new(CRASH_LIMIT, CRASH_WINDOW);
    let mut limiter = ReloadLimiter::new(RATE_WINDOW, MAX_RELOADS);
    let mut active = binary.clone();

    let outcome = loop {
        let mut child = match spawn_child(&active) {
            Ok(child) => child,
            Err(error) => break Err(error.into()),
        };
        let started = Instant::now();
        let mut pending: Option<Instant> = None;

        let status = loop {
            if STOP_REQUESTED.load(Ordering::Relaxed) {
                let _ = child.kill();
                let _ = child.wait();
                break None;
            }
            if watch.changed(&watch_root) {
                pending = Some(Instant::now());
            }
            if pending.is_some_and(|since| since.elapsed() >= DEBOUNCE)
                && limiter.allow(Instant::now())
            {
                pending = None;
                write_status(&state, "building");
                match build(&root, release, &state) {
                    Ok(()) => {
                        copy_binary(&binary, &last_good)?;
                        record(&state, "reload", "built");
                        write_status(&state, "reloading");
                        if web {
                            if let Some(mut old) = web_child.take() {
                                let _ = old.kill();
                                let _ = old.wait();
                            }
                            web_child = spawn_web(&binary, &state).ok();
                        }
                        request_reload(&child);
                    }
                    Err(error) => {
                        record(&state, "build-failed", &error.to_string());
                        write_status(&state, "build failed");
                    }
                }
            }
            match child.try_wait() {
                Ok(Some(status)) => break Some(status),
                Ok(None) => std::thread::sleep(POLL),
                Err(error) => {
                    record(&state, "wait-failed", &error.to_string());
                    let _ = child.kill();
                    let _ = child.wait();
                    break None;
                }
            }
        };

        restore_terminal(saved_termios.as_ref());

        let Some(status) = status else {
            break Ok(());
        };
        let code = status.code();
        if code == Some(0) {
            break Ok(());
        }
        if code == Some(RELOAD_EXIT_CODE) {
            continue;
        }
        if guard.record(started.elapsed(), code) == Verdict::Revert {
            record(&state, "crash", "reverted to last good binary");
            active.clone_from(&last_good);
        } else {
            active.clone_from(&binary);
        }
    };

    if let Some(mut web) = web_child {
        let _ = web.kill();
        let _ = web.wait();
    }
    restore_terminal(saved_termios.as_ref());
    outcome
}

/// Repository root, derived from the compiled crate location so the
/// supervisor does not depend on the caller's working directory.
/// The watched tree. Tests point this at a fixture; production uses the
/// workspace root.
fn watch_root(root: &Path) -> PathBuf {
    std::env::var_os("PIKU_DEV_WATCH_ROOT").map_or_else(|| root.to_path_buf(), PathBuf::from)
}

fn write_status(state: &Path, text: &str) {
    let _ = std::fs::write(state.join("status"), format!("{text}\n"));
}

fn spawn_web(binary: &Path, state: &Path) -> std::io::Result<Child> {
    let mut options = std::fs::OpenOptions::new();
    options.create(true).append(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let log = options.open(state.join("web.log"))?;
    Command::new(binary)
        .arg("web")
        .stdout(Stdio::from(log.try_clone()?))
        .stderr(Stdio::from(log))
        .spawn()
}

fn workspace_root() -> anyhow::Result<PathBuf> {
    let crate_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let root = crate_dir.parent().and_then(Path::parent).ok_or_else(|| {
        anyhow::anyhow!("cannot resolve workspace root from {}", crate_dir.display())
    })?;
    Ok(root.to_path_buf())
}

fn target_binary(root: &Path, release: bool) -> PathBuf {
    let target =
        std::env::var_os("CARGO_TARGET_DIR").map_or_else(|| root.join("target"), PathBuf::from);
    target_binary_in(&target, release)
}

fn target_binary_in(target: &Path, release: bool) -> PathBuf {
    let profile = if release { "release" } else { "debug" };
    target.join(profile).join("piku")
}

fn spawn_child(binary: &Path) -> std::io::Result<Child> {
    let mut command = Command::new(binary);
    command
        .env(DEV_ENV, "1")
        .stdin(Stdio::inherit())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit());
    command.spawn()
}

fn request_reload(child: &Child) {
    let pid = i32::try_from(child.id()).unwrap_or(-1);
    #[allow(unsafe_code)]
    unsafe {
        libc::kill(pid, libc::SIGUSR1);
    }
}

fn build(root: &Path, release: bool, state: &Path) -> anyhow::Result<()> {
    let log_path = state.join("build.log");
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create(true).truncate(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let log = options.open(&log_path)?;
    let mut command = if let Some(script) = std::env::var_os("PIKU_DEV_BUILD") {
        let mut command = Command::new(script);
        command.arg(root);
        command
    } else {
        let mut command = Command::new("cargo");
        command.arg("build").arg("-p").arg("piku").current_dir(root);
        if release {
            command.arg("--release");
        }
        command
    };
    let status = command
        .stdout(Stdio::from(log.try_clone()?))
        .stderr(Stdio::from(log))
        .status()?;
    anyhow::ensure!(
        status.success(),
        "cargo build failed; see {}",
        log_path.display()
    );
    Ok(())
}

fn copy_binary(from: &Path, to: &Path) -> std::io::Result<()> {
    // `copy` preserves the executable bit, which is what the restart needs.
    std::fs::copy(from, to).map(|_| ())
}

fn record(state: &Path, kind: &str, detail: &str) {
    let now = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map_or(0, |value| value.as_secs());
    let line = serde_json::json!({ "at": now, "kind": kind, "detail": detail });
    let _ = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(state.join("reloads.jsonl"))
        .and_then(|mut file| writeln!(file, "{line}"));
}

// ---------------------------------------------------------------------------
// Change detection
// ---------------------------------------------------------------------------

/// A bounded snapshot of the files that require a rebuild.
struct WatchSet {
    entries: BTreeMap<PathBuf, (Option<SystemTime>, u64)>,
}

impl WatchSet {
    fn capture(root: &Path) -> Self {
        let mut entries = BTreeMap::new();
        for base in [root.join("crates")] {
            for entry in walkdir::WalkDir::new(base)
                .into_iter()
                .filter_entry(|entry| !is_ignored(entry.path()))
                .filter_map(Result::ok)
            {
                if entry.file_type().is_file() && is_watched_source(entry.path()) {
                    insert(&mut entries, entry.path());
                }
            }
        }
        for path in [
            root.join("Cargo.toml"),
            root.join("Cargo.lock"),
            root.join(".piku").join("settings.toml"),
            global_settings_path(),
        ] {
            if path.is_file() {
                insert(&mut entries, &path);
            }
        }
        Self { entries }
    }

    /// Recapture and report whether anything changed since the last call.
    fn changed(&mut self, root: &Path) -> bool {
        let next = Self::capture(root);
        let changed = next.entries != self.entries;
        *self = next;
        changed
    }
}

fn insert(entries: &mut BTreeMap<PathBuf, (Option<SystemTime>, u64)>, path: &Path) {
    if let Ok(metadata) = std::fs::metadata(path) {
        entries.insert(
            path.to_path_buf(),
            (metadata.modified().ok(), metadata.len()),
        );
    }
}

fn is_ignored(path: &Path) -> bool {
    matches!(
        path.file_name().and_then(std::ffi::OsStr::to_str),
        Some("target" | "node_modules" | ".git" | ".agents")
    )
}

fn is_watched_source(path: &Path) -> bool {
    matches!(
        path.extension().and_then(std::ffi::OsStr::to_str),
        Some("rs" | "toml" | "js" | "css" | "html")
    )
}

fn global_settings_path() -> PathBuf {
    let base = std::env::var_os("XDG_CONFIG_HOME").map_or_else(
        || {
            std::env::var_os("HOME").map_or_else(
                || PathBuf::from(".config"),
                |home| PathBuf::from(home).join(".config"),
            )
        },
        PathBuf::from,
    );
    base.join("piku").join("settings.toml")
}

// ---------------------------------------------------------------------------
// Restart policy
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Verdict {
    Restart,
    Revert,
}

/// Revert after repeated fast crashes so a broken build cannot trap the
/// operator in a restart loop.
struct CrashGuard {
    limit: u32,
    window: Duration,
    fast_failures: u32,
}

impl CrashGuard {
    fn new(limit: u32, window: Duration) -> Self {
        Self {
            limit,
            window,
            fast_failures: 0,
        }
    }

    fn record(&mut self, elapsed: Duration, code: Option<i32>) -> Verdict {
        if code == Some(0) {
            self.fast_failures = 0;
            return Verdict::Restart;
        }
        if elapsed >= self.window {
            self.fast_failures = 0;
            return Verdict::Restart;
        }
        self.fast_failures += 1;
        if self.fast_failures >= self.limit {
            self.fast_failures = 0;
            Verdict::Revert
        } else {
            Verdict::Restart
        }
    }
}

/// Coalesced rebuilds still cannot outrun a save storm; cap them per minute.
struct ReloadLimiter {
    window: Duration,
    max: usize,
    recent: VecDeque<Instant>,
}

impl ReloadLimiter {
    fn new(window: Duration, max: usize) -> Self {
        Self {
            window,
            max,
            recent: VecDeque::new(),
        }
    }

    fn allow(&mut self, now: Instant) -> bool {
        while self
            .recent
            .front()
            .is_some_and(|earlier| now.duration_since(*earlier) >= self.window)
        {
            self.recent.pop_front();
        }
        if self.recent.len() >= self.max {
            return false;
        }
        self.recent.push_back(now);
        true
    }
}

// ---------------------------------------------------------------------------
// Signals and terminal
// ---------------------------------------------------------------------------

fn install_signal(signal: i32, handler: impl Fn() + Send + Sync + 'static) {
    #[allow(unsafe_code)]
    unsafe {
        let _ = signal_hook_registry::register(signal, handler);
    }
}

fn install_stop_handler() {
    for signal in [libc::SIGINT, libc::SIGTERM, libc::SIGHUP] {
        install_signal(signal, || {
            STOP_REQUESTED.store(true, Ordering::Relaxed);
        });
    }
}

#[cfg(unix)]
fn save_termios() -> Option<libc::termios> {
    #[allow(unsafe_code)]
    unsafe {
        let mut termios = std::mem::zeroed();
        if libc::tcgetattr(libc::STDIN_FILENO, &raw mut termios) == 0 {
            Some(termios)
        } else {
            None
        }
    }
}

#[cfg(not(unix))]
fn save_termios() -> Option<()> {
    None
}

#[cfg(unix)]
fn restore_terminal(saved: Option<&libc::termios>) {
    let mut stdout = std::io::stdout();
    let _ = stdout.write_all(b"\x1b[r\x1b[?25h\x1b[0m\n");
    let _ = stdout.flush();
    if let Some(termios) = saved {
        #[allow(unsafe_code)]
        unsafe {
            libc::tcsetattr(libc::STDIN_FILENO, libc::TCSANOW, termios);
        }
    }
}

#[cfg(not(unix))]
fn restore_terminal(_saved: Option<&()>) {}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn draft_round_trips_and_is_session_bound() {
        let dir = tempfile::tempdir().unwrap();
        write_draft_at(dir.path(), "session-a", "half typed").unwrap();
        assert_eq!(
            take_draft_at(dir.path(), "session-b").as_deref(),
            None,
            "a draft must not cross sessions"
        );
        assert_eq!(
            take_draft_at(dir.path(), "session-a").as_deref(),
            Some("half typed")
        );
        assert_eq!(
            take_draft_at(dir.path(), "session-a"),
            None,
            "taking a draft consumes it"
        );
    }

    #[test]
    fn watch_set_detects_source_changes_and_ignores_target() {
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("crates/piku/src");
        std::fs::create_dir_all(&source).unwrap();
        let target = dir.path().join("target/debug");
        std::fs::create_dir_all(&target).unwrap();
        std::fs::write(source.join("lib.rs"), "one").unwrap();

        let mut watch = WatchSet::capture(dir.path());
        assert!(!watch.changed(dir.path()));

        std::fs::write(source.join("lib.rs"), "two").unwrap();
        assert!(watch.changed(dir.path()));

        std::fs::write(target.join("artifact"), "ignored").unwrap();
        assert!(!watch.changed(dir.path()));
    }

    #[test]
    fn crash_guard_reverts_after_repeated_fast_failures() {
        let mut guard = CrashGuard::new(2, CRASH_WINDOW);
        assert_eq!(
            guard.record(Duration::from_millis(100), Some(1)),
            Verdict::Restart
        );
        assert_eq!(
            guard.record(Duration::from_millis(100), Some(1)),
            Verdict::Revert
        );
        // A long run clears the count.
        assert_eq!(
            guard.record(Duration::from_secs(10), Some(1)),
            Verdict::Restart
        );
        assert_eq!(
            guard.record(Duration::from_millis(100), Some(1)),
            Verdict::Restart
        );
    }

    #[test]
    fn reload_limiter_caps_a_save_storm() {
        let mut limiter = ReloadLimiter::new(RATE_WINDOW, 2);
        let start = Instant::now();
        assert!(limiter.allow(start));
        assert!(limiter.allow(start + Duration::from_secs(1)));
        assert!(!limiter.allow(start + Duration::from_secs(2)));
        assert!(limiter.allow(start + RATE_WINDOW));
    }

    #[test]
    fn target_binary_follows_the_selected_profile() {
        let target = Path::new("/work/piku/target");
        assert_eq!(target_binary_in(target, false), target.join("debug/piku"));
        assert_eq!(target_binary_in(target, true), target.join("release/piku"));
    }
}
