//! Cheap PTY smoke tests for the interactive TUI path.
//!
//! These drive the real `piku` binary over a PTY to exercise code paths
//! that pure unit tests can't reach — tokio runtime setup, `LocalSet`/
//! `spawn_local`, the concurrent keypress reader that fires on every turn.
//!
//! Contract: no LLM, no API calls, no external services. A fake API key
//! lets piku enter the TUI; the LLM request will fail but we only care
//! about what happens *before* that — the turn-start spawning that panicked
//! in `tui_repl.rs:1118` (`block_in_place` inside a `LocalSet` task).
//!
//! These are `#[ignore]`d + `#[serial]` and run isolated (`scripts/ci.sh pty`)
//! so the main suite stays fast; under full-workspace concurrency their PTY
//! teardown stalls.
//!
//! Run:
//!   cargo build -p piku
//!   cargo test --test `tui_smoke` -- --ignored

// Pedantic lints that are noise in this PTY harness, not in production:
// `use` lives inside test fns to keep platform glue local; `Winsize`/`ws_*`
// mirror C `struct winsize`; `_proc` is rexpect's own field we read to get the
// PTY fd. Production code stays under full pedantic-deny.
#![allow(
    clippy::items_after_statements,
    clippy::struct_field_names,
    clippy::used_underscore_binding
)]

use std::io::{Read, Write};
use std::path::PathBuf;
use std::process::Command;
use std::time::{Duration, Instant};

use serial_test::serial;

// The shared "seeing" model — a vt100 parser that renders the PTY byte stream
// into the grid a human would see. Same observer the agentic judge loop uses,
// so these smoke tests and the judge assert on one definition of "the screen".
#[path = "agentic/screen.rs"]
mod screen;
use screen::{ScreenObserver, ScreenSnapshot};

/// The fixed grid these PTY tests render into. The observer and the PTY winsize
/// are both built from this one constant, so the parsed grid can never disagree
/// with the size piku actually drew into.
const TEST_ROWS: u16 = 24;
const TEST_COLS: u16 = 80;

fn piku_binary() -> PathBuf {
    let exe = std::env::current_exe().unwrap();
    let profile_dir = exe.parent().unwrap().parent().unwrap();
    for candidate in [
        profile_dir.join("piku"),
        profile_dir.parent().unwrap().join("debug").join("piku"),
        profile_dir.parent().unwrap().join("release").join("piku"),
    ] {
        if candidate.exists() {
            return candidate;
        }
    }
    panic!("piku binary not found — run `cargo build -p piku` first");
}

struct Pty {
    _proc: rexpect::process::PtyProcess,
    writer: std::fs::File,
    reader: std::fs::File,
    buf: Vec<u8>,
    eof: bool,
    /// Persistent renderer: every byte drained from the PTY is also fed here so
    /// `screen()` returns what the user currently sees, not the raw log.
    observer: ScreenObserver,
}

impl Pty {
    fn spawn() -> Self {
        Self::spawn_in(std::env::temp_dir().as_path())
    }

    fn spawn_in(cwd: &std::path::Path) -> Self {
        Self::spawn_with_args(cwd, std::iter::empty::<&str>())
    }

    fn spawn_with_args<I, S>(cwd: &std::path::Path, args: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<std::ffi::OsStr>,
    {
        let mut cmd = Command::new(piku_binary());
        cmd.args(args)
            .current_dir(cwd)
            .env_clear()
            .env("PATH", std::env::var("PATH").unwrap_or_default())
            .env("HOME", std::env::var("HOME").unwrap_or_default())
            .env("TERM", "xterm-256color")
            // Fake key so piku enters TUI. No request will succeed; we
            // never wait for LLM output.
            .env("OPENROUTER_API_KEY", "sk-or-fake-smoke-test")
            // Disable terminal-restoring signal handlers. Under nextest,
            // each test runs in its own process group, and nextest forwards
            // SIGTERM to grandchildren on timeout/cancellation. Our handler
            // honors the signal by exiting promptly, which from the test's
            // POV looks like piku dying during startup. Production users
            // keep the handler. See signal-hook research 2026-04-20.
            .env("PIKU_NO_SIGNAL_HANDLERS", "1")
            .env("PIKU_RESTARTED", "1");

        let mut proc = rexpect::process::PtyProcess::new(cmd).expect("spawn piku");
        proc.set_kill_timeout(Some(3_000));

        let writer = proc.get_file_handle().expect("pty writer");
        let reader = proc.get_file_handle().expect("pty reader");

        // Fix the terminal size before piku's first `term_size()` so the layout
        // it draws and the grid the observer parses are the same shape. piku
        // reads its dimensions from the PTY ioctl (TIOCGWINSZ), not from
        // LINES/COLUMNS, so this is the only knob that controls the render.
        set_winsize(&writer, TEST_ROWS, TEST_COLS);

        use nix::fcntl::{fcntl, FcntlArg, OFlag};
        let flags = fcntl(&reader, FcntlArg::F_GETFL).unwrap();
        fcntl(
            &reader,
            FcntlArg::F_SETFL(OFlag::from_bits_truncate(flags) | OFlag::O_NONBLOCK),
        )
        .unwrap();

        Self {
            _proc: proc,
            writer,
            reader,
            buf: Vec::new(),
            eof: false,
            observer: ScreenObserver::new(TEST_ROWS, TEST_COLS),
        }
    }

    fn send(&mut self, bytes: &[u8]) {
        // Tolerant of closed PTY — writes after child exit return EIO.
        let _ = self.writer.write_all(bytes);
        let _ = self.writer.flush();
    }

    /// Current rendered screen — what the user sees right now. Drains any
    /// pending PTY output into the observer first so the snapshot is fresh.
    fn screen(&mut self) -> ScreenSnapshot {
        self.drain();
        self.observer.snapshot()
    }

    /// Poll until the rendered screen satisfies `pred` or the timeout elapses.
    /// Returns the last snapshot either way so a failing test can print what
    /// the user actually saw. This is the screen-level analogue of `wait_for`,
    /// which only checks the raw byte log.
    fn wait_until(
        &mut self,
        mut pred: impl FnMut(&ScreenSnapshot) -> bool,
        timeout: Duration,
    ) -> (bool, ScreenSnapshot) {
        let start = Instant::now();
        loop {
            let snap = self.screen();
            if pred(&snap) {
                return (true, snap);
            }
            if start.elapsed() >= timeout {
                return (false, snap);
            }
            std::thread::sleep(Duration::from_millis(50));
        }
    }

    fn drain(&mut self) {
        let mut chunk = [0u8; 4096];
        loop {
            match self.reader.read(&mut chunk) {
                Ok(0) => {
                    self.eof = true;
                    break;
                }
                Ok(n) => {
                    self.buf.extend_from_slice(&chunk[..n]);
                    self.observer.process(&chunk[..n]);
                }
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => break,
                Err(e) => {
                    if e.raw_os_error() == Some(libc::EIO) {
                        self.eof = true;
                    }
                    break;
                }
            }
        }
    }

    fn captured(&self) -> String {
        String::from_utf8_lossy(&self.buf).into_owned()
    }

    fn wait_for(&mut self, needle: &str, timeout: Duration) -> bool {
        let start = Instant::now();
        while start.elapsed() < timeout {
            self.drain();
            if self.captured().contains(needle) {
                return true;
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        false
    }

    fn wait(&mut self, dur: Duration) {
        let start = Instant::now();
        while start.elapsed() < dur {
            self.drain();
            std::thread::sleep(Duration::from_millis(25));
        }
    }

    /// Drive piku to a clean exit and wait for EOF. Called explicitly by tests
    /// that leave piku in a state where `Drop`'s lone Ctrl-D would be ignored
    /// (e.g. mid-turn), so the test cannot hang waiting on a still-running process.
    fn exit_cleanly(&mut self) {
        self.send(b"exit\r");
        let start = std::time::Instant::now();
        while !self.eof && start.elapsed() < Duration::from_secs(3) {
            self.drain();
            std::thread::sleep(Duration::from_millis(50));
        }
    }
}

impl Drop for Pty {
    fn drop(&mut self) {
        // Detached drop — rexpect's kill loop can hang on zombies.
        let () = self.send(b"\x04"); // Ctrl-D
    }
}

/// Regression: submitting any input used to panic at `tui_repl.rs:1118`
/// with "can call blocking only when running on the multi-threaded runtime"
/// because `block_in_place` was called inside a `spawn_local` task.
///
/// This test spawns piku, types a prompt, hits Enter, and verifies that
/// the process does NOT emit a Rust panic header before the LLM call
/// (which will legitimately fail with a fake API key).
#[test]
#[serial]
#[ignore = "PTY smoke: slow/fragile under concurrent-binary load; run isolated via `scripts/ci.sh pty`"]
fn submit_does_not_panic_on_turn_start() {
    let mut pty = Pty::spawn();

    // Wait for the TUI to paint its prompt. The ❯ glyph is the input row
    // marker. If we don't see it in 5s, the binary is broken in another way.
    let ready = pty.wait_for("❯", Duration::from_secs(5));
    assert!(
        ready,
        "piku never reached the prompt; output was:\n{}",
        pty.captured()
    );

    // Type "hi" and submit. This fires the turn-start path that spawns
    // the concurrent keypress reader via spawn_local.
    pty.send(b"hi\r");

    // Give the runtime time to spawn tasks and for any panic to propagate
    // to stderr. The panic is synchronous in the async task body, so it
    // surfaces quickly.
    pty.wait(Duration::from_secs(2));

    let out = pty.captured();

    // The precise failure mode we're guarding against. Any Rust panic is a
    // regression — but this message is the signature of the original bug.
    assert!(
        !out.contains("panicked at"),
        "piku panicked on turn start:\n{out}"
    );
    assert!(
        !out.contains("can call blocking only when running on the multi-threaded runtime"),
        "block_in_place panic regressed:\n{out}"
    );
}

/// Ctrl-D on an empty prompt should exit cleanly — no panic, no hang.
/// Startup/shutdown sanity.
#[test]
#[serial]
#[ignore = "PTY smoke: slow/fragile under concurrent-binary load; run isolated via `scripts/ci.sh pty`"]
fn ctrl_d_on_empty_prompt_exits_cleanly() {
    let mut pty = Pty::spawn();
    let ready = pty.wait_for("❯", Duration::from_secs(5));
    assert!(ready, "prompt not reached:\n{}", pty.captured());

    // Ctrl-D with no input — should exit.
    pty.send(b"\x04");

    // Wait for EOF or up to 3s.
    let start = Instant::now();
    while !pty.eof && start.elapsed() < Duration::from_secs(3) {
        pty.drain();
        std::thread::sleep(Duration::from_millis(50));
    }

    let out = pty.captured();
    assert!(!out.contains("panicked at"), "panic on Ctrl-D:\n{out}");
    assert!(
        pty.eof,
        "piku did not exit within 3s on Ctrl-D; output was:\n{out}"
    );
}

/// Bare --read-only should start the interactive TUI in read-only mode,
/// not fail as a headless-only flag.
#[test]
#[serial]
#[ignore = "PTY smoke: slow/fragile under concurrent-binary load; run isolated via `scripts/ci.sh pty`"]
fn read_only_flag_starts_read_only_tui() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let blocked_path = tmp.path().join("should-not-exist");
    let mut pty = Pty::spawn_with_args(tmp.path(), ["--read-only"]);
    let banner = pty.wait_for("read-only", Duration::from_secs(5));
    assert!(banner, "read-only banner not reached:\n{}", pty.captured());

    let ready = pty.wait_for("❯", Duration::from_secs(5));
    assert!(
        ready,
        "read-only TUI prompt not reached:\n{}",
        pty.captured()
    );

    let out = pty.captured();
    assert!(
        !out.contains("panicked at"),
        "panic on read-only TUI:\n{out}"
    );

    pty.send(b"!touch should-not-exist\r");
    let blocked = pty.wait_for("shell commands are disabled", Duration::from_secs(2));
    assert!(
        blocked,
        "read-only shell command was not blocked:\n{}",
        pty.captured()
    );
    assert!(
        !blocked_path.exists(),
        "read-only shell escape created {}",
        blocked_path.display()
    );
}

/// Regression guard for the raw-mode leak noted in the coverage audit:
/// `keypress_handle.abort()` skips the task's raw-mode cleanup, so a
/// second readline after a cancelled turn can receive raw input.
///
/// We submit two prompts back-to-back and check the second echoes as
/// normal dim text. If raw mode leaked, the second `hi` would be
/// interpreted as control bytes and wouldn't echo as characters.
#[test]
#[serial]
#[ignore = "PTY smoke: slow/fragile under concurrent-binary load; run isolated via `scripts/ci.sh pty`"]
fn two_consecutive_prompts_echo_normally() {
    let mut pty = Pty::spawn();
    let ready = pty.wait_for("❯", Duration::from_secs(5));
    assert!(ready, "prompt not reached:\n{}", pty.captured());

    // First prompt — kicks off a turn that will error out on fake API key.
    pty.send(b"first\r");
    // Wait for the error + return to prompt.
    let back_to_prompt = pty.wait_for("HTTP error 401", Duration::from_secs(5));
    assert!(
        back_to_prompt,
        "first turn did not produce expected error:\n{}",
        pty.captured()
    );
    // Let piku finish re-rendering the idle prompt and restart its readline.
    pty.wait(Duration::from_millis(500));

    // Clear what we've seen; focus on the second turn's echo.
    let before_second = pty.buf.len();
    // Type character-by-character with small pauses so each echo has time
    // to render — matches how a human types.
    for ch in b"second" {
        pty.send(&[*ch]);
        std::thread::sleep(Duration::from_millis(30));
        pty.drain();
    }
    pty.send(b"\r");
    pty.wait(Duration::from_secs(2));

    let second_segment = String::from_utf8_lossy(&pty.buf[before_second..]).into_owned();

    // Echo of the second prompt: characters must appear in the segment.
    // If raw mode leaked, each byte would be consumed by crossterm event
    // handling instead of being echoed as typed characters.
    assert!(
        second_segment.contains("second"),
        "second prompt did not echo (raw mode leak?):\n{second_segment}"
    );
    assert!(
        !second_segment.contains("panicked at"),
        "panic on second turn:\n{second_segment}"
    );
}

/// The /help slash command should render without panicking. /help has
/// zero prior test coverage per the audit.
#[test]
#[serial]
#[ignore = "PTY smoke: slow/fragile under concurrent-binary load; run isolated via `scripts/ci.sh pty`"]
fn help_slash_command_renders() {
    let mut pty = Pty::spawn();
    let ready = pty.wait_for("❯", Duration::from_secs(5));
    assert!(ready, "prompt not reached:\n{}", pty.captured());

    pty.send(b"/help\r");
    pty.wait(Duration::from_secs(1));

    let out = pty.captured();
    assert!(!out.contains("panicked at"), "panic on /help:\n{out}");
    // Help output should mention at least one known command.
    assert!(
        out.contains("/help") || out.contains("Commands") || out.contains("/permissions"),
        "/help did not render recognizable output:\n{out}"
    );
}

/// /permissions should render without panicking.
#[test]
#[serial]
#[ignore = "PTY smoke: slow/fragile under concurrent-binary load; run isolated via `scripts/ci.sh pty`"]
fn permissions_slash_command_renders() {
    let mut pty = Pty::spawn();
    let ready = pty.wait_for("❯", Duration::from_secs(5));
    assert!(ready, "prompt not reached:\n{}", pty.captured());

    pty.send(b"/permissions\r");
    pty.wait(Duration::from_secs(1));

    let out = pty.captured();
    assert!(
        !out.contains("panicked at"),
        "panic on /permissions:\n{out}"
    );
}

/// /hooks should render without panicking.
#[test]
#[serial]
#[ignore = "PTY smoke: slow/fragile under concurrent-binary load; run isolated via `scripts/ci.sh pty`"]
fn hooks_slash_command_renders() {
    let mut pty = Pty::spawn();
    let ready = pty.wait_for("❯", Duration::from_secs(5));
    assert!(ready, "prompt not reached:\n{}", pty.captured());

    pty.send(b"/hooks\r");
    pty.wait(Duration::from_secs(1));

    let out = pty.captured();
    assert!(!out.contains("panicked at"), "panic on /hooks:\n{out}");
}

/// /hooks should reflect a hooks.json that was written into `.piku/`.
/// Tests the `HookRegistry` load path through the real binary — complements
/// the in-process unit tests in hooks.rs.
#[test]
#[serial]
#[ignore = "PTY smoke: slow/fragile under concurrent-binary load; run isolated via `scripts/ci.sh pty`"]
fn hooks_config_loads_from_project_file() {
    let tmp = tempfile::tempdir().expect("tempdir");
    std::fs::create_dir_all(tmp.path().join(".piku")).unwrap();
    let hooks_json = serde_json::json!({
        "PreToolUse": [{
            "matcher": "bash",
            "hooks": [{
                "command": "echo hook-loaded-ok",
            }]
        }]
    });
    std::fs::write(
        tmp.path().join(".piku").join("hooks.json"),
        serde_json::to_string_pretty(&hooks_json).unwrap(),
    )
    .unwrap();

    let mut pty = Pty::spawn_in(tmp.path());
    let ready = pty.wait_for("❯", Duration::from_secs(5));
    assert!(ready, "prompt not reached:\n{}", pty.captured());

    pty.send(b"/hooks\r");
    pty.wait(Duration::from_secs(1));

    let out = pty.captured();
    assert!(!out.contains("panicked at"), "panic on /hooks:\n{out}");
    // /hooks summary should mention PreToolUse or the registered event in some form.
    assert!(
        out.contains("PreToolUse") || out.contains("bash") || out.contains("pre_tool_use"),
        "/hooks did not reflect the loaded config:\n{out}"
    );
}

/// Set PTY window size via TIOCSWINSZ. Crossterm reads the terminal
/// dimensions from the PTY ioctl (not $LINES / $COLUMNS), and sending
/// the ioctl also delivers SIGWINCH to the foreground process group.
#[allow(unsafe_code)]
fn set_pty_winsize(fd: &std::fs::File, rows: u16, cols: u16) {
    use std::os::unix::io::AsRawFd;
    #[cfg(target_os = "macos")]
    const TIOCSWINSZ: libc::c_ulong = 0x8008_7467;
    #[cfg(target_os = "linux")]
    const TIOCSWINSZ: libc::c_ulong = 0x5414;

    #[repr(C)]
    struct Winsize {
        ws_row: u16,
        ws_col: u16,
        ws_xpixel: u16,
        ws_ypixel: u16,
    }
    let ws = Winsize {
        ws_row: rows,
        ws_col: cols,
        ws_xpixel: 0,
        ws_ypixel: 0,
    };
    unsafe {
        libc::ioctl(fd.as_raw_fd(), TIOCSWINSZ, &ws);
    }
}

/// SIGWINCH (PTY resize) during a session should not panic piku. Does NOT
/// verify piku correctly relayouts — that would need VT100 parsing. This
/// is a regression guard for the crash class: crossterm's SIGWINCH handler
/// interacting with our own signal-hook registration, or rows=0 edge
/// cases in `setup_layout` when resize fires mid-startup.
#[test]
#[serial]
#[ignore = "PTY smoke: slow/fragile under concurrent-binary load; run isolated via `scripts/ci.sh pty`"]
fn pty_resize_does_not_panic() {
    let mut pty = Pty::spawn();
    let ready = pty.wait_for("❯", Duration::from_secs(5));
    assert!(ready, "prompt not reached:\n{}", pty.captured());

    // Set initial size, then shrink, then grow. Each ioctl delivers SIGWINCH.
    let fd = pty._proc.get_file_handle().expect("pty fd");
    set_pty_winsize(&fd, 30, 100);
    std::thread::sleep(Duration::from_millis(100));
    set_pty_winsize(&fd, 15, 50);
    std::thread::sleep(Duration::from_millis(100));
    set_pty_winsize(&fd, 50, 200);
    pty.wait(Duration::from_secs(1));

    // Still alive and not panicked.
    let out = pty.captured();
    assert!(!out.contains("panicked at"), "panic on resize:\n{out}");
    assert!(!pty.eof, "piku exited on SIGWINCH:\n{out}");
}

/// SIGTERM should trigger the terminal-restore signal handler, which
/// writes "\x1b[r\x1b[?25h\n" (reset scroll region + show cursor) to
/// stdout before re-raising the signal with the default disposition.
/// Without the handler, a kill(1) leaves DECSTBM set and the cursor
/// hidden for the user's shell.
///
/// Must opt into signal handlers explicitly via `PIKU_INSTALL_SIGNAL_HANDLERS=1`.
/// Production's `main()` sets this by default; tests leave it unset because
/// the nextest/rexpect harness delivers spurious SIGTERM to the child
/// during startup, tripping the handler before the test can interact.
#[test]
#[serial]
#[ignore = "PTY smoke: slow/fragile under concurrent-binary load; run isolated via `scripts/ci.sh pty`"]
fn sigterm_restores_terminal_before_exit() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let mut cmd = Command::new(piku_binary());
    cmd.current_dir(tmp.path())
        .env_clear()
        .env("PATH", std::env::var("PATH").unwrap_or_default())
        .env("HOME", std::env::var("HOME").unwrap_or_default())
        .env("TERM", "xterm-256color")
        .env("OPENROUTER_API_KEY", "sk-or-fake-smoke-test")
        .env("PIKU_INSTALL_SIGNAL_HANDLERS", "1")
        .env("PIKU_RESTARTED", "1");

    let mut proc = rexpect::process::PtyProcess::new(cmd).expect("spawn piku");
    proc.set_kill_timeout(Some(3_000));
    let writer = proc.get_file_handle().expect("pty writer");
    let reader = proc.get_file_handle().expect("pty reader");
    use nix::fcntl::{fcntl, FcntlArg, OFlag};
    let flags = fcntl(&reader, FcntlArg::F_GETFL).unwrap();
    fcntl(
        &reader,
        FcntlArg::F_SETFL(OFlag::from_bits_truncate(flags) | OFlag::O_NONBLOCK),
    )
    .unwrap();

    set_winsize(&writer, TEST_ROWS, TEST_COLS);
    let mut pty = Pty {
        _proc: proc,
        writer,
        reader,
        buf: Vec::new(),
        eof: false,
        observer: ScreenObserver::new(TEST_ROWS, TEST_COLS),
    };

    // Wait for piku to finish startup — prompt glyph is a reliable marker
    // that the signal handler has been installed (install fires before the
    // prompt renders).
    let ready = pty.wait_for("❯", Duration::from_secs(5));
    assert!(
        ready,
        "piku never started:\n{}",
        String::from_utf8_lossy(&pty.buf)
    );

    let before_signal = pty.buf.len();

    // Send SIGTERM to the child piku process.
    pty._proc
        .signal(nix::sys::signal::Signal::SIGTERM)
        .expect("signal SIGTERM");

    // Wait for exit.
    let start = Instant::now();
    while !pty.eof && start.elapsed() < Duration::from_secs(3) {
        pty.drain();
        std::thread::sleep(Duration::from_millis(25));
    }

    let after = &pty.buf[before_signal..];
    // The handler writes exactly b"\x1b[r\x1b[?25h\n". Looking for the full
    // sequence — `\x1b[?` alone matches many startup mode strings.
    const HANDLER_BYTES: &[u8] = b"\x1b[r\x1b[?25h\n";
    let has_handler_output = after
        .windows(HANDLER_BYTES.len())
        .any(|w| w == HANDLER_BYTES);
    assert!(pty.eof, "piku did not exit after SIGTERM within 3s");
    assert!(
        has_handler_output,
        "SIGTERM handler did not emit terminal-restore bytes:\nlooking for: {:?}\nin: {}",
        String::from_utf8_lossy(HANDLER_BYTES),
        String::from_utf8_lossy(after)
    );
}

/// Ctrl-C mid-turn should cancel the turn and return to the prompt
/// without panicking. This uses a test-local TCP listener that accepts
/// the connection and then hangs, so the LLM call is in-flight long
/// enough for the Ctrl-C to race it mid-turn.
///
/// Exercises the `CancelFlag` + keypress reader teardown path.
#[test]
#[serial]
#[ignore = "PTY smoke: slow/fragile under concurrent-binary load; run isolated via `scripts/ci.sh pty`"]
fn ctrl_c_mid_turn_cancels_cleanly() {
    use std::net::TcpListener;

    // Bind a listener that accepts but never responds. Child piku process
    // connects but hangs reading — the turn stays "in flight".
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let port = listener.local_addr().unwrap().port();
    // Accept in the background; hold the socket so piku blocks on read.
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            // Leak the socket — we want it to stay open and silent.
            std::mem::forget(stream);
        }
    });

    let tmp = tempfile::tempdir().expect("tempdir");
    let mut cmd = Command::new(piku_binary());
    cmd.current_dir(tmp.path())
        .env_clear()
        .env("PATH", std::env::var("PATH").unwrap_or_default())
        .env("HOME", std::env::var("HOME").unwrap_or_default())
        .env("TERM", "xterm-256color")
        .env("OPENROUTER_API_KEY", "sk-or-fake-smoke-test")
        .env("PIKU_BASE_URL", format!("http://127.0.0.1:{port}/v1"))
        .env("PIKU_NO_SIGNAL_HANDLERS", "1")
        .env("PIKU_RESTARTED", "1");

    let mut proc = rexpect::process::PtyProcess::new(cmd).expect("spawn piku");
    proc.set_kill_timeout(Some(3_000));
    let writer = proc.get_file_handle().expect("pty writer");
    let reader = proc.get_file_handle().expect("pty reader");
    use nix::fcntl::{fcntl, FcntlArg, OFlag};
    let flags = fcntl(&reader, FcntlArg::F_GETFL).unwrap();
    fcntl(
        &reader,
        FcntlArg::F_SETFL(OFlag::from_bits_truncate(flags) | OFlag::O_NONBLOCK),
    )
    .unwrap();

    set_winsize(&writer, TEST_ROWS, TEST_COLS);
    let mut pty = Pty {
        _proc: proc,
        writer,
        reader,
        buf: Vec::new(),
        eof: false,
        observer: ScreenObserver::new(TEST_ROWS, TEST_COLS),
    };

    let ready = pty.wait_for("❯", Duration::from_secs(5));
    assert!(ready, "prompt not reached:\n{}", pty.captured());

    pty.send(b"hello\r");
    // Let the turn begin: spinner + in-flight HTTP request to our hanging
    // server. The keypress reader is now live.
    std::thread::sleep(Duration::from_millis(500));

    // Send Ctrl-C.
    pty.send(b"\x03");
    pty.wait(Duration::from_secs(2));

    let out = pty.captured();
    assert!(!out.contains("panicked at"), "panic on Ctrl-C:\n{out}");
    // Process should not have exited — Ctrl-C cancels the turn, not the app.
    assert!(
        !pty.eof,
        "piku exited on Ctrl-C (should only cancel turn):\n{out}"
    );
}

// ── PTY window size ───────────────────────────────────────────────────────────

/// Set a PTY's window size via `ioctl(TIOCSWINSZ)`.
///
/// piku reads its dimensions from the PTY ioctl (`TIOCGWINSZ`), not from the
/// `LINES`/`COLUMNS` environment, so this is what makes the render deterministic.
/// Called once at spawn from the same `(TEST_ROWS, TEST_COLS)` the observer is
/// built with, so the grid piku draws and the grid the parser models agree.
#[allow(unsafe_code)]
fn set_winsize(file: &std::fs::File, rows: u16, cols: u16) {
    use std::os::unix::io::AsRawFd;
    #[cfg(target_os = "macos")]
    const TIOCSWINSZ: libc::c_ulong = 0x8008_7467;
    #[cfg(target_os = "linux")]
    const TIOCSWINSZ: libc::c_ulong = 0x5414;

    #[repr(C)]
    struct Winsize {
        ws_row: u16,
        ws_col: u16,
        ws_xpixel: u16,
        ws_ypixel: u16,
    }
    let ws = Winsize {
        ws_row: rows,
        ws_col: cols,
        ws_xpixel: 0,
        ws_ypixel: 0,
    };
    // SAFETY: TIOCSWINSZ writes a fixed-layout `struct winsize` to a valid PTY fd.
    unsafe {
        libc::ioctl(file.as_raw_fd(), TIOCSWINSZ, &ws);
    }
}

// ===========================================================================
// Rendered-screen TUI QA — assert on what a human SEES
// ---------------------------------------------------------------------------
// These tests drive the real piku binary over a PTY and assert on the rendered
// grid (`pty.screen()`), not the raw byte log. The renderer is the same vt100
// observer the agentic judge loop uses (`agentic/screen.rs`), so "the screen"
// has one definition across the suite.
//
// Why the rendered screen and not raw bytes: a user experiences the grid a
// terminal paints, not the escape sequence. vt100 is faithful enough that it
// *caught* the blank-screen bug these tests now guard against — DECSTBM homes
// the cursor (`ESC [ r` -> xterm `CursorSet(screen, 0, 0, ...)`), so piku's
// reset-region-then-erase wiped the whole frame on a real terminal too. See
// `agentic/screen.rs` for the full rationale.
//
// A fake API key means no network call succeeds; we assert only on the
// presentation that happens before / instead of the LLM response.
// ===========================================================================

/// Wait for piku's prompt to be rendered and return the ready snapshot.
fn wait_ready(pty: &mut Pty) -> ScreenSnapshot {
    let (ok, snap) = pty.wait_until(ScreenSnapshot::is_ready, Duration::from_secs(6));
    assert!(
        ok,
        "prompt never became ready; screen was:\n{}",
        snap.summary(24)
    );
    snap
}

#[test]
#[serial]
#[ignore = "PTY smoke: slow/fragile under concurrent-binary load; run isolated via `scripts/ci.sh pty`"]
fn header_is_pinned_and_shows_version_on_launch() {
    // The header the user sees at startup must carry the piku title, the
    // version, and the provider — and it must survive (not be wiped) once the
    // prompt is ready. This is the regression guard for "typing piku clears my
    // screen": if the frame were wiped, none of these would render.
    let mut pty = Pty::spawn();
    let snap = wait_ready(&mut pty);

    assert!(
        snap.shows("piku"),
        "title should be visible on launch; screen:\n{}",
        snap.summary(24)
    );
    assert!(
        snap.shows(env!("CARGO_PKG_VERSION")),
        "version {} should be visible in the header; screen:\n{}",
        env!("CARGO_PKG_VERSION"),
        snap.summary(24)
    );
    assert!(
        snap.shows("openrouter"),
        "provider should be visible in the header; screen:\n{}",
        snap.summary(24)
    );
    pty.exit_cleanly();
}

#[test]
#[serial]
#[ignore = "PTY smoke: slow/fragile under concurrent-binary load; run isolated via `scripts/ci.sh pty`"]
fn ready_caret_is_green() {
    // The caret signals "ready for input" and must be green — not the old red
    // that lingered for a whole prompt after a failed turn.
    let mut pty = Pty::spawn();
    let snap = wait_ready(&mut pty);

    let caret = snap
        .styled_input_row()
        .and_then(screen::StyledRow::first_glyph_fg);
    assert_eq!(
        caret,
        Some(screen::Color::Green),
        "ready caret should be green; input row was {:?}",
        snap.input_row()
    );
    pty.exit_cleanly();
}

#[test]
#[serial]
#[ignore = "PTY smoke: slow/fragile under concurrent-binary load; run isolated via `scripts/ci.sh pty`"]
fn typing_slash_preserves_the_frame() {
    // A slash prefix used to append an extra menu row while the editor was
    // anchored on the last terminal row, which scrolled the pinned header out
    // of view. Guard the invariant rather than the removed menu: after typing
    // `/`, the header and the footer hint are both still on screen.
    let mut pty = Pty::spawn();
    let initial = wait_ready(&mut pty);
    assert!(
        initial.shows("piku") && initial.shows("/help for commands"),
        "header and footer should render before typing; screen:\n{}",
        initial.summary(24)
    );

    pty.send(b"/");
    let (shown, snap) = pty.wait_until(
        |s| {
            s.shows("piku")
                && s.shows(env!("CARGO_PKG_VERSION"))
                && s.shows("/help for commands")
                && s.input_row().trim_start().ends_with('/')
        },
        Duration::from_secs(3),
    );
    assert!(
        shown,
        "typing a slash should echo into the prompt without scrolling away the \
         header or footer; screen:\n{}",
        snap.summary(24)
    );
    pty.exit_cleanly();
}

#[test]
#[serial]
#[ignore = "PTY smoke: slow/fragile under concurrent-binary load; run isolated via `scripts/ci.sh pty`"]
fn version_command_shows_version_in_tui() {
    // `/version` must print the version into the transcript the user sees.
    let mut pty = Pty::spawn();
    wait_ready(&mut pty);

    pty.send(b"/version\r");
    let want = format!("piku {}", env!("CARGO_PKG_VERSION"));
    let (shown, snap) = pty.wait_until(|s| s.shows(&want), Duration::from_secs(3));
    assert!(
        shown,
        "/version should show {want:?}; screen:\n{}",
        snap.summary(24)
    );
    pty.exit_cleanly();
}

#[test]
#[serial]
#[ignore = "PTY smoke: slow/fragile under concurrent-binary load; run isolated via `scripts/ci.sh pty`"]
fn submit_does_not_blank_the_screen() {
    // The core regression: submitting a prompt used to wipe the frame (DECSTBM
    // homed the cursor, then the editor erased from row 1 down). After the fix
    // the submitted text stays on screen and the frame is not blank.
    let mut pty = Pty::spawn();
    wait_ready(&mut pty);

    pty.send(b"hello there\r");
    // With a fake key the turn errors; we only need piku to have processed the
    // submit and returned to a drawn frame.
    let (settled, snap) = pty.wait_until(
        |s| s.shows("hello there") && s.non_empty_rows().len() >= 2,
        Duration::from_secs(8),
    );
    assert!(
        settled,
        "submitted input should stay visible and the frame should not blank; screen:\n{}",
        snap.summary(24)
    );
    // The header must still be there — proof the submit did not scroll-wipe it.
    assert!(
        snap.shows("piku"),
        "header should survive a submit; screen:\n{}",
        snap.summary(24)
    );
    pty.exit_cleanly();
}
