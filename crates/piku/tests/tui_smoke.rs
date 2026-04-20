//! Cheap PTY smoke tests for the interactive TUI path.
//!
//! These drive the real `piku` binary over a PTY to exercise code paths
//! that pure unit tests can't reach — tokio runtime setup, LocalSet/
//! spawn_local, the concurrent keypress reader that fires on every turn.
//!
//! Contract: no LLM, no API calls, no external services. A fake API key
//! lets piku enter the TUI; the LLM request will fail but we only care
//! about what happens *before* that — the turn-start spawning that panicked
//! in tui_repl.rs:1118 (block_in_place inside a LocalSet task).
//!
//! Run:
//!   cargo build -p piku
//!   cargo test --test tui_smoke

use std::io::{Read, Write};
use std::path::PathBuf;
use std::process::Command;
use std::time::{Duration, Instant};

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
}

impl Pty {
    fn spawn() -> Self {
        Self::spawn_in(std::env::temp_dir().as_path())
    }

    fn spawn_in(cwd: &std::path::Path) -> Self {
        let mut cmd = Command::new(piku_binary());
        cmd.current_dir(cwd)
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
        }
    }

    fn send(&mut self, bytes: &[u8]) {
        // Tolerant of closed PTY — writes after child exit return EIO.
        let _ = self.writer.write_all(bytes);
        let _ = self.writer.flush();
    }

    fn drain(&mut self) {
        let mut chunk = [0u8; 4096];
        loop {
            match self.reader.read(&mut chunk) {
                Ok(0) => {
                    self.eof = true;
                    break;
                }
                Ok(n) => self.buf.extend_from_slice(&chunk[..n]),
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
}

impl Drop for Pty {
    fn drop(&mut self) {
        // Detached drop — rexpect's kill loop can hang on zombies.
        let _ = self.send(b"\x04"); // Ctrl-D
    }
}

/// Regression: submitting any input used to panic at tui_repl.rs:1118
/// with "can call blocking only when running on the multi-threaded runtime"
/// because `block_in_place` was called inside a spawn_local task.
///
/// This test spawns piku, types a prompt, hits Enter, and verifies that
/// the process does NOT emit a Rust panic header before the LLM call
/// (which will legitimately fail with a fake API key).
#[test]
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

/// Regression guard for the raw-mode leak noted in the coverage audit:
/// `keypress_handle.abort()` skips the task's raw-mode cleanup, so a
/// second readline after a cancelled turn can receive raw input.
///
/// We submit two prompts back-to-back and check the second echoes as
/// normal dim text. If raw mode leaked, the second `hi` would be
/// interpreted as control bytes and wouldn't echo as characters.
#[test]
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
fn hooks_slash_command_renders() {
    let mut pty = Pty::spawn();
    let ready = pty.wait_for("❯", Duration::from_secs(5));
    assert!(ready, "prompt not reached:\n{}", pty.captured());

    pty.send(b"/hooks\r");
    pty.wait(Duration::from_secs(1));

    let out = pty.captured();
    assert!(!out.contains("panicked at"), "panic on /hooks:\n{out}");
}

/// SIGTERM should trigger the terminal-restore signal handler, which
/// writes "\x1b[r\x1b[?25h\n" (reset scroll region + show cursor) to
/// stdout before re-raising the signal with the default disposition.
/// Without the handler, a kill(1) leaves DECSTBM set and the cursor
/// hidden for the user's shell.
///
/// Must opt into signal handlers explicitly via PIKU_INSTALL_SIGNAL_HANDLERS=1.
/// Production's main() sets this by default; tests leave it unset because
/// the nextest/rexpect harness delivers spurious SIGTERM to the child
/// during startup, tripping the handler before the test can interact.
#[test]
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

    let mut pty = Pty {
        _proc: proc,
        writer,
        reader,
        buf: Vec::new(),
        eof: false,
    };

    // Small wait for piku to install its handler + print setup.
    pty.wait(Duration::from_millis(500));

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
/// Exercises the CancelFlag + keypress reader teardown path.
#[test]
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

    let mut pty = Pty {
        _proc: proc,
        writer,
        reader,
        buf: Vec::new(),
        eof: false,
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
