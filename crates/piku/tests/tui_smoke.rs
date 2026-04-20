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
        let mut cmd = Command::new(piku_binary());
        cmd.env_clear()
            .env("PATH", std::env::var("PATH").unwrap_or_default())
            .env("HOME", std::env::var("HOME").unwrap_or_default())
            .env("TERM", "xterm-256color")
            // Fake key so piku enters TUI. No request will succeed; we
            // never wait for LLM output.
            .env("OPENROUTER_API_KEY", "sk-or-fake-smoke-test")
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
        self.writer.write_all(bytes).unwrap();
        self.writer.flush().unwrap();
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
