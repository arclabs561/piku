use std::io::{Read, Write};
use std::path::Path;
use std::process::Command;
use std::time::{Duration, Instant};

use super::terminal::TerminalObserver;
use super::types::*;

pub struct PtyHandle {
    /// Kept for Drop — rexpect sends SIGTERM then SIGKILL after kill_timeout.
    /// Declared first so it drops last (after writer/reader close).
    _process: rexpect::process::PtyProcess,
    writer: std::fs::File,
    reader: std::fs::File,
    /// Raw bytes captured since last clear -- used to extract response text.
    pub raw_capture: Vec<u8>,
}

impl Drop for PtyHandle {
    fn drop(&mut self) {
        // Send Ctrl-D (EOF) to trigger piku's clean shutdown before the PTY
        // process is killed. Without this, SIGTERM arrives mid-render and can
        // leave zombie child processes (e.g. in-flight ollama requests).
        let _ = self.writer.write_all(b"\x04");
        let _ = self.writer.flush();
        std::thread::sleep(Duration::from_millis(300));
        // _process drops here via struct field order, triggering rexpect's kill loop.
    }
}

impl PtyHandle {
    pub fn spawn(
        workspace: &Path,
        piku_bin: &Path,
        provider_label: &str,
        model: &str,
        env_pairs: Vec<(String, String)>,
    ) -> Self {
        let mut cmd = Command::new("sh");
        cmd.arg("-c");
        let inner_cmd = format!(
            "cd {} && {} --provider {} --model {}",
            shell_escape(&workspace.to_string_lossy()),
            piku_bin.display(),
            provider_label,
            model,
        );
        cmd.arg(&inner_cmd);
        cmd.env_clear();
        for (k, v) in &env_pairs {
            cmd.env(k, v);
        }

        let mut process = rexpect::process::PtyProcess::new(cmd).expect("failed to spawn piku");
        process.set_kill_timeout(Some(5_000));

        // Set PTY window size so crossterm::terminal::size() returns correct dims.
        {
            let pty_fd = process.get_file_handle().expect("pty fd for winsize");
            set_pty_winsize(&pty_fd, 40, 120);
        }

        let writer = process.get_file_handle().expect("writer handle");
        let reader = process.get_file_handle().expect("reader handle");

        // Set reader to non-blocking
        use nix::fcntl::{fcntl, FcntlArg, OFlag};
        let flags = fcntl(&reader, FcntlArg::F_GETFL).expect("F_GETFL");
        fcntl(
            &reader,
            FcntlArg::F_SETFL(OFlag::from_bits_truncate(flags) | OFlag::O_NONBLOCK),
        )
        .expect("F_SETFL O_NONBLOCK");

        Self {
            _process: process,
            writer,
            reader,
            raw_capture: Vec::new(),
        }
    }

    pub fn send_bytes(&mut self, bytes: &[u8]) {
        let _ = self.writer.write_all(bytes);
        let _ = self.writer.flush();
    }

    pub fn send_str(&mut self, s: &str) {
        self.send_bytes(s.as_bytes());
    }

    pub fn send_line(&mut self, s: &str) {
        self.send_str(s);
        self.send_bytes(b"\r");
    }

    pub fn execute_action(&mut self, action: &Action, observer: &mut TerminalObserver) {
        match action {
            Action::Type(c) => {
                let mut buf = [0u8; 4];
                let bytes = c.encode_utf8(&mut buf);
                self.send_bytes(bytes.as_bytes());
                self.settle(observer, Duration::from_millis(30));
            }
            Action::Key(key) => {
                self.send_bytes(key.as_bytes());
                let settle = match key {
                    SpecialKey::Tab => Duration::from_millis(100),
                    SpecialKey::Enter => Duration::from_millis(50),
                    _ => Duration::from_millis(30),
                };
                self.settle(observer, settle);
            }
            Action::Observe => {
                self.drain(observer);
            }
            Action::Wait(d) => {
                std::thread::sleep(*d);
                self.drain(observer);
            }
            Action::TypeString { text, delay_ms } => {
                for c in text.chars() {
                    let mut buf = [0u8; 4];
                    let bytes = c.encode_utf8(&mut buf);
                    self.send_bytes(bytes.as_bytes());
                    std::thread::sleep(Duration::from_millis(*delay_ms));
                    self.drain(observer);
                }
            }
            Action::Submit(s) => {
                self.send_line(s);
                self.settle(observer, Duration::from_millis(50));
            }
        }
    }

    pub fn drain(&mut self, observer: &mut TerminalObserver) -> usize {
        let mut buf = [0u8; 4096];
        let mut total = 0;
        loop {
            match self.reader.read(&mut buf) {
                Ok(0) => break,
                Ok(n) => {
                    observer.process(&buf[..n]);
                    self.raw_capture.extend_from_slice(&buf[..n]);
                    total += n;
                }
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => break,
                Err(_) => break,
            }
        }
        total
    }

    /// Clear the raw capture buffer.
    pub fn clear_capture(&mut self) {
        self.raw_capture.clear();
    }

    /// Extract text content from captured raw bytes by stripping ANSI escape
    /// sequences and filtering thinking indicator noise.
    pub fn captured_text(&self) -> String {
        strip_ansi_bytes(&self.raw_capture)
    }

    pub fn settle(&mut self, observer: &mut TerminalObserver, max_wait: Duration) {
        let start = Instant::now();
        loop {
            let n = self.drain(observer);
            if n == 0 || start.elapsed() >= max_wait {
                break;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    pub fn wait_for_ready(
        &mut self,
        observer: &mut TerminalObserver,
        timeout: Duration,
    ) -> ScreenSnapshot {
        let deadline = Instant::now() + timeout;
        loop {
            self.drain(observer);
            let snap = observer.snapshot();
            if snap.is_ready() {
                return snap;
            }
            if Instant::now() >= deadline {
                eprintln!(
                    "[pty] ready-wait timed out after {timeout:?} \
                     (cursor_visible={}, cursor={:?}, cursor_row={:?}, \
                     non_empty_rows={})",
                    snap.cursor_visible,
                    snap.cursor,
                    snap.input_row(),
                    snap.rows.iter().filter(|r| !r.trim().is_empty()).count(),
                );
                return snap;
            }
            std::thread::sleep(Duration::from_millis(100));
        }
    }
}

fn shell_escape(s: &str) -> String {
    format!("'{}'", s.replace('\'', "'\\''"))
}

/// Set PTY window size via ioctl(TIOCSWINSZ).
#[allow(unsafe_code)]
fn set_pty_winsize(file: &std::fs::File, rows: u16, cols: u16) {
    use std::os::unix::io::AsRawFd;
    #[cfg(target_os = "macos")]
    const TIOCSWINSZ: libc::c_ulong = 0x80087467;
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
        libc::ioctl(file.as_raw_fd(), TIOCSWINSZ, &ws);
    }
}

/// Strip ANSI escape sequences from raw bytes, returning plain text.
/// Filters out thinking indicator frames and progressive prompt redraws.
fn strip_ansi_bytes(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        let b = bytes[i];
        if b == b'\x1b' {
            i += 1;
            if i >= bytes.len() {
                break;
            }
            match bytes[i] {
                b'[' => {
                    i += 1;
                    while i < bytes.len() {
                        let c = bytes[i];
                        i += 1;
                        if c.is_ascii_alphabetic() || c == b'~' {
                            break;
                        }
                    }
                }
                b']' => {
                    i += 1;
                    while i < bytes.len() {
                        if bytes[i] == b'\x07' || bytes[i] == b'\x1b' {
                            i += 1;
                            break;
                        }
                        i += 1;
                    }
                }
                _ => {
                    i += 1;
                }
            }
        } else if b == b'\r' {
            i += 1;
        } else if b == b'\n' {
            out.push('\n');
            i += 1;
        } else if b == b'\t' {
            out.push(' ');
            i += 1;
        } else if b < 0x20 && b != b'\n' {
            i += 1;
        } else if b < 0x80 {
            out.push(b as char);
            i += 1;
        } else {
            let remaining = &bytes[i..];
            match std::str::from_utf8(remaining) {
                Ok(s) => {
                    if let Some(c) = s.chars().next() {
                        out.push(c);
                        i += c.len_utf8();
                    } else {
                        i += 1;
                    }
                }
                Err(e) => {
                    let valid = e.valid_up_to();
                    if valid > 0 {
                        let s = std::str::from_utf8(&bytes[i..i + valid]).unwrap();
                        if let Some(c) = s.chars().next() {
                            out.push(c);
                            i += c.len_utf8();
                        } else {
                            i += 1;
                        }
                    } else {
                        i += 1;
                    }
                }
            }
        }
    }

    // Post-process: filter noise, collapse blanks
    let mut result = String::new();
    let mut nl_count = 0;
    for line in out.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            nl_count += 1;
            if nl_count <= 1 {
                result.push('\n');
            }
            continue;
        }
        nl_count = 0;
        // Skip thinking indicator lines
        if trimmed.contains("thinking\u{2026}") || trimmed.contains("thinking...") {
            continue;
        }
        // Skip progressive prompt redraws (multiple ❯ on one line)
        if trimmed.matches('\u{276F}').count() > 1 {
            continue;
        }
        result.push_str(trimmed);
        result.push('\n');
    }
    result
}
