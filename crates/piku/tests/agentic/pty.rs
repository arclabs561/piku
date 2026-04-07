use std::io::{Read, Write};
use std::path::Path;
use std::process::Command;
use std::time::{Duration, Instant};

use super::terminal::TerminalObserver;
use super::types::*;

pub struct PtyHandle {
    _process: rexpect::process::PtyProcess,
    writer: std::fs::File,
    reader: std::fs::File,
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
                    total += n;
                }
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => break,
                Err(_) => break,
            }
        }
        total
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
