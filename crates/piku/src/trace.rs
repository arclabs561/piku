#![allow(
    clippy::doc_markdown,
    clippy::must_use_candidate,
    clippy::needless_pass_by_value
)]

/// Session trace writer — appends structured JSONL events to
/// `~/.config/piku/traces/<session-id>.jsonl`.
///
/// Each line is a self-contained JSON object (one event per line).
/// Enabled by default; set `PIKU_NO_TRACE=1` to suppress.
///
/// Event schema:
/// ```json
/// {"ts": 1712345678.123, "event": "tool_start", "tool": "read_file", "id": "tc_1", "input": {...}}
/// {"ts": ..., "event": "tool_end",   "tool": "read_file", "id": "tc_1", "ok": true, "output": "..."}
/// {"ts": ..., "event": "text",       "text": "..."}
/// {"ts": ..., "event": "turn_end",   "iterations": 3, "input_tokens": 1200, "output_tokens": 400}
/// {"ts": ..., "event": "permission_denied", "tool": "bash", "reason": "..."}
/// ```
///
/// The trace file is a complete audit log for a session. One session = one file.
/// Replay scripts can reconstruct exactly what the agent did and at what cost.
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::PathBuf;

pub struct TraceWriter {
    file: Option<File>,
}

impl TraceWriter {
    /// Open (or create) the trace file for `session_id`.
    ///
    /// Returns a no-op writer if `PIKU_NO_TRACE=1` or the file can't be opened.
    pub fn open(traces_dir: &PathBuf, session_id: &str) -> Self {
        if std::env::var("PIKU_NO_TRACE")
            .map(|v| v == "1")
            .unwrap_or(false)
        {
            return Self { file: None };
        }

        if let Err(e) = std::fs::create_dir_all(traces_dir) {
            eprintln!("[piku] trace: could not create traces dir: {e}");
            return Self { file: None };
        }

        let path = traces_dir.join(format!("{session_id}.jsonl"));
        match OpenOptions::new().create(true).append(true).open(&path) {
            Ok(f) => {
                eprintln!("[piku] trace → {}", path.display());
                Self { file: Some(f) }
            }
            Err(e) => {
                eprintln!("[piku] trace: could not open {}: {e}", path.display());
                Self { file: None }
            }
        }
    }

    /// No-op writer (used in tests or when tracing is disabled).
    pub fn disabled() -> Self {
        Self { file: None }
    }

    fn write_event(&mut self, event: serde_json::Value) {
        let Some(f) = self.file.as_mut() else { return };
        let line = serde_json::to_string(&event).unwrap_or_default();
        let _ = writeln!(f, "{line}");
        let _ = f.flush();
    }

    fn now_secs() -> f64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0)
    }

    // ---------------------------------------------------------------------------
    // Event emitters
    // ---------------------------------------------------------------------------

    pub fn tool_start(&mut self, tool_name: &str, tool_id: &str, input: &serde_json::Value) {
        self.write_event(serde_json::json!({
            "ts": Self::now_secs(),
            "event": "tool_start",
            "tool": tool_name,
            "id": tool_id,
            "input": input,
        }));
    }

    pub fn tool_end(&mut self, tool_name: &str, tool_id: &str, output: &str, ok: bool) {
        // Truncate very large outputs (e.g. full file reads) to 2000 chars in the trace.
        // The session JSON already has the full content; trace is for analysis.
        let preview = if output.len() > 2000 {
            format!("{}…[{} chars]", &output[..2000], output.len())
        } else {
            output.to_string()
        };
        self.write_event(serde_json::json!({
            "ts": Self::now_secs(),
            "event": "tool_end",
            "tool": tool_name,
            "id": tool_id,
            "ok": ok,
            "output": preview,
        }));
    }

    pub fn text_chunk(&mut self, text: &str) {
        // Text comes in many tiny chunks — only record non-trivial ones
        // to keep the trace readable (not hundreds of single-char events).
        if text.len() < 20 {
            return;
        }
        self.write_event(serde_json::json!({
            "ts": Self::now_secs(),
            "event": "text",
            "text": text,
        }));
    }

    pub fn turn_end(&mut self, iterations: u32, input_tokens: u32, output_tokens: u32) {
        self.write_event(serde_json::json!({
            "ts": Self::now_secs(),
            "event": "turn_end",
            "iterations": iterations,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }));
    }

    pub fn permission_denied(&mut self, tool_name: &str, reason: &str) {
        self.write_event(serde_json::json!({
            "ts": Self::now_secs(),
            "event": "permission_denied",
            "tool": tool_name,
            "reason": reason,
        }));
    }

    pub fn prompt(&mut self, text: &str) {
        self.write_event(serde_json::json!({
            "ts": Self::now_secs(),
            "event": "prompt",
            "text": text,
        }));
    }
}
