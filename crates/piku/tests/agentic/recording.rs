//! Replayable, privacy-aware evidence for managed TUI evaluations.
//!
//! The `.cast` file is standard asciicast v2 for ordinary terminal playback.
//! The adjacent JSONL trace is Piku-owned evidence: it keeps semantic actions,
//! rendered-screen digests, and judge annotations without turning the cast
//! format into an unportable database.

use std::fs::{create_dir_all, File};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use serde_json::json;
use sha2::{Digest, Sha256};

use super::{Action, ScreenSnapshot};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecordingPaths {
    pub cast: PathBuf,
    pub trace: PathBuf,
}

/// Writes an asciicast v2 replay and a Piku trace in lockstep.
pub struct TuiRecording {
    started: Instant,
    cast: File,
    trace: File,
    record_raw_input_events: bool,
}

impl TuiRecording {
    pub fn create(
        directory: &Path,
        rows: u16,
        cols: u16,
        title: &str,
        record_raw_input_events: bool,
    ) -> io::Result<(Self, RecordingPaths)> {
        create_dir_all(directory)?;
        let paths = RecordingPaths {
            cast: directory.join("session.cast"),
            trace: directory.join("trace.jsonl"),
        };
        let mut cast = File::create(&paths.cast)?;
        let mut trace = File::create(&paths.trace)?;
        writeln!(
            cast,
            "{}",
            json!({
                "version": 2,
                "width": cols,
                "height": rows,
                "title": title,
            })
        )?;
        writeln!(
            trace,
            "{}",
            json!({
                "schema_version": 1,
                "kind": "header",
                "cast": "session.cast",
                "input_capture": {
                    "raw_input_events": if record_raw_input_events {
                        "recorded"
                    } else {
                        "omitted"
                    },
                    // Terminal echo arrives as PTY output, not an asciicast input
                    // event. Omitting raw input events therefore does not make a
                    // cast safe to share with an untrusted evaluator.
                    "echoed_input_in_output": "may_be_present",
                },
            })
        )?;
        cast.flush()?;
        trace.flush()?;
        Ok((
            Self {
                started: Instant::now(),
                cast,
                trace,
                record_raw_input_events,
            },
            paths,
        ))
    }

    fn elapsed_seconds(&self) -> f64 {
        self.started.elapsed().as_secs_f64()
    }

    fn trace(&mut self, value: &serde_json::Value) -> io::Result<()> {
        writeln!(self.trace, "{value}")?;
        self.trace.flush()
    }

    fn cast_event(&mut self, code: &str, data: &str) -> io::Result<()> {
        writeln!(self.cast, "{}", json!([self.elapsed_seconds(), code, data]))?;
        self.cast.flush()
    }

    pub fn input(&mut self, bytes: &[u8]) -> io::Result<()> {
        if self.record_raw_input_events {
            self.cast_event("i", &String::from_utf8_lossy(bytes))?;
        }
        self.trace(&json!({
            "t": self.elapsed_seconds(),
            "kind": "input",
            "byte_count": bytes.len(),
            "raw_input_recorded": self.record_raw_input_events,
        }))
    }

    pub fn output(&mut self, bytes: &[u8]) -> io::Result<()> {
        self.cast_event("o", &String::from_utf8_lossy(bytes))?;
        self.trace(&json!({
            "t": self.elapsed_seconds(),
            "kind": "output",
            "byte_count": bytes.len(),
            "echoed_input_in_output": "may_be_present",
        }))
    }

    pub fn action(&mut self, action: &Action) -> io::Result<()> {
        self.trace(&json!({
            "t": self.elapsed_seconds(),
            "kind": "action",
            "action": action_kind(action),
            "detail": action_detail(action),
        }))
    }

    pub fn observation(&mut self, reason: &str, screen: &ScreenSnapshot) -> io::Result<()> {
        self.trace(&json!({
            "t": self.elapsed_seconds(),
            "kind": "observation",
            "reason": reason,
            "screen": {
                "rows": screen.size.0,
                "columns": screen.size.1,
                "cursor": [screen.cursor.0, screen.cursor.1],
                "cursor_visible": screen.cursor_visible,
                "ready": screen.is_ready(),
                "visible_sha256": format!("{:x}", Sha256::digest(screen.contents.as_bytes())),
            },
        }))
    }

    pub fn annotation(&mut self, label: &str, value: &serde_json::Value) -> io::Result<()> {
        self.trace(&json!({
            "t": self.elapsed_seconds(),
            "kind": "annotation",
            "label": label,
            "value": value,
        }))
    }

    /// Marks the evidence bundle complete after all event writes have been
    /// durably flushed. A judge must reject a trace without this footer.
    pub fn finish(mut self) -> io::Result<()> {
        self.cast.flush()?;
        self.trace(&json!({
            "t": self.elapsed_seconds(),
            "kind": "footer",
            "disposition": "complete",
        }))
    }
}

fn action_kind(action: &Action) -> &'static str {
    match action {
        Action::Type(_) => "type",
        Action::Key(_) => "key",
        Action::Observe => "observe",
        Action::Wait(_) => "wait",
        Action::TypeString { .. } => "type_text",
        Action::Submit(_) => "submit",
    }
}

fn action_detail(action: &Action) -> serde_json::Value {
    match action {
        Action::Type(_) => json!({ "characters": 1 }),
        Action::Key(key) => json!({ "key": key.name() }),
        Action::Observe => json!({}),
        Action::Wait(duration) => json!({ "milliseconds": duration.as_millis() }),
        Action::TypeString { text, delay_ms } => json!({
            "characters": text.chars().count(),
            "delay_ms": delay_ms,
        }),
        Action::Submit(text) => json!({ "characters": text.chars().count() }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn screen(contents: &str) -> ScreenSnapshot {
        ScreenSnapshot {
            contents: contents.to_owned(),
            rows: vec![contents.to_owned()],
            cursor: (0, 0),
            cursor_visible: true,
            styled_rows: vec![],
            size: (24, 80),
        }
    }

    #[test]
    fn replay_is_standard_and_sidecar_omits_input_by_default() {
        let directory = tempdir().unwrap();
        let (mut recording, paths) =
            TuiRecording::create(directory.path(), 24, 80, "TUI evaluation", false).unwrap();
        recording.input(b"secret command\r").unwrap();
        recording.output(b"visible output\r\n").unwrap();
        recording
            .action(&Action::Submit("secret command".to_owned()))
            .unwrap();
        recording
            .observation("after_submit", &screen("visible output"))
            .unwrap();
        recording
            .annotation("judge", &json!({ "finding": "output visible" }))
            .unwrap();
        recording.finish().unwrap();

        let cast = std::fs::read_to_string(paths.cast).unwrap();
        assert!(cast.contains("\"version\":2"));
        assert!(cast.contains("visible output"));
        assert!(!cast.contains("secret command"));

        let trace = std::fs::read_to_string(paths.trace).unwrap();
        assert!(trace.contains("\"kind\":\"action\""));
        assert!(trace.contains("\"kind\":\"observation\""));
        assert!(trace.contains("\"kind\":\"annotation\""));
        assert!(trace.contains("\"raw_input_events\":\"omitted\""));
        assert!(trace.contains("\"echoed_input_in_output\":\"may_be_present\""));
        assert!(!trace.contains("secret command"));
    }

    #[test]
    fn omitted_raw_input_does_not_claim_that_terminal_echo_is_private() {
        let directory = tempdir().unwrap();
        let (mut recording, paths) =
            TuiRecording::create(directory.path(), 24, 80, "TUI evaluation", false).unwrap();

        recording.input(b"sensitive command\r").unwrap();
        // A real PTY can echo this command back on its output stream.
        recording.output(b"sensitive command\r\n").unwrap();
        recording.finish().unwrap();

        let cast = std::fs::read_to_string(paths.cast).unwrap();
        assert!(cast.contains("sensitive command"));

        let trace = std::fs::read_to_string(paths.trace).unwrap();
        assert!(trace.contains("\"raw_input_events\":\"omitted\""));
        assert!(trace.contains("\"echoed_input_in_output\":\"may_be_present\""));
    }

    #[test]
    fn write_failures_are_returned_to_the_harness() {
        let directory = tempdir().unwrap();
        let (mut recording, _) =
            TuiRecording::create(directory.path(), 24, 80, "TUI evaluation", false).unwrap();
        recording.cast = File::open(directory.path()).unwrap();

        let error = recording.output(b"visible output").unwrap_err();
        assert_ne!(error.kind(), io::ErrorKind::WouldBlock);
    }

    #[test]
    fn completed_recording_has_a_durable_footer() {
        let directory = tempdir().unwrap();
        let (recording, paths) =
            TuiRecording::create(directory.path(), 24, 80, "TUI evaluation", false).unwrap();

        recording.finish().unwrap();

        let trace = std::fs::read_to_string(paths.trace).unwrap();
        assert!(trace.contains("\"kind\":\"footer\""));
        assert!(trace.contains("\"disposition\":\"complete\""));
    }
}
