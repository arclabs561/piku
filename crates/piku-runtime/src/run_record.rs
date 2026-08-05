//! Durable, versioned records of an agent run.
//!
//! The run record is the semantic source shared by terminal, browser, editor,
//! and native projections. It is deliberately separate from diagnostic traces:
//! losing a trace may reduce observability, while losing this record would lose
//! user-visible work and provenance.

use serde::{Deserialize, Serialize};
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

/// Current on-disk schema for [`RunEventEnvelope`].
pub const RUN_RECORD_SCHEMA_VERSION: u32 = 1;

/// One ordered event in a durable run record.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunEventEnvelope {
    pub schema_version: u32,
    pub sequence: u64,
    pub recorded_at_ms: u64,
    pub session_id: String,
    pub turn_id: String,
    #[serde(flatten)]
    pub event: RunEvent,
}

/// A semantic event that a presentation surface may project differently.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum RunEvent {
    TurnStarted {
        provider: Option<String>,
        model: String,
        input: ContentRef,
    },
    ContextBuilt {
        manifest: ContextManifest,
    },
    CompactionApplied {
        before_messages: usize,
        after_messages: usize,
        masked_tool_results: usize,
        summary: ContentRef,
    },
    AssistantMessage {
        content: ContentRef,
    },
    ToolStarted {
        tool_call_id: String,
        name: String,
        arguments: serde_json::Value,
    },
    PermissionDecision {
        tool_call_id: String,
        decision: PermissionDecision,
    },
    ToolCompleted {
        tool_call_id: String,
        result: ContentRef,
        is_error: bool,
    },
    TurnCompleted {
        usage: UsageRecord,
        stop_reason: Option<String>,
    },
    Warning {
        message: String,
    },
}

/// Content retained inline, delegated to an artifact, or explicitly missing.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "storage", rename_all = "snake_case")]
pub enum ContentRef {
    Inline { text: String },
    Artifact(ArtifactRef),
    Unavailable { reason: String },
}

/// A run-relative pointer to a durable artifact.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArtifactRef {
    pub relative_path: PathBuf,
    pub media_type: String,
    pub bytes: u64,
}

/// Exact accounting of the context selected for a provider request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContextManifest {
    pub model: String,
    pub context_window_tokens: usize,
    pub estimated_input_tokens: usize,
    pub system_sections: Vec<ContextSection>,
    pub messages: Vec<ContextMessage>,
    pub tools: Vec<ContextTool>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContextSection {
    pub label: String,
    pub estimated_tokens: usize,
    pub selected: bool,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContextMessage {
    pub session_index: usize,
    pub role: String,
    pub estimated_tokens: usize,
    pub selected: bool,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContextTool {
    pub name: String,
    pub selected: bool,
    pub reason: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PermissionDecision {
    Allowed,
    Denied,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct UsageRecord {
    pub input_tokens: u64,
    pub output_tokens: u64,
}

/// Append-only writer that validates an existing record before resuming it.
#[derive(Debug)]
pub struct RunRecorder {
    path: PathBuf,
    session_id: String,
    next_sequence: u64,
    writer: BufWriter<File>,
}

impl RunRecorder {
    pub fn open(path: impl AsRef<Path>, session_id: impl Into<String>) -> io::Result<Self> {
        let path = path.as_ref().to_path_buf();
        let session_id = session_id.into();
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }

        let existing = read_run_record(&path)?;
        validate_existing(&existing, &session_id)?;
        let next_sequence = u64::try_from(existing.len())
            .map_err(|_| io::Error::other("run record contains too many events"))?;
        let file = OpenOptions::new().create(true).append(true).open(&path)?;

        Ok(Self {
            path,
            session_id,
            next_sequence,
            writer: BufWriter::new(file),
        })
    }

    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn append(&mut self, turn_id: impl Into<String>, event: RunEvent) -> io::Result<u64> {
        let sequence = self.next_sequence;
        let envelope = RunEventEnvelope {
            schema_version: RUN_RECORD_SCHEMA_VERSION,
            sequence,
            recorded_at_ms: now_ms()?,
            session_id: self.session_id.clone(),
            turn_id: turn_id.into(),
            event,
        };
        serde_json::to_writer(&mut self.writer, &envelope).map_err(io::Error::other)?;
        self.writer.write_all(b"\n")?;
        self.writer.flush()?;
        self.next_sequence = self
            .next_sequence
            .checked_add(1)
            .ok_or_else(|| io::Error::other("run event sequence overflow"))?;
        Ok(sequence)
    }
}

/// Read and strictly validate a complete JSONL run record.
pub fn read_run_record(path: impl AsRef<Path>) -> io::Result<Vec<RunEventEnvelope>> {
    let path = path.as_ref();
    let text = match fs::read_to_string(path) {
        Ok(text) => text,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) => return Err(error),
    };

    text.lines()
        .enumerate()
        .map(|(index, line)| {
            serde_json::from_str(line).map_err(|error| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("invalid run record at line {}: {error}", index + 1),
                )
            })
        })
        .collect()
}

fn validate_existing(events: &[RunEventEnvelope], session_id: &str) -> io::Result<()> {
    for (index, event) in events.iter().enumerate() {
        if event.schema_version != RUN_RECORD_SCHEMA_VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "unsupported run record schema {} at line {}",
                    event.schema_version,
                    index + 1
                ),
            ));
        }
        if event.session_id != session_id {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("run record session mismatch at line {}", index + 1),
            ));
        }
        let expected = u64::try_from(index)
            .map_err(|_| io::Error::other("run record contains too many events"))?;
        if event.sequence != expected {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "run record sequence {} at line {}, expected {expected}",
                    event.sequence,
                    index + 1
                ),
            ));
        }
    }
    Ok(())
}

fn now_ms() -> io::Result<u64> {
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(io::Error::other)?
        .as_millis();
    u64::try_from(millis).map_err(|_| io::Error::other("system time does not fit in u64"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_TEST_FILE: AtomicU64 = AtomicU64::new(0);

    fn test_path(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "piku-run-record-{}-{name}-{}.jsonl",
            std::process::id(),
            NEXT_TEST_FILE.fetch_add(1, Ordering::Relaxed)
        ))
    }

    fn started() -> RunEvent {
        RunEvent::TurnStarted {
            provider: Some("test".into()),
            model: "test-model".into(),
            input: ContentRef::Inline {
                text: "inspect this".into(),
            },
        }
    }

    #[test]
    fn appends_and_resumes_a_valid_record() {
        let path = test_path("resume");
        {
            let mut recorder = RunRecorder::open(&path, "session-1").unwrap();
            assert_eq!(recorder.append("turn-1", started()).unwrap(), 0);
        }
        {
            let mut recorder = RunRecorder::open(&path, "session-1").unwrap();
            assert_eq!(
                recorder
                    .append(
                        "turn-1",
                        RunEvent::TurnCompleted {
                            usage: UsageRecord {
                                input_tokens: 11,
                                output_tokens: 7,
                            },
                            stop_reason: Some("end_turn".into()),
                        },
                    )
                    .unwrap(),
                1
            );
        }

        let events = read_run_record(&path).unwrap();
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].sequence, 0);
        assert_eq!(events[1].sequence, 1);
        assert_eq!(events[0].session_id, "session-1");
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn refuses_to_append_to_another_session() {
        let path = test_path("session-mismatch");
        {
            let mut recorder = RunRecorder::open(&path, "session-1").unwrap();
            recorder.append("turn-1", started()).unwrap();
        }

        let error = RunRecorder::open(&path, "session-2").unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        assert!(error.to_string().contains("session mismatch"));
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn reports_corrupt_records_instead_of_silently_skipping_them() {
        let path = test_path("corrupt");
        fs::write(&path, "{not-json}\n").unwrap();

        let error = read_run_record(&path).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        assert!(error.to_string().contains("line 1"));
        fs::remove_file(path).unwrap();
    }
}
