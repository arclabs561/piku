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

use crate::{OutputSink, PostToolAction};
use piku_api::TokenUsage;

/// Current on-disk schema for [`RunEventEnvelope`].
///
/// Version 2 makes run-vs-turn scope explicit. The reader remains compatible
/// with version 1 records, where `turn_id` implied turn scope.
pub const RUN_RECORD_SCHEMA_VERSION: u32 = 2;

/// Content larger than this is stored beside the JSONL record as an artifact.
/// The event stream stays cheap to scan while retaining the complete value.
pub const RUN_INLINE_CONTENT_LIMIT_BYTES: usize = 16 * 1024;

/// One ordered event in a durable run record.
#[derive(Debug, Clone, PartialEq)]
pub struct RunEventEnvelope {
    pub schema_version: u32,
    pub sequence: u64,
    pub recorded_at_ms: u64,
    pub session_id: String,
    pub scope: EventScope,
    pub event: RunEvent,
}

/// The semantic owner of an event. Run-level decisions are not fabricated as
/// turns, and turn-level activity always names its turn.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EventScope {
    Run,
    Turn { turn_id: String },
}

impl EventScope {
    #[must_use]
    pub fn turn_id(&self) -> Option<&str> {
        match self {
            Self::Run => None,
            Self::Turn { turn_id } => Some(turn_id),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum EventScopeKind {
    Run,
    Turn,
}

#[derive(Serialize, Deserialize)]
struct RunEventEnvelopeWire {
    schema_version: u32,
    sequence: u64,
    recorded_at_ms: u64,
    session_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    scope: Option<EventScopeKind>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    turn_id: Option<String>,
    #[serde(flatten)]
    event: RunEvent,
}

impl Serialize for RunEventEnvelope {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let (scope, turn_id) = match &self.scope {
            EventScope::Run => (EventScopeKind::Run, None),
            EventScope::Turn { turn_id } => (EventScopeKind::Turn, Some(turn_id.clone())),
        };
        RunEventEnvelopeWire {
            schema_version: self.schema_version,
            sequence: self.sequence,
            recorded_at_ms: self.recorded_at_ms,
            session_id: self.session_id.clone(),
            scope: (self.schema_version >= 2).then_some(scope),
            turn_id,
            event: self.event.clone(),
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for RunEventEnvelope {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let wire = RunEventEnvelopeWire::deserialize(deserializer)?;
        if wire.schema_version == 1 && wire.scope.is_some() {
            return Err(serde::de::Error::custom(
                "schema v1 run event must use legacy turn_id scope",
            ));
        }
        let legacy_scope = wire.schema_version == 1 && wire.scope.is_none();
        let scope = match (wire.scope, wire.turn_id) {
            (Some(EventScopeKind::Run), None) => EventScope::Run,
            (Some(EventScopeKind::Turn), Some(turn_id)) => EventScope::Turn { turn_id },
            (None, Some(turn_id)) if legacy_scope => EventScope::Turn { turn_id },
            (Some(EventScopeKind::Run), Some(_)) => {
                return Err(serde::de::Error::custom(
                    "run-scoped event must not contain turn_id",
                ));
            }
            (Some(EventScopeKind::Turn), None) => {
                return Err(serde::de::Error::custom(
                    "turn-scoped event requires turn_id",
                ));
            }
            (None, _) => {
                return Err(serde::de::Error::custom(
                    "schema v2 run event requires explicit scope",
                ));
            }
        };
        Ok(Self {
            schema_version: wire.schema_version,
            sequence: wire.sequence,
            recorded_at_ms: wire.recorded_at_ms,
            session_id: wire.session_id,
            scope,
            event: wire.event,
        })
    }
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
    UserDisposition {
        disposition: RunDisposition,
        note: ContentRef,
    },
}

impl RunEvent {
    fn scope_kind(&self) -> EventScopeKind {
        match self {
            Self::UserDisposition { .. } => EventScopeKind::Run,
            Self::TurnStarted { .. }
            | Self::ContextBuilt { .. }
            | Self::CompactionApplied { .. }
            | Self::AssistantMessage { .. }
            | Self::ToolStarted { .. }
            | Self::PermissionDecision { .. }
            | Self::ToolCompleted { .. }
            | Self::TurnCompleted { .. }
            | Self::Warning { .. } => EventScopeKind::Turn,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunDisposition {
    Accepted,
    NeedsWork,
    Abandoned,
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
    pub estimated_tokens: usize,
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
    artifact_dir: PathBuf,
    artifact_relative_dir: PathBuf,
    session_id: String,
    next_sequence: u64,
    writer: BufWriter<File>,
}

/// Adds durable semantic recording to any existing presentation sink.
pub struct RecordingSink<'a> {
    inner: &'a mut dyn OutputSink,
    recorder: &'a mut RunRecorder,
    turn_id: String,
    record_error: Option<io::Error>,
}

impl<'a> RecordingSink<'a> {
    pub fn new(
        inner: &'a mut dyn OutputSink,
        recorder: &'a mut RunRecorder,
        turn_id: impl Into<String>,
    ) -> Self {
        Self {
            inner,
            recorder,
            turn_id: turn_id.into(),
            record_error: None,
        }
    }

    pub fn take_record_error(&mut self) -> Option<io::Error> {
        self.record_error.take()
    }
}

impl OutputSink for RecordingSink<'_> {
    fn on_text(&mut self, text: &str) {
        self.inner.on_text(text);
    }

    fn on_tool_start(&mut self, tool_name: &str, tool_id: &str, input: &serde_json::Value) {
        self.inner.on_tool_start(tool_name, tool_id, input);
    }

    fn on_tool_end(&mut self, tool_name: &str, result: &str, is_error: bool) -> PostToolAction {
        self.inner.on_tool_end(tool_name, result, is_error)
    }

    fn on_permission_denied(&mut self, tool_name: &str, reason: &str) {
        self.inner.on_permission_denied(tool_name, reason);
    }

    fn on_turn_complete(&mut self, usage: &TokenUsage, iterations: u32) {
        self.inner.on_turn_complete(usage, iterations);
    }

    fn on_interjection(&mut self, text: &str) {
        self.inner.on_interjection(text);
    }

    fn on_context_pressure(&mut self, pressure: f32) {
        self.inner.on_context_pressure(pressure);
    }

    fn on_provider_stream(&mut self, elapsed_ms: u64, blocks: usize, stop_reason: &str) {
        self.inner
            .on_provider_stream(elapsed_ms, blocks, stop_reason);
    }

    fn on_run_event(&mut self, event: &RunEvent) {
        self.inner.on_run_event(event);
        if self.record_error.is_none() {
            if let Err(error) = self.recorder.append(&self.turn_id, event.clone()) {
                self.record_error = Some(error);
            }
        }
    }
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
        let artifact_dir = path.with_extension("artifacts");
        let artifact_relative_dir = artifact_dir
            .file_name()
            .map(PathBuf::from)
            .ok_or_else(|| io::Error::other("run record path has no file name"))?;

        Ok(Self {
            path,
            artifact_dir,
            artifact_relative_dir,
            session_id,
            next_sequence,
            writer: BufWriter::new(file),
        })
    }

    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn append(&mut self, turn_id: impl Into<String>, mut event: RunEvent) -> io::Result<u64> {
        self.append_scoped(
            EventScope::Turn {
                turn_id: turn_id.into(),
            },
            &mut event,
        )
    }

    pub fn append_run(&mut self, mut event: RunEvent) -> io::Result<u64> {
        self.append_scoped(EventScope::Run, &mut event)
    }

    fn append_scoped(&mut self, scope: EventScope, event: &mut RunEvent) -> io::Result<u64> {
        let actual_scope = match scope {
            EventScope::Run => EventScopeKind::Run,
            EventScope::Turn { .. } => EventScopeKind::Turn,
        };
        if actual_scope != event.scope_kind() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "run event has incompatible scope",
            ));
        }
        let sequence = self.next_sequence;
        self.materialize_large_content(sequence, event)?;
        let envelope = RunEventEnvelope {
            schema_version: RUN_RECORD_SCHEMA_VERSION,
            sequence,
            recorded_at_ms: now_ms()?,
            session_id: self.session_id.clone(),
            scope,
            event: event.clone(),
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

    fn materialize_large_content(&self, sequence: u64, event: &mut RunEvent) -> io::Result<()> {
        let target = match event {
            RunEvent::TurnStarted { input, .. } => Some(("input", input)),
            RunEvent::CompactionApplied { summary, .. } => Some(("summary", summary)),
            RunEvent::AssistantMessage { content } => Some(("assistant", content)),
            RunEvent::ToolCompleted { result, .. } => Some(("tool-result", result)),
            RunEvent::UserDisposition { note, .. } => Some(("disposition", note)),
            RunEvent::ContextBuilt { .. }
            | RunEvent::ToolStarted { .. }
            | RunEvent::PermissionDecision { .. }
            | RunEvent::TurnCompleted { .. }
            | RunEvent::Warning { .. } => None,
        };
        if let Some((label, content)) = target {
            self.materialize_content(sequence, label, content)?;
        }
        Ok(())
    }

    fn materialize_content(
        &self,
        sequence: u64,
        label: &str,
        content: &mut ContentRef,
    ) -> io::Result<()> {
        let ContentRef::Inline { text } = content else {
            return Ok(());
        };
        if text.len() <= RUN_INLINE_CONTENT_LIMIT_BYTES {
            return Ok(());
        }

        fs::create_dir_all(&self.artifact_dir)?;
        let file_name = format!("{sequence:08}-{label}.txt");
        let destination = self.artifact_dir.join(&file_name);
        let temporary = self
            .artifact_dir
            .join(format!(".{file_name}.tmp-{}", std::process::id()));
        let bytes = u64::try_from(text.len())
            .map_err(|_| io::Error::other("artifact length does not fit in u64"))?;
        let write_result = (|| {
            let mut file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&temporary)?;
            file.write_all(text.as_bytes())?;
            file.flush()?;
            fs::rename(&temporary, &destination)
        })();
        if write_result.is_err() {
            let _ = fs::remove_file(&temporary);
        }
        write_result?;

        *content = ContentRef::Artifact(ArtifactRef {
            relative_path: self.artifact_relative_dir.join(file_name),
            media_type: "text/plain; charset=utf-8".to_string(),
            bytes,
        });
        Ok(())
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

    let events = text
        .lines()
        .enumerate()
        .map(|(index, line)| {
            serde_json::from_str(line).map_err(|error| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("invalid run record at line {}: {error}", index + 1),
                )
            })
        })
        .collect::<io::Result<Vec<_>>>()?;
    validate_record_structure(&events)?;
    Ok(events)
}

fn validate_existing(events: &[RunEventEnvelope], session_id: &str) -> io::Result<()> {
    validate_record_structure(events)?;
    for (index, event) in events.iter().enumerate() {
        if event.session_id != session_id {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("run record session mismatch at line {}", index + 1),
            ));
        }
    }
    Ok(())
}

fn validate_record_structure(events: &[RunEventEnvelope]) -> io::Result<()> {
    let session_id = events.first().map(|event| event.session_id.as_str());
    for (index, event) in events.iter().enumerate() {
        if !(1..=RUN_RECORD_SCHEMA_VERSION).contains(&event.schema_version) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "unsupported run record schema {} at line {}",
                    event.schema_version,
                    index + 1
                ),
            ));
        }
        let actual_scope = match &event.scope {
            EventScope::Run => EventScopeKind::Run,
            EventScope::Turn { .. } => EventScopeKind::Turn,
        };
        if actual_scope != event.event.scope_kind() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("run event has incompatible scope at line {}", index + 1),
            ));
        }
        if Some(event.session_id.as_str()) != session_id {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("run record changes session at line {}", index + 1),
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

    #[test]
    fn materializes_large_content_before_referencing_it() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("session-1.jsonl");
        let large = "evidence".repeat(RUN_INLINE_CONTENT_LIMIT_BYTES / 8 + 1);
        let mut recorder = RunRecorder::open(&path, "session-1").unwrap();
        recorder
            .append(
                "turn-0",
                RunEvent::AssistantMessage {
                    content: ContentRef::Inline {
                        text: large.clone(),
                    },
                },
            )
            .unwrap();
        drop(recorder);

        let events = read_run_record(&path).unwrap();
        let RunEvent::AssistantMessage {
            content: ContentRef::Artifact(artifact),
        } = &events[0].event
        else {
            panic!("large assistant message was not materialized");
        };
        assert_eq!(artifact.bytes, u64::try_from(large.len()).unwrap());
        assert_eq!(
            fs::read_to_string(path.parent().unwrap().join(&artifact.relative_path)).unwrap(),
            large
        );
    }

    #[test]
    fn leaves_small_content_inline() {
        let path = test_path("inline");
        let mut recorder = RunRecorder::open(&path, "session-1").unwrap();
        recorder
            .append(
                "turn-0",
                RunEvent::AssistantMessage {
                    content: ContentRef::Inline {
                        text: "small".to_string(),
                    },
                },
            )
            .unwrap();
        drop(recorder);

        let events = read_run_record(&path).unwrap();
        assert!(matches!(
            &events[0].event,
            RunEvent::AssistantMessage {
                content: ContentRef::Inline { text }
            } if text == "small"
        ));
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn reads_v1_turn_scope_and_resumes_with_v2_events() {
        let path = test_path("v1-resume");
        fs::write(
            &path,
            concat!(
                r#"{"schema_version":1,"sequence":0,"recorded_at_ms":0,"session_id":"session-1","turn_id":"turn-0","event":"warning","message":"legacy"}"#,
                "\n"
            ),
        )
        .unwrap();

        let legacy = read_run_record(&path).unwrap();
        assert_eq!(legacy[0].scope.turn_id(), Some("turn-0"));
        let legacy_json = serde_json::to_value(&legacy[0]).unwrap();
        assert!(legacy_json.get("scope").is_none());
        assert_eq!(legacy_json["turn_id"], "turn-0");
        let mut recorder = RunRecorder::open(&path, "session-1").unwrap();
        recorder
            .append(
                "turn-0",
                RunEvent::Warning {
                    message: "current".to_string(),
                },
            )
            .unwrap();
        drop(recorder);

        let lines = fs::read_to_string(&path).unwrap();
        let values = lines
            .lines()
            .map(|line| serde_json::from_str::<serde_json::Value>(line).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(values[0]["schema_version"], 1);
        assert_eq!(values[1]["schema_version"], 2);
        assert_eq!(values[1]["scope"], "turn");
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn rejects_missing_or_incompatible_v2_scope() {
        let missing = r#"{"schema_version":2,"sequence":0,"recorded_at_ms":0,"session_id":"session-1","turn_id":"turn-0","event":"warning","message":"x"}"#;
        assert!(serde_json::from_str::<RunEventEnvelope>(missing)
            .unwrap_err()
            .to_string()
            .contains("requires explicit scope"));

        let path = test_path("invalid-scope");
        let mut recorder = RunRecorder::open(&path, "session-1").unwrap();
        let error = recorder
            .append_run(RunEvent::Warning {
                message: "not run-level".to_string(),
            })
            .unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
        let error = recorder
            .append(
                "turn-0",
                RunEvent::UserDisposition {
                    disposition: RunDisposition::Accepted,
                    note: ContentRef::Inline {
                        text: "ship it".to_string(),
                    },
                },
            )
            .unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn reader_rejects_unsupported_schema_and_sequence_gaps() {
        let path = test_path("invalid-structure");
        fs::write(
            &path,
            concat!(
                r#"{"schema_version":99,"sequence":0,"recorded_at_ms":0,"session_id":"session-1","scope":"turn","turn_id":"turn-0","event":"warning","message":"x"}"#,
                "\n"
            ),
        )
        .unwrap();
        assert!(read_run_record(&path)
            .unwrap_err()
            .to_string()
            .contains("unsupported run record schema"));

        fs::write(
            &path,
            concat!(
                r#"{"schema_version":2,"sequence":1,"recorded_at_ms":0,"session_id":"session-1","scope":"turn","turn_id":"turn-0","event":"warning","message":"x"}"#,
                "\n"
            ),
        )
        .unwrap();
        assert!(read_run_record(&path)
            .unwrap_err()
            .to_string()
            .contains("expected 0"));
        fs::remove_file(path).unwrap();
    }
}
