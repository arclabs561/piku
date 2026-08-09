//! Shared ownership of one live session and its durable semantic record.

use std::io;
use std::path::Path;

use crate::{OutputSink, RecordingSink, RunRecorder, Session};

/// The runtime state that every interactive surface needs for a durable run.
///
/// Presentation, provider selection, tool policy, and session-file persistence
/// remain surface concerns. This handle only keeps the mutable conversation and
/// its append-only semantic record under one run identity.
pub struct RunHandle {
    session: Session,
    recorder: RunRecorder,
}

impl RunHandle {
    /// Open or resume the durable record belonging to `session`.
    pub fn open(session: Session, path: impl AsRef<Path>) -> io::Result<Self> {
        if session.id.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "run session ID must not be empty",
            ));
        }
        let recorder = RunRecorder::open(path, &session.id)?;
        Ok(Self { session, recorder })
    }

    #[must_use]
    pub fn id(&self) -> &str {
        &self.session.id
    }

    #[must_use]
    pub fn record_path(&self) -> &Path {
        self.recorder.path()
    }

    #[must_use]
    pub fn session(&self) -> &Session {
        &self.session
    }

    pub fn session_mut(&mut self) -> &mut Session {
        &mut self.session
    }

    /// Borrow the session and wrap a surface sink for one recorded turn.
    pub fn begin_turn<'a>(
        &'a mut self,
        inner: &'a mut dyn OutputSink,
        turn_id: impl Into<String>,
    ) -> RunTurn<'a> {
        RunTurn {
            session: &mut self.session,
            sink: RecordingSink::new(inner, &mut self.recorder, turn_id),
        }
    }

    #[must_use]
    pub fn into_session(self) -> Session {
        self.session
    }
}

/// A scoped turn borrow that makes deferred record failures explicit.
pub struct RunTurn<'a> {
    session: &'a mut Session,
    sink: RecordingSink<'a>,
}

impl<'a> RunTurn<'a> {
    pub fn parts(&mut self) -> (&mut Session, &mut RecordingSink<'a>) {
        (self.session, &mut self.sink)
    }

    pub fn finish(mut self) -> io::Result<()> {
        match self.sink.take_record_error() {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        read_run_record, ContextSourceSummary, PostToolAction, RunContentRef, RunEvent,
        Sha256Digest, SourceReference, TokenUsage, Trust,
    };
    use tempfile::tempdir;

    #[derive(Default)]
    struct Sink;

    impl OutputSink for Sink {
        fn on_text(&mut self, _text: &str) {}

        fn on_tool_start(&mut self, _tool_name: &str, _tool_id: &str, _input: &serde_json::Value) {}

        fn on_tool_end(
            &mut self,
            _tool_name: &str,
            _result: &str,
            _is_error: bool,
        ) -> PostToolAction {
            PostToolAction::Continue
        }

        fn on_permission_denied(&mut self, _tool_name: &str, _reason: &str) {}

        fn on_turn_complete(&mut self, _usage: &TokenUsage, _iterations: u32) {}
    }

    #[test]
    fn handle_keeps_session_and_record_identity_together() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("run.jsonl");
        let mut handle = RunHandle::open(Session::new("run-1".into()), &path).unwrap();
        let mut sink = Sink;

        let mut turn = handle.begin_turn(&mut sink, "turn-1");
        let (_, recording_sink) = turn.parts();
        recording_sink.on_run_event(&RunEvent::Warning {
            message: "observed".into(),
        });
        turn.finish().unwrap();

        assert_eq!(handle.id(), "run-1");
        assert_eq!(handle.record_path(), path);
        let events = read_run_record(path).unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].session_id, "run-1");
        assert_eq!(events[0].scope.turn_id(), Some("turn-1"));
    }

    #[test]
    fn handle_rejects_an_empty_session_identity() {
        let dir = tempdir().unwrap();
        let error = RunHandle::open(Session::default(), dir.path().join("run.jsonl"))
            .err()
            .expect("empty identity should fail");
        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
    }

    #[test]
    fn queued_context_provenance_immediately_follows_turn_start() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("run.jsonl");
        let mut handle = RunHandle::open(Session::new("run-1".into()), &path).unwrap();
        let mut sink = Sink;
        let mut turn = handle.begin_turn(&mut sink, "turn-1");
        let (_, recording_sink) = turn.parts();
        recording_sink
            .queue_after_turn_started(RunEvent::ContextSourcesResolved {
                sources: vec![ContextSourceSummary {
                    id: "note-1".into(),
                    sources: vec![SourceReference {
                        reference: "surface:scratch/object:note-1".into(),
                        sha256: Sha256Digest::of_bytes(b"note"),
                    }],
                    output_sha256: Sha256Digest::of_bytes(b"note"),
                    byte_size: 4,
                    trust: Trust::UntrustedEvidence,
                }],
            })
            .unwrap();
        recording_sink.on_run_event(&RunEvent::TurnStarted {
            provider: Some("test".into()),
            model: "model".into(),
            input: RunContentRef::Inline {
                text: "prompt".into(),
            },
        });
        recording_sink.on_run_event(&RunEvent::Warning {
            message: "later".into(),
        });
        turn.finish().unwrap();

        let events = read_run_record(path).unwrap();
        assert!(matches!(events[0].event, RunEvent::TurnStarted { .. }));
        assert!(matches!(
            events[1].event,
            RunEvent::ContextSourcesResolved { .. }
        ));
        assert!(matches!(events[2].event, RunEvent::Warning { .. }));
    }
}
