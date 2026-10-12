//! Fault injection through the existing runtime turn/record ownership boundary.
use crate::{
    audit_run_record, read_run_record, run_turn, AllowAll, OutputSink, PostToolAction, RunEvent,
    RunHandle, Session, TokenUsage,
};
use piku_api::{ApiError, Event, MessageRequest, Provider, StopReason};
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};

struct Scripted {
    target: std::path::PathBuf,
    requests: AtomicUsize,
}
impl Provider for Scripted {
    fn name(&self) -> &'static str {
        "record-fault-fixture"
    }
    fn stream_message(
        &self,
        _: MessageRequest,
    ) -> Pin<Box<dyn futures_util::Stream<Item = Result<Event, ApiError>> + Send + '_>> {
        let events = if self.requests.fetch_add(1, Ordering::SeqCst) == 0 {
            vec![
                Event::TextDelta { text: "x".repeat(crate::run_record::RUN_INLINE_CONTENT_LIMIT_BYTES + 1) },
                Event::ToolUseStart { id: "write-after-fault".into(), name: "write_file".into() },
                Event::ToolUseDelta { id: "write-after-fault".into(), partial_json: serde_json::json!({"path":self.target,"content":"actual effect after evidence failure"}).to_string() },
                Event::ToolUseEnd { id: "write-after-fault".into() },
                Event::MessageStop { stop_reason: StopReason::ToolUse },
            ]
        } else {
            vec![
                Event::TextDelta {
                    text: "done".into(),
                },
                Event::MessageStop {
                    stop_reason: StopReason::EndTurn,
                },
            ]
        };
        Box::pin(futures_util::stream::iter(events.into_iter().map(Ok)))
    }
}
#[derive(Default)]
struct Observed {
    completions: usize,
    tool_successes: usize,
}
impl OutputSink for Observed {
    fn on_text(&mut self, _: &str) {}
    fn on_tool_start(&mut self, _: &str, _: &str, _: &serde_json::Value) {}
    fn on_tool_end(&mut self, _: &str, _: &str, _: &str, is_error: bool) -> PostToolAction {
        self.tool_successes += usize::from(!is_error);
        PostToolAction::Continue
    }
    fn on_permission_denied(&mut self, _: &str, _: &str) {}
    fn on_turn_complete(&mut self, _: &TokenUsage, _: u32) {
        self.completions += 1;
    }
}

#[tokio::test]
async fn deferred_record_failure_reports_error_after_actual_tool_effect() {
    let dir = tempfile::tempdir().unwrap();
    let record = dir.path().join("run.jsonl");
    let target = dir.path().join("actual.txt");
    let mut run = RunHandle::open(Session::new("fault-fixture".into()), &record).unwrap();
    // Real filesystem failure, not a mocked recorder: a regular file blocks
    // creation of the artifact directory when the long assistant text arrives.
    std::fs::write(record.with_extension("artifacts"), "blocked").unwrap();
    let provider = Scripted {
        target: target.clone(),
        requests: AtomicUsize::new(0),
    };
    let mut observed = Observed::default();
    let mut turn = run.begin_turn(&mut observed, "turn-fault");
    let (session, sink) = turn.parts();
    let result = tokio::time::timeout(
        std::time::Duration::from_secs(2),
        run_turn(
            "write the fixture",
            session,
            &provider,
            "scripted",
            &[],
            piku_tools::all_tool_definitions(),
            &AllowAll,
            sink,
            Some(2),
            None,
        ),
    )
    .await
    .unwrap();
    assert!(result.stream_error.is_none());
    assert_eq!(
        std::fs::read_to_string(&target).unwrap(),
        "actual effect after evidence failure"
    );
    let error = turn
        .finish()
        .expect_err("deferred persistence error must reach caller");
    assert!(!error.to_string().is_empty());
    assert_eq!(observed.tool_successes, 1);
    assert_eq!(observed.completions, 1);
    assert_eq!(provider.requests.load(Ordering::SeqCst), 2);
    let reopened = read_run_record(&record).unwrap();
    assert!(reopened
        .iter()
        .any(|event| matches!(event.event, RunEvent::TurnStarted { .. })));
    assert!(!reopened.iter().any(|event| matches!(
        event.event,
        RunEvent::AssistantMessage { .. }
            | RunEvent::ToolStarted { .. }
            | RunEvent::ToolCompleted { .. }
            | RunEvent::TurnCompleted { .. }
    )));
    assert!(!audit_run_record(&reopened).is_structurally_complete());
    eprintln!("record fault observed: actual write=1; provider requests=2; presented completion=1; finish=error; reopened audit=incomplete; durable tool effects=0");
}
