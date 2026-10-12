use super::*;
use crate::run_record::{read_run_record, RecordingSink, RunEvent, RunRecorder};
use std::sync::atomic::Ordering;

struct Scripted {
    events: Vec<Event>,
    pending: bool,
}
impl Provider for Scripted {
    fn name(&self) -> &'static str {
        "scripted-cancellation"
    }
    fn stream_message(
        &self,
        _: MessageRequest,
    ) -> std::pin::Pin<
        Box<dyn futures_util::Stream<Item = Result<Event, piku_api::ApiError>> + Send + '_>,
    > {
        let events = futures_util::stream::iter(self.events.clone().into_iter().map(Ok));
        if self.pending {
            Box::pin(events.chain(futures_util::stream::pending()))
        } else {
            Box::pin(events)
        }
    }
}
struct CancelSink {
    flag: CancelFlag,
    on_text: bool,
    on_start: bool,
}
impl OutputSink for CancelSink {
    fn on_text(&mut self, text: &str) {
        if self.on_text && text == "partial" {
            self.flag.store(true, Ordering::Relaxed);
        }
    }
    fn on_tool_start(&mut self, _: &str, _: &str, _: &serde_json::Value) {
        if self.on_start {
            self.flag.store(true, Ordering::Relaxed);
        }
    }
    fn on_tool_end(&mut self, _: &str, id: &str, _: &str, _: bool) -> PostToolAction {
        if id == "first" {
            self.flag.store(true, Ordering::Relaxed);
            // Cancellation wins over a simultaneous self-update request and
            // still pairs the remaining assistant calls with skipped results.
            return PostToolAction::ReplaceAndExec("unused-replacement".into());
        }
        PostToolAction::Continue
    }
    fn on_permission_denied(&mut self, _: &str, _: &str) {}
    fn on_turn_complete(&mut self, _: &TokenUsage, _: u32) {}
}
async fn run(
    provider: Scripted,
    on_text: bool,
    on_start: bool,
    dir: &std::path::Path,
) -> (Session, Vec<crate::run_record::RunEventEnvelope>) {
    let flag = CancelFlag::default();
    if provider.pending && !on_text {
        let flag = flag.clone();
        tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_millis(40)).await;
            flag.store(true, Ordering::Relaxed);
        });
    }
    let mut inner = CancelSink {
        flag: flag.clone(),
        on_text,
        on_start,
    };
    let path = dir.join("run.jsonl");
    let mut recorder = RunRecorder::open(&path, "cancel-test").unwrap();
    let mut sink = RecordingSink::new(&mut inner, &mut recorder, "turn");
    let mut session = Session::new("cancel-test".into());
    let outcome = tokio::time::timeout(
        std::time::Duration::from_secs(1),
        run_turn_inner(
            "test",
            &mut session,
            &provider,
            "scripted",
            &[],
            piku_tools::all_tool_definitions(),
            &crate::permission::AllowAll,
            &mut sink,
            Some(1),
            None,
            None,
            0,
            &[],
            None,
            Some(&flag),
        ),
    )
    .await
    .expect("active cancellation must not wait for the provider");
    assert!(outcome.result.cancelled);
    assert!(outcome.result.replace_and_exec.is_none());
    assert!(matches!(outcome.exit, TurnExit::Cancelled));
    assert!(sink.take_record_error().is_none());
    drop(sink);
    drop(recorder);
    let events = read_run_record(path).unwrap();
    assert!(matches!(
        events.last().unwrap().event,
        RunEvent::TurnCancelled { .. }
    ));
    assert!(!events.iter().any(|e| matches!(
        e.event,
        RunEvent::TurnCompleted { .. } | RunEvent::TurnFailed { .. }
    )));
    (session, events)
}
#[tokio::test]
async fn pending_stream_preserves_text_and_cancels_without_dispatch() {
    let dir = tempfile::tempdir().unwrap();
    let (session, events) = run(
        Scripted {
            events: vec![Event::TextDelta {
                text: "partial".into(),
            }],
            pending: true,
        },
        true,
        false,
        dir.path(),
    )
    .await;
    assert!(session
        .messages
        .iter()
        .flat_map(|m| &m.blocks)
        .any(|b| matches!(b, ContentBlock::Text { text } if text == "partial")));
    assert!(!events
        .iter()
        .any(|e| matches!(e.event, RunEvent::ToolStarted { .. })));
}
#[tokio::test]
async fn cancellation_between_tools_preserves_first_effect_and_skips_second() {
    let dir = tempfile::tempdir().unwrap();
    let first = dir.path().join("first.txt");
    let second = dir.path().join("second.txt");
    let mut events = Vec::new();
    for (id, path) in [("first", &first), ("second", &second)] {
        events.extend([
            Event::ToolUseStart {
                id: id.into(),
                name: "write_file".into(),
            },
            Event::ToolUseDelta {
                id: id.into(),
                partial_json: serde_json::json!({"path": path, "content": id}).to_string(),
            },
            Event::ToolUseEnd { id: id.into() },
        ]);
    }
    events.push(Event::MessageStop {
        stop_reason: StopReason::ToolUse,
    });
    let (session, records) = run(
        Scripted {
            events,
            pending: false,
        },
        false,
        false,
        dir.path(),
    )
    .await;
    assert_eq!(std::fs::read_to_string(first).unwrap(), "first");
    assert!(!second.exists());
    for id in ["first", "second"] {
        assert_eq!(session.messages.iter().flat_map(|m| &m.blocks).filter(|b| matches!(b, ContentBlock::ToolResult { tool_use_id, .. } if tool_use_id == id)).count(), 1);
    }
    assert!(records.iter().any(|e| matches!(&e.event, RunEvent::ToolCompleted { tool_call_id, is_error: false, effects, .. } if tool_call_id == "first" && !effects.is_empty())));
    assert!(records.iter().any(|e| matches!(&e.event, RunEvent::ToolCompleted { tool_call_id, is_error: true, effects, .. } if tool_call_id == "second" && effects.is_empty())));
}

#[tokio::test]
async fn cancellation_wakes_idle_provider_and_discards_partial_tool_call() {
    let dir = tempfile::tempdir().unwrap();
    let target = dir.path().join("must-not-exist");
    let (session, events) = run(
        Scripted {
            events: vec![
                Event::TextDelta {
                    text: "observed".into(),
                },
                Event::ToolUseStart {
                    id: "incomplete".into(),
                    name: "write_file".into(),
                },
                Event::ToolUseDelta {
                    id: "incomplete".into(),
                    partial_json: serde_json::json!({"path": target, "content": "never"})
                        .to_string(),
                },
            ],
            pending: true,
        },
        false,
        false,
        dir.path(),
    )
    .await;
    assert!(!target.exists());
    assert!(!session
        .messages
        .iter()
        .flat_map(|m| &m.blocks)
        .any(|b| matches!(b, ContentBlock::ToolUse { .. })));
    assert!(session
        .messages
        .iter()
        .flat_map(|m| &m.blocks)
        .any(|b| matches!(b, ContentBlock::Text { text } if text == "observed")));
    assert!(!events
        .iter()
        .any(|e| matches!(e.event, RunEvent::ToolStarted { .. })));
}

#[tokio::test]
async fn cancel_on_final_text_is_cancelled_not_completed() {
    let dir = tempfile::tempdir().unwrap();
    run(
        Scripted {
            events: vec![
                Event::TextDelta {
                    text: "partial".into(),
                },
                Event::MessageStop {
                    stop_reason: StopReason::EndTurn,
                },
            ],
            pending: false,
        },
        true,
        false,
        dir.path(),
    )
    .await;
}

#[tokio::test]
async fn cancellation_in_start_callback_prevents_tool_effect() {
    let dir = tempfile::tempdir().unwrap();
    let target = dir.path().join("must-not-exist");
    let (_, events) = run(
        Scripted {
            events: vec![
                Event::ToolUseStart {
                    id: "first".into(),
                    name: "write_file".into(),
                },
                Event::ToolUseDelta {
                    id: "first".into(),
                    partial_json: serde_json::json!({"path": target, "content": "never"})
                        .to_string(),
                },
                Event::ToolUseEnd { id: "first".into() },
                Event::MessageStop {
                    stop_reason: StopReason::ToolUse,
                },
            ],
            pending: false,
        },
        false,
        true,
        dir.path(),
    )
    .await;
    assert!(!target.exists());
    assert!(events.iter().any(|e| matches!(&e.event, RunEvent::ToolCompleted { is_error: true, effects, .. } if effects.is_empty())));
}
