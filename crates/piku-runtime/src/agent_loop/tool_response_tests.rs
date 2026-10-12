use super::*;
use crate::run_record::{read_run_record, RecordingSink, RunEvent, RunRecorder};

struct Scripted(Vec<Event>);
impl Provider for Scripted {
    fn name(&self) -> &'static str {
        "scripted-tool-response"
    }
    fn stream_message(
        &self,
        _: MessageRequest,
    ) -> std::pin::Pin<
        Box<dyn futures_util::Stream<Item = Result<Event, piku_api::ApiError>> + Send + '_>,
    > {
        Box::pin(futures_util::stream::iter(
            self.0.clone().into_iter().map(Ok),
        ))
    }
}

struct Sink;
impl OutputSink for Sink {
    fn on_text(&mut self, _: &str) {}
    fn on_tool_start(&mut self, _: &str, _: &str, _: &serde_json::Value) {}
    fn on_tool_end(&mut self, _: &str, _: &str, _: &str, _: bool) -> PostToolAction {
        PostToolAction::Continue
    }
    fn on_permission_denied(&mut self, _: &str, _: &str) {}
    fn on_turn_complete(&mut self, _: &TokenUsage, _: u32) {}
}

fn tool(events: &mut Vec<Event>, id: &str, name: &str, json: &str) {
    events.extend([
        Event::ToolUseStart {
            id: id.into(),
            name: name.into(),
        },
        Event::ToolUseDelta {
            id: id.into(),
            partial_json: json.into(),
        },
        Event::ToolUseEnd { id: id.into() },
    ]);
}

async fn run(
    events: Vec<Event>,
    dir: &std::path::Path,
) -> (
    TurnResult,
    Session,
    Vec<crate::run_record::RunEventEnvelope>,
) {
    let path = dir.join("run.jsonl");
    let mut recorder = RunRecorder::open(&path, "tool-response").unwrap();
    let mut inner = Sink;
    let mut sink = RecordingSink::new(&mut inner, &mut recorder, "turn");
    let mut session = Session::new("tool-response".into());
    let result = run_turn(
        "test",
        &mut session,
        &Scripted(events),
        "scripted",
        &[],
        piku_tools::all_tool_definitions(),
        &crate::permission::AllowAll,
        &mut sink,
        Some(1),
        None,
    )
    .await;
    assert!(sink.take_record_error().is_none());
    drop(sink);
    drop(recorder);
    (result, session, read_run_record(path).unwrap())
}

#[tokio::test]
async fn malformed_arguments_reject_batch_before_any_effect() {
    let dir = tempfile::tempdir().unwrap();
    let target = dir.path().join("must-not-exist");
    let mut events = Vec::new();
    tool(
        &mut events,
        "valid",
        "write_file",
        &serde_json::json!({"path": target, "content": "unsafe partial batch"}).to_string(),
    );
    tool(&mut events, "invalid", "read_file", "{\"path\":");
    events.push(Event::MessageStop {
        stop_reason: StopReason::ToolUse,
    });
    let (result, session, records) = run(events, dir.path()).await;
    assert!(
        !target.exists(),
        "invalid response must not execute its valid prefix"
    );
    assert!(result.stream_error.is_some());
    assert!(!records
        .iter()
        .any(|e| matches!(e.event, RunEvent::ToolStarted { .. })));
    assert!(matches!(
        records.last().unwrap().event,
        RunEvent::TurnFailed { .. }
    ));
    assert!(!session
        .messages
        .iter()
        .flat_map(|m| &m.blocks)
        .any(|b| matches!(b, ContentBlock::ToolUse { .. })));
}

#[tokio::test]
async fn token_limit_response_does_not_execute_complete_tool_json() {
    let dir = tempfile::tempdir().unwrap();
    let target = dir.path().join("must-not-exist");
    let mut events = vec![Event::TextDelta {
        text: "partial answer".into(),
    }];
    tool(
        &mut events,
        "truncated",
        "write_file",
        &serde_json::json!({"path": target, "content": "unsafe truncated response"}).to_string(),
    );
    events.push(Event::MessageStop {
        stop_reason: StopReason::MaxTokens,
    });
    let (_, session, records) = run(events, dir.path()).await;
    assert!(
        !target.exists(),
        "token-limited response must not dispatch tools"
    );
    assert!(!records
        .iter()
        .any(|e| matches!(e.event, RunEvent::ToolStarted { .. })));
    assert!(!session
        .messages
        .iter()
        .flat_map(|m| &m.blocks)
        .any(|b| matches!(b, ContentBlock::ToolUse { .. })));
    assert!(session
        .messages
        .iter()
        .flat_map(|m| &m.blocks)
        .any(|b| matches!(b, ContentBlock::Text { text } if text == "partial answer")));
    assert!(
        matches!(&records.last().unwrap().event, RunEvent::TurnCompleted { stop_reason: Some(reason), .. } if reason == "max_tokens")
    );
}

#[tokio::test]
async fn valid_tool_response_still_writes_and_records_effect() {
    let dir = tempfile::tempdir().unwrap();
    let target = dir.path().join("written");
    let mut events = Vec::new();
    tool(
        &mut events,
        "valid",
        "write_file",
        &serde_json::json!({"path": target, "content": "accepted"}).to_string(),
    );
    events.push(Event::MessageStop {
        stop_reason: StopReason::ToolUse,
    });
    let (result, _, records) = run(events, dir.path()).await;
    assert!(result.stream_error.is_none());
    assert_eq!(std::fs::read_to_string(target).unwrap(), "accepted");
    assert_eq!(records.iter().filter(|e| matches!(&e.event, RunEvent::ToolCompleted { is_error: false, effects, .. } if !effects.is_empty())).count(), 1);
}

#[tokio::test]
async fn parameterless_call_without_delta_remains_supported() {
    let dir = tempfile::tempdir().unwrap();
    let events = vec![
        Event::ToolUseStart {
            id: "empty".into(),
            name: "list_dir".into(),
        },
        Event::ToolUseEnd { id: "empty".into() },
        Event::MessageStop {
            stop_reason: StopReason::ToolUse,
        },
    ];
    let (result, session, records) = run(events, dir.path()).await;
    assert!(result.stream_error.is_none());
    assert_eq!(records.iter().filter(|e| matches!(&e.event, RunEvent::ToolCompleted { tool_call_id, is_error: false, .. } if tool_call_id == "empty")).count(), 1);
    assert_eq!(session.messages.iter().flat_map(|m| &m.blocks).filter(|b| matches!(b, ContentBlock::ToolUse { id, name, input } if id == "empty" && name == "list_dir" && input == &serde_json::json!({}))).count(), 1);
    assert_eq!(session.messages.iter().flat_map(|m| &m.blocks).filter(|b| matches!(b, ContentBlock::ToolResult { tool_use_id, is_error: false, .. } if tool_use_id == "empty")).count(), 1);
}
