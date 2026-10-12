use super::*;
use crate::run_record::{read_run_record, RecordingSink, RunRecorder};
use std::sync::Mutex;

fn envelope(path: &std::path::Path, ciphertext: &str) -> piku_api::ReplayEnvelope {
    serde_json::from_value(serde_json::json!({
        "version": 1, "route": "chatgpt_responses", "registration": "account-registration-a", "model": "test-model",
        "output": [
            {"type":"reasoning", "id":"reasoning-1", "summary":[], "encrypted_content":ciphertext},
            {"type":"message", "id":"message-1", "role":"assistant", "status":"completed", "phase":"commentary", "content":[{"type":"output_text", "text":"writing", "annotations":[]}]},
            {"type":"function_call", "id":"item-1", "call_id":"real-call-1", "name":"write_file", "namespace":"piku", "arguments":serde_json::json!({"path":path,"content":"retained effect"}).to_string()}
        ]
    })).unwrap()
}

struct Scripted {
    name: &'static str,
    events: Vec<Event>,
    requests: Mutex<Vec<MessageRequest>>,
}
impl Provider for Scripted {
    fn name(&self) -> &str {
        self.name
    }
    fn stream_message(
        &self,
        request: MessageRequest,
    ) -> std::pin::Pin<
        Box<dyn futures_util::Stream<Item = Result<Event, piku_api::ApiError>> + Send + '_>,
    > {
        self.requests.lock().unwrap().push(request);
        Box::pin(futures_util::stream::iter(
            self.events.clone().into_iter().map(Ok),
        ))
    }
}

#[derive(Default)]
struct Sink {
    text: String,
}
impl OutputSink for Sink {
    fn on_text(&mut self, text: &str) {
        self.text.push_str(text);
    }
    fn on_tool_start(&mut self, _: &str, _: &str, _: &serde_json::Value) {}
    fn on_tool_end(&mut self, _: &str, _: &str, _: &str, _: bool) -> PostToolAction {
        PostToolAction::Continue
    }
    fn on_permission_denied(&mut self, _: &str, _: &str) {}
    fn on_turn_complete(&mut self, _: &TokenUsage, _: u32) {}
}

fn replay_values(request: &MessageRequest) -> Vec<serde_json::Value> {
    request
        .messages
        .iter()
        .flat_map(|m| &m.content)
        .filter_map(|block| match block {
            RequestContent::Replay { envelope } => Some(serde_json::to_value(envelope).unwrap()),
            _ => None,
        })
        .collect()
}

#[tokio::test]
async fn replay_survives_tool_execution_save_load_and_next_request_without_rendering() {
    let dir = tempfile::tempdir().unwrap();
    let target = dir.path().join("effect");
    let envelope = envelope(&target, "opaque-secret-ciphertext");
    let expected = serde_json::to_value(&envelope).unwrap();
    let provider = Scripted {
        name: "chatgpt",
        events: vec![
            Event::TextDelta {
                text: "writing".into(),
            },
            Event::ToolUseStart {
                id: "real-call-1".into(),
                name: "write_file".into(),
            },
            Event::ToolUseDelta {
                id: "real-call-1".into(),
                partial_json: serde_json::json!({"path":target,"content":"retained effect"})
                    .to_string(),
            },
            Event::ToolUseEnd {
                id: "real-call-1".into(),
            },
            Event::Replay { envelope },
            Event::MessageStop {
                stop_reason: StopReason::ToolUse,
            },
        ],
        requests: Mutex::default(),
    };
    let mut session = Session::new("replay-test".into());
    let mut presentation = Sink::default();
    let record_path = dir.path().join("run.jsonl");
    let mut recorder = RunRecorder::open(&record_path, "replay-test").unwrap();
    let mut sink = RecordingSink::new(&mut presentation, &mut recorder, "turn");
    let result = run_turn(
        "write",
        &mut session,
        &provider,
        "test-model",
        &[],
        piku_tools::all_tool_definitions(),
        &crate::permission::AllowAll,
        &mut sink,
        Some(1),
        None,
    )
    .await;
    assert!(result.stream_error.is_none());
    assert!(sink.take_record_error().is_none());
    drop(sink);
    drop(recorder);
    assert_eq!(std::fs::read_to_string(&target).unwrap(), "retained effect");
    assert!(!presentation.text.contains("opaque-secret"));
    let record = serde_json::to_string(&read_run_record(&record_path).unwrap()).unwrap();
    assert!(!record.contains("opaque-secret"));
    assert!(!format!("{session:?}").contains("opaque-secret"));
    let saved = dir.path().join("session.json");
    session.save(&saved).unwrap();
    let mut resumed = Session::load(&saved).unwrap();
    let next = Scripted {
        name: "chatgpt",
        events: vec![
            Event::TextDelta {
                text: "done".into(),
            },
            Event::MessageStop {
                stop_reason: StopReason::EndTurn,
            },
        ],
        requests: Mutex::default(),
    };
    let result = run_turn(
        "continue",
        &mut resumed,
        &next,
        "test-model",
        &[],
        piku_tools::all_tool_definitions(),
        &crate::permission::AllowAll,
        &mut presentation,
        Some(1),
        None,
    )
    .await;
    assert!(result.stream_error.is_none());
    let requests = next.requests.lock().unwrap();
    assert_eq!(requests[0].max_tokens, 0);
    assert_eq!(replay_values(&requests[0]), vec![expected]);
    assert_eq!(requests[0].messages.iter().flat_map(|m| &m.content).filter(|b| matches!(b, RequestContent::ToolResult { tool_use_id, .. } if tool_use_id=="real-call-1")).count(),1);
}

#[test]
fn replay_estimation_and_request_boundaries_do_not_duplicate_or_rebind_output() {
    let base = ConversationMessage::assistant(
        vec![ContentBlock::Text {
            text: "visible output".into(),
        }],
        None,
    );
    let mut with_replay = base.clone();
    let mut opaque = envelope(std::path::Path::new("file"), &"x".repeat(1_000_000));
    opaque.output.pop(); // No tool projection in this text-only response.
    with_replay
        .blocks
        .push(ContentBlock::Replay { envelope: opaque });
    assert_eq!(base.estimated_tokens(), with_replay.estimated_tokens());
    let mut session = Session::new("boundaries".into());
    session.messages = vec![with_replay.clone(), with_replay];
    let (request, _) = build_request(&session, "different-model", &[], &[]);
    assert_eq!(
        request.messages.len(),
        2,
        "distinct responses must not coalesce"
    );
    let values = replay_values(&request);
    assert_eq!(values.len(), 2);
    assert_eq!(values[0]["registration"], "account-registration-a");
    assert_eq!(
        values[0]["model"], "test-model",
        "runtime must not rebind stored output to the newly requested model"
    );
    assert_eq!(request.model, "different-model");
}

#[tokio::test]
async fn only_selected_chatgpt_route_removes_output_cap() {
    for (name, cap) in [
        ("chatgpt", 0),
        ("custom", 8192),
        ("chatgpt-compatible", 8192),
    ] {
        let provider = Scripted {
            name,
            events: vec![Event::MessageStop {
                stop_reason: StopReason::EndTurn,
            }],
            requests: Mutex::default(),
        };
        let mut session = Session::new("caps".into());
        run_turn(
            "test",
            &mut session,
            &provider,
            "chatgpt-model-name",
            &[],
            vec![],
            &crate::permission::AllowAll,
            &mut Sink::default(),
            Some(1),
            None,
        )
        .await;
        assert_eq!(provider.requests.lock().unwrap()[0].max_tokens, cap);
    }
}

fn exchange(envelope: piku_api::ReplayEnvelope) -> Vec<ConversationMessage> {
    vec![
        ConversationMessage::assistant(
            vec![
                ContentBlock::Text {
                    text: "writing".into(),
                },
                ContentBlock::ToolUse {
                    id: "real-call-1".into(),
                    name: "write_file".into(),
                    input: serde_json::json!({"path":"file","content":"retained effect"}),
                },
                ContentBlock::Replay { envelope },
            ],
            None,
        ),
        ConversationMessage::tool_result("real-call-1".into(), "written".into(), false),
    ]
}

#[tokio::test]
async fn opaque_only_response_remains_visible_and_incomplete_replay_is_not_retained() {
    for stop in [StopReason::EndTurn, StopReason::MaxTokens] {
        let mut opaque = envelope(std::path::Path::new("file"), "opaque-secret");
        opaque.output.truncate(1); // Reasoning only, no visible text or calls.
        let provider = Scripted {
            name: "chatgpt",
            events: vec![
                Event::Replay { envelope: opaque },
                Event::MessageStop {
                    stop_reason: stop.clone(),
                },
            ],
            requests: Mutex::default(),
        };
        let mut session = Session::new("opaque-only".into());
        let mut sink = Sink::default();
        run_turn(
            "test",
            &mut session,
            &provider,
            "test-model",
            &[],
            vec![],
            &crate::permission::AllowAll,
            &mut sink,
            Some(1),
            None,
        )
        .await;
        assert!(sink.text.contains("model returned an empty response"));
        assert!(!sink.text.contains("opaque-secret"));
        assert_eq!(
            session
                .messages
                .iter()
                .flat_map(|m| &m.blocks)
                .filter(|b| matches!(b, ContentBlock::Replay { .. }))
                .count(),
            usize::from(stop == StopReason::EndTurn)
        );
    }
}

#[test]
fn curation_and_compaction_preserve_surviving_replay_exchanges() {
    let opaque = envelope(std::path::Path::new("file"), "opaque-secret");
    let expected = serde_json::to_value(&opaque).unwrap();
    let mut session = Session::new("curation".into());
    session
        .messages
        .push(ConversationMessage::user("x".repeat(1_000_000)));
    session.messages.extend(exchange(opaque.clone()));
    session
        .messages
        .extend((0..6).map(|_| ConversationMessage::user("recent requirement")));
    let (request, manifest) = build_request(&session, "gpt-4o", &[], &[]);
    assert!(!manifest.messages[0].selected);
    assert_eq!(replay_values(&request), vec![expected.clone()]);
    assert_eq!(
        request
            .messages
            .iter()
            .flat_map(|m| &m.content)
            .filter(|b| matches!(b,RequestContent::ToolUse { id,.. } if id=="real-call-1"))
            .count(),
        1
    );
    assert_eq!(request.messages.iter().flat_map(|m| &m.content).filter(|b| matches!(b,RequestContent::ToolResult { tool_use_id,.. } if tool_use_id=="real-call-1")).count(),1);

    session.messages = vec![ConversationMessage::user("old task")];
    session.messages.extend(exchange(opaque));
    let config = crate::compact::CompactionConfig {
        preserve_recent_messages: 1,
        max_estimated_tokens: 1,
    };
    for compacted in [
        crate::compact::compact_session(&session, config),
        crate::compact::apply_compact_summary(&session, "old task summary", config),
    ] {
        assert_eq!(compacted.removed_message_count, 1);
        let (request, _) = build_request(&compacted.compacted_session, "test-model", &[], &[]);
        assert_eq!(replay_values(&request), vec![expected.clone()]);
        assert!(!compacted.summary.contains("opaque-secret"));
    }
    session
        .messages
        .extend((0..6).map(|_| ConversationMessage::user("new task")));
    let compacted = crate::compact::compact_session(
        &session,
        crate::compact::CompactionConfig {
            preserve_recent_messages: 6,
            max_estimated_tokens: 1,
        },
    );
    let (request, _) = build_request(&compacted.compacted_session, "test-model", &[], &[]);
    assert!(replay_values(&request).is_empty());
    assert!(!request
        .messages
        .iter()
        .flat_map(|m| &m.content)
        .any(|b| matches!(
            b,
            RequestContent::ToolUse { .. } | RequestContent::ToolResult { .. }
        )));
    assert!(!compacted.summary.contains("opaque-secret"));
}

#[test]
fn observation_masking_never_rewrites_a_retained_replay_tool_exchange() {
    let mut session = Session::new("masking".into());
    session.messages = exchange(envelope(std::path::Path::new("file"), "opaque-secret"));
    let output = "exact result ".repeat(1000);
    session.messages[1] =
        ConversationMessage::tool_result("real-call-1".into(), output.clone(), false);
    session
        .messages
        .extend((0..6).map(|_| ConversationMessage::user("new task")));
    let compacted = crate::compact::compact_session(
        &session,
        crate::compact::CompactionConfig {
            preserve_recent_messages: 6,
            max_estimated_tokens: 1000,
        },
    );
    let (request, _) = build_request(&compacted.compacted_session, "test-model", &[], &[]);
    for block in request.messages.iter().flat_map(|m| &m.content) {
        if let RequestContent::ToolResult {
            tool_use_id,
            content,
            ..
        } = block
        {
            assert_eq!(tool_use_id, "real-call-1");
            assert_eq!(content, &output, "retained replay result must remain exact");
        }
    }
    assert!(
        replay_values(&request).is_empty(),
        "whole over-budget exchange should be dropped"
    );
    assert_eq!(compacted.removed_message_count, 2);
    session.messages[0]
        .blocks
        .retain(|block| !matches!(block, ContentBlock::Replay { .. }));
    let ordinary = crate::compact::compact_session(
        &session,
        crate::compact::CompactionConfig {
            preserve_recent_messages: 6,
            max_estimated_tokens: 1000,
        },
    );
    assert_eq!(
        ordinary.removed_message_count, 0,
        "ordinary observation masking remains available"
    );
    assert!(ordinary
        .compacted_session
        .messages
        .iter()
        .flat_map(|m| &m.blocks)
        .any(
            |b| matches!(b,ContentBlock::ToolResult {output,..} if output.starts_with("[masked:"))
        ));
}

#[tokio::test]
async fn resumed_replay_identity_is_rejected_before_credentials_on_wrong_account_or_model() {
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };
    struct NoCredential {
        calls: AtomicUsize,
    }
    impl piku_api::AccessTokenSource for NoCredential {
        fn binding(&self) -> &'static str {
            "account-registration-a"
        }
        fn token(
            &self,
        ) -> std::pin::Pin<
            Box<
                dyn std::future::Future<Output = Result<piku_api::AccessToken, piku_api::ApiError>>
                    + Send
                    + '_,
            >,
        > {
            self.calls.fetch_add(1, Ordering::Relaxed);
            Box::pin(async {
                Err(piku_api::ApiError::Provider(
                    "credentials must not be requested".into(),
                ))
            })
        }
    }
    for wrong_account in [true, false] {
        let source = Arc::new(NoCredential {
            calls: AtomicUsize::new(0),
        });
        let provider = piku_api::ResponsesProvider::new(source.clone()).unwrap();
        let mut opaque = envelope(std::path::Path::new("file"), "opaque-secret");
        if wrong_account {
            opaque.registration = "other-registration".into();
        }
        let mut session = Session::new("isolation".into());
        session.messages = exchange(opaque);
        let result = run_turn(
            "resume",
            &mut session,
            &provider,
            if wrong_account {
                "test-model"
            } else {
                "other-model"
            },
            &[],
            piku_tools::all_tool_definitions(),
            &crate::permission::AllowAll,
            &mut Sink::default(),
            Some(1),
            None,
        )
        .await;
        assert!(result
            .stream_error
            .unwrap()
            .contains("binding/model mismatch"));
        assert_eq!(source.calls.load(Ordering::Relaxed), 0);
    }
}
