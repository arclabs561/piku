#[cfg(test)]
mod session {
    use super::tempdir;
    use crate::session::{ContentBlock, ConversationMessage, MessageRole, Session, UsageTracker};
    use piku_api::TokenUsage;

    #[test]
    fn new_session_is_empty() {
        let s = Session::new("test-1".to_string());
        assert_eq!(s.id, "test-1");
        assert!(s.messages.is_empty());
        assert_eq!(s.version, 1);
    }

    #[test]
    fn push_and_retrieve() {
        let mut s = Session::new("s".to_string());
        s.push(ConversationMessage::user("hello"));
        assert_eq!(s.messages.len(), 1);
        assert_eq!(s.messages[0].role, MessageRole::User);
    }

    #[test]
    fn estimated_tokens_nonzero_for_content() {
        let mut s = Session::new("s".to_string());
        s.push(ConversationMessage::user("a".repeat(400)));
        assert!(s.estimated_tokens() > 0);
    }

    #[test]
    fn save_and_load_roundtrip() {
        let dir = tempdir();
        let path = dir.join("session.json");
        let mut s = Session::new("rt".to_string());
        s.push(ConversationMessage::user("hi"));
        s.push(ConversationMessage::tool_result(
            "call_1".to_string(),
            "output".to_string(),
            false,
        ));
        s.save(&path).unwrap();

        let loaded = Session::load(&path).unwrap();
        assert_eq!(loaded.id, "rt");
        assert_eq!(loaded.messages.len(), 2);
    }

    #[test]
    fn content_block_text_len() {
        let b = ContentBlock::Text {
            text: "hello world".to_string(),
        };
        assert_eq!(b.text_len(), 11);
    }

    #[test]
    fn usage_tracker_accumulates() {
        let mut t = UsageTracker::default();
        t.record(TokenUsage {
            input_tokens: 10,
            output_tokens: 5,
            ..Default::default()
        });
        t.record(TokenUsage {
            input_tokens: 3,
            output_tokens: 2,
            ..Default::default()
        });
        assert_eq!(t.cumulative.input_tokens, 13);
        assert_eq!(t.cumulative.output_tokens, 7);
    }

    #[test]
    fn usage_tracker_from_session() {
        let mut s = Session::new("u".to_string());
        let mut msg = ConversationMessage::user("x");
        msg.usage = Some(TokenUsage {
            input_tokens: 100,
            output_tokens: 50,
            ..Default::default()
        });
        s.push(msg);
        let t = UsageTracker::from_session(&s);
        assert_eq!(t.cumulative.input_tokens, 100);
        assert_eq!(t.turns, 1);
    }

    #[test]
    fn finish_turn_increments_counter() {
        let mut t = UsageTracker::default();
        t.record(TokenUsage {
            input_tokens: 1,
            output_tokens: 1,
            ..Default::default()
        });
        t.finish_turn();
        assert_eq!(t.turns, 1);
        // current_turn resets
        assert_eq!(t.current_turn.input_tokens, 0);
    }
}

#[cfg(test)]
mod permissions {
    use super::tempdir;
    use crate::permission::{check_permission, AllowAll, DenyAll, PermissionOutcome};

    #[test]
    fn read_file_always_allowed() {
        let outcome = check_permission(
            "read_file",
            &serde_json::json!({ "path": "/any" }),
            &AllowAll,
        );
        assert!(matches!(outcome, PermissionOutcome::Allow));
    }

    #[test]
    fn list_dir_always_allowed() {
        let outcome = check_permission("list_dir", &serde_json::json!({}), &AllowAll);
        assert!(matches!(outcome, PermissionOutcome::Allow));
    }

    #[test]
    fn glob_always_allowed() {
        let outcome =
            check_permission("glob", &serde_json::json!({ "pattern": "*.rs" }), &AllowAll);
        assert!(matches!(outcome, PermissionOutcome::Allow));
    }

    #[test]
    fn grep_always_allowed() {
        let outcome = check_permission("grep", &serde_json::json!({ "pattern": "fn" }), &AllowAll);
        assert!(matches!(outcome, PermissionOutcome::Allow));
    }

    #[test]
    fn bash_needs_prompter_decision() {
        // With AllowAll, bash goes through and gets allowed
        let outcome = check_permission(
            "bash",
            &serde_json::json!({ "command": "echo hi" }),
            &AllowAll,
        );
        assert!(matches!(outcome, PermissionOutcome::Allow));
    }

    #[test]
    fn deny_all_blocks_bash() {
        let outcome = check_permission(
            "bash",
            &serde_json::json!({ "command": "echo hi" }),
            &DenyAll,
        );
        assert!(matches!(outcome, PermissionOutcome::Deny { .. }));
    }

    #[test]
    fn deny_all_blocks_write() {
        // write_file to existing file → Likely → goes to prompter → DenyAll
        let dir = tempdir();
        let path = dir.join("existing.txt");
        std::fs::write(&path, "x").unwrap();
        let outcome = check_permission(
            "write_file",
            &serde_json::json!({ "path": path, "content": "y" }),
            &DenyAll,
        );
        assert!(matches!(outcome, PermissionOutcome::Deny { .. }));
    }

    #[test]
    fn new_file_write_is_safe_even_with_deny_all() {
        // write_file to a non-existent path → Safe → auto-approved regardless of prompter
        let dir = tempdir();
        let path = dir.join("brand_new_file.txt");
        let outcome = check_permission(
            "write_file",
            &serde_json::json!({ "path": path, "content": "y" }),
            &DenyAll,
        );
        assert!(matches!(outcome, PermissionOutcome::Allow));
    }
}

#[cfg(test)]
mod agent_loop {
    #![allow(clippy::items_after_statements, clippy::unnecessary_literal_bound)]

    use std::pin::Pin;

    use futures_util::Stream;
    use piku_api::{ApiError, Event, MessageRequest, Provider, StopReason, TokenUsage};

    use super::tempdir;
    use crate::agent_loop::{run_turn, OutputSink};
    use crate::permission::AllowAll;
    use crate::session::Session;
    use piku_tools::all_tool_definitions;

    // -------------------------------------------------------------------
    // Mock provider that returns a fixed event sequence
    // -------------------------------------------------------------------

    struct MockProvider {
        events: Vec<Event>,
    }

    impl Provider for MockProvider {
        fn name(&self) -> &'static str {
            "mock"
        }

        fn stream_message(
            &self,
            _request: MessageRequest,
        ) -> Pin<Box<dyn Stream<Item = Result<Event, ApiError>> + Send + '_>> {
            let events = self.events.clone();
            Box::pin(async_stream::stream! {
                for event in events {
                    yield Ok(event);
                }
            })
        }
    }

    // -------------------------------------------------------------------
    // Mock sink that captures output
    // -------------------------------------------------------------------

    #[derive(Default)]
    struct CaptureSink {
        text: String,
        tool_starts: Vec<String>,
        tool_ends: Vec<(String, bool)>,
        denied: Vec<String>,
        complete: bool,
    }

    impl OutputSink for CaptureSink {
        fn on_text(&mut self, text: &str) {
            self.text.push_str(text);
        }
        fn on_tool_start(&mut self, tool_name: &str, _id: &str, _input: &serde_json::Value) {
            self.tool_starts.push(tool_name.to_string());
        }
        fn on_tool_end(
            &mut self,
            tool_name: &str,
            _result: &str,
            is_error: bool,
        ) -> crate::agent_loop::PostToolAction {
            self.tool_ends.push((tool_name.to_string(), is_error));
            crate::agent_loop::PostToolAction::Continue
        }
        fn on_permission_denied(&mut self, tool_name: &str, _reason: &str) {
            self.denied.push(tool_name.to_string());
        }
        fn on_turn_complete(&mut self, _usage: &TokenUsage, _iterations: u32) {
            self.complete = true;
        }
    }

    fn text_only_provider(text: &str) -> MockProvider {
        MockProvider {
            events: vec![
                Event::TextDelta {
                    text: text.to_string(),
                },
                Event::MessageStop {
                    stop_reason: StopReason::EndTurn,
                },
                Event::UsageDelta {
                    usage: TokenUsage {
                        input_tokens: 10,
                        output_tokens: 5,
                        ..Default::default()
                    },
                },
            ],
        }
    }

    #[tokio::test]
    async fn simple_text_turn() {
        let provider = text_only_provider("Hello, world!");
        let mut session = Session::new("t1".to_string());
        let mut sink = CaptureSink::default();

        run_turn(
            "hi",
            &mut session,
            &provider,
            "test-model",
            &["You are helpful.".to_string()],
            vec![],
            &AllowAll,
            &mut sink,
            None,
            None,
        )
        .await;

        assert_eq!(sink.text, "Hello, world!");
        assert!(sink.complete);
        // session should have user + assistant messages
        assert_eq!(session.messages.len(), 2);
    }

    #[tokio::test]
    async fn user_message_appended_to_session() {
        let provider = text_only_provider("ok");
        let mut session = Session::new("t2".to_string());
        let mut sink = CaptureSink::default();

        run_turn(
            "my question",
            &mut session,
            &provider,
            "m",
            &[],
            vec![],
            &AllowAll,
            &mut sink,
            None,
            None,
        )
        .await;

        let first = &session.messages[0];
        assert!(matches!(first.role, crate::session::MessageRole::User));
    }

    #[tokio::test]
    async fn tool_call_executes_read_file() {
        let dir = tempdir();
        let path = dir.join("data.txt");
        std::fs::write(&path, "file content here").unwrap();

        // Provider: one tool call then done
        let tool_input = serde_json::json!({ "path": path }).to_string();
        let provider = MockProvider {
            events: vec![
                Event::ToolUseStart {
                    id: "call_1".to_string(),
                    name: "read_file".to_string(),
                },
                Event::ToolUseDelta {
                    id: "call_1".to_string(),
                    partial_json: tool_input,
                },
                Event::ToolUseEnd {
                    id: "call_1".to_string(),
                },
                Event::MessageStop {
                    stop_reason: StopReason::ToolUse,
                },
                // second turn: just text
                Event::TextDelta {
                    text: "I read the file.".to_string(),
                },
                Event::MessageStop {
                    stop_reason: StopReason::EndTurn,
                },
            ],
        };

        let mut session = Session::new("t3".to_string());
        let mut sink = CaptureSink::default();

        run_turn(
            "read that file",
            &mut session,
            &provider,
            "m",
            &[],
            all_tool_definitions(),
            &AllowAll,
            &mut sink,
            None,
            None,
        )
        .await;

        assert!(sink.tool_starts.contains(&"read_file".to_string()));
        assert_eq!(sink.tool_ends[0].0, "read_file");
        assert!(!sink.tool_ends[0].1, "should not be an error");
        assert!(sink.text.contains("I read the file."));
    }

    #[tokio::test]
    async fn max_turns_stops_loop() {
        // Provider always requests a tool call → would loop forever without max_turns
        let events_per_call = vec![
            Event::ToolUseStart {
                id: "c".to_string(),
                name: "list_dir".to_string(),
            },
            Event::ToolUseDelta {
                id: "c".to_string(),
                partial_json: "{}".to_string(),
            },
            Event::ToolUseEnd {
                id: "c".to_string(),
            },
            Event::MessageStop {
                stop_reason: StopReason::ToolUse,
            },
        ];

        struct InfiniteProvider(Vec<Event>);
        impl Provider for InfiniteProvider {
            fn name(&self) -> &'static str {
                "infinite"
            }
            fn stream_message(
                &self,
                _: MessageRequest,
            ) -> Pin<Box<dyn Stream<Item = Result<Event, ApiError>> + Send + '_>> {
                let events = self.0.clone();
                Box::pin(async_stream::stream! {
                    for e in events { yield Ok(e); }
                })
            }
        }

        let provider = InfiniteProvider(events_per_call);
        let mut session = Session::new("max".to_string());
        let mut sink = CaptureSink::default();

        let result = run_turn(
            "go",
            &mut session,
            &provider,
            "m",
            &[],
            all_tool_definitions(),
            &AllowAll,
            &mut sink,
            Some(3),
            None,
        )
        .await;

        assert_eq!(result.iterations, 3);
        assert!(sink.complete);
    }
}

fn tempdir() -> std::path::PathBuf {
    let base = std::env::temp_dir().join(format!(
        "piku_rt_test_{}_{:?}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.subsec_nanos())
            .unwrap_or(0),
        std::thread::current().id(),
    ));
    std::fs::create_dir_all(&base).unwrap();
    base
}

// ---------------------------------------------------------------------------
// Extended P1/P2 tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod agent_loop_extended {
    use std::pin::Pin;

    use futures_util::Stream;
    use piku_api::{ApiError, Event, MessageRequest, Provider, StopReason, TokenUsage};

    use super::tempdir;
    use crate::agent_loop::{run_turn, OutputSink};
    use crate::permission::AllowAll;
    use crate::session::Session;
    use piku_tools::all_tool_definitions;

    #[derive(Default)]
    struct CaptureSink {
        text: String,
        tool_starts: Vec<String>,
        tool_ends: Vec<(String, bool)>,
        denied: Vec<String>,
    }

    impl OutputSink for CaptureSink {
        fn on_text(&mut self, t: &str) {
            self.text.push_str(t);
        }
        fn on_tool_start(&mut self, n: &str, _: &str, _: &serde_json::Value) {
            self.tool_starts.push(n.to_string());
        }
        fn on_tool_end(&mut self, n: &str, _: &str, e: bool) -> crate::agent_loop::PostToolAction {
            self.tool_ends.push((n.to_string(), e));
            crate::agent_loop::PostToolAction::Continue
        }
        fn on_permission_denied(&mut self, n: &str, _: &str) {
            self.denied.push(n.to_string());
        }
        fn on_turn_complete(&mut self, _: &TokenUsage, _: u32) {}
    }

    fn make_provider(events: Vec<Event>) -> impl Provider {
        struct P(Vec<Event>);
        impl Provider for P {
            fn name(&self) -> &'static str {
                "mock"
            }
            fn stream_message(
                &self,
                _: MessageRequest,
            ) -> Pin<Box<dyn Stream<Item = Result<Event, ApiError>> + Send + '_>> {
                let ev = self.0.clone();
                Box::pin(async_stream::stream! { for e in ev { yield Ok(e); } })
            }
        }
        P(events)
    }

    fn make_error_provider() -> impl Provider {
        struct EP;
        impl Provider for EP {
            fn name(&self) -> &'static str {
                "error"
            }
            fn stream_message(
                &self,
                _: MessageRequest,
            ) -> Pin<Box<dyn Stream<Item = Result<Event, ApiError>> + Send + '_>> {
                Box::pin(async_stream::stream! {
                    yield Ok(Event::TextDelta { text: "partial...".to_string() });
                    yield Err(ApiError::UnexpectedStreamEnd);
                })
            }
        }
        EP
    }

    #[tokio::test]
    async fn stream_error_sets_stream_error_field_and_no_panic() {
        // P1: stream error was silently swallowed, session could be corrupted
        let provider = make_error_provider();
        let mut session = Session::new("err".to_string());
        let mut sink = CaptureSink::default();

        let result = run_turn(
            "hi",
            &mut session,
            &provider,
            "test-model",
            &[],
            vec![],
            &AllowAll,
            &mut sink,
            None,
            None,
        )
        .await;

        assert!(result.stream_error.is_some(), "stream_error should be set");
        // session should have user message but no corrupted assistant message
        assert_eq!(
            session.messages.len(),
            1,
            "only user message, no partial assistant"
        );
    }

    #[tokio::test]
    async fn model_field_is_propagated_to_provider() {
        // P1: model was String::new() — verify it reaches the provider
        struct ModelCapture(std::sync::Arc<std::sync::Mutex<String>>);
        impl Provider for ModelCapture {
            fn name(&self) -> &'static str {
                "capture"
            }
            fn stream_message(
                &self,
                req: MessageRequest,
            ) -> Pin<Box<dyn Stream<Item = Result<Event, ApiError>> + Send + '_>> {
                *self.0.lock().unwrap() = req.model.clone();
                let ev = vec![Event::MessageStop {
                    stop_reason: StopReason::EndTurn,
                }];
                Box::pin(async_stream::stream! { for e in ev { yield Ok(e); } })
            }
        }

        let captured = std::sync::Arc::new(std::sync::Mutex::new(String::new()));
        let provider = ModelCapture(captured.clone());
        let mut session = Session::new("m".to_string());
        let mut sink = CaptureSink::default();

        run_turn(
            "hi",
            &mut session,
            &provider,
            "claude-sonnet-4.6",
            &[],
            vec![],
            &AllowAll,
            &mut sink,
            None,
            None,
        )
        .await;

        assert_eq!(*captured.lock().unwrap(), "claude-sonnet-4.6");
    }

    #[tokio::test]
    async fn unknown_tool_produces_error_result_in_session() {
        // P1: unknown tool name should produce an error ToolResult, not panic
        let events = vec![
            Event::ToolUseStart {
                id: "c1".to_string(),
                name: "nonexistent_tool".to_string(),
            },
            Event::ToolUseDelta {
                id: "c1".to_string(),
                partial_json: "{}".to_string(),
            },
            Event::ToolUseEnd {
                id: "c1".to_string(),
            },
            Event::MessageStop {
                stop_reason: StopReason::ToolUse,
            },
            Event::TextDelta {
                text: "done".to_string(),
            },
            Event::MessageStop {
                stop_reason: StopReason::EndTurn,
            },
        ];
        let provider = make_provider(events);
        let mut session = Session::new("unk".to_string());
        let mut sink = CaptureSink::default();

        let _result = run_turn(
            "call unknown",
            &mut session,
            &provider,
            "m",
            &[],
            all_tool_definitions(),
            &AllowAll,
            &mut sink,
            None,
            None,
        )
        .await;

        // tool_ends should have is_error=true for the unknown tool
        assert!(
            sink.tool_ends.iter().any(|(_, is_err)| *is_err),
            "unknown tool should produce error result"
        );
    }

    #[tokio::test]
    async fn multiple_tool_calls_in_single_turn_all_execute() {
        // P2: multi-tool turn — two reads in one response
        let dir = tempdir();
        let f1 = dir.join("a.txt");
        let f2 = dir.join("b.txt");
        std::fs::write(&f1, "content_a").unwrap();
        std::fs::write(&f2, "content_b").unwrap();

        let i1 = serde_json::json!({ "path": f1 }).to_string();
        let i2 = serde_json::json!({ "path": f2 }).to_string();

        let events = vec![
            Event::ToolUseStart {
                id: "t1".to_string(),
                name: "read_file".to_string(),
            },
            Event::ToolUseDelta {
                id: "t1".to_string(),
                partial_json: i1,
            },
            Event::ToolUseEnd {
                id: "t1".to_string(),
            },
            Event::ToolUseStart {
                id: "t2".to_string(),
                name: "read_file".to_string(),
            },
            Event::ToolUseDelta {
                id: "t2".to_string(),
                partial_json: i2,
            },
            Event::ToolUseEnd {
                id: "t2".to_string(),
            },
            Event::MessageStop {
                stop_reason: StopReason::ToolUse,
            },
            Event::TextDelta {
                text: "read both".to_string(),
            },
            Event::MessageStop {
                stop_reason: StopReason::EndTurn,
            },
        ];
        let provider = make_provider(events);
        let mut session = Session::new("multi".to_string());
        let mut sink = CaptureSink::default();

        run_turn(
            "read both files",
            &mut session,
            &provider,
            "m",
            &[],
            all_tool_definitions(),
            &AllowAll,
            &mut sink,
            None,
            None,
        )
        .await;

        assert_eq!(sink.tool_starts.len(), 2);
        assert_eq!(sink.tool_ends.len(), 2);
        assert!(
            sink.tool_ends.iter().all(|(_, e)| !e),
            "both reads should succeed"
        );
    }

    #[tokio::test]
    async fn permission_denied_tool_still_runs_others_in_same_turn() {
        // P2: partial denial — denied tool is skipped, others execute
        use crate::permission::{PermissionOutcome, PermissionPrompter, PermissionRequest};

        struct DenyBash;
        impl PermissionPrompter for DenyBash {
            fn decide(&self, req: &PermissionRequest) -> PermissionOutcome {
                if req.tool_name == "bash" {
                    PermissionOutcome::Deny {
                        reason: "bash denied in test".to_string(),
                    }
                } else {
                    PermissionOutcome::Allow
                }
            }
        }

        let dir = tempdir();
        let f = dir.join("ok.txt");
        std::fs::write(&f, "hello").unwrap();
        let input = serde_json::json!({ "path": f }).to_string();

        let events = vec![
            // bash call — will be denied
            Event::ToolUseStart {
                id: "b1".to_string(),
                name: "bash".to_string(),
            },
            Event::ToolUseDelta {
                id: "b1".to_string(),
                partial_json: r#"{"command":"echo hi"}"#.to_string(),
            },
            Event::ToolUseEnd {
                id: "b1".to_string(),
            },
            // read_file call — should still execute
            Event::ToolUseStart {
                id: "r1".to_string(),
                name: "read_file".to_string(),
            },
            Event::ToolUseDelta {
                id: "r1".to_string(),
                partial_json: input,
            },
            Event::ToolUseEnd {
                id: "r1".to_string(),
            },
            Event::MessageStop {
                stop_reason: StopReason::ToolUse,
            },
            Event::MessageStop {
                stop_reason: StopReason::EndTurn,
            },
        ];
        let provider = make_provider(events);
        let mut session = Session::new("partial".to_string());
        let mut sink = CaptureSink::default();

        run_turn(
            "try both",
            &mut session,
            &provider,
            "m",
            &[],
            all_tool_definitions(),
            &DenyBash,
            &mut sink,
            None,
            None,
        )
        .await;

        assert!(
            sink.denied.contains(&"bash".to_string()),
            "bash should be denied"
        );
        assert!(
            sink.tool_starts.contains(&"read_file".to_string()),
            "read_file should still run"
        );
    }

    #[tokio::test]
    async fn build_request_system_role_injected_as_user_with_wrapper() {
        // System-role messages (from compaction) must reach the API as
        // `<system>...</system>`-wrapped user messages, not as system prompts.
        // We verify this by capturing the actual RequestMessage list sent to
        // the provider, not just the session JSON.
        use crate::session::{ContentBlock, ConversationMessage, MessageRole};
        use futures_util::Stream;
        use piku_api::{ApiError, Event, MessageRequest, Provider, StopReason};
        use std::pin::Pin;
        use std::sync::{Arc, Mutex};

        struct Capture(Arc<Mutex<Vec<MessageRequest>>>);
        impl Provider for Capture {
            fn name(&self) -> &'static str {
                "capture"
            }
            fn stream_message(
                &self,
                req: MessageRequest,
            ) -> Pin<Box<dyn Stream<Item = Result<Event, ApiError>> + Send + '_>> {
                self.0.lock().unwrap().push(req);
                Box::pin(async_stream::stream! {
                    yield Ok(Event::TextDelta { text: "ok".to_string() });
                    yield Ok(Event::MessageStop { stop_reason: StopReason::EndTurn });
                })
            }
        }

        let captured = Arc::new(Mutex::new(Vec::new()));
        let provider = Capture(captured.clone());

        let mut session = Session::new("s".to_string());
        // Inject a System-role message as compaction would
        session.messages.push(ConversationMessage {
            role: MessageRole::System,
            blocks: vec![ContentBlock::Text {
                text: "compact summary here".to_string(),
            }],
            usage: None,
            importance: None,
        });
        session.push(ConversationMessage::user("hello"));

        let mut sink = crate::tests::agent_loop_extended::CaptureSink::default();
        crate::agent_loop::run_turn(
            "follow-up",
            &mut session,
            &provider,
            "m",
            &[],
            vec![],
            &crate::permission::AllowAll,
            &mut sink,
            None,
            None,
        )
        .await;

        let reqs = captured.lock().unwrap();
        assert!(!reqs.is_empty(), "provider should have been called");
        let messages = &reqs[0].messages;

        // Find the message that wraps the system summary
        let system_wrapper = messages.iter().find(|m| {
            m.content.iter().any(|c| {
                if let piku_api::RequestContent::Text { text } = c {
                    text.contains("<system>") && text.contains("compact summary here")
                } else {
                    false
                }
            })
        });

        assert!(
            system_wrapper.is_some(),
            "System-role message should be wrapped in <system>...</system> as a user message.\n\
             Actual messages: {:?}",
            messages
                .iter()
                .map(|m| (&m.role, m.content.len()))
                .collect::<Vec<_>>()
        );
        assert_eq!(
            system_wrapper.unwrap().role, "user",
            "System-role message wrapper must use 'user' role (Anthropic doesn't support mid-conversation system messages)"
        );
    }
}

#[cfg(test)]
mod session_extended {
    use crate::session::{ContentBlock, ConversationMessage, Session};

    #[test]
    fn load_malformed_json_returns_invalid_data_error() {
        // P1: corrupted session file must return error, not panic
        let dir = super::tempdir();
        let path = dir.join("bad.json");
        std::fs::write(&path, "not json {{ garbage").unwrap();
        let result = Session::load(&path);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().kind(), std::io::ErrorKind::InvalidData);
    }

    #[test]
    fn estimated_tokens_counts_all_block_types() {
        // P2: ToolUse and ToolResult blocks must also count
        let mut s = Session::new("t".to_string());
        s.messages.push(ConversationMessage::assistant(
            vec![
                ContentBlock::ToolUse {
                    id: "c1".to_string(),
                    name: "read_file".to_string(),
                    input: serde_json::json!({ "path": "/some/long/path/to/file.rs" }),
                },
                ContentBlock::ToolResult {
                    tool_use_id: "c1".to_string(),
                    output: "a".repeat(400),
                    is_error: false,
                },
            ],
            None,
        ));
        let tokens = s.estimated_tokens();
        assert!(tokens > 100, "expected >100 estimated tokens, got {tokens}");
    }

    #[test]
    fn empty_session_save_load_roundtrip() {
        let dir = super::tempdir();
        let path = dir.join("empty.json");
        let s = Session::new("empty-id".to_string());
        s.save(&path).unwrap();
        let loaded = Session::load(&path).unwrap();
        assert_eq!(loaded.id, "empty-id");
        assert!(loaded.messages.is_empty());
    }
}

#[cfg(test)]
mod prompt_extended {
    use super::tempdir;
    use crate::prompt::build_system_prompt;

    #[test]
    fn contains_cache_boundary_marker() {
        let dir = tempdir();
        let sections = build_system_prompt(&dir, "2026-04-03", "test-model", &[]);
        let full = sections.join("\n\n");
        assert!(
            full.contains("__PIKU_SYSTEM_PROMPT_DYNAMIC_BOUNDARY__"),
            "system prompt must contain cache boundary marker"
        );
    }

    #[test]
    fn contains_cwd_and_model() {
        let dir = tempdir();
        let sections = build_system_prompt(&dir, "2026-04-03", "my-model", &[]);
        let full = sections.join("\n\n");
        assert!(
            full.contains("my-model"),
            "model name should appear in prompt"
        );
        assert!(full.contains("2026-04-03"), "date should appear in prompt");
    }

    #[test]
    fn loads_piku_md_from_directory() {
        let dir = tempdir();
        std::fs::write(dir.join("PIKU.md"), "# Project rules\nAlways use tabs.").unwrap();
        let sections = build_system_prompt(&dir, "2026-04-03", "m", &[]);
        let full = sections.join("\n\n");
        assert!(
            full.contains("Always use tabs"),
            "PIKU.md content should be in prompt"
        );
    }

    #[test]
    fn truncates_large_piku_md() {
        let dir = tempdir();
        let large = "x".repeat(5000); // > MAX_PER_FILE (4000)
        std::fs::write(dir.join("PIKU.md"), &large).unwrap();
        let sections = build_system_prompt(&dir, "2026-04-03", "m", &[]);
        let full = sections.join("\n\n");
        assert!(
            full.contains("[truncated]"),
            "large PIKU.md should be truncated"
        );
    }
}

// ---------------------------------------------------------------------------
// Attempt tree simulation: full lifecycle without external deps
// ---------------------------------------------------------------------------

#[cfg(test)]
mod attempt_tree_simulation {
    use super::tempdir;
    use crate::embed_memory::{format_attempt_trees, AttemptTree, EntryType, MemoryStore, Outcome};

    /// Deterministic embedding from a seed string (hash-based, 768d).
    fn embed(text: &str) -> Vec<f32> {
        let mut v = vec![0.0f32; 768];
        // Simple hash-based embedding: spread bytes across dimensions
        for (i, b) in text.bytes().enumerate() {
            let idx = (i * 37 + b as usize) % 768;
            v[idx] += (f32::from(b) - 96.0) * 0.01;
        }
        // Normalize
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in &mut v {
                *x /= norm;
            }
        }
        v
    }

    /// Simulate: agent debugging a compile error, tries 3 approaches.
    /// Session 1: record attempts. Session 2: query and find them.
    #[test]
    fn full_lifecycle_record_then_query() {
        let dir = tempdir();
        let store_path = dir.join("memories.json");

        // === Session 1: agent encounters a compile error ===
        let mut store = MemoryStore::default();

        // Root attempt: the goal
        let root = store.record_attempt(
            "fix lifetime error in handler function".to_string(),
            "add explicit lifetime annotation".to_string(),
            None,
            embed("fix lifetime error | add explicit lifetime annotation"),
            7,
        );
        store.record_outcome(
            root,
            Outcome::Failure,
            Some("introduced second lifetime that conflicted with trait bound".to_string()),
        );

        // Sibling: different approach
        let attempt2 = store.record_attempt(
            "fix lifetime error in handler function".to_string(),
            "clone the borrowed value to avoid lifetime".to_string(),
            Some(root),
            embed("fix lifetime error | clone borrowed value"),
            6,
        );
        store.record_outcome(
            attempt2,
            Outcome::Failure,
            Some("value does not implement Clone".to_string()),
        );

        // Sibling: what worked
        let attempt3 = store.record_attempt(
            "fix lifetime error in handler function".to_string(),
            "restructure to use owned String instead of &str".to_string(),
            Some(root),
            embed("fix lifetime error | use owned String"),
            8,
        );
        store.record_outcome(
            attempt3,
            Outcome::Success,
            Some("compiles and tests pass".to_string()),
        );

        store.save(&store_path).unwrap();
        assert_eq!(store.entries.len(), 3);

        // === Session 2: new agent faces similar problem ===
        let store2 = MemoryStore::load(&store_path);
        assert_eq!(store2.entries.len(), 3);

        // Query with a similar goal
        let query_embed = embed("fix lifetime error in handler function");
        let trees = store2.find_attempt_trees(&query_embed, "fix lifetime error in handler", 5);

        assert!(!trees.is_empty(), "should find the attempt tree");

        // Verify tree structure
        let tree = &trees[0];
        assert!(tree.goal.as_deref().unwrap_or("").contains("lifetime"));
        // Root should have children
        assert!(!tree.children.is_empty(), "root should have child attempts");

        // Verify the formatted output is useful
        let formatted = format_attempt_trees(&trees);
        assert!(formatted.contains("[FAIL]"), "should show failed attempts");
        assert!(formatted.contains("[OK]"), "should show successful attempt");
        assert!(
            formatted.contains("owned String"),
            "should show the successful approach"
        );
        assert!(
            formatted.contains("does not implement Clone"),
            "should show failure reasons"
        );
    }

    /// Simulate: multiple independent goals in the same store.
    #[test]
    fn multiple_goals_stay_separate() {
        let mut store = MemoryStore::default();

        // Goal A: fix a compile error
        let a1 = store.record_attempt(
            "fix compile error".to_string(),
            "add missing import".to_string(),
            None,
            embed("fix compile error | add missing import"),
            6,
        );
        store.record_outcome(a1, Outcome::Success, None);

        // Goal B: optimize query performance
        let b1 = store.record_attempt(
            "optimize query performance".to_string(),
            "add database index".to_string(),
            None,
            embed("optimize query performance | add database index"),
            7,
        );
        store.record_outcome(b1, Outcome::Failure, Some("index too large".to_string()));

        let b2 = store.record_attempt(
            "optimize query performance".to_string(),
            "rewrite as batch query".to_string(),
            Some(b1),
            embed("optimize query performance | batch query"),
            7,
        );
        store.record_outcome(b2, Outcome::Success, None);

        // Query for compile error -- should find goal A, not B
        let trees = store.find_attempt_trees(&embed("fix compile error"), "fix compile error", 5);
        // The tree for "fix compile error" should be found
        assert!(!trees.is_empty());
        let first_goal = trees[0].goal.as_deref().unwrap_or("");
        assert!(
            first_goal.contains("compile"),
            "first tree should match the query goal, got: {first_goal}"
        );
    }

    /// Simulate: eviction preserves unresolved trees.
    #[test]
    fn eviction_preserves_unresolved_across_sessions() {
        let dir = tempdir();
        let store_path = dir.join("memories.json");

        // Session 1: record a failed attempt tree (unresolved)
        let mut store = MemoryStore::default();
        let root = store.record_attempt(
            "fix flaky test".to_string(),
            "add retry logic".to_string(),
            None,
            embed("fix flaky test | retry"),
            4, // low importance -- normally evictable
        );
        store.record_outcome(
            root,
            Outcome::Failure,
            Some("retry masks the real bug".to_string()),
        );
        let child = store.record_attempt(
            "fix flaky test".to_string(),
            "increase timeout".to_string(),
            Some(root),
            embed("fix flaky test | timeout"),
            4,
        );
        store.record_outcome(
            child,
            Outcome::Failure,
            Some("still flaky at 30s".to_string()),
        );
        // Make them old
        for entry in &mut store.entries {
            entry.last_accessed = 0;
            entry.created_at = 0;
        }
        store.save(&store_path).unwrap();

        // Session 2: run maintenance -- unresolved tree should survive
        let mut store2 = MemoryStore::load(&store_path);
        let (stale, weak) = store2.maintain();
        assert_eq!(
            stale + weak,
            0,
            "unresolved attempt tree should not be evicted"
        );
        assert_eq!(store2.valid_count(), 2);
    }

    /// Simulate: eviction CAN remove resolved trees after aging.
    #[test]
    fn eviction_removes_resolved_tree_leaves() {
        let mut store = MemoryStore::default();

        // Resolved tree: root -> child (success)
        let root = store.record_attempt(
            "goal".to_string(),
            "approach A".to_string(),
            None,
            embed("goal | approach A"),
            3, // low importance
        );
        store.record_outcome(root, Outcome::Failure, None);
        let child = store.record_attempt(
            "goal".to_string(),
            "approach B".to_string(),
            Some(root),
            embed("goal | approach B"),
            3,
        );
        store.record_outcome(child, Outcome::Success, None);

        // Make old
        for entry in &mut store.entries {
            entry.last_accessed = 0;
            entry.created_at = 0;
        }

        let (stale, weak) = store.maintain();
        // At least the leaf (child with no children) should be evictable
        // since the tree is resolved and both are old+low-importance
        assert!(
            stale + weak >= 1,
            "resolved tree leaves should be evictable"
        );
    }

    /// Simulate: extraction parser handles mixed output correctly.
    #[test]
    fn extraction_parser_simulation() {
        // Simulate what the LLM would return at compaction
        let llm_response = "\
- [importance:8] The handler function requires owned String, not &str {tags: rust, lifetime}
- [attempt] goal: fix lifetime error in handler | approach: add explicit lifetime | outcome: failure | detail: conflicted with trait bound
- [attempt] goal: fix lifetime error in handler | approach: use owned String | outcome: success | detail: compiles and tests pass
- [importance:6] Project uses tokio 1.x runtime {tags: rust, async}";

        let facts = crate::embed_memory::tests::parse_extraction_response_pub(llm_response);
        assert_eq!(facts.len(), 2, "should extract 2 facts (not attempt lines)");
        assert!(facts[0].0.contains("owned String"));
        assert!(facts[1].0.contains("tokio"));

        let attempts = crate::embed_memory::tests::parse_attempt_lines_pub(llm_response);
        assert_eq!(attempts.len(), 2, "should extract 2 attempts");
        assert_eq!(attempts[0].2, "failure");
        assert_eq!(attempts[1].2, "success");
    }

    /// Simulate: compaction dedup skips already-recorded attempts.
    #[test]
    fn compaction_dedup_skips_existing() {
        let mut store = MemoryStore::default();
        let e = embed("fix bug | add null check");

        // Agent manually recorded this attempt during the session
        let id = store.record_attempt(
            "fix bug".to_string(),
            "add null check".to_string(),
            None,
            e.clone(),
            7,
        );
        store.record_outcome(id, Outcome::Success, Some("fixed".to_string()));

        // Now simulate what compaction would do: search for similar before inserting
        let similar = store.search(&e, 1);
        assert!(
            similar
                .first()
                .is_some_and(|s| s.similarity > 0.85 && s.entry.entry_type == EntryType::Attempt),
            "should find the existing attempt as near-duplicate"
        );
        // Compaction would skip this attempt (similarity > 0.85)
    }

    /// Simulate: subagent prompt injection includes attempt trees.
    #[test]
    fn subagent_prompt_includes_attempts() {
        let mut store = MemoryStore::default();

        // Record some attempts
        let root = store.record_attempt(
            "deploy to production".to_string(),
            "use blue-green deployment".to_string(),
            None,
            embed("deploy to production | blue-green"),
            7,
        );
        store.record_outcome(
            root,
            Outcome::Failure,
            Some("load balancer config missing".to_string()),
        );
        let child = store.record_attempt(
            "deploy to production".to_string(),
            "use rolling deployment".to_string(),
            Some(root),
            embed("deploy to production | rolling"),
            7,
        );
        store.record_outcome(child, Outcome::Success, None);

        // Simulate what spawn_agent does: embed the task, find attempt trees
        let task_embed = embed("deploy the new version to production");
        let trees = store.find_attempt_trees(&task_embed, "deploy to production", 3);
        let formatted = format_attempt_trees(&trees);

        assert!(
            !formatted.is_empty(),
            "should find relevant attempt trees for subagent"
        );
        assert!(
            formatted.contains("Prior Attempts"),
            "should have the header"
        );
        assert!(
            formatted.contains("blue-green"),
            "should include the failed approach"
        );
        assert!(
            formatted.contains("rolling"),
            "should include the successful approach"
        );
    }

    /// Simulate: tree formatting renders correct structure.
    #[test]
    fn tree_formatting_structure() {
        let tree = AttemptTree {
            id: 0,
            approach: "root approach".to_string(),
            goal: Some("fix the system".to_string()),
            outcome: Some(Outcome::Failure),
            outcome_detail: Some("too broad".to_string()),
            children: vec![
                AttemptTree {
                    id: 1,
                    approach: "narrow to module A".to_string(),
                    goal: None,
                    outcome: Some(Outcome::Failure),
                    outcome_detail: Some("wrong module".to_string()),
                    children: vec![],
                },
                AttemptTree {
                    id: 2,
                    approach: "narrow to module B".to_string(),
                    goal: None,
                    outcome: Some(Outcome::Success),
                    outcome_detail: Some("found the bug".to_string()),
                    children: vec![AttemptTree {
                        id: 3,
                        approach: "apply fix to module B".to_string(),
                        goal: None,
                        outcome: Some(Outcome::Success),
                        outcome_detail: None,
                        children: vec![],
                    }],
                },
            ],
        };
        let formatted = tree.format(0);

        // Verify hierarchy through indentation
        assert!(formatted.contains("goal: fix the system"));
        assert!(formatted.contains("[FAIL] root approach -- too broad"));
        assert!(formatted.contains("  [FAIL] narrow to module A -- wrong module"));
        assert!(formatted.contains("  [OK] narrow to module B -- found the bug"));
        assert!(formatted.contains("    [OK] apply fix to module B"));
    }

    /// Simulate: persistence roundtrip preserves attempt tree across sessions.
    #[test]
    fn persistence_roundtrip_preserves_tree() {
        let dir = tempdir();
        let path = dir.join("mem.json");

        // Session 1: build a tree
        let mut store = MemoryStore::default();
        let root = store.record_attempt(
            "goal".to_string(),
            "approach 1".to_string(),
            None,
            embed("goal | approach 1"),
            7,
        );
        store.record_outcome(root, Outcome::Failure, Some("reason".to_string()));
        let child = store.record_attempt(
            "goal".to_string(),
            "approach 2".to_string(),
            Some(root),
            embed("goal | approach 2"),
            8,
        );
        store.record_outcome(child, Outcome::Success, Some("worked".to_string()));
        store.save(&path).unwrap();

        // Session 2: load and verify
        let loaded = MemoryStore::load(&path);
        assert_eq!(loaded.entries.len(), 2);
        assert_eq!(loaded.entries[0].entry_type, EntryType::Attempt);
        assert_eq!(loaded.entries[0].goal.as_deref(), Some("goal"));
        assert_eq!(loaded.entries[0].outcome, Some(Outcome::Failure));
        assert_eq!(loaded.entries[1].parent, Some(root));
        assert_eq!(loaded.entries[1].outcome, Some(Outcome::Success));

        // Verify tree operations work on loaded data
        let children = loaded.children(root);
        assert_eq!(children.len(), 1);
        assert_eq!(children[0].content, "approach 2");

        let path_to_root = loaded.path_to_root(child);
        assert_eq!(path_to_root.len(), 2);
    }
}
