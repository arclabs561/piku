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
