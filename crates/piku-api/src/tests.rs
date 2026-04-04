#[cfg(test)]
mod sse_parser {
    use crate::sse::SseParser;

    #[test]
    fn parses_single_event() {
        let mut p = SseParser::new();
        assert!(p.feed_line("event: content_block_delta").is_none());
        assert!(p.feed_line("data: {\"type\":\"delta\"}").is_none());
        let ev = p.feed_line("").unwrap();
        assert_eq!(ev.event_type.as_deref(), Some("content_block_delta"));
        assert_eq!(ev.data, "{\"type\":\"delta\"}");
    }

    #[test]
    fn ignores_comments_and_id() {
        let mut p = SseParser::new();
        p.feed_line(": this is a comment");
        p.feed_line("id: 42");
        assert!(p.feed_line("data: hello").is_none());
        let ev = p.feed_line("").unwrap();
        assert_eq!(ev.data, "hello");
        assert!(ev.event_type.is_none());
    }

    #[test]
    fn strips_trailing_newline_from_data() {
        let mut p = SseParser::new();
        p.feed_line("data: line");
        let ev = p.feed_line("").unwrap();
        assert_eq!(ev.data, "line");
    }

    #[test]
    fn no_dispatch_on_empty_data() {
        let mut p = SseParser::new();
        assert!(p.feed_line("").is_none()); // blank line with no data buffered
    }

    #[test]
    fn multiple_events_sequential() {
        let mut p = SseParser::new();
        p.feed_line("data: first");
        let ev1 = p.feed_line("").unwrap();
        p.feed_line("data: second");
        let ev2 = p.feed_line("").unwrap();
        assert_eq!(ev1.data, "first");
        assert_eq!(ev2.data, "second");
    }

    #[test]
    fn event_type_resets_between_events() {
        let mut p = SseParser::new();
        p.feed_line("event: ping");
        p.feed_line("data: {}");
        let ev1 = p.feed_line("").unwrap();
        assert_eq!(ev1.event_type.as_deref(), Some("ping"));

        // next event has no event: field
        p.feed_line("data: bare");
        let ev2 = p.feed_line("").unwrap();
        assert!(ev2.event_type.is_none());
    }
}

#[cfg(test)]
mod openai_sse_parsing {
    use crate::openai_compat::parse_openai_sse;
    use crate::types::{Event, StopReason};

    #[test]
    fn done_produces_no_events() {
        let events = parse_openai_sse("[DONE]").unwrap();
        assert!(events.is_empty());
    }

    #[test]
    fn text_delta() {
        let data = r#"{"choices":[{"delta":{"content":"hello"},"finish_reason":null}]}"#;
        let events = parse_openai_sse(data).unwrap();
        assert_eq!(events.len(), 1);
        assert!(matches!(&events[0], Event::TextDelta { text } if text == "hello"));
    }

    #[test]
    fn empty_text_delta_skipped() {
        let data = r#"{"choices":[{"delta":{"content":""},"finish_reason":null}]}"#;
        let events = parse_openai_sse(data).unwrap();
        assert!(events.is_empty());
    }

    #[test]
    fn finish_reason_stop() {
        let data = r#"{"choices":[{"delta":{},"finish_reason":"stop"}]}"#;
        let events = parse_openai_sse(data).unwrap();
        assert!(matches!(
            &events[0],
            Event::MessageStop {
                stop_reason: StopReason::EndTurn
            }
        ));
    }

    #[test]
    fn finish_reason_tool_calls() {
        let data = r#"{"choices":[{"delta":{},"finish_reason":"tool_calls"}]}"#;
        let events = parse_openai_sse(data).unwrap();
        assert!(matches!(
            &events[0],
            Event::MessageStop {
                stop_reason: StopReason::ToolUse
            }
        ));
    }

    #[test]
    fn tool_call_start_and_delta() {
        // First chunk: has id + name (start)
        let start = r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_abc","function":{"name":"read_file","arguments":""}}]},"finish_reason":null}]}"#;
        let events = parse_openai_sse(start).unwrap();
        assert!(matches!(&events[0], Event::ToolUseStart { name, .. } if name == "read_file"));

        // Second chunk: argument delta
        let delta = r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_abc","function":{"arguments":"{\"path\":"}}]},"finish_reason":null}]}"#;
        let events2 = parse_openai_sse(delta).unwrap();
        assert!(
            matches!(&events2[0], Event::ToolUseDelta { partial_json, .. } if partial_json.contains("path"))
        );
    }

    #[test]
    fn usage_in_final_chunk() {
        let data = r#"{"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":20}}"#;
        let events = parse_openai_sse(data).unwrap();
        assert!(matches!(
            &events[0],
            Event::UsageDelta { usage } if usage.input_tokens == 10 && usage.output_tokens == 20
        ));
    }

    #[test]
    fn bad_json_returns_parse_error() {
        let result = parse_openai_sse("not json at all");
        assert!(result.is_err());
    }
}

#[cfg(test)]
mod token_usage {
    use crate::types::TokenUsage;

    #[test]
    fn accumulate() {
        let mut a = TokenUsage {
            input_tokens: 10,
            output_tokens: 5,
            ..Default::default()
        };
        let b = TokenUsage {
            input_tokens: 3,
            output_tokens: 7,
            ..Default::default()
        };
        a.accumulate(&b);
        assert_eq!(a.input_tokens, 13);
        assert_eq!(a.output_tokens, 12);
    }

    #[test]
    fn total_tokens() {
        let u = TokenUsage {
            input_tokens: 100,
            output_tokens: 50,
            ..Default::default()
        };
        assert_eq!(u.total_tokens(), 150);
    }
}

#[cfg(test)]
mod openai_body_builder {
    use crate::openai_compat::build_openai_body;
    use crate::types::{MessageRequest, RequestContent, RequestMessage, ToolDefinition};

    fn simple_request(prompt: &str) -> MessageRequest {
        MessageRequest {
            model: "test-model".to_string(),
            max_tokens: 100,
            messages: vec![RequestMessage {
                role: "user".to_string(),
                content: vec![RequestContent::Text {
                    text: prompt.to_string(),
                }],
            }],
            system: None,
            tools: None,
            stream: true,
        }
    }

    #[test]
    fn system_prompt_injected() {
        use crate::types::SystemBlock;
        let mut req = simple_request("hi");
        req.system = Some(vec![SystemBlock::text("be helpful".to_string())]);
        let body = build_openai_body(&req);
        let msgs = body["messages"].as_array().unwrap();
        assert_eq!(msgs[0]["role"].as_str().unwrap(), "system");
        assert_eq!(msgs[0]["content"].as_str().unwrap(), "be helpful");
    }

    #[test]
    fn user_message_as_string() {
        let req = simple_request("hello");
        let body = build_openai_body(&req);
        let msgs = body["messages"].as_array().unwrap();
        assert_eq!(msgs[0]["role"].as_str().unwrap(), "user");
        assert_eq!(msgs[0]["content"].as_str().unwrap(), "hello");
    }

    #[test]
    fn tools_emitted_correctly() {
        let mut req = simple_request("use a tool");
        req.tools = Some(vec![ToolDefinition {
            name: "read_file".to_string(),
            description: "reads a file".to_string(),
            input_schema: serde_json::json!({ "type": "object" }),
        }]);
        let body = build_openai_body(&req);
        let tools = body["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["type"].as_str().unwrap(), "function");
        assert_eq!(tools[0]["function"]["name"].as_str().unwrap(), "read_file");
        assert_eq!(body["tool_choice"].as_str().unwrap(), "auto");
    }

    #[test]
    fn no_tools_key_when_empty() {
        let req = simple_request("no tools");
        let body = build_openai_body(&req);
        assert!(body.get("tools").is_none());
    }

    #[test]
    fn tool_result_emits_tool_role_message() {
        let req = MessageRequest {
            model: "m".to_string(),
            max_tokens: 10,
            messages: vec![RequestMessage {
                role: "user".to_string(),
                content: vec![RequestContent::ToolResult {
                    tool_use_id: "call_1".to_string(),
                    content: "file contents".to_string(),
                    is_error: Some(false),
                }],
            }],
            system: None,
            tools: None,
            stream: true,
        };
        let body = build_openai_body(&req);
        let msgs = body["messages"].as_array().unwrap();
        assert_eq!(msgs[0]["role"].as_str().unwrap(), "tool");
        assert_eq!(msgs[0]["tool_call_id"].as_str().unwrap(), "call_1");
        assert_eq!(msgs[0]["content"].as_str().unwrap(), "file contents");
    }

    #[test]
    fn error_tool_result_still_uses_tool_role() {
        // P1 fix: error results must NOT use "user" role
        let req = MessageRequest {
            model: "m".to_string(),
            max_tokens: 10,
            messages: vec![RequestMessage {
                role: "user".to_string(),
                content: vec![RequestContent::ToolResult {
                    tool_use_id: "call_err".to_string(),
                    content: "something failed".to_string(),
                    is_error: Some(true),
                }],
            }],
            system: None,
            tools: None,
            stream: true,
        };
        let body = build_openai_body(&req);
        let msgs = body["messages"].as_array().unwrap();
        // Must be "tool" not "user"
        assert_eq!(msgs[0]["role"].as_str().unwrap(), "tool");
        // Error content should be prefixed
        let content = msgs[0]["content"].as_str().unwrap();
        assert!(
            content.contains("Error:"),
            "expected Error: prefix, got: {content}"
        );
    }

    #[test]
    fn explicit_empty_tools_produces_no_tools_key() {
        let mut req = simple_request("no tools");
        req.tools = Some(vec![]);
        let body = build_openai_body(&req);
        assert!(body.get("tools").is_none());
        assert!(body.get("tool_choice").is_none());
    }

    #[test]
    fn unknown_finish_reason_maps_to_other() {
        use crate::openai_compat::parse_openai_sse;
        use crate::types::Event;
        let data = r#"{"choices":[{"delta":{},"finish_reason":"content_filter"}]}"#;
        let events = parse_openai_sse(data).unwrap();
        assert!(matches!(
            &events[0],
            Event::MessageStop { stop_reason: crate::types::StopReason::Other(s) } if s == "content_filter"
        ));
    }
}

#[cfg(test)]
mod anthropic_sse_parsing {
    use crate::anthropic::parse_anthropic_sse_pub as parse;
    use crate::types::{Event, StopReason};

    // Helper: use a borrowed event_type for the parser.
    fn et(s: &str) -> &str {
        s
    }

    #[test]
    fn done_produces_no_events() {
        assert!(parse(None, "[DONE]").unwrap().is_empty());
    }

    #[test]
    fn ping_produces_no_events() {
        let data = r#"{"type":"ping"}"#;
        assert!(parse(Some(et("ping")), data).unwrap().is_empty());
    }

    #[test]
    fn content_block_start_tool_use_emits_tool_use_start() {
        let data = r#"{"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_01","name":"read_file"}}"#;
        let events = parse(Some(et("content_block_start")), data).unwrap();
        assert_eq!(events.len(), 1);
        assert!(
            matches!(&events[0], Event::ToolUseStart { id, name } if id == "toolu_01" && name == "read_file")
        );
    }

    #[test]
    fn content_block_start_text_produces_no_events() {
        let data =
            r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#;
        let events = parse(Some(et("content_block_start")), data).unwrap();
        assert!(events.is_empty());
    }

    #[test]
    fn content_block_start_missing_content_block_key_does_not_panic() {
        // P1: was direct index `raw.rest["content_block"]` — would panic
        let data = r#"{"type":"content_block_start","index":0}"#;
        let result = parse(Some(et("content_block_start")), data);
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());
    }

    #[test]
    fn content_block_delta_text_emits_text_delta() {
        let data = r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hello"}}"#;
        let events = parse(Some(et("content_block_delta")), data).unwrap();
        assert!(matches!(&events[0], Event::TextDelta { text } if text == "hello"));
    }

    #[test]
    fn content_block_delta_missing_delta_key_does_not_panic() {
        // P1: was direct index `raw.rest["delta"]` — would panic
        let data = r#"{"type":"content_block_delta","index":0}"#;
        let result = parse(Some(et("content_block_delta")), data);
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());
    }

    #[test]
    fn content_block_delta_input_json_emits_tool_use_delta() {
        let data = r#"{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"path\":"}}"#;
        let events = parse(Some(et("content_block_delta")), data).unwrap();
        assert!(
            matches!(&events[0], Event::ToolUseDelta { partial_json, .. } if partial_json.contains("path"))
        );
        // id should be __idx_1
        if let Event::ToolUseDelta { id, .. } = &events[0] {
            assert_eq!(id, "__idx_1");
        }
    }

    #[test]
    fn content_block_stop_emits_tool_use_end() {
        let data = r#"{"type":"content_block_stop","index":1}"#;
        let events = parse(Some(et("content_block_stop")), data).unwrap();
        assert!(matches!(&events[0], Event::ToolUseEnd { id } if id == "__idx_1"));
    }

    #[test]
    fn message_delta_with_stop_reason_emits_message_stop() {
        let data = r#"{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"input_tokens":0,"output_tokens":42}}"#;
        let events = parse(Some(et("message_delta")), data).unwrap();
        assert!(matches!(
            &events[0],
            Event::MessageStop {
                stop_reason: StopReason::EndTurn
            }
        ));
        // usage delta also emitted
        assert!(matches!(&events[1], Event::UsageDelta { usage } if usage.output_tokens == 42));
    }

    #[test]
    fn message_delta_missing_delta_key_does_not_panic() {
        // P1: was direct index `raw.rest["delta"]` — would panic
        let data = r#"{"type":"message_delta"}"#;
        let result = parse(Some(et("message_delta")), data);
        assert!(result.is_ok());
    }

    #[test]
    fn message_delta_tool_use_stop_reason() {
        let data = r#"{"type":"message_delta","delta":{"stop_reason":"tool_use"}}"#;
        let events = parse(Some(et("message_delta")), data).unwrap();
        assert!(matches!(
            &events[0],
            Event::MessageStop {
                stop_reason: StopReason::ToolUse
            }
        ));
    }

    #[test]
    fn message_start_emits_usage_delta() {
        let data = r#"{"type":"message_start","message":{"id":"msg_01","type":"message","role":"assistant","content":[],"model":"claude-sonnet-4-5","stop_reason":null,"usage":{"input_tokens":100,"output_tokens":0}}}"#;
        let events = parse(Some(et("message_start")), data).unwrap();
        assert!(matches!(&events[0], Event::UsageDelta { usage } if usage.input_tokens == 100));
    }

    #[test]
    fn bad_json_returns_parse_error() {
        let result = parse(None, "not json {{ garbage");
        assert!(result.is_err());
    }
}

#[cfg(test)]
mod sse_extended {
    use crate::sse::SseParser;

    #[test]
    fn multiline_data_field_accumulates_with_newline_separator() {
        // SSE spec: multiple data: lines are concatenated with \n
        let mut p = SseParser::new();
        p.feed_line("data: {\"a\":");
        p.feed_line("data: 1}");
        let ev = p.feed_line("").unwrap();
        assert_eq!(ev.data, "{\"a\":\n1}");
    }

    #[test]
    fn crlf_line_endings_are_stripped() {
        let mut p = SseParser::new();
        assert!(p.feed_line("data: hello\r").is_none());
        let ev = p.feed_line("\r").unwrap(); // blank line with \r
        assert_eq!(ev.data, "hello");
    }

    #[test]
    fn event_field_with_no_space_after_colon() {
        // "event:ping" vs "event: ping" — both should work after trim_start
        let mut p = SseParser::new();
        p.feed_line("event:ping");
        p.feed_line("data: {}");
        let ev = p.feed_line("").unwrap();
        assert_eq!(ev.event_type.as_deref(), Some("ping"));
    }

    #[test]
    fn event_field_with_extra_spaces() {
        let mut p = SseParser::new();
        p.feed_line("event:  content_block_delta");
        p.feed_line("data: {}");
        let ev = p.feed_line("").unwrap();
        assert_eq!(ev.event_type.as_deref(), Some("content_block_delta"));
    }
}
