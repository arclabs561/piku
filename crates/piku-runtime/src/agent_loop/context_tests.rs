use super::*;
use std::collections::BTreeSet;

fn calls(ids: &[&str], padding: usize) -> ConversationMessage {
    let mut blocks = vec![ContentBlock::Text {
        text: "x".repeat(padding),
    }];
    blocks.extend(ids.iter().map(|id| ContentBlock::ToolUse {
        id: (*id).into(),
        name: "read_file".into(),
        input: serde_json::json!({"path":"fixture"}),
    }));
    ConversationMessage::assistant(blocks, None)
}

fn tail(messages: &mut Vec<ConversationMessage>, count: usize) {
    messages.extend((0..count).map(|_| ConversationMessage::user("current requirement")));
}

fn request(
    messages: Vec<ConversationMessage>,
) -> (MessageRequest, crate::run_record::ContextManifest) {
    let mut session = Session::new("curation-regression".into());
    session.messages = messages;
    build_request(&session, "gpt-4o", &[], &[])
}

// Check the actual provider request, after conversion and role coalescing.
// Every call must have exactly one result in the immediately following user
// message; no user result can exist without that preceding assistant call.
fn assert_complete_exchanges(request: &MessageRequest) -> usize {
    let mut pending = BTreeSet::new();
    let mut calls = 0;
    for message in &request.messages {
        let mut results = BTreeSet::new();
        for block in &message.content {
            match block {
                RequestContent::ToolUse { id, .. } => {
                    assert_eq!(message.role, "assistant");
                    assert!(pending.insert(id.clone()), "duplicate call {id}");
                    calls += 1;
                }
                RequestContent::ToolResult { tool_use_id, .. } => {
                    assert_eq!(message.role, "user");
                    assert!(results.insert(tool_use_id.clone()), "duplicate result");
                }
                _ => {}
            }
        }
        if message.role == "user" {
            assert_eq!(
                pending, results,
                "provider request has unmatched tool exchange"
            );
            pending.clear();
        }
    }
    assert!(pending.is_empty(), "request ends with unresolved calls");
    calls
}

#[test]
fn curation_does_not_select_cheap_result_without_expensive_call() {
    let mut result = ConversationMessage::tool_result("a".into(), "found".into(), false);
    result.importance = Some(1.0);
    let mut messages = vec![calls(&["a"], 1_000_000), result];
    tail(&mut messages, 6);
    let (request, _) = request(messages);
    assert_eq!(assert_complete_exchanges(&request), 0);
    assert!(request.messages.iter().any(|m| m.content.iter().any(
        |b| matches!(b, RequestContent::Text { text } if text.contains("current requirement"))
    )));
}

#[test]
fn curation_keeps_all_results_of_a_multi_call_or_none() {
    let mut call = calls(&["a", "b"], 0);
    call.importance = Some(1.0);
    let mut messages = vec![
        call,
        ConversationMessage::tool_result("a".into(), "small".into(), false),
        ConversationMessage::tool_result("b".into(), "x".repeat(1_000_000), false),
    ];
    tail(&mut messages, 6);
    let (request, _) = request(messages);
    assert_eq!(assert_complete_exchanges(&request), 0);
}

#[test]
fn mandatory_tail_expands_to_complete_exchange_and_reports_over_budget() {
    for trailing in 0..6 {
        let ids = ["a", "b", "c", "d", "e", "f", "g", "h"];
        let mut messages = vec![calls(&ids, 1_000_000)];
        messages.extend(
            ids.iter()
                .map(|id| ConversationMessage::tool_result((*id).into(), "found".into(), false)),
        );
        tail(&mut messages, trailing);
        let (request, manifest) = request(messages);
        assert_eq!(assert_complete_exchanges(&request), ids.len());
        assert!(manifest.estimated_input_tokens > manifest.context_window_tokens);
        assert!(manifest.messages.iter().all(|m| m.selected));
        assert!(manifest
            .messages
            .iter()
            .any(|m| m.reason == "mandatory_context_over_budget"));
    }
}

#[test]
fn important_user_role_result_keeps_its_complete_exchange() {
    let mut result = ConversationMessage::user("context accompanying results");
    result
        .blocks
        .extend(["a", "b"].iter().map(|id| ContentBlock::ToolResult {
            tool_use_id: (*id).into(),
            output: "found".into(),
            is_error: false,
        }));
    result.importance = Some(1.0);
    let mut messages = vec![
        calls(&["a", "b"], 20_000),
        result,
        ConversationMessage::user("x".repeat(1_000_000)),
    ];
    tail(&mut messages, 6);
    let (request, manifest) = request(messages);
    assert_eq!(assert_complete_exchanges(&request), 2);
    assert!(
        !manifest.messages[2].selected,
        "oversized unrelated history must be excluded"
    );
    assert!(manifest.estimated_input_tokens < manifest.context_window_tokens);
}

#[test]
fn mandatory_exchange_is_charged_once_before_ranking_other_history() {
    let mut high = ConversationMessage::user("important earlier requirement");
    high.importance = Some(1.0);
    let mut messages = vec![
        high,
        ConversationMessage::user("x".repeat(1_000_000)),
        calls(&["a", "b"], 20_000),
        ConversationMessage::tool_result("a".into(), "found".into(), false),
        ConversationMessage::tool_result("b".into(), "found".into(), false),
    ];
    tail(&mut messages, 4);
    let (request, manifest) = request(messages);
    assert_eq!(assert_complete_exchanges(&request), 2);
    assert!(manifest.messages[0].selected);
    assert!(!manifest.messages[1].selected);
    assert_eq!(manifest.messages[2].reason, "recent_exchange");
}

#[test]
fn compaction_preserves_large_exchange_in_provider_request() {
    let ids: Vec<String> = (0..13).map(|index| format!("call-{index}")).collect();
    for prefix in [false, true] {
        let mut session = Session::new("compaction-regression".into());
        if prefix {
            session
                .messages
                .push(ConversationMessage::user("old requirement"));
        }
        session.messages.push(calls(
            &ids.iter().map(String::as_str).collect::<Vec<_>>(),
            1000,
        ));
        session.messages.extend(
            ids.iter()
                .map(|id| ConversationMessage::tool_result(id.clone(), "result".into(), false)),
        );
        let config = crate::compact::CompactionConfig {
            preserve_recent_messages: 12,
            max_estimated_tokens: 1,
        };
        for compacted in [
            crate::compact::compact_session(&session, config),
            crate::compact::apply_compact_summary(&session, "old summary", config),
        ] {
            let (request, _) = build_request(&compacted.compacted_session, "gpt-4o", &[], &[]);
            assert_eq!(assert_complete_exchanges(&request), 13);
            assert_eq!(compacted.removed_message_count, usize::from(prefix));
            if !prefix {
                assert!(compacted.summary.is_empty());
                assert_eq!(
                    compacted.compacted_session.messages.len(),
                    session.messages.len()
                );
            }
        }
    }
}
