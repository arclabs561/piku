use super::HookRegistry;
use crate::{run_turn_with_registry, AllowAll, Session, TaskRegistry};
use piku_api::{ApiError, Event, MessageRequest, Provider, StopReason};
use std::pin::Pin;
use std::sync::Mutex;

struct SequenceProvider {
    sequences: Mutex<Vec<Vec<Event>>>,
    requests: Mutex<Vec<MessageRequest>>,
}

impl SequenceProvider {
    fn new(sequences: Vec<Vec<Event>>) -> Self {
        Self {
            sequences: Mutex::new(sequences),
            requests: Mutex::new(Vec::new()),
        }
    }
}

impl Provider for SequenceProvider {
    fn name(&self) -> &'static str {
        "hook-test"
    }
    fn stream_message(
        &self,
        request: MessageRequest,
    ) -> Pin<Box<dyn futures_util::Stream<Item = Result<Event, ApiError>> + Send + '_>> {
        self.requests.lock().unwrap().push(request);
        let events = self.sequences.lock().unwrap().remove(0);
        Box::pin(futures_util::stream::iter(events.into_iter().map(Ok)))
    }
}

fn tool_call_events(id: &str, name: &str, input: &str) -> Vec<Event> {
    vec![
        Event::ToolUseStart {
            id: id.into(),
            name: name.into(),
        },
        Event::ToolUseDelta {
            id: id.into(),
            partial_json: input.into(),
        },
        Event::ToolUseEnd { id: id.into() },
        Event::MessageStop {
            stop_reason: StopReason::ToolUse,
        },
    ]
}

#[tokio::test]
async fn e2e_hook_allow_context_reaches_model_and_later_deny_wins() {
    for deny in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let hooks_dir = dir.path().join(".piku");
        std::fs::create_dir(&hooks_dir).unwrap();
        let mut commands = Vec::new();
        for text in ["first context", "second context"] {
            let output = serde_json::json!({"hookSpecificOutput": {
                "hookEventName": "PreToolUse", "permissionDecision": "allow",
                "additionalContext": text
            }});
            commands.push(serde_json::json!({"command": format!("echo '{output}'")}));
        }
        if deny {
            commands.push(serde_json::json!({"command": "echo refused >&2; exit 2"}));
        }
        std::fs::write(
            hooks_dir.join("hooks.json"),
            serde_json::json!({
                "PreToolUse": [{"matcher": "list_dir", "hooks": commands}]
            })
            .to_string(),
        )
        .unwrap();
        let hooks = HookRegistry::load_project_only(dir.path());
        let provider = SequenceProvider::new(vec![
            tool_call_events(
                "hook-call",
                "list_dir",
                &serde_json::json!({"path": dir.path()}).to_string(),
            ),
            vec![Event::MessageStop {
                stop_reason: StopReason::EndTurn,
            }],
        ]);
        let mut session = Session::new("hook-context".to_string());
        let mut sink = crate::task::DevNullSink;
        run_turn_with_registry(
            "list files",
            &mut session,
            &provider,
            "test-model",
            &[],
            piku_tools::all_tool_definitions(),
            &AllowAll,
            &mut sink,
            Some(3),
            None,
            &TaskRegistry::new(),
            0,
            &[],
            Some(&hooks),
            None,
        )
        .await;
        let requests = provider.requests.lock().unwrap();
        assert_eq!(requests.len(), 2);
        let model_input = serde_json::to_string(&requests[1].messages).unwrap();
        if deny {
            assert!(model_input.contains("Blocked by hook: refused"));
            assert!(!model_input.contains("<hook-context>"));
        } else {
            assert!(
                model_input.contains("first context\\nsecond context")
                    || model_input.contains("first context\nsecond context"),
                "{model_input}"
            );
            assert!(model_input.contains("<hook-context>"));
        }
    }
}

#[cfg(unix)]
#[test]
fn hook_timeout_kills_descendants_that_hold_pipes() {
    let dir = tempfile::tempdir().unwrap();
    let result = super::run_hook_command_raw(
        r#"sh -c 'sleep 2; echo survived > "$PIKU_PROJECT_DIR/survived"' & echo $! > "$PIKU_PROJECT_DIR/descendant"; exit 0"#,
        "{}",
        1,
        Some(dir.path()),
    );
    assert!(matches!(result, super::HookCommandResult::Error(ref e) if e.contains("timed out")));
    let pid = std::fs::read_to_string(dir.path().join("descendant")).unwrap();
    // A killed orphan may briefly be a zombie while its new parent reaps it.
    let state = std::process::Command::new("ps")
        .args(["-p", pid.trim(), "-o", "stat="])
        .output()
        .unwrap();
    let state = String::from_utf8_lossy(&state.stdout);
    assert!(
        state.trim().is_empty() || state.trim().starts_with('Z'),
        "descendant still running: {state}"
    );
    std::thread::sleep(std::time::Duration::from_millis(1200));
    assert!(!dir.path().join("survived").exists());
}

#[test]
fn hook_output_overflow_fails_explicitly_for_each_stream() {
    for (stream, redirect) in [("stdout", ""), ("stderr", ">&2")] {
        let result = super::run_hook_command_raw(
            &format!("dd if=/dev/zero bs=65536 count=5 {redirect} 2>/dev/null"),
            "{}",
            3,
            None,
        );
        assert!(
            matches!(result, super::HookCommandResult::Error(ref e)
                if e.contains(stream) && e.contains("exceeded 262144-byte limit")),
            "{stream} overflow must fail rather than return partial JSON",
        );
    }
}

#[test]
fn hook_small_json_output_is_preserved() {
    let json = r#"{"hookSpecificOutput":{"permissionDecision":"deny","permissionDecisionReason":"bounded output ✓"}}"#;
    let result = super::run_hook_command_raw(&format!("printf '%s' '{json}'"), "{}", 3, None);
    assert!(matches!(result, super::HookCommandResult::Success(ref output) if output == json));
}
