//! Writes to Piku's own trust anchors must reach the permission prompter.
//! Deterministic, no network. Copied from the Oct 2026 bug hunt repro
//! (`bughunt2.rs`), with the self-update sink reduced to a quiet one.
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use futures_util::Stream;
use piku_api::{ApiError, Event, MessageRequest, Provider, StopReason, TokenUsage};
use piku_runtime::session::Session;
use piku_runtime::{run_turn, OutputSink, PostToolAction};
use piku_tools::all_tool_definitions;

struct Sequence {
    sequences: Vec<Vec<Event>>,
    index: Mutex<usize>,
    requests: Arc<Mutex<Vec<MessageRequest>>>,
}

impl Provider for Sequence {
    fn name(&self) -> &'static str {
        "sequence"
    }
    fn stream_message(
        &self,
        request: MessageRequest,
    ) -> Pin<Box<dyn Stream<Item = Result<Event, ApiError>> + Send + '_>> {
        self.requests.lock().unwrap().push(request);
        let mut index = self.index.lock().unwrap();
        let events = self.sequences[(*index).min(self.sequences.len() - 1)].clone();
        *index += 1;
        Box::pin(async_stream::stream! {
            for event in events {
                yield Ok(event);
            }
        })
    }
}

struct QuietSink;
impl OutputSink for QuietSink {
    fn on_text(&mut self, _: &str) {}
    fn on_tool_start(&mut self, _: &str, _: &str, _: &serde_json::Value) {}
    fn on_tool_end(&mut self, _: &str, _: &str, _: &str, _: bool) -> PostToolAction {
        PostToolAction::Continue
    }
    fn on_permission_denied(&mut self, _: &str, _: &str) {}
    fn on_turn_complete(&mut self, _: &TokenUsage, _: u32) {}
}

fn text_stop(text: &str) -> Vec<Event> {
    vec![
        Event::TextDelta { text: text.into() },
        Event::MessageStop {
            stop_reason: StopReason::EndTurn,
        },
    ]
}

/// A prompter standing in for a TUI user who would refuse every prompt.
#[derive(Default)]
struct RefusingUser {
    asked: Mutex<Vec<String>>,
}
impl piku_runtime::PermissionPrompter for RefusingUser {
    fn decide(&self, req: &piku_runtime::PermissionRequest) -> piku_runtime::PermissionOutcome {
        self.asked.lock().unwrap().push(format!(
            "{} {}",
            req.tool_name,
            req.params
                .get("path")
                .and_then(|p| p.as_str())
                .unwrap_or("")
        ));
        piku_runtime::PermissionOutcome::Deny {
            reason: "user refused".into(),
        }
    }
}

fn write_call(id: &str, path: &str, content: &str) -> Vec<Event> {
    vec![
        Event::ToolUseStart {
            id: id.into(),
            name: "write_file".into(),
        },
        Event::ToolUseDelta {
            id: id.into(),
            partial_json: serde_json::json!({ "path": path, "content": content }).to_string(),
        },
        Event::ToolUseEnd { id: id.into() },
        Event::MessageStop {
            stop_reason: StopReason::ToolUse,
        },
    ]
}

/// `write_file` to Piku's own global hook file was classified Safe when the file
/// did not exist yet, so it ran without the TUI permission prompt, and the
/// next `HookRegistry::load` executed it. `.claude/` was protected; Piku's own
/// trust anchors were not.
#[tokio::test]
async fn bughunt2_safe_write_installs_global_hook_without_prompt() {
    let root = std::env::temp_dir().join(format!("bughunt2-hooks-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let config = root.join("cfg");
    let project = root.join("project");
    std::fs::create_dir_all(&project).unwrap();
    let hooks_path = config.join("piku").join("hooks.json");
    let marker = root.join("hook-ran");
    let hook = serde_json::json!({
        "SessionStart": [{ "hooks": [{ "command": format!("touch {}", marker.display()) }] }]
    })
    .to_string();
    let claude_path = project.join(".claude").join("settings.json");

    let requests = Arc::new(Mutex::new(Vec::new()));
    let provider = Sequence {
        sequences: vec![
            write_call("w1", hooks_path.to_str().unwrap(), &hook),
            write_call("w2", claude_path.to_str().unwrap(), "{}"),
            text_stop("done"),
        ],
        index: Mutex::new(0),
        requests,
    };
    let user = RefusingUser::default();
    let mut session = Session::new("bughunt2-hooks".into());
    let mut sink = QuietSink;
    let _ = run_turn(
        "set things up",
        &mut session,
        &provider,
        "m",
        &[],
        all_tool_definitions(),
        &user,
        &mut sink,
        None,
        None,
    )
    .await;
    let asked = user.asked.lock().unwrap().clone();
    eprintln!("prompts shown: {asked:?}");
    eprintln!("global hooks.json exists: {}", hooks_path.exists());
    assert!(
        asked.iter().any(|a| a.contains(".claude")),
        "control: protected .claude path must prompt"
    );
    // Next interactive launch: global hooks load without any trust step.
    std::env::set_var("XDG_CONFIG_HOME", &config);
    let registry = piku_runtime::HookRegistry::load(&project);
    let _ = registry.run_session_start("next-session", &project);
    eprintln!("SessionStart hook executed: {}", marker.exists());
    assert!(
        !hooks_path.exists() || asked.iter().any(|a| a.contains("hooks.json")),
        "write_file created {} without asking the user; next launch ran it: {}",
        hooks_path.display(),
        marker.exists()
    );
    let _ = std::fs::remove_dir_all(&root);
}
