/// End-to-end integration tests for the piku agentic runtime.
///
/// These tests use:
/// - Real filesystem operations (tempdir per test)
/// - Mock providers with scripted event sequences
/// - Real tool implementations
/// - Real permission logic
/// - Real session serialisation/deserialisation
///
/// They simulate the full path a user prompt takes through piku:
///   prompt → stream → tool calls → execute → session persist → result
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use futures_util::Stream;
use piku_api::{ApiError, Event, MessageRequest, Provider, StopReason, TokenUsage};
use piku_runtime::{build_system_prompt, run_turn, AllowAll, OutputSink, PostToolAction, Session};
use piku_tools::all_tool_definitions;

// ---------------------------------------------------------------------------
// Test infrastructure
// ---------------------------------------------------------------------------

/// Scripted provider — replays a fixed list of events on every call.
struct ScriptedProvider {
    events: Vec<Event>,
}

impl ScriptedProvider {
    fn new(events: Vec<Event>) -> Self {
        Self { events }
    }
}

impl Provider for ScriptedProvider {
    fn name(&self) -> &'static str {
        "scripted"
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

/// Stateful provider — returns different event sequences per call (round-robin).
struct SequenceProvider {
    sequences: Arc<Mutex<Vec<Vec<Event>>>>,
    call_index: Arc<Mutex<usize>>,
}

impl SequenceProvider {
    fn new(sequences: Vec<Vec<Event>>) -> Self {
        Self {
            sequences: Arc::new(Mutex::new(sequences)),
            call_index: Arc::new(Mutex::new(0)),
        }
    }
}

impl Provider for SequenceProvider {
    fn name(&self) -> &'static str {
        "sequence"
    }

    fn stream_message(
        &self,
        _request: MessageRequest,
    ) -> Pin<Box<dyn Stream<Item = Result<Event, ApiError>> + Send + '_>> {
        let seqs = self.sequences.lock().unwrap();
        let mut idx = self.call_index.lock().unwrap();
        let events = seqs[(*idx).min(seqs.len() - 1)].clone();
        *idx += 1;
        drop(idx);
        drop(seqs);
        Box::pin(async_stream::stream! {
            for event in events {
                yield Ok(event);
            }
        })
    }
}

/// Collecting sink — records all events for assertions.
#[derive(Default)]
struct CollectSink {
    text: String,
    tool_starts: Vec<(String, String)>,     // (name, id)
    tool_ends: Vec<(String, String, bool)>, // (name, output, is_error)
    denied: Vec<(String, String)>,          // (name, reason)
    turn_complete: Option<(TokenUsage, u32)>,
    /// If set, return `ReplaceAndExec` for the next `tool_end` that matches
    trigger_replace_for_tool: Option<String>,
    replace_triggered: Option<PathBuf>,
}

impl OutputSink for CollectSink {
    fn on_text(&mut self, text: &str) {
        self.text.push_str(text);
    }

    fn on_tool_start(&mut self, name: &str, id: &str, _input: &serde_json::Value) {
        self.tool_starts.push((name.to_string(), id.to_string()));
    }

    fn on_tool_end(&mut self, name: &str, result: &str, is_error: bool) -> PostToolAction {
        self.tool_ends
            .push((name.to_string(), result.to_string(), is_error));

        if let Some(trigger) = &self.trigger_replace_for_tool {
            if name == trigger {
                let path = PathBuf::from("/tmp/fake-new-piku");
                self.replace_triggered = Some(path.clone());
                return PostToolAction::ReplaceAndExec(path);
            }
        }

        PostToolAction::Continue
    }

    fn on_permission_denied(&mut self, name: &str, reason: &str) {
        self.denied.push((name.to_string(), reason.to_string()));
    }

    fn on_turn_complete(&mut self, usage: &TokenUsage, iterations: u32) {
        self.turn_complete = Some((usage.clone(), iterations));
    }
}

fn tempdir() -> PathBuf {
    let base = std::env::temp_dir().join(format!(
        "piku_e2e_{}_{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.subsec_nanos())
            .unwrap_or(0),
        std::process::id(),
    ));
    std::fs::create_dir_all(&base).unwrap();
    base
}

fn text_stop(text: &str) -> Vec<Event> {
    vec![
        Event::TextDelta {
            text: text.to_string(),
        },
        Event::MessageStop {
            stop_reason: StopReason::EndTurn,
        },
        Event::UsageDelta {
            usage: TokenUsage {
                input_tokens: 50,
                output_tokens: 20,
                ..Default::default()
            },
        },
    ]
}

fn tool_call_events(id: &str, name: &str, input_json: &str) -> Vec<Event> {
    vec![
        Event::ToolUseStart {
            id: id.to_string(),
            name: name.to_string(),
        },
        Event::ToolUseDelta {
            id: id.to_string(),
            partial_json: input_json.to_string(),
        },
        Event::ToolUseEnd { id: id.to_string() },
        Event::MessageStop {
            stop_reason: StopReason::ToolUse,
        },
        Event::UsageDelta {
            usage: TokenUsage {
                input_tokens: 100,
                output_tokens: 30,
                ..Default::default()
            },
        },
    ]
}

// ---------------------------------------------------------------------------
// E2E scenario 1: Simple text response — no tools
// ---------------------------------------------------------------------------

#[tokio::test]
async fn e2e_simple_text_response() {
    let provider = ScriptedProvider::new(text_stop("The answer is 42."));
    let mut session = Session::new("e2e-1".to_string());
    let mut sink = CollectSink::default();

    let result = run_turn(
        "what is the answer to everything?",
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

    assert_eq!(sink.text, "The answer is 42.");
    assert!(sink.tool_starts.is_empty());
    assert_eq!(result.iterations, 1);
    assert!(result.stream_error.is_none());
    assert!(result.replace_and_exec.is_none());

    // session should have user + assistant
    assert_eq!(session.messages.len(), 2);
    let (usage, iters) = sink.turn_complete.unwrap();
    assert_eq!(iters, 1);
    assert_eq!(usage.input_tokens, 50);
}

// ---------------------------------------------------------------------------
// E2E scenario 2: Read a real file via tool call
// ---------------------------------------------------------------------------

#[tokio::test]
async fn e2e_reads_file_via_tool() {
    let dir = tempdir();
    let path = dir.join("project.md");
    std::fs::write(&path, "# My Project\n\nThis is a Rust project.").unwrap();

    let input = serde_json::json!({ "path": path }).to_string();

    let provider = SequenceProvider::new(vec![
        // Turn 1: call read_file
        tool_call_events("t1", "read_file", &input),
        // Turn 2: return text after seeing file contents
        text_stop("The file contains a Rust project description."),
    ]);

    let mut session = Session::new("e2e-2".to_string());
    let mut sink = CollectSink::default();

    let result = run_turn(
        "summarize the project.md file",
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

    // Tool was called
    assert_eq!(sink.tool_starts.len(), 1);
    assert_eq!(sink.tool_starts[0].0, "read_file");

    // Tool succeeded
    assert_eq!(sink.tool_ends.len(), 1);
    assert!(!sink.tool_ends[0].2, "read_file should not error");
    assert!(sink.tool_ends[0].1.contains("Rust project"));

    // Final text was produced
    assert!(sink.text.contains("Rust project description"));

    // Two iterations: one for tool call, one for final response
    assert_eq!(result.iterations, 2);

    // Session: user + assistant(tool_use) + tool_result(user) + assistant(text)
    assert_eq!(session.messages.len(), 4);
}

// ---------------------------------------------------------------------------
// E2E scenario 3: Write then read — multi-tool multi-turn
// ---------------------------------------------------------------------------

#[tokio::test]
async fn e2e_write_then_read_file() {
    let dir = tempdir();
    let path = dir.join("output.txt");

    let write_input = serde_json::json!({ "path": path, "content": "hello from piku" }).to_string();
    let read_input = serde_json::json!({ "path": path }).to_string();

    let provider = SequenceProvider::new(vec![
        // Turn 1: write the file
        tool_call_events("w1", "write_file", &write_input),
        // Turn 2: read it back
        tool_call_events("r1", "read_file", &read_input),
        // Turn 3: confirm
        text_stop("Done. I wrote and then read back the file successfully."),
    ]);

    let mut session = Session::new("e2e-3".to_string());
    let mut sink = CollectSink::default();

    run_turn(
        "write 'hello from piku' to output.txt then read it back",
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
    assert_eq!(sink.tool_starts[0].0, "write_file");
    assert_eq!(sink.tool_starts[1].0, "read_file");

    // write succeeded
    assert!(!sink.tool_ends[0].2);
    // read succeeded and returned the content
    assert!(!sink.tool_ends[1].2);
    assert!(sink.tool_ends[1].1.contains("hello from piku"));

    // file actually exists on disk
    assert_eq!(std::fs::read_to_string(&path).unwrap(), "hello from piku");

    assert!(sink.text.contains("Done"));
}

// ---------------------------------------------------------------------------
// E2E scenario 4: Edit file with surgical replacement
// ---------------------------------------------------------------------------

#[tokio::test]
async fn e2e_edit_file_surgical() {
    let dir = tempdir();
    let path = dir.join("src.rs");
    std::fs::write(&path, "fn main() {\n    println!(\"old\");\n}").unwrap();

    let edit_input = serde_json::json!({
        "path": path,
        "old_string": "println!(\"old\");",
        "new_string": "println!(\"new\");",
    })
    .to_string();

    let provider = SequenceProvider::new(vec![
        tool_call_events("e1", "edit_file", &edit_input),
        text_stop("Updated the println to say 'new'."),
    ]);

    let mut session = Session::new("e2e-4".to_string());
    let mut sink = CollectSink::default();

    run_turn(
        "change the println to say 'new'",
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

    assert!(!sink.tool_ends[0].2, "edit should succeed");
    let content = std::fs::read_to_string(&path).unwrap();
    assert!(content.contains("new"), "file should have been edited");
    assert!(!content.contains("old"), "old string should be gone");
}

// ---------------------------------------------------------------------------
// E2E scenario 5: Permission denial stops the tool, loop continues
// ---------------------------------------------------------------------------

#[tokio::test]
async fn e2e_permission_denied_stops_tool_not_loop() {
    use piku_runtime::permission::{PermissionOutcome, PermissionPrompter, PermissionRequest};

    struct DenyBash;
    impl PermissionPrompter for DenyBash {
        fn decide(&self, req: &PermissionRequest) -> PermissionOutcome {
            if req.tool_name == "bash" {
                PermissionOutcome::Deny {
                    reason: "bash not allowed in test".to_string(),
                }
            } else {
                PermissionOutcome::Allow
            }
        }
    }

    let dir = tempdir();
    let f = dir.join("safe.txt");
    std::fs::write(&f, "contents").unwrap();
    let read_input = serde_json::json!({ "path": f }).to_string();

    let provider = SequenceProvider::new(vec![
        // Turn 1: bash call (will be denied) + read_file call (will succeed)
        {
            let mut events = tool_call_events("b1", "bash", r#"{"command":"rm -rf /"}"#);
            // splice in a second tool call before MessageStop
            let stop_idx = events
                .iter()
                .position(|e| matches!(e, Event::MessageStop { .. }))
                .unwrap();
            events.splice(
                stop_idx..stop_idx,
                vec![
                    Event::ToolUseStart {
                        id: "r1".to_string(),
                        name: "read_file".to_string(),
                    },
                    Event::ToolUseDelta {
                        id: "r1".to_string(),
                        partial_json: read_input,
                    },
                    Event::ToolUseEnd {
                        id: "r1".to_string(),
                    },
                ],
            );
            events
        },
        // Turn 2: respond after seeing the tool results
        text_stop("bash was denied, but read_file succeeded."),
    ]);

    let mut session = Session::new("e2e-5".to_string());
    let mut sink = CollectSink::default();

    run_turn(
        "try bash then read a file",
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

    // bash was denied
    assert_eq!(sink.denied.len(), 1);
    assert_eq!(sink.denied[0].0, "bash");

    // read_file still ran
    assert!(sink.tool_starts.iter().any(|(n, _)| n == "read_file"));
    assert!(sink
        .tool_ends
        .iter()
        .any(|(n, _, e)| n == "read_file" && !e));

    // final response produced
    assert!(sink.text.contains("bash was denied"));
}

// ---------------------------------------------------------------------------
// E2E scenario 6: Session saves and loads correctly across turns
// ---------------------------------------------------------------------------

#[tokio::test]
async fn e2e_session_persists_and_resumes() {
    let dir = tempdir();
    let session_path = dir.join("session.json");
    let content_file = dir.join("data.txt");
    std::fs::write(&content_file, "persistent content").unwrap();

    let read_input = serde_json::json!({ "path": content_file }).to_string();

    // First turn: read file, get answer
    {
        let provider = SequenceProvider::new(vec![
            tool_call_events("t1", "read_file", &read_input),
            text_stop("I read the file. It says: persistent content."),
        ]);

        let mut session = Session::new("resume-test".to_string());
        let mut sink = CollectSink::default();

        run_turn(
            "read data.txt",
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

        session.save(&session_path).unwrap();
        assert_eq!(session.messages.len(), 4); // user + assistant + tool_result + assistant
    }

    // Second turn: load session, ask follow-up
    {
        let mut session = Session::load(&session_path).unwrap();
        let prior_message_count = session.messages.len();

        let provider =
            ScriptedProvider::new(text_stop("Yes, I remember — it said 'persistent content'."));
        let mut sink = CollectSink::default();

        run_turn(
            "what did the file say?",
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

        // Session grew by user + assistant
        assert_eq!(session.messages.len(), prior_message_count + 2);
        assert!(sink.text.contains("persistent content"));
    }
}

// ---------------------------------------------------------------------------
// E2E scenario 7: Glob + Grep realistic codebase search
// ---------------------------------------------------------------------------

#[tokio::test]
async fn e2e_glob_and_grep_codebase() {
    let dir = tempdir();
    // Create a fake Rust codebase
    std::fs::create_dir_all(dir.join("src")).unwrap();
    std::fs::write(
        dir.join("src").join("main.rs"),
        "fn main() {\n    run();\n}",
    )
    .unwrap();
    std::fs::write(
        dir.join("src").join("lib.rs"),
        "pub fn run() {\n    println!(\"running\");\n}",
    )
    .unwrap();
    std::fs::write(
        dir.join("src").join("utils.rs"),
        "fn helper() {}\n// TODO: implement",
    )
    .unwrap();

    let glob_input = serde_json::json!({ "pattern": "**/*.rs", "path": dir }).to_string();
    let grep_input = serde_json::json!({ "pattern": "fn ", "path": dir }).to_string();

    let provider = SequenceProvider::new(vec![
        tool_call_events("g1", "glob", &glob_input),
        tool_call_events("s1", "grep", &grep_input),
        text_stop("Found 3 Rust files. There are 3 function definitions."),
    ]);

    let mut session = Session::new("e2e-7".to_string());
    let mut sink = CollectSink::default();

    run_turn(
        "find all rust files and count functions",
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

    // glob found files
    let glob_result = &sink.tool_ends[0];
    assert_eq!(glob_result.0, "glob");
    assert!(!glob_result.2);
    assert!(glob_result.1.contains(".rs"));

    // grep found functions
    let grep_result = &sink.tool_ends[1];
    assert_eq!(grep_result.0, "grep");
    assert!(!grep_result.2);
    assert!(grep_result.1.contains("fn "));
}

// ---------------------------------------------------------------------------
// E2E scenario 8: Bash tool executes and captures output
// ---------------------------------------------------------------------------

#[tokio::test]
async fn e2e_bash_runs_real_command() {
    let dir = tempdir();

    let bash_input = serde_json::json!({
        "command": format!("echo 'hello from bash' > {}/output.txt && cat {}/output.txt", dir.display(), dir.display()),
    })
    .to_string();

    let provider = SequenceProvider::new(vec![
        tool_call_events("b1", "bash", &bash_input),
        text_stop("Command executed successfully."),
    ]);

    let mut session = Session::new("e2e-8".to_string());
    let mut sink = CollectSink::default();

    run_turn(
        "run the echo command",
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

    assert_eq!(sink.tool_ends[0].0, "bash");
    assert!(!sink.tool_ends[0].2, "bash should succeed");
    assert!(sink.tool_ends[0].1.contains("hello from bash"));
    assert!(dir.join("output.txt").exists());
}

// ---------------------------------------------------------------------------
// E2E scenario 9: max_turns halts an infinite tool loop
// ---------------------------------------------------------------------------

#[tokio::test]
async fn e2e_max_turns_prevents_infinite_loop() {
    // Provider always returns a tool call — simulates a model that loops
    let list_input = serde_json::json!({}).to_string();
    let provider = ScriptedProvider::new(tool_call_events("c1", "list_dir", &list_input));

    let mut session = Session::new("e2e-9".to_string());
    let mut sink = CollectSink::default();

    let result = run_turn(
        "list forever",
        &mut session,
        &provider,
        "m",
        &[],
        all_tool_definitions(),
        &AllowAll,
        &mut sink,
        Some(4),
        None,
    )
    .await;

    assert_eq!(result.iterations, 4);
    // At least 4 tool calls were made
    assert!(sink.tool_starts.len() >= 4);
}

// ---------------------------------------------------------------------------
// E2E scenario 10: Self-update signal propagates from sink to TurnResult
// ---------------------------------------------------------------------------

#[tokio::test]
async fn e2e_self_update_signal_breaks_loop() {
    let bash_input = serde_json::json!({ "command": "cargo build --release" }).to_string();

    let provider = SequenceProvider::new(vec![
        tool_call_events("b1", "bash", &bash_input),
        // This would be reached if the loop didn't break on ReplaceAndExec
        text_stop("Should not reach this."),
    ]);

    let mut session = Session::new("e2e-10".to_string());
    let mut sink = CollectSink {
        trigger_replace_for_tool: Some("bash".to_string()),
        ..Default::default()
    };

    let result = run_turn(
        "build piku",
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

    // Loop broke early — ReplaceAndExec is set
    assert!(result.replace_and_exec.is_some());
    assert_eq!(
        result.replace_and_exec.unwrap(),
        PathBuf::from("/tmp/fake-new-piku")
    );

    // The text "Should not reach this" was never streamed
    assert!(
        !sink.text.contains("Should not reach"),
        "loop should have broken before second turn"
    );

    // Session was persisted up to the point of interruption (bash result is in session)
    let has_tool_result = session.messages.iter().any(|m| {
        m.blocks
            .iter()
            .any(|b| matches!(b, piku_runtime::ContentBlock::ToolResult { .. }))
    });
    assert!(
        has_tool_result,
        "tool result should be in session before restart"
    );
}

// ---------------------------------------------------------------------------
// E2E scenario 11: Stream error — session not corrupted
// ---------------------------------------------------------------------------

#[tokio::test]
async fn e2e_stream_error_session_not_corrupted() {
    struct ErrorMidStream;
    impl Provider for ErrorMidStream {
        fn name(&self) -> &'static str {
            "error-mid"
        }
        fn stream_message(
            &self,
            _: MessageRequest,
        ) -> Pin<Box<dyn Stream<Item = Result<Event, ApiError>> + Send + '_>> {
            Box::pin(async_stream::stream! {
                yield Ok(Event::TextDelta { text: "partial text...".to_string() });
                yield Err(ApiError::UnexpectedStreamEnd);
            })
        }
    }

    let provider = ErrorMidStream;
    let mut session = Session::new("e2e-11".to_string());
    let mut sink = CollectSink::default();

    let result = run_turn(
        "this will error mid-stream",
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

    assert!(result.stream_error.is_some());
    // Only the user message — no corrupted assistant message
    assert_eq!(
        session.messages.len(),
        1,
        "session should only have user message"
    );
}

// ---------------------------------------------------------------------------
// E2E scenario 12: System prompt is included in API request
// ---------------------------------------------------------------------------

#[tokio::test]
async fn e2e_system_prompt_reaches_provider() {
    use piku_api::SystemBlock;

    struct SystemCapture(Arc<Mutex<Option<Vec<SystemBlock>>>>);
    impl Provider for SystemCapture {
        fn name(&self) -> &'static str {
            "capture"
        }
        fn stream_message(
            &self,
            req: MessageRequest,
        ) -> Pin<Box<dyn Stream<Item = Result<Event, ApiError>> + Send + '_>> {
            *self.0.lock().unwrap() = req.system.clone();
            Box::pin(async_stream::stream! {
                yield Ok(Event::TextDelta { text: "response".to_string() });
                yield Ok(Event::MessageStop { stop_reason: StopReason::EndTurn });
            })
        }
    }

    let captured_system: Arc<Mutex<Option<Vec<SystemBlock>>>> = Arc::new(Mutex::new(None));
    let provider = SystemCapture(captured_system.clone());
    let dir = tempdir();
    let system = build_system_prompt(&dir, "2026-04-03", "test-model", &[]);

    let mut session = Session::new("e2e-12".to_string());
    let mut sink = CollectSink::default();

    run_turn(
        "hello",
        &mut session,
        &provider,
        "test-model",
        &system,
        vec![],
        &AllowAll,
        &mut sink,
        None,
        None,
    )
    .await;

    // Flatten all system blocks into a single string for assertion
    let sys_blocks = captured_system.lock().unwrap().clone().unwrap_or_default();
    let sys: String = sys_blocks
        .iter()
        .map(|b| b.text.as_str())
        .collect::<Vec<_>>()
        .join("\n\n");
    // The static block should carry cache_control
    assert!(
        sys_blocks.iter().any(|b| b.cache_control.is_some()),
        "at least one system block must have cache_control"
    );
    assert!(
        sys.contains("test-model"),
        "system prompt must contain model name"
    );
}

// ===========================================================================
// Bug regression tests
// ===========================================================================

// ---------------------------------------------------------------------------
// Bug F: OAI multi-tool delta naming mismatch (__tc_N vs __idx_N)
// Two simultaneous tool calls — second should NOT get empty params.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn regression_bug_f_multi_tool_oai_delta_naming_second_tool_gets_params() {
    let dir = tempdir();
    let f1 = dir.join("file1.txt");
    let f2 = dir.join("file2.txt");
    std::fs::write(&f1, "content of file 1").unwrap();
    std::fs::write(&f2, "content of file 2").unwrap();

    let i1 = serde_json::json!({"path": f1}).to_string();
    let i2 = serde_json::json!({"path": f2}).to_string();

    // Simulate OAI-compat streaming: two tool calls with index-based __tc_N ids
    // First tool call uses __tc_0 for its deltas
    // Second tool call uses __tc_1 for its deltas
    let provider = ScriptedProvider::new(vec![
        // Tool 1: start with real id, deltas use __tc_0
        Event::ToolUseStart {
            id: "call_abc".to_string(),
            name: "read_file".to_string(),
        },
        Event::ToolUseDelta {
            id: "__tc_0".to_string(),
            partial_json: i1,
        },
        Event::ToolUseEnd {
            id: "call_abc".to_string(),
        },
        // Tool 2: start with real id, deltas use __tc_1
        Event::ToolUseStart {
            id: "call_def".to_string(),
            name: "read_file".to_string(),
        },
        Event::ToolUseDelta {
            id: "__tc_1".to_string(),
            partial_json: i2,
        },
        Event::ToolUseEnd {
            id: "call_def".to_string(),
        },
        Event::MessageStop {
            stop_reason: StopReason::ToolUse,
        },
        Event::TextDelta {
            text: "read both files".to_string(),
        },
        Event::MessageStop {
            stop_reason: StopReason::EndTurn,
        },
    ]);

    let mut session = Session::new("bug-f".to_string());
    let mut sink = CollectSink::default();

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

    assert_eq!(sink.tool_ends.len(), 2, "both tools should execute");

    // CRITICAL: neither tool should have errored
    // Before the fix, __tc_1 deltas were dropped → empty params → read_file error
    assert!(
        !sink.tool_ends[0].2,
        "first read_file should succeed (not error). output: {}",
        sink.tool_ends[0].1
    );
    assert!(
        !sink.tool_ends[1].2,
        "second read_file should succeed (not error with empty params). output: {}",
        sink.tool_ends[1].1
    );

    // Both reads should return actual content
    assert!(
        sink.tool_ends[0].1.contains("content of file 1"),
        "first read should return file1 content"
    );
    assert!(
        sink.tool_ends[1].1.contains("content of file 2"),
        "second read should return file2 content (before fix this would be empty params error)"
    );
}

// ---------------------------------------------------------------------------
// Bug C: coalesced {ToolResult + Text} drops Text block in build_openai_body
// After ReplaceAndExec restart, follow-up prompt must reach the provider.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn regression_bug_c_coalesced_tool_result_plus_text_not_dropped() {
    use futures_util::Stream;
    use piku_api::RequestContent;
    use piku_runtime::{ContentBlock, ConversationMessage};
    use std::pin::Pin;
    use std::sync::{Arc, Mutex};

    struct RequestCapture(Arc<Mutex<Vec<piku_api::MessageRequest>>>);
    impl piku_api::Provider for RequestCapture {
        fn name(&self) -> &'static str {
            "capture"
        }
        fn stream_message(
            &self,
            req: piku_api::MessageRequest,
        ) -> Pin<Box<dyn Stream<Item = Result<Event, piku_api::ApiError>> + Send + '_>> {
            self.0.lock().unwrap().push(req);
            Box::pin(async_stream::stream! {
                yield Ok(Event::TextDelta { text: "response".to_string() });
                yield Ok(Event::MessageStop { stop_reason: StopReason::EndTurn });
            })
        }
    }

    // Build a session that ends like post-ReplaceAndExec:
    // user: "build piku" → assistant: [ToolUse{bash}] → tool: [ToolResult{bash_output}]
    let mut session = Session::new("bug-c".to_string());
    session.push(ConversationMessage::user("build piku"));
    session.push(ConversationMessage::assistant(
        vec![ContentBlock::ToolUse {
            id: "b1".to_string(),
            name: "bash".to_string(),
            input: serde_json::json!({"command": "cargo build --release -p piku"}),
        }],
        None,
    ));
    session.push(ConversationMessage::tool_result(
        "b1".to_string(),
        "Finished release [optimized] target(s) in 42.0s".to_string(),
        false,
    ));

    let captured = Arc::new(Mutex::new(Vec::new()));
    let provider = RequestCapture(captured.clone());
    let mut sink = CollectSink::default();

    // Add follow-up prompt (the new user input after restart)
    run_turn(
        "what did you just build?",
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

    let requests = captured.lock().unwrap();
    assert!(!requests.is_empty());
    let req = &requests[0];

    // Find the last message in the API request — it should contain "what did you just build?"
    let all_text: String = req
        .messages
        .iter()
        .flat_map(|m| {
            m.content.iter().filter_map(|c| {
                if let RequestContent::Text { text } = c {
                    Some(text.as_str())
                } else {
                    None
                }
            })
        })
        .collect::<Vec<_>>()
        .join(" ");

    assert!(
        all_text.contains("what did you just build"),
        "follow-up prompt should be in the API request (before fix it was dropped).\n\
         All text in request: {all_text:?}\n\
         Messages: {:?}",
        req.messages
            .iter()
            .map(|m| (&m.role, m.content.len()))
            .collect::<Vec<_>>()
    );
}

// ---------------------------------------------------------------------------
// Bug D: edit_file CRLF normalization
// ---------------------------------------------------------------------------

#[tokio::test]
async fn regression_bug_d_edit_file_works_on_crlf_file() {
    let dir = tempdir();
    let path = dir.join("windows.rs");

    // Write file with Windows CRLF line endings
    let content_crlf = "fn main() {\r\n    println!(\"hello\");\r\n}\r\n";
    std::fs::write(&path, content_crlf.as_bytes()).unwrap();

    // LLM provides old_string with LF only (as LLMs always do)
    let result = piku_tools::edit_file::execute(serde_json::json!({
        "path": path,
        "old_string": "println!(\"hello\");",  // LF, no \r
        "new_string": "println!(\"goodbye\");",
    }));

    assert!(
        !result.is_error,
        "edit_file should succeed on CRLF file with LF old_string (before fix: 'not found'). \
         error: {}",
        result.output
    );

    let final_content = std::fs::read_to_string(&path).unwrap();
    assert!(
        final_content.contains("goodbye"),
        "edit should have replaced the text: {final_content:?}"
    );
    // CRLF line endings should be preserved
    assert!(
        final_content.contains("\r\n"),
        "CRLF line endings should be preserved after edit: {final_content:?}"
    );
}

#[tokio::test]
async fn regression_bug_d_edit_file_crlf_content_preserved_exactly() {
    let dir = tempdir();
    let path = dir.join("mixed.txt");

    // Three lines, CRLF endings
    let original = "first line\r\nsecond line\r\nthird line\r\n";
    std::fs::write(&path, original.as_bytes()).unwrap();

    let result = piku_tools::edit_file::execute(serde_json::json!({
        "path": path,
        "old_string": "second line",
        "new_string": "REPLACED LINE",
    }));

    assert!(!result.is_error, "{}", result.output);

    let bytes = std::fs::read(&path).unwrap();
    let text = String::from_utf8(bytes).unwrap();
    assert_eq!(
        text, "first line\r\nREPLACED LINE\r\nthird line\r\n",
        "CRLF endings should be preserved in all lines"
    );
}

// ---------------------------------------------------------------------------
// Bug E: glob path traversal sandbox
// ---------------------------------------------------------------------------

#[test]
fn regression_bug_e_glob_rejects_traversal_pattern() {
    let dir = tempdir();
    std::fs::write(dir.join("safe.txt"), "").unwrap();

    // Attempt to traverse out of base directory
    let result = piku_tools::glob::execute(serde_json::json!({
        "pattern": "../../../etc/passwd",
        "path": dir,
    }));

    assert!(
        result.is_error,
        "glob should reject ../../../ traversal pattern (before fix: would read /etc/passwd)"
    );
    assert!(
        result.output.contains("escapes") || result.output.contains("outside"),
        "error should mention traversal. output: {}",
        result.output
    );
}

#[test]
fn regression_bug_e_glob_allows_normal_patterns() {
    let dir = tempdir();
    std::fs::write(dir.join("a.rs"), "").unwrap();
    std::fs::write(dir.join("b.rs"), "").unwrap();

    let result = piku_tools::glob::execute(serde_json::json!({
        "pattern": "*.rs",
        "path": dir,
    }));

    assert!(
        !result.is_error,
        "normal glob should still work: {}",
        result.output
    );
    assert!(result.output.contains(".rs"), "should find .rs files");
}

#[test]
fn regression_bug_e_glob_allows_recursive_within_base() {
    let dir = tempdir();
    std::fs::create_dir_all(dir.join("src/nested")).unwrap();
    std::fs::write(dir.join("src/nested/deep.rs"), "").unwrap();

    let result = piku_tools::glob::execute(serde_json::json!({
        "pattern": "src/**/*.rs",
        "path": dir,
    }));

    assert!(
        !result.is_error,
        "recursive glob within base should work: {}",
        result.output
    );
    assert!(result.output.contains("deep.rs"), "should find nested file");
}

// ---------------------------------------------------------------------------
// Bug B: bash uses sh -c not sh -lc (no login shell profile sourcing)
// Verified by timing: sh -c should be faster than sh -lc with heavy .zshrc
// We can't easily control .zshrc in tests, so we verify process group works.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn regression_bug_b_sh_c_not_lc_used() {
    // sh -c does not source login profiles. We verify the command actually runs
    // by checking that simple commands complete quickly.
    let start = std::time::Instant::now();
    let result = piku_tools::bash::execute(serde_json::json!({
        "command": "echo PIKU_TEST_SH_C",
        "timeout_ms": 5000,
    }))
    .await;

    let elapsed = start.elapsed();
    assert!(
        !result.is_error,
        "sh -c command should succeed: {}",
        result.output
    );
    assert!(result.output.contains("PIKU_TEST_SH_C"));

    // With sh -c (not sh -lc), this should complete in well under 1s
    // even on systems with heavy shell profiles (no profiles are sourced).
    assert!(
        elapsed.as_secs() < 2,
        "sh -c should not source slow shell profiles: elapsed {}ms",
        elapsed.as_millis()
    );
}

#[tokio::test]
async fn regression_bug_b_kill_on_drop_prevents_zombie() {
    // Start a process that would run long, then let it timeout.
    // Verify no zombie processes are left (we can't easily check all
    // descendants, but we verify the timeout fires and returns cleanly).
    let result = piku_tools::bash::execute(serde_json::json!({
        "command": "sleep 60",
        "timeout_ms": 200,
    }))
    .await;

    assert!(result.is_error, "should timeout");
    assert!(
        result.output.contains("timed out"),
        "should indicate timeout: {}",
        result.output
    );
}

// ---------------------------------------------------------------------------
// Realistic codebase editing: real Rust syntax
// ---------------------------------------------------------------------------

#[tokio::test]
async fn e2e_edit_real_rust_source_adds_comment() {
    let dir = tempdir();
    let target = dir.join("lib.rs");

    // Use realistic Rust source with doc comments, imports, etc.
    let source = r"/// Public library surface — used by integration tests and main.rs.
pub mod self_update;
pub mod cli;
";
    std::fs::write(&target, source).unwrap();

    let edit_input = serde_json::json!({
        "path": target,
        "old_string": "pub mod cli;",
        "new_string": "// CLI argument parsing and provider resolution.\npub mod cli;",
    })
    .to_string();

    let provider = SequenceProvider::new(vec![
        tool_call_events("e1", "edit_file", &edit_input),
        text_stop("Added a comment above the cli module declaration."),
    ]);

    let mut session = Session::new("realistic-edit".to_string());
    let mut sink = CollectSink::default();

    run_turn(
        "Add a comment before `pub mod cli;` explaining what it does",
        &mut session,
        &provider,
        "test-model",
        &[],
        all_tool_definitions(),
        &AllowAll,
        &mut sink,
        None,
        None,
    )
    .await;

    assert!(
        !sink.tool_ends.is_empty() && !sink.tool_ends[0].2,
        "edit_file should succeed: {:?}",
        sink.tool_ends.first()
    );

    let edited = std::fs::read_to_string(&target).unwrap();
    assert!(edited.contains("// CLI argument parsing"));
    assert!(edited.contains("pub mod cli;"));

    // Comment precedes module declaration
    let comment_pos = edited.find("// CLI argument parsing").unwrap();
    let mod_pos = edited.find("pub mod cli;").unwrap();
    assert!(comment_pos < mod_pos, "comment should precede declaration");

    // Only one instance of the declaration
    assert_eq!(edited.matches("pub mod cli;").count(), 1);
}

#[tokio::test]
async fn e2e_edit_ambiguous_recovers_with_context() {
    let dir = tempdir();
    let target = dir.join("ambiguous.rs");

    std::fs::write(
        &target,
        r"struct Foo;
impl Foo {
    pub fn new() -> Self { Foo }
}

struct Bar;
impl Bar {
    pub fn new() -> Self { Bar }
}
",
    )
    .unwrap();

    // First call: ambiguous (two matches)
    let bad_edit = serde_json::json!({
        "path": target,
        "old_string": "pub fn new() -> Self {",
        "new_string": "/// Creates a new instance.\npub fn new() -> Self {",
    })
    .to_string();

    // Second call: precise with context
    let good_edit = serde_json::json!({
        "path": target,
        "old_string": "impl Foo {\n    pub fn new() -> Self { Foo }",
        "new_string": "impl Foo {\n    /// Creates a new Foo instance.\n    pub fn new() -> Self { Foo }",
    }).to_string();

    let provider = SequenceProvider::new(vec![
        tool_call_events("e1", "edit_file", &bad_edit),
        tool_call_events("e2", "edit_file", &good_edit),
        text_stop("Added the doc comment to Foo::new."),
    ]);

    let mut session = Session::new("ambig".to_string());
    let mut sink = CollectSink::default();

    run_turn(
        "Add doc comment to Foo::new",
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

    // First edit: must fail with ambiguity error
    assert!(sink.tool_ends[0].2, "first edit should error (ambiguous)");
    assert!(
        sink.tool_ends[0].1.contains("ambiguous") || sink.tool_ends[0].1.contains("matched"),
        "error should say 'ambiguous': {}",
        sink.tool_ends[0].1
    );

    // Second edit: must succeed
    assert!(
        !sink.tool_ends[1].2,
        "second edit should succeed: {}",
        sink.tool_ends[1].1
    );

    let final_content = std::fs::read_to_string(&target).unwrap();
    assert!(final_content.contains("/// Creates a new Foo instance."));
    // Bar::new untouched
    assert!(!final_content.contains("Creates a new Bar"));
}

// ===========================================================================
// Background agent / TaskRegistry integration
// ===========================================================================

/// Verify that `spawn_local` inside a `LocalSet` completes correctly.
/// This is the critical regression test for the "`spawn_local` without `LocalSet`" bug.
#[tokio::test]
async fn spawn_agent_completes_in_local_set() {
    use piku_runtime::{TaskRegistry, TaskStatus};
    use tokio::task::LocalSet;

    let registry = TaskRegistry::new();
    let reg_clone = registry.clone();

    let local = LocalSet::new();
    local
        .run_until(async move {
            // Simulate what execute_spawn_agent does: spawn_local a task that
            // registers completion in the registry.
            let task_id =
                reg_clone.register("test-agent".to_string(), "test task".to_string(), 1, None);
            let tid = task_id.clone();
            let reg2 = reg_clone.clone();
            tokio::task::spawn_local(async move {
                // Simulate subagent doing work
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
                reg2.complete(&tid, "done!", 1);
            });

            // Wait for completion via join channel
            let rx = reg_clone.wait_for(&task_id);
            let entry = tokio::time::timeout(std::time::Duration::from_secs(2), rx)
                .await
                .expect("timed out waiting for agent")
                .expect("channel closed");

            assert_eq!(entry.status, TaskStatus::Done);
            assert_eq!(entry.output.as_deref(), Some("done!"));
            assert_eq!(entry.turns_used, 1);
        })
        .await;
}

/// Verify completion notification reaches an interjection channel.
#[tokio::test]
async fn completion_notification_fires() {
    use piku_runtime::TaskRegistry;
    use tokio::task::LocalSet;

    let registry = TaskRegistry::new();
    let (notif_tx, mut notif_rx) = tokio::sync::mpsc::channel::<String>(8);
    registry.set_notification_channel(notif_tx);

    let local = LocalSet::new();
    local
        .run_until(async move {
            let task_id = registry.register(
                "notify-test".to_string(),
                "notification task".to_string(),
                1,
                None,
            );
            let tid = task_id.clone();
            let reg = registry.clone();
            tokio::task::spawn_local(async move {
                tokio::time::sleep(std::time::Duration::from_millis(5)).await;
                reg.complete(&tid, "result text", 2);
            });

            // Wait for notification message
            let msg = tokio::time::timeout(std::time::Duration::from_secs(2), notif_rx.recv())
                .await
                .expect("timed out")
                .expect("channel closed");

            assert!(
                msg.contains("notify-test"),
                "notification should include agent name: {msg}"
            );
            assert!(
                msg.contains("result text"),
                "notification should include output: {msg}"
            );
        })
        .await;
}

/// Verify `subagent_type` routing applies built-in system prompt.
#[test]
fn subagent_type_routes_to_builtin_system_prompt() {
    use piku_runtime::agents::find_built_in;
    use piku_tools::spawn_agent::validate_spawn_agent;

    let params = serde_json::json!({
        "task": "check if the tests pass",
        "subagent_type": "verification"
    });
    let p = validate_spawn_agent(params).unwrap();
    assert_eq!(p.subagent_type.as_deref(), Some("verification"));

    let def = find_built_in("verification").expect("verification agent must exist");
    assert!(
        def.system_prompt.contains("try to break it"),
        "wrong system prompt"
    );
    assert!(
        def.disallowed_tools.iter().any(|t| t == "write_file"),
        "write_file must be disallowed"
    );
    assert!(
        def.disallowed_tools.iter().any(|t| t == "edit_file"),
        "edit_file must be disallowed"
    );
}

/// Verify explorer agent exists and disallows bash.
#[test]
fn explorer_agent_disallows_bash() {
    use piku_runtime::agents::find_built_in;
    let def = find_built_in("explorer").expect("explorer agent must exist");
    assert!(
        def.disallowed_tools.iter().any(|t| t == "bash"),
        "explorer must disallow bash"
    );
    assert!(
        def.system_prompt.contains("read-only"),
        "must say read-only"
    );
}

/// Verify fork param is parsed correctly.
#[test]
fn fork_param_parsed() {
    use piku_tools::spawn_agent::validate_spawn_agent;

    let p = validate_spawn_agent(serde_json::json!({
        "task": "research something",
        "fork": true
    }))
    .unwrap();
    assert!(p.fork, "fork must be true");

    let p2 = validate_spawn_agent(serde_json::json!({
        "task": "do stuff"
    }))
    .unwrap();
    assert!(!p2.fork, "fork must default to false");
}

/// Verify auto-compaction triggers and summarises messages.
#[tokio::test]
async fn auto_compact_triggers_at_threshold() {
    use piku_runtime::compact::{compact_session, should_compact, CompactionConfig};
    use piku_runtime::session::{ContentBlock, ConversationMessage, MessageRole, Session};

    let cfg = CompactionConfig {
        preserve_recent_messages: 2,
        max_estimated_tokens: 1, // force trigger on any content
    };

    let mut s = Session::new("compact-test".to_string());
    for i in 0..6 {
        if i % 2 == 0 {
            s.push(ConversationMessage::user(format!(
                "user message {i} with enough content to count"
            )));
        } else {
            s.push(ConversationMessage::assistant(
                vec![ContentBlock::Text {
                    text: format!("assistant response {i} with enough content"),
                }],
                None,
            ));
        }
    }

    assert!(should_compact(&s, cfg), "should trigger compaction");
    let result = compact_session(&s, cfg);
    assert!(
        result.removed_message_count > 0,
        "must remove some messages"
    );
    assert!(
        result.compacted_session.messages.len() < s.messages.len(),
        "compacted session must be shorter"
    );
    // First message in compacted session is the system continuation
    assert_eq!(
        result.compacted_session.messages[0].role,
        MessageRole::System,
        "first compacted message must be system"
    );
}

/// Verify `TaskRegistry.fail()` marks task and fires notification.
#[tokio::test]
async fn task_failure_notification_fires() {
    use piku_runtime::TaskRegistry;
    use piku_runtime::TaskStatus;

    let registry = TaskRegistry::new();
    let (notif_tx, mut notif_rx) = tokio::sync::mpsc::channel::<String>(8);
    registry.set_notification_channel(notif_tx);

    let task_id = registry.register("fail-test".to_string(), "failing task".to_string(), 1, None);
    registry.fail(&task_id, "something went wrong");

    let entry = registry.status(&task_id).unwrap();
    assert_eq!(entry.status, TaskStatus::Failed);
    assert_eq!(entry.output.as_deref(), Some("something went wrong"));

    let msg = notif_rx
        .try_recv()
        .expect("notification should be immediate");
    assert!(
        msg.contains("failed"),
        "notification should say failed: {msg}"
    );
}
