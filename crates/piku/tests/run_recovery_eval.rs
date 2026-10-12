//! Paired evaluation of clean process-boundary recovery.
//!
//! This proves continuity across an orderly `Session::save`/`Session::load`
//! and `RunRecorder::open` boundary. It does not measure abrupt-crash recovery,
//! where the session file and append-only run record can diverge.

use std::collections::BTreeSet;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use futures_util::Stream;
use piku::run_view::{build_search_index_with_artifacts, render_html_with_artifacts, render_text};
use piku_api::{ApiError, Event, MessageRequest, Provider, RequestContent, StopReason, TokenUsage};
use piku_runtime::{
    audit_run_record, read_run_record, AllowAll, ContentBlock, OutputSink, PostToolAction,
    RecordingSink, RunEvent, RunRecorder, Session,
};

const MODEL: &str = "recovery-eval-model";
const PRIOR_MARKER: &str = "cobalt-heron-4821";
const RECOVERED_SESSION_ID: &str = "clean-boundary-session";

#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(clippy::struct_excessive_bools)] // The evaluation contract is an explicit metric vector.
struct RecoveryMetric {
    prior_marker_available: bool,
    prior_message_selected: bool,
    same_session_id: bool,
    sequence_contiguous: bool,
    attempted_turns: usize,
    completed_turns: usize,
}

/// What a surface exposes for one exact recovery-selection question.
///
/// This is a raw vector, not a score: each surface has a different trade-off
/// between immediate inspection and its ability to join the run record back to
/// the durable session message that supplied the selected context.
#[derive(Debug, Clone, PartialEq, Eq)]
struct RecoveryPresentationMetric {
    surface: &'static str,
    marker_candidates: usize,
    recovered_context_candidates: usize,
    false_positives: usize,
    target_rank: Option<usize>,
    chars_exposed_after_query: usize,
    exact_session_index_visible: bool,
    selected_flag_visible: bool,
    typed_marker_to_selection_link_resolved: bool,
    navigation_actions_to_context_event: Option<usize>,
    navigation_actions_to_end_to_end_proof: Option<usize>,
    provenance: BTreeSet<&'static str>,
}

struct CapturingProvider {
    response: String,
    requests: Arc<Mutex<Vec<MessageRequest>>>,
}

impl CapturingProvider {
    fn new(response: impl Into<String>) -> Self {
        Self {
            response: response.into(),
            requests: Arc::new(Mutex::new(Vec::new())),
        }
    }

    fn requests(&self) -> Arc<Mutex<Vec<MessageRequest>>> {
        Arc::clone(&self.requests)
    }
}

impl Provider for CapturingProvider {
    fn name(&self) -> &'static str {
        "recovery-eval"
    }

    fn stream_message(
        &self,
        request: MessageRequest,
    ) -> Pin<Box<dyn Stream<Item = Result<Event, ApiError>> + Send + '_>> {
        self.requests.lock().unwrap().push(request);
        let response = self.response.clone();
        Box::pin(async_stream::stream! {
            yield Ok(Event::TextDelta { text: response });
            yield Ok(Event::MessageStop {
                stop_reason: StopReason::EndTurn,
            });
            yield Ok(Event::UsageDelta {
                usage: TokenUsage {
                    input_tokens: 12,
                    output_tokens: 4,
                    ..Default::default()
                },
            });
        })
    }
}

#[derive(Default)]
struct QuietSink;

impl OutputSink for QuietSink {
    fn on_text(&mut self, _text: &str) {}

    fn on_tool_start(&mut self, _tool_name: &str, _tool_id: &str, _input: &serde_json::Value) {}

    fn on_tool_end(
        &mut self,
        _tool_name: &str,
        _tool_id: &str,
        _result: &str,
        _is_error: bool,
    ) -> PostToolAction {
        PostToolAction::Continue
    }

    fn on_permission_denied(&mut self, _tool_name: &str, _reason: &str) {}

    fn on_turn_complete(&mut self, _usage: &TokenUsage, _iterations: u32) {}
}

fn message_contains_marker(blocks: &[ContentBlock]) -> bool {
    blocks.iter().any(|block| match block {
        ContentBlock::Text { text } => text.contains(PRIOR_MARKER),
        ContentBlock::ToolResult { output, .. } => output.contains(PRIOR_MARKER),
        ContentBlock::ToolUse { input, .. } => input.to_string().contains(PRIOR_MARKER),
    })
}

fn request_contains_marker(request: &MessageRequest) -> bool {
    request.messages.iter().any(|message| {
        message.content.iter().any(|content| match content {
            RequestContent::Text { text } => text.contains(PRIOR_MARKER),
            RequestContent::ToolResult { content, .. } => content.contains(PRIOR_MARKER),
            RequestContent::ToolUse { input, .. } => input.to_string().contains(PRIOR_MARKER),
        })
    })
}

async fn run_recorded_turn(
    input: &str,
    turn_id: &str,
    session: &mut Session,
    provider: &dyn Provider,
    recorder: &mut RunRecorder,
) {
    let mut sink = QuietSink;
    let mut recording_sink = RecordingSink::new(&mut sink, recorder, turn_id);
    let result = piku_runtime::run_turn(
        input,
        session,
        provider,
        MODEL,
        &[],
        vec![],
        &AllowAll,
        &mut recording_sink,
        None,
        None,
    )
    .await;

    assert!(result.stream_error.is_none());
    assert!(recording_sink.take_record_error().is_none());
}

fn measure_recovery_presentation(
    events: &[piku_runtime::RunEventEnvelope],
    session: &Session,
    session_path: &std::path::Path,
    recovered_run_path: &std::path::Path,
    prior_marker_index: usize,
) -> Vec<RecoveryPresentationMetric> {
    let selection_query = format!("\"session_index\":{prior_marker_index}");
    let compact = render_text(events);
    let selected_context = events
        .iter()
        .filter(|event| {
            event.scope.turn_id() == Some("turn-1")
                && matches!(
                    &event.event,
                    RunEvent::ContextBuilt { manifest }
                        if manifest.messages.iter().any(|message| {
                            message.session_index == prior_marker_index && message.selected
                        })
                )
        })
        .collect::<Vec<_>>();
    let selected_context_chars = selected_context
        .iter()
        .map(|event| serde_json::to_string(event).unwrap().chars().count())
        .sum();
    let search_index = build_search_index_with_artifacts(events, recovered_run_path).unwrap();
    let selected_message = &session.messages[prior_marker_index];
    let session_file_contains_marker = std::fs::read_to_string(session_path)
        .unwrap()
        .contains(PRIOR_MARKER);
    let html = render_html_with_artifacts(events, recovered_run_path).unwrap();

    assert!(compact.contains("turn-0"));
    assert!(compact.contains("turn-1"));
    assert!(compact.contains("2 / 2 turns complete"));
    assert!(!compact.contains(&selection_query));
    assert!(html.contains("default-src 'none'"));
    assert!(html.contains("id=\"run-data\""));
    assert!(session_file_contains_marker);
    assert!(message_contains_marker(&selected_message.blocks));

    vec![
        RecoveryPresentationMetric {
            surface: "terminal_compact",
            marker_candidates: compact.match_indices(PRIOR_MARKER).count(),
            recovered_context_candidates: 0,
            false_positives: 0,
            target_rank: None,
            chars_exposed_after_query: compact.chars().count(),
            exact_session_index_visible: false,
            selected_flag_visible: false,
            typed_marker_to_selection_link_resolved: false,
            navigation_actions_to_context_event: None,
            navigation_actions_to_end_to_end_proof: None,
            provenance: BTreeSet::new(),
        },
        RecoveryPresentationMetric {
            surface: "terminal_composed",
            marker_candidates: usize::from(message_contains_marker(&selected_message.blocks)),
            recovered_context_candidates: selected_context.len(),
            false_positives: selected_context.len().saturating_sub(1),
            target_rank: Some(1),
            chars_exposed_after_query: selected_context_chars,
            exact_session_index_visible: true,
            selected_flag_visible: true,
            typed_marker_to_selection_link_resolved: true,
            navigation_actions_to_context_event: Some(2),
            navigation_actions_to_end_to_end_proof: Some(3),
            provenance: BTreeSet::from([
                "sequence",
                "scope",
                "event_kind",
                "session_index",
                "selected",
            ]),
        },
        browser_recovery_presentation(&search_index, &selection_query),
    ]
}

fn browser_recovery_presentation(
    search_index: &[piku::run_view::RunSearchEntry],
    selection_query: &str,
) -> RecoveryPresentationMetric {
    let matches = search_index
        .iter()
        .filter(|entry| entry.search_text.contains(selection_query))
        .collect::<Vec<_>>();
    RecoveryPresentationMetric {
        surface: "browser_workbench",
        marker_candidates: search_index
            .iter()
            .filter(|entry| entry.search_text.contains(PRIOR_MARKER))
            .count(),
        recovered_context_candidates: matches.len(),
        false_positives: matches.len().saturating_sub(1),
        target_rank: matches
            .iter()
            .position(|entry| entry.event_kind == "context_built" && entry.scope == "turn-1")
            .map(|index| index + 1),
        chars_exposed_after_query: matches
            .iter()
            .map(|entry| entry.preview_text.chars().count())
            .sum(),
        exact_session_index_visible: matches
            .iter()
            .any(|entry| entry.search_text.contains(selection_query)),
        selected_flag_visible: matches
            .iter()
            .any(|entry| entry.search_text.contains("\"selected\":true")),
        typed_marker_to_selection_link_resolved: false,
        navigation_actions_to_context_event: Some(1),
        navigation_actions_to_end_to_end_proof: None,
        provenance: BTreeSet::from([
            "sequence",
            "scope",
            "event_kind",
            "session_index",
            "selected",
        ]),
    }
}

fn assert_recovery_presentation(presentation: &[RecoveryPresentationMetric]) {
    let compact = &presentation[0];
    let composed = &presentation[1];
    let browser = &presentation[2];
    assert_eq!(compact.surface, "terminal_compact");
    assert_eq!(browser.surface, "browser_workbench");
    assert_eq!(composed.recovered_context_candidates, 1);
    assert_eq!(browser.recovered_context_candidates, 1);
    assert_eq!(composed.target_rank, Some(1));
    assert_eq!(browser.target_rank, Some(1));
    assert_eq!(composed.false_positives, 0);
    assert_eq!(browser.false_positives, 0);
    assert!(browser.chars_exposed_after_query < compact.chars_exposed_after_query);
    assert!(browser.exact_session_index_visible);
    assert!(browser.selected_flag_visible);
    assert!(!browser.typed_marker_to_selection_link_resolved);
    assert!(composed.typed_marker_to_selection_link_resolved);
    assert_eq!(browser.navigation_actions_to_context_event, Some(1));
    assert_eq!(browser.navigation_actions_to_end_to_end_proof, None);
    assert_eq!(composed.navigation_actions_to_end_to_end_proof, Some(3));
}

async fn evaluate_recovered_case(directory: &std::path::Path) -> RecoveryMetric {
    let session_path = directory.join("session.json");
    let recovered_run_path = directory.join("recovered-run.jsonl");
    // Process A completes one turn, then durably saves both projections.
    {
        let provider = CapturingProvider::new(format!("remember {PRIOR_MARKER}"));
        let mut session = Session::new(RECOVERED_SESSION_ID.to_string());
        session.record_provider(provider.name(), MODEL);
        let mut recorder = RunRecorder::open(&recovered_run_path, &session.id).unwrap();
        run_recorded_turn(
            "retain the fixed recovery marker",
            "turn-0",
            &mut session,
            &provider,
            &mut recorder,
        )
        .await;
        session.save(&session_path).unwrap();
    }

    // Process B reconstructs state solely from the two durable files.
    let mut recovered = Session::load(&session_path).unwrap();
    let prior_marker_index = recovered
        .messages
        .iter()
        .position(|message| message_contains_marker(&message.blocks));
    let prior_marker_available = prior_marker_index.is_some();
    let recovered_same_session = recovered.id == RECOVERED_SESSION_ID;
    let follow_up_provider = CapturingProvider::new("the prior marker remains available");
    let follow_up_requests = follow_up_provider.requests();
    {
        let mut recorder = RunRecorder::open(&recovered_run_path, &recovered.id).unwrap();
        run_recorded_turn(
            "which marker did the prior process retain?",
            "turn-1",
            &mut recovered,
            &follow_up_provider,
            &mut recorder,
        )
        .await;
    }

    {
        let requests = follow_up_requests.lock().unwrap();
        assert_eq!(requests.len(), 1);
        assert!(
            request_contains_marker(&requests[0]),
            "the recovered marker must reach the second provider request"
        );
    }

    let events = read_run_record(&recovered_run_path).unwrap();
    let audit = audit_run_record(&events);
    let turn_ids = events
        .iter()
        .filter_map(|event| event.scope.turn_id().map(str::to_string))
        .collect::<BTreeSet<_>>();
    let sequence_contiguous = events
        .iter()
        .enumerate()
        .all(|(index, event)| event.sequence == u64::try_from(index).unwrap());
    let prior_message_selected = events.iter().any(|event| {
        event.scope.turn_id() == Some("turn-1")
            && matches!(
                &event.event,
                RunEvent::ContextBuilt { manifest }
                    if prior_marker_index.is_some_and(|marker_index| {
                        manifest.messages.iter().any(|message| {
                            message.session_index == marker_index && message.selected
                        })
                    })
            )
    });
    let presentation = measure_recovery_presentation(
        &events,
        &recovered,
        &session_path,
        &recovered_run_path,
        prior_marker_index.expect("the positive recovery fixture has the marker"),
    );
    assert_recovery_presentation(&presentation);
    let metrics = RecoveryMetric {
        prior_marker_available,
        prior_message_selected,
        same_session_id: recovered_same_session
            && events
                .iter()
                .all(|event| event.session_id == RECOVERED_SESSION_ID),
        sequence_contiguous,
        attempted_turns: audit.turn_count,
        completed_turns: audit.completed_turn_count,
    };

    assert_eq!(
        turn_ids,
        BTreeSet::from(["turn-0".to_string(), "turn-1".to_string()])
    );
    assert!(audit.is_structurally_complete());
    metrics
}

async fn evaluate_fresh_control(directory: &std::path::Path) -> RecoveryMetric {
    // Paired negative: the same model gets a fresh session and must not acquire
    // the marker from model identity, ambient state, or the positive fixture.
    let fresh_run_path = directory.join("fresh-run.jsonl");
    let fresh_provider = CapturingProvider::new("no prior context");
    let fresh_requests = fresh_provider.requests();
    let mut fresh = Session::new("fresh-same-model-session".to_string());
    let fresh_marker_available = fresh
        .messages
        .iter()
        .any(|message| message_contains_marker(&message.blocks));
    {
        let mut recorder = RunRecorder::open(&fresh_run_path, &fresh.id).unwrap();
        run_recorded_turn(
            "which marker did the prior process retain?",
            "turn-0",
            &mut fresh,
            &fresh_provider,
            &mut recorder,
        )
        .await;
    }

    {
        let requests = fresh_requests.lock().unwrap();
        assert_eq!(requests.len(), 1);
        assert!(
            !request_contains_marker(&requests[0]),
            "a fresh same-model session must not expose the prior marker"
        );
    }

    let fresh_events = read_run_record(&fresh_run_path).unwrap();
    let fresh_audit = audit_run_record(&fresh_events);
    let metrics = RecoveryMetric {
        prior_marker_available: fresh_marker_available,
        prior_message_selected: false,
        same_session_id: fresh.id == RECOVERED_SESSION_ID,
        sequence_contiguous: fresh_events
            .iter()
            .enumerate()
            .all(|(index, event)| event.sequence == u64::try_from(index).unwrap()),
        attempted_turns: fresh_audit.turn_count,
        completed_turns: fresh_audit.completed_turn_count,
    };

    assert!(fresh_audit.is_structurally_complete());
    metrics
}

#[tokio::test]
async fn clean_process_boundary_recovery_preserves_selected_context_and_run_continuity() {
    let directory = tempfile::tempdir().unwrap();
    let positive = evaluate_recovered_case(directory.path()).await;
    assert_eq!(
        positive,
        RecoveryMetric {
            prior_marker_available: true,
            prior_message_selected: true,
            same_session_id: true,
            sequence_contiguous: true,
            attempted_turns: 2,
            completed_turns: 2,
        }
    );

    let negative = evaluate_fresh_control(directory.path()).await;
    assert_eq!(
        negative,
        RecoveryMetric {
            prior_marker_available: false,
            prior_message_selected: false,
            same_session_id: false,
            sequence_contiguous: true,
            attempted_turns: 1,
            completed_turns: 1,
        }
    );
}

/// Real terminal commands against the same OpenAI-compatible streaming fixture
/// shape used by `binary_tests`. Scripted answers exercise plumbing, not model quality.
mod terminal_workflow {
    use std::io::{Read, Seek, SeekFrom, Write};
    use std::path::Path;
    use std::process::{Command, Output};
    use std::sync::{Arc, Mutex};
    use std::time::{Duration, Instant};

    use serde_json::{json, Value};

    fn checked(command: &mut Command) -> Output {
        let mut stdout = tempfile::tempfile().unwrap();
        let mut stderr = tempfile::tempfile().unwrap();
        let mut child = command
            .stdout(stdout.try_clone().unwrap())
            .stderr(stderr.try_clone().unwrap())
            .spawn()
            .unwrap();
        let deadline = Instant::now() + Duration::from_secs(45);
        let status = loop {
            if let Some(status) = child.try_wait().unwrap() {
                break status;
            }
            if Instant::now() >= deadline {
                child.kill().unwrap();
                child.wait().unwrap();
                panic!("command exceeded 45 seconds: {command:?}");
            }
            std::thread::sleep(Duration::from_millis(10));
        };
        stdout.seek(SeekFrom::Start(0)).unwrap();
        stderr.seek(SeekFrom::Start(0)).unwrap();
        let mut output = Output {
            status,
            stdout: Vec::new(),
            stderr: Vec::new(),
        };
        stdout.read_to_end(&mut output.stdout).unwrap();
        stderr.read_to_end(&mut output.stderr).unwrap();
        assert!(
            output.status.success(),
            "{command:?}\n{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        output
    }

    fn cli(config: &Path, workspace: &Path, url: &str) -> Command {
        let mut command = Command::new(env!("CARGO_BIN_EXE_piku"));
        command
            .env_clear()
            .env("PATH", std::env::var_os("PATH").unwrap())
            .env("HOME", config)
            .env("XDG_CONFIG_HOME", config)
            .env("PIKU_NO_DOTENV", "1")
            .env("PIKU_BASE_URL", url)
            .env("PIKU_API_KEY", "synthetic-test-key")
            .env("PIKU_MODEL", "canned-model")
            .current_dir(workspace);
        command
    }

    fn git(workspace: &Path, args: &[&str]) -> String {
        String::from_utf8(
            checked(
                Command::new("git")
                    .current_dir(workspace)
                    .env("GIT_CONFIG_NOSYSTEM", "1")
                    .env("GIT_CONFIG_GLOBAL", "/dev/null")
                    .args(args),
            )
            .stdout,
        )
        .unwrap()
    }

    fn scripted_server(
        file: &Path,
    ) -> (String, Arc<Mutex<Vec<Value>>>, std::thread::JoinHandle<()>) {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let url = format!("http://{}/v1", listener.local_addr().unwrap());
        let requests = Arc::new(Mutex::new(Vec::new()));
        let captured = Arc::clone(&requests);
        let tool = |id: &str, debug: bool| json!({"choices":[{"delta":{"tool_calls":[{"index":0,"id":id,"type":"function","function":{"name":"write_file","arguments":json!({"path":file,"content":format!("{{\"port\":9090,\"debug\":{debug},\"service\":\"fixture\"}}\n")}).to_string()}}]},"finish_reason":null}]});
        let final_answer = json!({"choices":[{"delta":{"content":"Configuration complete."},"finish_reason":null}]});
        let responses = [
            (tool("partial-write", true), "tool_calls"),
            (final_answer.clone(), "stop"),
            (tool("corrected-write", false), "tool_calls"),
            (final_answer, "stop"),
        ];
        let handle = std::thread::spawn(move || {
            for (delta, stop) in responses {
                let deadline = Instant::now() + Duration::from_secs(30);
                let mut stream = loop {
                    match listener.accept() {
                        Ok((stream, _)) => break stream,
                        Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                            assert!(Instant::now() < deadline, "scripted request never arrived");
                            std::thread::sleep(Duration::from_millis(10));
                        }
                        Err(error) => panic!("accept: {error}"),
                    }
                };
                stream.set_nonblocking(false).unwrap();
                stream
                    .set_read_timeout(Some(Duration::from_secs(10)))
                    .unwrap();
                stream
                    .set_write_timeout(Some(Duration::from_secs(10)))
                    .unwrap();
                let mut request = Vec::new();
                let mut chunk = [0u8; 4096];
                let (header_end, length) = loop {
                    let n = stream.read(&mut chunk).unwrap();
                    assert_ne!(n, 0, "incomplete HTTP headers");
                    request.extend_from_slice(&chunk[..n]);
                    if let Some(end) = request.windows(4).position(|w| w == b"\r\n\r\n") {
                        let headers = String::from_utf8_lossy(&request[..end]).to_ascii_lowercase();
                        let length = headers
                            .lines()
                            .find_map(|line| line.strip_prefix("content-length:"))
                            .unwrap()
                            .trim()
                            .parse::<usize>()
                            .unwrap();
                        break (end + 4, length);
                    }
                };
                while request.len() < header_end + length {
                    let n = stream.read(&mut chunk).unwrap();
                    assert_ne!(n, 0, "incomplete HTTP body");
                    request.extend_from_slice(&chunk[..n]);
                }
                captured.lock().unwrap().push(
                    serde_json::from_slice(&request[header_end..header_end + length]).unwrap(),
                );
                let finish = json!({"choices":[{"delta":{},"finish_reason":stop}]});
                write!(stream, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\ndata: {delta}\n\ndata: {finish}\n\ndata: [DONE]\n\n").unwrap();
            }
        });
        (url, requests, handle)
    }

    // This oracle reads the actual workspace independently of model text and
    // tool status. Wrong types, extra keys, unrelated edits, or a dirty parent fail.
    fn workspace_accepted(workspace: &Path) -> bool {
        let actual: Value =
            serde_json::from_slice(&std::fs::read(workspace.join("settings.json")).unwrap())
                .unwrap();
        actual == json!({"port":9090,"debug":false,"service":"fixture"})
            && git(workspace, &["status", "--porcelain"]) == " M settings.json\n"
    }

    fn assert_review_history(
        record_path: &Path,
        rejected: &str,
        session: &str,
        note: &str,
        acceptance: &str,
    ) {
        let final_record = std::fs::read_to_string(record_path).unwrap();
        assert!(
            final_record.starts_with(rejected),
            "review history rewritten"
        );
        let reopened = piku_runtime::read_run_record(record_path).unwrap();
        let audit = piku_runtime::audit_run_record(&reopened);
        assert!(audit.is_structurally_complete(), "{audit:?}");
        assert_eq!(audit.turn_count, 2);
        assert_eq!(audit.completed_turn_count, 2);
        let events: Vec<Value> = final_record
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        for (sequence, event) in events.iter().enumerate() {
            assert_eq!(event["sequence"], sequence);
            assert_eq!(event["session_id"], session);
        }
        let dispositions: Vec<_> = events
            .iter()
            .filter(|event| event["event"] == "user_disposition")
            .collect();
        assert_eq!(dispositions.len(), 2);
        assert_eq!(dispositions[0]["disposition"], "needs_work");
        assert_eq!(dispositions[0]["note"]["text"], note);
        assert_eq!(dispositions[1]["disposition"], "accepted");
        assert_eq!(dispositions[1]["note"]["text"], acceptance);
        assert_eq!(
            events
                .iter()
                .filter(|event| event["event"] == "turn_completed")
                .count(),
            2
        );
        assert_eq!(
            events
                .iter()
                .filter(|event| event["event"] == "tool_completed" && event["is_error"] == false)
                .count(),
            2
        );
    }

    fn prepare_worktree(repo: &Path, retained: &Path) -> &'static str {
        git(repo, &["init", "-q"]);
        let original = "{\"port\":8080,\"debug\":true,\"service\":\"fixture\"}\n";
        std::fs::write(repo.join("settings.json"), original).unwrap();
        git(repo, &["add", "settings.json"]);
        git(
            repo,
            &[
                "-c",
                "user.name=Fixture",
                "-c",
                "user.email=fixture@example.invalid",
                "-c",
                "commit.gpgsign=false",
                "commit",
                "-qm",
                "fixture",
            ],
        );
        git(
            repo,
            &[
                "worktree",
                "add",
                "-qb",
                "recovery-fixture",
                retained.to_str().unwrap(),
            ],
        );
        original
    }

    #[test]
    fn cli_review_reject_correct_resume_accept_retains_work() {
        let fixture = tempfile::tempdir().unwrap();
        let repo = fixture.path().join("parent");
        let retained = fixture.path().join("retained");
        let config = fixture.path().join("config");
        std::fs::create_dir_all(&repo).unwrap();
        std::fs::create_dir_all(&config).unwrap();
        let original = prepare_worktree(&repo, &retained);
        let (url, requests, server) = scripted_server(&retained.join("settings.json"));
        let prompt = "Set settings.json port to 9090 and debug to false; preserve service and all other files.";
        let first = checked(cli(&config, &retained, &url).args(["-p", prompt]));
        let stderr = String::from_utf8(first.stderr).unwrap();
        let session = stderr
            .lines()
            .find_map(|line| line.split("sessions/").nth(1))
            .and_then(|rest| rest.strip_suffix(".json"))
            .unwrap()
            .to_owned();
        let partial: Value =
            serde_json::from_slice(&std::fs::read(retained.join("settings.json")).unwrap())
                .unwrap();
        assert_eq!(
            partial,
            json!({"port":9090,"debug":true,"service":"fixture"})
        );
        assert!(
            !workspace_accepted(&retained),
            "partial answer must fail independent review"
        );
        let inspect = checked(cli(&config, &retained, &url).args(["inspect", &session]));
        assert!(String::from_utf8_lossy(&inspect.stdout).contains("Configuration complete."));
        let record_path = config.join("piku/runs").join(format!("{session}.jsonl"));
        let before_rejection = std::fs::read_to_string(&record_path).unwrap();
        let note = format!("independent review failed: debug is true; record prefix {} events; branch recovery-fixture", before_rejection.lines().count());
        checked(cli(&config, &retained, &url).args([
            "conclude",
            &session,
            "--status",
            "needs-work",
            "--note",
            &note,
        ]));
        let rejected = std::fs::read_to_string(&record_path).unwrap();
        assert!(rejected.starts_with(&before_rejection));
        // Resumption is explicit: conclusions are evidence, not automatically
        // injected correction instructions or a restarted background task.
        let correction = "Independent review found debug=true. Correct debug to false in settings.json; preserve port=9090 and service=fixture.";
        checked(cli(&config, &retained, &url).args(["--resume", &session, "-p", correction]));
        server.join().unwrap();
        let captured = requests.lock().unwrap();
        assert_eq!(captured.len(), 4);
        let resumed = captured[2]["messages"].to_string();
        assert!(resumed.contains(prompt), "original session context lost");
        assert!(resumed.contains(correction), "explicit correction absent");
        assert!(
            resumed.contains("partial-write"),
            "prior tool evidence lost"
        );
        assert!(
            workspace_accepted(&retained),
            "corrected workspace must pass independent review"
        );
        let acceptance = "Independent workspace check passed: exact JSON keys/types/values; only settings.json changed; parent unchanged.";
        assert_eq!(
            std::fs::read_to_string(repo.join("settings.json")).unwrap(),
            original
        );
        assert!(git(&repo, &["status", "--porcelain"]).is_empty());
        checked(cli(&config, &retained, &url).args([
            "conclude", &session, "--status", "accepted", "--note", acceptance,
        ]));
        assert_review_history(&record_path, &rejected, &session, &note, acceptance);
        let final_inspect = checked(cli(&config, &retained, &url).args(["inspect", &session]));
        let text = String::from_utf8_lossy(&final_inspect.stdout);
        assert!(text.contains(&note) && text.contains(acceptance));
        assert!(retained.join("settings.json").exists());
        assert_eq!(
            git(&retained, &["branch", "--show-current"]),
            "recovery-fixture\n"
        );
        assert!(
            workspace_accepted(&retained),
            "acceptance must not retire unmerged work"
        );
        eprintln!("terminal workflow: initial workspace rejected; explicit same-session resume corrected; 2 review dispositions preserved; retained work and parent isolation verified; 4 synthetic provider requests");
    }
}
