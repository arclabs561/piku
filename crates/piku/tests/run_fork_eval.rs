//! Paired evaluation of parent-to-child fork-context continuity.
//!
//! The positive and negative cases differ only in `spawn_agent.fork`. The
//! metric vector keeps functional context transfer separate from durable child
//! evidence. The positive and negative controls both require a child session,
//! run record, and typed parent-child link on disk.

use std::pin::Pin;
use std::sync::{Arc, Mutex};

use futures_util::Stream;
use piku_api::{ApiError, Event, MessageRequest, Provider, RequestContent, StopReason};
use piku_runtime::{
    audit_run_record, read_run_record, run_turn_with_registry, AllowAll, OutputSink,
    PostToolAction, RecordingSink, RunContentRef, RunEvent, RunRecorder, SubagentEvidence,
    TaskRegistry, TokenUsage,
};

#[path = "support/review_workflow.rs"]
mod review_workflow;

const PARENT_MARKER: &str = "parent-context-cobalt-4182";
const CHILD_MARKER: &str = "child-result-amber-9271";
const CHILD_TASK: &str = "Inspect the delegated fork-context continuity fixture.";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(clippy::struct_excessive_bools)] // An explicit, unaggregated metric vector.
struct ForkContinuityMetric {
    parent_marker_in_child_request: bool,
    child_result_returned_to_parent: bool,
    spawn_join_recorded: bool,
    child_execution_durable: bool,
    typed_child_run_ref_in_parent_record: bool,
}

#[derive(Default)]
struct CaptureSink {
    text: String,
}

impl OutputSink for CaptureSink {
    fn on_text(&mut self, text: &str) {
        self.text.push_str(text);
    }

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

#[derive(Default)]
struct ProviderObservations {
    child_requests: Vec<String>,
}

#[derive(Clone)]
struct ForkRoutingProvider {
    registry: TaskRegistry,
    fork: bool,
    observations: Arc<Mutex<ProviderObservations>>,
    child_file: Option<std::path::PathBuf>,
}

impl ForkRoutingProvider {
    fn new(registry: TaskRegistry, fork: bool) -> Self {
        Self {
            registry,
            fork,
            observations: Arc::new(Mutex::new(ProviderObservations::default())),
            child_file: None,
        }
    }

    fn child_requests(&self) -> Vec<String> {
        self.observations
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .child_requests
            .clone()
    }
}

impl Provider for ForkRoutingProvider {
    fn name(&self) -> &'static str {
        "fork-routing-fixture"
    }

    fn stream_message(
        &self,
        request: MessageRequest,
    ) -> Pin<Box<dyn Stream<Item = Result<Event, ApiError>> + Send + '_>> {
        let request_text = request_text(&request);
        let events = if request_text.contains(CHILD_TASK) {
            self.observations
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .child_requests
                .push(request_text);
            if let Some(path) = self.child_file.as_ref().filter(|_| {
                !request.messages.iter().flat_map(|message| &message.content).any(|content| {
                    matches!(content, RequestContent::ToolResult { tool_use_id, .. } if tool_use_id == "partial-write")
                })
            }) {
                tool_response("partial-write", "write_file", &serde_json::json!({
                    "path": path,
                    "content": "{\"port\":9090,\"debug\":true,\"service\":\"fixture\"}\n",
                }))
            } else {
                text_response(CHILD_MARKER)
            }
        } else if request_text.contains(CHILD_MARKER) {
            text_response(&format!("parent received {CHILD_MARKER}"))
        } else if request_text.contains("spawned agent") {
            let task_id = self
                .registry
                .all()
                .into_iter()
                .next()
                .expect("spawn result must correspond to a registered task")
                .id
                .to_string();
            tool_response(
                "root-join",
                "agent_join",
                &serde_json::json!({"task_id": task_id, "timeout_secs": 5}),
            )
        } else {
            tool_response(
                "root-spawn",
                "spawn_agent",
                &serde_json::json!({
                    "task": CHILD_TASK,
                    "name": "fork-continuity-probe",
                    "max_turns": 2,
                    "background": false,
                    "fork": self.fork,
                }),
            )
        };

        Box::pin(async_stream::stream! {
            for event in events {
                yield Ok(event);
            }
        })
    }

    fn boxed_clone(&self) -> Box<dyn Provider + Send + Sync + 'static> {
        Box::new(self.clone())
    }
}

fn request_text(request: &MessageRequest) -> String {
    request
        .messages
        .iter()
        .flat_map(|message| &message.content)
        .filter_map(|content| match content {
            RequestContent::Text { text } => Some(text.as_str()),
            RequestContent::ToolResult { content, .. } => Some(content.as_str()),
            RequestContent::ToolUse { .. } | RequestContent::Replay { .. } => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn text_response(text: &str) -> Vec<Event> {
    vec![
        Event::TextDelta {
            text: text.to_string(),
        },
        Event::MessageStop {
            stop_reason: StopReason::EndTurn,
        },
    ]
}

fn tool_response(id: &str, name: &str, input: &serde_json::Value) -> Vec<Event> {
    vec![
        Event::ToolUseStart {
            id: id.to_string(),
            name: name.to_string(),
        },
        Event::ToolUseDelta {
            id: id.to_string(),
            partial_json: input.to_string(),
        },
        Event::ToolUseEnd { id: id.to_string() },
        Event::MessageStop {
            stop_reason: StopReason::ToolUse,
        },
    ]
}

fn inline_text(content: &RunContentRef) -> Option<&str> {
    match content {
        RunContentRef::Inline { text } => Some(text),
        RunContentRef::Artifact { .. } | RunContentRef::Unavailable { .. } => None,
    }
}

fn recorded_tool_pair(events: &[piku_runtime::RunEventEnvelope], name: &str, id: &str) -> bool {
    let started = events.iter().any(|envelope| {
        matches!(
            &envelope.event,
            RunEvent::ToolStarted {
                tool_call_id,
                name: tool_name,
                ..
            } if tool_call_id == id && tool_name == name
        )
    });
    let completed = events.iter().any(|envelope| {
        matches!(
            &envelope.event,
            RunEvent::ToolCompleted {
                tool_call_id,
                is_error: false,
                ..
            } if tool_call_id == id
        )
    });
    started && completed
}

fn child_execution_is_durable(registry: &TaskRegistry, parent_session_id: &str) -> bool {
    registry.all().into_iter().next().is_some_and(|entry| {
        entry.evidence.is_some_and(|evidence| {
            let link = std::fs::read(&evidence.link_path)
                .ok()
                .and_then(|bytes| serde_json::from_slice::<SubagentEvidence>(&bytes).ok());
            let child_session = piku_runtime::Session::load(&evidence.session_path).ok();
            let child_events = read_run_record(&evidence.run_record_path).ok();
            link.as_ref() == Some(&evidence)
                && child_session
                    .as_ref()
                    .is_some_and(|session| session.id == evidence.child_session_id)
                && child_events.as_ref().is_some_and(|events| {
                    !events.is_empty()
                        && audit_run_record(events).is_structurally_complete()
                        && events
                            .iter()
                            .all(|event| event.session_id == evidence.child_session_id)
                })
                && evidence.parent_session_id == parent_session_id
        })
    })
}

async fn run_case(fork: bool) -> ForkContinuityMetric {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("root-run.jsonl");
    let parent_session_id = format!("fork-eval-{fork}");
    let registry = TaskRegistry::with_persistence_run_path(
        &parent_session_id,
        directory.path().join("sessions"),
        directory.path().join("runs"),
        directory.path().join("links"),
        Some(path.clone()),
    );
    let provider = ForkRoutingProvider::new(registry.clone(), fork);
    let mut session = piku_runtime::Session::new(parent_session_id.clone());
    let mut capture = CaptureSink::default();
    let mut recorder = RunRecorder::open(&path, &session.id).unwrap();

    {
        let mut sink = RecordingSink::new(&mut capture, &mut recorder, "root-turn");
        let result = run_turn_with_registry(
            &format!("Preserve this marker for delegation: {PARENT_MARKER}"),
            &mut session,
            &provider,
            "fixture-model",
            &[],
            piku_tools::all_tool_definitions(),
            &AllowAll,
            &mut sink,
            Some(5),
            None,
            &registry,
            1,
            &[],
            None,
            None,
        )
        .await;
        assert!(result.stream_error.is_none(), "fixture stream failed");
        assert!(sink.take_record_error().is_none(), "recording failed");
    }

    let child_requests = provider.child_requests();
    assert_eq!(
        child_requests.len(),
        1,
        "exactly one child request should be routed"
    );
    let child_request = &child_requests[0];
    assert_eq!(
        child_request.contains("<fork_context>"),
        fork,
        "fork wrapper presence must follow the spawn flag"
    );

    let events = read_run_record(&path).unwrap();
    let join_returned_child_marker = events.iter().any(|envelope| {
        matches!(
            &envelope.event,
            RunEvent::ToolCompleted {
                tool_call_id,
                result,
                is_error: false,
                ..
            } if tool_call_id == "root-join"
                && inline_text(result).is_some_and(|text| {
                    text.contains(CHILD_MARKER) && text.contains("run record:")
                })
        )
    });
    let parent_confirmed_child_marker = events.iter().any(|envelope| {
        matches!(
            &envelope.event,
            RunEvent::AssistantMessage { content }
                if inline_text(content).is_some_and(|text| text.contains(CHILD_MARKER))
        )
    });
    let child_execution_durable = child_execution_is_durable(&registry, &parent_session_id);
    let typed_child_run_ref_in_parent_record = typed_child_run_ref_present(&events, &registry);
    ForkContinuityMetric {
        parent_marker_in_child_request: child_request.contains(PARENT_MARKER),
        child_result_returned_to_parent: join_returned_child_marker
            && parent_confirmed_child_marker,
        spawn_join_recorded: recorded_tool_pair(&events, "spawn_agent", "root-spawn")
            && recorded_tool_pair(&events, "agent_join", "root-join"),
        child_execution_durable,
        typed_child_run_ref_in_parent_record,
    }
}

/// Whether the parent run record carries a typed `ChildRunRef` pointing at the
/// spawned child session. This is the durable evidence a workbench follows;
/// `agent_join` prose alone does not survive reopening the run from disk.
fn typed_child_run_ref_present(
    events: &[piku_runtime::RunEventEnvelope],
    registry: &TaskRegistry,
) -> bool {
    let child_session_id = registry
        .all()
        .into_iter()
        .next()
        .and_then(|entry| entry.evidence)
        .map(|evidence| evidence.child_session_id);
    child_session_id.is_some_and(|child_id| {
        events.iter().any(|envelope| {
            matches!(
                &envelope.event,
                RunEvent::ChildRunRef { relationship, child_session_id, .. }
                    if relationship == "spawn_agent" && *child_session_id == child_id
            )
        })
    })
}

#[tokio::test(flavor = "current_thread")]
async fn fork_context_continuity_is_measured_against_a_no_fork_control() {
    let local = tokio::task::LocalSet::new();
    let (positive, negative) = local
        .run_until(async { (run_case(true).await, run_case(false).await) })
        .await;

    assert_eq!(
        positive,
        ForkContinuityMetric {
            parent_marker_in_child_request: true,
            child_result_returned_to_parent: true,
            spawn_join_recorded: true,
            child_execution_durable: true,
            typed_child_run_ref_in_parent_record: true,
        }
    );
    assert_eq!(
        negative,
        ForkContinuityMetric {
            parent_marker_in_child_request: false,
            child_result_returned_to_parent: true,
            spawn_join_recorded: true,
            child_execution_durable: true,
            typed_child_run_ref_in_parent_record: true,
        }
    );
}

fn sha256(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    format!("{:x}", Sha256::digest(bytes))
}

/// Descriptive review evidence follows option 1 in task-review-contract.md.
/// The fixture is bounded to Git-listed tracked/untracked files; this snapshot
/// does not claim containment of shell effects outside that workspace.
fn review_snapshot(workspace: &std::path::Path, record: &std::path::Path) -> serde_json::Value {
    use review_workflow::git;
    let prefix = std::fs::read(record).unwrap();
    let events = read_run_record(record).unwrap();
    let files: serde_json::Map<String, serde_json::Value> = git(
        workspace,
        &["ls-files", "--cached", "--others", "--exclude-standard"],
    )
    .lines()
    .map(|name| {
        (
            name.to_owned(),
            serde_json::json!(sha256(&std::fs::read(workspace.join(name)).unwrap())),
        )
    })
    .collect();
    serde_json::json!({
        "prefix_sha256": sha256(&prefix),
        "prefix_events": events.len(),
        "last_sequence": events.last().unwrap().sequence,
        "files": files,
        "coverage": "Git-listed tracked and untracked fixture files; external effects excluded",
        "path": workspace,
        "branch": git(workspace, &["branch", "--show-current"]).trim(),
        "head": git(workspace, &["rev-parse", "HEAD"]).trim(),
        "status": git(workspace, &["status", "--porcelain"]),
    })
}

#[tokio::test(flavor = "current_thread")]
#[allow(clippy::too_many_lines)] // Keep the ordered human-review journey visible together.
async fn spawned_child_human_reject_resume_accept_preserves_reviewed_evidence() {
    use review_workflow::*;

    // Freeze the human-review requirement and checker identity before execution.
    let required = "port=9090 integer; debug=false boolean; service=fixture; exact key set; only settings.json modified; parent unchanged";
    let fixture = tempfile::tempdir().unwrap();
    let repo = fixture.path().join("parent");
    let retained = fixture.path().join("retained");
    let config = fixture.path().join("config");
    std::fs::create_dir_all(&repo).unwrap();
    std::fs::create_dir_all(&config).unwrap();
    let original = prepare_worktree(&repo, &retained);
    let initial_sha256 = sha256(original.as_bytes());
    let checker_sha256 = sha256(include_bytes!("support/review_workflow.rs"));
    let parent_id = "spawned-child-human-review";
    let root_record = config.join("piku/runs").join(format!("{parent_id}.jsonl"));
    let registry = TaskRegistry::with_persistence_run_path(
        parent_id,
        config.join("piku/sessions"),
        config.join("piku/runs"),
        config.join("piku/links"),
        Some(root_record.clone()),
    );
    let mut provider = ForkRoutingProvider::new(registry.clone(), true);
    provider.child_file = Some(retained.join("settings.json"));
    let mut parent = piku_runtime::Session::new(parent_id.to_owned());
    let mut capture = CaptureSink::default();
    let mut recorder = RunRecorder::open(&root_record, parent_id).unwrap();
    tokio::task::LocalSet::new()
        .run_until(async {
            let mut sink = RecordingSink::new(&mut capture, &mut recorder, "root-turn");
            let result = run_turn_with_registry(
                &format!("Preserve {PARENT_MARKER}; required outcome: {required}"),
                &mut parent,
                &provider,
                "fixture-model",
                &[],
                piku_tools::all_tool_definitions(),
                &AllowAll,
                &mut sink,
                Some(5),
                None,
                &registry,
                1,
                &[],
                None,
                None,
            )
            .await;
            assert!(result.stream_error.is_none());
            assert!(sink.take_record_error().is_none());
        })
        .await;
    let tasks = registry.all();
    assert_eq!(tasks.len(), 1);
    let task = &tasks[0];
    assert_eq!(task.status, piku_runtime::TaskStatus::Done);
    let evidence = task.evidence.as_ref().unwrap();
    let child_id = evidence.child_session_id.clone();
    let record_path = evidence.run_record_path.clone();
    let parent_events = read_run_record(&root_record).unwrap();
    assert!(typed_child_run_ref_present(&parent_events, &registry));
    let references: Vec<_> = parent_events
        .iter()
        .filter_map(|event| match &event.event {
            RunEvent::ChildRunRef {
                reference_root,
                run_record_ref,
                session_ref,
                task_id,
                ..
            } => {
                assert_eq!(
                    *reference_root,
                    Some(piku_runtime::ChildReferenceRoot::ConfiguredEvidenceRoot)
                );
                assert_eq!(*task_id, task.id.to_string());
                Some((run_record_ref, session_ref))
            }
            _ => None,
        })
        .collect();
    assert_eq!(references.len(), 1);
    let (run_ref, session_ref) = references[0];
    let evidence_root = config.join("piku");
    assert_eq!(evidence_root.join(run_ref), evidence.run_record_path);
    assert_eq!(evidence_root.join(session_ref), evidence.session_path);
    assert!(piku::run_view::render_text(&parent_events).contains("basis configured_evidence_root"));
    assert!(child_execution_is_durable(&registry, parent_id));
    assert!(recorded_tool_pair(
        &parent_events,
        "agent_join",
        "root-join"
    ));
    assert!(capture.text.contains(CHILD_MARKER));
    assert!(
        !workspace_accepted(&retained),
        "Done must not establish task success"
    );
    let failed: serde_json::Value =
        serde_json::from_slice(&std::fs::read(retained.join("settings.json")).unwrap()).unwrap();
    assert_eq!(
        failed,
        serde_json::json!({"port":9090,"debug":true,"service":"fixture"})
    );
    let (url, requests, server) = scripted_server(&retained.join("settings.json"), true);
    let inspect = checked(cli(&config, &retained, &url).args(["inspect", &child_id]));
    assert!(String::from_utf8_lossy(&inspect.stdout).contains(CHILD_MARKER));
    let first_prefix = std::fs::read_to_string(&record_path).unwrap();
    let rejected_snapshot = review_snapshot(&retained, &record_path);
    let provenance = serde_json::json!({
        "task": task.id.to_string(), "parent": parent_id, "child": child_id,
        "reviewer": "fixture-human-reviewer", "owner": "fixture-retained-work-owner",
        "required": required, "checker_sha256": checker_sha256,
        "initial_sha256": initial_sha256,
    });
    let rejection = serde_json::json!({"provenance": provenance, "snapshot": rejected_snapshot,
        "result": "failed: debug=true", "disposition": "retain for explicit correction"})
    .to_string();
    checked(cli(&config, &retained, &url).args([
        "conclude",
        &child_id,
        "--status",
        "needs-work",
        "--note",
        &rejection,
    ]));
    let rejected = std::fs::read_to_string(&record_path).unwrap();
    assert!(rejected.starts_with(&first_prefix));
    assert_eq!(
        sha256(&std::fs::read(retained.join("settings.json")).unwrap()),
        rejected_snapshot["files"]["settings.json"]
            .as_str()
            .unwrap()
    );
    let link_bytes = std::fs::read(&evidence.link_path).unwrap();
    let link_path = evidence.link_path.clone();
    let task_id = task.id.clone();
    drop(provider);
    drop(registry);
    drop(recorder);
    // Fresh registry intentionally does not reconstruct old live task state.
    let reopened_registry = TaskRegistry::with_persistence_run_path(
        parent_id,
        config.join("piku/sessions"),
        config.join("piku/runs"),
        config.join("piku/links"),
        Some(root_record.clone()),
    );
    assert!(tokio::time::timeout(
        std::time::Duration::from_secs(1),
        reopened_registry.wait_for(&task_id),
    )
    .await
    .unwrap()
    .is_err());
    assert!(piku_runtime::Session::load(
        &config
            .join("piku/sessions")
            .join(format!("{child_id}.json"))
    )
    .is_ok());
    assert_eq!(read_run_record(&root_record).unwrap(), parent_events);
    let correction = format!("Independent human review found debug=true. Correct debug to false in settings.json; preserve port=9090 and service=fixture. Review references: {rejection}");
    checked(cli(&config, &retained, &url).args(["--resume", &child_id, "-p", &correction]));
    server.join().unwrap();
    let captured = requests.lock().unwrap();
    assert_eq!(captured.len(), 2);
    let resumed = captured[0]["messages"].to_string();
    assert!(resumed.contains(CHILD_TASK));
    assert!(resumed.contains("partial-write"));
    assert!(resumed.contains(&checker_sha256));
    assert!(resumed.contains("Independent human review found debug=true"));
    assert!(workspace_accepted(&retained));
    assert_eq!(
        std::fs::read_to_string(repo.join("settings.json")).unwrap(),
        original
    );
    assert!(git(&repo, &["status", "--porcelain"]).is_empty());
    let accepted_snapshot = review_snapshot(&retained, &record_path);
    assert_ne!(accepted_snapshot["files"], rejected_snapshot["files"]);
    assert_ne!(
        accepted_snapshot["prefix_sha256"],
        rejected_snapshot["prefix_sha256"]
    );
    let acceptance = serde_json::json!({"provenance": provenance, "snapshot": accepted_snapshot,
        "result": "passed: exact JSON keys/types/values; only settings.json changed; parent unchanged",
        "disposition": "retain unmerged work; integration and cleanup separate"}).to_string();
    let before_acceptance = std::fs::read_to_string(&record_path).unwrap();
    checked(cli(&config, &retained, &url).args([
        "conclude",
        &child_id,
        "--status",
        "accepted",
        "--note",
        &acceptance,
    ]));
    assert_review_history(&record_path, &rejected, &child_id, &rejection, &acceptance);
    assert!(std::fs::read_to_string(&record_path)
        .unwrap()
        .starts_with(&before_acceptance));
    let final_inspect =
        checked(cli(&config, &retained, &url).args(["inspect", &child_id, "--json"]));
    let text = String::from_utf8_lossy(&final_inspect.stdout);
    assert!(text.contains(&checker_sha256));
    assert!(text.contains(accepted_snapshot["prefix_sha256"].as_str().unwrap()));
    assert_eq!(std::fs::read(link_path).unwrap(), link_bytes);
    assert_eq!(
        git(&retained, &["branch", "--show-current"]),
        "recovery-fixture\n"
    );
    assert!(
        workspace_accepted(&retained),
        "Accepted must retain unmerged work"
    );
}
