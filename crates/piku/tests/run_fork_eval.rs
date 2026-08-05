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

    fn on_tool_end(&mut self, _tool_name: &str, _result: &str, _is_error: bool) -> PostToolAction {
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
}

impl ForkRoutingProvider {
    fn new(registry: TaskRegistry, fork: bool) -> Self {
        Self {
            registry,
            fork,
            observations: Arc::new(Mutex::new(ProviderObservations::default())),
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
            text_response(CHILD_MARKER)
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
            RequestContent::ToolUse { .. } => None,
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
    let registry = TaskRegistry::with_persistence(
        &parent_session_id,
        directory.path().join("sessions"),
        directory.path().join("runs"),
        directory.path().join("links"),
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

    ForkContinuityMetric {
        parent_marker_in_child_request: child_request.contains(PARENT_MARKER),
        child_result_returned_to_parent: join_returned_child_marker
            && parent_confirmed_child_marker,
        spawn_join_recorded: recorded_tool_pair(&events, "spawn_agent", "root-spawn")
            && recorded_tool_pair(&events, "agent_join", "root-join"),
        child_execution_durable,
    }
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
        }
    );
    assert_eq!(
        negative,
        ForkContinuityMetric {
            parent_marker_in_child_request: false,
            child_result_returned_to_parent: true,
            spawn_join_recorded: true,
            child_execution_durable: true,
        }
    );
}
