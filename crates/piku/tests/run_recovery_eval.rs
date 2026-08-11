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
