//! Controlled comparison of Piku's run-record projections.
//!
//! These measurements are retrieval and attention-load proxies. They do not
//! measure comprehension, preference, or usability, and they are deliberately
//! not collapsed into a single score.

use piku::run_view::{
    build_search_index_with_artifacts, render_html_with_artifacts, render_text, RunSearchEntry,
};
use piku_runtime::{
    read_run_record, ContextManifest, RunContentRef, RunDisposition, RunEvent,
    RunPermissionDecision, RunRecorder, UsageRecord, RUN_INLINE_CONTENT_LIMIT_BYTES,
};
use std::collections::BTreeSet;

#[derive(Debug, Clone, PartialEq, Eq)]
struct RetrievalMetric {
    target_found: bool,
    candidate_events: usize,
    false_positives: usize,
    target_rank: Option<usize>,
    chars_exposed_after_query: usize,
    navigation_actions_to_full_content: Option<usize>,
    provenance: ProvenanceMetric,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct ProvenanceMetric {
    present: BTreeSet<ProvenanceField>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum ProvenanceField {
    Sequence,
    Scope,
    EventKind,
    StorageRef,
    ByteCount,
}

fn inline(text: &str) -> RunContentRef {
    RunContentRef::Inline {
        text: text.to_string(),
    }
}

fn build_fixture(directory: &std::path::Path) -> std::path::PathBuf {
    let path = directory.join("surface-eval.jsonl");
    let mut recorder = RunRecorder::open(&path, "surface-eval").unwrap();
    recorder
        .append(
            "turn-0",
            RunEvent::TurnStarted {
                provider: Some("fixture".to_string()),
                model: "fixture-model".to_string(),
                input: inline("find the evidence and preserve its provenance"),
            },
        )
        .unwrap();
    recorder
        .append(
            "turn-0",
            RunEvent::ContextBuilt {
                manifest: ContextManifest {
                    model: "fixture-model".to_string(),
                    context_window_tokens: 32_000,
                    estimated_input_tokens: 20,
                    system_sections: Vec::new(),
                    messages: Vec::new(),
                    tools: Vec::new(),
                },
            },
        )
        .unwrap();
    recorder
        .append(
            "turn-0",
            RunEvent::ToolStarted {
                tool_call_id: "call-1".to_string(),
                name: "evidence_probe".to_string(),
                arguments: serde_json::json!({"path": "notes.txt"}),
            },
        )
        .unwrap();
    recorder
        .append(
            "turn-0",
            RunEvent::PermissionDecision {
                tool_call_id: "call-1".to_string(),
                decision: RunPermissionDecision::Allowed,
            },
        )
        .unwrap();
    recorder
        .append(
            "turn-0",
            RunEvent::ToolCompleted {
                tool_call_id: "call-1".to_string(),
                result: inline("ordinary distractor evidence"),
                is_error: false,
                effects: Vec::new(),
                verification: None,
            },
        )
        .unwrap();
    let deep_marker = "ultraviolet-otter-7319";
    let large = format!(
        "{}\n{deep_marker}\n{}",
        "distractor ".repeat(RUN_INLINE_CONTENT_LIMIT_BYTES / 11 + 64),
        "more distractor ".repeat(64)
    );
    recorder
        .append(
            "turn-0",
            RunEvent::AssistantMessage {
                content: inline(&large),
            },
        )
        .unwrap();
    recorder
        .append(
            "turn-0",
            RunEvent::TurnCompleted {
                usage: UsageRecord {
                    input_tokens: 20,
                    output_tokens: 10,
                },
                stop_reason: Some("end_turn".to_string()),
            },
        )
        .unwrap();
    recorder
        .append_run(RunEvent::UserDisposition {
            disposition: RunDisposition::NeedsWork,
            note: inline("confirm the deep artifact before accepting"),
        })
        .unwrap();
    path
}

fn browser_query(entries: &[RunSearchEntry], query: &str, target_kind: &str) -> RetrievalMetric {
    let hits = entries
        .iter()
        .filter(|entry| entry.search_text.contains(query))
        .collect::<Vec<_>>();
    let target = hits
        .iter()
        .position(|entry| entry.event_kind == target_kind);
    let provenance = target.map_or_else(ProvenanceMetric::default, |index| {
        let entry = hits[index];
        ProvenanceMetric {
            present: [
                Some(ProvenanceField::Sequence),
                (!entry.scope.is_empty()).then_some(ProvenanceField::Scope),
                (!entry.event_kind.is_empty()).then_some(ProvenanceField::EventKind),
                entry
                    .storage_ref
                    .is_some()
                    .then_some(ProvenanceField::StorageRef),
                entry
                    .byte_count
                    .is_some()
                    .then_some(ProvenanceField::ByteCount),
            ]
            .into_iter()
            .flatten()
            .collect(),
        }
    });
    RetrievalMetric {
        target_found: target.is_some(),
        candidate_events: hits.len(),
        false_positives: hits.len().saturating_sub(usize::from(target.is_some())),
        target_rank: target.map(|index| index + 1),
        chars_exposed_after_query: hits
            .iter()
            .map(|entry| entry.preview_text.chars().count())
            .sum(),
        navigation_actions_to_full_content: target.map(|_| 1),
        provenance,
    }
}

fn assert_status_retrieval(compact: &str, html: &str) {
    let status_line = compact
        .lines()
        .find(|line| line.contains("evidence complete"))
        .unwrap();
    let status_terminal = RetrievalMetric {
        target_found: true,
        candidate_events: 1,
        false_positives: 0,
        target_rank: Some(1),
        chars_exposed_after_query: status_line.chars().count(),
        navigation_actions_to_full_content: Some(0),
        provenance: ProvenanceMetric::default(),
    };
    let status_browser = RetrievalMetric {
        chars_exposed_after_query: "complete".chars().count(),
        ..status_terminal.clone()
    };
    assert!(html.contains("drawAudit"));
    assert!(html.contains("searchBySequence"));
    assert!(status_terminal.target_found && status_browser.target_found);
}

fn assert_metadata_retrieval(compact: &str, search_index: &[RunSearchEntry]) {
    let metadata_lines = compact
        .lines()
        .filter(|line| line.to_lowercase().contains("evidence_probe"))
        .collect::<Vec<_>>();
    let metadata_terminal = RetrievalMetric {
        target_found: metadata_lines.len() == 1,
        candidate_events: metadata_lines.len(),
        false_positives: metadata_lines.len().saturating_sub(1),
        target_rank: (metadata_lines.len() == 1).then_some(1),
        chars_exposed_after_query: metadata_lines.iter().map(|line| line.chars().count()).sum(),
        navigation_actions_to_full_content: Some(0),
        provenance: ProvenanceMetric {
            present: [ProvenanceField::Scope, ProvenanceField::EventKind]
                .into_iter()
                .collect(),
        },
    };
    let metadata_browser = browser_query(search_index, "evidence_probe", "tool_started");
    assert!(metadata_terminal.target_found);
    assert!(metadata_browser.target_found);
    assert_eq!(metadata_browser.candidate_events, 1);
    assert_eq!(metadata_browser.false_positives, 0);
    assert_eq!(metadata_browser.target_rank, Some(1));
    assert!(metadata_browser
        .provenance
        .present
        .contains(&ProvenanceField::Sequence));
    assert!(metadata_browser
        .provenance
        .present
        .contains(&ProvenanceField::Scope));
    assert!(metadata_browser
        .provenance
        .present
        .contains(&ProvenanceField::EventKind));
    assert!(!metadata_browser
        .provenance
        .present
        .contains(&ProvenanceField::StorageRef));
    assert!(!metadata_browser
        .provenance
        .present
        .contains(&ProvenanceField::ByteCount));
}

fn assert_artifact_retrieval(compact: &str, search_index: &[RunSearchEntry]) {
    let query = "ultraviolet-otter-7319";
    let compact_artifact = RetrievalMetric {
        target_found: compact.contains(query),
        candidate_events: usize::from(compact.contains(query)),
        false_positives: 0,
        target_rank: compact.contains(query).then_some(1),
        chars_exposed_after_query: compact.chars().count(),
        navigation_actions_to_full_content: None,
        provenance: ProvenanceMetric::default(),
    };
    let browser_artifact = browser_query(search_index, query, "assistant_message");
    let composed_target = search_index
        .iter()
        .find(|entry| entry.search_text.contains(query))
        .unwrap();
    let terminal_composed = RetrievalMetric {
        target_found: true,
        candidate_events: 1,
        false_positives: 0,
        target_rank: Some(1),
        chars_exposed_after_query: composed_target.full_content_chars,
        navigation_actions_to_full_content: Some(2),
        provenance: browser_artifact.provenance.clone(),
    };

    assert!(!compact_artifact.target_found);
    assert!(terminal_composed.target_found);
    assert!(browser_artifact.target_found);
    assert_eq!(browser_artifact.candidate_events, 1);
    assert_eq!(browser_artifact.false_positives, 0);
    assert_eq!(browser_artifact.target_rank, Some(1));
    assert_eq!(browser_artifact.navigation_actions_to_full_content, Some(1));
    assert_eq!(
        terminal_composed.navigation_actions_to_full_content,
        Some(2)
    );
    assert_eq!(
        browser_artifact.provenance,
        ProvenanceMetric {
            present: [
                ProvenanceField::Sequence,
                ProvenanceField::Scope,
                ProvenanceField::EventKind,
                ProvenanceField::StorageRef,
                ProvenanceField::ByteCount,
            ]
            .into_iter()
            .collect(),
        }
    );
}

#[test]
fn projection_metrics_compare_capabilities_without_declaring_a_winner() {
    let directory = tempfile::tempdir().unwrap();
    let record_path = build_fixture(directory.path());
    let events = read_run_record(&record_path).unwrap();
    let compact = render_text(&events);
    let html = render_html_with_artifacts(&events, &record_path).unwrap();
    let search_index = build_search_index_with_artifacts(&events, &record_path).unwrap();

    assert_status_retrieval(&compact, &html);
    assert_metadata_retrieval(&compact, &search_index);
    assert_artifact_retrieval(&compact, &search_index);
}
