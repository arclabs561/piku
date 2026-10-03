//! Cross-language admission checks for CLI, TUI, and web evaluation packets.

#[path = "agentic/surface_contract.rs"]
mod surface_contract;

use serde_json::{json, Value};
use surface_contract::{Surface, SurfacePacket};

const SCHEMA: &str = include_str!("../../../eval/surface-evaluation.schema.json");
fn packet(surface: &str) -> Value {
    let surface = match surface {
        "cli" => Surface::Cli,
        "tui" => Surface::Tui,
        "web" => Surface::Web,
        _ => panic!("unknown surface fixture: {surface}"),
    };
    let packet = SurfacePacket::output_recovery_room(surface);
    packet.validate().expect("typed packet is valid");
    serde_json::to_value(packet).expect("typed packet serializes")
}

fn validator() -> jsonschema::Validator {
    let schema = serde_json::from_str(SCHEMA).expect("surface evaluation schema is JSON");
    jsonschema::validator_for(&schema).expect("surface evaluation schema is valid Draft 2020-12")
}

#[test]
fn schema_accepts_the_shared_scenario_for_every_surface() {
    let validator = validator();
    for surface in ["cli", "tui", "web"] {
        assert!(
            validator.is_valid(&packet(surface)),
            "{surface} packet should be valid"
        );
    }
}

#[test]
fn schema_rejects_mismatched_observation_or_missing_surface_evidence() {
    let validator = validator();
    let mut wrong_observation = packet("tui");
    wrong_observation["observation"] = json!("bounded_subprocess");
    assert!(!validator.is_valid(&wrong_observation));

    let mut missing_evidence = packet("web");
    missing_evidence["evidence_kinds"] = json!(["browser_events", "screenshots", "run_record"]);
    assert!(!validator.is_valid(&missing_evidence));
}

#[test]
fn schema_rejects_unproven_scenario_claims_and_unknown_fields() {
    let validator = validator();
    let mut missing_claim = packet("cli");
    missing_claim["claims"] = json!(["output_visible"]);
    assert!(!validator.is_valid(&missing_claim));

    let mut extra_field = packet("cli");
    extra_field["judge_notes"] = json!("not part of the public evidence shape");
    assert!(!validator.is_valid(&extra_field));
}

#[test]
fn recording_metadata_requires_a_reason_when_evidence_is_incomplete_or_uninspectable() {
    let validator = validator();
    let mut incomplete = packet("tui");
    incomplete["recording"] = json!({
        "status": "incomplete",
        "inspectability": "inspectable",
        "source_artifacts": [{ "ref": "cast:failed", "retention": "local_private" }]
    });
    assert!(!validator.is_valid(&incomplete));

    incomplete["recording"]["reason"] = json!("cast flush failed");
    assert!(validator.is_valid(&incomplete));

    let mut uninspectable = packet("web");
    uninspectable["recording"] = json!({
        "status": "complete",
        "inspectability": "uninspectable",
        "source_artifacts": [{ "ref": "screen:missing", "retention": "ephemeral" }]
    });
    assert!(!validator.is_valid(&uninspectable));
    uninspectable["recording"]["reason"] = json!("remote artifact expired");
    assert!(validator.is_valid(&uninspectable));
}

#[test]
fn replay_timeline_schema_keeps_surface_payloads_and_annotations_aligned() {
    let schema = jsonschema::validator_for(
        &serde_json::from_str::<serde_json::Value>(include_str!(
            "../../../eval/evaluation-timeline.schema.json"
        ))
        .unwrap(),
    )
    .unwrap();
    for (surface, kind) in [
        ("cli", "process"),
        ("tui", "asciicast"),
        ("web", "playwright"),
    ] {
        let packet = serde_json::json!({
            "schema_version": 1,
            "surface": surface,
            "replay": {
                "kind": kind,
                "artifacts": ["evidence"],
                "recording": {
                    "status": "complete",
                    "inspectability": "inspectable",
                    "source_artifacts": [{ "ref": "evidence", "retention": "durable" }]
                }
            },
            "annotations": [{
                "t_ms": 0,
                "kind": "observation",
                "ref": "evidence",
                "summary": "initial state",
                "evidence_ids": ["e1"]
            }]
        });
        assert!(schema.is_valid(&packet), "{surface} packet should validate");
    }
}

#[test]
fn timeline_schema_records_incomplete_replay_without_admitting_a_silent_success() {
    let schema = jsonschema::validator_for(
        &serde_json::from_str::<serde_json::Value>(include_str!(
            "../../../eval/evaluation-timeline.schema.json"
        ))
        .unwrap(),
    )
    .unwrap();
    let mut packet = serde_json::json!({
        "schema_version": 1,
        "surface": "tui",
        "replay": {
            "kind": "asciicast",
            "artifacts": ["cast:partial"],
            "recording": {
                "status": "incomplete",
                "inspectability": "uninspectable",
                "source_artifacts": [{ "ref": "cast:partial", "retention": "ephemeral" }]
            }
        },
        "annotations": []
    });
    assert!(!schema.is_valid(&packet));
    packet["replay"]["recording"]["reason"] = json!("cast flush failed");
    assert!(schema.is_valid(&packet));
}
