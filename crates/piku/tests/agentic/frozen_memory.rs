//! Composition coverage for existing scenario evidence and frozen prompt assets.
//! This does not enforce a learning split, resource policy, or model capability.

use super::{tests::prompt_manifest, AttestedValue, PlaygroundLedger};
use crate::scenario::{self, Clause, Outcome, Scenario, Verification};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::fs;

#[test]
fn frozen_memory_preserves_scenario_evidence_through_asset_and_manifest() {
    let directory = tempfile::tempdir().unwrap();
    let scenario = Scenario {
        id: "acquisition-output",
        contexts: &["synthetic", "deterministic"],
        clauses: &[Clause {
            text: "The output file contains the required value.",
            check: Some(Verification::CommandSucceeds {
                program: "sh",
                args: &[
                    "-c",
                    "cat result.txt; test \"$(cat result.txt)\" = observed",
                ],
                timeout_secs: 2,
            }),
        }],
    };
    let mut observations = Vec::new();
    for (attempt, contents, expected) in [
        ("attempt-1", "incorrect\n", Outcome::Failed),
        ("attempt-2", "observed\n", Outcome::Passed),
    ] {
        fs::write(directory.path().join("result.txt"), contents).unwrap();
        let results = scenario::verify(scenario, directory.path());
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].outcome, expected);
        assert_eq!(results[0].evidence, contents);
        observations.push(json!({
            "attempt": attempt,
            "check": results[0].label,
            "outcome": results[0].outcome.label(),
            "evidence": results[0].evidence,
        }));
    }
    let ledger = PlaygroundLedger::open_at(directory.path().join("ledger.jsonl")).unwrap();
    let captured = json!({"scenario_id": scenario.id, "observations": observations});
    let payload = serde_json::to_string(&captured).unwrap();
    let asset = ledger
        .write_prompt_asset("subject", "acquisition", &payload)
        .unwrap();
    let asset_path = asset.path.clone();
    let asset_digest = format!("{:x}", Sha256::digest(payload.as_bytes()));
    assert_eq!(asset.sha256, asset_digest);
    assert_eq!(asset.size_bytes, payload.len());
    assert_eq!(fs::read(&asset_path).unwrap(), payload.as_bytes());
    assert_eq!(
        ledger
            .write_prompt_asset("subject", "acquisition", "replacement")
            .unwrap_err()
            .kind(),
        std::io::ErrorKind::AlreadyExists,
    );
    assert_eq!(fs::read(&asset_path).unwrap(), payload.as_bytes());

    let provenance = json!({"scenario_id": scenario.id, "source_refs": [asset_path]});
    let mut manifest = prompt_manifest(ledger.run_id(), "unused fixture asset");
    manifest.roles[0].role = "subject";
    manifest.roles[0].prompt_assets = vec![asset];
    manifest.roles[0].context_contract = AttestedValue::new(provenance.clone());
    let persisted = ledger.write_prompt_manifest(&manifest).unwrap();
    let bytes = fs::read(&persisted.path).unwrap();
    assert_eq!(persisted.sha256, format!("{:x}", Sha256::digest(&bytes)));
    let readback: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(readback["run_id"], ledger.run_id());
    let role = &readback["roles"][0];
    assert_eq!(role["role"], "subject");
    assert_eq!(role["context_contract"]["value"], provenance);
    assert_eq!(
        role["context_contract"]["sha256"],
        format!(
            "{:x}",
            Sha256::digest(serde_json::to_vec(&provenance).unwrap())
        ),
    );
    let refs = role["prompt_assets"].as_array().unwrap();
    assert_eq!(refs.len(), 1);
    assert_eq!(refs[0]["path"], asset_path);
    assert_eq!(refs[0]["sha256"], asset_digest);
    assert_eq!(refs[0]["size_bytes"], payload.len());
    let source = fs::read(refs[0]["path"].as_str().unwrap()).unwrap();
    assert_eq!(format!("{:x}", Sha256::digest(&source)), asset_digest);
    let evidence: Value = serde_json::from_slice(&source).unwrap();
    assert_eq!(evidence, captured);
    assert_eq!(evidence["observations"][0]["outcome"], "fail");
    assert_eq!(evidence["observations"][1]["outcome"], "pass");
    assert_eq!(evidence["observations"][0]["evidence"], "incorrect\n");
    assert_eq!(evidence["observations"][1]["evidence"], "observed\n");

    manifest.roles[0].context_contract = AttestedValue::new(json!({"source_refs": []}));
    assert_eq!(
        ledger.write_prompt_manifest(&manifest).unwrap_err().kind(),
        std::io::ErrorKind::AlreadyExists,
    );
    assert_eq!(fs::read(&persisted.path).unwrap(), bytes);
}
