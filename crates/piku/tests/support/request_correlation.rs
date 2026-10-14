//! Read production receipt bindings against the independently persisted record.

use std::collections::{HashMap, HashSet};

use serde_json::Value;

pub(super) fn assert_terminal_correlation(logs: &[String], record: &str, session_id: &str) {
    let events: Vec<Value> = record
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    let contexts: HashMap<u64, &Value> = events
        .iter()
        .filter(|event| event["event"] == "context_built")
        .map(|event| (event["sequence"].as_u64().unwrap(), event))
        .collect();
    assert_eq!(
        contexts.len(),
        4,
        "fixture prepared four actual foreground requests"
    );
    let mut namespaces = HashSet::new();
    let mut referenced = HashSet::new();
    for log in logs {
        let rows: Vec<Value> = log
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        let namespace = rows[0]["namespace"].as_str().unwrap();
        assert!(namespaces.insert(namespace.to_owned()));
        let mut foreground = 0;
        let mut exit_memory = 0;
        for start in rows.iter().filter(|row| row["event"] == "started") {
            let binding = &start["binding"];
            assert_eq!(binding["run_id"], session_id);
            assert!(binding["task_id"].is_null());
            assert!(binding["parent_origin"].is_null());
            let end = rows
                .iter()
                .find(|row| {
                    row["event"] == "finished"
                        && row["attempt_id"] == start["attempt_id"]
                        && row["namespace"] == start["namespace"]
                })
                .unwrap();
            assert_eq!(
                end["binding"], *binding,
                "receipt binding changed at finish"
            );
            assert_eq!(end["provider"], start["provider"]);
            assert!(end["reported_usage"].is_null());
            match binding["purpose"].as_str().unwrap() {
                "agent_turn" => {
                    foreground += 1;
                    let sequence = binding["context_sequence"].as_u64().unwrap();
                    assert!(
                        referenced.insert(sequence),
                        "fixture reused stale ContextBuilt"
                    );
                    let context = contexts.get(&sequence).unwrap();
                    assert_eq!(context["session_id"], binding["run_id"]);
                    assert_eq!(context["turn_id"], binding["turn_id"]);
                    assert_eq!(context["scope"], "turn");
                    assert!(context["manifest"]["estimated_input_tokens"]
                        .as_u64()
                        .is_some());
                }
                "exit_memory" => {
                    exit_memory += 1;
                    assert!(
                        binding["turn_id"].is_null(),
                        "exit memory carried a turn reference"
                    );
                    assert!(binding["context_sequence"].is_null());
                }
                other => panic!("unexpected fixture purpose: {other}"),
            }
            let mut keys: Vec<_> = binding
                .as_object()
                .unwrap()
                .keys()
                .map(String::as_str)
                .collect();
            keys.sort_unstable();
            assert_eq!(
                keys,
                [
                    "context_sequence",
                    "parent_origin",
                    "purpose",
                    "run_id",
                    "task_id",
                    "turn_id"
                ]
            );
        }
        assert_eq!(foreground, 2);
        assert_eq!(exit_memory, 1);
        let cutoff = rows.last().unwrap();
        assert_eq!(cutoff["producers_unsettled"], true);
        assert_eq!(cutoff["queue_full"], 0);
        assert_eq!(cutoff["queue_closed"], 0);
    }
    // The frozen fixture's independent record, not arrival order or clocks,
    // establishes these exact semantic references across fresh launch namespaces.
    assert_eq!(referenced.len(), contexts.len());
}
