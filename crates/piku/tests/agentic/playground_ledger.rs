//! Append-only evidence records for opt-in terminal playground runs.

use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::Serialize;

/// A reviewable terminal-playground turn. Provider labels deliberately exclude
/// credentials; terminal text is bounded by the caller before persistence.
#[derive(Serialize)]
pub struct TurnRecord<'a> {
    pub schema_version: u8,
    pub kind: &'static str,
    pub run_id: &'a str,
    pub timestamp_secs: u64,
    pub persona: &'a str,
    pub phase: &'a str,
    pub turn: usize,
    pub user_agent_provider: &'a str,
    pub user_agent_model: &'a str,
    pub piku_provider: &'a str,
    pub piku_model: &'a str,
    pub action: &'a str,
    pub viewport: &'a str,
    pub workspace_diff: &'a str,
    pub observations: &'a [String],
    pub bugs: &'a [String],
    pub deterministic_findings: &'a [String],
}

#[derive(Serialize)]
pub struct ReviewRecord<'a> {
    pub schema_version: u8,
    pub kind: &'static str,
    pub run_id: &'a str,
    pub timestamp_secs: u64,
    pub persona: &'a str,
    pub review: &'a str,
}

/// Resolved model drivers for one playground session. This is evidence, not a
/// credential record: provider labels and model identifiers only.
#[derive(Serialize)]
pub struct ConfigRecord<'a> {
    pub schema_version: u8,
    pub kind: &'static str,
    pub run_id: &'a str,
    pub timestamp_secs: u64,
    pub user_agent_provider: &'a str,
    pub user_agent_model: &'a str,
    pub judge_provider: &'a str,
    pub judge_model: &'a str,
    pub piku_provider: &'a str,
    pub piku_model: &'a str,
    pub user_agent_client: &'static str,
    pub judge_client: &'static str,
}

/// A bounded second-order review of the judge and the observed piku behavior.
#[derive(Serialize)]
pub struct ObserverRecord<'a> {
    pub schema_version: u8,
    pub kind: &'static str,
    pub run_id: &'a str,
    pub timestamp_secs: u64,
    pub persona: &'a str,
    pub judge_observations: &'a [String],
    pub piku_observations: &'a [String],
    pub verdict: &'a str,
    pub primary_review_grounded: bool,
}

/// The bridge from evaluation evidence to an engineering action. Verified
/// findings are safe to act on; hypotheses must first be reproduced.
#[derive(Serialize)]
pub struct ImprovementHandoffRecord<'a> {
    pub schema_version: u8,
    pub kind: &'static str,
    pub run_id: &'a str,
    pub timestamp_secs: u64,
    pub persona: &'a str,
    pub verified_findings: &'a [String],
    pub hypotheses: &'a [String],
    pub next_action: &'static str,
}

pub struct PlaygroundLedger {
    path: PathBuf,
    run_id: String,
}

impl PlaygroundLedger {
    pub fn open() -> std::io::Result<Self> {
        let path = std::env::var("PIKU_AGENTIC_LEDGER").map_or_else(
            |_| {
                PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .parent()
                    .and_then(|path| path.parent())
                    .expect("piku crate has workspace root")
                    .join("target/agentic-findings/playground.jsonl")
            },
            PathBuf::from,
        );
        Self::open_at(path)
    }

    fn open_at(path: PathBuf) -> std::io::Result<Self> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        Ok(Self {
            path,
            run_id: format!("{}-{}", now_secs(), std::process::id()),
        })
    }

    #[must_use]
    pub fn run_id(&self) -> &str {
        &self.run_id
    }

    pub fn append_turn(&self, record: &TurnRecord<'_>) -> std::io::Result<()> {
        self.append(record)
    }

    pub fn append_review(&self, record: &ReviewRecord<'_>) -> std::io::Result<()> {
        self.append(record)
    }

    pub fn append_config(&self, record: &ConfigRecord<'_>) -> std::io::Result<()> {
        self.append(record)
    }

    pub fn append_observer(&self, record: &ObserverRecord<'_>) -> std::io::Result<()> {
        self.append(record)
    }

    pub fn append_improvement_handoff(
        &self,
        record: &ImprovementHandoffRecord<'_>,
    ) -> std::io::Result<()> {
        self.append(record)
    }

    fn append<T: Serialize>(&self, record: &T) -> std::io::Result<()> {
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)?;
        serde_json::to_writer(&mut file, record)?;
        writeln!(file)
    }
}

#[must_use]
pub fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn writes_jsonl_turn_config_and_observer_records() {
        let directory =
            std::env::temp_dir().join(format!("piku-playground-ledger-{}", std::process::id()));
        let path = directory.join("ledger.jsonl");
        std::fs::create_dir_all(&directory).unwrap();
        let ledger = PlaygroundLedger::open_at(path.clone()).unwrap();
        let observations = vec!["response is readable".to_string()];
        ledger
            .append_turn(&TurnRecord {
                schema_version: 1,
                kind: "turn",
                run_id: ledger.run_id(),
                timestamp_secs: now_secs(),
                persona: "tester",
                phase: "startup",
                turn: 1,
                user_agent_provider: "ollama",
                user_agent_model: "test",
                piku_provider: "ollama",
                piku_model: "test",
                action: "Observe",
                viewport: "❯",
                workspace_diff: "no changes",
                observations: &observations,
                bugs: &[],
                deterministic_findings: &[],
            })
            .unwrap();
        ledger
            .append_config(&ConfigRecord {
                schema_version: 1,
                kind: "config",
                run_id: ledger.run_id(),
                timestamp_secs: 1,
                user_agent_provider: "ollama",
                user_agent_model: "test",
                judge_provider: "openrouter",
                judge_model: "test-judge",
                piku_provider: "ollama",
                piku_model: "test",
                user_agent_client: "direct-https/reqwest",
                judge_client: "direct-https/reqwest",
            })
            .unwrap();
        let judge_observations = vec!["the primary review is grounded".to_string()];
        let piku_observations = vec!["the prompt stayed visible".to_string()];
        ledger
            .append_observer(&ObserverRecord {
                schema_version: 1,
                kind: "recursive_observer",
                run_id: ledger.run_id(),
                timestamp_secs: 1,
                persona: "tester",
                judge_observations: &judge_observations,
                piku_observations: &piku_observations,
                verdict: "keep",
                primary_review_grounded: true,
            })
            .unwrap();
        let verified = vec!["a deterministic check failed".to_string()];
        let hypotheses = vec!["the agent may have missed a regression".to_string()];
        ledger
            .append_improvement_handoff(&ImprovementHandoffRecord {
                schema_version: 1,
                kind: "improvement_handoff",
                run_id: ledger.run_id(),
                timestamp_secs: 1,
                persona: "tester",
                verified_findings: &verified,
                hypotheses: &hypotheses,
                next_action: "reproduce_verified_findings_then_fix",
            })
            .unwrap();
        let content = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<_> = content.lines().collect();
        assert_eq!(lines.len(), 4);
        let record: serde_json::Value = serde_json::from_str(lines[0]).unwrap();
        assert_eq!(record["kind"], "turn");
        assert_eq!(record["observations"][0], "response is readable");
        let config: serde_json::Value = serde_json::from_str(lines[1]).unwrap();
        assert_eq!(config["kind"], "config");
        assert_eq!(config["judge_model"], "test-judge");
        let observer: serde_json::Value = serde_json::from_str(lines[2]).unwrap();
        assert_eq!(observer["kind"], "recursive_observer");
        assert_eq!(observer["verdict"], "keep");
        let handoff: serde_json::Value = serde_json::from_str(lines[3]).unwrap();
        assert_eq!(handoff["kind"], "improvement_handoff");
        assert_eq!(
            handoff["next_action"],
            "reproduce_verified_findings_then_fix"
        );
        std::fs::remove_dir_all(directory).unwrap();
    }
}
