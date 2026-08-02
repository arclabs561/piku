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
    /// Permission prompts observed in this turn and the harness response. This
    /// distinguishes an explicit approval from a silently unguarded tool call.
    pub permission_events: &'a [String],
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
    pub model_selection_seed: Option<&'a str>,
    pub user_agent_client: &'static str,
    pub judge_client: &'static str,
    /// `control` (pinned models and seed, comparable across builds),
    /// `discovery` (randomized, finds new failure shapes but is not a
    /// baseline), or `adhoc`. Comparing a randomized run to a randomized run
    /// measures the sample, not the change.
    pub run_role: &'a str,
    /// The piku revision under test, so a control run can be compared to the
    /// same control on another build.
    pub piku_revision: &'a str,
}

/// The filesystem-backed task contract selected for a run.
#[derive(Serialize)]
pub struct ScenarioContractRecord<'a> {
    pub schema_version: u8,
    pub kind: &'static str,
    pub run_id: &'a str,
    pub timestamp_secs: u64,
    pub scenario_id: &'a str,
    pub contexts: &'a [&'a str],
    pub goal: &'a str,
    pub verifications: &'a [String],
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
    /// `valid`, `skipped`, `provider_failure`, or `invalid_json`. Read
    /// `primary_review_grounded` as a judgment only when this is `valid`.
    pub status: &'a str,
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
    pub development_context_path: &'a str,
}

/// Deterministic context for the engineer who closes the evaluation loop.
/// It deliberately excludes free-form judge prose: only reproducible evidence
/// and the selected next action belong in a product-development handoff.
#[derive(Serialize)]
pub struct DevelopmentContextRecord<'a> {
    pub schema_version: u8,
    pub run_id: &'a str,
    pub persona: &'a str,
    pub prior_verified_history: &'a str,
    /// The scenario goal the run was working toward, empty when the persona
    /// has no filesystem contract.
    pub scenario_goal: &'a str,
    /// One line per executable acceptance check, pass or fail. These are the
    /// authoritative product outcomes; LLM review only annotates them.
    pub scenario_results: &'a [String],
    pub verified_findings: &'a [String],
    pub hypotheses: &'a [String],
    pub next_action: &'a str,
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

    pub fn append_scenario_contract(
        &self,
        record: &ScenarioContractRecord<'_>,
    ) -> std::io::Result<()> {
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

    pub fn write_development_context(
        &self,
        record: &DevelopmentContextRecord<'_>,
    ) -> std::io::Result<PathBuf> {
        let directory = self
            .path
            .parent()
            .unwrap_or_else(|| std::path::Path::new("."))
            .join("development-context");
        fs::create_dir_all(&directory)?;
        let path = directory.join(format!("{}.json", self.run_id));
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&path)?;
        serde_json::to_writer_pretty(file, record)?;
        Ok(path)
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
                permission_events: &[],
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
                model_selection_seed: Some("42"),
                user_agent_client: "direct-https/reqwest",
                judge_client: "direct-https/reqwest",
                run_role: "control",
                piku_revision: "abc1234",
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
                status: "valid",
            })
            .unwrap();
        let contract_verifications = vec!["cargo test --quiet".to_string()];
        ledger
            .append_scenario_contract(&ScenarioContractRecord {
                schema_version: 1,
                kind: "scenario_contract",
                run_id: ledger.run_id(),
                timestamp_secs: 1,
                scenario_id: "feature-line-numbers",
                contexts: &["feature-development"],
                goal: "return 1-based line numbers",
                verifications: &contract_verifications,
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
                development_context_path: "development-context/example.json",
            })
            .unwrap();
        let context_path = ledger
            .write_development_context(&DevelopmentContextRecord {
                schema_version: 1,
                run_id: ledger.run_id(),
                persona: "tester",
                prior_verified_history: "[MAJOR] previous deterministic failure",
                scenario_goal: "return 1-based line numbers",
                scenario_results: &["fail: cargo test --quiet".to_string()],
                verified_findings: &verified,
                hypotheses: &hypotheses,
                next_action: "fix_piku_for_failed_scenario_acceptance",
            })
            .unwrap();
        let content = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<_> = content.lines().collect();
        assert_eq!(lines.len(), 5);
        let record: serde_json::Value = serde_json::from_str(lines[0]).unwrap();
        assert_eq!(record["kind"], "turn");
        assert_eq!(record["observations"][0], "response is readable");
        let config: serde_json::Value = serde_json::from_str(lines[1]).unwrap();
        assert_eq!(config["kind"], "config");
        assert_eq!(config["judge_model"], "test-judge");
        assert_eq!(config["model_selection_seed"], "42");
        assert_eq!(config["run_role"], "control");
        assert_eq!(config["piku_revision"], "abc1234");
        let observer: serde_json::Value = serde_json::from_str(lines[2]).unwrap();
        assert_eq!(observer["kind"], "recursive_observer");
        assert_eq!(observer["verdict"], "keep");
        let contract: serde_json::Value = serde_json::from_str(lines[3]).unwrap();
        assert_eq!(contract["kind"], "scenario_contract");
        assert_eq!(contract["scenario_id"], "feature-line-numbers");
        assert_eq!(contract["verifications"][0], "cargo test --quiet");
        let handoff: serde_json::Value = serde_json::from_str(lines[4]).unwrap();
        assert_eq!(handoff["kind"], "improvement_handoff");
        assert_eq!(
            handoff["next_action"],
            "reproduce_verified_findings_then_fix"
        );
        let context_packet: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(context_path).unwrap()).unwrap();
        assert_eq!(context_packet["persona"], "tester");
        assert_eq!(
            context_packet["prior_verified_history"],
            "[MAJOR] previous deterministic failure"
        );
        assert_eq!(
            context_packet["verified_findings"][0],
            "a deterministic check failed"
        );
        assert_eq!(
            context_packet["scenario_goal"],
            "return 1-based line numbers"
        );
        assert_eq!(
            context_packet["scenario_results"][0],
            "fail: cargo test --quiet"
        );
        assert_eq!(
            context_packet["next_action"],
            "fix_piku_for_failed_scenario_acceptance"
        );
        std::fs::remove_dir_all(directory).unwrap();
    }
}
