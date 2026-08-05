//! Append-only evidence records for opt-in terminal playground runs.

use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
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
    /// `valid`, `invalid`, `skipped`, `provider_failure`, or `invalid_json`.
    pub status: &'a str,
    /// Machine-readable attestations retained only when the entire review
    /// validates against the run's frozen source/evidence catalog.
    pub claims: &'a [ReviewClaimRecord],
    /// Reasons an invalid review could not be admitted. Empty means either a
    /// valid review or an unavailable judge; those cases are named by `status`.
    pub invalid_reasons: &'a [String],
    /// Human-readable rendering for operator inspection; it is never the
    /// authoritative claim representation.
    pub review: &'a str,
}

/// One validated primary-review attestation.
#[derive(Serialize)]
pub struct ReviewClaimRecord {
    pub id: String,
    pub verdict: String,
    pub rationale: String,
    pub evidence_turns: Vec<u64>,
}

/// One validated second-order attestation of a primary-review claim.
///
/// This is deliberately separate from [`ReviewClaimRecord`]: an observer can
/// challenge an attestation, but must not overwrite its historical record.
#[derive(Debug, Serialize)]
pub struct ObserverClaimRecord {
    pub target_claim_id: String,
    pub disposition: String,
    pub rationale: String,
    pub evidence_turns: Vec<u64>,
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
    /// Request and environment parameters. Two runs of the same models can
    /// still differ on these, so a comparison that ignores them is comparing
    /// two different experiments.
    pub review_max_tokens: u32,
    pub turn_limit: usize,
    pub terminal_rows: u16,
    pub terminal_cols: u16,
    /// How the harness answers a permission prompt, which decides whether a
    /// run exercises the allow path or the deny path at all.
    pub permission_response: &'a str,
    /// True when both LLM review layers were skipped.
    pub fast_mode: bool,
    pub scenario_id: &'a str,
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

/// What the run cost. Harness figures come from the provider's own accounting;
/// piku's are the token counts it printed in its status footer, which is the
/// only usage it reports to an observer.
#[derive(Serialize)]
pub struct SpendRecord<'a> {
    pub schema_version: u8,
    pub kind: &'static str,
    pub run_id: &'a str,
    pub timestamp_secs: u64,
    pub harness_calls: u64,
    pub harness_prompt_tokens: u64,
    pub harness_completion_tokens: u64,
    /// Reported by the provider. Zero means it reported no cost, not free.
    pub harness_cost_usd: f64,
    pub piku_input_tokens: u64,
    pub piku_output_tokens: u64,
    /// Wall clock split between the harness's own review calls and waiting on
    /// piku. Cost and latency do not sit in the same place, and which half
    /// dominates decides what is worth changing.
    pub review_wall_ms: u64,
    pub piku_wait_wall_ms: u64,
    pub change_wait_wall_ms: u64,
    pub acceptance_wall_ms: u64,
}

/// Deterministic audit of the semantic evidence Piku retained for this run.
/// This sits beside judge records because it is an independent measurement,
/// not another opinion about the terminal transcript.
#[derive(Serialize)]
pub struct RunEvidenceRecord<'a> {
    pub schema_version: u8,
    pub kind: &'static str,
    pub run_id: &'a str,
    pub timestamp_secs: u64,
    pub run_record_path: &'a str,
    pub audit: &'a piku_runtime::RunAudit,
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
    /// Typed counter-attestations retained only when the observer accounted
    /// for every primary claim with known turn evidence.
    pub claim_assessments: &'a [ObserverClaimRecord],
    /// Why an invalid observer record could not be admitted.
    pub invalid_reasons: &'a [String],
    /// `valid`, `invalid`, `skipped`, `provider_failure`, or `invalid_json`.
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
    /// Copy of piku's own session file: the messages it sent, the tools it
    /// called with arguments and results, and per-turn usage. Empty when piku
    /// reported no session on exit.
    pub piku_session_path: &'a str,
    /// Frozen copy of the semantic event record and any referenced artifacts.
    pub piku_run_record_path: &'a str,
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
    /// Copy of piku's own session file, so the engineer can read what piku
    /// sent and called rather than only what the terminal showed.
    pub piku_session_path: &'a str,
    pub piku_run_record_path: &'a str,
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

    pub fn append_spend(&self, record: &SpendRecord<'_>) -> std::io::Result<()> {
        self.append(record)
    }

    pub fn append_run_evidence(&self, record: &RunEvidenceRecord<'_>) -> std::io::Result<()> {
        self.append(record)
    }

    /// Keep piku's own session file with the run that produced it.
    ///
    /// piku writes into a shared sessions directory that later runs add to and
    /// a user may prune, so a path recorded now can point at nothing later. A
    /// copy beside the ledger keeps the messages, tool calls, and usage
    /// readable for as long as the evidence is.
    /// Keep piku's event trace with the run, alongside its session.
    ///
    /// The session says what was exchanged; the trace says when, and how long
    /// each provider stream took. Separating a slow provider from a hang needs
    /// the second.
    pub fn copy_piku_trace(&self, source: &Path) -> std::io::Result<PathBuf> {
        let directory = self
            .path
            .parent()
            .unwrap_or_else(|| std::path::Path::new("."))
            .join("piku-traces");
        fs::create_dir_all(&directory)?;
        let destination = directory.join(format!("{}.jsonl", self.run_id));
        fs::copy(source, &destination)?;
        Ok(destination)
    }

    pub fn copy_piku_session(&self, source: &Path) -> std::io::Result<PathBuf> {
        let directory = self
            .path
            .parent()
            .unwrap_or_else(|| std::path::Path::new("."))
            .join("piku-sessions");
        fs::create_dir_all(&directory)?;
        let destination = directory.join(format!("{}.json", self.run_id));
        fs::copy(source, &destination)?;
        Ok(destination)
    }

    /// Freeze the JSONL record and its flat artifact directory as one bundle.
    /// Keeping the original file names preserves run-relative artifact paths.
    pub fn copy_piku_run(&self, source: &Path) -> std::io::Result<PathBuf> {
        let directory = self
            .path
            .parent()
            .unwrap_or_else(|| std::path::Path::new("."))
            .join("piku-runs")
            .join(&self.run_id);
        fs::create_dir_all(&directory)?;
        let file_name = source
            .file_name()
            .ok_or_else(|| std::io::Error::other("run record path has no file name"))?;
        let destination = directory.join(file_name);
        fs::copy(source, &destination)?;

        let artifacts = source.with_extension("artifacts");
        if artifacts.is_dir() {
            let artifact_name = artifacts
                .file_name()
                .ok_or_else(|| std::io::Error::other("artifact path has no file name"))?;
            let artifact_destination = directory.join(artifact_name);
            fs::create_dir_all(&artifact_destination)?;
            for entry in fs::read_dir(artifacts)? {
                let entry = entry?;
                if entry.file_type()?.is_file() {
                    fs::copy(entry.path(), artifact_destination.join(entry.file_name()))?;
                }
            }
        }
        Ok(destination)
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

    /// Append one record as a single write.
    ///
    /// Serialising straight into the file took two writes, the record and then
    /// the newline, so two runs appending at once could interleave and leave a
    /// line that parses as neither. Building the line first and writing it once
    /// lets concurrent runs share a ledger: an `O_APPEND` write is atomic up to a
    /// pipe buffer, which covers a bounded record. Oversized records can still
    /// tear, which is why turn evidence is truncated before it gets here.
    fn append<T: Serialize>(&self, record: &T) -> std::io::Result<()> {
        let mut line = serde_json::to_string(record)
            .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error))?;
        line.push('\n');
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)?;
        file.write_all(line.as_bytes())
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
    fn run_bundle_copy_preserves_relative_artifact_paths() {
        let directory = tempfile::tempdir().unwrap();
        let source_dir = directory.path().join("runtime/runs");
        let artifact_dir = source_dir.join("session-1.artifacts");
        fs::create_dir_all(&artifact_dir).unwrap();
        let source = source_dir.join("session-1.jsonl");
        fs::write(&source, "record\n").unwrap();
        fs::write(artifact_dir.join("00000001-tool-result.txt"), "evidence").unwrap();
        let ledger = PlaygroundLedger::open_at(directory.path().join("ledger.jsonl")).unwrap();

        let copied = ledger.copy_piku_run(&source).unwrap();

        assert_eq!(fs::read_to_string(&copied).unwrap(), "record\n");
        assert_eq!(
            fs::read_to_string(
                copied
                    .parent()
                    .unwrap()
                    .join("session-1.artifacts/00000001-tool-result.txt")
            )
            .unwrap(),
            "evidence"
        );
    }

    /// Two runs sharing a ledger must leave lines that each parse. Records are
    /// written as one call for this reason; the earlier two-call form could
    /// interleave a record and a newline from different processes.
    #[test]
    fn concurrent_writers_leave_parseable_lines() {
        let directory =
            std::env::temp_dir().join(format!("piku-ledger-concurrent-{}", std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();
        let path = directory.join("shared.jsonl");

        let writers: Vec<_> = (0..4)
            .map(|worker| {
                let path = path.clone();
                std::thread::spawn(move || {
                    let ledger = PlaygroundLedger::open_at(path).unwrap();
                    for turn in 0..25 {
                        ledger
                            .append_review(&ReviewRecord {
                                schema_version: 1,
                                kind: "review",
                                run_id: ledger.run_id(),
                                timestamp_secs: 1,
                                persona: "tester",
                                status: "valid",
                                claims: &[],
                                invalid_reasons: &[],
                                review: &format!("worker {worker} turn {turn}"),
                            })
                            .unwrap();
                    }
                })
            })
            .collect();
        for writer in writers {
            writer.join().unwrap();
        }

        let content = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<_> = content.lines().collect();
        assert_eq!(lines.len(), 100, "lost or split a record");
        for line in lines {
            serde_json::from_str::<serde_json::Value>(line)
                .unwrap_or_else(|error| panic!("torn line {line:?}: {error}"));
        }
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn review_record_keeps_typed_claims_separate_from_prose() {
        let directory =
            std::env::temp_dir().join(format!("piku-review-ledger-{}", std::process::id()));
        let path = directory.join("ledger.jsonl");
        std::fs::create_dir_all(&directory).unwrap();
        let ledger = PlaygroundLedger::open_at(path.clone()).unwrap();
        let claims = vec![ReviewClaimRecord {
            id: "user-bug-1-1".to_string(),
            verdict: "VALID".to_string(),
            rationale: "the prompt was absent on turn one".to_string(),
            evidence_turns: vec![1],
        }];
        ledger
            .append_review(&ReviewRecord {
                schema_version: 1,
                kind: "review",
                run_id: ledger.run_id(),
                timestamp_secs: 1,
                persona: "tester",
                status: "valid",
                claims: &claims,
                invalid_reasons: &[],
                review: "human-readable rendering",
            })
            .unwrap();

        let record: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(record["status"], "valid");
        assert_eq!(record["claims"][0]["id"], "user-bug-1-1");
        assert_eq!(
            record["claims"][0]["evidence_turns"],
            serde_json::json!([1])
        );
        assert_eq!(record["review"], "human-readable rendering");
        std::fs::remove_dir_all(directory).unwrap();
    }

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
                review_max_tokens: 2048,
                turn_limit: 6,
                terminal_rows: 40,
                terminal_cols: 120,
                permission_response: "n (deny)",
                fast_mode: false,
                scenario_id: "feature-line-numbers",
            })
            .unwrap();
        let judge_observations = vec!["the primary review is grounded".to_string()];
        let piku_observations = vec!["the prompt stayed visible".to_string()];
        let claim_assessments = vec![ObserverClaimRecord {
            target_claim_id: "user-bug-1-1".to_string(),
            disposition: "SUPPORTED".to_string(),
            rationale: "turn 1 contains the observed failure".to_string(),
            evidence_turns: vec![1],
        }];
        ledger
            .append_observer(&ObserverRecord {
                schema_version: 2,
                kind: "recursive_observer",
                run_id: ledger.run_id(),
                timestamp_secs: 1,
                persona: "tester",
                judge_observations: &judge_observations,
                piku_observations: &piku_observations,
                verdict: "keep",
                claim_assessments: &claim_assessments,
                invalid_reasons: &[],
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
                piku_session_path: "piku-sessions/example.json",
                piku_run_record_path: "piku-runs/example/session.jsonl",
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
                piku_session_path: "piku-sessions/example.json",
                piku_run_record_path: "piku-runs/example/session.jsonl",
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
        assert_eq!(observer["schema_version"], 2);
        assert_eq!(
            observer["claim_assessments"][0]["target_claim_id"],
            "user-bug-1-1"
        );
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
