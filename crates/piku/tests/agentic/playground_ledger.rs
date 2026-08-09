//! Append-only evidence records for opt-in terminal playground runs.

use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::Serialize;
use sha2::{Digest, Sha256};

pub const EVALUATION_CONTRACT: &str = "piku-evaluation-envelope-v1";
pub const EVALUATOR_VERSION: &str = concat!("agentic-playground-rust/", env!("CARGO_PKG_VERSION"));

#[derive(Debug, Serialize)]
pub struct PromptManifest {
    pub schema_version: u8,
    pub run_id: String,
    pub surface: &'static str,
    pub subject: PromptManifestSubject,
    pub evaluator: PromptManifestEvaluator,
    pub roles: Vec<PromptManifestRole>,
    pub effective_config: AttestedValue,
}

#[derive(Debug, Serialize)]
pub struct PromptManifestSubject {
    pub version: &'static str,
    pub revision: String,
    pub dirty: bool,
    pub model: String,
}

#[derive(Debug, Serialize)]
pub struct PromptManifestEvaluator {
    pub runtime: &'static str,
    pub version: &'static str,
    pub contract: &'static str,
}

#[derive(Debug, Serialize)]
pub struct PromptManifestRole {
    pub role: &'static str,
    pub provider: String,
    pub model: String,
    pub prompt_assets: Vec<PromptAsset>,
    pub context_contract: AttestedValue,
    pub tools: AttestedValue,
    pub limits: serde_json::Value,
}

#[derive(Debug, Serialize)]
pub struct PromptAsset {
    pub kind: &'static str,
    pub path: String,
    pub sha256: String,
    pub size_bytes: usize,
}

#[derive(Debug, Serialize)]
pub struct AttestedValue {
    pub sha256: String,
    pub value: serde_json::Value,
}

impl AttestedValue {
    #[must_use]
    pub fn new(value: serde_json::Value) -> Self {
        Self {
            sha256: sha256(canonical_json(&value).as_bytes()),
            value,
        }
    }
}

#[derive(Debug)]
pub struct PromptManifestArtifact {
    pub path: PathBuf,
    pub sha256: String,
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn canonical_json(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::Array(values) => format!(
            "[{}]",
            values
                .iter()
                .map(canonical_json)
                .collect::<Vec<_>>()
                .join(",")
        ),
        serde_json::Value::Object(values) => {
            let mut entries = values.iter().collect::<Vec<_>>();
            entries.sort_unstable_by_key(|(key, _)| *key);
            format!(
                "{{{}}}",
                entries
                    .into_iter()
                    .map(|(key, value)| format!(
                        "{}:{}",
                        serde_json::to_string(key).expect("JSON object key serializes"),
                        canonical_json(value)
                    ))
                    .collect::<Vec<_>>()
                    .join(",")
            )
        }
        _ => serde_json::to_string(value).expect("JSON value serializes"),
    }
}

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

/// Cross-domain metric vector for one run. Unmeasured human properties remain
/// named as unavailable instead of being replaced with simulator proxies.
#[derive(Serialize)]
pub struct PrincipleMetricsRecord<'a> {
    pub schema_version: u8,
    pub kind: &'static str,
    pub run_id: &'a str,
    pub timestamp_secs: u64,
    pub scenario_id: &'a str,
    pub outcome: OutcomeMetrics,
    pub attention: AttentionMetrics,
    pub evidence: EvidenceMetrics,
    pub control: ControlMetrics,
    pub understanding_measurement: &'static str,
    pub continuity_measurement: &'static str,
}

#[derive(Serialize)]
pub struct OutcomeMetrics {
    pub passed_checks: usize,
    pub failed_checks: usize,
    pub inconclusive_checks: usize,
    pub unverified_clauses: usize,
}

#[derive(Serialize)]
pub struct AttentionMetrics {
    pub observed_terminal_chars: usize,
    pub observed_terminal_lines: usize,
    pub semantic_event_count: usize,
    pub compact_projection_chars: usize,
    pub compact_projection_lines: usize,
    pub raw_record_bytes: u64,
    pub artifact_bytes: u64,
}

#[derive(Serialize)]
pub struct EvidenceMetrics {
    pub structurally_complete: Option<bool>,
    pub audit_errors: usize,
    pub audit_warnings: usize,
    pub context_messages_selected: usize,
    pub context_messages_excluded: usize,
    pub unavailable_content_items: usize,
    pub primary_claims: usize,
    pub primary_valid_claims: usize,
    pub observer_supported_claims: usize,
}

#[derive(Serialize)]
pub struct ControlMetrics {
    pub tool_calls_started: usize,
    pub permission_decisions_recorded: usize,
    pub permission_prompts_observed: usize,
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

/// The single shared-envelope projection of a completed playground run.
///
/// The detailed append-only playground ledger remains authoritative. This
/// deliberately small projection lets the CLI/TUI and web evaluators share
/// summaries without flattening the evidence bundle into a second ledger.
#[derive(Debug, Serialize)]
pub struct EvaluationSummaryRecord<'a> {
    pub schema_version: u8,
    pub run_id: &'a str,
    pub record_kind: &'static str,
    pub stage_id: &'static str,
    pub scenario_id: &'a str,
    pub surface: &'static str,
    pub subject_surface: &'static str,
    pub perspective: &'a str,
    pub subject_model: &'a str,
    pub explorer_model: &'a str,
    pub judge_model: &'a str,
    pub subject_version: &'static str,
    pub subject_revision: &'a str,
    pub subject_dirty: bool,
    pub evaluator_runtime: &'static str,
    pub evaluator_version: &'static str,
    pub evaluation_contract: &'static str,
    pub prompt_manifest: PromptManifestReference<'a>,
    pub task_contract: &'a str,
    pub run_status: &'a str,
    pub failure_class: &'a str,
    pub product_verdict: Option<&'a str>,
    pub finding_count: usize,
    pub evidence_ids: &'a [String],
    pub artifact_refs: &'a [String],
    pub followups: &'a [EvaluationFollowup],
    pub duration_ms: u64,
}

#[derive(Debug, Serialize)]
pub struct PromptManifestReference<'a> {
    pub path: &'a str,
    pub sha256: &'a str,
}

#[derive(Debug, Serialize, PartialEq, Eq)]
pub struct EvaluationFollowup {
    pub kind: &'static str,
    pub priority: &'static str,
    pub title: String,
    pub rationale: String,
    pub perspective: Option<String>,
    pub evidence_ids: Vec<String>,
}

#[must_use]
pub fn evaluation_followups(
    persona: &str,
    verified_findings: &[String],
    hypotheses: &[String],
) -> Vec<EvaluationFollowup> {
    let mut followups = Vec::with_capacity(verified_findings.len() + hypotheses.len());
    followups.extend(verified_findings.iter().map(|finding| EvaluationFollowup {
        kind: "todo",
        priority: "high",
        title: bounded_title(finding, "Address verified finding"),
        rationale: finding.clone(),
        perspective: Some(persona.to_string()),
        evidence_ids: Vec::new(),
    }));
    followups.extend(hypotheses.iter().map(|hypothesis| EvaluationFollowup {
        kind: "retest",
        priority: "medium",
        title: bounded_title(hypothesis, "Reproduce hypothesis"),
        rationale: hypothesis.clone(),
        perspective: Some(persona.to_string()),
        evidence_ids: Vec::new(),
    }));
    followups
}

fn bounded_title(value: &str, fallback: &str) -> String {
    let first_line = value.lines().next().unwrap_or_default().trim();
    let without_prefix = first_line
        .split_once("] ")
        .map_or(first_line, |(_, remainder)| remainder);
    let title: String = without_prefix.chars().take(120).collect();
    if title.is_empty() {
        fallback.to_string()
    } else {
        title
    }
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

    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
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

    /// Freeze evaluator identity before the first model call.
    ///
    /// The file is create-only. Mutable run state and summaries may reference
    /// it, but neither a later review nor a resumed stage may rewrite it.
    pub fn write_prompt_manifest(
        &self,
        manifest: &PromptManifest,
    ) -> std::io::Result<PromptManifestArtifact> {
        let directory = self
            .path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join("prompt-manifests");
        fs::create_dir_all(&directory)?;
        let path = directory.join(format!("{}.json", self.run_id));
        let mut bytes = serde_json::to_vec_pretty(manifest).map_err(std::io::Error::other)?;
        bytes.push(b'\n');
        let mut file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&path)?;
        file.write_all(&bytes)?;
        file.sync_all()?;
        Ok(PromptManifestArtifact {
            path,
            sha256: sha256(&bytes),
        })
    }

    pub fn write_prompt_asset(
        &self,
        role: &str,
        kind: &'static str,
        contents: &str,
    ) -> std::io::Result<PromptAsset> {
        let directory = self
            .path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join("prompt-manifests")
            .join(format!("{}.assets", self.run_id));
        fs::create_dir_all(&directory)?;
        let path = directory.join(format!("{role}-{kind}.txt"));
        let bytes = contents.as_bytes();
        let mut file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&path)?;
        file.write_all(bytes)?;
        file.sync_all()?;
        Ok(PromptAsset {
            kind,
            path: path.display().to_string(),
            sha256: sha256(bytes),
            size_bytes: bytes.len(),
        })
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

    pub fn append_principle_metrics(
        &self,
        record: &PrincipleMetricsRecord<'_>,
    ) -> std::io::Result<()> {
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

    /// Append the one canonical summary row for this run.
    ///
    /// `PIKU_LIVE_LEDGER` joins an explicitly selected shared ledger. Otherwise
    /// playground runs use the standard local live-ledger directory.
    pub fn append_evaluation_summary(
        &self,
        record: &EvaluationSummaryRecord<'_>,
    ) -> std::io::Result<PathBuf> {
        let path = std::env::var("PIKU_LIVE_LEDGER").map_or_else(
            |_| {
                PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .parent()
                    .and_then(|path| path.parent())
                    .expect("piku crate has workspace root")
                    .join("target/live-ledger/playground.jsonl")
            },
            PathBuf::from,
        );
        if path == self.path {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "shared evaluation ledger must differ from detailed playground ledger",
            ));
        }
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        append_json_line(&path, record)?;
        Ok(path)
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

fn append_json_line<T: Serialize>(path: &Path, record: &T) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut line = serde_json::to_string(record)
        .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error))?;
    line.push('\n');
    let mut file = OpenOptions::new().create(true).append(true).open(path)?;
    file.write_all(line.as_bytes())
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
    #![allow(dead_code)]

    use super::*;
    use serde::Deserialize;

    #[derive(Debug, Deserialize, Serialize)]
    #[serde(deny_unknown_fields)]
    struct ContextResolutionFixture {
        schema_version: u8,
        run_id: String,
        role: String,
        checkpoint: String,
        resolver: ResolverFixture,
        request: ResolutionRequestFixture,
        capability_profile: CapabilityProfileFixture,
        status: ResolutionStatusFixture,
        cache: CacheFixture,
        started_at: String,
        finished_at: String,
        items: Vec<ContextItemFixture>,
        warnings: Vec<String>,
        error: Option<ResolutionErrorFixture>,
        materialized_artifact_refs: Vec<String>,
    }

    #[derive(Debug, Deserialize, Serialize)]
    #[serde(deny_unknown_fields)]
    struct ResolverFixture {
        id: String,
        version: String,
        config_sha256: String,
        code_sha256: String,
    }

    #[derive(Debug, Deserialize, Serialize)]
    #[serde(deny_unknown_fields)]
    struct ResolutionRequestFixture {
        output_plane: OutputPlaneFixture,
        replay_mode: ReplayModeFixture,
        byte_budget: usize,
        token_budget: usize,
        deadline_ms: u64,
        freshness_policy: FreshnessPolicyFixture,
    }

    #[derive(Debug, Deserialize, Serialize)]
    #[serde(deny_unknown_fields)]
    struct CapabilityProfileFixture {
        id: String,
        sha256: String,
    }

    #[derive(Debug, Deserialize, Serialize)]
    #[serde(rename_all = "snake_case")]
    enum ResolutionStatusFixture {
        Succeeded,
        Failed,
    }

    #[derive(Debug, Deserialize, Serialize)]
    #[serde(deny_unknown_fields)]
    struct CacheFixture {
        decision: CacheDecisionFixture,
        key_sha256: String,
    }

    #[derive(Debug, Deserialize, Serialize)]
    #[serde(rename_all = "snake_case")]
    enum CacheDecisionFixture {
        Miss,
        Hit,
        Captured,
        Bypass,
    }

    #[derive(Debug, Deserialize, Serialize)]
    #[serde(deny_unknown_fields)]
    struct ContextItemFixture {
        id: String,
        resolver_id: String,
        resolver_version: String,
        output_plane: OutputPlaneFixture,
        media_type: String,
        sources: Vec<ContextSourceFixture>,
        trust: TrustFixture,
        freshness: FreshnessFixture,
        sensitivity: SensitivityFixture,
        priority: i64,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        inline_payload: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        payload_ref: Option<String>,
        byte_size: usize,
        token_estimate: usize,
        output_sha256: String,
        created_at: String,
        expires_at: Option<String>,
        warnings: Vec<String>,
    }

    #[derive(Debug, Deserialize, Serialize)]
    #[serde(deny_unknown_fields)]
    struct ContextSourceFixture {
        r#ref: String,
        sha256: String,
    }

    #[derive(Debug, Deserialize, Serialize)]
    #[serde(rename_all = "snake_case")]
    enum OutputPlaneFixture {
        Instruction,
        Message,
        Tool,
        State,
        Artifact,
    }

    #[derive(Debug, Deserialize, Serialize)]
    #[serde(rename_all = "snake_case")]
    enum ReplayModeFixture {
        Exact,
        Refresh,
        Fork,
    }

    #[derive(Debug, Deserialize, Serialize)]
    #[serde(rename_all = "snake_case")]
    enum FreshnessPolicyFixture {
        Captured,
        Current,
        MaxAge,
    }

    #[derive(Debug, Deserialize, Serialize)]
    #[serde(rename_all = "snake_case")]
    enum TrustFixture {
        Control,
        OperatorInstruction,
        HostFact,
        UntrustedEvidence,
        DerivedEvidence,
    }

    #[derive(Debug, Deserialize, Serialize)]
    #[serde(rename_all = "snake_case")]
    enum FreshnessFixture {
        Captured,
        Current,
        Stale,
        Unknown,
    }

    #[derive(Debug, Deserialize, Serialize)]
    #[serde(rename_all = "snake_case")]
    enum SensitivityFixture {
        Public,
        Workspace,
        Private,
        Secret,
    }

    #[derive(Debug, Deserialize, Serialize)]
    #[serde(deny_unknown_fields)]
    struct ResolutionErrorFixture {
        code: String,
        message: String,
    }

    fn summary_record<'a>(
        ledger: &'a PlaygroundLedger,
        artifacts: &'a [String],
        followups: &'a [EvaluationFollowup],
    ) -> EvaluationSummaryRecord<'a> {
        EvaluationSummaryRecord {
            schema_version: 1,
            run_id: ledger.run_id(),
            record_kind: "stage",
            stage_id: "summary",
            scenario_id: "fixture-repair",
            surface: "tui",
            subject_surface: "tui",
            perspective: "confident_dev",
            subject_model: "subject-model",
            explorer_model: "explorer-model",
            judge_model: "judge-model",
            subject_version: env!("CARGO_PKG_VERSION"),
            subject_revision: "0123456789abcdef0123456789abcdef01234567",
            subject_dirty: true,
            evaluator_runtime: "rust-agentic-playground",
            evaluator_version: EVALUATOR_VERSION,
            evaluation_contract: EVALUATION_CONTRACT,
            prompt_manifest: PromptManifestReference {
                path: "prompt-manifests/run.json",
                sha256: "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
            },
            task_contract: "repair the fixture",
            run_status: "product_failure",
            failure_class: "observed_product_behavior",
            product_verdict: Some("not_supported"),
            finding_count: 2,
            evidence_ids: &[],
            artifact_refs: artifacts,
            followups,
            duration_ms: 42,
        }
    }

    #[test]
    fn canonical_summary_is_one_projection_with_typed_followups() {
        let directory = tempfile::tempdir().unwrap();
        let detailed_path = directory.path().join("agentic-findings/playground.jsonl");
        let ledger = PlaygroundLedger::open_at(detailed_path.clone()).unwrap();
        let verified = vec!["[scenario:fixture] acceptance failed".to_string()];
        let hypotheses = vec!["[unreviewed] rerun after fixing output".to_string()];
        let followups = evaluation_followups("confident_dev", &verified, &hypotheses);
        let artifacts = vec![detailed_path.display().to_string()];
        let summary = summary_record(&ledger, &artifacts, &followups);
        let shared_path = directory.path().join("live-ledger/playground.jsonl");

        append_json_line(&shared_path, &summary).unwrap();

        let lines = fs::read_to_string(shared_path).unwrap();
        assert_eq!(lines.lines().count(), 1);
        let row: serde_json::Value = serde_json::from_str(lines.trim()).unwrap();
        assert_eq!(row["record_kind"], "stage");
        assert_eq!(row["stage_id"], "summary");
        assert_eq!(row["run_status"], "product_failure");
        assert_eq!(row["subject_revision"].as_str().unwrap().len(), 40);
        assert_eq!(row["subject_dirty"], true);
        assert_eq!(row["artifact_refs"][0], artifacts[0]);
        assert_eq!(
            row["prompt_manifest"]["sha256"],
            "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
        );
        assert_eq!(row["followups"][0]["kind"], "todo");
        assert_eq!(row["followups"][1]["kind"], "retest");
        assert_eq!(followups[0].rationale, verified[0]);
        assert_eq!(followups[1].rationale, hypotheses[0]);
    }

    fn prompt_manifest(run_id: &str, prompt: &str) -> PromptManifest {
        PromptManifest {
            schema_version: 1,
            run_id: run_id.to_string(),
            surface: "tui",
            subject: PromptManifestSubject {
                version: "0.1.0",
                revision: "0123456789abcdef0123456789abcdef01234567".to_string(),
                dirty: false,
                model: "subject-model".to_string(),
            },
            evaluator: PromptManifestEvaluator {
                runtime: "test",
                version: "test/1",
                contract: EVALUATION_CONTRACT,
            },
            roles: vec![PromptManifestRole {
                role: "judge",
                provider: "test".to_string(),
                model: "judge-model".to_string(),
                prompt_assets: vec![PromptAsset {
                    kind: "system",
                    path: "prompt.txt".to_string(),
                    sha256: sha256(prompt.as_bytes()),
                    size_bytes: prompt.len(),
                }],
                context_contract: AttestedValue::new(
                    serde_json::json!({ "source_refs": ["evidence"], "bounds": {} }),
                ),
                tools: AttestedValue::new(serde_json::json!({ "names": [], "authority": "none" })),
                limits: serde_json::json!({ "max_output_tokens": 10 }),
            }],
            effective_config: AttestedValue::new(serde_json::json!({ "mode": "test" })),
        }
    }

    #[test]
    fn prompt_manifest_is_create_only_and_content_addressed() {
        let directory = tempfile::tempdir().unwrap();
        let ledger = PlaygroundLedger::open_at(directory.path().join("ledger.jsonl")).unwrap();
        let first = ledger
            .write_prompt_manifest(&prompt_manifest(ledger.run_id(), "first"))
            .unwrap();
        let bytes = fs::read(&first.path).unwrap();

        assert_eq!(first.sha256.len(), 64);
        assert_eq!(sha256(&bytes), first.sha256);
        assert!(ledger
            .write_prompt_manifest(&prompt_manifest(ledger.run_id(), "changed"))
            .is_err());
        assert_eq!(fs::read(&first.path).unwrap(), bytes);
    }

    #[test]
    fn prompt_asset_digest_changes_with_prompt_bytes() {
        assert_ne!(sha256(b"first"), sha256(b"changed"));
    }

    #[test]
    fn tui_prompt_manifest_matches_shared_required_shape() {
        let manifest = prompt_manifest("run-1", "prompt");
        let value = serde_json::to_value(&manifest).unwrap();
        let schema: serde_json::Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../eval/evaluation-prompt-manifest.schema.json"
        )))
        .unwrap();
        for field in schema["required"].as_array().unwrap() {
            assert!(value.get(field.as_str().unwrap()).is_some(), "{field}");
        }
        let role = &value["roles"][0];
        for field in schema["properties"]["roles"]["items"]["required"]
            .as_array()
            .unwrap()
        {
            assert!(role.get(field.as_str().unwrap()).is_some(), "{field}");
        }
        assert_eq!(
            value["effective_config"]["sha256"],
            sha256(canonical_json(&value["effective_config"]["value"]).as_bytes())
        );
        assert_eq!(
            role["context_contract"]["sha256"],
            sha256(canonical_json(&role["context_contract"]["value"]).as_bytes())
        );
    }

    #[test]
    fn context_resolution_fixture_round_trips_exact_utf8_payload() {
        let source = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../eval/fixtures/context-resolution.v1.json"
        ));
        let resolution: ContextResolutionFixture = serde_json::from_str(source).unwrap();
        let item = &resolution.items[0];
        let payload = item.inline_payload.as_ref().unwrap();

        assert_eq!(resolution.schema_version, 1);
        assert!(matches!(
            resolution.status,
            ResolutionStatusFixture::Succeeded
        ));
        assert!(matches!(
            resolution.request.replay_mode,
            ReplayModeFixture::Exact
        ));
        assert!(matches!(
            resolution.cache.decision,
            CacheDecisionFixture::Captured
        ));
        assert!(item.payload_ref.is_none());
        assert_eq!(payload.len(), item.byte_size);
        assert_eq!(sha256(payload.as_bytes()), item.output_sha256);

        let round_trip = serde_json::to_value(&resolution).unwrap();
        let original: serde_json::Value = serde_json::from_str(source).unwrap();
        assert_eq!(round_trip, original);
    }

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

    #[test]
    fn principle_metrics_keep_human_dimensions_explicitly_unmeasured() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("ledger.jsonl");
        let ledger = PlaygroundLedger::open_at(path.clone()).unwrap();
        ledger
            .append_principle_metrics(&PrincipleMetricsRecord {
                schema_version: 1,
                kind: "principle_metrics",
                run_id: ledger.run_id(),
                timestamp_secs: 1,
                scenario_id: "delivery",
                outcome: OutcomeMetrics {
                    passed_checks: 2,
                    failed_checks: 0,
                    inconclusive_checks: 0,
                    unverified_clauses: 1,
                },
                attention: AttentionMetrics {
                    observed_terminal_chars: 500,
                    observed_terminal_lines: 20,
                    semantic_event_count: 8,
                    compact_projection_chars: 180,
                    compact_projection_lines: 9,
                    raw_record_bytes: 1_200,
                    artifact_bytes: 4_000,
                },
                evidence: EvidenceMetrics {
                    structurally_complete: Some(true),
                    audit_errors: 0,
                    audit_warnings: 0,
                    context_messages_selected: 4,
                    context_messages_excluded: 2,
                    unavailable_content_items: 0,
                    primary_claims: 2,
                    primary_valid_claims: 1,
                    observer_supported_claims: 1,
                },
                control: ControlMetrics {
                    tool_calls_started: 1,
                    permission_decisions_recorded: 1,
                    permission_prompts_observed: 0,
                },
                understanding_measurement: "not_measured_requires_human_trial",
                continuity_measurement: "not_measured_requires_recovery_or_fork_scenario",
            })
            .unwrap();

        let record: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(path).unwrap()).unwrap();
        assert_eq!(record["outcome"]["passed_checks"], 2);
        assert_eq!(record["attention"]["compact_projection_chars"], 180);
        assert_eq!(
            record["understanding_measurement"],
            "not_measured_requires_human_trial"
        );
        assert_eq!(
            record["continuity_measurement"],
            "not_measured_requires_recovery_or_fork_scenario"
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
