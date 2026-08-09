import { appendFile, mkdir } from "node:fs/promises";
import { execFileSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { randomUUID } from "node:crypto";
import path from "node:path";
import { resolvedCodexModel } from "./codex-exec.mjs";
import {
  assertEvaluationEnvelope,
  EVALUATION_SCHEMA_VERSION,
} from "../../../../scripts/evaluation-envelope.mjs";

export { assertEvaluationEnvelope, EVALUATION_SCHEMA_VERSION };

function commandOutput(command, args, cwd) {
  try {
    return execFileSync(command, args, { cwd, encoding: "utf8", stdio: ["ignore", "pipe", "ignore"] }).trim();
  } catch {
    return "unavailable";
  }
}

export function evaluationRuntimeMetadata(repoRoot) {
  const cargo = readFileSync(path.join(repoRoot, "Cargo.toml"), "utf8");
  const subjectVersion = cargo.match(/\[workspace\.package\][\s\S]*?\nversion\s*=\s*"([^"]+)"/)?.[1] || "unknown";
  return {
    subject_version: subjectVersion,
    subject_revision: commandOutput("git", ["rev-parse", "HEAD"], repoRoot),
    subject_dirty: Boolean(commandOutput("git", ["status", "--porcelain=v1"], repoRoot)),
    evaluator_runtime: "codex-cli",
    evaluator_version: commandOutput("codex", ["--version"], repoRoot),
    explorer_model: resolvedCodexModel(),
    evaluation_contract: "piku-evaluation-v2",
  };
}

export function evaluationRecord({
  runId,
  surface = null,
  stageId = surface ?? "synthesis",
  runStatus,
  failureClass,
  durationMs,
  report = null,
  artifactRefs = [],
  runtime = {},
}) {
  const identity = projectReportIdentity(report, runId, stageId);
  return {
    schema_version: EVALUATION_SCHEMA_VERSION,
    run_id: runId,
    record_kind: "stage",
    stage_id: stageId,
    scenario_id: "web-codex-replacement-thesis",
    surface: "web",
    subject_surface: surface,
    perspective: "integrated_explorer",
    subject_model: null,
    explorer_model: "codex",
    judge_model: null,
    task_contract: "agentic-legibility-evidence-board",
    run_status: runStatus,
    failure_class: failureClass,
    product_verdict: report?.product_thesis?.verdict ?? null,
    finding_count: report?.findings?.length ?? null,
    finding_refs: identity.findingRefs,
    evidence_ids: [],
    artifact_refs: artifactRefs,
    followups: identity.followups,
    duration_ms: durationMs,
    ...runtime,
  };
}

function localId(value, prefix, index, label) {
  const id = value ?? `${prefix}${index + 1}`;
  if (!new RegExp(`^${prefix}[1-9][0-9]*$`).test(id))
    throw new TypeError(`${label} ID must match ${prefix}N`);
  return id;
}

export function projectReportIdentity(report, runId, stageId) {
  if (runId.includes(":") || stageId.includes(":"))
    throw new TypeError("run and stage IDs used for scoped report identity must not contain colons");
  if (!report) return { findingRefs: [], followups: [] };
  const findingIds = (report.findings || []).map((finding, index) =>
    localId(finding.id, "f", index, "finding"));
  if (new Set(findingIds).size !== findingIds.length)
    throw new TypeError("report finding IDs must be unique");
  const findingByLocalId = new Map(findingIds.map((id) => [
    id, `${runId}:${stageId}:finding:${id}`,
  ]));
  const obligationIds = (report.followups || []).map((followup, index) =>
    localId(followup.id, "o", index, "followup"));
  if (new Set(obligationIds).size !== obligationIds.length)
    throw new TypeError("report followup IDs must be unique");
  const followups = (report.followups || []).map((followup, index) => {
    const findingRefs = (followup.finding_ids || []).map((id) => {
      const scoped = findingByLocalId.get(id);
      if (!scoped) throw new TypeError(`followup ${obligationIds[index]} cites unknown finding ID ${id}`);
      return scoped;
    });
    if ((followup.evidence_ids || []).length === 0 && findingRefs.length === 0)
      throw new TypeError(`followup ${obligationIds[index]} must cite evidence_ids or finding_ids`);
    if (followup.retest_of !== null && followup.retest_of !== undefined &&
        !/^[^:]+:[^:]+:obligation:o[1-9][0-9]*$/.test(followup.retest_of))
      throw new TypeError(`followup ${obligationIds[index]} retest_of must be a scoped obligation ID or null`);
    return {
      obligation_id: `${runId}:${stageId}:obligation:${obligationIds[index]}`,
      kind: followup.kind,
      priority: followup.priority,
      title: followup.title,
      rationale: followup.rationale,
      perspective: followup.perspective,
      evidence_ids: followup.evidence_ids || [],
      finding_refs: findingRefs,
      retest_of: followup.retest_of ?? null,
    };
  });
  return {
    findingRefs: [...findingByLocalId.values()],
    followups,
  };
}

export function evaluationAmendment({
  targetRecord,
  action,
  reasonCode,
  scope,
  basisRefs = [],
  basisHashes = [],
  replacement = null,
  replacementRunId = null,
  actor,
  toolVersion,
  eventId = randomUUID(),
  recordedAt = new Date().toISOString(),
  contractVersion = "piku-evaluation-amendment-v2",
}) {
  if (!targetRecord || targetRecord.record_kind === "amendment")
    throw new TypeError("an amendment must target an original run or stage record");
  return assertEvaluationEnvelope({
    ...targetRecord,
    record_kind: "amendment",
    stage_id: `amendment:${eventId}`,
    target_run_id: targetRecord.run_id,
    target_stage_id: targetRecord.stage_id,
    event_id: eventId,
    recorded_at: recordedAt,
    contract_version: contractVersion,
    amendment_action: action,
    reason_code: reasonCode,
    amendment_scope: scope,
    basis_refs: basisRefs,
    basis_hashes: basisHashes,
    replacement,
    replacement_run_id: replacementRunId,
    actor,
    tool_version: toolVersion,
    run_status: "completed",
    failure_class: "none",
    product_verdict: null,
    finding_count: null,
    finding_refs: [],
    evidence_ids: [],
    artifact_refs: basisRefs,
    followups: [],
    duration_ms: 0,
  });
}

export async function appendEvaluationRecord(ledgerPath, record) {
  assertEvaluationEnvelope(record, `evaluation record for ${ledgerPath}`);
  await mkdir(path.dirname(ledgerPath), { recursive: true });
  await appendFile(ledgerPath, `${JSON.stringify(record)}\n`, "utf8");
}
