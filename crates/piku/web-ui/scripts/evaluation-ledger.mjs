import { appendFile, mkdir } from "node:fs/promises";
import { execFileSync } from "node:child_process";
import { readFileSync } from "node:fs";
import path from "node:path";
import { resolvedCodexModel } from "./codex-exec.mjs";

export const EVALUATION_SCHEMA_VERSION = 1;

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
    evaluation_contract: "piku-evaluation-v1",
  };
}

export function evaluationRecord({
  runId,
  surface = null,
  runStatus,
  failureClass,
  durationMs,
  report = null,
  artifactRefs = [],
  runtime = {},
}) {
  return {
    schema_version: EVALUATION_SCHEMA_VERSION,
    run_id: runId,
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
    evidence_ids: [],
    artifact_refs: artifactRefs,
    followups: report?.followups ?? [],
    duration_ms: durationMs,
    ...runtime,
  };
}

export async function appendEvaluationRecord(ledgerPath, record) {
  await mkdir(path.dirname(ledgerPath), { recursive: true });
  await appendFile(ledgerPath, `${JSON.stringify(record)}\n`, "utf8");
}
