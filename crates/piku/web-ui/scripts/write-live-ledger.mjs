import path from "node:path";

import { evaluationRecord } from "./evaluation-ledger.mjs";

export function classifyWriteLive({ result, runnerFailure, elapsedMs }) {
  if (result?.status === "completed") {
    return { runStatus: "completed", failureClass: "none", verdict: "supported" };
  }
  if (result?.status === "product_failure") {
    return {
      runStatus: "product_failure",
      failureClass: result.failure_class || "write_journey_product_failure",
      verdict: "not_supported",
    };
  }
  const message = runnerFailure?.message || "";
  if (/timed?\s*out|timeout/i.test(message))
    return { runStatus: "timeout", failureClass: "evaluator_timeout", verdict: null };
  if (/server|attestation|executable|startup|spawn/i.test(message))
    return { runStatus: "infrastructure_failure", failureClass: "write_runtime_unavailable", verdict: null };
  return {
    runStatus: "harness_failure",
    failureClass: result?.failure_class || "write_live_evidence_incomplete",
    verdict: null,
  };
}

export function writeLiveEvaluationRecord({
  runId,
  artifactDir,
  repoRoot,
  result,
  runnerFailure,
  durationMs,
  runtime,
}) {
  const classification = classifyWriteLive({ result, runnerFailure, elapsedMs: durationMs });
  const relativeArtifact = (name) => path.relative(repoRoot, path.join(artifactDir, name));
  const artifactRefs = ["manifest.json", "server.log", "result.json", "workspace-write-complete.png"]
    .filter((name) => name !== "workspace-write-complete.png" || result?.screenshot === true)
    .filter((name) => name !== "result.json" || result)
    .map(relativeArtifact);
  const report = classification.verdict === null ? null : {
    product_thesis: { verdict: classification.verdict },
    findings: result?.findings || [],
    followups: result?.followups || [],
  };
  const record = evaluationRecord({
    runId,
    surface: "workspace-write-live",
    stageId: "workspace-write-live",
    scenarioId: "workspace-write-reviewed-mutation",
    perspective: "authority",
    taskContract: "reviewed-single-file-mutation-with-durable-evidence",
    runStatus: classification.runStatus,
    failureClass: classification.failureClass,
    durationMs,
    report,
    artifactRefs,
    runtime,
  });
  record.evidence_ids = result?.evidence_ids || [];
  return record;
}
