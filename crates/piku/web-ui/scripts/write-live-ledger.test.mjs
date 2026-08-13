import assert from "node:assert/strict";
import test from "node:test";

import { assertEvaluationEnvelope } from "./evaluation-ledger.mjs";
import { classifyWriteLive, writeLiveEvaluationRecord } from "./write-live-ledger.mjs";

test("write-live classifications preserve causal failure classes", () => {
  assert.deepEqual(classifyWriteLive({ result: { status: "completed" } }), {
    runStatus: "completed", failureClass: "none", verdict: "supported",
  });
  assert.equal(classifyWriteLive({ result: { status: "product_failure", failure_class: "effect_mismatch" } }).runStatus, "product_failure");
  assert.equal(classifyWriteLive({ runnerFailure: new Error("server did not become ready") }).runStatus, "infrastructure_failure");
  assert.equal(classifyWriteLive({ runnerFailure: new Error("Playwright timeout") }).runStatus, "timeout");
  assert.equal(classifyWriteLive({ runnerFailure: new Error("Playwright exited 1") }).runStatus, "harness_failure");
});

test("write-live produces a valid shared v2 evaluation envelope", () => {
  const record = writeLiveEvaluationRecord({
    runId: "write-live-test",
    artifactDir: "/repo/.artifacts/workspace-write-live/run",
    repoRoot: "/repo",
    result: {
      status: "completed",
      screenshot: true,
      evidence_ids: ["write-live:exact-bytes", "write-live:reload"],
      findings: [],
      followups: [],
    },
    durationMs: 42,
    runtime: { evaluator_runtime: "playwright", evaluator_version: "test" },
  });
  assert.equal(assertEvaluationEnvelope(record), record);
  assert.equal(record.scenario_id, "workspace-write-reviewed-mutation");
  assert.equal(record.surface, "web");
  assert.equal(record.run_status, "completed");
  assert.deepEqual(record.evidence_ids, ["write-live:exact-bytes", "write-live:reload"]);
});
