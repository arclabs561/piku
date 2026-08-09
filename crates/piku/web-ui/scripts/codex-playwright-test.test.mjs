import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { evaluationRecord, evaluationRuntimeMetadata } from "./evaluation-ledger.mjs";

const scriptsDir = path.dirname(fileURLToPath(import.meta.url));
const webUiDir = path.resolve(scriptsDir, "..");
const repoRoot = path.resolve(webUiDir, "../../..");

test("agent QA contract evaluates the product thesis, not only UI mechanics", async () => {
  const [prompt, schema, runner] = await Promise.all([
    readFile(path.join(webUiDir, "e2e", "codex-live-qa.md"), "utf8"),
    readFile(path.join(webUiDir, "e2e", "agent-report.schema.json"), "utf8"),
    readFile(path.join(scriptsDir, "codex-playwright-test.mjs"), "utf8"),
  ]);

  const dimensionNames = [
    "task_comprehension",
    "action_provenance",
    "state_visibility",
    "context_control",
    "rerun_semantics",
    "recovery",
    "authority_clarity",
    "spatial_utility",
  ];
  for (const name of dimensionNames) {
    assert.match(prompt, new RegExp(`\\b${name}\\b`));
    assert.match(schema, new RegExp(`"${name}"`));
    assert.match(runner, new RegExp(`"${name}"`));
  }
  assert.match(prompt, /one chat turn\s+and one selected-page-source change/);
  assert.match(prompt, /status.*evaluation journey/s);
  assert.match(prompt, /product_thesis\.verdict.*product/s);
  assert.match(runner, /supported thesis verdict contradicts dimension evidence/);
});

test("web evaluator records timeout separately from product failure", () => {
  const record = evaluationRecord({
    runId: "run-1",
    runStatus: "timeout",
    failureClass: "evaluator_timeout",
    durationMs: 900_000,
    artifactRefs: ["events.jsonl"],
  });

  assert.equal(record.surface, "web");
  assert.equal(record.run_status, "timeout");
  assert.equal(record.failure_class, "evaluator_timeout");
  assert.equal(record.product_verdict, null);
  assert.deepEqual(record.followups, []);
});

test("web records satisfy the shared CLI and web envelope", async () => {
  const schema = JSON.parse(
    await readFile(
      path.join(repoRoot, "eval", "evaluation-envelope.schema.json"),
      "utf8",
    ),
  );
  const record = evaluationRecord({
    runId: "run-2",
    runStatus: "completed",
    failureClass: "none",
    durationMs: 1,
  });

  for (const field of schema.required) {
    assert.ok(Object.hasOwn(record, field), `missing shared field: ${field}`);
  }
  assert.equal(record.schema_version, schema.properties.schema_version.const);
  assert.ok(schema.properties.surface.enum.includes(record.surface));
  assert.ok(schema.properties.run_status.enum.includes(record.run_status));
});

test("live evaluation metadata records exact subject and evaluator versions", () => {
  const runtime = evaluationRuntimeMetadata(repoRoot);
  assert.equal(runtime.subject_version, "0.1.0");
  assert.match(runtime.subject_revision, /^[0-9a-f]{40}$/);
  assert.equal(typeof runtime.subject_dirty, "boolean");
  assert.match(runtime.evaluator_version, /^codex-cli \d+/);
  assert.equal(runtime.explorer_model, "gpt-5.6-sol");
  assert.equal(runtime.evaluation_contract, "piku-evaluation-v1");
});
