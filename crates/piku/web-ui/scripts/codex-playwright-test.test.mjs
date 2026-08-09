import assert from "node:assert/strict";
import { access, mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import path from "node:path";
import {
  appendEvaluationRecord,
  assertEvaluationEnvelope,
  evaluationAmendment,
  evaluationRecord,
  evaluationRuntimeMetadata,
} from "./evaluation-ledger.mjs";

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

test("parallel evaluation separates causal mechanisms from verdicts", async () => {
  const files = await Promise.all([
    "explorer-coding-trace.md",
    "explorer-recovery.md",
    "synthesis.md",
    "explorer-report.schema.json",
    "synthesis-report.schema.json",
  ].map((name) => readFile(path.join(webUiDir, "e2e", name), "utf8")));
  const [tracePrompt, recoveryPrompt, synthesisPrompt, explorerText, synthesisText] = files;
  const explorerSchema = JSON.parse(explorerText);
  const synthesisSchema = JSON.parse(synthesisText);

  for (const prompt of [tracePrompt, recoveryPrompt, synthesisPrompt]) {
    assert.match(prompt, /mechanism/);
    assert.match(prompt, /prediction|predicts/);
    assert.match(prompt, /falsif/);
    assert.match(prompt, /confound/);
    assert.match(prompt, /alternative\s+explanation/);
    assert.match(prompt, /validity/);
  }
  for (const schema of [explorerSchema, synthesisSchema]) {
    const causal = schema.properties.causal_assessment;
    assert.ok(causal, "causal_assessment must be explicit");
    assert.ok(schema.required.includes("causal_assessment"));
    for (const field of ["mechanism", "prediction", "falsifier", "observed_outcome", "disposition", "confounders", "alternative_explanations"])
      assert.match(JSON.stringify(schema.$defs.hypothesis), new RegExp(`"${field}"`));
    for (const field of ["status", "compromised_by", "rationale", "evidence_ids"])
      assert.match(JSON.stringify(schema.$defs.validity), new RegExp(`"${field}"`));
    const serialized = JSON.stringify(schema);
    for (const unsupported of ["oneOf", "anyOf", "allOf"])
      assert.doesNotMatch(serialized, new RegExp(`"${unsupported}"`));
  }
  assert.match(synthesisPrompt, /none may substitute for mechanism\s+evidence/);
  assert.match(synthesisPrompt, /Do not infer source-level causation/);
  for (const field of ["producer_event_id", "producer_tool"])
    assert.match(explorerText, new RegExp(`"${field}"`));
  assert.match(synthesisPrompt, /producer binding compromises visual evidence/);
  assert.match(recoveryPrompt, /enumerate the exact visible text and status values/);
  assert.match(recoveryPrompt, /do not use absence from a fixed keyword regex/);
  assert.match(recoveryPrompt, /screenshot and\s+predicate disagree/);
  assert.match(synthesisPrompt, /resolve each\s+screenshot–predicate contradiction/);
  assert.match(synthesisPrompt, /Keyword-regex absence cannot override visible\s+screenshot text/);
  assert.match(tracePrompt, /Execution trace.*transient[\s\S]*not an authored or persisted workspace card/);
  assert.match(tracePrompt, /create a small seeded\s+page[\s\S]*submit a narrow heading-only change/);
  assert.match(tracePrompt, /initial empty-to-document creation is setup, not evidence/);
  assert.match(tracePrompt, /aggregate console count alone cannot support a finding/i);
  assert.match(recoveryPrompt, /raw\s+error count or HTTP status[\s\S]*cannot support a product\s+finding/);
  assert.match(synthesisPrompt, /aggregate console counts or generic DOM article counts/);
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
  assert.equal(record.record_kind, "stage");
  assert.equal(record.stage_id, "synthesis");
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

test("ledger rejects invalid records before creating or appending a file", async () => {
  const directory = await mkdtemp(path.join(tmpdir(), "piku-invalid-ledger-"));
  const ledger = path.join(directory, "nested", "runs.jsonl");
  try {
    const record = evaluationRecord({
      runId: "invalid-run",
      runStatus: "completed",
      failureClass: "none",
      durationMs: 1,
    });
    record.run_status = "banana";
    assert.throws(() => assertEvaluationEnvelope(record), /run_status is invalid/);
    await assert.rejects(appendEvaluationRecord(ledger, record), /run_status is invalid/);
    await assert.rejects(access(ledger));
  } finally {
    await rm(directory, { recursive: true });
  }
});

test("evaluation amendments retain their target and causal basis", () => {
  const target = evaluationRecord({
    runId: "amended-run",
    surface: "qa-stage",
    runStatus: "completed",
    failureClass: "none",
    durationMs: 10,
  });
  const record = evaluationAmendment({
    targetRecord: target,
    action: "invalidate",
    reasonCode: "judge_contamination",
    scope: { evidence_ids: [], finding_refs: [], verdict: true },
    basisRefs: ["audit.json"],
    basisHashes: [`sha256:${"a".repeat(64)}`],
    actor: "causal-auditor",
    toolVersion: "piku-audit/1",
    eventId: "amendment-event-1",
    recordedAt: "2026-08-09T00:00:00.000Z",
  });
  assert.equal(record.record_kind, "amendment");
  assert.equal(record.target_run_id, target.run_id);
  assert.equal(record.target_stage_id, target.stage_id);
  assert.notEqual(record.stage_id, target.stage_id);
  assert.equal(record.event_id, "amendment-event-1");
  assert.equal(record.contract_version, "piku-evaluation-amendment-v1");
  assert.equal(record.product_verdict, null);
  assert.deepEqual(record.basis_refs, ["audit.json"]);
  assert.throws(
    () => assertEvaluationEnvelope({ ...record, basis_hashes: ["sha256:abc"] }),
    /64 lowercase hex/,
  );
  assert.throws(
    () => assertEvaluationEnvelope({ ...record, stage_id: target.stage_id }),
    /must not reuse its target stage_id/,
  );
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
