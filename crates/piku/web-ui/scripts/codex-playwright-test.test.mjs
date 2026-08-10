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
  projectReportIdentity,
} from "./evaluation-ledger.mjs";
import { validateRequiredScreenshots } from "./playwright-authority.mjs";

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
  assert.match(prompt, /Number findings locally as `f1`/);
  assert.match(prompt, /`retest_of` only to an exact prior fully scoped obligation ID/);
  assert.match(prompt, /failed locator proves only that the locator failed/i);
  assert.match(prompt, /evaluator-generated errors.*compromised evidence/is);
  assert.match(prompt, /screenshot pixels and a predicate disagree/i);
  assert.match(prompt, /visually\s+apparent clipping, overlap, illegible density/i);
  const parsedSchema = JSON.parse(schema);
  for (const field of ["intent", "expected_outcome", "actions", "observation", "consequence", "next_probe"])
    assert.ok(parsedSchema.properties.journey.items.required.includes(field));
  assert.match(prompt, /expectation-gap probe/i);
  assert.match(prompt, /bounded observational self-talk/i);
  assert.equal(parsedSchema.properties.findings.items.properties.id.pattern, "^f[1-9][0-9]*$");
  for (const field of ["id", "finding_ids", "evidence_ids", "retest_of"])
    assert.ok(parsedSchema.properties.followups.items.required.includes(field));
});

test("single-agent QA uses the shared bounded Playwright authority", async () => {
  const runner = await readFile(
    path.join(scriptsDir, "codex-playwright-test.mjs"),
    "utf8",
  );

  assert.match(runner, /from "\.\/playwright-authority\.mjs"/);
  assert.match(runner, /withPlaywrightAuthority\(baseArgs, playwrightOutputDir\)/);
  assert.match(runner, /replaceAll\("\{\{RUN_DIR\}\}", playwrightOutputDir\)/);
  assert.doesNotMatch(runner, /browser_run_code_unsafe/);
});

test("single-agent QA binds each required screenshot to one absolute producer", () => {
  const root = "/tmp/piku-single/playwright-output";
  const event = (filename) => ({
    type: "item.completed",
    item: {
      type: "mcp_tool_call", server: "playwright", tool: "browser_take_screenshot",
      status: "completed", error: null, arguments: { filename },
    },
  });
  assert.deepEqual(
    validateRequiredScreenshots([event(`${root}/one.png`)], root, ["one.png"]),
    [`${root}/one.png`],
  );
  assert.throws(
    () => validateRequiredScreenshots([event("one.png")], root, ["one.png"]),
    /absolute filenames/,
  );
  assert.throws(
    () => validateRequiredScreenshots([event(`${root}/one.png`), event(`${root}/one.png`)], root, ["one.png"]),
    /exactly one successful producer/,
  );
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
  assert.ok(explorerSchema.required.includes("probes"));
  for (const field of ["intent", "expected_outcome", "action", "observed_outcome", "consequence", "next_probe", "evidence_ids"])
    assert.ok(explorerSchema.properties.probes.items.required.includes(field));
  for (const prompt of [tracePrompt, recoveryPrompt]) {
    assert.match(prompt, /expectation-gap probes/i);
    assert.match(prompt, /bounded observational self-talk/i);
  }
  assert.match(synthesisPrompt, /do not treat a probe alone as proof/i);
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
  assert.match(tracePrompt, /aggregate console count alone cannot\s+support a finding/i);
  assert.match(recoveryPrompt, /raw\s+error count or HTTP status[\s\S]*cannot support a product\s+finding/);
  assert.match(recoveryPrompt, /semantic selected-state predicate[^.]*`true`/i);
  assert.match(recoveryPrompt, /selection as transient interaction state/i);
  assert.match(recoveryPrompt, /distinctive, non-default canvas position/i);
  assert.match(recoveryPrompt, /deterministic delayed-provider fixture/i);
  assert.match(synthesisPrompt, /aggregate console counts or generic DOM article counts/);
  for (const prompt of [tracePrompt, recoveryPrompt, synthesisPrompt]) {
    assert.match(prompt, /evaluator(?:-generated)? noise/);
    assert.match(prompt, /product impact/);
  }
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
  assert.ok(schema.properties.schema_version.enum.includes(record.schema_version));
  assert.ok(schema.properties.surface.enum.includes(record.surface));
  assert.ok(schema.properties.run_status.enum.includes(record.run_status));
});

test("v2 report identities are scoped by run and stage without prose matching", () => {
  const report = {
    findings: [
      { id: "f1", title: "same prose" },
      { id: "f2", title: "same prose" },
    ],
    followups: [
      {
        id: "o1", kind: "retest", priority: "high", title: "same prose",
        rationale: "same rationale", perspective: null, evidence_ids: [],
        finding_ids: ["f2"], retest_of: "prior:stage:obligation:o7",
      },
    ],
  };
  const projected = projectReportIdentity(report, "run-a", "synthesis");
  assert.deepEqual(projected.findingRefs, [
    "run-a:synthesis:finding:f1",
    "run-a:synthesis:finding:f2",
  ]);
  assert.deepEqual(projected.followups[0], {
    obligation_id: "run-a:synthesis:obligation:o1",
    kind: "retest",
    priority: "high",
    title: "same prose",
    rationale: "same rationale",
    perspective: null,
    evidence_ids: [],
    finding_refs: ["run-a:synthesis:finding:f2"],
    retest_of: "prior:stage:obligation:o7",
  });
});

test("single-agent reports become valid v2 scoped ledger records", () => {
  const report = {
    product_thesis: { verdict: "partial" },
    findings: [{ id: "f1", title: "Attribution gap" }],
    followups: [{
      id: "o1", kind: "retest", priority: "high", title: "Retest attribution",
      rationale: "Fresh evidence is needed", perspective: "trust",
      evidence_ids: [], finding_ids: ["f1"], retest_of: null,
    }],
  };
  const record = evaluationRecord({
    runId: "single-run", surface: "qa-single", runStatus: "product_failure",
    failureClass: "high_impact_finding", durationMs: 12, report,
  });
  assert.equal(record.schema_version, 2);
  assert.deepEqual(record.finding_refs, ["single-run:qa-single:finding:f1"]);
  assert.equal(record.followups[0].obligation_id, "single-run:qa-single:obligation:o1");
  assert.deepEqual(record.followups[0].finding_refs, record.finding_refs);
  assert.doesNotThrow(() => assertEvaluationEnvelope(record));
});

test("v2 report identities require explicit basis and valid local references", () => {
  const base = {
    findings: [{ id: "f1" }],
    followups: [{
      id: "o1", kind: "todo", priority: "high", title: "t", rationale: "r",
      perspective: null, evidence_ids: [], finding_ids: [], retest_of: null,
    }],
  };
  assert.throws(() => projectReportIdentity(base, "run", "stage"), /must cite evidence_ids or finding_ids/);
  base.followups[0].finding_ids = ["f2"];
  assert.throws(() => projectReportIdentity(base, "run", "stage"), /unknown finding ID f2/);
});

test("shared validator reads v1 records and enforces v2 obligation integrity", () => {
  const current = evaluationRecord({
    runId: "run-v2", runStatus: "completed", failureClass: "none", durationMs: 1,
  });
  const legacy = structuredClone(current);
  legacy.schema_version = 1;
  delete legacy.finding_refs;
  assert.doesNotThrow(() => assertEvaluationEnvelope(legacy));

  const invalid = structuredClone(current);
  invalid.followups = [{
    obligation_id: "run-v2:synthesis:obligation:o1", kind: "todo", priority: "high",
    title: "t", rationale: "r", perspective: null, evidence_ids: [],
    finding_refs: [], retest_of: null,
  }];
  assert.throws(() => assertEvaluationEnvelope(invalid), /must cite evidence_ids or finding_refs/);
  invalid.followups[0].finding_refs = ["run-v2:synthesis:finding:f9"];
  assert.throws(() => assertEvaluationEnvelope(invalid), /unknown top-level finding ref/);

  const duplicate = structuredClone(current);
  duplicate.followups = [1, 2].map(() => ({
    obligation_id: "run-v2:synthesis:obligation:o1", kind: "todo", priority: "high",
    title: "t", rationale: "r", perspective: null, evidence_ids: ["e1"],
    finding_refs: [], retest_of: null,
  }));
  assert.throws(() => assertEvaluationEnvelope(duplicate), /obligation_id values must be unique/);
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
  assert.equal(record.contract_version, "piku-evaluation-amendment-v2");
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
  const commands = [];
  const runtime = evaluationRuntimeMetadata(repoRoot, (command, args, cwd) => {
    commands.push({ command, args, cwd });
    if (command === "git" && args[0] === "rev-parse") return "a".repeat(40);
    if (command === "git" && args[0] === "status") return "";
    if (command === "codex") return "codex-cli 0.146.0";
    throw new Error(`unexpected command: ${command}`);
  });
  assert.equal(runtime.subject_version, "0.1.0");
  assert.equal(runtime.subject_revision, "a".repeat(40));
  assert.equal(runtime.subject_dirty, false);
  assert.equal(runtime.evaluator_version, "codex-cli 0.146.0");
  assert.equal(runtime.explorer_model, "gpt-5.6-sol");
  assert.equal(runtime.evaluation_contract, "piku-evaluation-v2");
  assert.deepEqual(commands, [
    { command: "git", args: ["rev-parse", "HEAD"], cwd: repoRoot },
    { command: "git", args: ["status", "--porcelain=v1"], cwd: repoRoot },
    { command: "codex", args: ["--version"], cwd: repoRoot },
  ]);
});
