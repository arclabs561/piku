import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { explorerCallBudget, explorerHardCallLimit, explorerIdentity, explorerReportOutcome, safeRunId, validateSynthesis, writeRunManifest } from "./parallel-agent-eval.mjs";

const scriptsDir = path.dirname(fileURLToPath(import.meta.url));
const webUiDir = path.resolve(scriptsDir, "..");

test("explorers receive distinct surfaces and request IDs", () => {
  const runId = safeRunId("2026-08-08T12:34:56.000Z");
  const trace = explorerIdentity(runId, "coding_trace");
  const recovery = explorerIdentity(runId, "recovery");
  assert.notEqual(trace.surface, recovery.surface);
  assert.notEqual(trace.requestId, recovery.requestId);
  assert.match(trace.surface, /^qa-[a-z0-9-]+$/);
});

test("explorers get working budgets inside a separate hard runaway limit", () => {
  assert.equal(explorerCallBudget("coding_trace", {}), 40);
  assert.equal(explorerCallBudget("recovery", {}), 48);
  assert.equal(explorerCallBudget("recovery", { PIKU_EXPLORER_MAX_CALLS: "50" }), 50);
  assert.equal(explorerCallBudget("recovery", { PIKU_RECOVERY_MAX_CALLS: "44" }), 44);
  assert.equal(explorerHardCallLimit({}), 64);
  assert.equal(explorerHardCallLimit({ PIKU_EXPLORER_HARD_MAX_CALLS: "72" }), 72);
});

test("a blocked explorer cannot authorize product synthesis", () => {
  assert.deepEqual(explorerReportOutcome({ status: "blocked" }), {
    runStatus: "inconclusive",
    failureClass: "explorer_blocked",
  });
  assert.deepEqual(explorerReportOutcome({ status: "completed" }), {
    runStatus: "completed",
    failureClass: "none",
  });
});

test("synthesis rejects evidence IDs absent from raw packets", () => {
  const packets = [{ evidence: [{ id: "coding_trace:one" }] }, { evidence: [{ id: "recovery:one" }] }];
  const report = { verdict: "partial", evidence_ids: ["coding_trace:one", "unknown:id"], findings: [], followups: [] };
  assert.throws(() => validateSynthesis(report, packets), /unknown evidence IDs/);
});

test("synthesis rejects a supported verdict that contains a high finding", () => {
  const packets = [{ evidence: [{ id: "coding_trace:one" }] }];
  const report = {
    verdict: "supported",
    evidence_ids: ["coding_trace:one"],
    findings: [{ severity: "high", title: "contradiction", evidence_ids: ["coding_trace:one"] }],
    followups: [],
  };
  assert.throws(() => validateSynthesis(report, packets), /cannot contain a high-severity/);
});

test("orchestrator contract contains budgets, cleanup, isolation, and fresh synthesis", async () => {
  const [source, codexRuntime] = await Promise.all([
    readFile(path.join(scriptsDir, "parallel-agent-eval.mjs"), "utf8"),
    readFile(path.join(scriptsDir, "codex-exec.mjs"), "utf8"),
  ]);
  assert.match(source, /Promise\.all\(roles\.map/);
  assert.match(source, /budget_exceeded/);
  assert.match(source, /stopWithFallback/);
  assert.match(source, /progress calls=/);
  assert.match(source, /process\.kill\(-child\.pid/);
  assert.match(source, /forbidden_agent_action/);
  assert.match(source, /activeChildren/);
  assert.match(source, /cleanupSurface/);
  assert.match(source, /\{\{MANIFEST\}\}/);
  assert.match(codexRuntime, /--ephemeral/);
  assert.match(codexRuntime, /--ignore-user-config/);
  assert.match(codexRuntime, /--ignore-rules/);
  assert.match(codexRuntime, /playwright-mcp/);
  for (const file of ["explorer-coding-trace.md", "explorer-recovery.md", "synthesis.md", "explorer-report.schema.json", "synthesis-report.schema.json"])
    await readFile(path.join(webUiDir, "e2e", file), "utf8");
  const recoveryPrompt = await readFile(path.join(webUiDir, "e2e", "explorer-recovery.md"), "utf8");
  assert.match(recoveryPrompt, /dedicated,\s+non-destructive `browser_evaluate`/);
  assert.match(recoveryPrompt, /Do not combine verification, surface deletion/);
});

test("run manifest indexes role evidence and screenshots", async (t) => {
  const root = await import("node:fs/promises").then(({ mkdtemp }) => mkdtemp(path.join(process.env.TMPDIR || "/tmp", "piku-eval-")));
  t.after(async () => (await import("node:fs/promises")).rm(root, { recursive: true, force: true }));
  await (await import("node:fs/promises")).mkdir(path.join(root, "recovery"));
  await (await import("node:fs/promises")).writeFile(path.join(root, "recovery", "after.png"), "fixture");
  await (await import("node:fs/promises")).writeFile(path.join(root, "recovery", "events.jsonl"), "");
  await (await import("node:fs/promises")).writeFile(
    path.join(root, "recovery", "evidence.json"),
    JSON.stringify({ viewport: { width: 1440, height: 1000 } }),
  );
  const runtime = { subject_version: "0.1.0", subject_revision: "abc123" };
  const manifest = await writeRunManifest(root, "run-1", [{ role: "recovery", runStatus: "completed" }], null, runtime);
  assert.deepEqual(manifest.explorers.recovery.screenshots, ["recovery/after.png"]);
  assert.equal(manifest.explorers.recovery.events, "recovery/events.jsonl");
  assert.deepEqual(manifest.explorers.recovery.viewport, { width: 1440, height: 1000 });
  assert.deepEqual(manifest.runtime, runtime);
});
