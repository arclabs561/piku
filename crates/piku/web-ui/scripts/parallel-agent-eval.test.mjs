import assert from "node:assert/strict";
import { mkdir, mkdtemp, readFile, rm, symlink, writeFile } from "node:fs/promises";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { PLAYWRIGHT_TOOLS, attestEvidenceArtifacts, explorerCallBudget, explorerHardCallLimit, explorerIdentity, explorerReportOutcome, loadValidatedExplorerRun, nextSynthesisAttemptDir, playwrightAuthorityViolation, renderExplorerPrompt, restrictSynthesisPrompt, safeRunId, screenshotProducerIndex, traceAuthorityViolation, validateExplorerReport, validateSynthesis, withPlaywrightAuthority, writeRunManifest } from "./parallel-agent-eval.mjs";

const scriptsDir = path.dirname(fileURLToPath(import.meta.url));
const webUiDir = path.resolve(scriptsDir, "..");

const coverage = (coding = ["coding_trace:one"], recovery = ["recovery:one"]) => ({
  coding_trace: { status: "assessed", rationale: "trace covered", evidence_ids: coding },
  recovery: { status: "assessed", rationale: "recovery covered", evidence_ids: recovery },
});
const causal = (evidenceIds = ["coding_trace:one", "recovery:one"]) => ({
  hypotheses: [{ id: "h1", disposition: "supported", evidence_ids: evidenceIds }],
  validity: { status: "valid", compromised_by: [], evidence_ids: evidenceIds },
});

const screenshotEvent = (filename, { id = "item-shot", status = "completed", error = null } = {}) => JSON.stringify({
  type: "item.completed",
  item: { id, type: "mcp_tool_call", server: "playwright", tool: "browser_take_screenshot", arguments: { filename }, status, error },
});

test("explorers receive distinct surfaces and request IDs", () => {
  const runId = safeRunId("2026-08-08T12:34:56.000Z");
  const trace = explorerIdentity(runId, "coding_trace");
  const recovery = explorerIdentity(runId, "recovery");
  assert.notEqual(trace.surface, recovery.surface);
  assert.notEqual(trace.requestId, recovery.requestId);
  assert.match(trace.surface, /^qa-[a-z0-9-]+$/);
});

test("explorers get working budgets inside a separate hard runaway limit", () => {
  assert.equal(explorerCallBudget("coding_trace", {}), 48);
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
  const packets = [{ evidence: [{ id: "coding_trace:one", kind: "predicate" }] }, { evidence: [{ id: "recovery:one", kind: "predicate" }] }];
  const report = { verdict: "partial", evidence_ids: ["coding_trace:one", "unknown:id"], causal_assessment: causal(), coverage: coverage(), findings: [], followups: [] };
  assert.throws(() => validateSynthesis(report, packets), /unknown evidence IDs/);
});

test("synthesis rejects a supported verdict that contains a high finding", () => {
  const packets = [{ evidence: [{ id: "coding_trace:one", kind: "predicate" }] }, { evidence: [{ id: "recovery:one", kind: "predicate" }] }];
  const report = {
    verdict: "supported",
    evidence_ids: ["coding_trace:one", "recovery:one"],
    causal_assessment: causal(),
    coverage: coverage(),
    findings: [{ severity: "high", title: "contradiction", modality: "persistence", evidence_ids: ["coding_trace:one"] }],
    followups: [],
  };
  assert.throws(() => validateSynthesis(report, packets), /cannot contain a high-severity/);
});

test("artifact attestation records digest, size, and media type", async (t) => {
  const root = await mkdtemp(path.join(process.env.TMPDIR || "/tmp", "piku-evidence-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  const artifact = path.join(root, "screen.png");
  await writeFile(artifact, Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]));
  const report = { evidence: [{ id: "coding_trace:screen", kind: "screenshot", artifact }] };
  await attestEvidenceArtifacts(report, root, root, `${screenshotEvent(artifact)}\n`);
  assert.match(report.evidence[0].artifact, /^artifacts\/[a-f0-9]{16}\.png$/);
  assert.deepEqual(report.evidence[0].artifact_metadata, {
    sha256: "4c4b6a3be1314ab86138bef4314dde022e600960d8689a2c8f8631802d20dab6",
    size_bytes: 8,
    media_type: "image/png",
    producer_event_id: "item-shot",
    producer_tool: "browser_take_screenshot",
  });
});

test("artifact attestation rejects paths outside the explorer directory", async (t) => {
  const root = await mkdtemp(path.join(process.env.TMPDIR || "/tmp", "piku-evidence-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  const outside = `${root}-outside.png`;
  await writeFile(outside, "not png");
  t.after(() => rm(outside, { force: true }));
  const report = { evidence: [{ id: "coding_trace:escape", kind: "screenshot", artifact: outside }] };
  await assert.rejects(attestEvidenceArtifacts(report, root, root, `${screenshotEvent(outside)}\n`), /outside its allowed output directory/);
});

test("artifact attestation resolves symlinks before enforcing the output boundary", async (t) => {
  const root = await mkdtemp(path.join(process.env.TMPDIR || "/tmp", "piku-evidence-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  const output = path.join(root, "output");
  await mkdir(output);
  const outside = path.join(root, "outside.png");
  await writeFile(outside, Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]));
  await symlink(outside, path.join(output, "linked.png"));
  const report = { evidence: [{ id: "recovery:escape", kind: "screenshot", artifact: "linked.png" }] };
  await assert.rejects(attestEvidenceArtifacts(report, root, output, `${screenshotEvent("linked.png")}\n`), /outside its allowed output directory/);
});

test("screenshot attestation rejects missing, failed, and duplicate producers", async (t) => {
  const root = await mkdtemp(path.join(process.env.TMPDIR || "/tmp", "piku-producer-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  const artifact = path.join(root, "screen.png");
  await writeFile(artifact, Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]));
  const makeReport = () => ({ evidence: [{ id: "coding_trace:screen", kind: "screenshot", artifact }] });
  await assert.rejects(attestEvidenceArtifacts(makeReport(), root, root, ""), /lacks a successful producer/);
  await assert.rejects(
    attestEvidenceArtifacts(makeReport(), root, root, `${screenshotEvent(artifact, { status: "failed", error: { message: "nope" } })}\n`),
    /only failed producer/,
  );
  const duplicate = `${screenshotEvent(artifact, { id: "shot-1" })}\n${screenshotEvent(artifact, { id: "shot-2" })}\n`;
  await assert.rejects(attestEvidenceArtifacts(makeReport(), root, root, duplicate), /duplicate successful producer/);
});

test("screenshot producer index normalizes relative event filenames", () => {
  const index = screenshotProducerIndex(`${screenshotEvent("nested/screen.png")}\n`, "/tmp/role-output");
  assert.deepEqual(index.get("/tmp/role-output/nested/screen.png"), [{
    id: "item-shot", tool: "browser_take_screenshot", successful: true,
  }]);
});

test("Playwright authority config omits unsafe code and pins an output directory", () => {
  const args = ["exec", "--config", 'mcp_servers.playwright.args=["playwright-mcp"]', "prompt"];
  const hardened = withPlaywrightAuthority(args, "/tmp/piku-role-output");
  const encoded = hardened.join("\n");
  assert.match(encoded, /--output-dir/);
  assert.match(encoded, /enabled_tools=/);
  assert.equal(PLAYWRIGHT_TOOLS.includes("browser_run_code_unsafe"), false);
  assert.equal(encoded.includes("browser_run_code_unsafe"), false);
});

test("explorer prompt and Playwright MCP use the exact same output directory", () => {
  const outputDir = "/tmp/piku-role/playwright-output";
  const prompt = renderExplorerPrompt("write={{RUN_DIR}}", {
    baseUrl: new URL("http://127.0.0.1:9090"), surface: "qa-role", requestId: "run:role",
    playwrightOutputDir: outputDir, targetCalls: 40, maxSnapshots: 6,
  });
  const args = withPlaywrightAuthority(
    ["exec", "--config", 'mcp_servers.playwright.args=["playwright-mcp"]', prompt],
    outputDir,
  );
  const setting = args.find((arg) => arg.startsWith("mcp_servers.playwright.args="));
  const configured = JSON.parse(setting.slice(setting.indexOf("=") + 1));
  const configuredOutput = configured[configured.indexOf("--output-dir") + 1];
  assert.equal(prompt, `write=${configuredOutput}`);
  assert.equal(configuredOutput, path.resolve(outputDir));
});

test("trace authority fails closed for unsafe, unknown, and malformed events", () => {
  const unsafe = { type: "item.started", item: { type: "mcp_tool_call", server: "playwright", tool: "browser_run_code_unsafe" } };
  assert.equal(playwrightAuthorityViolation(unsafe), "forbidden_playwright_tool");
  assert.equal(traceAuthorityViolation(`${JSON.stringify(unsafe)}\n`), "forbidden_playwright_tool");
  assert.equal(traceAuthorityViolation("not-json\n"), "invalid_event_trace");
  const safe = { type: "item.completed", item: { type: "mcp_tool_call", server: "playwright", tool: "browser_snapshot" } };
  assert.equal(traceAuthorityViolation(`${JSON.stringify(safe)}\n`), null);
});

test("explorer causal citations and producer provenance fail closed", () => {
  const base = {
    evidence: [{ id: "coding_trace:one", kind: "predicate", artifact: null, artifact_metadata: null }],
    findings: [],
    causal_assessment: {
      hypotheses: [{ id: "h1", disposition: "supported", evidence_ids: [] }],
      validity: { status: "valid", compromised_by: [], evidence_ids: [] },
    },
  };
  assert.throws(() => validateExplorerReport(base), /tested causal hypothesis lacks evidence/);
  base.causal_assessment.hypotheses[0].disposition = "not_tested";
  base.causal_assessment.validity = { status: "compromised", compromised_by: [], evidence_ids: ["coding_trace:one"] };
  assert.throws(() => validateExplorerReport(base), /must name causes/);
  base.causal_assessment.validity.compromised_by = ["backend unavailable"];
  base.evidence.push({ id: "coding_trace:screen", kind: "screenshot", artifact: "artifacts/x.png", artifact_metadata: { producer_event_id: null, producer_tool: null } });
  assert.throws(() => validateExplorerReport(base), /lacks producer provenance/);
});

test("synthesis causal evidence must be known and load-bearing", () => {
  const packets = [{ evidence: [{ id: "coding_trace:one", kind: "predicate" }] }, { evidence: [{ id: "recovery:one", kind: "predicate" }] }];
  const report = {
    verdict: "partial",
    evidence_ids: ["coding_trace:one", "recovery:one"],
    causal_assessment: causal(["unknown:id"]),
    coverage: coverage(), findings: [], followups: [],
  };
  assert.throws(() => validateSynthesis(report, packets), /causal hypothesis cites unknown evidence/);
  report.causal_assessment = causal(["coding_trace:one"]);
  report.causal_assessment.validity.evidence_ids = ["unknown:id"];
  assert.throws(() => validateSynthesis(report, packets), /causal validity cites unknown evidence/);
  report.causal_assessment = causal(["coding_trace:one"]);
  report.evidence_ids = ["recovery:one"];
  assert.throws(() => validateSynthesis(report, packets), /causal hypothesis evidence is absent from verdict evidence/);
  report.causal_assessment.hypotheses[0].evidence_ids = ["recovery:one"];
  report.causal_assessment.validity.evidence_ids = ["coding_trace:one"];
  assert.throws(() => validateSynthesis(report, packets), /causal validity evidence is absent from verdict evidence/);
});

test("synthesis findings require modality evidence and verdict coverage", () => {
  const packets = [
    { evidence: [{ id: "coding_trace:dom", kind: "dom", artifact_metadata: null }] },
    { evidence: [{ id: "recovery:state", kind: "predicate", artifact_metadata: null }] },
  ];
  const report = {
    verdict: "partial",
    evidence_ids: ["coding_trace:dom", "recovery:state"],
    causal_assessment: causal(["coding_trace:dom", "recovery:state"]),
    coverage: coverage(["coding_trace:dom"], ["recovery:state"]),
    findings: [{ severity: "medium", title: "visual overlap", modality: "visual", evidence_ids: ["coding_trace:dom"] }],
    followups: [],
  };
  assert.throws(() => validateSynthesis(report, packets), /lacks visual-appropriate evidence/);
  report.findings[0] = { severity: "medium", title: "state loss", modality: "persistence", evidence_ids: ["recovery:state"] };
  report.evidence_ids = ["coding_trace:dom"];
  assert.throws(() => validateSynthesis(report, packets), /absent from verdict evidence/);
});

test("visual findings require an attested PNG screenshot", () => {
  const packets = [
    { evidence: [{ id: "coding_trace:screen", kind: "screenshot", artifact_metadata: null }] },
    { evidence: [{ id: "recovery:state", kind: "predicate", artifact_metadata: null }] },
  ];
  const report = {
    verdict: "partial",
    evidence_ids: ["coding_trace:screen", "recovery:state"],
    causal_assessment: causal(["coding_trace:screen", "recovery:state"]),
    coverage: coverage(["coding_trace:screen"], ["recovery:state"]),
    findings: [{ severity: "medium", title: "visual overlap", modality: "visual", evidence_ids: ["coding_trace:screen"] }],
    followups: [],
  };
  assert.throws(() => validateSynthesis(report, packets), /unattested screenshot/);
});

test("synthesis accepts cited, attested, modality-appropriate evidence from both perspectives", () => {
  const packets = [
    { evidence: [{ id: "coding_trace:screen", kind: "screenshot", artifact: "screen.png", artifact_metadata: { sha256: "a".repeat(64), size_bytes: 11, media_type: "image/png", producer_event_id: "shot-1", producer_tool: "browser_take_screenshot" } }] },
    { evidence: [{ id: "recovery:state", kind: "predicate", artifact: null, artifact_metadata: null }] },
  ];
  const report = {
    verdict: "partial",
    evidence_ids: ["coding_trace:screen", "recovery:state"],
    causal_assessment: causal(["coding_trace:screen", "recovery:state"]),
    coverage: coverage(["coding_trace:screen"], ["recovery:state"]),
    findings: [
      { severity: "medium", title: "visual overlap", modality: "visual", evidence_ids: ["coding_trace:screen"] },
      { severity: "medium", title: "state loss", modality: "persistence", evidence_ids: ["recovery:state"] },
    ],
    followups: [],
  };
  assert.doesNotThrow(() => validateSynthesis(report, packets));
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
  const synthesisPrompt = await readFile(path.join(webUiDir, "e2e", "synthesis.md"), "utf8");
  const synthesisSchema = await readFile(path.join(webUiDir, "e2e", "synthesis-report.schema.json"), "utf8");
  const explorerSchema = await readFile(path.join(webUiDir, "e2e", "explorer-report.schema.json"), "utf8");
  assert.match(synthesisPrompt, /Classify every finding by evidence modality/);
  assert.match(synthesisPrompt, /coverage for both perspectives/);
  assert.match(synthesisSchema, /"modality"/);
  assert.match(synthesisSchema, /"coverage"/);
  assert.match(explorerSchema, /"artifact_metadata"/);
  const recoveryPrompt = await readFile(path.join(webUiDir, "e2e", "explorer-recovery.md"), "utf8");
  assert.match(recoveryPrompt, /dedicated,\s+non-destructive `browser_evaluate`/);
  assert.match(recoveryPrompt, /Do not combine verification, surface deletion/);
  assert.match(recoveryPrompt, /full\s+absolute filename below `\{\{RUN_DIR\}\}`/);
  assert.match(recoveryPrompt, /never overwrite or recapture/);
  assert.match(recoveryPrompt, /semantic selected-state predicate[^.]*`true` before reload/i);
  assert.match(recoveryPrompt, /move\s+each card to a distinctive, non-default canvas position/i);
  assert.match(recoveryPrompt, /saved canvas coordinates[^.]*before\s+and after reload/i);
  assert.match(recoveryPrompt, /deterministic delayed-provider fixture/i);
  assert.match(recoveryPrompt, /do not treat a response\s+that merely completed quickly as cancellation evidence/i);
  const codingPrompt = await readFile(path.join(webUiDir, "e2e", "explorer-coding-trace.md"), "utf8");
  assert.match(codingPrompt, /Clicking empty canvas space opens the card\s+creation menu/);
  assert.match(codingPrompt, /unique full absolute filename/);
});

test("Codex response schemas avoid unsupported composition keywords", async () => {
  for (const file of ["explorer-report.schema.json", "synthesis-report.schema.json"]) {
    const schema = JSON.parse(await readFile(path.join(webUiDir, "e2e", file), "utf8"));
    const encoded = JSON.stringify(schema);
    for (const keyword of ["oneOf", "anyOf", "allOf"]) {
      assert.equal(encoded.includes(`\"${keyword}\"`), false, `${file} contains unsupported ${keyword}`);
    }
  }
});

test("run manifest indexes role evidence and screenshots", async (t) => {
  const root = await import("node:fs/promises").then(({ mkdtemp }) => mkdtemp(path.join(process.env.TMPDIR || "/tmp", "piku-eval-")));
  t.after(async () => (await import("node:fs/promises")).rm(root, { recursive: true, force: true }));
  await (await import("node:fs/promises")).mkdir(path.join(root, "recovery"));
  await (await import("node:fs/promises")).writeFile(path.join(root, "recovery", "after.png"), "fixture");
  await (await import("node:fs/promises")).writeFile(path.join(root, "recovery", "events.jsonl"), "");
  await (await import("node:fs/promises")).writeFile(
    path.join(root, "recovery", "evidence.json"),
    JSON.stringify({
      viewport: { width: 1440, height: 1000 },
      evidence: [{ kind: "screenshot", artifact: "artifacts/attested.png" }],
    }),
  );
  const runtime = { subject_version: "0.1.0", subject_revision: "abc123" };
  const manifest = await writeRunManifest(root, "run-1", [{ role: "recovery", runStatus: "completed" }], null, runtime);
  assert.deepEqual(manifest.explorers.recovery.screenshots, ["recovery/artifacts/attested.png"]);
  assert.equal(manifest.explorers.recovery.events, "recovery/events.jsonl");
  assert.deepEqual(manifest.explorers.recovery.viewport, { width: 1440, height: 1000 });
  assert.deepEqual(manifest.runtime, runtime);
});

test("run manifest safely indexes an absolute screenshot below the run", async (t) => {
  const root = await import("node:fs/promises").then(({ mkdtemp }) => mkdtemp(path.join(process.env.TMPDIR || "/tmp", "piku-eval-")));
  t.after(async () => (await import("node:fs/promises")).rm(root, { recursive: true, force: true }));
  const roleDir = path.join(root, "recovery");
  const screenshot = path.join(roleDir, "playwright-output", "after.png");
  await (await import("node:fs/promises")).mkdir(path.dirname(screenshot), { recursive: true });
  await (await import("node:fs/promises")).writeFile(screenshot, "fixture");
  await (await import("node:fs/promises")).writeFile(path.join(roleDir, "evidence.json"), JSON.stringify({
    evidence: [
      { kind: "screenshot", artifact: screenshot },
      { kind: "screenshot", artifact: "/outside/untrusted.png" },
    ],
  }));
  const manifest = await writeRunManifest(root, "run-absolute", [{ role: "recovery", runStatus: "harness_failure" }]);
  assert.deepEqual(manifest.explorers.recovery.screenshots, ["recovery/playwright-output/after.png"]);
});

async function writeResumeFixture(root, runId, statuses = { coding_trace: "completed", recovery: "completed" }) {
  const explorers = {};
  for (const role of ["coding_trace", "recovery"]) {
    const roleDir = path.join(root, role);
    await mkdir(path.join(roleDir, "playwright-output"), { recursive: true });
    const evidenceId = `${role}:one`;
    await writeFile(path.join(roleDir, "evidence.json"), JSON.stringify({
      perspective: role,
      request_id: `${runId}:${role}`,
      evidence: [{ id: evidenceId, kind: "predicate", artifact: null, artifact_metadata: null }],
      findings: [],
      causal_assessment: {
        hypotheses: [{ id: "h1", disposition: "supported", evidence_ids: [evidenceId] }],
        validity: { status: "valid", compromised_by: [], evidence_ids: [evidenceId] },
      },
    }));
    await writeFile(path.join(roleDir, "events.jsonl"), `${JSON.stringify({ type: "thread.started" })}\n`);
    explorers[role] = { status: statuses[role], evidence: `${role}/evidence.json`, events: `${role}/events.jsonl` };
  }
  await writeFile(path.join(root, "manifest.json"), JSON.stringify({
    run_id: runId, explorers, synthesis: { status: "timeout", report: null, events: "synthesis/events.jsonl" },
  }));
}

test("resume loader accepts only canonical completed explorer packets", async (t) => {
  const root = await mkdtemp(path.join(process.env.TMPDIR || "/tmp", "piku-resume-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  await writeResumeFixture(root, "run-1");
  const loaded = await loadValidatedExplorerRun(root, "run-1");
  assert.deepEqual(loaded.packets.map((packet) => packet.perspective), ["coding_trace", "recovery"]);
  const manifest = JSON.parse(await readFile(path.join(root, "manifest.json"), "utf8"));
  manifest.explorers.recovery.status = "timeout";
  await writeFile(path.join(root, "manifest.json"), JSON.stringify(manifest));
  await assert.rejects(loadValidatedExplorerRun(root, "run-1"), /requires completed explorer: recovery/);
});

test("resume attempts are numbered and never clobber prior synthesis output", async (t) => {
  const root = await mkdtemp(path.join(process.env.TMPDIR || "/tmp", "piku-attempt-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  await mkdir(path.join(root, "synthesis", "attempt-001"), { recursive: true });
  await writeFile(path.join(root, "synthesis", "attempt-001", "events.jsonl"), "original");
  const next = await nextSynthesisAttemptDir(root);
  assert.equal(path.basename(next), "attempt-002");
  assert.equal(await readFile(path.join(root, "synthesis", "attempt-001", "events.jsonl"), "utf8"), "original");
});

test("bounded synthesis prompt names only exact validated inputs", () => {
  const prompt = restrictSynthesisPrompt("judge", ["/tmp/run/coding/evidence.json", "/tmp/run/artifacts/a.png"], "/tmp/run/manifest.json");
  assert.match(prompt, /read only these exact files/);
  assert.match(prompt, /Do not inventory, search, or read any other repository path/);
  assert.match(prompt, /\/tmp\/run\/coding\/evidence\.json/);
  assert.match(prompt, /\/tmp\/run\/artifacts\/a\.png/);
});
