import { spawn } from "node:child_process";
import { createWriteStream } from "node:fs";
import { mkdir, readdir, readFile, writeFile } from "node:fs/promises";
import { createInterface } from "node:readline";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { appendEvaluationRecord, evaluationRecord, evaluationRuntimeMetadata } from "./evaluation-ledger.mjs";
import { codexExecArgs, codexJudgeEnvironment, resolvedCodexModel } from "./codex-exec.mjs";
import { cleanupStaleAutomationSurfaces, deleteSurface } from "./automation-surfaces.mjs";

const scriptsDir = path.dirname(fileURLToPath(import.meta.url));
const webUiDir = path.resolve(scriptsDir, "..");
const repoRoot = path.resolve(webUiDir, "../../..");
const roles = ["coding_trace", "recovery"];
const activeChildren = new Set();

function stopProcessGroup(child, signal) {
  if (!child.pid || child.exitCode !== null) return;
  try {
    if (process.platform === "win32") child.kill(signal);
    else process.kill(-child.pid, signal);
  } catch (error) {
    if (error.code !== "ESRCH") throw error;
  }
}

for (const [signal, exitCode] of [["SIGINT", 130], ["SIGTERM", 143], ["SIGHUP", 129]]) {
  process.once(signal, () => {
    for (const child of activeChildren) stopProcessGroup(child, signal);
    process.exit(exitCode);
  });
}

export function safeRunId(value = new Date().toISOString()) {
  return value.replaceAll(/[^A-Za-z0-9-]/g, "-");
}

export function explorerIdentity(runId, role) {
  const createdAt = Date.now();
  return {
    surface: `qa-${createdAt}-${role.replaceAll("_", "-")}`,
    requestId: `${runId}:${role}`,
  };
}

export function explorerCallBudget(role, environment = process.env) {
  const shared = Number(environment.PIKU_EXPLORER_MAX_CALLS || 40);
  if (role !== "recovery") return shared;
  return Number(environment.PIKU_RECOVERY_MAX_CALLS || Math.max(shared, 48));
}

export function explorerHardCallLimit(environment = process.env) {
  return Number(environment.PIKU_EXPLORER_HARD_MAX_CALLS || 64);
}

export function explorerReportOutcome(report) {
  return report.status === "blocked"
    ? { runStatus: "inconclusive", failureClass: "explorer_blocked" }
    : { runStatus: "completed", failureClass: "none" };
}

export function validateSynthesis(report, packets) {
  const known = new Set(packets.flatMap((packet) => packet.evidence.map((item) => item.id)));
  const cited = [
    ...report.evidence_ids,
    ...report.findings.flatMap((finding) => finding.evidence_ids),
    ...report.followups.flatMap((followup) => followup.evidence_ids),
  ];
  const unknown = cited.filter((id) => !known.has(id));
  if (unknown.length) throw new Error(`synthesis cited unknown evidence IDs: ${unknown.join(", ")}`);
  if (report.verdict !== "inconclusive" && report.evidence_ids.length === 0)
    throw new Error("synthesis verdict has no evidence");
  if (report.verdict === "supported" && report.findings.some((finding) => finding.severity === "high"))
    throw new Error("supported synthesis verdict cannot contain a high-severity finding");
}

export async function runCodex({ label = "judge", prompt, schemaPath, reportPath, eventsPath, timeoutMs, maxCalls = Infinity, maxSnapshots = Infinity, playwright = false, model = resolvedCodexModel() }) {
  await mkdir(path.dirname(reportPath), { recursive: true });
  const events = createWriteStream(eventsPath, { flags: "wx" });
  const child = spawn("codex", codexExecArgs({ schemaPath, reportPath, prompt, playwright, playwrightCwd: webUiDir, cwd: repoRoot, model }), {
    cwd: repoRoot,
    env: codexJudgeEnvironment(),
    detached: process.platform !== "win32",
    stdio: ["ignore", "pipe", "pipe"],
  });
  activeChildren.add(child);
  child.stdout.pipe(events);
  child.stderr.pipe(process.stderr);
  let calls = 0;
  let snapshots = 0;
  let reason = null;
  let killTimer;
  const stop = (signal) => {
    stopProcessGroup(child, signal);
  };
  const stopWithFallback = (nextReason) => {
    if (reason) return;
    reason = nextReason;
    console.error(`[piku eval] ${label} stopping reason=${reason} calls=${calls} snapshots=${snapshots}`);
    stop("SIGTERM");
    killTimer = setTimeout(() => stop("SIGKILL"), 5_000);
  };
  console.error(`[piku eval] ${label} started pid=${child.pid} timeout_ms=${timeoutMs} max_calls=${maxCalls} max_snapshots=${maxSnapshots}`);
  const reader = createInterface({ input: child.stdout });
  reader.on("line", (line) => {
    try {
      const event = JSON.parse(line);
      if (event.type === "item.completed" && event.item?.type === "mcp_tool_call" && event.item.server === "playwright") {
        calls += 1;
        if (event.item.tool === "browser_snapshot") snapshots += 1;
        console.error(`[piku eval] ${label} progress calls=${calls}/${maxCalls} snapshots=${snapshots}/${maxSnapshots}`);
        if (calls > maxCalls || snapshots > maxSnapshots) {
          stopWithFallback("budget_exceeded");
        }
      } else if (playwright && event.type?.startsWith("item.") && ["file_change", "command_execution"].includes(event.item?.type)) {
        stopWithFallback("forbidden_action");
      }
    } catch { /* The complete stream remains available for diagnosis. */ }
  });
  const timeout = setTimeout(() => {
    stopWithFallback("timeout");
  }, timeoutMs);
  const outcome = await new Promise((resolve) => {
    child.once("error", (error) => resolve({ code: null, signal: null, error }));
    child.once("exit", (code, signal) => resolve({ code, signal, error: null }));
  });
  clearTimeout(timeout);
  clearTimeout(killTimer);
  activeChildren.delete(child);
  await new Promise((resolve) => events.end(resolve));
  console.error(`[piku eval] ${label} finished code=${outcome.code} signal=${outcome.signal || "none"} reason=${reason || "none"} calls=${calls} snapshots=${snapshots}`);
  return { ...outcome, reason, calls, snapshots };
}

export async function cleanupSurface(baseUrl, surface) {
  await deleteSurface(baseUrl, surface);
}

export async function writeRunManifest(runDir, runId, explorers, synthesis = null, metadata = null) {
  const roles = {};
  for (const explorer of explorers) {
    const roleDir = path.join(runDir, explorer.role);
    let files = [];
    try { files = (await readdir(roleDir)).sort(); } catch { /* A failed launch may create no directory. */ }
    let viewport = null;
    if (files.includes("evidence.json")) {
      try {
        const report = JSON.parse(await readFile(path.join(roleDir, "evidence.json"), "utf8"));
        viewport = report.viewport || null;
      } catch { /* Invalid reports remain visible through explorer status. */ }
    }
    roles[explorer.role] = {
      status: explorer.runStatus,
      evidence: files.includes("evidence.json") ? `${explorer.role}/evidence.json` : null,
      events: files.includes("events.jsonl") ? `${explorer.role}/events.jsonl` : null,
      screenshots: files.filter((file) => file.endsWith(".png")).map((file) => `${explorer.role}/${file}`),
      viewport,
    };
  }
  const manifest = {
    schema_version: 1,
    run_id: runId,
    runtime: metadata,
    explorers: roles,
    synthesis: synthesis
      ? { status: synthesis.runStatus, report: synthesis.report ? "synthesis/report.json" : null, events: "synthesis/events.jsonl" }
      : { status: "not_run", report: null, events: null },
  };
  await writeFile(path.join(runDir, "manifest.json"), `${JSON.stringify(manifest, null, 2)}\n`, "utf8");
  return manifest;
}

async function runExplorer({ role, runId, runDir, baseUrl, ledgerPath, targetCalls, hardMaxCalls, maxSnapshots, timeoutMs, runtime }) {
  const started = Date.now();
  const identity = explorerIdentity(runId, role);
  const roleDir = path.join(runDir, role);
  const reportPath = path.join(roleDir, "evidence.json");
  const eventsPath = path.join(roleDir, "events.jsonl");
  const template = await readFile(path.join(webUiDir, "e2e", `explorer-${role.replaceAll("_", "-")}.md`), "utf8");
  const prompt = template
    .replaceAll("{{PIKU_WEB_URL}}", baseUrl.toString())
    .replaceAll("{{SURFACE}}", identity.surface)
    .replaceAll("{{REQUEST_ID}}", identity.requestId)
    .replaceAll("{{RUN_DIR}}", roleDir)
    .replaceAll("{{MAX_CALLS}}", String(targetCalls))
    .replaceAll("{{MAX_SNAPSHOTS}}", String(maxSnapshots));
  let outcome;
  let report = null;
  let runStatus = "harness_failure";
  let failureClass = "codex_exit";
  try {
    outcome = await runCodex({
      label: role,
      prompt,
      schemaPath: path.join(webUiDir, "e2e", "explorer-report.schema.json"),
      reportPath, eventsPath, timeoutMs, maxCalls: hardMaxCalls, maxSnapshots, playwright: true,
    });
    if (outcome.reason === "timeout") {
      runStatus = "timeout";
      failureClass = "evaluator_timeout";
    } else if (outcome.reason === "budget_exceeded") {
      failureClass = "evaluator_budget";
    } else if (outcome.reason === "forbidden_action") {
      failureClass = "forbidden_agent_action";
    } else if (outcome.code === 0) {
      report = JSON.parse(await readFile(reportPath, "utf8"));
      if (report.perspective !== role || report.surface !== identity.surface || report.request_id !== identity.requestId)
        throw new Error("explorer report identity does not match its isolated assignment");
      const prefix = `${role}:`;
      if (report.evidence.some((item) => !item.id.startsWith(prefix)))
        throw new Error(`explorer evidence IDs must start with ${prefix}`);
      const known = new Set(report.evidence.map((item) => item.id));
      if (report.findings.flatMap((item) => item.evidence_ids).some((id) => !known.has(id)))
        throw new Error("explorer finding cites unknown evidence");
      ({ runStatus, failureClass } = explorerReportOutcome(report));
    }
  } catch (error) {
    failureClass = "invalid_report";
    await writeFile(path.join(roleDir, "validation-error.txt"), `${error.message}\n`, "utf8");
  } finally {
    try { await cleanupSurface(baseUrl, identity.surface); }
    catch { if (runStatus === "completed") { runStatus = "harness_failure"; failureClass = "cleanup_failure"; } }
  }
  const refs = [path.relative(repoRoot, eventsPath)];
  if (report) refs.push(path.relative(repoRoot, reportPath));
  const record = evaluationRecord({ runId, surface: identity.surface, runStatus, failureClass, durationMs: Date.now() - started, report, artifactRefs: refs, runtime });
  record.perspective = role;
  record.explorer_model = resolvedCodexModel();
  record.evidence_ids = report?.evidence.map((item) => item.id) ?? [];
  await appendEvaluationRecord(ledgerPath, record);
  return { role, report, reportPath, runStatus };
}

export async function main() {
  const baseUrl = new URL(process.env.PIKU_WEB_URL || "http://127.0.0.1:9090");
  if (baseUrl.protocol !== "http:" || !["127.0.0.1", "localhost"].includes(baseUrl.hostname) || baseUrl.port !== "9090")
    throw new Error("PIKU_WEB_URL must be the local Piku server on port 9090");
  const response = await fetch(baseUrl, { signal: AbortSignal.timeout(3_000) });
  if (!response.ok) throw new Error(`Piku returned HTTP ${response.status}`);
  const removed = await cleanupStaleAutomationSurfaces(baseUrl);
  if (removed.length)
    console.error(`[piku eval] removed ${removed.length} stale automation surfaces`);
  const runId = safeRunId(process.env.PIKU_EVAL_RUN_ID);
  const runDir = path.join(repoRoot, ".artifacts", "playwright-agent", "parallel", runId);
  const ledgerPath = process.env.PIKU_LIVE_LEDGER || path.join(repoRoot, "target", "live-ledger", "web-agent.jsonl");
  const maxSnapshots = Number(process.env.PIKU_EXPLORER_MAX_SNAPSHOTS || 6);
  const timeoutMs = Number(process.env.PIKU_EXPLORER_TIMEOUT_MS || 600_000);
  const runtime = {
    ...evaluationRuntimeMetadata(repoRoot),
    viewport: { width: 1440, height: 1000 },
    explorer_target_calls: Object.fromEntries(roles.map((role) => [role, explorerCallBudget(role)])),
    explorer_hard_max_calls: explorerHardCallLimit(),
    explorer_max_snapshots: maxSnapshots,
    explorer_timeout_ms: timeoutMs,
  };
  const results = await Promise.all(roles.map((role) => runExplorer({
    role,
    runId,
    runDir,
    baseUrl,
    ledgerPath,
    targetCalls: explorerCallBudget(role),
    hardMaxCalls: explorerHardCallLimit(),
    maxSnapshots,
    timeoutMs,
    runtime,
  })));
  await writeRunManifest(runDir, runId, results, null, runtime);
  if (results.some((result) => result.runStatus !== "completed")) {
    console.error("At least one explorer failed; synthesis was not run.");
    process.exitCode = 1;
    return;
  }
  const packets = results.map((result) => result.report);
  const synthesisDir = path.join(runDir, "synthesis");
  const reportPath = path.join(synthesisDir, "report.json");
  const eventsPath = path.join(synthesisDir, "events.jsonl");
  const template = await readFile(path.join(webUiDir, "e2e", "synthesis.md"), "utf8");
  const prompt = template
    .replace("{{PACKETS}}", results.map((result) => path.relative(repoRoot, result.reportPath)).join("\n"))
    .replace("{{MANIFEST}}", path.relative(repoRoot, path.join(runDir, "manifest.json")))
    .replace("{{LEDGER}}", path.relative(repoRoot, ledgerPath));
  const started = Date.now();
  const synthesisModel = process.env.PIKU_SYNTHESIS_MODEL || resolvedCodexModel();
  const outcome = await runCodex({ label: "synthesis", prompt, schemaPath: path.join(webUiDir, "e2e", "synthesis-report.schema.json"), reportPath, eventsPath, timeoutMs: Number(process.env.PIKU_SYNTHESIS_TIMEOUT_MS || 240_000), model: synthesisModel });
  let report = null;
  let runStatus = "harness_failure";
  let failureClass = "synthesis_exit";
  try {
    if (outcome.reason === "timeout") { runStatus = "timeout"; failureClass = "synthesis_timeout"; }
    else if (outcome.code === 0) {
      report = JSON.parse(await readFile(reportPath, "utf8"));
      validateSynthesis(report, packets);
      runStatus = "completed";
      failureClass = "none";
    }
  } catch (error) {
    failureClass = "invalid_synthesis";
    await writeFile(path.join(synthesisDir, "validation-error.txt"), `${error.message}\n`, "utf8");
  }
  const record = evaluationRecord({ runId, runStatus, failureClass, durationMs: Date.now() - started, artifactRefs: [path.relative(repoRoot, reportPath), path.relative(repoRoot, eventsPath)], runtime });
  record.perspective = "synthesis";
  record.judge_model = synthesisModel;
  record.product_verdict = report?.verdict === "inconclusive" ? null : report?.verdict ?? null;
  record.finding_count = report?.findings.length ?? null;
  record.evidence_ids = report?.evidence_ids ?? [];
  record.followups = report?.followups ?? [];
  await appendEvaluationRecord(ledgerPath, record);
  await writeRunManifest(runDir, runId, results, { runStatus, report }, runtime);
  if (runStatus !== "completed") process.exitCode = 1;
  else console.error(`Parallel evaluation complete: ${runDir}`);
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url))
  await main();
