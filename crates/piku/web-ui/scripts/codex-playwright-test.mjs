import { spawn } from "node:child_process";
import { createWriteStream } from "node:fs";
import { access, mkdir, readFile } from "node:fs/promises";
import { createInterface } from "node:readline";
import { fileURLToPath } from "node:url";
import path from "node:path";
import {
  appendEvaluationRecord,
  evaluationRecord,
  evaluationRuntimeMetadata,
} from "./evaluation-ledger.mjs";
import { codexExecArgs, codexJudgeEnvironment } from "./codex-exec.mjs";
import { cleanupStaleAutomationSurfaces, deleteSurface } from "./automation-surfaces.mjs";
import {
  validateRequiredScreenshots,
  withPlaywrightAuthority,
} from "./playwright-authority.mjs";
import { runDeterministicFrontPorch } from "./deterministic-front-porch.mjs";

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const webUiDir = path.resolve(scriptDir, "..");
const repoRoot = path.resolve(webUiDir, "../../..");
const artifactsDir = path.join(repoRoot, ".artifacts", "playwright-agent");
const promptPath = path.join(webUiDir, "e2e", "codex-live-qa.md");
const schemaPath = path.join(webUiDir, "e2e", "agent-report.schema.json");
const runId = new Date().toISOString().replaceAll(/[:.]/g, "-");
const surfaceName = `qa-${Date.now()}-journey`;
const runDir = path.join(artifactsDir, "runs", runId);
const playwrightOutputDir = path.join(runDir, "playwright-output");
const reportPath = path.join(runDir, "report.json");
const eventsPath = path.join(runDir, "events.jsonl");
const ledgerPath =
  process.env.PIKU_LIVE_LEDGER ||
  path.join(repoRoot, "target", "live-ledger", "web-agent.jsonl");
const startedAt = Date.now();
const runtime = evaluationRuntimeMetadata(repoRoot);
const requiredPhases = ["ORIENT", "DISCOVER", "CONSTRUCT", "MANIPULATE", "TRUST", "STRESS", "REFLOW", "REFLECT"];
const requiredThesisDimensions = [
  "task_comprehension",
  "action_provenance",
  "state_visibility",
  "context_control",
  "rerun_semantics",
  "recovery",
  "authority_clarity",
  "spatial_utility",
];
const requiredScreenshots = ["01-empty-desktop.png", "02-create-menu.png", "03-workspace-desktop.png", "04-narrow.png", "05-final-desktop.png"];
const minPlaywrightCalls = 15;
const targetPlaywrightCalls = 105;
const hardMaxPlaywrightCalls = 120;
const maxSnapshotCalls = 7;

function argumentValue(name) {
  const index = process.argv.indexOf(name);
  return index >= 0 ? process.argv[index + 1] : undefined;
}

const target = argumentValue("--url") || process.env.PIKU_WEB_URL || "http://127.0.0.1:9090";
const parsed = new URL(target);
if (
  parsed.protocol !== "http:" ||
  !["localhost", "127.0.0.1"].includes(parsed.hostname) ||
  parsed.port !== "9090"
) {
  throw new Error("--url must be http://localhost:9090 or http://127.0.0.1:9090");
}

try {
  const response = await fetch(parsed, { signal: AbortSignal.timeout(3_000) });
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
} catch (error) {
  console.error(`Piku is not reachable at ${parsed}: ${error.message}`);
  console.error("Start it in your own terminal, then rerun this command.");
  process.exit(2);
}

const removed = await cleanupStaleAutomationSurfaces(parsed);
if (removed.length)
  console.error(`[piku qa] removed ${removed.length} stale automation surfaces`);

await mkdir(playwrightOutputDir, { recursive: true });
await runDeterministicFrontPorch({
  baseUrl: parsed,
  webUiDir,
  outputDir: path.join(runDir, "front-porch"),
});
const template = await readFile(promptPath, "utf8");
const prompt = template
  .replaceAll("{{PIKU_WEB_URL}}", parsed.toString())
  .replaceAll("{{RUN_ID}}", runId)
  .replaceAll("{{RUN_DIR}}", playwrightOutputDir)
  .replaceAll("{{SURFACE}}", surfaceName);
const baseArgs = codexExecArgs({
  schemaPath,
  reportPath,
  prompt,
  playwright: true,
  playwrightCwd: webUiDir,
  cwd: repoRoot,
});
const args = withPlaywrightAuthority(baseArgs, playwrightOutputDir);

console.error(`Running Codex Playwright QA against ${parsed}`);
console.error(`Structured report: ${reportPath}`);
console.error(`Codex event stream: ${eventsPath}`);

const events = createWriteStream(eventsPath, { flags: "wx" });
const child = spawn("codex", args, {
  cwd: repoRoot,
  env: codexJudgeEnvironment(),
  detached: process.platform !== "win32",
  stdio: ["ignore", "pipe", "pipe"],
});
child.stdout.pipe(events);
child.stderr.pipe(process.stderr);
let completedBrowserCalls = 0;
let successfulBrowserCalls = 0;
const eventReader = createInterface({ input: child.stdout });
eventReader.on("line", (line) => {
  try {
    const event = JSON.parse(line);
    if (event.type === "item.completed" && event.item?.type === "mcp_tool_call") {
      completedBrowserCalls += 1;
      const succeeded = event.item.status === "completed" && !event.item.error;
      if (succeeded) successfulBrowserCalls += 1;
      const outcome = succeeded ? "ok" : `failed: ${event.item.error?.message || event.item.status}`;
      console.error(
        `[qa ${String(completedBrowserCalls).padStart(3, "0")} · ${successfulBrowserCalls} ok] ${event.item.tool} ${outcome}`,
      );
    } else if (event.type === "turn.completed") {
      console.error(
        `[qa done] input=${event.usage?.input_tokens ?? "?"} output=${event.usage?.output_tokens ?? "?"}`,
      );
    } else if (event.type === "turn.failed" || event.type === "error") {
      console.error(`[qa error] ${event.message || JSON.stringify(event)}`);
    }
  } catch {
    console.error("[qa warning] Codex emitted a non-JSON event line");
  }
});
const timeoutMs = Number(process.env.PIKU_CODEX_TIMEOUT_MS || 900_000);
let timedOut = false;
let killTimer;
const stopChild = (signal) => {
  if (!child.pid || child.exitCode !== null) return;
  try {
    if (process.platform === "win32") child.kill(signal);
    else process.kill(-child.pid, signal);
  } catch (error) {
    if (error.code !== "ESRCH") throw error;
  }
};
for (const signal of ["SIGINT", "SIGTERM", "SIGHUP"]) {
  process.once(signal, async () => {
    stopChild(signal);
    try { await deleteSurface(parsed, surfaceName); } catch { /* Exit still owns process cleanup. */ }
    process.exit(128 + (signal === "SIGINT" ? 2 : signal === "SIGTERM" ? 15 : 1));
  });
}
const timeout = setTimeout(() => {
  timedOut = true;
  console.error(`Codex QA exceeded ${timeoutMs}ms; stopping its process group.`);
  stopChild("SIGTERM");
  killTimer = setTimeout(() => stopChild("SIGKILL"), 5_000);
}, timeoutMs);
child.on("error", (error) => {
  clearTimeout(timeout);
  clearTimeout(killTimer);
  events.end();
  console.error(`Could not start Codex: ${error.message}`);
  process.exit(1);
});
child.on("exit", async (code, signal) => {
  clearTimeout(timeout);
  clearTimeout(killTimer);
  await new Promise((resolve) => events.end(resolve));
  try { await deleteSurface(parsed, surfaceName); }
  catch (error) { console.error(`[piku qa] ${error.message}`); }
  if (timedOut) {
    await appendEvaluationRecord(
      ledgerPath,
      evaluationRecord({
        runId,
        runStatus: "timeout",
        failureClass: "evaluator_timeout",
        durationMs: Date.now() - startedAt,
        artifactRefs: [path.relative(repoRoot, eventsPath)],
        runtime,
      }),
    );
    console.error("Codex QA timed out before completing its browser journey.");
    process.exit(124);
  }
  if (signal) {
    await appendEvaluationRecord(
      ledgerPath,
      evaluationRecord({
        runId,
        runStatus: "harness_failure",
        failureClass: `signal_${signal.toLowerCase()}`,
        durationMs: Date.now() - startedAt,
        artifactRefs: [path.relative(repoRoot, eventsPath)],
        runtime,
      }),
    );
    console.error(`Codex ended from signal ${signal}`);
    process.exit(1);
  }
  if (code !== 0) {
    await appendEvaluationRecord(
      ledgerPath,
      evaluationRecord({
        runId,
        runStatus: "harness_failure",
        failureClass: "codex_exit",
        durationMs: Date.now() - startedAt,
        artifactRefs: [path.relative(repoRoot, eventsPath)],
        runtime,
      }),
    );
    process.exit(code ?? 1);
  }
  try {
    const report = JSON.parse(await readFile(reportPath, "utf8"));
    const eventLines = (await readFile(eventsPath, "utf8"))
      .split("\n")
      .filter(Boolean)
      .map((line) => JSON.parse(line));
    const completedPlaywrightCalls = eventLines.filter(
      (event) =>
        event.type === "item.completed" &&
        event.item?.type === "mcp_tool_call" &&
        event.item.server === "playwright",
    );
    const successfulPlaywrightCalls = completedPlaywrightCalls.filter(
      (event) => event.item.status === "completed" && !event.item.error,
    );
    const snapshotCalls = successfulPlaywrightCalls.filter(
      (event) => event.item.tool === "browser_snapshot",
    );
    const forbiddenActions = eventLines.filter((event) => {
      if (!event.type?.startsWith("item.")) return false;
      if (event.item?.type === "file_change") return true;
      if (event.item?.type !== "command_execution") return false;
      return !event.item.command?.includes("/.claude/hooks/ghostty-title.sh");
    });
    if (successfulPlaywrightCalls.length < minPlaywrightCalls)
      throw new Error(
        `event stream proves only ${successfulPlaywrightCalls.length} successful Playwright calls`,
      );
    if (completedPlaywrightCalls.length > hardMaxPlaywrightCalls)
      throw new Error(`browser journey exceeded its ${hardMaxPlaywrightCalls}-call hard limit (${completedPlaywrightCalls.length})`);
    if (completedPlaywrightCalls.length > targetPlaywrightCalls)
      console.error(
        `[qa warning] browser journey exceeded its ${targetPlaywrightCalls}-call target (${completedPlaywrightCalls.length})`,
      );
    if (snapshotCalls.length > maxSnapshotCalls)
      throw new Error(`browser journey exceeded its ${maxSnapshotCalls}-snapshot budget (${snapshotCalls.length})`);
    if (forbiddenActions.length > 0)
      throw new Error("agent used shell commands or edited files during browser QA");
    validateRequiredScreenshots(eventLines, playwrightOutputDir, requiredScreenshots);
    if (report.coverage?.length < requiredPhases.length || report.journey?.length < requiredPhases.length) {
      throw new Error("report lacks required phase coverage or journey evidence");
    }
    const reportedPhases = report.journey.map((entry) => entry.phase.toUpperCase());
    if (reportedPhases.length !== requiredPhases.length || requiredPhases.some((phase, index) => reportedPhases[index] !== phase))
      throw new Error(`journey phases must be exactly ${requiredPhases.join(", ")} in order`);
    if (report.surface !== surfaceName)
      throw new Error("report surface must match the harness-owned temporary surface");
    const thesisDimensions = report.product_thesis?.dimensions || [];
    if (
      thesisDimensions.length !== requiredThesisDimensions.length ||
      requiredThesisDimensions.some(
        (name, index) => thesisDimensions[index]?.name !== name,
      )
    )
      throw new Error(
        `product thesis dimensions must be exactly ${requiredThesisDimensions.join(", ")} in order`,
      );
    const thesisResults = thesisDimensions.map((dimension) => dimension.result);
    const absentCount = thesisResults.filter((result) => result === "absent").length;
    const thesisVerdict = report.product_thesis.verdict;
    if (
      thesisVerdict === "supported" &&
      thesisDimensions.some(
        (dimension) =>
          dimension.result !== "demonstrated" || dimension.score < 4,
      )
    )
      throw new Error("supported thesis verdict contradicts dimension evidence");
    if (
      thesisVerdict !== "not_supported" &&
      (thesisResults[0] === "absent" ||
        thesisResults[1] === "absent" ||
        absentCount >= 4)
    )
      throw new Error("product thesis verdict is too positive for absent core dimensions");
    if (report.artifacts?.length !== requiredScreenshots.length)
      throw new Error("report lacks the five required screenshots");
    const reportedArtifactNames = report.artifacts.map((artifact) => path.basename(artifact));
    if (requiredScreenshots.some((name, index) => reportedArtifactNames[index] !== name))
      throw new Error("report screenshots must use the required filenames in journey order");
    const screenshotCalls = successfulPlaywrightCalls.filter(
      (event) => event.item.tool === "browser_take_screenshot",
    );
    await Promise.all(
      report.artifacts.map(async (artifact, index) => {
        const resolved = path.resolve(repoRoot, artifact);
        if (!resolved.startsWith(`${runDir}${path.sep}`))
          throw new Error(`artifact is outside this QA run: ${artifact}`);
        await access(resolved);
        const header = (await readFile(resolved)).subarray(0, 8);
        if (!header.equals(Buffer.from([137, 80, 78, 71, 13, 10, 26, 10])))
          throw new Error(`artifact is not a PNG: ${artifact}`);
        if (!screenshotCalls.some((event) => JSON.stringify(event.item.arguments || {}).includes(requiredScreenshots[index])))
          throw new Error(`event stream does not prove screenshot capture: ${requiredScreenshots[index]}`);
      }),
    );
    const surfacesResponse = await fetch(new URL("/api/surfaces", parsed));
    if (!surfacesResponse.ok)
      throw new Error(`could not verify cleanup: HTTP ${surfacesResponse.status}`);
    const surfaces = await surfacesResponse.json();
    if (surfaces.includes(report.surface))
      throw new Error(`temporary surface still exists: ${report.surface}`);
    const highImpactFinding = report.findings.some((finding) =>
      ["critical", "high"].includes(finding.severity),
    );
    if (report.status === "passed" && highImpactFinding)
      throw new Error("passed report contradicts its critical or high finding");
    console.error(
      `QA complete: ${successfulPlaywrightCalls.length}/${completedPlaywrightCalls.length} browser calls succeeded, ${snapshotCalls.length} snapshots, ${report.findings.length} findings`,
    );
    await appendEvaluationRecord(
      ledgerPath,
      evaluationRecord({
        runId,
        surface: report.surface,
        runStatus: highImpactFinding ? "product_failure" : "completed",
        failureClass: highImpactFinding ? "high_impact_finding" : "none",
        durationMs: Date.now() - startedAt,
        report,
        artifactRefs: [
          path.relative(repoRoot, reportPath),
          path.relative(repoRoot, eventsPath),
          ...report.artifacts,
        ],
        runtime,
      }),
    );
    if (report.status !== "passed") {
      console.error(`QA verdict: ${report.status} — ${report.summary}`);
      process.exit(1);
    }
  } catch (error) {
    await appendEvaluationRecord(
      ledgerPath,
      evaluationRecord({
        runId,
        runStatus: "harness_failure",
        failureClass: "invalid_report",
        durationMs: Date.now() - startedAt,
        artifactRefs: [
          path.relative(repoRoot, reportPath),
          path.relative(repoRoot, eventsPath),
        ],
        runtime,
      }),
    );
    console.error(`Codex QA report is incomplete: ${error.message}`);
    process.exit(1);
  }
  process.exit(0);
});
