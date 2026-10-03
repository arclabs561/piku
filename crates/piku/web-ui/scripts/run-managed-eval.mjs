import { spawn } from "node:child_process";
import { createHash } from "node:crypto";
import { mkdir, readFile, realpath, writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { parseEnv } from "node:util";
import path from "node:path";
import { connectExternalEvaluationServer, startManagedEvaluationServer } from "./evaluation-server.mjs";
import { evaluationReviewRoot, webFocusPairReviewDirectory, webReviewDirectory, webRoleReviewDirectory } from "./evaluation-review.mjs";
import { writeAssetReview } from "./evaluation-asset-review.mjs";

export { evaluationReviewRoot, webFocusPairReviewDirectory, webReviewDirectory, webRoleReviewDirectory } from "./evaluation-review.mjs";

const scriptsDir = path.dirname(fileURLToPath(import.meta.url));
const webUiDir = path.resolve(scriptsDir, "..");
const repoRoot = path.resolve(webUiDir, "../../..");
const commands = {
  e2e: ["./node_modules/@playwright/test/cli.js", "test"],
  single: [path.join(scriptsDir, "codex-playwright-test.mjs")],
  parallel: [path.join(scriptsDir, "parallel-agent-eval.mjs")],
  "focus-pair": [path.join(scriptsDir, "focus-pair-eval.mjs")],
};

export function validateRunId(value) {
  if (typeof value !== "string" || value.length > 128
    || !/^[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*$/.test(value))
    throw new TypeError("PIKU_EVAL_RUN_ID must contain only alphanumeric hyphen-separated components");
  return value;
}

export function managedArtifactDir(root, runId) {
  const managedRoot = path.resolve(root, ".artifacts", "playwright-agent", "managed");
  const artifactDir = path.resolve(managedRoot, validateRunId(runId));
  const relative = path.relative(managedRoot, artifactDir);
  if (!relative || relative.startsWith("..") || path.isAbsolute(relative))
    throw new Error("managed evaluation artifact directory escaped its root");
  return artifactDir;
}

export function managedTerminalEnabled(mode) {
  return mode === "e2e";
}

export function managedEvaluationTimeoutMs(environment = process.env) {
  const configured = Number(environment.PIKU_MANAGED_EVAL_TIMEOUT_MS ?? 300_000);
  if (!Number.isInteger(configured) || configured < 1_000 || configured > 3_600_000)
    throw new TypeError("PIKU_MANAGED_EVAL_TIMEOUT_MS must be an integer from 1000 to 3600000");
  return configured;
}

export function managedPageBroker(mode, environment) {
  if (mode === "e2e") return null;
  if (typeof environment.OPENROUTER_API_KEY !== "string"
    || !environment.OPENROUTER_API_KEY.trim())
    throw new Error("managed model-driven evaluation requires OPENROUTER_API_KEY");
  return {
    model: environment.PIKU_EVAL_PAGE_MODEL || "openai/gpt-5.6-terra",
  };
}

async function ancestorOpenRouterKey(startDir) {
  let directory = path.resolve(startDir);
  while (true) {
    try {
      const parsed = parseEnv(await readFile(path.join(directory, ".env"), "utf8"));
      if (typeof parsed.OPENROUTER_API_KEY === "string"
        && parsed.OPENROUTER_API_KEY.trim())
        return parsed.OPENROUTER_API_KEY;
    } catch (error) {
      if (error.code !== "ENOENT")
        throw new Error("could not read managed evaluation credentials");
    }
    const parent = path.dirname(directory);
    if (parent === directory) return null;
    directory = parent;
  }
}

async function buildWebUi(environment) {
  const npm = process.platform === "win32" ? "npm.cmd" : "npm";
  await new Promise((resolve, reject) => {
    const build = spawn(npm, ["run", "build"], {
      cwd: webUiDir,
      env: environment,
      stdio: "inherit",
    });
    build.once("error", reject);
    build.once("exit", (code, signal) => {
      if (code === 0) resolve();
      else reject(new Error(`web asset build failed (${signal || `exit ${code}`})`));
    });
  });
}

export async function resolveManagedEvaluationEnvironment({ mode, environment, startDir }) {
  const resolved = { ...environment };
  if (mode === "e2e") return resolved;
  if (typeof resolved.OPENROUTER_API_KEY !== "string"
    || !resolved.OPENROUTER_API_KEY.trim())
    resolved.OPENROUTER_API_KEY = await ancestorOpenRouterKey(startDir);
  if (!resolved.OPENROUTER_API_KEY)
    throw new Error(
      "managed model-driven evaluation requires OPENROUTER_API_KEY "
      + "in the process environment or an ancestor .env",
    );
  return resolved;
}

export function managedJudgeEnvironment(environment) {
  const child = { ...environment };
  delete child.OPENROUTER_API_KEY;
  return child;
}

export function evaluationArtifactPaths(reviewRoot, mode, runId) {
  validateRunId(runId);
  if (mode === "single") return [path.join(webRoleReviewDirectory(reviewRoot, runId, "single"), "report.json")];
  if (mode === "parallel") return [
    path.join(webReviewDirectory(reviewRoot, runId), "manifest.json"),
    path.join(webReviewDirectory(reviewRoot, runId), "prompt-manifest.json"),
  ];
  if (mode === "focus-pair") return [
    path.join(webFocusPairReviewDirectory(reviewRoot, runId), "manifest.json"),
    path.join(webFocusPairReviewDirectory(reviewRoot, runId), "report.json"),
    ...["blind", "focused"].flatMap((arm) => [
      path.join(webReviewDirectory(reviewRoot, `${runId}-${arm}`), "manifest.json"),
      path.join(webReviewDirectory(reviewRoot, `${runId}-${arm}`), "prompt-manifest.json"),
    ]),
  ];
  return [];
}

async function fileAttestation(root, filePath) {
  const [resolvedRoot, resolvedFile] = await Promise.all([realpath(root), realpath(filePath)]);
  const resolvedRelative = path.relative(resolvedRoot, resolvedFile);
  if (!resolvedRelative || resolvedRelative.startsWith("..") || path.isAbsolute(resolvedRelative))
    throw new Error("managed lifecycle attestation escaped the repository");
  const contents = await readFile(resolvedFile);
  const relative = path.relative(root, filePath);
  if (!relative || relative.startsWith("..") || path.isAbsolute(relative))
    throw new Error("managed lifecycle attestation escaped the repository");
  return { path: relative, sha256: createHash("sha256").update(contents).digest("hex") };
}

async function attestExpectedArtifacts(root, filePaths) {
  const evaluationArtifacts = [];
  const expectedMissing = [];
  for (const filePath of filePaths) {
    try {
      evaluationArtifacts.push(await fileAttestation(root, filePath));
    } catch (error) {
      if (error.code !== "ENOENT") throw error;
      const relative = path.relative(root, filePath);
      if (!relative || relative.startsWith("..") || path.isAbsolute(relative))
        throw new Error("managed lifecycle attestation escaped the repository");
      expectedMissing.push(relative);
    }
  }
  return { evaluationArtifacts, expectedMissing };
}

export async function writeManagedLifecycleBinding({ root, reviewRoot, artifactDir, mode, runId, outcome }) {
  const lifecyclePath = path.join(artifactDir, "server", "lifecycle.json");
  const logPath = path.join(artifactDir, "server", "server.log");
  const lifecycle = JSON.parse(await readFile(lifecyclePath, "utf8"));
  if (lifecycle.ownership !== "managed" || lifecycle.status !== "stopped")
    throw new Error("managed lifecycle binding requires a stopped managed server");
  if (!outcome || (!Number.isInteger(outcome.code) && !outcome.signal))
    throw new Error("managed lifecycle binding requires a child outcome");
  const { evaluationArtifacts, expectedMissing } = await attestExpectedArtifacts(
    reviewRoot, evaluationArtifactPaths(reviewRoot, mode, runId),
  );
  const binding = {
    schema_version: 1,
    run_id: runId,
    mode,
    child: {
      exit_code: outcome.code ?? null,
      exit_signal: outcome.signal ?? null,
    },
    evaluation_artifacts_location: "local-review-cache",
    server: {
      lifecycle: await fileAttestation(root, lifecyclePath),
      log: await fileAttestation(root, logPath),
    },
    evaluation_artifacts: evaluationArtifacts,
    expected_but_missing: expectedMissing,
  };
  const bindingPath = path.join(artifactDir, "lifecycle-binding.json");
  await writeFile(bindingPath, `${JSON.stringify(binding, null, 2)}\n`, {
    encoding: "utf8", flag: "wx", mode: 0o600,
  });
  return { bindingPath, binding };
}

export async function writeBindingWithoutMaskingChildFailure({ outcome, writeBinding, reportError }) {
  try {
    return await writeBinding();
  } catch (error) {
    if (outcome.code === 0 && !outcome.signal) throw error;
    reportError(`Could not write managed lifecycle binding: ${error.message}`);
    return null;
  }
}

export async function writeReviewWithoutMaskingChildFailure({ outcome, writeReview, reportError }) {
  try {
    return await writeReview();
  } catch (error) {
    if (outcome.code === 0 && !outcome.signal) throw error;
    reportError(`Could not write evaluation asset review: ${error.message}`);
    return null;
  }
}

export async function runManagedEval({ argv = process.argv.slice(2), environment = process.env } = {}) {
  const [mode, ...args] = argv;
  if (!commands[mode]) throw new Error("usage: run-managed-eval.mjs e2e|single|parallel|focus-pair [args...]");
  const runId = environment.PIKU_EVAL_RUN_ID === undefined
    ? validateRunId(new Date().toISOString().replaceAll(/[^A-Za-z0-9-]/g, "-"))
    : validateRunId(environment.PIKU_EVAL_RUN_ID);
  const artifactDir = managedArtifactDir(repoRoot, runId);
  const reviewDir = webReviewDirectory(evaluationReviewRoot(environment), runId);
  await mkdir(artifactDir, { recursive: true });
  await mkdir(reviewDir, { recursive: true });
  const parentEnvironment = environment.PIKU_WEB_URL
    ? { ...environment }
    : await resolveManagedEvaluationEnvironment({ mode, environment, startDir: repoRoot });
  if (!environment.PIKU_WEB_URL)
    await buildWebUi(parentEnvironment);
  const server = environment.PIKU_WEB_URL
    ? await connectExternalEvaluationServer(environment.PIKU_WEB_URL)
    : await startManagedEvaluationServer({
      repoRoot,
      artifactDir,
      terminalEnabled: managedTerminalEnabled(mode),
      pageBroker: managedPageBroker(mode, parentEnvironment),
      parentEnv: parentEnvironment,
    });
  let child;
  let outcome;
  let timedOut = false;
  let stopping = false;
  for (const [signal, exitCode] of [["SIGINT", 130], ["SIGTERM", 143], ["SIGHUP", 129]]) {
    process.once(signal, async () => {
      if (stopping) return;
      stopping = true;
      child?.kill(signal);
      await server.stop();
      process.exit(exitCode);
    });
  }
  try {
    child = spawn(process.execPath, [...commands[mode], ...args], {
      cwd: webUiDir,
      env: {
        ...managedJudgeEnvironment(parentEnvironment),
        PIKU_EVAL_RUN_ID: runId,
        PIKU_EVAL_REVIEW_DIR: evaluationReviewRoot(environment),
        ...(mode === "focus-pair" ? { PIKU_EVAL_PAIR_ID: runId } : {}),
        PIKU_WEB_URL: server.baseUrl.toString(),
        PIKU_EVAL_SERVER_OWNERSHIP: server.metadata.ownership,
        PIKU_EVAL_FIXTURE_AVAILABLE: String(server.metadata.fixture_available),
        PIKU_WEB_EVAL_OUTPUT: path.join(reviewDir, "playwright-output"),
        PIKU_WEB_EVAL_REPORT: path.join(reviewDir, "playwright-report"),
        PIKU_REQUIRE_EVALUATION_FIXTURES: server.metadata.ownership === "managed" ? "1" : "0",
      },
      stdio: "inherit",
    });
    const timeoutMs = managedEvaluationTimeoutMs(environment);
    const watchdog = setTimeout(() => {
      timedOut = true;
      console.error(`[piku eval] managed ${mode} run exceeded ${timeoutMs}ms; stopping it`);
      child.kill("SIGTERM");
      setTimeout(() => child.kill("SIGKILL"), 5_000).unref();
    }, timeoutMs);
    try {
      outcome = await new Promise((resolve, reject) => {
        child.once("error", reject);
        child.once("exit", (code, signal) => resolve({ code, signal }));
      });
    } finally {
      clearTimeout(watchdog);
    }
    const managedManifest = mode === "e2e" ? "manifest.json" : "managed-lifecycle.json";
    await writeFile(path.join(reviewDir, managedManifest), `${JSON.stringify({
      schema_version: 1,
      surface: "web",
      run_id: runId,
      command: ["playwright", "test", ...args],
      exit_code: outcome.code ?? null,
      exit_signal: outcome.signal ?? null,
      timed_out: timedOut,
      evidence: ["browser_events", "screenshots", "dom_predicates", "run_record"],
      artifacts: ["playwright-output", "playwright-report"],
    }, null, 2)}\n`, { encoding: "utf8", flag: "wx", mode: 0o600 });
    await writeReviewWithoutMaskingChildFailure({
      outcome,
      writeReview: () => writeAssetReview(reviewDir, { surface: "web", runId }),
      reportError: (message) => console.error(message),
    });
    process.exitCode = outcome.code ?? 1;
  } finally {
    await server.stop();
    if (server.metadata.ownership === "managed" && outcome) {
      await writeBindingWithoutMaskingChildFailure({
        outcome,
        writeBinding: () => writeManagedLifecycleBinding({
          root: repoRoot, reviewRoot: evaluationReviewRoot(environment), artifactDir, mode, runId, outcome,
        }),
        reportError: (message) => console.error(message),
      });
    }
  }
  if (outcome.signal) process.kill(process.pid, outcome.signal);
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url))
  await runManagedEval();
