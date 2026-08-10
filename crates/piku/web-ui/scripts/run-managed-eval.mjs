import { spawn } from "node:child_process";
import { createHash } from "node:crypto";
import { mkdir, readFile, realpath, writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { connectExternalEvaluationServer, startManagedEvaluationServer } from "./evaluation-server.mjs";

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

export function evaluationArtifactPaths(root, mode, runId) {
  validateRunId(runId);
  const artifactsRoot = path.join(root, ".artifacts", "playwright-agent");
  if (mode === "single") return [path.join(artifactsRoot, "runs", runId, "report.json")];
  if (mode === "parallel") return [
    path.join(artifactsRoot, "parallel", runId, "manifest.json"),
    path.join(artifactsRoot, "parallel", runId, "prompt-manifest.json"),
  ];
  if (mode === "focus-pair") return [
    path.join(artifactsRoot, "focus-pairs", runId, "manifest.json"),
    path.join(artifactsRoot, "focus-pairs", runId, "report.json"),
    ...["blind", "focused"].flatMap((arm) => [
      path.join(artifactsRoot, "parallel", `${runId}-${arm}`, "manifest.json"),
      path.join(artifactsRoot, "parallel", `${runId}-${arm}`, "prompt-manifest.json"),
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

export async function writeManagedLifecycleBinding({ root, artifactDir, mode, runId, outcome }) {
  const lifecyclePath = path.join(artifactDir, "server", "lifecycle.json");
  const logPath = path.join(artifactDir, "server", "server.log");
  const lifecycle = JSON.parse(await readFile(lifecyclePath, "utf8"));
  if (lifecycle.ownership !== "managed" || lifecycle.status !== "stopped")
    throw new Error("managed lifecycle binding requires a stopped managed server");
  if (!outcome || (!Number.isInteger(outcome.code) && !outcome.signal))
    throw new Error("managed lifecycle binding requires a child outcome");
  const { evaluationArtifacts, expectedMissing } = await attestExpectedArtifacts(
    root, evaluationArtifactPaths(root, mode, runId),
  );
  const binding = {
    schema_version: 1,
    run_id: runId,
    mode,
    child: {
      exit_code: outcome.code ?? null,
      exit_signal: outcome.signal ?? null,
    },
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

export async function runManagedEval({ argv = process.argv.slice(2), environment = process.env } = {}) {
  const [mode, ...args] = argv;
  if (!commands[mode]) throw new Error("usage: run-managed-eval.mjs e2e|single|parallel|focus-pair [args...]");
  const runId = environment.PIKU_EVAL_RUN_ID === undefined
    ? validateRunId(new Date().toISOString().replaceAll(/[^A-Za-z0-9-]/g, "-"))
    : validateRunId(environment.PIKU_EVAL_RUN_ID);
  const artifactDir = managedArtifactDir(repoRoot, runId);
  await mkdir(artifactDir, { recursive: true });
  const server = environment.PIKU_WEB_URL
    ? await connectExternalEvaluationServer(environment.PIKU_WEB_URL)
    : await startManagedEvaluationServer({ repoRoot, artifactDir });
  let child;
  let outcome;
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
        ...environment,
        PIKU_EVAL_RUN_ID: runId,
        ...(mode === "focus-pair" ? { PIKU_EVAL_PAIR_ID: runId } : {}),
        PIKU_WEB_URL: server.baseUrl.toString(),
        PIKU_EVAL_SERVER_OWNERSHIP: server.metadata.ownership,
        PIKU_EVAL_FIXTURE_AVAILABLE: String(server.metadata.fixture_available),
        PIKU_REQUIRE_EVALUATION_FIXTURES: server.metadata.ownership === "managed" ? "1" : "0",
      },
      stdio: "inherit",
    });
    outcome = await new Promise((resolve, reject) => {
      child.once("error", reject);
      child.once("exit", (code, signal) => resolve({ code, signal }));
    });
    process.exitCode = outcome.code ?? 1;
  } finally {
    await server.stop();
    if (server.metadata.ownership === "managed" && outcome) {
      await writeBindingWithoutMaskingChildFailure({
        outcome,
        writeBinding: () => writeManagedLifecycleBinding({
          root: repoRoot, artifactDir, mode, runId, outcome,
        }),
        reportError: (message) => console.error(message),
      });
    }
  }
  if (outcome.signal) process.kill(process.pid, outcome.signal);
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url))
  await runManagedEval();
