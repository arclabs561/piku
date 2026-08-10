import { spawn } from "node:child_process";
import { createWriteStream } from "node:fs";
import { cp, mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { supervisePageBroker } from "./evaluation-page-broker.mjs";

const sleep = (milliseconds) => new Promise((resolve) => setTimeout(resolve, milliseconds));
const closeStream = (stream) => new Promise((resolve, reject) => {
  stream.end(resolve);
  stream.once("error", reject);
});

const INHERITED_ENV_ALLOWLIST = [
  "CARGO_HOME",
  "LANG",
  "LC_ALL",
  "PATH",
  "RUSTC_WRAPPER",
  "RUSTUP_HOME",
  "RUSTUP_TOOLCHAIN",
  "SSL_CERT_DIR",
  "SSL_CERT_FILE",
  "TEMP",
  "TMP",
  "TMPDIR",
];

function evaluationEnvironment(parentEnv, stateDir) {
  const env = Object.fromEntries(INHERITED_ENV_ALLOWLIST
    .filter((name) => parentEnv[name] !== undefined)
    .map((name) => [name, parentEnv[name]]));
  if (parentEnv.HOME) {
    env.CARGO_HOME ??= path.join(parentEnv.HOME, ".cargo");
    env.RUSTUP_HOME ??= path.join(parentEnv.HOME, ".rustup");
  }
  const homeDir = path.join(stateDir, "home");
  return {
    ...env,
    HOME: homeDir,
    XDG_CACHE_HOME: path.join(stateDir, "cache"),
    XDG_CONFIG_HOME: path.join(stateDir, "config"),
    XDG_DATA_HOME: path.join(stateDir, "data"),
    PIKU_NO_DOTENV: "1",
    PIKU_WEB_EVALUATION_FIXTURES: "1",
  };
}

async function prepareEvaluationWorkspace(repoRoot, stateDir) {
  const workspaceDir = path.join(stateDir, "workspace");
  const fixtureSource = path.join(
    repoRoot,
    "crates",
    "piku",
    "web-ui",
    "e2e",
    "fixtures",
    "operator-repo",
  );
  const fixtureTarget = path.join(
    workspaceDir,
    "crates",
    "piku",
    "web-ui",
    "e2e",
    "fixtures",
    "operator-repo",
  );
  await mkdir(workspaceDir, { recursive: true, mode: 0o700 });
  await cp(fixtureSource, fixtureTarget, { recursive: true, force: false });
  await writeFile(
    path.join(workspaceDir, "README.md"),
    "# Piku managed evaluation workspace\n\nThis workspace contains only deterministic test fixtures.\n",
    { encoding: "utf8", flag: "wx", mode: 0o600 },
  );
  return workspaceDir;
}

function stopProcessGroup(child, signal, killImpl = process.kill) {
  if (!child.pid || child.exitCode !== null) return;
  try {
    if (process.platform === "win32") child.kill(signal);
    else killImpl(-child.pid, signal);
  } catch (error) {
    if (error.code !== "ESRCH") throw error;
  }
}

export function validateEvaluationOrigin(value) {
  const url = new URL(value);
  if (url.protocol !== "http:" || !["127.0.0.1", "localhost"].includes(url.hostname)
    || !url.port || url.pathname !== "/")
    throw new Error("evaluation server URL must be a loopback HTTP origin with an explicit port");
  return url;
}

async function executorCatalog(baseUrl, fetchImpl) {
  const response = await fetchImpl(new URL("/api/executors", baseUrl), {
    signal: AbortSignal.timeout(2_000),
  });
  if (!response.ok) throw new Error(`executor catalog returned HTTP ${response.status}`);
  return response.json();
}

export async function connectExternalEvaluationServer(value, { fetchImpl = fetch } = {}) {
  const baseUrl = validateEvaluationOrigin(value);
  const catalog = await executorCatalog(baseUrl, fetchImpl);
  const fixture = catalog.executors?.find((item) => item.id === "evaluation_fixture");
  return {
    baseUrl,
    metadata: {
      ownership: "external",
      fixture_available: fixture?.available === true,
      ready_file: null,
    },
    stop: async () => {},
  };
}

export async function startManagedEvaluationServer({
  repoRoot,
  artifactDir,
  spawnImpl = spawn,
  fetchImpl = fetch,
  killImpl = process.kill,
  parentEnv = process.env,
  timeoutMs = 60_000,
  shutdownGraceMs = 3_000,
  terminalEnabled = false,
  pageBroker = null,
  serverBinary = null,
} = {}) {
  if (!repoRoot || !artifactDir) throw new TypeError("repoRoot and artifactDir are required");
  const serverDir = path.join(artifactDir, "server");
  const stateDir = path.join(serverDir, "state");
  const readyPath = path.join(serverDir, "ready.json");
  const logPath = path.join(serverDir, "server.log");
  const lifecyclePath = path.join(serverDir, "lifecycle.json");
  await mkdir(stateDir, { recursive: true });
  const env = evaluationEnvironment(parentEnv, stateDir);
  if (!terminalEnabled) env.PIKU_WEB_DISABLE_TERMINAL = "1";
  if (pageBroker) {
    if (typeof pageBroker.model !== "string" || !pageBroker.model)
      throw new TypeError("pageBroker.model is required");
    env.PIKU_PAGE_BROKER_FD = "3";
    env.PIKU_PAGE_BROKER_MODEL = pageBroker.model;
  }
  await Promise.all([
    mkdir(env.HOME, { recursive: true }),
    mkdir(env.XDG_CACHE_HOME, { recursive: true }),
    mkdir(env.XDG_CONFIG_HOME, { recursive: true }),
    mkdir(env.XDG_DATA_HOME, { recursive: true }),
  ]);
  const workspaceDir = await prepareEvaluationWorkspace(repoRoot, stateDir);
  const log = createWriteStream(logPath, { flags: "wx", mode: 0o600 });
  const startedAt = new Date().toISOString();
  const command = serverBinary || "cargo";
  const args = serverBinary
    ? ["web", "--port", "0"]
    : [
      "run",
      "--quiet",
      "--manifest-path",
      path.join(repoRoot, "Cargo.toml"),
      "-p",
      "piku",
      "--",
      "web",
      "--port",
      "0",
    ];
  const child = spawnImpl(command, args, {
    cwd: workspaceDir,
    env: { ...env, PIKU_WEB_READY_FILE: readyPath },
    detached: process.platform !== "win32",
    stdio: pageBroker ? ["ignore", "pipe", "pipe", "pipe"] : ["ignore", "pipe", "pipe"],
  });
  child.stdout?.pipe(log, { end: false });
  child.stderr?.pipe(log, { end: false });
  let closePageBroker = async () => {};
  if (pageBroker) {
    const brokerStream = child.stdio?.[3];
    if (!brokerStream) throw new Error("managed server did not expose page broker fd 3");
    closePageBroker = supervisePageBroker(brokerStream, {
      apiKey: parentEnv.OPENROUTER_API_KEY,
      model: pageBroker.model,
      fetchImpl: pageBroker.fetchImpl,
    });
  }
  let processOutcome = null;
  const terminated = new Promise((resolve) => {
    const settle = (outcome) => {
      if (processOutcome) return;
      processOutcome = outcome;
      resolve(processOutcome);
    };
    child.once("exit", (code, signal) => settle({ code, signal, error: null }));
    child.once("error", (error) => settle({ code: null, signal: null, error }));
  });
  const deadline = Date.now() + timeoutMs;
  let ready;
  let lastError = "ready file not published";
  while (Date.now() < deadline && !processOutcome) {
    try {
      ready = JSON.parse(await readFile(readyPath, "utf8"));
      if (ready.schema_version !== 1 || ready.fixture_enabled !== true
        || !Number.isSafeInteger(ready.pid) || ready.pid !== child.pid)
        throw new Error("ready file failed its strict contract");
      const baseUrl = validateEvaluationOrigin(ready.url);
      const catalog = await executorCatalog(baseUrl, fetchImpl);
      const fixture = catalog.executors?.find((item) => item.id === "evaluation_fixture");
      if (fixture?.available !== true) throw new Error("evaluation fixture is unavailable");
      const metadata = {
        ownership: "managed",
        fixture_available: true,
        terminal_enabled: terminalEnabled,
        page_broker_enabled: Boolean(pageBroker),
        page_broker_model: pageBroker?.model ?? null,
        ready_file: readyPath,
        workspace_root: workspaceDir,
        url: baseUrl.toString(),
        pid: child.pid,
        started_at: startedAt,
      };
      await writeFile(lifecyclePath, `${JSON.stringify({ ...metadata, status: "ready" }, null, 2)}\n`, { mode: 0o600 });
      let stopped = false;
      return {
        baseUrl,
        metadata,
        async stop() {
          if (stopped) return;
          stopped = true;
          stopProcessGroup(child, "SIGTERM", killImpl);
          const graceful = await Promise.race([
            terminated.then(() => true),
            sleep(shutdownGraceMs).then(() => false),
          ]);
          if (!graceful) {
            stopProcessGroup(child, "SIGKILL", killImpl);
            await terminated;
          }
          await closePageBroker();
          await closeStream(log);
          await writeFile(lifecyclePath, `${JSON.stringify({
            ...metadata,
            status: "stopped",
            stopped_at: new Date().toISOString(),
            exit_code: processOutcome?.code ?? null,
            exit_signal: processOutcome?.signal ?? null,
            forced: !graceful,
          }, null, 2)}\n`, { mode: 0o600 });
        },
      };
    } catch (error) {
      lastError = error.message;
      await Promise.race([terminated, sleep(100)]);
    }
  }
  const reason = processOutcome?.error
    ? `server failed to start: ${processOutcome.error.message}`
    : processOutcome
      ? `server exited before readiness (${processOutcome.signal || processOutcome.code})`
    : `server readiness timed out: ${lastError}`;
  stopProcessGroup(child, "SIGTERM", killImpl);
  await Promise.race([terminated, sleep(shutdownGraceMs)]);
  if (!processOutcome) {
    stopProcessGroup(child, "SIGKILL", killImpl);
    await terminated;
  }
  await closePageBroker();
  await closeStream(log);
  await writeFile(lifecyclePath, `${JSON.stringify({
    ownership: "managed",
    status: "failed",
    started_at: startedAt,
    failed_at: new Date().toISOString(),
    reason,
  }, null, 2)}\n`, { mode: 0o600 });
  throw new Error(reason);
}
