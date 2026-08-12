#!/usr/bin/env node

import { spawn } from "node:child_process";
import { constants as fsConstants } from "node:fs";
import { access, chmod, copyFile, mkdir, mkdtemp, readFile, rm, stat } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const LAUNCH_POLICY = JSON.parse(await readFile(new URL(
  "../crates/piku/src/web/codex-launch-policy.json",
  import.meta.url,
), "utf8"));

const DEFAULT_TIMEOUT_MS = 5_000;
const DEFAULT_INTERACTIVE_TIMEOUT_MS = 120_000;
const DEFAULT_OUTPUT_CAP = 16 * 1024;
const MAX_FIXTURES = 8;
const MAX_FIXTURE_TEXT = 512;
const MAX_RETAINED_NOTIFICATIONS = 32;

function cleanEnvironment(codexHome) {
  const env = {
    CODEX_HOME: codexHome,
    HOME: codexHome,
  };
  for (const key of LAUNCH_POLICY.child_env_allowlist) {
    if (process.env[key] !== undefined) env[key] = process.env[key];
  }
  env.LANG ??= "C";
  env.LC_ALL ??= "C";
  env.PATH ??= "/usr/bin:/bin";
  env.TMPDIR ??= tmpdir();
  return env;
}

function boundedText(chunks, cap) {
  return Buffer.concat(chunks).subarray(0, cap).toString("utf8");
}

function safeRpcDetail(value, fallback) {
  if (typeof value !== "string" && typeof value !== "number") return fallback;
  return String(value)
    .replace(/[\r\n\t]+/g, " ")
    .replace(/\b(api[_-]?key|token|secret|authorization)\s*[:=]\s*\S+/gi, "$1=[redacted]")
    .replace(/(?:\/[A-Za-z0-9._-]+){2,}/g, "[path]")
    .slice(0, 256);
}

async function copyInteractiveAuth(codexHome, explicitAuthFile) {
  const candidates = explicitAuthFile
    ? [explicitAuthFile]
    : [
        process.env.CODEX_HOME ? path.join(process.env.CODEX_HOME, "auth.json") : null,
        process.env.HOME ? path.join(process.env.HOME, ".codex", "auth.json") : null,
      ].filter(Boolean);
  for (const source of candidates) {
    try {
      const metadata = await stat(source);
      if (!metadata.isFile()) continue;
      const destination = path.join(codexHome, "auth.json");
      await copyFile(source, destination, fsConstants.COPYFILE_EXCL);
      await chmod(destination, 0o600);
      return;
    } catch {
      // Try the next structural Codex home candidate without exposing its path.
    }
  }
  throw new Error("Codex authentication unavailable");
}

async function exists(target) {
  try {
    await access(target, fsConstants.F_OK);
    return true;
  } catch {
    return false;
  }
}

function runVersion(executable, prefixArgs, env, timeoutMs, outputCap) {
  return new Promise((resolve, reject) => {
    const child = spawn(executable, [...prefixArgs, "--version"], {
      env,
      stdio: ["ignore", "pipe", "pipe"],
    });
    const stdout = [];
    let size = 0;
    const timer = setTimeout(() => {
      child.kill("SIGKILL");
      reject(new Error("Codex version check timed out"));
    }, timeoutMs);
    child.stdout.on("data", (chunk) => {
      if (size < outputCap) stdout.push(chunk.subarray(0, outputCap - size));
      size += chunk.length;
    });
    child.once("error", (error) => {
      clearTimeout(timer);
      reject(new Error(`Could not start Codex: ${error.code ?? "spawn error"}`));
    });
    child.once("close", (code) => {
      clearTimeout(timer);
      if (code !== 0) reject(new Error("Codex version check failed"));
      else resolve(boundedText(stdout, outputCap).trim());
    });
  });
}

class RpcClient {
  constructor(child, timeoutMs, outputCap) {
    this.child = child;
    this.timeoutMs = timeoutMs;
    this.outputCap = outputCap;
    this.nextId = 1;
    this.pending = new Map();
    this.notifications = [];
    this.waiters = [];
    this.buffer = "";
    this.stderrBytes = 0;
    child.stdout.setEncoding("utf8");
    child.stdout.on("data", (data) => this.onData(data));
    child.stderr.on("data", (data) => {
      this.stderrBytes += data.length;
      if (this.stderrBytes > outputCap) child.kill("SIGKILL");
    });
    child.once("error", () => this.failAll("App server process error"));
    child.once("close", () => this.failAll("App server exited before replying"));
  }

  onData(data) {
    this.buffer += data;
    if (Buffer.byteLength(this.buffer) > this.outputCap) {
      this.child.kill("SIGKILL");
      this.failAll("App server output exceeded limit");
      return;
    }
    for (;;) {
      const newline = this.buffer.indexOf("\n");
      if (newline < 0) return;
      const line = this.buffer.slice(0, newline).trim();
      this.buffer = this.buffer.slice(newline + 1);
      if (!line) continue;
      let message;
      try {
        message = JSON.parse(line);
      } catch {
        this.child.kill("SIGKILL");
        this.failAll("App server returned invalid JSON");
        return;
      }
      if (message.id === undefined) {
        const itemType = message.params?.item?.type;
        const retain = message.method === "turn/completed"
          || (["item/started", "item/completed"].includes(message.method)
            && ["commandExecution", "fileChange"].includes(itemType));
        if (retain) {
          if (this.notifications.length === MAX_RETAINED_NOTIFICATIONS) this.notifications.shift();
          this.notifications.push(message);
        }
        for (const waiter of [...this.waiters]) {
          if (!waiter.predicate(message)) continue;
          clearTimeout(waiter.timer);
          this.waiters.splice(this.waiters.indexOf(waiter), 1);
          waiter.resolve(message);
        }
        continue;
      }
      const pending = this.pending.get(message.id);
      if (!pending) continue;
      clearTimeout(pending.timer);
      this.pending.delete(message.id);
      if (message.error) {
        const code = safeRpcDetail(message.error.code, "unknown");
        const detail = safeRpcDetail(message.error.message, "no safe detail");
        pending.reject(new Error(`App server ${pending.method} failed (${code}): ${detail}`));
      }
      else pending.resolve(message.result);
    }
  }

  failAll(message) {
    for (const pending of this.pending.values()) {
      clearTimeout(pending.timer);
      pending.reject(new Error(message));
    }
    this.pending.clear();
  }

  request(method, params) {
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(id);
        this.child.kill("SIGKILL");
        reject(new Error(`App server ${method} timed out`));
      }, this.timeoutMs);
      this.pending.set(id, { method, resolve, reject, timer });
      this.child.stdin.write(`${JSON.stringify({ method, id, params })}\n`);
    });
  }

  notify(method, params = {}) {
    this.child.stdin.write(`${JSON.stringify({ method, params })}\n`);
  }

  waitForNotification(predicate, description) {
    const queued = this.notifications.find(predicate);
    if (queued) return Promise.resolve(queued);
    return new Promise((resolve, reject) => {
      const waiter = { predicate, resolve, reject };
      waiter.timer = setTimeout(() => {
        this.waiters.splice(this.waiters.indexOf(waiter), 1);
        this.child.kill("SIGKILL");
        reject(new Error(`App server ${description} timed out`));
      }, this.timeoutMs);
      this.waiters.push(waiter);
    });
  }
}

function writeCommand(target) {
  return [
    process.execPath,
    "-e",
    "require('node:fs').writeFileSync(process.argv[1], 'probe')",
    target,
  ];
}

function validExecResult(result) {
  return result && Number.isInteger(result.exitCode)
    && typeof result.stdout === "string" && typeof result.stderr === "string";
}

function workspacePolicy(workspace) {
  return {
    type: "workspaceWrite",
    writableRoots: [workspace],
    networkAccess: LAUNCH_POLICY.network_access,
    excludeSlashTmp: LAUNCH_POLICY.exclude_slash_tmp,
    excludeTmpdirEnvVar: LAUNCH_POLICY.exclude_tmpdir_env_var,
  };
}

function normalizeText(value, temporaryRoot, workspace) {
  if (typeof value !== "string") return null;
  return value
    .replaceAll(workspace, "$WORKSPACE")
    .replaceAll(temporaryRoot, "$TEMP_ROOT")
    .slice(0, MAX_FIXTURE_TEXT);
}

function itemFixture(message, temporaryRoot, workspace, turnIds) {
  const item = message?.params?.item;
  if (!item || !["commandExecution", "fileChange"].includes(item.type)) return null;
  const fixture = {
    event: message.method,
    turn: message.params?.turnId === turnIds.inside
      ? "workspaceWrite"
      : message.params?.turnId === turnIds.denied ? "siblingDenial" : null,
    type: item.type,
    status: typeof item.status === "string" ? item.status : null,
  };
  if (item.type === "commandExecution") {
    fixture.cwd = normalizeText(item.cwd, temporaryRoot, workspace);
    fixture.exitCode = Number.isInteger(item.exitCode) ? item.exitCode : null;
    fixture.outputBytes = typeof item.aggregatedOutput === "string"
      ? Math.min(Buffer.byteLength(item.aggregatedOutput), MAX_FIXTURE_TEXT)
      : 0;
  } else {
    fixture.changes = Array.isArray(item.changes) ? item.changes.slice(0, 8).map((change) => ({
      path: normalizeText(change.path, temporaryRoot, workspace),
      kind: change.kind?.type ?? change.kind ?? null,
    })) : [];
  }
  return fixture;
}

async function awaitTurn(rpc, threadId, turnId) {
  return rpc.waitForNotification(
    (message) => message.method === "turn/completed"
      && message.params?.threadId === threadId
      && message.params?.turn?.id === turnId,
    `turn ${turnId} completion`,
  );
}

async function containsProbe(target) {
  try {
    return (await readFile(target, "utf8")).trim() === "probe";
  } catch {
    return false;
  }
}

async function runInteractiveProbe(rpc, temporaryRoot, workspace) {
  const commandSentinel = path.join(workspace, "interactive-command-sentinel");
  const fileSentinel = path.join(workspace, "interactive-file-sentinel");
  const siblingSentinel = path.join(temporaryRoot, "interactive-sibling-sentinel");
  const policy = workspacePolicy(workspace);
  const start = await rpc.request("thread/start", {
    cwd: workspace,
    sandbox: LAUNCH_POLICY.thread_sandbox,
    approvalPolicy: LAUNCH_POLICY.approval_policy,
    ephemeral: false,
    developerInstructions: "Follow the probe instructions exactly. Do not use the network or modify any other path.",
  });
  const threadId = start?.thread?.id;
  if (typeof threadId !== "string") throw new Error("App server thread/start returned no thread id");
  const first = await rpc.request("turn/start", {
    threadId,
    approvalPolicy: LAUNCH_POLICY.approval_policy,
    sandboxPolicy: policy,
    input: [{ type: "text", text: `Run a shell command that writes exactly probe to ${commandSentinel}, then use the file-change tool to create ${fileSentinel} containing exactly probe. Do nothing else.` }],
  });
  const firstTurnId = first?.turn?.id;
  if (typeof firstTurnId !== "string") throw new Error("App server first turn/start returned no turn id");
  const firstCompleted = await awaitTurn(rpc, threadId, firstTurnId);

  const resumed = await rpc.request("thread/resume", {
    threadId,
    cwd: workspace,
    sandbox: LAUNCH_POLICY.thread_sandbox,
    approvalPolicy: LAUNCH_POLICY.approval_policy,
  });
  if (resumed?.thread?.id !== threadId) throw new Error("App server resumed an unexpected thread");
  const second = await rpc.request("turn/start", {
    threadId,
    approvalPolicy: LAUNCH_POLICY.approval_policy,
    sandboxPolicy: policy,
    input: [{ type: "text", text: `This is a sandbox-denial probe. Use the shell tool to execute exactly: printf probe > '${siblingSentinel}'. Execute it even though failure is expected. Do not ask for approval, reason about permissions first, or do anything else.` }],
  });
  const secondTurnId = second?.turn?.id;
  if (typeof secondTurnId !== "string") throw new Error("App server second turn/start returned no turn id");
  const secondCompleted = await awaitTurn(rpc, threadId, secondTurnId);

  const fixtures = rpc.notifications
    .map((message) => itemFixture(message, temporaryRoot, workspace, {
      inside: firstTurnId,
      denied: secondTurnId,
    }))
    .filter(Boolean)
    .slice(0, MAX_FIXTURES);
  const insideWritten = await containsProbe(commandSentinel) && await containsProbe(fileSentinel);
  const siblingDenied = !(await exists(siblingSentinel));
  const turnsCompleted = [firstCompleted, secondCompleted]
    .every((message) => message.params?.turn?.status === "completed");
  const lifecycleFixtures = fixtures.some((fixture) => fixture.event === "item/started" && fixture.type === "commandExecution")
    && fixtures.some((fixture) => fixture.event === "item/completed" && fixture.type === "commandExecution")
    && fixtures.some((fixture) => fixture.event === "item/started" && fixture.type === "fileChange")
    && fixtures.some((fixture) => fixture.event === "item/completed" && fixture.type === "fileChange");
  const denialObserved = fixtures.some((fixture) => fixture.turn === "siblingDenial"
    && fixture.event === "item/completed"
    && fixture.type === "commandExecution"
    && fixture.exitCode !== null && fixture.exitCode !== 0);
  return {
    // The deterministic command/exec probe above is the enforcement evidence.
    // A model may decline an obviously forbidden command before emitting an
    // item, so its attempted denial is useful behavioral evidence, not a gate.
    passed: insideWritten && siblingDenied && turnsCompleted && lifecycleFixtures,
    insideWritten,
    siblingDenied,
    denialObserved,
    denialEvidence: denialObserved ? "turn_command_denied" : "model_declined_before_tool",
    turnsCompleted,
    fixtures,
  };
}

export async function runProbe(options = {}) {
  const executable = options.executable ?? "codex";
  const prefixArgs = options.prefixArgs ?? [];
  const timeoutMs = options.timeoutMs
    ?? (options.interactive ? DEFAULT_INTERACTIVE_TIMEOUT_MS : DEFAULT_TIMEOUT_MS);
  const outputCap = options.outputCap ?? DEFAULT_OUTPUT_CAP;
  const temporaryRoot = await mkdtemp(path.join(tmpdir(), "piku-codex-probe-"));
  const codexHome = path.join(temporaryRoot, "codex-home");
  const workspace = path.join(temporaryRoot, "workspace");
  const readOnlySentinel = path.join(workspace, "read-only-sentinel");
  const workspaceSentinel = path.join(workspace, "workspace-sentinel");
  const siblingSentinel = path.join(temporaryRoot, "sibling-sentinel");
  let child;
  try {
    await mkdir(codexHome);
    await mkdir(workspace);
    if (options.interactive) await copyInteractiveAuth(codexHome, options.authFile);
    const env = cleanEnvironment(codexHome);
    const version = await runVersion(executable, prefixArgs, env, timeoutMs, outputCap);
    child = spawn(executable, [...prefixArgs, "app-server", "--listen", "stdio://"], {
      cwd: workspace,
      env,
      stdio: ["pipe", "pipe", "pipe"],
    });
    const rpc = new RpcClient(child, timeoutMs, outputCap);
    const initialization = await rpc.request("initialize", {
      clientInfo: { name: "piku_sandbox_probe", title: "Piku sandbox probe", version: "1" },
    });
    rpc.notify("initialized");

    const exec = (command, sandboxPolicy) => rpc.request("command/exec", {
      command,
      cwd: workspace,
      env: { CODEX_HOME: codexHome, HOME: codexHome },
      sandboxPolicy,
      timeoutMs,
      outputBytesCap: outputCap,
    });
    const readOnlyResult = await exec(writeCommand(readOnlySentinel), {
      type: "readOnly",
      networkAccess: false,
    });
    const insideResult = await exec(writeCommand(workspaceSentinel), workspacePolicy(workspace));
    const outsideResult = await exec(writeCommand(siblingSentinel), workspacePolicy(workspace));
    const resultsValid = [readOnlyResult, insideResult, outsideResult].every(validExecResult);
    const readOnlyAbsent = !(await exists(readOnlySentinel));
    const insidePresent = await exists(workspaceSentinel);
    const outsideAbsent = !(await exists(siblingSentinel));
    const readOnlyPassed = resultsValid && readOnlyResult.exitCode !== 0 && readOnlyAbsent;
    const workspaceWritePassed = resultsValid && insideResult.exitCode === 0 && insidePresent
      && outsideResult.exitCode !== 0 && outsideAbsent;
    const interactive = options.interactive
      ? await runInteractiveProbe(rpc, temporaryRoot, workspace)
      : undefined;
    return {
      ok: readOnlyPassed && workspaceWritePassed && (!interactive || interactive.passed),
      codexVersion: version,
      launchPolicy: LAUNCH_POLICY,
      protocol: {
        initialized: Boolean(initialization && typeof initialization === "object"),
        readOnly: { passed: readOnlyPassed, exitCode: readOnlyResult?.exitCode ?? null, sentinelAbsent: readOnlyAbsent },
        workspaceWrite: {
          passed: workspaceWritePassed,
          insideExitCode: insideResult?.exitCode ?? null,
          insideSentinelPresent: insidePresent,
          outsideExitCode: outsideResult?.exitCode ?? null,
          outsideSentinelAbsent: outsideAbsent,
        },
      },
      ...(interactive ? { interactive } : {}),
    };
  } finally {
    if (child && child.exitCode === null) child.kill("SIGKILL");
    await rm(temporaryRoot, { recursive: true, force: true });
  }
}

async function main() {
  try {
    const args = process.argv.slice(2);
    if (args.some((arg) => arg !== "--interactive")) throw new Error("Usage: codex-app-server-probe.mjs [--interactive]");
    const result = await runProbe({ interactive: args.includes("--interactive") });
    process.stdout.write(`${JSON.stringify(result, null, 2)}\n`);
    process.exitCode = result.ok ? 0 : 1;
  } catch (error) {
    process.stdout.write(`${JSON.stringify({ ok: false, error: error.message })}\n`);
    process.exitCode = 1;
  }
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  await main();
}
