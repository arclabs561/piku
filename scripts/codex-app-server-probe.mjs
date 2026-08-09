#!/usr/bin/env node

import { spawn } from "node:child_process";
import { constants as fsConstants } from "node:fs";
import { access, mkdir, mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const DEFAULT_TIMEOUT_MS = 5_000;
const DEFAULT_OUTPUT_CAP = 16 * 1024;

function cleanEnvironment(codexHome) {
  return {
    CODEX_HOME: codexHome,
    HOME: codexHome,
    LANG: "C",
    LC_ALL: "C",
    PATH: process.env.PATH ?? "/usr/bin:/bin",
    TMPDIR: tmpdir(),
  };
}

function boundedText(chunks, cap) {
  return Buffer.concat(chunks).subarray(0, cap).toString("utf8");
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
      if (message.id === undefined) continue;
      const pending = this.pending.get(message.id);
      if (!pending) continue;
      clearTimeout(pending.timer);
      this.pending.delete(message.id);
      if (message.error) pending.reject(new Error("App server RPC failed"));
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
      this.pending.set(id, { resolve, reject, timer });
      this.child.stdin.write(`${JSON.stringify({ method, id, params })}\n`);
    });
  }

  notify(method, params = {}) {
    this.child.stdin.write(`${JSON.stringify({ method, params })}\n`);
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

export async function runProbe(options = {}) {
  const executable = options.executable ?? "codex";
  const prefixArgs = options.prefixArgs ?? [];
  const timeoutMs = options.timeoutMs ?? DEFAULT_TIMEOUT_MS;
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
    const insideResult = await exec(writeCommand(workspaceSentinel), {
      type: "workspaceWrite",
      writableRoots: [workspace],
      networkAccess: false,
      excludeSlashTmp: true,
      excludeTmpdirEnvVar: true,
    });
    const outsideResult = await exec(writeCommand(siblingSentinel), {
      type: "workspaceWrite",
      writableRoots: [workspace],
      networkAccess: false,
      excludeSlashTmp: true,
      excludeTmpdirEnvVar: true,
    });
    const resultsValid = [readOnlyResult, insideResult, outsideResult].every(validExecResult);
    const readOnlyAbsent = !(await exists(readOnlySentinel));
    const insidePresent = await exists(workspaceSentinel);
    const outsideAbsent = !(await exists(siblingSentinel));
    const readOnlyPassed = resultsValid && readOnlyResult.exitCode !== 0 && readOnlyAbsent;
    const workspaceWritePassed = resultsValid && insideResult.exitCode === 0 && insidePresent
      && outsideResult.exitCode !== 0 && outsideAbsent;
    return {
      ok: readOnlyPassed && workspaceWritePassed,
      codexVersion: version,
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
    };
  } finally {
    if (child && child.exitCode === null) child.kill("SIGKILL");
    await rm(temporaryRoot, { recursive: true, force: true });
  }
}

async function main() {
  try {
    const result = await runProbe();
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
