import assert from "node:assert/strict";
import { readdir, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { runProbe } from "./codex-app-server-probe.mjs";

const test = process.argv[2] === "--fake-server" ? null : (await import("node:test")).test;

if (process.argv[2] === "--fake-server") {
  const mode = process.argv[3] ?? "pass";
  if (process.argv.includes("--version")) {
    process.stdout.write("codex-cli fake-1\n");
  } else if (mode === "timeout") {
    setInterval(() => {}, 1_000);
  } else {
    let buffer = "";
    process.stdin.setEncoding("utf8");
    process.stdin.on("data", async (data) => {
      buffer += data;
      for (;;) {
        const newline = buffer.indexOf("\n");
        if (newline < 0) break;
        const line = buffer.slice(0, newline);
        buffer = buffer.slice(newline + 1);
        const request = JSON.parse(line);
        if (request.method === "initialized") continue;
        if (request.method === "initialize") {
          process.stdout.write(`${JSON.stringify({ id: request.id, result: { userAgent: "fake" } })}\n`);
          continue;
        }
        if (request.method !== "command/exec") continue;
        const { command, cwd, sandboxPolicy } = request.params;
        const target = command.at(-1);
        const inside = path.dirname(target) === cwd;
        const validCommon = Array.isArray(command) && command.length === 4
          && request.params.timeoutMs > 0 && request.params.outputBytesCap > 0;
        const validReadOnly = validCommon && sandboxPolicy.type === "readOnly"
          && sandboxPolicy.networkAccess === false;
        const validWorkspace = validCommon && sandboxPolicy.type === "workspaceWrite"
          && sandboxPolicy.networkAccess === false
          && sandboxPolicy.excludeSlashTmp === true
          && sandboxPolicy.excludeTmpdirEnvVar === true
          && sandboxPolicy.writableRoots.length === 1
          && sandboxPolicy.writableRoots[0] === cwd;
        const shouldWrite = mode === "escape" || (validWorkspace && inside);
        if (shouldWrite) await writeFile(target, "probe");
        const denied = inside ? validReadOnly : validWorkspace;
        const exitCode = mode === "ambiguous" ? 0 : shouldWrite ? 0 : denied ? 1 : 0;
        process.stdout.write(`${JSON.stringify({ id: request.id, result: { exitCode, stdout: "", stderr: "" } })}\n`);
      }
    });
  }
}

const fakeOptions = (mode, overrides = {}) => ({
  executable: process.execPath,
  prefixArgs: [path.resolve(import.meta.dirname, "codex-app-server-probe.test.mjs"), "--fake-server", mode],
  ...overrides,
});

async function withFake(mode, options = {}) {
  return runProbe(fakeOptions(mode, options));
}

if (process.argv[2] !== "--fake-server") test("sends explicit sandbox policies and accepts only enforced boundaries", async () => {
  const result = await withFake("pass");
  assert.equal(result.ok, true);
  assert.equal(result.codexVersion, "codex-cli fake-1");
  assert.deepEqual(result.protocol.readOnly, { passed: true, exitCode: 1, sentinelAbsent: true });
  assert.equal(result.protocol.workspaceWrite.passed, true);
});

if (process.argv[2] !== "--fake-server") test("fails closed when a denied command reports success", async () => {
  const result = await withFake("ambiguous");
  assert.equal(result.ok, false);
  assert.equal(result.protocol.readOnly.passed, false);
  assert.equal(result.protocol.workspaceWrite.passed, false);
});

if (process.argv[2] !== "--fake-server") test("fails closed when workspace-write escapes its writable root", async () => {
  const result = await withFake("escape");
  assert.equal(result.ok, false);
  assert.equal(result.protocol.readOnly.sentinelAbsent, false);
  assert.equal(result.protocol.workspaceWrite.outsideSentinelAbsent, false);
});

if (process.argv[2] !== "--fake-server") test("bounds app-server response time and removes temporary state", async () => {
  const before = new Set(await readdir(tmpdir()));
  await assert.rejects(withFake("timeout", { timeoutMs: 50 }), /timed out/);
  const after = await readdir(tmpdir());
  assert.deepEqual(after.filter((name) => name.startsWith("piku-codex-probe-") && !before.has(name)), []);
});
