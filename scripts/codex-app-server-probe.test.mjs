import assert from "node:assert/strict";
import { existsSync, statSync } from "node:fs";
import { mkdtemp, readFile, readdir, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { runProbe, writeAttestation } from "./codex-app-server-probe.mjs";

const test = process.argv[2] === "--fake-server" ? null : (await import("node:test")).test;

if (process.argv[2] === "--fake-server") {
  const mode = process.argv[3] ?? "pass";
  if (process.argv.includes("--version")) {
    process.stdout.write("codex-cli fake-1\n");
  } else if (mode === "timeout") {
    setInterval(() => {}, 1_000);
  } else {
    let buffer = "";
    let interactiveTurn = 0;
    let interactiveWorkspace;
    const reply = (id, result) => process.stdout.write(`${JSON.stringify({ id, result })}\n`);
    const notify = (method, params) => process.stdout.write(`${JSON.stringify({ method, params })}\n`);
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
          const authFile = path.join(process.env.CODEX_HOME ?? "", "auth.json");
          const credentialEnvPresent = Object.keys(process.env)
            .some((key) => /(?:API_KEY|TOKEN|SECRET|AUTHORIZATION)$/i.test(key));
          if (existsSync(authFile)) {
            const authMode = statSync(authFile).mode & 0o777;
            if (credentialEnvPresent || authMode !== 0o600 || process.env.HOME !== process.env.CODEX_HOME) {
              process.stdout.write(`${JSON.stringify({ id: request.id, error: { code: "unsafe_env", message: "interactive environment policy failed" } })}\n`);
              continue;
            }
          }
          reply(request.id, { userAgent: "fake" });
          continue;
        }
        if (request.method === "thread/start") {
          const params = request.params;
          const valid = params.sandbox === "workspace-write"
            && params.approvalPolicy === "never"
            && params.ephemeral === false
            && typeof params.cwd === "string";
          if (!valid) {
            process.stdout.write(`${JSON.stringify({ id: request.id, error: { message: "invalid thread/start" } })}\n`);
            continue;
          }
          interactiveWorkspace = params.cwd;
          reply(request.id, { thread: { id: "fake-thread" } });
          continue;
        }
        if (request.method === "thread/resume") {
          const params = request.params;
          const valid = params.threadId === "fake-thread"
            && params.cwd === interactiveWorkspace
            && params.sandbox === "workspace-write"
            && params.approvalPolicy === "never";
          if (!valid) {
            process.stdout.write(`${JSON.stringify({ id: request.id, error: { message: "invalid thread/resume" } })}\n`);
            continue;
          }
          reply(request.id, { thread: { id: "fake-thread" } });
          continue;
        }
        if (request.method === "turn/start") {
          const params = request.params;
          const policy = params.sandboxPolicy;
          const valid = params.threadId === "fake-thread"
            && params.approvalPolicy === "never"
            && Array.isArray(params.input) && params.input.length === 1
            && params.input[0].type === "text"
            && policy.type === "workspaceWrite"
            && policy.networkAccess === false
            && policy.excludeSlashTmp === true
            && policy.excludeTmpdirEnvVar === true
            && policy.writableRoots.length === 1
            && policy.writableRoots[0] === interactiveWorkspace;
          if (!valid) {
            process.stdout.write(`${JSON.stringify({ id: request.id, error: { message: "invalid turn/start" } })}\n`);
            continue;
          }
          interactiveTurn += 1;
          const turnId = `fake-turn-${interactiveTurn}`;
          reply(request.id, { turn: { id: turnId } });
          const text = params.input[0].text;
          const itemBase = { commandActions: [], cwd: interactiveWorkspace };
          if (interactiveTurn === 1) {
            const commandTarget = path.join(interactiveWorkspace, "interactive-command-sentinel");
            const fileTarget = path.join(interactiveWorkspace, "interactive-file-sentinel");
            if (!text.includes(commandTarget) || !text.includes(fileTarget)) throw new Error("unexpected first turn prompt");
            notify("item/started", { threadId: "fake-thread", turnId, item: { ...itemBase, id: "command-1", type: "commandExecution", command: `write ${commandTarget}`, status: "inProgress" } });
            await writeFile(commandTarget, "probe");
            notify("item/completed", { threadId: "fake-thread", turnId, item: { ...itemBase, id: "command-1", type: "commandExecution", command: `write ${commandTarget}`, status: "completed", exitCode: 0, aggregatedOutput: "probe\n" } });
            notify("item/started", { threadId: "fake-thread", turnId, item: { id: "file-1", type: "fileChange", status: "inProgress", changes: [{ path: fileTarget, kind: "add" }] } });
            await writeFile(fileTarget, "probe");
            notify("item/completed", { threadId: "fake-thread", turnId, item: { id: "file-1", type: "fileChange", status: "completed", changes: [{ path: fileTarget, kind: "add" }] } });
          } else {
            const siblingTarget = path.join(path.dirname(interactiveWorkspace), "interactive-sibling-sentinel");
            if (!text.includes(siblingTarget)) throw new Error("unexpected second turn prompt");
            notify("item/started", { threadId: "fake-thread", turnId, item: { ...itemBase, id: "command-2", type: "commandExecution", command: `write ${siblingTarget}`, status: "inProgress" } });
            notify("item/completed", { threadId: "fake-thread", turnId, item: { ...itemBase, id: "command-2", type: "commandExecution", command: `write ${siblingTarget}`, status: "failed", exitCode: 1, aggregatedOutput: "denied" } });
          }
          notify("turn/completed", { threadId: "fake-thread", turn: { id: turnId, status: "completed" } });
          continue;
        }
        if (request.method !== "command/exec") continue;
        if (mode === "rpc-error") {
          process.stdout.write(`${JSON.stringify({ id: request.id, error: { code: "bad_policy", message: `token=do-not-report ${"x".repeat(500)}` } })}\n`);
          continue;
        }
        const { command, cwd, sandboxPolicy } = request.params;
        if (command[1] === "-e" && command[2]?.includes("createConnection")) {
          reply(request.id, { exitCode: 1, stdout: "", stderr: "denied" });
          continue;
        }
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
        reply(request.id, { exitCode, stdout: "", stderr: "" });
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
  assert.equal(result.launchPolicy.policy_id, "piku.codex.workspace-write.v1");
  assert.equal(result.launchPolicy.synthetic_home, true);
  assert.deepEqual(result.launchPolicy.child_env_allowlist, [
    "LANG",
    "LC_ALL",
    "PATH",
    "SHELL",
    "SSL_CERT_DIR",
    "SSL_CERT_FILE",
    "TERM",
    "TMPDIR",
  ]);
  assert.deepEqual(result.protocol.readOnly, { passed: true, exitCode: 1, sentinelAbsent: true });
  assert.equal(result.protocol.workspaceWrite.passed, true);
  assert.deepEqual(result.protocol.network, { passed: true, exitCode: 1, accepted: 0 });
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

if (process.argv[2] !== "--fake-server") test("interactive probe copies auth but excludes provider secrets from child env", async () => {
  const authRoot = await mkdtemp(path.join(tmpdir(), "piku-fake-auth-"));
  const authFile = path.join(authRoot, "auth.json");
  const priorApiKey = process.env.OPENAI_API_KEY;
  try {
    await writeFile(authFile, "{}", { mode: 0o644 });
    process.env.OPENAI_API_KEY = "must-not-reach-child";
    const result = await withFake("pass", { interactive: true, authFile });
    assert.equal(result.ok, true);
    assert.deepEqual(
      {
        passed: result.interactive.passed,
        insideWritten: result.interactive.insideWritten,
        siblingDenied: result.interactive.siblingDenied,
        turnsCompleted: result.interactive.turnsCompleted,
      },
      { passed: true, insideWritten: true, siblingDenied: true, turnsCompleted: true },
    );
    assert.deepEqual(result.interactive.fixtures.map(({ event, type }) => ({ event, type })), [
      { event: "item/started", type: "commandExecution" },
      { event: "item/completed", type: "commandExecution" },
      { event: "item/started", type: "fileChange" },
      { event: "item/completed", type: "fileChange" },
      { event: "item/started", type: "commandExecution" },
      { event: "item/completed", type: "commandExecution" },
    ]);
    assert.match(result.interactive.fixtures[0].cwd, /^\$WORKSPACE$/);
  } finally {
    if (priorApiKey === undefined) delete process.env.OPENAI_API_KEY;
    else process.env.OPENAI_API_KEY = priorApiKey;
    await rm(authRoot, { recursive: true, force: true });
  }
});

if (process.argv[2] !== "--fake-server") test("RPC errors identify the method with bounded redacted detail", async () => {
  await assert.rejects(withFake("rpc-error"), (error) => {
    assert.match(error.message, /^App server command\/exec failed \(bad_policy\):/);
    assert.match(error.message, /token=\[redacted\]/);
    assert.doesNotMatch(error.message, /do-not-report/);
    assert.ok(error.message.length < 350);
    return true;
  });
});

if (process.argv[2] !== "--fake-server") test("bounds app-server response time and removes temporary state", async () => {
  const before = new Set(await readdir(tmpdir()));
  await assert.rejects(withFake("timeout", { timeoutMs: 50 }), /timed out/);
  const after = await readdir(tmpdir());
  assert.deepEqual(after.filter((name) => name.startsWith("piku-codex-probe-") && !before.has(name)), []);
});

if (process.argv[2] !== "--fake-server") test("attestation records exact evidence and keeps unproven gates false", async () => {
  const directory = await mkdtemp(path.join(tmpdir(), "piku-attestation-test-"));
  const target = path.join(directory, "workspace-write-attestation.json");
  try {
    const authFile = path.join(directory, "auth.json");
    await writeFile(authFile, "{}", { mode: 0o600 });
    const result = await withFake("pass", { interactive: true, authFile });
    const written = await writeAttestation(target, result);
    assert.equal(written.complete, false);
    const attestation = JSON.parse(await readFile(target, "utf8"));
    assert.equal(attestation.schema, "piku.codex-write-attestation.v1");
    assert.ok(attestation.passed_gates.includes("command_write_inside"));
    assert.ok(attestation.passed_gates.includes("network_denied"));
    assert.ok(!attestation.passed_gates.includes("elevation_denied"));
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});
