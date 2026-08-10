import assert from "node:assert/strict";
import { EventEmitter } from "node:events";
import { PassThrough } from "node:stream";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { test } from "node:test";
import {
  connectExternalEvaluationServer,
  startManagedEvaluationServer,
  validateEvaluationOrigin,
} from "./evaluation-server.mjs";

test("evaluation origins require an explicit loopback port", () => {
  assert.equal(validateEvaluationOrigin("http://127.0.0.1:43210").port, "43210");
  assert.throws(() => validateEvaluationOrigin("https://127.0.0.1:43210"));
  assert.throws(() => validateEvaluationOrigin("http://example.com:43210"));
  assert.throws(() => validateEvaluationOrigin("http://localhost"));
});

test("external evaluation server reports fixture availability without claiming ownership", async () => {
  const server = await connectExternalEvaluationServer("http://localhost:43210", {
    fetchImpl: async () => ({
      ok: true,
      json: async () => ({ executors: [{ id: "evaluation_fixture", available: false }] }),
    }),
  });
  assert.deepEqual(server.metadata, {
    ownership: "external",
    fixture_available: false,
    ready_file: null,
  });
});

test("managed evaluation server verifies fixture readiness and records teardown", async () => {
  const root = await mkdtemp(path.join(os.tmpdir(), "piku-eval-server-"));
  await mkdir(path.join(root, "crates/piku/web-ui/e2e/fixtures/operator-repo"), { recursive: true });
  await writeFile(path.join(root, "crates/piku/web-ui/e2e/fixtures/operator-repo/Cargo.toml"), "[package]\nname='fixture'\n");
  let spawned;
  const spawnImpl = (command, args, options) => {
    const child = new EventEmitter();
    child.pid = 4242;
    child.exitCode = null;
    child.stdout = new PassThrough();
    child.stderr = new PassThrough();
    spawned = { args, child, command, options };
    setTimeout(async () => {
      await writeFile(options.env.PIKU_WEB_READY_FILE, JSON.stringify({
        schema_version: 1,
        url: "http://localhost:43210",
        fixture_enabled: true,
        pid: 4242,
      }));
    }, 5);
    return child;
  };
  const killImpl = (_pid, signal) => {
    if (signal === "SIGTERM") {
      spawned.child.exitCode = 0;
      spawned.child.emit("exit", 0, null);
    }
  };
  try {
    const server = await startManagedEvaluationServer({
      repoRoot: root,
      artifactDir: root,
      spawnImpl,
      killImpl,
      fetchImpl: async () => ({
        ok: true,
        json: async () => ({ executors: [{ id: "evaluation_fixture", available: true }] }),
      }),
      timeoutMs: 1_000,
    });
    assert.equal(server.baseUrl.port, "43210");
    assert.equal(spawned.options.env.PIKU_NO_DOTENV, "1");
    assert.equal(spawned.options.env.PIKU_WEB_EVALUATION_FIXTURES, "1");
    assert.equal(spawned.options.env.PIKU_WEB_DISABLE_TERMINAL, "1");
    assert.equal(spawned.options.env.HOME, path.join(root, "server", "state", "home"));
    assert.equal(spawned.options.env.XDG_CONFIG_HOME, path.join(root, "server", "state", "config"));
    assert.equal(spawned.options.cwd, path.join(root, "server", "state", "workspace"));
    assert.deepEqual(spawned.options.env.PIKU_WEB_READY_FILE, path.join(root, "server", "ready.json"));
    assert.equal(spawned.command, "cargo");
    assert.deepEqual(spawned.args.slice(0, 4), [
      "run",
      "--quiet",
      "--manifest-path",
      path.join(root, "Cargo.toml"),
    ]);
    assert.equal(
      await readFile(path.join(spawned.options.cwd, "README.md"), "utf8"),
      "# Piku managed evaluation workspace\n\nThis workspace contains only deterministic test fixtures.\n",
    );
    await server.stop();
    const lifecycle = JSON.parse(await readFile(path.join(root, "server", "lifecycle.json"), "utf8"));
    assert.equal(lifecycle.status, "stopped");
    assert.equal(lifecycle.forced, false);
    assert.equal(lifecycle.workspace_root, path.join(root, "server", "state", "workspace"));
    assert.equal(lifecycle.terminal_enabled, false);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("managed evaluation server allowlists its environment and reaps a startup SIGKILL", async () => {
  const root = await mkdtemp(path.join(os.tmpdir(), "piku-eval-server-"));
  await mkdir(path.join(root, "crates/piku/web-ui/e2e/fixtures/operator-repo"), { recursive: true });
  const signals = [];
  let child;
  let reaped = false;
  const spawnImpl = (_command, _args, options) => {
    child = new EventEmitter();
    child.pid = 4343;
    child.exitCode = null;
    child.stdout = new PassThrough();
    child.stderr = new PassThrough();
    child.options = options;
    return child;
  };
  const killImpl = (_pid, signal) => {
    signals.push(signal);
    if (signal === "SIGKILL") {
      setTimeout(() => {
        reaped = true;
        child.emit("exit", null, "SIGKILL");
      }, 10);
    }
  };
  try {
    await assert.rejects(startManagedEvaluationServer({
      repoRoot: root,
      artifactDir: root,
      spawnImpl,
      killImpl,
      parentEnv: {
        PATH: "/safe/bin",
        HOME: "/operator",
        AWS_SECRET_ACCESS_KEY: "must-not-leak",
        OPENAI_API_KEY: "must-not-leak",
      },
      timeoutMs: 1,
      shutdownGraceMs: 1,
    }), /server readiness timed out/);
    assert.equal(child.options.env.PATH, "/safe/bin");
    assert.equal(child.options.env.CARGO_HOME, "/operator/.cargo");
    assert.equal(child.options.env.RUSTUP_HOME, "/operator/.rustup");
    assert.notEqual(child.options.env.HOME, "/operator");
    assert.equal(child.options.env.AWS_SECRET_ACCESS_KEY, undefined);
    assert.equal(child.options.env.OPENAI_API_KEY, undefined);
    assert.deepEqual(signals, ["SIGTERM", "SIGKILL"]);
    assert.equal(reaped, true);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("managed evaluation server records a spawn error without waiting for an exit", async () => {
  const root = await mkdtemp(path.join(os.tmpdir(), "piku-eval-server-"));
  await mkdir(path.join(root, "crates/piku/web-ui/e2e/fixtures/operator-repo"), { recursive: true });
  const child = new EventEmitter();
  child.exitCode = null;
  child.stdout = new PassThrough();
  child.stderr = new PassThrough();
  const startedAt = Date.now();
  try {
    const starting = startManagedEvaluationServer({
      repoRoot: root,
      artifactDir: root,
      spawnImpl: () => {
        setTimeout(() => child.emit("error", new Error("spawn cargo ENOENT")), 5);
        return child;
      },
      timeoutMs: 10_000,
      shutdownGraceMs: 1,
    });
    await assert.rejects(starting, /server failed to start: spawn cargo ENOENT/);
    assert.ok(Date.now() - startedAt < 1_000, "spawn errors should reject promptly");
    const lifecycle = JSON.parse(await readFile(path.join(root, "server", "lifecycle.json"), "utf8"));
    assert.equal(lifecycle.status, "failed");
    assert.equal(lifecycle.reason, "server failed to start: spawn cargo ENOENT");
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});
