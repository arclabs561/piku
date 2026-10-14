import assert from "node:assert/strict";
import test from "node:test";
import { mkdtemp, mkdir, readFile, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { frontPorchArgs, frontPorchPattern, runDeterministicFrontPorch } from "./deterministic-front-porch.mjs";

// Exercise the production Node/Playwright CLI launch seam without a browser.
async function fixture(t, source) {
  const dir = await mkdtemp(path.join(os.tmpdir(), "piku-front-porch-"));
  const cliDir = path.join(dir, "node_modules/@playwright/test");
  await mkdir(cliDir, { recursive: true });
  await writeFile(path.join(cliDir, "cli.js"), source);
  t.after(() => rm(dir, { recursive: true, force: true }));
  return {
    baseUrl: new URL("http://127.0.0.1:1"),
    webUiDir: dir,
    outputDir: path.join(dir, "result"),
    timeoutMs: 300,
    maxOutputBytes: 64,
  };
}

test("front porch bounds a stuck CLI after its ready marker", async (t) => {
  // The fixture's own exit bounds the red run against the original helper.
  const options = await fixture(t, 'console.log("ready"); setTimeout(() => process.exit(0), 1500);');
  const start = Date.now();
  await assert.rejects(runDeterministicFrontPorch(options), /timeout after 300ms; signal SIGTERM/);
  assert.ok(Date.now() - start < 1200);
  assert.equal(await readFile(path.join(options.outputDir, "output.txt"), "utf8"), "ready\n");
});

test("front porch caps combined stdout and stderr even on exit zero", async (t) => {
  const options = await fixture(t, 'process.stdout.write("a".repeat(40)); setTimeout(() => process.stderr.write("b".repeat(40)), 30);');
  await assert.rejects(runDeterministicFrontPorch(options), /output exceeded 64 bytes/);
  const output = await readFile(path.join(options.outputDir, "output.txt"));
  assert.equal(output.length, 64);
  assert.equal(output.toString(), "a".repeat(40) + "b".repeat(24));
});

test("front porch preserves successful output through pipe close", async (t) => {
  const options = await fixture(t, 'process.stdout.write("out"); setTimeout(() => process.stderr.write("err"), 30);');
  assert.equal(await runDeterministicFrontPorch(options), "outerr");
  assert.equal(await readFile(path.join(options.outputDir, "output.txt"), "utf8"), "outerr");
});

test("front porch keeps the deadline after leader exit with inherited pipes", { skip: process.platform === "win32" }, async (t) => {
  const options = await fixture(t, `
    const { spawn } = require("node:child_process");
    spawn(process.execPath, ["-e", 'console.log("descendant-ready"); setTimeout(() => {}, 1500)'], { stdio: ["ignore", 1, 2] });
    process.exit(0);
  `);
  const start = Date.now();
  await assert.rejects(runDeterministicFrontPorch(options), /timeout after 300ms; exit 0/);
  assert.ok(Date.now() - start < 1200);
  assert.equal(await readFile(path.join(options.outputDir, "output.txt"), "utf8"), "descendant-ready\n");
});

test("front porch escalates when the child ignores TERM", { skip: process.platform === "win32" }, async (t) => {
  const options = await fixture(t, 'process.on("SIGTERM", () => console.error("term-observed")); console.log("ready"); setTimeout(() => process.exit(0), 8000);');
  const start = Date.now();
  await assert.rejects(runDeterministicFrontPorch(options), /timeout after 300ms; signal SIGKILL/);
  assert.ok(Date.now() - start >= 5000);
  assert.ok(Date.now() - start < 7000);
  assert.match(await readFile(path.join(options.outputDir, "output.txt"), "utf8"), /ready\nterm-observed\n/);
});

test("front porch writes at most the byte cap across split UTF-8", async (t) => {
  const options = await fixture(t, 'process.stdout.write("é");');
  options.maxOutputBytes = 1;
  await assert.rejects(runDeterministicFrontPorch(options), /output exceeded 1 bytes/);
  assert.deepEqual(await readFile(path.join(options.outputDir, "output.txt")), Buffer.from([0xc3]));
});

test("front porch permits output exactly at the cap", async (t) => {
  const options = await fixture(t, 'process.stdout.write("a".repeat(64));');
  assert.equal(await runDeterministicFrontPorch(options), "a".repeat(64));
});

test("front porch preserves nonzero exit and stderr", async (t) => {
  const options = await fixture(t, 'process.stderr.write("bad"); process.exitCode = 7;');
  await assert.rejects(runDeterministicFrontPorch(options), /exit 7\)\nbad/);
});

test("front porch preserves signal failure", { skip: process.platform === "win32" }, async (t) => {
  const options = await fixture(t, 'process.kill(process.pid, "SIGTERM");');
  await assert.rejects(runDeterministicFrontPorch(options), /signal SIGTERM/);
});

test("front porch records spawn failure", async (t) => {
  const options = await fixture(t, "");
  options.webUiDir = path.join(options.webUiDir, "missing");
  await assert.rejects(runDeterministicFrontPorch(options), /spawn.*ENOENT/);
  assert.equal((await readFile(path.join(options.outputDir, "output.txt"))).length, 0);
});

test("front porch rejects invalid limits before creating output", async (t) => {
  const options = await fixture(t, "");
  for (const name of ["timeoutMs", "maxOutputBytes"]) {
    for (const value of [0, -1, 0.5, NaN, Infinity, Number.MAX_SAFE_INTEGER + 1]) {
      await assert.rejects(runDeterministicFrontPorch({ ...options, [name]: value }), new RegExp(name));
    }
  }
  await assert.rejects(runDeterministicFrontPorch({ ...options, timeoutMs: 2_147_483_648 }), /timeoutMs/);
  await assert.rejects(readFile(path.join(options.outputDir, "output.txt")), /ENOENT/);
});

test("front porch runs the cheap operator and interaction predicates", () => {
  const args = frontPorchArgs();
  assert.deepEqual(args.slice(1, 4), [
    "test",
    "e2e/operator-journey.spec.js",
    "e2e/workspace.spec.js",
  ]);
  assert.equal(args.at(-2), "--grep");
  assert.equal(args.at(-1), frontPorchPattern);
  for (const behavior of [
    "inspect, contextualize, rerun, and resume",
    "blank-canvas click closes",
    "Escape dismisses",
    "drag and persist",
    "corner handles resize",
    "rerun from edited turns",
    "crosses browser contexts",
    "fixture cancellation survives fresh context",
    "crowded desktop cards remain reachable",
  ]) assert.match(frontPorchPattern, new RegExp(behavior));
});
