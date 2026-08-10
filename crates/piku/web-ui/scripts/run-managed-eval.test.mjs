import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import { test } from "node:test";
import path from "node:path";
import {
  evaluationArtifactPaths,
  managedArtifactDir,
  validateRunId,
  writeBindingWithoutMaskingChildFailure,
  writeManagedLifecycleBinding,
} from "./run-managed-eval.mjs";

test("managed run IDs accept only bounded filename components", () => {
  assert.equal(validateRunId("2026-08-10T12-00-00-000Z"), "2026-08-10T12-00-00-000Z");
  for (const runId of ["", ".", "..", "../escape", "/tmp/escape", "run/escape", "run\\escape", "run--escape", `${"a".repeat(129)}`])
    assert.throws(() => validateRunId(runId), /PIKU_EVAL_RUN_ID/);
});

test("managed artifact directories are confined below the managed root", () => {
  const root = path.resolve("/tmp/piku-managed-root");
  const artifactDir = managedArtifactDir(root, "run-one");
  assert.equal(artifactDir, path.join(root, ".artifacts", "playwright-agent", "managed", "run-one"));
  assert.throws(() => managedArtifactDir(root, "../escape"), /PIKU_EVAL_RUN_ID/);
});

test("managed lifecycle binding attests final server and parallel manifests without rewriting them", async (t) => {
  const root = await mkdtemp(path.join(os.tmpdir(), "piku-managed-binding-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  const runId = "run-one";
  const artifactDir = managedArtifactDir(root, runId);
  const serverDir = path.join(artifactDir, "server");
  const [manifestPath, promptManifestPath] = evaluationArtifactPaths(root, "parallel", runId);
  await Promise.all([
    mkdir(serverDir, { recursive: true }),
    mkdir(path.dirname(manifestPath), { recursive: true }),
  ]);
  const lifecycleBytes = `${JSON.stringify({ ownership: "managed", status: "stopped" })}\n`;
  await Promise.all([
    writeFile(path.join(serverDir, "lifecycle.json"), lifecycleBytes),
    writeFile(path.join(serverDir, "server.log"), "final log line\n"),
    writeFile(manifestPath, "parallel manifest\n"),
    writeFile(promptManifestPath, "immutable prompt manifest\n"),
  ]);

  const before = await readFile(promptManifestPath, "utf8");
  const { bindingPath, binding } = await writeManagedLifecycleBinding({
    root, artifactDir, mode: "parallel", runId, outcome: { code: 0, signal: null },
  });
  assert.equal(await readFile(promptManifestPath, "utf8"), before);
  assert.equal(binding.server.lifecycle.sha256,
    createHash("sha256").update(lifecycleBytes).digest("hex"));
  assert.deepEqual(binding.evaluation_artifacts.map((item) => item.path), [
    path.relative(root, manifestPath),
    path.relative(root, promptManifestPath),
  ]);
  assert.deepEqual(binding.child, { exit_code: 0, exit_signal: null });
  assert.deepEqual(binding.expected_but_missing, []);
  assert.deepEqual(JSON.parse(await readFile(bindingPath, "utf8")), binding);
  await assert.rejects(writeManagedLifecycleBinding({
    root, artifactDir, mode: "parallel", runId, outcome: { code: 0, signal: null },
  }), /EEXIST/);
});

test("failed managed runs attest partial artifacts and name expected missing outputs", async (t) => {
  const root = await mkdtemp(path.join(os.tmpdir(), "piku-managed-failed-binding-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  const runId = "failed-run";
  const artifactDir = managedArtifactDir(root, runId);
  const serverDir = path.join(artifactDir, "server");
  const [manifestPath, promptManifestPath] = evaluationArtifactPaths(root, "parallel", runId);
  await Promise.all([
    mkdir(serverDir, { recursive: true }),
    mkdir(path.dirname(manifestPath), { recursive: true }),
  ]);
  await Promise.all([
    writeFile(path.join(serverDir, "lifecycle.json"),
      `${JSON.stringify({ ownership: "managed", status: "stopped" })}\n`),
    writeFile(path.join(serverDir, "server.log"), "failed run log\n"),
    writeFile(promptManifestPath, "immutable prompt manifest\n"),
  ]);

  const { binding } = await writeManagedLifecycleBinding({
    root, artifactDir, mode: "parallel", runId, outcome: { code: 1, signal: null },
  });
  assert.deepEqual(binding.child, { exit_code: 1, exit_signal: null });
  assert.deepEqual(binding.evaluation_artifacts.map((item) => item.path), [
    path.relative(root, promptManifestPath),
  ]);
  assert.deepEqual(binding.expected_but_missing, [path.relative(root, manifestPath)]);
});

test("managed bindings preserve signal termination", async (t) => {
  const root = await mkdtemp(path.join(os.tmpdir(), "piku-managed-signal-binding-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  const runId = "signal-run";
  const artifactDir = managedArtifactDir(root, runId);
  const serverDir = path.join(artifactDir, "server");
  await mkdir(serverDir, { recursive: true });
  await Promise.all([
    writeFile(path.join(serverDir, "lifecycle.json"),
      `${JSON.stringify({ ownership: "managed", status: "stopped" })}\n`),
    writeFile(path.join(serverDir, "server.log"), "signal log\n"),
  ]);
  const { binding } = await writeManagedLifecycleBinding({
    root, artifactDir, mode: "e2e", runId, outcome: { code: null, signal: "SIGTERM" },
  });
  assert.deepEqual(binding.child, { exit_code: null, exit_signal: "SIGTERM" });
});

test("binding errors cannot replace a failed child outcome", async () => {
  const errors = [];
  const result = await writeBindingWithoutMaskingChildFailure({
    outcome: { code: 1, signal: null },
    writeBinding: async () => { throw new Error("disk full"); },
    reportError: (message) => errors.push(message),
  });
  assert.equal(result, null);
  assert.deepEqual(errors, ["Could not write managed lifecycle binding: disk full"]);
  await assert.rejects(writeBindingWithoutMaskingChildFailure({
    outcome: { code: 0, signal: null },
    writeBinding: async () => { throw new Error("disk full"); },
    reportError: () => assert.fail("successful child binding failures must propagate"),
  }), /disk full/);
});

test("focus-pair bindings cover the pair dossier and both immutable arm manifests", () => {
  const paths = evaluationArtifactPaths("/repo", "focus-pair", "pair-one")
    .map((item) => path.relative("/repo", item));
  assert.deepEqual(paths, [
    ".artifacts/playwright-agent/focus-pairs/pair-one/manifest.json",
    ".artifacts/playwright-agent/focus-pairs/pair-one/report.json",
    ".artifacts/playwright-agent/parallel/pair-one-blind/manifest.json",
    ".artifacts/playwright-agent/parallel/pair-one-blind/prompt-manifest.json",
    ".artifacts/playwright-agent/parallel/pair-one-focused/manifest.json",
    ".artifacts/playwright-agent/parallel/pair-one-focused/prompt-manifest.json",
  ]);
});
