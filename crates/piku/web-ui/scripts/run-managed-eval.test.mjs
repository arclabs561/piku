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
    root, artifactDir, mode: "parallel", runId,
  });
  assert.equal(await readFile(promptManifestPath, "utf8"), before);
  assert.equal(binding.server.lifecycle.sha256,
    createHash("sha256").update(lifecycleBytes).digest("hex"));
  assert.deepEqual(binding.evaluation_artifacts.map((item) => item.path), [
    path.relative(root, manifestPath),
    path.relative(root, promptManifestPath),
  ]);
  assert.deepEqual(JSON.parse(await readFile(bindingPath, "utf8")), binding);
  await assert.rejects(writeManagedLifecycleBinding({
    root, artifactDir, mode: "parallel", runId,
  }), /EEXIST/);
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
