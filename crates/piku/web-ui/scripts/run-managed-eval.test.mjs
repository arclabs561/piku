import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import { test } from "node:test";
import path from "node:path";
import {
  evaluationArtifactPaths,
  managedEvaluationTimeoutMs,
  managedTerminalEnabled,
  managedPageBroker,
  managedJudgeEnvironment,
  managedArtifactDir,
  evaluationReviewRoot,
  webFocusPairReviewDirectory,
  webReviewDirectory,
  webRoleReviewDirectory,
  resolveManagedEvaluationEnvironment,
  validateRunId,
  writeBindingWithoutMaskingChildFailure,
  writeReviewWithoutMaskingChildFailure,
  writeManagedLifecycleBinding,
} from "./run-managed-eval.mjs";

test("model-driven managed judges cannot reach the host terminal", () => {
  assert.equal(managedTerminalEnabled("e2e"), true);
  for (const mode of ["single", "parallel", "focus-pair"])
    assert.equal(managedTerminalEnabled(mode), false);
});

test("model-driven managed runs use a parent-only page broker when configured", () => {
  assert.equal(managedPageBroker("e2e", { OPENROUTER_API_KEY: "secret" }), null);
  assert.throws(() => managedPageBroker("parallel", {}), /requires OPENROUTER_API_KEY/);
  assert.deepEqual(managedPageBroker("parallel", { OPENROUTER_API_KEY: "secret" }), {
    model: "openai/gpt-5.6-terra",
  });
  assert.deepEqual(managedPageBroker("single", {
    OPENROUTER_API_KEY: "secret",
    PIKU_EVAL_PAGE_MODEL: "anthropic/custom",
  }), { model: "anthropic/custom" });
});

test("managed model-driven runs resolve only the broker key from the nearest ancestor env file", async (t) => {
  const root = await mkdtemp(path.join(os.tmpdir(), "piku-managed-env-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  const nested = path.join(root, "one", "two");
  await mkdir(nested, { recursive: true });
  await writeFile(path.join(root, ".env"), [
    "OPENROUTER_API_KEY='farther-secret'",
    "",
  ].join("\n"));
  await writeFile(path.join(root, "one", ".env"), [
    "OPENROUTER_API_KEY='nearest-secret'",
    "UNRELATED_SECRET=must-not-load",
    "",
  ].join("\n"));
  const source = { PATH: "/bin" };

  const resolved = await resolveManagedEvaluationEnvironment({
    mode: "parallel", environment: source, startDir: nested,
  });

  assert.deepEqual(source, { PATH: "/bin" });
  assert.equal(resolved.OPENROUTER_API_KEY, "nearest-secret");
  assert.equal(resolved.UNRELATED_SECRET, undefined);
});

test("process broker credentials take precedence and never enter the judge environment", async () => {
  const source = { OPENROUTER_API_KEY: "process-secret", PATH: "/bin" };
  const resolved = await resolveManagedEvaluationEnvironment({
    mode: "single", environment: source, startDir: "/path/that/need/not/exist",
  });
  assert.equal(resolved.OPENROUTER_API_KEY, "process-secret");
  assert.deepEqual(managedJudgeEnvironment(resolved), { PATH: "/bin" });
  assert.equal(source.OPENROUTER_API_KEY, "process-secret");
});

test("managed model-driven runs fail preflight without a broker credential", async (t) => {
  const root = await mkdtemp(path.join(os.tmpdir(), "piku-managed-no-env-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  await assert.rejects(resolveManagedEvaluationEnvironment({
    mode: "focus-pair", environment: {}, startDir: root,
  }), /requires OPENROUTER_API_KEY/);
});

test("deterministic e2e remains credential-free", async () => {
  assert.deepEqual(await resolveManagedEvaluationEnvironment({
    mode: "e2e", environment: { PATH: "/bin" }, startDir: "/does/not/matter",
  }), { PATH: "/bin" });
});

test("managed evaluation timeout is bounded and explicit", () => {
  assert.equal(managedEvaluationTimeoutMs({}), 300_000);
  assert.equal(managedEvaluationTimeoutMs({ PIKU_MANAGED_EVAL_TIMEOUT_MS: "1500" }), 1500);
  assert.throws(() => managedEvaluationTimeoutMs({ PIKU_MANAGED_EVAL_TIMEOUT_MS: "999" }), /1000 to 3600000/);
});

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

test("web review directories are confined below the human review root", () => {
  const root = path.resolve("/tmp/piku-managed-root");
  assert.equal(
    webReviewDirectory(path.join(root, "reviews"), "run-one"),
    path.join(root, "reviews", "web-run-one"),
  );
  assert.throws(() => webReviewDirectory(path.join(root, "reviews"), "../escape"), /PIKU_EVAL_RUN_ID/);
  assert.equal(
    webRoleReviewDirectory(path.join(root, "reviews"), "run-one", "recovery"),
    path.join(root, "reviews", "web-run-one", "roles", "recovery"),
  );
  assert.equal(
    webFocusPairReviewDirectory(path.join(root, "reviews"), "pair-one"),
    path.join(root, "reviews", "web-focus-pairs", "pair-one"),
  );
});

test("review artifacts default outside CloudDocs and permit an explicit override", () => {
  assert.equal(
    evaluationReviewRoot({}, "/Users/example"),
    "/Users/example/Library/Caches/piku/eval-review",
  );
  assert.equal(
    evaluationReviewRoot({ PIKU_EVAL_REVIEW_DIR: "/Volumes/evals" }, "/Users/example"),
    "/Volumes/evals",
  );
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
    root, reviewRoot: root, artifactDir, mode: "parallel", runId, outcome: { code: 0, signal: null },
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
    root, reviewRoot: root, artifactDir, mode: "parallel", runId, outcome: { code: 0, signal: null },
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
    root, reviewRoot: root, artifactDir, mode: "parallel", runId, outcome: { code: 1, signal: null },
  });
  assert.deepEqual(binding.child, { exit_code: 1, exit_signal: null });
  assert.deepEqual(binding.evaluation_artifacts.map((item) => item.path), [
    path.relative(root, promptManifestPath),
  ]);
  assert.deepEqual(binding.expected_but_missing, [path.relative(root, manifestPath)]);
});

test("managed lifecycle bindings attest review artifacts outside the repository", async (t) => {
  const root = await mkdtemp(path.join(os.tmpdir(), "piku-managed-repo-"));
  const reviewRoot = await mkdtemp(path.join(os.tmpdir(), "piku-managed-review-"));
  t.after(() => Promise.all([
    rm(root, { recursive: true, force: true }), rm(reviewRoot, { recursive: true, force: true }),
  ]));
  const runId = "external-review";
  const artifactDir = managedArtifactDir(root, runId);
  const serverDir = path.join(artifactDir, "server");
  const [manifestPath, promptManifestPath] = evaluationArtifactPaths(reviewRoot, "parallel", runId);
  await Promise.all([
    mkdir(serverDir, { recursive: true }), mkdir(path.dirname(manifestPath), { recursive: true }),
  ]);
  await Promise.all([
    writeFile(path.join(serverDir, "lifecycle.json"), `${JSON.stringify({ ownership: "managed", status: "stopped" })}\n`),
    writeFile(path.join(serverDir, "server.log"), "finished\n"),
    writeFile(manifestPath, "parallel manifest\n"), writeFile(promptManifestPath, "prompt manifest\n"),
  ]);
  const { binding } = await writeManagedLifecycleBinding({
    root, reviewRoot, artifactDir, mode: "parallel", runId, outcome: { code: 0, signal: null },
  });
  assert.equal(binding.evaluation_artifacts_location, "local-review-cache");
  assert.deepEqual(binding.evaluation_artifacts.map((item) => item.path), [
    path.relative(reviewRoot, manifestPath), path.relative(reviewRoot, promptManifestPath),
  ]);
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
    root, reviewRoot: root, artifactDir, mode: "e2e", runId, outcome: { code: null, signal: "SIGTERM" },
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

test("asset-review errors cannot replace a failed child outcome", async () => {
  const messages = [];
  const result = await writeReviewWithoutMaskingChildFailure({
    outcome: { code: 1, signal: null },
    writeReview: async () => { throw new Error("review disk full"); },
    reportError: (message) => messages.push(message),
  });
  assert.equal(result, null);
  assert.match(messages[0], /review disk full/);
  await assert.rejects(writeReviewWithoutMaskingChildFailure({
    outcome: { code: 0, signal: null },
    writeReview: async () => { throw new Error("review disk full"); },
    reportError: () => {},
  }), /review disk full/);
});

test("focus-pair bindings cover the local pair dossier and both immutable arm manifests", () => {
  const paths = evaluationArtifactPaths("/repo", "focus-pair", "pair-one")
    .map((item) => path.relative("/repo", item));
  assert.deepEqual(paths, [
    "web-focus-pairs/pair-one/manifest.json",
    "web-focus-pairs/pair-one/report.json",
    "web-pair-one-blind/manifest.json",
    "web-pair-one-blind/prompt-manifest.json",
    "web-pair-one-focused/manifest.json",
    "web-pair-one-focused/prompt-manifest.json",
  ]);
});
