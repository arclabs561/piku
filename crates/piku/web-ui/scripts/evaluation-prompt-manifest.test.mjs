import assert from "node:assert/strict";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { test } from "node:test";
import path from "node:path";
import {
  PROMPT_MANIFEST_SCHEMA, attestedFiles, attestedValue, canonicalJson, verifyPromptManifest, writePromptManifest,
} from "./evaluation-prompt-manifest.mjs";

async function fixture(t) {
  const root = await mkdtemp(path.join(process.env.TMPDIR || "/tmp", "piku-prompt-manifest-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  const repo = path.join(root, "repo");
  const runDir = path.join(repo, "run");
  await mkdir(path.join(repo, "inputs"), { recursive: true });
  const prompt = path.join(repo, "inputs", "prompt.md");
  const schema = path.join(repo, "inputs", "schema.json");
  await writeFile(prompt, "judge this exact surface\n");
  await writeFile(schema, '{"type":"object"}\n');
  const [promptAsset] = await attestedFiles(repo, [{ id: "explorer", filePath: prompt }]);
  const [schemaAsset] = await attestedFiles(repo, [{ id: "explorer", filePath: schema }]);
  const manifest = {
    schema_version: 1,
    run_id: "run-1",
    surface: "web",
    subject: { revision: "abc123" },
    evaluator: { runtime: "codex-cli" },
    roles: [{
      role: "explorer",
      provider: "codex",
      model: "gpt-5.6-sol",
      prompt_assets: [
        { kind: "prompt_template", path: promptAsset.path, sha256: promptAsset.sha256, size_bytes: promptAsset.size_bytes },
        { kind: "output_schema", path: schemaAsset.path, sha256: schemaAsset.sha256, size_bytes: schemaAsset.size_bytes },
      ],
      context_contract: attestedValue({ surface: "qa-explorer" }),
      tools: attestedValue({ executable: "codex", tools: ["browser_snapshot"] }),
      limits: { timeout_ms: 1000 },
    }],
    effective_config: attestedValue({ model: "gpt-5.6-sol", timeout_ms: 1000 }),
  };
  return { repo, runDir, prompt, manifest };
}

test("canonical JSON makes configuration hashes independent of insertion order", () => {
  assert.equal(canonicalJson({ b: 2, a: { d: 4, c: 3 } }), canonicalJson({ a: { c: 3, d: 4 }, b: 2 }));
});

test("module validation follows the shared manifest schema fields", () => {
  assert.deepEqual(PROMPT_MANIFEST_SCHEMA.required, [
    "schema_version", "run_id", "surface", "subject", "evaluator", "roles", "effective_config",
  ]);
});

test("prompt manifests are write-once and verify exact inputs", async (t) => {
  const { repo, runDir, manifest } = await fixture(t);
  const reference = await writePromptManifest(runDir, manifest);
  const verified = await verifyPromptManifest(runDir, "run-1", reference, repo);
  assert.equal(verified.manifest.effective_config.value.model, "gpt-5.6-sol");
  await assert.rejects(writePromptManifest(runDir, manifest), /EEXIST/);
});

test("resume verification rejects manifest and source-input drift", async (t) => {
  const { repo, runDir, prompt, manifest } = await fixture(t);
  const reference = await writePromptManifest(runDir, manifest);
  const manifestPath = path.join(runDir, reference.path);
  await writeFile(manifestPath, `${await readFile(manifestPath, "utf8")} `);
  await assert.rejects(verifyPromptManifest(runDir, "run-1", reference, repo), /digest mismatch/);

  const second = await fixture(t);
  const secondReference = await writePromptManifest(second.runDir, second.manifest);
  await writeFile(second.prompt, "changed prompt\n");
  await assert.rejects(
    verifyPromptManifest(second.runDir, "run-1", secondReference, second.repo),
    /input digest mismatch: prompt_template:inputs\/prompt\.md/,
  );
});

test("resume verification rejects run identity drift", async (t) => {
  const { repo, runDir, manifest } = await fixture(t);
  const reference = await writePromptManifest(runDir, manifest);
  await assert.rejects(verifyPromptManifest(runDir, "another-run", reference, repo), /run ID/);
});
