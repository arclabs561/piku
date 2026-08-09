import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { mkdir, readFile, realpath, writeFile } from "node:fs/promises";
import path from "node:path";

export const PROMPT_MANIFEST_FILE = "prompt-manifest.json";
export const PROMPT_MANIFEST_SCHEMA = JSON.parse(readFileSync(
  new URL("../../../../eval/evaluation-prompt-manifest.schema.json", import.meta.url),
  "utf8",
));

export function canonicalJson(value) {
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(",")}]`;
  if (value && typeof value === "object") {
    return `{${Object.keys(value).sort().map((key) => `${JSON.stringify(key)}:${canonicalJson(value[key])}`).join(",")}}`;
  }
  return JSON.stringify(value);
}

export function sha256(value) {
  return createHash("sha256").update(value).digest("hex");
}

export function attestedValue(value) {
  return { sha256: sha256(canonicalJson(value)), value };
}

export async function attestedFiles(repoRoot, files) {
  return Promise.all(files.map(async ({ id, filePath }) => {
    const bytes = await readFile(filePath);
    return {
      id,
      path: path.relative(repoRoot, filePath),
      sha256: sha256(bytes),
      size_bytes: bytes.byteLength,
    };
  }));
}

export function validatePromptManifest(manifest, runId = manifest?.run_id) {
  if (!manifest || typeof manifest !== "object" || Array.isArray(manifest))
    throw new Error("prompt manifest must be an object");
  for (const key of PROMPT_MANIFEST_SCHEMA.required) {
    if (!(key in manifest)) throw new Error(`prompt manifest lacks required field: ${key}`);
  }
  const rootProperties = new Set(Object.keys(PROMPT_MANIFEST_SCHEMA.properties));
  if (PROMPT_MANIFEST_SCHEMA.additionalProperties === false
    && Object.keys(manifest).some((key) => !rootProperties.has(key)))
    throw new Error("prompt manifest contains an unknown field");
  if (manifest.schema_version !== PROMPT_MANIFEST_SCHEMA.properties.schema_version.const)
    throw new Error("prompt manifest schema version is unsupported");
  if (manifest.run_id !== runId) throw new Error("prompt manifest run ID does not match requested run");
  if (typeof manifest.surface !== "string" || manifest.surface.length === 0
    || typeof manifest.subject !== "object" || typeof manifest.evaluator !== "object")
    throw new Error("prompt manifest has invalid identity fields");
  if (!Array.isArray(manifest.roles) || manifest.roles.length === 0)
    throw new Error("prompt manifest lacks evaluator roles");
  const roleSchema = PROMPT_MANIFEST_SCHEMA.properties.roles.items;
  const roleProperties = new Set(Object.keys(roleSchema.properties));
  const roleNames = new Set();
  for (const role of manifest.roles) {
    for (const key of roleSchema.required) {
      if (!(key in role)) throw new Error(`prompt manifest role lacks required field: ${key}`);
    }
    if (roleSchema.additionalProperties === false
      && Object.keys(role).some((key) => !roleProperties.has(key)))
      throw new Error("prompt manifest role contains an unknown field");
    if (typeof role.role !== "string" || role.role.length === 0 || roleNames.has(role.role)
      || typeof role.provider !== "string" || role.provider.length === 0
      || typeof role.model !== "string" || role.model.length === 0)
      throw new Error("prompt manifest contains an invalid evaluator role");
    roleNames.add(role.role);
    if (!Array.isArray(role.prompt_assets) || role.prompt_assets.length === 0)
      throw new Error("prompt manifest role lacks prompt assets");
    for (const item of role.prompt_assets) {
      if (typeof item.kind !== "string" || typeof item.path !== "string"
        || !/^[a-f0-9]{64}$/.test(item.sha256) || !Number.isSafeInteger(item.size_bytes)
        || item.size_bytes < 0)
        throw new Error("prompt manifest contains an invalid file attestation");
    }
    for (const item of [role.context_contract, role.tools]) {
      if (!item || !/^[a-f0-9]{64}$/.test(item.sha256)
        || sha256(canonicalJson(item.value)) !== item.sha256)
        throw new Error("prompt manifest contains an invalid role attestation");
    }
    if (!role.limits || typeof role.limits !== "object" || Array.isArray(role.limits))
      throw new Error("prompt manifest role has invalid limits");
  }
  if (!/^[a-f0-9]{64}$/.test(manifest.effective_config?.sha256 || "")
    || sha256(canonicalJson(manifest.effective_config.value)) !== manifest.effective_config.sha256)
    throw new Error("prompt manifest contains an invalid effective configuration attestation");
  return manifest;
}

export async function writePromptManifest(runDir, manifest) {
  validatePromptManifest(manifest);
  await mkdir(runDir, { recursive: true });
  const manifestPath = path.join(runDir, PROMPT_MANIFEST_FILE);
  const contents = `${JSON.stringify(manifest, null, 2)}\n`;
  await writeFile(manifestPath, contents, { encoding: "utf8", flag: "wx", mode: 0o600 });
  return {
    path: PROMPT_MANIFEST_FILE,
    sha256: sha256(contents),
  };
}

export async function verifyPromptManifest(runDir, runId, reference, repoRoot = null) {
  if (reference?.path !== PROMPT_MANIFEST_FILE || !/^[a-f0-9]{64}$/.test(reference?.sha256 || ""))
    throw new Error("run manifest lacks a canonical prompt manifest reference");
  const runRoot = await realpath(runDir);
  const manifestPath = await realpath(path.join(runRoot, reference.path));
  const relative = path.relative(runRoot, manifestPath);
  if (relative.startsWith("..") || path.isAbsolute(relative))
    throw new Error("prompt manifest escapes the run directory");
  const contents = await readFile(manifestPath, "utf8");
  if (sha256(contents) !== reference.sha256) throw new Error("prompt manifest digest mismatch");
  const manifest = validatePromptManifest(JSON.parse(contents), runId);
  if (repoRoot) {
    const repositoryRoot = await realpath(repoRoot);
    for (const item of manifest.roles.flatMap((role) => role.prompt_assets)) {
      const inputPath = await realpath(path.resolve(repositoryRoot, item.path));
      const inputRelative = path.relative(repositoryRoot, inputPath);
      if (inputRelative.startsWith("..") || path.isAbsolute(inputRelative))
        throw new Error(`prompt manifest input escapes the repository: ${item.kind}:${item.path}`);
      const bytes = await readFile(inputPath);
      if (bytes.byteLength !== item.size_bytes || sha256(bytes) !== item.sha256)
        throw new Error(`prompt manifest input digest mismatch: ${item.kind}:${item.path}`);
    }
  }
  return { manifest, manifestPath, reference: { ...reference } };
}
