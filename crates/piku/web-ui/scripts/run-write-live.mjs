import { execFile, spawn } from "node:child_process";
import { createHash } from "node:crypto";
import { createServer } from "node:net";
import { chmod, copyFile, mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { homedir, tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { promisify } from "node:util";

const execute = promisify(execFile);

const uiRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const repoRoot = path.resolve(uiRoot, "../../..");
const binary = path.join(repoRoot, "target", "debug", "piku");
const sourceConfig = process.env.XDG_CONFIG_HOME || path.join(homedir(), ".config");
const sourceAttestation = path.join(sourceConfig, "piku", "_codex", "workspace-write-attestation.json");
const temporary = await mkdtemp(path.join(tmpdir(), "piku-write-live-"));
const fixtureRoot = path.join(temporary, "workspace");
const configRoot = path.join(temporary, "config");
const codexRoot = path.join(configRoot, "piku", "_codex");
const runId = new Date().toISOString().replaceAll(":", "-");
const artifactDir = path.join(repoRoot, ".artifacts", "workspace-write-live", runId);

async function freePort() {
  return await new Promise((resolve, reject) => {
    const server = createServer();
    server.once("error", reject);
    server.listen(0, "127.0.0.1", () => {
      const { port } = server.address();
      server.close((error) => error ? reject(error) : resolve(port));
    });
  });
}

await mkdir(fixtureRoot, { recursive: true });
await mkdir(codexRoot, { recursive: true, mode: 0o700 });
await mkdir(artifactDir, { recursive: true });
await writeFile(path.join(fixtureRoot, "held-out.txt"), "before\n");
await copyFile(sourceAttestation, path.join(codexRoot, "workspace-write-attestation.json"));
await chmod(codexRoot, 0o700);
await chmod(path.join(codexRoot, "workspace-write-attestation.json"), 0o600);
const attestation = JSON.parse(await readFile(sourceAttestation, "utf8"));
const subjectRevision = (await execute("git", ["rev-parse", "HEAD"], { cwd: repoRoot })).stdout.trim();
const pikuVersion = (await execute(binary, ["--version"])).stdout.trim();
const subjectDirty = Boolean((await execute("git", ["status", "--porcelain"], { cwd: repoRoot })).stdout.trim());
const binarySha256 = createHash("sha256").update(await readFile(binary)).digest("hex");
const harnessSha256 = createHash("sha256")
  .update(await readFile(fileURLToPath(import.meta.url)))
  .update(await readFile(path.join(uiRoot, "e2e", "workspace-write-live.spec.js")))
  .digest("hex");

const port = await freePort();
const url = `http://127.0.0.1:${port}`;
const serverLog = [];
const server = spawn(binary, ["web", "--port", String(port)], {
  cwd: fixtureRoot,
  env: { ...process.env, XDG_CONFIG_HOME: configRoot },
  stdio: ["ignore", "pipe", "pipe"],
  detached: true,
});
for (const stream of [server.stdout, server.stderr]) {
  stream.setEncoding("utf8");
  stream.on("data", (chunk) => serverLog.push(chunk));
}

async function waitForServer() {
  const deadline = Date.now() + 20_000;
  while (Date.now() < deadline) {
    if (server.exitCode !== null) throw new Error(`Piku server exited ${server.exitCode}`);
    try {
      const response = await fetch(`${url}/api/executors`);
      if (response.ok) return;
    } catch {}
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
  throw new Error("Piku server did not become ready within 20 seconds");
}

let status = "failed";
let failure = null;
try {
  await waitForServer();
  const playwright = spawn(process.execPath, [
    path.join(uiRoot, "node_modules", "@playwright", "test", "cli.js"),
    "test",
    "e2e/workspace-write-live.spec.js",
    "--reporter=line",
  ], {
    cwd: uiRoot,
    env: {
      ...process.env,
      PIKU_WEB_URL: url,
      PIKU_LIVE_WRITE: "1",
      PIKU_WRITE_FIXTURE_ROOT: fixtureRoot,
      PIKU_WRITE_ARTIFACT_DIR: artifactDir,
    },
    stdio: "inherit",
  });
  const exitCode = await new Promise((resolve, reject) => {
    playwright.once("error", reject);
    playwright.once("exit", (code) => resolve(code ?? 1));
  });
  if (exitCode !== 0) throw new Error(`Playwright exited ${exitCode}`);
  status = "passed";
} catch (error) {
  failure = error;
} finally {
  if (server.exitCode === null) {
    try { process.kill(-server.pid, "SIGTERM"); } catch {}
  }
  await new Promise((resolve) => setTimeout(resolve, 100));
  await writeFile(path.join(artifactDir, "server.log"), serverLog.join(""));
  await writeFile(path.join(artifactDir, "manifest.json"), JSON.stringify({
    schema: "piku.workspace-write-live.v1",
    status,
    tested_at: new Date().toISOString(),
    subject_revision: subjectRevision,
    subject_worktree_dirty: subjectDirty,
    piku_binary_sha256: binarySha256,
    harness_sha256: harnessSha256,
    piku_version: pikuVersion,
    codex_version: attestation.codex_version,
    attestation_schema: attestation.schema,
    url,
    viewport: { width: 1280, height: 720 },
    fixture: "held-out.txt",
    failure: failure?.message || null,
  }, null, 2) + "\n");
  await rm(temporary, { recursive: true, force: true });
}

if (failure) throw failure;
console.log(`workspace-write live probe ${status}; artifacts: ${artifactDir}`);
