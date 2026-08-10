import { spawn } from "node:child_process";
import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";

export const frontPorchPattern = [
  "an operator can inspect, contextualize, rerun, and resume",
  "a second blank-canvas click closes the creation menu",
  "Escape dismisses the creation menu",
  "notes drag and persist through the server",
  "corner handles resize, reposition, and persist workspace cards",
  "chat cards persist isolated notebook history and rerun from edited turns",
].join("|");

export function frontPorchArgs() {
  return [
    "./node_modules/@playwright/test/cli.js",
    "test",
    "e2e/operator-journey.spec.js",
    "e2e/workspace.spec.js",
    "--grep",
    frontPorchPattern,
  ];
}

export async function runDeterministicFrontPorch({
  baseUrl,
  webUiDir,
  outputDir,
}) {
  await mkdir(outputDir, { recursive: true });
  const child = spawn(process.execPath, frontPorchArgs(), {
    cwd: webUiDir,
    env: {
      ...process.env,
      PIKU_WEB_URL: baseUrl.toString(),
      PLAYWRIGHT_OUTPUT_DIR: path.join(outputDir, "playwright-output"),
    },
    stdio: ["ignore", "pipe", "pipe"],
  });
  const chunks = [];
  child.stdout.on("data", (chunk) => chunks.push(chunk));
  child.stderr.on("data", (chunk) => chunks.push(chunk));
  const outcome = await new Promise((resolve, reject) => {
    child.once("error", reject);
    child.once("exit", (code, signal) => resolve({ code, signal }));
  });
  const output = Buffer.concat(chunks).toString("utf8");
  await writeFile(path.join(outputDir, "output.txt"), output, "utf8");
  if (outcome.code !== 0) {
    const detail = outcome.signal ? `signal ${outcome.signal}` : `exit ${outcome.code}`;
    throw new Error(`deterministic front porch failed (${detail})\n${output}`);
  }
  console.error("[piku eval] deterministic front porch passed");
  return output;
}
