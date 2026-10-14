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
  "workspace state crosses browser contexts while viewport state does not",
  "managed evaluation fixture cancellation survives fresh context",
  "crowded desktop cards remain reachable through the object picker",
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
  timeoutMs = 600_000,
  maxOutputBytes = 1_048_576,
}) {
  for (const [name, value] of Object.entries({ timeoutMs, maxOutputBytes })) {
    if (!Number.isSafeInteger(value) || value <= 0 || (name === "timeoutMs" && value > 2_147_483_647)) {
      throw new Error(`${name} must be a positive integer within the supported range`);
    }
  }
  await mkdir(outputDir, { recursive: true });
  const child = spawn(process.execPath, frontPorchArgs(), {
    cwd: webUiDir,
    env: {
      ...process.env,
      PIKU_WEB_URL: baseUrl.toString(),
      PLAYWRIGHT_OUTPUT_DIR: path.join(outputDir, "playwright-output"),
      PIKU_REQUIRE_EVALUATION_FIXTURES: "1",
    },
    stdio: ["ignore", "pipe", "pipe"],
    detached: process.platform !== "win32",
  });
  const chunks = [];
  let retainedBytes = 0;
  let failure;
  let spawnError;
  let exit = { code: null, signal: null };
  const outcome = await new Promise((resolve) => {
    let deadline;
    let killTimer;
    let finished = false;
    const stopGroup = (signal) => {
      if (!child.pid) return;
      try {
        // The leader may have exited while a descendant still owns its pipes.
        if (process.platform === "win32") child.kill(signal);
        else process.kill(-child.pid, signal);
      } catch (error) {
        if (error.code !== "ESRCH") spawnError ??= error;
      }
    };
    const finish = () => {
      if (finished) return;
      finished = true;
      clearTimeout(deadline);
      clearTimeout(killTimer);
      if (failure) stopGroup("SIGKILL");
      resolve(exit);
    };
    const stop = (reason) => {
      if (failure || finished) return;
      failure = reason;
      stopGroup("SIGTERM");
      killTimer = setTimeout(() => {
        stopGroup("SIGKILL");
        // Bound completion even if an escaped descendant retains a pipe.
        child.stdout.destroy();
        child.stderr.destroy();
        // Give Node a bounded turn to report the final exit/signal evidence.
        killTimer = setTimeout(finish, 1_000);
      }, 5_000);
    };
    const collect = (chunk) => {
      const remaining = maxOutputBytes - retainedBytes;
      if (remaining > 0) {
        const retained = chunk.subarray(0, remaining);
        chunks.push(Buffer.from(retained));
        retainedBytes += retained.length;
      }
      if (chunk.length > remaining) stop(`output exceeded ${maxOutputBytes} bytes`);
    };
    child.stdout.on("data", collect);
    child.stderr.on("data", collect);
    child.once("error", (error) => { spawnError = error; });
    child.once("exit", (code, signal) => { exit = { code, signal }; });
    child.once("close", finish);
    deadline = setTimeout(() => stop(`timeout after ${timeoutMs}ms`), timeoutMs);
  });
  const bytes = Buffer.concat(chunks, retainedBytes);
  const output = bytes.toString("utf8");
  // Keep the file byte bounded even when the cap splits a UTF-8 character.
  await writeFile(path.join(outputDir, "output.txt"), bytes);
  if (failure || spawnError || outcome.code !== 0) {
    const detail = outcome.signal ? `signal ${outcome.signal}` : `exit ${outcome.code}`;
    const reason = failure || (spawnError ? `spawn failed: ${spawnError.code || "unknown"}` : detail);
    throw new Error(`deterministic front porch failed (${reason}${reason === detail ? "" : `; ${detail}`})\n${output}`);
  }
  console.error("[piku eval] deterministic front porch passed");
  return output;
}
