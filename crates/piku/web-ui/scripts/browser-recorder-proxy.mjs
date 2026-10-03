#!/usr/bin/env node
/**
 * A transparent stdio MCP relay that owns browser-video lifecycle.
 *
 * Codex sees only the normal Playwright action vocabulary.  The proxy reserves
 * its own JSON-RPC IDs for recorder calls and never forwards those responses to
 * Codex, so recording cannot consume an evaluator action or become prompt
 * dependent.
 */
import { spawn } from "node:child_process";
import { createHash } from "node:crypto";
import { createWriteStream } from "node:fs";
import { mkdir, readFile, realpath, rename, rm, stat, writeFile } from "node:fs/promises";
import { createInterface } from "node:readline";
import { fileURLToPath } from "node:url";
import path from "node:path";

export const RECORDER_TOOLS = new Set([
  "browser_start_video",
  "browser_stop_video",
  "browser_video_chapter",
  "browser_video_show_actions",
  "browser_video_hide_actions",
]);
const CAPTURED_ACTIONS = new Set([
  "browser_click", "browser_drag", "browser_fill_form", "browser_navigate",
  "browser_navigate_back", "browser_press_key", "browser_resize", "browser_select_option",
  "browser_type", "browser_wait_for",
]);
const RECORDER_PREFIX = "piku-recorder:";

export function parseProxyArguments(argv) {
  const separator = argv.indexOf("--");
  if (separator < 0 || separator === argv.length - 1)
    throw new Error("browser recorder proxy requires an underlying command after --");
  const recordingIndex = argv.indexOf("--recording-dir");
  if (recordingIndex < 0 || !argv[recordingIndex + 1])
    throw new Error("browser recorder proxy requires --recording-dir");
  return {
    recordingDir: path.resolve(argv[recordingIndex + 1]),
    command: argv[separator + 1],
    commandArgs: argv.slice(separator + 2),
  };
}

export function recorderToolList(tools) {
  return tools.filter((tool) => !RECORDER_TOOLS.has(tool?.name));
}

export function isBrowserClose(message) {
  return message?.method === "tools/call" && message.params?.name === "browser_close";
}

export function isToolListRequest(message) {
  return message?.method === "tools/list" && message.id !== undefined;
}

export function isCapturedAction(name) {
  return CAPTURED_ACTIONS.has(name);
}

export async function confinedRecordingPath(root, filename = "journey.webm") {
  const resolvedRoot = path.resolve(root);
  const candidate = path.resolve(resolvedRoot, filename);
  const relative = path.relative(resolvedRoot, candidate);
  if (!relative || relative.startsWith("..") || path.isAbsolute(relative))
    throw new Error("recording path escaped its role review directory");
  return candidate;
}

export function preferredVideoCodecs(platform = process.platform) {
  return platform === "darwin" ? ["h264_videotoolbox", "libx264"] : ["libx264"];
}

function renderWithCodec(manifestPath, outputPath, codec, { spawnImpl, timeoutMs }) {
  return new Promise((resolve, reject) => {
    const child = spawnImpl("ffmpeg", [
      "-n", "-f", "concat", "-safe", "0", "-i", manifestPath,
      "-c:v", codec, "-pix_fmt", "yuv420p",
      "-movflags", "+faststart", outputPath,
    ], { stdio: ["ignore", "ignore", "pipe"] });
    let stderr = "";
    const timeout = setTimeout(() => child.kill("SIGTERM"), timeoutMs);
    child.stderr?.on("data", (chunk) => { stderr += chunk.toString(); });
    child.once("error", (error) => {
      clearTimeout(timeout);
      reject(error);
    });
    child.once("exit", (code, signal) => {
      clearTimeout(timeout);
      if (code === 0) resolve(outputPath);
      else reject(new Error(`${codec} failed code=${code ?? "none"} signal=${signal ?? "none"}: ${stderr.slice(-240)}`));
    });
  });
}

export async function renderFrameVideo(framePaths, outputPath, {
  spawnImpl = spawn,
  timeoutMs = 60_000,
  codecs = preferredVideoCodecs(),
} = {}) {
  if (!Array.isArray(framePaths) || framePaths.length === 0)
    throw new Error("cannot render a review video without captured frames");
  const manifestPath = `${outputPath}.ffconcat`;
  const quote = (filePath) => filePath.replaceAll("'", "'\\\\''");
  const lines = framePaths.flatMap((framePath) => [`file '${quote(framePath)}'`, "duration 0.75"]);
  lines.push(`file '${quote(framePaths.at(-1))}'`);
  await writeFile(manifestPath, `${lines.join("\n")}\n`, { encoding: "utf8", flag: "wx", mode: 0o600 });
  const errors = [];
  for (const codec of codecs) {
    const candidate = codec === codecs.at(-1)
      ? outputPath
      : outputPath.replace(/\.mp4$/, `.${codec}.mp4`);
    try {
      await renderWithCodec(manifestPath, candidate, codec, { spawnImpl, timeoutMs });
      if (candidate !== outputPath) await rename(candidate, outputPath);
      return outputPath;
    } catch (error) {
      errors.push(error.message);
      await rm(candidate, { force: true });
    }
  }
  throw new Error(`ffmpeg export failed: ${errors.join("; ")}`);
}

async function hashRegularFile(root, filePath) {
  const [resolvedRoot, resolvedFile] = await Promise.all([realpath(root), realpath(filePath)]);
  const relative = path.relative(resolvedRoot, resolvedFile);
  if (!relative || relative.startsWith("..") || path.isAbsolute(relative))
    throw new Error("recording file escaped its role review directory");
  const details = await stat(resolvedFile);
  if (!details.isFile() || details.size === 0)
    throw new Error("recording file is absent or empty");
  const bytes = await readFile(resolvedFile);
  return { path: relative, size_bytes: details.size, sha256: createHash("sha256").update(bytes).digest("hex") };
}

function writeJsonLine(stream, value) {
  stream.write(`${JSON.stringify(value)}\n`);
}

/** A minimal recorder state machine; protocol transport is deliberately thin. */
export class BrowserRecorder {
  constructor({ recordingDir, trace, sendInternal, renderFrames = renderFrameVideo, timeoutMs = 5_000, now = () => performance.now() }) {
    this.recordingDir = recordingDir;
    this.trace = trace;
    this.sendInternal = sendInternal;
    this.renderFrames = renderFrames;
    this.timeoutMs = timeoutMs;
    this.now = now;
    this.startedAt = now();
    this.state = "new";
    this.framePaths = [];
    this.frameNumber = 0;
    this.sequence = 0;
  }

  event(kind, fields = {}) {
    writeJsonLine(this.trace, { schema_version: 1, sequence: ++this.sequence, t_ms: Math.round(this.now() - this.startedAt), kind, ...fields });
  }

  async start() {
    if (this.state !== "new") return;
    this.state = "recording";
    this.event("recording_started", { video: "journey.mp4", strategy: "post_action_frames" });
  }

  async capture(tool) {
    if (this.state !== "recording") return;
    const filename = `frame-${String(++this.frameNumber).padStart(3, "0")}.png`;
    const framePath = await confinedRecordingPath(this.recordingDir, filename);
    try {
      await this.sendInternal("browser_take_screenshot", { filename: framePath, type: "png" }, this.timeoutMs);
      await hashRegularFile(this.recordingDir, framePath);
      this.framePaths.push(framePath);
      this.event("frame_captured", { frame: filename, tool });
    } catch (error) {
      this.event("frame_unavailable", { frame: filename, tool, error: error.message });
    }
  }

  async finish(reason) {
    if (this.state === "finalized") return;
    if (this.state !== "recording") {
      this.event("recording_incomplete", { reason, state: this.state });
      this.state = "finalized";
      return;
    }
    this.event("recording_stop_requested", { reason, captured_frames: this.framePaths.length });
    try {
      const mp4Path = await confinedRecordingPath(this.recordingDir, "journey.mp4");
      await this.renderFrames(this.framePaths, mp4Path);
      this.event("recording_finalized", { reason, media: [await hashRegularFile(this.recordingDir, mp4Path)] });
    } catch (error) {
      this.event("recording_incomplete", { reason, state: "render_failed", error: error.message });
    }
    this.state = "finalized";
  }
}

export async function runProxy({ argv = process.argv.slice(2), stdin = process.stdin, stdout = process.stdout, stderr = process.stderr } = {}) {
  const { recordingDir, command, commandArgs } = parseProxyArguments(argv);
  await mkdir(recordingDir, { recursive: true });
  const trace = createWriteStream(path.join(recordingDir, "recorder.jsonl"), { flags: "wx", mode: 0o600 });
  const child = spawn(command, commandArgs, { stdio: ["pipe", "pipe", "inherit"] });
  const internal = new Map();
  let counter = 0;
  let closed = false;
  const toolListIds = new Set();
  const recorder = new BrowserRecorder({
    recordingDir,
    trace,
    sendInternal: (name, arguments_, timeoutMs) => new Promise((resolve, reject) => {
      const id = `${RECORDER_PREFIX}${++counter}`;
      const timer = setTimeout(() => {
        internal.delete(id);
        reject(new Error(`${name} timed out after ${timeoutMs}ms`));
      }, timeoutMs);
      internal.set(id, { resolve, reject, timer });
      writeJsonLine(child.stdin, { jsonrpc: "2.0", id, method: "tools/call", params: { name, arguments: arguments_ } });
    }),
  });
  recorder.event("header", { proxy: "browser-recorder-proxy", video: "journey.mp4", input_capture: "tool_names_only" });

  let captureTail = Promise.resolve();
  const finalize = async (reason) => {
    if (closed) return;
    closed = true;
    await captureTail;
    await recorder.finish(reason);
    trace.end();
  };
  const serverReader = createInterface({ input: child.stdout });
  serverReader.on("line", (line) => {
    let message;
    try { message = JSON.parse(line); }
    catch { stdout.write(`${line}\n`); return; }
    if (typeof message.id === "string" && internal.has(message.id)) {
      const pending = internal.get(message.id);
      internal.delete(message.id);
      clearTimeout(pending.timer);
      if (message.error) pending.reject(new Error(message.error.message || "recorder tool failed"));
      else pending.resolve(message.result);
      return;
    }
    if (toolListIds.delete(message.id) && Array.isArray(message.result?.tools))
      message.result.tools = recorderToolList(message.result.tools);
    const tool = normalCalls.get(message.id);
    normalCalls.delete(message.id);
    if (tool && isCapturedAction(tool) && !message.error) {
      captureTail = captureTail.then(() => recorder.capture(tool));
      void captureTail.finally(() => stdout.write(`${JSON.stringify(message)}\n`));
      return;
    }
    if (tool === "browser_close") {
      void finalize("browser_close").finally(() => stdout.write(`${JSON.stringify(message)}\n`));
      return;
    }
    stdout.write(`${JSON.stringify(message)}\n`);
  });

  let serial = Promise.resolve();
  const normalCalls = new Map();
  const clientReader = createInterface({ input: stdin });
  clientReader.on("line", (line) => {
    serial = serial.then(async () => {
      let message;
      try { message = JSON.parse(line); }
      catch { child.stdin.write(`${line}\n`); return; }
      if (isToolListRequest(message)) toolListIds.add(message.id);
      if (message.method === "tools/call") {
        await recorder.start();
        recorder.event("tool_started", { tool: message.params?.name || "unknown" });
        if (message.id !== undefined) normalCalls.set(message.id, message.params?.name || "unknown");
      }
      child.stdin.write(`${line}\n`);
    }).catch(async (error) => {
      recorder.event("recording_incomplete", { reason: "proxy_error", error: error.message });
      await finalize("proxy_error");
    });
  });
  clientReader.on("close", () => { void serial.then(() => finalize("client_eof")).then(() => child.kill("SIGTERM")); });
  child.once("exit", async (code, signal) => {
    await finalize(signal ? `server_${signal}` : "server_exit");
    stderr.write(`[piku recorder] underlying MCP exited code=${code ?? "none"} signal=${signal ?? "none"}\n`);
    process.exitCode = code ?? 1;
  });
  child.once("error", async (error) => {
    recorder.event("recording_incomplete", { reason: "server_spawn_error", error: error.message });
    await finalize("server_spawn_error");
    stderr.write(`[piku recorder] could not start underlying MCP: ${error.message}\n`);
    process.exitCode = 1;
  });
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url))
  await runProxy();
