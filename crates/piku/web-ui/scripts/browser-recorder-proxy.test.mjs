import assert from "node:assert/strict";
import { PassThrough } from "node:stream";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { test } from "node:test";
import {
  BrowserRecorder,
  confinedRecordingPath,
  isCapturedAction,
  isBrowserClose,
  preferredVideoCodecs,
  parseProxyArguments,
  recorderToolList,
} from "./browser-recorder-proxy.mjs";

test("proxy requires a recorder root and an underlying command", () => {
  assert.throws(() => parseProxyArguments([]), /underlying command/);
  assert.throws(() => parseProxyArguments(["--", "node"]), /recording-dir/);
  assert.deepEqual(
    parseProxyArguments(["--recording-dir", "/tmp/review", "--", "npx", "playwright-mcp"]),
    { recordingDir: "/tmp/review", command: "npx", commandArgs: ["playwright-mcp"] },
  );
});

test("recorder-only tools never reach the evaluator tool list", () => {
  assert.deepEqual(
    recorderToolList([{ name: "browser_click" }, { name: "browser_start_video" }]),
    [{ name: "browser_click" }],
  );
  assert.equal(isBrowserClose({ method: "tools/call", params: { name: "browser_close" } }), true);
  assert.equal(isBrowserClose({ method: "tools/call", params: { name: "browser_click" } }), false);
  assert.equal(isCapturedAction("browser_click"), true);
  assert.equal(isCapturedAction("browser_snapshot"), false);
  assert.deepEqual(preferredVideoCodecs("darwin"), ["h264_videotoolbox", "libx264"]);
  assert.deepEqual(preferredVideoCodecs("linux"), ["libx264"]);
});

test("recording paths cannot escape the role review directory", async () => {
  assert.equal(await confinedRecordingPath("/tmp/piku-review"), "/tmp/piku-review/journey.webm");
  await assert.rejects(confinedRecordingPath("/tmp/piku-review", "../escape.webm"), /escaped/);
});

test("recorder renders a verified local video from harness-captured action frames", async (t) => {
  const root = await mkdtemp(path.join(tmpdir(), "piku-recorder-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  const trace = new PassThrough();
  const chunks = [];
  trace.on("data", (chunk) => chunks.push(chunk));
  const calls = [];
  const recorder = new BrowserRecorder({
    recordingDir: root,
    trace,
    sendInternal: async (name, arguments_) => {
      calls.push(name);
      if (name === "browser_take_screenshot") await writeFile(arguments_.filename, "frame bytes");
      return {};
    },
    renderFrames: async (frames, output) => {
      assert.equal(frames.length, 1);
      await writeFile(output, "mp4 bytes");
      return output;
    },
    now: (() => { let time = 0; return () => ++time; })(),
  });
  await recorder.start();
  await recorder.capture("browser_click");
  await recorder.finish("judge_exit");
  trace.end();
  await new Promise((resolve) => trace.once("end", resolve));
  const contents = Buffer.concat(chunks).toString("utf8");
  assert.deepEqual(calls, ["browser_take_screenshot"]);
  assert.match(contents, /recording_started/);
  assert.match(contents, /recording_finalized/);
  assert.match(contents, /journey\.mp4/);
  assert.match(contents, /frame_captured/);
  assert.doesNotMatch(contents, /arguments/);
  await writeFile(path.join(root, "recorder.jsonl"), contents);
  assert.equal((await readFile(path.join(root, "recorder.jsonl"), "utf8")).includes("recording_finalized"), true);
});
