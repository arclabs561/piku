import { fileURLToPath } from "node:url";
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
  isAdmittedNavigation,
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

test("JSON-RPC relay filters discovery and rejects denied calls without child forwarding", async (t) => {
  const { spawn } = await import("node:child_process");
  const { createInterface } = await import("node:readline");
  const root = await mkdtemp(path.join(tmpdir(), "piku-proxy-authority-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  const forwarded = path.join(root, "forwarded.txt");
  const launchFile = path.join(root, "launch.html");
  const capability = "c".repeat(64);
  await writeFile(launchFile, `<a href="http://localhost:43210/#token=${capability}">open</a>`, { mode: 0o600 });
  const server = `const fs=require('node:fs');const rl=require('node:readline').createInterface({input:process.stdin});rl.on('line',line=>{const m=JSON.parse(line);if(m.method==='tools/list')console.log(JSON.stringify({jsonrpc:'2.0',id:m.id,result:{tools:['browser_snapshot','browser_evaluate','browser_network_request','browser_network_requests','browser_run_code','browser_run_code_unsafe','browser_start_video'].map(name=>({name}))}}));else if(m.method==='tools/call'){fs.appendFileSync(${JSON.stringify(forwarded)},m.params.name+'\\n');console.log(JSON.stringify({jsonrpc:'2.0',id:m.id,result:{content:[{type:"text",text:"cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"}]}}));console.error("cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc")}});`;
  const proxy = spawn(process.execPath, [fileURLToPath(new URL("./browser-recorder-proxy.mjs", import.meta.url)),
    "--recording-dir", root, "--", process.execPath, "-e", server], { stdio: ["pipe", "pipe", "pipe"], env: { ...process.env, PIKU_WEB_LAUNCH_FILE: launchFile, PIKU_WEB_URL: "http://localhost:43210" } });
  t.after(() => { if (proxy.exitCode === null) proxy.kill("SIGTERM"); });
  let diagnostics = "";
  proxy.stderr.on("data", (chunk) => { diagnostics += chunk.toString(); });
  const replies = new Map();
  const reader = createInterface({ input: proxy.stdout });
  let resolveReplies;
  const complete = new Promise((resolve) => { resolveReplies = resolve; });
  reader.on("line", (line) => { const message = JSON.parse(line); replies.set(message.id, message); if (replies.size === 10) resolveReplies(); });
  const denied = ["browser_evaluate", "browser_network_request", "browser_network_requests", "browser_run_code", "browser_run_code_unsafe"];
  proxy.stdin.write(JSON.stringify({ jsonrpc: "2.0", id: 0, method: "tools/list" }) + "\n");
  denied.forEach((name, index) => proxy.stdin.write(JSON.stringify({ jsonrpc: "2.0", id: index + 1,
    method: "tools/call", params: { name, arguments: { secret: "must-not-forward" } } }) + "\n"));
  proxy.stdin.write(JSON.stringify({ jsonrpc: "2.0", id: 7, method: "tools/call", params: { name: "browser_navigate", arguments: { url: "javascript:console.log(sessionStorage)" } } }) + "\n");
  proxy.stdin.write(JSON.stringify({ jsonrpc: "2.0", id: 8, method: "tools/call", params: { name: "browser_tabs", arguments: { action: "new", url: "javascript:console.log(sessionStorage)" } } }) + "\n");
  proxy.stdin.write(JSON.stringify({ jsonrpc: "2.0", id: 9, method: "tools/call", params: { name: "browser_navigate", arguments: { url: "http://localhost:43210/?surface=qa" } } }) + "\n");
  proxy.stdin.write(JSON.stringify({ jsonrpc: "2.0", id: 6, method: "tools/call", params: { name: "browser_snapshot", arguments: {} } }) + "\n");
  await complete;
  assert.deepEqual(replies.get(0).result.tools, [{ name: "browser_snapshot" }]);
  for (let index = 1; index <= denied.length; index += 1) assert.equal(replies.get(index).error.code, -32601);
  assert.equal(replies.get(7).error.code, -32601);
  assert.equal(replies.get(8).error.code, -32601);
  assert.equal(replies.get(9).error, undefined);
  const forwardedTools = (await readFile(forwarded, "utf8")).trim().split("\n");
  assert.equal(forwardedTools.filter((name) => name === "browser_navigate").length, 1);
  assert.equal(forwardedTools.includes("browser_snapshot"), true);
  for (const forbidden of [...denied, "browser_tabs"]) assert.equal(forwardedTools.includes(forbidden), false);
  assert.equal(JSON.stringify([...replies.values()]).includes(capability), false);
  assert.match(replies.get(6).result.content[0].text, /redacted launch credential/);
  proxy.stdin.end();
  await new Promise((resolve) => proxy.once("exit", resolve));
  assert.equal(diagnostics.includes(capability), false);
  assert.match(diagnostics, /redacted launch credential/);
});


test("hosted navigation admits only the declared loopback HTTP origin", () => {
  const message = (url) => ({ method: "tools/call", params: { name: "browser_navigate", arguments: { url } } });
  assert.equal(isAdmittedNavigation(message("http://localhost:43210/?surface=qa"), "http://localhost:43210"), true);
  assert.equal(isAdmittedNavigation({ params: { name: "browser_tabs", arguments: { action: "new", url: "javascript:console.log(sessionStorage)" } } }, "http://localhost:43210"), false);
  for (const url of ["javascript:console.log(sessionStorage)", "data:text/html,<script></script>", "file:///tmp/file", "http://localhost:43211", "http://127.0.0.1:43210", "http://user@localhost:43210/"])
    assert.equal(isAdmittedNavigation(message(url), "http://localhost:43210"), false);
});
