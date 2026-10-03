import assert from "node:assert/strict";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { test } from "node:test";
import path from "node:path";
import { tmpdir } from "node:os";
import { automaticReviewFindings, classifyReviewAssets, recorderCoverage, writeAssetReview } from "./evaluation-asset-review.mjs";

test("asset review separates replay media, terminal text, and traces", () => {
  const assets = classifyReviewAssets(["journey.mp4", "frame-001.png", "session.cast", "stdout.txt", "trace.jsonl"]);
  assert.deepEqual(assets.videos, ["journey.mp4"]);
  assert.deepEqual(assets.images, ["frame-001.png"]);
  assert.deepEqual(assets.casts, ["session.cast"]);
  assert.deepEqual(assets.terminal_text, ["stdout.txt"]);
  assert.deepEqual(assets.traces, ["trace.jsonl"]);
});

test("recorder coverage treats a missing frame as a review gap", () => {
  const recorder = recorderCoverage([
    { kind: "frame_captured", frame: "frame-001.png", tool: "browser_click", t_ms: 20 },
    { kind: "frame_unavailable", frame: "frame-002.png", tool: "browser_click", t_ms: 30, error: "gone" },
  ]);
  const findings = automaticReviewFindings({ assets: classifyReviewAssets(["journey.mp4", "frame-001.png"]), recorder: [{ path: "recorder.jsonl", ...recorder }] });
  assert.equal(findings[0].kind, "missing_post_action_frame");
  assert.equal(findings[0].tool, "browser_click");
});

test("a malformed recorder becomes a review gap instead of masking the evaluation", async (t) => {
  const root = await mkdtemp(path.join(tmpdir(), "piku-asset-review-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  await writeFile(path.join(root, "recorder.jsonl"), "not-json\n");
  const review = await writeAssetReview(root, { surface: "web" });
  assert.equal(review.recorder[0].parse_error !== null, true);
  assert.equal(review.automatic_findings.some((finding) => finding.kind === "invalid_recorder_trace"), true);
});

test("a timed-out run is incomplete even when it retained viewable assets", async (t) => {
  const root = await mkdtemp(path.join(tmpdir(), "piku-asset-review-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  await writeFile(path.join(root, "manifest.json"), JSON.stringify({ timed_out: true, exit_signal: "SIGTERM" }));
  await writeFile(path.join(root, "video.webm"), "fixture");
  const review = await writeAssetReview(root, { surface: "web" });
  assert.equal(review.review_status, "incomplete_run_requires_retest");
  assert.equal(review.automatic_findings.some((finding) => finding.kind === "run_timed_out"), true);
});

test("a completed nonzero evaluator is reviewable but not a passing run", async (t) => {
  const root = await mkdtemp(path.join(tmpdir(), "piku-asset-review-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  await writeFile(path.join(root, "manifest.json"), JSON.stringify({ timed_out: false, exit_code: 1 }));
  await writeFile(path.join(root, "frame.png"), "fixture");
  const review = await writeAssetReview(root, { surface: "web" });
  assert.equal(review.review_status, "completed_with_failures");
  assert.equal(review.automatic_findings.some((finding) => finding.kind === "run_failed"), true);
});

test("asset review writes a bounded inspectable packet without overwriting it", async (t) => {
  const root = await mkdtemp(path.join(tmpdir(), "piku-asset-review-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  await mkdir(path.join(root, "playwright-output"));
  await writeFile(path.join(root, "playwright-output", "frame-001.png"), "fixture");
  await writeFile(path.join(root, "playwright-output", "recorder.jsonl"), `${JSON.stringify({ kind: "frame_captured", frame: "frame-001.png", tool: "browser_click", t_ms: 1 })}\n`);
  const review = await writeAssetReview(root, { surface: "web", runId: "run-1" });
  assert.equal(review.review_status, "ready_for_human_or_multimodal_review");
  assert.equal(review.recorder[0].captured[0].frame, "frame-001.png");
  assert.deepEqual(JSON.parse(await readFile(path.join(root, "asset-review.json"), "utf8")).assets.images, ["playwright-output/frame-001.png"]);
  await assert.rejects(writeAssetReview(root, { surface: "web", runId: "run-1" }), { code: "EEXIST" });
});
