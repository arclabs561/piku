import { readdir, readFile, stat, writeFile } from "node:fs/promises";
import path from "node:path";

const reviewFile = "asset-review.json";

async function filesBelow(root, directory = root) {
  const entries = await readdir(directory, { withFileTypes: true });
  const nested = await Promise.all(entries.map(async (entry) => {
    const candidate = path.join(directory, entry.name);
    if (entry.isDirectory()) return filesBelow(root, candidate);
    if (entry.isFile()) return [path.relative(root, candidate)];
    return [];
  }));
  return nested.flat().sort();
}

function artifactKind(file) {
  const filename = path.basename(file).toLowerCase();
  const extension = path.extname(file).toLowerCase();
  if (extension === ".mp4" || extension === ".webm") return "video";
  if (extension === ".png" || extension === ".jpg" || extension === ".jpeg") return "image";
  if (extension === ".cast") return "terminal_cast";
  if (filename === "stdout.txt" || filename === "output.txt" || extension === ".out") return "terminal_text";
  if (extension === ".json" || extension === ".jsonl" || extension === ".txt" || extension === ".yml") return "trace";
  return "other";
}

export function classifyReviewAssets(files) {
  const assets = files.map((file) => ({ path: file, kind: artifactKind(file) }));
  return {
    assets,
    videos: assets.filter((asset) => asset.kind === "video").map((asset) => asset.path),
    images: assets.filter((asset) => asset.kind === "image").map((asset) => asset.path),
    casts: assets.filter((asset) => asset.kind === "terminal_cast").map((asset) => asset.path),
    terminal_text: assets.filter((asset) => asset.kind === "terminal_text").map((asset) => asset.path),
    traces: assets.filter((asset) => asset.kind === "trace").map((asset) => asset.path),
  };
}

export function recorderCoverage(events) {
  const captured = [];
  const unavailable = [];
  for (const event of events) {
    if (event?.kind === "frame_captured" && typeof event.frame === "string")
      captured.push({ frame: event.frame, tool: event.tool ?? null, t_ms: event.t_ms ?? null });
    if (event?.kind === "frame_unavailable")
      unavailable.push({ frame: event.frame ?? null, tool: event.tool ?? null, t_ms: event.t_ms ?? null, reason: event.error ?? null });
  }
  return { captured, unavailable };
}

async function recorderSummaries(root, files) {
  const summaries = [];
  for (const file of files.filter((candidate) => path.basename(candidate) === "recorder.jsonl")) {
    const contents = await readFile(path.join(root, file), "utf8");
    try {
      const events = contents.split("\n").filter(Boolean).map((line) => JSON.parse(line));
      summaries.push({ path: file, ...recorderCoverage(events), parse_error: null });
    } catch (error) {
      summaries.push({ path: file, captured: [], unavailable: [], parse_error: error.message });
    }
  }
  return summaries;
}

async function terminalRunState(root, files) {
  if (!files.includes("manifest.json")) return null;
  try {
    const manifest = JSON.parse(await readFile(path.join(root, "manifest.json"), "utf8"));
    return {
      timed_out: manifest.timed_out === true,
      exit_code: manifest.exit_code ?? null,
      exit_signal: manifest.exit_signal ?? null,
    };
  } catch {
    return { timed_out: false, exit_code: null, exit_signal: null, manifest_invalid: true };
  }
}

export function automaticReviewFindings({ assets, recorder, terminal = null }) {
  const findings = [];
  if (terminal?.timed_out) {
    findings.push({
      severity: "review_gap",
      kind: "run_timed_out",
      rationale: "The evaluator did not reach a normal terminal outcome; retained artifacts are partial and require a retest.",
    });
  }
  if (terminal && !terminal.timed_out
    && (terminal.exit_signal || (Number.isInteger(terminal.exit_code) && terminal.exit_code !== 0))) {
    findings.push({
      severity: "evaluation_failure",
      kind: "run_failed",
      rationale: "The evaluator reached a terminal failure outcome; retained artifacts diagnose the failure but do not establish a passing run.",
    });
  }
  if (terminal?.manifest_invalid) {
    findings.push({
      severity: "review_gap",
      kind: "invalid_run_manifest",
      rationale: "The terminal manifest cannot be read, so completion state is unknown.",
    });
  }
  for (const summary of recorder) {
    if (summary.parse_error) {
      findings.push({
        severity: "review_gap",
        kind: "invalid_recorder_trace",
        artifact: summary.path,
        rationale: "The action-to-frame trace is malformed, so temporal claims cannot be checked against it.",
      });
    }
    for (const gap of summary.unavailable) {
      findings.push({
        severity: "review_gap",
        kind: "missing_post_action_frame",
        artifact: summary.path,
        frame: gap.frame,
        tool: gap.tool,
        t_ms: gap.t_ms,
        rationale: "An action lacks its post-action frame; do not infer its visible result from adjacent frames.",
      });
    }
  }
  if (assets.videos.length && !assets.images.length) {
    findings.push({
      severity: "review_gap",
      kind: "video_requires_frame_extraction",
      rationale: "A video exists without retained still frames; extract and inspect decisive transitions before visual critique.",
    });
  }
  if (!assets.videos.length && !assets.images.length && !assets.casts.length && !assets.terminal_text.length) {
    findings.push({
      severity: "review_gap",
      kind: "no_replayable_visual_asset",
      rationale: "No video, image, or terminal cast was retained for an operator-visible review.",
    });
  }
  return findings;
}

export async function writeAssetReview(bundleDir, { surface, runId = null } = {}) {
  if (!new Set(["cli", "tui", "web"]).has(surface)) throw new TypeError("asset review needs cli, tui, or web surface");
  const files = (await filesBelow(bundleDir)).filter((file) => file !== reviewFile);
  const assets = classifyReviewAssets(files);
  const recorder = await recorderSummaries(bundleDir, files);
  const terminal = await terminalRunState(bundleDir, files);
  const automaticFindings = automaticReviewFindings({ assets, recorder, terminal });
  const review = {
    schema_version: 1,
    surface,
    run_id: runId,
    review_status: terminal?.timed_out
      ? "incomplete_run_requires_retest"
      : terminal && (terminal.exit_signal || (Number.isInteger(terminal.exit_code) && terminal.exit_code !== 0))
        ? "completed_with_failures"
        : "ready_for_human_or_multimodal_review",
    review_contract: {
      operator_visible_claims: "cite retained image frames, rendered replay frames, terminal casts, or terminal text",
      temporal_claims: "cite timestamped trace events and before-after frames",
      behavioral_claims: "cite the native trace or predicate evidence",
      missing_coverage: "is a review gap, never product evidence",
    },
    assets,
    recorder,
    terminal,
    automatic_findings: automaticFindings,
  };
  await writeFile(path.join(bundleDir, reviewFile), `${JSON.stringify(review, null, 2)}\n`, {
    encoding: "utf8", flag: "wx", mode: 0o600,
  });
  return review;
}

async function main(argv) {
  const [surface, bundleDir, runId = null] = argv;
  if (!surface || !bundleDir || argv.length > 3)
    throw new Error("usage: evaluation-asset-review.mjs <cli|tui|web> <bundle-dir> [run-id]");
  await stat(bundleDir);
  const review = await writeAssetReview(bundleDir, { surface, runId });
  console.error(`[piku eval] review packet: ${path.join(bundleDir, reviewFile)} (${review.automatic_findings.length} automatic findings)`);
}

if (process.argv[1] && path.resolve(process.argv[1]) === import.meta.filename)
  await main(process.argv.slice(2));
