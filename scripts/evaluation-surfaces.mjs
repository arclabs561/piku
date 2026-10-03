import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

const schema = JSON.parse(readFileSync(
  fileURLToPath(new URL("../eval/surface-evaluation.schema.json", import.meta.url)),
  "utf8",
));
const RETAINED_ARTIFACTS = new Set(["durable", "local_private", "remote_redacted"]);

function requirements(surface) {
  const rule = schema.$defs?.[`${surface}_requirements`];
  const evidenceRules = rule?.properties?.evidence_kinds?.allOf;
  const observation = rule?.properties?.observation?.const;
  const requiredEvidence = evidenceRules?.map((entry) => entry.contains?.const);
  if (!observation || !requiredEvidence?.every((kind) => typeof kind === "string"))
    throw new TypeError(`surface evaluation schema has no usable ${surface} requirements`);
  return Object.freeze({ observation, required_evidence: Object.freeze(requiredEvidence) });
}

/** Node adapter for the schema-owned surface requirements. */
export const SURFACE_JUDGE_ADAPTERS = Object.freeze(Object.fromEntries(
  ["cli", "tui", "web"].map((surface) => [surface, requirements(surface)]),
));

export function surfaceJudgeAdapter(surface) {
  const adapter = SURFACE_JUDGE_ADAPTERS[surface];
  if (!adapter) throw new TypeError(`unknown judge surface: ${surface}`);
  return adapter;
}

export function assertSurfaceJudgeEvidence(surface, evidenceKinds, recording) {
  const adapter = surfaceJudgeAdapter(surface);
  if (!Array.isArray(evidenceKinds)) throw new TypeError("evidenceKinds must be an array");
  const supplied = new Set(evidenceKinds);
  const missing = adapter.required_evidence.filter((kind) => !supplied.has(kind));
  if (missing.length) throw new TypeError(`${surface} judge missing evidence: ${missing.join(", ")}`);
  if (recording === undefined)
    throw new TypeError("judge recording is ineligible: recording metadata is missing");
  assertJudgeEligibleRecording(recording);
  return adapter;
}

/**
 * Reject a recording that cannot support an inspectable judge claim.
 *
 * `recording` is optional on v1 packets for backwards-compatible schema
 * admission, but a judge must supply it and fail closed on incomplete or
 * uninspectable evidence rather than treating a screen trace as a verdict.
 */
export function assertJudgeEligibleRecording(recording) {
  if (recording === null || typeof recording !== "object" || Array.isArray(recording))
    throw new TypeError("judge recording metadata must be an object");
  if (recording.status !== "complete")
    throw new TypeError("judge recording is ineligible: recording is incomplete");
  if (recording.inspectability !== "inspectable")
    throw new TypeError("judge recording is ineligible: evidence is uninspectable");
  if (!Array.isArray(recording.source_artifacts) || recording.source_artifacts.length === 0)
    throw new TypeError("judge recording is ineligible: source artifacts are missing");
  for (const artifact of recording.source_artifacts) {
    if (artifact === null || typeof artifact !== "object" || Array.isArray(artifact)
      || typeof artifact.ref !== "string" || artifact.ref.length === 0
      || !RETAINED_ARTIFACTS.has(artifact.retention)) {
      throw new TypeError("judge recording is ineligible: source artifacts are not retained");
    }
  }
  return Object.freeze({ ...recording });
}

/** Reject timeline metadata whose claimed sources are absent from its replay. */
export function assertJudgeEligibleTimeline(timeline) {
  if (timeline === null || typeof timeline !== "object" || Array.isArray(timeline)
    || timeline.replay === null || typeof timeline.replay !== "object"
    || !Array.isArray(timeline.replay.artifacts)) {
    throw new TypeError("judge timeline is ineligible: replay artifacts are missing");
  }
  const recording = assertJudgeEligibleRecording(timeline.replay.recording);
  const artifacts = new Set(timeline.replay.artifacts);
  if (recording.source_artifacts.some(({ ref }) => !artifacts.has(ref)))
    throw new TypeError("judge timeline is ineligible: source artifact is not listed in replay");
  return recording;
}
