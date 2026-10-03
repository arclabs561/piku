import assert from "node:assert/strict";
import test from "node:test";
import {
  assertJudgeEligibleRecording,
  assertJudgeEligibleTimeline,
  assertSurfaceJudgeEvidence,
  surfaceJudgeAdapter,
} from "./evaluation-surfaces.mjs";
import { surfaceScenarioPacket } from "./evaluation-scenarios.mjs";

test("CLI, TUI, and web judges share one ledger contract with surface-specific evidence", () => {
  const recording = {
    status: "complete",
    inspectability: "inspectable",
    source_artifacts: [{ ref: "run:output-recovery", retention: "durable" }],
  };
  for (const [surface, evidence] of Object.entries({
    cli: ["argv", "exit_status", "stdout_or_artifact", "run_record"],
    tui: ["pty_transcript", "key_events", "resize_events", "run_record"],
    web: ["browser_events", "screenshots", "dom_predicates", "run_record"],
  })) {
    assert.equal(surfaceJudgeAdapter(surface).observation.length > 0, true);
    assert.equal(assertSurfaceJudgeEvidence(surface, evidence, recording).observation.length > 0, true);
    assert.equal(surfaceScenarioPacket({ surface, evidence, recording, claims: ["output_visible", "reference_addressable", "interruption_preserved", "recovery_without_regeneration"] }).scenario, "output-recovery-room");
  }
  assert.throws(() => assertSurfaceJudgeEvidence("tui", ["pty_transcript"], recording), /missing evidence/);
  assert.throws(
    () => assertSurfaceJudgeEvidence("tui", ["pty_transcript", "key_events", "resize_events", "run_record"]),
    /recording metadata is missing/,
  );
});

test("incomplete, uninspectable, and expired recordings cannot support judge claims", () => {
  const complete = {
    status: "complete",
    inspectability: "inspectable",
    source_artifacts: [{ ref: "cast:output-recovery", retention: "local_private" }],
  };
  assert.equal(assertJudgeEligibleRecording(complete).status, "complete");
  assert.equal(assertSurfaceJudgeEvidence(
    "tui",
    ["pty_transcript", "key_events", "resize_events", "run_record"],
    complete,
  ).observation, "managed_pty");

  assert.throws(
    () => assertJudgeEligibleRecording({ ...complete, status: "incomplete", reason: "cast flush failed" }),
    /recording is incomplete/,
  );
  assert.throws(
    () => assertJudgeEligibleRecording({ ...complete, inspectability: "uninspectable", reason: "artifact unavailable" }),
    /evidence is uninspectable/,
  );
  assert.throws(
    () => assertJudgeEligibleRecording({ ...complete, source_artifacts: [{ ref: "cast:expired", retention: "ephemeral" }] }),
    /source artifacts are not retained/,
  );
  assert.throws(
    () => assertJudgeEligibleRecording({ ...complete, source_artifacts: [{ ref: "cast:unknown", retention: "unknown" }] }),
    /source artifacts are not retained/,
  );

  const timeline = { replay: { artifacts: ["cast:output-recovery"], recording: complete } };
  assert.equal(assertJudgeEligibleTimeline(timeline).inspectability, "inspectable");
  assert.throws(
    () => assertJudgeEligibleTimeline({
      replay: { artifacts: ["cast:other"], recording: complete },
    }),
    /source artifact is not listed in replay/,
  );
});
