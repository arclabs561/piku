import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { evaluationRecord } from "./evaluation-ledger.mjs";

const scriptsDir = path.dirname(fileURLToPath(import.meta.url));

function clone(value) {
  return structuredClone(value);
}

function setAtPath(value, dottedPath, replacement) {
  const parts = dottedPath.split(".");
  const key = parts.pop();
  let target = value;
  for (const part of parts) target = target[part];
  target[key] = replacement;
}

// This deliberately small oracle models observable evidence, not UI internals.
// Each assertion below comes from a named thesis requirement in codex-live-qa.md.
function detectEvidenceFaults(packet) {
  const faults = [];
  const progress = new Set(packet.progress_states);
  if (!progress.has("queued") || !progress.has("running")) {
    faults.push({ dimension: "state_visibility", result: "absent" });
  }

  if (Object.values(packet.provenance).some((visible) => !visible)) {
    faults.push({ dimension: "action_provenance", result: "absent" });
  }

  if (
    packet.rerun.edited_input_id !== packet.rerun.result_input_id ||
    packet.rerun.prior_result_state !== "stale"
  ) {
    faults.push({ dimension: "rerun_semantics", result: "absent" });
  }

  if (Object.values(packet.authority).some((authority) => !authority)) {
    faults.push({ dimension: "authority_clarity", result: "absent" });
  }

  if (Object.values(packet.persistence).some((persisted) => !persisted)) {
    faults.push({ dimension: "recovery", result: "absent" });
  }

  return faults;
}

async function faultCases() {
  return JSON.parse(
    await readFile(
      path.join(scriptsDir, "fixtures", "evaluation-fault-cases.json"),
      "utf8",
    ),
  );
}

test("healthy evidence packet does not trigger a fault", async () => {
  const { baseline } = await faultCases();
  assert.deepEqual(detectEvidenceFaults(baseline), []);
});

test("each injected product fault is detected by the intended rubric dimension", async (t) => {
  const { baseline, cases } = await faultCases();

  for (const faultCase of cases) {
    await t.test(faultCase.id, () => {
      const packet = clone(baseline);
      setAtPath(packet, faultCase.mutation.path, faultCase.mutation.value);

      assert.deepEqual(detectEvidenceFaults(packet), [faultCase.expected]);
    });
  }
});

test("evaluator timeout is a harness result and cannot become a product verdict", () => {
  const record = evaluationRecord({
    runId: "fault-timeout",
    runStatus: "timeout",
    failureClass: "evaluator_timeout",
    durationMs: 45_000,
    artifactRefs: ["events.jsonl"],
  });

  assert.equal(record.run_status, "timeout");
  assert.equal(record.failure_class, "evaluator_timeout");
  assert.equal(record.product_verdict, null);
  assert.equal(record.finding_count, null);
});

test("live judge rubric explicitly asks for every injected fault signal", async () => {
  const prompt = await readFile(
    path.resolve(scriptsDir, "..", "e2e", "codex-live-qa.md"),
    "utf8",
  );

  const requiredSignals = [
    /queued\/running\/completed\/error\s+transitions/,
    /proposed actions, executed actions, files,\s+diffs, tool calls, verification/,
    /downstream output is\s+visibly invalidated/,
    /conversation, page, workspace, file, and terminal\s+powers are distinct/,
    /reload and verify content,\s+geometry, and stacking/,
  ];

  for (const signal of requiredSignals) {
    assert.match(prompt, signal);
  }
});
