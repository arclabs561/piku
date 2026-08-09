import assert from "node:assert/strict";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { test } from "node:test";
import path from "node:path";
import {
  buildEvidenceQualityDossier, focusPairOrder, runFocusPair, validatePairedContracts,
} from "./focus-pair-eval.mjs";

const revision = "a".repeat(40);

function result(runId, { status = "completed", model = "gpt", finding = runId } = {}) {
  const evidenceId = `coding_trace:${runId}`;
  return {
    runId, runDir: `/tmp/${runId}`, runStatus: status,
    runtime: { subject_revision: revision, subject_dirty: false, viewport: { width: 1440, height: 1000 } },
    promptManifestDocument: { roles: [{
      role: "coding_trace", model,
      prompt_assets: [{ path: "prompt.md", sha256: "b".repeat(64) }],
      tools: { argv: [`/tmp/${runId}/coding_trace/evidence.json`] },
      limits: { target_calls: 48, timeout_ms: 10 },
    }] },
    results: [{ report: { evidence: [{ id: evidenceId, kind: "predicate" }] } }],
    synthesis: { report: {
      verdict: "partial", evidence_ids: [evidenceId],
      findings: [{ id: finding, title: `finding ${finding}`, evidence_ids: [evidenceId] }],
      coverage: { coding_trace: { evidence_ids: [evidenceId] } },
    } },
  };
}

test("pair order alternates deterministically from the explicit ordinal", () => {
  assert.deepEqual(focusPairOrder(0), ["blind", "focused"]);
  assert.deepEqual(focusPairOrder(1), ["focused", "blind"]);
  assert.deepEqual(focusPairOrder(2), ["blind", "focused"]);
  assert.throws(() => focusPairOrder(-1), /non-negative integer/);
});

test("paired contract permits run paths but rejects evaluator drift", () => {
  assert.equal(validatePairedContracts(result("pair-blind"), result("pair-focused")).subject_revision, revision);
  assert.throws(() => validatePairedContracts(
    result("pair-blind"), result("pair-focused", { model: "other" }),
  ), /contract drift: models/);
});

test("pair arms are sequential and focus is isolated to the focused arm", async (t) => {
  const root = await mkdtemp(path.join(process.env.TMPDIR || "/tmp", "piku-pair-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  const focusPath = path.join(root, "events.jsonl");
  await writeFile(focusPath, `${JSON.stringify({ promotion_id: "promotion-1" })}\n`);
  const calls = [];
  let active = 0;
  const evaluate = async ({ environment }) => {
    active += 1;
    assert.equal(active, 1, "arms must never overlap");
    calls.push({ runId: environment.PIKU_EVAL_RUN_ID, focus: environment.PIKU_EVAL_FOCUS_EVENTS });
    await new Promise((resolve) => setImmediate(resolve));
    active -= 1;
    return result(environment.PIKU_EVAL_RUN_ID);
  };
  const paired = await runFocusPair({
    pairOrdinal: 1, focusEventsPath: focusPath, pairId: "pair-one", outputRoot: root,
    environment: { PIKU_EVAL_FOCUS_EVENTS: "/must/not/leak" }, evaluate,
  });
  assert.deepEqual(calls.map((call) => call.runId), ["pair-one-focused", "pair-one-blind"]);
  assert.equal(calls[0].focus, focusPath);
  assert.equal(calls[1].focus, undefined);
  assert.notEqual(paired.manifest.arms.blind.run_id, paired.manifest.arms.focused.run_id);
  assert.equal(JSON.parse(await readFile(path.join(paired.pairDir, "manifest.json"))).automatic_focus_mutation, false);
  await assert.rejects(runFocusPair({
    pairOrdinal: 1, focusEventsPath: focusPath, pairId: "pair-one", outputRoot: root, evaluate,
  }), /EEXIST/);
});

test("incomplete arm makes the dossier inconclusive without a winner or score", () => {
  const report = buildEvidenceQualityDossier({
    pairId: "pair", pairOrdinal: 0, order: focusPairOrder(0),
    blind: result("blind"), focused: result("focused", { status: "timeout" }),
  });
  assert.equal(report.status, "inconclusive");
  assert.equal(report.comparison_kind, "evidence_quality_dossier");
  assert.equal("winner" in report, false);
  assert.equal("score" in report, false);
  assert.deepEqual(report.arms.blind.cited_evidence_ids, ["coding_trace:blind"]);
  assert.equal(report.promotion_or_retirement, "operator_only");
});

test("focus echo check is narrow and requires a verbatim question without citations", () => {
  const focused = result("focused");
  focused.synthesis.report = {
    verdict: "partial",
    evidence_ids: [],
    findings: [{ id: "echo", title: "Can recovery be verified?", evidence_ids: [] }],
    coverage: {},
  };
  const report = buildEvidenceQualityDossier({
    pairId: "pair", pairOrdinal: 0, order: focusPairOrder(0),
    blind: result("blind"), focused, focusQuestions: ["Can recovery be verified?"],
  });
  assert.deepEqual(
    report.arms.focused.verbatim_focus_question_echo_without_cited_evidence,
    ["Can recovery be verified?"],
  );
});

test("a thrown arm is recorded as inconclusive and the other arm still runs", async (t) => {
  const root = await mkdtemp(path.join(process.env.TMPDIR || "/tmp", "piku-pair-failure-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  const focusPath = path.join(root, "events.jsonl");
  await writeFile(focusPath, `${JSON.stringify({ proposal_id: "proposal-1" })}\n`);
  let calls = 0;
  const paired = await runFocusPair({
    pairOrdinal: 0, focusEventsPath: focusPath, pairId: "pair-failure", outputRoot: root,
    evaluate: async ({ environment }) => {
      calls += 1;
      if (environment.PIKU_EVAL_RUN_ID.endsWith("-blind")) throw new Error("fixture failure");
      return result(environment.PIKU_EVAL_RUN_ID);
    },
  });
  assert.equal(calls, 2);
  assert.equal(paired.report.status, "inconclusive");
  assert.match(paired.report.effort_confounds[0], /blind arm failed/);
});
