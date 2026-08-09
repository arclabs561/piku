import assert from "node:assert/strict";
import { test } from "node:test";
import {
  canonicalEvaluationFocus,
  evaluationStageToFocusProposals,
  evaluationFocusEventErrors,
  projectEvaluationFocus,
} from "./evaluation-focus.mjs";

const HASH = `sha256:${"a".repeat(64)}`;
const OTHER_HASH = `sha256:${"b".repeat(64)}`;
const SCOPE = { surface: "web", scenario_id: "operator-journey", perspective: "recovery" };
const OPTIONS = {
  subjectStateHash: HASH,
  now: "2026-08-09T12:00:00.000Z",
  allowedTargets: [SCOPE],
  maxProjectionBytes: 4096,
  categoryQuotas: { recovery: 2 },
};

function stage(overrides = {}) {
  return {
    schema_version: 2,
    run_id: "run-2",
    record_kind: "stage",
    stage_id: "result",
    scenario_id: "operator-journey",
    surface: "web",
    perspective: "synthesis",
    subject_revision: "0123456789abcdef0123456789abcdef01234567",
    subject_dirty: false,
    task_contract: "Evaluate recovery",
    run_status: "completed",
    failure_class: "none",
    product_verdict: "supported",
    finding_count: 1,
    finding_refs: ["run-2:result:finding:f1"],
    evidence_ids: ["e-stage"],
    artifact_refs: [],
    followups: [{
      obligation_id: "run-2:result:obligation:o1",
      kind: "retest",
      priority: "high",
      title: "Recovery after cancellation",
      rationale: "Repeat the cancellation flow after the fix",
      perspective: "recovery",
      evidence_ids: ["e-followup"],
      finding_refs: ["run-2:result:finding:f1"],
      retest_of: "run-1:result:obligation:o1",
    }],
    duration_ms: 1,
    ...overrides,
  };
}

const CONVERSION_OPTIONS = {
  sourceRevision: "0123456789abcdef0123456789abcdef01234567",
  subjectStateHash: HASH,
  allowedTargets: [SCOPE],
  recordedAt: "2026-08-09T12:00:00.000Z",
  suggestedExpiresAt: "2026-08-12T00:00:00.000Z",
};

function proposal(overrides = {}) {
  return {
    schema_version: 1,
    event_id: "event-proposal-1",
    event_kind: "proposal",
    recorded_at: "2026-08-09T10:00:00.000Z",
    actor: { kind: "judge", id: "recovery-judge" },
    subject_state_hash: HASH,
    proposal_id: "proposal-1",
    source_run_id: "run-1",
    scope: SCOPE,
    evidence_refs: ["run-1:screenshot:1"],
    question: "Can the operator recover after a cancelled turn?",
    category: "recovery",
    suggested_expires_at: "2026-08-12T00:00:00.000Z",
    task_clause: "Resume an interrupted workspace",
    ...overrides,
  };
}

function promotion(overrides = {}) {
  return {
    schema_version: 1,
    event_id: "event-promotion-1",
    event_kind: "promotion",
    recorded_at: "2026-08-09T11:00:00.000Z",
    actor: { kind: "operator", id: "local-operator" },
    subject_state_hash: HASH,
    promotion_id: "promotion-1",
    proposal_id: "proposal-1",
    scope: SCOPE,
    activates_at: "2026-08-09T11:30:00.000Z",
    expires_at: "2026-08-10T00:00:00.000Z",
    max_prompt_bytes: 1024,
    retest_obligation: "run-1:result:obligation:o1",
    ...overrides,
  };
}

function retirement(overrides = {}) {
  return {
    schema_version: 1,
    event_id: "event-retirement-1",
    event_kind: "retirement",
    recorded_at: "2026-08-09T11:45:00.000Z",
    actor: { kind: "reviewer", id: "reviewer-1" },
    subject_state_hash: HASH,
    retirement_id: "retirement-1",
    promotion_id: "promotion-1",
    reason: "The behavior is covered by a deterministic test",
    ...overrides,
  };
}

test("proposals have no authority until an operator promotion activates them", () => {
  assert.deepEqual(projectEvaluationFocus([proposal()], OPTIONS).items, []);
  const projection = projectEvaluationFocus([proposal(), promotion()], OPTIONS);
  assert.equal(projection.items.length, 1);
  assert.equal(projection.items[0].question, proposal().question);
  assert.equal(projection.items[0].promotion_id, "promotion-1");
});

test("projection is canonical regardless of object key insertion order", () => {
  const projection = projectEvaluationFocus([proposal(), promotion()], OPTIONS);
  const reordered = { items: projection.items, projected_at: projection.projected_at,
    subject_state_hash: projection.subject_state_hash, schema_version: 1 };
  assert.equal(canonicalEvaluationFocus(projection), canonicalEvaluationFocus(reordered));
  assert.match(canonicalEvaluationFocus(projection), /"items"/);
});

test("retirement removes an active focus without rewriting history", () => {
  const projection = projectEvaluationFocus([proposal(), promotion(), retirement()], OPTIONS);
  assert.deepEqual(projection.items, []);
});

test("judge-authored promotions and non-question focus fail closed", () => {
  assert.ok(evaluationFocusEventErrors(promotion({ actor: { kind: "judge", id: "judge" } }))
    .some((error) => error.includes("operator or reviewer")));
  assert.ok(evaluationFocusEventErrors(proposal({ question: "Inspect recovery" }))
    .some((error) => error.includes("question-form")));
  assert.throws(() => projectEvaluationFocus([
    proposal(), promotion({ actor: { kind: "judge", id: "judge" } }),
  ], OPTIONS), /operator or reviewer/);
});

test("unknown targets and stale subject state fail closed", () => {
  assert.throws(() => projectEvaluationFocus([
    proposal({ scope: { ...SCOPE, scenario_id: "unknown" } }),
  ], OPTIONS), /unknown scoped target/);
  assert.throws(() => projectEvaluationFocus([
    proposal({ subject_state_hash: OTHER_HASH }),
  ], OPTIONS), /stale subject_state_hash/);
});

test("expired active promotions fail closed", () => {
  assert.throws(() => projectEvaluationFocus([
    proposal(), promotion({ expires_at: "2026-08-09T11:59:59.000Z" }),
  ], OPTIONS), /expired promotion/);
});

test("duplicate and conflicting event identities fail closed", () => {
  assert.throws(() => projectEvaluationFocus([proposal(), proposal()], OPTIONS), /duplicate event_id/);
  assert.throws(() => projectEvaluationFocus([
    proposal(), promotion(), promotion({ event_id: "event-promotion-2", promotion_id: "promotion-2" }),
  ], OPTIONS), /conflicting promotions/);
  assert.throws(() => projectEvaluationFocus([
    proposal(), promotion(), retirement(), retirement({ event_id: "event-retirement-2", retirement_id: "retirement-2" }),
  ], OPTIONS), /duplicate retirement/);
});

test("missing lineage and promotion scope conflicts fail closed", () => {
  assert.throws(() => projectEvaluationFocus([promotion()], OPTIONS), /unknown proposal/);
  assert.throws(() => projectEvaluationFocus([
    proposal(), promotion({ scope: { ...SCOPE, perspective: "coding_trace" } }),
  ], { ...OPTIONS, allowedTargets: [SCOPE, { ...SCOPE, perspective: "coding_trace" }] }), /scope conflicts/);
});

test("projection and per-promotion budgets reject instead of truncate", () => {
  assert.throws(() => projectEvaluationFocus([
    proposal(), promotion({ max_prompt_bytes: 128 }),
  ], OPTIONS), /max_prompt_bytes/);
  assert.throws(() => projectEvaluationFocus([
    proposal(), promotion(),
  ], { ...OPTIONS, maxProjectionBytes: 64 }), /projection exceeds/);
  assert.throws(() => projectEvaluationFocus([
    proposal(), promotion(),
  ], { ...OPTIONS, categoryQuotas: { recovery: 0 } }), /category quota exceeded/);
});

test("completed clean v2 stages produce only inert cited retest proposals", () => {
  const input = stage({ followups: [
    stage().followups[0],
    {
      ...stage().followups[0], obligation_id: "run-2:result:obligation:o2",
      kind: "todo", retest_of: null,
    },
    {
      ...stage().followups[0], obligation_id: "run-2:result:obligation:o3",
      perspective: null,
    },
  ] });
  const events = evaluationStageToFocusProposals(input, CONVERSION_OPTIONS);
  assert.equal(events.length, 1);
  assert.equal(events[0].event_kind, "proposal");
  assert.equal(Object.hasOwn(events[0], "promotion_id"), false);
  assert.deepEqual(events[0].evidence_refs,
    ["e-followup", "run-2:result:finding:f1"]);
  assert.equal(events[0].question, "Can we verify Recovery after cancellation?");
  assert.deepEqual(evaluationFocusEventErrors(events[0]), []);
  assert.deepEqual(projectEvaluationFocus(events, OPTIONS).items, []);
});

test("proposal identities and ordering are stable across followup order", () => {
  const second = {
    ...stage().followups[0],
    obligation_id: "run-2:result:obligation:o2",
    title: "A second retest",
  };
  const forward = evaluationStageToFocusProposals(
    stage({ followups: [stage().followups[0], second] }), CONVERSION_OPTIONS);
  const reverse = evaluationStageToFocusProposals(
    stage({ followups: [second, stage().followups[0]] }), CONVERSION_OPTIONS);
  assert.deepEqual(forward, reverse);
  assert.notEqual(forward[0].proposal_id, forward[1].proposal_id);
});

test("conversion rejects incomplete, dirty, stale, and unvalidated stages", () => {
  assert.throws(() => evaluationStageToFocusProposals(
    stage({ schema_version: 1 }), CONVERSION_OPTIONS), /schema_version 2/);
  assert.throws(() => evaluationStageToFocusProposals(
    stage({ run_status: "product_failure" }), CONVERSION_OPTIONS), /must be completed/);
  assert.throws(() => evaluationStageToFocusProposals(
    stage({ subject_dirty: true }), CONVERSION_OPTIONS), /must be clean/);
  assert.throws(() => evaluationStageToFocusProposals(
    stage(), { ...CONVERSION_OPTIONS, sourceRevision: "different" }), /revision is stale/);
  assert.throws(() => evaluationStageToFocusProposals(
    stage({ followups: [{ ...stage().followups[0], evidence_ids: [], finding_refs: [] }] }),
    CONVERSION_OPTIONS), /must cite evidence_ids or finding_refs/);
});

test("conversion allowlists exact scopes and bounds generated prompt text", () => {
  assert.deepEqual(evaluationStageToFocusProposals(stage(), {
    ...CONVERSION_OPTIONS,
    allowedTargets: [{ ...SCOPE, perspective: "coding_trace" }],
  }), []);
  const long = "word ".repeat(200);
  const [event] = evaluationStageToFocusProposals(stage({ followups: [{
    ...stage().followups[0], title: long, rationale: long,
  }] }), CONVERSION_OPTIONS);
  assert.ok(event.question.length <= 240);
  assert.ok(event.question.endsWith("?"));
  assert.ok(event.task_clause.length <= 500);
});
