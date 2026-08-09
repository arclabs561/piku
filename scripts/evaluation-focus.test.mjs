import assert from "node:assert/strict";
import { test } from "node:test";
import {
  canonicalEvaluationFocus,
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
