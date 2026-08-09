import { readdir, readFile } from "node:fs/promises";
import path from "node:path";
import { evaluationEnvelopeErrors } from "./evaluation-envelope.mjs";

const root = path.resolve(process.argv[2] || "target/live-ledger");
let files = [];
try {
  files = (await readdir(root))
    .filter((name) => name.endsWith(".jsonl"))
    .sort()
    .map((name) => path.join(root, name));
} catch (error) {
  if (error.code !== "ENOENT") throw error;
}

const rows = [];
const rowOrder = new WeakMap();
const legacyRows = [];
const quarantined = [];
for (const file of files) {
  for (const [index, line] of (await readFile(file, "utf8")).split("\n").entries()) {
    if (!line.trim()) continue;
    const source = `${file}:${index + 1}`;
    let row;
    try {
      row = JSON.parse(line);
    } catch (error) {
      quarantined.push({ source, errors: [`invalid JSON: ${error.message}`] });
      continue;
    }
    if (row !== null && typeof row === "object" && !Array.isArray(row) && !Object.hasOwn(row, "schema_version")) {
      legacyRows.push({ source, row });
      continue;
    }
    const errors = evaluationEnvelopeErrors(row);
    if (errors.length) quarantined.push({ source, errors });
    else {
      rowOrder.set(row, rows.length);
      rows.push(row);
    }
  }
}

const originals = rows.filter((row) => row.record_kind !== "amendment");
const amendments = rows.filter((row) => row.record_kind === "amendment");
const stageKey = (runId, stageId) => `${runId}\u001f${stageId}`;
const effective = new Map();
for (const row of originals) {
  effective.set(stageKey(row.run_id, row.stage_id), {
    run_id: row.run_id,
    stage_id: row.stage_id,
    original_product_verdict: row.product_verdict,
    product_verdict: row.product_verdict,
    finding_count: row.finding_count,
    evidence_ids: [...row.evidence_ids],
    eligibility: { evidence: true, findings: true, verdict: true },
    disposition: { evidence: "original", findings: "original", verdict: "original" },
    amendments_applied: 0,
    amendment_history: [],
  });
}
const orphanAmendments = [];
const duplicateAmendments = [];
const seenEventIds = new Set();
for (const amendment of amendments) {
  if (seenEventIds.has(amendment.event_id)) {
    duplicateAmendments.push({
      event_id: amendment.event_id,
      ledger_sequence: rowOrder.get(amendment),
      target_run_id: amendment.target_run_id,
      target_stage_id: amendment.target_stage_id,
    });
    continue;
  }
  seenEventIds.add(amendment.event_id);
  const state = effective.get(stageKey(amendment.target_run_id, amendment.target_stage_id));
  if (!state) {
    orphanAmendments.push({
      run_id: amendment.run_id,
      stage_id: amendment.stage_id,
      target_run_id: amendment.target_run_id,
      target_stage_id: amendment.target_stage_id,
      reason_code: amendment.reason_code,
    });
    continue;
  }
  state.amendments_applied += 1;
  state.amendment_history.push({
    action: amendment.amendment_action,
    event_id: amendment.event_id,
    recorded_at: amendment.recorded_at,
    ledger_sequence: rowOrder.get(amendment),
    reason_code: amendment.reason_code,
    scope: amendment.amendment_scope,
    basis_refs: amendment.basis_refs,
    basis_hashes: amendment.basis_hashes,
    actor: amendment.actor,
    tool_version: amendment.tool_version,
  });
  const selected = {
    evidence: amendment.amendment_scope.evidence_ids.length > 0,
    findings: amendment.amendment_scope.finding_refs.length > 0,
    verdict: amendment.amendment_scope.verdict,
  };
  for (const component of ["evidence", "findings", "verdict"]) {
    if (!selected[component]) continue;
    if (amendment.amendment_action === "qualify") {
      state.disposition[component] = "qualified";
      continue;
    }
    if (amendment.amendment_action === "reinstate") {
      state.eligibility[component] = true;
      state.disposition[component] = "reinstated";
      if (component === "verdict") state.product_verdict = state.original_product_verdict;
      continue;
    }
    const replacementState = amendment.replacement_run_id === null ? null :
      effective.get(stageKey(amendment.replacement_run_id, amendment.target_stage_id));
    const replacement = amendment.replacement ?? (replacementState ? {
      product_verdict: replacementState.product_verdict,
      finding_count: replacementState.finding_count,
      evidence_ids: replacementState.evidence_ids,
    } : null);
    const replacementField = component === "verdict" ? "product_verdict" : component === "findings" ? "finding_count" : "evidence_ids";
    const hasReplacement = amendment.amendment_action === "supersede" &&
      replacement !== null && Object.hasOwn(replacement, replacementField);
    state.eligibility[component] = hasReplacement;
    state.disposition[component] = hasReplacement ? "superseded_with_replacement" : amendment.amendment_action;
    if (hasReplacement) {
      if (component === "verdict") state.product_verdict = replacement.product_verdict;
      else if (component === "findings") state.finding_count = replacement.finding_count;
      else state.evidence_ids = [...replacement.evidence_ids];
    } else if (component === "verdict") {
      state.product_verdict = null;
    } else if (component === "evidence") {
      const invalid = new Set(amendment.amendment_scope.evidence_ids);
      state.evidence_ids = state.evidence_ids.filter((id) => !invalid.has(id));
    }
  }
}
const effectiveRecords = [...effective.values()];

const countBy = (field, sourceRows = rows) =>
  Object.fromEntries(
    [...new Set(sourceRows.map((row) => row[field] ?? "legacy"))]
      .sort()
      .map((value) => [value, sourceRows.filter((row) => (row[field] ?? "legacy") === value).length]),
  );
const verdictKey = (row) =>
  !Object.hasOwn(row, "product_verdict")
    ? "legacy"
    : row.product_verdict === null
      ? "unjudged"
      : row.product_verdict;
const eligibleVerdictRows = effectiveRecords
  .filter((state) => state.eligibility.verdict)
  .map((state) => ({ product_verdict: state.product_verdict }));
const productVerdicts = Object.fromEntries(
  [...new Set(eligibleVerdictRows.map(verdictKey))]
    .sort()
    .map((value) => [value, eligibleVerdictRows.filter((row) => verdictKey(row) === value).length]),
);
const priorityRank = { high: 0, medium: 1, low: 2 };
const rawFollowups = originals
  .flatMap((row) =>
    (Array.isArray(row.followups) ? row.followups : []).map((followup) => ({
      ...followup,
      source_run: row.run_id ?? null,
      source_surface: row.surface ?? "legacy",
    })),
  );
const groupedFollowups = new Map();
for (const followup of rawFollowups) {
  const evidence = [...(followup.evidence_ids || [])].sort();
  const identity = evidence.length
    ? `${followup.kind}|${followup.perspective || "all"}|${evidence.join("\u001f")}`
    : `${followup.kind}|${followup.perspective || "all"}|${followup.title.toLowerCase().replaceAll(/[^a-z0-9]+/g, " ").trim()}`;
  const existing = groupedFollowups.get(identity);
  if (!existing) {
    groupedFollowups.set(identity, {
      ...followup,
      occurrences: 1,
      sources: [{ run: followup.source_run, surface: followup.source_surface }],
    });
    continue;
  }
  existing.occurrences += 1;
  existing.sources.push({ run: followup.source_run, surface: followup.source_surface });
  if ((priorityRank[followup.priority] ?? 3) < (priorityRank[existing.priority] ?? 3)) {
    const { occurrences, sources } = existing;
    Object.assign(existing, followup, { occurrences, sources });
  }
}
const followups = [...groupedFollowups.values()]
  .sort(
    (left, right) =>
      (priorityRank[left.priority] ?? 3) - (priorityRank[right.priority] ?? 3) ||
      left.title.localeCompare(right.title),
  );

console.log(
  JSON.stringify(
    {
      ledger: root,
      runs: new Set(originals.map((row) => row.run_id).filter(Boolean)).size,
      stages: new Set(originals.map((row) => stageKey(row.run_id, row.stage_id))).size,
      records: rows.length,
      original_records: originals.length,
      amendment_records: amendments.length,
      legacy_records: legacyRows.length,
      quarantined_records: quarantined.length,
      quarantine: quarantined,
      by_record_kind: countBy("record_kind"),
      by_surface: countBy("surface", originals),
      by_status: countBy("run_status", originals),
      by_scenario: countBy("scenario_id", originals),
      by_subject_version: countBy("subject_version", originals),
      by_subject_revision: countBy("subject_revision", originals),
      by_evaluator_version: countBy("evaluator_version", originals),
      product_verdicts: productVerdicts,
      ineligible_verdicts: effectiveRecords.filter((state) => !state.eligibility.verdict).length,
      effective_records: effectiveRecords,
      orphan_amendments: orphanAmendments,
      duplicate_amendments: duplicateAmendments,
      followup_count: followups.length,
      raw_followup_count: rawFollowups.length,
      followups,
    },
    null,
    2,
  ),
);
