export const EVALUATION_SCHEMA_VERSION = 2;
const READABLE_SCHEMA_VERSIONS = new Set([1, EVALUATION_SCHEMA_VERSION]);

const RUN_STATUSES = new Set([
  "completed",
  "product_failure",
  "harness_failure",
  "infrastructure_failure",
  "timeout",
  "inconclusive",
]);
const PRODUCT_VERDICTS = new Set(["supported", "partial", "not_supported", null]);
const RECORD_KINDS = new Set(["run", "stage", "amendment"]);
const AMENDMENT_ACTIONS = new Set(["invalidate", "qualify", "supersede", "reinstate"]);
const SURFACES = new Set(["cli", "tui", "web"]);
const FOLLOWUP_KINDS = new Set(["todo", "idea", "retest"]);
const PRIORITIES = new Set(["high", "medium", "low"]);
const FINDING_REF = /^[^:]+:[^:]+:finding:f[1-9][0-9]*$/;
const OBLIGATION_REF = /^[^:]+:[^:]+:obligation:o[1-9][0-9]*$/;

function isObject(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function nonEmptyString(value) {
  return typeof value === "string" && value.length > 0;
}

function nullableString(value) {
  return value === null || typeof value === "string";
}

function stringArray(value, allowEmptyStrings = false) {
  return Array.isArray(value) && value.every((item) =>
    typeof item === "string" && (allowEmptyStrings || item.length > 0));
}

function validateFollowup(followup, index, errors, schemaVersion) {
  const prefix = `followups[${index}]`;
  if (!isObject(followup)) {
    errors.push(`${prefix} must be an object`);
    return;
  }
  const expected = new Set(["kind", "priority", "title", "rationale", "perspective", "evidence_ids"]);
  if (schemaVersion >= 2) {
    expected.add("obligation_id");
    expected.add("finding_refs");
    expected.add("retest_of");
  }
  for (const field of expected) {
    if (!Object.hasOwn(followup, field)) errors.push(`${prefix}.${field} is required`);
  }
  for (const field of Object.keys(followup)) {
    if (!expected.has(field)) errors.push(`${prefix}.${field} is not allowed`);
  }
  if (!FOLLOWUP_KINDS.has(followup.kind)) errors.push(`${prefix}.kind is invalid`);
  if (!PRIORITIES.has(followup.priority)) errors.push(`${prefix}.priority is invalid`);
  if (!nonEmptyString(followup.title)) errors.push(`${prefix}.title must be a non-empty string`);
  if (!nonEmptyString(followup.rationale)) errors.push(`${prefix}.rationale must be a non-empty string`);
  if (!nullableString(followup.perspective)) errors.push(`${prefix}.perspective must be a string or null`);
  if (!stringArray(followup.evidence_ids)) errors.push(`${prefix}.evidence_ids must contain non-empty strings`);
  if (schemaVersion >= 2) {
    if (!nonEmptyString(followup.obligation_id))
      errors.push(`${prefix}.obligation_id must be a non-empty string`);
    else if (!OBLIGATION_REF.test(followup.obligation_id))
      errors.push(`${prefix}.obligation_id must be a scoped obligation reference`);
    if (!stringArray(followup.finding_refs))
      errors.push(`${prefix}.finding_refs must contain non-empty strings`);
    if (!nullableString(followup.retest_of))
      errors.push(`${prefix}.retest_of must be a string or null`);
    else if (followup.retest_of !== null && !OBLIGATION_REF.test(followup.retest_of))
      errors.push(`${prefix}.retest_of must be a scoped obligation reference or null`);
    if (followup.evidence_ids?.length === 0 && followup.finding_refs?.length === 0)
      errors.push(`${prefix} must cite evidence_ids or finding_refs`);
  }
}

function validateAmendment(record, errors) {
  for (const field of ["target_run_id", "target_stage_id", "event_id", "contract_version", "reason_code", "actor", "tool_version"]) {
    if (!nonEmptyString(record[field])) errors.push(`${field} must be a non-empty string`);
  }
  if (!nonEmptyString(record.recorded_at) || Number.isNaN(Date.parse(record.recorded_at)))
    errors.push("recorded_at must be an ISO 8601 date-time");
  if (record.stage_id === record.target_stage_id)
    errors.push("an amendment must not reuse its target stage_id");
  if (!AMENDMENT_ACTIONS.has(record.amendment_action)) errors.push("amendment_action is invalid");
  if (!stringArray(record.basis_refs) || record.basis_refs.length === 0)
    errors.push("basis_refs must contain at least one non-empty string");
  if (!stringArray(record.basis_hashes) || record.basis_hashes.length === 0 ||
      record.basis_hashes.some((hash) => !/^sha256:[0-9a-f]{64}$/.test(hash)))
    errors.push("basis_hashes must contain at least one sha256:<64 lowercase hex> digest");
  if (Object.hasOwn(record, "replacement_run_id") && !nullableString(record.replacement_run_id))
    errors.push("replacement_run_id must be a string or null");
  const scope = record.amendment_scope;
  if (!isObject(scope)) errors.push("amendment_scope must be an object");
  else {
    if (!stringArray(scope.evidence_ids)) errors.push("amendment_scope.evidence_ids must contain non-empty strings");
    if (!stringArray(scope.finding_refs)) errors.push("amendment_scope.finding_refs must contain non-empty strings");
    if (typeof scope.verdict !== "boolean") errors.push("amendment_scope.verdict must be boolean");
    if (scope.verdict !== true && scope.evidence_ids?.length === 0 && scope.finding_refs?.length === 0)
      errors.push("amendment_scope must select evidence, findings, or verdict");
  }
  if (Object.hasOwn(record, "replacement") && record.replacement !== null) {
    const replacement = record.replacement;
    if (!isObject(replacement)) errors.push("replacement must be an object or null");
    else {
      if (Object.hasOwn(replacement, "product_verdict") && !PRODUCT_VERDICTS.has(replacement.product_verdict))
        errors.push("replacement.product_verdict is invalid");
      if (Object.hasOwn(replacement, "finding_count") &&
          !(replacement.finding_count === null || (Number.isInteger(replacement.finding_count) && replacement.finding_count >= 0)))
        errors.push("replacement.finding_count must be a non-negative integer or null");
      if (Object.hasOwn(replacement, "evidence_ids") && !stringArray(replacement.evidence_ids))
        errors.push("replacement.evidence_ids must contain non-empty strings");
      if (Object.hasOwn(replacement, "finding_refs") && !stringArray(replacement.finding_refs))
        errors.push("replacement.finding_refs must contain non-empty strings");
    }
  }
}

export function evaluationEnvelopeErrors(record) {
  if (!isObject(record)) return ["record must be an object"];
  const errors = [];
  const requiredStrings = [
    "run_id", "scenario_id", "perspective", "task_contract", "failure_class",
  ];
  for (const field of requiredStrings) {
    if (!nonEmptyString(record[field])) errors.push(`${field} must be a non-empty string`);
  }
  if (!READABLE_SCHEMA_VERSIONS.has(record.schema_version))
    errors.push(`schema_version must be 1 or ${EVALUATION_SCHEMA_VERSION}`);
  if (!SURFACES.has(record.surface)) errors.push("surface must be cli, tui, or web");
  if (!RUN_STATUSES.has(record.run_status)) errors.push("run_status is invalid");
  if (!PRODUCT_VERDICTS.has(record.product_verdict)) errors.push("product_verdict is invalid");
  if (!(record.finding_count === null || (Number.isInteger(record.finding_count) && record.finding_count >= 0)))
    errors.push("finding_count must be a non-negative integer or null");
  if (!stringArray(record.evidence_ids)) errors.push("evidence_ids must contain non-empty strings");
  if (!stringArray(record.artifact_refs)) errors.push("artifact_refs must contain non-empty strings");
  if (!Array.isArray(record.followups)) errors.push("followups must be an array");
  else record.followups.forEach((followup, index) => validateFollowup(followup, index, errors, record.schema_version));
  if (record.schema_version >= 2) {
    if (!stringArray(record.finding_refs)) errors.push("finding_refs must contain non-empty strings");
    else if (record.finding_refs.some((reference) => !FINDING_REF.test(reference)))
      errors.push("finding_refs must contain scoped finding references");
    const findingRefs = new Set(record.finding_refs || []);
    if (findingRefs.size !== (record.finding_refs || []).length)
      errors.push("finding_refs must be unique");
    const obligations = record.followups?.map((followup) => followup.obligation_id) || [];
    if (new Set(obligations).size !== obligations.length)
      errors.push("followup obligation_id values must be unique");
    for (const [index, followup] of (record.followups || []).entries()) {
      for (const findingRef of followup.finding_refs || []) {
        if (!findingRefs.has(findingRef))
          errors.push(`followups[${index}].finding_refs cites an unknown top-level finding ref`);
      }
    }
  }
  if (!(Number.isInteger(record.duration_ms) && record.duration_ms >= 0))
    errors.push("duration_ms must be a non-negative integer");

  for (const field of ["subject_surface", "subject_model", "explorer_model", "judge_model"]) {
    if (Object.hasOwn(record, field) && !nullableString(record[field]))
      errors.push(`${field} must be a string or null`);
  }
  if (Object.hasOwn(record, "record_kind") && !RECORD_KINDS.has(record.record_kind))
    errors.push("record_kind must be run, stage, or amendment");
  if (!Object.hasOwn(record, "record_kind")) errors.push("record_kind is required");
  if (Object.hasOwn(record, "stage_id") && !nonEmptyString(record.stage_id))
    errors.push("stage_id must be a non-empty string");
  if (!Object.hasOwn(record, "stage_id")) errors.push("stage_id is required");
  if (record.record_kind === "amendment") validateAmendment(record, errors);
  return errors;
}

export function assertEvaluationEnvelope(record, context = "evaluation record") {
  const errors = evaluationEnvelopeErrors(record);
  if (errors.length) throw new TypeError(`${context} is invalid: ${errors.join("; ")}`);
  return record;
}
