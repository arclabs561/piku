const DIGEST = /^sha256:[0-9a-f]{64}$/;
const ACTOR_KINDS = new Set(["judge", "harness", "operator", "reviewer"]);
const AUTHORITY_KINDS = new Set(["operator", "reviewer"]);
const EVENT_KINDS = new Set(["proposal", "promotion", "retirement"]);
const SURFACES = new Set(["cli", "tui", "web"]);

function isObject(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function string(value) {
  return typeof value === "string" && value.length > 0;
}

function date(value) {
  return string(value) && !Number.isNaN(Date.parse(value));
}

function exactFields(value, expected, prefix, errors) {
  for (const field of expected) {
    if (!Object.hasOwn(value, field)) errors.push(`${prefix}.${field} is required`);
  }
  for (const field of Object.keys(value)) {
    if (!expected.has(field)) errors.push(`${prefix}.${field} is not allowed`);
  }
}

function scopeErrors(scope, prefix, errors) {
  if (!isObject(scope)) {
    errors.push(`${prefix} must be an object`);
    return;
  }
  exactFields(scope, new Set(["surface", "scenario_id", "perspective"]), prefix, errors);
  if (!SURFACES.has(scope.surface)) errors.push(`${prefix}.surface is invalid`);
  if (!string(scope.scenario_id)) errors.push(`${prefix}.scenario_id must be a non-empty string`);
  if (!string(scope.perspective)) errors.push(`${prefix}.perspective must be a non-empty string`);
}

export function evaluationFocusEventErrors(event) {
  if (!isObject(event)) return ["event must be an object"];
  const errors = [];
  const common = ["schema_version", "event_id", "event_kind", "recorded_at", "actor", "subject_state_hash"];
  if (event.schema_version !== 1) errors.push("schema_version must be 1");
  if (!string(event.event_id)) errors.push("event_id must be a non-empty string");
  if (!EVENT_KINDS.has(event.event_kind)) errors.push("event_kind is invalid");
  if (!date(event.recorded_at)) errors.push("recorded_at must be an ISO 8601 date-time");
  if (!isObject(event.actor)) errors.push("actor must be an object");
  else {
    exactFields(event.actor, new Set(["kind", "id"]), "actor", errors);
    if (!ACTOR_KINDS.has(event.actor.kind)) errors.push("actor.kind is invalid");
    if (!string(event.actor.id)) errors.push("actor.id must be a non-empty string");
  }
  if (!DIGEST.test(event.subject_state_hash ?? ""))
    errors.push("subject_state_hash must be a sha256 digest");

  if (event.event_kind === "proposal") {
    exactFields(event, new Set([...common, "proposal_id", "source_run_id", "scope", "evidence_refs",
      "question", "category", "suggested_expires_at", "task_clause"]), "event", errors);
    for (const field of ["proposal_id", "source_run_id", "category", "task_clause"])
      if (!string(event[field])) errors.push(`${field} must be a non-empty string`);
    scopeErrors(event.scope, "scope", errors);
    if (!Array.isArray(event.evidence_refs) || event.evidence_refs.length === 0 ||
        event.evidence_refs.some((item) => !string(item)) ||
        new Set(event.evidence_refs).size !== event.evidence_refs.length)
      errors.push("evidence_refs must contain unique non-empty strings");
    if (!string(event.question) || !event.question.endsWith("?"))
      errors.push("question must be question-form and end with ?");
    if (!date(event.suggested_expires_at))
      errors.push("suggested_expires_at must be an ISO 8601 date-time");
  } else if (event.event_kind === "promotion") {
    exactFields(event, new Set([...common, "promotion_id", "proposal_id", "scope", "activates_at",
      "expires_at", "max_prompt_bytes", "retest_obligation"]), "event", errors);
    for (const field of ["promotion_id", "proposal_id"])
      if (!string(event[field])) errors.push(`${field} must be a non-empty string`);
    scopeErrors(event.scope, "scope", errors);
    if (!date(event.activates_at)) errors.push("activates_at must be an ISO 8601 date-time");
    if (!date(event.expires_at)) errors.push("expires_at must be an ISO 8601 date-time");
    if (date(event.activates_at) && date(event.expires_at) &&
        Date.parse(event.expires_at) <= Date.parse(event.activates_at))
      errors.push("expires_at must be after activates_at");
    if (!Number.isInteger(event.max_prompt_bytes) || event.max_prompt_bytes < 1)
      errors.push("max_prompt_bytes must be a positive integer");
    if (event.retest_obligation !== null && !string(event.retest_obligation))
      errors.push("retest_obligation must be a non-empty string or null");
    if (isObject(event.actor) && !AUTHORITY_KINDS.has(event.actor.kind))
      errors.push("only an operator or reviewer may author a promotion");
  } else if (event.event_kind === "retirement") {
    exactFields(event, new Set([...common, "retirement_id", "promotion_id", "reason"]), "event", errors);
    for (const field of ["retirement_id", "promotion_id", "reason"])
      if (!string(event[field])) errors.push(`${field} must be a non-empty string`);
    if (isObject(event.actor) && !AUTHORITY_KINDS.has(event.actor.kind))
      errors.push("only an operator or reviewer may author a retirement");
  }
  return errors;
}

export function assertEvaluationFocusEvent(event, context = "evaluation focus event") {
  const errors = evaluationFocusEventErrors(event);
  if (errors.length) throw new TypeError(`${context} is invalid: ${errors.join("; ")}`);
  return event;
}

function targetKey(scope) {
  return `${scope.surface}\u001f${scope.scenario_id}\u001f${scope.perspective}`;
}

function canonicalize(value) {
  if (Array.isArray(value)) return value.map(canonicalize);
  if (!isObject(value)) return value;
  return Object.fromEntries(Object.keys(value).sort().map((key) => [key, canonicalize(value[key])]));
}

export function canonicalEvaluationFocus(projection) {
  return `${JSON.stringify(canonicalize(projection))}\n`;
}

export function projectEvaluationFocus(events, options) {
  if (!Array.isArray(events)) throw new TypeError("events must be an array");
  if (!isObject(options)) throw new TypeError("projection options must be an object");
  const { subjectStateHash, now, allowedTargets, maxProjectionBytes, categoryQuotas = {} } = options;
  if (!DIGEST.test(subjectStateHash ?? "")) throw new TypeError("subjectStateHash must be a sha256 digest");
  if (!date(now)) throw new TypeError("now must be an ISO 8601 date-time");
  if (!Array.isArray(allowedTargets) || allowedTargets.some((scope) => {
    const errors = [];
    scopeErrors(scope, "allowedTargets[]", errors);
    return errors.length > 0;
  })) throw new TypeError("allowedTargets must contain valid scopes");
  if (!Number.isInteger(maxProjectionBytes) || maxProjectionBytes < 1)
    throw new TypeError("maxProjectionBytes must be a positive integer");
  if (!isObject(categoryQuotas) || Object.values(categoryQuotas).some((value) => !Number.isInteger(value) || value < 0))
    throw new TypeError("categoryQuotas must map categories to non-negative integers");

  const allowed = new Set(allowedTargets.map(targetKey));
  const eventIds = new Set();
  const proposals = new Map();
  const promotions = new Map();
  const promotedProposals = new Set();
  const retired = new Set();
  for (const [index, event] of events.entries()) {
    assertEvaluationFocusEvent(event, `events[${index}]`);
    if (eventIds.has(event.event_id)) throw new Error(`duplicate event_id: ${event.event_id}`);
    eventIds.add(event.event_id);
    if (event.subject_state_hash !== subjectStateHash)
      throw new Error(`stale subject_state_hash on event: ${event.event_id}`);
    if (event.scope && !allowed.has(targetKey(event.scope)))
      throw new Error(`unknown scoped target on event: ${event.event_id}`);
    if (event.event_kind === "proposal") {
      if (proposals.has(event.proposal_id)) throw new Error(`duplicate proposal_id: ${event.proposal_id}`);
      proposals.set(event.proposal_id, event);
    } else if (event.event_kind === "promotion") {
      const proposal = proposals.get(event.proposal_id);
      if (!proposal) throw new Error(`promotion references unknown proposal: ${event.proposal_id}`);
      if (targetKey(proposal.scope) !== targetKey(event.scope))
        throw new Error(`promotion scope conflicts with proposal: ${event.promotion_id}`);
      if (promotions.has(event.promotion_id)) throw new Error(`duplicate promotion_id: ${event.promotion_id}`);
      if (promotedProposals.has(event.proposal_id))
        throw new Error(`proposal has conflicting promotions: ${event.proposal_id}`);
      promotions.set(event.promotion_id, event);
      promotedProposals.add(event.proposal_id);
    } else {
      if (!promotions.has(event.promotion_id))
        throw new Error(`retirement references unknown promotion: ${event.promotion_id}`);
      if (retired.has(event.promotion_id))
        throw new Error(`duplicate retirement for promotion: ${event.promotion_id}`);
      retired.add(event.promotion_id);
    }
  }

  const nowMs = Date.parse(now);
  const categoryCounts = new Map();
  const items = [];
  for (const promotion of [...promotions.values()].sort((left, right) =>
    left.activates_at.localeCompare(right.activates_at) || left.promotion_id.localeCompare(right.promotion_id))) {
    if (retired.has(promotion.promotion_id)) continue;
    if (nowMs < Date.parse(promotion.activates_at)) continue;
    if (nowMs >= Date.parse(promotion.expires_at))
      throw new Error(`active focus projection encountered expired promotion: ${promotion.promotion_id}`);
    const proposal = proposals.get(promotion.proposal_id);
    const count = categoryCounts.get(proposal.category) ?? 0;
    const quota = Object.hasOwn(categoryQuotas, proposal.category) ? categoryQuotas[proposal.category] : Infinity;
    if (count >= quota) throw new Error(`category quota exceeded: ${proposal.category}`);
    categoryCounts.set(proposal.category, count + 1);
    items.push({
      promotion_id: promotion.promotion_id,
      proposal_id: proposal.proposal_id,
      scope: { ...promotion.scope },
      category: proposal.category,
      question: proposal.question,
      task_clause: proposal.task_clause,
      evidence_refs: [...proposal.evidence_refs].sort(),
      source_run_id: proposal.source_run_id,
      activates_at: promotion.activates_at,
      expires_at: promotion.expires_at,
      max_prompt_bytes: promotion.max_prompt_bytes,
      retest_obligation: promotion.retest_obligation,
    });
  }
  const projection = { schema_version: 1, subject_state_hash: subjectStateHash, projected_at: now, items };
  const bytes = Buffer.byteLength(canonicalEvaluationFocus(projection), "utf8");
  const itemBytes = items.reduce((sum, item) => sum + Buffer.byteLength(JSON.stringify(canonicalize(item)), "utf8"), 0);
  if (items.some((item) =>
    Buffer.byteLength(JSON.stringify(canonicalize(item)), "utf8") > item.max_prompt_bytes))
    throw new Error("promoted focus exceeds its max_prompt_bytes");
  if (itemBytes > maxProjectionBytes || bytes > maxProjectionBytes)
    throw new Error(`evaluation focus projection exceeds ${maxProjectionBytes} bytes`);
  return projection;
}
