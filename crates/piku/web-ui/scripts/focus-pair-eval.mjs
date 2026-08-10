import { createHash } from "node:crypto";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { runEvaluation, safeRunId } from "./parallel-agent-eval.mjs";

const scriptsDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(scriptsDir, "../../../..");
const armNames = Object.freeze(["blind", "focused"]);

export function focusPairOrder(pairOrdinal) {
  if (!Number.isSafeInteger(pairOrdinal) || pairOrdinal < 0)
    throw new TypeError("pair ordinal must be a non-negative integer");
  return pairOrdinal % 2 === 0 ? [...armNames] : [...armNames].reverse();
}

export function validatePairId(value) {
  if (typeof value !== "string" || value.length > 128
    || !/^[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*$/.test(value))
    throw new TypeError("pair ID must contain only alphanumeric hyphen-separated components");
  return value;
}

function digest(value) {
  return createHash("sha256").update(JSON.stringify(value)).digest("hex");
}

export function pairedContract(result) {
  const manifest = result?.promptManifestDocument ?? result?.promptManifest?.manifest;
  const roles = manifest?.roles;
  const normalize = (value) => JSON.parse(JSON.stringify(value, (_key, item) => {
    if (typeof item !== "string") return item;
    return item.replaceAll(result?.runDir || "\0", "{{RUN_DIR}}")
      .replaceAll(result?.runId || "\0", "{{RUN_ID}}");
  }));
  return {
    subject_revision: result?.runtime?.subject_revision,
    subject_dirty: result?.runtime?.subject_dirty,
    viewport: result?.runtime?.viewport,
    models: roles?.map(({ role, model }) => ({ role, model })),
    prompts: roles?.map(({ role, prompt_assets }) => ({ role, prompt_assets })),
    tools: roles?.map(({ role, tools }) => ({ role, tools: normalize(tools) })),
    limits: roles?.map(({ role, limits }) => ({ role, limits })),
  };
}

export function validatePairedContracts(blind, focused) {
  const left = pairedContract(blind);
  const right = pairedContract(focused);
  if (left.subject_dirty !== false || right.subject_dirty !== false)
    throw new Error("focus pair requires clean subject trees");
  if (!/^[0-9a-f]{40}$/.test(left.subject_revision || "")
    || left.subject_revision !== right.subject_revision)
    throw new Error("focus pair requires the same exact subject revision");
  for (const field of ["viewport", "models", "prompts", "tools", "limits"]) {
    if (digest(left[field]) !== digest(right[field]))
      throw new Error(`focus pair contract drift: ${field}`);
  }
  return left;
}

function evidenceFor(result) {
  return (result?.results || []).flatMap((entry) => entry.report?.evidence || []);
}

function findingsFor(result) {
  return result?.synthesis?.report?.findings || [];
}

function citedIds(result) {
  const report = result?.synthesis?.report;
  return [...new Set([
    ...(report?.evidence_ids || []),
    ...findingsFor(result).flatMap((finding) => finding.evidence_ids || []),
    ...Object.values(report?.coverage || {}).flatMap((item) => item.evidence_ids || []),
  ])].sort();
}

function armDossier(result, focusQuestions = []) {
  const evidence = evidenceFor(result);
  const findings = findingsFor(result);
  const cited = citedIds(result);
  const reportText = JSON.stringify(result?.synthesis?.report || {});
  return {
    run_id: result?.runId ?? null,
    status: result?.runStatus ?? "inconclusive",
    reproduction_status: result?.synthesis?.report?.verdict ?? "not_assessed",
    fresh_evidence_ids: evidence.map((item) => item.id).filter(Boolean).sort(),
    provenance_evidence_ids: evidence.filter((item) => item.artifact_metadata?.producer_event_id)
      .map((item) => item.id).filter(Boolean).sort(),
    verbatim_focus_question_echo_without_cited_evidence: focusQuestions
      .filter((question) => reportText.includes(question) && cited.length === 0).sort(),
    useful_specificity: findings.map((finding) => ({
      finding_id: finding.id,
      title: finding.title,
      cited_evidence_ids: [...(finding.evidence_ids || [])].sort(),
    })),
    coverage: result?.synthesis?.report?.coverage ?? null,
    effort: {
      explorer_target_calls: result?.runtime?.explorer_target_calls ?? null,
      explorer_hard_max_calls: result?.runtime?.explorer_hard_max_calls ?? null,
      explorer_max_snapshots: result?.runtime?.explorer_max_snapshots ?? null,
      explorer_timeout_ms: result?.runtime?.explorer_timeout_ms ?? null,
    },
    cited_evidence_ids: cited,
  };
}

export function buildEvidenceQualityDossier({ pairId, pairOrdinal, order, blind, focused, focusQuestions = [], confounds = [] }) {
  const blindArm = armDossier(blind);
  const focusedArm = armDossier(focused, focusQuestions);
  const blindTitles = new Set(blindArm.useful_specificity.map((item) => item.title));
  const focusedTitles = new Set(focusedArm.useful_specificity.map((item) => item.title));
  const incomplete = [blind, focused].some((result) => result?.runStatus !== "completed");
  return {
    schema_version: 1,
    pair_id: pairId,
    pair_ordinal: pairOrdinal,
    status: incomplete ? "inconclusive" : "completed",
    comparison_kind: "evidence_quality_dossier",
    order,
    arms: { blind: blindArm, focused: focusedArm },
    novel_collateral_findings: {
      blind: blindArm.useful_specificity.filter((item) => !focusedTitles.has(item.title)).map((item) => item.finding_id),
      focused: focusedArm.useful_specificity.filter((item) => !blindTitles.has(item.title)).map((item) => item.finding_id),
    },
    coverage_disagreement: JSON.stringify(blindArm.coverage) === JSON.stringify(focusedArm.coverage) ? [] : ["arm coverage differs"],
    effort_confounds: [...confounds],
    promotion_or_retirement: "operator_only",
  };
}

async function immutableJson(filePath, value) {
  await writeFile(filePath, `${JSON.stringify(value, null, 2)}\n`, { encoding: "utf8", flag: "wx", mode: 0o600 });
}

export async function runFocusPair({
  pairOrdinal,
  focusEventsPath,
  environment = process.env,
  evaluate = runEvaluation,
  pairId = safeRunId(),
  outputRoot = path.join(repoRoot, ".artifacts", "playwright-agent", "focus-pairs"),
} = {}) {
  const order = focusPairOrder(pairOrdinal);
  validatePairId(pairId);
  if (typeof focusEventsPath !== "string" || focusEventsPath.length === 0)
    throw new TypeError("focused arm requires an explicit focus events path");
  const focusBytes = await readFile(path.resolve(focusEventsPath));
  const events = focusBytes.toString("utf8")
    .split("\n").filter(Boolean).map((line) => JSON.parse(line));
  const promotedProposalIds = new Set(events
    .filter((event) => event.event_kind === "promotion")
    .map((event) => event.proposal_id));
  const focusQuestions = events
    .filter((event) => event.event_kind === "proposal" && promotedProposalIds.has(event.proposal_id))
    .map((event) => event.question);
  await mkdir(outputRoot, { recursive: true });
  const pairDir = path.join(outputRoot, pairId);
  await mkdir(pairDir, { recursive: false });
  const focusSnapshotFile = "focus-events.jsonl";
  const focusSnapshotPath = path.join(pairDir, focusSnapshotFile);
  await writeFile(focusSnapshotPath, focusBytes, { flag: "wx", mode: 0o600 });
  const results = {};
  const armFailures = [];
  for (const arm of order) {
    const armEnvironment = { ...environment, PIKU_EVAL_RUN_ID: `${pairId}-${arm}` };
    delete armEnvironment.PIKU_EVAL_RESUME_RUN_ID;
    if (arm === "focused") armEnvironment.PIKU_EVAL_FOCUS_EVENTS = focusSnapshotPath;
    else delete armEnvironment.PIKU_EVAL_FOCUS_EVENTS;
    try { results[arm] = await evaluate({ environment: armEnvironment, argv: [] }); }
    catch (error) {
      armFailures.push(`${arm} arm failed: ${error.message}`);
      results[arm] = {
        runId: `${pairId}-${arm}`, runDir: null, runStatus: "inconclusive",
        runtime: null, results: [], synthesis: null,
      };
    }
  }
  const confounds = [...armFailures];
  let contract = null;
  try { contract = validatePairedContracts(results.blind, results.focused); }
  catch (error) { confounds.push(error.message); }
  const focusSourceSha256 = createHash("sha256").update(focusBytes).digest("hex");
  const manifest = {
    schema_version: 1, pair_id: pairId, pair_ordinal: pairOrdinal, order,
    arms: Object.fromEntries(armNames.map((arm) => [arm, {
      run_id: results[arm]?.runId ?? `${pairId}-${arm}`,
      artifact_directory: results[arm]?.runDir ?? null,
      focus: arm === "focused" ? {
        source_sha256: focusSourceSha256,
        snapshot_path: focusSnapshotFile,
      } : null,
    }])),
    shared_contract: contract,
    automatic_focus_mutation: false,
  };
  const report = buildEvidenceQualityDossier({ pairId, pairOrdinal, order, ...results, focusQuestions, confounds });
  if (confounds.length) report.status = "inconclusive";
  await immutableJson(path.join(pairDir, "manifest.json"), manifest);
  await immutableJson(path.join(pairDir, "report.json"), report);
  return { pairId, pairDir, manifest, report, results };
}

export async function main() {
  const pairOrdinal = Number(process.env.PIKU_EVAL_PAIR_ORDINAL);
  const result = await runFocusPair({
    pairOrdinal,
    focusEventsPath: process.env.PIKU_EVAL_FOCUS_EVENTS,
    pairId: process.env.PIKU_EVAL_PAIR_ID || safeRunId(),
  });
  if (result.report.status !== "completed") process.exitCode = 1;
  else console.error(`Focus pair evaluation complete: ${result.pairDir}`);
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url))
  await main();
