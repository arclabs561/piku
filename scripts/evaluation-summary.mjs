import { readdir, readFile } from "node:fs/promises";
import path from "node:path";

const root = path.resolve(process.argv[2] || "target/live-ledger");
let files = [];
try {
  files = (await readdir(root))
    .filter((name) => name.endsWith(".jsonl"))
    .map((name) => path.join(root, name));
} catch (error) {
  if (error.code !== "ENOENT") throw error;
}

const rows = [];
for (const file of files) {
  for (const [index, line] of (await readFile(file, "utf8")).split("\n").entries()) {
    if (!line.trim()) continue;
    try {
      rows.push(JSON.parse(line));
    } catch {
      throw new Error(`${file}:${index + 1}: invalid JSONL row`);
    }
  }
}

const countBy = (field) =>
  Object.fromEntries(
    [...new Set(rows.map((row) => row[field] ?? "legacy"))]
      .sort()
      .map((value) => [value, rows.filter((row) => (row[field] ?? "legacy") === value).length]),
  );
const verdictKey = (row) =>
  !Object.hasOwn(row, "product_verdict")
    ? "legacy"
    : row.product_verdict === null
      ? "unjudged"
      : row.product_verdict;
const productVerdicts = Object.fromEntries(
  [...new Set(rows.map(verdictKey))]
    .sort()
    .map((value) => [value, rows.filter((row) => verdictKey(row) === value).length]),
);
const priorityRank = { high: 0, medium: 1, low: 2 };
const rawFollowups = rows
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
      runs: new Set(rows.map((row) => row.run_id).filter(Boolean)).size,
      records: rows.length,
      by_surface: countBy("surface"),
      by_status: countBy("run_status"),
      by_scenario: countBy("scenario_id"),
      by_subject_version: countBy("subject_version"),
      by_subject_revision: countBy("subject_revision"),
      by_evaluator_version: countBy("evaluator_version"),
      product_verdicts: productVerdicts,
      followup_count: followups.length,
      raw_followup_count: rawFollowups.length,
      followups,
    },
    null,
    2,
  ),
);
