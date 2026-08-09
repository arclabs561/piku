import assert from "node:assert/strict";
import { constants } from "node:fs";
import { mkdtemp, readFile, rm, stat, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { spawn, spawnSync } from "node:child_process";
import { test } from "node:test";

const SCRIPT = path.resolve("scripts/evaluation-focus-cli.mjs");
const HASH = `sha256:${"a".repeat(64)}`;
const OTHER_HASH = `sha256:${"b".repeat(64)}`;
const SCOPE = { surface: "web", scenario_id: "operator-journey", perspective: "recovery" };

function proposal(suffix = "1", overrides = {}) {
  return {
    schema_version: 1,
    event_id: `event-proposal-${suffix}`,
    event_kind: "proposal",
    recorded_at: "2026-08-09T10:00:00.000Z",
    actor: { kind: "judge", id: "recovery-judge" },
    subject_state_hash: HASH,
    proposal_id: `proposal-${suffix}`,
    source_run_id: `run-${suffix}`,
    scope: SCOPE,
    evidence_refs: [`run-${suffix}:screenshot:1`],
    question: `Can recovery ${suffix} be verified?`,
    category: "recovery",
    suggested_expires_at: "2026-08-12T00:00:00.000Z",
    task_clause: `Resume interrupted workspace ${suffix}`,
    ...overrides,
  };
}

function promotion(suffix = "1", overrides = {}) {
  return {
    schema_version: 1,
    event_id: `event-promotion-${suffix}`,
    event_kind: "promotion",
    recorded_at: "2026-08-09T11:00:00.000Z",
    actor: { kind: "operator", id: "local-operator" },
    subject_state_hash: HASH,
    promotion_id: `promotion-${suffix}`,
    proposal_id: `proposal-${suffix}`,
    scope: SCOPE,
    activates_at: "2026-08-09T11:30:00.000Z",
    expires_at: "2026-08-10T00:00:00.000Z",
    max_prompt_bytes: 1024,
    retest_obligation: `run-${suffix}:result:obligation:o1`,
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
    reason: "Covered by a deterministic test",
    ...overrides,
  };
}

function run(args) {
  return spawnSync(process.execPath, [SCRIPT, ...args], { encoding: "utf8" });
}

async function fixture(t) {
  const directory = await mkdtemp(path.join(tmpdir(), "piku-focus-cli-"));
  t.after(() => rm(directory, { recursive: true }));
  return { directory, ledger: path.join(directory, "events.jsonl") };
}

test("help states explicit paths and the provenance authority boundary", () => {
  const result = run(["--help"]);
  assert.equal(result.status, 0, result.stderr);
  assert.match(result.stdout, /caller-explicit/);
  assert.match(result.stdout, /provenance, not authentication/);
  assert.match(result.stdout, /mode 0600/);
});

test("proposal-only append round trips through inspect with private creation mode", async (t) => {
  const { directory, ledger } = await fixture(t);
  const source = path.join(directory, "proposals.jsonl");
  await writeFile(source, `${JSON.stringify(proposal())}\n`);
  const appended = run(["append", ledger, source]);
  assert.equal(appended.status, 0, appended.stderr);
  assert.equal(JSON.parse(appended.stdout).appended, 1);
  assert.equal((await stat(ledger)).mode & 0o777, 0o600);
  const inspected = run(["inspect", ledger]);
  assert.equal(inspected.status, 0, inspected.stderr);
  assert.deepEqual(JSON.parse(inspected.stdout).events, [proposal()]);
});

test("append rejects authority-bearing imports and preserves the ledger", async (t) => {
  const { directory, ledger } = await fixture(t);
  await writeFile(ledger, `${JSON.stringify(proposal())}\n`);
  const source = path.join(directory, "promotion.json");
  await writeFile(source, JSON.stringify(promotion()));
  const result = run(["append", ledger, source]);
  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /proposal events only/);
  assert.equal(await readFile(ledger, "utf8"), `${JSON.stringify(proposal())}\n`);
});

test("promote and retire restrict actor kinds and preserve lineage", async (t) => {
  const { ledger } = await fixture(t);
  await writeFile(ledger, `${JSON.stringify(proposal())}\n`, { mode: 0o600 });
  const denied = run(["promote", ledger, "--event", JSON.stringify(promotion("1", {
    actor: { kind: "judge", id: "judge" },
  }))]);
  assert.notEqual(denied.status, 0);
  assert.match(denied.stderr, /operator or reviewer/);
  const orphan = run(["promote", ledger, "--event", JSON.stringify(promotion("2"))]);
  assert.notEqual(orphan.status, 0);
  assert.match(orphan.stderr, /unknown proposal/);
  assert.equal(run(["promote", ledger, "--event", JSON.stringify(promotion())]).status, 0);
  const deniedRetire = run(["retire", ledger, "--event", JSON.stringify(retirement({
    actor: { kind: "harness", id: "harness" },
  }))]);
  assert.notEqual(deniedRetire.status, 0);
  assert.equal(run(["retire", ledger, "--event", JSON.stringify(retirement())]).status, 0);
  assert.equal((await readFile(ledger, "utf8")).trim().split("\n").length, 3);
});

test("inspect reports path and line for schema, staleness, and lineage errors", async (t) => {
  const { ledger } = await fixture(t);
  await writeFile(ledger, [
    JSON.stringify(proposal()),
    JSON.stringify(proposal("2", { subject_state_hash: OTHER_HASH })),
  ].join("\n") + "\n");
  let result = run(["inspect", ledger]);
  assert.notEqual(result.status, 0);
  assert.match(result.stderr, new RegExp(`${ledger.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}:2`));
  assert.match(result.stderr, /stale subject_state_hash/);
  await writeFile(ledger, `${JSON.stringify(promotion())}\n`);
  result = run(["inspect", ledger]);
  assert.match(result.stderr, /:1: promotion references unknown proposal/);
});

test("project emits canonical stdout and an identical atomic output file", async (t) => {
  const { directory, ledger } = await fixture(t);
  await writeFile(ledger, `${JSON.stringify(proposal())}\n${JSON.stringify(promotion())}\n`);
  const options = JSON.stringify({
    subjectStateHash: HASH,
    now: "2026-08-09T12:00:00.000Z",
    allowedTargets: [SCOPE],
    maxProjectionBytes: 4096,
    categoryQuotas: { recovery: 1 },
  });
  const stdout = run(["project", ledger, "--options", options]);
  assert.equal(stdout.status, 0, stdout.stderr);
  assert.equal(stdout.stdout, `${JSON.stringify(JSON.parse(stdout.stdout))}\n`);
  const output = path.join(directory, "focus.json");
  const written = run(["project", ledger, "--options", options, "--output", output]);
  assert.equal(written.status, 0, written.stderr);
  assert.equal(written.stdout, "");
  assert.equal(await readFile(output, "utf8"), stdout.stdout);
});

test("projection rejects stale state, expiry, and byte and category budgets", async (t) => {
  const { ledger } = await fixture(t);
  await writeFile(ledger, `${JSON.stringify(proposal())}\n${JSON.stringify(promotion())}\n`);
  const base = {
    subjectStateHash: HASH,
    now: "2026-08-09T12:00:00.000Z",
    allowedTargets: [SCOPE],
    maxProjectionBytes: 4096,
    categoryQuotas: { recovery: 1 },
  };
  for (const [options, pattern] of [
    [{ ...base, subjectStateHash: OTHER_HASH }, /stale subject_state_hash/],
    [{ ...base, now: "2026-08-10T00:00:00.000Z" }, /expired promotion/],
    [{ ...base, maxProjectionBytes: 64 }, /projection exceeds/],
    [{ ...base, categoryQuotas: { recovery: 0 } }, /category quota exceeded/],
  ]) {
    const result = run(["project", ledger, "--options", JSON.stringify(options)]);
    assert.notEqual(result.status, 0);
    assert.match(result.stderr, pattern);
  }
});

test("an existing lock fails busy and is never treated as stale", async (t) => {
  const { directory, ledger } = await fixture(t);
  const source = path.join(directory, "proposal.json");
  await writeFile(source, JSON.stringify(proposal()));
  await writeFile(`${ledger}.lock`, "old owner\n", { mode: 0o600 });
  const result = run(["append", ledger, source]);
  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /ledger is busy/);
  assert.equal(await readFile(`${ledger}.lock`, "utf8"), "old owner\n");
});

test("concurrent appenders serialize complete lines without lost writes", async (t) => {
  const { directory, ledger } = await fixture(t);
  const count = 12;
  const sources = [];
  for (let index = 0; index < count; index += 1) {
    const source = path.join(directory, `proposal-${index}.json`);
    await writeFile(source, JSON.stringify(proposal(String(index))));
    sources.push(source);
  }
  const attempt = (source) => new Promise((resolve) => {
    const child = spawn(process.execPath, [SCRIPT, "append", ledger, source], { stdio: ["ignore", "pipe", "pipe"] });
    let stderr = "";
    child.stderr.on("data", (chunk) => { stderr += chunk; });
    child.on("close", (status) => resolve({ status, stderr }));
  });
  const pending = [...sources];
  while (pending.length) {
    const batch = pending.splice(0);
    const results = await Promise.all(batch.map(attempt));
    for (let index = 0; index < results.length; index += 1) {
      if (results[index].status !== 0) {
        assert.match(results[index].stderr, /ledger is busy/);
        pending.push(batch[index]);
      }
    }
  }
  const lines = (await readFile(ledger, "utf8")).trim().split("\n");
  assert.equal(lines.length, count);
  const events = lines.map(JSON.parse);
  assert.equal(new Set(events.map((event) => event.event_id)).size, count);
  await assert.rejects(readFile(`${ledger}.lock`, "utf8"), { code: "ENOENT" });
});
