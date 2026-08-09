import assert from "node:assert/strict";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { spawnSync } from "node:child_process";
import { test } from "node:test";

function envelope(overrides) {
  return {
    schema_version: 1,
    run_id: "run",
    record_kind: "stage",
    stage_id: "result",
    scenario_id: "trace",
    surface: "cli",
    perspective: "test",
    task_contract: "trace",
    run_status: "completed",
    failure_class: "none",
    product_verdict: null,
    finding_count: null,
    evidence_ids: [],
    artifact_refs: [],
    followups: [],
    duration_ms: 1,
    ...overrides,
  };
}

function amendment(target, action, overrides = {}) {
  const eventId = `${target.run_id}-${action}-${overrides.event_suffix ?? "1"}`;
  return envelope({
    ...target,
    record_kind: "amendment",
    stage_id: `amendment:${eventId}`,
    run_status: "completed",
    failure_class: "none",
    product_verdict: null,
    finding_count: null,
    evidence_ids: [],
    artifact_refs: ["audit.json"],
    followups: [],
    duration_ms: 0,
    target_run_id: target.run_id,
    target_stage_id: target.stage_id,
    event_id: eventId,
    recorded_at: "2026-08-09T00:00:00.000Z",
    contract_version: "piku-evaluation-amendment-v1",
    amendment_action: action,
    reason_code: "causal_audit",
    amendment_scope: { evidence_ids: [], finding_refs: [], verdict: true },
    basis_refs: ["audit.json"],
    basis_hashes: [`sha256:${"a".repeat(64)}`],
    replacement: null,
    actor: "test-auditor",
    tool_version: "test-tool/1",
    ...Object.fromEntries(Object.entries(overrides).filter(([key]) => key !== "event_suffix")),
  });
}

test("cross-surface summary preserves prioritized judge followups", async () => {
  const directory = await mkdtemp(path.join(tmpdir(), "piku-eval-summary-"));
  try {
    const rows = [
      envelope({
        run_id: "cli-1",
        surface: "cli",
        run_status: "completed",
        scenario_id: "trace",
        product_verdict: null,
        followups: [
          {
            kind: "idea",
            priority: "low",
            title: "Try another model",
            rationale: "Check correlated priors",
            perspective: "adversarial",
            evidence_ids: [],
          },
          {
            kind: "retest",
            priority: "medium",
            title: "Run provenance again",
            rationale: "Same obligation from synthesis",
            perspective: "coding_trace",
            evidence_ids: ["web-1:timeline"],
          },
        ],
      }),
      envelope({
        run_id: "web-1",
        stage_id: "explorer",
        surface: "web",
        run_status: "product_failure",
        scenario_id: "trace",
        product_verdict: "partial",
        followups: [
          {
            kind: "retest",
            priority: "high",
            title: "Retest provenance",
            rationale: "Timeline evidence was missing",
            perspective: "coding_trace",
            evidence_ids: ["web-1:timeline"],
          },
        ],
      }),
      envelope({
        run_id: "web-1",
        stage_id: "synthesis",
        surface: "web",
        run_status: "completed",
        scenario_id: "trace",
        product_verdict: null,
        followups: [],
      }),
    ];
    await writeFile(
      path.join(directory, "runs.jsonl"),
      rows.map((row) => JSON.stringify(row)).join("\n") + "\n",
    );
    const script = path.resolve("scripts/evaluation-summary.mjs");
    const result = spawnSync(process.execPath, [script, directory], {
      encoding: "utf8",
    });
    assert.equal(result.status, 0, result.stderr);
    const summary = JSON.parse(result.stdout);
    assert.deepEqual(summary.by_surface, { cli: 1, web: 2 });
    assert.equal(summary.runs, 2);
    assert.equal(summary.stages, 3);
    assert.equal(summary.records, 3);
    assert.equal(summary.legacy_records, 0);
    assert.equal(summary.quarantined_records, 0);
    assert.deepEqual(summary.product_verdicts, { partial: 1, unjudged: 2 });
    assert.equal(summary.followup_count, 2);
    assert.equal(summary.raw_followup_count, 3);
    assert.equal(summary.followups[0].title, "Retest provenance");
    assert.equal(summary.followups[0].source_run, "web-1");
    assert.equal(summary.followups[0].occurrences, 2);
    assert.equal(summary.followups[1].title, "Try another model");
    assert.match(await readFile(path.join(directory, "runs.jsonl"), "utf8"), /web-1/);
  } finally {
    await rm(directory, { recursive: true });
  }
});

test("summary separates legacy rows and quarantines invalid versioned rows", async () => {
  const directory = await mkdtemp(path.join(tmpdir(), "piku-eval-quarantine-"));
  try {
    await writeFile(
      path.join(directory, "mixed.jsonl"),
      [
        JSON.stringify(envelope({ run_id: "valid" })),
        JSON.stringify({ run_id: "old-cli", result: "success" }),
        JSON.stringify(envelope({ run_id: "invalid", run_status: "banana" })),
        "{not-json",
        "null",
        JSON.stringify({ ...envelope({ run_id: "missing-stage" }), stage_id: undefined }),
      ].join("\n") + "\n",
    );
    const result = spawnSync(process.execPath, [path.resolve("scripts/evaluation-summary.mjs"), directory], {
      encoding: "utf8",
    });
    assert.equal(result.status, 0, result.stderr);
    const summary = JSON.parse(result.stdout);
    assert.equal(summary.runs, 1);
    assert.equal(summary.records, 1);
    assert.equal(summary.legacy_records, 1);
    assert.equal(summary.quarantined_records, 4);
    assert.match(summary.quarantine[0].source, /mixed\.jsonl:3$/);
    assert.ok(summary.quarantine[0].errors.some((error) => error.includes("run_status")));
    assert.match(summary.quarantine[1].source, /mixed\.jsonl:4$/);
    assert.match(summary.quarantine[2].source, /mixed\.jsonl:5$/);
    assert.match(summary.quarantine[3].source, /mixed\.jsonl:6$/);
    assert.ok(summary.quarantine[3].errors.some((error) => error.includes("stage_id is required")));
  } finally {
    await rm(directory, { recursive: true });
  }
});

test("playground fixture contributes one shared summary while detailed evidence stays lossless", async () => {
  const fixture = path.resolve("scripts/fixtures/playground-envelope");
  const result = spawnSync(process.execPath, [path.resolve("scripts/evaluation-summary.mjs"), fixture], {
    encoding: "utf8",
  });
  assert.equal(result.status, 0, result.stderr);
  const summary = JSON.parse(result.stdout);
  assert.equal(summary.runs, 1);
  assert.equal(summary.stages, 1);
  assert.equal(summary.records, 1);
  assert.deepEqual(summary.by_surface, { tui: 1 });
  assert.deepEqual(summary.by_status, { product_failure: 1 });
  assert.equal(summary.followup_count, 2);
  assert.deepEqual(summary.followups.map((item) => item.kind), ["todo", "retest"]);

  const detailed = await readFile(
    path.join(fixture, "detailed", "agentic-findings", "playground.jsonl"),
    "utf8",
  );
  const records = detailed.trim().split("\n").map((line) => JSON.parse(line));
  assert.deepEqual(records.map((row) => row.kind), ["config", "turn", "improvement_handoff"]);
  assert.equal(records[1].viewport, "full terminal output retained");
  assert.deepEqual(records[2].hypotheses, ["[unreviewed] rerun after fixing output"]);
});

test("append-only amendments derive eligibility without rewriting original verdicts", async () => {
  const directory = await mkdtemp(path.join(tmpdir(), "piku-eval-amendments-"));
  try {
    const invalidated = envelope({ run_id: "invalidated", product_verdict: "supported" });
    const qualified = envelope({ run_id: "qualified", product_verdict: "partial" });
    const superseded = envelope({ run_id: "superseded", product_verdict: "supported" });
    const reinstated = envelope({ run_id: "reinstated", product_verdict: "supported" });
    const rows = [
      invalidated,
      qualified,
      superseded,
      reinstated,
      amendment(invalidated, "invalidate"),
      amendment(qualified, "qualify"),
      amendment(superseded, "supersede", {
        replacement: { product_verdict: "not_supported" },
      }),
      amendment(reinstated, "invalidate", { event_suffix: "1" }),
      amendment(reinstated, "reinstate", { event_suffix: "2" }),
      amendment(envelope({ run_id: "missing" }), "invalidate"),
      amendment(invalidated, "invalidate"),
    ];
    await writeFile(
      path.join(directory, "amendments.jsonl"),
      rows.map((row) => JSON.stringify(row)).join("\n") + "\n",
    );
    const result = spawnSync(process.execPath, [path.resolve("scripts/evaluation-summary.mjs"), directory], {
      encoding: "utf8",
    });
    assert.equal(result.status, 0, result.stderr);
    const summary = JSON.parse(result.stdout);
    assert.equal(summary.records, 11);
    assert.equal(summary.original_records, 4);
    assert.equal(summary.amendment_records, 7);
    assert.equal(summary.runs, 4);
    assert.equal(summary.stages, 4);
    assert.equal(summary.ineligible_verdicts, 1);
    assert.deepEqual(summary.product_verdicts, { not_supported: 1, partial: 1, supported: 1 });
    assert.equal(summary.orphan_amendments.length, 1);
    assert.equal(summary.duplicate_amendments.length, 1);
    assert.equal(summary.duplicate_amendments[0].event_id, "invalidated-invalidate-1");
    const byRun = Object.fromEntries(summary.effective_records.map((record) => [record.run_id, record]));
    assert.equal(byRun.invalidated.disposition.verdict, "invalidate");
    assert.equal(byRun.invalidated.eligibility.verdict, false);
    assert.equal(byRun.invalidated.original_product_verdict, "supported");
    assert.equal(byRun.invalidated.product_verdict, null);
    assert.equal(byRun.qualified.disposition.verdict, "qualified");
    assert.equal(byRun.superseded.disposition.verdict, "superseded_with_replacement");
    assert.equal(byRun.superseded.product_verdict, "not_supported");
    assert.equal(byRun.reinstated.disposition.verdict, "reinstated");
    assert.equal(byRun.reinstated.eligibility.verdict, true);
    assert.deepEqual(
      byRun.reinstated.amendment_history.map((event) => event.action),
      ["invalidate", "reinstate"],
    );
    assert.ok(
      byRun.reinstated.amendment_history[0].ledger_sequence <
        byRun.reinstated.amendment_history[1].ledger_sequence,
    );
  } finally {
    await rm(directory, { recursive: true });
  }
});
