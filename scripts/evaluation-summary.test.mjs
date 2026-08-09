import assert from "node:assert/strict";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { spawnSync } from "node:child_process";
import { test } from "node:test";

test("cross-surface summary preserves prioritized judge followups", async () => {
  const directory = await mkdtemp(path.join(tmpdir(), "piku-eval-summary-"));
  try {
    const rows = [
      {
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
      },
      {
        run_id: "web-1",
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
      },
      {
        run_id: "web-1",
        surface: "web",
        run_status: "completed",
        scenario_id: "trace",
        product_verdict: null,
        followups: [],
      },
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
    assert.equal(summary.records, 3);
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
