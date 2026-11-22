# Design: shared cross-surface evaluation

Status: implemented first slice; projection and four-perspective expansion remain

## Problem

Piku has two evaluation systems that test related product behavior but cannot
currently learn from one another. The CLI harness has deterministic scenarios,
PTY personas, separate user and judge models, failure classes, and append-only
JSONL ledgers. The web harness has deterministic Playwright tests plus one
ephemeral Codex process that both explores the live workspace and judges it.
Its reports, screenshots, and tool events remain isolated under `.artifacts/`.

This separation hides whether a failure belongs to the runtime, the CLI
projection, or the web projection. The single web evaluator also anchors on its
own narrative and has too much responsibility. The first thesis-oriented web
run completed its browser work but timed out after 47 calls before producing a
report. That is a harness failure, not product evidence.

## Context

The existing live-ledger decision establishes JSONL as bounded evidence input,
not a score table or database. The agentic playground already separates the
subject, simulated user, primary judge, and recursive review. Deterministic
acceptance and trace facts outrank model prose. The web harness adds evidence
types the CLI cannot produce: screenshots, DOM state, browser actions, network
requests, and spatial layout measurements.

Piku surfaces and tabs are useful presentation and isolation primitives. They
do not replace process, browser-context, request, or artifact isolation.

## Non-goals

- Do not merge CLI and browser drivers. They produce different authoritative
  evidence and should remain replaceable adapters.
- Do not use majority voting among identical prompts. Deliberate perspective
  diversity and cited evidence matter more than vote count.
- Do not expose prior conclusions to explorers. Shared history must not turn
  retesting into confirmation.
- Do not make live-model evaluation a normal CI gate. Deterministic promoted
  regressions remain the release authority.
- Do not add a database. Append-only JSONL plus immutable run artifacts is
  sufficient until measured query or concurrency pressure proves otherwise.

## Options considered

### Keep independent CLI and web harnesses

This preserves local simplicity but duplicates failure classification, loses
cross-surface comparisons, and leaves web findings outside the established
promotion loop. Rejected.

### One large cross-surface judge

One agent could drive both the PTY and browser and return a unified opinion.
The timed-out thesis run is direct evidence against this shape: browser-only
execution already exceeded a useful single-agent responsibility budget.
Rejected.

### Shared evidence envelope with parallel explorers and fresh synthesis

Surface-specific explorers produce bounded evidence packets. A fresh-context
judge synthesizes them after execution. A shared ledger records run outcomes,
finding lifecycle, and perspective coverage. Chosen.

## Chosen approach

Introduce a surface-neutral evaluation envelope with these stable fields:

```text
schema_version, run_id, scenario_id, revision, surface, perspective,
subject_model, explorer_model, judge_model, task_contract, run_status,
failure_class, evidence, dimensions, findings, artifact_refs, duration_ms
```

`run_status` describes evidence production: `completed`, `product_failure`,
`harness_failure`, `infrastructure_failure`, `timeout`, or `inconclusive`.
Product quality is separate: each capability dimension records a score, result,
and evidence IDs. A completed run may still conclude that the product thesis is
only partial or unsupported.

The intended full parallel panel has four explorers:

1. `first_use`: comprehension and discoverability.
2. `coding_trace`: actions, files, diffs, verification, and provenance.
3. `recovery`: stale state, cancellation, errors, reload, and resume.
4. `authority`: context boundaries, mutation authority, terminal, and safety.

Each explorer owns a unique process, browser context or PTY, artifact directory,
and Piku surface named `qa-<run>-<perspective>`. Explorers receive the same task
contract and their own lens. They receive uncovered-perspective prompts and
explicit retest obligations from the ledger, but not historical verdicts.

Deterministic mechanics move out of explorer prompts into surface adapters.
The web adapter creates surfaces, performs stable menu/card operations, records
screenshots and DOM/network facts, and exposes a bounded action vocabulary. The
CLI adapter records PTY viewports, traces, workspace predicates, and tool effects.
Codex chooses what to investigate and interprets evidence; it does not spend
reasoning calls rediscovering brittle selectors.

After explorers finish, one fresh synthesis judge receives their raw evidence
packets, the full finding ledger, and no explorer chain of thought. It must cite
evidence IDs for every claim, distinguish disagreement from failure, and return
one cross-surface report. A later adversarial reviewer may challenge the rubric,
but is not required for the first implementation.

The web UI may render the panel as tabs inside one evaluation workspace and a
separate synthesis surface. This is observability, not execution isolation.

The implemented first slice runs `coding_trace` and `recovery` concurrently,
then starts a fresh synthesis process only after both evidence packets validate.
It emits the shared envelope, structured followups, raw JSONL events, screenshots,
and compact live progress. `first_use`, `authority`, deterministic browser setup,
and the evaluation-workspace projection remain deliberate later gates.

## Ledger and finding lifecycle

Extend the existing `target/live-ledger/*.jsonl` family rather than creating a
second web-only ledger. Large screenshots and event streams stay in immutable
run directories; ledger rows contain stable references and hashes.

Findings receive stable IDs and move through:

```text
observed → reproduced → fixed → regression_covered → independently_retested
```

The ledger also records uncovered perspectives and unverified task clauses.
Historical allegations never become current product failures without fresh
reproduction. Concurrent explorers append isolated run records; only the
synthesis step appends lifecycle transitions.

## Tradeoffs

- Parallel runs increase provider cost and can contend on rate limits. The
  orchestrator needs per-role budgets and must classify contention separately.
- A shared schema adds versioning work across Rust and JavaScript. This is less
  costly than silently drifting failure semantics.
- Scripted mechanics reduce exploratory freedom. Explorers retain direct
  Playwright or PTY access for hypothesis tests, but stable setup should be
  deterministic.
- Fresh synthesis improves independence but cannot eliminate correlated model
  priors. Periodic model and prompt diversity remains necessary.

## Implementation plan

1. Define and test the versioned envelope and evidence-ID contract in a small
   surface-neutral module. Reversible until both adapters emit it.
2. Map existing CLI live-ledger rows into the envelope without deleting the old
   fields. Reversible compatibility layer.
3. Make the web runner emit a ledger row even on timeout or invalid reports;
   retain its current artifacts as evidence references. Reversible.
4. Extract deterministic browser setup and measurements into a web adapter;
   cut the explorer journey to one perspective and a strict time budget.
5. Run two parallel perspectives plus one fresh synthesis judge. Expand to four
   only after cleanup, contention, and evidence validation pass.
6. Add the evaluation-workspace projection to Piku after the headless control
   plane is reliable. Presentation is intentionally last.

## Decision gates

- A forced explorer timeout must produce a `timeout` ledger row and no product
  failure.
- Two parallel explorers must use distinct surfaces, browser contexts, request
  IDs, and artifact directories, then leave no processes or temporary surfaces.
- The synthesis judge must reject an unknown evidence ID and a conclusion that
  contradicts deterministic predicates.
- The same scenario run through CLI and web must retain one scenario ID while
  keeping projection-specific evidence types.
- If two perspectives cannot complete within the cost and time of the current
  single run, reduce task scope before adding more judges.
- If a deterministic mutation of progress, provenance, stale rerun state, or
  authority labels is not detected, revise the rubric before trusting scores.

## Open questions

- Which model diversity provides useful disagreement rather than cost-only
  variation?
- Should synthesis disagreement block finding promotion or create a dedicated
  `contested` state?
- Which representative coding task can exercise both projections without
  permitting the web evaluator to mutate the real repository?
