# Roadmap: live dogfood quality loop

Status: proposal
Scope: live LLM dogfood, deterministic coverage, trace evidence, and the Codex Online comparison
Date: 2026-06-20

Grounded in:

- ADR-0004: deterministic agent loop coverage
- ADR-0005: live LLM matrix
- `docs/trace-backed-dogfood-design.md`
- `.claude/reports/research-2026-06-20-codex-online-comparison.md`

Review trigger:

- after the first 3 manual live matrix runs
- before adding any scheduled live model spend
- before adding a user-facing random or "top model" mode

## Where we are

Already done:

- PR-blocking agent coverage uses scripted providers for deterministic loop behavior.
- Runtime e2e tests already cover read-edit-verify and multi-file rename.
- Dogfood tests assert real trace JSONL instead of parsing model prose.
- The generated-tests dogfood now compiles and runs the generated test binary.
- A manual live matrix is committed in `d94cddd` with an owner gate and GitHub environment approval.
- ADR-0005 is tracked even though new ADR files are hidden by the global gitignore.

Not done yet:

- The live matrix has not run through GitHub with real environment secrets.
- There is no dogfood result ledger yet.
- There is no formal path from repeated live failures to deterministic PR-blocking tests.

Current drift:

- `.claude/reports/status.md` has stale findings. For example, clippy `-D warnings` is now wired into `scripts/ci.sh`.
- New ADR files under `docs/adr/` are hidden by `/Users/arc/.gitignore_global`.

## Phase 0: checkpoint the live-matrix branch

Status: done in `d94cddd`.

Goal: make the current branch reviewable before adding more ideas.

Work:

- Run `cargo fmt`.
- Keep `live-random` and the workflow matrix aligned.
- Force-add `docs/adr/0005-live-llm-matrix.md`, or add a repo-local ignore override if that is the preferred fix.
- Verify shell syntax, Rust test compilation, and whitespace.

Gate:

- `cargo fmt --check`
- `bash -n scripts/ci.sh`
- `cargo test -p piku --test llm_e2e --no-run`
- `cargo test -p piku --test dogfood --no-run`
- `git status --ignored` confirms the ADR is not silently lost.

## Phase 1: land the protected manual live matrix

Goal: get live dogfood without automatic spend or public-trigger risk.

Work:

- Commit and push the live matrix PR.
- Confirm the `live-llm` GitHub environment has required reviewer approval and provider keys as environment secrets.
- Run one manual `llm_e2e` dispatch.
- Do not add `pull_request`, `push`, or `schedule` triggers.

Gate:

- The workflow can only run by `workflow_dispatch`.
- The job is gated to `github.actor == 'arclabs561'`.
- The environment requires approval.
- Missing optional provider secrets skip only that provider row.

## Phase 2: add a dogfood ledger

Status: implemented after ADR-0006.

Goal: make model quality differences visible over time.

Smallest useful version:

- Live test helpers append one JSONL row per completed scenario when `PIKU_LIVE_LEDGER` is set.
- Fields: date, suite, provider, model, result, failure class, trace path, input tokens, output tokens, duration if available.
- No pricing table yet. Token counts are enough for the first version.

Decision:

- ADR-0006: live dogfood ledger.

Options:

- Post-run trace discovery script. Rejected because traces live under per-test temp config dirs.
- Test harness writes ledger rows directly. Chosen because the helper knows the trace path.
- GitHub artifact-only reporting. Rejected because local dogfood should produce the same rows.

Gate:

- One manual `llm_e2e` run and one manual `dogfood` run produce comparable rows.
- The ledger does not parse model prose.

## Phase 3: promote repeated live failures into deterministic tests

Status: policy accepted in ADR-0007.

Goal: live dogfood should improve the normal PR gate, not become a parallel universe.

Work:

- Use `docs/live-failure-promotions.md` as the promotion ledger.
- Classify failures as provider outage, provider behavior, piku parsing, piku tool loop, prompt weakness, test harness bug, or product decision.
- For each repeated live failure, either:
  - add a scripted-provider runtime e2e test,
  - add a parser or trace test,
  - quarantine the provider/model with a short note,
  - or mark it as product behavior that needs a design decision.

Gate:

- At least one real live finding is converted into a no-secret test or explicitly classified as provider-only.

## Phase 4: decide whether model selection belongs in the product

Goal: avoid turning a dogfood tool into product behavior without evidence.

Recommendation for now:

- Keep random model selection dogfood-only.
- Do not add `piku --random-top-model` or similar until the ledger shows repeated model differences that users benefit from.

Recommended ADR fork:

- ADR-0008: user-facing model selection.

Options:

- Keep randomness only in `scripts/ci.sh live-random`. Recommended now.
- Add a user-facing random/top model mode. Needs evidence and a threat model.
- Add a provider policy config table. Bigger surface, only worth it if users need repeatable provider routing.

Gate:

- At least 3 manual matrix runs show a stable pattern that product behavior could help.

## Later: hosted control plane

Codex Online's useful shape is hosted execution, GitHub review, worktrees, setup automation, and background tasks. That is not piku's best next lane.

Recommendation:

- Do not chase a hosted control plane now.
- Keep piku focused on local, provider-agnostic, trace-backed agent work.
- If hosted work becomes serious, write a separate ADR for the boundary before code.

## Next action

Do Phase 1 next: push the protected live matrix and verify the manual workflow can only run through the intended owner and environment gates.
