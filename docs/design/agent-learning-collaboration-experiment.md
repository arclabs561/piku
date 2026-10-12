# Design: evaluating collaboration and experience memory

Status: experiment proposal

## Current state

Piku already has attempt/outcome memory, parent/child run evidence, observed tool
effects, verification records, and an opt-in live dogfood ledger. These support
an experiment; they do not show that more agents or accumulated memory improve
held-out coding tasks. No live experiment is claimed by this design.

## Problem

Choose a useful built-in collaboration or learning mechanism using measured
task outcomes rather than message volume, stored memories, or an agent's claim
of success. Same-task retries, cross-task transfer, and coordination overhead
must be measured separately.

## Options considered

1. **Frozen cross-task experience memory.** Acquire experiences, extract
   conditional lessons, freeze the pool, then evaluate disjoint tasks. This
   offers a clear leakage boundary and strong retrieval-only baseline.
2. **Online memory updates after every task.** This fits continual use, but task
   order, outcome-label noise, and contamination make the first experiment
   harder to interpret.
3. **Reusable executable skills or deeper agent hierarchies.** These may help
   repeated procedures or distributed information. They add execution,
   authority, and coordination contracts, so test them only when the workload
   requires those mechanisms.

## Chosen approach

Start with option 1 as an isolated, opt-in experiment. Retain option 2 for a
later retention/order test and option 3 as an alternative if observed failures
are procedural or require independently held information. This is a proposed
sequence; it does not select a production learning architecture.

Use the existing live ledger, run records, and executable workspace predicates.
The [review contract](task-review-contract.md) keeps observed checks distinct
from human acceptance. Repair record integrity and task-join behavior before
relying on their evidence in a capability comparison.

## Learning comparison

Use one fixed model, tools, starting workspace and task set. Split acquisition
and evaluation by repository/task family or generator template; exclude hidden
tests, expected patches, duplicates and evaluation outcomes from the memory pool.

Compare:

- **No memory:** ordinary execution within the shared resource ceiling.
- **Retrieved experience:** one relevant successful acquisition experience,
  without a distilled lesson.
- **Conditional lesson:** a lesson drawn from observed failure/success contrasts,
  with supporting attempt references and applicability conditions.

Freeze all memory before held-out evaluation. Record source IDs, pool/split
hashes, injected context, retrieval exclusions, and the executed outcome checks.
An explanation of a failure remains an inference even when the source attempt
has deterministic evidence. Keep same-task reflection/retry results separate
from the primary cross-task transfer endpoint.

Set the same declared resource ceilings for all arms and report realized use.
Count injected context, retrieval, extraction, judging, and acquisition costs;
report deployment-only and acquisition-amortized views separately. Token,
wall-time and dollar budgets are different measures. Pin a pricing policy before
claiming a dollar-cost comparison; fewer steps alone do not imply lower cost.

Score hidden acceptance independently of the model. Report paired outcomes,
uncertainty, regressions, missing verification, timeouts, tool work and resource
use. Declare the practical gain and uncertainty criterion before execution;
choose retrieval-only if lesson extraction does not improve the comparison.

## Collaboration comparison

Choose a concrete task with independent evidence-finding or checking work.
Compare a single agent with simple parent/child delegation under the same total
resource ceilings, including coordinator and summary work. Keep file ownership,
authority, required checks and final parent review explicit.

Measure acceptance and missed defects alongside coordination cost, time and
handoff loss. Introduce a deeper hierarchy only after an observed information or
coordination bottleneck explains why the simple baseline fails. More messages,
parallel tasks or reviewer agreement do not establish improved correctness.

## First experiment scope

The first comparison uses existing record and ledger mechanisms, isolated
fixtures, declared input context and reset per-task state. It adds neither a
production memory-consolidation subsystem nor automatic task acceptance.
Scripted providers can test pool exclusion, provenance, budgets and recording;
they cannot demonstrate learned capability from a real model.

## Decision gates

- Evidence gate: every scored outcome has an executed acceptance predicate and
  an inspectable workspace/run identity; unavailable evidence is inconclusive.
- Leakage gate: held-out outcomes and expected solutions never enter acquisition
  memory or the model-visible context.
- Budget gate: a gain funded by extra total work is reported as a quality/cost
  tradeoff, not matched-resource superiority.
- Promotion gate: a practical held-out gain survives repeated seeds and negative
  or irrelevant-memory controls without unacceptable regressions. Otherwise keep
  the mechanism experimental or prefer the simpler baseline.
- Online gate: only after frozen transfer works, test score-before-update,
  task-order sensitivity, stale/conflicting memories and held-out retention.

## Primary sources and limits

- [ExpeL](https://arxiv.org/html/2308.10144), sections 4–5: separate acquisition,
  insight extraction and unseen single-attempt evaluation; retrieved experience
  is a serious component baseline. Its experiments do not establish this
  harness's performance at exactly matched total cost.
- [Reflexion](https://arxiv.org/html/2303.11366), evaluation and limitations:
  same-task feedback/retries and dependence on adequate verification.
- [Voyager](https://arxiv.org/html/2305.16291), skill-library and unseen-task
  evaluation: executable reuse is an alternative, with narrow environment and
  generalization coverage.
- [ReasoningBank](https://arxiv.org/html/2509.25140), methods, cost analysis and
  limitations: successful/failed experience distillation and coding evidence;
  model-judged labels and extra inference work need separate accounting.

These are primary-source-supported candidates, not an exhaustive literature
survey or a universal ranking. Workload selection, feasibility calibration,
resource policy, power and live execution remain experiment entry gates.
