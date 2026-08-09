# Grok: editable context as a durable investigation

## The argument in one sentence

The missing product is not a prettier terminal notebook. It is a local,
inspectable workbench where a person can construct agent context, attach
evidence, fork an earlier assumption, review concrete changes, and preserve a
human conclusion as an ordinary durable artifact.

## How the conversation gets there

The conversation begins with a surface question: what gives a terminal
Jupyter-like Markdown, math, cells, images, and plots? Euporie is the closest
terminal-native answer; cmux plus marimo or Quarto is the more polished answer.
But neither resolves the deeper problem.

The notebook critique is about semantics, not rendering:

- visible order and actual execution order can diverge;
- `.ipynb` is an awkward review and merge format;
- code, output, and environment drift apart;
- exploration and publication are forced into one object;
- cells become accidental software boundaries;
- navigation and refactoring degrade at project scale.

Current agent harnesses reproduce the same failure in a different form. Their
transcripts are fluid but ephemeral. Context selection, summarization, tool
state, approvals, and resulting changes are often implicit. The final diff is
reviewable, but the assumptions and evidence that produced it are hard to
recover or revise.

This reframes the original question:

> Can a long, branching human-agent interaction become a small set of durable,
> inspectable, regenerable objects without losing exploratory fluidity?

## Product thesis

The conversation arrives at two compatible formulations:

> A local lab book that turns coding-agent sessions into executable,
> reviewable, durable understanding.

> A native investigation companion that makes agent context, evidence, and
> changes inspectable, and lets you fork reasoning without forking your
> understanding.

The novel object is not a notebook containing chat cells. It is an editable,
executable construction of context.

## Core interaction

```text
write a question or hypothesis
  -> attach exact files, selections, and earlier conclusions
  -> run an existing agent
  -> inspect what it read, did, and changed
  -> attach commands, tests, plots, or citations as evidence
  -> write a human conclusion
  -> edit an earlier assumption and fork
  -> compare the downstream result
  -> export what remains useful
```

The interface should distinguish the user's interpretation from the agent's
proposal. It should keep repository files, scripts, tests, and notes canonical;
private traces and UI metadata may be disposable.

## Small universal model

The broad conceptual vocabulary is:

```text
Thread  Step  Reference  Run  Result  Branch  Decision
```

Only `Thread` and `Step` need to dominate the interface. The rest support
provenance, replay, comparison, and staleness. Exposing every concept as a new
cell type or mode would recreate the complexity the product is meant to tame.

The refined MVP uses one plain Markdown document and four block kinds:

```text
prose
shell
agent
file reference
```

Outputs live separately but remain attached to the operation that produced
them. Agent blocks show their exact context manifest. Source changes remain
ordinary Git patches.

## Why editable history matters

Changing an earlier prompt is not merely “rerun the whole chat.” The design
needs explicit operations:

- rerun this step;
- rerun from here;
- recompute affected steps;
- fork here;
- refresh against the current project.

That makes the user's mental model tangible. An assumption can be revised and
its consequences compared instead of silently overwriting the old path.
Automatic dependency inference should remain conservative: inferred staleness
is useful, but a false claim of reproducibility is worse than an explicit
manual checkpoint.

## Human understanding is a first-class result

Reading and writing manually remain important because their friction forces
prediction, selection, compression, and model revision. The tool should
automate boilerplate and mechanical work while preserving epistemic contact
with core abstractions, unfamiliar algorithms, concurrency, security,
performance, and mathematical reasoning.

A good learning loop is:

```text
read -> predict -> run -> modify -> explain -> compare with an agent
```

The product succeeds when the person can continue without reconstructing the
agent transcript, not when the transcript merely sounds clear.

## Domain shape

Different work has different canonical artifacts and evidence:

- software: symbols, commits, tests, diffs;
- data science: datasets, variables, kernels, transformations, plots;
- research: papers, passages, claims, citations;
- operations: services, logs, time ranges, queries, deployments;
- writing and design: sources, drafts, decisions, rendered artifacts.

The shared layer supplies context, provenance, branching, comparison,
staleness, and human interpretation. A domain adapter supplies what can be
referenced, executed, changed, replayed, and treated as evidence.

Software investigation is the right first wedge. Git already provides durable
identity, diffs, history, and review semantics; tests provide evidence; agent
use is common; and the design can be tested on real projects immediately.

## Implementation direction

For a Mac-first prototype, the conversation recommends:

```text
SwiftUI application frame
+ AppKit / TextKit 2 document surface
+ GRDB for disposable local metadata
+ Process / Pipe adapters for Codex and Claude CLIs
+ Git command integration
+ WKWebView only for rich rendered output
```

Start as a companion to Xcode, Zed, and the terminal. Open files and ranges in
the user's editor. Do not initially build an embedded terminal, source editor,
plot system, visual DAG, multi-agent orchestrator, or universal notebook
runtime. Consider `libghostty` only after the context/fork/review interaction
has proved useful.

## Decision-relevant legibility

The captured tail adds an important control rule. Raw activity overwhelms;
polished summaries can conceal correctness. The system should route attention
by consequence:

```text
routine activity      -> quiet
important evidence    -> summarized
model-changing event  -> salient
consequential choice  -> interruptive
full trace            -> available
```

It must also surface implementation choices whose alternatives change safety,
behavior, compatibility, performance, operability, or future flexibility.
“Detail on demand” is insufficient because the user may not know which hidden
detail matters. The agent should identify why a detail is decision-relevant and
link the compact claim to the diff, code path, reproduction, test, and affected
consumers.

This remains a trust boundary: the agent doing the work also classifies what
deserves attention. High-risk decisions need conservative escalation or an
independent check.

## Main design traps

- Too many first-class block types, modes, and task objects.
- A proprietary document format that traps the user's work.
- Capturing every terminal event instead of promoting useful evidence.
- Treating ownership labels as a substitute for visible action contracts.
- Claiming a complete reactive dependency graph for arbitrary agent work.
- Building cmux-quality shell infrastructure before testing the interaction.
- Parallel agents creating more supervisory and review debt than throughput.
- Invisible transcript compression and late, overwhelming review.

The refined answer is deliberately subtractive: plain files, visible context,
normal patches, selected evidence, explicit forks, and bounded review.

## What is still unsettled

- The exact persistent representation of steps, references, and branches.
- Replay semantics for stochastic, expensive, or side-effecting operations.
- How context snapshots survive file and repository changes.
- How staleness is inferred without false confidence.
- Which agent interfaces provide sufficiently structured events.
- How sensitive context and raw traces are retained or discarded.
- How the product measures understanding rather than apparent fluency.
- Which decision boundaries require independent verification.

## Source boundary

This synthesis is derived from the complete sanitized capture of 24 contiguous
rendered turns, numbered 1 through 24, from the exact canonical conversation
URL. The original full-page capture contained only a virtualized five-turn
window. External claims and citations inside the conversation were not
independently verified in this comprehension pass.
