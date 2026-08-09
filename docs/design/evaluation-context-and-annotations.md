# Design: operator-steered evaluation context

Status: proposed

## Problem

Piku's evaluators already produce useful findings, follow-ups, screenshots, and
run ledgers. They do not yet provide a safe way for an operator to add nuance,
correct a claim, or say what a future evaluation should examine. Repeating that
feedback in prompts loses provenance. Automatically feeding judge prose back
into later prompts creates a self-confirming loop and silently turns model
opinion into policy.

The evaluation systems also need reproducible prompt context. A result should
identify the exact reviewed configuration, prompt templates, tool profile, and
promoted operator focus that shaped it. A model name and repository revision are
not enough to reconstruct an evaluation.

## Context

The shared cross-surface design uses immutable run artifacts and a canonical
evaluation envelope. The review-claim design gives allegations stable IDs and
typed evidence addresses. The live-failure policy promotes repeated evidence
into deterministic coverage. Operator feedback should compose with those
systems rather than create another memory store or free-form instruction layer.

There are three different kinds of information that must remain distinct:

1. evidence and findings produced by a run;
2. operator annotations about that evidence;
3. reviewed focus that may influence a later run.

Only the third belongs in a judge prompt, and only after an explicit operator
action. An annotation is evidence about an interpretation, not an instruction.

## Non-goals

- Do not let a judge edit its own rubric, tool permissions, or active context.
- Do not treat an annotation as a product fact or deterministic test result.
- Do not create a general project-memory system or hidden personalization layer.
- Do not inject the entire historical ledger into each run.
- Do not add a web editor before the append-only record and CLI workflow are
  proven inspectable and reversible.

## Options considered

### Automatically inject recurring findings

The harness could summarize recent follow-ups and add them to every evaluator
prompt. This preserves momentum but rewards repeated phrasing, anchors new
explorers on old conclusions, and makes it difficult to distinguish discovery
from confirmation. Rejected.

### Let agents edit evaluation configuration

A judge could maintain a configuration or memory file directly. This is
convenient but conflates proposal and authority. Prompt injection or a mistaken
finding could permanently change later evaluations. Rejected.

### Append annotations and proposals, then promote explicitly

Judges may append evidence-linked proposals. Operators may append annotations
and explicitly promote a bounded proposal into active focus. A generated
projection supplies only promoted, unexpired focus to a run. Chosen.

### Derive all focus deterministically

The harness could select focus only from failure classes, coverage gaps, and
test results. This is useful for deterministic retest obligations but cannot
represent operator taste, disagreement, or qualitative design goals. Retained
as one input, not the whole system.

## Chosen approach

Use an append-only event log as the authority and a generated `focus.json` as a
disposable projection. Four layers keep authorship and authority visible.

### 1. Reviewed operator configuration

Repository configuration selects evaluator defaults: enabled perspectives,
named tool profiles, context budgets, expiration defaults, and allowed focus
categories. It grants no filesystem or network authority. Changes are normal
reviewed source changes and are identified by content hash.

### 2. Immutable run manifest

Every run snapshots the effective configuration before execution. The manifest
records repository revision and dirty state, evaluator and subject identities,
prompt-template hashes, response-schema hashes, tool-profile hashes, promoted
focus IDs, viewport and color scheme where relevant, budgets, and harness
versions. Resumed stages retain the same manifest and add an attempt record;
they do not silently render a new prompt.

### 3. Append-only annotations and proposals

An `EvaluationAnnotation` records:

```text
id, recorded_at, author_kind, author_id, auth_method, scope,
target_refs, target_hashes, stance, body, correction, supersedes
```

`scope` is one of `artifact`, `evidence`, `finding`, `stage`, `run`, or
`project`. `stance` is `agree`, `disagree`, `correct`, or `context`. Target
hashes prevent an annotation from appearing to apply after its evidence was
replaced. Records are immutable; a later record may supersede an earlier one
from the same authority class. A judge cannot supersede an operator record.

A `FocusProposal` is judge- or harness-authored and must include its source run,
model or deterministic producer, evidence references, proposed focus text,
category, suggested expiration, and the task clause it would help test. It has
no prompt authority by itself.

### 4. Explicit promotion

An operator creates a `FocusPromotion` referencing a proposal or authored focus.
It supplies a bounded scope, activation time, expiration condition, maximum
prompt budget, and optional retest obligation. Promotion never changes tools or
mutation authority. Retirement is another append-only event.

The harness validates the log and compiles active promotions into a sorted,
bounded `focus.json`. Runs consume only this projection plus deterministic
scenario context. They never consume arbitrary annotations, proposal prose, or
the whole ledger. The manifest records the projection hash and included IDs.

## Prompt and tool boundary

Prompt rendering separates trusted control text from untrusted product,
artifact, and historical content with explicit labeled sections. Promoted focus
is advisory: it may direct attention but cannot change the evidence hierarchy,
success criteria, output schema, or tool permissions. A judge must still cite
fresh run evidence. Prior findings are retest hypotheses, never current facts.

The first interface is operator-only CLI:

```text
piku eval annotate ...
piku eval propose ...
piku eval promote ...
piku eval retire ...
piku eval focus
```

If agent tooling is later justified, expose only
`propose_evaluation_focus`. It appends an evidence-linked proposal and cannot
promote, update, delete, select tools, or write configuration. The operator CLI
remains the sole promotion authority.

## Threats and controls

- **Self-confirmation:** explorers do not receive verdict history; promoted
  focus asks a question and requires fresh evidence.
- **Goodhart pressure:** focus cannot alter scoring rules or deterministic
  predicates, and expires by default.
- **Prompt injection:** annotation and evidence bodies are untrusted data;
  structural schema and tool restrictions remain outside the prompt.
- **Authority escalation:** promotion selects attention only, never tools,
  credentials, network access, or mutation scope.
- **Staleness:** target hashes, revision bounds, and expiration make old context
  visibly inapplicable.
- **Projection tampering:** the append-only log is authoritative; generated
  focus is validated and its hash is captured in the run manifest.
- **False independence:** manifests record judge roles and model identities so
  a same-model countercheck is not presented as independent review.
- **Context growth:** category quotas and a byte/token budget reject an
  oversized projection rather than silently truncate it.

## Implementation plan

1. Add a shared run-manifest schema and hash the effective prompts, schemas,
   config, tool profiles, and focus projection. Use it in CLI and web runners.
2. Add versioned annotation, proposal, promotion, and retirement records with a
   pure validator and byte-stable projection tests.
3. Add the external CLI workflow. Support annotations and authored proposals,
   but keep prompt injection disabled.
4. Enable bounded promoted-focus injection for one evaluator perspective. Run
   paired evaluations with and without focus and inspect evidence quality, not
   only scores.
5. Expand to other perspectives only if focus improves relevant discovery
   without increasing unsupported historical claims.
6. Consider the proposal-only agent tool after the CLI ledger has real use. Add
   a web projection later as a view over the same records, not a second store.

## Decision gates

- A judge-authored proposal must not affect a later prompt without a distinct
  operator promotion event.
- Replaying a run from its manifest must produce the same rendered prompt and
  tool contract or fail with a named unavailable dependency.
- Unknown targets, changed target hashes, invalid supersession, expired focus,
  and oversized projections must fail closed.
- A promoted historical concern must still be reported as inconclusive when
  fresh evidence does not reproduce it.
- If focused runs merely echo the promoted wording or reduce novel findings,
  disable injection and retain annotations as review-only context.
- If operators rarely promote proposals, stop before building agent tools or a
  web editor.

## Relationship to existing designs

- [Shared cross-surface evaluation](shared-cross-surface-evaluation.md) owns run
  envelopes, surface adapters, and synthesis.
- [Evidence-addressed review claims](review-claim-lifecycle.md) owns claim IDs,
  evidence references, attestations, and dispositions.
- [Live failure promotion](../live-failure-promotion-design.md) owns promotion
  from repeated live failures into deterministic engineering work.

This design owns only reproducible evaluation context and the path by which
operator judgment can annotate evidence or deliberately steer future attention.
