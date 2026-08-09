# Design: evidence-addressed review claims

Status: draft

## Problem

The terminal playground currently stores primary and recursive LLM reviews as
opaque prose plus a run-level grounding boolean. A structurally valid review can
therefore describe an allegation that Piku's user-agent never made, cite an
unrelated real turn, or be globally rejected without retaining which claim and
which evidence relation failed. The engineering handoff still derives
hypotheses from unclassified user-agent bugs.

This is not a request to make LLM review authoritative. ADR 0009 remains the
precedence rule: an executable deterministic predicate decides only the property
it directly measures. The problem is to preserve the lower-tier review evidence
well enough to audit, reproduce, and safely abstain.

## Context

The evidence hierarchy already separates workspace acceptance, deterministic
facts, terminal/session evidence, and model review. The missing representation
is between a source allegation and its review: a stable claim identity, concrete
evidence addresses, and a disposition history.

W3C PROV models entities, activities, and responsible agents separately. This
maps directly to an allegation entity, the review activity that assessed it, and
the model/provider configuration that produced the assessment. Assurance-case
work adds a needed distinction: counterevidence can rebut a claim, undermine an
evidence item, or undercut an inference. These are not interchangeable reasons
to erase text. LLM-as-a-judge research further shows order, prompt, and model
dependence, so an LLM verdict is provisional evidence rather than a correctness
oracle. Mature result formats make unavailable and malformed outcomes explicit
instead of encoding them as empty lists or failures.

## Options considered

### Keep review prose plus stronger run-level booleans

This preserves the current compact schema but cannot identify the allegation a
recursive observer contested. It also makes the final handoff depend on an
unclassified bag of source bugs. Rejected because it cannot meet roadmap
priority 2's claim-level gate.

### Stable claims with mutable final text

Each reviewer could edit a common claim record. This is mechanically simple but
destroys chronology: a later correction overwrites the original assertion and
cannot distinguish a changed scope from an error. Rejected because it weakens
auditability exactly where evidence is uncertain.

### Append-only claim records with typed evidence and dispositions

The producer creates immutable source claims; primary and recursive reviewers
append attestations that reference those IDs and evidence catalog entries. A
deterministic validator checks the record against the immutable catalog before
it may affect a handoff. Chosen because it keeps model opinions useful while
preserving their proof boundary.

## Chosen approach

Freeze an evidence catalog before either review starts. It contains producer
local source-claim IDs such as `user-bug-3-1`, turn IDs, and later trace,
transcript, workspace-snapshot, and deterministic-predicate IDs. A source claim
contains its producer, schema version, source turn, normalized allegation, and
the evidence refs that caused it to be raised. Its identity is never derived
from free-form description text or a line location.

The primary judge returns an attestation for a known source claim only:
`claim_id`, `evidence_refs`, verdict, rationale, and producer metadata. The
recursive observer may append an attestation that targets a known primary claim
only. It must name whether it rebuts the claim, undermines a cited evidence item,
or undercuts the inference. It cannot modify deterministic findings or make a
new product allegation.

The validator rejects the whole model-review record when any claimed ID is
empty, duplicate, unknown, unsupported by the frozen catalog, or uses an
unknown disposition/evidence kind. A malformed or unavailable reviewer yields a
separate availability record and changes no earlier disposition. This avoids
silently accepting a partially fabricated review. Missing evidence, conflicting
valid attestations, and order-instability are explicit `inconclusive` or
`abstained` outcomes, not `supported` or `retracted` by default.

The handoff is a deterministic projection, sorted by stable claim ID. A
retracted or abstained model claim cannot retain an actionable recommendation.
Deterministic scenario failures remain in their existing namespace and retain
their ADR 0009 precedence regardless of recursive review output.

## Record shape

The first schema needs only these concepts:

- `SourceClaim`: `id`, `schema_version`, `producer`, `source_turn`,
  `description`, and immutable `evidence_refs`.
- `ReviewAttestation`: `target_claim_id`, `producer`, `model`, `rubric_hash`,
  `candidate_order`, `evidence_refs`, `position`, and `rationale`.
- `EvidenceRef`: typed `turn`, `trace`, `transcript`, `workspace_snapshot`, or
  `deterministic_predicate` address; availability is `complete`, `unavailable`,
  `malformed`, `not_supported`, or `not_applicable`.
- `Disposition`: `proposed`, `supported`, `retracted`, `inconclusive`,
  `abstained`, or `superseded`; a counter-attestation separately records target
  kind `claim`, `evidence`, or `inference`.

This is intentionally a run-local ledger. Cross-run correlation, if later
needed, uses an optional versioned fingerprint and explicit baseline run ID;
it is not inferred from matching prose.

## Non-goals

- A general W3C PROV graph or RDF export. The playground needs an inspectable
  JSONL contract first, not a new interoperability surface.
- A confidence score or learned automatic disposition threshold. There is no
  representative adjudicated claim corpus or stated loss function to calibrate.
- Allowing a recursive reviewer to overturn an executable workspace predicate.
  That would contradict ADR 0009's evidence precedence.
- Cross-run deduplication or baseline suppression in the first change. Those
  require a stable comparison policy and should not be hidden behind a hash.
- Treating an unavailable reviewer as approval, rejection, or product failure.

## Implementation plan

1. Build and freeze the source-claim/evidence catalog before primary review.
   Add pure validation tests for empty, duplicate, unknown, and uncited IDs.
   This is reversible within the ignored playground harness.
2. Replace the primary review's free-form bug array with typed attestations and
   persist validated claims in the JSONL review record. Invalid records carry a
   reason and do not mutate handoff inputs.
3. Give recursive review only the primary IDs and catalog; accept claim-scoped
   counter-attestations, preserving its current observer prose as annotation.
4. Derive hypothesis/actionability from final dispositions, while retaining
   deterministic and scenario paths unchanged. Test byte-stable ordering.
5. Add controlled order-swap/repeat runs only after the schema exists. Record
   instability as abstention and inspect coverage; do not auto-tune a threshold.

## Decision gates

- If a claim can be traced to more than one semantic source after a correction,
  introduce explicit claim revisions linked by `supersedes`; do not reuse its
  ID for a changed predicate.
- If a legitimate run needs an evidence kind outside the initial catalog, add a
  typed variant and validator test before permitting it in model output.
- If an observer needs to overturn deterministic evidence, stop and revise ADR
  0009 rather than adding an exception here.
- If a labeled adjudication corpus and explicit error/coverage target emerge,
  reconsider calibrated selective disposition; until then retain deterministic
  abstention rules.

## Sources

- W3C, [PROV-O](https://www.w3.org/TR/prov-o/): entities, activities, agents,
  and derivation chains.
- Goodenough, Klein, and Weinstock, [Eliminative Argumentation: A Basis for
  Assurance](https://www.sei.cmu.edu/documents/1248/2015_005_001_434813.pdf),
  sections 2.4 and 3.5: rebutting, undermining, and undercutting defeaters.
- Zheng et al., [Judging LLM-as-a-Judge](https://proceedings.neurips.cc/paper_files/paper/2023/file/91f18a1287b398d378ef22505bf41832-Paper-Datasets_and_Benchmarks.pdf),
  sections 3–4: order sensitivity and limits of judge agreement.
- [SARIF 2.1.0 result objects](https://docs.oasis-open.org/sarif/sarif/v2.1.0/sarif-v2.1.0.html): stable result identity, versioned fingerprints, and baseline lifecycle.
- [OpenSSF Scorecard results](https://github.com/ossf/scorecard/blob/main/checker/check_result.go)
  and [findings](https://github.com/ossf/scorecard/blob/main/finding/finding.go):
  typed outcomes and explicit inconclusive/error states.
- Geifman and El-Yaniv, [Selective Classification for Deep Neural
  Networks](https://papers.neurips.cc/paper_files/paper/2017/file/4a8423d5e91fda00bb7e46540e2b0cf1-Paper.pdf): abstention has a measurable coverage tradeoff.

---
Decided: 2026-08-04 | Session: Codex 019fcd16
