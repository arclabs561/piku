# Design: computed context pipelines

Status: proposed

## Problem

Piku currently assembles context independently in the CLI/TUI runtime, web
chat, canvas-change agents, terminal playground, and Playwright evaluators.
Each path formats strings directly, chooses its own truncation unit, and loses
some source-level provenance. Values called "context" range from trusted
instructions to git status, persistent model memory, hook output, selected
cards, page HTML, DOM observations, judge reports, tool schemas, and runtime
capabilities.

A string-variable abstraction would make those paths look uniform without
making them safer or more reproducible. Some useful context is static. Some is
computed from current state. Some requires an arbitrary function, retrieval
step, or whole processing pipeline. Some should remain available as a tool and
never enter the prompt eagerly. Piku needs one representation that preserves
those distinctions.

## Context

The immutable evaluation prompt manifest records exact assets, role contracts,
tools, limits, and effective configuration. The operator-steering design keeps
annotations, proposals, and promoted focus separate. The durable investigation
design keeps raw evidence and projections distinct. A computed-context layer
should connect these decisions without becoming a second memory system or a
general plugin framework.

The live inventory found several concrete problems:

- TUI system context combines operator instructions, up to roughly 75 KB of
  persistent Markdown memory, and arbitrary SessionStart hook output at one
  instruction tier.
- Context selection budgets session messages before adding system text and tool
  schemas, so the stated reserve does not bound total model input.
- Web notebook attachments flatten typed cards into a truncated string and
  lose per-source lineage.
- TUI evaluators ask transcript-dependent questions during each turn without
  supplying a bounded transcript projection.
- Web evaluators already use the desired split in part: stable policy is in a
  template, operational values are computed, and large live observations are
  obtained through Playwright tools.

## Non-goals

- Do not build a broad third-party provider marketplace. Mature remote service
  integrations belong behind narrow tools, direct APIs, or MCP where justified.
- Do not make arbitrary producer code safe merely by giving it a schema.
- Do not inject every available value into every prompt.
- Do not promise deterministic replay when external APIs, models, clocks, or
  repository state are refreshed.
- Do not let a context producer alter tool authority, evaluator criteria, or
  operator promotion state.

## Options considered

### String callbacks

Each variable is a named callback returning text. This is simple but erases
trust, output destination, source identity, cost, and replay behavior. It also
encourages eager injection. Rejected.

### Treat every source as a tool

The model could fetch all context on demand. This controls prompt growth but
makes stable identity and safety policy model-optional, mixes read context with
mutation capabilities, and adds tool-selection failure to basic setup. Rejected
as the universal mechanism; retained for large, volatile, or task-specific
data.

### One mutable context/state object

Agents and middleware could share a mutable map. This is flexible but hides
which component changed the experiment and makes cross-agent contamination
easy. Rejected as the authority; typed state transitions may be one output
plane.

### Typed resolver pipeline with explicit output planes

A harness-selected resolver executes under a declared capability profile and
returns typed context items. Selection and rendering happen later. Every
resolution is traced and can be replayed from captured output. Chosen.

## Chosen approach

### Resolver contract

A `ContextResolver` is a versioned, host-registered producer, not a string
template:

```text
resolve(request, runtime) -> ContextResolution
```

It may be synchronous or asynchronous and may compose deterministic functions,
parsers, ranking/compression stages, retrieval, or registered read-only tools.
Only reviewed first-party deterministic functions run in-process. Arbitrary
functions and pipelines run in a bounded subprocess with an empty secret
environment, projected filesystem, network denied by default, and explicit
time, memory, concurrency, fan-out, and output limits. `runtime` exposes only
the intersection of the caller grant, producer manifest, and resolver policy.
Composition can only reduce that set. Secrets and service clients may exist in
runtime-local state, but are not model context and cannot be serialized into a
resolution.

The request names the run, role, turn/checkpoint, task clause, requested output
plane, source references, byte/token budget, deadline, freshness policy, and
replay mode. It contains references or digests rather than ambient access to
the whole workspace.

### Context item

Each successful producer returns bounded values and execution receipts. The
host resolver constructs one or more `ContextItem`s from those values and its
reviewed registry entry:

```text
id, resolver_id, resolver_version, output_plane, media_type,
source_refs, source_digests, trust, freshness, sensitivity,
priority, payload_ref|inline_payload, byte_size, token_estimate,
output_digest, created_at, expires_at, warnings
```

The initial output planes are:

- `instruction`: small reviewed invariants and explicitly promoted operator
  focus. Never produced from untrusted retrieval or model output.
- `message`: bounded model-visible evidence or current-turn attachments.
- `tool`: a capability descriptor made available for on-demand use; it does not
  execute during rendering.
- `state`: host-owned structured state for later deterministic stages.
- `artifact`: durable raw or derived material referenced rather than injected.

Plane, trust, sensitivity, freshness, and authoritative provenance are assigned
by the host, never accepted from producer output. A producer cannot invent a
tool schema: the tool plane only references a host-registered capability. State
accepts typed deterministic reducer events and remains model-invisible.

Presentation is downstream. A renderer decides how selected items become exact
model-visible bytes and records its own version and output digest. Typed items
alone do not improve model behavior; the rendered prompt remains the experiment.

### Trust and authority

Trust describes how content may be interpreted, not what its producer can do:

- `control`: versioned harness policy and output schema;
- `operator_instruction`: reviewed repository/user instruction or promoted
  focus;
- `host_fact`: deterministic host observation such as revision or viewport;
- `untrusted_evidence`: files, page content, tool output, hooks, memory, model
  reports, and retrieved text;
- `derived_evidence`: summaries, rankings, or compression derived from named
  sources.

Only control and explicit operator-instruction items can enter the instruction
plane. Retrieval, model, tool, and arbitrary-pipeline outputs may enter only
message or artifact planes. Derived text retains the lowest trust of its inputs.
Tool permissions are enforced outside prompts and are never expanded by
returned content. A digest proves byte identity, not truth or authority.

An execution graph is rejected if untrusted content can reach a model that also
has both high-sensitivity data access and general egress or consequential tools.
When arbitrary untrusted text is required, it is inspected in a quarantined
context or the receiving model's tools are reduced to guarded local reads.

### Selection and budgeting

Resolvers return candidates; a deterministic selector chooses what is visible.
The budget covers the complete request: control instructions, selected items,
history, and active tool schemas. Selection records included and excluded IDs,
reason, size, token estimate, priority, and any truncation or compression.

The default ladder is:

1. always visible: small identity, safety, authority, and output contracts;
2. computed snapshot: current environment, git/workspace state, task state, and
   explicitly attached small objects;
3. promoted focus: bounded operator-selected goals and retest obligations;
4. on demand: files, large DOM/console/network results, memory search, detailed
   ledgers, broad tool catalogs, and volatile external data;
5. excluded: secrets, unrelated workspace objects, ambient parent-agent memory,
   and historical verdicts without a retest contract.

Compression never destroys the raw source reference. Exact-copy/code tasks may
prefer raw excerpts; query-aware compression is an optional derived item whose
benefit must be measured locally.

### Resolution trace and replay

Every attempt records resolver identity, registry and code digests, sanitized
input references and digests, effective capability profile, start/end time,
cache decision, output IDs/digests/sizes, warnings, errors, and materialized
artifact refs. The run manifest records the selector and renderer versions plus
the digest of the final model-visible context. Logs and provenance use
structural allowlists; secret-bearing values and free-form errors do not enter
caches, traces, or provider requests without an explicit egress grant.

Cache keys include schema version, canonical typed arguments, producer identity
and code digest, source snapshots, output plane, trust policy, sensitivity,
tool profile, and resolver version. Entries are immutable and
content-addressed. Artifact references are run-relative, containment-checked,
symlink-safe, and verified for media type, size, and digest before use.

Replay is explicit:

- `exact`: consume captured resolution outputs and forbid resolver execution;
- `refresh`: execute again, compare digests, and emit a drift record;
- `fork`: inherit the exact parent snapshot to a checkpoint and resolve only
  explicitly refreshed inputs as a new branch. A broader provider, egress
  policy, sensitivity grant, or tool profile requires a new operator grant.

A resumed run never silently switches among these modes. External/model calls
after a checkpoint remain nondeterministic unless exact outputs were captured.

### Operator feedback

Annotations remain review-only records. A judge may propose a resolver, source,
or focus change, but cannot activate it. Operator promotion may select bounded,
expiring advisory questions or a reviewed resolver profile, never factual
verdicts. Promotion does not grant new filesystem, network, secret, provider,
or mutation authority. Lineage cycles from the run or claim being evaluated
are rejected, and blinded runs without promoted focus remain part of the eval
design.

## Initial providers

The first implementation should extract existing deterministic seams rather
than invent service plugins:

1. environment/revision/worktree status;
2. PIKU instruction files with per-file lineage;
3. bounded selected web-card attachments;
4. TUI evaluator terminal plus transcript projection;
5. evaluator operational variables already used by Playwright prompts.

Markdown memory, hook pipelines, broad tool catalogs, DOM inspection, and
external retrieval remain existing/on-demand mechanisms until the basic trace
and budget model is proven.

## Implementation plan

1. Define surface-neutral `ContextItem`, `ResolutionTrace`, trust, output-plane,
   and replay schemas. Add fixtures shared by Rust and JavaScript.
2. Extend prompt manifests with resolved-item and final-render attestations;
   include negative provenance for intentionally excluded ambient sources.
3. Extract one pure host-fact resolver in Rust and one typed card-attachment
   resolver in web code. Preserve current rendered output initially.
4. Centralize full-input budgeting and make included/excluded items inspectable.
5. Add exact/refresh tests with a deliberately nondeterministic resolver.
6. Add an internal composition API for registered read-only pipelines. Delay
   any agent-facing resolver proposal tool until real operator use exists.
7. Run paired tasks against the prior assembly path. Inspect success,
   unsupported claims, omitted decisive evidence, latency, cost, and operator
   comprehension rather than reporting only aggregate scores.

## Decision gates

- A resolver output cannot enter the instruction plane unless all inputs have
  instruction authority and the resolver is reviewed for that plane.
- An arbitrary pipeline cannot access filesystem, network, secrets, or tools
  outside its capability profile, regardless of schema or prompt text.
- Nested pipelines cannot increase their caller's authority. Unknown producer,
  plane, trust, source version, capability, or replay dependency fails closed.
- A producer cannot select its output plane, raise trust, author a tool schema,
  or write model-visible durable state.
- An execution graph containing untrusted input, high private-data access, and
  general egress or consequential tools must be rejected.
- Exact replay must execute zero resolvers and reproduce identical selected and
  rendered-context digests. Missing captured content is a named failure.
- Refresh must reveal input/output drift rather than overwrite the captured
  result.
- The selector must account for system, history, context items, and tool schemas
  in one total budget.
- Every compressed/derived item must retain source IDs and raw artifact access.
- Per-node and total time, byte, token, memory, concurrency, retry, and fan-out
  budgets must propagate cancellation and report partial exhaustion explicitly.
- Seeded credential markers must not appear in provider requests, traces,
  caches, artifacts, child context, logs, or errors without a provider-specific
  egress grant.
- If typed assembly does not improve inspectability or reduce drift while
  preserving task outcomes, retain the manifest/trace work and stop before a
  plugin API.
- If tool-backed context adds no benefit over explicit attachments in held-out
  tasks, keep it on demand and do not make it a default resolver stage.

## Evidence

- OpenAI Agents SDK separates arbitrary local run context from what is sent to
  the model and supports dynamic instruction callbacks:
  [context](https://openai.github.io/openai-agents-python/context/) and
  [dynamic instructions](https://openai.github.io/openai-agents-python/agents/#dynamic-instructions).
- LangGraph separates invocation context, mutable state, persistent store,
  messages, and tool runtime, and documents replay re-execution:
  [context](https://docs.langchain.com/oss/python/concepts/context) and
  [time travel](https://docs.langchain.com/oss/python/langgraph/use-time-travel).
- Letta uses a hierarchy of always-visible memory blocks and on-demand archival
  retrieval: [context hierarchy](https://docs.letta.com/guides/core-concepts/memory/context-hierarchy).
- LongLLMLingua provides primary evidence that query-aware selection and
  compression can improve relevance density and cost, with transfer to code
  tasks still requiring local tests:
  [ACL 2024 paper](https://aclanthology.org/2024.acl-long.91.pdf).
- ReAct demonstrates action/observation pipelines as a context-construction
  mechanism: [ICLR paper](https://arxiv.org/abs/2210.03629).
- Indirect prompt injection shows why retrieved and tool-produced content must
  remain untrusted regardless of how it enters context:
  [Greshake et al.](https://arxiv.org/abs/2302.12173).
- Benchmark-contamination work motivates held-out fixtures and producer/prompt
  provenance: [Oren et al., ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/file/46e624c244cff669223d488defd4e835-Paper-Conference.pdf).
