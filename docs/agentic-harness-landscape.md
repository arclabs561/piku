# Agentic harness landscape

Status: research snapshot

Reviewed: 2026-08-03

## Question

What do current agentic coding harnesses implement well, what do users repeatedly
value or reject, and which lessons fit Piku without turning it into a hosted
control plane or a feature-parity project?

## Coverage and limits

The survey covers Codex, Claude Code, OpenCode, Hermes Agent, Pi, Aider, Cline,
Goose, Roo Code, SWE-agent, Qwen Code, Kimi Code, Crush, Antigravity CLI, and
Copilot CLI. The evidence set combines implementation files, tests, official
documentation and changelogs, and high-signal GitHub issues and pull requests.
Issue reactions and comments are popularity signals, not a user census. Praise
is less likely than failure to become an issue. Claude Code's runtime is closed,
and the Antigravity and Copilot repositories do not publish their runtime source,
so their implementation cannot be inferred from product documentation.

The selection is purposive: it represents recurring design tensions, not every
open ticket in each repository. Repository stars and release counts can identify
projects worth inspecting, but do not establish adoption, satisfaction, or
quality. No comparable telemetry or download census exists across this set.

## Summary

The best harnesses converge on a narrow execution core with multiple presentation
surfaces, typed lifecycle state, capability-scoped tools, explicit context
management, and deterministic evidence around tool effects. The most persistent
complaints are not about benchmark intelligence. They are about hidden context
loss, weak or exhausting permission systems, unreliable terminals and edits,
unrecoverable sessions, excessive tool schemas, unsafe background work, and
unclear cost.

Piku should remain a small local harness, but “small” should mean a narrow and
observable core, not missing safety boundaries. Its strongest differentiator is
the trace and workspace-backed quality loop. Its highest-risk gaps are launch-turn
and subagent authority, full-catalog tool loading, and a transcript that hides
some successful details while lacking a focused review surface. The governing
product objective should be verified change per unit of scarce human attention,
not maximum autonomy or minimum reading in isolation.

## What people value

| Theme | Evidence | What users appear to value |
| --- | --- | --- |
| Scoped autonomy | [Aider #3362](https://github.com/Aider-AI/aider/issues/3362), [Cline #11](https://github.com/cline/cline/issues/11), [Goose #2806](https://github.com/aaif-goose/goose/issues/2806) | Let cheap inspection flow while bounding writes, commands, paths, time, and spend. |
| Focused delegation | [Claude subagents](https://code.claude.com/docs/en/sub-agents), [OpenCode task implementation](https://github.com/anomalyco/opencode/blob/dev/packages/opencode/src/tool/task.ts), [Hermes delegation](https://github.com/NousResearch/hermes-agent/blob/main/tools/delegate_tool.py) | Separate context, visible lifecycle, bounded authority, and a result that returns to the parent. |
| Reversible work | [Aider #649](https://github.com/Aider-AI/aider/issues/649), [Codex #11626](https://github.com/openai/codex/issues/11626) | Preview, Git-backed recovery, and clear separation between conversation rewind and workspace restoration. |
| Long-session continuity | [Goose context management](https://github.com/aaif-goose/goose/blob/10df80e409d6aca6636f72c8f4fd7758f263acf0/documentation/docs/guides/sessions/smart-context-management.md), [Cline #4389](https://github.com/cline/cline/issues/4389), [Claude #5996](https://github.com/anthropics/claude-code/issues/5996) | Sessions that degrade visibly and recoverably instead of silently forgetting the goal or failing at the context limit. |
| Programmable governance | [Codex #2109](https://github.com/openai/codex/issues/2109), [Codex hook runtime](https://github.com/openai/codex/blob/main/codex-rs/core/src/hook_runtime.rs), [Claude hooks](https://code.claude.com/docs/en/hooks) | Hooks with stable events, explicit block semantics, context injection, and audit uses. |
| Portable instructions | [Claude #6235](https://github.com/anthropics/claude-code/issues/6235) | Repository guidance that moves between harnesses rather than binding a project to one vendor file. |
| Provider choice | [OpenCode offline discussion](https://github.com/anomalyco/opencode/issues/10416), [Hermes architecture](https://github.com/NousResearch/hermes-agent/blob/main/website/docs/developer-guide/architecture.md) | One workflow across hosted and local models, with an explicit offline contract when requested. |
| Terminal fidelity | [Claude #3648](https://github.com/anthropics/claude-code/issues/3648), [Cline terminal consolidation](https://github.com/cline/cline/issues/4356) | Stable scrollback, observable completion, correct shell integration, and output that does not flood context. |
| Controllable reading | [Pi #4916](https://github.com/earendil-works/pi/issues/4916), [Kimi #2212](https://github.com/MoonshotAI/kimi-code/issues/2212), [Copilot CLI changelog](https://github.com/github/copilot-cli/blob/main/changelog.md) | Stable scrollback, compact defaults, selective expansion, and side questions that do not pollute durable task history. |

## What people reject

- Approval fatigue when every operation prompts but the policy is not expressive
  enough to grant a narrow reusable capability.
- “Workspace-only” claims enforced by prompts or tool names while shell execution
  can still cross the boundary.
- Defaults that silently regain network access, bundled skills, or broad tool
  surfaces after a user narrows them.
- Tool failures that trigger repeated model turns and spend without changing
  workspace state.
- Context compaction that hides what was removed or makes a session impossible to
  resume.
- Background agents that block the parent, lose their result, share mutable state
  without ownership, or inherit surprising authority.
- Model-generated diagnoses treated as proof without a trace, profile, test, or
  workspace invariant.
- A large central agent class that becomes the compatibility surface before its
  prompt, transport, tool, persistence, and finalization seams are explicit.
- Output reduction that removes the evidence needed to review a change, or
  verbosity that makes the relevant decision disappear in process narration.

## Different philosophies, not one maturity ladder

Pi is the clearest counterexample to treating every omitted feature as a gap. Its
default exposes only four tools, keeps the system prompt short, and explicitly
leaves native MCP, subagents, plan mode, permission popups, to-do lists, and
background shell execution to extensions or external composition. Its core is
not small: it supports interactive, print, JSON/RPC, and SDK surfaces over a
layered provider, agent, session, and coding runtime. Pi's minimalism is a choice
to omit workflow policy from the default product.

That philosophy does not transfer wholesale to Piku. Pi and its TypeScript
extensions intentionally run with the launching user's authority. Project trust
guards which project resources are loaded, not what tools can do. Piku has
already chosen permission rules, hooks, destructiveness classes, and optional
worktree allocation, so silently adopting full-user authority would contradict its own
direction. Pi is useful prior art for a lean prompt, append-only session tree,
real-loop fake-provider tests, and extension seams, not for Piku's safety
boundary.

Sources: [Pi transfer and canonical home](https://pi.dev/news/2026/5/7/pi-has-a-new-home),
[Pi package boundaries](https://github.com/earendil-works/pi/blob/305c014dcccfe97ebd3f4057ac16c436f1e2c71e/README.md#L13-L45),
[Pi omissions](https://github.com/earendil-works/pi/blob/305c014dcccfe97ebd3f4057ac16c436f1e2c71e/packages/coding-agent/README.md#L492-L508),
[Pi default tools](https://github.com/earendil-works/pi/blob/305c014dcccfe97ebd3f4057ac16c436f1e2c71e/packages/coding-agent/src/core/tools/index.ts#L81-L145),
[Pi session tree](https://github.com/earendil-works/pi/blob/305c014dcccfe97ebd3f4057ac16c436f1e2c71e/packages/coding-agent/src/core/session-manager.ts#L30-L153),
[Pi deterministic harness](https://github.com/earendil-works/pi/blob/305c014dcccfe97ebd3f4057ac16c436f1e2c71e/packages/coding-agent/test/suite/harness.ts#L101-L224).

The emerging inspectable systems point in a different direction:

- Qwen Code is developing a bounded workspace daemon for multiple sessions and
  clients. Its `/btw` command handles a side question outside the durable main
  history while the user keeps typing.
- Kimi Code implements ACP, hooks, plugin trust, and isolated coder, explorer,
  and planner agents. An open issue reports that some configured hooks and
  permission rules can fail open in an interactive Windows path, illustrating
  why the same authority contract must apply across every surface.
- Crush combines durable SQLite sessions, LSP, MCP, skills, hooks, and
  permissions. A fixed command-prefix bypass shows why command structure must be
  parsed rather than treated as a safe string prefix.
- Antigravity CLI and Copilot CLI have fast public release cadences and broad
  documented surfaces, but their public repositories do not contain the engine
  source. Their architecture claims remain vendor claims.

Sources: [Qwen daemon design](https://github.com/QwenLM/qwen-code/blob/main/docs/design/2026-07-31-daemon-capacity-model-and-memory-bounds.md),
[Qwen commands](https://github.com/QwenLM/qwen-code/blob/main/docs/users/features/commands.md),
[Kimi Code](https://github.com/MoonshotAI/kimi-code),
[Kimi #2070](https://github.com/MoonshotAI/kimi-code/issues/2070),
[Crush](https://github.com/charmbracelet/crush),
[Crush #2882](https://github.com/charmbracelet/crush/issues/2882),
[Crush PR #2893](https://github.com/charmbracelet/crush/pull/2893),
[Antigravity source request](https://github.com/google-antigravity/antigravity-cli/issues/4),
[Copilot CLI](https://github.com/github/copilot-cli).

## Implementation patterns that transfer

### One core, multiple surfaces

Codex keeps reusable behavior in `codex-core` and exposes explicit thread, turn,
and item primitives through its app server. OpenCode's next-generation design
also treats the harness as a service/API platform with durable session events and
separate client surfaces. Hermes routes CLI, gateway, ACP, batch, and API modes
through one `AIAgent` core.

Piku already has a narrow provider/runtime split, but the TUI and launch-turn paths
do not yet consume one stable external event model for messages, approvals,
tools, agents, and compaction. Extending `OutputSink` ad hoc will become expensive
if another UI or automation client appears. The next surface should trigger a
typed event-contract design, not another set of callbacks.

Sources: [Codex app server](https://github.com/openai/codex/blob/main/codex-rs/app-server/README.md),
[OpenCode context architecture](https://github.com/anomalyco/opencode/blob/dev/CONTEXT.md),
[Hermes architecture](https://github.com/NousResearch/hermes-agent/blob/main/website/docs/developer-guide/architecture.md).

### Policy is not containment

Codex and Claude document prompt guidance, permission decisions, and OS-level
filesystem/network containment as separate layers. OpenCode's sandbox issue
contains a maintainer acknowledgement that working-directory restrictions are
bypassable through shell execution. Goose tests that denial remains dominant
through later inspection.

Piku's TUI permission policy is useful operator control, but it is not a sandbox.
The distinction is especially important because every writable launch turn and
every child agent loop uses `AllowAll`. This includes a prompt that will enter
the TUI after its first turn. Piku should expose an explicit non-interactive
policy lease or default those paths to a narrower catalog. It should not claim
workspace containment until the boundary is enforced below `sh -c`.

Sources: [Codex sandbox contract](https://github.com/openai/codex/blob/main/codex-rs/core/README.md),
[Claude permissions](https://code.claude.com/docs/en/permissions),
[OpenCode #2242](https://github.com/anomalyco/opencode/issues/2242),
[Goose permission tests](https://github.com/aaif-goose/goose/blob/10df80e409d6aca6636f72c8f4fd7758f263acf0/crates/goose/tests/tool_inspection_permission_precedence.rs).

### Progressive disclosure must remove schemas from context

Claude added agent-only MCP servers and lazy tool loading after strong demand to
keep specialist schemas out of the main context. Hermes users report large fixed
prompt costs from broad default tools and continue to debate retrieval misses
versus the latency of two-stage loading. Aider's repo map demonstrates the same
budgeted-selection principle for code context.

Piku's `tool_search` currently searches a catalog already sent in full. It helps
the model find a name, but does not reduce prompt size or selection noise. A real
implementation would keep a small safety/orchestration set hot, select additional
schemas on demand, and provide a deterministic full-catalog fallback when
retrieval misses.

Sources: [Claude #6915](https://github.com/anthropics/claude-code/issues/6915),
[Hermes #4379](https://github.com/NousResearch/hermes-agent/issues/4379),
[Hermes #6839](https://github.com/NousResearch/hermes-agent/issues/6839),
[Aider repo map](https://github.com/Aider-AI/aider/blob/5dc9490bb35f9729ef2c95d00a19ccd30c26339c/aider/website/docs/repomap.md).

### Delegation is a lifecycle and authority boundary

Codex centralizes parent-child edges, capacity, messaging, and completion in one
shared control tree. Claude scopes each subagent's context, model, tools,
permissions, hooks, memory, and optional worktree. Hermes tests that a child tool
set cannot exceed its parent. OpenCode first exposed background polling, then
removed it after durable completion notification could re-enter the parent
session. Roo's tests show why lifecycle matters: starting a child invalidates
later tools from the same parent message.

Piku already has depth, turn budgets, task IDs, tool filtering, status/join, and
optional worktrees. Its TUI also injects child completion through a bounded
channel. That delivery is best-effort, unacknowledged, absent from launch turns,
and can inject an unbounded final response. Before adding more concurrent writers
Piku should make permission inheritance, cancellation, delivery disposition,
file ownership, worktree result, and parent verification explicit.

Sources: [Codex agent control](https://github.com/openai/codex/blob/main/codex-rs/core/src/agent/control.rs),
[Hermes scope test](https://github.com/NousResearch/hermes-agent/blob/main/tests/tools/test_delegate_toolset_scope.py),
[OpenCode PR #29179](https://github.com/anomalyco/opencode/pull/29179),
[Roo isolation test](https://github.com/RooCodeInc/Roo-Code/blob/b867ec9145750d0ae1ff7f02d35406e9bf2a0b16/src/core/task/__tests__/new-task-isolation.spec.ts).

### Context management is part of the product contract

Goose exposes configurable compaction and fallback strategies. SWE-agent uses a
simple last-N observation processor with explicit omission markers. OpenCode
coordinates compaction with safe turn boundaries and tests known provider
headroom gaps. Cline reports show that irrelevant file and terminal output can
make a paid session unrecoverable.

Piku's observation-masking-first approach is aligned with this evidence. The next
gain is visibility: record what was masked or structurally summarized, why the
threshold fired, the token contribution by content class, and the artifact that
would let a user inspect or reverse the rewrite.

Sources: [SWE-agent history processors](https://github.com/SWE-agent/SWE-agent/blob/3ea751c087f32b16e039a2233dd6eefecef325d5/sweagent/agent/history_processors.py),
[OpenCode compaction tests](https://github.com/anomalyco/opencode/blob/dev/packages/opencode/test/session/compaction.test.ts#L441-L500),
[Cline #4389](https://github.com/cline/cline/issues/4389).

### Tool reliability outranks apparent intelligence

Cline's terminal and edit megathreads link hangs, shell-boundary confusion,
formatter races, truncation, exact-match failures, and retry loops directly to
lost time and API spend. OpenCode's most productive memory investigations use
heap snapshots, exact versions, and controlled toggles rather than generated
diagnoses. SWE-agent maintainers required evaluation results before prioritizing
a sophisticated code-search proposal.

Piku's traces and scenario acceptance checks are the right foundation. A tool
result should have a bounded output, authoritative completion state, changed-file
inventory, stale-read handling, retry ceiling, and a recoverable failure class.
Commit `0a49f7b` now caps each shell output stream at 256 KiB and marks
truncation. That closes the unbounded in-memory shell-output path, but does not
provide the durable full artifact or uniform result metadata needed by the wider
contract.

Sources: [Cline terminal tests](https://github.com/cline/cline/blob/5fdd840d5fc5fc575ae60ab8ee3e43315c4ba0e5/apps/vscode/src/hosts/vscode/terminal/VscodeTerminalProcess.test.ts),
[Cline #4384](https://github.com/cline/cline/issues/4384),
[OpenCode #20695](https://github.com/anomalyco/opencode/issues/20695),
[SWE-agent #38](https://github.com/SWE-agent/SWE-agent/issues/38).

### Memory and self-improvement need provenance

Hermes implements skill learning, session search, memory, and curation, but users
still ask which skill version produced an output and who authorized its mutation.
Its persistent-identity discussion also separates ephemeral delegation from
durable task coordination. OpenCode users similarly ask for typed, persistent
goals with pause/resume, budgets, and completion verification rather than prompt
templates.

Piku should keep historical attempt memory separate from active goal lifecycle.
Its existing automatically extracted semantic memory already lacks source
session, model, evidence, and authorization provenance; that is a present audit
gap, not only a concern for future mutable skills. If Piku adds mutable skills or
procedural memory, every run should also record the governing version, mutation
authority, evidence, and rollback path first.

Sources: [Hermes skills](https://github.com/NousResearch/hermes-agent/blob/main/website/docs/user-guide/features/skills.md),
[Hermes curator tests](https://github.com/NousResearch/hermes-agent/blob/main/tests/agent/test_curator.py),
[Hermes #11692](https://github.com/NousResearch/hermes-agent/issues/11692),
[Hermes #344](https://github.com/NousResearch/hermes-agent/issues/344),
[OpenCode #27167](https://github.com/anomalyco/opencode/issues/27167).

## Human attention: fatigue versus productive friction

“Reading fatigue” is useful only when made testable. In a harness it means the
extraneous review cost of repeated status prose, raw logs, long undifferentiated
diffs, approval chatter, plan revisions, handoffs, unstable scrollback, and
notifications that do not improve a decision. It is not every moment of effort.
Predicting behavior, recalling an invariant, explaining a choice, inspecting a
semantic diff, and debugging a failed assumption can be productive friction
because they build or test the user's mental model.

A practical removal test is: if hiding an element leaves comprehension, defect
detection, and ownership unchanged while reducing decision time or scrolling, it
was probably noise. If the user becomes less able to explain, modify, or debug
the result, the removed work was likely germane. This is a design hypothesis to
measure, not a validated universal “reading fatigue” scale for agentic harnesses.

The manual-coding critique deserves more weight than a preference survey. Writing
code by hand couples prediction, choice, error, and correction. A randomized
study of developers unfamiliar with a library found lower immediate mastery with
AI assistance and only a small, statistically uncertain speed improvement. A
separate controlled study found much higher median task completeness alongside a
lower ability to answer technical questions about code the participants had just
implemented. Earlier novice evidence found no retention harm in a setting that
paired generation with active modification, and workplace studies report real
throughput gains. The responsible conclusion is neither “AI causes brain
atrophy” nor “slower is always better.” Autonomy can remove valuable reasoning
loops, while forcing people through boilerplate can waste attention.

Sources: [Anthropic coding-skills study](https://www.anthropic.com/research/AI-assistance-coding-skills),
[Martin-Lopez et al.](https://doi.org/10.1109/TSE.2026.3679627),
[Kazemitabaar et al.](https://arxiv.org/abs/2302.07427),
[generation effect](https://doi.org/10.1037/0278-7393.4.6.592),
[cognitive offloading review](https://pubmed.ncbi.nlm.nih.gov/27542527/),
[workplace randomized trials](https://doi.org/10.1287/mnsc.2025.00535).

The harness should therefore support at least two intents rather than imposing
one friction level:

- In delivery-oriented work, collapse routine successful reads and searches,
  coalesce activity, preserve errors and mutations, and present a semantic review
  packet with goal, changed files, tests, invariants, risks, and drill-down.
- In learning- or ownership-oriented work, ask for an expected behavior or
  invariant before generation, require active modification or a compact
  teach-back, and check whether the user can later explain, change, and debug the
  result.
- In both, keep every collapsed detail recoverable. Concision without an
  inspectable source is information loss, not progressive disclosure.

Pi's one-line collapsed reads with explicit expansion, Qwen's out-of-history
side questions, Copilot's selective expansion, and repeated complaints about
scroll hijacking support the presentation half of this design. Research on code
review cognitive load and change decomposition supports grouping review by
meaning rather than displaying one long transcript.

Sources: [Pi #4916](https://github.com/earendil-works/pi/issues/4916),
[Kimi #2212](https://github.com/MoonshotAI/kimi-code/issues/2212),
[code-review cognitive load](https://link.springer.com/article/10.1007/s10664-022-10123-8),
[review decomposition](https://arxiv.org/abs/1805.10978),
[targeted code highlighting](https://arxiv.org/abs/2302.07248).

## Piku alignment

### Already aligned

- One in-process Rust loop consumes provider-neutral stream events.
- Provider choice is separate from the user workflow.
- Normal CI uses deterministic providers; live-model work is opt-in.
- Traces prove tool shape and workspace checks prove task effects.
- Real PTY tests treat terminal behavior as product behavior.
- Hooks, background agents, bounded recursion, tool filtering, and optional
  worktree allocation are implemented rather than only described.
- Automatic compaction masks observations before rewriting older history.

### Misaligned implementations

| Priority | Current implementation | Why it conflicts with the evidence |
| --- | --- | --- |
| Load-bearing | Writable launch turns and subagent turns use `AllowAll`; TUI prompting begins only after the launch turn. | Approval has become an implicit broad capability lease, and no lower sandbox enforces workspace containment. |
| Load-bearing | `Safe` calls bypass TUI configuration rules, per-turn allow-all precedes deny, and some mutations are classified `Safe`. | Deny policy is not dominant even after the TUI starts; a new unprotected file, Markdown-memory write, or attempt write can occur without a prompt. |
| Load-bearing | `tool_search` receives a catalog whose schemas were already sent to the model. | Discovery exists, but the context and selection cost it is meant to address remains. |
| Load-bearing | TUI child completion is delivered through a bounded, best-effort channel; launch turns have no task registry, and changed worktrees lack a typed integration/verification state. | Background execution has partial notification but no complete acknowledgement, ownership, or verification contract. |
| Load-bearing | A requested subagent worktree changes the prompt, not the executor or file-tool working directory; bash-only changes can evade its dirty check. | “Isolation” is not enforced and cleanup can discard work that the event heuristic did not observe. |
| High | Piku reads `PIKU.md` but not the portable `AGENTS.md` convention. | Repository instructions remain harness-specific despite unusually strong interoperability demand. |
| High | `OutputSink` is an internal callback surface rather than a stable thread/turn/item event contract. | A second UI or API would duplicate lifecycle semantics and drift from the TUI. |
| High | Context reduction is visible only through a brief message, not an inspectable ledger. | Users cannot see which evidence disappeared, why, or what it cost. |
| High | Automatically extracted semantic memory lacks source session, model, evidence, and authorization provenance. | Retrieval cannot be audited well enough to distinguish useful continuity from stale or misattributed influence. |
| High | Successful tool cards are truncated to a few lines with only an omitted-line count, while every call still adds a separate transcript row. | The UI reduces volume but provides neither lossless expansion nor a decision-focused summary; it can create both hidden evidence and scrolling. |
| High | Session resume replays a fixed recent tail and child completion can inject an unbounded final response. | Neither is organized around the goal, changed state, verification, or unresolved risk a human needs to regain ownership. |
| High | There is no first-class completion or semantic-diff review surface. | Users must reconstruct the result from chat and tool chronology, the most tiring and least decision-relevant representation. |
| Medium | Foreground tools execute sequentially and background writers have optional prompt-routed worktrees. | More concurrency would increase conflict risk before ownership and permission inheritance are explicit. |
| Medium | The main loop, TUI, and agentic playground are large responsibility clusters. | Late extraction creates compatibility shims and makes each behavioral change harder to localize. |

## Recommended sequence

1. Correct surface and evaluator truth: distinguish pass, product failure, and
   inconclusive evidence; stop advertising unavailable or non-blocking behavior.
2. Define explicit authority for launch-turn and subagent execution. Separate prompt
   guidance, permission leases, executor working directory, and process
   containment in both code and docs.
3. Carry dependable tool results through the model, session, trace, UI, resume,
   and evaluator before optimizing schema discovery.
4. Add provenance to automatically injected child memory, then finish reliable
   completion delivery and acknowledgement, cancellation, inherited authority,
   ownership, result disposition, and parent verification.
5. Add a lossless decision view: coalesced activity, stable scrollback, a
   semantic change-and-evidence packet, and drill-down to full tool output.
   Measure reading reduction and defect detection together.
6. Add an opt-in learning and ownership mode that preserves prediction,
   modification, and explanation instead of treating all friction as waste.
7. Require a failing Piku scenario before native `AGENTS.md`; if it exists, test
   bounded discovery and a documented `PIKU.md` overlay.
8. Make tool disclosure genuinely lazy with a small always-hot set and a
   deterministic fallback. Measure selection misses and prompt savings.
9. Introduce a typed external event contract only when a second client or
   surface is being built. Preserve option value until then.

## Explicit deferrals

- Do not add MCP merely for parity. First prove that a concrete Piku workflow
  needs a portable remote tool surface rather than an existing CLI or direct API.
- Do not promise full workspace rewind while arbitrary shell effects are outside
  the mutation ledger. Conversation rewind can be designed separately.
- Do not build a hosted control plane as an extension of the local loop. It needs
  a separate identity, isolation, secrets, network, and operations decision.
- Do not add true parallel writes before task ownership and merge disposition are
  explicit.
- Do not promote mutable self-authored skills without versioned provenance and
  rollback.

## What would change this recommendation

- Repeated dogfood evidence that the current full tool catalog has negligible
  context and selection cost would lower the priority of lazy disclosure.
- A real second client would raise the typed event contract above compatibility
  work.
- A concrete remote or multi-user deployment request would reopen the hosted and
  identity boundaries.
- Measured task results showing that concurrent writers outperform isolated
  sequential handoffs without more recovery cost would reopen the concurrency
  deferral.
- A compact presentation that reduces scrolling but also lowers defect detection,
  delayed comprehension, or successful modification should be rejected.
- A learning-mode intervention that adds time without improving later explanation,
  modification, or debugging should be removed.
