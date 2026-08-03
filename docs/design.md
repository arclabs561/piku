# Piku design

Status: current implementation guide

Piku is a local, terminal-native coding agent written in Rust. It owns its
provider transport, agent loop, tools, sessions, permissions, hooks, memory,
background agents, and terminal UI. The repository uses deterministic tests for
the normal quality gate and keeps live-model dogfood opt-in.

This document describes the implementation as it exists. Focused tracked design
documents under `docs/` explain individual mechanisms and their evidence
boundaries. This checkout also keeps an intentionally ignored, owner-local ADR
ledger under `docs/adr/`; public claims must not depend on that ledger being
present in a fresh clone.

## Design principles

- Own the loop. Provider calls, tool dispatch, persistence, and compaction stay
  in process rather than behind an agent framework.
- Keep provider transport behind one event model. The runtime consumes normalized
  streaming events rather than provider-specific wire formats.
- Prefer observable evidence. Traces and workspace acceptance checks carry more
  weight than rendered terminal text or an LLM judge's prose.
- Keep the normal gate deterministic. Live providers are useful for discovery,
  but their failures become blocking only after reproduction in a deterministic
  test or a provider-specific parser test.
- Keep hosted control-plane concerns out of the runtime until a concrete need
  justifies the authentication, isolation, and operations surface.

## Workspace

```text
crates/piku-api      provider clients and normalized request/event types
        ^
        |
crates/piku-tools    built-in tools and tool metadata
        ^
        |
crates/piku-runtime  agent loop, sessions, permissions, hooks, memory, agents
        ^
        |
crates/piku          CLI, terminal UI, configuration, tracing, self-update

tests/fixture        isolated crate used by agentic dogfood
scripts              canonical CI and local dogfood entrypoints
docs                 focused designs, roadmaps, and operating notes
```

The binary depends on both `piku-runtime` and `piku-tools`. `piku-runtime`
depends on `piku-api` and `piku-tools`; `piku-tools` depends on `piku-api` for
provider-neutral tool definitions.

## Core abstractions

| Abstraction | Location | Responsibility |
| --- | --- | --- |
| `Provider` and `Event` | `crates/piku-api/src/provider.rs`, `types.rs` | Stream a request and normalize text, tool-use, stop, and usage events. |
| `run_turn_with_registry` | `crates/piku-runtime/src/agent_loop.rs` | Drive model iterations, permission checks, tool dispatch, hooks, compaction, memory extraction, and background agents. |
| `Session` | `crates/piku-runtime/src/session.rs` | Persist attributed conversation messages and usage as JSON. |
| `ToolEntry` | `crates/piku-tools/src/lib.rs` | Pair a tool name and schema with destructiveness metadata; dispatch is separate. |
| `TaskRegistry` | `crates/piku-runtime/src/task.rs` | Track background subagents with depth/turn bounds, completion notices, and optional worktree allocation. |
| `HookRegistry` | `crates/piku-runtime/src/hooks.rs` | Run configured lifecycle commands and let pre-tool hooks deny calls. |

## Provider boundary

`piku-api` exposes one object-safe streaming `Provider` trait. Native Anthropic
and OpenAI-compatible clients translate provider responses into the same `Event`
enum before the runtime sees them. Runtime provider resolution currently supports:

- OpenRouter
- Anthropic
- Groq
- Ollama
- a custom OpenAI-compatible endpoint

Provider selection is opportunistic unless `--provider` is supplied. Credentials
come from environment variables; project settings do not store provider keys.

## Agent loop

```text
user input
  -> append to Session
  -> mask or structurally compact old observations when required
  -> build MessageRequest from prompt, history, and tool definitions
  -> stream normalized provider events
  -> collect complete tool calls
  -> for each call, in order:
       validate advertised availability
       run PreToolUse hooks
       apply permission policy
       reject repeated read-only calls
       dispatch the tool or background-agent operation
       run PostToolUse hooks
       append the result to Session
  -> repeat until the model ends, fails, is cancelled, or reaches max turns
```

Tool calls returned in one model message execute sequentially. Background agents
are the current concurrency mechanism.

Automatic context reduction is deterministic. It masks older large tool results
first and falls back to a structural summary if masking is insufficient. The
threshold is half of the resolved model context window, with twelve recent
messages preserved. An unused LLM-summary helper remains in the runtime, but no
manual `/compact` command is exposed.

## Tools and capability discovery

The built-in registry includes file reads and writes, shell execution, search,
directory listing, background-agent control, persistent memory, semantic memory,
attempt trees, and `tool_search`.

`tool_search` searches metadata for definitions that were already placed in the
request. It is not yet lazy schema loading: the normal writable loop still sends
the full built-in tool catalog to the model on every request. Read-only mode is
the exception; it advertises only `read_file`, `glob`, `grep`, and `list_dir`.

Shell stdout and stderr are each capped at 256 KiB with an explicit truncation
marker. This bounds one high-volume path, but Piku does not yet provide a uniform
tool-result envelope or a durable full-output artifact behind truncated output.

There is no implemented MCP client, LSP client, browser, web-search tool, image
input, or patch-edit protocol. Those capabilities should remain described as
possible future integrations, not current behavior.

## Permissions and execution boundaries

Each tool classifies a call as `Safe`, `Likely`, or `Definite`.

- `Safe` calls run before the configured prompter, so configuration deny rules
  cannot block them. This includes new unprotected files, Markdown-memory writes,
  and attempt recording.
- For calls that reach the interactive TUI prompter, a prior per-turn allow-all
  wins. Otherwise configured deny rules win over allow rules; remaining calls
  prompt for one-time allow, deny, or allow-all for the rest of the turn.
- Pre-tool hooks run before the permission prompt and can deny a call.
- `--read-only` removes mutating and agent tools from the advertised catalog,
  but does not confine reads, provider disclosure, or Piku's own persistence.

There is no AI safety classifier in the current implementation, despite older
design text describing one. `Likely` and `Definite` differ in presentation, not
in the decision mechanism.

Every writable launch turn uses `AllowAll`, including `-p`, a prompt that later
enters the TUI, and a resumed prompt. Permission prompts begin only inside the
TUI loop. A launch turn is therefore an automation interface, not a sandbox or
approval boundary. `--read-only` narrows the advertised tools, but its read
paths are not workspace-confined and project/user memory can still reach the
provider, so it is not a confidentiality boundary.

Path checks in individual file tools reduce some risk, but they do not create
process isolation. The shell tool executes through `sh -c` with the invoking
user's authority. A subagent worktree is currently conveyed through prompt text;
the executor and file tools retain the parent process working directory, so this
is routing guidance rather than an enforced workspace boundary.

## Background agents

`spawn_agent` starts a fresh-session agent as a background Tokio task. The parent
can poll or join it through `TaskRegistry`. In the TUI, a bounded notification
channel also injects completed child output as a user-role interjection. The
launch-turn path has no task registry, so agent tools advertised there fail as
unavailable even when that turn will later enter the TUI. The implementation
provides:

- a hard recursion depth of four;
- per-agent turn budgets;
- built-in and `.piku/agents/*.md` agent definitions;
- tool allowlists and blocklists applied to the parent's advertised catalog;
- optional parent-context forking with per-block truncation but no aggregate
  fork-context bound;
- optional temporary Git worktree allocation with prompt-routed working-directory
  guidance;
- lifecycle hooks around subagent start and stop.

A changed worktree is left for the caller to inspect. Piku does not merge or
apply a child branch automatically. The dirty check observes `write_file` and
`edit_file` tool events, so changes made only through `bash` can be missed during
cleanup. The `background: false` field changes the returned hint but does not
make spawning synchronous. Agent output is a report to the parent, not
independent proof that the task succeeded.

## Sessions, memory, and hooks

Sessions are JSON files under `$XDG_CONFIG_HOME/piku/sessions/`, falling back to
`~/.config/piku/sessions/`. Provider/model attribution is session-level startup
metadata: it is overwritten on resume and can become stale after `/model`.
Per-turn attribution and pricing are not implemented.

Memory has three distinct surfaces:

- Markdown memory for explicit user, project, and local notes;
- embedding-backed semantic memory with extraction during compaction;
- attempt trees for recording and querying prior approaches.

These surfaces are implemented, but their presence does not prove that retrieval
improves coding outcomes. Automatically extracted semantic entries currently
record content and retrieval metadata without source session, model, evidence,
or authorization provenance. The dogfood loop should measure whether retrieved
material is relevant and used, and should make provenance inspectable, before
the system adds more automatic recall.

The interactive TUI loads hooks from global and project JSON files. They cover
pre/post tool use, session start, stop, pre-compaction, and subagent lifecycle
events. Read-only mode deliberately disables hooks, and launch turns do not load
them. Hook names and schemas are modeled after Claude Code, but Piku
executes them locally and does not inherit Claude Code's authorization boundary.

## Configuration and CLI

Settings are JSON, not TOML:

```text
$XDG_CONFIG_HOME/piku/settings.json   global settings (`~/.config` fallback)
.piku/settings.json           project overrides
CLI flags                      provider and model overrides
```

Settings cover provider/model choice, maximum turns, and tool allow/deny
patterns. Environment variables carry provider endpoints, credentials, and
embedding configuration.

The CLI supports interactive use, one-shot prompts, `-p` headless output,
read-only mode, provider/model selection, session resume, and provider status.
The TUI is implemented directly with Crossterm and Syntect rather than Ratatui.
Its current slash-command surface covers help, status, token cost, model/provider
status, saved sessions, background tasks, permission rules, hooks, clear, and
exit. Older design entries for `/diff`, `/init`, `/export`, and interactive
session switching are not implemented.

## Human attention and review

The TUI currently truncates successful tool output to short cards and reports an
omitted-line count, but does not provide a lossless expansion path. Each tool
event still occupies a separate transcript row. Session resume shows a fixed
recent tail rather than a semantic task checkpoint, and background-agent
completion may inject the child's full final response into the parent. There is
no first-class `/diff` or completion-review view.

The intended direction is not simply less text. Routine successful activity can
be coalesced, but errors, mutations, invariant failures, and divergence should
remain visible. Every collapsed item should be expandable or backed by a durable
full artifact. Completion should be grouped around the goal, changed files,
tests, risks, and unresolved decisions so a user can verify and own the result
without reconstructing it from chronology.

Some friction is productive. Predicting behavior, naming an invariant, actively
modifying generated code, explaining a choice, and debugging a failed assumption
can build or test the user's mental model. Piku should eventually distinguish a
delivery-oriented presentation from an opt-in learning or ownership mode, and
measure both attention cost and later ability to explain, modify, or debug the
change.

## Quality loop

The repository separates four evidence tiers:

1. Unit and integration tests for provider parsing, tools, runtime behavior, CLI,
   rendering, hooks, memory, and input.
2. Scripted-provider runtime tests for deterministic multi-iteration agent-loop
   behavior.
3. Isolated PTY smoke tests for the real binary and terminal state.
4. Opt-in live-provider and agentic-user suites for behavior discovery.

The terminal playground adds personas, keystroke and viewport evidence, workspace
acceptance checks, an append-only ledger, a primary LLM review, and a bounded
recursive review. Deterministic checks outrank reviewer prose only for the exact
property they assert. Several scenario goals are broader than their predicates,
and verifier spawn or timeout errors currently enter the same failure path as a
failed product assertion. Until those are separated into pass, fail, and
inconclusive states, an acceptance result is not automatically a product verdict.

## Known implementation gaps

- Every writable launch turn lacks an explicit non-interactive approval policy
  and currently allows every advertised tool call; TUI permission policy begins
  only after that turn.
- Launch turns advertise agent tools without providing the task
  registry required to execute them. Memory, attempt-tree, and `tool_search`
  operations use shared runtime paths and remain available there.
- Read-only paths are not workspace-confined, project/user memory can enter the
  provider context, and Piku still writes its own runtime state.
- The `Likely` versus `Definite` distinction does not change the decision path;
  the documented classifier layer is absent.
- Configuration deny rules do not dominate `Safe` classification or a prior
  per-turn allow-all. Some state-changing tools are classified `Safe`, including
  creation of a new unprotected path, Markdown-memory writes, and attempt writes.
- The full tool schema is sent on normal requests, so `tool_search` improves
  naming discovery but not context size or tool-selection load.
- Successful tool output is truncated without lossless expansion, while the
  transcript has no semantic completion or change-review surface.
- Display and trace previews use byte-indexed UTF-8 slicing and can panic at a
  non-ASCII boundary; launch-turn stdout can contain ANSI escapes when redirected.
- Session replay and background completion are sized by recent chronology and
  child output rather than by the information needed to resume or verify a goal.
- Background completion is best-effort and TUI-only; cancellation,
  acknowledgement, worktree disposition, and parent verification are not a
  complete typed lifecycle.
- Worktree routing is prompt-based, `background: false` is not blocking, and
  bash-only worktree changes can evade the current dirty check.
- Built-in agent-role labels are not capability boundaries: the explorer blocks
  workspace writes and shell but can still mutate memory/attempt state.
- Forked parent context truncates individual blocks but has no aggregate bound.
- Read-only compaction can still trigger semantic extraction into persistent
  state because that path lacks the mode flag.
- Semantic-memory extraction lacks source and authorization provenance.
- Foreground tool calls are sequential even when independent.
- MCP and LSP integration are absent.
- Configuration ownership remains in the binary crate even though provider
  resolution moved into `piku-runtime`.
- `agent_loop.rs`, `tui_repl.rs`, and the agentic playground are large change
  surfaces with several responsibilities each.
- The live dogfood evaluator is better grounded than before, but LLM reviews
  still require deterministic citation validation and cannot establish product
  correctness by themselves.

These are inventory items, not an instruction to implement all of them. Each
candidate change should be justified by a failing user scenario, a repeated
trace-backed finding, or a clear reduction in existing complexity.

## Non-goals

- Do not build a hosted execution control plane without a separate decision on
  identity, isolation, secrets, networking, and operations.
- Do not chase feature parity with larger harnesses as a goal in itself.
- Do not make live-model behavior a pull-request gate.
- Do not treat model-generated review prose as ground truth.
- Do not make GitHub artifact ingestion part of the product runtime while the
  local dogfood corpus supplies the needed evidence.
- Do not add an integration merely because another harness exposes it; require a
  Piku user scenario and a boundary test.

## Where to read next

1. `crates/piku-runtime/src/agent_loop.rs` for the execution path.
2. `crates/piku-tools/src/lib.rs` for the advertised capability surface.
3. `crates/piku/src/tui_repl.rs` for interactive permissions and rendering.
4. `crates/piku-runtime/tests/e2e.rs` for deterministic loop invariants.
5. `crates/piku/tests/agentic_user.rs` and `crates/piku/tests/agentic/` for the
   live evaluation boundary.
6. `docs/live-dogfood-roadmap.md` for the operational evidence loop.
