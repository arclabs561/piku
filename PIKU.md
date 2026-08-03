# piku project context

Piku is a Rust agentic coding harness with a CLI and sticky-bottom TUI.

## Workspace

```text
crates/piku-api/       Provider trait, clients, and streaming event types
crates/piku-tools/     Built-in tool metadata and executors
crates/piku-runtime/   Agent loop, sessions, permissions, compaction, memory
crates/piku/           CLI, TUI, configuration, tracing, and self-update
```

The dependency direction is `piku-api <- piku-tools`, with `piku-runtime`
depending on both and the `piku` binary depending on the runtime and tools.
Keep provider protocol details in `piku-api`, orchestration in `piku-runtime`,
tool implementations in `piku-tools`, and presentation/configuration in `piku`.

## Current behavior

- No prompt opens the interactive TUI; a prompt continues into the TUI after
  the first turn unless `-p` or `--read-only` is used.
- Sessions are persisted and can be resumed with `--resume <id>`.
- Context is compacted automatically with observation masking and a structural
  fallback. There is no manual `/compact` command.
- Writable modes share Markdown and semantic memory, attempt tracking, and tool
  search. Once entered, the TUI additionally provides permission prompts, a
  background-agent registry, and hooks.
- Read-only mode exposes only `read_file`, `glob`, `grep`, and `list_dir` and
  disables hooks. It does not confine reads or provider disclosure, and Piku's
  own persistence still writes session/history state; prompt-at-launch runs also
  write trace state.
- Every writable launch turn uses `AllowAll`, including a prompt that later
  enters the TUI. It has no confirmation or sandbox and advertises agent tools
  without a task registry, so those calls return unavailable stubs. Permission
  prompts and agent lifecycle support begin only inside the TUI loop.
- In the TUI, `Safe` calls bypass configuration rules and a prior per-turn
  allow-all precedes deny rules. New unprotected files and agent-memory writes
  are among the mutations currently classified `Safe`.
- Project instructions come from `PIKU.md`, `PIKU.local.md`, and
  `.piku/PIKU.md`. Piku does not load `AGENTS.md` and has no MCP, LSP,
  browser/web-search, or image integration.

## Working in this repository

Read the target, its callers, and nearby tests before editing. Keep user and
peer changes intact. Prefer a small, evidence-backed change over broad cleanup.
Treat model review as a hypothesis until deterministic evidence reproduces it.

Run the canonical repository gate:

```bash
just check
```

It owns formatting, script self-tests, strict Clippy, deterministic tests,
isolated PTY smoke tests, and the release build. Use narrower commands while
iterating, but `just check` is the completion gate.

Live LLM suites are opt-in and require explicit provider credentials. Use the
`justfile` recipes rather than inventing an invocation. Do not report an LLM
judge statement as a verified defect, and distinguish an acceptance failure
from an inconclusive verifier timeout or transport failure.

## Self-hosting

`cargo build --release -p piku` writes the default candidate binary. A writable
source-built piku may detect it, save the current session, atomically replace
its executable, and restart via `exec`. Do not assume that happened: verify the
restart banner or process behavior. Read-only mode does not self-update, and
the mechanism provides no confirmation, signature verification, or automatic
rollback.

## Key references

- `crates/piku-runtime/src/agent_loop.rs`: orchestration and tool routing
- `crates/piku-runtime/src/session.rs`: session model and persistence
- `crates/piku-runtime/src/permission.rs`: permission policy
- `crates/piku-tools/src/lib.rs`: built-in tool catalog
- `crates/piku/src/main.rs`: CLI and headless entry point
- `crates/piku/src/tui_repl.rs`: interactive surface and registries
- `crates/piku/src/self_update.rs`: self-update implementation
- `docs/design.md`: current architecture, boundaries, and known gaps
- `docs/live-dogfood-roadmap.md`: evidence loop and active priorities
- `docs/self-update.md`: detailed self-update note
