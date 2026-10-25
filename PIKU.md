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
- Writable launch turns permit tool calls after configured deny rules, including
  a prompt that later enters the TUI. They have no interactive confirmation or
  sandbox and do not advertise agent tools because no task registry exists on
  that path. Permission prompts and agent lifecycle support begin only inside
  the TUI loop.
- Configured deny rules precede `Safe` classification and per-turn allow-all.
  New unprotected files and agent-memory writes are among the mutations
  classified `Safe` after restrictions have been checked.
- Direct-file rules check resolved target identity against the process cwd.
  Scoped grants require both the raw spelling and resolved literal scope to
  match; symlink aliases and exact `./` spellings can require fresh approval.
  Protected resolved parents also classify missing file targets as `Definite`.
  This application policy does not provide OS confinement or race protection.
- Project instructions come from `PIKU.md`, `PIKU.local.md`, and
  `.piku/PIKU.md`. Piku does not load `AGENTS.md` and has no MCP, LSP,
  browser/web-search, or image integration.
- The loopback browser surface requires per-launch admission for private reads
  and actions. The printed private launch file admits a tab; HTTP requests use
  bearer credentials, and terminal WebSockets authenticate before PTY creation.
  Host/Origin checks bind to the listener. Browser terminals require the explicit
  `web --terminal` launch flag and retain ambient host authority.

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

Self-update is disabled by default. Developer launches can explicitly approve
one absolute Cargo target directory with `--self-update-root /absolute/build/root`.
A writable opted-in Piku checks only that pinned directory's `release/piku`,
saves the session, snapshots the opened candidate, atomically replaces its
executable, and restarts via `exec`. Release-directory and candidate symlinks
are rejected. Working-directory builds, project settings, environment values,
and Cargo output cannot enable replacement. Read-only mode never self-updates.
Verify the restart banner or process behavior; the mechanism has no signature
verification or automatic rollback. This approval is an application rule, not
OS containment: executed code retains the process's host authority.

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
