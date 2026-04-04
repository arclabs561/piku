# piku project context

This is the piku source repository. piku is a terminal AI coding agent written in Rust.

## Workspace layout

```
crates/
  piku-api/       # Provider trait + Anthropic/OpenAI/Groq/Ollama/Custom implementations
  piku-tools/     # Built-in tools: read_file, write_file, edit_file, bash, glob, grep, list_dir
  piku-runtime/   # Agentic loop, session, permissions, system prompt, compaction
  piku/           # Binary: CLI entrypoint, StdoutSink, self_update
```

## Self-hosting rules

- After editing any Rust source file, run: `cargo build --release -p piku`
- piku detects the new binary and restarts itself automatically (same PID, same session)
- Do NOT ask the user to restart — build and let piku restart itself
- The session is auto-saved before restart; nothing is lost

## Build commands

```bash
cargo build --release -p piku     # Build binary only (fast)
cargo build --release              # Build all crates
cargo test --workspace             # Run all tests (155 tests)
cargo test -p piku-runtime         # Run runtime tests only
```

## Key files

- `crates/piku-runtime/src/agent_loop.rs` — the main agentic loop
- `crates/piku-runtime/src/session.rs` — session model and persistence
- `crates/piku-runtime/src/permission.rs` — permission tiers
- `crates/piku-api/src/openai_compat.rs` — OpenAI-compatible SSE provider
- `crates/piku/src/main.rs` — CLI entrypoint, provider resolution
- `crates/piku/src/self_update.rs` — self-update mechanism
- `DESIGN.md` — full architecture spec
- `SELF_UPDATE.md` — self-update design doc

## Current state

- v0 single-shot mode works: `piku "prompt"`
- TUI not yet built (phase 6)
- Session resume not yet wired to CLI (--resume flag missing)
- Compaction not yet implemented
- MCP client not yet implemented
