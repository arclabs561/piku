# piku

`piku` is a Rust agentic coding harness with a sticky-bottom TUI, tool execution,
session persistence, and local dogfood workflows.

## Layout

```text
.
├── crates/
│   ├── piku/         # CLI + TUI entrypoint
│   ├── piku-api/     # provider clients and streaming/event types
│   ├── piku-runtime/ # agent loop, session, permissions
│   └── piku-tools/   # built-in tools
├── tests/fixture/    # isolated play-dir used by agentic dogfood
└── justfile          # repo entrypoints
```

## Build

```bash
cargo build --workspace
```

## Test

`just check` runs the full gate (fmt, clippy with `-D warnings`, tests, release
build) — the exact commands CI runs, defined once in `scripts/ci.sh`:

```bash
just check
```

Just the default test suite (fast, deterministic — no live LLM, no PTY):

```bash
cargo test --workspace
```

The PTY smoke tests (`tui_smoke`) drive the real binary over a pseudo-terminal;
they are isolated into their own stage because they stall under full-workspace
concurrency. `just check` runs them; standalone:

```bash
cargo test --test tui_smoke -- --ignored
```

The end-to-end suites that drive a real LLM (`llm_e2e`, `dogfood`, and the
`agentic_user` personas) are `#[ignore]`d so the default run reports them as
*ignored* rather than silently passing. They are opt-in, need a provider key,
and run per-suite (not `--workspace -- --ignored`, which would also wake the PTY
tests):

```bash
export OPENROUTER_API_KEY=sk-or-...        # or ANTHROPIC_API_KEY / GROQ_API_KEY
cargo test --test llm_e2e -- --ignored
cargo test --test dogfood -- --ignored
just agentic-user                          # one persona; see justfile for more
```

## Run

Interactive (TUI REPL):

```bash
cargo run -p piku -- --help
piku "explain src/main.rs"
```

Headless (run once, print, exit — for scripts and pipelines, like `aider -m` or
`claude -p`):

```bash
piku -p "explain src/main.rs" > explanation.txt
```

## Dogfood

Default isolated smoke run:

```bash
just agentic-user
```

Run against a temp copy of this repo's real code:

```bash
just agentic-user-real
```

Use a custom play dir:

```bash
PIKU_AGENTIC_PLAYDIR=/path/to/playdir just agentic-user
```

Full multi-turn mode:

```bash
just agentic-user-full
```

