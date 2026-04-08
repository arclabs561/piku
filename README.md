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

```bash
cargo test --workspace -- --test-threads=1
```

## Run

```bash
cargo run -p piku -- --help
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

