# piku

Piku is an agentic coding harness with tool execution and a terminal UI. It is
written in Rust and currently intended to be built from source.

Piku keeps the agent loop, provider clients, tools, and interface separate:

```text
piku (CLI + TUI)  -+-> piku-runtime -+-> piku-api
                   |                 +-> piku-tools -> piku-api
                   +-------------------> piku-tools
```

The practical goal is verified change per unit of scarce human attention. Piku
tries to keep routine execution compact while leaving authority, important
decisions, and evidence visible. Faster generation should not remove the slower
thinking needed to understand or own a change.

## Build and run

Requires the stable Rust toolchain and `just` for the canonical repository
check.

```bash
cargo build --release -p piku
./target/release/piku --help
./target/release/piku "explain this repository"
```

With no prompt, piku opens its sticky-bottom TUI. A prompt runs one turn and
then continues in the TUI unless `-p` or `--read-only` is selected. Both of
those modes run one turn, print the result, and exit:

```bash
piku
piku "explain src/main.rs"
piku -p "summarize the public API"
piku --read-only "inspect this repository"
piku --resume <session-id>
```

Sessions are saved under `$XDG_CONFIG_HOME/piku/sessions/`, falling back to
`~/.config/piku/sessions/`. Durable run records are available locally at
`http://127.0.0.1:8080` with `piku web` (or choose a port with
`piku web --port 3000`). Long sessions are compacted automatically with
observation masking and a deterministic structural fallback while preserving a
recent tail. There is currently no manual `/compact` command.

## Providers

Piku supports OpenRouter, Anthropic, Groq, Ollama, and custom OpenAI-compatible
servers. Inspect availability with:

```bash
piku --providers
```

Provider selection can be explicit with `--provider` and `--model`, or resolved
from `PIKU_BASE_URL`, `OPENROUTER_API_KEY`, `ANTHROPIC_API_KEY`, `GROQ_API_KEY`,
and `OLLAMA_HOST`. User settings live in
`$XDG_CONFIG_HOME/piku/settings.toml`, falling back to
`~/.config/piku/settings.toml`; project overrides live in
`.piku/settings.toml`.

When a provider variable is absent from the launch environment, Piku reads only
recognized provider variables from the nearest `.env`; unrelated entries are
not imported. Set `PIKU_NO_DOTENV=1` to disable this discovery. Inherited
variables always win, and Piku never logs credential values.

## Tools and safety

Interactive piku exposes tools for:

- file inspection and editing, search, directory listing, and shell commands;
- background agents and their status/results;
- Markdown and semantic memory;
- attempt-tree recording and lookup; and
- tool metadata search.

`--read-only` advertises only `read_file`, `glob`, `grep`, and `list_dir`, but it
does not confine reads or provider disclosure. Piku still persists session and
history state, and prompt-at-launch runs also write trace state. It is a
mutation-restricted tool catalog, not a
confidentiality or filesystem sandbox.

Every writable launch turn uses `AllowAll`, including `-p`, `piku "prompt"`,
and a resumed prompt. Permission prompts begin only after control enters the
TUI. Launch turns therefore auto-allow every advertised tool call and are not a
sandbox. Run them only inside a boundary whose authority you accept. Their
stdout can also contain ANSI escapes when redirected.

Background-agent tools lack a task registry during any launch turn and return
unavailable stub results there. Markdown memory, semantic memory, attempt trees,
and `tool_search` do run through the shared runtime. Hooks and completion
notifications are wired only after entering the writable TUI and are disabled
in read-only mode.

Inside the TUI, `Safe` calls bypass configuration rules, and a prior per-turn
allow-all precedes deny rules. Some mutations, including creation of a new
unprotected file and agent-memory writes, are currently classified `Safe`.

Piku reads `PIKU.md`, `PIKU.local.md`, and `.piku/PIKU.md` from the working
directory and its ancestors. It does not currently load `AGENTS.md`, and it has
no MCP, LSP, browser/web-search, or image tool integration.

## Self-update while dogfooding

When a source-built piku detects a newer `target/release/piku`, it saves the
session, replaces the running executable, and execs the new binary. This path is
automatic in writable modes and limited to the local default release build. It
has no interactive confirmation, signature verification, or automatic rollback;
verify the restart banner and keep source control as the recovery boundary. See
[the self-update implementation note](docs/self-update.md).

## Develop and verify

`just check` is the canonical local gate and runs the same stages as CI:
formatting, shell self-tests, strict Clippy, deterministic tests, isolated PTY
smoke tests, and the release build.

```bash
just check
```

Live LLM and simulated-user suites are opt-in because they need provider
credentials, cost money, and produce variable outcomes:

```bash
just live
just live-dogfood
just agentic-user
just playground-control
```

These runs keep deterministic scenario checks separate from model review. A
model-reported issue is a hypothesis until reproduced. The intended contract
treats a verifier transport or timeout failure as inconclusive, but current
scenario handling still conflates some infrastructure failures with acceptance
failures. That implementation gap is tracked in the roadmap. Detailed recipes
are in the `justfile`.

### Rendered-workspace QA

The web workspace has two complementary browser gates. Both target a Piku
server you already started at `http://localhost:9090`; the QA clients use the
equivalent explicit IPv4 address `http://127.0.0.1:9090` to avoid local IPv6
resolution drift. Neither command starts,
stops, or restarts it.

```bash
# Stable assertions for interaction, persistence, and authority boundaries.
just web-e2e

# Autonomous Codex exploration through the project-scoped Playwright MCP.
just web-qa-agent

# Two isolated explorers followed by a fresh evidence-only synthesis judge.
just web-agent-parallel
```

The agent run is a full visual user journey. Codex creates a temporary surface,
performs one bounded chat request and one page-source request, tests notebook
reruns, provenance, context, recovery, authority, spatial utility, desktop and
narrow layouts, then removes the surface. It never starts the terminal or
requests repository mutation. Each run writes a structured report, five
screenshots, and the raw Codex JSONL event stream under
`.artifacts/playwright-agent/runs/`.

The runner rejects reports that lack the eight ordered phases, 15 successful
Playwright actions, five event-proven PNG screenshots, the eight ordered product
thesis dimensions, or successful surface cleanup. Exploration is capped at 55
browser actions and seven accessibility snapshots so repeated state dumps
cannot masquerade as deeper QA. The run verdict and product-thesis verdict are
separate: a credible evaluation can complete while finding the product partial
or unsupported. The agentic critique remains exploratory evidence; `just
web-e2e` is the deterministic regression oracle.

Exploration is bounded to 15 minutes by default. Override it with
`PIKU_CODEX_TIMEOUT_MS` when debugging a slower browser or model. The terminal
prints one compact line per completed browser action; the complete Codex event
payload remains in `events.jsonl` for audit and replay.
The command exits nonzero for a `failed` or `blocked` exploratory verdict, while
still preserving the complete evidence bundle.

The parallel command currently runs `coding_trace` and `recovery` explorers in
separate ephemeral Codex processes, browser contexts, Piku surfaces, request
IDs, and artifact directories. It reports each role's browser-call and snapshot
progress, enforces process-group cleanup after budget or timeout termination,
and only starts a third fresh Codex synthesis process when both evidence packets
validate. Explorer conclusions are not passed as hidden narrative: synthesis
receives the cited evidence packets and shared ledger. Parallel artifacts live
under `.artifacts/playwright-agent/parallel/<run>/<perspective>/`; each run has
a `manifest.json` indexing role status, evidence, raw events, screenshots, and
synthesis. Each explorer is capped at 40
browser calls and six snapshots; the recovery lens deliberately uses only file,
chat, and note cards so setup mechanics do not consume its investigation budget.

CLI and web live evaluations emit the same versioned envelope into the
`target/live-ledger/` artifact family. Rows distinguish product, harness,
infrastructure, timeout, and inconclusive outcomes and may carry structured
todos, ideas, and retest obligations for later judges. Summarize all local rows
with `just eval-summary`. The shared schema lives at
`eval/evaluation-envelope.schema.json`; projection-specific traces and
screenshots remain separate authoritative evidence.

## Architecture and direction

- [Architecture and current gaps](docs/design.md)
- [Evidence-driven dogfood roadmap](docs/live-dogfood-roadmap.md)
- [Shared CLI and web evaluation design](docs/design/shared-cross-surface-evaluation.md)
- [Contributing](CONTRIBUTING.md)

## License

MIT. See [LICENSE](LICENSE).
