# piku

Piku runs a local coding agent in a terminal or browser workspace. It is
experimental and currently built from source.

## Run

Build the binary with stable Rust:

```bash
cargo build --release -p piku
./target/release/piku --help
```

The help output begins with:

```text
terminal AI coding agent

Usage: piku [OPTIONS] [COMMAND]
```

Check configured providers, then start a session:

```bash
piku providers
piku
piku "explain src/main.rs"
piku --resume <session-id>
```

With no prompt, Piku opens its terminal UI. A prompt runs one turn and then
continues there. `-p` prints one result and exits; `--read-only` runs one turn
with inspection tools and exits. Provider and model overrides are available as
`--provider` and `--model`.

Piku can use OpenRouter, Anthropic, Groq, Ollama, or a custom OpenAI-compatible
server. It reads provider settings from the environment, user configuration,
and `.piku/settings.toml`. Run `piku providers` to see what the current launch
can use.

Sessions are stored under `$XDG_CONFIG_HOME/piku/sessions/`, or
`~/.config/piku/sessions/` when `XDG_CONFIG_HOME` is unset.

## Browser workspace

```bash
piku web --port 9090
```

The local browser workspace is under active development. Its canvas can hold
chat, source-change, file, note, page-preview, and terminal cards. Chat cards
keep their own editable turn history and optional context. Page previews render
Markdown, KaTeX, and Mermaid.

## Safety

Piku is not a sandbox. Writable launch turns may run advertised file and shell
tools without confirmation. `--read-only` narrows the tool catalog, but it does
not confine filesystem reads or provider disclosure. The browser terminal is an
unrestricted host shell and requires an explicit start action.

Source-built writable sessions can replace a running Piku process with a newer
local release binary. See [self-update behavior](docs/self-update.md) before
using that workflow.

## Develop

The repository check is:

```bash
just check
```

Live-model and browser evaluations are opt-in. Their commands, evidence format,
and current limitations are documented in
[the evaluation design](docs/design/shared-cross-surface-evaluation.md) and the
[`justfile`](justfile).

More detail:

- [Architecture and current gaps](docs/design.md)
- [Evidence-driven dogfood roadmap](docs/live-dogfood-roadmap.md)
- [Contributing](CONTRIBUTING.md)

## License

MIT. See [LICENSE](LICENSE).
