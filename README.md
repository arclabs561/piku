# piku

Piku runs a local coding agent in a terminal or browser workspace. It is
experimental and currently built from source.

For a mature terminal agent, use Claude Code, Codex CLI, aider, or opencode;
piku pairs a provider-agnostic Rust agent with a browser canvas that holds
chat, terminal, file, and preview cards side by side.

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

Piku can use ChatGPT plan access, OpenRouter, Anthropic, Groq, Ollama, or a custom
OpenAI-compatible server. Provider selection comes from the launch environment,
user configuration and explicit flags. Repository `.env` files and project
settings cannot choose credential destinations or override provider/model
selection. Run `piku providers` to see what the current launch can use.

For eligible ChatGPT accounts, Sign in with ChatGPT keeps Piku's own agent loop
and tools. Login works from captured command runners as well as terminals and
returns a local browser link for explicit consent:

```bash
piku auth chatgpt login
piku auth chatgpt accounts
piku auth chatgpt select ACCOUNT_ID
piku auth chatgpt models
piku --provider chatgpt --model MODEL
```

Select the account ID and model from the listed values. `auth chatgpt status`
shows login progress and stored metadata. Open the link and choose Continue
within three minutes; `auth chatgpt cancel-login` stops a pending attempt.
The printed link contains no OAuth credentials or identity hints.
`auth chatgpt logout ACCOUNT_ID` removes local credentials and attempts remote
revocation. Account credentials remain separate from
project settings. Session replay is bound to the original account and model.

This preview route requires plan-use permission and has no hard output-token
cap, including child and memory calls. Allowance and optional credit usage are
unknown; Piku shows token counts without applying API-dollar estimates. It does
not switch to a paid API or another executor on failure. Embeddings remain a
separate route. Live account interoperability still needs validation.

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

Developer self-update is disabled unless `--self-update-root` names an approved
absolute Cargo target directory. See [self-update behavior](docs/self-update.md)
before using that workflow.

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
