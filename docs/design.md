# piku — Design Specification

> A personal, from-scratch, terminal-native AI coding agent.
> Written in Rust. Provider-agnostic. Minimal UI. No inherited baggage.

---

## 1. Philosophy

- **Own your loop.** The agentic execution loop is fully in-process, not delegated to a framework.
- **Provider-agnostic by design.** Anthropic, OpenAI, Gemini, Ollama, anything OpenAI-compatible — all behind a single trait. No provider lock-in.
- **Minimal TUI, maximum signal.** Terminal-native. No panels or sidebars in v0. The chat is the interface.
- **Adaptive permissions, not mode-switching.** Destructive operations are identified by heuristics + cheap AI classification. No global "danger mode" toggle — every action is evaluated on its own merit.
- **Sessions are first-class.** Everything persists automatically. You never lose context.

---

## 2. Workspace Layout

```
piku/
├── Cargo.toml              # Workspace root
├── DESIGN.md               # This file
└── crates/
    ├── piku-api/           # Provider trait + all provider implementations
    ├── piku-tools/         # Built-in tool implementations + MCP client
    ├── piku-runtime/       # Agentic loop, session, config, permissions, compaction
    └── piku/               # Binary: TUI + CLI entrypoint
```

Runtime data lives in `~/.config/piku/`:
```
~/.config/piku/
├── config.toml             # Global config
└── sessions/               # Auto-persisted session files
    └── <id>.json
```

Per-project context: `PIKU.md` (walks up from cwd, like `CLAUDE.md`).

---

## 3. Crate Responsibilities

### `piku-api`

Provider-agnostic API layer. Defines the core trait and all provider implementations.

```rust
pub trait Provider: Send + Sync {
    fn stream_message(
        &self,
        request: MessageRequest,
    ) -> impl Stream<Item = Result<Event, ApiError>> + Send;
}
```

**Implementations:**
- `AnthropicProvider` — native SSE, `x-api-key` or Bearer token, prompt caching headers
- `OpenAiProvider` — OpenAI-compatible SSE (covers OpenAI, Gemini via compat endpoint, any base-URL-swappable service)
- `OllamaProvider` — local, no auth, OpenAI-compatible wire format

Provider selected via `config.toml` `[provider]` section or `--provider` CLI flag. All providers normalized to the same `Event` enum before the runtime sees them.

**Key types:**
- `MessageRequest` — model, messages, tools, system prompt, max_tokens, streaming flag
- `Event` — `TextDelta`, `ToolUseStart`, `ToolUseDelta`, `ToolUseEnd`, `MessageStop`, `UsageDelta`
- `ToolDefinition` — name, description, JSON schema (provider-agnostic)

---

### `piku-tools`

All built-in tool implementations and the MCP client.

**v0 built-in tools:**

| Tool | Description |
|------|-------------|
| `read_file` | Read file contents with optional line range |
| `write_file` | Write/overwrite a file |
| `edit_file` | Surgical old→new string replacement (errors on ambiguous match) |
| `bash` | Execute shell command with timeout and working directory |
| `glob` | Find files matching a glob pattern |
| `grep` | Search file contents with regex, returns file:line matches |
| `list_dir` | List directory contents |
| `mcp_call` | Dynamically dispatch to any discovered MCP tool (post-v0) |

**Tool trait:**
```rust
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn schema(&self) -> serde_json::Value;
    fn execute(&self, params: serde_json::Value) -> impl Future<Output = ToolResult> + Send;
    fn is_destructive_heuristic(&self, params: &serde_json::Value) -> Destructiveness;
}

pub enum Destructiveness {
    Safe,       // auto-approve
    Likely,     // run cheap classifier
    Definite,   // always prompt user
}
```

**Concurrent execution:** independent tool calls in a single assistant turn are dispatched
concurrently via `tokio::spawn`. Tool calls with overlapping write targets are serialized.

---

### `piku-runtime`

The brain. Owns the agentic loop, session state, config loading, permission checks, and compaction.

**Agentic loop:**

```
user input
  → build MessageRequest (system prompt + history + tools)
  → stream from provider
  → on TextDelta:    stream to output
  → on ToolUse*:     collect tool call
  → on MessageStop:
      if tool calls present:
          check permissions for each (heuristic → classifier → user)
          execute tools (sequentially in v0, concurrent later)
          append tool results to history
          loop (bounded by max_turns, default 20)
      else:
          turn complete
```

**Permission model — three tiers:**

1. **Static heuristics (free):**
   - `read_file`, `glob`, `grep`, `list_dir` → always `Safe`
   - `write_file` on new path → `Safe`
   - `write_file` overwriting existing → `Likely`
   - `edit_file` within cwd → `Likely`
   - `edit_file` outside cwd → `Definite`
   - `bash` matching `rm`, `dd`, `mkfs`, `>` redirect to existing, `sudo` → `Definite`
   - `bash` otherwise → `Likely`

2. **Cheap AI classifier (for `Likely`):** single call to haiku/gpt-4o-mini.
   Prompt: "Is this tool call safe to auto-approve? YES or NO + one sentence reason."
   YES → proceed. NO → escalate to user.

3. **User confirmation (for `Definite` and classifier NO):** inline in message history,
   waits for `y`/`n`.

**Sub-agent recursion:** `agent` tool spawns nested runtime, max depth 3 (configurable).

**System prompt — static/dynamic split:**

```
[STATIC — prompt-cacheable]
  intro, task guidelines, safety rules

__PIKU_SYSTEM_PROMPT_DYNAMIC_BOUNDARY__

[DYNAMIC — rebuilt per session]
  environment (cwd, date, platform, model)
  project context (git status, branch)
  PIKU.md contents (walked up from cwd)
  config overrides
```

**Compaction:**
- Triggered at ~80k estimated tokens (4 chars/token heuristic)
- Summarize oldest messages via LLM call, strip `<analysis>` scratch block
- Inject summary as `System`-role message with explicit "resume directly" instruction
- Preserve N most recent messages verbatim

**Config — three layers (deep merge, last wins on conflict):**
1. `~/.config/piku/config.toml` — global user config
2. `.piku/settings.toml` — per-project config
3. `.piku/settings.local.toml` — local overrides (gitignored)

---

### `piku` (binary)

TUI and CLI entrypoint. Fast-path dispatch before building full runtime.

**Fast-path tree:**
```
piku --version      → print version, exit
piku --help         → print help, exit
piku "prompt"       → single-shot mode (self-hosting checkpoint)
piku                → full TUI REPL
```

**TUI layout (ratatui, v0):**
```
┌─────────────────────────────────────────────┐
│                                             │
│  [scrollable message history]               │
│                                             │
│  > user message                             │
│  assistant response streaming...            │
│                                             │
│  ╔ edit_file: src/main.rs ════════════════╗ │
│  ║ - old line                             ║ │
│  ║ + new line                             ║ │
│  ╚════════════════════════════════════════╝ │
│                                             │
│  [!] bash: rm -rf build/  — approve? [y/n]  │
│                                             │
├─────────────────────────────────────────────┤
│ ›  [input box]                              │
├─────────────────────────────────────────────┤
│ claude-sonnet-4-5 │ ~4.2k tok │ session-1   │
└─────────────────────────────────────────────┘
```

**Slash commands (v0):**

| Command | Description |
|---------|-------------|
| `/help` | Command reference |
| `/status` | Model, session, usage, permissions |
| `/cost` | Token usage + estimated cost |
| `/model [name]` | Show or switch model |
| `/provider [name]` | Show or switch provider |
| `/clear [--confirm]` | Clear session history |
| `/compact` | Summarize and compact old messages |
| `/session [list\|switch <id>]` | Manage sessions |
| `/diff` | Git diff of session changes |
| `/init` | Create `PIKU.md` in cwd |
| `/export [path]` | Export session transcript |
| `/exit` `/quit` | Exit |

---

## 4. Configuration Reference

`~/.config/piku/config.toml`:

```toml
[provider]
default = "anthropic"

[anthropic]
api_key = ""                   # or ANTHROPIC_API_KEY env var
model = "claude-sonnet-4-5"
base_url = "https://api.anthropic.com"

[openai]
api_key = ""                   # or OPENAI_API_KEY env var
model = "gpt-4o"
base_url = "https://api.openai.com/v1"

[ollama]
base_url = "http://localhost:11434"
model = "llama3.1"

[permissions]
classifier_model = "claude-haiku-4-5"
classifier_provider = "anthropic"

[session]
sessions_dir = "~/.config/piku/sessions"
max_turns = 20
compaction_threshold_tokens = 80000

[mcp]
servers = []
# servers = [
#   { name = "fs", command = "npx", args = ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"] },
# ]
```

---

## 5. v0 Scope

**In v0:**
- All 4 crates scaffolded and building
- Anthropic + OpenAI-compatible providers
- 7 built-in tools (read, write, edit, bash, glob, grep, list_dir)
- Full agentic loop with streaming
- Sub-agent recursion (bounded, depth 3)
- Adaptive permission model (heuristic + classifier + user confirm)
- Auto-persist sessions to `~/.config/piku/sessions/`
- PIKU.md per-project context
- Minimal ratatui TUI: scrolling history, streaming, inline diffs, status bar
- All slash commands listed above
- Single-shot CLI mode (`piku "prompt"`)

**Explicitly deferred (post-v0):**
- OAuth / browser auth (API key only in v0)
- Image input, voice, vim mode
- Multi-pane TUI
- Remote runtime
- MCP client (stub only in v0)
- Memdir long-term memory (PIKU.md only in v0)
- Concurrent tool execution (sequential in v0)
- Streaming cost estimation (approximate token count in v0)

---

## 6. Key Dependencies

```toml
# piku-api
tokio = { features = ["full"] }
reqwest = { features = ["stream", "json"] }
futures-util = "0.3"
serde = { features = ["derive"] }
serde_json = "1"

# piku-tools
tokio = { features = ["full", "process"] }
glob = "0.3"
regex = "1"
walkdir = "2"

# piku-runtime
tokio = { features = ["full"] }

# piku (binary)
ratatui = "0.29"
crossterm = "0.28"
clap = { features = ["derive"] }
tokio = { features = ["full"] }
```

---

## 7. Non-Goals

- Not a fork of Claude Code, rusty-claude-cli, or claw-code. Zero shared code.
- Not a general-purpose LLM library. Purpose-built for interactive coding agent use.
- Not a hosted service. Runs locally, talks directly to provider APIs.
- Not a plugin platform (yet). MCP is the extensibility story.
- Not cross-platform in v0. macOS first, Linux later, Windows not planned.
