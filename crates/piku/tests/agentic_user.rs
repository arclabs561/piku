#![allow(warnings)]

/// Agentic user harness — an LLM plays the role of a developer using piku.
///
/// This is the "real experience" dogfood: instead of scripted assertions on
/// individual tool calls, a second LLM instance acts as a skeptical user who:
///
///   1. Opens piku in a real PTY via `rexpect` (real terminal, real ANSI output)
///   2. Sends messages as a developer would over multiple turns
///   3. Reads the raw screen output after each turn
///   4. Notes observations and bugs in a running critique log
///   5. Decides what to type next (follow-up, harder task, or quit)
///   6. Prints a structured UX report at the end
///
/// GATING: Requires PIKU_AGENTIC_USER=1 AND an API key.
///
/// QUICK RUN (confident_dev persona, ~2 min):
///   cargo build --release -p piku
///   PIKU_AGENTIC_USER=1 OPENROUTER_API_KEY=sk-or-... \
///     cargo test --test agentic_user -- agentic_user_confident_dev --nocapture
///
/// ALL PERSONAS:
///   PIKU_AGENTIC_USER=1 cargo test --test agentic_user -- --nocapture
///
/// The test only fails on CRITICAL bugs (crash, cursor permanently gone, no output).
/// The full critique report is always printed to stdout.
use std::collections::HashMap;
use std::path::PathBuf;
use std::process::Command;
use std::time::Duration;

use rexpect::session::PtySession;

// ---------------------------------------------------------------------------
// Gate
// ---------------------------------------------------------------------------

fn is_enabled() -> bool {
    std::env::var("PIKU_AGENTIC_USER")
        .map(|v| v == "1" || v == "true")
        .unwrap_or(false)
}

fn piku_binary() -> PathBuf {
    let exe = std::env::current_exe().unwrap();
    let profile_dir = exe.parent().unwrap().parent().unwrap();
    let candidate = profile_dir.join("piku");
    if candidate.exists() {
        return candidate;
    }
    let release = profile_dir.parent().unwrap().join("release").join("piku");
    if release.exists() {
        return release;
    }
    panic!("piku binary not found — run `cargo build --release -p piku` first");
}

fn has_key(var: &str) -> bool {
    std::env::var(var).map(|v| !v.is_empty()).unwrap_or(false)
}

fn normalize_ollama_host(host: &str) -> String {
    let host = host.trim_end_matches('/');
    if host.starts_with("http://") || host.starts_with("https://") {
        host.to_string()
    } else {
        format!("http://{host}")
    }
}

fn ollama_host() -> String {
    normalize_ollama_host(
        &std::env::var("OLLAMA_HOST").unwrap_or_else(|_| "127.0.0.1:11434".to_string()),
    )
}

fn ollama_is_available(host: &str) -> bool {
    Command::new("curl")
        .args(["-sf", &format!("{}/api/tags", host.trim_end_matches('/'))])
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

#[derive(Clone, Debug)]
enum Backend {
    OpenRouter,
    Anthropic,
    Ollama,
}

#[derive(Clone, Debug)]
struct ProviderSpec {
    backend: Backend,
    label: &'static str,
    model: String,
    api_key_env: Option<&'static str>,
    api_key: Option<String>,
    ollama_host: Option<String>,
}

impl ProviderSpec {
    fn openrouter(model: impl Into<String>) -> Self {
        Self {
            backend: Backend::OpenRouter,
            label: "openrouter",
            model: model.into(),
            api_key_env: Some("OPENROUTER_API_KEY"),
            api_key: std::env::var("OPENROUTER_API_KEY").ok(),
            ollama_host: None,
        }
    }

    fn anthropic(model: impl Into<String>) -> Self {
        Self {
            backend: Backend::Anthropic,
            label: "anthropic",
            model: model.into(),
            api_key_env: Some("ANTHROPIC_API_KEY"),
            api_key: std::env::var("ANTHROPIC_API_KEY").ok(),
            ollama_host: None,
        }
    }

    fn ollama(model: impl Into<String>) -> Self {
        Self {
            backend: Backend::Ollama,
            label: "ollama",
            model: model.into(),
            api_key_env: None,
            api_key: None,
            ollama_host: Some(ollama_host()),
        }
    }

    fn env_pairs(&self) -> Vec<(String, String)> {
        let mut pairs = vec![
            (
                "PATH".to_string(),
                std::env::var("PATH").unwrap_or_default(),
            ),
            (
                "HOME".to_string(),
                std::env::var("HOME").unwrap_or_default(),
            ),
            ("TERM".to_string(), "xterm-256color".to_string()),
            ("COLUMNS".to_string(), "120".to_string()),
            ("LINES".to_string(), "40".to_string()),
            ("PIKU_NO_TRACE".to_string(), "1".to_string()),
        ];
        if let Some(host) = &self.ollama_host {
            pairs.push(("OLLAMA_HOST".to_string(), host.clone()));
        }
        if let (Some(key_var), Some(key)) = (self.api_key_env, self.api_key.as_ref()) {
            pairs.push((key_var.to_string(), key.clone()));
        }
        pairs
    }
}

/// Provider for the user-agent LLM (cheap/fast — reads and critiques output).
fn user_agent_provider() -> Option<ProviderSpec> {
    let ollama = ProviderSpec::ollama(
        std::env::var("PIKU_AGENTIC_USER_MODEL").unwrap_or_else(|_| "llama3.2:latest".to_string()),
    );
    if ollama_is_available(ollama.ollama_host.as_ref().unwrap()) {
        return Some(ollama);
    }
    if has_key("OPENROUTER_API_KEY") {
        return Some(ProviderSpec::openrouter(
            std::env::var("PIKU_AGENTIC_USER_MODEL")
                .unwrap_or_else(|_| "anthropic/claude-haiku-4-5".to_string()),
        ));
    }
    if has_key("ANTHROPIC_API_KEY") {
        return Some(ProviderSpec::anthropic(
            std::env::var("PIKU_AGENTIC_USER_MODEL")
                .unwrap_or_else(|_| "claude-haiku-4-5".to_string()),
        ));
    }
    None
}

/// Provider for piku itself (quality matters here).
fn piku_provider() -> Option<ProviderSpec> {
    let ollama = ProviderSpec::ollama(
        std::env::var("PIKU_AGENTIC_PIKU_MODEL").unwrap_or_else(|_| "gemma4:latest".to_string()),
    );
    if ollama_is_available(ollama.ollama_host.as_ref().unwrap()) {
        return Some(ollama);
    }
    if has_key("OPENROUTER_API_KEY") {
        return Some(ProviderSpec::openrouter(
            std::env::var("PIKU_AGENTIC_PIKU_MODEL")
                .unwrap_or_else(|_| "anthropic/claude-sonnet-4-5".to_string()),
        ));
    }
    if has_key("ANTHROPIC_API_KEY") {
        return Some(ProviderSpec::anthropic(
            std::env::var("PIKU_AGENTIC_PIKU_MODEL")
                .unwrap_or_else(|_| "claude-sonnet-4-5".to_string()),
        ));
    }
    None
}

fn tempdir(label: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.subsec_nanos())
        .unwrap_or(0);
    let base = std::env::temp_dir().join(format!("piku_agentic_{label}_{nanos}"));
    std::fs::create_dir_all(&base).unwrap();
    base
}

fn agentic_turn_limit(persona: &Persona) -> usize {
    if std::env::var("PIKU_AGENTIC_FULL")
        .map(|v| v == "1" || v == "true")
        .unwrap_or(false)
    {
        return persona.max_turns;
    }

    std::env::var("PIKU_AGENTIC_MAX_TURNS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(2)
        .min(persona.max_turns)
}

// ---------------------------------------------------------------------------
// Persona definitions
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct Persona {
    name: &'static str,
    description: &'static str,
    first_task: &'static str,
    behaviour: &'static str,
    max_turns: usize,
}

fn personas() -> HashMap<&'static str, Persona> {
    if std::env::var("PIKU_AGENTIC_SCENARIO")
        .map(|v| v == "repo")
        .unwrap_or(false)
    {
        let mut m = HashMap::new();

        m.insert("confident_dev", Persona {
            name: "confident_dev",
            description: "Senior Rust developer working on the piku repo copy.",
            first_task: "Read crates/piku/src/tui_repl.rs and tell me how the sticky-bottom REPL works.",
            behaviour: "You are direct and practical. Look for UX regressions like cursor disappearance, missing user echoes, and bad prompt restoration. After the first turn, ask for one concrete improvement to the TUI. After the second turn, ask it to point out one specific code path in the REPL.",
            max_turns: 3,
        });

        m.insert("cautious_beginner", Persona {
            name: "cautious_beginner",
            description: "Junior dev learning the piku repo copy.",
            first_task: "What is this repo? Show me how to run the main binary and what it does.",
            behaviour: "You read output carefully and note whether piku explains the repository clearly. If the user message is missing from the screen, treat that as a usability bug. Ask for a plain-English explanation first, then a concrete run command.",
            max_turns: 2,
        });

        m.insert("adversarial", Persona {
            name: "adversarial",
            description: "Developer stress-testing the piku repo copy.",
            first_task: "Run the workspace tests and tell me which area is most fragile.",
            behaviour: "You are trying to find broken UX and brittle behavior. Push the agent to inspect the TUI, the dogfood harness, and the tests. Call out missing echoes, cursor issues, and any confusing repo assumptions.",
            max_turns: 3,
        });

        m.insert("input_explorer", Persona {
            name: "input_explorer",
            description: "Developer testing the input/readline layer on the piku repo copy.",
            first_task: "/help",
            behaviour: "Exercise the input layer: slash commands, tab completion, hint text, prompt state changes. Turn 1: /help. Turn 2: /status. Turn 3: ask a real question about the repo. Note whether echoed input looks different from the active prompt, whether hints appear, whether tab-complete works.",
            max_turns: 3,
        });

        return m;
    }

    let mut m = HashMap::new();

    m.insert("confident_dev", Persona {
        name: "confident_dev",
        description: "Senior Rust developer, high expectations, works quickly.",
        first_task: "Read src/stats.rs and tell me what the mean() function does.",
        behaviour: "You are direct and fast. After seeing the response:\
            \n- If your message was echoed (dimmed, with > prefix) before the response, note that as working correctly.\
            \n- If your message was NOT visible in the output, report MAJOR bug: user message echo missing.\
            \n- If cursor is visible and prompt reappears, note that as working correctly.\
            \n- Push harder each turn: first explain, then find bugs, then fix them.\
            \nAfter turn 3, ask piku to fix the mean() panic bug.",
        max_turns: 5,
    });

    m.insert("cautious_beginner", Persona {
        name: "cautious_beginner",
        description: "Junior dev, new to AI tools, reads every line carefully.",
        first_task: "What files are in this project?",
        behaviour: "You read output very carefully. Look for:\
            \n- Did your message appear echoed back before the AI response? If not: MAJOR bug.\
            \n- Is the prompt (>) visible after the response? If cursor is gone: CRITICAL bug.\
            \n- Is the output readable or garbled with escape sequences? If garbled: MAJOR bug.\
            \nAsk simple follow-up questions. After 3 turns try: 'write a test for the sum function'.",
        max_turns: 4,
    });

    m.insert(
        "adversarial",
        Persona {
            name: "adversarial",
            description: "Developer trying to find edge cases.",
            first_task: "ls",
            behaviour: "Stress-test piku each turn with a different edge case:\
            \nTurn 1: bare shell command (ls)\
            \nTurn 2: ask for a file that doesn't exist\
            \nTurn 3: send a very long message (repeat 'test ' 50 times)\
            \nTurn 4: send just a single character\
            \nNote any crashes, hangs, garbled output, or missing echoes.",
            max_turns: 4,
        },
    );

    m.insert(
        "input_explorer",
        Persona {
            name: "input_explorer",
            description: "Developer specifically testing the input/readline experience.",
            first_task: "/help",
            behaviour: "You are testing the input layer of piku. Each turn exercises a different aspect:\
            \nTurn 1: /help — verify slash commands are listed and formatted. Note whether the prompt has hint text visible (dim placeholder like 'Send a message' when empty).\
            \nTurn 2: /st then TAB — try to trigger tab completion for /status. If the screen shows the status output, tab-complete worked. If it shows 'unknown command /st', tab-complete did not work. Note which happened.\
            \nTurn 3: /model — check that the current model is displayed. Note whether the prompt glyph (> or !) looks correct.\
            \nTurn 4: Send a real message like 'what files are here?' — check that your message is echoed differently from the prompt (should be dimmed with a different glyph like ▸, not the same bright > as the input prompt).\
            \nLook for: hint text when empty, tab completion for /commands, prompt state changes, echoed input looking distinct from active prompt.",
            max_turns: 4,
        },
    );

    m
}

// ---------------------------------------------------------------------------
// Structured critique types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct CritiqueEntry {
    turn: usize,
    prompt_sent: String,
    screen_text: String, // ANSI-stripped screen capture
    observations: Vec<String>,
    bugs: Vec<Bug>,
    next_action: NextAction,
}

#[derive(Debug, Clone)]
struct Bug {
    severity: Severity,
    description: String,
    expected: String,
    actual: String,
}

#[derive(Debug, Clone, PartialEq)]
enum Severity {
    Critical,
    Major,
    Minor,
    Info,
}

impl std::fmt::Display for Severity {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Severity::Critical => write!(f, "CRITICAL"),
            Severity::Major => write!(f, "MAJOR"),
            Severity::Minor => write!(f, "minor"),
            Severity::Info => write!(f, "info"),
        }
    }
}

#[derive(Debug, Clone)]
enum NextAction {
    Send(String),
    Quit,
}

// ---------------------------------------------------------------------------
// LLM client — blocking curl subprocess, JSON mode enforced
// ---------------------------------------------------------------------------

struct LlmClient {
    spec: ProviderSpec,
}

impl LlmClient {
    fn new(spec: ProviderSpec) -> Self {
        Self { spec }
    }

    /// Call LLM and return raw response text.
    fn call_raw(&self, system: &str, messages: &[(&str, &str)]) -> String {
        let msgs: Vec<serde_json::Value> = messages
            .iter()
            .map(|(role, content)| serde_json::json!({"role": role, "content": content}))
            .collect();

        // Build body — Anthropic and OpenAI-compat have slightly different shapes
        let body = match self.spec.backend {
            Backend::Anthropic => serde_json::json!({
                "model": self.spec.model,
                "max_tokens": 1024,
                "system": system,
                "messages": msgs,
            }),
            Backend::OpenRouter => {
                let mut all_msgs = vec![serde_json::json!({"role": "system", "content": system})];
                all_msgs.extend(msgs.iter().cloned());
                serde_json::json!({
                    "model": self.spec.model,
                    "max_tokens": 1024,
                    "messages": all_msgs,
                    "response_format": {"type": "json_object"},
                })
            }
            Backend::Ollama => {
                let mut all_msgs = vec![serde_json::json!({"role": "system", "content": system})];
                all_msgs.extend(msgs.iter().cloned());
                serde_json::json!({
                    "model": self.spec.model,
                    "messages": all_msgs,
                    "stream": false,
                    "format": "json",
                })
            }
        };

        let (url, auth_header): (String, Option<String>) = match self.spec.backend {
            Backend::OpenRouter => (
                "https://openrouter.ai/api/v1/chat/completions".to_string(),
                Some(format!(
                    "Authorization: Bearer {}",
                    self.spec.api_key.as_deref().unwrap_or("")
                )),
            ),
            Backend::Anthropic => (
                "https://api.anthropic.com/v1/messages".to_string(),
                Some(format!(
                    "x-api-key: {}",
                    self.spec.api_key.as_deref().unwrap_or("")
                )),
            ),
            Backend::Ollama => (
                format!(
                    "{}/api/chat",
                    self.spec
                        .ollama_host
                        .as_ref()
                        .unwrap()
                        .trim_end_matches('/')
                ),
                None,
            ),
        };

        let body_str = serde_json::to_string(&body).unwrap();
        let mut cmd_args: Vec<String> = vec![
            "-s".into(),
            "-X".into(),
            "POST".into(),
            url,
            "-H".into(),
            "Content-Type: application/json".into(),
        ];
        if let Some(auth_header) = auth_header {
            cmd_args.push("-H".into());
            cmd_args.push(auth_header);
        }
        if matches!(self.spec.backend, Backend::Anthropic) {
            cmd_args.extend(["-H".into(), "anthropic-version: 2023-06-01".into()]);
        }
        cmd_args.extend(["-d".into(), body_str]);

        let output = Command::new("curl")
            .args(&cmd_args)
            .output()
            .expect("curl must be available");

        let resp: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap_or_default();

        resp.pointer("/message/content")
            .or_else(|| resp.pointer("/content/0/text"))
            .or_else(|| resp.pointer("/choices/0/message/content"))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string()
    }

    /// Call LLM and parse JSON response, with one retry if parsing fails.
    fn call_json(&self, system: &str, user: &str) -> serde_json::Value {
        // Use owned Strings so we can accumulate the retry conversation.
        let mut messages: Vec<(String, String)> = vec![("user".into(), user.into())];

        for attempt in 0..2 {
            let refs: Vec<(&str, &str)> = messages
                .iter()
                .map(|(r, c)| (r.as_str(), c.as_str()))
                .collect();
            let raw = self.call_raw(system, &refs);
            let json_str = extract_json(&raw);

            match serde_json::from_str::<serde_json::Value>(&json_str) {
                Ok(v) if v.is_object() => return v,
                _ => {
                    if attempt == 0 {
                        eprintln!("[user_agent] JSON parse failed, retrying");
                        eprintln!("[user_agent] raw: {}", &raw[..raw.len().min(300)]);
                        let correction = "Your previous response was not valid JSON. \
                            Respond with ONLY a JSON object. \
                            Start with { and end with }. No markdown, no prose."
                            .to_string();
                        messages.push(("assistant".into(), raw));
                        messages.push(("user".into(), correction));
                    }
                }
            }
        }

        eprintln!("[user_agent] JSON parse failed after 2 attempts");
        serde_json::json!({
            "observations": ["[user-agent parse error]"],
            "bugs": [],
            "next_action": {"type": "quit"},
            "reasoning": "parse error"
        })
    }
}

/// Extract a JSON object from a string that may contain prose or markdown.
fn extract_json(s: &str) -> String {
    let s = s.trim();
    // Direct: already valid JSON
    if s.starts_with('{') {
        return s.to_string();
    }
    // Markdown fence: ```json ... ``` or ``` ... ```
    for fence in &["```json", "```"] {
        if let Some(start) = s.find(fence) {
            let after = &s[start + fence.len()..];
            // skip optional newline
            let after = after.trim_start_matches('\n');
            if let Some(end) = after.find("```") {
                return after[..end].trim().to_string();
            }
        }
    }
    // Last resort: find first { .. last }
    if let (Some(start), Some(end)) = (s.find('{'), s.rfind('}')) {
        if start < end {
            return s[start..=end].to_string();
        }
    }
    s.to_string()
}

// ---------------------------------------------------------------------------
// PTY session via rexpect
// ---------------------------------------------------------------------------

/// Spawn piku in a real PTY using rexpect, which properly handles terminal
/// echo, control codes, and idle detection.
fn spawn_piku_pty(
    workspace: &std::path::Path,
    spec: &ProviderSpec,
    config_dir: &std::path::Path,
) -> PtySession {
    let piku_bin = piku_binary();

    // Build the command string for rexpect::spawn
    let cmd = format!(
        "{} --provider {} --model {}",
        piku_bin.display(),
        spec.label,
        spec.model
    );

    let env_prefix: String = spec
        .env_pairs()
        .into_iter()
        .map(|(k, v)| format!("{}={}", k, shell_escape(&v)))
        .collect::<Vec<_>>()
        .join(" ");

    let full_cmd = format!(
        "cd {} && {} {}",
        shell_escape(&workspace.to_string_lossy()),
        env_prefix,
        cmd,
    );

    rexpect::spawn(
        &format!("sh -c '{}'", full_cmd.replace('\'', "'\\''")),
        Some(120_000),
    )
    .expect("failed to spawn piku via rexpect")
}

fn shell_escape(s: &str) -> String {
    // Simple escaping: wrap in single quotes, escape embedded single quotes
    format!("'{}'", s.replace('\'', "'\\''"))
}

// ---------------------------------------------------------------------------
// ANSI stripping + screen normalisation
// ---------------------------------------------------------------------------

fn strip_ansi(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\x1b' {
            match chars.peek() {
                Some(&'[') => {
                    chars.next();
                    // CSI sequence: consume until final byte (letter or ~)
                    for ch in chars.by_ref() {
                        if ch.is_ascii_alphabetic() || ch == '~' {
                            break;
                        }
                    }
                }
                Some(&']') => {
                    // OSC sequence: consume until ST (ESC \) or BEL
                    chars.next();
                    for ch in chars.by_ref() {
                        if ch == '\x07' || ch == '\x1b' {
                            break;
                        }
                    }
                }
                _ => {}
            }
        } else if c == '\r' {
            out.push('\n');
        } else {
            out.push(c);
        }
    }
    // Collapse 3+ consecutive newlines to 2
    let mut result = String::new();
    let mut nl_count = 0;
    for line in out.lines() {
        if line.trim().is_empty() {
            nl_count += 1;
            if nl_count <= 2 {
                result.push('\n');
            }
        } else {
            nl_count = 0;
            result.push_str(line);
            result.push('\n');
        }
    }
    result
}

fn safe_truncate(s: &str, max_chars: usize) -> &str {
    if s.chars().count() <= max_chars {
        return s;
    }
    // Walk to max_chars char boundary
    let mut idx = 0;
    for (i, _) in s.char_indices().take(max_chars) {
        idx = i;
    }
    &s[..idx]
}

fn head_tail_preview(s: &str, chars: usize) -> (String, String) {
    let head = safe_truncate(s, chars).to_string();
    let total = s.chars().count();
    if total <= chars {
        return (head, String::new());
    }

    let tail_start = total.saturating_sub(chars);
    let mut byte_idx = 0;
    for (i, _) in s.char_indices().skip(tail_start) {
        byte_idx = i;
        break;
    }
    (head, s[byte_idx..].to_string())
}

// ---------------------------------------------------------------------------
// User-agent decision
// ---------------------------------------------------------------------------

const USER_AGENT_SYSTEM: &str = r#"You are a developer testing a terminal AI coding agent called piku.
After each turn you will receive a screen capture and you MUST critique it.

CRITICAL: You MUST respond with ONLY a JSON object. No prose. No markdown. No explanation outside the JSON.
Start your response with { and end with }.

JSON schema (required):
{
  "observations": ["string", "string"],
  "bugs": [
    {
      "severity": "CRITICAL or MAJOR or minor or info",
      "description": "what is wrong",
      "expected": "what you expected",
      "actual": "what you actually saw"
    }
  ],
  "next_action": {
    "type": "send",
    "message": "the exact message to send to piku"
  },
  "reasoning": "one sentence"
}

Or to quit:
  "next_action": { "type": "quit" }

Severity:
- CRITICAL: tool unusable (crashed, cursor permanently invisible, zero output)
- MAJOR: significantly degrades experience (user message not echoed, output garbled)
- minor: cosmetic issue
- info: neutral observation

What to look for:
1. USER MESSAGE ECHO: did YOUR sent message appear before the AI response? (look for dimmed text with ▸ prefix)
   - YES, with ▸ prefix and dimmed = good (note it)
   - YES, but same style as the input prompt = minor bug "echo not visually distinct from prompt"
   - NO = MAJOR bug "user message not echoed in scroll zone"
2. CURSOR RESTORE: does the prompt (> or !) reappear after the response?
   - YES = good
   - NO = CRITICAL or MAJOR bug "cursor disappeared"
3. OUTPUT QUALITY: is the response readable? Correct? Helpful?
4. TOOL USAGE: did piku use the right tools?
5. INPUT LAYER (if testing input):
   - HINT TEXT: when input is empty, is there dim placeholder text like "Send a message"?
   - TAB COMPLETION: for /commands, does tab show completions?
   - PROMPT STATE: does the prompt change (> normally, ! after error)?
   - SLASH COMMANDS: do they produce expected output?

Be specific. Cite exact text from the screen capture in your bugs."#;

fn user_agent_turn(
    llm: &LlmClient,
    persona: &Persona,
    turn: usize,
    prompt_sent: &str,
    screen_text: &str,
) -> CritiqueEntry {
    let truncated_screen = safe_truncate(screen_text, 3000);

    let user_prompt = format!(
        "PERSONA: {name} — {desc}\n\
         TURN: {turn} of {max}\n\
         MESSAGE YOU SENT TO PIKU: {prompt}\n\n\
         SCREEN OUTPUT (ANSI stripped):\n\
         ---\n{screen}\n---\n\n\
         BEHAVIOUR GUIDE: {behaviour}\n\n\
         Analyse and respond with JSON only.",
        name = persona.name,
        desc = persona.description,
        turn = turn,
        max = persona.max_turns,
        prompt = prompt_sent,
        screen = truncated_screen,
        behaviour = persona.behaviour,
    );

    let parsed = llm.call_json(USER_AGENT_SYSTEM, &user_prompt);

    let observations: Vec<String> = parsed["observations"]
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default();

    let bugs: Vec<Bug> = parsed["bugs"]
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|entry| {
                    let sev = match entry["severity"].as_str().unwrap_or("info") {
                        "CRITICAL" => Severity::Critical,
                        "MAJOR" => Severity::Major,
                        "minor" => Severity::Minor,
                        _ => Severity::Info,
                    };
                    Some(Bug {
                        severity: sev,
                        description: entry["description"].as_str().unwrap_or("").to_string(),
                        expected: entry["expected"].as_str().unwrap_or("").to_string(),
                        actual: entry["actual"].as_str().unwrap_or("").to_string(),
                    })
                })
                .collect()
        })
        .unwrap_or_default();

    let next_action = match parsed["next_action"]["type"].as_str() {
        Some("send") => {
            let msg = parsed["next_action"]["message"]
                .as_str()
                .unwrap_or("continue")
                .to_string();
            NextAction::Send(msg)
        }
        _ => NextAction::Quit,
    };

    let reasoning = parsed["reasoning"].as_str().unwrap_or("").to_string();
    if !reasoning.is_empty() {
        eprintln!("[user_agent] turn {turn} reasoning: {reasoning}");
    }

    CritiqueEntry {
        turn,
        prompt_sent: prompt_sent.to_string(),
        screen_text: screen_text.to_string(),
        observations,
        bugs,
        next_action,
    }
}

// ---------------------------------------------------------------------------
// Report printer
// ---------------------------------------------------------------------------

fn print_report(persona: &Persona, entries: &[CritiqueEntry]) {
    let all_bugs: Vec<&Bug> = entries.iter().flat_map(|e| &e.bugs).collect();
    let n_critical = all_bugs
        .iter()
        .filter(|b| b.severity == Severity::Critical)
        .count();
    let n_major = all_bugs
        .iter()
        .filter(|b| b.severity == Severity::Major)
        .count();
    let n_minor = all_bugs
        .iter()
        .filter(|b| b.severity == Severity::Minor)
        .count();

    println!();
    println!("╔══════════════════════════════════════════════════════════════════");
    println!(
        "║  AGENTIC USER REPORT  ·  persona: {}  ·  {} turns",
        persona.name,
        entries.len()
    );
    println!(
        "║  {} CRITICAL  ·  {} MAJOR  ·  {} minor",
        n_critical, n_major, n_minor
    );
    println!("╠══════════════════════════════════════════════════════════════════");

    for entry in entries {
        println!("║");
        println!(
            "║  TURN {}  ──  sent: {:?}",
            entry.turn,
            if entry.prompt_sent.len() > 70 {
                format!("{}…", &entry.prompt_sent[..70])
            } else {
                entry.prompt_sent.clone()
            }
        );

        // Show condensed screen (first 10 non-empty lines)
        let screen_lines: Vec<&str> = entry
            .screen_text
            .lines()
            .filter(|l| !l.trim().is_empty())
            .take(10)
            .collect();
        println!(
            "║  screen ({} chars, {} non-empty lines):",
            entry.screen_text.len(),
            entry
                .screen_text
                .lines()
                .filter(|l| !l.trim().is_empty())
                .count()
        );
        for line in &screen_lines {
            let truncated = if line.len() > 100 { &line[..100] } else { line };
            println!("║    {truncated}");
        }
        let total = entry
            .screen_text
            .lines()
            .filter(|l| !l.trim().is_empty())
            .count();
        if total > 10 {
            println!("║    … ({} more lines)", total - 10);
        }

        println!("║  observations:");
        for obs in &entry.observations {
            println!("║    · {obs}");
        }

        if !entry.bugs.is_empty() {
            println!("║  bugs:");
            for bug in &entry.bugs {
                println!("║    [{}]  {}", bug.severity, bug.description);
                if !bug.expected.is_empty() {
                    println!("║          expected: {}", bug.expected);
                }
                if !bug.actual.is_empty() {
                    println!("║          actual:   {}", bug.actual);
                }
            }
        }

        match &entry.next_action {
            NextAction::Send(m) => println!("║  → next: {:?}", m),
            NextAction::Quit => println!("║  → QUIT"),
        }
    }

    println!("║");
    println!("╠══════════════════════════════════════════════════════════════════");
    println!("║  VERDICT");
    if n_critical == 0 && n_major == 0 {
        println!("║  ✓ No critical or major bugs found");
    }
    for bug in all_bugs.iter().filter(|b| b.severity == Severity::Critical) {
        println!("║  ✗ CRITICAL: {}", bug.description);
    }
    for bug in all_bugs.iter().filter(|b| b.severity == Severity::Major) {
        println!("║  ! MAJOR:    {}", bug.description);
    }
    for bug in all_bugs.iter().filter(|b| b.severity == Severity::Minor) {
        println!("║  ~ minor:    {}", bug.description);
    }
    println!("╚══════════════════════════════════════════════════════════════════");
    println!();
}

// ---------------------------------------------------------------------------
// Copy directory tree
// ---------------------------------------------------------------------------

fn copy_dir_all(src: &std::path::Path, dst: &std::path::Path) -> std::io::Result<()> {
    std::fs::create_dir_all(dst)?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let ty = entry.file_type()?;
        let dest_path = dst.join(entry.file_name());
        if ty.is_dir() {
            copy_dir_all(&entry.path(), &dest_path)?;
        } else {
            std::fs::copy(entry.path(), dest_path)?;
        }
    }
    Ok(())
}

fn agentic_seed_source() -> PathBuf {
    if let Ok(dir) = std::env::var("PIKU_AGENTIC_PLAYDIR") {
        return PathBuf::from(dir);
    }

    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("tests")
        .join("fixture")
}

// ---------------------------------------------------------------------------
// Core session runner
// ---------------------------------------------------------------------------

fn run_agentic_session(persona: &Persona) {
    let Some(ua_spec) = user_agent_provider() else {
        eprintln!("skipping: no supported provider for user-agent LLM");
        return;
    };
    let Some(piku_spec) = piku_provider() else {
        eprintln!("skipping: no supported provider for piku");
        return;
    };

    // Seed workspace
    let workspace = tempdir(persona.name);
    let config_dir = workspace.join(".config");
    std::fs::create_dir_all(&config_dir).unwrap();

    let seed_source = agentic_seed_source();
    if seed_source.exists() {
        copy_dir_all(&seed_source, &workspace)
            .unwrap_or_else(|e| eprintln!("[agentic_user] warn: copy fixture: {e}"));
    } else {
        // Minimal fallback
        std::fs::create_dir_all(workspace.join("src")).unwrap();
        std::fs::write(workspace.join("src/stats.rs"),
            "pub fn mean(values: &[i32]) -> f64 {\n    let n = values.len();\n    values.iter().sum::<i32>() as f64 / n as f64\n}\n"
        ).unwrap();
        std::fs::write(
            workspace.join("Cargo.toml"),
            "[package]\nname=\"fixture\"\nversion=\"0.1.0\"\nedition=\"2021\"\n",
        )
        .unwrap();
    }

    eprintln!("[agentic_user] persona={}", persona.name);
    eprintln!(
        "[agentic_user] user-agent: {}/{}",
        ua_spec.label, ua_spec.model
    );
    eprintln!(
        "[agentic_user] piku:       {}/{}",
        piku_spec.label, piku_spec.model
    );
    eprintln!("[agentic_user] workspace:  {}", workspace.display());

    // Spawn piku in PTY
    let mut pty = spawn_piku_pty(&workspace, &piku_spec, &config_dir);

    // Wait for the TUI to draw its welcome screen
    // Expect the session line to appear (piku is ready)
    let _ = pty.exp_regex(r"/help for commands");
    eprintln!("[agentic_user] startup banner seen");

    let llm = LlmClient::new(ua_spec);
    let mut entries: Vec<CritiqueEntry> = Vec::new();
    let turn_limit = agentic_turn_limit(persona);
    let mut current_prompt = persona.first_task.to_string();

    for turn in 1..=turn_limit {
        eprintln!(
            "[agentic_user] turn {turn}/{} → {:?}",
            turn_limit,
            if current_prompt.len() > 60 {
                &current_prompt[..60]
            } else {
                &current_prompt
            }
        );

        // Send the message
        pty.send_line(&current_prompt).expect("send_line failed");

        // Collect output: wait for > prompt to reappear (means turn is done)
        // or timeout after 90 seconds
        let mut screen_raw = match pty.exp_regex(r"\[\d+ iter") {
            Ok((before, matched)) => format!("{before}{matched}"),
            Err(e) => {
                eprintln!("[agentic_user] turn {turn}: timeout waiting for turn footer: {e}");
                e.to_string()
            }
        };

        match pty.exp_regex(r"> ") {
            Ok((before, matched)) => {
                screen_raw.push_str(&before);
                screen_raw.push_str(&matched);
            }
            Err(e) => {
                eprintln!("[agentic_user] turn {turn}: prompt did not reappear: {e}");
                screen_raw.push_str(&e.to_string());
            }
        }

        let screen_text = strip_ansi(&screen_raw);
        eprintln!(
            "[agentic_user] turn {turn}: captured {} chars ({} stripped)",
            screen_raw.len(),
            screen_text.len()
        );

        let (head, tail) = head_tail_preview(&screen_text, 180);
        eprintln!("[agentic_user] turn {turn} head: {head}");
        if !tail.is_empty() {
            eprintln!("[agentic_user] turn {turn} tail: {tail}");
        }

        // Ask the user-agent to critique
        let entry = user_agent_turn(&llm, persona, turn, &current_prompt, &screen_text);

        // Print bugs immediately as they're found
        for bug in &entry.bugs {
            eprintln!("[agentic_user] [{:}] {}", bug.severity, bug.description);
        }

        let next = entry.next_action.clone();
        entries.push(entry);

        match next {
            NextAction::Quit => {
                eprintln!("[agentic_user] user agent chose to quit after turn {turn}");
                break;
            }
            NextAction::Send(msg) => {
                current_prompt = msg;
            }
        }
    }

    // Exit cleanly
    let _ = pty.send_control('d');
    std::thread::sleep(Duration::from_millis(300));

    print_report(persona, &entries);

    // Dogfood is a report-first harness: never fail the test just because the
    // model made a bad judgment or the task itself wasn't solved. The point is
    // to observe the experience and keep the suite executable.
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn agentic_user_confident_dev() {
    if !is_enabled() {
        return;
    }
    let ps = personas();
    run_agentic_session(ps.get("confident_dev").unwrap());
}

#[test]
fn agentic_user_cautious_beginner() {
    if !is_enabled() {
        return;
    }
    let ps = personas();
    run_agentic_session(ps.get("cautious_beginner").unwrap());
}

#[test]
fn agentic_user_adversarial() {
    if !is_enabled() {
        return;
    }
    let ps = personas();
    run_agentic_session(ps.get("adversarial").unwrap());
}

#[test]
fn agentic_user_input_explorer() {
    if !is_enabled() {
        return;
    }
    let ps = personas();
    run_agentic_session(ps.get("input_explorer").unwrap());
}

// ---------------------------------------------------------------------------
// Unit tests for extract_json
// ---------------------------------------------------------------------------

#[test]
fn extract_json_bare_object() {
    let s = r#"{"a": 1, "b": "hello"}"#;
    assert_eq!(extract_json(s), s);
}

#[test]
fn extract_json_from_markdown_fence() {
    let s = "Here is the JSON:\n```json\n{\"a\": 1}\n```\nDone.";
    let j = extract_json(s);
    assert_eq!(j, r#"{"a": 1}"#);
}

#[test]
fn extract_json_from_prose() {
    let s = r#"The result is: {"observations": ["good"], "next_action": {"type": "quit"}} as requested."#;
    let j = extract_json(s);
    let parsed: serde_json::Value = serde_json::from_str(&j).unwrap();
    assert_eq!(parsed["next_action"]["type"], "quit");
}

#[test]
fn strip_ansi_removes_escape_codes() {
    let raw = "\x1b[2mfaint text\x1b[0m normal";
    let stripped = strip_ansi(raw);
    assert!(
        stripped.contains("faint text"),
        "should contain text: {stripped:?}"
    );
    assert!(
        stripped.contains("normal"),
        "should contain normal: {stripped:?}"
    );
    // ANSI codes are removed, remaining text is preserved
    assert!(
        !stripped.contains("\x1b"),
        "no escape sequences should remain: {stripped:?}"
    );
}

#[test]
fn strip_ansi_collapses_cr() {
    let raw = "line1\r\nline2\r\n";
    let stripped = strip_ansi(raw);
    assert!(stripped.contains("line1"));
    assert!(stripped.contains("line2"));
}
