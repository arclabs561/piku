use piku::cli::{parse_args, CliAction};
use piku::config::PikuConfig;
use piku::self_update;
use piku::trace::TraceWriter;
use piku::tui_repl;

use std::env;
use std::io::{self, Write};

use piku_runtime::{build_system_prompt, run_turn, AllowAll, OutputSink, Session, TurnResult};
use piku_runtime::{provider_availability, PostToolAction, ResolvedProvider, TokenUsage};
use piku_tools::all_tool_definitions;

const VERSION: &str = env!("CARGO_PKG_VERSION"); // self-update demo

// ---------------------------------------------------------------------------
// Entry point — fast-path dispatch
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args: Vec<String> = env::args().skip(1).collect();
    let action = parse_args(&args);

    // Extract CLI-level overrides for config loading.
    let (cli_model, cli_provider) = match &action {
        CliAction::SingleShot {
            model,
            provider_override,
            ..
        }
        | CliAction::Resume {
            model,
            provider_override,
            ..
        }
        | CliAction::Repl {
            model,
            provider_override,
            ..
        } => (model.as_deref(), provider_override.as_deref()),
        _ => (None, None),
    };

    let cwd = env::current_dir().ok();
    let config = PikuConfig::load(cli_provider, cli_model, cwd.as_deref());

    match action {
        CliAction::Version => println!("piku {VERSION}"),
        CliAction::Help => print_help(),
        CliAction::Providers => print_providers(),
        CliAction::ArgError(msg) => {
            eprintln!("error: {msg}");
            eprintln!("Run `piku --help` for usage.");
            std::process::exit(1);
        }
        CliAction::SingleShot {
            prompt,
            print,
            read_only,
            ..
        } => {
            if !print && !read_only && self_update::was_restarted() {
                if let Some(session) = try_load_restart_session(&config) {
                    return run_tui_repl_post_restart(session, &config, read_only).await;
                }
            }
            run_single_shot(&prompt, None, &config, print, read_only).await?;
        }
        CliAction::Resume {
            session_id,
            prompt,
            print,
            read_only,
            ..
        } => {
            run_resume(&session_id, prompt.as_deref(), &config, print, read_only).await?;
        }
        CliAction::Repl { read_only, .. } => {
            if self_update::was_restarted() {
                if let Some(session) = try_load_restart_session(&config) {
                    return run_tui_repl_post_restart(session, &config, read_only).await;
                }
            }
            tui_repl::run_tui_repl_with_mode(&config, read_only).await?;
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Post-restart helpers (self-update seamless resume)
// ---------------------------------------------------------------------------

/// Load the session indicated by `PIKU_SESSION_ID` (set by `tui_repl` before exec).
/// Returns None if the env var is missing or the session can't be loaded.
fn try_load_restart_session(config: &PikuConfig) -> Option<Session> {
    let session_id = std::env::var("PIKU_SESSION_ID").ok()?;
    std::env::remove_var("PIKU_SESSION_ID");
    let sessions_dir = config.sessions_dir();
    let path = sessions_dir.join(format!("{session_id}.json"));
    match Session::load(&path) {
        Ok(s) => Some(s),
        Err(e) => {
            eprintln!("[piku] could not reload session after restart: {e}");
            None
        }
    }
}

/// Enter the TUI REPL with a session that was just restored after a self-rebuild.
async fn run_tui_repl_post_restart(
    session: Session,
    config: &PikuConfig,
    read_only: bool,
) -> anyhow::Result<()> {
    tui_repl::run_tui_repl_post_restart(config, Some(session), read_only).await
}

// ---------------------------------------------------------------------------
// Single-shot mode
// ---------------------------------------------------------------------------

async fn run_resume(
    session_id: &str,
    prompt: Option<&str>,
    config: &PikuConfig,
    print: bool,
    read_only: bool,
) -> anyhow::Result<()> {
    let sessions_dir = config.sessions_dir();
    std::fs::create_dir_all(&sessions_dir)?;
    let session_path = sessions_dir.join(format!("{session_id}.json"));

    if !session_path.exists() {
        // Try partial match: find any session file whose name contains session_id
        let matches: Vec<_> = std::fs::read_dir(&sessions_dir)?
            .filter_map(std::result::Result::ok)
            .filter(|e| e.file_name().to_string_lossy().contains(session_id))
            .collect();

        match matches.len() {
            0 => anyhow::bail!(
                "session '{session_id}' not found in {}\n\
                 List saved sessions with: ls {}",
                sessions_dir.display(),
                sessions_dir.display()
            ),
            1 => {
                let matched_path = matches[0].path();
                eprintln!("[piku] resuming {}", matched_path.display());
                let session = Session::load(&matched_path)
                    .map_err(|e| anyhow::anyhow!("failed to load session: {e}"))?;
                return run_single_shot(
                    prompt.unwrap_or("Continue where we left off."),
                    Some(session),
                    config,
                    print,
                    read_only,
                )
                .await;
            }
            n => {
                eprintln!("error: '{session_id}' is ambiguous — {n} sessions match:");
                for m in &matches {
                    eprintln!("  {}", m.file_name().to_string_lossy());
                }
                anyhow::bail!("use a more specific session ID");
            }
        }
    }

    eprintln!("[piku] resuming {}", session_path.display());
    let session =
        Session::load(&session_path).map_err(|e| anyhow::anyhow!("failed to load session: {e}"))?;

    eprintln!(
        "[piku] loaded {} messages from prior session",
        session.messages.len()
    );

    run_single_shot(
        prompt.unwrap_or("Continue where we left off."),
        Some(session),
        config,
        print,
        read_only,
    )
    .await
}

// ---------------------------------------------------------------------------
// Single-shot mode
//
// Runs one prompt to completion. In headless mode (`print = true`, set by
// `-p`/`--print`) it exits afterward (aider `-m` / `claude -p` / `codex exec`
// style). Otherwise it drops into the TUI REPL with the same session so the
// conversation can continue interactively.
// ---------------------------------------------------------------------------

async fn run_single_shot(
    prompt: &str,
    existing_session: Option<Session>,
    config: &PikuConfig,
    print: bool,
    read_only: bool,
) -> anyhow::Result<()> {
    let resolved = ResolvedProvider::resolve(config.provider.as_deref())?;
    let model = config
        .model
        .as_deref()
        .unwrap_or(&resolved.default_model)
        .to_string();

    eprintln!("[piku] provider={} model={model}", resolved.name());

    let cwd = env::current_dir()?;
    let date = current_date();
    let mut system_sections = build_system_prompt(&cwd, &date, &model, &[]);
    if read_only {
        system_sections.push(piku::read_only_system_prompt_section());
    }

    let (session_id, mut session) = if let Some(s) = existing_session {
        eprintln!("[piku] continuing session {}", s.id);
        (s.id.clone(), s)
    } else {
        let id = new_session_id();
        (id.clone(), Session::new(id))
    };

    let tool_defs = if read_only {
        eprintln!("[piku] read-only mode: file-inspection tools only");
        piku_tools::read_only_tool_definitions()
    } else {
        all_tool_definitions()
    };
    let prompter = AllowAll;
    let traces_dir = config.traces_dir();
    std::fs::create_dir_all(&traces_dir).ok();
    let trace = TraceWriter::open(&traces_dir, &session_id);
    let mut sink = StdoutSink::new(trace);
    sink.trace.prompt(prompt);

    let result: TurnResult = run_turn(
        prompt,
        &mut session,
        resolved.as_provider(),
        &model,
        &system_sections,
        tool_defs,
        &prompter,
        &mut sink,
        None,
        None,
    )
    .await;

    if let Some(err) = &result.stream_error {
        eprintln!("[piku] stream error: {err}");
    }

    // Persist session BEFORE self-update — nothing should be lost on restart
    let sessions_dir = config.sessions_dir();
    std::fs::create_dir_all(&sessions_dir)?;
    let session_path = sessions_dir.join(format!("{session_id}.json"));
    if let Err(e) = session.save(&session_path) {
        eprintln!("warning: could not save session: {e}");
    } else {
        eprintln!("[piku] session saved → {}", session_path.display());
    }

    // Self-update: sink detected a new binary — replace and exec
    if let Some(new_binary) = result.replace_and_exec {
        eprintln!("[piku] rebuilt — restarting with new binary...");
        if let Err(e) = self_update::replace_and_exec(&new_binary) {
            eprintln!("[piku] self-update failed: {e}");
            eprintln!("[piku] continuing with old binary");
        }
        // replace_and_exec does not return on success
    }

    // Check if this is a post-restart invocation
    if self_update::was_restarted() {
        eprintln!("[piku] restarted after self-rebuild ✓");
    }

    // Headless: stop here. The turn output has already streamed to stdout and
    // the session is saved; entering the REPL would block on a (likely
    // non-TTY) stdin, which is exactly what scripts and pipelines don't want.
    if print || read_only {
        return Ok(());
    }

    // Drop into TUI REPL with the same session for continued conversation
    println!(); // blank line between single-shot output and REPL
    tui_repl::run_tui_repl_with_session(config, Some(session), result.usage).await
}

// ---------------------------------------------------------------------------
// Stdout sink
// ---------------------------------------------------------------------------

struct StdoutSink {
    stdout: io::Stdout,
    trace: TraceWriter,
    /// Track `tool_id` so `on_tool_end` can log it (the trait doesn't pass it there).
    /// Maps `tool_name` → most recent `tool_id` (good enough for sequential execution).
    pending_tool_id: std::collections::HashMap<String, String>,
    /// Streaming markdown renderer.
    md: piku::markdown::StreamingMarkdown,
}

impl StdoutSink {
    fn new(trace: TraceWriter) -> Self {
        Self {
            stdout: io::stdout(),
            trace,
            pending_tool_id: std::collections::HashMap::new(),
            md: piku::markdown::StreamingMarkdown::new_stdout(),
        }
    }
}

impl OutputSink for StdoutSink {
    fn on_text(&mut self, text: &str) {
        let rendered = self.md.push(text);
        if !rendered.is_empty() {
            let _ = self.stdout.write_all(rendered.as_bytes());
            let _ = self.stdout.flush();
        }
        self.trace.text_chunk(text);
    }

    fn on_tool_start(&mut self, tool_name: &str, tool_id: &str, input: &serde_json::Value) {
        let flushed = self.md.flush();
        if !flushed.is_empty() {
            let _ = self.stdout.write_all(flushed.as_bytes());
        }
        let args = piku::format_tool_input(tool_name, input);
        let line = if args.is_empty() {
            format!("\n\x1b[2m[{tool_name} …]\x1b[0m")
        } else {
            format!("\n\x1b[2m[{tool_name} {args} …]\x1b[0m")
        };
        let _ = writeln!(self.stdout, "{line}");
        let _ = self.stdout.flush();

        self.trace.tool_start(tool_name, tool_id, input);
        self.pending_tool_id
            .insert(tool_name.to_string(), tool_id.to_string());
    }

    fn on_tool_end(&mut self, tool_name: &str, result: &str, is_error: bool) -> PostToolAction {
        let tag = if is_error {
            "\x1b[31merr\x1b[0m"
        } else {
            "\x1b[32mok\x1b[0m"
        };
        let preview = if result.len() > 400 {
            format!("{}…", &result[..400])
        } else {
            result.to_string()
        };
        let _ = writeln!(
            self.stdout,
            "\x1b[2m[{tool_name} → {tag}]\x1b[0m\n{preview}\n"
        );
        let _ = self.stdout.flush();

        let tool_id = self.pending_tool_id.remove(tool_name).unwrap_or_default();
        self.trace.tool_end(tool_name, &tool_id, result, !is_error);

        if tool_name == "bash" {
            if let Some(new_binary) = self_update::detect_self_build(result, !is_error) {
                eprintln!("[piku] detected self-rebuild → {}", new_binary.display());
                return PostToolAction::ReplaceAndExec(new_binary);
            }
        }

        PostToolAction::Continue
    }

    fn on_permission_denied(&mut self, tool_name: &str, reason: &str) {
        let _ = writeln!(
            self.stdout,
            "\x1b[33m[permission denied: {tool_name}]\x1b[0m {reason}"
        );
        let _ = self.stdout.flush();
        self.trace.permission_denied(tool_name, reason);
    }

    fn on_turn_complete(&mut self, usage: &TokenUsage, iterations: u32) {
        let flushed = self.md.flush();
        if !flushed.is_empty() {
            let _ = self.stdout.write_all(flushed.as_bytes());
        }
        let _ = writeln!(
            self.stdout,
            "\n\x1b[2m[{iterations} iter · {}↑ {}↓ tokens]\x1b[0m",
            usage.input_tokens, usage.output_tokens
        );
        let _ = self.stdout.flush();
        self.trace
            .turn_end(iterations, usage.input_tokens, usage.output_tokens);
    }
}

// ---------------------------------------------------------------------------
// Help
// ---------------------------------------------------------------------------

fn print_help() {
    use piku_runtime::{
        DEFAULT_MODEL_ANTHROPIC, DEFAULT_MODEL_GROQ, DEFAULT_MODEL_OLLAMA, DEFAULT_MODEL_OPENROUTER,
    };
    println!(
        "piku {VERSION} — terminal AI coding agent

USAGE:
    piku [OPTIONS] [PROMPT]

OPTIONS:
    -p, --print          Headless: run the prompt, print the result, and exit
                         (no interactive REPL). For scripts and pipelines.
    --read-only          Only read_file, glob, grep, and list_dir may run
    --model <name>       Override model (default: provider-dependent)
    --provider <name>    Force provider: openrouter | anthropic | groq | ollama | custom
    --providers          Show provider status and exit
    --resume <id>        Resume a previous session by ID (partial match ok)
    --version            Print version
    --help               Print this help

PROVIDER SELECTION (opportunistic — first available wins):
    PIKU_BASE_URL        → custom OpenAI-compatible server
    OPENROUTER_API_KEY   → openrouter  (default: {DEFAULT_MODEL_OPENROUTER})
    ANTHROPIC_API_KEY    → anthropic   (default: {DEFAULT_MODEL_ANTHROPIC})
    GROQ_API_KEY         → groq        (default: {DEFAULT_MODEL_GROQ})
    OLLAMA_HOST          → ollama      (default: {DEFAULT_MODEL_OLLAMA}, no key needed)

EXAMPLES:
    piku \"explain src/main.rs\"
    piku -p \"explain src/main.rs\" > explanation.txt   # headless, exits after
    piku --model anthropic/claude-opus-4 \"refactor the permission system\"
    piku --provider anthropic \"what does the agentic loop do?\"

NOTES:
    Sessions are auto-saved to ~/.config/piku/sessions/
    Per-project context: add a PIKU.md file in your project root"
    );
}

fn print_providers() {
    println!("PROVIDERS:");
    for provider in provider_availability() {
        let marker = if provider.available {
            "available"
        } else {
            "missing"
        };
        println!(
            "    {:<10} {:<9} default={} ({})",
            provider.name, marker, provider.default_model, provider.note
        );
    }
}

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

fn new_session_id() -> String {
    piku::new_session_id()
}

fn current_date() -> String {
    piku::current_date()
}
