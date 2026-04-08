use piku::cli::{parse_args, CliAction, ResolvedProvider};
use piku::self_update;
use piku::trace::TraceWriter;
use piku::tui_repl;

use std::env;
use std::io::{self, Write};

use piku_api::TokenUsage;
use piku_runtime::PostToolAction;
use piku_runtime::{build_system_prompt, run_turn, AllowAll, OutputSink, Session, TurnResult};
use piku_tools::all_tool_definitions;

const VERSION: &str = env!("CARGO_PKG_VERSION"); // self-update demo

// ---------------------------------------------------------------------------
// Entry point — fast-path dispatch
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args: Vec<String> = env::args().skip(1).collect();

    match parse_args(&args) {
        CliAction::Version => println!("piku {VERSION}"),
        CliAction::Help => print_help(),
        CliAction::ArgError(msg) => {
            eprintln!("error: {msg}");
            eprintln!("Run `piku --help` for usage.");
            std::process::exit(1);
        }
        CliAction::SingleShot {
            prompt,
            model,
            provider_override,
        } => {
            // If we're restarting after a self-build, recover the session
            // instead of re-running the original prompt.
            if self_update::was_restarted() {
                if let Some(session) = try_load_restart_session() {
                    return run_tui_repl_post_restart(
                        session,
                        model.as_deref(),
                        provider_override.as_deref(),
                    )
                    .await;
                }
            }
            run_single_shot_then_repl(
                &prompt,
                None,
                model.as_deref(),
                provider_override.as_deref(),
            )
            .await?;
        }
        CliAction::Resume {
            session_id,
            prompt,
            model,
            provider_override,
        } => {
            run_resume(
                &session_id,
                prompt.as_deref(),
                model.as_deref(),
                provider_override.as_deref(),
            )
            .await?;
        }
        CliAction::Repl {
            model,
            provider_override,
        } => {
            // If we're restarting after a self-build, reload the session.
            if self_update::was_restarted() {
                if let Some(session) = try_load_restart_session() {
                    return run_tui_repl_post_restart(
                        session,
                        model.as_deref(),
                        provider_override.as_deref(),
                    )
                    .await;
                }
            }
            tui_repl::run_tui_repl(model.as_deref(), provider_override.as_deref()).await?;
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Post-restart helpers (self-update seamless resume)
// ---------------------------------------------------------------------------

/// Load the session indicated by PIKU_SESSION_ID (set by tui_repl before exec).
/// Returns None if the env var is missing or the session can't be loaded.
fn try_load_restart_session() -> Option<Session> {
    let session_id = std::env::var("PIKU_SESSION_ID").ok()?;
    std::env::remove_var("PIKU_SESSION_ID");
    let sessions_dir = sessions_dir().ok()?;
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
/// Prints a restart banner in the scroll zone.
async fn run_tui_repl_post_restart(
    session: Session,
    model_override: Option<&str>,
    provider_override: Option<&str>,
) -> anyhow::Result<()> {
    tui_repl::run_tui_repl_post_restart(model_override, provider_override, Some(session)).await
}

// ---------------------------------------------------------------------------
// Single-shot mode
// ---------------------------------------------------------------------------

async fn run_resume(
    session_id: &str,
    prompt: Option<&str>,
    model_override: Option<&str>,
    provider_override: Option<&str>,
) -> anyhow::Result<()> {
    let sessions_dir = sessions_dir()?;
    let session_path = sessions_dir.join(format!("{session_id}.json"));

    if !session_path.exists() {
        // Try partial match: find any session file whose name contains session_id
        let matches: Vec<_> = std::fs::read_dir(&sessions_dir)?
            .filter_map(|e| e.ok())
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
                return run_single_shot_then_repl(
                    prompt.unwrap_or("Continue where we left off."),
                    Some(session),
                    model_override,
                    provider_override,
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

    run_single_shot_then_repl(
        prompt.unwrap_or("Continue where we left off."),
        Some(session),
        model_override,
        provider_override,
    )
    .await
}

// ---------------------------------------------------------------------------
// Single-shot mode → drops into TUI REPL for continued chat
// ---------------------------------------------------------------------------

async fn run_single_shot_then_repl(
    prompt: &str,
    existing_session: Option<Session>,
    model_override: Option<&str>,
    provider_override: Option<&str>,
) -> anyhow::Result<()> {
    let resolved = ResolvedProvider::resolve(provider_override)?;
    let model = model_override
        .unwrap_or(&resolved.default_model)
        .to_string();

    eprintln!("[piku] provider={} model={model}", resolved.name());

    let cwd = env::current_dir()?;
    let date = current_date();
    let system_sections = build_system_prompt(&cwd, &date, &model, &[]);

    let (session_id, mut session) = if let Some(s) = existing_session {
        eprintln!("[piku] continuing session {}", s.id);
        (s.id.clone(), s)
    } else {
        let id = new_session_id();
        (id.clone(), Session::new(id))
    };

    let tool_defs = all_tool_definitions();
    let prompter = AllowAll;
    let trace = traces_dir()
        .map(|dir| TraceWriter::open(&dir, &session_id))
        .unwrap_or_else(|_| TraceWriter::disabled());
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
    let sessions_dir = sessions_dir()?;
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

    // Drop into TUI REPL with the same session for continued conversation
    println!(); // blank line between single-shot output and REPL
    tui_repl::run_tui_repl_with_session(
        model_override,
        provider_override,
        Some(session),
        result.usage,
    )
    .await
}

// ---------------------------------------------------------------------------
// Stdout sink
// ---------------------------------------------------------------------------

struct StdoutSink {
    stdout: io::Stdout,
    trace: TraceWriter,
    /// Track tool_id so on_tool_end can log it (the trait doesn't pass it there).
    /// Maps tool_name → most recent tool_id (good enough for sequential execution).
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
    use piku::cli::{
        DEFAULT_MODEL_ANTHROPIC, DEFAULT_MODEL_GROQ, DEFAULT_MODEL_OLLAMA, DEFAULT_MODEL_OPENROUTER,
    };
    println!(
        "piku {VERSION} — terminal AI coding agent

USAGE:
    piku [OPTIONS] [PROMPT]

OPTIONS:
    --model <name>       Override model (default: provider-dependent)
    --provider <name>    Force provider: openrouter | anthropic | groq | ollama | custom
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
    piku --model anthropic/claude-opus-4 \"refactor the permission system\"
    piku --provider anthropic \"what does the agentic loop do?\"

NOTES:
    Sessions are auto-saved to ~/.config/piku/sessions/
    Per-project context: add a PIKU.md file in your project root"
    );
}

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

fn sessions_dir() -> anyhow::Result<std::path::PathBuf> {
    piku::sessions_dir()
}

fn traces_dir() -> anyhow::Result<std::path::PathBuf> {
    let base = env::var("XDG_CONFIG_HOME")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| {
            env::var("HOME")
                .map(|h| std::path::PathBuf::from(h).join(".config"))
                .unwrap_or_else(|_| std::path::PathBuf::from(".config"))
        });
    let path = base.join("piku").join("traces");
    std::fs::create_dir_all(&path)?;
    Ok(path)
}

fn new_session_id() -> String {
    piku::new_session_id()
}

fn current_date() -> String {
    piku::current_date()
}
