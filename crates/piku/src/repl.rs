/// Interactive REPL for piku.
///
/// Drops into a persistent session loop when piku is run with no prompt argument.
/// Supports slash commands, multi-line input, and graceful exit.
use std::env;
use std::io::{self, Write};

use piku_api::TokenUsage;
use piku_runtime::{
    build_system_prompt, run_turn, AllowAll, OutputSink, PostToolAction, Session, TurnResult,
};
use piku_tools::all_tool_definitions;

use crate::cli::ResolvedProvider;
use crate::input_helper;
use crate::self_update;

// ---------------------------------------------------------------------------
// Slash commands
// ---------------------------------------------------------------------------

enum SlashCmd {
    Help,
    Status,
    Clear,
    Sessions,
    Exit,
    Model(Option<String>),
    Cost,
    Unknown(String),
}

fn parse_slash(line: &str) -> Option<SlashCmd> {
    let line = line.trim();
    if !line.starts_with('/') {
        return None;
    }
    let mut parts = line[1..].splitn(2, ' ');
    let cmd = parts.next().unwrap_or("").to_lowercase();
    let arg = parts.next().map(|s| s.trim().to_string());
    Some(match cmd.as_str() {
        "help" | "h" => SlashCmd::Help,
        "status" => SlashCmd::Status,
        "clear" => SlashCmd::Clear,
        "sessions" | "session" => SlashCmd::Sessions,
        "exit" | "quit" | "q" => SlashCmd::Exit,
        "model" => SlashCmd::Model(arg),
        "cost" => SlashCmd::Cost,
        other => SlashCmd::Unknown(other.to_string()),
    })
}

// ---------------------------------------------------------------------------
// REPL entry point
// ---------------------------------------------------------------------------

pub async fn run_repl(
    model_override: Option<&str>,
    provider_override: Option<&str>,
) -> anyhow::Result<()> {
    let resolved = ResolvedProvider::resolve(provider_override)?;
    let mut model = model_override
        .unwrap_or(&resolved.default_model)
        .to_string();

    let cwd = env::current_dir()?;
    let date = crate::current_date();

    // Create a new session
    let session_id = crate::new_session_id();
    let mut session = Session::new(session_id.clone());
    let sessions_dir = crate::sessions_dir()?;
    let session_path = sessions_dir.join(format!("{session_id}.json"));

    let mut total_usage = TokenUsage::default();

    // Print welcome banner
    println!("{}", banner(&model, resolved.name(), &session_id));

    let mut editor = input_helper::LineEditor::new("\x1b[34m❯\x1b[0m ");
    let history_path = sessions_dir.join(".repl_history");
    editor.load_history_file(&history_path);

    loop {
        let system_sections = build_system_prompt(&cwd, &date, &model, &[]);

        let readline = editor.read_line();
        match readline {
            Ok(input_helper::ReadOutcome::Submit(line)) => {
                let trimmed = line.trim().to_string();
                if trimmed.is_empty() {
                    continue;
                }
                editor.push_history(&trimmed);

                // Slash command?
                if let Some(cmd) = parse_slash(&trimmed) {
                    let current_model_snap = model.clone();
                    let should_exit = handle_slash_cmd(
                        cmd,
                        &mut session,
                        &mut total_usage,
                        &current_model_snap,
                        resolved.name(),
                        &mut model,
                    );
                    if should_exit {
                        break;
                    }
                    continue;
                }

                // Normal prompt — run a turn
                let tool_defs = all_tool_definitions();
                let prompter = AllowAll;
                let mut sink = ReplSink::new();

                println!();

                let result: TurnResult = run_turn(
                    &trimmed,
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

                println!();

                total_usage.accumulate(&result.usage);

                if let Some(err) = &result.stream_error {
                    eprintln!("\x1b[31m[error]\x1b[0m {err}");
                }

                // Persist session after each turn
                if let Err(e) = session.save(&session_path) {
                    eprintln!("\x1b[33m[warn]\x1b[0m could not save session: {e}");
                }

                // Self-update: sink detected a new binary
                if let Some(new_binary) = result.replace_and_exec {
                    eprintln!("\x1b[2m[piku] rebuilt — saving session and restarting…\x1b[0m");
                    if let Err(e) = self_update::replace_and_exec(&new_binary) {
                        eprintln!("\x1b[31m[piku]\x1b[0m self-update failed: {e}");
                    }
                    // replace_and_exec does not return on success
                }
            }

            Ok(input_helper::ReadOutcome::Cancel) => {}
            Ok(input_helper::ReadOutcome::Exit) => {
                break;
            }
            Err(err) => {
                eprintln!("\x1b[31m[input error]\x1b[0m {err}");
                break;
            }
        }
    }

    // Save session and history on clean exit
    if let Err(e) = session.save(&session_path) {
        eprintln!("\x1b[33m[warn]\x1b[0m could not save session: {e}");
    } else if !session.messages.is_empty() {
        eprintln!("\x1b[2m[session saved → {}]\x1b[0m", session_path.display());
    }
    editor.save_history_file(&history_path);

    Ok(())
}

// ---------------------------------------------------------------------------
// Slash command handler — returns true if should exit
// ---------------------------------------------------------------------------

fn handle_slash_cmd(
    cmd: SlashCmd,
    session: &mut Session,
    total_usage: &mut TokenUsage,
    current_model: &str,
    provider_name: &str,
    model: &mut String,
) -> bool {
    match cmd {
        SlashCmd::Help => {
            println!(
                "\n\x1b[1mSlash commands:\x1b[0m
  /help          This message
  /status        Session info (model, provider, messages, tokens)
  /cost          Token usage for this session
  /model [name]  Show or switch model
  /sessions      List saved sessions
  /clear         Clear session history (start fresh)
  /exit, /quit   Exit piku\n"
            );
        }
        SlashCmd::Status => {
            println!(
                "\n\x1b[1mStatus:\x1b[0m
  Provider:  {provider_name}
  Model:     {current_model}
  Session:   {}
  Messages:  {}
  Est. tokens: {}
  Usage ↑:   {}  ↓: {}\n",
                session.id,
                session.messages.len(),
                session.estimated_tokens(),
                total_usage.input_tokens,
                total_usage.output_tokens,
            );
        }
        SlashCmd::Cost => {
            println!(
                "\n\x1b[1mToken usage:\x1b[0m
  Input:  {}
  Output: {}
  Total:  {}
  Cache reads:    {}
  Cache creates:  {}\n",
                total_usage.input_tokens,
                total_usage.output_tokens,
                total_usage.total_tokens(),
                total_usage.cache_read_input_tokens,
                total_usage.cache_creation_input_tokens,
            );
        }
        SlashCmd::Model(None) => {
            println!("\nCurrent model: {current_model}\n");
        }
        SlashCmd::Model(Some(new_model)) => {
            new_model.clone_into(model);
            println!("\x1b[2m[model → {new_model}]\x1b[0m");
        }
        SlashCmd::Sessions => {
            if let Ok(dir) = crate::sessions_dir() {
                println!("\n\x1b[1mSaved sessions:\x1b[0m  ({})\n", dir.display());
                if let Ok(entries) = std::fs::read_dir(&dir) {
                    let mut files: Vec<_> = entries
                        .filter_map(std::result::Result::ok)
                        .filter(|e| e.path().extension().is_some_and(|x| x == "json"))
                        .collect();
                    files.sort_by_key(|e| {
                        e.metadata()
                            .and_then(|m| m.modified())
                            .unwrap_or(std::time::SystemTime::UNIX_EPOCH)
                    });
                    files.reverse();
                    if files.is_empty() {
                        println!("  (none)");
                    } else {
                        for f in files.iter().take(20) {
                            let raw_name = f.file_name();
                            let name = raw_name.to_string_lossy();
                            let name = name.trim_end_matches(".json");
                            let size = f.metadata().map(|m| m.len()).unwrap_or(0);
                            println!("  {name}  \x1b[2m({size}b)\x1b[0m");
                        }
                    }
                }
                println!();
            }
        }
        SlashCmd::Clear => {
            let old_id = session.id.clone();
            session.messages.clear();
            session.id = crate::new_session_id();
            *total_usage = TokenUsage::default();
            println!("\x1b[2m[session cleared  {old_id} → {}]\x1b[0m", session.id);
        }
        SlashCmd::Exit => {
            return true;
        }
        SlashCmd::Unknown(name) => {
            println!("\x1b[33m[unknown command]\x1b[0m /{name} — try /help");
        }
    }
    false
}

// ---------------------------------------------------------------------------
// REPL output sink — richer than single-shot StdoutSink
// ---------------------------------------------------------------------------

pub struct ReplSink {
    stdout: io::Stdout,
    md: crate::markdown::StreamingMarkdown,
}

impl Default for ReplSink {
    fn default() -> Self {
        Self::new()
    }
}

impl ReplSink {
    #[must_use]
    pub fn new() -> Self {
        Self {
            stdout: io::stdout(),
            md: crate::markdown::StreamingMarkdown::new_stdout(),
        }
    }
}

impl OutputSink for ReplSink {
    fn on_text(&mut self, text: &str) {
        let rendered = self.md.push(text);
        if !rendered.is_empty() {
            let _ = self.stdout.write_all(rendered.as_bytes());
            let _ = self.stdout.flush();
        }
    }

    fn on_tool_start(&mut self, tool_name: &str, tool_id: &str, input: &serde_json::Value) {
        // Flush pending markdown before tool label
        let flushed = self.md.flush();
        if !flushed.is_empty() {
            let _ = self.stdout.write_all(flushed.as_bytes());
        }
        let icon = tool_icon(tool_name);
        let args = crate::format_tool_input(tool_name, input);
        let line = if args.is_empty() {
            format!("\n\x1b[2m{icon} {tool_name} …\x1b[0m")
        } else {
            format!("\n\x1b[2m{icon} {tool_name} {args} …\x1b[0m")
        };
        let _ = writeln!(self.stdout, "{line}");
        let _ = self.stdout.flush();
        let _ = tool_id;
    }

    fn on_tool_end(&mut self, tool_name: &str, result: &str, is_error: bool) -> PostToolAction {
        let (status_color, tag) = if is_error {
            ("\x1b[31m", "err")
        } else {
            ("\x1b[32m", "ok")
        };

        let preview = format_tool_result(tool_name, result, is_error);

        let _ = writeln!(
            self.stdout,
            "\x1b[2m[{tool_name} → {status_color}{tag}\x1b[0m\x1b[2m]\x1b[0m",
        );
        if !preview.is_empty() {
            let _ = writeln!(self.stdout, "{preview}");
        }
        let _ = self.stdout.flush();

        if tool_name == "bash" {
            if let Some(new_binary) = self_update::detect_self_build(result, !is_error) {
                eprintln!(
                    "\x1b[2m[piku] detected self-rebuild → {}\x1b[0m",
                    new_binary.display()
                );
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
    }

    fn on_turn_complete(&mut self, usage: &TokenUsage, iterations: u32) {
        let flushed = self.md.flush();
        if !flushed.is_empty() {
            let _ = self.stdout.write_all(flushed.as_bytes());
        }
        let _ = writeln!(
            self.stdout,
            "\n\x1b[2m[{iterations} iter · {}↑ {}↓]\x1b[0m",
            usage.input_tokens, usage.output_tokens
        );
        let _ = self.stdout.flush();
    }
}

// ---------------------------------------------------------------------------
// Tool result formatting
// ---------------------------------------------------------------------------

fn tool_icon(tool_name: &str) -> &'static str {
    match tool_name {
        "read_file" => "📄",
        "write_file" => "✏️ ",
        "edit_file" => "✂️ ",
        "bash" => "🔧",
        "glob" => "🔍",
        "grep" => "🔎",
        "list_dir" => "📁",
        _ => "⚙️ ",
    }
}

/// Format a tool result for display in the REPL.
/// - `edit_file`: show a compact diff-style preview
/// - bash: show output lines, capped
/// - others: brief preview
fn format_tool_result(tool_name: &str, result: &str, is_error: bool) -> String {
    const MAX_LINES: usize = 12;
    const MAX_CHARS: usize = 600;

    if is_error {
        // Always show errors in full (up to cap)
        return truncate_with_notice(result, MAX_CHARS);
    }

    match tool_name {
        "bash" => {
            // Show stdout/stderr lines, capped
            let lines: Vec<&str> = result.lines().collect();
            if lines.is_empty() {
                return String::new();
            }
            let shown: Vec<&str> = lines.iter().copied().take(MAX_LINES).collect();
            let mut out = shown
                .iter()
                .map(|l| format!("  \x1b[2m{l}\x1b[0m"))
                .collect::<Vec<_>>()
                .join("\n");
            if lines.len() > MAX_LINES {
                out.push_str(&format!(
                    "\n  \x1b[2m… {} more lines\x1b[0m",
                    lines.len() - MAX_LINES
                ));
            }
            out
        }
        "read_file" => {
            // Show line count and first few lines
            let lines: Vec<&str> = result.lines().collect();
            let shown = lines.iter().copied().take(6).collect::<Vec<_>>();
            let mut out = shown
                .iter()
                .map(|l| format!("  \x1b[2m{l}\x1b[0m"))
                .collect::<Vec<_>>()
                .join("\n");
            if lines.len() > 6 {
                out.push_str(&format!("\n  \x1b[2m… {} lines total\x1b[0m", lines.len()));
            }
            out
        }
        "edit_file" | "write_file" => {
            // These return success messages — show them compactly
            format!("  \x1b[2m{}\x1b[0m", result.trim())
        }
        "glob" | "list_dir" => {
            // File lists — show line count
            let lines: Vec<&str> = result.lines().collect();
            let count = lines.len();
            let shown = lines.iter().copied().take(8).collect::<Vec<_>>();
            let mut out = shown
                .iter()
                .map(|l| format!("  \x1b[2m{l}\x1b[0m"))
                .collect::<Vec<_>>()
                .join("\n");
            if count > 8 {
                out.push_str(&format!("\n  \x1b[2m… {count} total\x1b[0m"));
            }
            out
        }
        "grep" => {
            let lines: Vec<&str> = result.lines().collect();
            let count = lines.len();
            let shown = lines.iter().copied().take(MAX_LINES).collect::<Vec<_>>();
            let mut out = shown
                .iter()
                .map(|l| format!("  \x1b[2m{l}\x1b[0m"))
                .collect::<Vec<_>>()
                .join("\n");
            if count > MAX_LINES {
                out.push_str(&format!("\n  \x1b[2m… {count} matches total\x1b[0m"));
            }
            out
        }
        _ => truncate_with_notice(result, MAX_CHARS),
    }
}

fn truncate_with_notice(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!(
            "{}…\x1b[2m [{} chars truncated]\x1b[0m",
            &s[..max],
            s.len() - max
        )
    }
}

// ---------------------------------------------------------------------------
// Banner
// ---------------------------------------------------------------------------

fn banner(model: &str, provider: &str, session_id: &str) -> String {
    format!(
        "\x1b[1mpiku\x1b[0m  \x1b[2m{provider} · {model}\x1b[0m\n\
         \x1b[2msession: {session_id}\x1b[0m\n\
         \x1b[2mType a message, or /help for commands. Ctrl-D to exit.\x1b[0m\n"
    )
}

// ---------------------------------------------------------------------------
// rustyline editor factory
// ---------------------------------------------------------------------------

// Editor factory moved to crate::input_helper::build_editor()
