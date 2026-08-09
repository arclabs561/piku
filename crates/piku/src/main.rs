use clap::Parser;

use piku::cli::{Cli, Commands, InspectFormat};
use piku::config::{load_provider_dotenv, PikuConfig};
use piku::self_update;
use piku::trace::TraceWriter;
use piku::tui_repl;

use std::env;
use std::io::{self, Write};

use piku_runtime::{
    build_system_prompt, run_turn, AllowAll, OutputSink, RecordingSink, RunRecorder, Session,
    TurnResult,
};
use piku_runtime::{provider_availability, PostToolAction, ResolvedProvider, TokenUsage};
use piku_tools::all_tool_definitions;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    piku::telemetry::init();
    let cwd = env::current_dir().ok();
    if env::var_os("PIKU_NO_DOTENV").is_none() {
        match load_provider_dotenv(cwd.as_deref()) {
            Ok(0) => {}
            Ok(count) => {
                tracing::info!(count, source = "nearest_dotenv", "provider settings loaded");
            }
            Err(error) => {
                tracing::warn!(%error, "provider dotenv could not be read");
            }
        }
    }
    let cli = Cli::parse();

    match &cli.command {
        Some(Commands::Providers) => {
            print_providers();
            return Ok(());
        }
        Some(Commands::Inspect {
            session_id,
            json,
            html,
        }) => {
            let config = PikuConfig::load(None, None, env::current_dir().ok().as_deref());
            let format = if *html {
                InspectFormat::Html
            } else if *json {
                InspectFormat::Json
            } else {
                InspectFormat::Text
            };
            inspect_run(&config, session_id, format)?;
            return Ok(());
        }
        Some(Commands::Conclude {
            session_id,
            status,
            note,
        }) => {
            let config = PikuConfig::load(None, None, env::current_dir().ok().as_deref());
            conclude_run(&config, session_id, status, note)?;
            return Ok(());
        }
        Some(Commands::Web { port }) => {
            let config = PikuConfig::load(None, None, env::current_dir().ok().as_deref());
            piku::web::serve(&config, *port).await?;
            return Ok(());
        }
        None => {}
    }

    let config = PikuConfig::load(
        cli.provider.as_deref(),
        cli.model.as_deref(),
        cwd.as_deref(),
    );

    let prompt_str = if cli.prompt.is_empty() {
        None
    } else {
        Some(cli.prompt.join(" "))
    };

    if let Some(session_id) = &cli.resume {
        run_resume(
            session_id,
            prompt_str.as_deref(),
            &config,
            cli.print,
            cli.read_only,
        )
        .await?;
    } else if let Some(prompt) = prompt_str {
        if !cli.print && !cli.read_only && self_update::was_restarted() {
            if let Some(session) = try_load_restart_session(&config) {
                return run_tui_repl_post_restart(session, &config, cli.read_only).await;
            }
        }
        run_single_shot(&prompt, None, &config, cli.print, cli.read_only).await?;
    } else {
        if self_update::was_restarted() {
            if let Some(session) = try_load_restart_session(&config) {
                return run_tui_repl_post_restart(session, &config, cli.read_only).await;
            }
        }
        tui_repl::run_tui_repl_with_mode(&config, cli.read_only).await?;
    }

    Ok(())
}

fn inspect_run(config: &PikuConfig, session_id: &str, format: InspectFormat) -> anyhow::Result<()> {
    let path = config.runs_dir().join(format!("{session_id}.jsonl"));
    let events = piku_runtime::read_run_record(&path)?;
    if events.is_empty() {
        anyhow::bail!("no durable run record found at {}", path.display());
    }
    match format {
        InspectFormat::Text => print!("{}", piku::run_view::render_text(&events)),
        InspectFormat::Json => println!("{}", piku::run_view::render_json(&events)?),
        InspectFormat::Html => {
            let output_path = config.runs_dir().join(format!("{session_id}.html"));
            std::fs::write(
                &output_path,
                piku::run_view::render_html_with_artifacts(&events, &path)?,
            )?;
            println!("{}", output_path.display());
        }
    }
    Ok(())
}

fn conclude_run(
    config: &PikuConfig,
    session_id: &str,
    status: &str,
    note: &str,
) -> anyhow::Result<()> {
    let disposition = match status {
        "accepted" => piku_runtime::RunDisposition::Accepted,
        "needs-work" => piku_runtime::RunDisposition::NeedsWork,
        "abandoned" => piku_runtime::RunDisposition::Abandoned,
        _ => anyhow::bail!("--status must be accepted, needs-work, or abandoned"),
    };
    let path = config.runs_dir().join(format!("{session_id}.jsonl"));
    if !path.exists() {
        anyhow::bail!("no durable run record found at {}", path.display());
    }
    let mut recorder = RunRecorder::open(&path, session_id)?;
    recorder.append_run(piku_runtime::RunEvent::UserDisposition {
        disposition,
        note: piku_runtime::RunContentRef::Inline {
            text: note.to_string(),
        },
    })?;
    println!("recorded {disposition:?} for {session_id}");
    Ok(())
}

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

async fn run_tui_repl_post_restart(
    session: Session,
    config: &PikuConfig,
    read_only: bool,
) -> anyhow::Result<()> {
    tui_repl::run_tui_repl_post_restart(config, Some(session), read_only).await
}

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
        let matches: Vec<_> = std::fs::read_dir(&sessions_dir)?
            .filter_map(std::result::Result::ok)
            .filter(|e| e.file_name().to_string_lossy().contains(session_id))
            .collect();

        match matches.len() {
            0 => anyhow::bail!(
                "session '{session_id}' not found in {}",
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
    let date = piku::current_date();
    let mut system_sections = build_system_prompt(&cwd, &date, &model, &[]);
    if read_only {
        system_sections.push(piku::read_only_system_prompt_section());
    }

    let (session_id, mut session) = if let Some(s) = existing_session {
        eprintln!("[piku] continuing session {}", s.id);
        (s.id.clone(), s)
    } else {
        let id = piku::new_session_id();
        (id.clone(), Session::new(id))
    };
    if let Some((prior_provider, prior_model)) = session.record_provider(resolved.name(), &model) {
        eprintln!(
            "[piku] warning: session was written by {prior_provider}/{prior_model}, continuing with {}/{model}",
            resolved.name()
        );
    }

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
    let run_path = config.runs_dir().join(format!("{session_id}.jsonl"));
    let turn_id = format!("turn-{}", session.messages.len());
    let mut recorder = RunRecorder::open(&run_path, &session_id)?;
    let mut recording_sink = RecordingSink::new(&mut sink, &mut recorder, turn_id);

    let result: TurnResult = run_turn(
        prompt,
        &mut session,
        resolved.as_provider(),
        &model,
        &system_sections,
        tool_defs,
        &prompter,
        &mut recording_sink,
        None,
        None,
    )
    .await;
    if let Some(error) = recording_sink.take_record_error() {
        anyhow::bail!(
            "could not persist run record {}: {error}",
            run_path.display()
        );
    }

    if let Some(err) = &result.stream_error {
        eprintln!("[piku] stream error: {err}");
    }

    let sessions_dir = config.sessions_dir();
    std::fs::create_dir_all(&sessions_dir)?;
    let session_path = sessions_dir.join(format!("{session_id}.json"));
    if let Err(e) = session.save(&session_path) {
        eprintln!("warning: could not save session: {e}");
    } else {
        eprintln!("[piku] session saved → {}", session_path.display());
    }

    if let Some(new_binary) = result.replace_and_exec {
        eprintln!("[piku] rebuilt — restarting with new binary...");
        if let Err(e) = self_update::replace_and_exec(&new_binary) {
            eprintln!("[piku] self-update failed: {e}");
            eprintln!("[piku] continuing with old binary");
        }
    }

    if self_update::was_restarted() {
        eprintln!("[piku] restarted after self-rebuild ✓");
    }

    if print || read_only {
        return Ok(());
    }

    println!();
    tui_repl::run_tui_repl_with_session(config, Some(session), result.usage).await
}

struct StdoutSink {
    stdout: io::Stdout,
    trace: TraceWriter,
    pending_tool_id: std::collections::HashMap<String, String>,
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
            format!("{}…", piku::truncate_on_char_boundary(result, 400))
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
