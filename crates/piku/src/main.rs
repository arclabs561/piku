mod auth_cli;

use clap::Parser;

use piku::cli::{Cli, Commands, InspectFormat};
use piku::config::PikuConfig;
use piku::request_log::RequestLog;
use piku::self_update;
use piku::trace::TraceWriter;
use piku::tui_repl;

use std::env;
use std::io::{self, IsTerminal, Write};

use piku_runtime::{
    build_system_prompt, run_turn, OutputSink, PermissionOutcome, PermissionPrompter,
    PermissionRequest, RunHandle, RunRecorder, Session, TurnResult,
};
use piku_runtime::{provider_availability, PostToolAction, TokenUsage};
use piku_tools::launch_tool_definitions;

#[allow(clippy::too_many_lines)] // Command dispatch stays beside its subcommand wiring.
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    if auth_cli::helper_requested() {
        return auth_cli::run_helper().await;
    }
    piku::telemetry::init();
    let cwd = env::current_dir().ok();
    let cli = Cli::parse();
    cli.validate_request_log()?;

    match &cli.command {
        Some(Commands::Permissions { command }) => {
            piku::project_permissions::run(
                command,
                cwd.as_deref().unwrap_or(std::path::Path::new(".")),
            )?;
            return Ok(());
        }
        Some(Commands::Auth { command, .. }) => {
            auth_cli::run(command).await?;
            return Ok(());
        }
        Some(Commands::Providers) => {
            print_providers();
            return Ok(());
        }
        Some(Commands::Inspect {
            session_id,
            cells,
            activity,
            cell,
            receipt,
            raw,
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
            inspect_run(
                &config,
                session_id,
                &InspectOptions {
                    format,
                    cells: *cells,
                    activity: *activity,
                    selected_cell: cell.as_deref(),
                    selected_receipt: receipt.as_deref(),
                    raw_cell: raw.as_deref(),
                },
            )?;
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
        Some(Commands::Web { port, terminal }) => {
            let config = PikuConfig::load(
                cli.provider.as_deref(),
                cli.model.as_deref(),
                cwd.as_deref(),
            );
            piku::web::serve(&config, *port, *terminal).await?;
            return Ok(());
        }
        None => {}
    }

    let request_log = cli
        .request_log
        .as_deref()
        .map(RequestLog::create)
        .transpose()?;
    let result = run_terminal(&cli, cwd.as_deref(), request_log.as_ref()).await;
    let recording = request_log.map(RequestLog::finish).transpose();
    if let Err(error) = &recording {
        eprintln!("[piku] request-log recording error: {error}");
    }
    result?;
    recording?;
    Ok(())
}

async fn run_terminal(
    cli: &Cli,
    cwd: Option<&std::path::Path>,
    request_log: Option<&RequestLog>,
) -> anyhow::Result<()> {
    let self_update = self_update::SelfUpdatePolicy::from_operator_root(
        cli.self_update_root.as_deref(),
        cli.read_only,
    )?;
    let config = PikuConfig::load(cli.provider.as_deref(), cli.model.as_deref(), cwd);

    // A referenced interactive successor resumes its checkpoint, not the
    // original argv's prompt or --resume action. Marker-only/headless launches
    // retain their existing behavior.
    if !cli.print
        && (!cli.read_only || cli.prompt.is_empty())
        && std::env::var_os("PIKU_SESSION_ID").is_some()
        && self_update::was_restarted()
    {
        let session = load_restart_session(&config)?;
        return run_tui_repl_post_restart(
            session,
            &config,
            cli.read_only,
            &self_update,
            request_log,
        )
        .await;
    }

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
            &self_update,
            request_log,
        )
        .await?;
    } else if let Some(prompt) = prompt_str {
        if !cli.print && !cli.read_only {
            let _ = self_update::was_restarted();
        }
        run_single_shot(
            &prompt,
            None,
            &config,
            cli.print,
            cli.read_only,
            &self_update,
            request_log,
        )
        .await?;
    } else {
        let _ = self_update::was_restarted();
        tui_repl::run_tui_repl_with_policy_observed(
            &config,
            None,
            TokenUsage::default(),
            false,
            cli.read_only,
            &self_update,
            request_log,
        )
        .await?;
    }

    Ok(())
}

struct InspectOptions<'a> {
    format: InspectFormat,
    cells: bool,
    activity: bool,
    selected_cell: Option<&'a str>,
    selected_receipt: Option<&'a str>,
    raw_cell: Option<&'a str>,
}

#[allow(clippy::too_many_lines)] // Validation and format rendering form one CLI contract.
fn inspect_run(
    config: &PikuConfig,
    session_id: &str,
    options: &InspectOptions<'_>,
) -> anyhow::Result<()> {
    let InspectOptions {
        format,
        cells,
        activity,
        selected_cell,
        selected_receipt,
        raw_cell,
    } = *options;
    let path = config.runs_dir().join(format!("{session_id}.jsonl"));
    let events = piku_runtime::read_run_record(&path)?;
    if events.is_empty() {
        anyhow::bail!("no durable run record found at {}", path.display());
    }
    // Selecting one item implies the listing it belongs to.
    let cells = cells || selected_cell.is_some();
    let activity = activity || selected_receipt.is_some();
    if cells && activity {
        anyhow::bail!("--cells and --activity cannot be combined");
    }
    if let Some(reference) = raw_cell {
        if !matches!(format, InspectFormat::Text)
            || cells
            || activity
            || selected_cell.is_some()
            || selected_receipt.is_some()
        {
            anyhow::bail!("--raw cannot be combined with --json, --html, --cells, --activity, --cell, or --receipt");
        }
        let output = piku::actions::execute(
            &piku::actions::Action::InspectCell {
                reference: Some(reference.to_string()),
                open_in_pager: true,
            },
            &events,
            &path,
        )?
        .require_exact_output()?;
        print!("{output}");
        return Ok(());
    }
    let selected_cells = if cells {
        let action = selected_cell.map_or(
            piku::actions::Action::InspectCells {
                open_in_pager: false,
            },
            |reference| piku::actions::Action::InspectCell {
                reference: Some(reference.to_string()),
                open_in_pager: false,
            },
        );
        let cells = piku::actions::execute(&action, &events, &path)?.require_cells()?;
        if selected_cell.is_some() && cells.is_empty() {
            anyhow::bail!("unknown cell reference; use piku inspect {session_id} --cells");
        }
        cells
    } else {
        Vec::new()
    };
    let selected_receipts = if activity {
        let action = selected_receipt.map_or(
            piku::actions::Action::InspectReceipts {
                open_in_pager: false,
            },
            |reference| piku::actions::Action::InspectReceipt {
                reference: Some(reference.to_string()),
                open_in_pager: false,
            },
        );
        let receipts = piku::actions::execute(&action, &events, &path)?.require_receipts()?;
        if selected_receipt.is_some() && receipts.is_empty() {
            anyhow::bail!("unknown receipt reference; use piku inspect {session_id} --activity");
        }
        receipts
    } else {
        Vec::new()
    };
    match format {
        InspectFormat::Text if cells => {
            print!("{}", piku::cell_view::render_cells_text(&selected_cells));
        }
        InspectFormat::Json if cells => {
            println!("{}", serde_json::to_string_pretty(&selected_cells)?);
        }
        InspectFormat::Text if activity => {
            if selected_receipts.is_empty() {
                eprintln!("no interaction receipts recorded for {session_id}");
            }
            print!(
                "{}",
                piku::receipt_view::render_receipts_text(&selected_receipts)
            );
        }
        InspectFormat::Json if activity => {
            println!("{}", serde_json::to_string_pretty(&selected_receipts)?);
        }
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

fn load_restart_session(config: &PikuConfig) -> anyhow::Result<Session> {
    let session_id = std::env::var("PIKU_SESSION_ID")
        .map_err(|error| anyhow::anyhow!("invalid restart checkpoint reference: {error}"))?;
    std::env::remove_var("PIKU_SESSION_ID");
    let sessions_dir = config.sessions_dir();
    let path = sessions_dir.join(format!("{session_id}.json"));
    let session = Session::load(&path).map_err(|error| {
        anyhow::anyhow!(
            "could not load restart checkpoint {}: {error}; recover the saved session explicitly",
            path.display()
        )
    })?;
    if session.id != session_id {
        anyhow::bail!(
            "restart checkpoint session ID mismatch at {}; recover the saved session explicitly",
            path.display()
        );
    }
    Ok(session)
}

async fn run_tui_repl_post_restart(
    session: Session,
    config: &PikuConfig,
    read_only: bool,
    self_update: &self_update::SelfUpdatePolicy,
    request_log: Option<&RequestLog>,
) -> anyhow::Result<()> {
    tui_repl::run_tui_repl_with_policy_observed(
        config,
        Some(session),
        TokenUsage::default(),
        true,
        read_only,
        self_update,
        request_log,
    )
    .await
}

async fn run_resume(
    session_id: &str,
    prompt: Option<&str>,
    config: &PikuConfig,
    print: bool,
    read_only: bool,
    self_update: &self_update::SelfUpdatePolicy,
    request_log: Option<&RequestLog>,
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
                    self_update,
                    request_log,
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
        self_update,
        request_log,
    )
    .await
}

fn print_route_status(provider: &str, model: &str) {
    eprintln!("[piku] provider={provider} model={model}");
    if provider == "chatgpt" {
        eprintln!("[piku] ChatGPT subscription funding; allowance and credits unknown; no hard output-token cap for foreground, children, or memory calls");
    }
}

#[allow(clippy::too_many_lines)] // Keep launch persistence and update admission together.
async fn run_single_shot(
    prompt: &str,
    existing_session: Option<Session>,
    config: &PikuConfig,
    print: bool,
    read_only: bool,
    self_update: &self_update::SelfUpdatePolicy,
    request_log: Option<&RequestLog>,
) -> anyhow::Result<()> {
    let resolved = piku::request_log::resolve(config.provider.as_deref(), request_log)?;
    let model = config
        .model
        .as_deref()
        .unwrap_or(&resolved.default_model)
        .to_string();

    print_route_status(resolved.name(), &model);

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
        eprintln!(
            "[piku] tool calls run without prompting; {} deny rule(s) from settings.toml apply",
            config.deny.len()
        );
        launch_tool_definitions()
    };
    let prompter = ConfiguredAllowAll::new(&config.deny);
    let traces_dir = config.traces_dir();
    std::fs::create_dir_all(&traces_dir).ok();
    let trace = TraceWriter::open(&traces_dir, &session_id);
    let mut sink = StdoutSink::new(trace, self_update.clone());
    sink.request_observation = request_log.map(RequestLog::request_observation);
    sink.trace.prompt(prompt);
    let run_path = config.runs_dir().join(format!("{session_id}.jsonl"));
    let mut turn_nonce = [0; 16];
    getrandom::fill(&mut turn_nonce).map_err(|error| anyhow::anyhow!("turn ID: {error}"))?;
    let turn_id = format!("turn-{:032x}", u128::from_ne_bytes(turn_nonce));
    let mut run = RunHandle::open(session, &run_path)?;
    let result: TurnResult = {
        let mut turn = run.begin_turn(&mut sink, turn_id);
        let (session, recording_sink) = turn.parts();
        let result = run_turn(
            prompt,
            session,
            resolved.as_provider(),
            &model,
            &system_sections,
            tool_defs,
            &prompter,
            recording_sink,
            config.max_turns,
            None,
        )
        .await;
        turn.finish().map_err(|error| {
            anyhow::anyhow!(
                "could not persist run record {}: {error}",
                run_path.display()
            )
        })?;
        result
    };
    let session = run.into_session();

    let sessions_dir = config.sessions_dir();
    std::fs::create_dir_all(&sessions_dir)?;
    let session_path = sessions_dir.join(format!("{session_id}.json"));
    let checkpoint_saved = if let Err(e) = session.save(&session_path) {
        eprintln!("warning: could not save session: {e}");
        false
    } else {
        eprintln!("[piku] session saved → {}", session_path.display());
        true
    };

    report_launch_stream_error(result.stream_error.as_deref(), print || read_only)?;

    let failed = result.stream_error.is_some();
    if result.replace_and_exec.is_some() && !failed && !checkpoint_saved {
        eprintln!("[piku] self-update refused: restart checkpoint could not be saved; continuing with old binary");
    }
    if let Some(new_binary) = result
        .replace_and_exec
        .filter(|_| !failed && checkpoint_saved)
    {
        eprintln!("[piku] rebuilt — restarting with new binary...");
        if let Some(log) = request_log {
            log.prepare_exec()?;
        }
        if let Err(e) = self_update.replace_and_exec(&new_binary, &[]) {
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
    tui_repl::run_tui_repl_with_policy_observed(
        config,
        Some(session),
        result.usage,
        false,
        read_only,
        self_update,
        request_log,
    )
    .await
}

fn report_launch_stream_error(error: Option<&str>, headless: bool) -> anyhow::Result<()> {
    if let Some(error) = error {
        if headless {
            anyhow::bail!("stream error: {error}");
        }
        eprintln!("[piku] stream error: {error}");
    }
    Ok(())
}

struct ConfiguredAllowAll {
    deny_rules: Vec<String>,
}

impl ConfiguredAllowAll {
    fn new(deny_rules: &[String]) -> Self {
        Self {
            deny_rules: deny_rules.to_vec(),
        }
    }
}

impl PermissionPrompter for ConfiguredAllowAll {
    fn delegated_policy(&self) -> Option<std::sync::Arc<dyn PermissionPrompter>> {
        Some(std::sync::Arc::new(Self::new(&self.deny_rules)))
    }

    fn decide(&self, _req: &PermissionRequest) -> PermissionOutcome {
        PermissionOutcome::Allow
    }

    fn denies(&self, tool_name: &str, params: &serde_json::Value) -> Option<String> {
        self.deny_rules
            .iter()
            .find(|pattern| piku::config::matches_tool_pattern(pattern, tool_name, params))
            .map(|pattern| format!("denied by settings.toml rule: {pattern}"))
    }
}

struct StdoutSink {
    request_observation: Option<piku_runtime::RequestObservation>,
    self_update: self_update::SelfUpdatePolicy,
    stdout: io::Stdout,
    trace: TraceWriter,
    md: piku::markdown::StreamingMarkdown,
    color: bool,
}

impl StdoutSink {
    fn new(trace: TraceWriter, self_update: self_update::SelfUpdatePolicy) -> Self {
        let stdout = io::stdout();
        let color = stdout.is_terminal() && env::var_os("NO_COLOR").is_none();
        Self {
            request_observation: None,
            self_update,
            stdout,
            trace,
            md: piku::markdown::StreamingMarkdown::new_stdout(),
            color,
        }
    }

    /// Write rendered output, dropping escape sequences when stdout is a file,
    /// a pipe, or `NO_COLOR` is set.
    fn emit(&mut self, text: &str) {
        if self.color {
            let _ = self.stdout.write_all(text.as_bytes());
        } else {
            let _ = self
                .stdout
                .write_all(piku::markdown::strip_ansi(text).as_bytes());
        }
        let _ = self.stdout.flush();
    }
}

impl OutputSink for StdoutSink {
    fn request_observation(&self) -> Option<piku_runtime::RequestObservation> {
        self.request_observation.clone()
    }

    fn on_text(&mut self, text: &str) {
        let rendered = self.md.push(text);
        if !rendered.is_empty() {
            self.emit(&rendered);
        }
        self.trace.text_chunk(text);
    }

    fn on_tool_start(&mut self, tool_name: &str, tool_id: &str, input: &serde_json::Value) {
        let flushed = self.md.flush();
        if !flushed.is_empty() {
            self.emit(&flushed);
        }
        let args = piku::format_tool_input(tool_name, input);
        let line = if args.is_empty() {
            format!("\n\x1b[2m[{tool_name} …]\x1b[0m")
        } else {
            format!("\n\x1b[2m[{tool_name} {args} …]\x1b[0m")
        };
        self.emit(&format!("{line}\n"));

        self.trace.tool_start(tool_name, tool_id, input);
    }

    fn on_tool_end(
        &mut self,
        tool_name: &str,
        tool_id: &str,
        result: &str,
        is_error: bool,
    ) -> PostToolAction {
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
        self.emit(&format!(
            "\x1b[2m[{tool_name} → {tag}]\x1b[0m\n{preview}\n\n"
        ));

        self.trace.tool_end(tool_name, tool_id, result, !is_error);

        if tool_name == "bash" {
            match self.self_update.detect_self_build(result, !is_error) {
                Ok(Some(new_binary)) => {
                    eprintln!(
                        "[piku] detected approved self-rebuild → {}",
                        new_binary.display()
                    );
                    return PostToolAction::ReplaceAndExec(new_binary);
                }
                Ok(None) => {}
                Err(error) => eprintln!("[piku] self-update check failed: {error}"),
            }
        }

        PostToolAction::Continue
    }

    fn on_permission_denied(&mut self, tool_name: &str, reason: &str) {
        self.emit(&format!(
            "\x1b[33m[permission denied: {tool_name}]\x1b[0m {reason}\n"
        ));
        self.trace.permission_denied(tool_name, reason);
    }

    fn on_turn_complete(&mut self, usage: &TokenUsage, iterations: u32) {
        let flushed = self.md.flush();
        if !flushed.is_empty() {
            self.emit(&flushed);
        }
        self.emit(&format!(
            "\n\x1b[2m[{iterations} iter · {}↑ {}↓ tokens]\x1b[0m\n",
            usage.input_tokens, usage.output_tokens
        ));
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
