/// Sticky-bottom REPL: output scrolls above a fixed readline prompt.
///
/// Layout (terminal rows, 1-indexed):
///   1 … (H-2)   scrolling output area   ← agent text / tool results
///   H-1         thin divider             ← "─── piku ──────────────"
///   H           readline input           ← "› your message"
///
/// We achieve this with ANSI DECSTBM (`CSI <top>;<bot> r`) which tells the
/// terminal to only scroll rows 1..(H-2).  Everything written with a trailing
/// newline in that zone scrolls upward; the bottom two rows stay fixed.
///
/// rustyline owns row H.  Between readline calls we:
///   1. move cursor into the scroll zone (row H-2)
///   2. call `run_turn` (agent streams into scroll zone)
///   3. redraw the divider at H-1
///   4. call readline again (it redraws at H)
///
/// Multiline: paste text with newlines, or press Enter mid-input to add
/// a line. Submit by pressing Enter on an empty continuation line.
use std::io::{self, Write};

use crate::cli::ResolvedProvider;
use crate::input_helper::{LineEditor, ReadOutcome};
use crate::markdown::StreamingMarkdown;
use crate::self_update;
use crossterm::event::{self as cxevent, Event, KeyCode, KeyEvent, KeyModifiers};
use crossterm::terminal;
use piku_api::TokenUsage;
use piku_runtime::{
    build_system_prompt, run_turn_with_registry, InterjectionRx, InterjectionTx, OutputSink,
    PermissionOutcome, PermissionPrompter, PermissionRequest, PostToolAction, Session,
    TaskRegistry, TaskStatus, TurnResult,
};
use piku_tools::{all_tool_definitions, Destructiveness};

// ── Permission prompter ───────────────────────────────────────────────────────

/// Interactive permission prompter for the TUI.
///
/// When the agent wants to run a destructive tool the agent loop calls
/// `decide()` synchronously (before execution).  We:
///   1. Move to the input row and print a one-line prompt.
///   2. Enable raw mode and read a single keypress.
///   3. Restore raw-mode state and return Allow / Deny.
///
/// Key bindings:
///   y / Enter  → Allow
///   n / Escape → Deny
///   a          → Allow all (upgrades self to `AllowAll` for the rest of the turn)
///
pub struct TuiPrompter {
    /// If true, skip all future prompts and allow everything.
    allow_all: std::sync::atomic::AtomicBool,
    /// Pre-configured allow/deny rules from settings.json.
    allow_rules: Vec<String>,
    deny_rules: Vec<String>,
}

impl Default for TuiPrompter {
    fn default() -> Self {
        Self::new(&[], &[])
    }
}

impl TuiPrompter {
    #[must_use]
    pub fn new(allow: &[String], deny: &[String]) -> Self {
        Self {
            allow_all: std::sync::atomic::AtomicBool::new(false),
            allow_rules: allow.to_vec(),
            deny_rules: deny.to_vec(),
        }
    }
}

impl PermissionPrompter for TuiPrompter {
    fn decide(&self, req: &PermissionRequest) -> PermissionOutcome {
        // Fast path: user already said "allow all" earlier this turn.
        if self.allow_all.load(std::sync::atomic::Ordering::Relaxed) {
            return PermissionOutcome::Allow;
        }

        // Config-based rules: deny > allow > fall through to interactive.
        for pattern in &self.deny_rules {
            if crate::config::matches_tool_pattern(pattern, &req.tool_name, &req.params) {
                return PermissionOutcome::Deny {
                    reason: format!("denied by settings.json rule: {pattern}"),
                };
            }
        }
        for pattern in &self.allow_rules {
            if crate::config::matches_tool_pattern(pattern, &req.tool_name, &req.params) {
                return PermissionOutcome::Allow;
            }
        }

        let (cols, rows) = term_size();
        let input_row = rows;
        let footer_row = rows.saturating_sub(1);

        // ── Build prompt text ─────────────────────────────────────────────────
        // Claude Code aesthetic: no icons, color only as signal, dim hints.
        // Definite → red tool name; Likely → yellow; Safe → skip.
        let (color, label) = match req.destructiveness {
            Destructiveness::Definite => ("\x1b[31m", req.tool_name.as_str()),
            Destructiveness::Likely => ("\x1b[33m", req.tool_name.as_str()),
            Destructiveness::Safe => return PermissionOutcome::Allow,
        };
        let reset = "\x1b[0m";
        let dim = "\x1b[2m";
        let desc = &req.description;
        // Truncate description so it fits: cols minus ~25 for the framing
        let max_desc = (cols as usize).saturating_sub(25);
        let short_desc = if desc.len() > max_desc {
            format!("{}…", &desc[..max_desc.saturating_sub(1)])
        } else {
            desc.clone()
        };
        // Format: "  bash  rm -rf build/   y/n/a? "
        let prompt =
            format!("\x1b[2K\r  {color}{label}{reset}  {short_desc}  {dim}y/n/a?{reset} ",);

        // ── Show prompt on the input row ──────────────────────────────────────
        goto(input_row, 1);
        print!("\x1b[?25h{prompt}"); // ensure cursor visible before waiting
        let _ = io::stdout().flush();

        // ── Read one keypress in raw mode ─────────────────────────────────────
        let was_raw = terminal::is_raw_mode_enabled().unwrap_or(false);
        if !was_raw {
            let _ = terminal::enable_raw_mode();
        }

        let outcome = loop {
            match cxevent::read() {
                Ok(Event::Key(KeyEvent {
                    code, modifiers, ..
                })) => {
                    // Ctrl-C / Ctrl-D → deny
                    if modifiers.contains(KeyModifiers::CONTROL)
                        && matches!(code, KeyCode::Char('c' | 'd'))
                    {
                        break PermissionOutcome::Deny {
                            reason: "user aborted with Ctrl-C".to_string(),
                        };
                    }
                    match code {
                        KeyCode::Char('y') | KeyCode::Enter => {
                            break PermissionOutcome::Allow;
                        }
                        KeyCode::Char('a') => {
                            self.allow_all
                                .store(true, std::sync::atomic::Ordering::Relaxed);
                            break PermissionOutcome::Allow;
                        }
                        KeyCode::Char('n') | KeyCode::Esc => {
                            break PermissionOutcome::Deny {
                                reason: "user denied".to_string(),
                            };
                        }
                        _ => {
                            // ignore unknown keys, keep waiting
                        }
                    }
                }
                Ok(_) => {} // resize events etc. — ignore
                Err(_) => {
                    // I/O error — deny for safety
                    break PermissionOutcome::Deny {
                        reason: "I/O error reading keypress".to_string(),
                    };
                }
            }
        };

        if !was_raw {
            let _ = terminal::disable_raw_mode();
        }

        // ── Restore input row ──────────────────────────────────────────────────
        // Clear the prompt line and redraw the footer so the TUI looks tidy
        // before the tool executes (or the denial message prints).
        goto(input_row, 1);
        print!("\x1b[2K"); // erase prompt
                           // Echo decision into scroll zone
        let (_, scroll_bot_rows) = term_size();
        let scroll_bot = scroll_bot_rows.saturating_sub(2);
        goto(scroll_bot, 1);
        match &outcome {
            PermissionOutcome::Allow => {
                let suffix = if self.allow_all.load(std::sync::atomic::Ordering::Relaxed) {
                    " (all remaining)"
                } else {
                    ""
                };
                println!("  \x1b[32m{label}\x1b[0m\x1b[2m  {short_desc}{suffix}\x1b[0m\r");
            }
            PermissionOutcome::Deny { .. } => {
                println!("  \x1b[31m{label}\x1b[0m\x1b[2m  {short_desc}  denied\x1b[0m\r");
            }
        }
        goto(footer_row, 1);
        print!("\x1b[2K");
        goto(rows, 1);
        print!("\x1b[2K\x1b[?25h");
        let _ = io::stdout().flush();

        outcome
    }
}

// ── ANSI helpers ──────────────────────────────────────────────────────────────

/// Set terminal scrolling region rows [top..=bot] (1-indexed).
fn set_scroll_region(top: u16, bot: u16) {
    print!("\x1b[{top};{bot}r");
}

/// Reset scrolling region to the full screen.
fn reset_scroll_region() {
    print!("\x1b[r");
}

/// Move cursor to (row, col) — 1-indexed.
fn goto(row: u16, col: u16) {
    print!("\x1b[{row};{col}H");
}

fn term_size() -> (u16, u16) {
    terminal::size().unwrap_or((80, 24))
}

// ── Divider ───────────────────────────────────────────────────────────────────

// ── Footer ────────────────────────────────────────────────────────────────────

/// All the state needed to render the footer bar.
pub(crate) struct FooterState<'a> {
    pub(crate) provider: &'a str,
    pub(crate) model: &'a str,
    pub(crate) session_id: &'a str,
    pub(crate) input_tokens: u32,
    pub(crate) output_tokens: u32,
    pub(crate) turns: u32,
    pub(crate) running_agents: usize,
    pub(crate) context_pct: Option<u8>,
}

/// Draw the sticky footer at `row`.
///
/// Layout (left → right, separated by dim │):
///   provider · model  │  ↑123k ↓45  │  sess-…abc  │  /help
///
/// Everything is on one line. If the terminal is too narrow, rightmost
/// segments are dropped first.
fn draw_footer(row: u16, cols: u16, s: &FooterState) {
    let line = render_footer(cols, s);
    goto(row, 1);
    print!("\x1b[2K\x1b[7m{line}\x1b[0m");
}

/// Render the footer bar content as a String (no cursor movement).
/// Testable independently of terminal state.
pub(crate) fn render_footer(cols: u16, s: &FooterState) -> String {
    render_footer_inner(cols, s)
}

fn render_footer_inner(cols: u16, s: &FooterState) -> String {
    // ── Segments ──────────────────────────────────────────────────────────────
    // Left anchor: provider · model (normal weight, always shown)
    let model_seg = format!(" {} · {} ", s.provider, s.model);

    // Token usage (dim, omitted when zero)
    let tok_seg = if s.input_tokens > 0 || s.output_tokens > 0 {
        format!(
            " ↑{} ↓{} ",
            fmt_tokens(s.input_tokens),
            fmt_tokens(s.output_tokens)
        )
    } else {
        String::new()
    };

    // Turns (dim, omitted when zero)
    let turns_seg = if s.turns > 0 {
        format!(" {} turns ", s.turns)
    } else {
        String::new()
    };

    // Context window % — amber >75%, red >90%
    let ctx_seg = match s.context_pct {
        Some(pct) if pct >= 90 => format!(" \x1b[31m{pct}%\x1b[0m "),
        Some(pct) if pct >= 75 => format!(" \x1b[33m{pct}%\x1b[0m "),
        Some(pct) if pct >= 50 => format!(" {pct}% "),
        _ => String::new(),
    };

    // Running agents (amber, omitted when zero)
    let agents_seg = if s.running_agents > 0 {
        format!(" \x1b[33m{} bg\x1b[0m ", s.running_agents)
    } else {
        String::new()
    };

    // Session id — show last 8 chars so it's recognisable but short
    let short_id = short_session_id(s.session_id);
    let sess_seg = format!(" {short_id} ");

    // Right anchor: hint (always shown if space allows)
    let hint_seg = " /help ";

    let sep = "│";

    // ── Measure ───────────────────────────────────────────────────────────────
    let cols = cols as usize;

    // Helper: visible width of a segment (strips ANSI escape codes).
    let vis = |s: &str| -> usize { crate::input_helper::visible_width(s) };

    let sep_vis = vis(sep); // │ is 1 display column but 3 UTF-8 bytes

    // Required: model_seg + separator + hint_seg
    let base_width = vis(&model_seg) + sep_vis + vis(hint_seg);
    let mut budget = cols.saturating_sub(base_width);

    let mut show_ctx = false;
    let mut show_tok = false;
    let mut show_turns = false;
    let mut show_agents = false;
    let mut show_sess = false;

    // agents always shown if there are any (highest priority optional)
    if !agents_seg.is_empty() && budget >= vis(&agents_seg) + sep_vis {
        budget -= vis(&agents_seg) + sep_vis;
        show_agents = true;
    }
    // context % — shown when ≥50%, high priority (user needs to know)
    if !ctx_seg.is_empty() && budget >= vis(&ctx_seg) + sep_vis {
        budget -= vis(&ctx_seg) + sep_vis;
        show_ctx = true;
    }
    if !tok_seg.is_empty() && budget >= vis(&tok_seg) + sep_vis {
        budget -= vis(&tok_seg) + sep_vis;
        show_tok = true;
    }
    if !turns_seg.is_empty() && budget >= vis(&turns_seg) + sep_vis {
        budget -= vis(&turns_seg) + sep_vis;
        show_turns = true;
    }
    if budget >= vis(&sess_seg) + sep_vis {
        show_sess = true;
    }

    // ── Render ────────────────────────────────────────────────────────────────
    // We render into a plain String first, then pad to `cols`.
    let dim = "\x1b[2m";
    let reset = "\x1b[0m";
    let sep_s = format!("{dim}{sep}{reset}");

    let mut line = String::new();

    // model (normal weight)
    line.push_str(&model_seg);

    if show_agents {
        line.push_str(&sep_s);
        line.push_str(&agents_seg);
    }
    if show_ctx {
        line.push_str(&sep_s);
        line.push_str(&ctx_seg);
    }
    if show_tok {
        line.push_str(&sep_s);
        line.push_str(dim);
        line.push_str(&tok_seg);
        line.push_str(reset);
    }
    if show_turns {
        line.push_str(&sep_s);
        line.push_str(dim);
        line.push_str(&turns_seg);
        line.push_str(reset);
    }
    if show_sess {
        line.push_str(&sep_s);
        line.push_str(dim);
        line.push_str(&sess_seg);
        line.push_str(reset);
    }

    // Right-align the hint: pad with spaces to fill the row.
    // visible_len uses vis() for ANSI-containing segments.
    let visible_len = vis(&model_seg)
        + if show_agents {
            sep.len() + vis(&agents_seg)
        } else {
            0
        }
        + if show_ctx {
            sep.len() + vis(&ctx_seg)
        } else {
            0
        }
        + if show_tok {
            sep.len() + vis(&tok_seg)
        } else {
            0
        }
        + if show_turns {
            sep.len() + vis(&turns_seg)
        } else {
            0
        }
        + if show_sess {
            sep.len() + vis(&sess_seg)
        } else {
            0
        }
        + sep_vis
        + vis(hint_seg);

    let padding = cols.saturating_sub(visible_len);
    line.push_str(&" ".repeat(padding));
    line.push_str(&sep_s);
    line.push_str(dim);
    line.push_str(hint_seg);
    line.push_str(reset);

    line
}

/// Format a token count compactly: 1234 → "1.2k", 123456 → "123k", 12 → "12".
/// Estimate context window % from cumulative input tokens.
/// Uses 200k as the default window size (accurate for claude-3.x/4.x).
/// Returns None when tokens are zero (no usage yet).
pub(crate) fn context_pct(input_tokens: u32) -> Option<u8> {
    if input_tokens == 0 {
        return None;
    }
    const WINDOW: u32 = 200_000;
    let pct = ((input_tokens as f32 / WINDOW as f32) * 100.0).round() as u8;
    Some(pct.min(100))
}

pub(crate) fn fmt_tokens(n: u32) -> String {
    if n >= 10_000 {
        format!("{}k", n / 1000)
    } else if n >= 1_000 {
        format!("{:.1}k", n as f32 / 1000.0)
    } else {
        n.to_string()
    }
}

/// Return a short recognisable suffix of a session id.
/// "session-1775307696040353000-75082" → "…75082"
pub(crate) fn short_session_id(id: &str) -> String {
    // Take the last segment after the final '-', or last 8 chars
    if let Some(pos) = id.rfind('-') {
        let suffix = &id[pos + 1..];
        if suffix.len() <= 8 {
            return format!("…{suffix}");
        }
    }
    let n = id.len().min(8);
    format!("…{}", &id[id.len() - n..])
}

// ── Setup / teardown ──────────────────────────────────────────────────────────

fn setup_layout(rows: u16, cols: u16, model: &str, provider: &str, session_id: &str) {
    let scroll_bot = rows.saturating_sub(2);

    // Clear screen, position at top
    print!("\x1b[2J\x1b[H");
    set_scroll_region(1, scroll_bot);
    draw_footer(
        rows.saturating_sub(1),
        cols,
        &FooterState {
            provider,
            model,
            session_id,
            input_tokens: 0,
            output_tokens: 0,
            turns: 0,
            running_agents: 0,
            context_pct: None,
        },
    );
    // Park cursor in scroll zone bottom so first readline renders at row H
    goto(scroll_bot, 1);
    let _ = io::stdout().flush();
}

fn teardown_layout(rows: u16) {
    reset_scroll_region();
    print!("\x1b[?25h"); // always restore cursor on exit
    goto(rows, 1);
    println!();
    let _ = io::stdout().flush();
}

// ── Output sink that writes into the scroll zone ──────────────────────────────

pub struct TuiSink {
    stdout: io::Stdout,
    /// Mtime of the running binary at startup — used to detect self-rebuilds
    /// even when `current_exe()` and the build output path resolve to the same file.
    binary_mtime_baseline: Option<std::time::SystemTime>,
    /// Path to check for a new build.
    build_candidate: std::path::PathBuf,
    /// Whether the thinking indicator on the input row needs clearing.
    needs_indicator_clear: bool,
    /// Signal to stop the background thinking ticker.
    thinking_stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
    /// Current activity label shown in the indicator (changes during the turn).
    indicator_label: std::sync::Arc<std::sync::Mutex<String>>,
    /// Streaming markdown renderer for assistant text.
    md: StreamingMarkdown,
}

impl TuiSink {
    fn new(_model: &str, binary_mtime_baseline: Option<std::time::SystemTime>) -> Self {
        Self {
            stdout: io::stdout(),
            binary_mtime_baseline,
            build_candidate: self_update::default_build_output(),
            needs_indicator_clear: true,
            thinking_stop: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
            indicator_label: std::sync::Arc::new(std::sync::Mutex::new("thinking".to_string())),
            md: StreamingMarkdown::new(),
        }
    }

    /// Clear the thinking indicator from the input row and move cursor
    /// back into the scroll zone. Called once on first output.
    fn clear_thinking_indicator(&mut self) {
        if self.needs_indicator_clear {
            self.needs_indicator_clear = false;
            // Stop the background ticker
            self.thinking_stop
                .store(true, std::sync::atomic::Ordering::Relaxed);
            let (_, rows) = term_size();
            let scroll_bot = rows.saturating_sub(2);
            // Clear the thinking text from the input row
            goto(rows, 1);
            let _ = self.stdout.write_all(b"\x1b[2K");
            // Move cursor into scroll zone for output
            goto(scroll_bot, 1);
            let _ = self.stdout.flush();
        }
    }

    fn print(&mut self, s: &str) {
        let _ = self.stdout.write_all(s.as_bytes());
    }

    fn println(&mut self, s: &str) {
        self.print(s);
        self.print("\r\n");
    }
}

impl OutputSink for TuiSink {
    fn on_text(&mut self, text: &str) {
        self.clear_thinking_indicator();
        // Render markdown formatting as text streams in.
        // The renderer buffers partial lines and emits \r\n-terminated output.
        let rendered = self.md.push(text);
        if !rendered.is_empty() {
            self.print("\x1b[?25h");
            self.print(&rendered);
            let _ = self.stdout.flush();
        }
    }

    fn on_tool_start(&mut self, tool_name: &str, _tool_id: &str, input: &serde_json::Value) {
        self.clear_thinking_indicator();
        // Flush any pending markdown before showing the tool label.
        let flushed = self.md.flush();
        if !flushed.is_empty() {
            self.print(&flushed);
        }
        let name = tool_display_name(tool_name);
        let args = crate::format_tool_input(tool_name, input);
        self.print("\x1b[?25h");
        if args.is_empty() {
            self.println(&format!("\r\n\x1b[33m⏺\x1b[0m \x1b[1m{name}\x1b[0m"));
        } else {
            self.println(&format!(
                "\r\n\x1b[33m⏺\x1b[0m \x1b[1m{name}\x1b[0m\x1b[2m({args})\x1b[0m"
            ));
        }
        // Update indicator label to show tool execution
        *self.indicator_label.lock().unwrap() = format!("running {name}");
        let _ = self.stdout.flush();
    }

    fn on_tool_end(&mut self, tool_name: &str, result: &str, is_error: bool) -> PostToolAction {
        let preview = format_tool_result(tool_name, result, is_error);
        if !preview.is_empty() {
            self.println(&preview);
        }

        // Status dot: green success, red error
        let dot = if is_error {
            "\x1b[31m⏺\x1b[0m"
        } else {
            "\x1b[32m⏺\x1b[0m"
        };
        self.println(dot);
        // Restore label for next iteration (model will think again)
        *self.indicator_label.lock().unwrap() = "thinking".to_string();
        let _ = self.stdout.flush();

        if tool_name == "bash" {
            // Use the mtime baseline if available — this correctly detects a
            // rebuild even when current_exe() and build_candidate are the same path.
            let is_new = if let Some(baseline) = self.binary_mtime_baseline {
                self_update::is_newer_than_mtime(&self.build_candidate, baseline)
            } else {
                self_update::is_newer_than_running(&self.build_candidate)
            };
            // Also require the bash call itself to have succeeded and mentioned cargo
            let cargo_output_ok = !is_error
                && (result.contains("Finished")
                    && (result.contains("Compiling piku v")
                        || result.contains("target/release/piku")));
            if cargo_output_ok && is_new {
                return PostToolAction::ReplaceAndExec(self.build_candidate.clone());
            }
        }

        PostToolAction::Continue
    }

    fn on_permission_denied(&mut self, tool_name: &str, reason: &str) {
        self.println(&format!(
            "\x1b[33m[permission denied: {tool_name}]\x1b[0m {reason}"
        ));
        let _ = self.stdout.flush();
    }

    fn on_turn_complete(&mut self, usage: &TokenUsage, iterations: u32) {
        // Stop the background spinner (prevents task leak).
        self.thinking_stop
            .store(true, std::sync::atomic::Ordering::Relaxed);
        // Clear any lingering indicator from the input row.
        self.clear_thinking_indicator();
        // Flush any remaining markdown before the turn summary.
        let flushed = self.md.flush();
        if !flushed.is_empty() {
            self.print(&flushed);
        }
        self.println(&format!(
            "\r\n\x1b[2m[{iterations} iter · {}↑ {}↓]\x1b[0m",
            usage.input_tokens, usage.output_tokens
        ));
        let _ = self.stdout.flush();
    }
}

// ── Entry point ───────────────────────────────────────────────────────────────

pub async fn run_tui_repl(config: &crate::config::PikuConfig) -> anyhow::Result<()> {
    run_tui_repl_with_session(config, None, TokenUsage::default()).await
}

/// Like `run_tui_repl` but resumes from an existing session (e.g. after a
/// single-shot turn).  `initial_usage` is added to the running total so
/// `/cost` reflects the full conversation.
pub async fn run_tui_repl_with_session(
    config: &crate::config::PikuConfig,
    existing_session: Option<Session>,
    initial_usage: TokenUsage,
) -> anyhow::Result<()> {
    run_tui_repl_inner(config, existing_session, initial_usage, false).await
}

/// Like `run_tui_repl_with_session` but shows a "restarted" banner --
/// used after a seamless self-rebuild exec.
pub async fn run_tui_repl_post_restart(
    config: &crate::config::PikuConfig,
    existing_session: Option<Session>,
) -> anyhow::Result<()> {
    run_tui_repl_inner(config, existing_session, TokenUsage::default(), true).await
}

async fn run_tui_repl_inner(
    config: &crate::config::PikuConfig,
    existing_session: Option<Session>,
    initial_usage: TokenUsage,
    post_restart: bool,
) -> anyhow::Result<()> {
    // Wrap the entire REPL in a LocalSet so `tokio::task::spawn_local` works
    // for background subagents. Without this, spawn_local panics at runtime
    // with "there is no reactor running".
    let local = tokio::task::LocalSet::new();
    local
        .run_until(run_tui_repl_core(
            config,
            existing_session,
            initial_usage,
            post_restart,
        ))
        .await
}

async fn run_tui_repl_core(
    config: &crate::config::PikuConfig,
    existing_session: Option<Session>,
    initial_usage: TokenUsage,
    post_restart: bool,
) -> anyhow::Result<()> {
    let resolved = ResolvedProvider::resolve(config.provider.as_deref())?;
    let mut model = config
        .model
        .as_deref()
        .unwrap_or(&resolved.default_model)
        .to_string();

    let cwd = std::env::current_dir()?;
    let date = crate::current_date();
    let custom_agents = piku_runtime::load_custom_agents(&cwd);
    let hook_registry = piku_runtime::HookRegistry::load(&cwd);
    if hook_registry.has_hooks() {
        eprintln!("\x1b[2m[hooks loaded from .piku/hooks.json]\x1b[0m");
    }

    // Session-start maintenance: evict stale/weak memories from embedding store.
    {
        let store_path = piku_runtime::default_store_path(&cwd);
        let mut store = piku_runtime::MemoryStore::load(&store_path);
        if store.valid_count() > 0 {
            let (stale, weak) = store.maintain();
            if stale + weak > 0 {
                let _ = store.save(&store_path);
                eprintln!(
                    "\x1b[2m[memory maintenance: {stale} stale + {weak} weak entries evicted]\x1b[0m"
                );
            }
        }
    }

    let task_registry = TaskRegistry::new();
    // Wire a notification channel so background agent completions inject
    // a user-role message into the parent's interjection stream.
    let (notif_tx, notif_rx): (InterjectionTx, InterjectionRx) = tokio::sync::mpsc::channel(32);
    task_registry.set_notification_channel(notif_tx);
    let mut notif_rx = notif_rx;

    let sessions_dir = config.sessions_dir();
    std::fs::create_dir_all(&sessions_dir)?;

    let (session_id, mut session) = if let Some(s) = existing_session {
        let id = s.id.clone();
        (id, s)
    } else {
        let id = crate::new_session_id();
        (id.clone(), Session::new(id))
    };
    let session_path = sessions_dir.join(format!("{session_id}.json"));

    let mut total_usage = initial_usage;

    // Run SessionStart hooks -- captured context is appended to system prompt each turn.
    let hook_session_context = hook_registry.run_session_start(&session_id, &cwd);
    if let Some(ref ctx) = hook_session_context {
        eprintln!(
            "\x1b[2m[session-start hook injected {} chars]\x1b[0m",
            ctx.len()
        );
    }

    // Capture the running binary's mtime as a baseline. After a `cargo build`
    // the file is overwritten in-place (we exec a new copy), so comparing
    // against this baseline is the only reliable way to detect a newer build
    // when current_exe and the build output resolve to the same inode.
    let binary_mtime_baseline = self_update::running_mtime();
    let build_candidate = self_update::default_build_output();

    // ── Startup self-update check ─────────────────────────────────────────────
    // If a newer binary exists at the default cargo output path, replace and
    // exec before drawing any UI. The session (possibly empty) is persisted
    // first so the restarted process can resume it via PIKU_SESSION_ID.
    if !post_restart {
        let newer = binary_mtime_baseline.map_or_else(
            || self_update::is_newer_than_running(&build_candidate),
            |b| self_update::is_newer_than_mtime(&build_candidate, b),
        );
        if newer {
            // Persist session (may be empty — that's fine)
            if !session.messages.is_empty() {
                let _ = session.save(&session_path);
            }
            // No TUI layout yet — just print a plain status line and exec.
            eprintln!("[piku] newer binary detected — restarting...");
            if let Err(e) = self_update::replace_and_exec_with_env(
                &build_candidate,
                &[("PIKU_SESSION_ID", &session_id)],
            ) {
                eprintln!("[piku] self-update failed: {e}");
                // Fall through and start normally with old binary
            }
        }
    }

    // ── Terminal safety ────────────────────────────────────────────────────────
    // Install a panic hook that resets the terminal before printing the panic.
    // Without this, a panic leaves DECSTBM set and cursor hidden.
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        // Best-effort terminal reset
        let _ = io::stdout().write_all(b"\x1b[r\x1b[?25h\n");
        let _ = io::stdout().flush();
        default_hook(info);
    }));

    // ── Terminal setup ────────────────────────────────────────────────────────
    let (cols, rows) = term_size();
    setup_layout(rows, cols, &model, resolved.name(), &session_id);

    // Print a brief welcome (or restart banner) into the scroll zone,
    // then replay the tail of the session history so context is visible.
    {
        let scroll_bot = rows.saturating_sub(2);
        goto(1, 1); // top of scroll zone
        if post_restart {
            println!(
                "\x1b[1;32m↺ restarted with new binary\x1b[0m  \x1b[2msession: {session_id}\x1b[0m\r"
            );
        } else {
            println!(
                "\x1b[1mpiku\x1b[0m  \x1b[2m{} · {}\x1b[0m\r\n\x1b[2m/help for commands · Shift+Enter for newline · Ctrl-D to exit\x1b[0m\r",
                resolved.name(), model,
            );
        }

        // Replay tail of session history into scroll zone
        print_session_tail(&session, scroll_bot);

        goto(scroll_bot, 1);
        let _ = io::stdout().flush();
    }

    // ── Line editor ──────────────────────────────────────────────────────────
    let history_path = sessions_dir.join(".repl_history");
    let mut editor = LineEditor::new("\x1b[34m❯\x1b[0m ");
    editor.load_history_file(&history_path);

    // ── Main loop ─────────────────────────────────────────────────────────────
    // Track whether the last turn had an error, to change the prompt glyph.
    let mut last_turn_error = false;

    loop {
        // Check for a newer binary on every iteration — catches the case where
        // `cargo build` was run externally while the REPL was already running.
        {
            let newer = binary_mtime_baseline.map_or_else(
                || self_update::is_newer_than_running(&build_candidate),
                |b| self_update::is_newer_than_mtime(&build_candidate, b),
            );
            if newer {
                let _ = session.save(&session_path);
                let (_, rows) = term_size();
                let scroll_bot = rows.saturating_sub(2);
                goto(scroll_bot, 1);
                announce_self_update(&build_candidate);
                let _ = self_update::replace_and_exec_with_env(
                    &build_candidate,
                    &[("PIKU_SESSION_ID", &session_id)],
                );
                // only reached on failure — fall through and keep running
            }
        }

        // Position editor at the bottom row.
        let (_, rows) = term_size();
        let input_row = rows;
        goto(input_row, 1);
        print!("\x1b[2K"); // clear current line before editor redraws
        let _ = io::stdout().flush();

        // Prompt glyph reflects state: red after error, blue normally.
        if last_turn_error {
            editor.set_prompt("\x1b[31m❯\x1b[0m ");
        } else {
            editor.set_prompt("\x1b[34m❯\x1b[0m ");
        }

        // Use read_line_raw so the editor doesn't emit post-submit
        // cursor movement that conflicts with the DECSTBM layout.
        // We temporarily reset the scroll region so MoveUp/MoveDown
        // in the editor's redraw work without scroll-region clipping.
        reset_scroll_region();
        let readline = editor.read_line_raw();
        // Restore scroll region and re-position cursor.
        let (_cols, rows) = term_size();
        set_scroll_region(1, rows.saturating_sub(2));
        // Park cursor in scroll zone bottom so output lands there.
        let scroll_bot = rows.saturating_sub(2);
        goto(scroll_bot, 1);
        print!("\x1b[?25h");
        let _ = io::stdout().flush();

        match readline {
            Ok(ReadOutcome::Submit(line)) => {
                // Expand paste pills before processing
                let expanded = editor.expand_paste_pills(&line);
                editor.clear_paste_pills();
                editor.push_history(&expanded);
                let full_input = expanded.trim().to_string();

                if full_input.is_empty() {
                    continue;
                }

                // Bare exit/quit keywords — no slash required
                if matches!(full_input.to_lowercase().as_str(), "exit" | "quit" | "q") {
                    break;
                }

                // ! prefix: direct bash command (bypass AI)
                // TODO: route through runtime's bash tool so the command appears in
                // session history, respects permissions, and shows in session replay.
                if let Some(cmd) = full_input.strip_prefix('!') {
                    let cmd = cmd.trim();
                    if !cmd.is_empty() {
                        let (_, rows) = term_size();
                        let scroll_bot = rows.saturating_sub(2);
                        goto(rows, 1);
                        print!("\x1b[2K");
                        goto(scroll_bot, 1);
                        // Echo command
                        println!("\r\n\x1b[2;35m!\x1b[0m \x1b[2m{cmd}\x1b[0m\r");
                        // Run it
                        let _ = io::stdout().flush();
                        reset_scroll_region();
                        let output = std::process::Command::new("sh").arg("-c").arg(cmd).output();
                        let (_, rows) = term_size();
                        set_scroll_region(1, rows.saturating_sub(2));
                        let scroll_bot = rows.saturating_sub(2);
                        goto(scroll_bot, 1);
                        match output {
                            Ok(out) => {
                                let stdout_str = String::from_utf8_lossy(&out.stdout);
                                let stderr_str = String::from_utf8_lossy(&out.stderr);
                                for line in stdout_str.lines().take(20) {
                                    println!("  {line}\r");
                                }
                                if stdout_str.lines().count() > 20 {
                                    println!(
                                        "  \x1b[2m… +{} lines\x1b[0m\r",
                                        stdout_str.lines().count() - 20
                                    );
                                }
                                if !stderr_str.is_empty() {
                                    for line in stderr_str.lines().take(5) {
                                        println!("  \x1b[31m{line}\x1b[0m\r");
                                    }
                                }
                            }
                            Err(e) => {
                                println!("  \x1b[31m{e}\x1b[0m\r");
                            }
                        }
                        let (cols, rows) = term_size();
                        draw_footer(
                            rows.saturating_sub(1),
                            cols,
                            &FooterState {
                                provider: resolved.name(),
                                model: &model,
                                session_id: &session_id,
                                input_tokens: total_usage.input_tokens,
                                output_tokens: total_usage.output_tokens,
                                turns: 0,
                                running_agents: 0,
                                context_pct: context_pct(total_usage.input_tokens),
                            },
                        );
                        let _ = io::stdout().flush();
                    }
                    continue;
                }

                // Slash commands — clear the input row first
                if full_input.starts_with('/') {
                    let (_, rows) = term_size();
                    goto(rows, 1);
                    print!("\x1b[2K");
                    let _ = io::stdout().flush();
                    let current_model_name = model.clone();
                    let should_exit = handle_slash_cmd(
                        &full_input,
                        &mut session,
                        &total_usage,
                        &current_model_name,
                        resolved.name(),
                        &session_id,
                        &mut model,
                        &task_registry,
                        config,
                        &hook_registry,
                    );
                    if should_exit {
                        break;
                    }
                    continue;
                }

                // ── Agent turn ────────────────────────────────────────────────
                // Move cursor into scroll zone, echo the user message so it
                // stays visible in the scroll history, then clear the input row.
                let (_, rows) = term_size();
                let scroll_bot = rows.saturating_sub(2);
                // Clear input row first so the stale prompt disappears
                goto(rows, 1);
                print!("\x1b[2K");
                // Echo user message into scroll zone — visually distinct from
                // the active prompt: dimmed text with a different glyph so the
                // user can distinguish "what I typed" from "where I type next".
                goto(scroll_bot, 1);
                print!("\r\n"); // fresh line before the user message
                let display_input = {
                    let lines: Vec<&str> = full_input.lines().collect();
                    if lines.len() <= 3 {
                        full_input.replace('\n', "\r\n")
                    } else {
                        let head = lines[..3].join("\r\n");
                        format!("{head}\r\n\x1b[2m… ({} more lines)\x1b[0m", lines.len() - 3)
                    }
                };
                // Echo: dim blue glyph + dim text, visually distinct from
                // the bright active prompt.
                println!("\x1b[2;34m❯\x1b[0m \x1b[2m{display_input}\x1b[0m\r");
                let _ = io::stdout().flush();

                let mut system_sections = build_system_prompt(&cwd, &date, &model, &custom_agents);
                if let Some(ref ctx) = hook_session_context {
                    system_sections.push(format!("# Hook Context\n\n{ctx}"));
                }
                let tool_defs = all_tool_definitions();
                let prompter = TuiPrompter::new(&config.allow, &config.deny);
                let mut sink = TuiSink::new(&model, binary_mtime_baseline);

                // Show a ticking thinking indicator on the input row.
                // A background task updates it every second with elapsed time.
                // Cleared by clear_thinking_indicator() on first output.
                let stop_flag = sink.thinking_stop.clone();
                stop_flag.store(false, std::sync::atomic::Ordering::Relaxed);
                let label_ref = sink.indicator_label.clone();
                *label_ref.lock().unwrap() = "thinking".to_string();
                // Show dimmed prompt with animated indicator on the input row.
                goto(rows, 1);
                print!("\x1b[2K\x1b[2m❯ · thinking\x1b[0m\x1b[?25h");
                goto(rows, 3);
                let _ = io::stdout().flush();
                let indicator_row = rows;
                tokio::task::spawn_local(async move {
                    const FRAMES: &[&str] = &["·", "✦", "✶", "✻", "✽", "✻", "✶", "✦"];
                    const STALL_SECS: u64 = 30;
                    let start = std::time::Instant::now();
                    let mut tick: usize = 0;
                    loop {
                        tokio::time::sleep(std::time::Duration::from_millis(120)).await;
                        if stop_flag.load(std::sync::atomic::Ordering::Relaxed) {
                            break;
                        }
                        let elapsed = start.elapsed().as_secs();
                        let frame = FRAMES[tick % FRAMES.len()];
                        tick += 1;
                        let time_str = crate::fmt_duration(elapsed);
                        let label = label_ref.lock().unwrap().clone();
                        // Stalled state: after 30s, interpolate color toward red
                        let color = if elapsed > STALL_SECS {
                            let t = ((elapsed - STALL_SECS) as f32 / 30.0).min(1.0);
                            let r = (153.0 + 18.0 * t) as u8;
                            let g = (153.0 - 110.0 * t) as u8;
                            let b = (153.0 - 90.0 * t) as u8;
                            format!("\x1b[38;2;{r};{g};{b}m")
                        } else {
                            "\x1b[2m".to_string()
                        };
                        let mut out = io::stdout();
                        let _ = out.write_all(b"\x1b[s");
                        goto(indicator_row, 1);
                        let _ = out.write_all(
                            format!("\x1b[2K{color}❯ {frame} {label} ({time_str})\x1b[0m")
                                .as_bytes(),
                        );
                        let _ = out.write_all(b"\x1b[u");
                        let _ = out.flush();
                    }
                });

                let result: TurnResult = run_turn_with_registry(
                    &full_input,
                    &mut session,
                    resolved.as_provider(),
                    &model,
                    &system_sections,
                    tool_defs,
                    &prompter,
                    &mut sink,
                    config.max_turns,
                    Some(&mut notif_rx),
                    &task_registry,
                    0,
                    &custom_agents,
                    Some(&hook_registry),
                )
                .await;

                total_usage.accumulate(&result.usage);
                last_turn_error = result.stream_error.is_some();

                if let Some(err) = &result.stream_error {
                    // Print error into scroll zone
                    let (_, rows) = term_size();
                    let scroll_bot = rows.saturating_sub(2);
                    goto(scroll_bot, 1);
                    println!("\x1b[31m[error]\x1b[0m {err}\r");
                }

                // Turn complete — redraw footer with updated token counts.
                let (cols, rows) = term_size();
                draw_footer(
                    rows.saturating_sub(1),
                    cols,
                    &FooterState {
                        provider: resolved.name(),
                        model: &model,
                        session_id: &session_id,
                        input_tokens: total_usage.input_tokens,
                        output_tokens: total_usage.output_tokens,
                        turns: result.iterations,
                        running_agents: task_registry.running().len(),
                        context_pct: context_pct(total_usage.input_tokens),
                    },
                );
                // Position input row, clear it, and restore cursor visibility.
                // rustyline may have left the cursor hidden; always show it here
                // so the › prompt is reliably visible before the next readline.
                goto(rows, 1);
                print!("\x1b[2K\x1b[?25h"); // clear input line + show cursor
                let _ = io::stdout().flush();

                // Persist session
                if let Err(e) = session.save(&session_path) {
                    eprintln!("\x1b[33m[warn]\x1b[0m could not save session: {e}");
                }

                // Self-update: show animation, then exec the new binary in-place.
                // We do NOT call teardown_layout — the new process calls setup_layout
                // which redraws the whole screen fresh. PIKU_SESSION_ID lets the
                // restarted process resume the session seamlessly.
                if let Some(new_binary) = result.replace_and_exec {
                    // Place animation in the scroll zone, below the turn output
                    let (_, rows) = term_size();
                    let scroll_bot = rows.saturating_sub(2);
                    goto(scroll_bot, 1);
                    announce_self_update(&new_binary);
                    if let Err(e) = self_update::replace_and_exec_with_env(
                        &new_binary,
                        &[("PIKU_SESSION_ID", &session_id)],
                    ) {
                        eprintln!("\x1b[31m[piku]\x1b[0m self-update failed: {e}");
                        // Re-enter layout on failure so REPL stays usable
                        let (cols, rows) = term_size();
                        setup_layout(rows, cols, &model, resolved.name(), &session_id);
                    }
                    // replace_and_exec_with_env does not return on success
                }
            }

            Ok(ReadOutcome::Cancel) => {
                // Ctrl-C / Esc: cancel current input
                let (_, rows) = term_size();
                let scroll_bot = rows.saturating_sub(2);
                goto(scroll_bot, 1);
                println!("\r\n\x1b[2m[interrupted]\x1b[0m\r");
                let (cols, rows) = term_size();
                draw_footer(
                    rows.saturating_sub(1),
                    cols,
                    &FooterState {
                        provider: resolved.name(),
                        model: &model,
                        session_id: &session_id,
                        input_tokens: total_usage.input_tokens,
                        output_tokens: total_usage.output_tokens,
                        turns: 0,
                        running_agents: 0,
                        context_pct: None,
                    },
                );
                print!("\x1b[?25h");
                let _ = io::stdout().flush();
            }

            Ok(ReadOutcome::Exit) => {
                // Ctrl-D / Ctrl-C on empty: exit
                break;
            }

            Err(err) => {
                eprintln!("\x1b[31m[input error]\x1b[0m {err}");
                break;
            }
        }
    }

    // ── Cleanup ───────────────────────────────────────────────────────────────
    let (_, rows) = term_size();
    teardown_layout(rows);

    if let Err(e) = session.save(&session_path) {
        eprintln!("\x1b[33m[warn]\x1b[0m could not save session: {e}");
    } else if !session.messages.is_empty() {
        eprintln!("\x1b[2m[session saved → {}]\x1b[0m", session_path.display());
    }

    // Session-end memory extraction: distill atomic facts from the conversation.
    if session.messages.len() > 2 {
        let transcript = piku_runtime::build_extraction_transcript(&session.messages);
        if !transcript.trim().is_empty() {
            let store_path = piku_runtime::default_store_path(&cwd);
            let mut store = piku_runtime::MemoryStore::load(&store_path);
            let embed_config = piku_runtime::embed_memory::EmbedConfig::from_env();
            // Use a short timeout — don't make the user wait forever on exit
            let extraction_future = piku_runtime::extract_and_store(
                &transcript,
                resolved.as_provider(),
                &model,
                &mut store,
                &embed_config,
            );
            match tokio::time::timeout(std::time::Duration::from_secs(15), extraction_future).await
            {
                Ok(n) if n > 0 => {
                    let _ = store.save(&store_path);
                    eprintln!("\x1b[2m[{n} memories extracted from session]\x1b[0m");
                }
                _ => {} // timeout or no memories — exit silently
            }
        }
    }

    editor.save_history_file(&history_path);

    Ok(())
}

// ── Session tail replay ────────────────────────────────────────────────────────

/// Print the last few exchanges from `session` into the scroll zone so the
/// user always has context at startup. Text is dimmed to distinguish it from
/// live output.
fn print_session_tail(session: &Session, _scroll_bot: u16) {
    use piku_runtime::session::{ContentBlock, MessageRole};

    const MAX_TAIL_MESSAGES: usize = 10; // last N messages to replay
    const MAX_TEXT_LINES: usize = 20; // cap per text block

    if session.messages.is_empty() {
        return;
    }

    let msgs = &session.messages;
    let start = msgs.len().saturating_sub(MAX_TAIL_MESSAGES);
    let tail = &msgs[start..];

    // Print a separator
    println!("\x1b[2m── prior conversation ──────────────────────────────────\x1b[0m\r");

    for msg in tail {
        match msg.role {
            MessageRole::User => {
                // Show user messages in a recognisable colour
                for block in &msg.blocks {
                    if let ContentBlock::Text { text } = block {
                        let trimmed = text.trim();
                        if trimmed.is_empty() {
                            continue;
                        }
                        println!(
                            "\x1b[36m❯ \x1b[0m\x1b[2m{}\x1b[0m\r",
                            first_n_lines(trimmed, MAX_TEXT_LINES)
                        );
                    }
                }
            }
            MessageRole::Assistant => {
                for block in &msg.blocks {
                    match block {
                        ContentBlock::Text { text } => {
                            let trimmed = text.trim();
                            if trimmed.is_empty() {
                                continue;
                            }
                            println!("\x1b[2m{}\x1b[0m\r", first_n_lines(trimmed, MAX_TEXT_LINES));
                        }
                        ContentBlock::ToolUse { name, .. } => {
                            let dname = tool_display_name(name);
                            println!("\x1b[2m  ● {dname}\x1b[0m\r");
                        }
                        ContentBlock::ToolResult { .. } => {}
                    }
                }
            }
            MessageRole::Tool => {
                // Summarise tool results briefly
                for block in &msg.blocks {
                    if let ContentBlock::ToolResult {
                        output, is_error, ..
                    } = block
                    {
                        let status = if *is_error {
                            "\x1b[31merr\x1b[0m"
                        } else {
                            "\x1b[32mok\x1b[0m"
                        };
                        println!("  \x1b[2m→ {status}\x1b[0m\r");
                        for l in output.lines().take(3) {
                            println!("    \x1b[2m{l}\x1b[0m\r");
                        }
                    }
                }
            }
            MessageRole::System => {}
        }
    }

    println!("\x1b[2m── continue ────────────────────────────────────────────\x1b[0m\r");
}

/// Return the first `n` lines of `s`, appending "… (N more lines)" if truncated.
fn first_n_lines(s: &str, n: usize) -> String {
    let lines: Vec<&str> = s.lines().collect();
    if lines.len() <= n {
        return lines.join("\r\n");
    }
    let mut out = lines[..n].join("\r\n");
    out.push_str(&format!(
        "\r\n\x1b[2m… ({} more lines)\x1b[0m",
        lines.len() - n
    ));
    out
}

// ── Slash commands ─────────────────────────────────────────────────────────────

/// Returns true if the REPL should exit.
fn handle_slash_cmd(
    input: &str,
    session: &mut Session,
    total_usage: &TokenUsage,
    current_model: &str,
    provider_name: &str,
    session_id: &str,
    model: &mut String,
    task_registry: &TaskRegistry,
    config: &crate::config::PikuConfig,
    hook_registry: &piku_runtime::HookRegistry,
) -> bool {
    let mut parts = input.trim()[1..].splitn(2, ' ');
    let cmd = parts.next().unwrap_or("").to_lowercase();
    let arg = parts.next().map(|s| s.trim().to_string());

    // Print into scroll zone
    let (_, rows) = term_size();
    let scroll_bot = rows.saturating_sub(2);
    goto(scroll_bot, 1);
    print!("\r\n");

    match cmd.as_str() {
        "help" | "h" => {
            println!(
                "\x1b[1mCommands:\x1b[0m\r
  /help          This message\r
  !command       Run shell command directly\r
  /status        Session info\r
  /cost          Token usage\r
  /model [name]  Show or switch model\r
  /tasks         List background agents\r
  /sessions      List saved sessions\r
  /clear         Clear session context\r
  /exit, /quit   Exit piku\r
\r
\x1b[1mKeys:\x1b[0m\r
  Shift+Enter    Insert newline (multiline input)\r
  Ctrl+J         Insert newline (alternative)\r
  Ctrl+A / E     Move to start / end of line\r
  Ctrl+K / U     Kill to end / start of line\r
  Ctrl+W         Kill previous word\r
  Ctrl+Y         Yank (paste killed text)\r
  Right          Accept history suggestion\r
  Esc            Clear input\r"
            );
        }
        "status" => {
            println!(
                "\x1b[1mStatus:\x1b[0m  provider={provider_name}  model={current_model}  msgs={}\r",
                session.messages.len(),
            );
        }
        "cost" => {
            println!(
                "\x1b[1mTokens:\x1b[0m  in={}  out={}  total={}\r",
                total_usage.input_tokens,
                total_usage.output_tokens,
                total_usage.total_tokens(),
            );
        }
        "model" => match arg {
            None => println!("model: {current_model}\r"),
            Some(m) => {
                m.clone_into(model);
                println!("\x1b[2m[model → {m}]\x1b[0m\r");
                let (cols, rows) = term_size();
                draw_footer(
                    rows.saturating_sub(1),
                    cols,
                    &FooterState {
                        provider: provider_name,
                        model: &m,
                        session_id,
                        input_tokens: total_usage.input_tokens,
                        output_tokens: total_usage.output_tokens,
                        turns: 0,
                        running_agents: 0,
                        context_pct: None,
                    },
                );
            }
        },
        "sessions" => {
            if let Ok(dir) = crate::sessions_dir() {
                println!("\x1b[1mSessions:\x1b[0m  {}\r", dir.display());
                if let Ok(entries) = std::fs::read_dir(&dir) {
                    let mut files: Vec<_> = entries
                        .filter_map(std::result::Result::ok)
                        .filter(|e| e.path().extension().is_some_and(|x| x == "json"))
                        .collect();
                    files.sort_by_key(|e| {
                        e.metadata()
                            .and_then(|m| m.modified())
                            .unwrap_or(std::time::UNIX_EPOCH)
                    });
                    files.reverse();
                    for f in files.iter().take(15) {
                        let raw_name = f.file_name();
                        let name = raw_name.to_string_lossy();
                        let name = name.trim_end_matches(".json");
                        println!("  {name}\r");
                    }
                }
            }
        }
        "tasks" | "agents" => {
            let tasks = task_registry.all();
            if tasks.is_empty() {
                println!("\x1b[2mno background agents\x1b[0m\r");
            } else {
                println!("\x1b[1mBackground agents:\x1b[0m\r");
                for t in &tasks {
                    let status_color = match t.status {
                        TaskStatus::Running => "\x1b[33m",
                        TaskStatus::Done => "\x1b[32m",
                        TaskStatus::Failed => "\x1b[31m",
                    };
                    let elapsed = t.elapsed().as_secs();
                    println!(
                        "  {status_color}{}\x1b[0m  \x1b[2m{}\x1b[0m  {}s  depth={}\r",
                        t.name, t.id, elapsed, t.depth
                    );
                    if let Some(ref out) = t.output {
                        let preview: String = out.chars().take(80).collect();
                        println!("    \x1b[2m{preview}\x1b[0m\r");
                    }
                }
            }
        }
        "clear" => {
            session.messages.clear();
            println!("\x1b[2m[session cleared]\x1b[0m\r");
        }
        "permissions" | "perms" => {
            if config.allow.is_empty() && config.deny.is_empty() {
                println!("\x1b[2m[no permission rules configured]\x1b[0m\r");
                println!("\x1b[2mAdd \"allow\"/\"deny\" arrays to ~/.config/piku/settings.json or .piku/settings.json\x1b[0m\r");
            } else {
                if !config.allow.is_empty() {
                    println!("\x1b[32mAllow:\x1b[0m\r");
                    for rule in &config.allow {
                        println!("  {rule}\r");
                    }
                }
                if !config.deny.is_empty() {
                    println!("\x1b[31mDeny:\x1b[0m\r");
                    for rule in &config.deny {
                        println!("  {rule}\r");
                    }
                }
            }
        }
        "hooks" => {
            if hook_registry.has_hooks() {
                println!("\x1b[1mActive hooks:\x1b[0m\r");
                let summary = hook_registry.summary();
                for line in summary.lines() {
                    println!("  {line}\r");
                }
            } else {
                println!("\x1b[2m[no hooks configured]\x1b[0m\r");
                println!(
                    "\x1b[2mAdd hooks to ~/.config/piku/hooks.json or .piku/hooks.json\x1b[0m\r"
                );
            }
        }
        "exit" | "quit" | "q" => {
            return true;
        }
        other => {
            println!("\x1b[33m[unknown]\x1b[0m /{other} — try /help\r");
        }
    }

    let (cols, rows) = term_size();
    draw_footer(
        rows.saturating_sub(1),
        cols,
        &FooterState {
            provider: provider_name,
            model,
            session_id,
            input_tokens: total_usage.input_tokens,
            output_tokens: total_usage.output_tokens,
            turns: 0,
            running_agents: 0,
            context_pct: None,
        },
    );
    let _ = io::stdout().flush();
    false
}

// ── Tool formatting ────────────────────────────────────────────────────────────

/// Friendly display name for a tool (Title case, no underscores).
fn tool_display_name(tool_name: &str) -> &str {
    match tool_name {
        "read_file" => "Read",
        "write_file" => "Write",
        "edit_file" => "Update",
        "bash" => "Bash",
        "glob" => "Glob",
        "grep" => "Grep",
        "list_dir" => "List",
        _ => tool_name,
    }
}

/// Format tool result lines with tree connector and truncation.
/// `max_lines`: how many preview lines to show.
/// `dim`: whether the content lines should be dim.
pub(crate) fn format_result_lines(result: &str, max_lines: usize, dim: bool) -> String {
    const MAX_LINE_WIDTH: usize = 200;

    if result.trim().is_empty() {
        return String::new();
    }

    let connector = "\x1b[2m⎿\x1b[0m ";
    let lines: Vec<&str> = result.lines().collect();
    let total = lines.len();
    let mut out = Vec::new();

    // +1 exception: if only 1 extra line, just show it (Claude Code pattern)
    let show_lines = if total == max_lines + 1 {
        max_lines + 1
    } else {
        max_lines
    };

    for (i, line) in lines.iter().take(show_lines).enumerate() {
        // Truncate very long lines (minified JSON, binary-ish content)
        let display = if line.len() > MAX_LINE_WIDTH {
            format!("{}…", &line[..MAX_LINE_WIDTH])
        } else {
            (*line).to_string()
        };

        let prefix = if i == 0 { connector } else { "  " };
        if dim {
            out.push(format!("{prefix}\x1b[2m{display}\x1b[0m"));
        } else {
            out.push(format!("{prefix}{display}"));
        }
    }

    if total > show_lines {
        out.push(format!("  \x1b[2m… +{} lines\x1b[0m", total - show_lines));
    }

    out.join("\r\n")
}

pub(crate) fn format_tool_result(tool_name: &str, result: &str, is_error: bool) -> String {
    if result.trim().is_empty() {
        return String::new();
    }

    if is_error {
        // Errors in red, show generously
        let connector = "\x1b[2m⎿\x1b[0m ";
        let lines: Vec<&str> = result.lines().collect();
        let mut out = Vec::new();
        for (i, line) in lines.iter().take(8).enumerate() {
            let prefix = if i == 0 { connector } else { "  " };
            out.push(format!("{prefix}\x1b[31m{line}\x1b[0m"));
        }
        if lines.len() > 8 {
            out.push(format!("  \x1b[2m… +{} lines\x1b[0m", lines.len() - 8));
        }
        return out.join("\r\n");
    }

    match tool_name {
        // Bash output is primary content — not dimmed, generous preview.
        // Try to pretty-print JSON output for readability.
        "bash" => {
            let formatted = crate::try_pretty_json(result);
            format_result_lines(&formatted, 8, false)
        }
        // File reads are context — dim, shorter
        "read_file" => format_result_lines(result, 4, true),
        // Edit/write just show the success summary
        "edit_file" | "write_file" => {
            let trimmed = result.trim();
            if trimmed.is_empty() {
                String::new()
            } else {
                format!("\x1b[2m└\x1b[0m {trimmed}")
            }
        }
        // File lists — dim, moderate
        "glob" | "list_dir" => format_result_lines(result, 6, true),
        // Search results — dim, moderate
        "grep" => format_result_lines(result, 6, true),
        // Unknown tools — dim, short
        _ => format_result_lines(result, 4, true),
    }
}

// ── Self-update animation ──────────────────────────────────────────────────────

/// Print an animated banner in the scroll zone then return.
/// Caller must position cursor in the scroll zone before calling.
fn announce_self_update(new_binary: &std::path::Path) {
    use std::io::Write;
    let mut stdout = io::stdout();

    let frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
    let bin_name = new_binary
        .file_name()
        .map(|n| n.to_string_lossy())
        .unwrap_or_default();

    // Save cursor position, then spin on a fresh line
    let _ = write!(stdout, "\r\n\x1b[s"); // newline + save position
    let _ = stdout.flush();

    for i in 0..12 {
        let frame = frames[i % frames.len()];
        let _ = write!(
            stdout,
            "\x1b[u\x1b[2K\x1b[1;33m{frame} rebuilding — preparing restart…\x1b[0m \x1b[2m({bin_name})\x1b[0m"
        );
        let _ = stdout.flush();
        std::thread::sleep(std::time::Duration::from_millis(80));
    }

    // Final line
    let _ = writeln!(
        stdout,
        "\x1b[u\x1b[2K\x1b[1;32m↺ new binary ready — restarting now\x1b[0m  \x1b[2m{bin_name}\x1b[0m"
    );
    let _ = stdout.flush();
}

// ── Unit tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn strip_ansi(s: &str) -> String {
        let mut out = String::new();
        let mut chars = s.chars().peekable();
        while let Some(ch) = chars.next() {
            if ch == '\x1b' {
                if chars.peek() == Some(&'[') {
                    chars.next();
                    for c in chars.by_ref() {
                        if c.is_ascii_alphabetic() {
                            break;
                        }
                    }
                }
            } else {
                out.push(ch);
            }
        }
        out
    }

    fn make_footer() -> FooterState<'static> {
        FooterState {
            provider: "openrouter",
            model: "claude-sonnet",
            session_id: "session-123456789-42",
            input_tokens: 5000,
            output_tokens: 200,
            turns: 3,
            running_agents: 0,
            context_pct: Some(60),
        }
    }

    // ── Footer ──────────────────────────────────────────────────────

    #[test]
    fn footer_contains_provider_and_model() {
        let s = make_footer();
        let line = render_footer(120, &s);
        let plain = strip_ansi(&line);
        assert!(plain.contains("openrouter"), "provider: {plain}");
        assert!(plain.contains("claude-sonnet"), "model: {plain}");
    }

    #[test]
    fn footer_contains_token_counts() {
        let s = make_footer();
        let line = render_footer(120, &s);
        let plain = strip_ansi(&line);
        assert!(plain.contains("5.0k"), "input tokens: {plain}");
    }

    #[test]
    fn footer_contains_context_pct() {
        let s = make_footer();
        let line = render_footer(120, &s);
        let plain = strip_ansi(&line);
        assert!(plain.contains("60%"), "context pct: {plain}");
    }

    #[test]
    fn footer_contains_help_hint() {
        let s = make_footer();
        let line = render_footer(120, &s);
        let plain = strip_ansi(&line);
        assert!(plain.contains("/help"), "hint: {plain}");
    }

    #[test]
    fn footer_drops_segments_when_narrow() {
        let s = make_footer();
        let line = render_footer(40, &s);
        let plain = strip_ansi(&line);
        assert!(plain.contains("openrouter"), "model always shown: {plain}");
    }

    #[test]
    fn footer_shows_agents_when_nonzero() {
        let mut s = make_footer();
        s.running_agents = 2;
        let line = render_footer(120, &s);
        let plain = strip_ansi(&line);
        assert!(plain.contains("2 bg"), "agents: {plain}");
    }

    #[test]
    fn footer_hides_tokens_when_zero() {
        let mut s = make_footer();
        s.input_tokens = 0;
        s.output_tokens = 0;
        let line = render_footer(120, &s);
        let plain = strip_ansi(&line);
        assert!(!plain.contains('↑'), "no token arrows when zero: {plain}");
    }

    #[test]
    fn footer_amber_at_75_pct() {
        let mut s = make_footer();
        s.context_pct = Some(80);
        let line = render_footer(120, &s);
        assert!(line.contains("\x1b[33m"), "80% should be amber");
    }

    #[test]
    fn footer_red_at_90_pct() {
        let mut s = make_footer();
        s.context_pct = Some(95);
        let line = render_footer(120, &s);
        assert!(line.contains("\x1b[31m"), "95% should be red");
    }

    #[test]
    fn footer_all_segments_present_wide() {
        let s = make_footer();
        let line = render_footer(120, &s);
        let plain = strip_ansi(&line);
        // All segments should be present at 120 cols
        assert!(plain.contains("openrouter"), "provider");
        assert!(plain.contains("60%"), "context");
        assert!(plain.contains("5.0k"), "tokens");
        assert!(plain.contains("turns"), "turns");
        assert!(plain.contains("…42"), "session");
        assert!(plain.contains("/help"), "hint");
    }

    // ── fmt_tokens ──────────────────────────────────────────────────

    #[test]
    fn fmt_tokens_small() {
        assert_eq!(fmt_tokens(42), "42");
        assert_eq!(fmt_tokens(999), "999");
    }

    #[test]
    fn fmt_tokens_thousands() {
        assert_eq!(fmt_tokens(1500), "1.5k");
    }

    #[test]
    fn fmt_tokens_large() {
        assert_eq!(fmt_tokens(15000), "15k");
        assert_eq!(fmt_tokens(123_456), "123k");
    }

    // ── short_session_id ────────────────────────────────────────────

    #[test]
    fn session_id_extracts_suffix() {
        assert_eq!(short_session_id("session-12345-678"), "…678");
    }

    // ── context_pct ─────────────────────────────────────────────────

    #[test]
    fn context_pct_zero_is_none() {
        assert_eq!(context_pct(0), None);
    }

    #[test]
    fn context_pct_half() {
        assert_eq!(context_pct(100_000), Some(50));
    }

    #[test]
    fn context_pct_caps() {
        assert_eq!(context_pct(300_000), Some(100));
    }

    // ── format_result_lines ─────────────────────────────────────────

    #[test]
    fn result_lines_empty() {
        assert!(format_result_lines("", 5, false).is_empty());
    }

    #[test]
    fn result_lines_connector_first() {
        let out = format_result_lines("line1\nline2", 5, false);
        let plain = strip_ansi(&out);
        assert!(plain.contains('⎿'), "connector: {plain}");
    }

    #[test]
    fn result_lines_plus_one_exception() {
        let out = format_result_lines("a\nb\nc\nd\ne", 4, false);
        let plain = strip_ansi(&out);
        assert!(plain.contains('e'), "5th line should show");
        assert!(!plain.contains('+'), "no +1 truncation");
    }

    #[test]
    fn result_lines_truncates_long() {
        let long = "x".repeat(300);
        let out = format_result_lines(&long, 5, false);
        let plain = strip_ansi(&out);
        assert!(plain.contains('…'), "long line truncated");
    }

    #[test]
    fn result_lines_dim_mode() {
        let out = format_result_lines("hello", 5, true);
        assert!(out.contains("\x1b[2m"), "dim mode");
    }

    // ── format_tool_result ──────────────────────────────────────────

    #[test]
    fn tool_result_bash_not_dimmed() {
        let out = format_tool_result("bash", "output", false);
        let after = out.split('⎿').nth(1).unwrap_or("");
        assert!(!after.starts_with(" \x1b[2m"), "bash not dimmed");
    }

    #[test]
    fn tool_result_read_dimmed() {
        let out = format_tool_result("read_file", "content", false);
        assert!(out.contains("\x1b[2m"), "read_file dimmed");
    }

    #[test]
    fn tool_result_error_red() {
        let out = format_tool_result("bash", "fail", true);
        assert!(out.contains("\x1b[31m"), "error red");
    }

    #[test]
    fn tool_result_empty() {
        assert!(format_tool_result("bash", "", false).is_empty());
    }
}
