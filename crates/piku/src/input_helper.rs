#![allow(warnings)]

//! Shared rustyline Helper for piku REPLs.
//!
//! Provides:
//! - Tab-completion for slash commands
//! - Dim hint text when the input line is empty
//! - Cyan syntax highlighting for `/commands`
//! - Multiline via Validator (pasted newlines accumulate; Enter on a
//!   complete line submits)

use std::borrow::Cow;

use rustyline::completion::{Completer, Pair};
use rustyline::highlight::Highlighter;
use rustyline::hint::Hinter;
use rustyline::validate::{ValidationContext, ValidationResult, Validator};
use rustyline::{Context, Helper};

/// All known slash commands (used for completion and validation).
pub const SLASH_CMDS: &[&str] = &[
    "/help", "/status", "/cost", "/model", "/tasks", "/agents", "/sessions", "/clear", "/exit",
    "/quit",
];

// ── Helper ──────────────────────────────────────────────────────────────────

pub struct PikuHelper {
    /// Hint shown when the input line is empty.
    pub placeholder: &'static str,
}

impl PikuHelper {
    pub fn new() -> Self {
        Self {
            placeholder: "Send a message or /help",
        }
    }
}

impl Helper for PikuHelper {}

// ── Completer ───────────────────────────────────────────────────────────────

impl Completer for PikuHelper {
    type Candidate = Pair;

    fn complete(
        &self,
        line: &str,
        pos: usize,
        _ctx: &Context<'_>,
    ) -> rustyline::Result<(usize, Vec<Pair>)> {
        let word = &line[..pos];
        // Only complete if we're in the first token and it starts with /
        if !word.starts_with('/') || word.contains(' ') {
            return Ok((0, vec![]));
        }
        let candidates: Vec<Pair> = SLASH_CMDS
            .iter()
            .filter(|cmd| cmd.starts_with(word))
            .map(|cmd| Pair {
                display: cmd.to_string(),
                replacement: cmd.to_string(),
            })
            .collect();
        Ok((0, candidates))
    }
}

// ── Hinter ──────────────────────────────────────────────────────────────────

impl Hinter for PikuHelper {
    type Hint = String;

    fn hint(&self, line: &str, _pos: usize, _ctx: &Context<'_>) -> Option<String> {
        if line.is_empty() {
            return Some(self.placeholder.to_string());
        }
        // Slash command prefix hint: show the first matching command
        if line.starts_with('/') && !line.contains(' ') {
            for cmd in SLASH_CMDS {
                if cmd.starts_with(line) && *cmd != line {
                    // Show the remaining suffix
                    return Some(cmd[line.len()..].to_string());
                }
            }
        }
        None
    }
}

// ── Highlighter ─────────────────────────────────────────────────────────────

impl Highlighter for PikuHelper {
    fn highlight<'l>(&self, line: &'l str, _pos: usize) -> Cow<'l, str> {
        if line.starts_with('/') {
            // Cyan for slash commands
            Cow::Owned(format!("\x1b[36m{line}\x1b[0m"))
        } else {
            Cow::Borrowed(line)
        }
    }

    fn highlight_hint<'h>(&self, hint: &'h str) -> Cow<'h, str> {
        // Dim hint text
        Cow::Owned(format!("\x1b[2m{hint}\x1b[0m"))
    }

    fn highlight_char(&self, line: &str, _pos: usize, _forced: bool) -> bool {
        // Re-highlight when typing a slash command
        line.starts_with('/')
    }
}

// ── Validator ───────────────────────────────────────────────────────────────

impl Validator for PikuHelper {
    fn validate(&self, _ctx: &mut ValidationContext) -> rustyline::Result<ValidationResult> {
        // Always accept on Enter. This is a chat REPL, not a code editor --
        // users expect Enter to submit. Pasted multiline text arrives as a
        // single buffer anyway (rustyline accumulates rapid keystrokes).
        Ok(ValidationResult::Valid(None))
    }
}

// ── Editor factory ──────────────────────────────────────────────────────────

pub type PikuEditor = rustyline::Editor<PikuHelper, rustyline::history::DefaultHistory>;

pub fn build_editor() -> anyhow::Result<PikuEditor> {
    let mut builder = rustyline::Config::builder()
        .history_ignore_space(true)
        .completion_type(rustyline::CompletionType::List);
    if std::env::var("PIKU_VI").is_ok() {
        builder = builder.edit_mode(rustyline::EditMode::Vi);
    }
    let mut rl = PikuEditor::with_config(builder.build())?;
    rl.set_helper(Some(PikuHelper::new()));
    Ok(rl)
}
