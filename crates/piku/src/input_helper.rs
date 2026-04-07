//! Custom line editor for piku TUI.
//!
//! Built on crossterm raw mode. Supports:
//! - Shift+Enter / Ctrl+J for newline insertion (multiline input)
//! - Emacs readline keybindings (Ctrl+A/E/K/U/W/D, Alt+B/F)
//! - History navigation (Up/Down, with draft preservation)
//! - Tab completion for slash commands
//! - Continuation prompt for multiline display
//! - Dim placeholder hint when input is empty

use std::io::{self, IsTerminal, Write};

use crossterm::cursor::{MoveDown, MoveToColumn, MoveUp};
use crossterm::event::{
    self, Event, KeyCode, KeyEvent, KeyModifiers, KeyboardEnhancementFlags,
    PopKeyboardEnhancementFlags, PushKeyboardEnhancementFlags,
};
use crossterm::terminal::{self, Clear, ClearType};
use crossterm::{execute, queue};

/// All known slash commands (used for completion).
pub const SLASH_CMDS: &[&str] = &[
    "/help", "/status", "/cost", "/model", "/tasks", "/agents", "/sessions", "/clear", "/exit",
    "/quit",
];

// ── Input buffer ────────────────────────────────────────────────────────────

/// Characters that define word boundaries (matches Codex's WORD_SEPARATORS).
const WORD_SEPS: &str = " \t\n`~!@#$%^&*()-=+[{]}\\|;:'\",.<>/?";

fn is_word_sep(c: char) -> bool {
    WORD_SEPS.contains(c)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InputBuffer {
    buffer: String,
    cursor: usize,
    /// Kill ring (single entry, like Codex). Ctrl+K/U/W store here, Ctrl+Y yanks.
    kill_buffer: String,
}

impl InputBuffer {
    #[must_use]
    pub fn new() -> Self {
        Self {
            buffer: String::new(),
            cursor: 0,
            kill_buffer: String::new(),
        }
    }

    pub fn insert(&mut self, ch: char) {
        self.buffer.insert(self.cursor, ch);
        self.cursor += ch.len_utf8();
    }

    pub fn insert_newline(&mut self) {
        self.insert('\n');
    }

    pub fn backspace(&mut self) {
        if self.cursor == 0 {
            return;
        }
        let prev = self.buffer[..self.cursor]
            .char_indices()
            .last()
            .map_or(0, |(i, _)| i);
        self.buffer.drain(prev..self.cursor);
        self.cursor = prev;
    }

    /// Delete character under cursor (Ctrl+D behavior when buffer non-empty).
    pub fn delete_char(&mut self) {
        if self.cursor >= self.buffer.len() {
            return;
        }
        if let Some(ch) = self.buffer[self.cursor..].chars().next() {
            self.buffer.drain(self.cursor..self.cursor + ch.len_utf8());
        }
    }

    pub fn move_left(&mut self) {
        if self.cursor == 0 {
            return;
        }
        self.cursor = self.buffer[..self.cursor]
            .char_indices()
            .last()
            .map_or(0, |(i, _)| i);
    }

    pub fn move_right(&mut self) {
        if self.cursor >= self.buffer.len() {
            return;
        }
        if let Some(ch) = self.buffer[self.cursor..].chars().next() {
            self.cursor += ch.len_utf8();
        }
    }

    pub fn move_home(&mut self) {
        self.cursor = 0;
    }

    pub fn move_end(&mut self) {
        self.cursor = self.buffer.len();
    }

    /// Move cursor back one word (Alt+B / Ctrl+Left).
    /// Uses separator characters for boundaries (like Codex).
    pub fn move_word_back(&mut self) {
        if self.cursor == 0 {
            return;
        }
        let before = &self.buffer[..self.cursor];
        let chars: Vec<char> = before.chars().collect();
        let mut i = chars.len();
        // Skip separators
        while i > 0 && is_word_sep(chars[i - 1]) {
            i -= 1;
        }
        // Skip word chars
        while i > 0 && !is_word_sep(chars[i - 1]) {
            i -= 1;
        }
        self.cursor = chars[..i].iter().map(|c| c.len_utf8()).sum();
    }

    /// Move cursor forward one word (Alt+F / Ctrl+Right).
    pub fn move_word_forward(&mut self) {
        let after = &self.buffer[self.cursor..];
        let chars: Vec<char> = after.chars().collect();
        let mut i = 0;
        // Skip current word chars
        while i < chars.len() && !is_word_sep(chars[i]) {
            i += 1;
        }
        // Skip separators
        while i < chars.len() && is_word_sep(chars[i]) {
            i += 1;
        }
        self.cursor += chars[..i].iter().map(|c| c.len_utf8()).sum::<usize>();
    }

    /// Transpose the two characters before the cursor (Ctrl+T).
    pub fn transpose_chars(&mut self) {
        if self.cursor == 0 || self.buffer.len() < 2 {
            return;
        }
        // If cursor is at end, transpose the last two chars
        let at_end = self.cursor >= self.buffer.len();
        if at_end {
            self.move_left();
        }
        if self.cursor == 0 {
            return;
        }
        let prev_start = self.buffer[..self.cursor]
            .char_indices()
            .last()
            .map_or(0, |(i, _)| i);
        let curr_char = self.buffer[self.cursor..].chars().next();
        let prev_char = self.buffer[prev_start..self.cursor].chars().next();
        if let (Some(p), Some(c)) = (prev_char, curr_char) {
            let after = self.cursor + c.len_utf8();
            let replacement = format!("{c}{p}");
            self.buffer.replace_range(prev_start..after, &replacement);
            self.cursor = prev_start + replacement.len();
        }
    }

    /// Kill from cursor to end of line (Ctrl+K). Stores killed text.
    pub fn kill_to_end(&mut self) {
        self.kill_buffer = self.buffer[self.cursor..].to_string();
        self.buffer.truncate(self.cursor);
    }

    /// Kill from start to cursor (Ctrl+U). Stores killed text.
    pub fn kill_to_start(&mut self) {
        self.kill_buffer = self.buffer[..self.cursor].to_string();
        self.buffer.drain(..self.cursor);
        self.cursor = 0;
    }

    /// Kill previous word (Ctrl+W). Stores killed text.
    pub fn kill_word_back(&mut self) {
        let old_cursor = self.cursor;
        self.move_word_back();
        self.kill_buffer = self.buffer[self.cursor..old_cursor].to_string();
        self.buffer.drain(self.cursor..old_cursor);
    }

    /// Yank (paste) the kill buffer at cursor (Ctrl+Y).
    pub fn yank(&mut self) {
        if self.kill_buffer.is_empty() {
            return;
        }
        let text = self.kill_buffer.clone();
        for ch in text.chars() {
            self.insert(ch);
        }
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.buffer
    }

    #[must_use] 
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    pub fn clear(&mut self) {
        self.buffer.clear();
        self.cursor = 0;
    }

    pub fn replace(&mut self, value: impl Into<String>) {
        self.buffer = value.into();
        self.cursor = self.buffer.len();
    }

    /// Try to complete a slash command prefix. Returns true if anything changed.
    pub fn complete_slash_command(&mut self, candidates: &[&str]) -> bool {
        // Only complete if cursor is at end and input starts with /
        if self.cursor != self.buffer.len() {
            return false;
        }
        let prefix = &self.buffer[..self.cursor];
        if prefix.contains(char::is_whitespace) || !prefix.starts_with('/') {
            return false;
        }
        let matches: Vec<&str> = candidates
            .iter()
            .filter(|c| c.starts_with(prefix))
            .copied()
            .collect();
        if matches.is_empty() {
            return false;
        }
        let replacement = longest_common_prefix(&matches);
        if replacement == prefix {
            return false;
        }
        self.replace(replacement);
        true
    }

    #[cfg(test)]
    #[must_use]
    pub fn cursor(&self) -> usize {
        self.cursor
    }
}

// ── Read outcome ────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReadOutcome {
    Submit(String),
    Cancel,
    Exit,
}

// ── Line editor ─────────────────────────────────────────────────────────────

pub struct LineEditor {
    prompt: String,
    continuation_prompt: String,
    placeholder: String,
    history: Vec<String>,
    history_index: Option<usize>,
    draft: Option<String>,
}

impl LineEditor {
    #[must_use]
    pub fn new(prompt: impl Into<String>) -> Self {
        Self {
            prompt: prompt.into(),
            continuation_prompt: String::from("  "),
            placeholder: String::from("Send a message or /help"),
            history: Vec::new(),
            history_index: None,
            draft: None,
        }
    }

    pub fn set_prompt(&mut self, prompt: impl Into<String>) {
        self.prompt = prompt.into();
    }

    pub fn push_history(&mut self, entry: impl Into<String>) {
        let entry = entry.into();
        if entry.trim().is_empty() {
            return;
        }
        // Deduplicate consecutive entries
        if self.history.last().map(String::as_str) != Some(entry.as_str()) {
            self.history.push(entry);
        }
        self.history_index = None;
        self.draft = None;
    }

    pub fn load_history_file(&mut self, path: &std::path::Path) {
        if let Ok(contents) = std::fs::read_to_string(path) {
            for line in contents.lines() {
                let trimmed = line.trim();
                if !trimmed.is_empty() {
                    self.history.push(trimmed.to_string());
                }
            }
        }
    }

    pub fn save_history_file(&self, path: &std::path::Path) {
        // Keep last 500 entries
        let start = self.history.len().saturating_sub(500);
        let contents: String = self.history[start..]
            .iter()
            .map(|s| format!("{s}\n"))
            .collect();
        let _ = std::fs::write(path, contents);
    }

    /// Read a line of input. After submit, moves the cursor past the
    /// rendered input and emits a newline. Use `read_line_raw` when the
    /// caller manages cursor positioning (e.g. TUI scroll regions).
    pub fn read_line(&mut self) -> io::Result<ReadOutcome> {
        let (outcome, rendered_lines) = self.read_line_inner()?;

        // Move cursor past the rendered input so subsequent output
        // doesn't overwrite it.
        if let ReadOutcome::Submit(_) = &outcome {
            let mut stdout = io::stdout();
            if rendered_lines > 1 {
                let _ = execute!(stdout, MoveDown(sat_u16(rendered_lines - 1)));
            }
            let _ = execute!(stdout, MoveToColumn(0));
            let _ = write!(stdout, "\r\n");
            let _ = stdout.flush();
        }

        Ok(outcome)
    }

    /// Read a line without post-submit cursor movement.
    /// The caller is responsible for positioning after this returns.
    /// Use this in TUI modes with scroll regions.
    pub fn read_line_raw(&mut self) -> io::Result<ReadOutcome> {
        let (outcome, _) = self.read_line_inner()?;
        Ok(outcome)
    }

    fn read_line_inner(&mut self) -> io::Result<(ReadOutcome, usize)> {
        if !io::stdin().is_terminal() || !io::stdout().is_terminal() {
            return self.read_line_fallback().map(|o| (o, 1));
        }

        let was_raw = terminal::is_raw_mode_enabled().unwrap_or(false);
        if !was_raw {
            terminal::enable_raw_mode()?;
        }

        // Enable kitty keyboard protocol so Shift+Enter is distinguishable
        // from plain Enter. Terminals that don't support it silently ignore this.
        let mut stdout = io::stdout();
        let _ = execute!(
            stdout,
            PushKeyboardEnhancementFlags(
                KeyboardEnhancementFlags::DISAMBIGUATE_ESCAPE_CODES
            )
        );

        let mut input = InputBuffer::new();
        let mut rendered_lines = 1usize;
        self.redraw(&mut stdout, &input, rendered_lines)?;

        let outcome = loop {
            match event::read() {
                Ok(Event::Key(key)) => match self.handle_key(key, &mut input) {
                    Action::Continue => {
                        rendered_lines = self.redraw(&mut stdout, &input, rendered_lines)?;
                    }
                    Action::Submit => {
                        break ReadOutcome::Submit(input.as_str().to_owned());
                    }
                    Action::Cancel => {
                        break ReadOutcome::Cancel;
                    }
                    Action::Exit => {
                        break ReadOutcome::Exit;
                    }
                },
                Ok(Event::Resize(..)) => {
                    rendered_lines = self.redraw(&mut stdout, &input, rendered_lines)?;
                }
                Ok(_) => {}
                Err(e) => {
                    let _ = execute!(stdout, PopKeyboardEnhancementFlags);
                    if !was_raw {
                        let _ = terminal::disable_raw_mode();
                    }
                    return Err(e);
                }
            }
        };

        let _ = execute!(stdout, PopKeyboardEnhancementFlags);
        if !was_raw {
            terminal::disable_raw_mode()?;
        }

        self.history_index = None;
        self.draft = None;

        Ok((outcome, rendered_lines))
    }

    fn read_line_fallback(&self) -> io::Result<ReadOutcome> {
        let mut stdout = io::stdout();
        write!(stdout, "{}", self.prompt)?;
        stdout.flush()?;
        let mut buf = String::new();
        let n = io::stdin().read_line(&mut buf)?;
        if n == 0 {
            return Ok(ReadOutcome::Exit);
        }
        while matches!(buf.chars().last(), Some('\n' | '\r')) {
            buf.pop();
        }
        Ok(ReadOutcome::Submit(buf))
    }

    fn handle_key(&mut self, key: KeyEvent, input: &mut InputBuffer) -> Action {
        match key {
            // ── Ctrl combos ─────────────────────────────────────────────
            KeyEvent {
                code: KeyCode::Char('c'),
                modifiers,
                ..
            } if modifiers.contains(KeyModifiers::CONTROL) => {
                if input.is_empty() {
                    Action::Exit
                } else {
                    input.clear();
                    self.history_index = None;
                    self.draft = None;
                    Action::Cancel
                }
            }
            KeyEvent {
                code: KeyCode::Char('d'),
                modifiers,
                ..
            } if modifiers.contains(KeyModifiers::CONTROL) => {
                if input.is_empty() {
                    Action::Exit
                } else {
                    input.delete_char();
                    Action::Continue
                }
            }
            // Ctrl+J / Shift+Enter → insert newline
            KeyEvent {
                code: KeyCode::Char('j'),
                modifiers,
                ..
            } if modifiers.contains(KeyModifiers::CONTROL) => {
                input.insert_newline();
                Action::Continue
            }
            KeyEvent {
                code: KeyCode::Enter,
                modifiers,
                ..
            } if modifiers.contains(KeyModifiers::SHIFT) => {
                input.insert_newline();
                Action::Continue
            }
            // Readline: Ctrl+A = home, Ctrl+E = end
            KeyEvent {
                code: KeyCode::Char('a'),
                modifiers,
                ..
            } if modifiers.contains(KeyModifiers::CONTROL) => {
                input.move_home();
                Action::Continue
            }
            KeyEvent {
                code: KeyCode::Char('e'),
                modifiers,
                ..
            } if modifiers.contains(KeyModifiers::CONTROL) => {
                input.move_end();
                Action::Continue
            }
            // Readline: Ctrl+K = kill to end, Ctrl+U = kill to start
            KeyEvent {
                code: KeyCode::Char('k'),
                modifiers,
                ..
            } if modifiers.contains(KeyModifiers::CONTROL) => {
                input.kill_to_end();
                Action::Continue
            }
            KeyEvent {
                code: KeyCode::Char('u'),
                modifiers,
                ..
            } if modifiers.contains(KeyModifiers::CONTROL) => {
                input.kill_to_start();
                Action::Continue
            }
            // Readline: Ctrl+W = kill word back
            KeyEvent {
                code: KeyCode::Char('w'),
                modifiers,
                ..
            } if modifiers.contains(KeyModifiers::CONTROL) => {
                input.kill_word_back();
                Action::Continue
            }
            // Readline: Ctrl+Y = yank (paste kill buffer)
            KeyEvent {
                code: KeyCode::Char('y'),
                modifiers,
                ..
            } if modifiers.contains(KeyModifiers::CONTROL) => {
                input.yank();
                Action::Continue
            }
            // Readline: Ctrl+B = left, Ctrl+F = right
            KeyEvent {
                code: KeyCode::Char('b'),
                modifiers,
                ..
            } if modifiers.contains(KeyModifiers::CONTROL) => {
                input.move_left();
                Action::Continue
            }
            KeyEvent {
                code: KeyCode::Char('f'),
                modifiers,
                ..
            } if modifiers.contains(KeyModifiers::CONTROL) => {
                input.move_right();
                Action::Continue
            }
            // Readline: Ctrl+T = transpose chars
            KeyEvent {
                code: KeyCode::Char('t'),
                modifiers,
                ..
            } if modifiers.contains(KeyModifiers::CONTROL) => {
                input.transpose_chars();
                Action::Continue
            }
            // Readline: Ctrl+P = history up, Ctrl+N = history down
            KeyEvent {
                code: KeyCode::Char('p'),
                modifiers,
                ..
            } if modifiers.contains(KeyModifiers::CONTROL) => {
                self.history_up(input);
                Action::Continue
            }
            KeyEvent {
                code: KeyCode::Char('n'),
                modifiers,
                ..
            } if modifiers.contains(KeyModifiers::CONTROL) => {
                self.history_down(input);
                Action::Continue
            }
            // Alt+B = word back, Alt+F = word forward
            KeyEvent {
                code: KeyCode::Char('b'),
                modifiers,
                ..
            } if modifiers.contains(KeyModifiers::ALT) => {
                input.move_word_back();
                Action::Continue
            }
            KeyEvent {
                code: KeyCode::Char('f'),
                modifiers,
                ..
            } if modifiers.contains(KeyModifiers::ALT) => {
                input.move_word_forward();
                Action::Continue
            }
            // ── Basic keys ──────────────────────────────────────────────
            KeyEvent {
                code: KeyCode::Enter,
                ..
            } => Action::Submit,
            KeyEvent {
                code: KeyCode::Backspace,
                ..
            } => {
                input.backspace();
                Action::Continue
            }
            KeyEvent {
                code: KeyCode::Delete,
                ..
            } => {
                input.delete_char();
                Action::Continue
            }
            KeyEvent {
                code: KeyCode::Left,
                modifiers,
                ..
            } if modifiers.contains(KeyModifiers::ALT) => {
                input.move_word_back();
                Action::Continue
            }
            KeyEvent {
                code: KeyCode::Right,
                modifiers,
                ..
            } if modifiers.contains(KeyModifiers::ALT) => {
                input.move_word_forward();
                Action::Continue
            }
            KeyEvent {
                code: KeyCode::Left,
                ..
            } => {
                input.move_left();
                Action::Continue
            }
            KeyEvent {
                code: KeyCode::Right,
                ..
            } => {
                // At end of input: accept ghost text hint if available
                if input.cursor >= input.as_str().len() {
                    if let Some(ghost) = self.history_ghost(input.as_str()) {
                        input.replace(format!("{}{ghost}", input.as_str()));
                        self.history_index = None;
                        self.draft = None;
                    }
                } else {
                    input.move_right();
                }
                Action::Continue
            }
            KeyEvent {
                code: KeyCode::Up, ..
            } => {
                self.history_up(input);
                Action::Continue
            }
            KeyEvent {
                code: KeyCode::Down,
                ..
            } => {
                self.history_down(input);
                Action::Continue
            }
            KeyEvent {
                code: KeyCode::Home,
                ..
            } => {
                input.move_home();
                Action::Continue
            }
            KeyEvent {
                code: KeyCode::End, ..
            } => {
                input.move_end();
                Action::Continue
            }
            KeyEvent {
                code: KeyCode::Tab, ..
            } => {
                input.complete_slash_command(SLASH_CMDS);
                Action::Continue
            }
            KeyEvent {
                code: KeyCode::Esc, ..
            } => {
                if !input.is_empty() {
                    input.clear();
                    self.history_index = None;
                    self.draft = None;
                }
                Action::Continue
            }
            // Regular character input
            KeyEvent {
                code: KeyCode::Char(ch),
                modifiers,
                ..
            } if modifiers.is_empty() || modifiers == KeyModifiers::SHIFT => {
                input.insert(ch);
                self.history_index = None;
                self.draft = None;
                Action::Continue
            }
            _ => Action::Continue,
        }
    }

    fn history_up(&mut self, input: &mut InputBuffer) {
        if self.history.is_empty() {
            return;
        }
        match self.history_index {
            Some(0) => {}
            Some(i) => {
                let next = i - 1;
                input.replace(self.history[next].clone());
                self.history_index = Some(next);
            }
            None => {
                self.draft = Some(input.as_str().to_owned());
                let next = self.history.len() - 1;
                input.replace(self.history[next].clone());
                self.history_index = Some(next);
            }
        }
    }

    fn history_down(&mut self, input: &mut InputBuffer) {
        let Some(i) = self.history_index else {
            return;
        };
        if i + 1 < self.history.len() {
            let next = i + 1;
            input.replace(self.history[next].clone());
            self.history_index = Some(next);
        } else {
            input.replace(self.draft.take().unwrap_or_default());
            self.history_index = None;
        }
    }

    /// Find the most recent history entry that starts with `prefix` and
    /// return the suffix (the part after the prefix). Returns None if no match.
    fn history_ghost(&self, prefix: &str) -> Option<String> {
        // Don't ghost slash commands (tab completion handles those)
        if prefix.starts_with('/') {
            return None;
        }
        // Search from most recent backward
        for entry in self.history.iter().rev() {
            if entry.starts_with(prefix) && entry.len() > prefix.len() {
                return Some(entry[prefix.len()..].to_string());
            }
        }
        None
    }

    // ── Rendering ───────────────────────────────────────────────────────

    fn render(&self, input: &InputBuffer) -> Rendered {
        let text = input.as_str();

        // Cursor position within the text
        let before_cursor = &text[..input.cursor];
        let cursor_row = before_cursor.chars().filter(|&c| c == '\n').count();
        let cursor_line = before_cursor.rsplit('\n').next().unwrap_or_default();
        let cursor_prompt = if cursor_row == 0 {
            &self.prompt
        } else {
            &self.continuation_prompt
        };
        // Account for visible prompt width (strip ANSI)
        let prompt_width = visible_width(cursor_prompt);
        let cursor_col = prompt_width + cursor_line.chars().count();

        // Build display lines
        let mut lines = Vec::new();
        if text.is_empty() {
            // Show placeholder hint
            lines.push(format!(
                "{}\x1b[2m{}\x1b[0m",
                self.prompt, self.placeholder
            ));
        } else {
            for (i, line) in text.split('\n').enumerate() {
                let prefix = if i == 0 {
                    &self.prompt
                } else {
                    &self.continuation_prompt
                };
                lines.push(format!("{prefix}{line}"));
            }
            // Ghost text: show dim hint from matching history entry (single-line only).
            // Only when cursor is at end and input has no newlines.
            if !text.contains('\n') && input.cursor == text.len() && text.len() >= 2 {
                if let Some(hint) = self.history_ghost(text) {
                    if let Some(first) = lines.first_mut() {
                        first.push_str(&format!("\x1b[2m{hint}\x1b[0m"));
                    }
                }
            }
        }

        Rendered {
            lines,
            cursor_row: sat_u16(cursor_row),
            cursor_col: sat_u16(cursor_col),
            line_count: text.split('\n').count().max(1),
        }
    }

    fn redraw(
        &self,
        out: &mut impl Write,
        input: &InputBuffer,
        prev_lines: usize,
    ) -> io::Result<usize> {
        let rendered = self.render(input);

        // Move up to the first line of previous render
        if prev_lines > 1 {
            queue!(out, MoveUp(sat_u16(prev_lines - 1)))?;
        }
        queue!(out, MoveToColumn(0), Clear(ClearType::FromCursorDown))?;

        // Write lines
        for (i, line) in rendered.lines.iter().enumerate() {
            if i > 0 {
                write!(out, "\r\n")?;
            }
            write!(out, "{line}")?;
        }

        // Position cursor
        let lines_below = rendered.line_count.saturating_sub(1);
        if lines_below > 0 {
            // We're at the last line; move up to cursor row
            let up = lines_below.saturating_sub(rendered.cursor_row as usize);
            if up > 0 {
                queue!(out, MoveUp(sat_u16(up)))?;
            }
        }
        queue!(out, MoveToColumn(rendered.cursor_col))?;
        // Ensure cursor is visible
        write!(out, "\x1b[?25h")?;
        out.flush()?;

        Ok(rendered.line_count)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Action {
    Continue,
    Submit,
    Cancel,
    Exit,
}

struct Rendered {
    lines: Vec<String>,
    cursor_row: u16,
    cursor_col: u16,
    line_count: usize,
}

// ── Utilities ───────────────────────────────────────────────────────────────

fn longest_common_prefix(values: &[&str]) -> String {
    let Some(first) = values.first() else {
        return String::new();
    };
    let mut prefix = (*first).to_string();
    for value in values.iter().skip(1) {
        while !value.starts_with(&prefix) {
            prefix.pop();
            if prefix.is_empty() {
                break;
            }
        }
    }
    prefix
}

/// Count visible characters (strip ANSI escape sequences).
pub fn visible_width(s: &str) -> usize {
    let mut width = 0;
    let mut in_escape = false;
    for ch in s.chars() {
        if in_escape {
            if ch.is_ascii_alphabetic() {
                in_escape = false;
            }
        } else if ch == '\x1b' {
            in_escape = true;
        } else {
            width += 1;
        }
    }
    width
}

fn sat_u16(v: usize) -> u16 {
    u16::try_from(v).unwrap_or(u16::MAX)
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }

    fn ctrl(ch: char) -> KeyEvent {
        KeyEvent::new(KeyCode::Char(ch), KeyModifiers::CONTROL)
    }

    fn _alt(ch: char) -> KeyEvent {
        KeyEvent::new(KeyCode::Char(ch), KeyModifiers::ALT)
    }

    #[test]
    fn basic_insert_and_backspace() {
        let mut buf = InputBuffer::new();
        buf.insert('h');
        buf.insert('i');
        assert_eq!(buf.as_str(), "hi");
        buf.backspace();
        assert_eq!(buf.as_str(), "h");
    }

    #[test]
    fn newline_insertion() {
        let mut buf = InputBuffer::new();
        buf.insert('a');
        buf.insert_newline();
        buf.insert('b');
        assert_eq!(buf.as_str(), "a\nb");
    }

    #[test]
    fn delete_char() {
        let mut buf = InputBuffer::new();
        for ch in "abc".chars() {
            buf.insert(ch);
        }
        buf.move_home();
        buf.delete_char();
        assert_eq!(buf.as_str(), "bc");
    }

    #[test]
    fn kill_to_end() {
        let mut buf = InputBuffer::new();
        for ch in "hello world".chars() {
            buf.insert(ch);
        }
        buf.move_home();
        // Move to after "hello"
        for _ in 0..5 {
            buf.move_right();
        }
        buf.kill_to_end();
        assert_eq!(buf.as_str(), "hello");
    }

    #[test]
    fn kill_to_start() {
        let mut buf = InputBuffer::new();
        for ch in "hello world".chars() {
            buf.insert(ch);
        }
        // Cursor at end, move back to after space
        for _ in 0..5 {
            buf.move_left();
        }
        buf.kill_to_start();
        assert_eq!(buf.as_str(), "world");
    }

    #[test]
    fn kill_word_back() {
        let mut buf = InputBuffer::new();
        for ch in "hello world foo".chars() {
            buf.insert(ch);
        }
        buf.kill_word_back();
        assert_eq!(buf.as_str(), "hello world ");
    }

    #[test]
    fn word_movement() {
        let mut buf = InputBuffer::new();
        for ch in "hello world foo".chars() {
            buf.insert(ch);
        }
        buf.move_word_back();
        assert_eq!(buf.cursor(), 12); // before "foo"
        buf.move_word_back();
        assert_eq!(buf.cursor(), 6); // before "world"
        buf.move_word_forward();
        assert_eq!(buf.cursor(), 12); // before "foo"
    }

    #[test]
    fn slash_command_completion() {
        let mut buf = InputBuffer::new();
        for ch in "/he".chars() {
            buf.insert(ch);
        }
        assert!(buf.complete_slash_command(SLASH_CMDS));
        assert_eq!(buf.as_str(), "/help");
    }

    #[test]
    fn history_navigation() {
        let mut editor = LineEditor::new("> ");
        editor.push_history("first");
        editor.push_history("second");

        let mut input = InputBuffer::new();
        for ch in "draft".chars() {
            input.insert(ch);
        }

        let _ = editor.handle_key(key(KeyCode::Up), &mut input);
        assert_eq!(input.as_str(), "second");

        let _ = editor.handle_key(key(KeyCode::Up), &mut input);
        assert_eq!(input.as_str(), "first");

        let _ = editor.handle_key(key(KeyCode::Down), &mut input);
        assert_eq!(input.as_str(), "second");

        let _ = editor.handle_key(key(KeyCode::Down), &mut input);
        assert_eq!(input.as_str(), "draft");
    }

    #[test]
    fn ctrl_j_inserts_newline() {
        let mut editor = LineEditor::new("> ");
        let mut input = InputBuffer::new();
        input.insert('a');
        let action = editor.handle_key(ctrl('j'), &mut input);
        assert_eq!(action, Action::Continue);
        assert_eq!(input.as_str(), "a\n");
    }

    #[test]
    fn shift_enter_inserts_newline() {
        let mut editor = LineEditor::new("> ");
        let mut input = InputBuffer::new();
        input.insert('a');
        let action = editor.handle_key(
            KeyEvent::new(KeyCode::Enter, KeyModifiers::SHIFT),
            &mut input,
        );
        assert_eq!(action, Action::Continue);
        assert_eq!(input.as_str(), "a\n");
    }

    #[test]
    fn ctrl_a_e_home_end() {
        let mut editor = LineEditor::new("> ");
        let mut input = InputBuffer::new();
        for ch in "hello".chars() {
            input.insert(ch);
        }
        assert_eq!(input.cursor(), 5);

        let _ = editor.handle_key(ctrl('a'), &mut input);
        assert_eq!(input.cursor(), 0);

        let _ = editor.handle_key(ctrl('e'), &mut input);
        assert_eq!(input.cursor(), 5);
    }

    #[test]
    fn ctrl_k_u_w() {
        let mut editor = LineEditor::new("> ");
        let mut input = InputBuffer::new();
        for ch in "hello world".chars() {
            input.insert(ch);
        }

        // Ctrl+W kills last word
        let _ = editor.handle_key(ctrl('w'), &mut input);
        assert_eq!(input.as_str(), "hello ");

        // Ctrl+U kills to start
        let _ = editor.handle_key(ctrl('u'), &mut input);
        assert_eq!(input.as_str(), "");
    }

    #[test]
    fn ctrl_d_exits_on_empty_deletes_on_non_empty() {
        let mut editor = LineEditor::new("> ");

        let mut empty = InputBuffer::new();
        let action = editor.handle_key(ctrl('d'), &mut empty);
        assert_eq!(action, Action::Exit);

        let mut filled = InputBuffer::new();
        filled.insert('x');
        filled.move_home();
        let action = editor.handle_key(ctrl('d'), &mut filled);
        assert_eq!(action, Action::Continue);
        assert_eq!(filled.as_str(), "");
    }

    #[test]
    fn esc_clears_input() {
        let mut editor = LineEditor::new("> ");
        let mut input = InputBuffer::new();
        for ch in "some text".chars() {
            input.insert(ch);
        }
        let _ = editor.handle_key(key(KeyCode::Esc), &mut input);
        assert!(input.is_empty());
    }

    #[test]
    fn visible_width_strips_ansi() {
        assert_eq!(visible_width("\x1b[34m❯\x1b[0m "), 2); // "❯ "
        assert_eq!(visible_width("hello"), 5);
        assert_eq!(visible_width("\x1b[1m\x1b[36m##\x1b[0m"), 2);
    }

    #[test]
    fn render_shows_placeholder_when_empty() {
        let editor = LineEditor::new("> ");
        let input = InputBuffer::new();
        let rendered = editor.render(&input);
        assert_eq!(rendered.lines.len(), 1);
        assert!(rendered.lines[0].contains("Send a message"));
    }

    #[test]
    fn render_multiline_with_continuation() {
        let editor = LineEditor::new("> ");
        let mut input = InputBuffer::new();
        for ch in "hello".chars() {
            input.insert(ch);
        }
        input.insert_newline();
        for ch in "world".chars() {
            input.insert(ch);
        }
        let rendered = editor.render(&input);
        assert_eq!(rendered.lines.len(), 2);
        assert!(rendered.lines[0].starts_with("> "));
        assert!(rendered.lines[1].starts_with("  "));
    }

    #[test]
    fn transpose_chars() {
        let mut buf = InputBuffer::new();
        for ch in "ab".chars() {
            buf.insert(ch);
        }
        buf.transpose_chars();
        assert_eq!(buf.as_str(), "ba");
    }

    #[test]
    fn ctrl_p_n_history() {
        let mut editor = LineEditor::new("> ");
        editor.push_history("old");
        let mut input = InputBuffer::new();
        // Ctrl+P = history up
        let _ = editor.handle_key(ctrl('p'), &mut input);
        assert_eq!(input.as_str(), "old");
        // Ctrl+N = history down (back to draft)
        let _ = editor.handle_key(ctrl('n'), &mut input);
        assert_eq!(input.as_str(), "");
    }

    #[test]
    fn dedup_consecutive_history() {
        let mut editor = LineEditor::new("> ");
        editor.push_history("same");
        editor.push_history("same");
        editor.push_history("different");
        editor.push_history("different");
        assert_eq!(editor.history.len(), 2);
    }
}
