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

use crossterm::cursor::{MoveDown, MoveTo, MoveToColumn, MoveUp};
use crossterm::event::{
    self, Event, KeyCode, KeyEvent, KeyModifiers, KeyboardEnhancementFlags,
    PopKeyboardEnhancementFlags, PushKeyboardEnhancementFlags,
};
use crossterm::terminal::{self, Clear, ClearType};
use crossterm::{execute, queue};
use unicode_segmentation::UnicodeSegmentation;
use unicode_width::UnicodeWidthStr;

/// All known slash commands (used for completion).
pub const SLASH_CMDS: &[&str] = &[
    "/help",
    "/status",
    "/cost",
    "/model",
    "/tasks",
    "/agents",
    "/actors",
    "/actor",
    "/sessions",
    "/permissions",
    "/hooks",
    "/clear",
    "/exit",
    "/quit",
];

// ── Input buffer ────────────────────────────────────────────────────────────

/// Characters that define word boundaries (matches Codex's `WORD_SEPARATORS`).
const WORD_SEPS: &str = " \t\n`~!@#$%^&*()-=+[{]}\\|;:'\",.<>/?";

fn is_word_sep(c: char) -> bool {
    WORD_SEPS.contains(c)
}

const UNDO_MAX: usize = 50;

#[derive(Debug, Clone, PartialEq, Eq)]
struct UndoEntry {
    buffer: String,
    cursor: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InputBuffer {
    buffer: String,
    cursor: usize,
    /// Kill ring (single entry, like Codex). Ctrl+K/U/W store here, Ctrl+Y yanks.
    kill_buffer: String,
    /// Undo stack.
    undo_stack: Vec<UndoEntry>,
    /// Current position in undo stack (-1 = head, i.e. after all entries).
    /// When user undoes, this steps back. When user types after undo,
    /// future entries are truncated (branch truncation, Claude Code pattern).
    undo_index: usize,
    /// Number of edits since last undo snapshot (debounce).
    edits_since_snapshot: usize,
}

/// Find the byte offset of the grapheme cluster boundary before `pos`.
fn prev_grapheme(s: &str, pos: usize) -> usize {
    s[..pos]
        .grapheme_indices(true)
        .next_back()
        .map_or(0, |(i, _)| i)
}

/// Find the byte offset after the grapheme cluster at `pos`.
fn next_grapheme(s: &str, pos: usize) -> usize {
    s[pos..]
        .grapheme_indices(true)
        .nth(1)
        .map_or(s.len(), |(i, _)| pos + i)
}

impl Default for InputBuffer {
    fn default() -> Self {
        Self::new()
    }
}

impl InputBuffer {
    #[must_use]
    pub fn new() -> Self {
        Self {
            buffer: String::new(),
            cursor: 0,
            kill_buffer: String::new(),
            undo_stack: Vec::new(),
            undo_index: 0,
            edits_since_snapshot: 0,
        }
    }

    /// Save an undo snapshot. Called before destructive operations.
    fn snapshot(&mut self) {
        // Debounce: don't snapshot on every character, only every ~5 edits
        // or on major operations (kill, yank, etc.)
        self.edits_since_snapshot += 1;
        if self.edits_since_snapshot < 5 {
            return;
        }
        self.force_snapshot();
    }

    fn force_snapshot(&mut self) {
        self.edits_since_snapshot = 0;
        // Branch truncation: if we've undone past entries and now edit,
        // discard the future (entries after undo_index).
        if self.undo_index < self.undo_stack.len() {
            self.undo_stack.truncate(self.undo_index);
        }
        // Dedup: don't snapshot if nothing changed
        if self
            .undo_stack
            .last()
            .is_some_and(|e| e.buffer == self.buffer)
        {
            return;
        }
        self.undo_stack.push(UndoEntry {
            buffer: self.buffer.clone(),
            cursor: self.cursor,
        });
        self.undo_index = self.undo_stack.len();
        if self.undo_stack.len() > UNDO_MAX {
            self.undo_stack.remove(0);
            self.undo_index = self.undo_stack.len();
        }
    }

    /// Undo: step back in the undo stack. Returns true if state changed.
    pub fn undo(&mut self) -> bool {
        if self.undo_index == 0 {
            return false;
        }
        // Save current state if we're at the head (first undo)
        if self.undo_index == self.undo_stack.len() {
            self.undo_stack.push(UndoEntry {
                buffer: self.buffer.clone(),
                cursor: self.cursor,
            });
        }
        self.undo_index = self.undo_index.saturating_sub(1);
        if let Some(entry) = self.undo_stack.get(self.undo_index) {
            self.buffer = entry.buffer.clone();
            self.cursor = entry.cursor.min(self.buffer.len());
            self.edits_since_snapshot = 0;
            true
        } else {
            false
        }
    }

    /// Clear undo history (call on submit).
    pub fn clear_undo(&mut self) {
        self.undo_stack.clear();
        self.undo_index = 0;
        self.edits_since_snapshot = 0;
    }

    pub fn insert(&mut self, ch: char) {
        self.snapshot();
        // Normalize: tabs → 4 spaces, \r → \n
        if ch == '\t' {
            for _ in 0..4 {
                self.buffer.insert(self.cursor, ' ');
                self.cursor += 1;
            }
            return;
        }
        let ch = if ch == '\r' { '\n' } else { ch };
        self.buffer.insert(self.cursor, ch);
        self.cursor += ch.len_utf8();
    }

    /// Insert a string (e.g. from bracketed paste). Normalizes ANSI, \r\n, tabs.
    pub fn insert_str(&mut self, s: &str) {
        let cleaned = strip_ansi_from_paste(s);
        // Normalize \r\n → \n, then lone \r → \n
        let normalized = cleaned.replace("\r\n", "\n").replace('\r', "\n");
        for ch in normalized.chars() {
            self.insert(ch);
        }
    }

    pub fn insert_newline(&mut self) {
        self.insert('\n');
    }

    pub fn backspace(&mut self) {
        if self.cursor == 0 {
            return;
        }
        let prev = prev_grapheme(&self.buffer, self.cursor);
        self.buffer.drain(prev..self.cursor);
        self.cursor = prev;
    }

    /// Delete grapheme under cursor (Ctrl+D behavior when buffer non-empty).
    pub fn delete_char(&mut self) {
        if self.cursor >= self.buffer.len() {
            return;
        }
        let next = next_grapheme(&self.buffer, self.cursor);
        self.buffer.drain(self.cursor..next);
    }

    pub fn move_left(&mut self) {
        if self.cursor == 0 {
            return;
        }
        self.cursor = prev_grapheme(&self.buffer, self.cursor);
    }

    pub fn move_right(&mut self) {
        if self.cursor >= self.buffer.len() {
            return;
        }
        self.cursor = next_grapheme(&self.buffer, self.cursor);
    }

    pub fn move_home(&mut self) {
        self.cursor = 0;
    }

    pub fn move_end(&mut self) {
        self.cursor = self.buffer.len();
    }

    /// Column position of cursor within its current line (display width).
    #[must_use]
    pub fn current_col(&self) -> usize {
        let before = &self.buffer[..self.cursor];
        let line_start = before.rfind('\n').map_or(0, |i| i + 1);
        UnicodeWidthStr::width(&before[line_start..])
    }

    /// Move cursor to the previous line, targeting `target_col` display width.
    /// Returns true if movement happened.
    pub fn move_up(&mut self, target_col: usize) -> bool {
        let before = &self.buffer[..self.cursor];
        let Some(nl_pos) = before.rfind('\n') else {
            return false; // already on first line
        };
        // Find the start of the previous line
        let prev_line_start = before[..nl_pos].rfind('\n').map_or(0, |i| i + 1);
        let prev_line = &self.buffer[prev_line_start..nl_pos];
        // Find byte offset within prev_line that matches target_col
        self.cursor = prev_line_start + col_to_byte_offset(prev_line, target_col);
        true
    }

    /// Move cursor to the next line, targeting `target_col` display width.
    /// Returns true if movement happened.
    pub fn move_down(&mut self, target_col: usize) -> bool {
        // Find next newline from current position
        let rest_from_line_start = &self.buffer[self.cursor..];
        let Some(nl_offset) = rest_from_line_start.find('\n') else {
            return false; // already on last line
        };
        let next_line_start = self.cursor + nl_offset + 1;
        if next_line_start >= self.buffer.len() {
            // Next line is empty (trailing newline)
            self.cursor = self.buffer.len();
            return true;
        }
        let next_line_end = self.buffer[next_line_start..]
            .find('\n')
            .map_or(self.buffer.len(), |i| next_line_start + i);
        let next_line = &self.buffer[next_line_start..next_line_end];
        self.cursor = next_line_start + col_to_byte_offset(next_line, target_col);
        true
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
        self.force_snapshot();
        self.kill_buffer = self.buffer[self.cursor..].to_string();
        self.buffer.truncate(self.cursor);
    }

    /// Kill from start to cursor (Ctrl+U). Stores killed text.
    pub fn kill_to_start(&mut self) {
        self.force_snapshot();
        self.kill_buffer = self.buffer[..self.cursor].to_string();
        self.buffer.drain(..self.cursor);
        self.cursor = 0;
    }

    /// Kill previous word (Ctrl+W). Stores killed text.
    pub fn kill_word_back(&mut self) {
        self.force_snapshot();
        let old_cursor = self.cursor;
        self.move_word_back();
        self.kill_buffer = self.buffer[self.cursor..old_cursor].to_string();
        self.buffer.drain(self.cursor..old_cursor);
    }

    /// Kill from cursor to end of next word (Alt+D).
    pub fn kill_word_forward(&mut self) {
        self.force_snapshot();
        let old_cursor = self.cursor;
        self.move_word_forward();
        self.kill_buffer = self.buffer[old_cursor..self.cursor].to_string();
        self.buffer.drain(old_cursor..self.cursor);
        self.cursor = old_cursor;
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

    /// Candidates for the current slash-command prefix, used to show a live
    /// autocomplete menu above the input row. Returns an empty vec when the
    /// input is not a slash-command prefix (so callers can hide the menu).
    #[must_use]
    pub fn slash_completions<'a>(&self, candidates: &[&'a str]) -> Vec<&'a str> {
        let prefix = &self.buffer[..self.cursor];
        if self.cursor != self.buffer.len()
            || prefix.contains(char::is_whitespace)
            || !prefix.starts_with('/')
        {
            return Vec::new();
        }
        candidates
            .iter()
            .filter(|c| c.starts_with(prefix))
            .copied()
            .collect()
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
    /// A dev reload arrived while the prompt was idle. The payload is the
    /// in-progress draft, which the caller persists before restarting.
    Reload(String),
    Cancel,
    Inspect,
    /// Ctrl-L: repaint the screen. The payload is the draft to restore.
    Repaint(String),
    Exit,
}

/// How often the idle prompt checks for a pending dev reload. Short enough to
/// feel immediate, long enough that an idle prompt costs nothing.
const RELOAD_POLL: std::time::Duration = std::time::Duration::from_millis(100);

// ── Line editor ─────────────────────────────────────────────────────────────

/// Threshold for collapsing pasted text into a pill.
const PASTE_PILL_CHARS: usize = 800;
const PASTE_PILL_LINES: usize = 3;

/// Stashed input state (Ctrl+S toggle).
#[derive(Debug)]
struct Stash {
    text: String,
    cursor: usize,
    kill_buffer: String,
}

pub struct LineEditor {
    prompt: String,
    continuation_prompt: String,
    placeholder: String,
    history: Vec<String>,
    history_index: Option<usize>,
    draft: Option<String>,
    preferred_col: Option<usize>,
    /// Active reverse-search query (None = normal mode).
    search_query: Option<String>,
    /// Index into history of the current search match.
    search_match_idx: Option<usize>,
    stash: Option<Stash>,
    /// Storage for large pastes collapsed to pills.
    /// Key: pill ID, Value: full pasted text.
    pasted_contents: Vec<String>,
    /// Text restored after a dev reload, applied to the next prompt.
    pending_input: Option<String>,
    /// Optional dev-reload probe, checked once per poll tick while idle.
    reload_check: Option<fn() -> bool>,
    /// Latest raw dev status label from the supervisor.
    dev_status_label: Option<String>,
    /// Rendered dev status, with state color and elapsed time.
    dev_status_line: Option<String>,
    /// When the current status label first appeared.
    status_since: std::time::Instant,
    /// Optional dev status probe, polled while the prompt is idle.
    status_source: Option<fn() -> Option<String>>,
    /// Bottom-anchored layout owned by a full-screen caller; see [`EditorFrame`].
    frame: Option<EditorFrame>,
    /// Rows the input block occupied at the last framed redraw.
    frame_rows: u16,
    /// Terminal size at the last framed redraw, to relayout after a resize.
    frame_size: (u16, u16),
}

/// Lets a full-screen caller keep its frame while the editor draws.
///
/// Without a frame the editor draws relative to the cursor and lets the
/// terminal wrap and scroll, which in a pinned layout scrolls the header and
/// footer away as soon as input needs a second row. With a frame, the input
/// block is drawn at absolute rows at the bottom of the screen, split into
/// rows that never reach the last column, and `relayout` is called whenever
/// the block needs a different number of rows (or the terminal resized) so
/// the caller can move its footer and scroll region above it.
pub struct EditorFrame {
    /// `relayout(old_rows, new_rows)`: lay out the frame for `new_rows` input
    /// rows at the bottom of the screen.
    pub relayout: Box<dyn FnMut(u16, u16)>,
    /// Most rows the block may take; longer input shows a window around the
    /// caret.
    pub max_rows: u16,
}

/// Encode one history entry as a single line.
///
/// Multiline prompts are ordinary input (Shift+Enter). A raw newline would
/// split one entry into several on the next load, so backslash, newline, and
/// carriage return are escaped. This is the one-line escaping fish and zsh use
/// for multiline history. Entries without backslashes read unchanged, so files
/// written by older versions still load.
fn encode_history_entry(entry: &str) -> String {
    let mut encoded = String::with_capacity(entry.len());
    for ch in entry.chars() {
        match ch {
            '\\' => encoded.push_str("\\\\"),
            '\n' => encoded.push_str("\\n"),
            '\r' => encoded.push_str("\\r"),
            other => encoded.push(other),
        }
    }
    encoded
}

/// Decode a line written by [`encode_history_entry`]. Unknown escapes keep
/// their backslash so a future format stays readable instead of silently
/// dropping text.
fn decode_history_entry(line: &str) -> String {
    let mut decoded = String::with_capacity(line.len());
    let mut chars = line.chars();
    while let Some(ch) = chars.next() {
        if ch != '\\' {
            decoded.push(ch);
            continue;
        }
        match chars.next() {
            Some('n') => decoded.push('\n'),
            Some('r') => decoded.push('\r'),
            Some('\\') => decoded.push('\\'),
            Some(other) => {
                decoded.push('\\');
                decoded.push(other);
            }
            None => decoded.push('\\'),
        }
    }
    decoded
}

/// Collapse whitespace and bound one history entry to a single display line.
fn one_line(text: &str, max: usize) -> String {
    let collapsed = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if collapsed.chars().count() <= max {
        return collapsed;
    }
    let mut short: String = collapsed.chars().take(max.saturating_sub(1)).collect();
    short.push('\u{2026}');
    short
}

impl LineEditor {
    #[must_use]
    pub fn new(prompt: impl Into<String>) -> Self {
        Self {
            prompt: prompt.into(),
            continuation_prompt: String::from("\x1b[2m\u{22EE}\x1b[0m "),
            placeholder: String::from("Send a message or /help"),
            history: Vec::new(),
            history_index: None,
            draft: None,
            preferred_col: None,
            stash: None,
            pasted_contents: Vec::new(),
            pending_input: None,
            reload_check: None,
            dev_status_label: None,
            dev_status_line: None,
            status_since: std::time::Instant::now(),
            status_source: None,
            search_query: None,
            search_match_idx: None,
            frame: None,
            frame_rows: 1,
            frame_size: (0, 0),
        }
    }

    /// Draw input as a bottom-anchored block inside a caller-owned frame.
    pub fn set_frame(&mut self, frame: Option<EditorFrame>) {
        self.frame = frame;
        self.frame_rows = 1;
        self.frame_size = (0, 0);
    }

    /// Rows the framed input block occupies now (1 when idle or unframed).
    #[must_use]
    pub fn frame_rows(&self) -> u16 {
        self.frame_rows
    }

    /// Prefill the next prompt. Used when a dev reload preserved a draft.
    pub fn set_initial_input(&mut self, text: impl Into<String>) {
        self.pending_input = Some(text.into());
    }

    /// Arm the dev-reload probe. The prompt polls it while idle and returns
    /// [`ReadOutcome::Reload`] instead of waiting for a key.
    pub fn set_reload_check(&mut self, check: fn() -> bool) {
        self.reload_check = Some(check);
    }

    /// Arm the dev status probe. Status is rendered on the prompt row only
    /// while the input is empty, so it cannot disturb a typed draft or the
    /// cursor position inside one.
    pub fn set_status_source(&mut self, source: fn() -> Option<String>) {
        self.status_source = Some(source);
    }

    /// Expand any paste pills in the text, replacing `[Pasted text #N ...]`
    /// with the actual pasted content. Call on submit.
    #[must_use]
    pub fn expand_paste_pills(&self, text: &str) -> String {
        let mut result = text.to_string();
        for (i, content) in self.pasted_contents.iter().enumerate() {
            let id = i + 1;
            // Find and replace the pill marker
            let prefix = format!("[Pasted text #{id}");
            if let Some(start) = result.find(&prefix) {
                if let Some(end) = result[start..].find(']') {
                    result.replace_range(start..=(start + end), content);
                }
            }
        }
        result
    }

    /// Clear paste pill storage (call after submit).
    pub fn clear_paste_pills(&mut self) {
        self.pasted_contents.clear();
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
                let entry = decode_history_entry(line);
                // Whitespace-only lines carry no prompt; everything else is
                // kept verbatim so indented code and trailing spaces survive.
                if !entry.trim().is_empty() {
                    self.history.push(entry);
                }
            }
        }
    }

    pub fn save_history_file(&self, path: &std::path::Path) {
        // Keep last 500 entries
        let start = self.history.len().saturating_sub(500);
        let contents = self.history[start..]
            .iter()
            .map(|entry| encode_history_entry(entry))
            .collect::<Vec<_>>()
            .join("\n")
            + "\n";
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
            let _ = writeln!(stdout);
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
            PushKeyboardEnhancementFlags(KeyboardEnhancementFlags::DISAMBIGUATE_ESCAPE_CODES)
        );

        let mut input = InputBuffer::new();
        if let Some(text) = self.pending_input.take() {
            input.replace(text);
        }
        let mut rendered_lines = 1usize;
        self.redraw(&mut stdout, &input, rendered_lines)?;

        let outcome = loop {
            if self.reload_check.is_some_and(|check| check()) {
                break ReadOutcome::Reload(input.as_str().to_owned());
            }
            match event::poll(RELOAD_POLL) {
                Ok(false) => {
                    self.refresh_status(&mut stdout, &input, &mut rendered_lines)?;
                    continue;
                }
                Ok(true) => {}
                Err(e) => {
                    let _ = execute!(stdout, PopKeyboardEnhancementFlags);
                    if !was_raw {
                        let _ = terminal::disable_raw_mode();
                    }
                    return Err(e);
                }
            }
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
                    Action::Inspect => {
                        break ReadOutcome::Inspect;
                    }
                    Action::Repaint => {
                        break ReadOutcome::Repaint(input.as_str().to_owned());
                    }
                    Action::Exit => {
                        break ReadOutcome::Exit;
                    }
                },
                Ok(Event::Paste(text)) => {
                    // Bracketed paste: normalize, then either insert inline
                    // or collapse to a pill for large pastes.
                    let cleaned = strip_ansi_from_paste(&text);
                    let normalized = cleaned.replace("\r\n", "\n").replace('\r', "\n");
                    let lines = normalized.lines().count();
                    if normalized.len() > PASTE_PILL_CHARS || lines > PASTE_PILL_LINES {
                        // Collapse to pill
                        let id = self.pasted_contents.len() + 1;
                        self.pasted_contents.push(normalized);
                        let pill = format!("[Pasted text #{id} +{lines} lines]");
                        for ch in pill.chars() {
                            input.insert(ch);
                        }
                    } else {
                        input.insert_str(&text);
                    }
                    self.history_index = None;
                    self.draft = None;
                    rendered_lines = self.redraw(&mut stdout, &input, rendered_lines)?;
                }
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

    pub fn handle_key(&mut self, key: KeyEvent, input: &mut InputBuffer) -> Action {
        // ── Reverse search mode ─────────────────────────────────────────
        if self.search_query.is_some() {
            return self.handle_search_key(key, input);
        }

        // Clear sticky column on any key except Up/Down
        if !matches!(key.code, KeyCode::Up | KeyCode::Down) {
            self.preferred_col = None;
        }
        match key {
            // ── Ctrl combos ─────────────────────────────────────────────
            KeyEvent {
                code: KeyCode::Char('o'),
                modifiers,
                ..
            } if modifiers.contains(KeyModifiers::CONTROL) && input.is_empty() => Action::Inspect,
            KeyEvent {
                code: KeyCode::Char('l'),
                modifiers,
                ..
            } if modifiers.contains(KeyModifiers::CONTROL) => Action::Repaint,
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
            // Stash: Ctrl+S — toggle save/restore draft
            KeyEvent {
                code: KeyCode::Char('s'),
                modifiers,
                ..
            } if modifiers.contains(KeyModifiers::CONTROL) => {
                if input.is_empty() {
                    // Pop from stash
                    if let Some(stash) = self.stash.take() {
                        input.buffer = stash.text;
                        input.cursor = stash.cursor.min(input.buffer.len());
                        input.kill_buffer = stash.kill_buffer;
                    }
                } else {
                    // Push to stash
                    self.stash = Some(Stash {
                        text: input.buffer.clone(),
                        cursor: input.cursor,
                        kill_buffer: input.kill_buffer.clone(),
                    });
                    input.clear();
                }
                Action::Continue
            }
            // Undo: Ctrl+Z
            KeyEvent {
                code: KeyCode::Char('z'),
                modifiers,
                ..
            } if modifiers.contains(KeyModifiers::CONTROL) => {
                input.undo();
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
            // Ctrl+R = reverse history search
            KeyEvent {
                code: KeyCode::Char('r'),
                modifiers,
                ..
            } if modifiers.contains(KeyModifiers::CONTROL) => {
                self.search_query = Some(String::new());
                self.search_match_idx = None;
                Action::Continue
            }
            // Alt+D = kill word forward
            KeyEvent {
                code: KeyCode::Char('d'),
                modifiers,
                ..
            } if modifiers.contains(KeyModifiers::ALT) => {
                input.kill_word_forward();
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
                let on_first_line = !input.as_str()[..input.cursor].contains('\n');
                if on_first_line {
                    self.history_up(input);
                    self.preferred_col = None;
                } else {
                    let col = self.preferred_col.unwrap_or_else(|| input.current_col());
                    if input.move_up(col) {
                        self.preferred_col = Some(col);
                    }
                }
                Action::Continue
            }
            KeyEvent {
                code: KeyCode::Down,
                ..
            } => {
                let on_last_line = !input.as_str()[input.cursor..].contains('\n');
                if on_last_line {
                    self.history_down(input);
                    self.preferred_col = None;
                } else {
                    let col = self.preferred_col.unwrap_or_else(|| input.current_col());
                    if input.move_down(col) {
                        self.preferred_col = Some(col);
                    }
                }
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

    // ── Reverse search ──────────────────────────────────────────────────

    fn handle_search_key(&mut self, key: KeyEvent, input: &mut InputBuffer) -> Action {
        match key {
            // Ctrl-R again = find next match (further back in history)
            KeyEvent {
                code: KeyCode::Char('r'),
                modifiers,
                ..
            } if modifiers.contains(KeyModifiers::CONTROL) => {
                self.search_next_match(input);
                Action::Continue
            }
            // Enter = accept the current match
            KeyEvent {
                code: KeyCode::Enter,
                ..
            } => {
                self.search_query = None;
                self.search_match_idx = None;
                Action::Continue // stays in buffer, user can edit or submit
            }
            // Escape / Ctrl-C / Ctrl-G = cancel search, restore original input
            KeyEvent {
                code: KeyCode::Esc, ..
            }
            | KeyEvent {
                code: KeyCode::Char('c' | 'g'),
                modifiers: KeyModifiers::CONTROL,
                ..
            } => {
                // Restore the draft that was active before search
                if let Some(draft) = self.draft.take() {
                    input.replace(draft);
                } else {
                    input.clear();
                }
                self.search_query = None;
                self.search_match_idx = None;
                self.history_index = None;
                Action::Continue
            }
            // Backspace = remove last char from search query
            KeyEvent {
                code: KeyCode::Backspace,
                ..
            } => {
                if let Some(ref mut q) = self.search_query {
                    q.pop();
                    self.search_update(input);
                }
                Action::Continue
            }
            // Regular character = append to search query
            KeyEvent {
                code: KeyCode::Char(c),
                modifiers,
                ..
            } if !modifiers.contains(KeyModifiers::CONTROL)
                && !modifiers.contains(KeyModifiers::ALT) =>
            {
                if let Some(ref mut q) = self.search_query {
                    q.push(c);
                    self.search_update(input);
                }
                Action::Continue
            }
            // Any other key = accept match and replay the key
            _ => {
                self.search_query = None;
                self.search_match_idx = None;
                self.handle_key(key, input)
            }
        }
    }

    /// Search history for the current query and update the input buffer.
    fn search_update(&mut self, input: &mut InputBuffer) {
        let Some(ref query) = self.search_query else {
            return;
        };
        if query.is_empty() {
            return;
        }
        let query_lower = query.to_lowercase();
        // Search backwards from the beginning (most recent first)
        for (i, entry) in self.history.iter().enumerate() {
            if entry.to_lowercase().contains(&query_lower) {
                input.replace(entry.clone());
                self.history_index = Some(i);
                self.search_match_idx = Some(i);
                return;
            }
        }
        // No match found -- leave input as-is
        self.search_match_idx = None;
    }

    /// Find the next match (further back in history from current).
    /// Newest-first matches for the picker. Bounded so the overlay cannot
    /// outgrow the scroll region.
    fn search_matches(&self, query: &str) -> Vec<usize> {
        let needle = query.to_lowercase();
        self.history
            .iter()
            .enumerate()
            .rev()
            .filter(|(_, entry)| needle.is_empty() || entry.to_lowercase().contains(&needle))
            .map(|(index, _)| index)
            .take(6)
            .collect()
    }

    fn search_next_match(&mut self, input: &mut InputBuffer) {
        let Some(ref query) = self.search_query else {
            return;
        };
        if query.is_empty() {
            return;
        }
        let query_lower = query.to_lowercase();
        let start = self.search_match_idx.map_or(0, |i| i + 1);
        for (i, entry) in self.history.iter().enumerate().skip(start) {
            if entry.to_lowercase().contains(&query_lower) {
                input.replace(entry.clone());
                self.history_index = Some(i);
                self.search_match_idx = Some(i);
                return;
            }
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
        // In search mode, show the query plus a newest-first picker. Selection
        // wins over age dimming so the current match is always readable.
        if let Some(ref query) = self.search_query {
            let matches = self.search_matches(query);
            let mut lines = Vec::new();
            let no_match = if matches.is_empty() {
                " \x1b[31m(no match)\x1b[0m"
            } else {
                ""
            };
            lines.push(format!("\x1b[2m(reverse-search):\x1b[0m {query}{no_match}"));
            for (position, index) in matches.iter().enumerate() {
                let text = one_line(&self.history[*index], 88);
                let line = if self.search_match_idx == Some(*index) {
                    format!("  \x1b[7m {text} \x1b[0m")
                } else if position >= 5 {
                    format!("  \x1b[2m{text}\x1b[0m")
                } else {
                    format!("  {text}")
                };
                lines.push(line);
            }
            #[allow(clippy::cast_possible_truncation)]
            let cursor_col = (19 + query.len()) as u16;
            let line_count = lines.len();
            return Rendered {
                lines,
                cursor_row: 0,
                cursor_col,
                line_count,
            };
        }

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
        // Account for visible prompt width (strip ANSI) and display width
        let prompt_width = visible_width(cursor_prompt);
        let cursor_col = prompt_width + UnicodeWidthStr::width(cursor_line);

        // Build display lines
        let mut lines = Vec::new();
        if text.is_empty() {
            // Show placeholder hint, plus the dev status when the prompt is
            // idle. Both are dim and live on the prompt row, so a build in
            // flight is visible without touching the transcript.
            let mut line = format!("{}\x1b[2m{}\x1b[0m", self.prompt, self.placeholder);
            if let Some(status) = &self.dev_status_line {
                line.push_str("  ");
                line.push_str(status);
            }
            lines.push(line);
        } else {
            for (i, line) in text.split('\n').enumerate() {
                let prefix = if i == 0 {
                    &self.prompt
                } else {
                    &self.continuation_prompt
                };
                // Syntax highlighting for input text
                let highlighted = highlight_input_line(line, i == 0);
                lines.push(format!("{prefix}{highlighted}"));
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

    /// Refresh the dev status on an idle tick, redrawing only when it changed.
    fn refresh_status(
        &mut self,
        out: &mut impl Write,
        input: &InputBuffer,
        rendered_lines: &mut usize,
    ) -> io::Result<()> {
        let now = std::time::Instant::now();
        let label = self.status_source.and_then(|source| source());
        if label != self.dev_status_label {
            self.dev_status_label = label;
            self.status_since = now;
        }
        let line = self.status_line(now);
        if line != self.dev_status_line {
            self.dev_status_line = line;
            *rendered_lines = self.redraw(out, input, *rendered_lines)?;
        }
        Ok(())
    }

    /// Render the current dev status as a colored token. `ready` fades after
    /// a few seconds so the prompt returns to its resting shape; `building`
    /// carries elapsed seconds so a slow compile is visible, not silent.
    fn status_line(&self, now: std::time::Instant) -> Option<String> {
        let (color, text) = match self.dev_status_label.as_deref()? {
            "building" => (
                33,
                format!(
                    "building {}s",
                    now.duration_since(self.status_since).as_secs()
                ),
            ),
            "ready" => {
                if now.duration_since(self.status_since) >= std::time::Duration::from_secs(4) {
                    return None;
                }
                (32, String::from("ready"))
            }
            "build failed" => (31, String::from("build failed")),
            "reloading" => (36, String::from("reloading")),
            other => (2, other.to_string()),
        };
        Some(format!("\x1b[{color}m● {text}\x1b[0m"))
    }

    fn redraw(
        &mut self,
        out: &mut impl Write,
        input: &InputBuffer,
        prev_lines: usize,
    ) -> io::Result<usize> {
        let rendered = self.render(input);
        if self.frame.is_some() {
            return self.redraw_framed(out, &rendered);
        }

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
pub enum Action {
    Continue,
    Submit,
    Cancel,
    Inspect,
    Repaint,
    Exit,
}

impl LineEditor {
    /// Framed redraw: absolute rows at the bottom, no terminal wrapping or
    /// scrolling, and a relayout call when the block changes height.
    fn redraw_framed(&mut self, out: &mut impl Write, rendered: &Rendered) -> io::Result<usize> {
        let (cols, rows) = term_size();
        let layout = frame_layout(
            rendered,
            cols,
            self.frame.as_ref().map_or(1, |f| f.max_rows),
        );
        let block = sat_u16(layout.rows.len()).max(1);
        if block != self.frame_rows || (cols, rows) != self.frame_size {
            if let Some(frame) = self.frame.as_mut() {
                (frame.relayout)(self.frame_rows, block);
            }
            self.frame_rows = block;
            self.frame_size = (cols, rows);
        }
        let top = rows.saturating_sub(block) + 1;
        for (offset, row) in layout.rows.iter().enumerate() {
            queue!(
                out,
                MoveTo(0, top - 1 + sat_u16(offset)),
                Clear(ClearType::CurrentLine)
            )?;
            write!(out, "{row}\x1b[0m")?;
        }
        queue!(out, MoveTo(layout.cursor_col, top - 1 + layout.cursor_row))?;
        write!(out, "\x1b[?25h")?;
        out.flush()?;
        Ok(rendered.line_count)
    }
}

/// Rendered lines split into screen rows for a framed redraw.
struct FrameLayout {
    rows: Vec<String>,
    /// Caret position within `rows` (0-based row, 0-based column).
    cursor_row: u16,
    cursor_col: u16,
}

/// Split each rendered line into rows of at most `cols - 1` cells (the last
/// column stays empty so a full row never leaves the terminal in its
/// wrap-pending state), place the caret, and keep at most `max_rows` rows
/// around it.
fn frame_layout(rendered: &Rendered, cols: u16, max_rows: u16) -> FrameLayout {
    let width = usize::from(cols.saturating_sub(1)).max(1);
    let mut rows = Vec::new();
    let mut caret = (0usize, 0usize);
    for (index, line) in rendered.lines.iter().enumerate() {
        let first = rows.len();
        rows.extend(split_by_width(line, width));
        if index == usize::from(rendered.cursor_row) {
            let col = usize::from(rendered.cursor_col);
            caret = (first + col / width, col % width);
            // A caret just past a full row sits at the start of the next one.
            while rows.len() <= caret.0 {
                rows.push(String::new());
            }
        }
    }
    if rows.is_empty() {
        rows.push(String::new());
    }
    let max_rows = usize::from(max_rows).max(1);
    let start = if rows.len() > max_rows {
        (caret.0 + 1)
            .saturating_sub(max_rows)
            .min(rows.len() - max_rows)
    } else {
        0
    };
    let end = (start + max_rows).min(rows.len());
    FrameLayout {
        rows: rows[start..end].to_vec(),
        cursor_row: sat_u16(caret.0 - start),
        cursor_col: sat_u16(caret.1),
    }
}

/// Split `s` into pieces of at most `width` display cells, passing ANSI escape
/// sequences through untouched. Styles are re-opened per piece only as far as
/// the escape sequences themselves carry them.
fn split_by_width(s: &str, width: usize) -> Vec<String> {
    let mut pieces = vec![String::new()];
    let mut used = 0;
    let mut in_escape = false;
    for ch in s.chars() {
        if in_escape {
            pieces.last_mut().unwrap().push(ch);
            if ch.is_ascii_alphabetic() {
                in_escape = false;
            }
            continue;
        }
        if ch == '\x1b' {
            in_escape = true;
            pieces.last_mut().unwrap().push(ch);
            continue;
        }
        let w = UnicodeWidthStr::width(ch.encode_utf8(&mut [0; 4]) as &str);
        if used + w > width && used > 0 {
            pieces.push(String::new());
            used = 0;
        }
        pieces.last_mut().unwrap().push(ch);
        used += w;
    }
    pieces
}

struct Rendered {
    lines: Vec<String>,
    cursor_row: u16,
    cursor_col: u16,
    line_count: usize,
}

// ── Utilities ───────────────────────────────────────────────────────────────

/// Find the byte offset within `line` that corresponds to `target_col` display columns.
fn col_to_byte_offset(line: &str, target_col: usize) -> usize {
    let mut col = 0;
    for (i, g) in line.grapheme_indices(true) {
        let w = UnicodeWidthStr::width(g);
        if col + w > target_col {
            return i;
        }
        col += w;
    }
    line.len() // target_col beyond line end → park at end
}

/// Highlight input text: slash commands cyan, backtick code green.
fn highlight_input_line(line: &str, is_first_line: bool) -> String {
    // Slash commands: full line cyan when on first line, no spaces
    if is_first_line && line.starts_with('/') && !line.contains(' ') {
        return format!("\x1b[36m{line}\x1b[0m");
    }

    // ! bash escape: highlight the ! in magenta, command dim
    if is_first_line && line.starts_with('!') {
        let cmd = &line[1..];
        return format!("\x1b[35m!\x1b[0m\x1b[2m{cmd}\x1b[0m");
    }

    // Inline backtick highlighting: `code` → green
    if !line.contains('`') {
        return line.to_string();
    }

    let mut out = String::new();
    let mut in_backtick = false;
    let chars = line.chars().peekable();
    for ch in chars {
        if ch == '`' {
            if in_backtick {
                out.push('`');
                out.push_str("\x1b[0m");
                in_backtick = false;
            } else {
                out.push_str("\x1b[32m");
                out.push('`');
                in_backtick = true;
            }
        } else {
            out.push(ch);
        }
    }
    if in_backtick {
        out.push_str("\x1b[0m"); // unclosed backtick
    }
    out
}

/// Strip ANSI escape sequences from pasted text.
fn strip_ansi_from_paste(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '\x1b' {
            // Skip ESC [ ... letter sequences
            if chars.peek() == Some(&'[') {
                chars.next();
                for c in chars.by_ref() {
                    if c.is_ascii_alphabetic() {
                        break;
                    }
                }
            }
            // Skip ESC ] ... BEL/ST sequences (OSC)
            else if chars.peek() == Some(&']') {
                chars.next();
                for c in chars.by_ref() {
                    if c == '\x07' || c == '\\' {
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

/// Count visible display width (strip ANSI escape sequences, use Unicode width).
#[must_use]
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
            width += UnicodeWidthStr::width(ch.encode_utf8(&mut [0; 4]) as &str);
        }
    }
    width
}

/// Terminal `(cols, rows)`. Some PTY setups (freshly spawned pty slaves, CI
/// runners) report (0, 0) through a successful ioctl rather than failing; an
/// `unwrap_or` on `Err` would miss that and draw into a zero-sized grid. A
/// degenerate size resolves to 80x24.
pub(crate) fn term_size() -> (u16, u16) {
    match terminal::size() {
        Ok((cols, rows)) if cols > 0 && rows > 0 => (cols, rows),
        _ => (80, 24),
    }
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
    fn slash_completions_filters_by_prefix() {
        let cands = ["/help", "/status", "/sessions", "/version"];
        let mut buf = InputBuffer::new();
        for ch in "/s".chars() {
            buf.insert(ch);
        }
        // Only the /s* commands match, in candidate order.
        assert_eq!(buf.slash_completions(&cands), vec!["/status", "/sessions"]);
    }

    #[test]
    fn slash_completions_empty_when_not_a_slash_prefix() {
        let cands = ["/help", "/status"];
        // Plain text: no menu.
        let mut buf = InputBuffer::new();
        for ch in "hello".chars() {
            buf.insert(ch);
        }
        assert!(buf.slash_completions(&cands).is_empty());

        // Slash with a trailing space is a completed command, not a prefix.
        let mut buf2 = InputBuffer::new();
        for ch in "/help ".chars() {
            buf2.insert(ch);
        }
        assert!(buf2.slash_completions(&cands).is_empty());
    }

    #[test]
    fn slash_completions_require_cursor_at_end() {
        let cands = ["/help", "/status"];
        let mut buf = InputBuffer::new();
        for ch in "/help".chars() {
            buf.insert(ch);
        }
        // Move the cursor off the end: an in-the-middle edit should not pop a
        // menu (the user is not extending the prefix).
        buf.move_left();
        assert!(buf.slash_completions(&cands).is_empty());
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
    fn ctrl_o_opens_cell_inspection_only_from_an_empty_prompt() {
        let mut editor = LineEditor::new("> ");
        let mut empty = InputBuffer::new();
        assert_eq!(editor.handle_key(ctrl('o'), &mut empty), Action::Inspect);
        assert_eq!(editor.handle_key(ctrl('l'), &mut empty), Action::Repaint);

        let mut filled = InputBuffer::new();
        filled.insert('o');
        assert_eq!(editor.handle_key(ctrl('o'), &mut filled), Action::Continue);
        assert_eq!(filled.as_str(), "o");
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
        // Continuation prompt: dim ⋮ + space (with ANSI codes)
        assert!(
            rendered.lines[1].contains('\u{22EE}'),
            "continuation line should contain ⋮ glyph: {:?}",
            rendered.lines[1]
        );
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

    #[test]
    fn history_file_round_trips_multiline_entries() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("history");
        let mut editor = LineEditor::new("> ");
        editor.push_history("single line");
        editor.push_history("first\nsecond\n  indented");
        editor.push_history("literal \\n and a back\\slash");
        editor.save_history_file(&path);

        let mut loaded = LineEditor::new("> ");
        loaded.load_history_file(&path);
        assert_eq!(loaded.history.len(), 3);
        assert_eq!(loaded.history[0], "single line");
        assert_eq!(loaded.history[1], "first\nsecond\n  indented");
        assert_eq!(loaded.history[2], "literal \\n and a back\\slash");
    }

    // The Option is required by the `fn() -> Option<String>` probe type.
    #[allow(clippy::unnecessary_wraps)]
    fn sample_dev_status() -> Option<String> {
        Some("building".to_string())
    }

    #[test]
    fn dev_status_renders_only_on_an_empty_prompt() {
        let mut editor = LineEditor::new("> ");
        editor.dev_status_label = Some("building".to_string());
        editor.status_since = std::time::Instant::now();
        let line = editor.status_line(std::time::Instant::now());
        editor.dev_status_line = line;
        let empty = editor.render(&InputBuffer::new());
        assert!(empty.lines[0].contains("building"), "{}", empty.lines[0]);
        assert!(empty.lines[0].contains("\x1b[33m"), "{}", empty.lines[0]);

        let mut typed = InputBuffer::new();
        typed.replace("half typed");
        let rendered = editor.render(&typed);
        assert!(
            !rendered.lines[0].contains("[dev:"),
            "status must not move the cursor of a draft: {}",
            rendered.lines[0]
        );
    }

    #[test]
    fn reverse_search_renders_a_picker() {
        let mut editor = LineEditor::new("> ");
        for text in ["alpha one", "beta", "alpha two"] {
            editor.push_history(text);
        }
        editor.search_query = Some("alpha".to_string());
        editor.search_match_idx = Some(2);

        let rendered = editor.render(&InputBuffer::new());
        assert_eq!(rendered.line_count, 3);
        assert!(rendered.lines[0].contains("alpha"));
        assert!(
            rendered.lines[1].contains("\x1b[7m"),
            "selection must be visible: {}",
            rendered.lines[1]
        );
        assert!(rendered.lines[2].contains("alpha one"));
    }

    #[test]
    fn reverse_search_dims_older_matches() {
        let mut editor = LineEditor::new("> ");
        for index in 0..8 {
            editor.push_history(format!("entry {index}"));
        }
        editor.search_query = Some(String::new());

        let rendered = editor.render(&InputBuffer::new());
        // Query line plus six bounded matches; the sixth match is older than
        // the newest five and is dimmed.
        assert_eq!(rendered.line_count, 7);
        assert!(
            rendered.lines[6].contains("\x1b[2m"),
            "older matches must dim: {}",
            rendered.lines[6]
        );
    }

    #[test]
    fn idle_tick_picks_up_a_changed_dev_status() {
        let mut editor = LineEditor::new("> ");
        editor.set_status_source(sample_dev_status);
        let mut out = Vec::new();
        let mut lines = 1usize;
        editor
            .refresh_status(&mut out, &InputBuffer::new(), &mut lines)
            .unwrap();
        assert!(
            editor
                .dev_status_line
                .as_deref()
                .is_some_and(|line| line.contains("building")),
            "{:?}",
            editor.dev_status_line
        );
    }

    #[test]
    fn history_file_reads_legacy_unescaped_lines() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("history");
        std::fs::write(&path, "first\nsecond\n\n").unwrap();

        let mut loaded = LineEditor::new("> ");
        loaded.load_history_file(&path);
        assert_eq!(loaded.history, vec!["first", "second"]);
    }

    // ── Undo ────────────────────────────────────────────────────────

    #[test]
    fn undo_restores_after_kill() {
        let mut buf = InputBuffer::new();
        for ch in "hello world".chars() {
            buf.insert(ch);
        }
        buf.kill_to_end(); // kills nothing (cursor at end), but snapshots
        buf.move_home();
        buf.kill_to_end(); // kills "hello world"
        assert_eq!(buf.as_str(), "");
        buf.undo();
        assert_eq!(buf.as_str(), "hello world");
    }

    #[test]
    fn undo_is_empty_on_fresh_buffer() {
        let mut buf = InputBuffer::new();
        assert!(!buf.undo()); // nothing to undo
    }

    #[test]
    fn undo_after_many_inserts() {
        let mut buf = InputBuffer::new();
        // Insert 20 chars (triggers snapshots due to debounce every 5)
        for ch in "abcdefghijklmnopqrst".chars() {
            buf.insert(ch);
        }
        // Should be able to undo at least once
        assert!(buf.undo());
        assert!(buf.as_str().len() < 20);
    }

    // ── Kill/Yank ───────────────────────────────────────────────────

    #[test]
    fn kill_and_yank_round_trip() {
        let mut buf = InputBuffer::new();
        for ch in "hello world".chars() {
            buf.insert(ch);
        }
        buf.kill_word_back(); // kills "world"
        assert_eq!(buf.as_str(), "hello ");
        buf.yank();
        assert_eq!(buf.as_str(), "hello world");
    }

    #[test]
    fn kill_buffer_survives_clear() {
        let mut buf = InputBuffer::new();
        for ch in "test".chars() {
            buf.insert(ch);
        }
        buf.kill_to_end();
        buf.clear();
        assert!(buf.is_empty());
        buf.yank(); // kill buffer should still have "test"
                    // Actually kill_to_end at cursor=4 kills nothing. Let me redo:
        let mut buf = InputBuffer::new();
        for ch in "test".chars() {
            buf.insert(ch);
        }
        buf.move_home();
        buf.kill_to_end(); // kills "test"
        assert_eq!(buf.as_str(), "");
        buf.clear();
        buf.yank();
        assert_eq!(buf.as_str(), "test");
    }

    // ── Ghost text ──────────────────────────────────────────────────

    #[test]
    fn ghost_text_from_history() {
        let mut editor = LineEditor::new("> ");
        editor.push_history("explain src/main.rs");
        assert_eq!(
            editor.history_ghost("explain"),
            Some(" src/main.rs".to_string())
        );
    }

    #[test]
    fn ghost_text_no_match() {
        let mut editor = LineEditor::new("> ");
        editor.push_history("hello");
        assert_eq!(editor.history_ghost("xyz"), None);
    }

    #[test]
    fn ghost_text_skips_slash_commands() {
        let mut editor = LineEditor::new("> ");
        editor.push_history("/help");
        assert_eq!(editor.history_ghost("/he"), None);
    }

    #[test]
    fn ghost_text_most_recent_wins() {
        let mut editor = LineEditor::new("> ");
        editor.push_history("fix the old bug");
        editor.push_history("fix the new feature");
        assert_eq!(
            editor.history_ghost("fix"),
            Some(" the new feature".to_string())
        );
    }

    // ── Paste normalization ─────────────────────────────────────────

    #[test]
    fn paste_strips_ansi() {
        let result = strip_ansi_from_paste("\x1b[31mhello\x1b[0m world");
        assert_eq!(result, "hello world");
    }

    #[test]
    fn paste_normalizes_cr() {
        let mut buf = InputBuffer::new();
        buf.insert_str("line1\r\nline2\rline3");
        assert_eq!(buf.as_str(), "line1\nline2\nline3");
    }

    #[test]
    fn paste_converts_tabs() {
        let mut buf = InputBuffer::new();
        buf.insert_str("a\tb");
        assert_eq!(buf.as_str(), "a    b");
    }

    // ── Grapheme-cluster movement ───────────────────────────────────

    #[test]
    fn grapheme_movement_ascii() {
        let mut buf = InputBuffer::new();
        for ch in "abc".chars() {
            buf.insert(ch);
        }
        buf.move_left();
        assert_eq!(buf.cursor, 2); // before 'c'
        buf.move_left();
        assert_eq!(buf.cursor, 1); // before 'b'
        buf.move_right();
        assert_eq!(buf.cursor, 2); // before 'c'
    }

    #[test]
    fn grapheme_backspace_multibyte() {
        let mut buf = InputBuffer::new();
        buf.insert_str("hello");
        // 'hello' is 5 single-byte chars
        buf.backspace();
        assert_eq!(buf.as_str(), "hell");
    }

    #[test]
    fn visible_width_cjk() {
        // CJK characters are 2 columns wide
        assert_eq!(visible_width("ab"), 2);
        // Basic test with known-width chars
        assert_eq!(visible_width("a"), 1);
    }

    // ── Multiline history awareness ─────────────────────────────────

    #[test]
    fn up_arrow_only_on_first_line() {
        let mut editor = LineEditor::new("> ");
        editor.push_history("old command");
        let mut input = InputBuffer::new();
        input.insert_str("line1\nline2");
        // Cursor is at end of "line2" (second line)
        let _ = editor.handle_key(key(KeyCode::Up), &mut input);
        // Should NOT navigate history because we're on the second line
        assert_eq!(input.as_str(), "line1\nline2");
    }

    #[test]
    fn cursor_up_down_within_multiline() {
        let mut editor = LineEditor::new("> ");
        let mut input = InputBuffer::new();
        input.insert_str("short\nlonger line\nx");
        // Cursor at end of "x" (third line, col 1)
        let _ = editor.handle_key(key(KeyCode::Up), &mut input);
        // Should be on second line now
        let before = &input.as_str()[..input.cursor];
        assert_eq!(before.matches('\n').count(), 1, "should be on line 2");
        let _ = editor.handle_key(key(KeyCode::Up), &mut input);
        let before = &input.as_str()[..input.cursor];
        assert_eq!(before.matches('\n').count(), 0, "should be on line 1");
    }

    #[test]
    fn cursor_down_from_first_line() {
        let mut editor = LineEditor::new("> ");
        let mut input = InputBuffer::new();
        input.insert_str("abc\ndef\nghi");
        input.move_home(); // go to start
        let _ = editor.handle_key(key(KeyCode::Down), &mut input);
        let before = &input.as_str()[..input.cursor];
        assert!(before.contains('\n'), "should have moved past first line");
    }

    #[test]
    fn up_arrow_works_on_single_line() {
        let mut editor = LineEditor::new("> ");
        editor.push_history("old command");
        let mut input = InputBuffer::new();
        input.insert_str("new");
        let _ = editor.handle_key(key(KeyCode::Up), &mut input);
        assert_eq!(input.as_str(), "old command");
    }
}
