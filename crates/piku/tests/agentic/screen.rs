#![allow(dead_code)]
// This is test-support code included via `#[path]` into several test
// binaries; each uses a subset. The style lints below match those already
// allowed in the agentic harness this observer is shared with.
#![allow(clippy::struct_excessive_bools, clippy::format_push_string)]
//! Shared "seeing" model for PTY-driven TUI tests.
//!
//! # Why a rendered screen, not raw bytes
//!
//! A user does not experience the byte stream piku writes; they experience the
//! *grid a terminal paints from it*. Asserting on raw bytes tests the wrong
//! artifact: it passes when piku emits the right escapes in the wrong order and
//! fails on a cosmetic reordering that renders identically. This module feeds
//! the real PTY output through a `vt100::Parser` and exposes the resulting
//! screen — the same thing the agentic judge loop observes — so a test asserts
//! on what a human sees.
//!
//! # vt100 is faithful enough (it caught a real bug)
//!
//! An earlier note claimed `vt100` was "not realistic enough" because a pinned
//! header vanished in the parser. That was backwards. DECSTBM (`ESC [ r`, and
//! the parameterized `ESC [ t;b r`) *homes the cursor* as a side effect — the
//! VT100 user guide says "the cursor is placed in the home position", and xterm
//! implements the sequence as `set_tb_margins(...)` then `CursorSet(screen, 0,
//! 0, ...)`. So when piku reset its scroll region and then erased from the
//! cursor down, the erase started at row 1 and wiped the whole frame. `vt100`
//! reproduced that exactly. A real xterm does the same thing; the blank screen
//! was a genuine product bug, and the parser was the tool that exposed it. The
//! lesson is the opposite of the note: this is the realistic surface, so keep
//! testing on it.
//!
//! # Misuse resistance
//!
//! The grid size is owned by the observer and fixed at construction. The old
//! ad-hoc helpers took the size as free arguments next to the byte buffer
//! (`rendered_rows(buf, cols, rows)`), which let a caller parse at a size that
//! disagreed with the PTY's actual winsize and silently get a wrong render.
//! Here `ScreenObserver::new(rows, cols)` is the single source of the grid, and
//! the PTY harness constructs the observer from the same winsize it set, so the
//! two cannot drift.

/// Colors distilled from vt100 cells for style assertions (e.g. the ready caret
/// is green). Kept small on purpose: tests assert on the handful of colors piku
/// actually uses as signals.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Color {
    Default,
    Black,
    Red,
    Green,
    Yellow,
    Blue,
    Magenta,
    Cyan,
    White,
    Idx(u8),
    Rgb(u8, u8, u8),
}

impl From<vt100::Color> for Color {
    fn from(c: vt100::Color) -> Self {
        match c {
            vt100::Color::Default => Color::Default,
            vt100::Color::Idx(0) => Color::Black,
            vt100::Color::Idx(1) => Color::Red,
            vt100::Color::Idx(2) => Color::Green,
            vt100::Color::Idx(3) => Color::Yellow,
            vt100::Color::Idx(4) => Color::Blue,
            vt100::Color::Idx(5) => Color::Magenta,
            vt100::Color::Idx(6) => Color::Cyan,
            vt100::Color::Idx(7) => Color::White,
            vt100::Color::Idx(n) => Color::Idx(n),
            vt100::Color::Rgb(r, g, b) => Color::Rgb(r, g, b),
        }
    }
}

/// One rendered cell with the styling a human perceives.
#[derive(Debug, Clone)]
pub struct StyledCell {
    pub ch: String,
    pub bold: bool,
    pub dim: bool,
    pub italic: bool,
    pub inverse: bool,
    pub fg: Color,
    pub bg: Color,
}

/// One rendered row, with its cells and the trimmed plain text.
#[derive(Debug, Clone)]
pub struct StyledRow {
    pub row_index: u16,
    pub cells: Vec<StyledCell>,
    pub text: String,
}

impl StyledRow {
    /// The foreground color of the first non-space glyph, or `None` for a blank
    /// row. Useful for "the caret is green" style assertions without indexing
    /// into cells by hand.
    #[must_use]
    pub fn first_glyph_fg(&self) -> Option<Color> {
        self.cells
            .iter()
            .find(|c| !c.ch.trim().is_empty())
            .map(|c| c.fg)
    }
}

/// An immutable view of the rendered screen at one instant — what the user sees.
#[derive(Debug, Clone)]
pub struct ScreenSnapshot {
    /// Full rendered screen text (rows joined by newline).
    pub contents: String,
    /// Individual rows, right-trimmed.
    pub rows: Vec<String>,
    /// Cursor position `(row, col)`, 0-indexed.
    pub cursor: (u16, u16),
    /// Whether the cursor is visible.
    pub cursor_visible: bool,
    /// Fully-styled rows (all rows), for color/attribute assertions.
    pub styled_rows: Vec<StyledRow>,
    /// Grid size `(rows, cols)`.
    pub size: (u16, u16),
}

impl ScreenSnapshot {
    /// The row the cursor sits on (where the user is typing). Follows the
    /// cursor rather than assuming a fixed row, because piku's DECSTBM layout
    /// places the prompt at different absolute rows depending on terminal size.
    #[must_use]
    pub fn input_row(&self) -> &str {
        let r = self.cursor.0 as usize;
        self.rows.get(r).map_or_else(
            || self.rows.last().map_or("", String::as_str),
            String::as_str,
        )
    }

    /// The row above the cursor (typically the footer/status bar).
    #[must_use]
    pub fn footer_row(&self) -> &str {
        let r = self.cursor.0.saturating_sub(1) as usize;
        self.rows.get(r).map_or("", String::as_str)
    }

    /// The styled view of a specific row, if present.
    #[must_use]
    pub fn styled_row(&self, row: u16) -> Option<&StyledRow> {
        self.styled_rows.iter().find(|r| r.row_index == row)
    }

    /// The styled view of the row the cursor is on.
    #[must_use]
    pub fn styled_input_row(&self) -> Option<&StyledRow> {
        self.styled_row(self.cursor.0)
    }

    /// Every non-empty visible row, oldest first.
    #[must_use]
    pub fn non_empty_rows(&self) -> Vec<&str> {
        self.rows
            .iter()
            .map(String::as_str)
            .filter(|l| !l.trim().is_empty())
            .collect()
    }

    /// Whether any visible row contains `needle`. This is the "did the user see
    /// it" predicate — it looks at the rendered grid, not the byte stream, so an
    /// erased or overdrawn string correctly reads as *not* visible.
    #[must_use]
    pub fn shows(&self, needle: &str) -> bool {
        self.contents.contains(needle)
    }

    /// Whether piku is ready for input (prompt present, not thinking/streaming).
    #[must_use]
    pub fn is_ready(&self) -> bool {
        if !self.cursor_visible {
            return false;
        }
        let input = self.input_row().trim_start();
        let has_prompt = input.starts_with('\u{276F}')
            || input.starts_with('>')
            || input.starts_with('!')
            || input.contains("Send a message");
        if !has_prompt {
            return false;
        }
        !(input.contains("thinking") || input.contains('\u{00B7}') || input.contains('\u{273B}'))
    }

    /// Whether a tool-permission prompt (`y/n/a?`) is showing on the input row.
    #[must_use]
    pub fn has_permission_prompt(&self) -> bool {
        self.input_row().contains("y/n/a?")
    }

    /// The screen as a bounded string for diagnostics / LLM critique.
    #[must_use]
    pub fn summary(&self, max_lines: usize) -> String {
        let visible = self.non_empty_rows();
        let mut out = String::new();
        for (i, line) in visible.iter().enumerate() {
            if i >= max_lines {
                out.push_str(&format!("  ... ({} more lines)\n", visible.len() - i));
                break;
            }
            let truncated: String = line.chars().take(120).collect();
            out.push_str(&truncated);
            out.push('\n');
        }
        out
    }
}

/// A persistent VT100 parser: feed it the PTY byte stream, ask it what the
/// screen looks like. Size is fixed at construction and never taken again, so a
/// snapshot is always rendered at the grid the terminal actually uses.
pub struct ScreenObserver {
    parser: vt100::Parser,
    rows: u16,
    cols: u16,
}

impl ScreenObserver {
    /// Create an observer for a `rows`×`cols` grid. Use the *same* size the PTY
    /// winsize is set to (the harness does this for you) so the render matches
    /// what piku drew into.
    #[must_use]
    pub fn new(rows: u16, cols: u16) -> Self {
        Self {
            parser: vt100::Parser::new(rows, cols, 500),
            rows,
            cols,
        }
    }

    /// Feed raw PTY bytes into the parser (incremental; call as bytes arrive).
    pub fn process(&mut self, bytes: &[u8]) {
        self.parser.process(bytes);
    }

    /// The grid size this observer renders at.
    #[must_use]
    pub fn size(&self) -> (u16, u16) {
        (self.rows, self.cols)
    }

    /// Render the current screen.
    #[must_use]
    pub fn snapshot(&self) -> ScreenSnapshot {
        let screen = self.parser.screen();
        let (term_rows, term_cols) = screen.size();

        let mut rows = Vec::with_capacity(term_rows as usize);
        let mut styled_rows = Vec::with_capacity(term_rows as usize);
        for r in 0..term_rows {
            let mut row = String::new();
            let mut cells = Vec::new();
            for c in 0..term_cols {
                if let Some(cell) = screen.cell(r, c) {
                    let ch = cell.contents().to_string();
                    row.push_str(&ch);
                    cells.push(StyledCell {
                        ch,
                        bold: cell.bold(),
                        dim: cell.dim(),
                        italic: cell.italic(),
                        inverse: cell.inverse(),
                        fg: cell.fgcolor().into(),
                        bg: cell.bgcolor().into(),
                    });
                }
            }
            let text = row.trim_end().to_string();
            styled_rows.push(StyledRow {
                row_index: r,
                cells,
                text,
            });
            rows.push(row.trim_end().to_string());
        }

        ScreenSnapshot {
            contents: screen.contents(),
            rows,
            cursor: screen.cursor_position(),
            cursor_visible: !screen.hide_cursor(),
            styled_rows,
            size: (term_rows, term_cols),
        }
    }
}
