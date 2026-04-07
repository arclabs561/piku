use std::time::Duration;

// ===========================================================================
// Action space — keystroke-level interaction
// ===========================================================================

#[derive(Debug, Clone)]
pub enum SpecialKey {
    Enter,
    Tab,
    Escape,
    Backspace,
    Delete,
    ArrowUp,
    ArrowDown,
    ArrowLeft,
    ArrowRight,
    Home,
    End,
    CtrlC,
    CtrlD,
    CtrlL,
    CtrlA,
    CtrlE,
    CtrlW,
    CtrlU,
}

impl SpecialKey {
    pub fn as_bytes(&self) -> &[u8] {
        match self {
            SpecialKey::Enter => b"\r",
            SpecialKey::Tab => b"\t",
            SpecialKey::Escape => b"\x1b",
            SpecialKey::Backspace => b"\x7f",
            SpecialKey::Delete => b"\x1b[3~",
            SpecialKey::ArrowUp => b"\x1b[A",
            SpecialKey::ArrowDown => b"\x1b[B",
            SpecialKey::ArrowLeft => b"\x1b[D",
            SpecialKey::ArrowRight => b"\x1b[C",
            SpecialKey::Home => b"\x1b[H",
            SpecialKey::End => b"\x1b[F",
            SpecialKey::CtrlC => b"\x03",
            SpecialKey::CtrlD => b"\x04",
            SpecialKey::CtrlL => b"\x0c",
            SpecialKey::CtrlA => b"\x01",
            SpecialKey::CtrlE => b"\x05",
            SpecialKey::CtrlW => b"\x17",
            SpecialKey::CtrlU => b"\x15",
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            SpecialKey::Enter => "Enter",
            SpecialKey::Tab => "Tab",
            SpecialKey::Escape => "Escape",
            SpecialKey::Backspace => "Backspace",
            SpecialKey::Delete => "Delete",
            SpecialKey::ArrowUp => "ArrowUp",
            SpecialKey::ArrowDown => "ArrowDown",
            SpecialKey::ArrowLeft => "ArrowLeft",
            SpecialKey::ArrowRight => "ArrowRight",
            SpecialKey::Home => "Home",
            SpecialKey::End => "End",
            SpecialKey::CtrlC => "Ctrl-C",
            SpecialKey::CtrlD => "Ctrl-D",
            SpecialKey::CtrlL => "Ctrl-L",
            SpecialKey::CtrlA => "Ctrl-A",
            SpecialKey::CtrlE => "Ctrl-E",
            SpecialKey::CtrlW => "Ctrl-W",
            SpecialKey::CtrlU => "Ctrl-U",
        }
    }
}

impl std::fmt::Display for SpecialKey {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

#[derive(Debug, Clone)]
pub enum Action {
    Type(char),
    Key(SpecialKey),
    Observe,
    Wait(Duration),
    TypeString { text: String, delay_ms: u64 },
    Submit(String),
}

impl std::fmt::Display for Action {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Action::Type(c) => write!(f, "Type('{c}')"),
            Action::Key(k) => write!(f, "Key({k})"),
            Action::Observe => write!(f, "Observe"),
            Action::Wait(d) => write!(f, "Wait({d:?})"),
            Action::TypeString { text, .. } => {
                let preview = if text.len() > 30 {
                    format!("{}...", &text[..30])
                } else {
                    text.clone()
                };
                write!(f, "TypeString({preview:?})")
            }
            Action::Submit(s) => {
                let preview = if s.len() > 40 {
                    format!("{}...", &s[..40])
                } else {
                    s.clone()
                };
                write!(f, "Submit({preview:?})")
            }
        }
    }
}

// ===========================================================================
// Screen snapshot — structured VT100 observation
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq)]
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

#[derive(Debug, Clone)]
pub struct StyledRow {
    pub row_index: u16,
    pub cells: Vec<StyledCell>,
    pub text: String,
}

#[derive(Debug, Clone)]
pub struct ScreenSnapshot {
    pub contents: String,
    pub rows: Vec<String>,
    pub cursor: (u16, u16),
    pub cursor_visible: bool,
    pub styled_rows: Vec<StyledRow>,
    pub size: (u16, u16),
}

impl ScreenSnapshot {
    /// The row the cursor is on (follows cursor, DECSTBM-aware).
    pub fn input_row(&self) -> &str {
        let r = self.cursor.0 as usize;
        if r < self.rows.len() {
            &self.rows[r]
        } else {
            self.rows.last().map(|s| s.as_str()).unwrap_or("")
        }
    }

    /// The row above the cursor (typically the footer/status bar).
    pub fn footer_row(&self) -> &str {
        let r = self.cursor.0.saturating_sub(1) as usize;
        if r < self.rows.len() {
            &self.rows[r]
        } else {
            ""
        }
    }

    /// Check if piku is ready for input.
    pub fn is_ready(&self) -> bool {
        if !self.cursor_visible {
            return false;
        }
        let input = self.input_row().trim_start();
        input.starts_with('\u{276F}')
            || input.starts_with('>')
            || input.starts_with('!')
            || input.contains("Send a message")
    }

    /// All non-empty visible rows for LLM context.
    pub fn summary(&self, max_lines: usize) -> String {
        let visible: Vec<&str> = self
            .rows
            .iter()
            .map(|s| s.as_str())
            .filter(|l| !l.trim().is_empty())
            .collect();

        let mut out = String::new();
        for (i, line) in visible.iter().enumerate() {
            if i >= max_lines {
                out.push_str(&format!("  ... ({} more lines)\n", visible.len() - i));
                break;
            }
            let truncated = if line.len() > 120 { &line[..120] } else { line };
            out.push_str(truncated);
            out.push('\n');
        }
        out
    }
}

// ===========================================================================
// Bug / Severity / Finding types
// ===========================================================================

#[derive(Debug, Clone)]
pub struct Bug {
    pub severity: Severity,
    pub description: String,
    pub expected: String,
    pub actual: String,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Severity {
    Critical,
    Major,
    Minor,
    Info,
}

impl std::fmt::Display for Severity {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Severity::Critical => write!(f, "CRITICAL"),
            Severity::Major => write!(f, "MAJOR"),
            Severity::Minor => write!(f, "minor"),
            Severity::Info => write!(f, "info"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Finding {
    pub severity: Severity,
    pub description: String,
    pub expected: String,
    pub actual: String,
}

#[derive(Debug, Clone)]
pub struct CritiqueEntry {
    pub phase: String,
    pub action_desc: String,
    pub screen_text: String,
    pub observations: Vec<String>,
    pub bugs: Vec<Bug>,
    pub deterministic_findings: Vec<Finding>,
    pub workspace_diff: String,
    pub next_action: NextAction,
}

#[derive(Debug, Clone)]
pub enum NextAction {
    Send(String),
    Quit,
}

pub fn safe_truncate(s: &str, max_chars: usize) -> &str {
    if s.chars().count() <= max_chars {
        return s;
    }
    let mut idx = 0;
    for (i, _) in s.char_indices().take(max_chars) {
        idx = i;
    }
    &s[..idx]
}
