/// Streaming markdown renderer for terminal output.
///
/// Processes text chunk-by-chunk as it arrives from the API, rendering
/// markdown constructs with ANSI formatting:
///
///   - Code blocks: box-drawn frame with syntect highlighting
///   - Headings: bold + cyan
///   - List items: bullet with indent
///   - Block quotes: gutter, dim
///   - Inline: **bold**, *italic*, `code` (green)
///   - Horizontal rules
///
/// Output uses `\r\n` line endings for DECSTBM scroll-region compatibility.
use std::fmt::Write;
use std::sync::OnceLock;

use syntect::easy::HighlightLines;
use syntect::highlighting::{Theme, ThemeSet};
use syntect::parsing::SyntaxSet;
use syntect::util::{as_24_bit_terminal_escaped, LinesWithEndings};

// ── ANSI constants ──────────────────────────────────────────────────────────

const RESET: &str = "\x1b[0m";
const BOLD: &str = "\x1b[1m";
const DIM: &str = "\x1b[2m";
const ITALIC: &str = "\x1b[3m";
const CYAN: &str = "\x1b[36m";
const GREEN: &str = "\x1b[32m";
const YELLOW: &str = "\x1b[33m";
const MAGENTA: &str = "\x1b[35m";

// ── Shared syntect state (heavy -- load once) ───────────────────────────────

fn syntax_set() -> &'static SyntaxSet {
    static SS: OnceLock<SyntaxSet> = OnceLock::new();
    SS.get_or_init(SyntaxSet::load_defaults_newlines)
}

fn syntax_theme() -> &'static Theme {
    static TH: OnceLock<Theme> = OnceLock::new();
    TH.get_or_init(|| {
        ThemeSet::load_defaults()
            .themes
            .remove("base16-ocean.dark")
            .unwrap_or_default()
    })
}

// ── Streaming renderer ──────────────────────────────────────────────────────

pub struct StreamingMarkdown {
    /// Partial line not yet terminated by `\n`.
    line_buf: String,
    /// Currently inside a fenced code block.
    in_code_block: bool,
    /// Language tag from the opening fence.
    code_lang: String,
    /// Accumulated code block content.
    code_buf: String,
}

impl Default for StreamingMarkdown {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamingMarkdown {
    #[must_use]
    pub fn new() -> Self {
        Self {
            line_buf: String::new(),
            in_code_block: false,
            code_lang: String::new(),
            code_buf: String::new(),
        }
    }

    /// Push a streaming text chunk, return rendered ANSI output.
    pub fn push(&mut self, text: &str) -> String {
        self.line_buf.push_str(text);
        let mut out = String::new();

        // Process all complete lines in the buffer.
        while let Some(nl) = self.line_buf.find('\n') {
            let line = self.line_buf[..nl].to_string();
            self.line_buf = self.line_buf[nl + 1..].to_string();
            self.process_line(&line, &mut out);
        }

        out
    }

    /// Flush any remaining buffered content (call at end of turn / before tool).
    pub fn flush(&mut self) -> String {
        let mut out = String::new();

        // Flush partial line
        if !self.line_buf.is_empty() {
            let line = std::mem::take(&mut self.line_buf);
            if self.in_code_block {
                self.code_buf.push_str(&line);
                self.code_buf.push('\n');
            } else {
                out.push_str(&render_inline(&line));
            }
        }

        // If we're mid-code-block, render what we have
        if self.in_code_block {
            out.push_str(&render_code_block(&self.code_lang, &self.code_buf));
            self.in_code_block = false;
            self.code_lang.clear();
            self.code_buf.clear();
        }

        out
    }

    fn process_line(&mut self, line: &str, out: &mut String) {
        let trimmed = line.trim_end();

        // ── Code fence detection ────────────────────────────────────────
        if let Some(rest) = trimmed.strip_prefix("```") {
            if self.in_code_block {
                // Closing fence
                out.push_str(&render_code_block(&self.code_lang, &self.code_buf));
                self.in_code_block = false;
                self.code_lang.clear();
                self.code_buf.clear();
                return;
            }
            // Opening fence
            self.in_code_block = true;
            self.code_lang = rest.trim().to_string();
            self.code_buf.clear();
            return;
        }

        if self.in_code_block {
            self.code_buf.push_str(line);
            self.code_buf.push('\n');
            return;
        }

        // ── Block-level constructs ──────────────────────────────────────

        // Headings
        if let Some((prefix, text)) = try_strip_heading(trimmed) {
            let _ = write!(
                out,
                "\r\n{BOLD}{CYAN}{prefix}{RESET}{BOLD}{CYAN}{}{RESET}\r\n",
                render_inline(text)
            );
            return;
        }

        // Horizontal rule
        if is_horizontal_rule(trimmed) {
            let _ = write!(out, "{DIM}────────────────────────────────────────{RESET}\r\n");
            return;
        }

        // Block quote
        if let Some(rest) = trimmed.strip_prefix("> ") {
            let _ = write!(out, "{DIM}│ {}{RESET}\r\n", render_inline(rest));
            return;
        }
        if trimmed == ">" {
            let _ = write!(out, "{DIM}│{RESET}\r\n");
            return;
        }

        // Unordered list items
        if let Some((indent, text)) = try_strip_list_item(trimmed) {
            let pad = indent * 2;
            let _ = write!(
                out,
                "{:pad$}  {DIM}{CYAN}{RESET} {}\r\n",
                "",
                render_inline(text)
            );
            return;
        }

        // Ordered list items (e.g. "1. text")
        if let Some((indent, num, text)) = try_strip_ordered_item(trimmed) {
            let pad = indent * 2;
            let _ = write!(
                out,
                "{:pad$}{DIM}{num}.{RESET} {}\r\n",
                "",
                render_inline(text)
            );
            return;
        }

        // Regular paragraph line
        out.push_str(&render_inline(trimmed));
        out.push_str("\r\n");
    }
}

// ── Block rendering ─────────────────────────────────────────────────────────

fn render_code_block(lang: &str, code: &str) -> String {
    let mut out = String::new();
    let code = code.trim_end_matches('\n');

    // Top frame
    if lang.is_empty() {
        let _ = write!(out, "{DIM}╭──{RESET}\r\n");
    } else {
        let _ = write!(out, "{DIM}╭─ {CYAN}{lang}{RESET}\r\n");
    }

    // Highlighted code lines
    let ss = syntax_set();
    let theme = syntax_theme();
    let syntax = ss
        .find_syntax_by_token(lang)
        .unwrap_or_else(|| ss.find_syntax_plain_text());
    let mut hl = HighlightLines::new(syntax, theme);

    for line in LinesWithEndings::from(code) {
        let _ = write!(out, "{DIM}│{RESET} ");
        match hl.highlight_line(line, ss) {
            Ok(ranges) => {
                let escaped = as_24_bit_terminal_escaped(&ranges, false);
                let escaped = escaped.trim_end_matches('\n');
                out.push_str(escaped);
                out.push_str(RESET);
            }
            Err(_) => {
                out.push_str(line.trim_end_matches('\n'));
            }
        }
        out.push_str("\r\n");
    }

    // Handle empty code blocks
    if code.is_empty() {
        let _ = write!(out, "{DIM}│{RESET}\r\n");
    }

    // Bottom frame
    let _ = write!(out, "{DIM}╰──{RESET}\r\n");
    out
}

// ── Inline rendering ────────────────────────────────────────────────────────

/// Render inline markdown formatting within a single line.
fn render_inline(text: &str) -> String {
    let mut out = String::new();
    let chars: Vec<char> = text.chars().collect();
    let len = chars.len();
    let mut i = 0;

    while i < len {
        // Inline code: `...`
        if chars[i] == '`' {
            if let Some(end) = find_char_from(&chars, '`', i + 1) {
                let code: String = chars[i + 1..end].iter().collect();
                let _ = write!(out, "{GREEN}`{code}`{RESET}");
                i = end + 1;
                continue;
            }
        }

        // Bold: **...**
        if i + 1 < len && chars[i] == '*' && chars[i + 1] == '*' {
            if let Some(end) = find_double_star(&chars, i + 2) {
                let inner: String = chars[i + 2..end].iter().collect();
                let _ = write!(out, "{BOLD}{YELLOW}{inner}{RESET}");
                i = end + 2;
                continue;
            }
        }

        // Italic: *...*  (single star, not followed by another star)
        if chars[i] == '*' && (i + 1 >= len || chars[i + 1] != '*') {
            if let Some(end) = find_single_star(&chars, i + 1) {
                let inner: String = chars[i + 1..end].iter().collect();
                let _ = write!(out, "{ITALIC}{MAGENTA}{inner}{RESET}");
                i = end + 1;
                continue;
            }
        }

        out.push(chars[i]);
        i += 1;
    }

    out
}

fn find_char_from(chars: &[char], target: char, start: usize) -> Option<usize> {
    chars.iter().enumerate().skip(start).find_map(|(i, &c)| {
        if c == target {
            Some(i)
        } else {
            None
        }
    })
}

fn find_double_star(chars: &[char], start: usize) -> Option<usize> {
    chars
        .windows(2)
        .enumerate()
        .skip(start)
        .find_map(|(i, w)| {
            if w[0] == '*' && w[1] == '*' {
                Some(i)
            } else {
                None
            }
        })
}

fn find_single_star(chars: &[char], start: usize) -> Option<usize> {
    for (i, &c) in chars.iter().enumerate().skip(start) {
        if c == '*' {
            // Don't match ** as end of single-star span
            if chars.get(i + 1) == Some(&'*') {
                continue;
            }
            // Must have content (not empty span)
            if i > start {
                return Some(i);
            }
        }
    }
    None
}

// ── Line classification helpers ─────────────────────────────────────────────

/// Try to parse a heading. Returns (prefix, remaining text).
fn try_strip_heading(line: &str) -> Option<(&str, &str)> {
    for (prefix, skip) in [("#### ", 5), ("### ", 4), ("## ", 3), ("# ", 2)] {
        if let Some(rest) = line.strip_prefix(prefix) {
            let _ = rest;
            return Some((prefix, &line[skip..]));
        }
    }
    None
}

fn is_horizontal_rule(line: &str) -> bool {
    let trimmed = line.trim();
    if trimmed.len() < 3 {
        return false;
    }
    let first = trimmed.chars().next().unwrap_or(' ');
    matches!(first, '-' | '*' | '_') && trimmed.chars().all(|c| c == first || c == ' ')
}

/// Try to parse an unordered list item. Returns (`indent_level`, text).
fn try_strip_list_item(line: &str) -> Option<(usize, &str)> {
    let indent = line.len() - line.trim_start().len();
    let trimmed = line.trim_start();
    if let Some(rest) = trimmed.strip_prefix("- ") {
        Some((indent / 2, rest))
    } else {
        trimmed.strip_prefix("* ").map(|rest| (indent / 2, rest))
    }
}

/// Try to parse an ordered list item like "1. text". Returns (indent, number, text).
fn try_strip_ordered_item(line: &str) -> Option<(usize, &str, &str)> {
    let indent = line.len() - line.trim_start().len();
    let trimmed = line.trim_start();
    let dot_pos = trimmed.find(". ")?;
    let num_str = &trimmed[..dot_pos];
    if num_str.len() <= 3 && num_str.chars().all(|c| c.is_ascii_digit()) {
        Some((indent / 2, num_str, &trimmed[dot_pos + 2..]))
    } else {
        None
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

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

    #[test]
    fn heading_rendered() {
        let mut md = StreamingMarkdown::new();
        let out = md.push("## Hello world\n");
        let plain = strip_ansi(&out);
        assert!(plain.contains("## Hello world"));
    }

    #[test]
    fn code_block_framed() {
        let mut md = StreamingMarkdown::new();
        let out = md.push("```rust\nfn main() {}\n```\n");
        let plain = strip_ansi(&out);
        assert!(plain.contains("rust"), "should show language");
        assert!(plain.contains("fn main"), "should contain code");
        assert!(plain.contains('╭'), "should have top frame");
        assert!(plain.contains('╰'), "should have bottom frame");
    }

    #[test]
    fn inline_code_highlighted() {
        let mut md = StreamingMarkdown::new();
        let out = md.push("Use `foo()` here\n");
        assert!(out.contains(GREEN), "inline code should be green");
        let plain = strip_ansi(&out);
        assert!(plain.contains("`foo()`"));
    }

    #[test]
    fn list_bullets() {
        let mut md = StreamingMarkdown::new();
        let out = md.push("- item one\n- item two\n");
        let plain = strip_ansi(&out);
        assert!(plain.contains("item one"));
        assert!(plain.contains("item two"));
    }

    #[test]
    fn bold_and_italic() {
        let mut md = StreamingMarkdown::new();
        let out = md.push("This is **bold** and *italic*\n");
        assert!(out.contains(BOLD), "should contain bold");
        assert!(out.contains(ITALIC), "should contain italic");
    }

    #[test]
    fn streaming_partial_lines() {
        let mut md = StreamingMarkdown::new();
        let out1 = md.push("## Hel");
        assert!(out1.is_empty(), "partial line should buffer");
        let out2 = md.push("lo\n");
        let plain = strip_ansi(&out2);
        assert!(plain.contains("## Hello"));
    }

    #[test]
    fn flush_pending() {
        let mut md = StreamingMarkdown::new();
        let _ = md.push("partial text");
        let out = md.flush();
        let plain = strip_ansi(&out);
        assert!(plain.contains("partial text"));
    }

    #[test]
    fn code_block_flush_mid_block() {
        let mut md = StreamingMarkdown::new();
        let _ = md.push("```python\nprint('hi')\n");
        let out = md.flush();
        let plain = strip_ansi(&out);
        assert!(plain.contains("python"));
        assert!(plain.contains("print"));
    }

    #[test]
    fn horizontal_rule() {
        let mut md = StreamingMarkdown::new();
        let out = md.push("---\n");
        let plain = strip_ansi(&out);
        assert!(plain.contains("───"));
    }

    #[test]
    fn block_quote() {
        let mut md = StreamingMarkdown::new();
        let out = md.push("> quoted text\n");
        let plain = strip_ansi(&out);
        assert!(plain.contains("quoted text"));
    }
}
