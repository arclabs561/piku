//! Line buffer, cursor, and prompt rendering.

use std::io;

/// The current input line. `chars` holds one element per character so cursor
/// movement and the handoff blob's char-index cursor never split UTF-8.
pub struct LineEditor {
    chars: Vec<char>,
    cursor: usize,
}

impl LineEditor {
    pub fn new(text: &str, cursor: usize) -> Self {
        let chars: Vec<char> = text.chars().collect();
        Self {
            cursor: cursor.min(chars.len()),
            chars,
        }
    }

    pub fn text(&self) -> String {
        self.chars.iter().collect()
    }

    pub fn cursor(&self) -> usize {
        self.cursor
    }

    pub fn is_empty(&self) -> bool {
        self.chars.is_empty()
    }

    pub fn insert(&mut self, character: char) {
        self.chars.insert(self.cursor, character);
        self.cursor += 1;
    }

    pub fn backspace(&mut self) {
        if self.cursor > 0 {
            self.cursor -= 1;
            self.chars.remove(self.cursor);
        }
    }

    pub fn left(&mut self) {
        self.cursor = self.cursor.saturating_sub(1);
    }

    pub fn right(&mut self) {
        if self.cursor < self.chars.len() {
            self.cursor += 1;
        }
    }

    pub fn home(&mut self) {
        self.cursor = 0;
    }

    pub fn end(&mut self) {
        self.cursor = self.chars.len();
    }

    pub fn clear(&mut self) {
        self.chars.clear();
        self.cursor = 0;
    }

    /// Replaces the line, used by history recall.
    pub fn set(&mut self, text: &str, cursor: usize) {
        *self = Self::new(text, cursor);
    }

    /// Replaces `start..end` with `text` and parks the cursor after it. Used by
    /// completion, which must leave the rest of the line untouched.
    pub fn replace_range(&mut self, start: usize, end: usize, text: &str) {
        let end = end.min(self.chars.len());
        let start = start.min(end);
        let replacement: Vec<char> = text.chars().collect();
        self.cursor = start + replacement.len();
        self.chars.splice(start..end, replacement);
    }

    /// Repaints the prompt line and parks the cursor at its character offset.
    ///
    /// The buffer is drawn with syntax highlighting; ANSI escapes occupy no
    /// columns, so the visible text is exactly the buffer. Display width is
    /// approximated as one column per char, as agreed for this slice. Lines
    /// wider than the terminal wrap incorrectly; see the report.
    pub fn render(&self, prompt: &str) -> io::Result<()> {
        let text = self.text();
        let mut line = String::with_capacity(prompt.len() + text.len() + 24);
        line.push_str("\r\x1b[K");
        line.push_str(prompt);
        line.push_str(&crate::highlight::render(&text));
        line.push('\r');
        let column = prompt.chars().count() + self.cursor;
        if column > 0 {
            line.push_str("\x1b[");
            line.push_str(&column.to_string());
            line.push('C');
        }
        crate::term::write(line.as_bytes())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn replace_range_keeps_the_rest_of_the_line() {
        let mut editor = LineEditor::new("cat al tail", 6);
        editor.replace_range(4, 6, "alpha.txt");
        assert_eq!(editor.text(), "cat alpha.txt tail");
        assert_eq!(editor.cursor(), 13);
    }

    #[test]
    fn replace_range_clamps_out_of_bounds_indices() {
        let mut editor = LineEditor::new("ab", 2);
        editor.replace_range(1, 99, "x");
        assert_eq!(editor.text(), "ax");
        assert_eq!(editor.cursor(), 2);
    }
}
