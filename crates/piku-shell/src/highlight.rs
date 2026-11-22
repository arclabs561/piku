//! Syntax highlighting for the line being typed.
//!
//! `highlight` is pure: it splits a line into styled spans and never touches the
//! terminal, so the rules can be unit-tested. `render` wraps those spans in SGR
//! sequences; escape codes occupy no columns, so the visible text and the cursor
//! column are unchanged.
//!
//! This is presentation only. The line is still handed to `sh -c` verbatim, and
//! nothing here parses, expands, or validates it.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Style {
    Default,
    Command,
    String,
    Variable,
    Operator,
}

impl Style {
    /// SGR parameters for this style, or `None` to leave the terminal default.
    fn sgr(self) -> Option<&'static str> {
        match self {
            Self::Default => None,
            Self::Command => Some("1;32"),
            Self::String => Some("33"),
            Self::Variable => Some("36"),
            Self::Operator => Some("1;35"),
        }
    }
}

/// Splits `line` into spans. Concatenating the span text always reproduces the
/// input exactly; only the accompanying style changes.
pub fn highlight(line: &str) -> Vec<(String, Style)> {
    let chars: Vec<char> = line.chars().collect();
    let mut spans: Vec<(String, Style)> = Vec::new();
    let mut index = 0;
    // A word here starts a simple command rather than an argument.
    let mut expect_command = true;

    while index < chars.len() {
        let character = chars[index];
        if character.is_whitespace() {
            push_char(&mut spans, character, Style::Default);
            index += 1;
            continue;
        }
        if character == '\'' || character == '"' {
            let end = quoted_end(&chars, index, character);
            let text: String = chars[index..end].iter().collect();
            push(&mut spans, &text, Style::String);
            expect_command = false;
            index = end;
            continue;
        }
        if character == '$' {
            let end = variable_end(&chars, index);
            if end > index + 1 {
                let text: String = chars[index..end].iter().collect();
                push(&mut spans, &text, Style::Variable);
                if expect_command {
                    expect_command = false;
                }
            } else {
                push_char(&mut spans, '$', Style::Default);
            }
            index = end;
            continue;
        }
        if is_operator(character) {
            let mut end = index;
            while end < chars.len() && is_operator(chars[end]) {
                end += 1;
            }
            let text: String = chars[index..end].iter().collect();
            push(&mut spans, &text, Style::Operator);
            // `|`, `;`, `&&`, and `||` start a new command; `>` and `<` do not.
            expect_command = !matches!(character, '<' | '>');
            index = end;
            continue;
        }

        let mut end = index;
        while end < chars.len() && !is_word_break(chars[end]) {
            end += 1;
        }
        let text: String = chars[index..end].iter().collect();
        let redirection_target = end < chars.len()
            && matches!(chars[end], '<' | '>')
            && !text.is_empty()
            && text.chars().all(|digit| digit.is_ascii_digit());
        if redirection_target {
            // `2>` and `1<`: the descriptor belongs to the operator.
            push(&mut spans, &text, Style::Operator);
        } else if expect_command {
            push(&mut spans, &text, Style::Command);
            expect_command = false;
        } else {
            push(&mut spans, &text, Style::Default);
        }
        index = end;
    }
    spans
}

/// Renders highlighted spans for the terminal. Non-printing bytes are added,
/// so the text still occupies the same columns.
pub fn render(line: &str) -> String {
    let mut out = String::with_capacity(line.len() + 32);
    for (text, style) in highlight(line) {
        match style.sgr() {
            Some(sgr) => {
                out.push_str("\x1b[");
                out.push_str(sgr);
                out.push('m');
                out.push_str(&text);
                out.push_str("\x1b[0m");
            }
            None => out.push_str(&text),
        }
    }
    out
}

fn is_operator(character: char) -> bool {
    matches!(character, '|' | '&' | ';' | '<' | '>')
}

fn is_word_break(character: char) -> bool {
    character.is_whitespace() || is_operator(character) || matches!(character, '\'' | '"' | '$')
}

/// One past a quoted span. An unterminated quote runs to the end of the line.
fn quoted_end(chars: &[char], start: usize, quote: char) -> usize {
    let mut index = start + 1;
    while index < chars.len() {
        if chars[index] == quote {
            return index + 1;
        }
        index += 1;
    }
    chars.len()
}

/// One past `$NAME` or `${NAME}`. A lone `$` is not a variable.
fn variable_end(chars: &[char], start: usize) -> usize {
    let brace = chars.get(start + 1) == Some(&'{');
    let mut index = start + if brace { 2 } else { 1 };
    if brace {
        while index < chars.len() {
            if chars[index] == '}' {
                return index + 1;
            }
            index += 1;
        }
        // Unterminated `${`: style through the end of the line.
        return chars.len();
    }
    while index < chars.len() && (chars[index].is_alphanumeric() || chars[index] == '_') {
        index += 1;
    }
    index
}

fn push(spans: &mut Vec<(String, Style)>, text: &str, style: Style) {
    if text.is_empty() {
        return;
    }
    spans.push((text.to_string(), style));
}

fn push_char(spans: &mut Vec<(String, Style)>, character: char, style: Style) {
    let mut buffer = [0u8; 4];
    push(spans, character.encode_utf8(&mut buffer), style);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Asserts the exact spans. Every expected text is compared as an owned
    /// string so the helper does not borrow a temporary.
    fn assert_styles(line: &str, expected: &[(&str, Style)]) {
        let expected: Vec<(String, Style)> = expected
            .iter()
            .map(|(text, style)| ((*text).to_string(), *style))
            .collect();
        assert_eq!(highlight(line), expected);
    }

    /// Spans must reproduce the line exactly; visible width depends on it.
    fn assert_text_preserved(line: &str) {
        let joined: String = highlight(line).into_iter().map(|(text, _)| text).collect();
        assert_eq!(joined, line);
    }

    #[test]
    fn first_word_is_the_command() {
        assert_styles(
            "echo hi",
            &[
                ("echo", Style::Command),
                (" ", Style::Default),
                ("hi", Style::Default),
            ],
        );
    }

    #[test]
    fn word_after_a_pipe_is_a_command() {
        assert_styles(
            "echo hi | grep x",
            &[
                ("echo", Style::Command),
                (" ", Style::Default),
                ("hi", Style::Default),
                (" ", Style::Default),
                ("|", Style::Operator),
                (" ", Style::Default),
                ("grep", Style::Command),
                (" ", Style::Default),
                ("x", Style::Default),
            ],
        );
    }

    #[test]
    fn words_after_and_and_semicolon_are_commands() {
        assert_styles(
            "a && b; c",
            &[
                ("a", Style::Command),
                (" ", Style::Default),
                ("&&", Style::Operator),
                (" ", Style::Default),
                ("b", Style::Command),
                (";", Style::Operator),
                (" ", Style::Default),
                ("c", Style::Command),
            ],
        );
    }

    #[test]
    fn quoted_spans_are_strings() {
        assert_styles(
            "echo 'a b' \"c\"",
            &[
                ("echo", Style::Command),
                (" ", Style::Default),
                ("'a b'", Style::String),
                (" ", Style::Default),
                ("\"c\"", Style::String),
            ],
        );
    }

    #[test]
    fn unterminated_quote_runs_to_the_end() {
        assert_styles(
            "echo 'open",
            &[
                ("echo", Style::Command),
                (" ", Style::Default),
                ("'open", Style::String),
            ],
        );
    }

    #[test]
    fn variables_are_marked() {
        assert_styles(
            "echo $HOME ${PATH}x $",
            &[
                ("echo", Style::Command),
                (" ", Style::Default),
                ("$HOME", Style::Variable),
                (" ", Style::Default),
                ("${PATH}", Style::Variable),
                ("x", Style::Default),
                (" ", Style::Default),
                ("$", Style::Default),
            ],
        );
    }

    #[test]
    fn redirections_are_operators() {
        assert_styles(
            "echo hi >out 2>&1",
            &[
                ("echo", Style::Command),
                (" ", Style::Default),
                ("hi", Style::Default),
                (" ", Style::Default),
                (">", Style::Operator),
                ("out", Style::Default),
                (" ", Style::Default),
                ("2", Style::Operator),
                (">&", Style::Operator),
                ("1", Style::Default),
            ],
        );
    }

    #[test]
    fn text_is_preserved_so_width_does_not_change() {
        for line in [
            "",
            "echo hi",
            "  ls -la ~/Developer | grep wish  ",
            "cat 'a b' \"$HOME\" $VAR ${OTHER} > /dev/null 2>&1",
            "printf '%s\\n' αβγ",
            "echo 'unterminated",
            "&& <<< ''",
        ] {
            assert_text_preserved(line);
        }
    }

    #[test]
    fn rendered_output_keeps_the_visible_text() {
        let line = "echo 'hi' $HOME > out";
        let rendered = render(line);
        let stripped = strip_sgr(&rendered);
        assert_eq!(stripped, line);
        assert!(rendered.contains("\x1b[1;32mecho\x1b[0m"));
    }

    fn strip_sgr(input: &str) -> String {
        let mut out = String::new();
        let mut chars = input.chars();
        while let Some(character) = chars.next() {
            if character == '\x1b' {
                // Skip `[` parameters and the final `m`.
                for next in chars.by_ref() {
                    if next == 'm' {
                        break;
                    }
                }
            } else {
                out.push(character);
            }
        }
        out
    }
}
