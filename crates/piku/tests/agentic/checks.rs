use super::types::*;

/// Deterministic checks on screen snapshots — no LLM needed.
pub fn deterministic_checks(
    before: &ScreenSnapshot,
    after: &ScreenSnapshot,
    action: &Action,
) -> Vec<Finding> {
    let mut findings = Vec::new();

    // 1. Cursor visibility
    if !after.cursor_visible {
        findings.push(Finding {
            severity: Severity::Major,
            description: "cursor hidden after action".to_string(),
            expected: "cursor should be visible after every action".to_string(),
            actual: format!(
                "cursor at ({}, {}), hidden=true",
                after.cursor.0, after.cursor.1
            ),
        });
    }

    // 2. Prompt glyph presence (only check after submit + response)
    if matches!(action, Action::Submit(_)) && after.is_ready() {
        let input = after.input_row().trim_start();
        let has_glyph = input.starts_with('\u{276F}')
            || input.starts_with('>')
            || input.starts_with('!')
            || input.contains("Send a message");
        if !has_glyph {
            findings.push(Finding {
                severity: Severity::Major,
                description: "prompt glyph missing from input row".to_string(),
                expected: "input row should start with \u{276F}, >, or !".to_string(),
                actual: format!("input row: {:?}", &input[..input.len().min(40)]),
            });
        }
    }

    // 3. Footer presence (check reverse-video on footer row)
    if after.styled_rows.len() >= 2 {
        let footer = &after.styled_rows[1];
        let has_inverse = footer
            .cells
            .iter()
            .any(|c| c.inverse && !c.ch.trim().is_empty());
        if !footer.text.trim().is_empty() && !has_inverse {
            findings.push(Finding {
                severity: Severity::Minor,
                description: "footer row not rendered in reverse video".to_string(),
                expected: "footer should use reverse video for status bar".to_string(),
                actual: format!(
                    "footer text: {:?}",
                    &footer.text[..footer.text.len().min(60)]
                ),
            });
        }
    }

    // 4. Echo presence after submit
    if let Action::Submit(text) = action {
        if !text.is_empty() && after.is_ready() {
            let scroll_rows = &after.rows[..after.rows.len().saturating_sub(2)];
            let echo_found = scroll_rows.iter().any(|r| r.contains(text.as_str()));
            if echo_found {
                findings.push(Finding {
                    severity: Severity::Info,
                    description: "user message echoed in scroll zone".to_string(),
                    expected: String::new(),
                    actual: format!("found echo of: {:?}", &text[..text.len().min(40)]),
                });
            }
        }
    }

    // 5. Screen corruption: control chars
    for (i, row) in after.rows.iter().enumerate() {
        if row
            .chars()
            .any(|c| c.is_control() && c != '\n' && c != '\t')
        {
            findings.push(Finding {
                severity: Severity::Major,
                description: format!("control characters in rendered row {i}"),
                expected: "rendered rows should contain only printable text".to_string(),
                actual: format!("row {i}: {:?}", &row[..row.len().min(60)]),
            });
        }
    }

    // 6. Tab completion change detection
    if matches!(action, Action::Key(SpecialKey::Tab)) {
        let before_input = before.input_row();
        let after_input = after.input_row();
        if before_input != after_input {
            findings.push(Finding {
                severity: Severity::Info,
                description: "tab completion changed input".to_string(),
                expected: String::new(),
                actual: format!(
                    "{:?} -> {:?}",
                    &before_input[..before_input.len().min(40)],
                    &after_input[..after_input.len().min(40)]
                ),
            });
        } else {
            findings.push(Finding {
                severity: Severity::Info,
                description: "tab had no effect on input".to_string(),
                expected: String::new(),
                actual: format!(
                    "input unchanged: {:?}",
                    &after_input[..after_input.len().min(40)]
                ),
            });
        }
    }

    findings
}
