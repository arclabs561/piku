/// Functional rendering tests.
///
/// These test the actual output of the markdown renderer and tool input
/// formatter with realistic inputs, not source-code pattern matching.

// ── Markdown rendering ──────────────────────────────────────────────────────

use piku::markdown::StreamingMarkdown;

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

/// Realistic assistant response with multiple markdown features.
#[test]
fn full_assistant_response() {
    let mut md = StreamingMarkdown::new_stdout();
    let input = "\
## Summary

Here's what I found:

- The **main function** calls `parse_args()`
- It handles *three modes*: single-shot, REPL, and resume

```rust
fn main() {
    let args = parse_args();
    match args {
        Action::SingleShot { .. } => run_single_shot(),
        Action::Repl { .. } => run_repl(),
    }
}
```

> Note: the resume path loads from disk first.

1. Parse CLI args
2. Resolve provider
3. Enter main loop

---

That's the overview.
";
    let mut all = String::new();
    // Simulate token-by-token streaming (chunks of ~20 chars)
    for chunk in input.as_bytes().chunks(20) {
        let s = std::str::from_utf8(chunk).unwrap();
        all.push_str(&md.push(s));
    }
    all.push_str(&md.flush());

    let plain = strip_ansi(&all);

    // Headings
    assert!(plain.contains("## Summary"), "heading");
    // Bold
    assert!(plain.contains("main function"), "bold text");
    // Inline code
    assert!(plain.contains("`parse_args()`"), "inline code");
    // List items (bullet character)
    assert!(plain.contains("\u{2022}"), "bullet character");
    // Code block frame
    assert!(plain.contains("rust"), "code block language label");
    assert!(plain.contains("fn main"), "code block content");
    assert!(plain.contains('╭'), "code block top frame");
    assert!(plain.contains('╰'), "code block bottom frame");
    // Block quote
    assert!(plain.contains("Note:"), "block quote content");
    // Ordered list
    assert!(plain.contains("1."), "ordered list");
    // Horizontal rule
    assert!(plain.contains("───"), "horizontal rule");
    // Normal text
    assert!(plain.contains("That's the overview."), "trailing text");
}

/// Streaming: code block split across many tiny chunks.
#[test]
fn code_block_streamed_byte_by_byte() {
    let mut md = StreamingMarkdown::new_stdout();
    let input = "```python\ndef hello():\n    print('hi')\n```\n";
    let mut all = String::new();
    for ch in input.chars() {
        all.push_str(&md.push(&ch.to_string()));
    }
    all.push_str(&md.flush());

    let plain = strip_ansi(&all);
    assert!(plain.contains("python"));
    assert!(plain.contains("def hello"));
    assert!(plain.contains("print"));
    assert_eq!(plain.matches('╭').count(), 1);
    assert_eq!(plain.matches('╰').count(), 1);
}

/// Multiple code blocks with text between them.
#[test]
fn interleaved_code_and_text() {
    let mut md = StreamingMarkdown::new_stdout();
    let input = "First block:\n\n```sh\nls -la\n```\n\nSecond block:\n\n```rust\nlet x = 1;\n```\n\nDone.\n";
    let mut all = String::new();
    all.push_str(&md.push(input));
    all.push_str(&md.flush());

    let plain = strip_ansi(&all);
    assert!(plain.contains("First block:"));
    assert!(plain.contains("ls -la"));
    assert!(plain.contains("Second block:"));
    assert!(plain.contains("let x"));
    assert!(plain.contains("Done."));
    assert_eq!(plain.matches('╭').count(), 2);
}

/// TUI mode uses \r\n, stdout mode uses \n.
#[test]
fn eol_modes_consistent() {
    let input = "line one\nline two\n";

    let mut tui = StreamingMarkdown::new();
    let tui_out = tui.push(input);
    assert!(tui_out.contains("\r\n"), "TUI mode must use \\r\\n");

    let mut stdout = StreamingMarkdown::new_stdout();
    let stdout_out = stdout.push(input);
    assert!(!stdout_out.contains("\r\n"), "stdout mode must not use \\r\\n");
    assert!(stdout_out.contains('\n'), "stdout mode must use \\n");
}

// ── Tool input formatting ───────────────────────────────────────────────────

#[test]
fn bash_multiline_command() {
    let input = serde_json::json!({
        "command": "cd /tmp && \\\n  ls -la && \\\n  echo done"
    });
    let result = piku::format_tool_input("bash", &input);
    assert!(result.contains("(+2 lines)"), "multiline bash: {result}");
}

#[test]
fn bash_single_line_command() {
    let input = serde_json::json!({ "command": "cargo test --lib" });
    let result = piku::format_tool_input("bash", &input);
    assert_eq!(result, "cargo test --lib");
}

#[test]
fn bash_long_command_truncated() {
    let long_cmd = "a".repeat(100);
    let input = serde_json::json!({ "command": long_cmd });
    let result = piku::format_tool_input("bash", &input);
    assert!(result.ends_with('…'), "should be truncated: {result}");
    assert!(result.len() < 80, "should be shorter than original");
}

#[test]
fn read_file_with_line_range() {
    let input = serde_json::json!({
        "path": "/Users/arc/Documents/dev/piku/crates/piku/src/tui_repl.rs",
        "start_line": 100,
        "end_line": 200,
    });
    let result = piku::format_tool_input("read_file", &input);
    // Long path should be shortened
    assert!(result.contains("…/"), "long path shortened: {result}");
    assert!(result.contains(":100-200"), "line range: {result}");
}

#[test]
fn read_file_short_path() {
    let input = serde_json::json!({ "path": "src/main.rs" });
    let result = piku::format_tool_input("read_file", &input);
    assert_eq!(result, "src/main.rs");
}

#[test]
fn grep_with_path() {
    let input = serde_json::json!({
        "pattern": "fn main",
        "path": "/Users/arc/Documents/dev/piku/crates/piku/src/"
    });
    let result = piku::format_tool_input("grep", &input);
    assert!(result.contains("fn main"), "pattern present: {result}");
    assert!(result.contains("…/"), "long path shortened: {result}");
}

#[test]
fn unknown_tool_shows_first_string_arg() {
    let input = serde_json::json!({ "query": "hello world", "count": 5 });
    let result = piku::format_tool_input("custom_tool", &input);
    assert!(result.contains("hello world"), "first string arg: {result}");
}

#[test]
fn empty_tool_input() {
    let input = serde_json::json!({});
    let result = piku::format_tool_input("bash", &input);
    assert!(result.is_empty(), "empty input: {result}");
}
