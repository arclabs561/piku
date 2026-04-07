/// Tests for TUI formatting: tool input, path shortening, visible_width, markdown edge cases.
/// Footer/tool-result tests are unit tests in tui_repl.rs (need pub(crate) access).

// ── fmt_duration ────────────────────────────────────────────────────────────

#[test]
fn duration_seconds() {
    assert_eq!(piku::fmt_duration(0), "0s");
    assert_eq!(piku::fmt_duration(45), "45s");
}

#[test]
fn duration_minutes() {
    assert_eq!(piku::fmt_duration(120), "2m");
    assert_eq!(piku::fmt_duration(150), "2m 30s");
}

#[test]
fn duration_hours() {
    assert_eq!(piku::fmt_duration(3661), "1h 01m 01s");
}

// ── try_pretty_json ─────────────────────────────────────────────────────────

#[test]
fn pretty_json_object() {
    let result = piku::try_pretty_json(r#"{"a":1,"b":"hello"}"#);
    assert!(result.contains("  \"a\""), "should be indented: {result}");
}

#[test]
fn pretty_json_not_json() {
    assert_eq!(piku::try_pretty_json("not json"), "not json");
}

#[test]
fn pretty_json_too_large() {
    let big = format!("{{\"x\":\"{}\"}}", "a".repeat(11000));
    assert_eq!(piku::try_pretty_json(&big), big);
}
mod test_helpers;
use test_helpers::strip_ansi;

// ── format_tool_input ───────────────────────────────────────────────────────

#[test]
fn tool_input_bash_empty() {
    let input = serde_json::json!({});
    assert!(piku::format_tool_input("bash", &input).is_empty());
}

#[test]
fn tool_input_bash_short() {
    let input = serde_json::json!({ "command": "ls -la" });
    assert_eq!(piku::format_tool_input("bash", &input), "ls -la");
}

#[test]
fn tool_input_bash_multiline_shows_count() {
    let input = serde_json::json!({
        "command": "cd /tmp\nls -la\necho done"
    });
    let result = piku::format_tool_input("bash", &input);
    assert!(
        result.contains("(+2 lines)"),
        "should show extra line count: {result}"
    );
    assert!(
        result.contains("cd /tmp"),
        "should show first line: {result}"
    );
}

#[test]
fn tool_input_bash_very_long_truncated() {
    let cmd = "a".repeat(200);
    let input = serde_json::json!({ "command": cmd });
    let result = piku::format_tool_input("bash", &input);
    assert!(
        result.len() < 80,
        "should be truncated: len={}",
        result.len()
    );
    assert!(result.ends_with('…'));
}

#[test]
fn tool_input_read_file_with_lines() {
    let input = serde_json::json!({
        "path": "src/main.rs",
        "start_line": 10,
        "end_line": 20,
    });
    assert_eq!(
        piku::format_tool_input("read_file", &input),
        "src/main.rs:10-20"
    );
}

#[test]
fn tool_input_read_file_long_path_shortened() {
    let input = serde_json::json!({
        "path": "/Users/arc/Documents/dev/piku/crates/piku/src/tui_repl.rs"
    });
    let result = piku::format_tool_input("read_file", &input);
    assert!(
        result.starts_with("…/"),
        "long path should be shortened: {result}"
    );
    assert!(result.contains("tui_repl.rs"));
}

#[test]
fn tool_input_read_file_short_path_unchanged() {
    let input = serde_json::json!({ "path": "src/lib.rs" });
    assert_eq!(piku::format_tool_input("read_file", &input), "src/lib.rs");
}

#[test]
fn tool_input_grep_pattern_only() {
    let input = serde_json::json!({ "pattern": "fn main" });
    assert_eq!(piku::format_tool_input("grep", &input), "fn main");
}

#[test]
fn tool_input_grep_with_long_path() {
    let input = serde_json::json!({
        "pattern": "TODO",
        "path": "/Users/arc/Documents/dev/piku/crates/"
    });
    let result = piku::format_tool_input("grep", &input);
    assert!(result.contains("TODO"), "pattern: {result}");
    assert!(result.contains("…/"), "path shortened: {result}");
}

#[test]
fn tool_input_unknown_tool_first_string() {
    let input = serde_json::json!({ "count": 5, "query": "hello" });
    let result = piku::format_tool_input("custom", &input);
    assert!(
        result.contains("hello"),
        "should show first string value: {result}"
    );
}

// ── visible_width ───────────────────────────────────────────────────────────

#[test]
fn visible_width_plain() {
    assert_eq!(piku::input_helper::visible_width("hello"), 5);
}

#[test]
fn visible_width_ansi() {
    assert_eq!(piku::input_helper::visible_width("\x1b[31mred\x1b[0m"), 3);
}

#[test]
fn visible_width_prompt() {
    assert_eq!(piku::input_helper::visible_width("\x1b[34m❯\x1b[0m "), 2);
}

#[test]
fn visible_width_empty() {
    assert_eq!(piku::input_helper::visible_width(""), 0);
}

#[test]
fn visible_width_complex_ansi() {
    assert_eq!(
        piku::input_helper::visible_width("\x1b[38;2;100;200;50mtext\x1b[0m"),
        4
    );
}

// ── StreamingMarkdown edge cases ────────────────────────────────────────────

#[test]
fn markdown_fast_path_plain_text() {
    let mut md = piku::markdown::StreamingMarkdown::new_stdout();
    let out = md.push("This is plain text with no special chars.\n");
    let plain = strip_ansi(&out);
    assert_eq!(plain.trim(), "This is plain text with no special chars.");
}

#[test]
fn markdown_fast_path_still_handles_code_blocks() {
    let mut md = piku::markdown::StreamingMarkdown::new_stdout();
    let out = md.push("plain line\n```\ncode\n```\n");
    let plain = strip_ansi(&out);
    assert!(plain.contains("plain line"));
    assert!(plain.contains("code"));
    assert!(plain.contains('╭'));
}

#[test]
fn markdown_flush_adds_eol_to_partial() {
    let mut md = piku::markdown::StreamingMarkdown::new_stdout();
    let _ = md.push("partial");
    let flushed = md.flush();
    assert!(flushed.ends_with('\n'), "flush must add EOL: {flushed:?}");
}

#[test]
fn markdown_blockquote_uses_quarter_block() {
    let mut md = piku::markdown::StreamingMarkdown::new_stdout();
    let out = md.push("> quoted\n");
    let plain = strip_ansi(&out);
    assert!(
        plain.contains('\u{258e}'),
        "should use ▎ (U+258E): {plain:?}"
    );
}
