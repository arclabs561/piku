/// VT100-based rendering tests.
///
/// These write ANSI output to a vt100::Parser (in-memory terminal emulator)
/// and read back the screen contents. This tests the full ANSI pipeline
/// as a real terminal would see it -- no strip_ansi heuristics needed.
///
/// Pattern from Codex: CrosstermBackend<vt100::Parser> → screen().contents()
use piku::markdown::StreamingMarkdown;

/// Create a VT100 parser, write text to it, return screen contents.
fn render_to_vt100(ansi_text: &str) -> String {
    let mut parser = vt100::Parser::new(50, 120, 0);
    parser.process(ansi_text.as_bytes());
    // Get screen contents (strips ANSI, gives plain text with positioning)
    parser.screen().contents()
}

/// Same but returns the specific row contents (trimmed).
fn render_row(ansi_text: &str, row: u16) -> String {
    let mut parser = vt100::Parser::new(50, 120, 0);
    parser.process(ansi_text.as_bytes());
    let screen = parser.screen();
    let mut line = String::new();
    for col in 0..120 {
        let cell = screen.cell(row, col);
        if let Some(cell) = cell {
            line.push_str(&cell.contents());
        }
    }
    line.trim_end().to_string()
}

// ── Markdown through VT100 ─────────────────────────────────────────────────

#[test]
fn heading_renders_as_text() {
    let mut md = StreamingMarkdown::new_stdout();
    let out = md.push("## Hello World\n");
    let screen = render_to_vt100(&out);
    assert!(
        screen.contains("## Hello World"),
        "heading text on screen: {screen:?}"
    );
}

#[test]
fn code_block_has_frame_chars() {
    let mut md = StreamingMarkdown::new_stdout();
    let out = md.push("```rust\nlet x = 1;\n```\n");
    let screen = render_to_vt100(&out);
    assert!(screen.contains("rust"), "language label: {screen:?}");
    assert!(screen.contains("let x"), "code content: {screen:?}");
    // Frame characters should survive ANSI processing
    assert!(
        screen.contains('╭') && screen.contains('╰'),
        "frame chars: {screen:?}"
    );
}

#[test]
fn inline_formatting_shows_plain_text() {
    let mut md = StreamingMarkdown::new_stdout();
    let out = md.push("This is **bold** and *italic* and `code`\n");
    let screen = render_to_vt100(&out);
    assert!(screen.contains("bold"), "bold: {screen:?}");
    assert!(screen.contains("italic"), "italic: {screen:?}");
    assert!(screen.contains("`code`"), "code: {screen:?}");
}

#[test]
fn list_bullets_render() {
    let mut md = StreamingMarkdown::new_stdout();
    let out = md.push("- first\n- second\n");
    let screen = render_to_vt100(&out);
    assert!(screen.contains("first"), "list item 1: {screen:?}");
    assert!(screen.contains("second"), "list item 2: {screen:?}");
    assert!(screen.contains('\u{2022}'), "bullet char: {screen:?}");
}

#[test]
fn blockquote_renders_with_bar() {
    let mut md = StreamingMarkdown::new_stdout();
    let out = md.push("> quoted text\n");
    let screen = render_to_vt100(&out);
    assert!(screen.contains("quoted text"), "quote content: {screen:?}");
    assert!(screen.contains('\u{258e}'), "quarter block bar: {screen:?}");
}

#[test]
fn horizontal_rule_renders() {
    let mut md = StreamingMarkdown::new_stdout();
    let out = md.push("---\n");
    let screen = render_to_vt100(&out);
    assert!(screen.contains("───"), "hr dashes: {screen:?}");
}

// ── Tool formatting through VT100 ──────────────────────────────────────────

#[test]
fn tool_start_renders_dot_and_name() {
    // Simulate what TuiSink.on_tool_start would emit
    let ansi = "\x1b[33m⏺\x1b[0m \x1b[1mBash\x1b[0m\x1b[2m(ls -la)\x1b[0m\n";
    let screen = render_to_vt100(ansi);
    assert!(screen.contains('⏺'), "dot: {screen:?}");
    assert!(screen.contains("Bash"), "tool name: {screen:?}");
    assert!(screen.contains("ls -la"), "args: {screen:?}");
}

#[test]
fn tool_result_connector_renders() {
    let ansi = "\x1b[2m⎿\x1b[0m output line\n";
    let screen = render_to_vt100(ansi);
    assert!(screen.contains('⎿'), "connector: {screen:?}");
    assert!(screen.contains("output line"), "content: {screen:?}");
}

// ── Footer bar through VT100 ───────────────────────────────────────────────

#[test]
fn footer_reverse_video() {
    // Simulate footer rendering with reverse video
    let ansi = "\x1b[7m openrouter · claude-sonnet │ /help \x1b[0m";
    let screen = render_to_vt100(ansi);
    assert!(
        screen.contains("openrouter"),
        "provider in footer: {screen:?}"
    );
    assert!(screen.contains("/help"), "hint in footer: {screen:?}");
}

// ── Thinking indicator ──────────────────────────────────────────────────────

#[test]
fn thinking_indicator_renders() {
    let ansi = "\x1b[2m❯ ✶ thinking… (3s)\x1b[0m";
    let screen = render_to_vt100(ansi);
    assert!(screen.contains("thinking"), "thinking text: {screen:?}");
    assert!(screen.contains("3s"), "elapsed: {screen:?}");
}

// ── Multi-element composition ───────────────────────────────────────────────

#[test]
fn full_turn_output() {
    // Simulate a complete agent turn: text + tool + result + summary
    let mut md = StreamingMarkdown::new_stdout();
    let mut output = String::new();

    // Assistant text
    output.push_str(&md.push("Let me check that file.\n"));
    output.push_str(&md.flush());

    // Tool call
    output.push_str("\x1b[33m⏺\x1b[0m \x1b[1mRead\x1b[0m\x1b[2m(src/main.rs)\x1b[0m\n");

    // Tool result
    output.push_str("\x1b[2m⎿\x1b[0m \x1b[2mfn main() {\x1b[0m\n");

    // Status dot
    output.push_str("\x1b[32m⏺\x1b[0m\n");

    // More text
    let mut md2 = StreamingMarkdown::new_stdout();
    output.push_str(&md2.push("The file contains the entry point.\n"));
    output.push_str(&md2.flush());

    // Summary
    output.push_str("\x1b[2m[1 iter · 1.2k↑ 500↓]\x1b[0m\n");

    let screen = render_to_vt100(&output);
    assert!(screen.contains("Let me check"), "assistant text");
    assert!(screen.contains("Read"), "tool name");
    assert!(screen.contains("fn main"), "tool result");
    assert!(screen.contains("entry point"), "follow-up text");
    assert!(screen.contains("1.2k"), "summary");
}
