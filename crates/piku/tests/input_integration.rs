/// Realistic multi-turn input integration tests.
///
/// These simulate the sequence of operations a human would perform
/// during a real piku session: type, edit, undo, stash, navigate
/// history, accept ghost text, paste, etc.
///
/// Tests use only the public API (handle_key, as_str, is_empty, etc.)
/// without accessing private fields.
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use piku::input_helper::{Action, InputBuffer, LineEditor};

fn key(code: KeyCode) -> KeyEvent {
    KeyEvent::new(code, KeyModifiers::NONE)
}

fn ctrl(ch: char) -> KeyEvent {
    KeyEvent::new(KeyCode::Char(ch), KeyModifiers::CONTROL)
}

fn char_key(ch: char) -> KeyEvent {
    KeyEvent::new(KeyCode::Char(ch), KeyModifiers::NONE)
}

/// Type a string into the editor via key events.
fn type_str(editor: &mut LineEditor, input: &mut InputBuffer, s: &str) {
    for ch in s.chars() {
        editor.handle_key(char_key(ch), input);
    }
}

/// Scenario 1: Type, clear with Ctrl+U, retype, submit, recall from history.
#[test]
fn type_clear_retype_recall() {
    let mut editor = LineEditor::new("> ");
    let mut input = InputBuffer::new();

    type_str(&mut editor, &mut input, "explain the parser");
    assert_eq!(input.as_str(), "explain the parser");

    editor.handle_key(ctrl('u'), &mut input);
    assert!(input.is_empty());

    type_str(&mut editor, &mut input, "fix the bug");
    let action = editor.handle_key(key(KeyCode::Enter), &mut input);
    assert_eq!(action, Action::Submit);

    editor.push_history("fix the bug");
    let mut input = InputBuffer::new();
    editor.handle_key(key(KeyCode::Up), &mut input);
    assert_eq!(input.as_str(), "fix the bug");
}

/// Scenario 2: Multiline editing with Ctrl+J, then Up/Down within text.
#[test]
fn multiline_editing() {
    let mut editor = LineEditor::new("> ");
    let mut input = InputBuffer::new();

    type_str(&mut editor, &mut input, "line one");
    editor.handle_key(ctrl('j'), &mut input);
    type_str(&mut editor, &mut input, "line two");
    editor.handle_key(ctrl('j'), &mut input);
    type_str(&mut editor, &mut input, "line three");

    assert_eq!(input.as_str(), "line one\nline two\nline three");

    // Up from line 3 → should NOT trigger history (multiline aware)
    editor.handle_key(key(KeyCode::Up), &mut input);
    // After Up, text should be unchanged (cursor moved, not history replaced)
    assert_eq!(input.as_str(), "line one\nline two\nline three");
}

/// Scenario 3: Stash (Ctrl+S) — save draft, quick query, restore.
#[test]
fn stash_workflow() {
    let mut editor = LineEditor::new("> ");
    let mut input = InputBuffer::new();

    type_str(&mut editor, &mut input, "long draft message");
    editor.handle_key(ctrl('s'), &mut input);
    assert!(input.is_empty(), "stash should clear input");

    type_str(&mut editor, &mut input, "quick question");
    let action = editor.handle_key(key(KeyCode::Enter), &mut input);
    assert_eq!(action, Action::Submit);

    let mut input = InputBuffer::new();
    editor.handle_key(ctrl('s'), &mut input);
    assert_eq!(input.as_str(), "long draft message", "stash should restore");
}

/// Scenario 4: Kill, yank, undo flow.
#[test]
fn kill_yank_undo() {
    let mut editor = LineEditor::new("> ");
    let mut input = InputBuffer::new();

    type_str(&mut editor, &mut input, "hello beautiful world");

    // Kill last word
    editor.handle_key(ctrl('w'), &mut input);
    assert_eq!(input.as_str(), "hello beautiful ");

    // Kill again
    editor.handle_key(ctrl('w'), &mut input);
    assert_eq!(input.as_str(), "hello ");

    // Undo should restore
    editor.handle_key(ctrl('z'), &mut input);
    assert!(
        input.as_str().len() > "hello ".len(),
        "undo should restore: {}",
        input.as_str()
    );

    // Yank at end
    editor.handle_key(ctrl('e'), &mut input);
    editor.handle_key(ctrl('y'), &mut input);
    // Kill buffer has the last killed text
    assert!(input.as_str().len() > 10, "yank should add text");
}

/// Scenario 5: Ghost text acceptance with Right arrow.
#[test]
fn ghost_text_accept() {
    let mut editor = LineEditor::new("> ");
    editor.push_history("explain src/main.rs in detail");

    let mut input = InputBuffer::new();
    type_str(&mut editor, &mut input, "explain");

    // Right arrow at end accepts ghost text
    editor.handle_key(key(KeyCode::Right), &mut input);
    assert_eq!(input.as_str(), "explain src/main.rs in detail");
}

/// Scenario 6: Paste with ANSI and CRLF normalization.
#[test]
fn paste_normalization() {
    let mut input = InputBuffer::new();
    input.insert_str("\x1b[31mhello\x1b[0m\r\nworld\ttabbed");
    assert_eq!(input.as_str(), "hello\nworld    tabbed");
}

/// Scenario 7: Tab completion for slash commands.
#[test]
fn tab_completion() {
    let mut editor = LineEditor::new("> ");
    let mut input = InputBuffer::new();

    type_str(&mut editor, &mut input, "/he");
    editor.handle_key(key(KeyCode::Tab), &mut input);
    assert_eq!(input.as_str(), "/help");
}

/// Scenario 8: Esc clears, history still works after.
#[test]
fn esc_then_history() {
    let mut editor = LineEditor::new("> ");
    editor.push_history("old command");

    let mut input = InputBuffer::new();
    type_str(&mut editor, &mut input, "draft");
    editor.handle_key(key(KeyCode::Esc), &mut input);
    assert!(input.is_empty());

    editor.handle_key(key(KeyCode::Up), &mut input);
    assert_eq!(input.as_str(), "old command");
}

/// Scenario 9: Ctrl+C on non-empty clears, on empty exits.
#[test]
fn ctrl_c_behavior() {
    let mut editor = LineEditor::new("> ");
    let mut input = InputBuffer::new();

    type_str(&mut editor, &mut input, "text");
    let action = editor.handle_key(ctrl('c'), &mut input);
    assert_eq!(action, Action::Cancel);
    assert!(input.is_empty());

    let action = editor.handle_key(ctrl('c'), &mut input);
    assert_eq!(action, Action::Exit);
}

/// Scenario 10: Ctrl+D on non-empty deletes char, on empty exits.
#[test]
fn ctrl_d_behavior() {
    let mut editor = LineEditor::new("> ");
    let mut input = InputBuffer::new();

    type_str(&mut editor, &mut input, "ab");
    editor.handle_key(ctrl('a'), &mut input); // go home
    let action = editor.handle_key(ctrl('d'), &mut input);
    assert_eq!(action, Action::Continue);
    assert_eq!(input.as_str(), "b");

    input.clear();
    let action = editor.handle_key(ctrl('d'), &mut input);
    assert_eq!(action, Action::Exit);
}

/// Scenario 11: Full readline sequence: type, word-back, kill-word, yank elsewhere.
#[test]
fn readline_sequence() {
    let mut editor = LineEditor::new("> ");
    let mut input = InputBuffer::new();

    type_str(&mut editor, &mut input, "the quick brown fox");

    // Alt+B: move back one word (to "fox")
    editor.handle_key(
        KeyEvent::new(KeyCode::Char('b'), KeyModifiers::ALT),
        &mut input,
    );
    // Alt+B again: to "brown"
    editor.handle_key(
        KeyEvent::new(KeyCode::Char('b'), KeyModifiers::ALT),
        &mut input,
    );

    // Ctrl+K: kill to end (kills "brown fox")
    editor.handle_key(ctrl('k'), &mut input);
    assert_eq!(input.as_str(), "the quick ");

    // Ctrl+A: go home
    editor.handle_key(ctrl('a'), &mut input);

    // Ctrl+Y: yank "brown fox" at beginning
    editor.handle_key(ctrl('y'), &mut input);
    assert!(
        input.as_str().starts_with("brown fox"),
        "yanked at start: {}",
        input.as_str()
    );
}
