/// TUI output regression tests.
///
/// These tests verify that the TUI code emits the correct ANSI sequences
/// for the bugs that were fixed:
///   1. User messages must be echoed into the scroll zone before a turn starts
///   2. Cursor must be restored (ESC[?25h) after a turn and on teardown
///   3. Tool output must be dimmed (ESC[2m) but bash output must NOT be dimmed
///
/// We test the source-level logic (not a subprocess) by checking that
/// the relevant ANSI sequences are present in the tui_repl.rs source.
/// This is a canary test — if someone removes the fix, the test fails.
///
/// For full end-to-end TUI testing, see the PIKU_DOGFOOD suite which
/// runs the binary against real scenarios (though it uses single-shot mode).
use std::path::PathBuf;

fn tui_repl_source() -> String {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("src")
        .join("tui_repl.rs");
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("could not read tui_repl.rs: {e}"))
}

// ── Cursor visibility ─────────────────────────────────────────────────────────

/// After an agent turn completes, the cursor must be restored.
/// ESC[?25h = show cursor. This must appear in the turn-complete path,
/// not just in teardown.
#[test]
fn cursor_show_sequence_present_after_turn() {
    let src = tui_repl_source();

    // Find the region between "run_turn(" and "goto(rows, 1)" that follows it
    // (the post-turn cleanup block). The \x1b[?25h must appear there.
    // We look for the string near the "clear input line" comment.
    assert!(
        src.contains("\\x1b[?25h"),
        "tui_repl.rs must contain ESC[?25h (show cursor) — cursor was disappearing after turns"
    );

    // More specific: must be present in the post-turn cleanup block that clears
    // the input line and restores cursor visibility.
    let post_turn_pos = src
        .find("clear it, and restore cursor visibility")
        .expect("post-turn cleanup block must exist in tui_repl.rs");
    let post_turn_section = &src[post_turn_pos..post_turn_pos + 500];
    assert!(
        post_turn_section.contains("?25h"),
        "ESC[?25h must appear within ~500 chars of the post-turn cleanup block.\n\
          Post-turn section:\n{post_turn_section}"
    );
}

/// teardown_layout must restore cursor so the terminal isn't left with
/// an invisible cursor after piku exits.
#[test]
fn cursor_restored_in_teardown() {
    let src = tui_repl_source();
    let teardown_pos = src
        .find("fn teardown_layout(")
        .expect("teardown_layout must exist");
    // Extract function body safely (char-boundary aware)
    let fn_slice = src[teardown_pos..].chars().take(300).collect::<String>();
    assert!(
        fn_slice.contains("?25h"),
        "teardown_layout must contain ESC[?25h to restore cursor on exit.\n\
         teardown_layout body:\n{fn_slice}"
    );
}

/// Ctrl-C / Cancel path must also restore cursor.
#[test]
fn cursor_restored_on_ctrl_c() {
    let src = tui_repl_source();
    // We replaced ReadlineError::Interrupted with ReadOutcome::Cancel
    let cancel_pos = src
        .find("ReadOutcome::Cancel")
        .expect("Cancel arm must exist in tui_repl.rs");
    let arm: String = src[cancel_pos..].chars().take(1500).collect();
    assert!(
        arm.contains("?25h"),
        "Cancel (Ctrl-C) arm must restore cursor with ESC[?25h.\narm:\n{arm}"
    );
}

// ── User message echo ─────────────────────────────────────────────────────────

/// Before run_turn is called, the user's input must be echoed into the scroll zone.
/// We verify that the user-input echo appears in the code BEFORE the run_turn
/// call site, not after. The echo uses a dimmed style with ▸ glyph to
/// distinguish it from the active prompt.
#[test]
fn user_message_echoed_before_run_turn() {
    let src = tui_repl_source();

    // Find the agent-turn block: it starts with the "Agent turn" comment
    let agent_turn_pos = src
        .find("── Agent turn")
        .expect("Agent turn comment must exist in tui_repl.rs");

    let run_turn_pos = src[agent_turn_pos..]
        .find("let result: TurnResult = run_turn")
        .map(|p| agent_turn_pos + p)
        .expect("run_turn call must exist after agent turn comment");

    let block = &src[agent_turn_pos..run_turn_pos];

    // The echo uses a dim style (\x1b[2m) with the ▸ glyph — visually
    // distinct from the active prompt.
    let has_echo = block.contains("display_input") && block.contains("\\x1b[2m");

    assert!(
        has_echo,
        "User message echo (dimmed with display_input) must appear BEFORE run_turn in the agent turn block.\n\
         Agent turn block (first 800 chars):\n{}",
        &src[agent_turn_pos..agent_turn_pos + 800.min(run_turn_pos - agent_turn_pos)]
    );
}

/// The user-echo must include the actual user input text, not a hardcoded placeholder.
#[test]
fn user_message_echo_includes_input_variable() {
    let src = tui_repl_source();

    let agent_turn_pos = src
        .find("── Agent turn")
        .expect("Agent turn comment must exist");
    let run_turn_pos = src[agent_turn_pos..]
        .find("let result: TurnResult = run_turn")
        .map(|p| agent_turn_pos + p)
        .expect("run_turn call must exist");

    let block = &src[agent_turn_pos..run_turn_pos];

    // The echo should reference display_input or full_input — not a hardcoded string
    assert!(
        block.contains("display_input") || block.contains("full_input"),
        "User echo block must reference display_input or full_input (actual user text).\n\
         Block:\n{block}"
    );
}

// ── Tool output styling ───────────────────────────────────────────────────────

/// Tool results (non-bash) must be dimmed so they don't compete with agent prose.
/// bash output must NOT be dimmed — it's primary content the user needs to read.
#[test]
fn bash_output_not_dimmed_in_format_tool_result() {
    let src = tui_repl_source();
    let fmt_fn_pos = src
        .find("fn format_tool_result(")
        .expect("format_tool_result must exist");
    let fn_end = src[fmt_fn_pos..]
        .find("\nfn ")
        .map(|p| fmt_fn_pos + p)
        .unwrap_or(src.len());
    let fn_body = &src[fmt_fn_pos..fn_end];

    // bash arm must NOT start its output lines with \x1b[2m (dim)
    // Find the bash arm
    let bash_arm_start = fn_body.find("\"bash\"").expect("bash arm must exist");
    let bash_arm = &fn_body[bash_arm_start..bash_arm_start + 400];

    // Should NOT contain dim escape on the output lines themselves
    // (error output is ok to be dimmed, but normal bash output should be full brightness)
    // We check that the primary output format string does NOT have [2m before the content
    assert!(
        !bash_arm.contains("\\x1b[2m{l}"),
        "bash output lines must not be dimmed (\\x1b[2m).\nbash arm:\n{bash_arm}"
    );
}
