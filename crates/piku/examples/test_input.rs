/// Standalone test for the LineEditor outside the TUI layout.
/// Run with: cargo run -p piku --example test_input
use piku::input_helper::{LineEditor, ReadOutcome};

fn main() {
    println!("Line editor test. Type text, press Enter to submit.");
    println!("Shift+Enter or Ctrl+J should insert a newline.");
    println!("Ctrl+D or Ctrl+C on empty to exit.\n");

    let mut editor = LineEditor::new("\x1b[34m›\x1b[0m ");

    loop {
        match editor.read_line() {
            Ok(ReadOutcome::Submit(text)) => {
                editor.push_history(&text);
                let lines = text.lines().count();
                println!(
                    "\x1b[32mSubmitted\x1b[0m ({lines} line{}): {:?}\n",
                    if lines == 1 { "" } else { "s" },
                    text
                );
            }
            Ok(ReadOutcome::Cancel) => {
                println!("\x1b[33mCancelled\x1b[0m\n");
            }
            Ok(ReadOutcome::Exit) => {
                println!("\x1b[2mExit\x1b[0m");
                break;
            }
            Err(e) => {
                eprintln!("\x1b[31mError:\x1b[0m {e}");
                break;
            }
        }
    }
}
