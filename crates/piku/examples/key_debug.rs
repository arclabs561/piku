/// Debug what key events crossterm sees for each keypress.
/// Run: cargo run -p piku --example key_debug
/// Press keys to see their crossterm representation. Ctrl+C to exit.
use crossterm::event::{
    self, Event, KeyCode, KeyEvent, KeyModifiers, KeyboardEnhancementFlags,
    PopKeyboardEnhancementFlags, PushKeyboardEnhancementFlags,
};
use crossterm::execute;
use crossterm::terminal;
use std::io::stdout;

fn main() -> std::io::Result<()> {
    println!("Key event debugger. Press keys to see crossterm events.");
    println!("Try: Enter, Shift+Enter, Ctrl+J, Ctrl+A, Alt+B, End, Home");
    println!("Press Ctrl+C to exit.\n");

    terminal::enable_raw_mode()?;

    // Enable kitty keyboard protocol for shift+enter detection
    let enhanced = execute!(
        stdout(),
        PushKeyboardEnhancementFlags(KeyboardEnhancementFlags::DISAMBIGUATE_ESCAPE_CODES)
    )
    .is_ok();
    print!("\r\x1b[2K");
    println!(
        "Keyboard enhancement: {}\r",
        if enhanced { "enabled" } else { "NOT supported" }
    );

    loop {
        match event::read()? {
            Event::Key(KeyEvent {
                code,
                modifiers,
                kind,
                state,
            }) => {
                print!("\r\x1b[2K");

                // Human-readable description
                let desc = match (&code, &modifiers) {
                    (KeyCode::Enter, m) if m.contains(KeyModifiers::SHIFT) => {
                        "SHIFT+ENTER (newline) "
                    }
                    (KeyCode::Enter, _) => "ENTER (submit) ",
                    (KeyCode::Char('j'), m) if m.contains(KeyModifiers::CONTROL) => {
                        "CTRL+J (newline) "
                    }
                    _ => "",
                };

                println!(
                    "{desc}code={code:?}  mod={modifiers:?}  kind={kind:?}  state={state:?}\r"
                );

                if modifiers.contains(KeyModifiers::CONTROL) && code == KeyCode::Char('c') {
                    break;
                }
            }
            Event::Resize(w, h) => {
                print!("\r\x1b[2K");
                println!("Resize: {w}x{h}\r");
            }
            _ => {}
        }
    }

    let _ = execute!(stdout(), PopKeyboardEnhancementFlags);
    terminal::disable_raw_mode()?;
    println!("\r\nDone.");
    Ok(())
}
