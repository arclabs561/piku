//! CLI argument parsing.
//!
//! Split into `lib.rs` so integration tests can access these types.

// ---------------------------------------------------------------------------
// CLI action
// ---------------------------------------------------------------------------

pub enum CliAction {
    Version,
    Help,
    /// Interactive REPL — no prompt given.
    Repl {
        model: Option<String>,
        provider_override: Option<String>,
        /// Run the REPL with file-inspection tools only.
        read_only: bool,
    },
    /// Prompt without a prior session.
    SingleShot {
        prompt: String,
        model: Option<String>,
        provider_override: Option<String>,
        /// Run with file-inspection tools only, then exit.
        read_only: bool,
        /// Headless: run the prompt to completion and exit instead of
        /// dropping into the interactive REPL (`-p` / `--print`).
        print: bool,
    },
    /// Resume a previous session by ID, then run prompt.
    Resume {
        session_id: String,
        prompt: Option<String>,
        model: Option<String>,
        provider_override: Option<String>,
        /// Run with file-inspection tools only, then exit.
        read_only: bool,
        /// Headless: run-and-exit, no REPL (`-p` / `--print`).
        print: bool,
    },
    /// Error in argument parsing — surface to user.
    ArgError(String),
}

#[must_use]
pub fn parse_args(args: &[String]) -> CliAction {
    let mut model: Option<String> = None;
    let mut provider_override: Option<String> = None;
    let mut resume_session: Option<String> = None;
    let mut print = false;
    let mut read_only = false;
    let mut rest: Vec<String> = Vec::new();
    let mut i = 0;

    while i < args.len() {
        match args[i].as_str() {
            "--version" | "-V" => return CliAction::Version,
            "--help" | "-h" => return CliAction::Help,
            "--print" | "-p" => {
                print = true;
                i += 1;
            }
            "--read-only" => {
                read_only = true;
                i += 1;
            }
            "--model" => match args.get(i + 1) {
                Some(v) if !v.starts_with('-') => {
                    model = Some(v.clone());
                    i += 2;
                }
                _ => {
                    return CliAction::ArgError(
                        "--model requires a value (e.g. --model=claude-opus-4)".to_string(),
                    )
                }
            },
            flag if flag.starts_with("--model=") => {
                let val = &flag[8..];
                if val.is_empty() {
                    return CliAction::ArgError("--model= requires a non-empty value".to_string());
                }
                model = Some(val.to_string());
                i += 1;
            }
            "--provider" => match args.get(i + 1) {
                Some(v) if !v.starts_with('-') => {
                    provider_override = Some(v.clone());
                    i += 2;
                }
                _ => {
                    return CliAction::ArgError(
                        "--provider requires a value (e.g. --provider=openrouter)".to_string(),
                    )
                }
            },
            flag if flag.starts_with("--provider=") => {
                let val = &flag[11..];
                if val.is_empty() {
                    return CliAction::ArgError(
                        "--provider= requires a non-empty value".to_string(),
                    );
                }
                provider_override = Some(val.to_string());
                i += 1;
            }
            "--resume" => match args.get(i + 1) {
                Some(v) if !v.starts_with('-') => {
                    resume_session = Some(v.clone());
                    i += 2;
                }
                _ => {
                    return CliAction::ArgError(
                        "--resume requires a session ID (e.g. --resume session-123)".to_string(),
                    )
                }
            },
            flag if flag.starts_with("--resume=") => {
                let val = &flag[9..];
                if val.is_empty() {
                    return CliAction::ArgError(
                        "--resume= requires a non-empty session ID".to_string(),
                    );
                }
                resume_session = Some(val.to_string());
                i += 1;
            }
            other => {
                rest.push(other.to_string());
                i += 1;
            }
        }
    }

    let prompt_str = if rest.is_empty() {
        None
    } else {
        Some(rest.join(" "))
    };

    if let Some(session_id) = resume_session {
        CliAction::Resume {
            session_id,
            prompt: prompt_str,
            model,
            provider_override,
            read_only,
            print,
        }
    } else if let Some(prompt) = prompt_str {
        CliAction::SingleShot {
            prompt,
            model,
            provider_override,
            read_only,
            print,
        }
    } else {
        // No prompt given → interactive REPL (`-p` is meaningless without a
        // prompt, so it is silently ignored here).
        CliAction::Repl {
            model,
            provider_override,
            read_only,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(s: &[&str]) -> Vec<String> {
        s.iter().map(|x| (*x).to_string()).collect()
    }

    #[test]
    fn single_shot_defaults_to_interactive() {
        match parse_args(&args(&["explain", "main.rs"])) {
            CliAction::SingleShot { prompt, print, .. } => {
                assert_eq!(prompt, "explain main.rs");
                assert!(!print, "without -p, single-shot drops into the REPL");
            }
            _ => panic!("expected SingleShot"),
        }
    }

    #[test]
    fn print_flag_makes_single_shot_headless() {
        for flag in ["-p", "--print"] {
            match parse_args(&args(&[flag, "explain", "main.rs"])) {
                CliAction::SingleShot { prompt, print, .. } => {
                    assert_eq!(prompt, "explain main.rs");
                    assert!(print, "{flag} should set headless mode");
                }
                _ => panic!("expected SingleShot for {flag}"),
            }
        }
    }

    #[test]
    fn print_flag_threads_through_resume() {
        match parse_args(&args(&["--resume", "sess-1", "-p", "do the thing"])) {
            CliAction::Resume { print, prompt, .. } => {
                assert!(print);
                assert_eq!(prompt.as_deref(), Some("do the thing"));
            }
            _ => panic!("expected Resume"),
        }
    }

    #[test]
    fn read_only_flag_threads_through_single_shot() {
        match parse_args(&args(&["--read-only", "explain", "main.rs"])) {
            CliAction::SingleShot {
                prompt, read_only, ..
            } => {
                assert_eq!(prompt, "explain main.rs");
                assert!(read_only);
            }
            _ => panic!("expected SingleShot"),
        }
    }

    #[test]
    fn read_only_flag_threads_through_resume() {
        match parse_args(&args(&["--resume", "sess-1", "--read-only"])) {
            CliAction::Resume { read_only, .. } => {
                assert!(read_only);
            }
            _ => panic!("expected Resume"),
        }
    }

    #[test]
    fn bare_read_only_with_no_prompt_enters_read_only_repl() {
        match parse_args(&args(&["--read-only"])) {
            CliAction::Repl { read_only, .. } => assert!(read_only),
            _ => panic!("expected Repl"),
        }
    }

    #[test]
    fn bare_print_with_no_prompt_is_interactive_repl() {
        // `-p` alone has no prompt → REPL (headless flag has nothing to run).
        assert!(matches!(parse_args(&args(&["-p"])), CliAction::Repl { .. }));
    }
}
