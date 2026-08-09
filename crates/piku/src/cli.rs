use clap::{Parser, Subcommand, ValueEnum};

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum InspectFormat {
    Text,
    Json,
    Html,
}

#[derive(Debug, Parser)]
#[command(name = "piku", version, about = "terminal AI coding agent")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Option<Commands>,

    #[arg(
        long,
        global = true,
        help = "Override model (default: provider-dependent)"
    )]
    pub model: Option<String>,

    #[arg(
        long,
        global = true,
        help = "Force provider: openrouter | anthropic | groq | ollama | custom"
    )]
    pub provider: Option<String>,

    #[arg(
        long,
        global = true,
        help = "Run with file-inspection tools only, then exit"
    )]
    pub read_only: bool,

    #[arg(
        short = 'p',
        long = "print",
        global = true,
        help = "Headless: run the prompt, print the result, and exit"
    )]
    pub print: bool,

    #[arg(
        long,
        global = true,
        help = "Resume a previous session by ID (partial match ok)"
    )]
    pub resume: Option<String>,

    #[arg(trailing_var_arg = true, allow_hyphen_values = true, hide = true)]
    pub prompt: Vec<String>,
}

#[derive(Debug, Subcommand)]
pub enum Commands {
    #[command(about = "Show provider status")]
    Providers,
    #[command(about = "Inspect a run record")]
    Inspect {
        #[arg(help = "Session ID")]
        session_id: String,
        #[arg(long, help = "JSON output")]
        json: bool,
        #[arg(long, help = "HTML output")]
        html: bool,
    },
    #[command(about = "Conclude a run with a disposition and note")]
    Conclude {
        #[arg(help = "Session ID")]
        session_id: String,
        #[arg(long, required = true, help = "accepted | needs-work | abandoned")]
        status: String,
        #[arg(long, required = true, help = "Reason for the disposition")]
        note: String,
    },
    #[command(about = "Start the run surface web server")]
    Web {
        #[arg(long, default_value = "8080", help = "Port to listen on")]
        port: u16,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_shot_defaults_to_interactive() {
        let cli = Cli::try_parse_from(["piku", "explain", "main.rs"]).unwrap();
        assert!(cli.command.is_none());
        assert_eq!(cli.prompt, vec!["explain", "main.rs"]);
        assert!(!cli.print);
    }

    #[test]
    fn print_flag_makes_single_shot_headless() {
        for flag in ["-p", "--print"] {
            let cli = Cli::try_parse_from(["piku", flag, "explain", "main.rs"]).unwrap();
            assert!(cli.print);
        }
    }

    #[test]
    fn print_flag_threads_through_resume() {
        let cli = Cli::try_parse_from(["piku", "--resume", "sess-1", "-p", "do", "the", "thing"])
            .unwrap();
        assert!(cli.print);
        assert_eq!(cli.resume.as_deref(), Some("sess-1"));
        assert_eq!(cli.prompt, vec!["do", "the", "thing"]);
    }

    #[test]
    fn read_only_flag_threads_through_single_shot() {
        let cli = Cli::try_parse_from(["piku", "--read-only", "explain", "main.rs"]).unwrap();
        assert!(cli.read_only);
        assert_eq!(cli.prompt, vec!["explain", "main.rs"]);
    }

    #[test]
    fn bare_read_only_with_no_prompt_enters_read_only_repl() {
        let cli = Cli::try_parse_from(["piku", "--read-only"]).unwrap();
        assert!(cli.read_only);
        assert!(cli.command.is_none());
    }

    #[test]
    fn providers_subcommand() {
        let cli = Cli::try_parse_from(["piku", "providers"]).unwrap();
        assert!(matches!(cli.command, Some(Commands::Providers)));
    }

    #[test]
    fn web_subcommand() {
        let cli = Cli::try_parse_from(["piku", "web"]).unwrap();
        assert!(matches!(cli.command, Some(Commands::Web { port: 8080 })));
    }

    #[test]
    fn web_subcommand_with_port() {
        let cli = Cli::try_parse_from(["piku", "web", "--port", "3000"]).unwrap();
        assert!(matches!(cli.command, Some(Commands::Web { port: 3000 })));
    }

    #[test]
    fn inspect_subcommand_with_html() {
        let cli = Cli::try_parse_from(["piku", "inspect", "session-1", "--html"]).unwrap();
        match cli.command {
            Some(Commands::Inspect {
                session_id,
                html,
                json,
            }) => {
                assert_eq!(session_id, "session-1");
                assert!(html);
                assert!(!json);
            }
            _ => panic!("expected Inspect"),
        }
    }

    #[test]
    fn inspect_defaults_to_text() {
        let cli = Cli::try_parse_from(["piku", "inspect", "session-1"]).unwrap();
        match cli.command {
            Some(Commands::Inspect { html, json, .. }) => {
                assert!(!html);
                assert!(!json);
            }
            _ => panic!("expected Inspect"),
        }
    }

    #[test]
    fn conclude_requires_both_status_and_note() {
        let cli = Cli::try_parse_from([
            "piku",
            "conclude",
            "session-1",
            "--status",
            "needs-work",
            "--note",
            "verify the browser projection",
        ])
        .unwrap();
        assert!(matches!(
            cli.command,
            Some(Commands::Conclude {
                session_id,
                status,
                note,
            }) if session_id == "session-1" && status == "needs-work" && note == "verify the browser projection"
        ));
    }

    #[test]
    fn conclude_missing_args_fails() {
        assert!(
            Cli::try_parse_from(["piku", "conclude", "session-1", "--status", "accepted"]).is_err()
        );
    }
}
