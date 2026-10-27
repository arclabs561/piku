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
        help = "Force provider: chatgpt | openrouter | anthropic | groq | ollama | custom"
    )]
    pub provider: Option<String>,

    #[arg(
        long,
        global = true,
        value_name = "PATH",
        help = "Create a fresh private request metadata JSONL log for terminal, print, or resume launches"
    )]
    pub request_log: Option<std::path::PathBuf>,

    #[arg(long, help = "Run with file-inspection tools only, then exit")]
    pub read_only: bool,

    #[arg(
        long,
        value_name = "ABSOLUTE_BUILD_ROOT",
        help = "Enable developer self-update from an approved Cargo target directory"
    )]
    pub self_update_root: Option<std::path::PathBuf>,

    #[arg(
        short = 'p',
        long = "print",
        help = "Headless: run the prompt, print the result, and exit. Tool calls run without prompting except those matching deny rules in settings.toml"
    )]
    pub print: bool,

    #[arg(long, help = "Resume a previous session by ID (partial match ok)")]
    pub resume: Option<String>,

    #[arg(
        long,
        value_name = "N",
        num_args = 0..=1,
        default_missing_value = "20",
        help = "Attach the last N commands from the piku shell ledger to the prompt (default 20; or set PIKU_SHELL_CONTEXT)"
    )]
    pub shell_context: Option<usize>,

    #[arg(trailing_var_arg = true, allow_hyphen_values = true, hide = true)]
    pub prompt: Vec<String>,
}

impl Cli {
    /// Validate before subcommand dispatch, authentication, or provider resolution.
    pub fn validate_request_log(&self) -> anyhow::Result<()> {
        if self.request_log.is_some() && self.command.is_some() {
            anyhow::bail!("--request-log supports terminal, --print, and --resume launches only; see piku --help");
        }
        if self.request_log.is_some() && self.self_update_root.is_some() {
            anyhow::bail!("--request-log cannot combine with --self-update-root: restart explicitly with a fresh request-log path; see piku --help");
        }
        Ok(())
    }

    /// `shell` owns the terminal and runs no agent turn, so a launch option
    /// that would silently do nothing is an error instead.
    pub fn validate_shell_launch(&self) -> anyhow::Result<()> {
        if !matches!(self.command, Some(Commands::Shell)) {
            return Ok(());
        }
        let launch_only = [
            (self.print, "--print"),
            (self.read_only, "--read-only"),
            (self.resume.is_some(), "--resume"),
            (self.model.is_some(), "--model"),
            (self.provider.is_some(), "--provider"),
            (self.self_update_root.is_some(), "--self-update-root"),
            (!self.prompt.is_empty(), "a prompt"),
        ];
        let used: Vec<&str> = launch_only
            .iter()
            .filter(|(present, _)| *present)
            .map(|(_, flag)| *flag)
            .collect();
        if used.is_empty() {
            return Ok(());
        }
        anyhow::bail!(
            "piku shell runs an interactive shell; {} cannot apply to it; see piku --help",
            used.join(", ")
        )
    }
}

#[derive(Debug, Subcommand)]
pub enum Commands {
    #[command(about = "Run the interactive shell (agent launch options do not apply)")]
    Shell,
    #[command(about = "Print shell integration; add `eval \"$(piku init zsh)\"` to ~/.zshrc")]
    Init {
        #[arg(value_parser = ["zsh"])]
        shell: String,
    },
    #[command(
        hide = true,
        about = "Append one shell ledger record (called by the zsh hook)"
    )]
    LedgerAppend {
        #[arg(long)]
        exit: i32,
        #[arg(long)]
        cwd: std::path::PathBuf,
        #[arg(allow_hyphen_values = true)]
        command: String,
    },
    #[command(about = "Review and approve this project's tool permission rules")]
    Permissions {
        #[command(subcommand)]
        command: PermissionCommand,
    },
    #[command(about = "Show provider status")]
    Providers,
    #[command(about = "Manage explicitly selected subscription authentication")]
    Auth {
        #[arg(value_parser = ["chatgpt"])]
        provider: String,
        #[command(subcommand)]
        command: AuthCommand,
    },
    #[command(about = "Inspect a run record")]
    Inspect {
        #[arg(help = "Session ID")]
        session_id: String,
        #[arg(long, help = "Render addressable cells instead of the event stream")]
        cells: bool,
        #[arg(
            long,
            help = "Render addressable interaction receipts instead of the event stream"
        )]
        activity: bool,
        #[arg(
            long,
            value_name = "REFERENCE",
            help = "Show one cell (@@N or canonical cell reference; @cN remains accepted)"
        )]
        cell: Option<String>,
        #[arg(
            long,
            value_name = "REFERENCE",
            help = "Show one receipt (##N or canonical receipt reference)"
        )]
        receipt: Option<String>,
        #[arg(
            long,
            value_name = "REFERENCE",
            help = "Print one cell's retained text output"
        )]
        raw: Option<String>,
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
    #[command(
        about = "Watch sources and settings, rebuild, and restart the TUI at a turn boundary"
    )]
    Dev {
        #[arg(long, help = "Build the release profile instead of debug")]
        release: bool,
        #[arg(long, help = "Run and restart the web surface beside the TUI")]
        web: bool,
    },
    #[command(about = "Start the run surface web server")]
    Web {
        #[arg(long, default_value = "8080", help = "Port to listen on")]
        port: u16,
        #[arg(
            long,
            help = "Allow the authenticated browser to open a host terminal for this launch"
        )]
        terminal: bool,
    },
}

#[derive(Debug, Subcommand)]
pub enum PermissionCommand {
    #[command(about = "Display the current project rules and their approval digest")]
    Review,
    #[command(about = "Approve the exact reviewed rules for this canonical project root")]
    Approve {
        #[arg(long, value_name = "CANONICAL_ROOT")]
        root: std::path::PathBuf,
        #[arg(long, value_name = "SHA256")]
        digest: String,
    },
}

#[derive(Debug, Subcommand)]
pub enum AuthCommand {
    #[command(about = "Start bounded browser sign-in and print a local continuation link")]
    Login {
        #[arg(
            long,
            value_name = "REGISTRATION_ID",
            help = "Reuse a stored registration"
        )]
        registration: Option<String>,
        #[arg(long, help = "Request consent again during this login")]
        request_consent: bool,
    },
    #[command(about = "Cancel the active browser sign-in attempt")]
    CancelLogin,
    #[command(about = "List stored account metadata without contacting the provider")]
    Accounts,
    #[command(about = "Show selected account and funding status without a token probe")]
    Status,
    #[command(about = "Explicitly select an existing account for new runs")]
    Select {
        #[arg(value_name = "ACCOUNT_ID")]
        account_id: String,
    },
    #[command(about = "Remove local credentials for one account")]
    Logout {
        #[arg(value_name = "ACCOUNT_ID")]
        account_id: String,
    },
    #[command(about = "Fetch model availability for the selected account")]
    Models,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_log_is_explicit_and_terminal_only() {
        let default = Cli::try_parse_from(["piku", "-p", "hello"]).unwrap();
        assert!(default.request_log.is_none());
        for args in [
            vec!["piku", "--request-log", "new.jsonl", "-p", "hello"],
            vec![
                "piku",
                "--resume",
                "session-id",
                "--request-log",
                "new.jsonl",
            ],
            vec!["piku", "--request-log", "new.jsonl"],
        ] {
            let cli = Cli::try_parse_from(args).unwrap();
            assert_eq!(
                cli.request_log.as_deref(),
                Some(std::path::Path::new("new.jsonl"))
            );
            cli.validate_request_log().unwrap();
        }
        let restart = Cli::try_parse_from([
            "piku",
            "--request-log",
            "new.jsonl",
            "--self-update-root",
            "/build-root",
        ])
        .unwrap();
        assert!(restart
            .validate_request_log()
            .unwrap_err()
            .to_string()
            .contains("fresh request-log path"));
        for args in [
            vec!["piku", "--request-log", "new.jsonl", "providers"],
            vec!["piku", "web", "--request-log", "new.jsonl"],
            vec![
                "piku",
                "auth",
                "chatgpt",
                "models",
                "--request-log",
                "new.jsonl",
            ],
            vec![
                "piku",
                "inspect",
                "session-id",
                "--request-log",
                "new.jsonl",
            ],
        ] {
            let cli = Cli::try_parse_from(args).unwrap();
            assert!(cli
                .validate_request_log()
                .unwrap_err()
                .to_string()
                .contains("piku --help"));
        }
    }

    #[test]
    fn shell_rejects_launch_options_that_would_do_nothing() {
        Cli::try_parse_from(["piku", "shell"])
            .unwrap()
            .validate_shell_launch()
            .unwrap();
        // Launch options stay valid for the commands that use them.
        Cli::try_parse_from(["piku", "--model", "m", "providers"])
            .unwrap()
            .validate_shell_launch()
            .unwrap();
        for args in [
            vec!["piku", "-p", "shell"],
            vec!["piku", "--model", "m", "shell"],
            vec!["piku", "--provider", "openrouter", "shell"],
            vec!["piku", "--resume", "session-id", "shell"],
        ] {
            let cli = Cli::try_parse_from(args).unwrap();
            assert!(
                cli.validate_shell_launch()
                    .unwrap_err()
                    .to_string()
                    .contains("cannot apply"),
                "a launch option on shell must be an error"
            );
        }
    }

    #[test]
    fn subscription_auth_requires_explicit_provider_and_action() {
        assert!(Cli::try_parse_from(["piku", "auth"]).is_err());
        assert!(Cli::try_parse_from(["piku", "auth", "anthropic", "login"]).is_err());
        assert!(Cli::try_parse_from(["piku", "auth", "chatgpt", "select"]).is_err());
        assert!(Cli::try_parse_from(["piku", "auth", "chatgpt", "logout"]).is_err());
        let cli =
            Cli::try_parse_from(["piku", "auth", "chatgpt", "login", "--request-consent"]).unwrap();
        assert!(matches!(
            cli.command,
            Some(Commands::Auth {
                command: AuthCommand::Login {
                    request_consent: true,
                    registration: None
                },
                ..
            })
        ));
    }

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
        assert!(matches!(
            cli.command,
            Some(Commands::Web {
                port: 8080,
                terminal: false
            })
        ));
    }

    #[test]
    fn web_subcommand_with_port() {
        let cli = Cli::try_parse_from(["piku", "web", "--port", "3000"]).unwrap();
        assert!(matches!(
            cli.command,
            Some(Commands::Web {
                port: 3000,
                terminal: false
            })
        ));
    }

    #[test]
    fn web_terminal_requires_explicit_launch_flag() {
        let cli = Cli::try_parse_from(["piku", "web", "--terminal"]).unwrap();
        assert!(matches!(
            cli.command,
            Some(Commands::Web { terminal: true, .. })
        ));
    }

    #[test]
    fn inspect_subcommand_with_html() {
        let cli = Cli::try_parse_from(["piku", "inspect", "session-1", "--html"]).unwrap();
        match cli.command {
            Some(Commands::Inspect {
                session_id,
                cells,
                activity,
                cell,
                receipt,
                raw,
                html,
                json,
            }) => {
                assert_eq!(session_id, "session-1");
                assert!(!cells);
                assert!(!activity);
                assert!(cell.is_none());
                assert!(receipt.is_none());
                assert!(raw.is_none());
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
            Some(Commands::Inspect {
                html,
                json,
                cells,
                activity,
                cell,
                receipt,
                raw,
                ..
            }) => {
                assert!(!html);
                assert!(!json);
                assert!(!cells);
                assert!(!activity);
                assert!(cell.is_none());
                assert!(receipt.is_none());
                assert!(raw.is_none());
            }
            _ => panic!("expected Inspect"),
        }
    }

    #[test]
    fn inspect_raw_accepts_a_cell_reference() {
        let cli = Cli::try_parse_from(["piku", "inspect", "session-1", "--raw", "@@2"]).unwrap();
        assert!(matches!(
            cli.command,
            Some(Commands::Inspect { raw: Some(reference), .. }) if reference == "@@2"
        ));
    }

    #[test]
    fn inspect_cells_accepts_a_short_reference() {
        let cli = Cli::try_parse_from(["piku", "inspect", "session-1", "--cells", "--cell", "@@2"])
            .unwrap();
        assert!(matches!(
            cli.command,
            Some(Commands::Inspect { cells: true, cell: Some(reference), .. }) if reference == "@@2"
        ));
    }

    #[test]
    fn inspect_activity_accepts_a_receipt_reference() {
        let cli = Cli::try_parse_from([
            "piku",
            "inspect",
            "session-1",
            "--activity",
            "--receipt",
            "##2",
        ])
        .unwrap();
        assert!(matches!(
            cli.command,
            Some(Commands::Inspect {
                activity: true,
                receipt: Some(reference),
                ..
            }) if reference == "##2"
        ));
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
