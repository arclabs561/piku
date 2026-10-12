//! Explicit operator commands; authorization URLs are presented only for login.
use std::io::{self, IsTerminal, Write};
use std::sync::Arc;

use piku::cli::AuthCommand;
use piku_runtime::chatgpt_auth::{default_directory, AccountInfo, AuthManager};

const FUNDING: &str = "ChatGPT subscription funding; allowance and credits unknown; no hard output-token cap for foreground, children, or memory calls";

pub async fn run(command: &AuthCommand) -> anyhow::Result<()> {
    if matches!(command, AuthCommand::Login { .. })
        && (!io::stdin().is_terminal() || !io::stdout().is_terminal())
    {
        anyhow::bail!("ChatGPT login requires an attended terminal for stdin and stdout; run `piku auth chatgpt login` directly in your terminal without pipes or redirection");
    }
    let manager = AuthManager::open(default_directory()?)?;
    match command {
        AuthCommand::Login {
            registration,
            request_consent,
        } => {
            let pending = manager
                .begin_login(registration.as_deref(), *request_consent)
                .await?;
            eprintln!("Open this private authorization URL in your browser; do not share it:");
            // May contain an identity hint. Never send this URL to tracing or
            // account status; only the explicit interactive login presents it.
            println!("{}", pending.authorization_url());
            io::stdout().flush()?;
            let account = manager.complete_login(pending).await?;
            print_account(&account, false);
            println!(
                "Select explicitly with: piku auth chatgpt select {}",
                account.registration_id
            );
        }
        AuthCommand::Accounts | AuthCommand::Status => {
            let selected = manager.selected_registration_id()?;
            let accounts = manager.list().await?;
            if accounts.is_empty() {
                println!("No ChatGPT accounts stored. Run: piku auth chatgpt login");
            }
            for account in &accounts {
                if matches!(command, AuthCommand::Accounts)
                    || selected.as_deref() == Some(account.registration_id.as_str())
                {
                    print_account(
                        account,
                        selected.as_deref() == Some(account.registration_id.as_str()),
                    );
                }
            }
            if selected.is_none() {
                println!("No account selected.");
            }
            println!("{FUNDING}");
            println!("Stored metadata only; no authentication or quota probe was performed.");
        }
        AuthCommand::Select { account_id } => {
            let selected = manager.select(account_id).await?;
            println!(
                "Selected ChatGPT account {} for new runs.",
                selected.binding()
            );
            println!("{FUNDING}");
        }
        AuthCommand::Logout { account_id } => {
            let result = manager.logout(account_id).await?;
            println!("Local ChatGPT credentials removed for {account_id}.");
            println!(
                "Remote revocation confirmed: {}",
                result.remote_revocation_confirmed
            );
        }
        AuthCommand::Models => {
            let selected = manager.selected()?.ok_or_else(|| {
                anyhow::anyhow!(
                    "no ChatGPT account selected; use `piku auth chatgpt select ACCOUNT_ID`"
                )
            })?;
            let provider = piku_api::ResponsesProvider::new(Arc::new(selected))?;
            for model in provider.models().await? {
                println!(
                    "{}\t{}",
                    terminal_text(&model.slug),
                    terminal_text(&model.display_name)
                );
            }
        }
    }
    Ok(())
}

fn print_account(account: &AccountInfo, selected: bool) {
    println!(
        "{}{} signed_in={} plan_permission={}",
        if selected { "* " } else { "  " },
        account.registration_id,
        account.signed_in,
        account.plan_enabled
    );
}

// Remote catalog labels are data, never terminal control sequences.
fn terminal_text(value: &str) -> String {
    value
        .chars()
        .flat_map(|ch| {
            if ch.is_control() {
                ch.escape_default().collect::<Vec<_>>()
            } else {
                vec![ch]
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    #[test]
    fn catalog_labels_cannot_control_terminal() {
        assert_eq!(
            super::terminal_text("model\x1b[2J\nname"),
            "model\\u{1b}[2J\\nname"
        );
        assert_eq!(super::terminal_text("Modèle"), "Modèle");
    }
}
