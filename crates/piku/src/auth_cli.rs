//! Explicit operator commands; captured login exposes only a local bootstrap.
mod login_helper;
use std::sync::Arc;

use piku::cli::AuthCommand;
use piku_runtime::chatgpt_auth::{default_directory, AccountInfo, AuthManager};

const FUNDING: &str = "ChatGPT subscription funding; allowance and credits unknown; no hard output-token cap for foreground, children, or memory calls";

pub fn helper_requested() -> bool {
    login_helper::requested()
}

pub async fn run_helper() -> anyhow::Result<()> {
    login_helper::run().await
}

pub async fn run(command: &AuthCommand) -> anyhow::Result<()> {
    if let AuthCommand::Login {
        registration,
        request_consent,
    } = command
    {
        let url = login_helper::launch(registration.clone(), *request_consent).await?;
        println!("Open this local sign-in link, then choose Continue in your browser:");
        println!("{url}");
        println!("The attempt expires in 3 minutes. Check `piku auth chatgpt status` or stop it with `piku auth chatgpt cancel-login`.");
        return Ok(());
    }
    let manager = AuthManager::open(default_directory()?)?;
    match command {
        AuthCommand::Login { .. } => {
            unreachable!("login was dispatched before opening the account store")
        }
        AuthCommand::CancelLogin => {
            if manager.cancel_login()? {
                println!("Login cancellation requested. Check `piku auth chatgpt status` for the terminal outcome.");
            } else {
                println!("No active login attempt.");
            }
        }
        AuthCommand::Accounts | AuthCommand::Status => {
            if matches!(command, AuthCommand::Status) {
                if let Some(status) = manager.login_status()? {
                    println!(
                        "Login attempt: {} phase={} expires_at={}",
                        status.attempt_id,
                        serde_json::to_value(status.phase)?
                            .as_str()
                            .unwrap_or("unknown"),
                        status.expires_at
                    );
                    if let Some(id) = status.registration_id {
                        println!("Select explicitly with: piku auth chatgpt select {id}");
                    }
                }
            }
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
