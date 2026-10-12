//! A bounded login process with a credential-free readiness handshake.
use std::path::PathBuf;
use std::process::Stdio;
use std::time::Duration;

use piku_runtime::chatgpt_auth::{default_directory, AuthManager};
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};

const HELPER_ARG: &str = "__chatgpt-login-helper";
const MAX_FRAME: u64 = 16 * 1024;

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Options {
    directory: PathBuf,
    registration: Option<String>,
    request_consent: bool,
}

#[derive(Serialize, Deserialize)]
#[serde(untagged)]
enum Ready {
    Started { bootstrap_url: String },
    Failed { error: String },
}

pub(super) fn requested() -> bool {
    std::env::args_os()
        .nth(1)
        .is_some_and(|arg| arg == HELPER_ARG)
}

pub(super) async fn launch(
    registration: Option<String>,
    request_consent: bool,
) -> anyhow::Result<String> {
    let options = Options {
        directory: default_directory()?,
        registration,
        request_consent,
    };
    let mut child = tokio::process::Command::new(std::env::current_exe()?)
        .arg(HELPER_ARG)
        .env_clear()
        .current_dir("/")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()?;
    let exchange = async {
        let mut input = child
            .stdin
            .take()
            .ok_or_else(|| anyhow::anyhow!("login helper input unavailable"))?;
        input.write_all(&serde_json::to_vec(&options)?).await?;
        input.shutdown().await?;
        drop(input);
        let output = child
            .stdout
            .take()
            .ok_or_else(|| anyhow::anyhow!("login helper output unavailable"))?;
        let mut reader = BufReader::new(output.take(MAX_FRAME));
        let mut frame = String::new();
        reader.read_line(&mut frame).await?;
        let ready = serde_json::from_str::<Ready>(&frame).map_err(|_| {
            anyhow::anyhow!("login helper could not start; check protected auth storage and retry")
        })?;
        match ready {
            Ready::Started { bootstrap_url } => Ok(bootstrap_url),
            Ready::Failed { error } => Err(anyhow::anyhow!("{error}")),
        }
    };
    match tokio::time::timeout(Duration::from_secs(5), exchange).await {
        Ok(Ok(url)) => Ok(url),
        failure => {
            // Reap failures; successful helpers own their bounded runtime lifetime.
            let _ = child.kill().await;
            let _ = child.wait().await;
            match failure {
                Ok(Err(error)) => Err(error),
                _ => anyhow::bail!(
                    "login helper did not become ready; retry `piku auth chatgpt login`"
                ),
            }
        }
    }
}

pub(super) async fn run() -> anyhow::Result<()> {
    // This is the freshly exec'd helper, not a fork of a multithreaded process.
    // Detach from the captured runner's terminal/session without launching a daemon.
    nix::unistd::setsid()?;
    let mut input = Vec::new();
    tokio::time::timeout(
        Duration::from_secs(5),
        tokio::io::stdin().take(MAX_FRAME).read_to_end(&mut input),
    )
    .await??;
    let options: Options = serde_json::from_slice(&input)?;
    let manager = match AuthManager::open(options.directory) {
        Ok(manager) => manager,
        Err(error) => {
            write_ready(&Ready::Failed {
                error: error.to_string(),
            })
            .await?;
            return Ok(());
        }
    };
    let attempt = match manager
        .prepare_browser_login(options.registration, options.request_consent)
        .await
    {
        Ok(attempt) => attempt,
        Err(error) => {
            write_ready(&Ready::Failed {
                error: error.to_string(),
            })
            .await?;
            return Ok(());
        }
    };
    write_ready(&Ready::Started {
        bootstrap_url: attempt.bootstrap_url().to_string(),
    })
    .await?;
    // Runtime records a sanitized terminal outcome; no diagnostics go to the shell.
    let _ = attempt.run().await;
    Ok(())
}

async fn write_ready(ready: &Ready) -> anyhow::Result<()> {
    let mut frame = serde_json::to_vec(ready)?;
    frame.push(b'\n');
    let mut output = tokio::io::stdout();
    output.write_all(&frame).await?;
    output.flush().await?;
    Ok(())
}
