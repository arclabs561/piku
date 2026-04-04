use std::time::Duration;

use serde::Deserialize;
use tokio::process::Command;
use tokio::time::timeout;

use crate::{Destructiveness, ToolResult};

const DEFAULT_TIMEOUT_MS: u64 = 30_000;

// Patterns that make a bash command Definite
const DEFINITE_PATTERNS: &[&str] = &[
    "rm ",
    "rm\t",
    "rm\n",
    "rmdir",
    "dd ",
    "mkfs",
    "sudo ",
    ":(){:|:&};:", // fork bomb
    "chmod 777",
    "chown ",
    "curl | sh",
    "curl|sh",
    "wget | sh",
    "wget|sh",
    "| bash",
    "|bash",
    "| sh",
    "|sh",
    ">/dev/",
    "2>/dev/",
    " > ", // output redirect with spaces — could overwrite any file
    "1> ", // explicit stdout redirect
];

// Patterns that need closer inspection (Likely)
const LIKELY_PATTERNS: &[&str] = &[
    ">>",  // append redirect (less destructive than overwrite)
    "> /", // redirect to absolute path (belt-and-suspenders catch)
    "mv ", "cp ", "truncate", "shred", "kill", "pkill", "killall",
];

#[derive(Debug, Deserialize)]
pub struct BashParams {
    pub command: String,
    /// Timeout in milliseconds. Default 30s.
    pub timeout_ms: Option<u64>,
    /// Description for display in permission prompts.
    pub description: Option<String>,
}

#[must_use]
pub fn schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "command": { "type": "string", "description": "Shell command to execute" },
            "timeout_ms": { "type": "integer", "description": "Timeout in milliseconds (default 30000)" },
            "description": { "type": "string", "description": "Short description of what this command does" }
        },
        "required": ["command"]
    })
}

#[must_use]
pub fn destructiveness(params: &serde_json::Value) -> Destructiveness {
    let cmd = params.get("command").and_then(|v| v.as_str()).unwrap_or("");
    for pat in DEFINITE_PATTERNS {
        if cmd.contains(pat) {
            return Destructiveness::Definite;
        }
    }
    for pat in LIKELY_PATTERNS {
        if cmd.contains(pat) {
            return Destructiveness::Likely;
        }
    }
    Destructiveness::Likely // bash is always at least Likely
}

#[must_use]
pub async fn execute(params: serde_json::Value) -> ToolResult {
    let p: BashParams = match serde_json::from_value(params) {
        Ok(v) => v,
        Err(e) => return ToolResult::error(format!("invalid params: {e}")),
    };

    let timeout_duration = Duration::from_millis(p.timeout_ms.unwrap_or(DEFAULT_TIMEOUT_MS));

    // Use `sh -c` not `sh -lc` to avoid sourcing login shell profiles.
    // Login shell startup (nvm, pyenv, etc.) can consume 200-800ms before
    // any command runs, causing spurious timeouts on short timeout_ms values.
    // Users who need login shell behaviour can prefix: `bash -lc '...'`.
    let mut cmd = Command::new("sh");
    cmd.arg("-c").arg(&p.command).kill_on_drop(true);

    // On Unix, put the child in its own process group so that on timeout we
    // can SIGKILL the entire group (kills grandchildren too).
    #[cfg(unix)]
    cmd.process_group(0);

    let fut = cmd.output();

    match timeout(timeout_duration, fut).await {
        Err(_) => ToolResult::error(format!(
            "bash: command timed out after {}ms: {}",
            p.timeout_ms.unwrap_or(DEFAULT_TIMEOUT_MS),
            p.command
        )),
        Ok(Err(e)) => ToolResult::error(format!("bash: spawn failed: {e}")),
        Ok(Ok(output)) => {
            let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
            let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
            let code = output.status.code().unwrap_or(-1);

            if code != 0 {
                let mut msg = format!("exit code {code}");
                if !stdout.is_empty() {
                    msg.push_str("\nstdout:\n");
                    msg.push_str(&stdout);
                }
                if !stderr.is_empty() {
                    msg.push_str("\nstderr:\n");
                    msg.push_str(&stderr);
                }
                ToolResult::error(msg)
            } else {
                let mut out = stdout;
                if !stderr.is_empty() {
                    if !out.is_empty() {
                        out.push('\n');
                    }
                    out.push_str("stderr:\n");
                    out.push_str(&stderr);
                }
                ToolResult::ok(out)
            }
        }
    }
}
