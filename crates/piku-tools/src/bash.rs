use std::fmt::Write as _;
use std::time::Duration;

use serde::Deserialize;
use tokio::process::Command;
use tokio::time::timeout;

use crate::{
    Destructiveness, ToolEffect, ToolResult, VerificationIndeterminate, VerificationRecord,
    VerificationStatus,
};

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
    #[serde(default)]
    pub purpose: BashPurpose,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BashPurpose {
    #[default]
    Operation,
    Verification,
}

#[must_use]
pub fn schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "command": { "type": "string", "description": "Shell command to execute" },
            "timeout_ms": { "type": "integer", "description": "Timeout in milliseconds (default 30000)" },
            "description": { "type": "string", "description": "Short description of what this command does" },
            "purpose": {
                "type": "string",
                "enum": ["operation", "verification"],
                "description": "Use verification only when this command checks a claimed result"
            }
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

/// Cap on each captured stream.
///
/// `read_file` already refuses files over 10 MB to avoid an OOM on a huge or
/// special file. A command's output is the same exposure by another route, and
/// worse for being unbounded twice: it grows in memory here, then travels into
/// the conversation, where a stray `find /` or `cat` of a log costs a context
/// window and the tokens to fill it. Kept generous enough that ordinary build
/// and test output is untouched.
const MAX_STREAM_BYTES: usize = 256 * 1024;

/// Decode a captured stream, keeping the head and saying what was dropped.
///
/// The head is what a reader wants: a compiler's first errors, a test run's
/// first failures. Truncating silently would let a model reason from a partial
/// result it believes is complete.
fn bound_stream(bytes: &[u8], label: &str) -> String {
    if bytes.len() <= MAX_STREAM_BYTES {
        return String::from_utf8_lossy(bytes).into_owned();
    }
    // Back off to a UTF-8 boundary so the cut does not split a character;
    // a continuation byte has the bit pattern 0b10xx_xxxx.
    let mut end = MAX_STREAM_BYTES;
    while end > 0 && (bytes[end] & 0b1100_0000) == 0b1000_0000 {
        end -= 1;
    }
    let mut out = String::from_utf8_lossy(&bytes[..end]).into_owned();
    let _ = write!(
        out,
        "\n[{label} truncated: {} of {} bytes shown]",
        end,
        bytes.len()
    );
    out
}

#[must_use]
pub async fn execute(params: serde_json::Value) -> ToolResult {
    let p: BashParams = match serde_json::from_value(params) {
        Ok(v) => v,
        Err(e) => return ToolResult::error(format!("invalid params: {e}")),
    };

    let timeout_duration = Duration::from_millis(p.timeout_ms.unwrap_or(DEFAULT_TIMEOUT_MS));
    let verification_description = p.description.clone().unwrap_or_else(|| p.command.clone());

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
        Err(_) => attach_verification(
            ToolResult::error(format!(
                "bash: command timed out after {}ms: {}",
                p.timeout_ms.unwrap_or(DEFAULT_TIMEOUT_MS),
                p.command
            ))
            .with_effect(ToolEffect::ShellCommand {
                command: p.command.clone(),
                exit_code: None,
            }),
            p.purpose,
            verification_description,
            VerificationStatus::Indeterminate {
                reason: VerificationIndeterminate::TimedOut,
            },
        ),
        Ok(Err(e)) => attach_verification(
            ToolResult::error(format!("bash: spawn failed: {e}")).with_effect(
                ToolEffect::ShellCommand {
                    command: p.command.clone(),
                    exit_code: None,
                },
            ),
            p.purpose,
            verification_description,
            VerificationStatus::Indeterminate {
                reason: VerificationIndeterminate::SpawnFailed,
            },
        ),
        Ok(Ok(output)) => {
            let stdout = bound_stream(&output.stdout, "stdout");
            let stderr = bound_stream(&output.stderr, "stderr");
            let code = output.status.code().unwrap_or(-1);

            let result = if code != 0 {
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
            .with_effect(ToolEffect::ShellCommand {
                command: p.command.clone(),
                exit_code: output.status.code(),
            });
            attach_verification(
                result,
                p.purpose,
                verification_description,
                if output.status.success() {
                    VerificationStatus::Passed
                } else {
                    VerificationStatus::Failed {
                        exit_code: output.status.code(),
                    }
                },
            )
        }
    }
}

fn attach_verification(
    result: ToolResult,
    purpose: BashPurpose,
    description: String,
    status: VerificationStatus,
) -> ToolResult {
    if purpose == BashPurpose::Verification {
        result.with_verification(VerificationRecord {
            description,
            status,
        })
    } else {
        result
    }
}
