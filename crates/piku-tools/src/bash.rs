use std::fmt::Write as _;
use std::process::Stdio;
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
fn bound_stream(bytes: &[u8], total: usize, label: &str) -> String {
    if total <= MAX_STREAM_BYTES {
        return String::from_utf8_lossy(bytes).into_owned();
    }
    // Back off to a UTF-8 boundary so the cut does not split a character;
    // a continuation byte has the bit pattern 0b10xx_xxxx.
    let mut end = MAX_STREAM_BYTES;
    while end > 0 && (bytes[end] & 0b1100_0000) == 0b1000_0000 {
        end -= 1;
    }
    let mut out = String::from_utf8_lossy(&bytes[..end]).into_owned();
    let _ = write!(out, "\n[{label} truncated: {end} of {total} bytes shown]");
    out
}

#[allow(clippy::too_many_lines)]
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
    cmd.arg("-c")
        .arg(&p.command)
        .kill_on_drop(true)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    // On Unix, put the child in its own process group so that on timeout we
    // can SIGKILL the entire group (kills grandchildren too).
    #[cfg(unix)]
    cmd.process_group(0);

    let child = match cmd.spawn() {
        Ok(child) => child,
        Err(e) => {
            return attach_verification(
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
            );
        }
    };
    let process_group = child.id();
    let mut output = CapturedOutput::default();
    let fut = async {
        let status = capture_output(child, &mut output).await?;
        let stdout = bound_stream(&output.stdout, output.stdout_total, "stdout");
        let stderr = bound_stream(&output.stderr, output.stderr_total, "stderr");
        // Decoding is bounded by the retained prefixes. Yield before returning
        // so the deadline also covers this last synchronous work.
        tokio::task::yield_now().await;
        Ok::<_, std::io::Error>((status, stdout, stderr))
    };

    match timeout(timeout_duration, fut).await {
        Err(_) => {
            #[cfg(unix)]
            kill_process_group(process_group);
            let mut message = format!(
                "bash: command timed out after {}ms: {}",
                p.timeout_ms.unwrap_or(DEFAULT_TIMEOUT_MS),
                p.command
            );
            // Only bytes already observed by the cancelled drain are retained;
            // these counts are not complete process-output totals.
            for (label, bytes, total) in [
                ("stdout", &output.stdout, output.stdout_total),
                ("stderr", &output.stderr, output.stderr_total),
            ] {
                if !bytes.is_empty() {
                    let partial = bound_stream(bytes, total, label);
                    let _ = write!(
                        message,
                        "\npartial {label} (observed {total} bytes):\n{partial}"
                    );
                }
            }
            attach_verification(
                ToolResult::error(message).with_effect(ToolEffect::ShellCommand {
                    command: p.command.clone(),
                    exit_code: None,
                }),
                p.purpose,
                verification_description,
                VerificationStatus::Indeterminate {
                    reason: VerificationIndeterminate::TimedOut,
                },
            )
        }
        Ok(Err(e)) => {
            #[cfg(unix)]
            kill_process_group(process_group);
            attach_verification(
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
            )
        }
        Ok(Ok((status, stdout, stderr))) => {
            let code = status.code().unwrap_or(-1);

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
                exit_code: status.code(),
            });
            attach_verification(
                result,
                p.purpose,
                verification_description,
                if status.success() {
                    VerificationStatus::Passed
                } else {
                    VerificationStatus::Failed {
                        exit_code: status.code(),
                    }
                },
            )
        }
    }
}

#[cfg(unix)]
fn kill_process_group(process_group: Option<u32>) {
    let Some(process_group) = process_group.and_then(|id| i32::try_from(id).ok()) else {
        return;
    };
    // A negative PID directs SIGKILL to the process group created above.
    // Ignore ESRCH: the direct child may have exited while the timeout fired.
    let _ = nix::sys::signal::kill(
        nix::unistd::Pid::from_raw(-process_group),
        nix::sys::signal::Signal::SIGKILL,
    );
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

#[derive(Default)]
struct CapturedOutput {
    stdout: Vec<u8>,
    stderr: Vec<u8>,
    stdout_total: usize,
    stderr_total: usize,
}

async fn capture_output(
    mut child: tokio::process::Child,
    output: &mut CapturedOutput,
) -> std::io::Result<std::process::ExitStatus> {
    let stdout = child.stdout.take().expect("piped stdout");
    let stderr = child.stderr.take().expect("piped stderr");
    let ((), (), status) = tokio::try_join!(
        capture_stream(stdout, &mut output.stdout, &mut output.stdout_total),
        capture_stream(stderr, &mut output.stderr, &mut output.stderr_total),
        child.wait(),
    )?;
    Ok(status)
}

async fn capture_stream(
    mut stream: impl tokio::io::AsyncRead + Unpin,
    prefix: &mut Vec<u8>,
    total: &mut usize,
) -> std::io::Result<()> {
    use tokio::io::AsyncReadExt;

    // One lookahead byte lets bound_stream preserve its UTF-8 cut behavior.
    // Continue draining excess: noisy commands retain their normal exit status.
    let limit = MAX_STREAM_BYTES + 1;
    prefix.reserve_exact(limit);
    let mut buffer = Box::new([0_u8; 8192]);
    loop {
        let count = stream.read(&mut buffer[..]).await?;
        if count == 0 {
            return Ok(());
        }
        *total = total
            .checked_add(count)
            .ok_or_else(|| std::io::Error::other("shell output byte count overflow"))?;
        let keep = count.min(limit - prefix.len());
        prefix.extend_from_slice(&buffer[..keep]);
    }
}

#[cfg(test)]
mod capture_tests {
    use super::*;

    #[tokio::test]
    async fn shell_capture_retains_only_bounded_prefixes() {
        let child = Command::new("sh")
            .arg("-c")
            .arg("dd if=/dev/zero bs=65536 count=32 2>/dev/null; dd if=/dev/zero bs=65536 count=32 >&2 2>/dev/null")
            .stdout(Stdio::piped()).stderr(Stdio::piped()).spawn().unwrap();
        let mut output = CapturedOutput::default();
        let status = capture_output(child, &mut output).await.unwrap();
        assert!(status.success());
        assert_eq!(output.stdout_total, 2 * 1024 * 1024);
        assert_eq!(output.stderr_total, 2 * 1024 * 1024);
        assert!(
            output.stdout.capacity() <= MAX_STREAM_BYTES + 1,
            "stdout allocation {} exceeds bounded capture",
            output.stdout.capacity()
        );
        assert!(
            output.stderr.capacity() <= MAX_STREAM_BYTES + 1,
            "stderr allocation {} exceeds bounded capture",
            output.stderr.capacity()
        );
    }

    #[tokio::test]
    async fn shell_noisy_streams_preserve_exit_and_disclose_totals() {
        for code in [0, 7] {
            let result = execute(serde_json::json!({
                "command": format!("dd if=/dev/zero bs=65536 count=32 2>/dev/null & dd if=/dev/zero bs=65536 count=32 >&2 2>/dev/null & wait; exit {code}"),
                "purpose": "verification",
            })).await;
            assert_eq!(result.is_error, code != 0);
            assert!(result
                .output
                .contains("[stdout truncated: 262144 of 2097152 bytes shown]"));
            assert!(result
                .output
                .contains("[stderr truncated: 262144 of 2097152 bytes shown]"));
            assert!(result.output.len() < 2 * MAX_STREAM_BYTES + 200);
            assert!(matches!(&result.effects[0],
                ToolEffect::ShellCommand { exit_code: Some(actual), .. } if *actual == code));
        }
    }

    #[tokio::test]
    async fn shell_infinite_output_respects_deadline() {
        let start = std::time::Instant::now();
        let result = execute(serde_json::json!({
            "command": "yes noisy & yes noisy >&2 & wait",
            "timeout_ms": 100,
        }))
        .await;
        assert!(result.is_error && result.output.contains("timed out"));
        assert!(start.elapsed() < Duration::from_secs(2));
    }

    #[tokio::test]
    async fn shell_timeout_preserves_emitted_diagnostics_and_classification() {
        // Encoded printf arguments prevent the echoed timeout command from
        // masquerading as retained output. The late marker must never appear.
        let prints = r"printf '\117\125\124\055\104\111\101\107'; printf '\105\122\122\055\104\111\101\107' >&2";
        let positive =
            execute(serde_json::json!({"command": prints, "purpose":"verification"})).await;
        assert_eq!(positive.output, "OUT-DIAG\nstderr:\nERR-DIAG");
        assert!(!positive.is_error);
        assert_eq!(
            positive.verification.unwrap().status,
            VerificationStatus::Passed
        );
        let marker =
            std::env::temp_dir().join(format!("piku-shell-diagnostics-{}", std::process::id()));
        assert!(!marker.exists());
        let command = format!(
            r"{prints}; printf ready > '{}'; sleep 2; printf '\114\101\124\105'",
            marker.display()
        );
        let start = std::time::Instant::now();
        let result = execute(
            serde_json::json!({"command":command,"timeout_ms":300,"purpose":"verification"}),
        )
        .await;
        assert_eq!(std::fs::read_to_string(&marker).unwrap(), "ready");
        std::fs::remove_file(&marker).unwrap();
        assert!(result.is_error && result.output.contains("timed out after 300ms"));
        assert!(result
            .output
            .contains("partial stdout (observed 8 bytes):\nOUT-DIAG"));
        assert!(result
            .output
            .contains("partial stderr (observed 8 bytes):\nERR-DIAG"));
        assert!(!result.output.contains("LATE"));
        assert_eq!(
            result.verification.unwrap().status,
            VerificationStatus::Indeterminate {
                reason: VerificationIndeterminate::TimedOut
            }
        );
        assert!(
            matches!(result.effects.as_slice(), [ToolEffect::ShellCommand { command: actual, exit_code: None }] if actual == &command)
        );
        assert!(start.elapsed() < Duration::from_secs(2));
    }

    #[tokio::test]
    async fn shell_timeout_noisy_partial_streams_remain_bounded() {
        let marker =
            std::env::temp_dir().join(format!("piku-shell-noisy-timeout-{}", std::process::id()));
        assert!(!marker.exists());
        let result = execute(serde_json::json!({
            "command":format!("dd if=/dev/zero bs=65536 count=8 2>/dev/null; dd if=/dev/zero bs=65536 count=8 >&2 2>/dev/null; printf ready > '{}'; sleep 2", marker.display()),
            "timeout_ms":300,"purpose":"verification",
        })).await;
        assert_eq!(std::fs::read_to_string(&marker).unwrap(), "ready");
        std::fs::remove_file(&marker).unwrap();
        assert!(result.is_error);
        for stream in ["stdout", "stderr"] {
            assert!(result
                .output
                .contains(&format!("partial {stream} (observed 524288 bytes):")));
            assert!(result.output.contains(&format!(
                "[{stream} truncated: 262144 of 524288 bytes shown]"
            )));
        }
        assert!(result.output.len() < 2 * MAX_STREAM_BYTES + 1000);
        assert_eq!(
            result.verification.unwrap().status,
            VerificationStatus::Indeterminate {
                reason: VerificationIndeterminate::TimedOut
            }
        );
    }

    #[test]
    fn shell_truncation_keeps_utf8_boundary_and_small_output() {
        let mut prefix = vec![b'a'; MAX_STREAM_BYTES - 1];
        prefix.extend_from_slice("✓".as_bytes());
        let output = bound_stream(&prefix[..=MAX_STREAM_BYTES], prefix.len(), "stdout");
        assert!(!output.contains('�'));
        assert!(output.ends_with("262143 of 262146 bytes shown]"));
        assert_eq!(bound_stream("✓ ok".as_bytes(), 6, "stdout"), "✓ ok");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn shell_timeout_kills_pipe_holding_descendant() {
        let dir =
            std::env::temp_dir().join(format!("piku-shell-descendant-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let marker = dir.join("survived");
        assert!(!marker.exists());
        let result = execute(serde_json::json!({
            "command": format!("(sleep 1; echo survived > '{}') & exit 0", marker.display()),
            "timeout_ms": 100,
        }))
        .await;
        assert!(result.is_error && result.output.contains("timed out"));
        tokio::time::sleep(Duration::from_millis(1200)).await;
        assert!(!marker.exists(), "descendant survived timeout cleanup");
        std::fs::remove_dir(&dir).unwrap();
    }
}
