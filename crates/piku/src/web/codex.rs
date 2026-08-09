use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::{error::Error, fmt};

use anyhow::{anyhow, Context};
use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};

use super::ChatMessage;

const DEVELOPER_INSTRUCTIONS: &str = "You are the read-only conversation executor inside Piku. Answer the user's question directly and concisely. The Piku workspace and its files are not implicit context. Do not inspect files, run commands, use tools, or mutate the workspace. Use only the conversation and optional context supplied in this turn.";
const CHILD_ENV_ALLOWLIST: &[&str] = &[
    // The installed Codex launcher resolves its binary relative to HOME. Codex
    // configuration still comes exclusively from the explicit CODEX_HOME.
    "HOME",
    "LANG",
    "LC_ALL",
    "PATH",
    "SHELL",
    "SSL_CERT_DIR",
    "SSL_CERT_FILE",
    "TERM",
    "TMPDIR",
];

#[derive(Debug, Clone, serde::Serialize)]
pub(super) struct CodexReadiness {
    pub available: bool,
    pub authenticated: bool,
    pub isolated: bool,
    pub model: &'static str,
    pub detail: String,
}

#[derive(Debug)]
pub(super) enum CodexEvent {
    Started {
        model: String,
        thread_id: String,
        turn_id: String,
        input: String,
    },
    Delta(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct CodexUsage {
    pub input_tokens: u64,
    pub output_tokens: u64,
}

#[derive(Debug)]
pub(super) struct CodexResult {
    pub output: String,
    pub model: String,
    pub thread_id: String,
    pub turn_id: String,
    pub usage: Option<CodexUsage>,
}

/// A native Codex turn failure plus any assistant text received before it.
///
/// The partial output is kept separate from the display message so logging the
/// error cannot accidentally duplicate conversation content.
#[derive(Debug)]
pub(super) struct CodexFailure {
    message: String,
    partial_output: String,
}

impl CodexFailure {
    fn new(message: impl Into<String>, partial_output: String) -> Self {
        Self {
            message: message.into(),
            partial_output,
        }
    }

    pub(super) fn partial_output(&self) -> &str {
        &self.partial_output
    }
}

impl fmt::Display for CodexFailure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl Error for CodexFailure {}

impl From<anyhow::Error> for CodexFailure {
    fn from(error: anyhow::Error) -> Self {
        Self::new(error.to_string(), String::new())
    }
}

pub(super) fn readiness() -> CodexReadiness {
    let available = command_exists("codex");
    let authenticated = codex_auth_path().is_some_and(|path| path.is_file());
    let isolated = available && authenticated;
    let detail = match (available, authenticated) {
        (true, true) => "app-server · read-only · isolated configuration".to_string(),
        (true, false) => "Codex is installed, but local authentication is unavailable".to_string(),
        (false, _) => "Codex CLI is unavailable on PATH".to_string(),
    };
    CodexReadiness {
        available,
        authenticated,
        isolated,
        model: "account default",
        detail,
    }
}

pub(super) async fn run_chat<F>(
    workspace_root: &Path,
    codex_root: &Path,
    message: &str,
    context: Option<&str>,
    history: &[ChatMessage],
    thread_id: Option<&str>,
    mut on_event: F,
) -> Result<CodexResult, CodexFailure>
where
    F: FnMut(CodexEvent),
{
    let mut server = CodexServer::spawn(codex_root)?;
    server.initialize().await?;
    let (thread, model) = if let Some(thread_id) = thread_id.filter(|id| !id.trim().is_empty()) {
        server.resume_thread(workspace_root, thread_id).await?
    } else {
        server.start_thread(workspace_root).await?
    };
    let input = compose_input(message, context, history);
    let turn_id = server.start_turn(&thread, &input).await?;
    on_event(CodexEvent::Started {
        model: model.clone(),
        thread_id: thread.clone(),
        turn_id: turn_id.clone(),
        input,
    });

    let mut output = String::new();
    let mut usage = None;
    loop {
        let message = server
            .read_message()
            .await
            .map_err(|error| CodexFailure::new(error.to_string(), output.clone()))?;
        if message.get("id").is_some() && message.get("method").is_some() {
            return Err(CodexFailure::new(
                "Codex requested an interactive action outside Piku's read-only contract",
                output,
            ));
        }
        if apply_stream_event(
            parse_stream_event(&message),
            &mut output,
            &mut usage,
            &mut on_event,
        )? {
            break;
        }
    }
    server.stop().await;
    if output.trim().is_empty() {
        return Err(CodexFailure::new(
            "Codex returned an empty response",
            output,
        ));
    }
    Ok(CodexResult {
        output,
        model,
        thread_id: thread,
        turn_id,
        usage,
    })
}

fn apply_stream_event<F>(
    event: StreamEvent,
    output: &mut String,
    usage: &mut Option<CodexUsage>,
    on_event: &mut F,
) -> Result<bool, CodexFailure>
where
    F: FnMut(CodexEvent),
{
    match event {
        StreamEvent::Delta(delta) => {
            output.push_str(&delta);
            on_event(CodexEvent::Delta(delta));
            Ok(false)
        }
        StreamEvent::TurnStarted | StreamEvent::Ignore => Ok(false),
        StreamEvent::Usage(value) => {
            *usage = Some(value);
            Ok(false)
        }
        StreamEvent::Completed => Ok(true),
        StreamEvent::Failed(reason) => Err(CodexFailure::new(reason, std::mem::take(output))),
    }
}

pub(super) fn compose_input(
    message: &str,
    context: Option<&str>,
    history: &[ChatMessage],
) -> String {
    let mut input = String::new();
    if !history.is_empty() {
        input.push_str("Conversation so far:\n");
        for item in history {
            input.push_str(if item.role == "assistant" {
                "Assistant: "
            } else {
                "User: "
            });
            input.push_str(&item.content);
            input.push('\n');
        }
        input.push('\n');
    }
    if let Some(context) = context.filter(|value| !value.trim().is_empty()) {
        input.push_str("Optional context explicitly attached by the user:\n<context>\n");
        input.push_str(context);
        input.push_str("\n</context>\n\n");
    }
    input.push_str("Current turn:\n");
    input.push_str(message);
    input
}

struct CodexServer {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

impl CodexServer {
    fn spawn(codex_root: &Path) -> anyhow::Result<Self> {
        let auth = codex_auth_path()
            .filter(|path| path.is_file())
            .ok_or_else(|| {
                anyhow!("Codex authentication is unavailable; run `codex login` first")
            })?;
        prepare_codex_home(codex_root, &auth)?;
        let mut command = Command::new("codex");
        command
            .env_clear()
            .args(["app-server", "--listen", "stdio://"])
            .env("CODEX_HOME", codex_root)
            .current_dir(codex_root)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        for key in CHILD_ENV_ALLOWLIST {
            if let Some(value) = std::env::var_os(key) {
                command.env(key, value);
            }
        }
        let mut child = command.spawn().context("start Codex app-server")?;
        let stdin = child.stdin.take().context("open Codex stdin")?;
        let stdout = BufReader::new(child.stdout.take().context("open Codex stdout")?);
        let stderr = child.stderr.take().context("open Codex stderr")?;
        tokio::spawn(async move {
            let mut lines = BufReader::new(stderr).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                tracing::warn!(executor = "codex", message = %line, "Codex app-server diagnostic");
            }
        });
        Ok(Self {
            child,
            stdin,
            stdout,
        })
    }

    async fn initialize(&mut self) -> anyhow::Result<()> {
        self.send(json!({"method":"initialize","id":1,"params":{"clientInfo":{"name":"piku","title":"Piku","version":env!("CARGO_PKG_VERSION")}}})).await?;
        self.expect_response(1).await?;
        self.send(json!({"method":"initialized","params":{}})).await
    }

    async fn start_thread(&mut self, workspace_root: &Path) -> anyhow::Result<(String, String)> {
        self.send(json!({"method":"thread/start","id":2,"params":{
            "cwd": workspace_root,
            "sandbox":"read-only",
            "approvalPolicy":"never",
            "personality":"pragmatic",
            "ephemeral":false,
            "developerInstructions":DEVELOPER_INSTRUCTIONS
        }}))
        .await?;
        let response = self.expect_response(2).await?;
        let thread_id = response
            .pointer("/result/thread/id")
            .and_then(Value::as_str)
            .map(str::to_string)
            .ok_or_else(|| protocol_error(&response, "thread/start failed"))?;
        let model = response
            .pointer("/result/model")
            .or_else(|| response.pointer("/result/thread/model"))
            .and_then(Value::as_str)
            .unwrap_or("account default")
            .to_string();
        Ok((thread_id, model))
    }

    async fn resume_thread(
        &mut self,
        workspace_root: &Path,
        thread_id: &str,
    ) -> anyhow::Result<(String, String)> {
        self.send(json!({"method":"thread/resume","id":2,"params":{
            "threadId":thread_id,
            "cwd":workspace_root,
            "sandbox":"read-only",
            "approvalPolicy":"never",
            "personality":"pragmatic",
            "developerInstructions":DEVELOPER_INSTRUCTIONS
        }}))
        .await?;
        let response = self.expect_response(2).await?;
        let resumed_id = response
            .pointer("/result/thread/id")
            .and_then(Value::as_str)
            .map(str::to_string)
            .ok_or_else(|| protocol_error(&response, "thread/resume failed"))?;
        if resumed_id != thread_id {
            return Err(anyhow!("Codex resumed an unexpected thread"));
        }
        let model = response
            .pointer("/result/model")
            .or_else(|| response.pointer("/result/thread/model"))
            .and_then(Value::as_str)
            .unwrap_or("account default")
            .to_string();
        Ok((resumed_id, model))
    }

    async fn start_turn(&mut self, thread_id: &str, input: &str) -> anyhow::Result<String> {
        self.send(json!({"method":"turn/start","id":3,"params":{"threadId":thread_id,"input":[{"type":"text","text":input}]}})).await?;
        let response = self.expect_response(3).await?;
        response
            .pointer("/result/turn/id")
            .and_then(Value::as_str)
            .map(str::to_string)
            .ok_or_else(|| protocol_error(&response, "turn/start failed"))
    }

    async fn send(&mut self, value: Value) -> anyhow::Result<()> {
        let mut bytes = serde_json::to_vec(&value)?;
        bytes.push(b'\n');
        self.stdin.write_all(&bytes).await?;
        self.stdin.flush().await?;
        Ok(())
    }

    async fn read_message(&mut self) -> anyhow::Result<Value> {
        let mut line = String::new();
        let count = self.stdout.read_line(&mut line).await?;
        if count == 0 {
            return Err(anyhow!("Codex app-server closed unexpectedly"));
        }
        serde_json::from_str(&line).context("decode Codex app-server message")
    }

    async fn expect_response(&mut self, id: u64) -> anyhow::Result<Value> {
        loop {
            let message = self.read_message().await?;
            if message.get("id").and_then(Value::as_u64) == Some(id) {
                return Ok(message);
            }
            if message.get("id").is_some() && message.get("method").is_some() {
                return Err(anyhow!(
                    "Codex requested an unauthorized interactive action"
                ));
            }
        }
    }

    async fn stop(&mut self) {
        let _ = self.child.kill().await;
        let _ = self.child.wait().await;
    }
}

#[derive(Debug, PartialEq)]
enum StreamEvent {
    TurnStarted,
    Usage(CodexUsage),
    Delta(String),
    Completed,
    Failed(String),
    Ignore,
}

fn parse_stream_event(message: &Value) -> StreamEvent {
    match message.get("method").and_then(Value::as_str) {
        Some("turn/started") => StreamEvent::TurnStarted,
        Some("thread/tokenUsage/updated") => {
            let input_tokens = message
                .pointer("/params/tokenUsage/last/inputTokens")
                .and_then(Value::as_u64);
            let output_tokens = message
                .pointer("/params/tokenUsage/last/outputTokens")
                .and_then(Value::as_u64);
            match (input_tokens, output_tokens) {
                (Some(input_tokens), Some(output_tokens)) => StreamEvent::Usage(CodexUsage {
                    input_tokens,
                    output_tokens,
                }),
                _ => StreamEvent::Ignore,
            }
        }
        Some("item/agentMessage/delta") => message
            .pointer("/params/delta")
            .and_then(Value::as_str)
            .map_or(StreamEvent::Ignore, |value| {
                StreamEvent::Delta(value.to_string())
            }),
        Some("turn/completed") => {
            let status = message
                .pointer("/params/turn/status")
                .and_then(Value::as_str);
            if status.is_some_and(|value| value != "completed") {
                let reason = message
                    .pointer("/params/turn/error/message")
                    .and_then(Value::as_str)
                    .unwrap_or("Codex turn failed");
                StreamEvent::Failed(reason.to_string())
            } else {
                StreamEvent::Completed
            }
        }
        _ => StreamEvent::Ignore,
    }
}

fn protocol_error(message: &Value, fallback: &str) -> anyhow::Error {
    let detail = message
        .pointer("/error/message")
        .and_then(Value::as_str)
        .unwrap_or(fallback);
    anyhow!(detail.to_string())
}

fn codex_auth_path() -> Option<PathBuf> {
    std::env::var_os("CODEX_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".codex")))
        .map(|home| home.join("auth.json"))
}

fn command_exists(command: &str) -> bool {
    std::env::var_os("PATH")
        .is_some_and(|paths| std::env::split_paths(&paths).any(|path| path.join(command).is_file()))
}

#[cfg(unix)]
fn link_auth(source: &Path, target: &Path) -> std::io::Result<()> {
    std::os::unix::fs::symlink(source, target)
}

fn prepare_codex_home(codex_root: &Path, auth: &Path) -> anyhow::Result<()> {
    std::fs::create_dir_all(codex_root).context("create isolated Codex state directory")?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;
        std::fs::set_permissions(codex_root, std::fs::Permissions::from_mode(0o700))?;
    }
    let target = codex_root.join("auth.json");
    if target.exists() || target.is_symlink() {
        let existing = std::fs::canonicalize(&target).context("resolve Codex auth reference")?;
        let expected = std::fs::canonicalize(auth).context("resolve source Codex auth")?;
        if existing != expected {
            return Err(anyhow!(
                "isolated Codex state contains an unexpected auth reference"
            ));
        }
        return Ok(());
    }
    link_auth(auth, &target).context("link Codex authentication into isolated state")
}

#[cfg(not(unix))]
fn link_auth(source: &Path, target: &Path) -> std::io::Result<()> {
    std::fs::copy(source, target).map(|_| ())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn composes_only_explicit_context_and_history() {
        let history = vec![ChatMessage {
            role: "user".into(),
            content: "first".into(),
        }];
        let input = compose_input("now", Some("attached"), &history);
        assert!(input.contains("User: first"));
        assert!(input.contains("<context>\nattached\n</context>"));
        assert!(input.ends_with("Current turn:\nnow"));
    }

    #[test]
    fn parses_stream_delta_and_failure() {
        assert_eq!(
            parse_stream_event(&json!({
                "method":"thread/tokenUsage/updated",
                "params":{"tokenUsage":{"last":{"inputTokens":12,"outputTokens":7}}}
            })),
            StreamEvent::Usage(CodexUsage {
                input_tokens: 12,
                output_tokens: 7,
            })
        );
        assert_eq!(
            parse_stream_event(
                &json!({"method":"item/agentMessage/delta","params":{"delta":"hello"}})
            ),
            StreamEvent::Delta("hello".into())
        );
        assert_eq!(
            parse_stream_event(
                &json!({"method":"turn/completed","params":{"turn":{"status":"failed","error":{"message":"nope"}}}})
            ),
            StreamEvent::Failed("nope".into())
        );
        assert_eq!(
            parse_stream_event(&json!({
                "method":"thread/tokenUsage/updated",
                "params":{"tokenUsage":{"last":{"inputTokens":12}}}
            })),
            StreamEvent::Ignore
        );
    }

    #[test]
    fn preserves_streamed_assistant_text_in_typed_failure() {
        let mut output = String::new();
        let mut usage = None;
        let mut observed = Vec::new();
        let mut on_event = |event| {
            if let CodexEvent::Delta(delta) = event {
                observed.push(delta);
            }
        };

        assert!(!apply_stream_event(
            StreamEvent::Delta("partial ".into()),
            &mut output,
            &mut usage,
            &mut on_event,
        )
        .unwrap());
        assert!(!apply_stream_event(
            StreamEvent::Delta("answer".into()),
            &mut output,
            &mut usage,
            &mut on_event,
        )
        .unwrap());

        let failure = apply_stream_event(
            StreamEvent::Failed("provider failed".into()),
            &mut output,
            &mut usage,
            &mut on_event,
        )
        .unwrap_err();

        assert_eq!(failure.to_string(), "provider failed");
        assert_eq!(failure.partial_output(), "partial answer");
        assert_eq!(observed, ["partial ", "answer"]);
        assert!(output.is_empty());
    }

    #[test]
    #[cfg(unix)]
    fn durable_home_is_private_and_refuses_a_different_auth_reference() {
        use std::os::unix::fs::PermissionsExt as _;

        let directory = tempfile::tempdir().unwrap();
        let first = directory.path().join("first-auth.json");
        let second = directory.path().join("second-auth.json");
        std::fs::write(&first, "first").unwrap();
        std::fs::write(&second, "second").unwrap();
        let home = directory.path().join("isolated");

        prepare_codex_home(&home, &first).unwrap();
        assert_eq!(
            std::fs::metadata(&home).unwrap().permissions().mode() & 0o777,
            0o700
        );
        assert_eq!(
            std::fs::canonicalize(home.join("auth.json")).unwrap(),
            std::fs::canonicalize(&first).unwrap()
        );
        assert!(prepare_codex_home(&home, &second).is_err());
    }

    #[test]
    fn child_environment_excludes_credentials_and_agent_configuration() {
        assert!(CHILD_ENV_ALLOWLIST.contains(&"PATH"));
        for forbidden in [
            "OPENROUTER_API_KEY",
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "CODEX_HOME",
            "RUST_LOG",
        ] {
            assert!(!CHILD_ENV_ALLOWLIST.contains(&forbidden));
        }
    }
}
