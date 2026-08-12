use std::collections::HashMap;
use std::io::Read as _;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::{Duration, SystemTime};
use std::{error::Error, fmt};

use anyhow::{anyhow, Context};
use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};
use tokio::sync::watch;

use piku_runtime::{RunContentChange, RunContentRef, RunEffectCategory, RunToolEffect};

use super::ChatMessage;

const DEVELOPER_INSTRUCTIONS: &str = "You are the read-only conversation executor inside Piku. Answer the user's question directly and concisely. The Piku workspace and its files are not implicit context. Do not inspect files, run commands, use tools, or mutate the workspace. Use only the conversation and optional context supplied in this turn.";
const WRITE_DEVELOPER_INSTRUCTIONS: &str = "You are the workspace-write coding executor inside Piku for one explicitly approved turn. Work only inside the provided workspace root, do not use the network, do not request broader approval, and do not access sibling paths. Use native command and file-change tools as needed. Explain what you changed, what you verified, and any uncertainty.";
const CHILD_ENV_ALLOWLIST: &[&str] = &[
    "LANG",
    "LC_ALL",
    "PATH",
    "SHELL",
    "SSL_CERT_DIR",
    "SSL_CERT_FILE",
    "TERM",
    "TMPDIR",
];
const LAUNCH_POLICY_JSON: &str = include_str!("codex-launch-policy.json");
const INTERRUPT_RESPONSE_ID: u64 = 4;
const INTERRUPT_GRACE: Duration = Duration::from_secs(2);

pub(super) fn launch_policy() -> Value {
    serde_json::from_str(LAUNCH_POLICY_JSON).expect("embedded Codex launch policy is valid JSON")
}

pub(super) fn workspace_write_attestation(codex_root: &Path) -> Result<(), &'static str> {
    let attestation = codex_root.join("workspace-write-attestation.json");
    if !attestation.is_file() {
        return Err("workspace-write attestation is unavailable");
    }
    let mut command = std::process::Command::new("codex");
    command
        .env_clear()
        .arg("--version")
        .env("CODEX_HOME", codex_root)
        .env("HOME", codex_root);
    for key in CHILD_ENV_ALLOWLIST {
        if let Some(value) = std::env::var_os(key) {
            command.env(key, value);
        }
    }
    command.stdout(Stdio::piped()).stderr(Stdio::null());
    let mut child = command
        .spawn()
        .map_err(|_| "Codex version probe is unavailable")?;
    let deadline = std::time::Instant::now() + Duration::from_secs(2);
    let status = loop {
        if let Some(status) = child.try_wait().map_err(|_| "Codex version probe failed")? {
            break status;
        }
        if std::time::Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            return Err("Codex version probe timed out");
        }
        std::thread::sleep(Duration::from_millis(20));
    };
    let mut stdout = Vec::new();
    child
        .stdout
        .take()
        .ok_or("Codex version probe has no output")?
        .take(257)
        .read_to_end(&mut stdout)
        .map_err(|_| "Codex version probe failed")?;
    if !status.success() || stdout.len() > 256 {
        return Err("Codex version probe failed");
    }
    let version = std::str::from_utf8(&stdout)
        .map_err(|_| "Codex version probe returned invalid text")?
        .trim();
    super::codex_attestation::verify(
        &attestation,
        LAUNCH_POLICY_JSON.as_bytes(),
        version,
        SystemTime::now(),
    )
}

#[derive(Debug, Clone)]
pub(super) struct CodexCancellation {
    sender: watch::Sender<bool>,
}

impl CodexCancellation {
    pub(super) fn request(&self) {
        self.sender.send_replace(true);
    }
}

pub(super) fn cancellation_channel() -> (CodexCancellation, watch::Receiver<bool>) {
    let (sender, receiver) = watch::channel(false);
    (CodexCancellation { sender }, receiver)
}

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
    ToolStarted {
        tool_call_id: String,
        name: String,
        arguments: Value,
    },
    ToolCompleted {
        tool_call_id: String,
        result: RunContentRef,
        is_error: bool,
        effects: Vec<RunToolEffect>,
    },
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum CodexTurnPolicy {
    ReadOnly,
    WorkspaceWrite,
}

impl CodexTurnPolicy {
    fn thread_sandbox(self) -> &'static str {
        match self {
            Self::ReadOnly => "read-only",
            Self::WorkspaceWrite => "workspace-write",
        }
    }

    fn developer_instructions(self) -> &'static str {
        match self {
            Self::ReadOnly => DEVELOPER_INSTRUCTIONS,
            Self::WorkspaceWrite => WRITE_DEVELOPER_INSTRUCTIONS,
        }
    }

    fn turn_sandbox(self, workspace_root: &Path) -> Value {
        match self {
            Self::ReadOnly => json!({"type":"readOnly","networkAccess":false}),
            Self::WorkspaceWrite => json!({
                "type":"workspaceWrite",
                "writableRoots":[workspace_root],
                "networkAccess":false,
                "excludeSlashTmp":true,
                "excludeTmpdirEnvVar":true
            }),
        }
    }
}

/// A native Codex turn failure plus any assistant text received before it.
///
/// The partial output is kept separate from the display message so logging the
/// error cannot accidentally duplicate conversation content.
#[derive(Debug)]
pub(super) struct CodexFailure {
    message: String,
    partial_output: String,
    interrupted: bool,
}

impl CodexFailure {
    fn new(message: impl Into<String>, partial_output: String) -> Self {
        Self {
            message: message.into(),
            partial_output,
            interrupted: false,
        }
    }

    fn interrupted(partial_output: String) -> Self {
        Self {
            message: "Codex turn was interrupted".into(),
            partial_output,
            interrupted: true,
        }
    }

    fn interruption_failed(message: impl Into<String>, partial_output: String) -> Self {
        Self {
            message: message.into(),
            partial_output,
            interrupted: true,
        }
    }

    pub(super) fn partial_output(&self) -> &str {
        &self.partial_output
    }

    pub(super) fn is_interrupted(&self) -> bool {
        self.interrupted
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
    let _launch_policy = launch_policy();
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

pub(super) async fn run_chat_cancellable<F>(
    workspace_root: &Path,
    codex_root: &Path,
    message: &str,
    context: Option<&str>,
    history: &[ChatMessage],
    thread_id: Option<&str>,
    policy: CodexTurnPolicy,
    cancellation: watch::Receiver<bool>,
    on_event: F,
) -> Result<CodexResult, CodexFailure>
where
    F: FnMut(CodexEvent),
{
    run_chat_inner(
        workspace_root,
        codex_root,
        message,
        context,
        history,
        thread_id,
        policy,
        Some(cancellation),
        on_event,
    )
    .await
}

async fn run_chat_inner<F>(
    workspace_root: &Path,
    codex_root: &Path,
    message: &str,
    context: Option<&str>,
    history: &[ChatMessage],
    thread_id: Option<&str>,
    policy: CodexTurnPolicy,
    mut cancellation: Option<watch::Receiver<bool>>,
    mut on_event: F,
) -> Result<CodexResult, CodexFailure>
where
    F: FnMut(CodexEvent),
{
    let mut server = CodexServer::spawn(codex_root)?;
    server.initialize().await?;
    let (thread, model) = if let Some(thread_id) = thread_id.filter(|id| !id.trim().is_empty()) {
        server
            .resume_thread(workspace_root, thread_id, policy)
            .await?
    } else {
        server.start_thread(workspace_root, policy).await?
    };
    let input = compose_input(message, context, history);
    let turn_id = server
        .start_turn(&thread, &input, workspace_root, policy)
        .await?;
    on_event(CodexEvent::Started {
        model: model.clone(),
        thread_id: thread.clone(),
        turn_id: turn_id.clone(),
        input,
    });

    let mut output = String::new();
    let mut usage = None;
    let mut tools = ToolLifecycle::default();
    loop {
        if cancellation
            .as_ref()
            .is_some_and(|receiver| *receiver.borrow())
        {
            let result = server.interrupt_turn(&thread, &turn_id).await;
            server.stop().await;
            return match result {
                Ok(()) => Err(CodexFailure::interrupted(output)),
                Err(error) => Err(CodexFailure::interruption_failed(
                    format!("Codex interruption did not complete cleanly: {error}"),
                    output,
                )),
            };
        }
        let message = if cancellation.is_some() {
            enum NextMessage {
                Message(anyhow::Result<Value>),
                Cancel,
                CancellationClosed,
            }
            let next = {
                let receiver = cancellation.as_mut().expect("checked above");
                tokio::select! {
                    message = server.read_message() => NextMessage::Message(message),
                    changed = receiver.changed() => {
                        if changed.is_ok() && *receiver.borrow() {
                            NextMessage::Cancel
                        } else {
                            NextMessage::CancellationClosed
                        }
                    }
                }
            };
            match next {
                NextMessage::Message(message) => {
                    message.map_err(|error| CodexFailure::new(error.to_string(), output.clone()))?
                }
                NextMessage::Cancel => {
                    let result = server.interrupt_turn(&thread, &turn_id).await;
                    server.stop().await;
                    return match result {
                        Ok(()) => Err(CodexFailure::interrupted(output)),
                        Err(error) => Err(CodexFailure::interruption_failed(
                            format!("Codex interruption did not complete cleanly: {error}"),
                            output,
                        )),
                    };
                }
                NextMessage::CancellationClosed => {
                    cancellation = None;
                    continue;
                }
            }
        } else {
            server
                .read_message()
                .await
                .map_err(|error| CodexFailure::new(error.to_string(), output.clone()))?
        };
        if message.get("id").is_some() && message.get("method").is_some() {
            return Err(CodexFailure::new(
                "Codex requested an interactive action outside Piku's no-elevation contract",
                output,
            ));
        }
        if apply_stream_event(
            parse_stream_event(&message),
            &mut output,
            &mut usage,
            &mut tools,
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
    tools: &mut ToolLifecycle,
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
        StreamEvent::ToolStarted(tool) => {
            let event = tools
                .start(tool)
                .map_err(|error| CodexFailure::new(error.to_string(), std::mem::take(output)))?;
            on_event(event);
            Ok(false)
        }
        StreamEvent::ToolCompleted(tool) => {
            let event = tools
                .complete(tool)
                .map_err(|error| CodexFailure::new(error.to_string(), std::mem::take(output)))?;
            on_event(event);
            Ok(false)
        }
        StreamEvent::TurnStarted | StreamEvent::Ignore => Ok(false),
        StreamEvent::Usage(value) => {
            *usage = Some(value);
            Ok(false)
        }
        StreamEvent::Completed => {
            if tools.active.is_empty() {
                Ok(true)
            } else {
                Err(CodexFailure::new(
                    "Codex completed the turn with unfinished tool items",
                    std::mem::take(output),
                ))
            }
        }
        StreamEvent::Interrupted => Err(CodexFailure::interrupted(std::mem::take(output))),
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
            .env("HOME", codex_root)
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
        #[cfg(unix)]
        command.process_group(0);
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

    async fn start_thread(
        &mut self,
        workspace_root: &Path,
        policy: CodexTurnPolicy,
    ) -> anyhow::Result<(String, String)> {
        self.send(json!({"method":"thread/start","id":2,"params":{
            "cwd": workspace_root,
            "sandbox":policy.thread_sandbox(),
            "approvalPolicy":"never",
            "personality":"pragmatic",
            "ephemeral":false,
            "developerInstructions":policy.developer_instructions()
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
        policy: CodexTurnPolicy,
    ) -> anyhow::Result<(String, String)> {
        self.send(json!({"method":"thread/resume","id":2,"params":{
            "threadId":thread_id,
            "cwd":workspace_root,
            "sandbox":policy.thread_sandbox(),
            "approvalPolicy":"never",
            "personality":"pragmatic",
            "developerInstructions":policy.developer_instructions()
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

    async fn start_turn(
        &mut self,
        thread_id: &str,
        input: &str,
        workspace_root: &Path,
        policy: CodexTurnPolicy,
    ) -> anyhow::Result<String> {
        self.send(json!({"method":"turn/start","id":3,"params":{
            "threadId":thread_id,
            "approvalPolicy":"never",
            "sandboxPolicy":policy.turn_sandbox(workspace_root),
            "input":[{"type":"text","text":input}]
        }}))
        .await?;
        let response = self.expect_response(3).await?;
        response
            .pointer("/result/turn/id")
            .and_then(Value::as_str)
            .map(str::to_string)
            .ok_or_else(|| protocol_error(&response, "turn/start failed"))
    }

    async fn interrupt_turn(&mut self, thread_id: &str, turn_id: &str) -> anyhow::Result<()> {
        self.send(interrupt_request(thread_id, turn_id)).await?;
        with_interrupt_timeout(INTERRUPT_GRACE, self.await_interrupted()).await?;
        Ok(())
    }

    async fn await_interrupted(&mut self) -> anyhow::Result<()> {
        let mut tracker = InterruptTracker::default();
        loop {
            let message = self.read_message().await?;
            if tracker.observe(&message)? {
                return Ok(());
            }
        }
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
        #[cfg(unix)]
        if let Some(pid) = self.child.id().and_then(|pid| i32::try_from(pid).ok()) {
            let _ = nix::sys::signal::killpg(
                nix::unistd::Pid::from_raw(pid),
                nix::sys::signal::Signal::SIGKILL,
            );
        }
        let _ = self.child.kill().await;
        let _ = self.child.wait().await;
    }
}

#[derive(Debug, PartialEq)]
enum StreamEvent {
    TurnStarted,
    Usage(CodexUsage),
    Delta(String),
    ToolStarted(NativeToolItem),
    ToolCompleted(NativeToolItem),
    Completed,
    Interrupted,
    Failed(String),
    Ignore,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NativeToolKind {
    CommandExecution,
    FileChange,
}

#[derive(Debug, PartialEq)]
struct NativeToolItem {
    id: String,
    kind: NativeToolKind,
    value: Value,
}

#[derive(Default)]
struct ToolLifecycle {
    active: HashMap<String, NativeToolKind>,
}

impl ToolLifecycle {
    fn start(&mut self, tool: NativeToolItem) -> Result<CodexEvent, CodexFailure> {
        if self.active.insert(tool.id.clone(), tool.kind).is_some() {
            return Err(CodexFailure::new(
                format!("Codex started duplicate tool item {}", tool.id),
                String::new(),
            ));
        }
        let (name, arguments) = match tool.kind {
            NativeToolKind::CommandExecution => {
                let command = required_string(&tool.value, "command", &tool.id)?;
                let cwd = tool.value.get("cwd").and_then(Value::as_str);
                (
                    "commandExecution".to_string(),
                    json!({"command": command, "cwd": cwd}),
                )
            }
            NativeToolKind::FileChange => {
                let changes = required_array(&tool.value, "changes", &tool.id)?;
                ("fileChange".to_string(), json!({"changes": changes}))
            }
        };
        Ok(CodexEvent::ToolStarted {
            tool_call_id: tool.id,
            name,
            arguments,
        })
    }

    fn complete(&mut self, tool: NativeToolItem) -> Result<CodexEvent, CodexFailure> {
        match self.active.remove(&tool.id) {
            Some(kind) if kind == tool.kind => {}
            Some(_) => {
                return Err(CodexFailure::new(
                    format!("Codex changed tool item type for {}", tool.id),
                    String::new(),
                ))
            }
            None => {
                return Err(CodexFailure::new(
                    format!("Codex completed tool item {} without a start", tool.id),
                    String::new(),
                ))
            }
        }
        let status = required_string(&tool.value, "status", &tool.id)?;
        let (result, is_error, effects) = match tool.kind {
            NativeToolKind::CommandExecution => {
                let command = required_string(&tool.value, "command", &tool.id)?.to_string();
                let exit_code = tool
                    .value
                    .get("exitCode")
                    .and_then(Value::as_i64)
                    .and_then(|value| i32::try_from(value).ok());
                let output = tool
                    .value
                    .get("aggregatedOutput")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                (
                    RunContentRef::Inline { text: output },
                    status != "completed" || exit_code.is_some_and(|code| code != 0),
                    vec![
                        RunToolEffect::ShellCommand { command, exit_code },
                        RunToolEffect::Unattributed {
                            category: RunEffectCategory::FileSystem,
                            reason: "a shell command may mutate paths that Codex native events do not enumerate"
                                .to_string(),
                        },
                    ],
                )
            }
            NativeToolKind::FileChange => {
                let changes = required_array(&tool.value, "changes", &tool.id)?;
                let effects = changes.iter().filter_map(file_change_effect).collect();
                (
                    RunContentRef::Inline {
                        text: format!("Codex reported {} file changes", changes.len()),
                    },
                    status != "completed",
                    effects,
                )
            }
        };
        Ok(CodexEvent::ToolCompleted {
            tool_call_id: tool.id,
            result,
            is_error,
            effects,
        })
    }
}

fn required_string<'a>(value: &'a Value, field: &str, id: &str) -> Result<&'a str, CodexFailure> {
    value
        .get(field)
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| {
            CodexFailure::new(
                format!("Codex tool item {id} has no {field}"),
                String::new(),
            )
        })
}

fn required_array<'a>(
    value: &'a Value,
    field: &str,
    id: &str,
) -> Result<&'a Vec<Value>, CodexFailure> {
    value.get(field).and_then(Value::as_array).ok_or_else(|| {
        CodexFailure::new(
            format!("Codex tool item {id} has no {field} array"),
            String::new(),
        )
    })
}

fn file_change_effect(change: &Value) -> Option<RunToolEffect> {
    let path = change.get("path").and_then(Value::as_str)?;
    let kind = change
        .pointer("/kind/type")
        .or_else(|| change.get("kind"))
        .and_then(Value::as_str)?;
    match kind {
        "add" | "create" => Some(RunToolEffect::FileWrite {
            path: PathBuf::from(path),
            content_change: RunContentChange::Created,
        }),
        "update" | "modify" => Some(RunToolEffect::FileWrite {
            path: PathBuf::from(path),
            content_change: RunContentChange::Modified,
        }),
        // The current schema cannot encode delete or rename as FileWrite.
        // Preserve the native report as explicit incomplete attribution.
        _ => Some(RunToolEffect::Unattributed {
            category: RunEffectCategory::FileSystem,
            reason: format!(
                "Codex reported file change kind {kind} for {path}; this effect is not representable as a file write"
            ),
        }),
    }
}

fn parse_tool_item(message: &Value) -> Option<NativeToolItem> {
    let value = message.pointer("/params/item")?.clone();
    let id = value.get("id")?.as_str()?.to_string();
    if id.is_empty() {
        return None;
    }
    let kind = match value.get("type")?.as_str()? {
        "commandExecution" => NativeToolKind::CommandExecution,
        "fileChange" => NativeToolKind::FileChange,
        _ => return None,
    };
    Some(NativeToolItem { id, kind, value })
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
        Some("item/started") => {
            parse_tool_item(message).map_or(StreamEvent::Ignore, StreamEvent::ToolStarted)
        }
        Some("item/completed") => {
            parse_tool_item(message).map_or(StreamEvent::Ignore, StreamEvent::ToolCompleted)
        }
        Some("turn/completed") => {
            let status = message
                .pointer("/params/turn/status")
                .and_then(Value::as_str);
            if status == Some("interrupted") {
                StreamEvent::Interrupted
            } else if status.is_some_and(|value| value != "completed") {
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

fn interrupt_request(thread_id: &str, turn_id: &str) -> Value {
    json!({"method":"turn/interrupt","id":INTERRUPT_RESPONSE_ID,"params":{
        "threadId":thread_id,
        "turnId":turn_id
    }})
}

#[derive(Default)]
struct InterruptTracker {
    acknowledged: bool,
    interrupted: bool,
}

impl InterruptTracker {
    fn observe(&mut self, message: &Value) -> anyhow::Result<bool> {
        if message.get("id").and_then(Value::as_u64) == Some(INTERRUPT_RESPONSE_ID) {
            if message.get("error").is_some() {
                return Err(protocol_error(message, "turn/interrupt failed"));
            }
            self.acknowledged = true;
            return Ok(self.interrupted);
        }
        if message.get("id").is_some() && message.get("method").is_some() {
            return Err(anyhow!(
                "Codex requested an unauthorized interactive action"
            ));
        }
        if parse_stream_event(message) == StreamEvent::Interrupted {
            self.interrupted = true;
            return Ok(self.acknowledged);
        }
        Ok(false)
    }
}

async fn with_interrupt_timeout<F>(duration: Duration, future: F) -> anyhow::Result<()>
where
    F: std::future::Future<Output = anyhow::Result<()>>,
{
    tokio::time::timeout(duration, future)
        .await
        .map_err(|_| anyhow!("timed out waiting for Codex interruption"))?
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
    fn turn_policies_are_explicit_and_networkless() {
        let root = Path::new("/workspace");
        assert_eq!(
            CodexTurnPolicy::ReadOnly.turn_sandbox(root),
            json!({"type":"readOnly","networkAccess":false})
        );
        assert_eq!(
            CodexTurnPolicy::WorkspaceWrite.turn_sandbox(root),
            json!({
                "type":"workspaceWrite",
                "writableRoots":["/workspace"],
                "networkAccess":false,
                "excludeSlashTmp":true,
                "excludeTmpdirEnvVar":true
            })
        );
        assert_eq!(
            CodexTurnPolicy::WorkspaceWrite.thread_sandbox(),
            "workspace-write"
        );
        assert!(CodexTurnPolicy::WorkspaceWrite
            .developer_instructions()
            .contains("one explicitly approved turn"));
    }

    #[test]
    fn parses_stream_delta_and_failure() {
        assert_eq!(
            parse_stream_event(
                &json!({"method":"turn/completed","params":{"turn":{"status":"interrupted"}}})
            ),
            StreamEvent::Interrupted
        );
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
        let mut tools = ToolLifecycle::default();
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
            &mut tools,
            &mut on_event,
        )
        .unwrap());
        assert!(!apply_stream_event(
            StreamEvent::Delta("answer".into()),
            &mut output,
            &mut usage,
            &mut tools,
            &mut on_event,
        )
        .unwrap());

        let failure = apply_stream_event(
            StreamEvent::Failed("provider failed".into()),
            &mut output,
            &mut usage,
            &mut tools,
            &mut on_event,
        )
        .unwrap_err();

        assert_eq!(failure.to_string(), "provider failed");
        assert_eq!(failure.partial_output(), "partial answer");
        assert_eq!(observed, ["partial ", "answer"]);
        assert!(output.is_empty());
    }

    #[test]
    fn projects_command_lifecycle_with_native_id() {
        let started = parse_stream_event(&json!({
            "method":"item/started",
            "params":{"item":{"id":"native-7","type":"commandExecution","status":"inProgress","command":"cargo test","cwd":"/work"}}
        }));
        let completed = parse_stream_event(&json!({
            "method":"item/completed",
            "params":{"item":{"id":"native-7","type":"commandExecution","status":"completed","command":"cargo test","cwd":"/work","exitCode":0,"aggregatedOutput":"ok"}}
        }));
        let mut tools = ToolLifecycle::default();
        let CodexEvent::ToolStarted {
            tool_call_id,
            name,
            arguments,
        } = tools
            .start(match started {
                StreamEvent::ToolStarted(tool) => tool,
                event => panic!("unexpected event: {event:?}"),
            })
            .unwrap()
        else {
            panic!("expected tool start")
        };
        assert_eq!(tool_call_id, "native-7");
        assert_eq!(name, "commandExecution");
        assert_eq!(arguments, json!({"command":"cargo test","cwd":"/work"}));

        let CodexEvent::ToolCompleted {
            tool_call_id,
            result,
            is_error,
            effects,
        } = tools
            .complete(match completed {
                StreamEvent::ToolCompleted(tool) => tool,
                event => panic!("unexpected event: {event:?}"),
            })
            .unwrap()
        else {
            panic!("expected tool completion")
        };
        assert_eq!(tool_call_id, "native-7");
        assert_eq!(result, RunContentRef::Inline { text: "ok".into() });
        assert!(!is_error);
        assert_eq!(
            effects,
            vec![
                RunToolEffect::ShellCommand {
                    command: "cargo test".into(),
                    exit_code: Some(0)
                },
                RunToolEffect::Unattributed {
                    category: RunEffectCategory::FileSystem,
                    reason:
                        "a shell command may mutate paths that Codex native events do not enumerate"
                            .into(),
                }
            ]
        );
    }

    #[test]
    fn file_changes_preserve_unrepresentable_deletes_and_renames_as_unattributed() {
        let changes = json!([
            {"path":"new.rs","kind":{"type":"add"}},
            {"path":"old.rs","kind":{"type":"delete"}},
            {"path":"from.rs","kind":{"type":"rename"}},
            {"path":"lib.rs","kind":{"type":"update"}}
        ]);
        let mut tools = ToolLifecycle::default();
        tools
            .start(NativeToolItem {
                id: "files-1".into(),
                kind: NativeToolKind::FileChange,
                value: json!({"changes":changes}),
            })
            .unwrap();
        let CodexEvent::ToolCompleted { effects, .. } = tools
            .complete(NativeToolItem {
                id: "files-1".into(),
                kind: NativeToolKind::FileChange,
                value: json!({"status":"completed","changes":changes}),
            })
            .unwrap()
        else {
            panic!("expected tool completion")
        };
        assert_eq!(
            effects,
            vec![
                RunToolEffect::FileWrite {
                    path: PathBuf::from("new.rs"),
                    content_change: RunContentChange::Created
                },
                RunToolEffect::Unattributed {
                    category: RunEffectCategory::FileSystem,
                    reason: "Codex reported file change kind delete for old.rs; this effect is not representable as a file write".into(),
                },
                RunToolEffect::Unattributed {
                    category: RunEffectCategory::FileSystem,
                    reason: "Codex reported file change kind rename for from.rs; this effect is not representable as a file write".into(),
                },
                RunToolEffect::FileWrite {
                    path: PathBuf::from("lib.rs"),
                    content_change: RunContentChange::Modified
                },
            ]
        );
    }

    #[test]
    fn rejects_duplicate_starts_and_completion_without_start() {
        let item = || NativeToolItem {
            id: "tool-1".into(),
            kind: NativeToolKind::CommandExecution,
            value: json!({"command":"true","status":"completed","exitCode":0}),
        };
        let mut tools = ToolLifecycle::default();
        tools.start(item()).unwrap();
        assert!(tools
            .start(item())
            .unwrap_err()
            .to_string()
            .contains("duplicate"));

        let mut tools = ToolLifecycle::default();
        assert!(tools
            .complete(item())
            .unwrap_err()
            .to_string()
            .contains("without a start"));
    }

    #[test]
    fn interrupt_request_has_app_server_shape() {
        assert_eq!(
            interrupt_request("thread-1", "turn-2"),
            json!({
                "method": "turn/interrupt",
                "id": 4,
                "params": {"threadId": "thread-1", "turnId": "turn-2"}
            })
        );
    }

    #[test]
    fn interruption_requires_ack_and_terminal_state_in_either_order() {
        let mut tracker = InterruptTracker::default();
        assert!(!tracker.observe(&json!({"id":4,"result":{}})).unwrap());
        assert!(tracker
            .observe(&json!({
                "method":"turn/completed",
                "params":{"turn":{"status":"interrupted"}}
            }))
            .unwrap());

        let mut terminal_first = InterruptTracker::default();
        assert!(!terminal_first
            .observe(&json!({
                "method":"turn/completed",
                "params":{"turn":{"status":"interrupted"}}
            }))
            .unwrap());
        assert!(terminal_first
            .observe(&json!({"id":4,"result":{}}))
            .unwrap());
    }

    #[test]
    fn interruption_refuses_interactive_requests() {
        let mut tracker = InterruptTracker::default();
        let error = tracker
            .observe(&json!({"id":91,"method":"item/tool/call","params":{}}))
            .unwrap_err();
        assert!(error
            .to_string()
            .contains("unauthorized interactive action"));
    }

    #[tokio::test]
    async fn interruption_timeout_is_a_bounded_fallback_seam() {
        let pending = std::future::pending::<anyhow::Result<()>>();
        let error = with_interrupt_timeout(Duration::from_millis(1), pending)
            .await
            .unwrap_err();
        assert!(error.to_string().contains("timed out"));
    }

    #[test]
    fn interrupted_failure_is_distinct_and_keeps_partial_output() {
        let mut output = "partial".to_string();
        let mut usage = None;
        let mut tools = ToolLifecycle::default();
        let failure = apply_stream_event(
            StreamEvent::Interrupted,
            &mut output,
            &mut usage,
            &mut tools,
            &mut |_| {},
        )
        .unwrap_err();
        assert!(failure.is_interrupted());
        assert_eq!(failure.partial_output(), "partial");
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
            "HOME",
            "OPENROUTER_API_KEY",
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "CODEX_HOME",
            "RUST_LOG",
        ] {
            assert!(!CHILD_ENV_ALLOWLIST.contains(&forbidden));
        }
    }

    #[test]
    fn embedded_launch_policy_matches_the_production_boundary() {
        let policy = launch_policy();
        assert_eq!(policy["schema_version"], 1);
        assert_eq!(policy["policy_id"], "piku.codex.workspace-write.v1");
        assert_eq!(policy["thread_sandbox"], "workspace-write");
        assert_eq!(policy["approval_policy"], "never");
        assert_eq!(policy["network_access"], false);
        assert_eq!(policy["exclude_slash_tmp"], true);
        assert_eq!(policy["exclude_tmpdir_env_var"], true);
        assert_eq!(policy["synthetic_home"], true);
        let allowlist = policy["child_env_allowlist"].as_array().unwrap();
        let expected = CHILD_ENV_ALLOWLIST
            .iter()
            .map(|key| Value::String((*key).to_string()))
            .collect::<Vec<_>>();
        assert_eq!(allowlist, &expected);
    }
}
