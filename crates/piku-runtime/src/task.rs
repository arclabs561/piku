/// Background task registry — tracks spawned subagent runs.
///
/// A parent agent calls `spawn_agent` and gets an `AgentTaskId` back
/// immediately. The subagent runs concurrently in a tokio task.
/// The parent can poll with `agent_status` or block with `agent_join`.
///
/// # Depth and budget
///
/// Every spawned task carries a `depth` counter (root = 0). The hard cap is
/// `MAX_SPAWN_DEPTH = 4`. Budget is expressed as max turns; the parent
/// allocates a fraction of its own remaining turns.
///
/// # Thread safety
///
/// `TaskRegistry` is `Clone + Send + Sync` — it wraps an `Arc<Mutex<_>>`
/// so it can be handed to the TUI, the agent loop, and tool executors.
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use tokio::sync::{mpsc, oneshot};

/// Maximum recursive spawn depth. Beyond this, `spawn_agent` returns an error.
pub const MAX_SPAWN_DEPTH: u32 = 4;

// ---------------------------------------------------------------------------
// Git worktree helpers
// ---------------------------------------------------------------------------

/// Create a temporary git worktree for agent isolation.
/// Returns `(worktree_path, branch_name)` on success.
pub fn create_worktree(
    repo_root: &std::path::Path,
    task_id: &AgentTaskId,
) -> Result<(std::path::PathBuf, String), String> {
    let branch = format!("piku-agent-{}", &task_id.0[..16.min(task_id.0.len())]);
    let wt_path = std::env::temp_dir().join(format!("piku-wt-{}", &task_id.0));

    let out = std::process::Command::new("git")
        .args([
            "worktree",
            "add",
            "-b",
            &branch,
            wt_path.to_str().unwrap_or("."),
            "HEAD",
        ])
        .current_dir(repo_root)
        .output()
        .map_err(|e| format!("git worktree add failed: {e}"))?;

    if !out.status.success() {
        return Err(format!(
            "git worktree add failed: {}",
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    Ok((wt_path, branch))
}

/// Remove a git worktree and its branch.
/// `changed` indicates whether the agent made any file changes.
/// Returns the worktree path if changes were made (caller can report it).
#[must_use]
pub fn cleanup_worktree(
    repo_root: &std::path::Path,
    wt_path: &std::path::Path,
    branch: &str,
    changed: bool,
) -> Option<std::path::PathBuf> {
    if changed {
        // Leave the worktree intact — return path for the parent to report
        return Some(wt_path.to_path_buf());
    }
    // No changes — clean up silently
    let _ = std::process::Command::new("git")
        .args([
            "worktree",
            "remove",
            "--force",
            wt_path.to_str().unwrap_or("."),
        ])
        .current_dir(repo_root)
        .output();
    let _ = std::process::Command::new("git")
        .args(["branch", "-D", branch])
        .current_dir(repo_root)
        .output();
    None
}

// ---------------------------------------------------------------------------
// DevNullSink — discards all output from background subagents
// ---------------------------------------------------------------------------

/// An `OutputSink` that silently discards everything.
/// Used by background subagents that don't have a terminal to write to.
pub struct DevNullSink;

impl crate::agent_loop::OutputSink for DevNullSink {
    fn on_text(&mut self, _text: &str) {}
    fn on_tool_start(&mut self, _tool_name: &str, _tool_id: &str, _input: &serde_json::Value) {}
    fn on_tool_end(
        &mut self,
        _tool_name: &str,
        _result: &str,
        _is_error: bool,
    ) -> crate::agent_loop::PostToolAction {
        crate::agent_loop::PostToolAction::Continue
    }
    fn on_permission_denied(&mut self, _tool_name: &str, _reason: &str) {}
    fn on_turn_complete(&mut self, _usage: &piku_api::TokenUsage, _iterations: u32) {}
}

/// Default max turns for a spawned subagent.
pub const DEFAULT_SUBAGENT_MAX_TURNS: u32 = 20;

// ---------------------------------------------------------------------------
// IDs
// ---------------------------------------------------------------------------

/// Opaque identifier for a background task.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AgentTaskId(pub String);

impl Default for AgentTaskId {
    fn default() -> Self {
        Self::new()
    }
}

impl AgentTaskId {
    #[must_use]
    pub fn new() -> Self {
        // short unique id: timestamp-nanos + random suffix
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .subsec_nanos();
        let pid = std::process::id();
        Self(format!("agent-{pid}-{nanos:08x}"))
    }
}

impl std::fmt::Display for AgentTaskId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

// ---------------------------------------------------------------------------
// Task state
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum TaskStatus {
    Running,
    Done,
    Failed,
}

impl std::fmt::Display for TaskStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Running => write!(f, "running"),
            Self::Done => write!(f, "done"),
            Self::Failed => write!(f, "failed"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct TaskEntry {
    pub id: AgentTaskId,
    pub name: String,
    pub description: String,
    pub status: TaskStatus,
    pub depth: u32,
    pub started_at: Instant,
    /// Final output — set when status transitions to Done or Failed.
    pub output: Option<String>,
    /// Turn count used.
    pub turns_used: u32,
    /// Worktree path if isolation=worktree was requested.
    pub worktree_path: Option<std::path::PathBuf>,
}

impl TaskEntry {
    #[must_use]
    pub fn elapsed(&self) -> Duration {
        self.started_at.elapsed()
    }
}

// ---------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct TaskRegistry {
    inner: Arc<Mutex<RegistryInner>>,
}

struct RegistryInner {
    tasks: HashMap<AgentTaskId, TaskEntry>,
    /// Completions channel — callers waiting on join receive via oneshot.
    waiters: HashMap<AgentTaskId, Vec<oneshot::Sender<TaskEntry>>>,
    /// If set, background task completions inject a notification message
    /// into the parent agent's interjection channel.
    notification_tx: Option<mpsc::Sender<String>>,
}

impl Default for TaskRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl TaskRegistry {
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(RegistryInner {
                tasks: HashMap::new(),
                waiters: HashMap::new(),
                notification_tx: None,
            })),
        }
    }

    /// Register an interjection channel so background task completions
    /// auto-inject a notification into the parent's agent loop.
    pub fn set_notification_channel(&self, tx: mpsc::Sender<String>) {
        self.inner.lock().unwrap().notification_tx = Some(tx);
    }

    /// Register a new running task. Returns the id.
    #[must_use]
    pub fn register(
        &self,
        name: String,
        description: String,
        depth: u32,
        worktree_path: Option<std::path::PathBuf>,
    ) -> AgentTaskId {
        let id = AgentTaskId::new();
        let entry = TaskEntry {
            id: id.clone(),
            name,
            description,
            status: TaskStatus::Running,
            depth,
            started_at: Instant::now(),
            output: None,
            turns_used: 0,
            worktree_path,
        };
        self.inner.lock().unwrap().tasks.insert(id.clone(), entry);
        id
    }

    /// Mark a task as complete with its final output.
    pub fn complete(&self, id: &AgentTaskId, output: &str, turns_used: u32) {
        let mut inner = self.inner.lock().unwrap();
        let notification = if let Some(entry) = inner.tasks.get_mut(id) {
            entry.status = TaskStatus::Done;
            entry.output = Some(output.to_string());
            entry.turns_used = turns_used;
            Some(format!(
                "[agent \"{}\" ({}) completed in {} turns]\n\n{}",
                entry.name, id, turns_used, output
            ))
        } else {
            None
        };
        Self::notify_waiters(&mut inner, id);
        if let (Some(msg), Some(tx)) = (notification, &inner.notification_tx) {
            let _ = tx.try_send(msg);
        }
    }

    /// Mark a task as failed.
    pub fn fail(&self, id: &AgentTaskId, reason: &str) {
        let mut inner = self.inner.lock().unwrap();
        let notification = if let Some(entry) = inner.tasks.get_mut(id) {
            entry.status = TaskStatus::Failed;
            entry.output = Some(reason.to_string());
            Some(format!(
                "[agent \"{}\" ({}) failed: {}]",
                entry.name, id, reason
            ))
        } else {
            None
        };
        Self::notify_waiters(&mut inner, id);
        if let (Some(msg), Some(tx)) = (notification, &inner.notification_tx) {
            let _ = tx.try_send(msg);
        }
    }

    /// Poll the status of a task without blocking.
    #[must_use]
    pub fn status(&self, id: &AgentTaskId) -> Option<TaskEntry> {
        self.inner.lock().unwrap().tasks.get(id).cloned()
    }

    /// All tasks, sorted by start time (most recent first).
    #[must_use]
    pub fn all(&self) -> Vec<TaskEntry> {
        let inner = self.inner.lock().unwrap();
        let mut tasks: Vec<_> = inner.tasks.values().cloned().collect();
        tasks.sort_by(|a, b| b.started_at.cmp(&a.started_at));
        tasks
    }

    /// Active (running) tasks.
    #[must_use]
    pub fn running(&self) -> Vec<TaskEntry> {
        self.all()
            .into_iter()
            .filter(|t| t.status == TaskStatus::Running)
            .collect()
    }

    /// Register a waiter that will be notified when the task completes.
    /// Returns a receiver; the sender is stored in the registry.
    #[must_use]
    pub fn wait_for(&self, id: &AgentTaskId) -> oneshot::Receiver<TaskEntry> {
        let (tx, rx) = oneshot::channel();
        let mut inner = self.inner.lock().unwrap();
        // If already done, notify immediately
        if let Some(entry) = inner.tasks.get(id) {
            if entry.status != TaskStatus::Running {
                let _ = tx.send(entry.clone());
                return rx;
            }
        }
        inner.waiters.entry(id.clone()).or_default().push(tx);
        rx
    }

    fn notify_waiters(inner: &mut RegistryInner, id: &AgentTaskId) {
        if let Some(waiters) = inner.waiters.remove(id) {
            if let Some(entry) = inner.tasks.get(id) {
                for tx in waiters {
                    let _ = tx.send(entry.clone());
                }
            }
        }
    }
}
