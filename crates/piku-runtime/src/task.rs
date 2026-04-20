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
    remove_worktree_and_branch(repo_root, wt_path, branch);
    None
}

fn remove_worktree_and_branch(
    repo_root: &std::path::Path,
    wt_path: &std::path::Path,
    branch: &str,
) {
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
}

/// Drop-based cleanup for worktrees. If the task panics or is aborted
/// before it can call `cleanup_worktree`, the Drop impl removes the
/// worktree and branch so they don't accumulate in /tmp and `git branch -a`.
///
/// The happy path calls [`WorktreeGuard::defuse`] to surrender ownership
/// to the existing `cleanup_worktree` logic (which may keep the worktree
/// when the agent made changes).
pub struct WorktreeGuard {
    repo_root: std::path::PathBuf,
    wt_path: std::path::PathBuf,
    branch: String,
    armed: bool,
}

impl WorktreeGuard {
    #[must_use]
    pub fn new(repo_root: std::path::PathBuf, wt_path: std::path::PathBuf, branch: String) -> Self {
        Self {
            repo_root,
            wt_path,
            branch,
            armed: true,
        }
    }

    /// Surrender ownership — the caller will handle cleanup explicitly.
    /// Drop will not run the git commands.
    pub fn defuse(&mut self) {
        self.armed = false;
    }

    pub fn wt_path(&self) -> &std::path::Path {
        &self.wt_path
    }

    pub fn branch(&self) -> &str {
        &self.branch
    }
}

impl Drop for WorktreeGuard {
    fn drop(&mut self) {
        if self.armed {
            // Panic or abort path: unconditionally clean up. Any partial
            // work inside the worktree is lost, but we avoid a permanent
            // resource leak. Losing half-complete panicking-subagent work
            // is a better outcome than accumulating zombie worktrees.
            remove_worktree_and_branch(&self.repo_root, &self.wt_path, &self.branch);
        }
    }
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

#[cfg(test)]
mod worktree_guard_tests {
    use super::*;

    /// Initialize a throwaway git repo for worktree tests. Returns the path.
    fn init_repo() -> tempfile::TempDir {
        let dir = tempfile::tempdir().expect("tempdir");
        let repo_root = dir.path();
        for args in [
            &["init", "-q", "-b", "main"][..],
            &["config", "user.email", "t@t.t"][..],
            &["config", "user.name", "t"][..],
            &["commit", "--allow-empty", "-q", "-m", "init"][..],
        ] {
            let status = std::process::Command::new("git")
                .args(args)
                .current_dir(repo_root)
                .status()
                .expect("git");
            assert!(status.success(), "git {args:?} failed");
        }
        dir
    }

    /// A guard that panics inside a scope should still remove the worktree.
    #[test]
    fn guard_cleans_up_on_panic() {
        let repo = init_repo();
        let tid = AgentTaskId::new();
        let (wt_path, branch) = create_worktree(repo.path(), &tid).expect("worktree");
        assert!(wt_path.exists(), "worktree dir should exist after create");

        let wt_clone = wt_path.clone();
        let branch_clone = branch.clone();
        let repo_clone = repo.path().to_path_buf();
        // Simulate a panic inside the scope holding the guard.
        let r = std::panic::catch_unwind(move || {
            let _guard = WorktreeGuard::new(repo_clone, wt_clone, branch_clone);
            panic!("simulated subagent panic");
        });
        assert!(r.is_err(), "panic should have propagated");

        // Guard's Drop should have removed the worktree dir.
        assert!(
            !wt_path.exists(),
            "worktree dir still exists after panic cleanup"
        );

        // Branch should also be gone.
        let branches = std::process::Command::new("git")
            .args(["branch", "-l"])
            .current_dir(repo.path())
            .output()
            .expect("git branch -l");
        let out = String::from_utf8_lossy(&branches.stdout);
        assert!(
            !out.contains(&branch),
            "branch {branch} still present after panic: {out}"
        );
    }

    /// A defused guard should leave the worktree intact so the caller can
    /// run its own `cleanup_worktree(changed=true)` logic.
    #[test]
    fn defused_guard_does_not_touch_worktree() {
        let repo = init_repo();
        let tid = AgentTaskId::new();
        let (wt_path, branch) = create_worktree(repo.path(), &tid).expect("worktree");

        {
            let mut guard =
                WorktreeGuard::new(repo.path().to_path_buf(), wt_path.clone(), branch.clone());
            guard.defuse();
        }

        assert!(
            wt_path.exists(),
            "defused guard should have left worktree alone"
        );
        // Clean up by hand so we don't leak.
        remove_worktree_and_branch(repo.path(), &wt_path, &branch);
    }
}
