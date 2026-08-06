/// Background task registry — tracks spawned subagent runs.
///
/// A parent agent calls `spawn_agent` and gets an `AgentTaskId` back
/// immediately. The subagent runs concurrently in a tokio task.
/// The parent can poll with `agent_status` or block with `agent_join`.
///
/// # Depth and budget
///
/// Every spawned task carries a `depth` counter (root = 0). The hard cap is
/// `MAX_SPAWN_DEPTH = 4`. Budget is an explicit per-agent maximum turn count,
/// using the spawn request, agent definition, or default rather than a fraction
/// of the parent's remaining turns.
///
/// # Thread safety
///
/// `TaskRegistry` is `Clone + Send + Sync` — it wraps an `Arc<Mutex<_>>`
/// so it can be handed to the TUI, the agent loop, and tool executors.
use std::collections::HashMap;
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc, Mutex, MutexGuard,
};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use tokio::sync::{mpsc, oneshot};

/// Maximum recursive spawn depth. Beyond this, `spawn_agent` returns an error.
pub const MAX_SPAWN_DEPTH: u32 = 4;

// ---------------------------------------------------------------------------
// Git worktree helpers
// ---------------------------------------------------------------------------

/// Allocate a temporary git worktree for prompt-directed task routing.
/// Returns `(worktree_path, branch_name)` on success.
pub fn create_worktree(
    repo_root: &std::path::Path,
    task_id: &AgentTaskId,
) -> Result<(std::path::PathBuf, String), String> {
    let branch = format!("piku-agent-{}", &task_id.0[..16.min(task_id.0.len())]);
    let wt_path = std::env::temp_dir().join(format!("piku-wt-{}", task_id.0));

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

/// Drop-based, best-effort cleanup for worktrees. If the owning future unwinds
/// or is dropped before explicit cleanup, the Drop implementation attempts to
/// remove the worktree and branch. It cannot run after process termination.
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

    #[must_use]
    pub fn wt_path(&self) -> &std::path::Path {
        &self.wt_path
    }

    #[must_use]
    pub fn branch(&self) -> &str {
        &self.branch
    }
}

impl Drop for WorktreeGuard {
    fn drop(&mut self) {
        if self.armed {
            // Early-drop path: unconditionally clean up. Any partial
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
        static NEXT_ID: AtomicU64 = AtomicU64::new(0);

        // Put the process-local counter before the timestamp because
        // worktree branch names use a truncated task id prefix.
        let sequence = NEXT_ID.fetch_add(1, Ordering::Relaxed);
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let pid = std::process::id();
        Self(format!("agent-{pid}-{sequence:x}-{nanos:x}"))
    }
}

impl std::fmt::Display for AgentTaskId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

#[cfg(test)]
mod task_id_tests {
    use super::*;

    #[test]
    fn task_ids_are_unique_in_tight_loop() {
        let mut seen = std::collections::HashSet::new();

        for _ in 0..1024 {
            let id = AgentTaskId::new();
            assert!(seen.insert(id), "duplicate task id generated");
        }
    }

    #[test]
    fn persistent_registry_writes_a_typed_parent_child_link() {
        let root = tempfile::tempdir().unwrap();
        let registry = TaskRegistry::with_persistence(
            "parent-session",
            root.path().join("sessions"),
            root.path().join("runs"),
            root.path().join("links"),
        );
        let id = registry.register("child".into(), "inspect".into(), 1, None);

        registry.persist_evidence_link(&id).unwrap();

        let evidence = registry.evidence(&id).unwrap();
        let decoded: SubagentEvidence =
            serde_json::from_slice(&std::fs::read(&evidence.link_path).unwrap()).unwrap();
        assert_eq!(decoded, evidence);
        assert_eq!(decoded.parent_session_id, "parent-session");
        assert_eq!(decoded.child_session_id, format!("subagent-{id}"));

        let nested = registry.register_for_parent(
            Some(&decoded.child_session_id),
            "grandchild".into(),
            "inspect deeper".into(),
            2,
            None,
        );
        assert_eq!(
            registry.evidence(&nested).unwrap().parent_session_id,
            decoded.child_session_id
        );
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
    /// Durable relationship and evidence locations when persistence is configured.
    pub evidence: Option<SubagentEvidence>,
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

/// Typed relationship between a parent run and one spawned child run.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SubagentEvidence {
    pub schema_version: u32,
    pub task_id: AgentTaskId,
    pub parent_session_id: String,
    pub child_session_id: String,
    pub session_path: std::path::PathBuf,
    pub run_record_path: std::path::PathBuf,
    pub link_path: std::path::PathBuf,
}

#[derive(Debug, Clone)]
struct SubagentPersistence {
    parent_session_id: String,
    sessions_dir: std::path::PathBuf,
    runs_dir: std::path::PathBuf,
    links_dir: std::path::PathBuf,
    /// Parent run record, used to compute run-relative child references.
    parent_run_path: Option<std::path::PathBuf>,
}

struct RegistryInner {
    tasks: HashMap<AgentTaskId, TaskEntry>,
    persistence: Option<SubagentPersistence>,
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
                persistence: None,
                waiters: HashMap::new(),
                notification_tx: None,
            })),
        }
    }

    /// Create a registry whose child sessions, run records, and parent links
    /// survive the process that spawned them.
    #[must_use]
    pub fn with_persistence(
        parent_session_id: impl Into<String>,
        sessions_dir: impl Into<std::path::PathBuf>,
        runs_dir: impl Into<std::path::PathBuf>,
        links_dir: impl Into<std::path::PathBuf>,
    ) -> Self {
        Self::with_persistence_run_path(parent_session_id, sessions_dir, runs_dir, links_dir, None)
    }

    /// Like [`with_persistence`], but records the parent run record so child
    /// references can be expressed relative to it.
    #[must_use]
    pub fn with_persistence_run_path(
        parent_session_id: impl Into<String>,
        sessions_dir: impl Into<std::path::PathBuf>,
        runs_dir: impl Into<std::path::PathBuf>,
        links_dir: impl Into<std::path::PathBuf>,
        parent_run_path: Option<std::path::PathBuf>,
    ) -> Self {
        let registry = Self::new();
        registry.inner().persistence = Some(SubagentPersistence {
            parent_session_id: parent_session_id.into(),
            sessions_dir: sessions_dir.into(),
            runs_dir: runs_dir.into(),
            links_dir: links_dir.into(),
            parent_run_path,
        });
        registry
    }

    /// Compute child run-record and session references relative to the parent
    /// run record, if persistence and a parent run path are configured.
    #[must_use]
    ///
    /// References stay inside the config tree: the child run lives under
    /// `runs_dir` and the session under `sessions_dir`. We fail closed (return
    /// `None`) if either path escapes that expected layout, so the durable
    /// record never names a path outside the run directory graph.
    pub fn child_refs_relative_to_parent(
        &self,
        child_session_id: &str,
    ) -> Option<(std::path::PathBuf, std::path::PathBuf)> {
        let inner = self.inner();
        let persistence = inner.persistence.as_ref()?;
        let parent_run_path = persistence.parent_run_path.as_ref()?;
        let parent_dir = parent_run_path.parent()?;
        let run_record_ref = persistence
            .runs_dir
            .join(format!("{child_session_id}.jsonl"));
        let session_ref = persistence
            .sessions_dir
            .join(format!("{child_session_id}.json"));
        let rel_run = relative_within(parent_dir, &run_record_ref)?;
        let rel_session = relative_within(parent_dir, &session_ref)?;
        if rel_run.is_absolute()
            || rel_session.is_absolute()
            || rel_run
                .components()
                .any(|c| !matches!(c, std::path::Component::Normal(_)))
            || rel_session
                .components()
                .any(|c| !matches!(c, std::path::Component::Normal(_)))
        {
            return None;
        }
        Some((rel_run, rel_session))
    }

    fn inner(&self) -> MutexGuard<'_, RegistryInner> {
        self.inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Register an interjection channel so background task completions
    /// auto-inject a notification into the parent's agent loop.
    pub fn set_notification_channel(&self, tx: mpsc::Sender<String>) {
        self.inner().notification_tx = Some(tx);
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
        self.register_for_parent(None, name, description, depth, worktree_path)
    }

    /// Register a task with the session that directly spawned it.
    #[must_use]
    pub fn register_for_parent(
        &self,
        parent_session_id: Option<&str>,
        name: String,
        description: String,
        depth: u32,
        worktree_path: Option<std::path::PathBuf>,
    ) -> AgentTaskId {
        let id = AgentTaskId::new();
        let evidence = self.inner().persistence.as_ref().map(|persistence| {
            let child_session_id = format!("subagent-{id}");
            SubagentEvidence {
                schema_version: 1,
                task_id: id.clone(),
                parent_session_id: parent_session_id
                    .unwrap_or(&persistence.parent_session_id)
                    .to_string(),
                session_path: persistence
                    .sessions_dir
                    .join(format!("{child_session_id}.json")),
                run_record_path: persistence
                    .runs_dir
                    .join(format!("{child_session_id}.jsonl")),
                link_path: persistence.links_dir.join(format!("{id}.json")),
                child_session_id,
            }
        });
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
            evidence,
        };
        self.inner().tasks.insert(id.clone(), entry);
        id
    }

    /// Return the durable evidence contract for a spawned task, if enabled.
    #[must_use]
    pub fn evidence(&self, id: &AgentTaskId) -> Option<SubagentEvidence> {
        self.inner()
            .tasks
            .get(id)
            .and_then(|entry| entry.evidence.clone())
    }

    /// Atomically persist the typed parent-child relationship.
    pub fn persist_evidence_link(&self, id: &AgentTaskId) -> std::io::Result<()> {
        let evidence = self
            .evidence(id)
            .ok_or_else(|| std::io::Error::other("subagent persistence is not configured"))?;
        if let Some(parent) = evidence.link_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let json = serde_json::to_vec_pretty(&evidence).map_err(std::io::Error::other)?;
        let tmp = evidence
            .link_path
            .with_extension(format!("json.tmp.{}", std::process::id()));
        std::fs::write(&tmp, json)?;
        std::fs::rename(tmp, &evidence.link_path)
    }

    /// Mark a task as complete with its final output.
    pub fn complete(&self, id: &AgentTaskId, output: &str, turns_used: u32) {
        let mut inner = self.inner();
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
        let mut inner = self.inner();
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
        self.inner().tasks.get(id).cloned()
    }

    /// All tasks, sorted by start time (most recent first).
    #[must_use]
    pub fn all(&self) -> Vec<TaskEntry> {
        let inner = self.inner();
        let mut tasks: Vec<_> = inner.tasks.values().cloned().collect();
        tasks.sort_by_key(|t| std::cmp::Reverse(t.started_at));
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
        let mut inner = self.inner();
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

/// Compute `target` relative to `base` when both share an ancestry, or `None`
/// if `target` is not nested under `base`. Used to keep durable child
/// references inside the run-record graph instead of leaking absolute paths.
fn relative_within(base: &std::path::Path, target: &std::path::Path) -> Option<std::path::PathBuf> {
    let base_components: Vec<_> = base.components().collect();
    let target_components: Vec<_> = target.components().collect();
    if target_components.len() < base_components.len() {
        return None;
    }
    for (base_part, target_part) in base_components.iter().zip(&target_components) {
        if base_part != target_part {
            return None;
        }
    }
    let mut relative = std::path::PathBuf::new();
    for part in &target_components[base_components.len()..] {
        relative.push(part.as_os_str());
    }
    Some(relative)
}
