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

#[cfg(test)]
mod waiter_tests;

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

/// Retire a clean worktree whose HEAD and branch are integrated in the parent.
/// `changed` is a conservative retention hint; false still checks actual Git state.
/// Returns the path when work is retained or cleanup could not finish.
#[must_use]
pub fn cleanup_worktree(
    repo_root: &std::path::Path,
    wt_path: &std::path::Path,
    branch: &str,
    changed: bool,
) -> Option<std::path::PathBuf> {
    if changed || !worktree_is_integrated(repo_root, wt_path, branch) {
        return Some(wt_path.to_path_buf());
    }
    if remove_worktree_and_branch(repo_root, wt_path, branch) {
        None
    } else {
        Some(wt_path.to_path_buf())
    }
}

// Unknown state is retained. Include ignored files: they may contain child work
// too, and `worktree remove` alone does not protect them.
fn worktree_is_integrated(
    repo_root: &std::path::Path,
    wt_path: &std::path::Path,
    branch: &str,
) -> bool {
    let Ok(status) = std::process::Command::new("git")
        .args([
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--ignored",
        ])
        .current_dir(wt_path)
        .output()
    else {
        return false;
    };
    if !status.status.success() || !status.stdout.is_empty() {
        return false;
    }
    for revision in ["HEAD", branch] {
        let Ok(tip) = std::process::Command::new("git")
            .args(["rev-parse", "--verify", revision])
            .current_dir(wt_path)
            .output()
        else {
            return false;
        };
        if !tip.status.success() {
            return false;
        }
        let Ok(integrated) = std::process::Command::new("git")
            .args([
                "merge-base",
                "--is-ancestor",
                String::from_utf8_lossy(&tip.stdout).trim(),
                "HEAD",
            ])
            .current_dir(repo_root)
            .status()
        else {
            return false;
        };
        if !integrated.success() {
            return false;
        }
    }
    true
}

fn remove_worktree_and_branch(
    repo_root: &std::path::Path,
    wt_path: &std::path::Path,
    branch: &str,
) -> bool {
    let removed = std::process::Command::new("git")
        .args(["worktree", "remove"])
        .arg(wt_path)
        .current_dir(repo_root)
        .output()
        .is_ok_and(|out| out.status.success());
    if !removed {
        return false;
    }
    std::process::Command::new("git")
        .args(["branch", "-d", branch])
        .current_dir(repo_root)
        .output()
        .is_ok_and(|out| out.status.success())
}

/// Drop-based, best-effort cleanup for worktrees. If the owning future unwinds
/// or is dropped before explicit cleanup, the Drop implementation attempts to
/// retire a clean, integrated worktree and branch. Partial or unknown work is
/// retained. It cannot run after process termination.
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
            let _ = cleanup_worktree(&self.repo_root, &self.wt_path, &self.branch, false);
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
        _tool_id: &str,
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

/// Delivery policy at one parent-child delegation edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HandoffDelivery {
    Join,
    Notice,
    Bubble,
}

/// A descendant reaches an attended ancestor only when every intervening edge
/// opted into bubbling. UI and evaluators use this same recursion rule.
#[must_use]
pub fn handoff_bubbles_to_ancestor(path: &[HandoffDelivery]) -> bool {
    !path.is_empty()
        && path
            .iter()
            .all(|delivery| *delivery == HandoffDelivery::Bubble)
}

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

// A notification is a bounded hint; agent_join retains the full result and
// durable evidence even when the bounded channel cannot accept another hint.
fn notification_preview(id: &AgentTaskId, message: &str) -> String {
    use std::fmt::Write as _;

    let mut preview: String = message.chars().take(1024).collect();
    if message.chars().count() > 1024 {
        preview.push('…');
    }
    let _ = write!(
        preview,
        "\nUse agent_join with task_id {id} for the full result and evidence."
    );
    preview
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
        if !normal_relative_reference(&rel_run) || !normal_relative_reference(&rel_session) {
            return None;
        }
        Some((rel_run, rel_session))
    }

    /// Build a durable pointer without redefining legacy parent-relative refs.
    /// A configured-root basis is allowed only for declared sibling stores and
    /// a parent record directly in the declared run store.
    pub(crate) fn child_run_reference(
        &self,
        id: &AgentTaskId,
    ) -> Option<crate::run_record::RunEvent> {
        let evidence = self.evidence(id)?;
        let (reference_root, run_record_ref, session_ref) = if let Some((run, session)) =
            self.child_refs_relative_to_parent(&evidence.child_session_id)
        {
            (None, run, session)
        } else {
            let inner = self.inner();
            let persistence = inner.persistence.as_ref()?;
            let parent_dir = persistence.parent_run_path.as_ref()?.parent()?;
            let root = persistence.runs_dir.parent()?;
            if parent_dir != persistence.runs_dir
                || persistence.sessions_dir.parent()? != root
                || persistence.sessions_dir == persistence.runs_dir
            {
                return None;
            }
            let run = relative_within(root, &evidence.run_record_path)?;
            let session = relative_within(root, &evidence.session_path)?;
            if !normal_relative_reference(&run) || !normal_relative_reference(&session) {
                return None;
            }
            (
                Some(crate::run_record::ChildReferenceRoot::ConfiguredEvidenceRoot),
                run,
                session,
            )
        };
        Some(crate::run_record::RunEvent::ChildRunRef {
            relationship: "spawn_agent".to_owned(),
            child_session_id: evidence.child_session_id,
            task_id: id.to_string(),
            reference_root,
            run_record_ref,
            session_ref,
        })
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
            let _ = tx.try_send(notification_preview(id, &msg));
        }
    }

    /// Mark a task as failed.
    pub fn fail(&self, id: &AgentTaskId, reason: &str) {
        self.fail_after_turns(id, reason, 0);
    }

    pub(crate) fn fail_after_turns(&self, id: &AgentTaskId, reason: &str, turns_used: u32) {
        let mut inner = self.inner();
        let notification = if let Some(entry) = inner.tasks.get_mut(id) {
            entry.status = TaskStatus::Failed;
            entry.turns_used = turns_used;
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
            let _ = tx.try_send(notification_preview(id, &msg));
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
    /// Unknown tasks return a closed receiver. Senders for dropped receivers
    /// are pruned on registration; terminal tasks notify immediately.
    #[must_use]
    pub fn wait_for(&self, id: &AgentTaskId) -> oneshot::Receiver<TaskEntry> {
        let (tx, rx) = oneshot::channel();
        let mut inner = self.inner();
        inner.waiters.retain(|_, waiters| {
            waiters.retain(|sender| !sender.is_closed());
            !waiters.is_empty()
        });
        let Some(entry) = inner.tasks.get(id) else {
            // Drop tx rather than retaining a sender no task can ever notify.
            return rx;
        };
        if entry.status != TaskStatus::Running {
            let _ = tx.send(entry.clone());
            return rx;
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
        std::fs::write(repo_root.join("tracked.txt"), "initial").unwrap();
        std::fs::write(repo_root.join(".gitignore"), "ignored.txt\n").unwrap();
        for args in [
            &["init", "-q", "-b", "main"][..],
            &["config", "user.email", "t@t.t"][..],
            &["config", "user.name", "t"][..],
            &["add", "tracked.txt", ".gitignore"][..],
            &["commit", "-q", "-m", "init"][..],
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

    #[test]
    fn cleanup_preserves_shell_edits_and_child_commits() {
        for command in [
            "printf work > untracked.txt",
            "printf changed > tracked.txt",
            "printf work > ignored.txt",
            "printf work > committed.txt && git add committed.txt && git commit -qm child",
        ] {
            let repo = init_repo();
            let (wt, branch) = create_worktree(repo.path(), &AgentTaskId::new()).unwrap();
            assert!(std::process::Command::new("sh")
                .args(["-c", command])
                .current_dir(&wt)
                .status()
                .unwrap()
                .success());
            assert_eq!(
                cleanup_worktree(repo.path(), &wt, &branch, false),
                Some(wt.clone()),
                "lost shell work: {command}"
            );
            assert!(wt.exists());
            // Explicit disposal of this test's fixture after proving retention.
            assert!(std::process::Command::new("git")
                .args(["worktree", "remove", "--force"])
                .arg(&wt)
                .current_dir(repo.path())
                .status()
                .unwrap()
                .success());
        }
    }

    #[test]
    fn guard_preserves_partial_work_on_early_drop() {
        let repo = init_repo();
        let (wt, branch) = create_worktree(repo.path(), &AgentTaskId::new()).unwrap();
        {
            let _guard = WorktreeGuard::new(repo.path().to_path_buf(), wt.clone(), branch);
            std::fs::write(wt.join("partial.txt"), "recover me").unwrap();
        }
        assert_eq!(
            std::fs::read_to_string(wt.join("partial.txt")).unwrap(),
            "recover me"
        );
        assert!(std::process::Command::new("git")
            .args(["worktree", "remove", "--force"])
            .arg(&wt)
            .current_dir(repo.path())
            .status()
            .unwrap()
            .success());
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

fn normal_relative_reference(path: &std::path::Path) -> bool {
    !path.as_os_str().is_empty()
        && !path.is_absolute()
        && path
            .components()
            .all(|part| matches!(part, std::path::Component::Normal(_)))
}

#[cfg(test)]
mod child_reference_tests {
    use super::*;
    use crate::run_record::{ChildReferenceRoot, RunEvent};

    #[test]
    fn reference_basis_preserves_legacy_and_handles_only_declared_siblings() {
        let root = tempfile::tempdir().unwrap();
        let runs = root.path().join("runs");
        let sessions = root.path().join("sessions");
        for (parent, expected_basis) in [
            (root.path().join("parent.jsonl"), None),
            (
                runs.join("parent.jsonl"),
                Some(ChildReferenceRoot::ConfiguredEvidenceRoot),
            ),
        ] {
            let registry = TaskRegistry::with_persistence_run_path(
                "parent",
                &sessions,
                &runs,
                root.path().join("links"),
                Some(parent),
            );
            let id = registry.register("fixture".into(), "task".into(), 1, None);
            let evidence = registry.evidence(&id).unwrap();
            assert_eq!(
                registry
                    .child_refs_relative_to_parent(&evidence.child_session_id)
                    .is_some(),
                expected_basis.is_none()
            );
            let RunEvent::ChildRunRef {
                reference_root,
                run_record_ref,
                session_ref,
                ..
            } = registry.child_run_reference(&id).unwrap()
            else {
                panic!("expected typed pointer")
            };
            assert_eq!(reference_root, expected_basis);
            assert_eq!(root.path().join(run_record_ref), evidence.run_record_path);
            assert_eq!(root.path().join(session_ref), evidence.session_path);
        }
        for (sessions, parent) in [
            (
                root.path().join("external/sessions"),
                runs.join("parent.jsonl"),
            ),
            (
                root.path().join("sessions"),
                runs.join("nested/parent.jsonl"),
            ),
            (root.path().join("../sessions"), runs.join("parent.jsonl")),
        ] {
            let registry = TaskRegistry::with_persistence_run_path(
                "parent",
                sessions,
                &runs,
                root.path().join("links"),
                Some(parent),
            );
            let id = registry.register("fixture".into(), "task".into(), 1, None);
            assert!(registry.child_run_reference(&id).is_none());
        }
    }
}
