#![allow(clippy::doc_markdown)]

/// `spawn_agent` / `agent_status` / `agent_join` tools.
///
/// These tools let a running agent delegate work to background subagents
/// and collect their results. Subagents run concurrently and can themselves
/// spawn further subagents, up to `MAX_SPAWN_DEPTH`.
///
/// # Tools
///
/// - `spawn_agent`   — start a background task, returns task_id immediately
/// - `agent_status`  — poll status of one task (or list all)
/// - `agent_join`    — block until a task completes and return its output
///
/// The actual subagent execution happens in the runtime layer
/// (`piku-runtime`), which owns the provider and session. The tools here
/// are thin parameter validators that the runtime wires to real execution
/// via `TaskRegistry`.
use serde::Deserialize;

use crate::{Destructiveness, ToolResult};

// ---------------------------------------------------------------------------
// spawn_agent
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum Isolation {
    #[default]
    /// Run in the current working directory (default).
    None,
    /// Run in a temporary git worktree. Auto-cleaned if no changes; if changes
    /// are made, the worktree path and branch are returned in the result.
    Worktree,
}

#[derive(Debug, Deserialize)]
pub struct SpawnAgentParams {
    /// Natural language description of the task to perform.
    pub task: String,
    /// Short human-readable name for the agent (shown in status).
    pub name: Option<String>,
    /// Built-in agent type (e.g. "verification", "explorer").
    /// Omit for general-purpose.
    pub subagent_type: Option<String>,
    /// Optional list of file paths to include as initial context.
    #[serde(default)]
    pub context_files: Vec<String>,
    /// Maximum turns the subagent may use. Defaults to 20.
    pub max_turns: Option<u32>,
    /// Isolation mode. Use "worktree" to run in a temporary git worktree.
    #[serde(default)]
    pub isolation: Isolation,
    /// Run in background (default true). When false, agent_join is implicit.
    #[serde(default = "default_true")]
    pub background: bool,
    /// Fork mode: inherit parent session context. Default false.
    #[serde(default)]
    pub fork: bool,
}

fn default_true() -> bool {
    true
}

#[must_use]
pub fn spawn_agent_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": "Clear description of the task. Be specific — include file paths, expected outputs, and success criteria. Brief the agent like a smart colleague: explain context, what you've tried, what's in scope."
            },
            "name": {
                "type": "string",
                "description": "Short human-readable name for this agent (1-3 words). Shown in status display."
            },
            "context_files": {
                "type": "array",
                "items": { "type": "string" },
                "description": "Optional file paths to read and include as context."
            },
            "max_turns": {
                "type": "integer",
                "description": "Maximum number of turns the subagent may use (default 20, max 50)."
            },
            "subagent_type": {
                "type": "string",
                "description": "Built-in agent type: 'verification' (verify correctness, returns PASS/FAIL) or 'explorer' (read-only research). Omit for general-purpose."
            },
            "isolation": {
                "type": "string",
                "enum": ["none", "worktree"],
                "description": "Isolation mode. Use 'worktree' to run the agent in a temporary git worktree — auto-cleaned if no changes are made. Recommended for any agent that will edit files."
            },
            "background": {
                "type": "boolean",
                "description": "Run in background (default true). Set false to block until done — equivalent to spawn + immediate agent_join."
            },
            "fork": {
                "type": "boolean",
                "description": "Fork mode: inherit parent session context (default false). Use for research tasks where the agent benefits from knowing what you already know. Forks share prompt cache with the parent."
            }
        },
        "required": ["task"]
    })
}

#[must_use]
pub fn spawn_agent_destructiveness(_params: &serde_json::Value) -> Destructiveness {
    // Spawning agents is potentially destructive — they can write files and run bash.
    // Always prompt so the user knows a subagent is being started.
    Destructiveness::Likely
}

/// Parsed, validated spawn_agent parameters.
#[derive(Debug)]
pub struct ValidatedSpawnParams {
    pub task: String,
    pub name: String,
    pub subagent_type: Option<String>,
    pub context_files: Vec<String>,
    pub max_turns: u32,
    pub isolation: Isolation,
    pub background: bool,
    pub fork: bool,
}

/// Validate spawn_agent params.
pub fn validate_spawn_agent(params: serde_json::Value) -> Result<ValidatedSpawnParams, String> {
    let p: SpawnAgentParams =
        serde_json::from_value(params).map_err(|e| format!("invalid params: {e}"))?;
    if p.task.trim().is_empty() {
        return Err("task must not be empty".to_string());
    }
    let max_turns = p.max_turns.unwrap_or(20).min(50);
    let name = p.name.filter(|n| !n.trim().is_empty()).unwrap_or_else(|| {
        p.subagent_type.as_deref().map_or_else(
            || {
                p.task
                    .split_whitespace()
                    .take(4)
                    .collect::<Vec<_>>()
                    .join("-")
                    .to_lowercase()
            },
            str::to_string,
        )
    });
    Ok(ValidatedSpawnParams {
        task: p.task,
        name,
        subagent_type: p.subagent_type,
        context_files: p.context_files,
        max_turns,
        isolation: p.isolation,
        background: p.background,
        fork: p.fork,
    })
}

// ---------------------------------------------------------------------------
// agent_status
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct AgentStatusParams {
    /// Task ID to query. If omitted, lists all tasks.
    pub task_id: Option<String>,
}

#[must_use]
pub fn agent_status_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "task_id": {
                "type": "string",
                "description": "ID returned by spawn_agent. Omit to list all background tasks."
            }
        }
    })
}

#[must_use]
pub fn agent_status_destructiveness(_params: &serde_json::Value) -> Destructiveness {
    Destructiveness::Safe
}

// ---------------------------------------------------------------------------
// agent_join
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct AgentJoinParams {
    /// Task ID to wait for.
    pub task_id: String,
    /// Timeout in seconds. Defaults to 300s (5 minutes).
    pub timeout_secs: Option<u64>,
}

#[must_use]
pub fn agent_join_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "task_id": {
                "type": "string",
                "description": "ID returned by spawn_agent to wait for."
            },
            "timeout_secs": {
                "type": "integer",
                "description": "Maximum seconds to wait before returning (default 300)."
            }
        },
        "required": ["task_id"]
    })
}

#[must_use]
pub fn agent_join_destructiveness(_params: &serde_json::Value) -> Destructiveness {
    Destructiveness::Safe
}

// ---------------------------------------------------------------------------
// Stub executors (real execution happens in piku-runtime via TaskRegistry)
// These are called when the runtime has NOT wired up a TaskRegistry,
// e.g. in tests or single-shot mode.
// ---------------------------------------------------------------------------

#[must_use]
pub fn execute_spawn_agent_stub(params: serde_json::Value) -> ToolResult {
    match validate_spawn_agent(params) {
        Ok(p) => ToolResult::error(format!(
            "spawn_agent requires an interactive session with a task registry. \
             Task: {}\nStart piku in TUI mode to use background agents.",
            p.task
        )),
        Err(e) => ToolResult::error(e),
    }
}

#[must_use]
pub fn execute_agent_status_stub(_params: serde_json::Value) -> ToolResult {
    ToolResult::error(
        "agent_status requires an interactive session with a task registry.".to_string(),
    )
}

#[must_use]
pub fn execute_agent_join_stub(_params: serde_json::Value) -> ToolResult {
    ToolResult::error(
        "agent_join requires an interactive session with a task registry.".to_string(),
    )
}
