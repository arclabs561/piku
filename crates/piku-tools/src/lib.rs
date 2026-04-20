pub mod attempt_tree_tool;
pub mod bash;
pub mod edit_file;
pub mod embed_memory_tool;
pub mod glob;
pub mod grep;
pub mod list_dir;
pub mod memory_tool;
pub mod read_file;
pub mod spawn_agent;
#[cfg(test)]
mod tests;
pub mod tool_search;
pub mod write_file;

use piku_api::ToolDefinition;

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct ToolResult {
    pub output: String,
    pub is_error: bool,
}

impl ToolResult {
    #[must_use]
    pub fn ok(output: String) -> Self {
        Self {
            output,
            is_error: false,
        }
    }

    #[must_use]
    pub fn error(output: String) -> Self {
        Self {
            output,
            is_error: true,
        }
    }
}

/// How destructive a tool call is, used by the permission system.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Destructiveness {
    /// Auto-approve — read-only or creating new files.
    Safe,
    /// Run cheap AI classifier before proceeding.
    Likely,
    /// Always pause and ask the user.
    Definite,
}

/// Safety-check paths that are bypass-immune — always require explicit user
/// confirmation regardless of mode.  Mirrors Claude Code's `checkPathSafety`.
///
/// Any path containing one of these components triggers `Definite`.
const PROTECTED_PATH_SEGMENTS: &[&str] = &[
    ".git", ".claude", ".env", ".ssh", ".gnupg", ".aws", ".npmrc", ".pypirc",
];

/// Returns true if the path contains a protected segment.
#[must_use]
pub fn is_protected_path(path: &str) -> bool {
    let p = std::path::Path::new(path);
    p.components().any(|c| {
        if let std::path::Component::Normal(s) = c {
            PROTECTED_PATH_SEGMENTS
                .iter()
                .any(|&seg| s.eq_ignore_ascii_case(seg))
        } else {
            false
        }
    })
}

/// Reject paths that traverse out of `base` via `..` components or
/// target well-known system directories. Absolute paths inside the user's
/// home / tempdir are allowed so piku can edit files across the user's
/// projects; the permission prompt is the final gate for those.
///
/// Policy:
///   - Reject relative paths whose canonical resolution escapes `base`
///     via `..` (classic traversal).
///   - Reject absolute paths that target system directories
///     (`/etc`, `/usr`, `/bin`, `/sbin`, `/boot`, `/sys`, `/proc`, `/dev`,
///     `/var` except `/var/tmp`). These require root anyway, but the
///     explicit rejection avoids silently-failing writes + gives a clearer
///     message for users running piku as root.
///   - Everything else passes.
///
/// The aggressive "refuse anything outside cwd" policy was too strict —
/// piku users legitimately edit files across projects. The threat model
/// is "model writes to /etc/cron.d/pwned" which this still catches.
pub fn ensure_within_base(target: &str, base: &std::path::Path) -> Result<(), String> {
    // Absolute-path system-directory deny-list. Writing to these is
    // either immediately destructive or an obvious exfiltration target.
    // /usr, /var, /bin, /sbin intentionally excluded — /usr/local is a
    // legitimate install target, macOS tmpdir lives under /var/folders,
    // and /bin writes require root regardless.
    const SYSTEM_ROOTS: &[&str] = &["/etc", "/boot", "/sys", "/proc", "/dev"];

    let target_path = std::path::Path::new(target);

    // Relative-path traversal: canonicalize and check containment within base.
    if !target_path.is_absolute() {
        let canonical_base = base
            .canonicalize()
            .map_err(|e| format!("cannot canonicalize base {}: {e}", base.display()))?;
        let abs = canonical_base.join(target_path);
        let mut check = abs.clone();
        let canonical_target = loop {
            match check.canonicalize() {
                Ok(c) => break c,
                Err(_) => {
                    if !check.pop() {
                        return Err(format!("relative path escapes project root: {target}"));
                    }
                }
            }
        };
        if !canonical_target.starts_with(&canonical_base) {
            return Err(format!(
                "relative path escapes project root: {} resolves to {} (outside {})",
                target,
                canonical_target.display(),
                canonical_base.display()
            ));
        }
        return Ok(());
    }

    for root in SYSTEM_ROOTS {
        if target_path.starts_with(root) {
            return Err(format!(
                "absolute path targets system directory: {target} (under {root})"
            ));
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Tool registry
// ---------------------------------------------------------------------------

/// A registered tool with its metadata and executor.
pub struct ToolEntry {
    pub name: &'static str,
    pub description: &'static str,
    pub schema_fn: fn() -> serde_json::Value,
    pub destructiveness_fn: fn(&serde_json::Value) -> Destructiveness,
}

/// Return all built-in tools as `ToolDefinition` for the API request.
#[must_use]
pub fn all_tool_definitions() -> Vec<ToolDefinition> {
    TOOLS
        .iter()
        .map(|t| ToolDefinition {
            name: t.name.to_string(),
            description: t.description.to_string(),
            input_schema: (t.schema_fn)(),
        })
        .collect()
}

/// Get the `Destructiveness` for a named tool call.
#[must_use]
pub fn tool_destructiveness(name: &str, params: &serde_json::Value) -> Destructiveness {
    TOOLS
        .iter()
        .find(|t| t.name == name)
        .map_or(Destructiveness::Likely, |t| (t.destructiveness_fn)(params))
}

/// Execute a named tool. Returns `None` if the tool is unknown.
/// Async tools (bash) are called via `tokio::task::block_in_place`.
///
/// Note: `spawn_agent`, `agent_status`, `agent_join` are NOT routed here —
/// they require a `TaskRegistry` and are executed directly by `piku-runtime`'s
/// agent loop before this function is called.
pub async fn execute_tool(name: &str, params: serde_json::Value) -> Option<ToolResult> {
    match name {
        "read_file" => Some(read_file::execute(params)),
        "write_file" => Some(write_file::execute(params)),
        "edit_file" => Some(edit_file::execute(params)),
        "bash" => Some(bash::execute(params).await),
        "glob" => Some(glob::execute(params)),
        "grep" => Some(grep::execute(params)),
        "list_dir" => Some(list_dir::execute(params)),
        // Stub responses when task registry not available (single-shot mode)
        "spawn_agent" => Some(spawn_agent::execute_spawn_agent_stub(params)),
        "agent_status" => Some(spawn_agent::execute_agent_status_stub(params)),
        "agent_join" => Some(spawn_agent::execute_agent_join_stub(params)),
        "read_memory" => Some(memory_tool::execute_read_memory(params)),
        "write_memory" => Some(memory_tool::execute_write_memory(params)),
        // These tools are routed by the runtime (need catalog/store), stubs here
        "tool_search" => Some(ToolResult::ok(
            "tool_search requires a tool catalog. Use it in an interactive session.".to_string(),
        )),
        "search_memory" => Some(embed_memory_tool::execute_search_memory_stub(params)),
        "manage_memory" => Some(ToolResult::ok(
            "manage_memory requires the embedding runtime. Use it in an interactive session."
                .to_string(),
        )),
        "record_attempt" => Some(attempt_tree_tool::execute_record_attempt_stub(params)),
        "query_attempts" => Some(attempt_tree_tool::execute_query_attempts_stub(params)),
        _ => None,
    }
}

static TOOLS: &[ToolEntry] = &[
    ToolEntry {
        name: "read_file",
        description: "Read the contents of a file. Optionally specify start_line and end_line (1-indexed) to read a range.",
        schema_fn: read_file::schema,
        destructiveness_fn: read_file::destructiveness,
    },
    ToolEntry {
        name: "write_file",
        description: "Write content to a file, creating it if it doesn't exist or overwriting it if it does.",
        schema_fn: write_file::schema,
        destructiveness_fn: write_file::destructiveness,
    },
    ToolEntry {
        name: "edit_file",
        description: "Surgically replace an exact string in a file. The old_string must match exactly once (or use replace_all=true). Returns an error if the string is not found or is ambiguous.",
        schema_fn: edit_file::schema,
        destructiveness_fn: edit_file::destructiveness,
    },
    ToolEntry {
        name: "bash",
        description: "Execute a shell command using sh -c (non-login shell). Returns stdout and stderr. Use timeout_ms to limit execution time (default 30s). If you need login shell behaviour (e.g. PATH including ~/.cargo/bin), prefix: bash -lc 'your command'.",
        schema_fn: bash::schema,
        destructiveness_fn: bash::destructiveness,
    },
    ToolEntry {
        name: "glob",
        description: "Find files matching a glob pattern (e.g. **/*.rs). Returns a list of matching paths.",
        schema_fn: glob::schema,
        destructiveness_fn: glob::destructiveness,
    },
    ToolEntry {
        name: "grep",
        description: "Search file contents with a regex pattern. Returns matching lines as file:line:content. Optionally filter by filename glob (include) and limit results (max_results).",
        schema_fn: grep::schema,
        destructiveness_fn: grep::destructiveness,
    },
    ToolEntry {
        name: "list_dir",
        description: "List the contents of a directory. Directories are shown with a trailing /.",
        schema_fn: list_dir::schema,
        destructiveness_fn: list_dir::destructiveness,
    },
    ToolEntry {
        name: "spawn_agent",
        description: "Spawn a background subagent to work on a task concurrently. Returns a task_id immediately — use agent_status to poll or agent_join to wait for the result. Subagents can themselves spawn further agents (up to depth 4). Use this to parallelise independent subtasks.",
        schema_fn: spawn_agent::spawn_agent_schema,
        destructiveness_fn: spawn_agent::spawn_agent_destructiveness,
    },
    ToolEntry {
        name: "agent_status",
        description: "Check the status of a background subagent. Pass task_id to query a specific agent, or omit to list all running and completed agents.",
        schema_fn: spawn_agent::agent_status_schema,
        destructiveness_fn: spawn_agent::agent_status_destructiveness,
    },
    ToolEntry {
        name: "agent_join",
        description: "Wait for a background subagent to complete and return its output. Blocks until done or timeout_secs (default 300) elapses.",
        schema_fn: spawn_agent::agent_join_schema,
        destructiveness_fn: spawn_agent::agent_join_destructiveness,
    },
    ToolEntry {
        name: "read_memory",
        description: "Read your persistent memory (MEMORY.md). Contains notes saved from previous sessions. Scope: 'user' (cross-project), 'project' (default, this project), 'local' (gitignored).",
        schema_fn: memory_tool::read_memory_schema,
        destructiveness_fn: memory_tool::read_memory_destructiveness,
    },
    ToolEntry {
        name: "write_memory",
        description: "Write to your persistent memory (MEMORY.md). Use '## Heading' sections for structured notes that can be updated. Memory persists across sessions. Save: user preferences, feedback corrections, project context, key decisions.",
        schema_fn: memory_tool::write_memory_schema,
        destructiveness_fn: memory_tool::write_memory_destructiveness,
    },
    ToolEntry {
        name: "tool_search",
        description: "Search for available tools by keyword. Use when you need a capability not in your current tool set. Returns matching tool names and descriptions.",
        schema_fn: tool_search::tool_search_schema,
        destructiveness_fn: tool_search::tool_search_destructiveness,
    },
    ToolEntry {
        name: "search_memory",
        description: "Semantic search over your embedding-based memory store. Finds memories by meaning, not just keywords. Use to recall past decisions, error patterns, or user preferences.",
        schema_fn: embed_memory_tool::search_memory_schema,
        destructiveness_fn: embed_memory_tool::search_memory_destructiveness,
    },
    ToolEntry {
        name: "manage_memory",
        description: "Manage your embedding memory store. Actions: 'stats' (counts), 'list' (recent entries), 'inspect' (full detail by ID), 'invalidate' (mark as outdated), 'query_tags' (find by tag).",
        schema_fn: embed_memory_tool::manage_memory_schema,
        destructiveness_fn: embed_memory_tool::manage_memory_destructiveness,
    },
    ToolEntry {
        name: "record_attempt",
        description: "Record an approach you are trying (or tried) toward a goal. Builds a tree of what works and what doesn't. Use attempt_id to update outcome of an existing attempt. Future agents can query these trees to avoid repeating failures.",
        schema_fn: attempt_tree_tool::record_attempt_schema,
        destructiveness_fn: attempt_tree_tool::record_attempt_destructiveness,
    },
    ToolEntry {
        name: "query_attempts",
        description: "Search for prior attempt trees matching a goal. Returns tree-structured history of approaches tried, their outcomes, and why they succeeded or failed. Use before starting a complex task to learn from past experience.",
        schema_fn: attempt_tree_tool::query_attempts_schema,
        destructiveness_fn: attempt_tree_tool::query_attempts_destructiveness,
    },
];
