/// `read_memory` and `write_memory` tools.
///
/// Gives agents persistent storage across sessions via MEMORY.md files.
/// Three scopes: user (~/.config/piku/memory/), project (.piku/memory/),
/// local (.piku/memory-local/).
use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::{Destructiveness, ToolResult};

// Memory paths — duplicated from piku-runtime::memory to avoid circular deps.
fn memory_dir(scope: &str, cwd: &Path) -> PathBuf {
    match scope {
        "user" => {
            let base = std::env::var("XDG_CONFIG_HOME").map_or_else(
                |_| {
                    std::env::var("HOME").map_or_else(
                        |_| PathBuf::from(".config"),
                        |h| PathBuf::from(h).join(".config"),
                    )
                },
                PathBuf::from,
            );
            base.join("piku").join("memory")
        }
        "local" => cwd.join(".piku").join("memory-local"),
        _ => cwd.join(".piku").join("memory"), // project (default)
    }
}

#[derive(Debug, Deserialize)]
struct ReadMemoryParams {
    /// Scope: "user", "project", or "local". Default "project".
    scope: Option<String>,
}

#[derive(Debug, Deserialize)]
struct WriteMemoryParams {
    /// Markdown entry to write. Should start with a `## Heading` for merging.
    entry: String,
    /// Scope: "user", "project", or "local". Default "project".
    scope: Option<String>,
}

#[must_use]
pub fn read_memory_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "scope": {
                "type": "string",
                "enum": ["user", "project", "local"],
                "description": "Memory scope. 'project' is project-local (default). 'user' is cross-project personal memory. 'local' is gitignored project memory."
            }
        }
    })
}

#[must_use]
pub fn read_memory_destructiveness(_params: &serde_json::Value) -> Destructiveness {
    Destructiveness::Safe
}

#[must_use]
pub fn execute_read_memory(params: serde_json::Value) -> ToolResult {
    let p: ReadMemoryParams = match serde_json::from_value(params) {
        Ok(v) => v,
        Err(e) => return ToolResult::error(format!("invalid params: {e}")),
    };
    let scope = p.scope.as_deref().unwrap_or("project");
    let cwd = std::env::current_dir().unwrap_or_default();
    let path = memory_dir(scope, &cwd).join("MEMORY.md");
    match std::fs::read_to_string(&path) {
        Ok(c) if !c.trim().is_empty() => ToolResult::ok(c),
        _ => ToolResult::ok(format!(
            "(no memory in {scope} scope — create some with write_memory)"
        )),
    }
}

#[must_use]
pub fn write_memory_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "entry": {
                "type": "string",
                "description": "Markdown entry to save. Start with '## Heading' to allow section replacement. Plain text is appended as-is."
            },
            "scope": {
                "type": "string",
                "enum": ["user", "project", "local"],
                "description": "Memory scope (default 'project')."
            }
        },
        "required": ["entry"]
    })
}

#[must_use]
pub fn write_memory_destructiveness(_params: &serde_json::Value) -> Destructiveness {
    Destructiveness::Safe
}

#[must_use]
pub fn execute_write_memory(params: serde_json::Value) -> ToolResult {
    let p: WriteMemoryParams = match serde_json::from_value(params) {
        Ok(v) => v,
        Err(e) => return ToolResult::error(format!("invalid params: {e}")),
    };
    if p.entry.trim().is_empty() {
        return ToolResult::error("entry must not be empty".to_string());
    }
    let scope = p.scope.as_deref().unwrap_or("project");
    let cwd = std::env::current_dir().unwrap_or_default();
    let dir = memory_dir(scope, &cwd);
    if let Err(e) = std::fs::create_dir_all(&dir) {
        return ToolResult::error(format!("create_dir_all: {e}"));
    }
    let path = dir.join("MEMORY.md");
    let existing = std::fs::read_to_string(&path).unwrap_or_default();

    // Replace section if heading exists, otherwise append
    let heading = p.entry.lines().find(|l| l.starts_with("## "));
    let updated = if let Some(h) = heading {
        replace_section(&existing, h, &p.entry)
    } else if existing.trim().is_empty() {
        p.entry.clone()
    } else {
        let mut s = existing.trim_end().to_string();
        s.push_str("\n\n");
        s.push_str(&p.entry);
        s
    };

    match std::fs::write(&path, updated) {
        Ok(()) => ToolResult::ok(format!("memory written ({scope} scope)")),
        Err(e) => ToolResult::error(format!("write failed: {e}")),
    }
}

fn replace_section(existing: &str, heading: &str, entry: &str) -> String {
    let mut search = heading.to_string();
    search.push('\n');
    if let Some(start) = existing.find(&search) {
        let after = &existing[start + search.len()..];
        let end = after
            .find("\n## ")
            .map_or(existing.len(), |p| start + search.len() + p + 1);
        format!("{}{}\n{}", &existing[..start], entry, &existing[end..])
    } else if existing.trim().is_empty() {
        entry.to_string()
    } else {
        let mut s = existing.trim_end().to_string();
        s.push_str("\n\n");
        s.push_str(entry);
        s
    }
}
