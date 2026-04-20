use serde::Deserialize;

use crate::{Destructiveness, ToolResult};

#[derive(Debug, Deserialize)]
pub struct ListDirParams {
    pub path: Option<String>,
}

#[must_use]
pub fn schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "path": { "type": "string", "description": "Directory to list (default: cwd)" }
        }
    })
}

#[must_use]
pub fn destructiveness(_params: &serde_json::Value) -> Destructiveness {
    Destructiveness::Safe
}

#[must_use]
pub fn execute(params: serde_json::Value) -> ToolResult {
    let p: ListDirParams = match serde_json::from_value(params) {
        Ok(v) => v,
        Err(e) => return ToolResult::error(format!("invalid params: {e}")),
    };

    let dir = p.path.unwrap_or_else(|| {
        std::env::current_dir()
            .unwrap_or_default()
            .to_string_lossy()
            .into_owned()
    });

    let entries = match std::fs::read_dir(&dir) {
        Ok(e) => e,
        Err(e) => return ToolResult::error(format!("list_dir: {dir}: {e}")),
    };

    let mut names: Vec<String> = Vec::new();
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        let suffix = if entry.file_type().is_ok_and(|t| t.is_dir()) {
            "/"
        } else {
            ""
        };
        names.push(format!("{name}{suffix}"));
    }
    names.sort();

    if names.is_empty() {
        ToolResult::ok("(empty directory)".to_string())
    } else {
        ToolResult::ok(names.join("\n"))
    }
}
