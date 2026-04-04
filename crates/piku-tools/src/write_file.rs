use std::path::Path;

use serde::Deserialize;

use crate::{Destructiveness, ToolResult};

#[derive(Debug, Deserialize)]
pub struct WriteFileParams {
    pub path: String,
    pub content: String,
}

#[must_use]
pub fn schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "path": { "type": "string", "description": "Path to write" },
            "content": { "type": "string", "description": "Full file content to write" }
        },
        "required": ["path", "content"]
    })
}

#[must_use]
pub fn destructiveness(params: &serde_json::Value) -> Destructiveness {
    let path = params.get("path").and_then(|v| v.as_str()).unwrap_or("");
    if crate::is_protected_path(path) {
        return Destructiveness::Definite;
    }
    if Path::new(path).exists() {
        Destructiveness::Likely
    } else {
        Destructiveness::Safe
    }
}

#[must_use]
pub fn execute(params: serde_json::Value) -> ToolResult {
    let p: WriteFileParams = match serde_json::from_value(params) {
        Ok(v) => v,
        Err(e) => return ToolResult::error(format!("invalid params: {e}")),
    };

    // create parent dirs if needed
    if let Some(parent) = Path::new(&p.path).parent() {
        if !parent.as_os_str().is_empty() {
            if let Err(e) = std::fs::create_dir_all(parent) {
                return ToolResult::error(format!("write_file: create dirs: {e}"));
            }
        }
    }

    match std::fs::write(&p.path, &p.content) {
        Ok(()) => ToolResult::ok(format!("wrote {} bytes to {}", p.content.len(), p.path)),
        Err(e) => ToolResult::error(format!("write_file: {}: {e}", p.path)),
    }
}
