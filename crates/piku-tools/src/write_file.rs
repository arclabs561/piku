use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::{ContentChange, Destructiveness, ToolEffect, ToolResult};

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

    let cwd = std::env::current_dir().unwrap_or_default();
    let requested_path = PathBuf::from(&p.path);
    let resolved_path = if requested_path.is_absolute() {
        requested_path.clone()
    } else {
        cwd.join(&requested_path)
    };
    let content_change = match std::fs::read(&resolved_path) {
        Ok(previous) if previous == p.content.as_bytes() => ContentChange::Unchanged,
        Ok(_) => ContentChange::Modified,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => ContentChange::Created,
        Err(_) => ContentChange::Unknown,
    };

    // Limited path guard: reject traversal and selected system roots. This is
    // not workspace containment; most absolute paths remain eligible.
    // PIKU_ALLOW_WRITE_ANY=1 disables the guard.
    if std::env::var("PIKU_ALLOW_WRITE_ANY").as_deref() != Ok("1") {
        if let Err(e) = crate::ensure_within_base(&p.path, &cwd) {
            return ToolResult::error(format!("write_file refused: {e}"));
        }
    }

    // create parent dirs if needed
    if let Some(parent) = Path::new(&p.path).parent() {
        if !parent.as_os_str().is_empty() {
            if let Err(e) = std::fs::create_dir_all(parent) {
                return ToolResult::error(format!("write_file: create dirs: {e}"));
            }
        }
    }

    match std::fs::write(&p.path, &p.content) {
        Ok(()) => ToolResult::ok(format!("wrote {} bytes to {}", p.content.len(), p.path))
            .with_effect(ToolEffect::FileWrite {
                path: resolved_path.canonicalize().unwrap_or(resolved_path),
                content_change,
            }),
        Err(e) => ToolResult::error(format!("write_file: {}: {e}", p.path)),
    }
}
