use serde::Deserialize;

use crate::{Destructiveness, ToolResult};

#[derive(Debug, Deserialize)]
pub struct EditFileParams {
    pub path: String,
    pub old_string: String,
    pub new_string: String,
    /// Replace all occurrences instead of requiring exactly one. Default false.
    #[serde(default)]
    pub replace_all: bool,
}

#[must_use]
pub fn schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "path": { "type": "string", "description": "Path to the file to edit" },
            "old_string": { "type": "string", "description": "Exact string to replace (must match exactly once unless replace_all is true)" },
            "new_string": { "type": "string", "description": "String to replace it with" },
            "replace_all": { "type": "boolean", "description": "Replace all occurrences (default false)" }
        },
        "required": ["path", "old_string", "new_string"]
    })
}

#[must_use]
pub fn destructiveness(params: &serde_json::Value) -> Destructiveness {
    let path = params.get("path").and_then(|v| v.as_str()).unwrap_or("");
    // Safety-check paths are always Definite regardless of location.
    if crate::is_protected_path(path) {
        return Destructiveness::Definite;
    }
    // Check if path is outside cwd
    let cwd = std::env::current_dir().unwrap_or_default();
    let abs = if std::path::Path::new(path).is_absolute() {
        std::path::PathBuf::from(path)
    } else {
        cwd.join(path)
    };
    if abs.starts_with(&cwd) {
        Destructiveness::Likely
    } else {
        Destructiveness::Definite
    }
}

#[must_use]
pub fn execute(params: serde_json::Value) -> ToolResult {
    let p: EditFileParams = match serde_json::from_value(params) {
        Ok(v) => v,
        Err(e) => return ToolResult::error(format!("invalid params: {e}")),
    };

    // Reject empty old_string early — it would "match" len+1 times and corrupt.
    if p.old_string.is_empty() {
        return ToolResult::error("edit_file: old_string must not be empty".to_string());
    }

    let raw = match std::fs::read_to_string(&p.path) {
        Ok(c) => c,
        Err(e) => return ToolResult::error(format!("edit_file: read {}: {e}", p.path)),
    };

    // Normalize CRLF → LF so that files with Windows line endings can be
    // edited with LF-only old_string (which is what LLMs always provide).
    // We preserve the original line ending style when writing back.
    let has_crlf = raw.contains("\r\n");
    let content = if has_crlf {
        raw.replace("\r\n", "\n")
    } else {
        raw.clone()
    };
    // Also normalize old_string and new_string for comparison
    let old_string_normalized = p.old_string.replace("\r\n", "\n");
    let new_string_normalized = p.new_string.replace("\r\n", "\n");

    // Helper: restore original line endings if needed
    let restore = |s: String| -> String {
        if has_crlf {
            s.replace('\n', "\r\n")
        } else {
            s
        }
    };

    if p.replace_all {
        let new_content = content.replace(&old_string_normalized, &new_string_normalized);
        let count = content.matches(old_string_normalized.as_str()).count();
        if count == 0 {
            return ToolResult::error(format!("edit_file: old_string not found in {}", p.path));
        }
        return match std::fs::write(&p.path, restore(new_content)) {
            Ok(()) => ToolResult::ok(format!("replaced {count} occurrences in {}", p.path)),
            Err(e) => ToolResult::error(format!("edit_file: write {}: {e}", p.path)),
        };
    }

    // require exactly one match
    let count = content.matches(old_string_normalized.as_str()).count();
    match count {
        0 => ToolResult::error(format!(
            "edit_file: old_string not found in {}",
            p.path
        )),
        1 => {
            let new_content = content.replacen(&old_string_normalized, &new_string_normalized, 1);
            match std::fs::write(&p.path, restore(new_content)) {
                Ok(()) => ToolResult::ok(format!("edited {}", p.path)),
                Err(e) => ToolResult::error(format!("edit_file: write {}: {e}", p.path)),
            }
        }
        n => ToolResult::error(format!(
            "edit_file: ambiguous — old_string matched {n} times in {}. Use replace_all=true or provide more context.",
            p.path
        )),
    }
}
