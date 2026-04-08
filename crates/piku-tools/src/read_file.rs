use serde::Deserialize;

use crate::{Destructiveness, ToolResult};

#[derive(Debug, Deserialize)]
pub struct ReadFileParams {
    pub path: String,
    /// 1-indexed start line (inclusive). None = from beginning.
    pub start_line: Option<usize>,
    /// 1-indexed end line (inclusive). None = to end.
    pub end_line: Option<usize>,
}

#[must_use]
pub fn schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "path": { "type": "string", "description": "Absolute or relative path to the file" },
            "start_line": { "type": "integer", "description": "First line to read (1-indexed, inclusive)" },
            "end_line": { "type": "integer", "description": "Last line to read (1-indexed, inclusive)" }
        },
        "required": ["path"]
    })
}

#[must_use]
pub fn destructiveness(_params: &serde_json::Value) -> Destructiveness {
    Destructiveness::Safe
}

#[must_use]
pub fn execute(params: serde_json::Value) -> ToolResult {
    const MAX_FILE_SIZE: u64 = 10 * 1024 * 1024; // 10 MB

    let p: ReadFileParams = match serde_json::from_value(params) {
        Ok(v) => v,
        Err(e) => return ToolResult::error(format!("invalid params: {e}")),
    };

    // Size guard: prevent OOM on huge files or special files (/dev/zero).
    match std::fs::metadata(&p.path) {
        Ok(meta) if meta.len() > MAX_FILE_SIZE => {
            return ToolResult::error(format!(
                "read_file: {} is too large ({} bytes, limit {})",
                p.path,
                meta.len(),
                MAX_FILE_SIZE
            ));
        }
        Err(e) => return ToolResult::error(format!("read_file: {}: {e}", p.path)),
        _ => {}
    }

    let content = match std::fs::read_to_string(&p.path) {
        Ok(c) => c,
        Err(e) => return ToolResult::error(format!("read_file: {}: {e}", p.path)),
    };

    match (p.start_line, p.end_line) {
        (None, None) => ToolResult::ok(content),
        (start, end) => {
            let lines: Vec<&str> = content.lines().collect();
            let total = lines.len();
            // 1-indexed → 0-indexed, clamped to valid range
            let start = start.unwrap_or(1).saturating_sub(1).min(total);
            let end = end.unwrap_or(total).min(total);
            if start > end {
                return ToolResult::error(format!(
                    "read_file: start_line ({}) > end_line ({})",
                    start + 1,
                    end
                ));
            }
            ToolResult::ok(lines[start..end].join("\n"))
        }
    }
}
