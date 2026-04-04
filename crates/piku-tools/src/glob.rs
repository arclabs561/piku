use serde::Deserialize;

use crate::{Destructiveness, ToolResult};

#[derive(Debug, Deserialize)]
pub struct GlobParams {
    pub pattern: String,
    /// Root directory to search from. Defaults to cwd.
    pub path: Option<String>,
}

#[must_use]
pub fn schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "pattern": { "type": "string", "description": "Glob pattern (e.g. **/*.rs)" },
            "path": { "type": "string", "description": "Root directory to search from (default: cwd)" }
        },
        "required": ["pattern"]
    })
}

#[must_use]
pub fn destructiveness(_params: &serde_json::Value) -> Destructiveness {
    Destructiveness::Safe
}

#[must_use]
pub fn execute(params: serde_json::Value) -> ToolResult {
    let p: GlobParams = match serde_json::from_value(params) {
        Ok(v) => v,
        Err(e) => return ToolResult::error(format!("invalid params: {e}")),
    };

    let base = p.path.unwrap_or_else(|| {
        std::env::current_dir()
            .unwrap_or_default()
            .to_string_lossy()
            .into_owned()
    });

    let full_pattern = if p.pattern.starts_with('/') {
        p.pattern.clone()
    } else {
        format!("{base}/{}", p.pattern)
    };

    // Sandbox check: verify the pattern doesn't escape the base directory via ../
    // Strategy: check if the literal prefix (before any glob special chars) is
    // contained within base after canonicalization.
    let canonical_base = std::path::Path::new(&base)
        .canonicalize()
        .unwrap_or_else(|_| std::path::PathBuf::from(&base));

    let prefix_end = full_pattern
        .find(['*', '?', '[', '{'])
        .unwrap_or(full_pattern.len());

    // Strip trailing path separators so `/base/src/` → `/base/src` before canonicalize.
    // This avoids canonicalize failing on a path that ends with `/`.
    let literal_str = full_pattern[..prefix_end].trim_end_matches('/');

    if !literal_str.is_empty() {
        // Build the closest existing ancestor for canonicalization.
        // Walk up until we find a path that exists.
        let mut check_path = std::path::PathBuf::from(literal_str);
        let canonical_prefix = loop {
            match check_path.canonicalize() {
                Ok(c) => break c,
                Err(_) => {
                    if !check_path.pop() {
                        // Fell off the root — use base as fallback (safe)
                        break canonical_base.clone();
                    }
                }
            }
        };

        if !canonical_prefix.starts_with(&canonical_base) {
            return ToolResult::error(format!(
                "glob: pattern escapes base directory — pattern is outside '{base}'"
            ));
        }
    }

    let entries = match glob::glob(&full_pattern) {
        Ok(paths) => paths,
        Err(e) => return ToolResult::error(format!("glob: invalid pattern: {e}")),
    };

    let mut results: Vec<String> = Vec::new();
    for entry in entries {
        match entry {
            Ok(path) => results.push(path.to_string_lossy().into_owned()),
            Err(e) => results.push(format!("(error: {e})")),
        }
    }

    if results.is_empty() {
        ToolResult::ok("(no matches)".to_string())
    } else {
        ToolResult::ok(results.join("\n"))
    }
}
