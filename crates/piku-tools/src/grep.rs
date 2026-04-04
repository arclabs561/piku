use serde::Deserialize;
use walkdir::WalkDir;

use crate::{Destructiveness, ToolResult};

#[derive(Debug, Deserialize)]
pub struct GrepParams {
    pub pattern: String,
    /// Directory or file to search. Defaults to cwd.
    pub path: Option<String>,
    /// File glob filter (e.g. "*.rs"). None = all files.
    pub include: Option<String>,
    /// Maximum number of results to return. Default 100.
    pub max_results: Option<usize>,
}

#[must_use]
pub fn schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "pattern": { "type": "string", "description": "Regex pattern to search for" },
            "path": { "type": "string", "description": "Directory or file to search (default: cwd)" },
            "include": { "type": "string", "description": "Filename glob filter (e.g. *.rs)" },
            "max_results": { "type": "integer", "description": "Maximum matches to return (default 100)" }
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
    let p: GrepParams = match serde_json::from_value(params) {
        Ok(v) => v,
        Err(e) => return ToolResult::error(format!("invalid params: {e}")),
    };

    let re = match regex::Regex::new(&p.pattern) {
        Ok(r) => r,
        Err(e) => return ToolResult::error(format!("grep: invalid pattern: {e}")),
    };

    let root = p.path.unwrap_or_else(|| {
        std::env::current_dir()
            .unwrap_or_default()
            .to_string_lossy()
            .into_owned()
    });

    let max = p.max_results.unwrap_or(100);
    let mut results: Vec<String> = Vec::new();
    let mut total = 0usize;

    let root_path = std::path::Path::new(&root);

    // If root is a file, search just that file
    if root_path.is_file() {
        search_file(
            root_path,
            &re,
            p.include.as_deref(),
            max,
            &mut results,
            &mut total,
        );
        if total >= max {
            results.push(format!("(truncated at {max} results)"));
        }
    } else {
        let mut truncated = false;
        for entry in WalkDir::new(root_path)
            .follow_links(false)
            .into_iter()
            .filter_map(std::result::Result::ok)
            .filter(|e| e.file_type().is_file())
        {
            if total >= max {
                truncated = true;
                break;
            }
            search_file(
                entry.path(),
                &re,
                p.include.as_deref(),
                max - total,
                &mut results,
                &mut total,
            );
            if total >= max {
                truncated = true;
                break;
            }
        }
        if truncated {
            results.push(format!("(truncated at {max} results)"));
        }
    }

    if results.is_empty() {
        ToolResult::ok("(no matches)".to_string())
    } else {
        ToolResult::ok(results.join("\n"))
    }
}

fn search_file(
    path: &std::path::Path,
    re: &regex::Regex,
    include_glob: Option<&str>,
    limit: usize,
    results: &mut Vec<String>,
    total: &mut usize,
) {
    // apply filename filter
    if let Some(glob_pat) = include_glob {
        let file_name = path.file_name().unwrap_or_default().to_string_lossy();
        let pattern = glob::Pattern::new(glob_pat)
            .unwrap_or_else(|_| glob::Pattern::new("*").expect("* is always a valid pattern"));
        if !pattern.matches(&file_name) {
            return;
        }
    }

    let Ok(content) = std::fs::read_to_string(path) else {
        return;
    }; // skip binary / unreadable files

    for (line_no, line) in content.lines().enumerate() {
        if *total >= limit {
            break;
        }
        if re.is_match(line) {
            results.push(format!("{}:{}: {}", path.display(), line_no + 1, line));
            *total += 1;
        }
    }
}
