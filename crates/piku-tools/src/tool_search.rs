/// `tool_search` — lazy tool discovery meta-tool.
///
/// When the tool catalog grows (MCP servers, custom tools), dumping all
/// schemas into context causes tool-selection errors (ITR paper, Toolshed).
/// Instead, expose only core tools by default and let the model discover
/// others on demand via keyword search over tool descriptions.
///
/// The model calls `tool_search(query: "file editing")` and gets back
/// matching tool names + descriptions, which it can then use by name.
use serde::Deserialize;

use crate::{Destructiveness, ToolResult};

/// A searchable tool entry (name + description, no full schema).
#[derive(Debug, Clone)]
pub struct SearchableToolEntry {
    pub name: String,
    pub description: String,
}

#[derive(Debug, Deserialize)]
struct ToolSearchParams {
    /// Search query — keywords to match against tool names and descriptions.
    query: String,
    /// Maximum results to return (default 5).
    max_results: Option<usize>,
}

#[must_use]
pub fn tool_search_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Keywords to search for in tool names and descriptions. \
                    Example: 'file editing', 'search code', 'run command'."
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results (default 5)."
            }
        },
        "required": ["query"]
    })
}

#[must_use]
pub fn tool_search_destructiveness(_params: &serde_json::Value) -> Destructiveness {
    Destructiveness::Safe
}

/// Execute a tool search against a catalog of searchable entries.
/// The catalog is passed in because the tool module doesn't own the registry.
#[must_use]
pub fn execute_tool_search(
    params: serde_json::Value,
    catalog: &[SearchableToolEntry],
) -> ToolResult {
    let p: ToolSearchParams = match serde_json::from_value(params) {
        Ok(v) => v,
        Err(e) => return ToolResult::error(format!("invalid params: {e}")),
    };

    if p.query.trim().is_empty() {
        return ToolResult::error("query must not be empty".to_string());
    }

    let max = p.max_results.unwrap_or(5).min(20);
    let query_terms: Vec<String> = p
        .query
        .to_lowercase()
        .split_whitespace()
        .map(String::from)
        .collect();

    // Score each tool by how many query terms appear in name + description.
    let mut scored: Vec<(usize, &SearchableToolEntry)> = catalog
        .iter()
        .map(|entry| {
            let haystack = format!("{} {}", entry.name, entry.description).to_lowercase();
            let score = query_terms
                .iter()
                .filter(|term| haystack.contains(term.as_str()))
                .count();
            (score, entry)
        })
        .filter(|(score, _)| *score > 0)
        .collect();

    // Sort by score descending, then by name for stability.
    scored.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.name.cmp(&b.1.name)));

    let results: Vec<&SearchableToolEntry> = scored
        .into_iter()
        .take(max)
        .map(|(_, entry)| entry)
        .collect();

    if results.is_empty() {
        return ToolResult::ok(format!(
            "No tools matching '{}'. Try broader keywords.",
            p.query
        ));
    }

    let mut output = format!(
        "Found {} tool(s) matching '{}':\n\n",
        results.len(),
        p.query
    );
    for entry in &results {
        output.push_str("- **");
        output.push_str(&entry.name);
        output.push_str("**: ");
        output.push_str(&entry.description);
        output.push('\n');
    }
    output.push_str("\nUse these tools by name in subsequent tool calls.");

    ToolResult::ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_catalog() -> Vec<SearchableToolEntry> {
        vec![
            SearchableToolEntry {
                name: "read_file".to_string(),
                description: "Read the contents of a file".to_string(),
            },
            SearchableToolEntry {
                name: "write_file".to_string(),
                description: "Write content to a file".to_string(),
            },
            SearchableToolEntry {
                name: "edit_file".to_string(),
                description: "Surgically replace an exact string in a file".to_string(),
            },
            SearchableToolEntry {
                name: "bash".to_string(),
                description: "Execute a shell command".to_string(),
            },
            SearchableToolEntry {
                name: "grep".to_string(),
                description: "Search file contents with a regex pattern".to_string(),
            },
        ]
    }

    #[test]
    fn search_finds_matching_tools() {
        let catalog = sample_catalog();
        let result = execute_tool_search(serde_json::json!({"query": "file"}), &catalog);
        assert!(!result.is_error);
        assert!(result.output.contains("read_file"));
        assert!(result.output.contains("write_file"));
        assert!(result.output.contains("edit_file"));
    }

    #[test]
    fn search_respects_max_results() {
        let catalog = sample_catalog();
        let result = execute_tool_search(
            serde_json::json!({"query": "file", "max_results": 1}),
            &catalog,
        );
        assert!(!result.is_error);
        // Should contain exactly 1 result
        assert!(result.output.contains("1 tool(s)"));
    }

    #[test]
    fn search_no_matches() {
        let catalog = sample_catalog();
        let result =
            execute_tool_search(serde_json::json!({"query": "database migration"}), &catalog);
        assert!(!result.is_error);
        assert!(result.output.contains("No tools matching"));
    }

    #[test]
    fn search_empty_query_is_error() {
        let catalog = sample_catalog();
        let result = execute_tool_search(serde_json::json!({"query": "  "}), &catalog);
        assert!(result.is_error);
    }

    #[test]
    fn search_multi_term_scoring() {
        let catalog = sample_catalog();
        let result = execute_tool_search(
            serde_json::json!({"query": "search regex pattern"}),
            &catalog,
        );
        assert!(!result.is_error);
        // grep should rank highest (matches "search", "regex", "pattern")
        assert!(result.output.starts_with("Found 1 tool") || result.output.contains("grep"));
    }
}
