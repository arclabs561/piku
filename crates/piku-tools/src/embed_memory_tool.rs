/// `search_memory` and `manage_memory` tools for embedding-based memory.
///
/// `search_memory`: semantic search over the embedding store.
/// `manage_memory`: list, inspect, invalidate, query by tags.
///
/// These give the agent (and operator) full visibility into what's stored,
/// what's being retrieved, and the ability to curate memory actively.
use std::fmt::Write;

use serde::Deserialize;

use crate::{Destructiveness, ToolResult};

// ---------------------------------------------------------------------------
// search_memory
// ---------------------------------------------------------------------------

#[must_use]
pub fn search_memory_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural language query to search semantic memory. Finds memories by meaning, not just keywords."
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum results to return (default 5)."
            }
        },
        "required": ["query"]
    })
}

#[must_use]
pub fn search_memory_destructiveness(_params: &serde_json::Value) -> Destructiveness {
    Destructiveness::Safe
}

/// Execute `search_memory`. Requires embedding the query first (async, handled by runtime).
/// This is a stub -- the actual embedding + search is done in the runtime's agent loop.
#[must_use]
pub fn execute_search_memory_stub(_params: serde_json::Value) -> ToolResult {
    ToolResult::ok(
        "search_memory requires the embedding runtime. Use it in an interactive session."
            .to_string(),
    )
}

// ---------------------------------------------------------------------------
// manage_memory
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct ManageMemoryParams {
    /// Action: `list`, `inspect`, `invalidate`, `stats`, `query_tags`.
    action: String,
    /// Memory ID (for inspect/invalidate).
    id: Option<u64>,
    /// Tag query (for `query_tags`).
    tag: Option<String>,
    /// Max results for `list`/`query_tags` (default 10).
    max_results: Option<usize>,
}

#[must_use]
pub fn manage_memory_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["list", "inspect", "invalidate", "stats", "query_tags"],
                "description": "Action to perform on the memory store."
            },
            "id": {
                "type": "integer",
                "description": "Memory ID (for inspect/invalidate actions)."
            },
            "tag": {
                "type": "string",
                "description": "Tag to search for (for query_tags action)."
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum results for list/query_tags (default 10)."
            }
        },
        "required": ["action"]
    })
}

#[must_use]
pub fn manage_memory_destructiveness(params: &serde_json::Value) -> Destructiveness {
    match params.get("action").and_then(|a| a.as_str()) {
        Some("invalidate") => Destructiveness::Likely,
        _ => Destructiveness::Safe,
    }
}

/// Execute `manage_memory` against a `MemoryStore`.
/// Called by the runtime with the actual store instance.
pub fn execute_manage_memory(
    params: serde_json::Value,
    store: &mut dyn piku_runtime_types::MemoryStoreView,
) -> ToolResult {
    let p: ManageMemoryParams = match serde_json::from_value(params) {
        Ok(v) => v,
        Err(e) => return ToolResult::error(format!("invalid params: {e}")),
    };

    match p.action.as_str() {
        "stats" => {
            let total = store.total_count();
            let valid = store.valid_count();
            let invalid = total - valid;
            let attempts = store.attempt_count();
            ToolResult::ok(format!(
                "Memory store stats:\n- Total entries: {total}\n- Valid: {valid}\n- Invalid: {invalid}\n- Attempts: {attempts}"
            ))
        }
        "list" => {
            let max = p.max_results.unwrap_or(10);
            let entries = store.list_recent(max);
            if entries.is_empty() {
                return ToolResult::ok("No memories stored.".to_string());
            }
            let mut out = String::from("Recent memories:\n");
            for (id, content, valid, access_count) in entries {
                let status = if valid { "valid" } else { "INVALID" };
                let _ = write!(
                    out,
                    "\n[{id}] ({status}, accessed {access_count}x) {content}"
                );
            }
            ToolResult::ok(out)
        }
        "inspect" => {
            let Some(id) = p.id else {
                return ToolResult::error("inspect requires an 'id' parameter".to_string());
            };
            match store.inspect(id) {
                Some(detail) => ToolResult::ok(detail),
                None => ToolResult::error(format!("memory {id} not found")),
            }
        }
        "invalidate" => {
            let Some(id) = p.id else {
                return ToolResult::error("invalidate requires an 'id' parameter".to_string());
            };
            if store.invalidate(id) {
                ToolResult::ok(format!("memory {id} marked as invalid"))
            } else {
                ToolResult::error(format!("memory {id} not found"))
            }
        }
        "query_tags" => {
            let Some(tag) = &p.tag else {
                return ToolResult::error("query_tags requires a 'tag' parameter".to_string());
            };
            let max = p.max_results.unwrap_or(10);
            let entries = store.query_by_tag(tag, max);
            if entries.is_empty() {
                return ToolResult::ok(format!("No memories with tag '{tag}'."));
            }
            let mut out = format!("Memories tagged '{tag}':\n");
            for (id, content) in entries {
                let _ = write!(out, "\n[{id}] {content}");
            }
            ToolResult::ok(out)
        }
        other => ToolResult::error(format!(
            "unknown action '{other}'. Use: list, inspect, invalidate, stats, query_tags"
        )),
    }
}

/// Trait for memory store operations needed by `manage_memory`.
/// This avoids a circular dep between piku-tools and piku-runtime.
pub mod piku_runtime_types {
    pub trait MemoryStoreView {
        fn total_count(&self) -> usize;
        fn valid_count(&self) -> usize;
        fn list_recent(&self, max: usize) -> Vec<(u64, String, bool, u32)>;
        fn inspect(&self, id: u64) -> Option<String>;
        fn invalidate(&mut self, id: u64) -> bool;
        fn query_by_tag(&self, tag: &str, max: usize) -> Vec<(u64, String)>;
        /// Count of attempt-type entries (valid only).
        fn attempt_count(&self) -> usize;
        /// Record outcome on an attempt. Returns false if not found.
        fn record_outcome(
            &mut self,
            attempt_id: u64,
            outcome: &str,
            detail: Option<String>,
        ) -> bool;
    }
}
