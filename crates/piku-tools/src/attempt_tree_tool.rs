/// `record_attempt` and `query_attempts` tools for tree-structured experiential memory.
///
/// Agents use `record_attempt` to log what they try and whether it worked.
/// `query_attempts` retrieves past attempt trees for similar goals, letting
/// agents avoid repeating failed approaches and build on successful ones.
use crate::{Destructiveness, ToolResult};

// ---------------------------------------------------------------------------
// record_attempt
// ---------------------------------------------------------------------------

#[must_use]
pub fn record_attempt_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "goal": {
                "type": "string",
                "description": "What you are trying to achieve. Used to match future queries."
            },
            "approach": {
                "type": "string",
                "description": "The specific approach being tried."
            },
            "parent_id": {
                "type": "integer",
                "description": "ID of the parent attempt if this is a sub-approach or retry. Omit for root attempts."
            },
            "outcome": {
                "type": "string",
                "enum": ["pending", "success", "failure"],
                "description": "Outcome of this attempt. Use 'pending' if still in progress, then call again with attempt_id to update."
            },
            "outcome_detail": {
                "type": "string",
                "description": "Why this approach succeeded or failed. Important for guiding future attempts."
            },
            "attempt_id": {
                "type": "integer",
                "description": "If updating an existing attempt's outcome, pass its ID here. Omit when creating a new attempt."
            },
            "importance": {
                "type": "integer",
                "description": "Importance 1-10 (default 6). Higher = more likely to be retained and surfaced."
            }
        },
        "required": ["goal", "approach"]
    })
}

#[must_use]
pub fn record_attempt_destructiveness(_params: &serde_json::Value) -> Destructiveness {
    Destructiveness::Safe
}

/// Stub for single-shot mode (no embedding runtime).
#[must_use]
pub fn execute_record_attempt_stub(_params: serde_json::Value) -> ToolResult {
    ToolResult::ok(
        "record_attempt requires the embedding runtime. Use it in an interactive session."
            .to_string(),
    )
}

// ---------------------------------------------------------------------------
// query_attempts
// ---------------------------------------------------------------------------

#[must_use]
pub fn query_attempts_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "goal": {
                "type": "string",
                "description": "Describe what you are trying to achieve. Returns attempt trees from similar past goals."
            },
            "max_trees": {
                "type": "integer",
                "description": "Maximum number of attempt trees to return (default 3)."
            }
        },
        "required": ["goal"]
    })
}

#[must_use]
pub fn query_attempts_destructiveness(_params: &serde_json::Value) -> Destructiveness {
    Destructiveness::Safe
}

/// Stub for single-shot mode (no embedding runtime).
#[must_use]
pub fn execute_query_attempts_stub(_params: serde_json::Value) -> ToolResult {
    ToolResult::ok(
        "query_attempts requires the embedding runtime. Use it in an interactive session."
            .to_string(),
    )
}
