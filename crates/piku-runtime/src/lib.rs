pub mod agent_loop;
pub mod agents;
pub mod compact;
pub mod memory;
pub mod permission;
pub mod prompt;
pub mod session;
pub mod task;
#[cfg(test)]
mod tests;

pub use agent_loop::{
    run_turn, run_turn_with_registry, InterjectionRx, InterjectionTx, OutputSink, PostToolAction,
    TurnResult,
};
pub use agents::{agent_listing_prompt, all_built_ins, find_built_in, AgentDef};
pub use compact::{
    apply_compact_summary, compact_session, compact_system_prompt, estimate_session_tokens,
    format_compact_summary, get_compact_continuation_message, should_compact, CompactionConfig,
    CompactionResult,
};
pub use memory::{build_memory_prompt, read_memory, write_memory, MemoryScope};
pub use permission::{AllowAll, PermissionOutcome, PermissionPrompter, PermissionRequest};
pub use piku_api::Provider;
pub use piku_api::TokenUsage;
pub use prompt::build_system_prompt;
pub use session::{ContentBlock, ConversationMessage, MessageRole, Session, UsageTracker};
pub use task::{
    AgentTaskId, TaskEntry, TaskRegistry, TaskStatus, DEFAULT_SUBAGENT_MAX_TURNS, MAX_SPAWN_DEPTH,
};
