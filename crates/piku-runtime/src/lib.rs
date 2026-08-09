//! Runtime orchestration for piku.
//!
//! This crate owns the agent loop, sessions, memory, hooks, permissions,
//! provider resolution, and subagent task tracking. Protocol clients live in
//! `piku-api`; executable tools live in `piku-tools`.

pub mod agent_loop;
pub mod agents;
pub mod compact;
pub mod context;
pub mod embed_memory;
pub mod hooks;
pub mod memory;
pub mod permission;
pub mod prompt;
pub mod provider;
pub mod run_audit;
pub mod run_handle;
pub mod run_record;
pub mod session;
pub mod task;
#[cfg(test)]
mod tests;

pub use agent_loop::{
    run_turn, run_turn_with_registry, CancelFlag, InterjectionRx, InterjectionTx, OutputSink,
    PostToolAction, TurnResult,
};
pub use agents::{
    agent_listing_prompt_with_custom, find_agent, find_built_in, load_custom_agents, AgentDef,
};
pub use compact::{
    apply_compact_summary, compact_session, should_compact, CompactionConfig, CompactionResult,
};
pub use context::{
    render_captured_attachments, resolve_captured_attachments, CacheDecision, CapabilityProfile,
    CapturedAttachment, ContextBudget, ContextError, ContextItem, ContextPayload,
    ContextResolution, Freshness, FreshnessPolicy, OutputPlane, RenderedContext, ReplayMode,
    ResolutionCache, ResolutionError, ResolutionRequest, ResolutionStatus, ResolverIdentity,
    Sensitivity, Sha256Digest, SourceReference, Trust, CONTEXT_RESOLUTION_SCHEMA_VERSION,
};
pub use embed_memory::{
    build_extraction_transcript, default_store_path, embed_text, embed_text_with_config,
    extract_and_store, extract_memories, format_retrieved_memories, EmbedBackend, EmbedConfig,
    MemoryEntry, MemoryJudgment, MemoryStore, RetrievedMemory,
};
pub use hooks::{HookConfig, HookRegistry};
pub use memory::{
    build_agent_memory_prompt, build_memory_prompt, read_agent_memory, read_memory,
    write_agent_memory, write_memory, MemoryScope,
};
pub use permission::{AllowAll, PermissionOutcome, PermissionPrompter, PermissionRequest};
pub use piku_api::Provider;
pub use piku_api::TokenUsage;
pub use prompt::build_system_prompt;
pub use provider::{
    provider_availability, ProviderAvailability, ResolvedProvider, DEFAULT_MODEL_ANTHROPIC,
    DEFAULT_MODEL_GROQ, DEFAULT_MODEL_OLLAMA, DEFAULT_MODEL_OPENROUTER,
};
pub use run_audit::{
    audit_run_record, AuditSeverity, ContentAudit, ContextAudit, RunAudit, RunAuditFinding,
};
pub use run_handle::{RunHandle, RunTurn};
pub use run_record::{
    read_run_record, ArtifactRef, ContentChange as RunContentChange, ContentRef as RunContentRef,
    ContextManifest, ContextMessage, ContextSection, ContextSourceSummary, ContextTool,
    EventScope as RunEventScope, PermissionDecision as RunPermissionDecision, RecordingSink,
    RunDisposition, RunEvent, RunEventEnvelope, RunRecorder, ToolEffect as RunToolEffect,
    UsageRecord, VerificationIndeterminate, VerificationRecord, VerificationStatus,
    RUN_INLINE_CONTENT_LIMIT_BYTES, RUN_RECORD_SCHEMA_VERSION,
};
pub use session::{ContentBlock, ConversationMessage, MessageRole, Session, UsageTracker};
pub use task::{
    AgentTaskId, SubagentEvidence, TaskEntry, TaskRegistry, TaskStatus, DEFAULT_SUBAGENT_MAX_TURNS,
    MAX_SPAWN_DEPTH,
};
