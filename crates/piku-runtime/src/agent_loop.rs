#![allow(
    clippy::doc_markdown,
    clippy::match_wildcard_for_single_variants,
    clippy::needless_borrows_for_generic_args,
    clippy::too_many_arguments,
    clippy::too_many_lines
)]

use std::fmt::Write;

use futures_util::StreamExt;
use tokio::sync::mpsc;

use piku_api::{
    Event, MessageRequest, Provider, RequestContent, RequestMessage, StopReason, TokenUsage,
    ToolDefinition,
};
use piku_tools::execute_tool;

use crate::permission::{check_permission, PermissionOutcome, PermissionPrompter};
use crate::session::{ContentBlock, ConversationMessage, MessageRole, Session, UsageTracker};
use crate::task::{TaskRegistry, MAX_SPAWN_DEPTH};

/// A channel the TUI can use to inject interjections mid-turn.
/// Send a plain string; the agent wraps it in `<interjection>` tags so
/// the model sees it as an in-band user message without breaking the
/// tool-use alternation protocol.
pub type InterjectionRx = mpsc::Receiver<String>;
pub type InterjectionTx = mpsc::Sender<String>;

const DEFAULT_MAX_TURNS: u32 = 20;

/// Action the loop should take after a tool call completes.
#[derive(Debug, Default)]
pub enum PostToolAction {
    /// Continue the loop normally.
    #[default]
    Continue,
    /// Stop the loop and replace the running binary before resuming.
    /// The `PathBuf` is the path to the new binary.
    ReplaceAndExec(std::path::PathBuf),
}

/// Callbacks for streaming output so the caller (TUI or CLI) can render in real time.
pub trait OutputSink: Send {
    fn on_text(&mut self, text: &str);
    fn on_tool_start(&mut self, tool_name: &str, tool_id: &str, input: &serde_json::Value);
    /// Called after a tool completes. Return `PostToolAction::ReplaceAndExec`
    /// to trigger a self-update before the loop continues.
    fn on_tool_end(&mut self, tool_name: &str, result: &str, is_error: bool) -> PostToolAction;
    fn on_permission_denied(&mut self, tool_name: &str, reason: &str);
    fn on_turn_complete(&mut self, usage: &TokenUsage, iterations: u32);
    /// Called when a user interjection is injected mid-turn (so the sink
    /// can echo it visually). Default: no-op.
    fn on_interjection(&mut self, _text: &str) {}
}

/// A turn result after the full agentic loop for one user message.
#[derive(Debug)]
pub struct TurnResult {
    pub iterations: u32,
    pub usage: TokenUsage,
    /// Set if a stream-level error was encountered during any iteration.
    pub stream_error: Option<String>,
    /// Set if the loop was interrupted for a self-update restart.
    pub replace_and_exec: Option<std::path::PathBuf>,
}

/// Run a full agentic turn: stream, collect tool calls, execute, loop.
///
/// Mutates `session` in place (appends messages).
/// Calls `sink` for streaming output.
/// Calls `prompter` for permission checks.
/// If `interjections` is Some, the loop drains it between tool calls and
/// injects any queued messages so the model can respond to them.
///
/// If `task_registry` is Some, `spawn_agent` / `agent_status` / `agent_join`
/// calls are handled natively — subagents run as background tokio tasks and
/// report back through the registry.
pub async fn run_turn(
    input: &str,
    session: &mut Session,
    provider: &dyn Provider,
    model: &str,
    system_prompt: &[String],
    tool_defs: Vec<ToolDefinition>,
    prompter: &dyn PermissionPrompter,
    sink: &mut dyn OutputSink,
    max_turns: Option<u32>,
    interjections: Option<&mut InterjectionRx>,
) -> TurnResult {
    run_turn_inner(
        input,
        session,
        provider,
        model,
        system_prompt,
        tool_defs,
        prompter,
        sink,
        max_turns,
        interjections,
        None,
        0,
        &[],
    )
    .await
}

/// Like `run_turn` but with a `TaskRegistry` for background agent spawning
/// and explicit `depth` tracking for recursive spawn safety.
pub async fn run_turn_with_registry(
    input: &str,
    session: &mut Session,
    provider: &dyn Provider,
    model: &str,
    system_prompt: &[String],
    tool_defs: Vec<ToolDefinition>,
    prompter: &dyn PermissionPrompter,
    sink: &mut dyn OutputSink,
    max_turns: Option<u32>,
    interjections: Option<&mut InterjectionRx>,
    registry: &TaskRegistry,
    depth: u32,
    custom_agents: &[crate::agents::CustomAgentDef],
) -> TurnResult {
    run_turn_inner(
        input,
        session,
        provider,
        model,
        system_prompt,
        tool_defs,
        prompter,
        sink,
        max_turns,
        interjections,
        Some(registry),
        depth,
        custom_agents,
    )
    .await
}

async fn run_turn_inner(
    input: &str,
    session: &mut Session,
    provider: &dyn Provider,
    model: &str,
    system_prompt: &[String],
    tool_defs: Vec<ToolDefinition>,
    prompter: &dyn PermissionPrompter,
    sink: &mut dyn OutputSink,
    max_turns: Option<u32>,
    interjections: Option<&mut InterjectionRx>,
    task_registry: Option<&TaskRegistry>,
    depth: u32,
    custom_agents: &[crate::agents::CustomAgentDef],
) -> TurnResult {
    let max = max_turns.unwrap_or(DEFAULT_MAX_TURNS);
    let mut interjections = interjections;

    // push user message
    session.push(ConversationMessage::user(input));

    let mut tracker = UsageTracker::default();
    let mut iterations = 0u32;
    let mut stream_error: Option<String> = None;
    let mut replace_and_exec: Option<std::path::PathBuf> = None;

    // Dedup detection: canonical string key (tool_name + args JSON) for exact match.
    // Uses String keys instead of hashing to avoid collision risk on safety-critical dedup.
    let mut seen_tool_calls: std::collections::HashSet<String> = std::collections::HashSet::new();

    // Track where we last extracted memories (message index).
    // Periodic extraction happens after compaction events.
    let mut last_extraction_idx: usize = session.messages.len();
    // Guard against concurrent extraction tasks racing on the store file.
    let extraction_in_flight = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));

    // Auto-compaction config — compact when session tokens exceed this threshold.
    // 10k tokens is ~40k chars, roughly 200 exchanges of average length.
    let compact_cfg = crate::compact::CompactionConfig::default();

    loop {
        if iterations >= max {
            break;
        }
        iterations += 1;

        // Auto-compact: if the session is getting long, summarise old messages.
        // Tries LLM summarisation first (richer summaries); falls back to
        // structural (no-LLM) if the provider call fails or times out.
        if crate::compact::should_compact(session, compact_cfg) {
            let (result, method) =
                match try_llm_compact(session, provider, model, compact_cfg).await {
                    Some(r) => (r, "llm"),
                    None => (
                        crate::compact::compact_session(session, compact_cfg),
                        "structural",
                    ),
                };
            if result.removed_message_count > 0 {
                // Trigger memory extraction on the messages being compacted away.
                // Guarded: skip if a previous extraction is still in flight (prevents
                // race conditions on the store file).
                let new_messages = &session.messages[last_extraction_idx..];
                if !new_messages.is_empty()
                    && !extraction_in_flight.load(std::sync::atomic::Ordering::Relaxed)
                {
                    let transcript = crate::embed_memory::build_extraction_transcript(new_messages);
                    if !transcript.trim().is_empty() {
                        let provider_clone = provider.boxed_clone();
                        let model_owned = model.to_string();
                        let cwd = std::env::current_dir().unwrap_or_default();
                        let store_path = crate::embed_memory::default_store_path(&cwd);
                        let ollama_url = std::env::var("OLLAMA_HOST")
                            .unwrap_or_else(|_| "http://localhost:11434".to_string());
                        let embed_model = std::env::var("PIKU_EMBED_MODEL")
                            .unwrap_or_else(|_| "nomic-embed-text".to_string());
                        let flag = extraction_in_flight.clone();
                        flag.store(true, std::sync::atomic::Ordering::Relaxed);
                        tokio::task::spawn_local(async move {
                            let mut store = crate::embed_memory::MemoryStore::load(&store_path);
                            let n = crate::embed_memory::extract_and_store(
                                &transcript,
                                provider_clone.as_ref(),
                                &model_owned,
                                &mut store,
                                &ollama_url,
                                &embed_model,
                            )
                            .await;
                            if n > 0 {
                                let _ = store.save(&store_path);
                            }
                            flag.store(false, std::sync::atomic::Ordering::Relaxed);
                        });
                    }
                }

                *session = result.compacted_session;
                last_extraction_idx = 0; // reset after compaction replaces messages
                sink.on_text(&format!(
                    "\x1b[2m[context compacted: {} messages summarised ({method})]\x1b[0m\n",
                    result.removed_message_count
                ));
            }
        }

        let request = build_request(session, model, system_prompt, &tool_defs);

        // stream the response
        let (assistant_blocks, usage, mut stop_reason, maybe_err) =
            stream_response(provider, request, sink).await;

        if let Some(err) = maybe_err {
            stream_error = Some(err);
            // Don't push a partial assistant message on stream error — break cleanly
            tracker.record(usage);
            break;
        }

        tracker.record(usage.clone());

        // extract tool calls from this assistant message
        let tool_calls: Vec<(String, String, serde_json::Value)> = assistant_blocks
            .iter()
            .filter_map(|b| {
                if let ContentBlock::ToolUse { id, name, input } = b {
                    Some((id.clone(), name.clone(), input.clone()))
                } else {
                    None
                }
            })
            .collect();

        // push assistant message
        session.push(ConversationMessage::assistant(
            assistant_blocks,
            Some(usage),
        ));

        if tool_calls.is_empty() {
            break;
        }

        // execute each tool call sequentially (v0 — concurrent in a future phase)
        for (tool_use_id, tool_name, params) in tool_calls {
            // permission check
            match check_permission(&tool_name, &params, prompter) {
                PermissionOutcome::Allow => {}
                PermissionOutcome::Deny { reason } => {
                    sink.on_permission_denied(&tool_name, &reason);
                    session.push(ConversationMessage::tool_result(
                        tool_use_id,
                        format!("Permission denied: {reason}"),
                        true,
                    ));
                    continue;
                }
            }

            // Dedup detection: skip if we've seen this exact (tool, args) before.
            // Read-only tools (read_file, glob, grep, list_dir) are checked;
            // write tools and agent tools are exempt (side effects may differ).
            let is_dedup_eligible = matches!(
                tool_name.as_str(),
                "read_file" | "glob" | "grep" | "list_dir"
            );
            if is_dedup_eligible {
                let call_key = format!("{tool_name}:{params}");
                if !seen_tool_calls.insert(call_key) {
                    let dedup_msg = format!(
                        "You already called {tool_name} with the same arguments. \
                         The result hasn't changed — try a different approach."
                    );
                    sink.on_tool_start(&tool_name, &tool_use_id, &params);
                    sink.on_tool_end(&tool_name, &dedup_msg, true);
                    session.push(ConversationMessage::tool_result(
                        tool_use_id,
                        dedup_msg,
                        true,
                    ));
                    continue;
                }
            }

            sink.on_tool_start(&tool_name, &tool_use_id, &params);

            // Route special tools through the runtime; everything else through execute_tool.
            let (output, is_error) = if tool_name == "search_memory" {
                // Semantic search over embedding store -- needs embedding the query
                let cwd = std::env::current_dir().unwrap_or_default();
                let store_path = crate::embed_memory::default_store_path(&cwd);
                let mut store = crate::embed_memory::MemoryStore::load(&store_path);
                if store.valid_count() == 0 {
                    (
                        "No memories stored yet. Use write_memory to save facts.".to_string(),
                        false,
                    )
                } else {
                    let query = params.get("query").and_then(|v| v.as_str()).unwrap_or("");
                    if query.trim().is_empty() {
                        ("search_memory requires a non-empty query".to_string(), true)
                    } else {
                        #[allow(clippy::cast_possible_truncation)] // max_results is always small
                        let max_k = params
                            .get("max_results")
                            .and_then(serde_json::Value::as_u64)
                            .unwrap_or(5) as usize;
                        let ollama_url = std::env::var("OLLAMA_HOST")
                            .unwrap_or_else(|_| "http://localhost:11434".to_string());
                        let embed_model = std::env::var("PIKU_EMBED_MODEL")
                            .unwrap_or_else(|_| "nomic-embed-text".to_string());
                        // Use .await directly -- safe in both parent and subagent async contexts.
                        // Do NOT use block_in_place here (panics inside spawn_local).
                        let embed_result = tokio::time::timeout(
                            std::time::Duration::from_secs(5),
                            crate::embed_memory::embed_text(query, &ollama_url, &embed_model),
                        )
                        .await;
                        match embed_result {
                            Ok(Ok(query_vec)) => {
                                let retrieved = store.hybrid_retrieve(&query_vec, query, max_k);
                                let _ = store.save(&store_path);
                                if retrieved.is_empty() {
                                    (
                                        "No relevant memories found for that query.".to_string(),
                                        false,
                                    )
                                } else {
                                    (
                                        crate::embed_memory::format_retrieved_memories(&retrieved),
                                        false,
                                    )
                                }
                            }
                            _ => (
                                "search_memory: embedding service unavailable".to_string(),
                                true,
                            ),
                        }
                    }
                }
            } else if tool_name == "manage_memory" {
                // Memory management -- direct store access, no embedding needed
                let cwd = std::env::current_dir().unwrap_or_default();
                let store_path = crate::embed_memory::default_store_path(&cwd);
                let mut store = crate::embed_memory::MemoryStore::load(&store_path);
                let is_mutating =
                    params.get("action").and_then(|a| a.as_str()) == Some("invalidate");
                let result =
                    piku_tools::embed_memory_tool::execute_manage_memory(params, &mut store);
                // Only save on mutating actions
                if is_mutating {
                    let _ = store.save(&store_path);
                }
                (result.output, result.is_error)
            } else if tool_name == "tool_search" {
                // Build catalog from current tool_defs for on-demand search
                let catalog: Vec<piku_tools::tool_search::SearchableToolEntry> = tool_defs
                    .iter()
                    .map(|t| piku_tools::tool_search::SearchableToolEntry {
                        name: t.name.clone(),
                        description: t.description.clone(),
                    })
                    .collect();
                let r = piku_tools::tool_search::execute_tool_search(params, &catalog);
                (r.output, r.is_error)
            } else if let Some(registry) = task_registry {
                match tool_name.as_str() {
                    "spawn_agent" => execute_spawn_agent(
                        &params,
                        registry,
                        provider,
                        model,
                        system_prompt,
                        &tool_defs,
                        depth,
                        &session.messages,
                        custom_agents,
                    ),
                    "agent_status" => execute_agent_status(&params, registry),
                    "agent_join" => execute_agent_join(&params, registry).await,
                    _ => {
                        let r = execute_tool(&tool_name, params).await;
                        match r {
                            Some(r) => (r.output, r.is_error),
                            None => (format!("unknown tool: {tool_name}"), true),
                        }
                    }
                }
            } else {
                let result = execute_tool(&tool_name, params).await;
                match result {
                    Some(r) => (r.output, r.is_error),
                    None => (format!("unknown tool: {tool_name}"), true),
                }
            };

            let action = sink.on_tool_end(&tool_name, &output, is_error);
            session.push(ConversationMessage::tool_result(
                tool_use_id,
                output,
                is_error,
            ));

            // Self-update: sink detected a new binary — break cleanly after
            // persisting this tool result, then signal the caller.
            if let PostToolAction::ReplaceAndExec(path) = action {
                replace_and_exec = Some(path);
                break;
            }
        }

        // Self-update was requested — exit the loop cleanly
        if replace_and_exec.is_some() {
            break;
        }

        // Drain any queued interjections and inject them as user messages.
        // This lets the model see "steer me" signals without breaking the
        // tool-use alternation protocol.
        if let Some(rx) = interjections.as_deref_mut() {
            let mut injected = false;
            while let Ok(msg) = rx.try_recv() {
                sink.on_interjection(&msg);
                session.push(ConversationMessage::user(&format!(
                    "<interjection>\n{msg}\n</interjection>"
                )));
                injected = true;
            }
            if injected {
                // Force another iteration so the model can respond to the interjection.
                stop_reason = StopReason::ToolUse;
            }
        }

        // only loop if stop_reason was tool_use
        if stop_reason != StopReason::ToolUse {
            break;
        }
    }

    tracker.finish_turn();
    sink.on_turn_complete(&tracker.cumulative, iterations);

    TurnResult {
        iterations,
        usage: tracker.cumulative.clone(),
        stream_error,
        replace_and_exec,
    }
}

// ---------------------------------------------------------------------------
// Stream a single API call, collect events into blocks
// ---------------------------------------------------------------------------

async fn stream_response(
    provider: &dyn Provider,
    request: MessageRequest,
    sink: &mut dyn OutputSink,
) -> (Vec<ContentBlock>, TokenUsage, StopReason, Option<String>) {
    let mut stream = provider.stream_message(request);

    let mut text_buf = String::new();
    let mut tool_calls: std::collections::HashMap<String, (String, String)> =
        std::collections::HashMap::new();
    let mut tool_order: Vec<String> = Vec::new();
    let mut usage = TokenUsage::default();
    let mut stop_reason = StopReason::EndTurn;
    let mut stream_err: Option<String> = None;

    // map __idx_N → real tool id for delta correlation
    let mut idx_to_id: std::collections::HashMap<String, String> = std::collections::HashMap::new();

    while let Some(event) = stream.next().await {
        match event {
            Err(e) => {
                stream_err = Some(e.to_string());
                break;
            }
            Ok(Event::TextDelta { text }) => {
                sink.on_text(&text);
                text_buf.push_str(&text);
            }
            Ok(Event::ToolUseStart { id, name }) => {
                tool_calls.insert(id.clone(), (name, String::new()));
                tool_order.push(id.clone());
                // Map any index-based placeholder ids → the real id.
                // Two naming conventions in use:
                //   Anthropic: __idx_N  (from anthropic.rs, 0-indexed by arrival order)
                //   OAI-compat: __tc_N  (from openai_compat.rs, uses tc.index from wire)
                // We register both so either convention resolves correctly.
                let arrival_idx = tool_order.len() - 1;
                idx_to_id.insert(format!("__idx_{arrival_idx}"), id.clone());
                idx_to_id.insert(format!("__tc_{arrival_idx}"), id.clone());
                // Also register the id itself as an identity mapping (no-op lookup)
                idx_to_id.insert(id.clone(), id.clone());
            }
            Ok(Event::ToolUseDelta { id, partial_json }) => {
                // resolve placeholder id → real id (or use directly if already real)
                let real_id = idx_to_id.get(&id).cloned().unwrap_or_else(|| id.clone());
                if let Some(entry) = tool_calls.get_mut(&real_id) {
                    entry.1.push_str(&partial_json);
                }
            }
            Ok(Event::ToolUseEnd { .. }) => {} // nothing needed
            Ok(Event::MessageStop { stop_reason: sr }) => {
                stop_reason = sr;
            }
            Ok(Event::UsageDelta { usage: u }) => {
                usage.accumulate(&u);
            }
        }
    }

    // assemble content blocks
    let mut blocks: Vec<ContentBlock> = Vec::new();

    if !text_buf.is_empty() {
        blocks.push(ContentBlock::Text { text: text_buf });
    }

    for id in tool_order {
        if let Some((name, json_str)) = tool_calls.remove(&id) {
            let input: serde_json::Value = serde_json::from_str(&json_str)
                .unwrap_or(serde_json::Value::Object(serde_json::Map::new()));
            blocks.push(ContentBlock::ToolUse { id, name, input });
        }
    }

    (blocks, usage, stop_reason, stream_err)
}

// ---------------------------------------------------------------------------
// Build MessageRequest from session history
// ---------------------------------------------------------------------------

const DYNAMIC_BOUNDARY: &str = "__PIKU_SYSTEM_PROMPT_DYNAMIC_BOUNDARY__";

fn build_request(
    session: &Session,
    model: &str,
    system_prompt: &[String],
    tool_defs: &[ToolDefinition],
) -> MessageRequest {
    // Split system prompt at the boundary marker.
    // Everything before it is static and can be cached by Anthropic.
    // Everything after is dynamic (cwd, date, git status) and changes per turn.
    let boundary_pos = system_prompt.iter().position(|s| s == DYNAMIC_BOUNDARY);

    let system = {
        let mut blocks: Vec<piku_api::SystemBlock> = Vec::new();
        if let Some(idx) = boundary_pos {
            // Static prefix → single cached block
            let static_text = system_prompt[..idx].join("\n\n");
            if !static_text.is_empty() {
                blocks.push(piku_api::SystemBlock::cached(static_text));
            }
            // Dynamic suffix → uncached block
            let dynamic_text = system_prompt[idx + 1..].join("\n\n");
            if !dynamic_text.is_empty() {
                blocks.push(piku_api::SystemBlock::text(dynamic_text));
            }
        } else {
            // No boundary — send everything as a single uncached block
            let text = system_prompt.join("\n\n");
            if !text.is_empty() {
                blocks.push(piku_api::SystemBlock::text(text));
            }
        }
        if blocks.is_empty() {
            None
        } else {
            Some(blocks)
        }
    };

    // Convert session messages → API messages
    // System-role messages from compaction are injected as user-role text
    // with a clear wrapper (the API doesn't support mid-conversation system messages).
    let mut api_messages: Vec<RequestMessage> = Vec::new();

    for msg in &session.messages {
        match msg.role {
            MessageRole::System => {
                // inject as user message with wrapper
                let text = msg
                    .blocks
                    .iter()
                    .filter_map(|b| {
                        if let ContentBlock::Text { text } = b {
                            Some(text.as_str())
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                api_messages.push(RequestMessage {
                    role: "user".to_string(),
                    content: vec![RequestContent::Text {
                        text: format!("<system>\n{text}\n</system>"),
                    }],
                });
            }
            MessageRole::User => {
                let content: Vec<RequestContent> = msg
                    .blocks
                    .iter()
                    .map(|b| match b {
                        ContentBlock::Text { text } => RequestContent::Text { text: text.clone() },
                        ContentBlock::ToolResult {
                            tool_use_id,
                            output,
                            is_error,
                        } => RequestContent::ToolResult {
                            tool_use_id: tool_use_id.clone(),
                            content: output.clone(),
                            is_error: Some(*is_error),
                        },
                        _ => RequestContent::Text {
                            text: String::new(),
                        },
                    })
                    .collect();
                api_messages.push(RequestMessage {
                    role: "user".to_string(),
                    content,
                });
            }
            MessageRole::Tool => {
                // tool results go as user role
                let content: Vec<RequestContent> = msg
                    .blocks
                    .iter()
                    .filter_map(|b| {
                        if let ContentBlock::ToolResult {
                            tool_use_id,
                            output,
                            is_error,
                        } = b
                        {
                            Some(RequestContent::ToolResult {
                                tool_use_id: tool_use_id.clone(),
                                content: output.clone(),
                                is_error: Some(*is_error),
                            })
                        } else {
                            None
                        }
                    })
                    .collect();
                if !content.is_empty() {
                    api_messages.push(RequestMessage {
                        role: "user".to_string(),
                        content,
                    });
                }
            }
            MessageRole::Assistant => {
                let content: Vec<RequestContent> = msg
                    .blocks
                    .iter()
                    .map(|b| match b {
                        ContentBlock::Text { text } => RequestContent::Text { text: text.clone() },
                        ContentBlock::ToolUse { id, name, input } => RequestContent::ToolUse {
                            id: id.clone(),
                            name: name.clone(),
                            input: input.clone(),
                        },
                        _ => RequestContent::Text {
                            text: String::new(),
                        },
                    })
                    .collect();
                api_messages.push(RequestMessage {
                    role: "assistant".to_string(),
                    content,
                });
            }
        }
    }

    // Coalesce consecutive same-role messages to avoid protocol violations.
    // This can happen after a ReplaceAndExec: the session ends with a
    // Tool-role result (rendered as user), and run_turn prepends a new user
    // message — producing two consecutive user messages which most providers
    // reject with a 400.
    let api_messages = coalesce_consecutive_roles(api_messages);

    MessageRequest {
        model: model.to_string(),
        max_tokens: 8192,
        messages: api_messages,
        system,
        tools: Some(tool_defs.to_vec()),
        stream: true,
    }
}

// ---------------------------------------------------------------------------
// Background agent execution
// ---------------------------------------------------------------------------

// Execute `spawn_agent`: fork a background tokio task running `run_turn_inner`
// with a fresh session and a budget cap. Returns immediately with the task_id.
// Note: foreground mode runs inline; background uses spawn_local.
fn execute_spawn_agent(
    params: &serde_json::Value,
    registry: &TaskRegistry,
    provider: &dyn Provider,
    model: &str,
    system_prompt: &[String],
    tool_defs: &[ToolDefinition],
    depth: u32,
    parent_session_messages: &[crate::session::ConversationMessage],
    custom_agents: &[crate::agents::CustomAgentDef],
) -> (String, bool) {
    if depth >= MAX_SPAWN_DEPTH {
        return (
            format!("spawn_agent refused: maximum recursion depth ({MAX_SPAWN_DEPTH}) reached"),
            true,
        );
    }

    let p = match piku_tools::spawn_agent::validate_spawn_agent(params.clone()) {
        Ok(v) => v,
        Err(e) => return (format!("spawn_agent: {e}"), true),
    };

    // Worktree isolation: create a temp git worktree if requested
    let (worktree_path, worktree_branch, cwd_override) =
        if p.isolation == piku_tools::spawn_agent::Isolation::Worktree {
            let repo_root = std::env::current_dir().unwrap_or_default();
            // We need a task_id early to name the worktree
            let tmp_id = crate::task::AgentTaskId::new();
            match crate::task::create_worktree(&repo_root, &tmp_id) {
                Ok((wt_path, branch)) => {
                    let cwd = wt_path.clone();
                    (Some(wt_path), Some(branch), Some(cwd))
                }
                Err(e) => return (format!("worktree creation failed: {e}"), true),
            }
        } else {
            (None, None, None)
        };

    // Resolve agent definition: check custom agents first, then built-ins.
    let agent_def = p
        .subagent_type
        .as_deref()
        .and_then(|t| crate::agents::find_agent(t, custom_agents));

    // Build system prompt for subagent:
    // - named agent type overrides the main prompt
    // - otherwise use the parent's system prompt unchanged
    // Agent-type-specific memory is appended when available.
    let sub_system_prompt: Vec<String> = if let Some(ref def) = agent_def {
        let mut prompt = def.system_prompt().to_string();
        // Inject per-agent-type persistent memory
        let cwd = std::env::current_dir().unwrap_or_default();
        if let Some(mem_prompt) = crate::memory::build_agent_memory_prompt(&cwd, def.agent_type()) {
            prompt.push_str(&mem_prompt);
        }
        vec![prompt]
    } else {
        system_prompt.to_vec()
    };

    // Filter tool defs via the agent's allowlist/blocklist.
    let sub_tool_defs: Vec<ToolDefinition> = if let Some(ref def) = agent_def {
        tool_defs
            .iter()
            .filter(|t| def.is_tool_allowed(&t.name))
            .cloned()
            .collect()
    } else {
        tool_defs.to_vec()
    };

    // Build the task prompt — prepend context file contents if requested
    let mut prompt = p.task.clone();

    // Fork: prepend parent session history as <fork_context>.
    // Includes user text, assistant text, and tool call names+results
    // (results masked to 200 chars to keep context lean).
    if p.fork && !parent_session_messages.is_empty() {
        let mut fork_ctx = String::from("\n\n<fork_context>\n");
        fork_ctx.push_str(
            "The following is the conversation history from the parent agent. \
             Use it for context — you do not need to repeat work already done.\n\n",
        );
        for msg in parent_session_messages {
            let role = match msg.role {
                crate::session::MessageRole::User => "user",
                crate::session::MessageRole::Assistant => "assistant",
                crate::session::MessageRole::Tool => "tool",
                crate::session::MessageRole::System => continue,
            };
            for block in &msg.blocks {
                match block {
                    crate::session::ContentBlock::Text { text } if !text.trim().is_empty() => {
                        fork_ctx.push('[');
                        fork_ctx.push_str(role);
                        fork_ctx.push_str("]: ");
                        let trunc = if text.chars().count() > 500 {
                            let s: String = text.chars().take(500).collect();
                            format!("{s}...")
                        } else {
                            text.clone()
                        };
                        fork_ctx.push_str(&trunc);
                        fork_ctx.push('\n');
                    }
                    crate::session::ContentBlock::ToolUse { name, .. } => {
                        fork_ctx.push_str("[tool_use]: ");
                        fork_ctx.push_str(name);
                        fork_ctx.push('\n');
                    }
                    crate::session::ContentBlock::ToolResult { output, .. }
                        if !output.trim().is_empty() =>
                    {
                        fork_ctx.push_str("[tool_result]: ");
                        if output.len() > 200 {
                            let preview: String = output.chars().take(100).collect();
                            let _ = write!(fork_ctx, "{preview}... ({} chars)", output.len());
                        } else {
                            fork_ctx.push_str(output);
                        }
                        fork_ctx.push('\n');
                    }
                    _ => {}
                }
            }
        }
        fork_ctx.push_str("</fork_context>");
        prompt.push_str(&fork_ctx);
    }

    if !p.context_files.is_empty() {
        let mut ctx = String::from("\n\n<context>\n");
        for path in &p.context_files {
            match std::fs::read_to_string(path) {
                Ok(content) => {
                    ctx.push_str("# ");
                    ctx.push_str(path);
                    ctx.push('\n');
                    ctx.push_str(&content);
                    ctx.push_str("\n\n");
                }
                Err(e) => {
                    ctx.push_str("# ");
                    ctx.push_str(path);
                    ctx.push('\n');
                    ctx.push_str("[could not read: ");
                    ctx.push_str(&e.to_string());
                    ctx.push_str("]\n\n");
                }
            }
        }
        ctx.push_str("</context>");
        prompt.push_str(&ctx);
    }
    // Proactive recall: embed the task prompt and retrieve relevant memories.
    // Only for top-level spawns (depth == 0) -- subagents already get focused context,
    // and block_in_place panics inside spawn_local (which subagents run in).
    if depth == 0 {
        let cwd = std::env::current_dir().unwrap_or_default();
        let store_path = crate::embed_memory::default_store_path(&cwd);
        let mut store = crate::embed_memory::MemoryStore::load(&store_path);
        if store.valid_count() > 0 {
            let ollama_url = std::env::var("OLLAMA_HOST")
                .unwrap_or_else(|_| "http://localhost:11434".to_string());
            let embed_model = std::env::var("PIKU_EMBED_MODEL")
                .unwrap_or_else(|_| "nomic-embed-text".to_string());
            let query_text: String = p.task.chars().take(500).collect();
            let query_result = tokio::task::block_in_place(|| {
                tokio::runtime::Handle::current().block_on(async {
                    tokio::time::timeout(
                        std::time::Duration::from_secs(5),
                        crate::embed_memory::embed_text(&query_text, &ollama_url, &embed_model),
                    )
                    .await
                })
            });
            if let Ok(Ok(query_vec)) = query_result {
                let retrieved = store.hybrid_retrieve(&query_vec, &query_text, 5);
                if !retrieved.is_empty() {
                    let mem_section = crate::embed_memory::format_retrieved_memories(&retrieved);
                    prompt.push_str(&mem_section);
                    let _ = store.save(&store_path);
                }
            }
        }
    }

    if let Some(ref wt) = worktree_path {
        prompt.push_str("\n\nYou are running in an isolated git worktree at ");
        prompt.push_str(&wt.display().to_string());
        prompt.push_str(". You can freely edit files here without affecting the main checkout.");
    }

    // Per-agent turn limit: agent def's max_turns overrides the default
    // when the model didn't explicitly set one in the spawn params.
    let effective_max_turns = if let Some(ref def) = agent_def {
        def.max_turns().unwrap_or(p.max_turns)
    } else {
        p.max_turns
    };

    let task_id = registry.register(
        p.name.clone(),
        p.task.clone(),
        depth + 1,
        worktree_path.clone(),
    );

    let registry_clone = registry.clone();
    let task_id_clone = task_id.clone();
    let provider_clone: Box<dyn Provider + Send + Sync + 'static> = provider.boxed_clone();
    let model_owned = model.to_string();
    let wt_branch = worktree_branch;
    let wt_path_clone = cwd_override;

    // Build transparent spawn hint showing what the agent got.
    let tool_count = sub_tool_defs.len();
    let type_info = if let Some(ref def) = agent_def {
        format!(
            " [type={}, tools={tool_count}, max_turns={effective_max_turns}]",
            def.agent_type()
        )
    } else {
        format!(" [tools={tool_count}, max_turns={effective_max_turns}]")
    };

    let hint = if p.background {
        format!("spawned agent {} ({}){type_info}", p.name, task_id)
    } else {
        format!(
            "spawned agent {} ({}){type_info} — use agent_join({}) to wait for result",
            p.name, task_id, task_id
        )
    };

    let custom_agents_owned = custom_agents.to_vec();
    let _handle = tokio::task::spawn_local(run_subagent_task(
        task_id_clone,
        prompt,
        provider_clone,
        model_owned,
        sub_system_prompt,
        sub_tool_defs,
        effective_max_turns,
        registry_clone,
        depth + 1,
        wt_path_clone,
        wt_branch,
        custom_agents_owned,
    ));

    (hint, false)
}

async fn run_subagent_task(
    task_id: crate::task::AgentTaskId,
    prompt: String,
    provider: Box<dyn Provider + Send + Sync + 'static>,
    model: String,
    system_prompt: Vec<String>,
    tool_defs: Vec<ToolDefinition>,
    max_turns: u32,
    registry: crate::task::TaskRegistry,
    depth: u32,
    worktree_cwd: Option<std::path::PathBuf>,
    worktree_branch: Option<String>,
    custom_agents: Vec<crate::agents::CustomAgentDef>,
) {
    let mut session = crate::session::Session::new(format!("subagent-{task_id}"));
    let mut sink = crate::task::DevNullSink;

    // For worktree isolation: inject cwd as the first user message so the
    // agent knows to cd there. We do NOT call set_current_dir — that mutates
    // global process state and races with the parent's tool calls.
    let effective_prompt = if let Some(ref wt) = worktree_cwd {
        let cwd_str = wt.display().to_string();
        format!(
            "{prompt}\n\n<system-reminder>Your working directory is {cwd_str}. \
             Start every bash command with `cd {cwd_str} &&` or use absolute paths. \
             This is an isolated git worktree — your changes do not affect the main checkout.\
             </system-reminder>"
        )
    } else {
        prompt
    };

    let repo_root = std::env::current_dir().unwrap_or_default();

    let result = run_turn_inner(
        &effective_prompt,
        &mut session,
        provider.as_ref(),
        &model,
        &system_prompt,
        tool_defs,
        &crate::permission::AllowAll,
        &mut sink,
        Some(max_turns),
        None,
        Some(&registry),
        depth,
        &custom_agents,
    )
    .await;

    // Worktree cleanup
    let worktree_result = if let (Some(ref wt_path), Some(ref branch)) =
        (&worktree_cwd, &worktree_branch)
    {
        let changed = session.messages.iter().any(|m| {
            m.blocks.iter().any(|b| matches!(b, crate::session::ContentBlock::ToolUse { name, .. } if matches!(name.as_str(), "write_file" | "edit_file")))
        });
        crate::task::cleanup_worktree(&repo_root, wt_path, branch, changed)
    } else {
        None
    };

    if let Some(err) = result.stream_error {
        registry.fail(&task_id, &err);
    } else {
        let mut output = extract_last_assistant_text(&session);
        if let Some(wt_kept) = worktree_result {
            output.push_str("\n\n[Changes saved in worktree: ");
            output.push_str(&wt_kept.display().to_string());
            output.push(']');
        }
        registry.complete(&task_id, &output, result.iterations);
    }
}

fn extract_last_assistant_text(session: &crate::session::Session) -> String {
    session
        .messages
        .iter()
        .rev()
        .find(|m| m.role == crate::session::MessageRole::Assistant)
        .and_then(|m| {
            m.blocks.iter().find_map(|b| match b {
                crate::session::ContentBlock::Text { text } if !text.trim().is_empty() => {
                    Some(text.clone())
                }
                _ => None,
            })
        })
        .unwrap_or_else(|| "(no output)".to_string())
}

/// Execute `agent_status`: poll or list tasks.
fn execute_agent_status(params: &serde_json::Value, registry: &TaskRegistry) -> (String, bool) {
    let task_id = params.get("task_id").and_then(|v| v.as_str());
    if let Some(id_str) = task_id {
        let id = crate::task::AgentTaskId(id_str.to_string());
        match registry.status(&id) {
            Some(entry) => {
                let elapsed = entry.elapsed().as_secs();
                let out = entry.output.as_deref().unwrap_or("");
                let summary = if out.chars().count() > 300 {
                    let s: String = out.chars().take(300).collect();
                    format!("{s}…")
                } else {
                    out.to_string()
                };
                (
                    format!(
                        "task {id_str}\nstatus: {}\ndepth: {}\nelapsed: {elapsed}s\nturns: {}\n\
                         output:\n{summary}",
                        entry.status, entry.depth, entry.turns_used
                    ),
                    false,
                )
            }
            None => (format!("unknown task: {id_str}"), true),
        }
    } else {
        let tasks = registry.all();
        if tasks.is_empty() {
            return ("no background tasks".to_string(), false);
        }
        let lines: Vec<String> = tasks
            .iter()
            .map(|t| {
                format!(
                    "{} [{}] depth={} elapsed={}s turns={}",
                    t.id,
                    t.status,
                    t.depth,
                    t.elapsed().as_secs(),
                    t.turns_used
                )
            })
            .collect();
        (lines.join("\n"), false)
    }
}

/// Execute `agent_join`: block until a task completes.
async fn execute_agent_join(params: &serde_json::Value, registry: &TaskRegistry) -> (String, bool) {
    let Some(id_str) = params.get("task_id").and_then(|v| v.as_str()) else {
        return ("agent_join requires task_id".to_string(), true);
    };
    let timeout_secs = params
        .get("timeout_secs")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(300);

    let id = crate::task::AgentTaskId(id_str.to_string());
    let rx = registry.wait_for(&id);

    match tokio::time::timeout(std::time::Duration::from_secs(timeout_secs), rx).await {
        Ok(Ok(entry)) => {
            let out = entry.output.as_deref().unwrap_or("(no output)");
            (format!("task {id_str} {}\n\n{out}", entry.status), false)
        }
        Ok(Err(_)) => (format!("task {id_str}: channel closed unexpectedly"), true),
        Err(_) => (
            format!("task {id_str}: timed out after {timeout_secs}s"),
            true,
        ),
    }
}

/// Attempt LLM-based compaction: send old messages to the model with the compact
/// prompt, collect the summary, and apply it. Returns `None` if the LLM call
/// fails (caller should fall back to structural compaction).
///
/// Uses a 15-second timeout to avoid blocking the loop on slow models.
async fn try_llm_compact(
    session: &Session,
    provider: &dyn Provider,
    model: &str,
    config: crate::compact::CompactionConfig,
) -> Option<crate::compact::CompactionResult> {
    let keep_from = session
        .messages
        .len()
        .saturating_sub(config.preserve_recent_messages);
    if keep_from == 0 {
        return None;
    }

    let removed = &session.messages[..keep_from];

    // Build the conversation to summarise (as a single user message).
    let mut conversation_text = String::new();
    for msg in removed {
        let role = match msg.role {
            MessageRole::User => "User",
            MessageRole::Assistant => "Assistant",
            MessageRole::Tool => "Tool",
            MessageRole::System => "System",
        };
        for block in &msg.blocks {
            let text = match block {
                ContentBlock::Text { text } => text.clone(),
                ContentBlock::ToolUse { name, input, .. } => {
                    format!("[tool_use: {name}({input})]")
                }
                ContentBlock::ToolResult { output, .. } => {
                    if output.len() > 300 {
                        let preview: String = output.chars().take(150).collect();
                        format!("[tool_result: {preview}... ({} chars)]", output.len())
                    } else {
                        format!("[tool_result: {output}]")
                    }
                }
            };
            if !text.trim().is_empty() {
                conversation_text.push_str(role);
                conversation_text.push_str(": ");
                conversation_text.push_str(&text);
                conversation_text.push('\n');
            }
        }
    }

    let compact_prompt = crate::compact::compact_system_prompt(None);
    let request = MessageRequest {
        model: model.to_string(),
        max_tokens: 4096,
        messages: vec![RequestMessage {
            role: "user".to_string(),
            content: vec![RequestContent::Text {
                text: conversation_text,
            }],
        }],
        system: Some(vec![piku_api::SystemBlock::text(compact_prompt)]),
        tools: None,
        stream: true,
    };

    // Stream with timeout
    let timeout = std::time::Duration::from_secs(15);
    let mut stream = provider.stream_message(request);
    let mut summary = String::new();

    let collect_result = tokio::time::timeout(timeout, async {
        while let Some(event) = stream.next().await {
            match event {
                Ok(Event::TextDelta { text }) => summary.push_str(&text),
                Err(_) => return Err(()),
                _ => {}
            }
        }
        Ok(())
    })
    .await;

    match collect_result {
        Ok(Ok(())) if !summary.trim().is_empty() => Some(crate::compact::apply_compact_summary(
            session, &summary, config,
        )),
        _ => None, // Timeout or error — caller falls back to structural
    }
}

/// Merge consecutive messages of the same role by combining their content blocks.
///
/// After `ReplaceAndExec`, the session ends with a Tool-role message (mapped to
/// user-role) followed by a new user message — two consecutive user-role messages
/// which most providers reject with a 400. Merging fixes the alternation.
fn coalesce_consecutive_roles(messages: Vec<RequestMessage>) -> Vec<RequestMessage> {
    let mut out: Vec<RequestMessage> = Vec::with_capacity(messages.len());
    for msg in messages {
        if let Some(last) = out.last_mut() {
            if last.role == msg.role {
                last.content.extend(msg.content);
                continue;
            }
        }
        out.push(msg);
    }
    out
}
