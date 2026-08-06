#![allow(
    clippy::doc_markdown,
    clippy::match_wildcard_for_single_variants,
    clippy::needless_borrows_for_generic_args,
    clippy::too_many_arguments,
    clippy::too_many_lines
)]

use std::{collections::HashSet, fmt::Write};

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

/// Shared cancellation flag. Set to `true` to abort the current turn.
pub type CancelFlag = std::sync::Arc<std::sync::atomic::AtomicBool>;

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

    /// Called with the context-window pressure after each turn completes
    /// (cumulative input tokens / model window size, clamped to [0, 1]).
    /// The TUI uses this to light up color thresholds in the footer and
    /// future work will use it to trigger rolling-summary compaction.
    /// Default: no-op.
    fn on_context_pressure(&mut self, _pressure: f32) {}

    /// Called after each provider stream completes, with how long it took.
    ///
    /// A turn's wall clock cannot separate a slow provider from a hang in the
    /// loop or the terminal; they are different defects with the same symptom.
    /// Default: no-op.
    fn on_provider_stream(&mut self, _elapsed_ms: u64, _blocks: usize, _stop_reason: &str) {}

    /// Called for durable semantic events. Presentation-only sinks may ignore
    /// these; [`crate::run_record::RecordingSink`] persists them.
    fn on_run_event(&mut self, _event: &crate::run_record::RunEvent) {}
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
    /// Set if the user cancelled the turn mid-execution.
    pub cancelled: bool,
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
        None,
        None,
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
    custom_agents: &[crate::agents::AgentDef],
    hook_registry: Option<&crate::hooks::HookRegistry>,
    cancel_flag: Option<&CancelFlag>,
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
        hook_registry,
        cancel_flag,
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
    custom_agents: &[crate::agents::AgentDef],
    hook_registry: Option<&crate::hooks::HookRegistry>,
    cancel_flag: Option<&CancelFlag>,
) -> TurnResult {
    let max = max_turns.unwrap_or(DEFAULT_MAX_TURNS);
    let mut interjections = interjections;
    let mut cancelled = false;

    // push user message
    session.push(ConversationMessage::user(input));
    sink.on_run_event(&crate::run_record::RunEvent::TurnStarted {
        provider: Some(provider.name().to_string()),
        model: model.to_string(),
        input: crate::run_record::ContentRef::Inline {
            text: input.to_string(),
        },
    });

    let turn_start = std::time::Instant::now();
    let mut tracker = UsageTracker::default();
    let mut iterations = 0u32;
    let mut stream_error: Option<String> = None;
    let mut replace_and_exec: Option<std::path::PathBuf> = None;

    // Dedup detection: canonical string key (tool_name + args JSON) for exact match.
    // Uses String keys instead of hashing to avoid collision risk on safety-critical dedup.
    let mut seen_tool_calls: std::collections::HashSet<String> = std::collections::HashSet::new();
    let advertised_tool_names: HashSet<String> = tool_defs.iter().map(|t| t.name.clone()).collect();

    // Track where we last extracted memories (message index).
    // Periodic extraction happens after compaction events.
    let mut last_extraction_idx: usize = session.messages.len();
    // Guard against concurrent extraction tasks racing on the store file.
    let extraction_in_flight = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));

    // Auto-compaction config — compact when session tokens exceed this threshold.
    // 10k tokens is ~40k chars, roughly 200 exchanges of average length.
    // Scale compaction trigger to the model's context window. For Claude
    // (200k) compaction fires at ~100k tokens; for Gemini (1M) it fires
    // much later. Step-2 curation handles normal-case budget pressure —
    // compaction here is the emergency release valve that rewrites old
    // content as a summary so curation has less to drop.
    let compact_cfg =
        crate::compact::CompactionConfig::for_window(piku_context_window_for_model(model));

    loop {
        if iterations >= max {
            break;
        }
        // Check cancellation flag before each iteration.
        if let Some(flag) = cancel_flag {
            if flag.load(std::sync::atomic::Ordering::Relaxed) {
                cancelled = true;
                sink.on_text("\n\x1b[2m[cancelled by user]\x1b[0m\n");
                break;
            }
        }
        iterations += 1;

        // Auto-compact: if the session is getting long, shed bulk.
        //
        // Strategy: observation-masking first, structural summary as fallback.
        // Never call an LLM in the auto path. Research (JetBrains NeurIPS
        // 2025, ACON Oct 2025, Anthropic engineering Sep 2025) converges on
        // masking > LLM summarization for automatic compaction:
        //   - LLM summarization causes 15% longer trajectories (smooths
        //     stopping signals) and ~7% per-instance cost overhead with
        //     near-zero cache reuse
        //   - Summaries drop exactly the load-bearing details that only
        //     matter later ("context rot")
        //   - Observation masking keeps reasoning + tool calls verbatim,
        //     sheds bulk from tool-output observations only
        // The richer `try_llm_compact` path is retained for a possible future
        // explicit compaction command. No such command is currently exposed.
        if crate::compact::should_compact(session, compact_cfg) {
            // PreCompact hook: let hooks veto compaction (exit 2 = block).
            let vetoed = hook_registry.is_some_and(|hooks| {
                let cwd = std::env::current_dir().unwrap_or_default();
                !hooks.run_pre_compact(&session.id, &cwd, session.messages.len(), "auto")
            });
            if vetoed {
                // Hook vetoed compaction -- skip this cycle.
            } else {
                let before_messages = session.messages.len();
                let result = crate::compact::compact_session(session, compact_cfg);
                let method = if result.removed_message_count == 0 {
                    "mask"
                } else {
                    "structural"
                };
                if result.removed_message_count > 0 {
                    let masked_tool_results = result
                        .compacted_session
                        .messages
                        .iter()
                        .flat_map(|message| &message.blocks)
                        .filter(|block| {
                            matches!(
                                block,
                                ContentBlock::ToolResult { output, .. }
                                    if output.starts_with("[masked:")
                            )
                        })
                        .count();
                    sink.on_run_event(&crate::run_record::RunEvent::CompactionApplied {
                        before_messages,
                        after_messages: result.compacted_session.messages.len(),
                        masked_tool_results,
                        summary: crate::run_record::ContentRef::Inline {
                            text: result.formatted_summary.clone(),
                        },
                    });
                    // Trigger memory extraction on the messages being compacted away.
                    // Guarded: skip if a previous extraction is still in flight (prevents
                    // race conditions on the store file).
                    let new_messages = &session.messages[last_extraction_idx..];
                    if !new_messages.is_empty()
                        && !extraction_in_flight.load(std::sync::atomic::Ordering::Relaxed)
                    {
                        let transcript =
                            crate::embed_memory::build_extraction_transcript(new_messages);
                        if !transcript.trim().is_empty() {
                            let provider_clone = provider.boxed_clone();
                            let model_owned = model.to_string();
                            let cwd = std::env::current_dir().unwrap_or_default();
                            let store_path = crate::embed_memory::default_store_path(&cwd);
                            let embed_config = crate::embed_memory::EmbedConfig::from_env();
                            let flag = extraction_in_flight.clone();
                            flag.store(true, std::sync::atomic::Ordering::Relaxed);
                            tokio::task::spawn_local(async move {
                                let mut store = crate::embed_memory::MemoryStore::load(&store_path);
                                let n = crate::embed_memory::extract_and_store(
                                    &transcript,
                                    provider_clone.as_ref(),
                                    &model_owned,
                                    &mut store,
                                    &embed_config,
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
            } // else compact_allowed
        }

        let (request, manifest) = build_request(session, model, system_prompt, &tool_defs);
        sink.on_run_event(&crate::run_record::RunEvent::ContextBuilt { manifest });

        // stream the response
        let stream_started = std::time::Instant::now();
        let (mut assistant_blocks, usage, mut stop_reason, maybe_err) =
            stream_response(provider, request, sink).await;
        #[allow(clippy::cast_possible_truncation)]
        sink.on_provider_stream(
            stream_started.elapsed().as_millis() as u64,
            assistant_blocks.len(),
            &format!("{stop_reason:?}"),
        );

        if let Some(err) = maybe_err {
            stream_error = Some(err);
            // Don't push a partial assistant message on stream error — break cleanly
            tracker.record(usage);
            break;
        }

        tracker.record(usage.clone());

        // A provider can legally end an SSE stream without text or a tool
        // call. Do not turn that into a silent user-visible no-op: preserve a
        // clear, retryable assistant message in both the sink and session.
        if assistant_blocks.is_empty() {
            const EMPTY_RESPONSE: &str =
                "The model returned an empty response. Please retry your request.";
            sink.on_text(EMPTY_RESPONSE);
            sink.on_text("\n");
            assistant_blocks.push(ContentBlock::Text {
                text: EMPTY_RESPONSE.to_string(),
            });
        }

        let assistant_text = assistant_blocks
            .iter()
            .filter_map(|block| match block {
                ContentBlock::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("");
        sink.on_run_event(&crate::run_record::RunEvent::AssistantMessage {
            content: crate::run_record::ContentRef::Inline {
                text: assistant_text,
            },
        });

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
        let cwd_for_hooks = std::env::current_dir().unwrap_or_default();
        let session_id_for_hooks = session.id.clone();

        for (tool_use_id, tool_name, params) in &tool_calls {
            sink.on_run_event(&crate::run_record::RunEvent::ToolStarted {
                tool_call_id: tool_use_id.clone(),
                name: tool_name.clone(),
                arguments: params.clone(),
            });
            if !advertised_tool_names.contains(tool_name) {
                let msg = format!("{tool_name} is not available in this mode");
                sink.on_tool_start(tool_name, tool_use_id, params);
                sink.on_tool_end(tool_name, &msg, true);
                session.push(ConversationMessage::tool_result(
                    tool_use_id.clone(),
                    msg.clone(),
                    true,
                ));
                sink.on_run_event(&crate::run_record::RunEvent::ToolCompleted {
                    tool_call_id: tool_use_id.clone(),
                    result: crate::run_record::ContentRef::Inline { text: msg },
                    is_error: true,
                    effects: Vec::new(),
                    verification: None,
                });
                continue;
            }

            // PreToolUse hooks (can block before permission check)
            let mut hook_context: Option<String> = None;
            if let Some(hooks) = hook_registry {
                let hook_result = hooks.run_pre_tool_use(
                    tool_name,
                    params,
                    &session_id_for_hooks,
                    &cwd_for_hooks,
                );
                if let crate::hooks::HookDecision::Deny(reason) = hook_result.decision {
                    sink.on_tool_start(tool_name, tool_use_id, params);
                    sink.on_tool_end(tool_name, &reason, true);
                    session.push(ConversationMessage::tool_result(
                        tool_use_id.clone(),
                        format!("Blocked by hook: {reason}"),
                        true,
                    ));
                    sink.on_run_event(&crate::run_record::RunEvent::ToolCompleted {
                        tool_call_id: tool_use_id.clone(),
                        result: crate::run_record::ContentRef::Inline {
                            text: format!("Blocked by hook: {reason}"),
                        },
                        is_error: true,
                        effects: Vec::new(),
                        verification: None,
                    });
                    continue;
                }
                hook_context = hook_result.context;
            }

            // permission check
            match check_permission(tool_name, params, prompter) {
                PermissionOutcome::Allow => {
                    sink.on_run_event(&crate::run_record::RunEvent::PermissionDecision {
                        tool_call_id: tool_use_id.clone(),
                        decision: crate::run_record::PermissionDecision::Allowed,
                    });
                }
                PermissionOutcome::Deny { reason } => {
                    sink.on_run_event(&crate::run_record::RunEvent::PermissionDecision {
                        tool_call_id: tool_use_id.clone(),
                        decision: crate::run_record::PermissionDecision::Denied,
                    });
                    sink.on_permission_denied(tool_name, &reason);
                    let result = format!("Permission denied: {reason}");
                    session.push(ConversationMessage::tool_result(
                        tool_use_id.clone(),
                        result.clone(),
                        true,
                    ));
                    sink.on_run_event(&crate::run_record::RunEvent::ToolCompleted {
                        tool_call_id: tool_use_id.clone(),
                        result: crate::run_record::ContentRef::Inline { text: result },
                        is_error: true,
                        effects: Vec::new(),
                        verification: None,
                    });
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
                    sink.on_tool_start(tool_name, tool_use_id, params);
                    sink.on_tool_end(tool_name, &dedup_msg, true);
                    session.push(ConversationMessage::tool_result(
                        tool_use_id.clone(),
                        dedup_msg.clone(),
                        true,
                    ));
                    sink.on_run_event(&crate::run_record::RunEvent::ToolCompleted {
                        tool_call_id: tool_use_id.clone(),
                        result: crate::run_record::ContentRef::Inline { text: dedup_msg },
                        is_error: true,
                        effects: Vec::new(),
                        verification: None,
                    });
                    continue;
                }
            }

            sink.on_tool_start(tool_name, tool_use_id, params);

            // Route special tools through the runtime; everything else through execute_tool.
            let mut tool_effects = Vec::new();
            let mut verification = None;
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
                        let embed_config = crate::embed_memory::EmbedConfig::from_env();
                        // Use .await directly -- safe in both parent and subagent async contexts.
                        // Do NOT use block_in_place here (panics inside spawn_local).
                        let embed_result = tokio::time::timeout(
                            std::time::Duration::from_secs(5),
                            crate::embed_memory::embed_text_with_config(query, &embed_config),
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
                let result = piku_tools::embed_memory_tool::execute_manage_memory(
                    params.clone(),
                    &mut store,
                );
                // Only save on mutating actions
                if is_mutating {
                    let _ = store.save(&store_path);
                }
                unpack_tool_result(result, &mut tool_effects, &mut verification)
            } else if tool_name == "record_attempt" {
                // Record an attempt in the embedding store -- needs embedding
                let cwd = std::env::current_dir().unwrap_or_default();
                let store_path = crate::embed_memory::default_store_path(&cwd);
                let mut store = crate::embed_memory::MemoryStore::load(&store_path);

                let goal = params.get("goal").and_then(|v| v.as_str()).unwrap_or("");
                let approach = params
                    .get("approach")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let attempt_id = params.get("attempt_id").and_then(serde_json::Value::as_u64);

                if goal.trim().is_empty() || approach.trim().is_empty() {
                    (
                        "record_attempt requires non-empty 'goal' and 'approach'".to_string(),
                        true,
                    )
                } else if let Some(existing_id) = attempt_id {
                    // Updating an existing attempt's outcome
                    let outcome = params
                        .get("outcome")
                        .and_then(|v| v.as_str())
                        .unwrap_or("pending");
                    let detail = params
                        .get("outcome_detail")
                        .and_then(|v| v.as_str())
                        .map(String::from);
                    let outcome_enum = match outcome {
                        "success" => crate::embed_memory::Outcome::Success,
                        "failure" => crate::embed_memory::Outcome::Failure,
                        _ => crate::embed_memory::Outcome::Pending,
                    };
                    if store.record_outcome(existing_id, outcome_enum, detail) {
                        let _ = store.save(&store_path);
                        (format!("attempt {existing_id} updated: {outcome}"), false)
                    } else {
                        (format!("attempt {existing_id} not found"), true)
                    }
                } else {
                    // Creating a new attempt -- need to embed the approach
                    let embed_config = crate::embed_memory::EmbedConfig::from_env();
                    let embed_text = format!("{goal} | {approach}");
                    let embed_result = tokio::time::timeout(
                        std::time::Duration::from_secs(5),
                        crate::embed_memory::embed_text_with_config(&embed_text, &embed_config),
                    )
                    .await;
                    match embed_result {
                        Ok(Ok(embedding)) => {
                            let parent_id =
                                params.get("parent_id").and_then(serde_json::Value::as_u64);
                            #[allow(clippy::cast_possible_truncation)]
                            let importance = params
                                .get("importance")
                                .and_then(serde_json::Value::as_u64)
                                .unwrap_or(6) as u8;
                            let id = store.record_attempt(
                                goal.to_string(),
                                approach.to_string(),
                                parent_id,
                                embedding,
                                importance,
                            );
                            // Apply immediate outcome if provided
                            let outcome = params.get("outcome").and_then(|v| v.as_str());
                            if let Some(outcome_str) = outcome {
                                let detail = params
                                    .get("outcome_detail")
                                    .and_then(|v| v.as_str())
                                    .map(String::from);
                                let outcome_enum = match outcome_str {
                                    "success" => crate::embed_memory::Outcome::Success,
                                    "failure" => crate::embed_memory::Outcome::Failure,
                                    _ => crate::embed_memory::Outcome::Pending,
                                };
                                store.record_outcome(id, outcome_enum, detail);
                            }
                            let _ = store.save(&store_path);
                            let status = params
                                .get("outcome")
                                .and_then(|v| v.as_str())
                                .unwrap_or("pending");
                            (
                                format!("attempt recorded (id={id}, status={status})"),
                                false,
                            )
                        }
                        _ => (
                            "record_attempt: embedding service unavailable".to_string(),
                            true,
                        ),
                    }
                }
            } else if tool_name == "query_attempts" {
                // Query attempt trees by goal similarity -- needs embedding
                let cwd = std::env::current_dir().unwrap_or_default();
                let store_path = crate::embed_memory::default_store_path(&cwd);
                let store = crate::embed_memory::MemoryStore::load(&store_path);
                let goal = params.get("goal").and_then(|v| v.as_str()).unwrap_or("");
                if goal.trim().is_empty() {
                    (
                        "query_attempts requires a non-empty 'goal'".to_string(),
                        true,
                    )
                } else if store.valid_count() == 0 {
                    (
                        "No attempts recorded yet. Use record_attempt to log what you try."
                            .to_string(),
                        false,
                    )
                } else {
                    let embed_config = crate::embed_memory::EmbedConfig::from_env();
                    let embed_result = tokio::time::timeout(
                        std::time::Duration::from_secs(5),
                        crate::embed_memory::embed_text_with_config(goal, &embed_config),
                    )
                    .await;
                    match embed_result {
                        Ok(Ok(goal_vec)) => {
                            #[allow(clippy::cast_possible_truncation)]
                            let max_trees = params
                                .get("max_trees")
                                .and_then(serde_json::Value::as_u64)
                                .unwrap_or(3) as usize;
                            // find_attempt_trees takes &self, no need for mut
                            let trees = store.find_attempt_trees(&goal_vec, goal, max_trees);
                            if trees.is_empty() {
                                (
                                    "No prior attempts found for a similar goal.".to_string(),
                                    false,
                                )
                            } else {
                                (crate::embed_memory::format_attempt_trees(&trees), false)
                            }
                        }
                        _ => (
                            "query_attempts: embedding service unavailable".to_string(),
                            true,
                        ),
                    }
                }
            } else if tool_name == "tool_search" {
                // Build catalog from current tool_defs for on-demand search
                let catalog: Vec<piku_tools::tool_search::SearchableToolEntry> = tool_defs
                    .iter()
                    .map(|t| piku_tools::tool_search::SearchableToolEntry {
                        name: t.name.clone(),
                        description: t.description.clone(),
                    })
                    .collect();
                let r = piku_tools::tool_search::execute_tool_search(params.clone(), &catalog);
                unpack_tool_result(r, &mut tool_effects, &mut verification)
            } else if let Some(registry) = task_registry {
                match tool_name.as_str() {
                    "spawn_agent" => {
                        let (hint, is_error, child_ref) = execute_spawn_agent(
                            params,
                            registry,
                            provider,
                            model,
                            system_prompt,
                            &tool_defs,
                            depth,
                            &session.id,
                            &session.messages,
                            custom_agents,
                            hook_registry,
                        );
                        if let Some(child_ref) = child_ref {
                            sink.on_run_event(&child_ref);
                        }
                        (hint, is_error)
                    }
                    "agent_status" => execute_agent_status(params, registry),
                    "agent_join" => execute_agent_join(params, registry).await,
                    _ => {
                        let r = execute_tool(tool_name, params.clone()).await;
                        match r {
                            Some(r) => unpack_tool_result(r, &mut tool_effects, &mut verification),
                            None => (format!("unknown tool: {tool_name}"), true),
                        }
                    }
                }
            } else {
                let result = execute_tool(tool_name, params.clone()).await;
                match result {
                    Some(r) => unpack_tool_result(r, &mut tool_effects, &mut verification),
                    None => (format!("unknown tool: {tool_name}"), true),
                }
            };

            // Append hook-injected context to the tool output so the model sees it.
            let output = if let Some(ctx) = hook_context {
                format!("{output}\n\n<hook-context>\n{ctx}\n</hook-context>")
            } else {
                output
            };

            let action = sink.on_tool_end(tool_name, &output, is_error);
            if !is_error && matches!(tool_name.as_str(), "write_file" | "edit_file" | "bash") {
                seen_tool_calls.clear();
            }

            // PostToolUse hooks -- params is still available since we borrow tool_calls.
            if let Some(hooks) = hook_registry {
                hooks.run_post_tool_use(
                    tool_name,
                    params,
                    &output,
                    is_error,
                    &session_id_for_hooks,
                    &cwd_for_hooks,
                );
            }

            session.push(ConversationMessage::tool_result(
                tool_use_id.clone(),
                output.clone(),
                is_error,
            ));
            sink.on_run_event(&crate::run_record::RunEvent::ToolCompleted {
                tool_call_id: tool_use_id.clone(),
                result: crate::run_record::ContentRef::Inline { text: output },
                is_error,
                effects: tool_effects,
                verification,
            });

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

    // Score message relevance for context curation.
    session.score_messages();

    tracker.finish_turn();
    sink.on_turn_complete(&tracker.cumulative, iterations);

    let completion_reason = if replace_and_exec.is_some() {
        "replace_and_exec"
    } else if cancelled {
        "cancelled"
    } else if stream_error.is_some() {
        "error"
    } else if iterations >= max {
        "max_turns"
    } else {
        "end_turn"
    };
    sink.on_run_event(&crate::run_record::RunEvent::TurnCompleted {
        usage: crate::run_record::UsageRecord {
            input_tokens: u64::from(tracker.cumulative.input_tokens),
            output_tokens: u64::from(tracker.cumulative.output_tokens),
        },
        stop_reason: Some(completion_reason.to_string()),
    });

    // Context pressure: clamped input_tokens / model_window. We only need
    // ~2 significant figures for a 0-1 ratio, so u32→f32 precision loss
    // is fine. `pressure` is always in [0, 1].
    #[allow(clippy::cast_precision_loss)]
    let window = piku_context_window_for_model(model) as f32;
    #[allow(clippy::cast_precision_loss)]
    let pressure = (tracker.cumulative.input_tokens as f32 / window).clamp(0.0, 1.0);
    sink.on_context_pressure(pressure);

    // Stop hooks -- fire after turn completes (notifications, logging, cleanup).
    if let Some(hooks) = hook_registry {
        let cwd = std::env::current_dir().unwrap_or_default();
        #[allow(clippy::cast_possible_truncation)] // turn durations won't exceed u64::MAX ms
        let duration_ms = turn_start.elapsed().as_millis() as u64;
        hooks.run_stop(
            &session.id,
            &cwd,
            iterations,
            completion_reason,
            &tracker.cumulative,
            duration_ms,
        );
    }

    TurnResult {
        iterations,
        usage: tracker.cumulative.clone(),
        stream_error,
        replace_and_exec,
        cancelled,
    }
}

fn unpack_tool_result(
    result: piku_tools::ToolResult,
    effects: &mut Vec<crate::run_record::ToolEffect>,
    verification: &mut Option<crate::run_record::VerificationRecord>,
) -> (String, bool) {
    effects.extend(result.effects.into_iter().map(Into::into));
    *verification = result.verification.map(Into::into);
    (result.output, result.is_error)
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
    // The tool call whose content block is currently streaming.
    let mut open_tool_id: Option<String> = None;

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
                //   Anthropic: __idx_N  (from anthropic.rs, the wire's content
                //     block index, which counts text blocks too)
                //   OAI-compat: __tc_N  (from openai_compat.rs, uses tc.index)
                // We register both so either convention resolves correctly.
                let arrival_idx = tool_order.len() - 1;
                idx_to_id.insert(format!("__idx_{arrival_idx}"), id.clone());
                idx_to_id.insert(format!("__tc_{arrival_idx}"), id.clone());
                // Also register the id itself as an identity mapping (no-op lookup)
                idx_to_id.insert(id.clone(), id.clone());
                // Arrival order and the Anthropic content-block index agree only
                // when no text block precedes the tool. When one does, deltas
                // arrive under a placeholder nobody registered, and the
                // arguments were being dropped: the tool then ran on an empty
                // object. Content blocks stream one at a time, so an
                // unrecognised delta belongs to the most recently started call.
                open_tool_id = Some(id.clone());
            }
            Ok(Event::ToolUseDelta { id, partial_json }) => {
                // Resolve placeholder → real id, falling back to the open call
                // for a placeholder that was never registered.
                let real_id = idx_to_id
                    .get(&id)
                    .cloned()
                    .or_else(|| {
                        (!tool_calls.contains_key(&id))
                            .then(|| open_tool_id.clone())
                            .flatten()
                    })
                    .unwrap_or_else(|| id.clone());
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

/// Context-window estimate used by curation. Mirrors (a subset of)
/// piku's `context_window_for` in tui_repl.rs. Duplicated because
/// piku-runtime can't depend on the piku binary crate. Override via
/// `PIKU_CONTEXT_WINDOW=<tokens>`.
fn piku_context_window_for_model(model: &str) -> usize {
    if let Ok(s) = std::env::var("PIKU_CONTEXT_WINDOW") {
        if let Ok(n) = s.parse::<usize>() {
            if n > 0 {
                return n;
            }
        }
    }
    let m = model.to_ascii_lowercase();
    if m.contains("gemini-2") || m.contains("gemini-3") || m.contains("gemini-1.5") {
        return 1_000_000;
    }
    if m.contains("gpt-4o") || m.contains("gpt-4-turbo") || m.contains("gpt-4.1") {
        return 128_000;
    }
    // Claude, GPT-5, o1, o3, Ollama fallback — 200k is a safe default.
    200_000
}

/// Budget-aware curation tail size: always keep the last N messages
/// regardless of importance. Empirically 6 preserves the recent task
/// context + any open tool_use/tool_result pair without being so large
/// that it blows the budget on its own.
const CURATION_TAIL_SIZE: usize = 6;

/// Fraction of the context window reserved for system prompt, tool
/// definitions, and the LLM's response. Anything above this budget gets
/// curated out.
const CONTEXT_USAGE_CAP: f32 = 0.70;

/// Select which messages to include in the LLM request when the session
/// exceeds the budget. Zone assembly:
///   1. Last `CURATION_TAIL_SIZE` messages are always kept.
///   2. Remaining "head" messages are ranked by importance (from
///      Session::score_messages) and filled into the remaining budget.
///   3. Output is in chronological order.
///
/// Pair invariant caveat: an assistant's tool_use and the next user's
/// tool_result are distinct messages. If a tool_use is kept but its
/// result is dropped (or vice versa), Anthropic / OpenAI will 400. The
/// tail usually includes recent pairs together; for older pairs we keep
/// them atomic by walking in chronological pairs when the earlier
/// message of a pair is selected.
///
/// `model_window` is the provider's context window in tokens.
#[cfg(test)]
pub(crate) fn curate_messages(
    messages: &[crate::session::ConversationMessage],
    model_window: usize,
) -> Vec<&crate::session::ConversationMessage> {
    curate_message_indices(messages, model_window)
        .into_iter()
        .map(|index| &messages[index])
        .collect()
}

fn curate_message_indices(
    messages: &[crate::session::ConversationMessage],
    model_window: usize,
) -> Vec<usize> {
    let total: usize = messages
        .iter()
        .map(crate::session::ConversationMessage::estimated_tokens)
        .sum();
    // usize→f32→usize for a 0.7 scaling factor. Precision loss is immaterial
    // at the token budgets we deal with (10k–1M); the usize cast rounds toward
    // zero which is what we want (underestimate the budget, not overestimate).
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    let budget = ((model_window as f32) * CONTEXT_USAGE_CAP) as usize;
    if total <= budget || messages.len() <= CURATION_TAIL_SIZE {
        return (0..messages.len()).collect();
    }

    let tail_start = messages.len() - CURATION_TAIL_SIZE;
    let tail_tokens: usize = messages[tail_start..]
        .iter()
        .map(crate::session::ConversationMessage::estimated_tokens)
        .sum();
    let mut remaining_budget = budget.saturating_sub(tail_tokens);

    // Rank head messages by importance desc, with chronological index as
    // tiebreaker (prefer more recent when importance ties).
    let mut head_ranked: Vec<usize> = (0..tail_start).collect();
    head_ranked.sort_by(|&a, &b| {
        let ia = messages[a].importance.unwrap_or(0.0);
        let ib = messages[b].importance.unwrap_or(0.0);
        ib.partial_cmp(&ia)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| b.cmp(&a))
    });

    // Include messages greedily by importance until budget exhausted.
    // Pair invariant: if we include an Assistant message that has a
    // ToolUse block, the following Tool message (its result) must also
    // be included or the request is malformed.
    let mut included: std::collections::BTreeSet<usize> = (tail_start..messages.len()).collect();
    for idx in head_ranked {
        if included.contains(&idx) {
            continue;
        }
        // Compute pair: this message + its paired follower, if any.
        let pair_end = pair_end_for(messages, idx);
        let pair_cost: usize = (idx..=pair_end)
            .map(|i| messages[i].estimated_tokens())
            .sum();
        if pair_cost <= remaining_budget {
            for i in idx..=pair_end {
                included.insert(i);
            }
            remaining_budget -= pair_cost;
        }
    }

    included.into_iter().collect()
}

/// Find the last index of the pair that starts at `idx`. If the message
/// at `idx` has a ToolUse block, the pair extends to include the
/// following tool_result (role=Tool) message. Otherwise the pair is just
/// `idx` itself.
fn pair_end_for(messages: &[crate::session::ConversationMessage], idx: usize) -> usize {
    let has_tool_use = messages[idx].role == crate::session::MessageRole::Assistant
        && messages[idx]
            .blocks
            .iter()
            .any(|b| matches!(b, crate::session::ContentBlock::ToolUse { .. }));
    if has_tool_use
        && idx + 1 < messages.len()
        && messages[idx + 1].role == crate::session::MessageRole::Tool
    {
        idx + 1
    } else {
        idx
    }
}

fn build_request(
    session: &Session,
    model: &str,
    system_prompt: &[String],
    tool_defs: &[ToolDefinition],
) -> (MessageRequest, crate::run_record::ContextManifest) {
    // Split system prompt at the boundary marker.
    // Everything before it is static and can be cached by Anthropic.
    // Everything after is dynamic (cwd, date, git status) and changes per turn.
    let boundary_pos = system_prompt.iter().position(|s| s == DYNAMIC_BOUNDARY);

    let mut manifest_system_sections = Vec::new();
    let system = {
        let mut blocks: Vec<piku_api::SystemBlock> = Vec::new();
        if let Some(idx) = boundary_pos {
            // Static prefix → single cached block
            let static_text = system_prompt[..idx].join("\n\n");
            if !static_text.is_empty() {
                manifest_system_sections.push(crate::run_record::ContextSection {
                    label: "static".to_string(),
                    estimated_tokens: static_text.len() / 4 + 1,
                    selected: true,
                    reason: "cached_static_prefix".to_string(),
                });
                blocks.push(piku_api::SystemBlock::cached(static_text));
            }
            // Dynamic suffix → uncached block
            let dynamic_text = system_prompt[idx + 1..].join("\n\n");
            if !dynamic_text.is_empty() {
                manifest_system_sections.push(crate::run_record::ContextSection {
                    label: "dynamic".to_string(),
                    estimated_tokens: dynamic_text.len() / 4 + 1,
                    selected: true,
                    reason: "uncached_dynamic_suffix".to_string(),
                });
                blocks.push(piku_api::SystemBlock::text(dynamic_text));
            }
        } else {
            // No boundary — send everything as a single uncached block
            let text = system_prompt.join("\n\n");
            if !text.is_empty() {
                manifest_system_sections.push(crate::run_record::ContextSection {
                    label: "system".to_string(),
                    estimated_tokens: text.len() / 4 + 1,
                    selected: true,
                    reason: "uncached_system_prompt".to_string(),
                });
                blocks.push(piku_api::SystemBlock::text(text));
            }
        }
        if blocks.is_empty() {
            None
        } else {
            Some(blocks)
        }
    };

    // Convert session messages → API messages.
    //
    // Curation: when the session exceeds ~70% of the model's context
    // window, drop lower-importance messages to stay under budget.
    // Always keeps the last 6 messages (tail) + any high-importance
    // older messages that fit. See `curate_messages`.
    let window = piku_context_window_for_model(model);
    let selected_indices = curate_message_indices(&session.messages, window);
    let selected_set: std::collections::BTreeSet<usize> =
        selected_indices.iter().copied().collect();
    let all_messages_selected = selected_indices.len() == session.messages.len();
    let tail_start = session.messages.len().saturating_sub(CURATION_TAIL_SIZE);
    let manifest_messages = session
        .messages
        .iter()
        .enumerate()
        .map(|(index, message)| {
            let selected = selected_set.contains(&index);
            let reason = if all_messages_selected {
                "within_context_budget"
            } else if selected && index >= tail_start {
                "recent_tail"
            } else if selected {
                "importance_budget"
            } else {
                "excluded_by_importance_budget"
            };
            crate::run_record::ContextMessage {
                session_index: index,
                role: message_role_name(&message.role).to_string(),
                estimated_tokens: message.estimated_tokens(),
                selected,
                reason: reason.to_string(),
            }
        })
        .collect::<Vec<_>>();
    let selected = selected_indices
        .iter()
        .map(|&index| &session.messages[index]);

    // System-role messages from compaction are injected as user-role text
    // with a clear wrapper (the API doesn't support mid-conversation system messages).
    let mut api_messages: Vec<RequestMessage> = Vec::new();

    for msg in selected {
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

    let manifest_tools = tool_defs
        .iter()
        .map(|tool| crate::run_record::ContextTool {
            name: tool.name.clone(),
            estimated_tokens: serde_json::to_string(tool).map_or(0, |json| json.len() / 4 + 1),
            selected: true,
            reason: "advertised_to_provider".to_string(),
        })
        .collect::<Vec<_>>();
    let estimated_input_tokens = manifest_system_sections
        .iter()
        .map(|section| section.estimated_tokens)
        .sum::<usize>()
        + manifest_messages
            .iter()
            .filter(|message| message.selected)
            .map(|message| message.estimated_tokens)
            .sum::<usize>()
        + manifest_tools
            .iter()
            .map(|tool| tool.estimated_tokens)
            .sum::<usize>();
    let manifest = crate::run_record::ContextManifest {
        model: model.to_string(),
        context_window_tokens: window,
        estimated_input_tokens,
        system_sections: manifest_system_sections,
        messages: manifest_messages,
        tools: manifest_tools,
    };

    (
        MessageRequest {
            model: model.to_string(),
            max_tokens: 8192,
            messages: api_messages,
            system,
            tools: Some(tool_defs.to_vec()),
            stream: true,
        },
        manifest,
    )
}

fn message_role_name(role: &MessageRole) -> &'static str {
    match role {
        MessageRole::User => "user",
        MessageRole::Assistant => "assistant",
        MessageRole::Tool => "tool",
        MessageRole::System => "system",
    }
}

// ---------------------------------------------------------------------------
// Background agent execution
// ---------------------------------------------------------------------------

// Execute `spawn_agent`: fork a background tokio task running `run_turn_inner`
// with a fresh session and a budget cap. Returns immediately with the task_id.
// The current implementation always uses spawn_local. A non-background spawn
// is still asynchronous, so its response must point callers to agent_join
// rather than implying that execution completed inline.
fn execute_spawn_agent(
    params: &serde_json::Value,
    registry: &TaskRegistry,
    provider: &dyn Provider,
    model: &str,
    system_prompt: &[String],
    tool_defs: &[ToolDefinition],
    depth: u32,
    parent_session_id: &str,
    parent_session_messages: &[crate::session::ConversationMessage],
    custom_agents: &[crate::agents::AgentDef],
    hook_registry: Option<&crate::hooks::HookRegistry>,
) -> (String, bool, Option<crate::run_record::RunEvent>) {
    if depth >= MAX_SPAWN_DEPTH {
        return (
            format!("spawn_agent refused: maximum recursion depth ({MAX_SPAWN_DEPTH}) reached"),
            true,
            None,
        );
    }

    let p = match piku_tools::spawn_agent::validate_spawn_agent(params.clone()) {
        Ok(v) => v,
        Err(e) => return (format!("spawn_agent: {e}"), true, None),
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
                Err(e) => return (format!("worktree creation failed: {e}"), true, None),
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
        let mut prompt = def.system_prompt.clone();
        // Inject per-agent-type persistent memory
        let cwd = std::env::current_dir().unwrap_or_default();
        if let Some(mem_prompt) = crate::memory::build_agent_memory_prompt(&cwd, &def.agent_type) {
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
        let project_root = std::env::current_dir().unwrap_or_default();
        let mut ctx = String::from("\n\n<context>\n");
        for path in &p.context_files {
            // Path validation: restrict to project directory to prevent
            // sensitive file exfiltration (e.g. ~/.ssh/id_rsa sent to LLM).
            let resolved = std::path::Path::new(path);
            let abs_path = if resolved.is_absolute() {
                resolved.to_path_buf()
            } else {
                project_root.join(resolved)
            };
            if let Ok(canonical) = abs_path.canonicalize() {
                if !canonical.starts_with(&project_root) {
                    ctx.push_str("# ");
                    ctx.push_str(path);
                    ctx.push_str("\n[blocked: path is outside the project directory]\n\n");
                    continue;
                }
            }
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
    if depth == 0 {
        let cwd = std::env::current_dir().unwrap_or_default();
        let store_path = crate::embed_memory::default_store_path(&cwd);
        let mut store = crate::embed_memory::MemoryStore::load(&store_path);
        if store.valid_count() > 0 {
            let embed_config = crate::embed_memory::EmbedConfig::from_env();
            let query_text: String = p.task.chars().take(500).collect();
            // This function is sync but the outer caller runs on a LocalSet
            // (current-thread scheduler), where block_in_place panics. Use a
            // dedicated std::thread + mini runtime to block until the embed
            // call finishes, without touching the outer runtime.
            let query_result = {
                let query_text_cl = query_text.clone();
                let embed_config_cl = embed_config.clone();
                let (tx, rx) = std::sync::mpsc::channel();
                std::thread::spawn(move || {
                    let Ok(rt) = tokio::runtime::Builder::new_current_thread()
                        .enable_all()
                        .build()
                    else {
                        return;
                    };
                    let result = rt.block_on(tokio::time::timeout(
                        std::time::Duration::from_secs(5),
                        crate::embed_memory::embed_text_with_config(
                            &query_text_cl,
                            &embed_config_cl,
                        ),
                    ));
                    let _ = tx.send(result);
                });
                rx.recv_timeout(std::time::Duration::from_secs(6))
                    .unwrap_or(Ok(Err("embed thread did not report".to_string())))
            };
            if let Ok(Ok(query_vec)) = query_result {
                let retrieved = store.hybrid_retrieve(&query_vec, &query_text, 5);
                if !retrieved.is_empty() {
                    let mem_section = crate::embed_memory::format_retrieved_memories(&retrieved);
                    prompt.push_str(&mem_section);
                    let _ = store.save(&store_path);
                }
                // Also inject relevant attempt trees so subagents avoid repeating failures.
                let attempt_trees = store.find_attempt_trees(&query_vec, &query_text, 3);
                if !attempt_trees.is_empty() {
                    prompt.push_str(&crate::embed_memory::format_attempt_trees(&attempt_trees));
                }
            }
        }
    }

    if let Some(ref wt) = worktree_path {
        prompt.push_str("\n\nA separate git worktree was allocated at ");
        prompt.push_str(&wt.display().to_string());
        prompt.push_str(
            ". Route all file and shell work there. This is prompt guidance, not enforced containment.",
        );
    }

    // Per-agent turn limit: agent def's max_turns overrides the default
    // when the model didn't explicitly set one in the spawn params.
    let effective_max_turns = if let Some(ref def) = agent_def {
        def.max_turns.unwrap_or(p.max_turns)
    } else {
        p.max_turns
    };

    let task_id = registry.register_for_parent(
        Some(parent_session_id),
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
            def.agent_type
        )
    } else {
        format!(" [tools={tool_count}, max_turns={effective_max_turns}]")
    };

    let hint = if p.background {
        format!(
            "spawned background agent {} ({}){type_info}",
            p.name, task_id
        )
    } else {
        format!(
            "spawned agent {} ({}){type_info} — use agent_join({}) to wait for result",
            p.name, task_id, task_id
        )
    };

    let custom_agents_owned = custom_agents.to_vec();
    let hooks_owned = hook_registry.cloned();

    // SubagentStart hook: fire synchronously before the subagent task
    // is spawned. The parent continues regardless of hook outcome —
    // subagent hooks are observability, not policy.
    if let Some(hr) = &hook_registry {
        let cwd = std::env::current_dir().unwrap_or_default();
        hr.run_subagent_start(
            &task_id.to_string(),
            agent_def.as_ref().map(|d| d.agent_type.as_str()),
            &p.task,
            &cwd,
        );
    }

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
        hooks_owned,
    ));

    // Build a typed parent→child run pointer for the durable parent record.
    // This is the evidence the static browser follows; parsing `agent_join`
    // prose is not sufficient because the relationship must survive reopening
    // the run from disk. The caller emits it through the parent sink.
    let child_ref = registry.evidence(&task_id).and_then(|evidence| {
        registry
            .child_refs_relative_to_parent(&evidence.child_session_id)
            .map(
                |(run_record_ref, session_ref)| crate::run_record::RunEvent::ChildRunRef {
                    relationship: "spawn_agent".to_string(),
                    child_session_id: evidence.child_session_id.clone(),
                    task_id: task_id.to_string(),
                    run_record_ref,
                    session_ref,
                },
            )
    });

    (hint, false, child_ref)
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
    custom_agents: Vec<crate::agents::AgentDef>,
    hook_registry: Option<crate::hooks::HookRegistry>,
) {
    let evidence = registry.evidence(&task_id);
    if evidence.is_some() {
        if let Err(error) = registry.persist_evidence_link(&task_id) {
            fail_subagent_setup(
                &registry,
                &task_id,
                &format!("could not persist subagent link: {error}"),
                worktree_cwd.as_deref(),
                worktree_branch.as_deref(),
                hook_registry.as_ref(),
            );
            return;
        }
    }

    let child_session_id = evidence.as_ref().map_or_else(
        || format!("subagent-{task_id}"),
        |item| item.child_session_id.clone(),
    );
    let mut session = crate::session::Session::new(child_session_id);
    session.record_provider(provider.name(), &model);
    let mut recorder = match evidence.as_ref() {
        Some(item) => match crate::run_record::RunRecorder::open(
            &item.run_record_path,
            &item.child_session_id,
        ) {
            Ok(recorder) => Some(recorder),
            Err(error) => {
                fail_subagent_setup(
                    &registry,
                    &task_id,
                    &format!("could not open subagent run record: {error}"),
                    worktree_cwd.as_deref(),
                    worktree_branch.as_deref(),
                    hook_registry.as_ref(),
                );
                return;
            }
        },
        None => None,
    };
    let mut sink = crate::task::DevNullSink;

    // Worktree guard: Drop attempts to clean up the worktree and branch if
    // this future unwinds or returns before the explicit cleanup below. Defused
    // on the happy path so the existing `changed`-aware cleanup can decide
    // whether to keep the worktree.
    let repo_root_for_guard = std::env::current_dir().unwrap_or_default();
    let mut worktree_guard = match (&worktree_cwd, &worktree_branch) {
        (Some(wt), Some(branch)) => Some(crate::task::WorktreeGuard::new(
            repo_root_for_guard,
            wt.clone(),
            branch.clone(),
        )),
        _ => None,
    };

    // For worktree routing: inject cwd as the first user message so the agent
    // knows to cd there. This is guidance, not containment. We do NOT call
    // set_current_dir because that mutates global process state and races with
    // the parent's tool calls.
    let effective_prompt = if let Some(ref wt) = worktree_cwd {
        let cwd_str = wt.display().to_string();
        format!(
            "{prompt}\n\n<system-reminder>Your working directory is {cwd_str}. \
             Start every bash command with `cd {cwd_str} &&` or use absolute paths. \
             A separate git worktree was allocated for this task. Keep every change there.\
             </system-reminder>"
        )
    } else {
        prompt
    };

    let repo_root = std::env::current_dir().unwrap_or_default();

    let (result, record_error) = if let Some(recorder) = recorder.as_mut() {
        let mut recording_sink =
            crate::run_record::RecordingSink::new(&mut sink, recorder, format!("turn-{depth}-0"));
        let result = run_turn_inner(
            &effective_prompt,
            &mut session,
            provider.as_ref(),
            &model,
            &system_prompt,
            tool_defs,
            &crate::permission::AllowAll,
            &mut recording_sink,
            Some(max_turns),
            None,
            Some(&registry),
            depth,
            &custom_agents,
            hook_registry.as_ref(),
            None, // subagents don't support mid-turn cancel
        )
        .await;
        (result, recording_sink.take_record_error())
    } else {
        (
            run_turn_inner(
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
                hook_registry.as_ref(),
                None, // subagents don't support mid-turn cancel
            )
            .await,
            None,
        )
    };

    let durability_error = record_error.map(|error| format!("run record failed: {error}"));
    let durability_error = match (durability_error, evidence.as_ref()) {
        (None, Some(item)) => session
            .save(&item.session_path)
            .err()
            .map(|error| format!("session save failed: {error}")),
        (error, _) => error,
    };

    // Worktree cleanup. Happy path: defuse the guard so its Drop doesn't
    // unconditionally remove the worktree; then delegate to the existing
    // `changed`-aware cleanup (which keeps the worktree when the agent
    // produced file changes).
    let worktree_result = if let (Some(ref wt_path), Some(ref branch)) =
        (&worktree_cwd, &worktree_branch)
    {
        if let Some(g) = worktree_guard.as_mut() {
            g.defuse();
        }
        let changed = session.messages.iter().any(|m| {
            m.blocks.iter().any(|b| matches!(b, crate::session::ContentBlock::ToolUse { name, .. } if matches!(name.as_str(), "write_file" | "edit_file")))
        });
        crate::task::cleanup_worktree(&repo_root, wt_path, branch, changed)
    } else {
        None
    };
    // Drop order: if we somehow skipped the defuse branch above, the guard
    // will still fire on function return.
    drop(worktree_guard);

    let (status, iterations) = if let Some(error) = durability_error {
        registry.fail(&task_id, &error);
        ("failed", result.iterations)
    } else if let Some(err) = result.stream_error {
        registry.fail(&task_id, &err);
        ("failed", result.iterations)
    } else {
        let mut output = extract_last_assistant_text(&session);
        if let Some(wt_kept) = worktree_result {
            output.push_str("\n\n[Changes saved in worktree: ");
            output.push_str(&wt_kept.display().to_string());
            output.push(']');
        }
        registry.complete(&task_id, &output, result.iterations);
        ("done", result.iterations)
    };

    // SubagentStop hook. Observational; no veto. We don't have the
    // agent_type here — it was consumed in execute_spawn_agent — so pass
    // None. Could thread it through if needed later.
    if let Some(hr) = &hook_registry {
        hr.run_subagent_stop(&task_id.to_string(), None, status, iterations, &repo_root);
    }
}

fn fail_subagent_setup(
    registry: &crate::task::TaskRegistry,
    task_id: &crate::task::AgentTaskId,
    reason: &str,
    worktree_cwd: Option<&std::path::Path>,
    worktree_branch: Option<&str>,
    hook_registry: Option<&crate::hooks::HookRegistry>,
) {
    let repo_root = std::env::current_dir().unwrap_or_default();
    if let (Some(worktree), Some(branch)) = (worktree_cwd, worktree_branch) {
        let _ = crate::task::cleanup_worktree(&repo_root, worktree, branch, false);
    }
    registry.fail(task_id, reason);
    if let Some(hooks) = hook_registry {
        hooks.run_subagent_stop(&task_id.to_string(), None, "failed", 0, &repo_root);
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
                         {}output:\n{summary}",
                        entry.status,
                        entry.depth,
                        entry.turns_used,
                        task_evidence_summary(&entry)
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
            (
                format!(
                    "task {id_str} {}\n{}\n{out}",
                    entry.status,
                    task_evidence_summary(&entry)
                ),
                false,
            )
        }
        Ok(Err(_)) => (format!("task {id_str}: channel closed unexpectedly"), true),
        Err(_) => (
            format!("task {id_str}: timed out after {timeout_secs}s"),
            true,
        ),
    }
}

fn task_evidence_summary(entry: &crate::task::TaskEntry) -> String {
    entry
        .evidence
        .as_ref()
        .map_or_else(String::new, |evidence| {
            format!(
                "child session: {}\nrun record: {}\nparent link: {}\n",
                evidence.session_path.display(),
                evidence.run_record_path.display(),
                evidence.link_path.display()
            )
        })
}

/// Attempt LLM-based compaction: send old messages to the model with the compact
/// prompt, collect the summary, and apply it. Returns `None` if the LLM call
/// fails (caller should fall back to structural compaction).
///
/// Uses a 15-second timeout to avoid blocking the loop on slow models.
///
/// Currently unused from the auto-compact path (research on cliff-edge
/// vs observation-masking moved the auto path to masking). Retained
/// because the richer summary is still the right choice for manual
/// `/compact` if that slash command is ever added.
#[allow(dead_code)]
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

// ---------------------------------------------------------------------------
// Runtime-invariant tests: the LocalSet + block_in_place trap
// ---------------------------------------------------------------------------

#[cfg(test)]
mod runtime_tests {
    //! These tests encode the runtime-shape invariants that two production
    //! panics have already violated (tui_repl.rs:1118 and this file's
    //! execute_spawn_agent proactive-recall block).
    //!
    //! The rule: inside a LocalSet task (which the TUI REPL wraps the whole
    //! agent loop in), `tokio::task::block_in_place` panics because a
    //! LocalSet pins tasks to the current thread. The fix is to use either
    //! `.await` directly (if the call site is async) or a dedicated
    //! `std::thread` with its own mini runtime (if the call site is sync).
    use std::time::Duration;

    /// Mirrors the dedicated-thread pattern used in `execute_spawn_agent`.
    fn blocking_async_from_sync_context() -> Result<u32, String> {
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let rt = match tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
            {
                Ok(r) => r,
                Err(e) => {
                    let _ = tx.send(Err(format!("runtime build: {e}")));
                    return;
                }
            };
            let result = rt.block_on(async {
                tokio::time::sleep(Duration::from_millis(10)).await;
                Ok::<u32, String>(42)
            });
            let _ = tx.send(result);
        });
        rx.recv_timeout(Duration::from_secs(2))
            .unwrap_or(Err("thread did not report".to_string()))
    }

    /// The fix in `execute_spawn_agent` uses std::thread + its own runtime
    /// to bridge from the sync caller back to async without touching the
    /// outer LocalSet runtime. This test verifies the bridge works from
    /// inside a LocalSet (the production context for the TUI REPL).
    ///
    /// If this test panics with "can call blocking only when running on the
    /// multi-threaded runtime", the fix has regressed to the block_in_place
    /// anti-pattern.
    #[tokio::test(flavor = "current_thread")]
    async fn dedicated_thread_bridge_works_inside_local_set() {
        let local = tokio::task::LocalSet::new();
        local
            .run_until(async {
                // Sync helper called from inside LocalSet — same shape as
                // execute_spawn_agent reaching embed_text_with_config.
                let result = blocking_async_from_sync_context();
                assert_eq!(result, Ok(42));
            })
            .await;
    }

    /// Negative control: demonstrates why the dedicated-thread pattern is
    /// needed. Calling `block_in_place` inside a LocalSet task panics. We
    /// don't want the code to do this; the test pins the invariant so a
    /// future refactor can't reintroduce it without this test failing.
    #[tokio::test(flavor = "current_thread")]
    async fn block_in_place_inside_local_set_panics() {
        let local = tokio::task::LocalSet::new();
        let panicked = local
            .run_until(async {
                let result = tokio::task::spawn_local(async {
                    // This is the anti-pattern. Catch the panic so the test
                    // process stays alive.
                    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        tokio::task::block_in_place(|| 42_u32)
                    }));
                    r.is_err()
                })
                .await
                .unwrap_or(false);
                result
            })
            .await;
        assert!(
            panicked,
            "block_in_place inside a LocalSet should panic — if this test \
             starts failing, tokio's semantics have changed and the \
             runtime-context rules guarding execute_spawn_agent / \
             tui_repl.rs may no longer apply"
        );
    }
}

#[cfg(test)]
mod curation_tests {
    use super::*;
    use crate::session::{ContentBlock, ConversationMessage, MessageRole};

    fn user_msg(text: &str, importance: Option<f32>) -> ConversationMessage {
        ConversationMessage {
            role: MessageRole::User,
            blocks: vec![ContentBlock::Text {
                text: text.to_string(),
            }],
            usage: None,
            importance,
        }
    }

    fn assistant_text(text: &str, importance: Option<f32>) -> ConversationMessage {
        ConversationMessage {
            role: MessageRole::Assistant,
            blocks: vec![ContentBlock::Text {
                text: text.to_string(),
            }],
            usage: None,
            importance,
        }
    }

    fn assistant_tool_use(tool_id: &str, name: &str) -> ConversationMessage {
        ConversationMessage {
            role: MessageRole::Assistant,
            blocks: vec![ContentBlock::ToolUse {
                id: tool_id.to_string(),
                name: name.to_string(),
                input: serde_json::json!({}),
            }],
            usage: None,
            importance: None,
        }
    }

    fn tool_result(tool_id: &str, output: &str) -> ConversationMessage {
        ConversationMessage {
            role: MessageRole::Tool,
            blocks: vec![ContentBlock::ToolResult {
                tool_use_id: tool_id.to_string(),
                output: output.to_string(),
                is_error: false,
            }],
            usage: None,
            importance: None,
        }
    }

    #[test]
    fn within_budget_returns_everything() {
        let msgs = vec![user_msg("a", None), assistant_text("b", None)];
        let kept = curate_messages(&msgs, 200_000);
        assert_eq!(kept.len(), 2);
    }

    #[test]
    fn tail_is_always_kept() {
        // 20 messages, tiny window → budget will bite. Last 6 must survive.
        let big = "x".repeat(100_000);
        let mut msgs: Vec<ConversationMessage> = (0..20)
            .map(|i| user_msg(&format!("{i}:{big}"), Some(0.0)))
            .collect();
        // Give one oldest message high importance.
        msgs[0].importance = Some(1.0);

        let kept = curate_messages(&msgs, 1_000_000);
        let kept_texts: Vec<String> = kept
            .iter()
            .flat_map(|m| {
                m.blocks.iter().filter_map(|b| match b {
                    ContentBlock::Text { text } => Some(text.clone()),
                    _ => None,
                })
            })
            .collect();
        // Last 6 are always present.
        for i in 14..20 {
            assert!(
                kept_texts.iter().any(|t| t.starts_with(&format!("{i}:"))),
                "tail message {i} missing"
            );
        }
    }

    #[test]
    fn higher_importance_beats_lower() {
        // Construct a scenario: tail is unimportant but fits; two head
        // candidates compete for remaining budget.
        let filler = "x".repeat(40_000); // ~10k tokens each
        let mut msgs: Vec<ConversationMessage> = Vec::new();
        // Head: 3 messages, importance 0.1, 0.9, 0.2 — only one fits after tail.
        msgs.push(user_msg(&format!("LOW_A:{filler}"), Some(0.1)));
        msgs.push(user_msg(&format!("HIGH:{filler}"), Some(0.9)));
        msgs.push(user_msg(&format!("LOW_B:{filler}"), Some(0.2)));
        // Tail: 6 tiny messages
        for i in 0..6 {
            msgs.push(user_msg(&format!("tail{i}"), Some(0.5)));
        }
        // Budget ~20k tokens → room for tail + 1 head candidate.
        let kept = curate_messages(&msgs, 30_000);
        let has_high = kept.iter().any(|m| match &m.blocks[0] {
            ContentBlock::Text { text } => text.starts_with("HIGH:"),
            _ => false,
        });
        assert!(has_high, "high-importance message should be kept");
    }

    #[test]
    #[allow(clippy::cast_precision_loss)] // test indices 0..10, exact f32 value irrelevant
    fn preserves_chronological_order() {
        let msgs: Vec<ConversationMessage> = (0..10)
            .map(|i| user_msg(&format!("m{i}"), Some(i as f32 / 10.0)))
            .collect();
        let kept = curate_messages(&msgs, 200_000);
        let kept_texts: Vec<&str> = kept
            .iter()
            .filter_map(|m| match &m.blocks[0] {
                ContentBlock::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect();
        // Should still be in order m0, m1, m2, ...
        let is_sorted = kept_texts
            .windows(2)
            .all(|w| w[0] < w[1] || w[0].len() < w[1].len());
        assert!(is_sorted, "order mangled: {kept_texts:?}");
    }

    #[test]
    #[allow(clippy::cast_precision_loss, clippy::float_cmp)] // clamp(0,1) yields exact 1.0
    fn pressure_matches_input_tokens_over_window() {
        // Direct unit test for the calculation. The actual `on_context_pressure`
        // wiring fires at end of run_turn — we exercise the formula only.
        let window = piku_context_window_for_model("claude-sonnet-4-6") as f32;
        // 50k input tokens against 200k window → 0.25 pressure.
        let p = (50_000_f32 / window).clamp(0.0, 1.0);
        assert!((p - 0.25).abs() < 0.01, "got {p}");
        // Overflow: 1M tokens → clamped to 1.0
        let p2 = (1_000_000_f32 / window).clamp(0.0, 1.0);
        assert_eq!(p2, 1.0);
    }

    #[test]
    fn tool_use_result_pair_kept_atomically_or_dropped() {
        // Construct: tail of 6 + old tool-use/result pair under budget pressure.
        // If the pair is dropped, neither appears; if kept, both appear.
        let filler = "x".repeat(40_000);
        let mut msgs: Vec<ConversationMessage> = Vec::new();
        // Head pair (importance 0.9): assistant tool_use + tool_result
        let mut tu = assistant_tool_use("t1", "bash");
        tu.importance = Some(0.9);
        msgs.push(tu);
        msgs.push(tool_result("t1", "output"));
        // More head filler with lower importance
        for i in 0..3 {
            msgs.push(user_msg(&format!("low_{i}:{filler}"), Some(0.0)));
        }
        // Tail
        for i in 0..6 {
            msgs.push(user_msg(&format!("tail{i}"), Some(0.5)));
        }
        let kept = curate_messages(&msgs, 30_000);
        let has_use = kept.iter().any(|m| {
            m.blocks
                .iter()
                .any(|b| matches!(b, ContentBlock::ToolUse { id, .. } if id == "t1"))
        });
        let has_result = kept.iter().any(|m| {
            m.blocks.iter().any(
                |b| matches!(b, ContentBlock::ToolResult { tool_use_id, .. } if tool_use_id == "t1"),
            )
        });
        assert_eq!(
            has_use, has_result,
            "tool_use and tool_result must be included or dropped together (use={has_use} result={has_result})"
        );
    }
}
