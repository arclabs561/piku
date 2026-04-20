/// Session compaction — summarise old context to stay within the context window.
///
/// When `should_compact` returns true, `compact_system_prompt` returns a system
/// message that the caller should prepend to the session (replacing old messages)
/// so the model can continue without losing the thread.
///
/// The compaction prompt structure mirrors Claude Code's
/// `src/services/compact/prompt.ts` — 9 sections, <analysis> scratchpad
/// stripped before returning the formatted summary.
///
/// piku supports LLM-based compaction via `try_llm_compact` in the agent loop,
/// which calls the same model to write a summary. If that fails or times out,
/// the structural `compact_session` is used as a fallback.
use crate::session::{ContentBlock, ConversationMessage, MessageRole, Session};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Tunable thresholds for compaction.
#[derive(Debug, Clone, Copy)]
pub struct CompactionConfig {
    /// Keep at least this many recent messages verbatim after the summary.
    /// Step-2 curation already handles ordinary budget pressure, so
    /// compaction should preserve enough recent history for continuity
    /// (12 turns ≈ 3 user+assistant+tool cycles of context).
    pub preserve_recent_messages: usize,
    /// Trigger compaction when estimated tokens exceed this value.
    /// Set to roughly 50% of the model's context window so compaction is
    /// the emergency release valve, not the first line of defense.
    pub max_estimated_tokens: usize,
}

impl Default for CompactionConfig {
    fn default() -> Self {
        Self::for_window(200_000)
    }
}

impl CompactionConfig {
    /// Build a config scaled to a given context window. Trigger is 50%
    /// of the window; below that, step-2 curation handles budget without
    /// destroying history. At 50%+ we summarize so curation has less to
    /// drop.
    #[must_use]
    pub fn for_window(window_tokens: usize) -> Self {
        Self {
            preserve_recent_messages: 12,
            max_estimated_tokens: window_tokens / 2,
        }
    }
}

// ---------------------------------------------------------------------------
// Token estimation
// ---------------------------------------------------------------------------

/// Rough token estimate for a session (4 chars ≈ 1 token).
#[must_use]
pub fn estimate_session_tokens(session: &Session) -> usize {
    session.messages.iter().map(estimate_message_tokens).sum()
}

fn estimate_message_tokens(msg: &ConversationMessage) -> usize {
    msg.blocks
        .iter()
        .map(|b| match b {
            ContentBlock::Text { text } => text.len() / 4 + 1,
            ContentBlock::ToolUse { name, input, .. } => {
                (name.len() + input.to_string().len()) / 4 + 1
            }
            ContentBlock::ToolResult { output, .. } => output.len() / 4 + 1,
        })
        .sum()
}

// ---------------------------------------------------------------------------
// Decision
// ---------------------------------------------------------------------------

/// Returns true when the session is large enough to warrant compaction.
#[must_use]
pub fn should_compact(session: &Session, config: CompactionConfig) -> bool {
    session.messages.len() > config.preserve_recent_messages
        && estimate_session_tokens(session) >= config.max_estimated_tokens
}

// ---------------------------------------------------------------------------
// The compact system prompt (for the LLM summarisation call)
// ---------------------------------------------------------------------------

/// Returns the system prompt to send to a summarisation model.
///
/// The caller should:
///   1. Send this + the messages to be summarised as a new single-turn request.
///   2. Pass the response to `format_compact_summary`.
///   3. Build a continuation message via `get_compact_continuation_message`.
///   4. Replace the old messages with `[continuation_message] + recent_kept`.
#[must_use]
pub fn compact_system_prompt(custom_instructions: Option<&str>) -> String {
    let mut prompt = format!("{NO_TOOLS_PREAMBLE}{BASE_COMPACT_PROMPT}");
    if let Some(extra) = custom_instructions {
        if !extra.trim().is_empty() {
            prompt.push_str("\n\nAdditional Instructions:\n");
            prompt.push_str(extra);
        }
    }
    prompt.push_str(NO_TOOLS_TRAILER);
    prompt
}

// ---------------------------------------------------------------------------
// Summary formatting
// ---------------------------------------------------------------------------

/// Strip the `<analysis>` scratchpad and normalise `<summary>` tags.
/// Mirrors `formatCompactSummary` from `compact/prompt.ts`.
#[must_use]
pub fn format_compact_summary(raw: &str) -> String {
    // Strip <analysis>…</analysis>
    let without_analysis = strip_tag_block(raw, "analysis");

    // Replace <summary>…</summary> with "Summary:\n…"
    let formatted = if let Some(content) = extract_tag_block(&without_analysis, "summary") {
        without_analysis.replace(
            &format!("<summary>{content}</summary>"),
            &format!("Summary:\n{}", content.trim()),
        )
    } else {
        without_analysis
    };

    collapse_blank_lines(&formatted).trim().to_string()
}

/// Build the user-visible continuation message that replaces old context.
/// Mirrors `getCompactUserSummaryMessage` from `compact/prompt.ts`.
#[must_use]
pub fn get_compact_continuation_message(
    summary: &str,
    suppress_follow_up_questions: bool,
    recent_messages_preserved: bool,
) -> String {
    let formatted = format_compact_summary(summary);
    let mut msg = format!(
        "This session is being continued from a previous conversation that ran out of context. \
         The summary below covers the earlier portion of the conversation.\n\n{formatted}"
    );
    if recent_messages_preserved {
        msg.push_str("\n\nRecent messages are preserved verbatim.");
    }
    if suppress_follow_up_questions {
        msg.push_str(
            "\nContinue the conversation from where it left off without asking the user any \
             further questions. Resume directly — do not acknowledge the summary, do not recap \
             what was happening, and do not preface with \"I'll continue\" or similar. \
             Pick up the last task as if the break never happened.",
        );
    }
    msg
}

// ---------------------------------------------------------------------------
// Compact result
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct CompactionResult {
    /// Raw summary text (may contain <analysis> tags — use `format_compact_summary`).
    pub summary: String,
    /// Pre-formatted summary (analysis stripped).
    pub formatted_summary: String,
    /// The new session after compaction.
    pub compacted_session: Session,
    /// How many messages were removed.
    pub removed_message_count: usize,
}

/// Compact a session without calling an LLM — builds a structural summary
/// from the message metadata (useful as a fast fallback or for testing).
///
/// Two-phase compaction (mirrors Claude Code's approach):
///   1. **Observation masking**: large tool results (>200 chars) in older
///      messages are replaced with a short preview. This preserves the
///      reasoning chain (text blocks, tool call names) while shedding bulk.
///   2. **Hard compaction**: if still over threshold after masking, the old
///      messages are replaced with a structural summary.
///
/// For a richer summary, call the LLM with `compact_system_prompt()` instead,
/// then pass its response to `apply_compact_summary`.
#[must_use]
pub fn compact_session(session: &Session, config: CompactionConfig) -> CompactionResult {
    if !should_compact(session, config) {
        return CompactionResult {
            summary: String::new(),
            formatted_summary: String::new(),
            compacted_session: session.clone(),
            removed_message_count: 0,
        };
    }

    // Phase 1: observation masking on older messages (keep last N untouched).
    let keep_from = session
        .messages
        .len()
        .saturating_sub(config.preserve_recent_messages);
    let masked = mask_observations(&session.messages[..keep_from]);
    let preserved = session.messages[keep_from..].to_vec();

    // Check if masking alone brought us under threshold.
    let masked_tokens: usize = masked.iter().map(estimate_message_tokens).sum::<usize>()
        + preserved.iter().map(estimate_message_tokens).sum::<usize>();

    if masked_tokens < config.max_estimated_tokens {
        // Masking was enough — reassemble without hard compaction.
        let mut all = masked;
        all.extend(preserved);
        return CompactionResult {
            summary: String::new(),
            formatted_summary: String::new(),
            compacted_session: Session {
                version: session.version,
                id: session.id.clone(),
                messages: all,
            },
            removed_message_count: 0,
        };
    }

    // Phase 2: hard compaction — summarise the masked old messages.
    let summary = build_structural_summary(&masked);
    let formatted_summary = format_compact_summary(&summary);
    let continuation = get_compact_continuation_message(&summary, true, !preserved.is_empty());

    let mut compacted_messages = vec![ConversationMessage::system(continuation)];
    compacted_messages.extend(preserved);

    CompactionResult {
        summary,
        formatted_summary,
        compacted_session: Session {
            version: session.version,
            id: session.id.clone(),
            messages: compacted_messages,
        },
        removed_message_count: masked.len(),
    }
}

/// Apply an LLM-generated summary to produce a compacted session.
#[must_use]
pub fn apply_compact_summary(
    session: &Session,
    summary: &str,
    config: CompactionConfig,
) -> CompactionResult {
    let keep_from = session
        .messages
        .len()
        .saturating_sub(config.preserve_recent_messages);
    let removed_count = keep_from;
    let preserved = session.messages[keep_from..].to_vec();

    let formatted_summary = format_compact_summary(summary);
    let continuation = get_compact_continuation_message(summary, true, !preserved.is_empty());

    let mut compacted_messages = vec![ConversationMessage::system(continuation)];
    compacted_messages.extend(preserved);

    CompactionResult {
        summary: summary.to_string(),
        formatted_summary,
        compacted_session: Session {
            version: session.version,
            id: session.id.clone(),
            messages: compacted_messages,
        },
        removed_message_count: removed_count,
    }
}

// ---------------------------------------------------------------------------
// Observation masking (phase 1 of compaction)
// ---------------------------------------------------------------------------

/// Threshold in chars above which a tool result is masked.
const MASK_THRESHOLD: usize = 200;

/// Replace large tool results with a short preview.
/// Preserves text blocks and tool-use blocks (reasoning chain) intact.
/// Only `ToolResult` outputs above `MASK_THRESHOLD` chars are masked.
fn mask_observations(messages: &[ConversationMessage]) -> Vec<ConversationMessage> {
    messages
        .iter()
        .map(|msg| {
            let blocks: Vec<ContentBlock> = msg
                .blocks
                .iter()
                .map(|b| match b {
                    ContentBlock::ToolResult {
                        tool_use_id,
                        output,
                        is_error,
                    } if output.len() > MASK_THRESHOLD => {
                        let preview: String = output.chars().take(100).collect();
                        ContentBlock::ToolResult {
                            tool_use_id: tool_use_id.clone(),
                            output: format!("[masked: {preview}... ({} chars)]", output.len()),
                            is_error: *is_error,
                        }
                    }
                    other => other.clone(),
                })
                .collect();
            ConversationMessage {
                role: msg.role.clone(),
                blocks,
                usage: msg.usage.clone(),
                importance: msg.importance,
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Structural summary (no LLM needed)
// ---------------------------------------------------------------------------

fn build_structural_summary(messages: &[ConversationMessage]) -> String {
    let user_count = messages
        .iter()
        .filter(|m| m.role == MessageRole::User)
        .count();
    let assistant_count = messages
        .iter()
        .filter(|m| m.role == MessageRole::Assistant)
        .count();
    let tool_count = messages
        .iter()
        .filter(|m| m.role == MessageRole::Tool)
        .count();

    let mut tool_names: Vec<&str> = messages
        .iter()
        .flat_map(|m| m.blocks.iter())
        .filter_map(|b| match b {
            ContentBlock::ToolUse { name, .. } => Some(name.as_str()),
            _ => None,
        })
        .collect();
    tool_names.sort_unstable();
    tool_names.dedup();

    let recent_user_requests: Vec<String> = messages
        .iter()
        .filter(|m| m.role == MessageRole::User)
        .rev()
        .filter_map(first_text_block)
        .take(3)
        .map(|t| truncate(t, 160))
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();

    let key_files = collect_key_files(messages);
    let current_work = messages
        .iter()
        .rev()
        .filter_map(first_text_block)
        .find(|t| !t.trim().is_empty())
        .map(|t| truncate(t, 200));

    let mut lines = vec![
        "<summary>".to_string(),
        "Conversation summary:".to_string(),
        format!(
            "- Scope: {} earlier messages compacted (user={user_count}, assistant={assistant_count}, tool={tool_count}).",
            messages.len()
        ),
    ];

    if !tool_names.is_empty() {
        lines.push(format!("- Tools mentioned: {}.", tool_names.join(", ")));
    }
    if !recent_user_requests.is_empty() {
        lines.push("- Recent user requests:".to_string());
        for r in &recent_user_requests {
            lines.push(format!("  - {r}"));
        }
    }
    if !key_files.is_empty() {
        lines.push(format!("- Key files referenced: {}.", key_files.join(", ")));
    }
    if let Some(work) = current_work {
        lines.push(format!("- Current work: {work}"));
    }

    lines.push("- Key timeline:".to_string());
    for msg in messages {
        let role = match msg.role {
            MessageRole::System => "system",
            MessageRole::User => "user",
            MessageRole::Assistant => "assistant",
            MessageRole::Tool => "tool",
        };
        let content = msg
            .blocks
            .iter()
            .map(summarise_block)
            .collect::<Vec<_>>()
            .join(" | ");
        lines.push(format!("  - {role}: {content}"));
    }
    lines.push("</summary>".to_string());
    lines.join("\n")
}

fn summarise_block(b: &ContentBlock) -> String {
    let raw = match b {
        ContentBlock::Text { text } => text.clone(),
        ContentBlock::ToolUse { name, input, .. } => format!("tool_use {name}({input})"),
        ContentBlock::ToolResult { output, .. } => output.clone(),
    };
    truncate(&raw, 160)
}

fn first_text_block(msg: &ConversationMessage) -> Option<&str> {
    msg.blocks.iter().find_map(|b| match b {
        ContentBlock::Text { text } if !text.trim().is_empty() => Some(text.as_str()),
        _ => None,
    })
}

fn collect_key_files(messages: &[ConversationMessage]) -> Vec<String> {
    let mut files: Vec<String> = messages
        .iter()
        .flat_map(|m| m.blocks.iter())
        .flat_map(|b| {
            let text = match b {
                ContentBlock::Text { text } => text.as_str(),
                ContentBlock::ToolUse { input, .. } => input.as_str().unwrap_or(""),
                ContentBlock::ToolResult { output, .. } => output.as_str(),
            };
            extract_file_candidates(text)
        })
        .collect();
    files.sort();
    files.dedup();
    files.into_iter().take(8).collect()
}

fn extract_file_candidates(text: &str) -> Vec<String> {
    text.split_whitespace()
        .filter_map(|token| {
            let c = token.trim_matches(|ch: char| {
                matches!(ch, ',' | '.' | ':' | ';' | ')' | '(' | '"' | '\'')
            });
            if c.contains('/') && has_code_extension(c) {
                Some(c.to_string())
            } else {
                None
            }
        })
        .collect()
}

fn has_code_extension(path: &str) -> bool {
    std::path::Path::new(path)
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| {
            matches!(
                e,
                "rs" | "ts" | "tsx" | "js" | "json" | "md" | "toml" | "yaml" | "py"
            )
        })
}

fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let mut t: String = s.chars().take(max.saturating_sub(1)).collect();
        t.push('…');
        t
    }
}

// ---------------------------------------------------------------------------
// String helpers (mirrors compact/prompt.ts formatCompactSummary helpers)
// ---------------------------------------------------------------------------

fn extract_tag_block(content: &str, tag: &str) -> Option<String> {
    let open = format!("<{tag}>");
    let close = format!("</{tag}>");
    let start = content.find(&open)? + open.len();
    let end = content[start..].find(&close)? + start;
    Some(content[start..end].to_string())
}

fn strip_tag_block(content: &str, tag: &str) -> String {
    let open = format!("<{tag}>");
    let close = format!("</{tag}>");
    let mut result = content.to_string();
    // Strip all occurrences (handles multiple <analysis> blocks from reasoning models)
    while let Some(s) = result.find(&open) {
        let Some(e_rel) = result[s + open.len()..].find(&close) else {
            break;
        };
        let e = s + open.len() + e_rel + close.len();
        result = format!("{}{}", &result[..s], &result[e..]);
    }
    result
}

fn collapse_blank_lines(s: &str) -> String {
    let mut out = String::new();
    let mut last_blank = false;
    for line in s.lines() {
        let blank = line.trim().is_empty();
        if blank && last_blank {
            continue;
        }
        out.push_str(line);
        out.push('\n');
        last_blank = blank;
    }
    out
}

// ---------------------------------------------------------------------------
// Prompt constants (from compact/prompt.ts)
// ---------------------------------------------------------------------------

const NO_TOOLS_PREAMBLE: &str = "CRITICAL: Respond with TEXT ONLY. Do NOT call any tools.\n\n\
- Do NOT use Read, Bash, Grep, Glob, Edit, Write, or ANY other tool.\n\
- You already have all the context you need in the conversation above.\n\
- Tool calls will be REJECTED and will waste your only turn — you will fail the task.\n\
- Your entire response must be plain text: an <analysis> block followed by a <summary> block.\n\n";

const NO_TOOLS_TRAILER: &str =
    "\n\nREMINDER: Do NOT call any tools. Respond with plain text only — \
an <analysis> block followed by a <summary> block. \
Tool calls will be rejected and you will fail the task.";

const BASE_COMPACT_PROMPT: &str = "Your task is to create a detailed summary of the conversation \
so far, paying close attention to the user's explicit requests and your previous actions.\n\
This summary should be thorough in capturing technical details, code patterns, and architectural \
decisions that would be essential for continuing development work without losing context.\n\n\
Before providing your final summary, wrap your analysis in <analysis> tags to organize your \
thoughts and ensure you've covered all necessary points.\n\n\
Your summary should include the following sections:\n\n\
1. Primary Request and Intent: Capture all of the user's explicit requests and intents in detail\n\
2. Key Technical Concepts: List all important technical concepts, technologies, and frameworks discussed.\n\
3. Files and Code Sections: Enumerate specific files and code sections examined, modified, or created.\n\
4. Errors and fixes: List all errors that you ran into, and how you fixed them.\n\
5. Problem Solving: Document problems solved and any ongoing troubleshooting efforts.\n\
6. All user messages: List ALL user messages that are not tool results.\n\
7. Pending Tasks: Outline any pending tasks that you have explicitly been asked to work on.\n\
8. Current Work: Describe in detail precisely what was being worked on immediately before this summary request.\n\
9. Optional Next Step: List the next step that you will take that is related to the most recent work.\n\n\
Wrap your summary in <summary>…</summary> tags.";

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::Session;

    fn make_session(n_messages: usize, chars_each: usize) -> Session {
        let mut s = Session::new("test".to_string());
        for i in 0..n_messages {
            if i % 2 == 0 {
                s.push(ConversationMessage::user("x".repeat(chars_each)));
            } else {
                s.push(ConversationMessage::assistant(
                    vec![ContentBlock::Text {
                        text: "y".repeat(chars_each),
                    }],
                    None,
                ));
            }
        }
        s
    }

    #[test]
    fn small_session_not_compacted() {
        let s = make_session(2, 10);
        let r = compact_session(&s, CompactionConfig::default());
        assert_eq!(r.removed_message_count, 0);
        assert_eq!(r.compacted_session.messages.len(), s.messages.len());
    }

    #[test]
    fn large_session_is_compacted() {
        let s = make_session(10, 5000);
        let cfg = CompactionConfig {
            preserve_recent_messages: 2,
            max_estimated_tokens: 1,
        };
        let r = compact_session(&s, cfg);
        assert!(r.removed_message_count > 0);
        assert_eq!(r.compacted_session.messages[0].role, MessageRole::System);
    }

    #[test]
    fn format_strips_analysis_keeps_summary() {
        let raw = "<analysis>scratch</analysis>\n<summary>The work</summary>";
        let f = format_compact_summary(raw);
        assert!(!f.contains("<analysis>"));
        assert!(f.contains("Summary:"));
        assert!(f.contains("The work"));
    }

    #[test]
    fn strip_tag_block_multiple_occurrences() {
        let input = "<analysis>first</analysis>text<analysis>second</analysis>more";
        let result = super::strip_tag_block(input, "analysis");
        assert!(!result.contains("<analysis>"));
        assert!(result.contains("text"));
        assert!(result.contains("more"));
    }

    #[test]
    fn continuation_message_contains_summary() {
        let msg = get_compact_continuation_message("<summary>Did stuff</summary>", true, false);
        assert!(msg.contains("Did stuff"));
        assert!(msg.contains("Continue the conversation"));
    }

    #[test]
    fn masking_replaces_large_tool_results() {
        let msgs = vec![ConversationMessage {
            importance: None,
            role: MessageRole::Tool,
            blocks: vec![ContentBlock::ToolResult {
                tool_use_id: "t1".to_string(),
                output: "x".repeat(500),
                is_error: false,
            }],
            usage: None,
        }];
        let masked = super::mask_observations(&msgs);
        let ContentBlock::ToolResult { output, .. } = &masked[0].blocks[0] else {
            panic!("expected ToolResult");
        };
        assert!(output.starts_with("[masked:"));
        assert!(output.contains("500 chars"));
    }

    #[test]
    fn masking_preserves_small_tool_results() {
        let msgs = vec![ConversationMessage {
            importance: None,
            role: MessageRole::Tool,
            blocks: vec![ContentBlock::ToolResult {
                tool_use_id: "t1".to_string(),
                output: "small".to_string(),
                is_error: false,
            }],
            usage: None,
        }];
        let masked = super::mask_observations(&msgs);
        let ContentBlock::ToolResult { output, .. } = &masked[0].blocks[0] else {
            panic!("expected ToolResult");
        };
        assert_eq!(output, "small");
    }

    #[test]
    fn masking_preserves_text_blocks() {
        let msgs = vec![ConversationMessage::assistant(
            vec![ContentBlock::Text {
                text: "x".repeat(1000),
            }],
            None,
        )];
        let masked = super::mask_observations(&msgs);
        match &masked[0].blocks[0] {
            ContentBlock::Text { text } => assert_eq!(text.len(), 1000),
            _ => panic!("expected Text"),
        }
    }

    #[test]
    fn compact_uses_masking_before_hard_compaction() {
        // Build a session with large tool results -- masking should reduce size
        let mut s = Session::new("test".to_string());
        for _ in 0..8 {
            s.push(ConversationMessage::user("do something"));
            s.push(ConversationMessage::assistant(
                vec![ContentBlock::ToolUse {
                    id: "t1".to_string(),
                    name: "read_file".to_string(),
                    input: serde_json::json!({"path": "foo.rs"}),
                }],
                None,
            ));
            s.push(ConversationMessage {
                importance: None,
                role: MessageRole::Tool,
                blocks: vec![ContentBlock::ToolResult {
                    tool_use_id: "t1".to_string(),
                    output: "x".repeat(500),
                    is_error: false,
                }],
                usage: None,
            });
        }
        let cfg = CompactionConfig {
            preserve_recent_messages: 2,
            max_estimated_tokens: 100,
        };
        let result = compact_session(&s, cfg);
        // Should have been compacted (hard phase, since masking alone won't
        // bring 22 messages with tool-use blocks under 100 tokens)
        assert!(
            result.removed_message_count > 0
                || result.compacted_session.messages.len() < s.messages.len()
        );
    }

    #[test]
    fn apply_compact_preserves_recent() {
        let s = make_session(6, 100);
        let cfg = CompactionConfig {
            preserve_recent_messages: 2,
            max_estimated_tokens: 1,
        };
        let r = apply_compact_summary(&s, "<summary>summary</summary>", cfg);
        // system msg + 2 recent
        assert_eq!(r.compacted_session.messages.len(), 3);
        assert_eq!(r.removed_message_count, 4);
    }

    #[test]
    fn for_window_scales_threshold() {
        // 200k window → trigger at 100k tokens.
        let cfg = CompactionConfig::for_window(200_000);
        assert_eq!(cfg.max_estimated_tokens, 100_000);
        assert_eq!(cfg.preserve_recent_messages, 12);

        // 1M window → trigger at 500k.
        let cfg = CompactionConfig::for_window(1_000_000);
        assert_eq!(cfg.max_estimated_tokens, 500_000);
    }

    #[test]
    fn small_window_doesnt_starve_recent_messages() {
        // Tiny Ollama window (8k) → trigger at 4k. Even then preserve
        // enough recent messages for continuity.
        let cfg = CompactionConfig::for_window(8_000);
        assert_eq!(cfg.max_estimated_tokens, 4_000);
        assert!(
            cfg.preserve_recent_messages >= 12,
            "recent preserve should not shrink with window"
        );
    }
}
