#![allow(
    clippy::io_other_error,
    clippy::must_use_candidate,
    clippy::needless_pass_by_value
)]

use serde::{Deserialize, Serialize};

use piku_api::TokenUsage;

// ---------------------------------------------------------------------------
// Session data model
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Session {
    pub version: u32,
    pub id: String,
    pub messages: Vec<ConversationMessage>,
}

impl Session {
    pub fn new(id: String) -> Self {
        Self {
            version: 1,
            id,
            messages: Vec::new(),
        }
    }

    pub fn push(&mut self, msg: ConversationMessage) {
        self.messages.push(msg);
    }

    /// Approximate token count (4 chars/token heuristic).
    pub fn estimated_tokens(&self) -> usize {
        self.messages
            .iter()
            .flat_map(|m| &m.blocks)
            .map(|b| b.text_len() / 4 + 1)
            .sum()
    }

    pub fn save(&self, path: &std::path::Path) -> std::io::Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        std::fs::write(path, json)
    }

    pub fn load(path: &std::path::Path) -> std::io::Result<Self> {
        let json = std::fs::read_to_string(path)?;
        serde_json::from_str(&json)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }

    /// Score each message's relevance to the most recent user message.
    /// Uses keyword overlap (cheap, no LLM). Recent messages get a recency boost.
    /// User messages always score 1.0 (they are the task definition).
    #[allow(clippy::cast_precision_loss)] // scoring heuristic -- f32 precision is fine
    pub fn score_messages(&mut self) {
        // Extract keywords from the last user message.
        let query_words: std::collections::HashSet<String> = self
            .messages
            .iter()
            .rev()
            .find(|m| m.role == MessageRole::User)
            .map(extract_keywords)
            .unwrap_or_default();

        if query_words.is_empty() {
            return;
        }

        let len = self.messages.len();
        for (idx, msg) in self.messages.iter_mut().enumerate() {
            if msg.role == MessageRole::User {
                msg.importance = Some(1.0);
                continue;
            }

            let msg_words = extract_keywords(msg);
            let overlap = query_words.intersection(&msg_words).count();
            let denom = query_words.len().min(msg_words.len()).max(1) as f32;
            let keyword_score = overlap as f32 / denom;

            let recency = if len > 1 {
                0.5 * (idx as f32) / ((len - 1) as f32)
            } else {
                0.5
            };

            msg.importance = Some((keyword_score * 0.6 + recency).min(1.0));
        }
    }
}

/// Extract lowercase keywords from a message (words > 3 chars, no stop words).
fn extract_keywords(msg: &ConversationMessage) -> std::collections::HashSet<String> {
    let mut words = std::collections::HashSet::new();
    for block in &msg.blocks {
        let text = match block {
            ContentBlock::Text { text } => text.as_str(),
            ContentBlock::ToolUse { name, .. } => name.as_str(),
            ContentBlock::ToolResult { output, .. } => output.as_str(),
        };
        for word in text.split(|c: char| !c.is_alphanumeric() && c != '_') {
            let lower = word.to_lowercase();
            if lower.len() > 3 {
                words.insert(lower);
            }
        }
    }
    words
}

// ---------------------------------------------------------------------------
// Message types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationMessage {
    pub role: MessageRole,
    pub blocks: Vec<ContentBlock>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<TokenUsage>,
    /// Relevance score for context curation (0.0-1.0, None = unscored).
    /// Set by `score_messages` after each turn. Higher = more relevant to keep.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub importance: Option<f32>,
}

impl ConversationMessage {
    pub fn user(text: impl Into<String>) -> Self {
        Self {
            role: MessageRole::User,
            blocks: vec![ContentBlock::Text { text: text.into() }],
            usage: None,
            importance: None,
        }
    }

    pub fn assistant(blocks: Vec<ContentBlock>, usage: Option<TokenUsage>) -> Self {
        Self {
            role: MessageRole::Assistant,
            blocks,
            usage,
            importance: None,
        }
    }

    pub fn tool_result(tool_use_id: String, output: String, is_error: bool) -> Self {
        Self {
            role: MessageRole::Tool,
            blocks: vec![ContentBlock::ToolResult {
                tool_use_id,
                output,
                is_error,
            }],
            usage: None,
            importance: None,
        }
    }

    pub fn system(text: impl Into<String>) -> Self {
        Self {
            role: MessageRole::System,
            blocks: vec![ContentBlock::Text { text: text.into() }],
            usage: None,
            importance: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum MessageRole {
    User,
    Assistant,
    Tool,
    System,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    Text {
        text: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    ToolResult {
        tool_use_id: String,
        output: String,
        is_error: bool,
    },
}

impl ContentBlock {
    pub fn text_len(&self) -> usize {
        match self {
            Self::Text { text } => text.len(),
            Self::ToolUse { input, .. } => input.to_string().len(),
            Self::ToolResult { output, .. } => output.len(),
        }
    }
}

// ---------------------------------------------------------------------------
// Usage tracking
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default)]
pub struct UsageTracker {
    pub cumulative: TokenUsage,
    pub current_turn: TokenUsage,
    pub turns: u32,
}

impl UsageTracker {
    pub fn record(&mut self, usage: TokenUsage) {
        self.current_turn.accumulate(&usage);
        self.cumulative.accumulate(&usage);
    }

    pub fn finish_turn(&mut self) {
        self.turns += 1;
        self.current_turn = TokenUsage::default();
    }

    pub fn from_session(session: &Session) -> Self {
        let mut tracker = Self::default();
        for msg in &session.messages {
            if let Some(usage) = &msg.usage {
                tracker.cumulative.accumulate(usage);
                tracker.turns += 1;
            }
        }
        tracker
    }
}
