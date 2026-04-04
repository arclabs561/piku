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
}

impl ConversationMessage {
    pub fn user(text: impl Into<String>) -> Self {
        Self {
            role: MessageRole::User,
            blocks: vec![ContentBlock::Text { text: text.into() }],
            usage: None,
        }
    }

    pub fn assistant(blocks: Vec<ContentBlock>, usage: Option<TokenUsage>) -> Self {
        Self {
            role: MessageRole::Assistant,
            blocks,
            usage,
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
        }
    }

    pub fn system(text: impl Into<String>) -> Self {
        Self {
            role: MessageRole::System,
            blocks: vec![ContentBlock::Text { text: text.into() }],
            usage: None,
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
