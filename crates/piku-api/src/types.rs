use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Outbound request types
// ---------------------------------------------------------------------------

/// A single block in the `system` array sent to Anthropic.
/// Using an array (instead of a plain string) enables prompt caching:
/// the last block before the dynamic boundary can carry
/// `cache_control: {type: "ephemeral"}` so Anthropic caches the static
/// prefix across turns, significantly reducing costs for long sessions.
#[derive(Debug, Clone, Serialize)]
pub struct SystemBlock {
    #[serde(rename = "type")]
    pub block_type: String, // always "text"
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<CacheControl>,
}

impl SystemBlock {
    #[must_use]
    pub fn text(text: String) -> Self {
        Self {
            block_type: "text".to_string(),
            text,
            cache_control: None,
        }
    }

    #[must_use]
    pub fn cached(text: String) -> Self {
        Self {
            block_type: "text".to_string(),
            text,
            cache_control: Some(CacheControl {
                cache_type: "ephemeral".to_string(),
            }),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct CacheControl {
    #[serde(rename = "type")]
    pub cache_type: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct MessageRequest {
    pub model: String,
    pub max_tokens: u32,
    pub messages: Vec<RequestMessage>,
    /// System prompt as an array of blocks.  Using an array instead of a plain
    /// string lets us mark the static prefix with `cache_control` for Anthropic
    /// prompt caching.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<Vec<SystemBlock>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ToolDefinition>>,
    pub stream: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct RequestMessage {
    pub role: String,
    pub content: Vec<RequestContent>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RequestContent {
    Text {
        text: String,
    },
    ToolResult {
        tool_use_id: String,
        content: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        is_error: Option<bool>,
    },
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
}

// ---------------------------------------------------------------------------
// Streaming events (provider-agnostic)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub enum Event {
    /// A chunk of assistant text.
    TextDelta { text: String },

    /// A tool call is starting.
    ToolUseStart { id: String, name: String },

    /// A chunk of the tool call's JSON input.
    ToolUseDelta { id: String, partial_json: String },

    /// The tool call's input is complete.
    ToolUseEnd { id: String },

    /// The message is fully done.
    MessageStop { stop_reason: StopReason },

    /// Token usage update (may arrive at end of stream).
    UsageDelta { usage: TokenUsage },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StopReason {
    EndTurn,
    ToolUse,
    MaxTokens,
    StopSequence,
    Other(String),
}

impl StopReason {
    #[must_use]
    pub fn from_wire_str(s: &str) -> Self {
        match s {
            "end_turn" => Self::EndTurn,
            "tool_use" => Self::ToolUse,
            "max_tokens" => Self::MaxTokens,
            "stop_sequence" => Self::StopSequence,
            other => Self::Other(other.to_string()),
        }
    }
}

// ---------------------------------------------------------------------------
// Token usage
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TokenUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    #[serde(default)]
    pub cache_creation_input_tokens: u32,
    #[serde(default)]
    pub cache_read_input_tokens: u32,
}

impl TokenUsage {
    #[must_use]
    pub fn total_tokens(&self) -> u32 {
        self.input_tokens + self.output_tokens
    }

    pub fn accumulate(&mut self, other: &Self) {
        self.input_tokens += other.input_tokens;
        self.output_tokens += other.output_tokens;
        self.cache_creation_input_tokens += other.cache_creation_input_tokens;
        self.cache_read_input_tokens += other.cache_read_input_tokens;
    }
}
