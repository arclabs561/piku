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

    /// Save atomically: write to a sibling tmp file, then rename over the
    /// destination. `std::fs::write` is O_TRUNC + write, so a crash between
    /// the two syscalls leaves a zero-byte session file. `rename(2)` is
    /// atomic on the same filesystem — readers either see the old file or
    /// the new one, never a truncated partial.
    pub fn save(&self, path: &std::path::Path) -> std::io::Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

        // Tmp file sibling to the destination so rename is same-filesystem.
        // Include pid to avoid collisions between concurrent saves.
        let tmp_path = match path.file_name() {
            Some(name) => {
                let tmp_name = format!("{}.tmp.{}", name.to_string_lossy(), std::process::id());
                path.with_file_name(tmp_name)
            }
            None => path.with_extension("tmp"),
        };
        std::fs::write(&tmp_path, &json)?;
        // Rename is atomic. If this returns error the tmp file is left
        // on disk; it'll be overwritten by the next save or cleaned on
        // next startup scan.
        std::fs::rename(&tmp_path, path)
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

#[cfg(test)]
mod save_atomicity_tests {
    use super::*;

    #[test]
    fn save_does_not_leave_tmp_files_in_place_on_success() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let path = tmp.path().join("session.json");
        let session = Session::default();
        session.save(&path).expect("save");

        let leftover: Vec<_> = std::fs::read_dir(tmp.path())
            .unwrap()
            .flatten()
            .filter(|e| e.file_name().to_string_lossy().contains(".tmp."))
            .collect();
        assert!(
            leftover.is_empty(),
            "tmp files left after save: {leftover:?}"
        );
    }

    #[test]
    fn save_replaces_existing_file_atomically() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let path = tmp.path().join("session.json");

        // Pre-existing content simulates a previous successful save.
        std::fs::write(&path, b"{\"old\":true}").unwrap();

        // The atomic save should replace the file with valid new content.
        let session = Session::default();
        session.save(&path).expect("save");

        let content = std::fs::read_to_string(&path).unwrap();
        // New content must parse as a Session and not contain the old payload.
        assert!(!content.contains("\"old\""), "old content survived save");
        let _: Session = serde_json::from_str(&content).expect("reparse");
    }

    #[test]
    fn readers_never_see_partial_file_during_concurrent_saves() {
        // The atomic rename guarantees the destination file is always either
        // old content or new content, never empty / partial. This test races
        // a reader against many saves and checks the invariant holds.
        let tmp = tempfile::tempdir().expect("tempdir");
        let path = tmp.path().join("session.json");
        let session = Session::default();
        session.save(&path).unwrap();

        let reader = std::thread::spawn({
            let path = path.clone();
            move || {
                let deadline = std::time::Instant::now() + std::time::Duration::from_millis(50);
                while std::time::Instant::now() < deadline {
                    if let Ok(b) = std::fs::read(&path) {
                        assert!(!b.is_empty(), "zero-byte session observed");
                        assert!(
                            serde_json::from_slice::<Session>(&b).is_ok(),
                            "session did not parse: {:?}",
                            String::from_utf8_lossy(&b)
                        );
                    }
                }
            }
        });
        for _ in 0..50 {
            session.save(&path).unwrap();
        }
        reader.join().unwrap();
    }
}
