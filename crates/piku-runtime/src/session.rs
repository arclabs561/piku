#![allow(
    clippy::io_other_error,
    clippy::must_use_candidate,
    clippy::needless_pass_by_value
)]

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use piku_api::TokenUsage;

// ---------------------------------------------------------------------------
// Session data model
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Session {
    pub version: u32,
    pub id: String,
    /// Most recently recorded provider and model for this session.
    ///
    /// `--resume` takes its model from the current config rather than from the
    /// file, so without this a resumed session silently continues under a
    /// different model without a warning. These session-level fields are
    /// replaced on resume; they do not provide per-turn attribution or pricing.
    /// Optional and defaulted so older sessions still load.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    pub messages: Vec<ConversationMessage>,
    /// Private histories keyed by configured foreground actor. The active
    /// actor's history remains in `messages` for compatibility with the loop.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub actor_histories: BTreeMap<String, Vec<ConversationMessage>>,
    #[serde(default = "default_actor_id")]
    pub active_actor: String,
}

impl Session {
    pub fn new(id: String) -> Self {
        Self {
            version: 1,
            id,
            provider: None,
            model: None,
            messages: Vec::new(),
            actor_histories: BTreeMap::new(),
            active_actor: default_actor_id(),
        }
    }

    /// Stamp the run's provider and model onto the session.
    ///
    /// Returns the previous pair when it differs, so a resume can say that the
    /// transcript it is about to extend was written by something else.
    pub fn record_provider(&mut self, provider: &str, model: &str) -> Option<(String, String)> {
        let previous = match (&self.provider, &self.model) {
            (Some(old_provider), Some(old_model))
                if old_provider != provider || old_model != model =>
            {
                Some((old_provider.clone(), old_model.clone()))
            }
            _ => None,
        };
        self.provider = Some(provider.to_string());
        self.model = Some(model.to_string());
        previous
    }

    pub fn push(&mut self, msg: ConversationMessage) {
        self.messages.push(msg);
    }

    /// Change foreground actor without merging private model histories.
    pub fn select_actor(&mut self, actor: impl Into<String>) {
        let actor = actor.into();
        if actor == self.active_actor {
            return;
        }
        let previous = std::mem::take(&mut self.messages);
        self.actor_histories
            .insert(self.active_actor.clone(), previous);
        self.messages = self.actor_histories.remove(&actor).unwrap_or_default();
        self.active_actor = actor;
    }

    /// Approximate token count (4 chars/token heuristic).
    pub fn estimated_tokens(&self) -> usize {
        self.messages
            .iter()
            .map(ConversationMessage::estimated_tokens)
            .sum()
    }

    /// Save atomically: write to a sibling tmp file, then rename over the
    /// destination. `std::fs::write` is `O_TRUNC` + write, so a crash between
    /// the two syscalls leaves a zero-byte session file. `rename(2)` is
    /// atomic on the same filesystem — readers either see the old file or
    /// the new one, never a truncated partial.
    pub fn save(&self, path: &std::path::Path) -> std::io::Result<()> {
        use std::io::Write;
        use std::sync::atomic::{AtomicU64, Ordering};
        static NEXT_STAGE: AtomicU64 = AtomicU64::new(0);

        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

        // Exclusive creation protects concurrent saves and retained staging
        // files from earlier processes that happened to use the same PID.
        let name = path.file_name().ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "session path has no filename",
            )
        })?;
        let (tmp_path, mut staged) = loop {
            let sequence = NEXT_STAGE.fetch_add(1, Ordering::Relaxed);
            let mut tmp_name = name.to_os_string();
            tmp_name.push(format!(".tmp.{}.{sequence}", std::process::id()));
            let tmp_path = path.with_file_name(tmp_name);
            match std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&tmp_path)
            {
                Ok(file) => break (tmp_path, file),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
                Err(error) => return Err(error),
            }
        };
        staged.write_all(json.as_bytes())?;
        drop(staged);
        // Same-filesystem rename publishes one complete value. Failed writes
        // or renames retain their staging file; there is no crash/fsync guarantee.
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

fn default_actor_id() -> String {
    "primary".to_string()
}

#[cfg(test)]
mod actor_tests {
    use super::*;

    #[test]
    fn actor_switches_keep_private_histories_separate() {
        let mut session = Session::new("room-1".to_string());
        session.push(ConversationMessage::user("private researcher note"));
        session.select_actor("critic");
        assert!(session.messages.is_empty());
        session.push(ConversationMessage::user("private critic note"));
        session.select_actor("primary");
        assert_eq!(session.messages.len(), 1);
        session.select_actor("critic");
        assert_eq!(session.messages.len(), 1);
        assert!(matches!(
            &session.messages[0].blocks[0],
            ContentBlock::Text { text } if text == "private critic note"
        ));
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

    /// Estimated token cost of this message (4 chars/token heuristic,
    /// plus 1 for role framing).
    #[must_use]
    pub fn estimated_tokens(&self) -> usize {
        self.blocks
            .iter()
            .map(|b| b.text_len() / 4 + 1)
            .sum::<usize>()
            + 1
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
    fn same_process_writers_publish_only_complete_sessions() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("session.json");
        Session::new("initial".into()).save(&path).unwrap();
        let barrier = std::sync::Barrier::new(2);
        std::thread::scope(|scope| {
            let handles: Vec<_> = ["a", "b"].into_iter().map(|id| {
                let path = &path;
                let barrier = &barrier;
                scope.spawn(move || {
                    let mut session = Session::new(id.into());
                    session.push(ConversationMessage::user(id.repeat(256 * 1024)));
                    let mut errors = Vec::new();
                    for _ in 0..32 {
                        barrier.wait();
                        if let Err(error) = session.save(path) { errors.push(error.to_string()); }
                        match Session::load(path) {
                            Ok(value) => {
                                let expected = value.id.repeat(256 * 1024);
                                if !matches!(value.messages.first().and_then(|message| message.blocks.first()), Some(ContentBlock::Text { text }) if *text == expected) {
                                    errors.push("mixed session payload".into());
                                }
                            }
                            Err(error) => errors.push(error.to_string()),
                        }
                    }
                    errors
                })
            }).collect();
            let errors: Vec<_> = handles
                .into_iter()
                .flat_map(|handle| handle.join().unwrap())
                .collect();
            assert!(errors.is_empty(), "concurrent saves failed: {errors:?}");
        });
    }

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
    fn provenance_round_trips_and_reports_a_model_change() {
        let mut session = Session::new("s1".to_string());
        assert_eq!(session.record_provider("openrouter", "model-a"), None);
        assert_eq!(session.provider.as_deref(), Some("openrouter"));

        // Same pair on a later run is not a change.
        assert_eq!(session.record_provider("openrouter", "model-a"), None);

        // A different model is what a resume needs to be told about, since it
        // takes the model from config rather than from the file.
        assert_eq!(
            session.record_provider("anthropic", "model-b"),
            Some(("openrouter".to_string(), "model-a".to_string()))
        );
        assert_eq!(session.model.as_deref(), Some("model-b"));
    }

    #[test]
    fn a_session_written_before_provenance_existed_still_loads() {
        let directory = tempfile::tempdir().expect("tempdir");
        let path = directory.path().join("old.json");
        std::fs::write(&path, r#"{"version":1,"id":"old","messages":[]}"#).expect("write");

        let session = Session::load(&path).expect("load");
        assert_eq!(session.id, "old");
        assert_eq!(session.provider, None);
        assert_eq!(session.model, None);
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
