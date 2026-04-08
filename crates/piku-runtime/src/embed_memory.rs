/// Embedding-based memory — semantic retrieval over persistent memories.
///
/// Uses ollama's `/api/embed` endpoint (nomic-embed-text, 768d) for embeddings.
/// Storage is a JSON file of `MemoryEntry` structs with pre-normalized vectors.
/// Retrieval is brute-force dot product (< 1ms at 1k entries, < 5ms at 10k).
///
/// Key design decisions (from research):
/// - Confidence-gated retrieval: no results below threshold (CTIM-Rover finding)
/// - Write-time investment: tag and validate at insert, not retrieval
/// - Atomic notes: one fact per entry, not paragraphs (A-MEM pattern)
/// - Mark invalid rather than delete (preserves audit trail)

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// A single memory entry with its embedding vector.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEntry {
    /// Unique ID (monotonic counter).
    pub id: u64,
    /// The memory content (one atomic fact).
    pub content: String,
    /// LLM-generated keyword tags for hybrid retrieval.
    pub tags: Vec<String>,
    /// When this memory was created (Unix timestamp).
    pub created_at: u64,
    /// Last time this memory was retrieved (for access-frequency eviction).
    pub last_accessed: u64,
    /// How many times this memory has been retrieved.
    pub access_count: u32,
    /// Whether this memory is still valid (false = superseded/contradicted).
    pub is_valid: bool,
    /// ID of the memory this one supersedes (if any).
    pub supersedes: Option<u64>,
    /// Pre-normalized embedding vector (768d for nomic-embed-text).
    pub embedding: Vec<f32>,
}

/// A retrieved memory with its similarity score.
#[derive(Debug, Clone)]
pub struct RetrievedMemory {
    pub entry: MemoryEntry,
    pub similarity: f32,
}

/// The in-memory store (loaded from / saved to JSON).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MemoryStore {
    pub next_id: u64,
    pub entries: Vec<MemoryEntry>,
}

/// Minimum similarity threshold for retrieval.
/// Below this, no memory is returned (confidence-gated retrieval).
const MIN_SIMILARITY: f32 = 0.3;

// ---------------------------------------------------------------------------
// Embedding client (ollama /api/embed)
// ---------------------------------------------------------------------------

/// Embed text using ollama's /api/embed endpoint.
/// Returns a normalized 768-dim vector.
pub async fn embed_text(
    text: &str,
    ollama_url: &str,
    model: &str,
) -> Result<Vec<f32>, String> {
    let client = reqwest::Client::new();
    let body = serde_json::json!({
        "model": model,
        "input": text,
    });

    let resp = client
        .post(format!("{ollama_url}/api/embed"))
        .json(&body)
        .send()
        .await
        .map_err(|e| format!("embed request failed: {e}"))?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        return Err(format!("embed failed ({status}): {body}"));
    }

    let json: serde_json::Value = resp
        .json()
        .await
        .map_err(|e| format!("embed parse failed: {e}"))?;

    let embedding = json
        .get("embeddings")
        .and_then(|e| e.get(0))
        .ok_or_else(|| "no embeddings in response".to_string())?;

    let mut vec: Vec<f32> = serde_json::from_value(embedding.clone())
        .map_err(|e| format!("embedding parse failed: {e}"))?;

    // Normalize to unit length — ollama usually returns normalized vectors,
    // but normalize defensively so dot product = cosine similarity.
    normalize(&mut vec);

    Ok(vec)
}

// ---------------------------------------------------------------------------
// Vector math
// ---------------------------------------------------------------------------

/// Dot product of two pre-normalized vectors (= cosine similarity).
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// L2-normalize a vector in place.
fn normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        v.iter_mut().for_each(|x| *x /= norm);
    }
}

// ---------------------------------------------------------------------------
// Store operations
// ---------------------------------------------------------------------------

impl MemoryStore {
    /// Load from a JSON file. Returns empty store if file doesn't exist.
    pub fn load(path: &Path) -> Self {
        std::fs::read_to_string(path)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default()
    }

    /// Save to a JSON file.
    pub fn save(&self, path: &Path) -> Result<(), String> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("create_dir_all failed: {e}"))?;
        }
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| format!("serialize failed: {e}"))?;
        std::fs::write(path, json).map_err(|e| format!("write failed: {e}"))
    }

    /// Insert a new memory entry with a pre-computed embedding.
    pub fn insert(&mut self, content: String, tags: Vec<String>, mut embedding: Vec<f32>) -> u64 {
        normalize(&mut embedding);
        let id = self.next_id;
        self.next_id += 1;
        let now = now_unix();
        self.entries.push(MemoryEntry {
            id,
            content,
            tags,
            created_at: now,
            last_accessed: now,
            access_count: 0,
            is_valid: true,
            supersedes: None,
            embedding,
        });
        id
    }

    /// Insert a memory that supersedes an older one (contradiction resolution).
    pub fn insert_superseding(
        &mut self,
        content: String,
        tags: Vec<String>,
        embedding: Vec<f32>,
        supersedes_id: u64,
    ) -> u64 {
        // Mark the old memory as invalid
        if let Some(old) = self.entries.iter_mut().find(|e| e.id == supersedes_id) {
            old.is_valid = false;
        }
        let id = self.insert(content, tags, embedding);
        if let Some(new) = self.entries.iter_mut().find(|e| e.id == id) {
            new.supersedes = Some(supersedes_id);
        }
        id
    }

    /// Retrieve the top-k most similar valid memories above the confidence threshold.
    /// Updates access metadata on retrieved entries.
    pub fn retrieve(&mut self, query_embedding: &[f32], k: usize) -> Vec<RetrievedMemory> {
        let mut scored: Vec<(f32, usize)> = self
            .entries
            .iter()
            .enumerate()
            .filter(|(_, e)| e.is_valid)
            .map(|(i, e)| (dot(query_embedding, &e.embedding), i))
            .filter(|(sim, _)| *sim >= MIN_SIMILARITY)
            .collect();

        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        let now = now_unix();
        scored
            .into_iter()
            .take(k)
            .map(|(sim, idx)| {
                self.entries[idx].last_accessed = now;
                self.entries[idx].access_count += 1;
                RetrievedMemory {
                    entry: self.entries[idx].clone(),
                    similarity: sim,
                }
            })
            .collect()
    }

    /// Read-only retrieval (doesn't update access metadata).
    #[must_use]
    pub fn search(&self, query_embedding: &[f32], k: usize) -> Vec<RetrievedMemory> {
        let mut scored: Vec<(f32, &MemoryEntry)> = self
            .entries
            .iter()
            .filter(|e| e.is_valid)
            .map(|e| (dot(query_embedding, &e.embedding), e))
            .filter(|(sim, _)| *sim >= MIN_SIMILARITY)
            .collect();

        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        scored
            .into_iter()
            .take(k)
            .map(|(sim, entry)| RetrievedMemory {
                entry: entry.clone(),
                similarity: sim,
            })
            .collect()
    }

    /// Count of valid entries.
    #[must_use]
    pub fn valid_count(&self) -> usize {
        self.entries.iter().filter(|e| e.is_valid).count()
    }

    /// Find the top-k most similar existing memories to a candidate
    /// (for contradiction detection at write time).
    #[must_use]
    pub fn find_similar(&self, embedding: &[f32], k: usize) -> Vec<RetrievedMemory> {
        self.search(embedding, k)
    }
}

/// Default store path for a project.
#[must_use]
pub fn default_store_path(cwd: &Path) -> PathBuf {
    cwd.join(".piku").join("embed-memory").join("memories.json")
}

fn now_unix() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

// ---------------------------------------------------------------------------
// Format retrieved memories for prompt injection
// ---------------------------------------------------------------------------

/// Format retrieved memories for injection into the system prompt.
/// Transparent: shows similarity scores so the operator can inspect relevance.
#[must_use]
pub fn format_retrieved_memories(memories: &[RetrievedMemory]) -> String {
    if memories.is_empty() {
        return String::new();
    }
    let mut out = String::from("\n\n# Retrieved Memories\n\n");
    out.push_str("The following memories were retrieved by semantic similarity to the current task.\n\n");
    for m in memories {
        out.push_str(&format!(
            "- [sim={:.2}] {}\n",
            m.similarity, m.entry.content
        ));
        if !m.entry.tags.is_empty() {
            out.push_str(&format!("  tags: {}\n", m.entry.tags.join(", ")));
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Session-end memory extraction
// ---------------------------------------------------------------------------

/// Prompt for extracting atomic facts from a conversation.
const EXTRACTION_PROMPT: &str = "\
You are a memory extraction system. Given a conversation between a user and an AI assistant, \
extract 0-5 atomic facts worth remembering for future sessions.

Rules:
- Each fact must be self-contained (no pronouns, no \"the project\")
- One fact per line, prefixed with `- `
- Include: user preferences, project conventions, error patterns, key decisions
- Exclude: transient task details, file contents, code snippets, things derivable from git
- If nothing is worth remembering, output exactly: NONE

Output format:
- fact one
- fact two
TAGS: comma, separated, keywords

Or:
NONE";

/// Extract memories from a conversation transcript.
/// Returns a list of (fact, tags) pairs.
pub async fn extract_memories(
    conversation: &str,
    provider: &dyn piku_api::Provider,
    model: &str,
) -> Vec<(String, Vec<String>)> {
    use futures_util::StreamExt;

    let request = piku_api::MessageRequest {
        model: model.to_string(),
        max_tokens: 1024,
        messages: vec![piku_api::RequestMessage {
            role: "user".to_string(),
            content: vec![piku_api::RequestContent::Text {
                text: conversation.to_string(),
            }],
        }],
        system: Some(vec![piku_api::SystemBlock::text(
            EXTRACTION_PROMPT.to_string(),
        )]),
        tools: None,
        stream: true,
    };

    let mut stream = provider.stream_message(request);
    let mut response = String::new();

    let timeout = std::time::Duration::from_secs(30);
    let result = tokio::time::timeout(timeout, async {
        while let Some(event) = stream.next().await {
            if let Ok(piku_api::Event::TextDelta { text }) = event {
                response.push_str(&text);
            }
        }
    })
    .await;

    if result.is_err() || response.trim() == "NONE" || response.trim().is_empty() {
        return Vec::new();
    }

    parse_extraction_response(&response)
}

/// Parse the LLM's extraction response into (fact, tags) pairs.
fn parse_extraction_response(response: &str) -> Vec<(String, Vec<String>)> {
    let mut facts = Vec::new();
    let mut current_tags: Vec<String> = Vec::new();

    // Extract TAGS line if present
    for line in response.lines().rev() {
        let trimmed = line.trim();
        if let Some(tags_str) = trimmed.strip_prefix("TAGS:") {
            current_tags = tags_str
                .split(',')
                .map(|t| t.trim().to_lowercase())
                .filter(|t| !t.is_empty())
                .collect();
            break;
        }
    }

    // Extract facts (lines starting with "- ")
    for line in response.lines() {
        let trimmed = line.trim();
        if let Some(fact) = trimmed.strip_prefix("- ") {
            let fact = fact.trim();
            if !fact.is_empty() && !fact.starts_with("TAGS:") {
                facts.push((fact.to_string(), current_tags.clone()));
            }
        }
    }

    facts
}

/// Full session-end memory pipeline:
/// 1. Extract atomic facts from the conversation
/// 2. Embed each fact
/// 3. Check for contradictions against existing memories
/// 4. Insert or supersede
///
/// Returns the number of memories added/updated.
pub async fn extract_and_store(
    conversation: &str,
    provider: &dyn piku_api::Provider,
    model: &str,
    store: &mut MemoryStore,
    ollama_url: &str,
    embed_model: &str,
) -> usize {
    let facts = extract_memories(conversation, provider, model).await;
    if facts.is_empty() {
        return 0;
    }

    let mut count = 0;
    for (fact, tags) in facts {
        // Embed the new fact
        let embedding = match embed_text(&fact, ollama_url, embed_model).await {
            Ok(v) => v,
            Err(_) => continue,
        };

        // Check for contradictions: find top-3 most similar existing memories
        let similar = store.find_similar(&embedding, 3);

        // If the most similar memory is very close (>0.85), it might be a
        // duplicate or contradiction. For now, skip near-duplicates.
        // A future version can use an LLM call to decide add/update/ignore.
        if similar.first().is_some_and(|s| s.similarity > 0.85) {
            // Near-duplicate — skip
            continue;
        }

        store.insert(fact, tags, embedding);
        count += 1;
    }

    count
}

// ---------------------------------------------------------------------------
// MemoryStoreView trait impl (for manage_memory tool)
// ---------------------------------------------------------------------------

impl piku_tools::embed_memory_tool::piku_runtime_types::MemoryStoreView for MemoryStore {
    fn total_count(&self) -> usize {
        self.entries.len()
    }

    fn valid_count(&self) -> usize {
        self.entries.iter().filter(|e| e.is_valid).count()
    }

    fn list_recent(&self, max: usize) -> Vec<(u64, String, bool, u32)> {
        let mut sorted: Vec<&MemoryEntry> = self.entries.iter().collect();
        sorted.sort_by(|a, b| b.created_at.cmp(&a.created_at));
        sorted
            .into_iter()
            .take(max)
            .map(|e| (e.id, e.content.clone(), e.is_valid, e.access_count))
            .collect()
    }

    fn inspect(&self, id: u64) -> Option<String> {
        self.entries.iter().find(|e| e.id == id).map(|e| {
            format!(
                "ID: {}\nContent: {}\nTags: {}\nCreated: {}\nLast accessed: {}\nAccess count: {}\nValid: {}\nSupersedes: {:?}\nEmbedding dims: {}",
                e.id,
                e.content,
                e.tags.join(", "),
                e.created_at,
                e.last_accessed,
                e.access_count,
                e.is_valid,
                e.supersedes,
                e.embedding.len()
            )
        })
    }

    fn invalidate(&mut self, id: u64) -> bool {
        if let Some(e) = self.entries.iter_mut().find(|e| e.id == id) {
            e.is_valid = false;
            true
        } else {
            false
        }
    }

    fn query_by_tag(&self, tag: &str, max: usize) -> Vec<(u64, String)> {
        let tag_lower = tag.to_lowercase();
        self.entries
            .iter()
            .filter(|e| e.is_valid && e.tags.iter().any(|t| t.to_lowercase().contains(&tag_lower)))
            .take(max)
            .map(|e| (e.id, e.content.clone()))
            .collect()
    }
}

/// Build a compact conversation transcript for memory extraction.
/// Includes only user and assistant text blocks, truncated to keep
/// the extraction prompt manageable.
pub fn build_extraction_transcript(
    messages: &[crate::session::ConversationMessage],
) -> String {
    let mut transcript = String::new();
    for msg in messages {
        let role = match msg.role {
            crate::session::MessageRole::User => "User",
            crate::session::MessageRole::Assistant => "Assistant",
            _ => continue,
        };
        for block in &msg.blocks {
            if let crate::session::ContentBlock::Text { text } = block {
                if !text.trim().is_empty() {
                    transcript.push_str(role);
                    transcript.push_str(": ");
                    if text.len() > 500 {
                        let trunc: String = text.chars().take(500).collect();
                        transcript.push_str(&trunc);
                        transcript.push_str("...");
                    } else {
                        transcript.push_str(text);
                    }
                    transcript.push('\n');
                }
            }
        }
    }
    transcript
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_embedding(seed: f32) -> Vec<f32> {
        let mut v: Vec<f32> = (0..768).map(|i| ((i as f32 + seed) * 0.01).sin()).collect();
        normalize(&mut v);
        v
    }

    #[test]
    fn insert_and_retrieve() {
        let mut store = MemoryStore::default();
        let e1 = make_embedding(1.0);
        let e2 = make_embedding(2.0);
        store.insert("fact about rust".to_string(), vec!["rust".to_string()], e1.clone());
        store.insert("fact about python".to_string(), vec!["python".to_string()], e2);

        let results = store.retrieve(&e1, 5);
        assert!(!results.is_empty());
        assert_eq!(results[0].entry.content, "fact about rust");
        assert!(results[0].similarity > 0.9); // should be near-identical
    }

    #[test]
    fn confidence_gating() {
        let mut store = MemoryStore::default();
        let e1 = make_embedding(1.0);
        // Insert with very different embedding
        let mut orthogonal = vec![0.0f32; 768];
        orthogonal[0] = 1.0;
        normalize(&mut orthogonal);
        store.insert("unrelated fact".to_string(), vec![], orthogonal);

        let results = store.retrieve(&e1, 5);
        // Should be empty if similarity is below threshold
        // (depends on actual vectors, but orthogonal should be near 0)
        for r in &results {
            assert!(r.similarity >= MIN_SIMILARITY);
        }
    }

    #[test]
    fn superseding_marks_old_invalid() {
        let mut store = MemoryStore::default();
        let e1 = make_embedding(1.0);
        let id1 = store.insert("old fact".to_string(), vec![], e1.clone());
        store.insert_superseding("new fact".to_string(), vec![], e1.clone(), id1);

        assert!(!store.entries.iter().find(|e| e.id == id1).unwrap().is_valid);
        assert_eq!(store.valid_count(), 1);

        let results = store.retrieve(&e1, 5);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].entry.content, "new fact");
    }

    #[test]
    fn save_and_load_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test-memories.json");

        let mut store = MemoryStore::default();
        let e1 = make_embedding(1.0);
        store.insert("test fact".to_string(), vec!["test".to_string()], e1);
        store.save(&path).unwrap();

        let loaded = MemoryStore::load(&path);
        assert_eq!(loaded.entries.len(), 1);
        assert_eq!(loaded.entries[0].content, "test fact");
        assert_eq!(loaded.next_id, 1);
    }

    #[test]
    fn access_tracking() {
        let mut store = MemoryStore::default();
        let e1 = make_embedding(1.0);
        store.insert("tracked fact".to_string(), vec![], e1.clone());

        assert_eq!(store.entries[0].access_count, 0);
        store.retrieve(&e1, 5);
        assert_eq!(store.entries[0].access_count, 1);
        store.retrieve(&e1, 5);
        assert_eq!(store.entries[0].access_count, 2);
    }

    #[test]
    fn format_memories_shows_scores() {
        let entry = MemoryEntry {
            id: 0,
            content: "rust is good".to_string(),
            tags: vec!["rust".to_string()],
            created_at: 0,
            last_accessed: 0,
            access_count: 0,
            is_valid: true,
            supersedes: None,
            embedding: vec![],
        };
        let retrieved = vec![RetrievedMemory {
            entry,
            similarity: 0.85,
        }];
        let formatted = format_retrieved_memories(&retrieved);
        assert!(formatted.contains("[sim=0.85]"));
        assert!(formatted.contains("rust is good"));
        assert!(formatted.contains("tags: rust"));
    }

    #[test]
    fn parse_extraction_basic() {
        let response = "- The user prefers snake_case in Rust code\n\
                        - The project uses tokio for async\n\
                        TAGS: rust, style, async";
        let facts = super::parse_extraction_response(response);
        assert_eq!(facts.len(), 2);
        assert_eq!(facts[0].0, "The user prefers snake_case in Rust code");
        assert_eq!(facts[0].1, vec!["rust", "style", "async"]);
    }

    #[test]
    fn parse_extraction_none() {
        let response = "NONE";
        let facts = super::parse_extraction_response(response);
        assert!(facts.is_empty());
    }

    #[test]
    fn parse_extraction_no_tags() {
        let response = "- Single fact without tags";
        let facts = super::parse_extraction_response(response);
        assert_eq!(facts.len(), 1);
        assert!(facts[0].1.is_empty());
    }

    #[test]
    fn build_transcript_filters_roles() {
        use crate::session::{ConversationMessage, ContentBlock};
        let messages = vec![
            ConversationMessage::user("hello"),
            ConversationMessage::assistant(
                vec![ContentBlock::Text { text: "hi there".to_string() }],
                None,
            ),
            ConversationMessage {
                role: crate::session::MessageRole::Tool,
                blocks: vec![ContentBlock::ToolResult {
                    tool_use_id: "t1".to_string(),
                    output: "tool output".to_string(),
                    is_error: false,
                }],
                usage: None,
            },
        ];
        let transcript = super::build_extraction_transcript(&messages);
        assert!(transcript.contains("User: hello"));
        assert!(transcript.contains("Assistant: hi there"));
        assert!(!transcript.contains("tool output")); // tool results excluded
    }

    #[test]
    fn normalize_works() {
        let mut v = vec![3.0, 4.0];
        normalize(&mut v);
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn dot_product_of_normalized() {
        let mut a = vec![1.0, 0.0, 0.0];
        let mut b = vec![1.0, 0.0, 0.0];
        normalize(&mut a);
        normalize(&mut b);
        assert!((dot(&a, &b) - 1.0).abs() < 1e-6);
    }
}
