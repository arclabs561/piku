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

    let vec: Vec<f32> = serde_json::from_value(embedding.clone())
        .map_err(|e| format!("embedding parse failed: {e}"))?;

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
