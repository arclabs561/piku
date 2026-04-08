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
use std::fmt::Write;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Whether this entry is a static fact or a recorded attempt.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EntryType {
    /// An atomic fact (the original memory type).
    #[default]
    Fact,
    /// A recorded attempt: an approach tried toward a goal, with outcome tracking.
    Attempt,
}

/// Outcome of an attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Outcome {
    /// Attempt is still in progress (no result yet).
    Pending,
    /// Approach succeeded.
    Success,
    /// Approach failed.
    Failure,
}

/// A single memory entry with its embedding vector.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEntry {
    /// Unique ID (monotonic counter).
    pub id: u64,
    /// The memory content (one atomic fact or attempt description).
    pub content: String,
    /// LLM-generated keyword tags for hybrid retrieval.
    pub tags: Vec<String>,
    /// LLM-rated importance (1-10). Memories below `MIN_IMPORTANCE` are not stored.
    /// Higher = more likely to be retained long-term and surface in retrieval.
    #[serde(default = "default_importance")]
    pub importance: u8,
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

    // --- Tree / attempt fields (v2) ---
    /// Entry type: Fact (default, backward-compat) or Attempt.
    #[serde(default)]
    pub entry_type: EntryType,
    /// Parent attempt ID (forms the tree). Root attempts have `None`.
    #[serde(default)]
    pub parent: Option<u64>,
    /// Outcome of this attempt (`None` for Facts).
    #[serde(default)]
    pub outcome: Option<Outcome>,
    /// Why this attempt succeeded or failed.
    #[serde(default)]
    pub outcome_detail: Option<String>,
    /// The goal this attempt is working toward (for tree roots and retrieval).
    #[serde(default)]
    pub goal: Option<String>,
}

fn default_importance() -> u8 {
    5
}

/// A retrieved memory with its composite score.
#[derive(Debug, Clone)]
pub struct RetrievedMemory {
    pub entry: MemoryEntry,
    /// Cosine similarity component.
    pub similarity: f32,
    /// Composite score: relevance + recency + importance + access frequency.
    pub score: f32,
}

/// The in-memory store (loaded from / saved to JSON).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MemoryStore {
    pub next_id: u64,
    pub entries: Vec<MemoryEntry>,
}

/// Minimum cosine similarity for retrieval (confidence gate).
const MIN_SIMILARITY: f32 = 0.3;

/// Minimum importance score for admission (1-10 scale).
/// Facts below this are not stored. Prevents error propagation.
pub const MIN_IMPORTANCE: u8 = 3;

/// Scoring weights (Park et al. formula + access frequency).
/// score = `w_rel` * relevance + `w_rec` * recency + `w_imp` * importance + `w_acc` * access
const W_RELEVANCE: f32 = 0.4;
const W_RECENCY: f32 = 0.3;
const W_IMPORTANCE: f32 = 0.2;
const W_ACCESS: f32 = 0.1;

/// Recency half-life in seconds (7 days). After this, recency score = 0.5.
const RECENCY_HALF_LIFE: f64 = 7.0 * 24.0 * 3600.0;

/// Compute composite retrieval score.
/// All components normalized to [0, 1].
fn composite_score(similarity: f32, entry: &MemoryEntry, now: u64) -> f32 {
    let relevance = similarity; // already [0, 1] for normalized vectors

    // Recency: exponential decay with configurable half-life
    #[allow(clippy::cast_precision_loss)] // age_secs: u64 -> f64, fine for time deltas
    let age_secs = now.saturating_sub(entry.last_accessed) as f64;
    #[allow(clippy::cast_possible_truncation)] // f64 -> f32 is intentional (scoring precision)
    let recency = (-(age_secs * (2.0_f64.ln())) / RECENCY_HALF_LIFE).exp() as f32;

    // Importance: normalize 1-10 to [0, 1], clamping at 1 for legacy data with 0
    let importance = (f32::from(entry.importance.max(1)) - 1.0) / 9.0;

    // Access frequency: log scale, capped at 1.0
    #[allow(clippy::cast_precision_loss)] // u32 -> f32, access_count won't exceed 2^23
    let access = (1.0 + entry.access_count as f32).ln() / 5.0_f32.ln(); // ln(1+n)/ln(5)
    let access = access.min(1.0);

    W_RELEVANCE * relevance + W_RECENCY * recency + W_IMPORTANCE * importance + W_ACCESS * access
}

// ---------------------------------------------------------------------------
// Embedding client (ollama /api/embed)
// ---------------------------------------------------------------------------

/// Embed text using ollama's /api/embed endpoint.
/// Returns a normalized 768-dim vector.
pub async fn embed_text(text: &str, ollama_url: &str, model: &str) -> Result<Vec<f32>, String> {
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
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

// ---------------------------------------------------------------------------
// Store operations
// ---------------------------------------------------------------------------

impl MemoryStore {
    /// Load from a JSON file. Returns empty store if file doesn't exist.
    /// If the file exists but is corrupt, backs it up before returning empty.
    #[must_use]
    pub fn load(path: &Path) -> Self {
        let Ok(content) = std::fs::read_to_string(path) else {
            return Self::default(); // File doesn't exist
        };
        match serde_json::from_str(&content) {
            Ok(store) => store,
            Err(e) => {
                // Corrupt file — back up before returning empty to avoid data loss
                let backup = path.with_extension("json.bak");
                let _ = std::fs::copy(path, &backup);
                eprintln!(
                    "[piku] warning: corrupt memories at {}, backed up to {}: {e}",
                    path.display(),
                    backup.display()
                );
                Self::default()
            }
        }
    }

    /// Save to a JSON file.
    pub fn save(&self, path: &Path) -> Result<(), String> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| format!("create_dir_all failed: {e}"))?;
        }
        let json =
            serde_json::to_string_pretty(self).map_err(|e| format!("serialize failed: {e}"))?;
        std::fs::write(path, json).map_err(|e| format!("write failed: {e}"))
    }

    /// Insert a new memory entry with a pre-computed embedding and importance score.
    /// Returns `None` if importance is below `MIN_IMPORTANCE` (admission gate).
    pub fn insert(
        &mut self,
        content: String,
        tags: Vec<String>,
        mut embedding: Vec<f32>,
        importance: u8,
    ) -> Option<u64> {
        if importance < MIN_IMPORTANCE {
            return None; // Importance gate: don't store low-value facts
        }
        normalize(&mut embedding);
        let id = self.next_id;
        self.next_id += 1;
        let now = now_unix();
        self.entries.push(MemoryEntry {
            id,
            content,
            tags,
            importance,
            created_at: now,
            last_accessed: now,
            access_count: 0,
            is_valid: true,
            supersedes: None,
            embedding,
            entry_type: EntryType::Fact,
            parent: None,
            outcome: None,
            outcome_detail: None,
            goal: None,
        });
        Some(id)
    }

    /// Insert a memory that supersedes an older one (contradiction resolution).
    /// Superseding entries bypass the importance gate (contradictions are always important).
    pub fn insert_superseding(
        &mut self,
        content: String,
        tags: Vec<String>,
        mut embedding: Vec<f32>,
        supersedes_id: u64,
        importance: u8,
    ) -> u64 {
        // Mark the old memory as invalid
        if let Some(old) = self.entries.iter_mut().find(|e| e.id == supersedes_id) {
            old.is_valid = false;
        }
        normalize(&mut embedding);
        let id = self.next_id;
        self.next_id += 1;
        let now = now_unix();
        self.entries.push(MemoryEntry {
            id,
            content,
            tags,
            importance: importance.max(MIN_IMPORTANCE), // always at least MIN
            created_at: now,
            last_accessed: now,
            access_count: 0,
            is_valid: true,
            supersedes: Some(supersedes_id),
            embedding,
            entry_type: EntryType::Fact,
            parent: None,
            outcome: None,
            outcome_detail: None,
            goal: None,
        });
        id
    }

    /// Retrieve the top-k valid memories ranked by composite score.
    /// Composite: relevance(cosine) + `recency(exp_decay)` + importance + `access_frequency`.
    /// Confidence-gated: cosine similarity must exceed `MIN_SIMILARITY`.
    /// Updates access metadata on retrieved entries.
    pub fn retrieve(&mut self, query_embedding: &[f32], k: usize) -> Vec<RetrievedMemory> {
        let now = now_unix();
        let mut scored: Vec<(f32, f32, usize)> = self
            .entries
            .iter()
            .enumerate()
            .filter(|(_, e)| e.is_valid)
            .map(|(i, e)| {
                let sim = dot(query_embedding, &e.embedding);
                let score = composite_score(sim, e, now);
                (sim, score, i)
            })
            .filter(|(sim, _, _)| *sim >= MIN_SIMILARITY)
            .collect();

        // Sort by composite score descending
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        scored
            .into_iter()
            .take(k)
            .map(|(sim, score, idx)| {
                self.entries[idx].last_accessed = now;
                self.entries[idx].access_count += 1;
                RetrievedMemory {
                    entry: self.entries[idx].clone(),
                    similarity: sim,
                    score,
                }
            })
            .collect()
    }

    /// Read-only retrieval (doesn't update access metadata).
    #[must_use]
    pub fn search(&self, query_embedding: &[f32], k: usize) -> Vec<RetrievedMemory> {
        let now = now_unix();
        let mut scored: Vec<(f32, f32, &MemoryEntry)> = self
            .entries
            .iter()
            .filter(|e| e.is_valid)
            .map(|e| {
                let sim = dot(query_embedding, &e.embedding);
                let score = composite_score(sim, e, now);
                (sim, score, e)
            })
            .filter(|(sim, _, _)| *sim >= MIN_SIMILARITY)
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        scored
            .into_iter()
            .take(k)
            .map(|(sim, score, entry)| RetrievedMemory {
                entry: entry.clone(),
                similarity: sim,
                score,
            })
            .collect()
    }

    /// Count of valid entries.
    #[must_use]
    pub fn valid_count(&self) -> usize {
        self.entries.iter().filter(|e| e.is_valid).count()
    }

    /// Find the top-k most similar existing memories to a candidate
    /// Hybrid retrieval: combines embedding similarity with keyword matching.
    /// Boosts memories whose tags or content match query keywords.
    /// This is the default search mode -- works at all scales.
    pub fn hybrid_retrieve(
        &mut self,
        query_embedding: &[f32],
        query_text: &str,
        k: usize,
    ) -> Vec<RetrievedMemory> {
        let now = now_unix();
        let query_terms: Vec<String> = query_text
            .to_lowercase()
            .split_whitespace()
            .filter(|w| w.len() > 2) // skip short words
            .map(String::from)
            .collect();

        let mut scored: Vec<(f32, f32, f32, usize)> = self
            .entries
            .iter()
            .enumerate()
            .filter(|(_, e)| e.is_valid)
            .map(|(i, e)| {
                let sim = dot(query_embedding, &e.embedding);
                let base_score = composite_score(sim, e, now);

                // Keyword boost: check tags and content for query term matches
                let haystack = format!(
                    "{} {}",
                    e.content.to_lowercase(),
                    e.tags.join(" ").to_lowercase()
                );
                let keyword_hits = query_terms
                    .iter()
                    .filter(|t| haystack.contains(t.as_str()))
                    .count();
                #[allow(clippy::cast_precision_loss)]
                // keyword_hits is always small (< query terms)
                let keyword_boost = (keyword_hits as f32 * 0.05).min(0.15);

                let final_score = base_score + keyword_boost;
                (sim, final_score, keyword_boost, i)
            })
            .filter(|(sim, _, _, _)| *sim >= MIN_SIMILARITY)
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        scored
            .into_iter()
            .take(k)
            .map(|(sim, score, _, idx)| {
                self.entries[idx].last_accessed = now;
                self.entries[idx].access_count += 1;
                RetrievedMemory {
                    entry: self.entries[idx].clone(),
                    similarity: sim,
                    score,
                }
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Tree operations (attempt tracking)
// ---------------------------------------------------------------------------

impl MemoryStore {
    /// Record a new attempt. Returns the attempt's ID.
    ///
    /// `goal`: what the agent is trying to achieve (used for retrieval).
    /// `approach`: what specific approach is being tried (stored as `content`).
    /// `parent`: parent attempt ID if this is a sub-approach or alternative.
    /// `embedding`: pre-computed embedding of the approach description.
    pub fn record_attempt(
        &mut self,
        goal: String,
        approach: String,
        parent: Option<u64>,
        mut embedding: Vec<f32>,
        importance: u8,
    ) -> u64 {
        normalize(&mut embedding);
        let id = self.next_id;
        self.next_id += 1;
        let now = now_unix();
        self.entries.push(MemoryEntry {
            id,
            content: approach,
            tags: vec![],
            importance: importance.max(MIN_IMPORTANCE),
            created_at: now,
            last_accessed: now,
            access_count: 0,
            is_valid: true,
            supersedes: None,
            embedding,
            entry_type: EntryType::Attempt,
            parent,
            outcome: Some(Outcome::Pending),
            outcome_detail: None,
            goal: Some(goal),
        });
        id
    }

    /// Record the outcome of an existing attempt.
    /// Returns `false` if the attempt ID was not found.
    pub fn record_outcome(
        &mut self,
        attempt_id: u64,
        outcome: Outcome,
        detail: Option<String>,
    ) -> bool {
        if let Some(entry) = self.entries.iter_mut().find(|e| e.id == attempt_id) {
            entry.outcome = Some(outcome);
            entry.outcome_detail = detail;
            entry.last_accessed = now_unix();
            true
        } else {
            false
        }
    }

    /// Get direct children of an attempt (other attempts with this ID as parent).
    #[must_use]
    pub fn children(&self, parent_id: u64) -> Vec<&MemoryEntry> {
        self.entries
            .iter()
            .filter(|e| e.parent == Some(parent_id) && e.entry_type == EntryType::Attempt)
            .collect()
    }

    /// Get siblings of an attempt (same parent, excluding self).
    #[must_use]
    pub fn siblings(&self, id: u64) -> Vec<&MemoryEntry> {
        let parent = self
            .entries
            .iter()
            .find(|e| e.id == id)
            .and_then(|e| e.parent);
        self.entries
            .iter()
            .filter(|e| e.id != id && e.parent == parent && e.entry_type == EntryType::Attempt)
            .collect()
    }

    /// Trace from an attempt back to the root. Returns entries from leaf to root.
    #[must_use]
    pub fn path_to_root(&self, id: u64) -> Vec<&MemoryEntry> {
        let mut path = Vec::new();
        let mut current_id = Some(id);
        // Guard against cycles (max 50 depth).
        let mut depth = 0;
        while let Some(cid) = current_id {
            if depth > 50 {
                break;
            }
            if let Some(entry) = self.entries.iter().find(|e| e.id == cid) {
                path.push(entry);
                current_id = entry.parent;
            } else {
                break;
            }
            depth += 1;
        }
        path
    }

    /// Find attempt trees relevant to a goal. Returns root attempts (no parent)
    /// that match the goal embedding, along with their full subtrees.
    #[must_use]
    pub fn find_attempt_trees(
        &self,
        goal_embedding: &[f32],
        goal_text: &str,
        max_roots: usize,
    ) -> Vec<AttemptTree> {
        let now = now_unix();
        let query_terms: Vec<String> = goal_text
            .to_lowercase()
            .split_whitespace()
            .filter(|w| w.len() > 2)
            .map(String::from)
            .collect();

        // Score all attempt entries by goal similarity.
        let mut scored: Vec<(f32, &MemoryEntry)> = self
            .entries
            .iter()
            .filter(|e| e.entry_type == EntryType::Attempt && e.is_valid)
            .map(|e| {
                // Match against goal text if present, otherwise content.
                let match_text = e.goal.as_deref().unwrap_or(&e.content).to_lowercase();
                let sim = dot(goal_embedding, &e.embedding);
                let keyword_hits = query_terms
                    .iter()
                    .filter(|t| match_text.contains(t.as_str()))
                    .count();
                #[allow(clippy::cast_precision_loss)]
                let boost = (keyword_hits as f32 * 0.05).min(0.15);
                let score = composite_score(sim, e, now) + boost;
                (score, e)
            })
            .filter(|(score, _)| *score > MIN_SIMILARITY)
            .collect();

        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        // Collect unique root IDs from top-scoring attempts.
        let mut seen_roots = Vec::new();
        let mut trees = Vec::new();

        for (_, entry) in &scored {
            let root_id = self
                .path_to_root(entry.id)
                .last()
                .map_or(entry.id, |r| r.id);
            if seen_roots.contains(&root_id) {
                continue;
            }
            seen_roots.push(root_id);
            if let Some(root) = self.entries.iter().find(|e| e.id == root_id) {
                trees.push(self.build_attempt_tree(root));
            }
            if trees.len() >= max_roots {
                break;
            }
        }

        trees
    }

    /// Build a tree structure from a root attempt.
    fn build_attempt_tree(&self, root: &MemoryEntry) -> AttemptTree {
        let children_entries = self.children(root.id);
        let children = children_entries
            .into_iter()
            .map(|c| self.build_attempt_tree(c))
            .collect();
        AttemptTree {
            id: root.id,
            approach: root.content.clone(),
            goal: root.goal.clone(),
            outcome: root.outcome,
            outcome_detail: root.outcome_detail.clone(),
            children,
        }
    }
}

/// A tree of attempts for display/retrieval.
#[derive(Debug, Clone)]
pub struct AttemptTree {
    pub id: u64,
    pub approach: String,
    pub goal: Option<String>,
    pub outcome: Option<Outcome>,
    pub outcome_detail: Option<String>,
    pub children: Vec<AttemptTree>,
}

impl AttemptTree {
    /// Format as an indented tree string.
    #[must_use]
    pub fn format(&self, indent: usize) -> String {
        let prefix = "  ".repeat(indent);
        let icon = match self.outcome {
            Some(Outcome::Success) => "[OK]",
            Some(Outcome::Failure) => "[FAIL]",
            Some(Outcome::Pending) => "[...]",
            None => "[-]",
        };
        let detail = self
            .outcome_detail
            .as_deref()
            .map(|d| format!(" -- {d}"))
            .unwrap_or_default();
        let goal_line = if indent == 0 {
            self.goal
                .as_deref()
                .map(|g| format!("{prefix}goal: {g}\n"))
                .unwrap_or_default()
        } else {
            String::new()
        };
        let mut out = format!("{goal_line}{prefix}{icon} {}{detail}\n", self.approach);
        for child in &self.children {
            out.push_str(&child.format(indent + 1));
        }
        out
    }
}

/// Format attempt trees for prompt injection.
#[must_use]
pub fn format_attempt_trees(trees: &[AttemptTree]) -> String {
    if trees.is_empty() {
        return String::new();
    }
    let mut out = String::from("\n\n# Prior Attempts\n\n");
    out.push_str(
        "The following attempt trees were found for similar goals. \
         Avoid repeating failed approaches.\n\n",
    );
    for tree in trees {
        out.push_str(&tree.format(0));
        out.push('\n');
    }
    out
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
    out.push_str(
        "The following memories were retrieved by semantic similarity to the current task.\n\n",
    );
    for m in memories {
        let _ = writeln!(
            out,
            "- [score={:.2}, sim={:.2}, imp={}] {}",
            m.score, m.similarity, m.entry.importance, m.entry.content
        );
        if !m.entry.tags.is_empty() {
            let _ = writeln!(out, "  tags: {}", m.entry.tags.join(", "));
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Session-end memory extraction
// ---------------------------------------------------------------------------

/// Prompt for extracting atomic facts and attempts from a conversation.
const EXTRACTION_PROMPT: &str = "\
You are a memory extraction system. Given a conversation between a user and an AI assistant, \
extract two kinds of memories:

## 1. Facts (0-5)
Atomic facts worth remembering for future sessions.
Format: `- [importance:N] fact text {tags: a, b, c}`

## 2. Attempts (0-3)
Approaches tried during the conversation and their outcomes. Only extract attempts where \
the outcome (success or failure) provides useful guidance for future similar tasks.
Format: `- [attempt] goal: GOAL | approach: APPROACH | outcome: success|failure | detail: WHY`

Rules for facts:
- Each fact must be self-contained (no pronouns, no \"the project\")
  - importance is 1-10:
    - 1-2: trivial, not worth storing
    - 3-4: mildly useful context
    - 5-6: useful preference or convention
    - 7-8: important decision or error pattern
    - 9-10: critical correction or safety constraint
  - tags: 1-4 lowercase keywords for this specific fact
- Include: user preferences, project conventions, error patterns, key decisions
- Exclude: transient task details, file contents, code snippets, things derivable from git

Rules for attempts:
- Only extract attempts where something was tried and the outcome was observed
- goal: what the agent was trying to achieve (specific enough to match future queries)
- approach: the specific strategy or method used
- outcome: success or failure (skip pending/unclear)
- detail: concise reason WHY it succeeded or failed (one sentence)
- Skip trivial operations (file reads, simple edits) -- only meaningful problem-solving attempts

If nothing is worth remembering, output exactly: NONE

Output format:
- [importance:7] The user prefers snake_case in Rust code {tags: rust, style}
- [importance:9] Never use byte slicing on UTF-8 strings {tags: rust, safety, unicode}
- [attempt] goal: fix UTF-8 rendering | approach: byte slicing for truncation | outcome: failure | detail: panics on multi-byte characters
- [attempt] goal: fix UTF-8 rendering | approach: char iterator with take(n) | outcome: success | detail: handles all unicode correctly

Or:
NONE";

/// Raw extraction from the LLM -- returns the full response text.
async fn extract_raw(
    conversation: &str,
    provider: &dyn piku_api::Provider,
    model: &str,
) -> Option<String> {
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
        return None;
    }

    Some(response)
}

/// Extract memories from a conversation transcript.
/// Returns a list of (fact, tags, importance) tuples.
pub async fn extract_memories(
    conversation: &str,
    provider: &dyn piku_api::Provider,
    model: &str,
) -> Vec<(String, Vec<String>, u8)> {
    match extract_raw(conversation, provider, model).await {
        Some(response) => parse_extraction_response(&response),
        None => Vec::new(),
    }
}

/// Parse the LLM's extraction response into (fact, tags, importance) tuples.
/// Supports per-fact tags: `- [importance:N] fact text {tags: a, b, c}`
/// Falls back to a shared `TAGS:` line if per-fact tags are absent.
fn parse_extraction_response(response: &str) -> Vec<(String, Vec<String>, u8)> {
    let mut facts = Vec::new();

    // Shared TAGS fallback (last TAGS: line applies to facts without inline tags)
    let mut shared_tags: Vec<String> = Vec::new();
    for line in response.lines().rev() {
        let trimmed = line.trim();
        if let Some(tags_str) = trimmed.strip_prefix("TAGS:") {
            shared_tags = tags_str
                .split(',')
                .map(|t| t.trim().to_lowercase())
                .filter(|t| !t.is_empty())
                .collect();
            break;
        }
    }

    for line in response.lines() {
        let trimmed = line.trim();
        let Some(fact_text) = trimmed.strip_prefix("- ") else {
            continue;
        };
        let fact_text = fact_text.trim();
        if fact_text.is_empty()
            || fact_text.starts_with("TAGS:")
            || fact_text.starts_with("[attempt]")
        {
            continue;
        }

        // Parse optional [importance:N] prefix
        let (importance, rest) = if fact_text.starts_with("[importance:") {
            if let Some(end) = fact_text.find(']') {
                let num_str = &fact_text[12..end];
                let imp = num_str.parse::<u8>().unwrap_or(5);
                (imp, fact_text[end + 1..].trim())
            } else {
                (5, fact_text)
            }
        } else {
            (5, fact_text)
        };

        // Parse optional inline {tags: a, b, c} suffix
        let (fact, tags) = if let Some(tag_start) = rest.rfind("{tags:") {
            let tag_end = rest[tag_start..].find('}').map(|p| tag_start + p + 1);
            let tag_str = &rest[tag_start + 6..tag_end.unwrap_or(rest.len()).saturating_sub(1)];
            let inline_tags: Vec<String> = tag_str
                .split(',')
                .map(|t| t.trim().to_lowercase())
                .filter(|t| !t.is_empty())
                .collect();
            (rest[..tag_start].trim().to_string(), inline_tags)
        } else {
            (rest.to_string(), shared_tags.clone())
        };

        if !fact.is_empty() {
            facts.push((fact, tags, importance));
        }
    }

    facts
}

/// A parsed attempt extraction from the LLM response.
#[derive(Debug, Clone)]
struct ExtractedAttempt {
    goal: String,
    approach: String,
    outcome: Outcome,
    detail: String,
}

/// Parse attempt lines from the extraction response.
/// Format: `- [attempt] goal: GOAL | approach: APPROACH | outcome: success|failure | detail: WHY`
fn parse_attempt_lines(response: &str) -> Vec<ExtractedAttempt> {
    let mut attempts = Vec::new();
    for line in response.lines() {
        let trimmed = line.trim();
        let Some(rest) = trimmed.strip_prefix("- [attempt]") else {
            continue;
        };
        let rest = rest.trim();
        // Parse pipe-separated fields
        let fields: std::collections::HashMap<&str, &str> = rest
            .split('|')
            .filter_map(|part| {
                let part = part.trim();
                let colon = part.find(':')?;
                let key = part[..colon].trim();
                let val = part[colon + 1..].trim();
                Some((key, val))
            })
            .collect();

        let Some(goal) = fields.get("goal") else {
            continue;
        };
        let Some(approach) = fields.get("approach") else {
            continue;
        };
        let Some(outcome_str) = fields.get("outcome") else {
            continue;
        };
        let outcome = match *outcome_str {
            "success" => Outcome::Success,
            "failure" => Outcome::Failure,
            _ => continue, // skip unclear outcomes
        };
        let detail = fields.get("detail").unwrap_or(&"");

        if !goal.is_empty() && !approach.is_empty() {
            attempts.push(ExtractedAttempt {
                goal: (*goal).to_string(),
                approach: (*approach).to_string(),
                outcome,
                detail: (*detail).to_string(),
            });
        }
    }
    attempts
}

/// Full session-end memory pipeline:
/// 1. Extract atomic facts and attempts from the conversation
/// 2. Embed each fact/attempt
/// 3. Check for contradictions against existing memories
/// 4. Insert or supersede (facts) / record as attempt tree nodes (attempts)
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
    let Some(response) = extract_raw(conversation, provider, model).await else {
        return 0;
    };

    let facts = parse_extraction_response(&response);
    let attempts = parse_attempt_lines(&response);

    if facts.is_empty() && attempts.is_empty() {
        return 0;
    }

    let mut count = 0;

    // Process facts (existing pipeline)
    for (fact, tags, importance) in facts {
        let Ok(embedding) = embed_text(&fact, ollama_url, embed_model).await else {
            continue;
        };

        // Find top-3 similar existing memories for contradiction check
        let similar = store.search(&embedding, 3);

        if similar.first().is_some_and(|s| s.similarity > 0.7) {
            // High similarity -- ask LLM to judge: ADD, UPDATE, or IGNORE
            let existing_texts: Vec<String> = similar
                .iter()
                .filter(|s| s.similarity > 0.5)
                .map(|s| format!("[id:{}] {}", s.entry.id, s.entry.content))
                .collect();
            let decision = judge_memory_conflict(&fact, &existing_texts, provider, model).await;
            match decision {
                MemoryJudgment::Add => {
                    if store.insert(fact, tags, embedding, importance).is_some() {
                        count += 1;
                    }
                }
                MemoryJudgment::Update(supersedes_id) => {
                    store.insert_superseding(fact, tags, embedding, supersedes_id, importance);
                    count += 1;
                }
                MemoryJudgment::Ignore => {} // duplicate or irrelevant
            }
        } else {
            // No similar memories -- just insert
            if store.insert(fact, tags, embedding, importance).is_some() {
                count += 1;
            }
        }
    }

    // Process attempts -- group by goal, create tree structure.
    // Dedup: if an agent already called record_attempt during the session,
    // the extracted attempt would duplicate it. Check embedding similarity
    // against existing attempts before inserting.
    let mut goal_roots: std::collections::HashMap<String, u64> = std::collections::HashMap::new();
    for attempt in attempts {
        let embed_text_str = format!("{} | {}", attempt.goal, attempt.approach);
        let Ok(embedding) = embed_text(&embed_text_str, ollama_url, embed_model).await else {
            continue;
        };
        // Dedup: check if a similar attempt already exists
        let similar = store.search(&embedding, 1);
        if similar
            .first()
            .is_some_and(|s| s.similarity > 0.85 && s.entry.entry_type == EntryType::Attempt)
        {
            // Near-duplicate attempt already recorded (likely via record_attempt tool)
            continue;
        }
        // Reuse or create a root for this goal
        let parent_id = goal_roots.get(&attempt.goal).copied();
        let id = store.record_attempt(
            attempt.goal.clone(),
            attempt.approach,
            parent_id,
            embedding,
            7, // attempts extracted at compaction are inherently important
        );
        store.record_outcome(id, attempt.outcome, Some(attempt.detail));
        // First attempt for this goal becomes the root
        goal_roots.entry(attempt.goal).or_insert(id);
        count += 1;
    }

    count
}

// ---------------------------------------------------------------------------
// LLM memory judge (Mem0-style ADD/UPDATE/IGNORE)
// ---------------------------------------------------------------------------

/// Decision from the LLM judge.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MemoryJudgment {
    /// New fact, no conflict with existing memories.
    Add,
    /// Updates/supersedes an existing memory (by ID).
    Update(u64),
    /// Duplicate or irrelevant -- don't store.
    Ignore,
}

const JUDGE_PROMPT: &str = "\
You are a memory conflict resolver. Given a NEW fact and EXISTING memories, decide:

- ADD: the new fact is genuinely new information (not a duplicate or update)
- UPDATE <id>: the new fact supersedes/corrects an existing memory (give the id to replace)
- IGNORE: the new fact is a duplicate or already covered by existing memories

Respond with exactly one line: ADD, UPDATE <id>, or IGNORE.";

/// Ask the LLM to judge whether a new fact should be added, updated, or ignored.
async fn judge_memory_conflict(
    new_fact: &str,
    existing: &[String],
    provider: &dyn piku_api::Provider,
    model: &str,
) -> MemoryJudgment {
    use futures_util::StreamExt;

    let existing_text = existing.join("\n");
    let user_msg =
        format!("NEW FACT: {new_fact}\n\nEXISTING MEMORIES:\n{existing_text}\n\nDecision:");

    let request = piku_api::MessageRequest {
        model: model.to_string(),
        max_tokens: 32,
        messages: vec![piku_api::RequestMessage {
            role: "user".to_string(),
            content: vec![piku_api::RequestContent::Text { text: user_msg }],
        }],
        system: Some(vec![piku_api::SystemBlock::text(JUDGE_PROMPT.to_string())]),
        tools: None,
        stream: true,
    };

    let mut stream = provider.stream_message(request);
    let mut response = String::new();

    let timeout = std::time::Duration::from_secs(10);
    let result = tokio::time::timeout(timeout, async {
        while let Some(event) = stream.next().await {
            if let Ok(piku_api::Event::TextDelta { text }) = event {
                response.push_str(&text);
            }
        }
    })
    .await;

    if result.is_err() {
        return MemoryJudgment::Add; // timeout → safe default: add
    }

    parse_judgment(&response)
}

fn parse_judgment(response: &str) -> MemoryJudgment {
    let trimmed = response.trim().to_uppercase();
    if trimmed == "IGNORE" {
        return MemoryJudgment::Ignore;
    }
    if trimmed == "ADD" {
        return MemoryJudgment::Add;
    }
    if let Some(rest) = trimmed.strip_prefix("UPDATE") {
        let id_str = rest.trim();
        if let Ok(id) = id_str.parse::<u64>() {
            return MemoryJudgment::Update(id);
        }
    }
    MemoryJudgment::Add // unparseable → safe default
}

// ---------------------------------------------------------------------------
// Ebbinghaus-inspired eviction
// ---------------------------------------------------------------------------

/// Strength score for eviction decisions.
/// strength = `importance_norm` * `recency_decay` * (1 + log(1 + `access_count`))
/// Low strength = candidate for eviction.
#[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
fn memory_strength(entry: &MemoryEntry, now: u64) -> f32 {
    let importance_norm = f32::from(entry.importance.max(1)) / 10.0;
    let age_secs = now.saturating_sub(entry.last_accessed) as f64;
    let recency = (-(age_secs * (2.0_f64.ln())) / RECENCY_HALF_LIFE).exp() as f32;
    let access_boost = (1.0 + entry.access_count as f32).ln() + 1.0;
    importance_norm * recency * access_boost
}

impl MemoryStore {
    /// Whether an attempt has any valid children.
    fn has_valid_children(&self, id: u64) -> bool {
        self.entries
            .iter()
            .any(|e| e.parent == Some(id) && e.is_valid)
    }

    /// Whether an attempt tree rooted at `id` is resolved (has at least one
    /// successful descendant). Unresolved trees (all failures, still searching)
    /// should be retained longer.
    fn tree_is_resolved(&self, id: u64) -> bool {
        // Check direct children for success
        for child in &self.entries {
            if child.parent == Some(id) && child.is_valid {
                if child.outcome == Some(Outcome::Success) {
                    return true;
                }
                // Recurse into child subtrees
                if self.tree_is_resolved(child.id) {
                    return true;
                }
            }
        }
        false
    }

    /// Check whether an attempt entry is safe to evict.
    /// Returns false if eviction would orphan children or discard an unresolved tree.
    fn attempt_evictable(&self, entry: &MemoryEntry) -> bool {
        if entry.entry_type != EntryType::Attempt {
            return true; // facts use normal eviction rules
        }
        // Never orphan children
        if self.has_valid_children(entry.id) {
            return false;
        }
        // Find the tree root
        let root_id = self
            .path_to_root(entry.id)
            .last()
            .map_or(entry.id, |r| r.id);
        // Keep unresolved trees (still useful for guiding future attempts)
        if !self.tree_is_resolved(root_id) {
            return false;
        }
        true
    }

    /// Evict (invalidate) memories with strength below `threshold`.
    /// Returns the number of entries evicted.
    /// Does NOT delete -- marks `is_valid = false` to preserve audit trail.
    /// Tree-aware: won't orphan attempt children or discard unresolved attempt trees.
    pub fn evict_weak(&mut self, threshold: f32) -> usize {
        let now = now_unix();
        // Collect IDs to evict (can't mutate while iterating with tree checks).
        let to_evict: Vec<u64> = self
            .entries
            .iter()
            .filter(|e| {
                e.is_valid
                    && e.importance < 5
                    && memory_strength(e, now) < threshold
            })
            .map(|e| e.id)
            .collect();
        let mut evicted = 0;
        for id in to_evict {
            // Re-check tree safety (earlier evictions in this pass may have changed the tree)
            let dominated_by_tree = self
                .entries
                .iter()
                .find(|e| e.id == id)
                .is_some_and(|e| !self.attempt_evictable(e));
            if dominated_by_tree {
                continue;
            }
            if let Some(entry) = self.entries.iter_mut().find(|e| e.id == id) {
                entry.is_valid = false;
                evicted += 1;
            }
        }
        evicted
    }

    /// Evict memories that haven't been accessed in `max_age_days` days
    /// AND have low importance (< 5).
    /// Tree-aware: won't orphan attempt children or discard unresolved attempt trees.
    pub fn evict_stale(&mut self, max_age_days: u32) -> usize {
        let now = now_unix();
        let max_age_secs = u64::from(max_age_days) * 24 * 3600;
        let to_evict: Vec<u64> = self
            .entries
            .iter()
            .filter(|e| {
                e.is_valid
                    && e.importance < 5
                    && now.saturating_sub(e.last_accessed) > max_age_secs
            })
            .map(|e| e.id)
            .collect();
        let mut evicted = 0;
        for id in to_evict {
            let dominated_by_tree = self
                .entries
                .iter()
                .find(|e| e.id == id)
                .is_some_and(|e| !self.attempt_evictable(e));
            if dominated_by_tree {
                continue;
            }
            if let Some(entry) = self.entries.iter_mut().find(|e| e.id == id) {
                entry.is_valid = false;
                evicted += 1;
            }
        }
        evicted
    }

    /// Run maintenance: evict weak memories and report.
    /// Call at session start or periodically.
    pub fn maintain(&mut self) -> (usize, usize) {
        let stale = self.evict_stale(30); // 30-day stale threshold
        let weak = self.evict_weak(0.05); // very low strength threshold
        (stale, weak)
    }
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
        // Show all entries (including invalid) so operator has full visibility.
        // Sort valid first, then by recency within each group.
        let mut sorted: Vec<&MemoryEntry> = self.entries.iter().collect();
        sorted.sort_by(|a, b| {
            b.is_valid
                .cmp(&a.is_valid)
                .then(b.created_at.cmp(&a.created_at))
        });
        sorted
            .into_iter()
            .take(max)
            .map(|e| (e.id, e.content.clone(), e.is_valid, e.access_count))
            .collect()
    }

    fn inspect(&self, id: u64) -> Option<String> {
        self.entries.iter().find(|e| e.id == id).map(|e| {
            let mut out = format!(
                "ID: {}\nType: {:?}\nContent: {}\nTags: {}\nCreated: {}\nLast accessed: {}\nAccess count: {}\nValid: {}\nSupersedes: {:?}\nEmbedding dims: {}",
                e.id,
                e.entry_type,
                e.content,
                e.tags.join(", "),
                e.created_at,
                e.last_accessed,
                e.access_count,
                e.is_valid,
                e.supersedes,
                e.embedding.len()
            );
            if e.entry_type == EntryType::Attempt {
                if let Some(ref goal) = e.goal {
                    let _ = write!(out, "\nGoal: {goal}");
                }
                if let Some(parent) = e.parent {
                    let _ = write!(out, "\nParent: {parent}");
                }
                if let Some(ref outcome) = e.outcome {
                    let _ = write!(out, "\nOutcome: {outcome:?}");
                }
                if let Some(ref detail) = e.outcome_detail {
                    let _ = write!(out, "\nOutcome detail: {detail}");
                }
                let children = self.children(e.id);
                if !children.is_empty() {
                    let child_ids: Vec<String> = children.iter().map(|c| c.id.to_string()).collect();
                    let _ = write!(out, "\nChildren: [{}]", child_ids.join(", "));
                }
            }
            out
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

    fn attempt_count(&self) -> usize {
        self.entries
            .iter()
            .filter(|e| e.is_valid && e.entry_type == EntryType::Attempt)
            .count()
    }

    fn record_outcome(&mut self, attempt_id: u64, outcome: &str, detail: Option<String>) -> bool {
        let outcome_enum = match outcome {
            "success" => Outcome::Success,
            "failure" => Outcome::Failure,
            "pending" => Outcome::Pending,
            _ => return false,
        };
        self.record_outcome(attempt_id, outcome_enum, detail)
    }
}

/// Build a compact conversation transcript for memory extraction.
/// Includes only user and assistant text blocks, truncated to keep
/// the extraction prompt manageable.
#[must_use]
pub fn build_extraction_transcript(messages: &[crate::session::ConversationMessage]) -> String {
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
                    if text.chars().count() > 500 {
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

    #[allow(clippy::cast_precision_loss)] // test helper, precision doesn't matter
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
        store.insert(
            "fact about rust".to_string(),
            vec!["rust".to_string()],
            e1.clone(),
            7,
        );
        store.insert(
            "fact about python".to_string(),
            vec!["python".to_string()],
            e2,
            5,
        );

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
        store.insert("unrelated fact".to_string(), vec![], orthogonal, 5);

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
        let id1 = store
            .insert("old fact".to_string(), vec![], e1.clone(), 7)
            .unwrap();
        store.insert_superseding("new fact".to_string(), vec![], e1.clone(), id1, 8);

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
        store.insert("test fact".to_string(), vec!["test".to_string()], e1, 6);
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
        store.insert("tracked fact".to_string(), vec![], e1.clone(), 5);

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
            importance: 7,
            created_at: 0,
            last_accessed: 0,
            access_count: 0,
            is_valid: true,
            supersedes: None,
            embedding: vec![],
            entry_type: EntryType::Fact,
            parent: None,
            outcome: None,
            outcome_detail: None,
            goal: None,
        };
        let retrieved = vec![RetrievedMemory {
            entry,
            similarity: 0.85,
            score: 0.72,
        }];
        let formatted = format_retrieved_memories(&retrieved);
        assert!(formatted.contains("score=0.72"));
        assert!(formatted.contains("sim=0.85"));
        assert!(formatted.contains("rust is good"));
        assert!(formatted.contains("tags: rust"));
    }

    #[test]
    fn parse_extraction_per_fact_tags() {
        let response = "- [importance:7] The user prefers snake_case {tags: rust, style}\n\
                        - [importance:5] The project uses tokio {tags: rust, async}";
        let facts = super::parse_extraction_response(response);
        assert_eq!(facts.len(), 2);
        assert_eq!(facts[0].0, "The user prefers snake_case");
        assert_eq!(facts[0].1, vec!["rust", "style"]);
        assert_eq!(facts[0].2, 7);
        assert_eq!(facts[1].1, vec!["rust", "async"]);
    }

    #[test]
    fn parse_extraction_shared_tags_fallback() {
        // When no inline {tags:}, shared TAGS: line is used
        let response = "- [importance:7] The user prefers snake_case\n\
                        TAGS: rust, style";
        let facts = super::parse_extraction_response(response);
        assert_eq!(facts.len(), 1);
        assert_eq!(facts[0].1, vec!["rust", "style"]);
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
        assert_eq!(facts[0].2, 5); // default importance
    }

    #[test]
    fn parse_extraction_importance_gate() {
        let response = "- [importance:2] Trivial fact\n\
                        - [importance:8] Important fact";
        let facts = super::parse_extraction_response(response);
        assert_eq!(facts.len(), 2);
        // Both parsed, but the store's insert() will gate the importance:2 one
        assert_eq!(facts[0].2, 2);
        assert_eq!(facts[1].2, 8);
    }

    #[test]
    fn importance_gate_on_insert() {
        let mut store = super::MemoryStore::default();
        let e = make_embedding(1.0);
        // Below MIN_IMPORTANCE (3) -- should be rejected
        assert!(store
            .insert("low importance".to_string(), vec![], e.clone(), 2)
            .is_none());
        // At MIN_IMPORTANCE -- should be accepted
        assert!(store
            .insert("medium importance".to_string(), vec![], e.clone(), 3)
            .is_some());
        assert_eq!(store.valid_count(), 1);
    }

    #[test]
    fn composite_scoring_prefers_recent_important() {
        let mut store = super::MemoryStore::default();
        let e = make_embedding(1.0);
        // Old, low importance
        store.insert("old low".to_string(), vec![], e.clone(), 3);
        // Make it old by tweaking created_at
        store.entries[0].created_at = 0;
        store.entries[0].last_accessed = 0;
        // New, high importance
        store.insert("new high".to_string(), vec![], e.clone(), 9);

        let results = store.retrieve(&e, 2);
        assert_eq!(results.len(), 2);
        // New high-importance should rank first due to composite scoring
        assert_eq!(results[0].entry.content, "new high");
        assert!(results[0].score > results[1].score);
    }

    #[test]
    fn build_transcript_filters_roles() {
        use crate::session::{ContentBlock, ConversationMessage};
        let messages = vec![
            ConversationMessage::user("hello"),
            ConversationMessage::assistant(
                vec![ContentBlock::Text {
                    text: "hi there".to_string(),
                }],
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

    // --- MemoryStoreView trait tests ---

    #[test]
    fn store_view_stats() {
        use piku_tools::embed_memory_tool::piku_runtime_types::MemoryStoreView;
        let mut store = MemoryStore::default();
        let e = make_embedding(1.0);
        store.insert("valid".to_string(), vec![], e.clone(), 5);
        store.insert("also valid".to_string(), vec![], e.clone(), 7);
        assert_eq!(store.total_count(), 2);
        assert_eq!(store.valid_count(), 2);
    }

    #[test]
    fn store_view_invalidate() {
        use piku_tools::embed_memory_tool::piku_runtime_types::MemoryStoreView;
        let mut store = MemoryStore::default();
        let e = make_embedding(1.0);
        let id = store
            .insert("to invalidate".to_string(), vec![], e, 5)
            .unwrap();
        assert!(store.invalidate(id));
        assert_eq!(store.valid_count(), 0);
        assert!(!store.invalidate(999)); // nonexistent
    }

    #[test]
    fn store_view_query_tags() {
        use piku_tools::embed_memory_tool::piku_runtime_types::MemoryStoreView;
        let mut store = MemoryStore::default();
        let e = make_embedding(1.0);
        store.insert(
            "rust fact".to_string(),
            vec!["rust".to_string(), "lang".to_string()],
            e.clone(),
            5,
        );
        store.insert(
            "python fact".to_string(),
            vec!["python".to_string()],
            e.clone(),
            5,
        );
        let results = store.query_by_tag("rust", 10);
        assert_eq!(results.len(), 1);
        assert!(results[0].1.contains("rust fact"));
        // Case insensitive
        let results2 = store.query_by_tag("RUST", 10);
        assert_eq!(results2.len(), 1);
    }

    #[test]
    fn store_view_list_recent() {
        use piku_tools::embed_memory_tool::piku_runtime_types::MemoryStoreView;
        let mut store = MemoryStore::default();
        let e = make_embedding(1.0);
        store.insert("first".to_string(), vec![], e.clone(), 5);
        // Manually set different timestamps since both inserts happen in same ms
        store.entries[0].created_at = 100;
        store.insert("second".to_string(), vec![], e.clone(), 5);
        store.entries[1].created_at = 200;
        let recent = store.list_recent(1);
        assert_eq!(recent.len(), 1);
        assert!(recent[0].1.contains("second")); // most recent first
    }

    #[test]
    fn store_view_inspect() {
        use piku_tools::embed_memory_tool::piku_runtime_types::MemoryStoreView;
        let mut store = MemoryStore::default();
        let e = make_embedding(1.0);
        let id = store
            .insert("inspectable".to_string(), vec!["tag1".to_string()], e, 8)
            .unwrap();
        let detail = store.inspect(id).unwrap();
        assert!(detail.contains("inspectable"));
        assert!(detail.contains("tag1"));
        assert!(detail.contains("Valid: true"));
        assert!(store.inspect(999).is_none());
    }

    // --- Corrupt file handling ---

    #[test]
    fn corrupt_json_backed_up() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("corrupt.json");
        std::fs::write(&path, "not valid json{{{").unwrap();
        let store = MemoryStore::load(&path);
        assert!(store.entries.is_empty()); // returns empty
        assert!(path.with_extension("json.bak").exists()); // backup created
    }

    // --- Composite scoring edge cases ---

    #[test]
    fn composite_score_zero_importance() {
        // importance=0 should not cause division or NaN
        let e = MemoryEntry {
            id: 0,
            content: String::new(),
            tags: vec![],
            importance: 0,
            created_at: 0,
            last_accessed: 0,
            access_count: 0,
            is_valid: true,
            supersedes: None,
            embedding: vec![],
            entry_type: EntryType::Fact,
            parent: None,
            outcome: None,
            outcome_detail: None,
            goal: None,
        };
        let score = super::composite_score(0.5, &e, super::now_unix());
        assert!(score.is_finite());
        assert!(score >= 0.0);
    }

    #[test]
    fn composite_score_high_access_count_capped() {
        let e = MemoryEntry {
            id: 0,
            content: String::new(),
            tags: vec![],
            importance: 5,
            created_at: super::now_unix(),
            last_accessed: super::now_unix(),
            access_count: 1_000_000,
            is_valid: true,
            supersedes: None,
            embedding: vec![],
            entry_type: EntryType::Fact,
            parent: None,
            outcome: None,
            outcome_detail: None,
            goal: None,
        };
        let score = super::composite_score(0.5, &e, super::now_unix());
        assert!(score.is_finite());
        assert!(score <= 1.0); // should be bounded
    }

    // --- Search with no valid entries ---

    #[test]
    fn hybrid_retrieve_boosts_keyword_matches() {
        let mut store = MemoryStore::default();
        let e = make_embedding(1.0);
        // Both have similar embeddings, but one has a tag matching the query
        store.insert(
            "rust fact".to_string(),
            vec!["rust".to_string()],
            e.clone(),
            5,
        );
        store.insert(
            "python fact".to_string(),
            vec!["python".to_string()],
            e.clone(),
            5,
        );

        // Query with "rust" keyword should boost the rust-tagged entry
        let results = store.hybrid_retrieve(&e, "rust programming", 2);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].entry.content, "rust fact");
        assert!(results[0].score >= results[1].score);
    }

    #[test]
    fn search_empty_store_returns_empty() {
        let store = MemoryStore::default();
        let e = make_embedding(1.0);
        let results = store.search(&e, 5);
        assert!(results.is_empty());
    }

    // --- Eviction tests ---

    #[test]
    fn evict_stale_removes_old_low_importance() {
        let mut store = MemoryStore::default();
        let e = make_embedding(1.0);
        store.insert("old low".to_string(), vec![], e.clone(), 3);
        store.entries[0].last_accessed = 0; // epoch = very old
        store.entries[0].created_at = 0;
        store.insert("recent high".to_string(), vec![], e.clone(), 9);

        let evicted = store.evict_stale(30);
        assert_eq!(evicted, 1);
        assert!(!store.entries[0].is_valid); // old low evicted
        assert!(store.entries[1].is_valid); // recent high kept
    }

    #[test]
    fn evict_stale_keeps_high_importance_old() {
        let mut store = MemoryStore::default();
        let e = make_embedding(1.0);
        store.insert("old but important".to_string(), vec![], e.clone(), 8);
        store.entries[0].last_accessed = 0;
        let evicted = store.evict_stale(30);
        assert_eq!(evicted, 0); // importance >= 5, kept
    }

    #[test]
    fn evict_weak_removes_zero_strength() {
        let mut store = MemoryStore::default();
        let e = make_embedding(1.0);
        store.insert("zero strength".to_string(), vec![], e.clone(), 3);
        store.entries[0].last_accessed = 0;
        store.entries[0].created_at = 0;
        store.entries[0].access_count = 0;
        let evicted = store.evict_weak(0.05);
        assert!(evicted > 0);
    }

    #[test]
    fn maintain_returns_counts() {
        let mut store = MemoryStore::default();
        let e = make_embedding(1.0);
        store.insert("healthy".to_string(), vec![], e.clone(), 7);
        let (stale, weak) = store.maintain();
        // Recent, medium importance -- should not be evicted
        assert_eq!(stale, 0);
        assert_eq!(weak, 0);
    }

    #[test]
    fn maintain_preserves_high_importance_old_memories() {
        // Regression: evict_weak's multiplicative strength formula would kill
        // high-importance old memories that evict_stale protects. The importance
        // floor in evict_weak must prevent this.
        let mut store = MemoryStore::default();
        let e = make_embedding(1.0);

        // High importance, very old (90 days), accessed only once
        store.insert("critical safety rule".to_string(), vec![], e.clone(), 9);
        store.entries[0].last_accessed = 0;
        store.entries[0].created_at = 0;
        store.entries[0].access_count = 1;

        // Low importance, also old
        store.insert("trivial note".to_string(), vec![], e.clone(), 3);
        store.entries[1].last_accessed = 0;
        store.entries[1].created_at = 0;
        store.entries[1].access_count = 0;

        let (stale, weak) = store.maintain();

        // The trivial note should be evicted (low importance + old)
        assert!(stale + weak > 0, "trivial note should be evicted");

        // The critical rule MUST survive both eviction passes
        assert!(
            store.entries[0].is_valid,
            "high-importance old memory must survive maintain()"
        );
    }

    // --- Embedding edge cases ---

    #[test]
    fn search_with_mismatched_dimensions_does_not_panic() {
        // dot() uses zip which silently truncates to the shorter vector.
        // This test documents the behavior: wrong scores, no panic.
        let mut store = MemoryStore::default();
        let e768 = make_embedding(1.0); // 768d
        store.insert("fact one".to_string(), vec![], e768, 7);

        // Query with a 384d vector (dimension mismatch)
        let short_query: Vec<f32> = make_embedding(1.0).into_iter().take(384).collect();
        let results = store.search(&short_query, 5);
        // Should not panic -- zip truncates silently.
        // Results may be returned but scores are meaningless.
        assert!(results.len() <= 1);
    }

    #[test]
    fn empty_store_search_returns_empty() {
        let store = MemoryStore::default();
        let query = make_embedding(1.0);
        let results = store.search(&query, 10);
        assert!(results.is_empty());
    }

    #[test]
    fn search_excludes_invalid_entries() {
        let mut store = MemoryStore::default();
        let e = make_embedding(1.0);
        store.insert("valid fact".to_string(), vec![], e.clone(), 7);
        store.insert("invalid fact".to_string(), vec![], e.clone(), 7);
        store.entries[1].is_valid = false;

        let results = store.search(&e, 10);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].entry.content, "valid fact");
    }

    // --- LLM judge parsing ---

    #[test]
    fn parse_judgment_add() {
        assert_eq!(super::parse_judgment("ADD"), super::MemoryJudgment::Add);
        assert_eq!(super::parse_judgment("add"), super::MemoryJudgment::Add);
        assert_eq!(super::parse_judgment("  ADD  "), super::MemoryJudgment::Add);
    }

    #[test]
    fn parse_judgment_ignore() {
        assert_eq!(
            super::parse_judgment("IGNORE"),
            super::MemoryJudgment::Ignore
        );
    }

    #[test]
    fn parse_judgment_update() {
        assert_eq!(
            super::parse_judgment("UPDATE 42"),
            super::MemoryJudgment::Update(42)
        );
        assert_eq!(
            super::parse_judgment("update 7"),
            super::MemoryJudgment::Update(7)
        );
    }

    #[test]
    fn parse_judgment_garbage_defaults_to_add() {
        assert_eq!(super::parse_judgment("dunno"), super::MemoryJudgment::Add);
        assert_eq!(super::parse_judgment(""), super::MemoryJudgment::Add);
    }

    #[test]
    fn search_all_invalid_returns_empty() {
        let mut store = MemoryStore::default();
        let e = make_embedding(1.0);
        let id = store
            .insert("will invalidate".to_string(), vec![], e.clone(), 5)
            .unwrap();
        store
            .entries
            .iter_mut()
            .find(|x| x.id == id)
            .unwrap()
            .is_valid = false;
        let results = store.search(&e, 5);
        assert!(results.is_empty());
    }

    // --- Attempt tree tests ---

    #[test]
    fn record_attempt_creates_attempt_entry() {
        let mut store = MemoryStore::default();
        let e = make_embedding(1.0);
        let id = store.record_attempt(
            "fix the bug".to_string(),
            "change return type".to_string(),
            None,
            e,
            6,
        );
        assert_eq!(store.entries.len(), 1);
        let entry = &store.entries[0];
        assert_eq!(entry.id, id);
        assert_eq!(entry.entry_type, EntryType::Attempt);
        assert_eq!(entry.goal.as_deref(), Some("fix the bug"));
        assert_eq!(entry.content, "change return type");
        assert_eq!(entry.outcome, Some(Outcome::Pending));
        assert!(entry.parent.is_none());
    }

    #[test]
    fn record_outcome_updates_attempt() {
        let mut store = MemoryStore::default();
        let e = make_embedding(1.0);
        let id = store.record_attempt(
            "fix the bug".to_string(),
            "change return type".to_string(),
            None,
            e,
            6,
        );
        assert!(store.record_outcome(id, Outcome::Failure, Some("broke 3 call sites".to_string())));
        let entry = &store.entries[0];
        assert_eq!(entry.outcome, Some(Outcome::Failure));
        assert_eq!(entry.outcome_detail.as_deref(), Some("broke 3 call sites"));
    }

    #[test]
    fn record_outcome_nonexistent_returns_false() {
        let mut store = MemoryStore::default();
        assert!(!store.record_outcome(999, Outcome::Success, None));
    }

    #[test]
    fn children_returns_direct_children() {
        let mut store = MemoryStore::default();
        let e = make_embedding(1.0);
        let root = store.record_attempt("goal".into(), "root".into(), None, e.clone(), 6);
        let child1 = store.record_attempt("goal".into(), "child1".into(), Some(root), e.clone(), 6);
        let _child2 =
            store.record_attempt("goal".into(), "child2".into(), Some(root), e.clone(), 6);
        // Grandchild should not appear
        store.record_attempt(
            "goal".into(),
            "grandchild".into(),
            Some(child1),
            e.clone(),
            6,
        );

        let children = store.children(root);
        assert_eq!(children.len(), 2);
        assert!(children.iter().any(|c| c.content == "child1"));
        assert!(children.iter().any(|c| c.content == "child2"));
    }

    #[test]
    fn siblings_excludes_self() {
        let mut store = MemoryStore::default();
        let e = make_embedding(1.0);
        let root = store.record_attempt("goal".into(), "root".into(), None, e.clone(), 6);
        let child1 = store.record_attempt("goal".into(), "child1".into(), Some(root), e.clone(), 6);
        store.record_attempt("goal".into(), "child2".into(), Some(root), e.clone(), 6);

        let siblings = store.siblings(child1);
        assert_eq!(siblings.len(), 1);
        assert_eq!(siblings[0].content, "child2");
    }

    #[test]
    fn path_to_root_traces_ancestry() {
        let mut store = MemoryStore::default();
        let e = make_embedding(1.0);
        let root = store.record_attempt("goal".into(), "root".into(), None, e.clone(), 6);
        let child = store.record_attempt("goal".into(), "child".into(), Some(root), e.clone(), 6);
        let grandchild =
            store.record_attempt("goal".into(), "gc".into(), Some(child), e.clone(), 6);

        let path = store.path_to_root(grandchild);
        assert_eq!(path.len(), 3);
        assert_eq!(path[0].content, "gc"); // leaf first
        assert_eq!(path[1].content, "child");
        assert_eq!(path[2].content, "root"); // root last
    }

    #[test]
    fn path_to_root_single_node() {
        let mut store = MemoryStore::default();
        let e = make_embedding(1.0);
        let root = store.record_attempt("goal".into(), "root".into(), None, e, 6);
        let path = store.path_to_root(root);
        assert_eq!(path.len(), 1);
        assert_eq!(path[0].content, "root");
    }

    #[test]
    fn build_attempt_tree_structure() {
        let mut store = MemoryStore::default();
        let e = make_embedding(1.0);
        let root = store.record_attempt("fix bug".into(), "approach A".into(), None, e.clone(), 6);
        store.record_outcome(root, Outcome::Failure, Some("type error".into()));
        let child = store.record_attempt(
            "fix bug".into(),
            "approach B".into(),
            Some(root),
            e.clone(),
            6,
        );
        store.record_outcome(child, Outcome::Success, Some("worked".into()));

        let tree = store.build_attempt_tree(&store.entries[0].clone());
        assert_eq!(tree.approach, "approach A");
        assert_eq!(tree.outcome, Some(Outcome::Failure));
        assert_eq!(tree.children.len(), 1);
        assert_eq!(tree.children[0].approach, "approach B");
        assert_eq!(tree.children[0].outcome, Some(Outcome::Success));
    }

    #[test]
    fn format_attempt_tree_rendering() {
        let tree = AttemptTree {
            id: 0,
            approach: "try X".to_string(),
            goal: Some("fix the thing".to_string()),
            outcome: Some(Outcome::Failure),
            outcome_detail: Some("broke it".to_string()),
            children: vec![AttemptTree {
                id: 1,
                approach: "try Y".to_string(),
                goal: None,
                outcome: Some(Outcome::Success),
                outcome_detail: None,
                children: vec![],
            }],
        };
        let formatted = tree.format(0);
        assert!(formatted.contains("goal: fix the thing"));
        assert!(formatted.contains("[FAIL] try X -- broke it"));
        assert!(formatted.contains("  [OK] try Y"));
    }

    #[test]
    fn format_attempt_trees_empty() {
        let formatted = format_attempt_trees(&[]);
        assert!(formatted.is_empty());
    }

    #[test]
    fn find_attempt_trees_returns_matching() {
        let mut store = MemoryStore::default();
        let e = make_embedding(1.0);
        // Create two separate attempt trees
        let root1 = store.record_attempt(
            "fix auth bug".into(),
            "try reset".into(),
            None,
            e.clone(),
            6,
        );
        store.record_outcome(root1, Outcome::Failure, Some("didn't work".into()));
        store.record_attempt(
            "fix auth bug".into(),
            "try logout".into(),
            Some(root1),
            e.clone(),
            6,
        );

        let root2 = store.record_attempt(
            "improve perf".into(),
            "add cache".into(),
            None,
            make_embedding(100.0),
            6,
        );
        store.record_outcome(root2, Outcome::Success, None);

        // Query with embedding similar to root1
        let trees = store.find_attempt_trees(&e, "fix auth bug", 5);
        // Should find at least root1's tree (both have similar embeddings)
        assert!(!trees.is_empty());
    }

    #[test]
    fn attempt_backward_compat_serde() {
        // Old-format JSON (no tree fields) should deserialize with defaults
        let json = r#"{
            "next_id": 1,
            "entries": [{
                "id": 0,
                "content": "old fact",
                "tags": ["test"],
                "importance": 5,
                "created_at": 1000,
                "last_accessed": 1000,
                "access_count": 0,
                "is_valid": true,
                "supersedes": null,
                "embedding": [1.0, 0.0, 0.0]
            }]
        }"#;
        let store: MemoryStore = serde_json::from_str(json).unwrap();
        assert_eq!(store.entries.len(), 1);
        assert_eq!(store.entries[0].entry_type, EntryType::Fact);
        assert!(store.entries[0].parent.is_none());
        assert!(store.entries[0].outcome.is_none());
        assert!(store.entries[0].goal.is_none());
    }

    #[test]
    fn attempt_serde_roundtrip() {
        let mut store = MemoryStore::default();
        let e = make_embedding(1.0);
        let id = store.record_attempt("goal".into(), "approach".into(), None, e, 7);
        store.record_outcome(id, Outcome::Failure, Some("reason".into()));

        let json = serde_json::to_string(&store).unwrap();
        let loaded: MemoryStore = serde_json::from_str(&json).unwrap();
        assert_eq!(loaded.entries[0].entry_type, EntryType::Attempt);
        assert_eq!(loaded.entries[0].outcome, Some(Outcome::Failure));
        assert_eq!(loaded.entries[0].outcome_detail.as_deref(), Some("reason"));
        assert_eq!(loaded.entries[0].goal.as_deref(), Some("goal"));
    }

    #[test]
    fn attempt_count_via_store_view() {
        use piku_tools::embed_memory_tool::piku_runtime_types::MemoryStoreView;
        let mut store = MemoryStore::default();
        let e = make_embedding(1.0);
        store.insert("a fact".into(), vec![], e.clone(), 5);
        store.record_attempt("goal".into(), "approach".into(), None, e, 6);
        assert_eq!(store.attempt_count(), 1);
        assert_eq!(store.valid_count(), 2); // both fact and attempt
    }

    #[test]
    fn record_outcome_via_store_view() {
        use piku_tools::embed_memory_tool::piku_runtime_types::MemoryStoreView;
        let mut store = MemoryStore::default();
        let e = make_embedding(1.0);
        let id = store.record_attempt("goal".into(), "approach".into(), None, e, 6);
        assert!(MemoryStoreView::record_outcome(
            &mut store, id, "success", None
        ));
        assert_eq!(store.entries[0].outcome, Some(Outcome::Success));
        // Invalid outcome string
        assert!(!MemoryStoreView::record_outcome(
            &mut store, id, "bogus", None
        ));
    }

    #[test]
    fn inspect_shows_attempt_fields() {
        use piku_tools::embed_memory_tool::piku_runtime_types::MemoryStoreView;
        let mut store = MemoryStore::default();
        let e = make_embedding(1.0);
        let root = store.record_attempt("fix bug".into(), "try A".into(), None, e.clone(), 6);
        store.record_outcome(root, Outcome::Failure, Some("broke it".into()));
        store.record_attempt("fix bug".into(), "try B".into(), Some(root), e, 6);

        let detail = store.inspect(root).unwrap();
        assert!(detail.contains("Type: Attempt"));
        assert!(detail.contains("Goal: fix bug"));
        assert!(detail.contains("Outcome: Failure"));
        assert!(detail.contains("Outcome detail: broke it"));
        assert!(detail.contains("Children:"));
    }

    // --- Attempt extraction parser tests ---

    #[test]
    fn parse_attempt_lines_basic() {
        let response = "\
- [importance:7] Some fact {tags: test}
- [attempt] goal: fix UTF-8 rendering | approach: byte slicing | outcome: failure | detail: panics on multi-byte
- [attempt] goal: fix UTF-8 rendering | approach: char iterator | outcome: success | detail: handles all unicode";
        let attempts = super::parse_attempt_lines(response);
        assert_eq!(attempts.len(), 2);
        assert_eq!(attempts[0].goal, "fix UTF-8 rendering");
        assert_eq!(attempts[0].approach, "byte slicing");
        assert_eq!(attempts[0].outcome, Outcome::Failure);
        assert_eq!(attempts[0].detail, "panics on multi-byte");
        assert_eq!(attempts[1].outcome, Outcome::Success);
    }

    #[test]
    fn parse_attempt_lines_skips_pending() {
        let response =
            "- [attempt] goal: test | approach: thing | outcome: pending | detail: still going";
        let attempts = super::parse_attempt_lines(response);
        assert!(attempts.is_empty()); // pending outcomes are skipped
    }

    #[test]
    fn parse_attempt_lines_skips_incomplete() {
        let response = "- [attempt] goal: test | approach: thing";
        let attempts = super::parse_attempt_lines(response);
        assert!(attempts.is_empty()); // missing outcome field
    }

    #[test]
    fn parse_attempt_lines_none_response() {
        let response = "NONE";
        let attempts = super::parse_attempt_lines(response);
        assert!(attempts.is_empty());
    }

    #[test]
    fn parse_mixed_facts_and_attempts() {
        let response = "\
- [importance:8] User prefers explicit error handling {tags: rust, style}
- [attempt] goal: reduce compile time | approach: split into subcrates | outcome: success | detail: 40% faster builds
- [importance:5] Project uses tokio runtime {tags: rust, async}";
        let facts = super::parse_extraction_response(response);
        let attempts = super::parse_attempt_lines(response);
        assert_eq!(facts.len(), 2);
        assert_eq!(attempts.len(), 1);
        assert_eq!(attempts[0].goal, "reduce compile time");
    }

    // --- Tree-aware eviction tests ---

    #[test]
    fn evict_stale_wont_orphan_children() {
        let mut store = MemoryStore::default();
        let e = make_embedding(1.0);
        // Parent attempt (old, low importance -- normally evictable)
        let root = store.record_attempt("goal".into(), "root".into(), None, e.clone(), 3);
        store.entries[0].last_accessed = 0;
        store.entries[0].created_at = 0;
        // Child attempt (recent)
        store.record_attempt("goal".into(), "child".into(), Some(root), e.clone(), 3);

        let evicted = store.evict_stale(30);
        // Root should NOT be evicted because it has a valid child
        assert_eq!(evicted, 0);
        assert!(store.entries[0].is_valid);
    }

    #[test]
    fn evict_stale_keeps_unresolved_tree() {
        let mut store = MemoryStore::default();
        let e = make_embedding(1.0);
        // Unresolved tree: root with one failed child, no successes
        let root = store.record_attempt("goal".into(), "root".into(), None, e.clone(), 3);
        store.entries[0].last_accessed = 0;
        store.entries[0].created_at = 0;
        let child = store.record_attempt("goal".into(), "child".into(), Some(root), e.clone(), 3);
        store.record_outcome(child, Outcome::Failure, Some("didn't work".into()));
        store.entries[1].last_accessed = 0;
        store.entries[1].created_at = 0;

        let evicted = store.evict_stale(30);
        // Unresolved tree: no successes anywhere, should be kept
        assert_eq!(evicted, 0);
    }

    #[test]
    fn evict_stale_can_evict_resolved_leaf() {
        let mut store = MemoryStore::default();
        let e = make_embedding(1.0);
        // Resolved tree: root -> child (success)
        let root = store.record_attempt("goal".into(), "root".into(), None, e.clone(), 3);
        let child = store.record_attempt("goal".into(), "child".into(), Some(root), e.clone(), 3);
        store.record_outcome(child, Outcome::Success, Some("worked".into()));
        // Make both old
        for entry in &mut store.entries {
            entry.last_accessed = 0;
            entry.created_at = 0;
        }

        let evicted = store.evict_stale(30);
        // Child (leaf with no children) in resolved tree CAN be evicted.
        // Root still has a valid child so it's protected until child goes first.
        // But the child has no children and tree is resolved, so child is evictable.
        assert!(evicted >= 1);
    }

    #[test]
    fn evict_weak_wont_orphan_children() {
        let mut store = MemoryStore::default();
        let e = make_embedding(1.0);
        let root = store.record_attempt("goal".into(), "root".into(), None, e.clone(), 3);
        store.entries[0].last_accessed = 0;
        store.entries[0].access_count = 0;
        store.record_attempt("goal".into(), "child".into(), Some(root), e.clone(), 3);

        let evicted = store.evict_weak(0.05);
        // Root protected by child
        assert!(store.entries[0].is_valid);
        assert_eq!(evicted, 0);
    }

    #[test]
    fn tree_is_resolved_checks_descendants() {
        let mut store = MemoryStore::default();
        let e = make_embedding(1.0);
        let root = store.record_attempt("g".into(), "root".into(), None, e.clone(), 6);
        let child = store.record_attempt("g".into(), "child".into(), Some(root), e.clone(), 6);
        let grandchild =
            store.record_attempt("g".into(), "gc".into(), Some(child), e.clone(), 6);
        store.record_outcome(grandchild, Outcome::Success, None);

        assert!(store.tree_is_resolved(root));
        assert!(store.tree_is_resolved(child));
    }

    #[test]
    fn tree_is_resolved_false_when_all_fail() {
        let mut store = MemoryStore::default();
        let e = make_embedding(1.0);
        let root = store.record_attempt("g".into(), "root".into(), None, e.clone(), 6);
        let child = store.record_attempt("g".into(), "child".into(), Some(root), e.clone(), 6);
        store.record_outcome(child, Outcome::Failure, None);

        assert!(!store.tree_is_resolved(root));
    }

    #[test]
    fn facts_still_evict_normally_with_tree_logic() {
        // Ensure tree-aware eviction doesn't accidentally protect facts
        let mut store = MemoryStore::default();
        let e = make_embedding(1.0);
        store.insert("old fact".into(), vec![], e.clone(), 3);
        store.entries[0].last_accessed = 0;
        store.entries[0].created_at = 0;

        let evicted = store.evict_stale(30);
        assert_eq!(evicted, 1);
        assert!(!store.entries[0].is_valid);
    }
}
