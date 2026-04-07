#[derive(Debug, Clone)]
pub struct TurnSummary {
    pub turn: usize,
    pub action_desc: String,
    pub observations: Vec<String>,
    pub bugs: Vec<String>,
    pub prompt_visible: bool,
    pub cursor_visible: bool,
    pub workspace_changes: String,
}

pub struct ConversationMemory {
    pub entries: Vec<TurnSummary>,
}

impl ConversationMemory {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    pub fn push(&mut self, summary: TurnSummary) {
        self.entries.push(summary);
    }

    pub fn format_for_llm(&self) -> String {
        if self.entries.is_empty() {
            return String::new();
        }
        let mut out = String::from("PRIOR TURNS:\n");
        for e in &self.entries {
            out.push_str(&format!(
                "  Turn {}: {} | prompt={} cursor={} | {} obs, {} bugs",
                e.turn,
                e.action_desc,
                if e.prompt_visible { "ok" } else { "MISSING" },
                if e.cursor_visible { "ok" } else { "HIDDEN" },
                e.observations.len(),
                e.bugs.len(),
            ));
            if !e.workspace_changes.is_empty() && e.workspace_changes != "no changes" {
                out.push_str(&format!(" | fs: {}", e.workspace_changes));
            }
            out.push('\n');
        }
        out
    }
}
