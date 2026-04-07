/// Agent memory — persistent MEMORY.md files across sessions.
///
/// Three scopes mirror Claude Code's `agentMemory.ts`:
///
/// - `User`    — `~/.config/piku/memory/MEMORY.md`  (personal, cross-project)
/// - `Project` — `.piku/memory/MEMORY.md`            (project-local, checked-in)
/// - `Local`   — `.piku/memory-local/MEMORY.md`      (project-local, gitignored)
///
/// Each scope holds a single `MEMORY.md` file.  The agent reads it at turn
/// start (injected into the system prompt dynamic section) and can write to it
/// via the `write_memory` tool.
///
/// # Memory types (from CC's taxonomy)
/// - **user**     — who the user is, expertise, preferences
/// - **feedback** — corrections and confirmations from the user
/// - **project**  — ongoing work, goals, bugs, incidents
/// - **reference** — stable facts that don't expire
///
/// The file format is plain Markdown.  Entries are headings + body.  There is
/// no schema enforcement — the LLM writes free-form Markdown and we just inject
/// it verbatim.  Max 200 lines / 25 KB before truncation.
use std::path::{Path, PathBuf};

const MEMORY_FILE: &str = "MEMORY.md";
const MAX_LINES: usize = 200;
const MAX_BYTES: usize = 25_000;

// ---------------------------------------------------------------------------
// Scope
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryScope {
    /// `~/.config/piku/memory/` — personal, cross-project.
    User,
    /// `<cwd>/.piku/memory/` — project-local, suitable for VCS.
    Project,
    /// `<cwd>/.piku/memory-local/` — project-local, gitignored.
    Local,
}

impl MemoryScope {
    pub fn dir(&self, cwd: &Path) -> PathBuf {
        match self {
            Self::User => {
                let base = std::env::var("XDG_CONFIG_HOME").map_or_else(
                    |_| {
                        std::env::var("HOME").map_or_else(
                            |_| PathBuf::from(".config"),
                            |h| PathBuf::from(h).join(".config"),
                        )
                    },
                    PathBuf::from,
                );
                base.join("piku").join("memory")
            }
            Self::Project => cwd.join(".piku").join("memory"),
            Self::Local => cwd.join(".piku").join("memory-local"),
        }
    }

    #[must_use]
    pub fn memory_file(&self, cwd: &Path) -> PathBuf {
        self.dir(cwd).join(MEMORY_FILE)
    }

    #[must_use]
    pub fn display_name(&self) -> &'static str {
        match self {
            Self::User => "User memory",
            Self::Project => "Project memory",
            Self::Local => "Local memory",
        }
    }
}

// ---------------------------------------------------------------------------
// Read
// ---------------------------------------------------------------------------

/// Read and truncate a MEMORY.md file.
/// Returns `None` if the file doesn't exist or is empty.
#[must_use]
pub fn read_memory(scope: MemoryScope, cwd: &Path) -> Option<String> {
    let path = scope.memory_file(cwd);
    let raw = std::fs::read_to_string(&path).ok()?.trim().to_string();
    if raw.is_empty() {
        return None;
    }
    Some(truncate_memory(&raw))
}

fn truncate_memory(raw: &str) -> String {
    let lines: Vec<&str> = raw.lines().collect();
    let was_line_truncated = lines.len() > MAX_LINES;
    let truncated_by_lines: String = if was_line_truncated {
        lines[..MAX_LINES].join("\n")
    } else {
        raw.to_string()
    };

    if truncated_by_lines.len() > MAX_BYTES {
        // Truncate at last newline before MAX_BYTES
        let cutoff = truncated_by_lines[..MAX_BYTES]
            .rfind('\n')
            .unwrap_or(MAX_BYTES);
        let mut out = truncated_by_lines[..cutoff].to_string();
        out.push_str("\n\n[memory truncated at byte limit]");
        out
    } else if was_line_truncated {
        truncated_by_lines + "\n\n[memory truncated at line limit]"
    } else {
        truncated_by_lines
    }
}

// ---------------------------------------------------------------------------
// Write
// ---------------------------------------------------------------------------

/// Append or replace a section in MEMORY.md.
///
/// `entry` should be a Markdown heading + body.  If `heading` is provided and
/// already exists in the file, the section is replaced.  Otherwise the entry
/// is appended.
pub fn write_memory(scope: MemoryScope, cwd: &Path, entry: &str) -> Result<(), String> {
    let dir = scope.dir(cwd);
    std::fs::create_dir_all(&dir).map_err(|e| format!("create_dir_all failed: {e}"))?;
    let path = scope.memory_file(cwd);

    // Read existing content
    let existing = std::fs::read_to_string(&path).unwrap_or_default();

    // Extract heading from entry (first `##` line if present)
    let heading = entry.lines().find(|l| l.starts_with("## ")).map(str::trim);

    let updated = if let Some(h) = heading {
        replace_or_append_section(&existing, h, entry)
    } else if existing.trim().is_empty() {
        entry.to_string()
    } else {
        let mut s = existing.trim_end().to_string();
        s.push_str("\n\n");
        s.push_str(entry);
        s
    };

    std::fs::write(&path, updated).map_err(|e| format!("write failed: {e}"))
}

fn replace_or_append_section(existing: &str, heading: &str, entry: &str) -> String {
    // Find the heading in the existing file
    let search = {
        let mut s = heading.to_string();
        s.push('\n');
        s
    };
    if let Some(start) = existing.find(&search) {
        // Find the next `## ` heading after start, or end of file
        let after = &existing[start + search.len()..];
        let end = after
            .find("\n## ")
            .map_or(existing.len(), |p| start + search.len() + p + 1); // keep the newline before next heading
        format!("{}{}\n{}", &existing[..start], entry, &existing[end..])
    } else {
        // Append
        if existing.trim().is_empty() {
            entry.to_string()
        } else {
            format!("{}\n\n{}", existing.trim_end(), entry)
        }
    }
}

// ---------------------------------------------------------------------------
// System prompt injection
// ---------------------------------------------------------------------------

/// Build the memory section to inject into the dynamic system prompt.
/// Reads all three scopes and concatenates non-empty ones.
#[must_use]
pub fn build_memory_prompt(cwd: &Path) -> Option<String> {
    let scopes = [MemoryScope::User, MemoryScope::Project, MemoryScope::Local];
    let mut sections: Vec<String> = Vec::new();

    for scope in &scopes {
        if let Some(content) = read_memory(*scope, cwd) {
            sections.push(format!(
                "## {} ({})\n\n{}",
                scope.display_name(),
                scope.memory_file(cwd).display(),
                content
            ));
        }
    }

    if sections.is_empty() {
        None
    } else {
        Some(format!(
            "# Memory\n\nThe following is your persistent memory from previous sessions. \
             Use it to inform your work. Update it when you learn something worth remembering.\n\n{}",
            sections.join("\n\n---\n\n")
        ))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    fn tempdir() -> TempDir {
        tempfile::tempdir().unwrap()
    }

    #[test]
    fn write_and_read_project_memory() {
        let dir = tempdir();
        let cwd = dir.path();
        write_memory(MemoryScope::Project, cwd, "## Test\n\nsome content").unwrap();
        let content = read_memory(MemoryScope::Project, cwd).unwrap();
        assert!(content.contains("some content"));
    }

    #[test]
    fn replace_section() {
        let dir = tempdir();
        let cwd = dir.path();
        write_memory(MemoryScope::Project, cwd, "## Feedback\n\nold content").unwrap();
        write_memory(MemoryScope::Project, cwd, "## Feedback\n\nnew content").unwrap();
        let content = read_memory(MemoryScope::Project, cwd).unwrap();
        assert!(content.contains("new content"));
        assert!(!content.contains("old content"));
    }

    #[test]
    fn truncate_at_line_limit() {
        let big: String = (0..300)
            .map(|i| {
                let mut s = "line ".to_string();
                s.push_str(&i.to_string());
                s.push('\n');
                s
            })
            .collect();
        let truncated = truncate_memory(&big);
        let line_count = truncated.lines().count();
        assert!(line_count <= MAX_LINES + 2); // +2 for truncation notice
        assert!(truncated.contains("truncated"));
    }

    #[test]
    fn empty_file_returns_none() {
        let dir = tempdir();
        let cwd = dir.path();
        // File doesn't exist
        assert!(read_memory(MemoryScope::Project, cwd).is_none());
        // Empty file
        let path = MemoryScope::Project.memory_file(cwd);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, "").unwrap();
        assert!(read_memory(MemoryScope::Project, cwd).is_none());
    }

    #[test]
    fn build_memory_prompt_none_when_empty() {
        let dir = tempdir();
        assert!(build_memory_prompt(dir.path()).is_none());
    }

    #[test]
    fn build_memory_prompt_some_when_content() {
        let dir = tempdir();
        write_memory(MemoryScope::Project, dir.path(), "## Note\n\nhello").unwrap();
        let prompt = build_memory_prompt(dir.path()).unwrap();
        assert!(prompt.contains("hello"));
        assert!(prompt.contains("Project memory"));
    }
}
