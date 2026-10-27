//! Prompt history: load, append, and Up/Down navigation.
//!
//! The file lives at `${XDG_DATA_HOME:-$HOME/.local/share}/piku/history`, one
//! entry per line. Consecutive duplicates are dropped. The prefix came from
//! the standalone wish shell; its file is still read when the piku one does not
//! exist yet, so existing history survives the move and new lines go to piku.

use std::fs::{self, OpenOptions};
use std::io::{Read, Write};
use std::path::PathBuf;

/// Keep startup bounded even if the history file grew without limit.
const MAX_ENTRIES: usize = 2000;
const MAX_BYTES: u64 = 4 * 1024 * 1024;

pub struct History {
    entries: Vec<String>,
    /// One past the last entry: the "editing a new line" position.
    index: usize,
    /// The line as it was before the first Up, restored by Down.
    draft: String,
    path: Option<PathBuf>,
}

impl History {
    pub fn load() -> Self {
        let path = history_path("piku/history");
        let entries = match path.as_deref().filter(|path| path.exists()) {
            Some(path) => read_entries(path),
            None => history_path("wish/history")
                .as_deref()
                .map_or_else(Vec::new, read_entries),
        };
        let index = entries.len();
        Self {
            entries,
            index,
            draft: String::new(),
            path,
        }
    }

    /// Leaves history browsing and forgets the draft. Call after any line is
    /// submitted or the buffer is cleared.
    pub fn reset(&mut self) {
        self.index = self.entries.len();
        self.draft.clear();
    }

    /// Recalls the previous entry, remembering `current` as the draft.
    pub fn previous(&mut self, current: &str) -> Option<String> {
        if self.entries.is_empty() {
            return None;
        }
        if self.index == self.entries.len() {
            self.draft = current.to_string();
        }
        if self.index > 0 {
            self.index -= 1;
        }
        self.entries.get(self.index).cloned()
    }

    /// Moves towards newer entries and finally back to the draft.
    pub fn next(&mut self) -> Option<String> {
        if self.entries.is_empty() || self.index >= self.entries.len() {
            return None;
        }
        self.index += 1;
        if self.index == self.entries.len() {
            Some(self.draft.clone())
        } else {
            self.entries.get(self.index).cloned()
        }
    }

    /// Records a submitted line in memory and on disk. Failures are reported
    /// to the caller but never abort the session.
    pub fn append(&mut self, line: &str) -> Result<(), String> {
        if line.is_empty() {
            return Ok(());
        }
        let duplicate = self
            .entries
            .last()
            .is_some_and(|last| last.as_str() == line);
        if !duplicate {
            self.entries.push(line.to_string());
        }
        self.reset();
        if duplicate {
            return Ok(());
        }
        let Some(path) = &self.path else {
            return Ok(());
        };
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .map_err(|error| format!("create {}: {error}", parent.display()))?;
        }
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .map_err(|error| format!("open {}: {error}", path.display()))?;
        writeln!(file, "{line}").map_err(|error| format!("write {}: {error}", path.display()))
    }
}

fn history_path(suffix: &str) -> Option<PathBuf> {
    let base = match std::env::var_os("XDG_DATA_HOME") {
        Some(dir) if !dir.is_empty() => PathBuf::from(dir),
        _ => {
            let home = std::env::var_os("HOME").filter(|home| !home.is_empty())?;
            PathBuf::from(home).join(".local/share")
        }
    };
    Some(base.join(suffix))
}

fn read_entries(path: &std::path::Path) -> Vec<String> {
    let Ok(file) = fs::File::open(path) else {
        return Vec::new();
    };
    let mut bytes = Vec::new();
    if file.take(MAX_BYTES).read_to_end(&mut bytes).is_err() {
        return Vec::new();
    }
    let text = String::from_utf8_lossy(&bytes);
    let mut entries: Vec<String> = Vec::new();
    for line in text.lines() {
        if line.is_empty() || entries.last().is_some_and(|last| last.as_str() == line) {
            continue;
        }
        entries.push(line.to_string());
    }
    if entries.len() > MAX_ENTRIES {
        entries.drain(..entries.len() - MAX_ENTRIES);
    }
    entries
}
