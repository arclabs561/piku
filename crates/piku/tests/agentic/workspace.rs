use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

pub struct WorkspaceObserver {
    root: PathBuf,
    baseline: HashMap<PathBuf, (SystemTime, u64)>,
}

impl WorkspaceObserver {
    pub fn new(root: PathBuf) -> Self {
        let mut ws = Self {
            root,
            baseline: HashMap::new(),
        };
        ws.checkpoint();
        ws
    }

    pub fn checkpoint(&mut self) {
        self.baseline = self.scan_files();
    }

    pub fn diff_since_checkpoint(&self) -> WorkspaceDiff {
        let current = self.scan_files();
        WorkspaceDiff {
            created: current
                .keys()
                .filter(|k| !self.baseline.contains_key(*k))
                .cloned()
                .collect(),
            modified: current
                .iter()
                .filter(|(k, (mtime, size))| {
                    self.baseline
                        .get(*k)
                        .map_or(false, |(bt, bs)| mtime != bt || size != bs)
                })
                .map(|(k, _)| k.clone())
                .collect(),
            deleted: self
                .baseline
                .keys()
                .filter(|k| !current.contains_key(*k))
                .cloned()
                .collect(),
        }
    }

    #[allow(dead_code)]
    pub fn read_file(&self, relative: &Path) -> Option<String> {
        std::fs::read_to_string(self.root.join(relative)).ok()
    }

    fn scan_files(&self) -> HashMap<PathBuf, (SystemTime, u64)> {
        let mut map = HashMap::new();
        self.scan_dir(&self.root, &mut map);
        map
    }

    fn scan_dir(&self, dir: &Path, map: &mut HashMap<PathBuf, (SystemTime, u64)>) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path
                .file_name()
                .and_then(|n| n.to_str())
                .map_or(false, |n| n.starts_with('.'))
            {
                continue;
            }
            if path.is_dir() {
                self.scan_dir(&path, map);
            } else if let Ok(meta) = path.metadata() {
                let mtime = meta.modified().unwrap_or(SystemTime::UNIX_EPOCH);
                let relative = path.strip_prefix(&self.root).unwrap_or(&path).to_path_buf();
                map.insert(relative, (mtime, meta.len()));
            }
        }
    }
}

#[derive(Debug)]
pub struct WorkspaceDiff {
    pub created: Vec<PathBuf>,
    pub modified: Vec<PathBuf>,
    pub deleted: Vec<PathBuf>,
}

impl WorkspaceDiff {
    pub fn is_empty(&self) -> bool {
        self.created.is_empty() && self.modified.is_empty() && self.deleted.is_empty()
    }

    pub fn summary(&self) -> String {
        if self.is_empty() {
            return "no changes".to_string();
        }
        let mut parts = Vec::new();
        if !self.created.is_empty() {
            let files: Vec<String> = self
                .created
                .iter()
                .map(|p| p.display().to_string())
                .collect();
            parts.push(format!("created: {}", files.join(", ")));
        }
        if !self.modified.is_empty() {
            let files: Vec<String> = self
                .modified
                .iter()
                .map(|p| p.display().to_string())
                .collect();
            parts.push(format!("modified: {}", files.join(", ")));
        }
        if !self.deleted.is_empty() {
            let files: Vec<String> = self
                .deleted
                .iter()
                .map(|p| p.display().to_string())
                .collect();
            parts.push(format!("deleted: {}", files.join(", ")));
        }
        parts.join("; ")
    }
}
