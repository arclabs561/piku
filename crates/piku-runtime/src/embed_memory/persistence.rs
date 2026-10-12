use super::{EmbedConfig, MemoryStore};
use std::path::{Path, PathBuf};

// Resolve once per operation so aliases use the target's lock and replacement
// path. In particular, rename must never replace the caller's symlink itself.
fn resolve_store_path(path: &Path) -> Result<PathBuf, String> {
    match std::fs::symlink_metadata(path) {
        Ok(_) => {
            let resolved = path
                .canonicalize()
                .map_err(|e| format!("resolve memory store path failed: {e}"))?;
            if !std::fs::metadata(&resolved)
                .map_err(|e| format!("inspect memory store target failed: {e}"))?
                .is_file()
            {
                return Err("memory store target is not a regular file".into());
            }
            Ok(resolved)
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            let parent = path
                .parent()
                .filter(|p| !p.as_os_str().is_empty())
                .unwrap_or(Path::new("."));
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("create memory directory failed: {e}"))?;
            let parent = parent
                .canonicalize()
                .map_err(|e| format!("resolve memory directory failed: {e}"))?;
            let name = path
                .file_name()
                .ok_or("memory store path has no filename")?;
            Ok(parent.join(name))
        }
        Err(error) => Err(format!("inspect memory store path failed: {error}")),
    }
}

struct MemoryLock {
    file: std::fs::File,
}

impl Drop for MemoryLock {
    fn drop(&mut self) {
        // Close alone keeps the lock alive while a concurrent fork still has
        // its inherited descriptor open. Release the shared description's lock
        // explicitly on success, error, and unwind, before closing our handle.
        if let Err(error) = self.file.unlock() {
            eprintln!("[piku] memory lock release failed: {error}");
        }
    }
}

impl MemoryLock {
    fn acquire(path: &Path) -> Result<Self, String> {
        let mut lock_name = path.as_os_str().to_os_string();
        lock_name.push(".lock");
        let mut options = std::fs::OpenOptions::new();
        options.read(true).write(true).create(true).truncate(false);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        // Keep this inode stable across atomic replacement of the data file.
        let lock = options
            .open(lock_name)
            .map_err(|e| format!("open memory lock failed: {e}"))?;
        lock.try_lock()
            .map_err(|e| format!("memory store busy or lock unavailable; retry: {e}"))?;
        Ok(Self { file: lock })
    }
}

impl MemoryStore {
    /// Reload and mutate under a process-shared lock. The closure must not
    /// perform network I/O. Contention fails promptly so an async caller never
    /// blocks its executor waiting for another writer.
    pub fn transaction<T>(
        path: &Path,
        mutate: impl FnOnce(&mut Self) -> Result<T, String>,
    ) -> Result<T, String> {
        Self::transaction_resolved(&resolve_store_path(path)?, mutate)
    }

    fn transaction_resolved<T>(
        path: &Path,
        mutate: impl FnOnce(&mut Self) -> Result<T, String>,
    ) -> Result<T, String> {
        let _lock = MemoryLock::acquire(path)?;
        let mut store = Self::try_load(path)?;
        let before =
            serde_json::to_vec(&store).map_err(|e| format!("serialize memories failed: {e}"))?;
        let result = mutate(&mut store)?;
        if serde_json::to_vec(&store).map_err(|e| format!("serialize memories failed: {e}"))?
            != before
        {
            store.save(path)?;
        }
        Ok(result)
    }
}

/// Extract and persist memories for compaction and session shutdown.
pub async fn extract_and_persist(
    conversation: &str,
    provider: &dyn piku_api::Provider,
    model: &str,
    path: &Path,
    embed_config: &EmbedConfig,
) -> Result<usize, String> {
    let path = resolve_store_path(path)?;
    // Extraction is independent of store state. Keep its candidates while
    // conflict resolution is retried against a more recent snapshot.
    let Some(response) = super::extract_raw(conversation, provider, model).await else {
        return Ok(0);
    };
    for _ in 0..3 {
        let mut store = MemoryStore::try_load(&path)?;
        let before = serde_json::to_vec(&store).map_err(|e| e.to_string())?;
        let count =
            super::apply_extraction(&response, provider, model, &mut store, embed_config).await;
        let committed = MemoryStore::transaction_resolved(&path, |fresh| {
            if serde_json::to_vec(fresh).map_err(|e| e.to_string())? != before {
                return Ok(false);
            }
            if count > 0 {
                *fresh = store;
            }
            Ok(true)
        })?;
        if committed {
            return Ok(count);
        }
    }
    Err("memory store changed during extraction three times; extraction not persisted".into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transaction_serializes_processes_and_reloads_after_release() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("memories.json");
        let child = |expect_busy: bool| {
            let status = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "embed_memory::persistence::tests::process_contender",
                    "--nocapture",
                ])
                .env("PIKU_MEMORY_LOCK_TEST_PATH", &path)
                .env(
                    "PIKU_MEMORY_LOCK_EXPECT_BUSY",
                    if expect_busy { "1" } else { "0" },
                )
                .status()
                .unwrap();
            assert!(status.success(), "contender failed: {status}");
        };
        MemoryStore::transaction(&path, |store| {
            store.insert("parent".into(), vec![], vec![1.0], 6);
            child(true);
            Ok(())
        })
        .unwrap();
        child(false);
        let store = MemoryStore::try_load(&path).unwrap();
        assert_eq!(store.entries.len(), 2);
        assert_eq!(store.entries[0].content, "parent");
        assert_eq!(store.entries[1].content, "child");
        assert_eq!(store.entries[1].id, 1);
    }

    #[cfg(unix)]
    #[test]
    fn alias_and_target_share_process_lock_and_preserve_symlink() {
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("target.json");
        let alias = dir.path().join("alias.json");
        MemoryStore::transaction(&target, |store| {
            store.insert("initial".into(), vec![], vec![1.0], 6);
            Ok(())
        })
        .unwrap();
        std::os::unix::fs::symlink("target.json", &alias).unwrap();
        let child = |expect_busy: bool| {
            let status = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "embed_memory::persistence::tests::process_contender",
                    "--nocapture",
                ])
                .env("PIKU_MEMORY_LOCK_TEST_PATH", &alias)
                .env(
                    "PIKU_MEMORY_LOCK_EXPECT_BUSY",
                    if expect_busy { "1" } else { "0" },
                )
                .status()
                .unwrap();
            assert!(status.success(), "alias contender failed: {status}");
        };
        MemoryStore::transaction(&target, |store| {
            store.insert("parent".into(), vec![], vec![1.0], 6);
            child(true);
            Ok(())
        })
        .unwrap();
        child(false);
        assert!(std::fs::symlink_metadata(&alias)
            .unwrap()
            .file_type()
            .is_symlink());
        let store = MemoryStore::try_load(&target).unwrap();
        assert_eq!(store.entries.len(), 3);
        assert_eq!(store.entries[2].content, "child");
        assert_eq!(store.entries[2].id, 2);
        assert_eq!(
            std::fs::read(&alias).unwrap(),
            std::fs::read(&target).unwrap()
        );
    }

    #[cfg(unix)]
    #[test]
    fn invalid_alias_targets_fail_without_mutation() {
        let dir = tempfile::tempdir().unwrap();
        for target in [dir.path().join("missing.json"), dir.path().to_path_buf()] {
            let alias = dir.path().join(if target.is_dir() {
                "directory-link"
            } else {
                "dangling-link"
            });
            std::os::unix::fs::symlink(&target, &alias).unwrap();
            let mut called = false;
            let result = MemoryStore::transaction(&alias, |_| {
                called = true;
                Ok(())
            });
            assert!(result.is_err(), "invalid alias target must fail");
            assert!(!called);
            assert!(std::fs::symlink_metadata(&alias)
                .unwrap()
                .file_type()
                .is_symlink());
        }
    }

    #[test]
    fn process_contender() {
        let Some(path) = std::env::var_os("PIKU_MEMORY_LOCK_TEST_PATH") else {
            return;
        };
        let mut called = false;
        let result = MemoryStore::transaction(Path::new(&path), |store| {
            called = true;
            store.insert("child".into(), vec![], vec![1.0], 6);
            Ok(())
        });
        if std::env::var("PIKU_MEMORY_LOCK_EXPECT_BUSY").unwrap() == "1" {
            assert!(result.unwrap_err().contains("busy"));
            assert!(!called, "contender must not mutate while the lock is held");
        } else {
            result.unwrap();
            assert!(called);
        }
    }

    #[test]
    fn lock_release_does_not_wait_for_duplicated_handle() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("memories.json");
        let lock = MemoryLock::acquire(&path).unwrap();
        // A duplicated handle shares the open file description, just as the
        // descriptor inherited between fork and exec does on Unix.
        let inherited = lock.file.try_clone().unwrap();
        drop(lock);
        let next = MemoryLock::acquire(&path);
        assert!(
            next.is_ok(),
            "transaction lock must release before inherited handles close"
        );
        drop(inherited);
    }

    #[test]
    fn failed_mutation_preserves_previous_file_and_releases_lock() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("memories.json");
        MemoryStore::transaction(&path, |store| {
            store.insert("acknowledged".into(), vec![], vec![1.0], 6);
            Ok(())
        })
        .unwrap();
        let before = std::fs::read(&path).unwrap();
        let result = MemoryStore::transaction(&path, |store| {
            store.entries.clear();
            Err::<(), _>("mutation failed".into())
        });
        assert_eq!(result.unwrap_err(), "mutation failed");
        assert_eq!(std::fs::read(&path).unwrap(), before);
        MemoryStore::transaction(&path, |_| Ok(())).unwrap();
        std::fs::write(&path, "corrupt").unwrap();
        assert!(MemoryStore::transaction(&path, |_| Ok(())).is_err());
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "corrupt");
    }
}
