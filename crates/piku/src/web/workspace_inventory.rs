use std::collections::BTreeMap;
use std::fs::File;
use std::io::Read as _;
use std::path::{Path, PathBuf};

use piku_runtime::{RunContentChange, RunEffectCategory, RunToolEffect};
use sha2::{Digest, Sha256};

const MAX_FILES: usize = 20_000;
const MAX_HASHED_BYTES: u64 = 512 * 1024 * 1024;
const EXCLUDED_ROOTS: &[&str] = &[".git", "node_modules", "target"];

#[derive(Debug, Clone, PartialEq, Eq)]
struct FileIdentity {
    bytes: u64,
    sha256: [u8; 32],
}

#[derive(Debug)]
pub(super) struct WorkspaceSnapshot {
    files: BTreeMap<PathBuf, FileIdentity>,
    incomplete_reasons: Vec<String>,
}

pub(super) fn capture(root: &Path) -> WorkspaceSnapshot {
    let mut files = BTreeMap::new();
    let mut incomplete_reasons = Vec::new();
    let mut hashed_bytes = 0_u64;
    let mut entries = 0_usize;

    for entry in walkdir::WalkDir::new(root)
        .follow_links(false)
        .into_iter()
        .filter_entry(|entry| {
            entry.depth() == 0
                || !entry.file_type().is_dir()
                || !EXCLUDED_ROOTS.contains(&entry.file_name().to_string_lossy().as_ref())
        })
    {
        let Ok(entry) = entry else {
            incomplete_reasons.push("one or more workspace entries could not be read".into());
            continue;
        };
        if entry.depth() == 0 || !entry.file_type().is_file() {
            continue;
        }
        entries += 1;
        if entries > MAX_FILES {
            incomplete_reasons.push(format!("workspace inventory exceeded {MAX_FILES} files"));
            break;
        }
        let Ok(relative) = entry.path().strip_prefix(root) else {
            incomplete_reasons.push("a workspace entry escaped the inventory root".into());
            continue;
        };
        let relative = relative.to_path_buf();
        let Ok(metadata) = entry.metadata() else {
            incomplete_reasons.push(format!("cannot inspect {}", relative.display()));
            continue;
        };
        let Some(next_bytes) = hashed_bytes.checked_add(metadata.len()) else {
            incomplete_reasons.push("workspace inventory byte count overflowed".into());
            break;
        };
        if next_bytes > MAX_HASHED_BYTES {
            incomplete_reasons.push(format!(
                "workspace inventory exceeded {MAX_HASHED_BYTES} hashed bytes"
            ));
            break;
        }
        match hash_file(entry.path()) {
            Ok(sha256) => {
                hashed_bytes = next_bytes;
                files.insert(
                    relative,
                    FileIdentity {
                        bytes: metadata.len(),
                        sha256,
                    },
                );
            }
            Err(()) => incomplete_reasons.push(format!("cannot hash {}", relative.display())),
        }
    }
    incomplete_reasons.sort();
    incomplete_reasons.dedup();
    WorkspaceSnapshot {
        files,
        incomplete_reasons,
    }
}

fn hash_file(path: &Path) -> Result<[u8; 32], ()> {
    let mut file = File::open(path).map_err(|_| ())?;
    let mut digest = Sha256::new();
    let mut buffer = vec![0_u8; 64 * 1024];
    loop {
        let read = file.read(&mut buffer).map_err(|_| ())?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    Ok(digest.finalize().into())
}

pub(super) fn diff(before: WorkspaceSnapshot, after: WorkspaceSnapshot) -> Vec<RunToolEffect> {
    let mut effects = Vec::new();
    for (path, identity) in &after.files {
        match before.files.get(path) {
            None => effects.push(RunToolEffect::FileWrite {
                path: path.clone(),
                content_change: RunContentChange::Created,
            }),
            Some(previous) if previous != identity => effects.push(RunToolEffect::FileWrite {
                path: path.clone(),
                content_change: RunContentChange::Modified,
            }),
            Some(_) => {}
        }
    }
    for path in before.files.keys() {
        if !after.files.contains_key(path) {
            effects.push(RunToolEffect::FileDelete { path: path.clone() });
        }
    }
    for reason in before
        .incomplete_reasons
        .into_iter()
        .chain(after.incomplete_reasons)
    {
        effects.push(RunToolEffect::Unattributed {
            category: RunEffectCategory::FileSystem,
            reason,
        });
    }
    effects
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn observes_created_modified_deleted_hidden_and_same_size_files() {
        let directory = tempfile::tempdir().unwrap();
        std::fs::write(directory.path().join("modified"), b"aaaa").unwrap();
        std::fs::write(directory.path().join("deleted"), b"gone").unwrap();
        std::fs::write(directory.path().join(".hidden"), b"before").unwrap();
        let before = capture(directory.path());

        std::fs::write(directory.path().join("modified"), b"bbbb").unwrap();
        std::fs::remove_file(directory.path().join("deleted")).unwrap();
        std::fs::write(directory.path().join(".hidden"), b"after!").unwrap();
        std::fs::write(directory.path().join("created"), b"new").unwrap();
        let effects = diff(before, capture(directory.path()));

        assert!(effects.contains(&RunToolEffect::FileWrite {
            path: "modified".into(),
            content_change: RunContentChange::Modified,
        }));
        assert!(effects.contains(&RunToolEffect::FileWrite {
            path: ".hidden".into(),
            content_change: RunContentChange::Modified,
        }));
        assert!(effects.contains(&RunToolEffect::FileWrite {
            path: "created".into(),
            content_change: RunContentChange::Created,
        }));
        assert!(effects.contains(&RunToolEffect::FileDelete {
            path: "deleted".into(),
        }));
    }

    #[cfg(unix)]
    #[test]
    fn does_not_follow_symlinks() {
        use std::os::unix::fs::symlink;

        let workspace = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        std::fs::write(outside.path().join("secret"), b"outside").unwrap();
        symlink(outside.path(), workspace.path().join("external")).unwrap();

        let snapshot = capture(workspace.path());
        assert!(snapshot.files.is_empty());
        assert!(snapshot.incomplete_reasons.is_empty());
    }

    #[test]
    fn excludes_vcs_and_build_roots_from_the_declared_authoring_scope() {
        let workspace = tempfile::tempdir().unwrap();
        for root in EXCLUDED_ROOTS {
            std::fs::create_dir(workspace.path().join(root)).unwrap();
            std::fs::write(workspace.path().join(root).join("ignored"), b"ignored").unwrap();
        }
        std::fs::write(workspace.path().join("observed"), b"observed").unwrap();

        let snapshot = capture(workspace.path());

        assert_eq!(snapshot.files.len(), 1);
        assert!(snapshot.files.contains_key(Path::new("observed")));
    }
}
