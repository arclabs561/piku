//! Explicit project permission review, independent of provider and hook trust.
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context};
use sha2::{Digest, Sha256};

use crate::cli::PermissionCommand;
use crate::config::SettingsFile;

const STORE: &str = "approved-permissions.json";

#[derive(serde::Serialize)]
struct Policy<'a> {
    schema: u32,
    allow: &'a [String],
    deny: &'a [String],
}

pub(crate) fn digest(settings: &SettingsFile) -> String {
    let bytes = serde_json::to_vec(&Policy {
        schema: 1,
        allow: &settings.allow,
        deny: &settings.deny,
    })
    .expect("permission policy serializes");
    format!("{:x}", Sha256::digest(bytes))
}

fn read_store(config_dir: &Path) -> anyhow::Result<BTreeMap<PathBuf, String>> {
    let path = config_dir.join(STORE);
    match std::fs::symlink_metadata(&path) {
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(BTreeMap::new()),
        Err(error) => return Err(error.into()),
        Ok(metadata) if !metadata.is_file() || metadata.file_type().is_symlink() => {
            bail!("permission receipt must be a regular file")
        }
        Ok(_) => {}
    }
    serde_json::from_slice(&std::fs::read(path)?).context("permission receipt is invalid")
}

pub(crate) fn approved(config_dir: &Path, project: &Path, settings: &SettingsFile) -> bool {
    let Ok(root) = project.canonicalize() else {
        return false;
    };
    match read_store(config_dir) {
        Ok(store) => store.get(&root) == Some(&digest(settings)),
        Err(error) => {
            tracing::warn!(%error, "project permission approval could not be read; project allow rules skipped");
            false
        }
    }
}

/// Run an operator-requested review or approval without loading any provider.
pub fn run(command: &PermissionCommand, project: &Path) -> anyhow::Result<()> {
    run_in(command, project, &crate::config::global_config_dir())
}

fn run_in(command: &PermissionCommand, project: &Path, config_dir: &Path) -> anyhow::Result<()> {
    let root = project.canonicalize().context("canonical project root")?;
    let path = root.join(".piku/settings.toml");
    let settings: SettingsFile = toml::from_str(
        &std::fs::read_to_string(&path).context("read project settings for permission review")?,
    )
    .map_err(|_| {
        anyhow::anyhow!(
            "project settings contain invalid TOML; repair syntax before reviewing permissions"
        )
    })?;
    let current = digest(&settings);
    match command {
        PermissionCommand::Review => {
            println!(
                "Root: {}\nDigest: {current}",
                crate::terminal_text(&root.display().to_string())
            );
            println!("Project allow (grants code execution under your OS identity):");
            for rule in &settings.allow {
                println!("  {}", serde_json::to_string(rule)?);
            }
            println!("Project deny (always applies):");
            for rule in &settings.deny {
                println!("  {}", serde_json::to_string(rule)?);
            }
            println!("Approved: {}", approved(config_dir, &root, &settings));
        }
        PermissionCommand::Approve {
            root: reviewed_root,
            digest: reviewed_digest,
        } => {
            if reviewed_root != &root || reviewed_digest != &current {
                bail!("root or permission rules changed; review this project again")
            }
            std::fs::create_dir_all(config_dir)?;
            let lock = config_dir.join(".permissions.lock");
            let _lock_file = std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&lock)
                .context(
                    "permission approval already in progress or lock requires operator recovery",
                )?;
            struct Lock(std::path::PathBuf);
            impl Drop for Lock {
                fn drop(&mut self) {
                    let _ = std::fs::remove_file(&self.0);
                }
            }
            let _lock = Lock(lock);
            let mut store = read_store(config_dir)?;
            store.insert(root, current);
            let mut nonce = [0_u8; 16];
            getrandom::fill(&mut nonce)
                .map_err(|error| anyhow::anyhow!("receipt nonce: {error}"))?;
            let temporary = config_dir.join(format!(
                ".permissions-{:032x}.tmp",
                u128::from_ne_bytes(nonce)
            ));
            let mut options = std::fs::OpenOptions::new();
            options.create_new(true).write(true);
            #[cfg(unix)]
            {
                use std::os::unix::fs::OpenOptionsExt;
                options.mode(0o600);
            }
            let mut file = options.open(&temporary)?;
            use std::io::Write;
            file.write_all(&serde_json::to_vec_pretty(&store)?)?;
            file.sync_all()?;
            // Recheck the destination before atomic replacement. A damaged or
            // symlinked receipt never becomes an empty replacement store.
            read_store(config_dir)?;
            std::fs::rename(temporary, config_dir.join(STORE))?;
            println!("Approved reviewed project permission rules.");
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> (tempfile::TempDir, tempfile::TempDir, SettingsFile) {
        let project = tempfile::tempdir().unwrap();
        let config = tempfile::tempdir().unwrap();
        std::fs::create_dir(project.path().join(".piku")).unwrap();
        std::fs::write(
            project.path().join(".piku/settings.toml"),
            "allow = [\"bash(cargo test)\"]\ndeny = [\"bash(rm *)\"]\n",
        )
        .unwrap();
        let settings =
            crate::config::load_settings_file(&project.path().join(".piku/settings.toml"));
        (project, config, settings)
    }

    fn approve(project: &Path, config: &Path, settings: &SettingsFile) -> anyhow::Result<()> {
        run_in(
            &PermissionCommand::Approve {
                root: project.canonicalize().unwrap(),
                digest: digest(settings),
            },
            project,
            config,
        )
    }

    #[test]
    fn approval_binds_root_and_only_permission_policy() {
        let (project, config, mut settings) = fixture();
        assert!(!approved(config.path(), project.path(), &settings));
        approve(project.path(), config.path(), &settings).unwrap();
        assert!(approved(config.path(), project.path(), &settings));
        settings.model = Some("different-model".into());
        settings.max_turns = Some(3);
        assert!(approved(config.path(), project.path(), &settings));
        std::fs::write(project.path().join("code.rs"), "unrelated edit").unwrap();
        assert!(approved(config.path(), project.path(), &settings));
        let second = tempfile::tempdir().unwrap();
        assert!(!approved(config.path(), second.path(), &settings));
        settings.allow.push("bash(cargo build)".into());
        assert!(!approved(config.path(), project.path(), &settings));
        settings.allow.pop();
        settings.deny.push("write_file".into());
        assert!(!approved(config.path(), project.path(), &settings));
    }

    #[test]
    fn stale_digest_wrong_root_and_damaged_receipt_never_grant() {
        let (project, config, settings) = fixture();
        let command = PermissionCommand::Approve {
            root: project.path().canonicalize().unwrap(),
            digest: "stale".into(),
        };
        assert!(run_in(&command, project.path(), config.path()).is_err());
        let command = PermissionCommand::Approve {
            root: config.path().to_path_buf(),
            digest: digest(&settings),
        };
        assert!(run_in(&command, project.path(), config.path()).is_err());
        std::fs::write(config.path().join(STORE), "broken receipt").unwrap();
        assert!(approve(project.path(), config.path(), &settings).is_err());
        assert_eq!(
            std::fs::read_to_string(config.path().join(STORE)).unwrap(),
            "broken receipt"
        );
        assert!(!approved(config.path(), project.path(), &settings));
    }

    #[cfg(unix)]
    #[test]
    fn canonical_alias_works_and_symlinked_receipt_is_refused() {
        let (project, config, settings) = fixture();
        let aliases = tempfile::tempdir().unwrap();
        let alias = aliases.path().join("alias");
        std::os::unix::fs::symlink(project.path(), &alias).unwrap();
        approve(&alias, config.path(), &settings).unwrap();
        assert!(approved(config.path(), &alias, &settings));

        let second = tempfile::tempdir().unwrap();
        let target = second.path().join("target");
        std::fs::write(&target, "owner data").unwrap();
        std::os::unix::fs::symlink(&target, second.path().join(STORE)).unwrap();
        assert!(approve(project.path(), second.path(), &settings).is_err());
        assert_eq!(std::fs::read_to_string(target).unwrap(), "owner data");
    }
}
