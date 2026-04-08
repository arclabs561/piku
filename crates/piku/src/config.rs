/// Unified piku configuration loaded from file + env + CLI overrides.
///
/// Precedence: CLI flags > env vars > settings file > defaults.
///
/// Config file: `~/.config/piku/settings.json` (user-global).
/// Project-local overrides: `.piku/settings.json` (merged on top).
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// Settings file schema
// ---------------------------------------------------------------------------

/// On-disk settings file shape (`settings.json`).
#[derive(Debug, Clone, Default, serde::Deserialize)]
pub struct SettingsFile {
    /// Default provider name (openrouter, anthropic, groq, ollama, custom).
    pub provider: Option<String>,
    /// Default model override.
    pub model: Option<String>,
    /// Maximum turns per agent turn (overrides the runtime default).
    pub max_turns: Option<u32>,
}

// ---------------------------------------------------------------------------
// Resolved config
// ---------------------------------------------------------------------------

/// Fully resolved configuration after merging file + env + CLI.
#[derive(Debug, Clone)]
pub struct PikuConfig {
    /// Provider name override (from CLI > file).
    pub provider: Option<String>,
    /// Model name override (from CLI > file).
    pub model: Option<String>,
    /// Max turns per agent turn.
    pub max_turns: Option<u32>,
    /// Path to user-global config dir (`~/.config/piku/`).
    pub config_dir: PathBuf,
}

impl PikuConfig {
    /// Load config: global settings file, then project-local, then CLI overrides.
    #[must_use]
    pub fn load(
        cli_provider: Option<&str>,
        cli_model: Option<&str>,
        project_dir: Option<&Path>,
    ) -> Self {
        let config_dir = global_config_dir();

        // Layer 1: global settings file
        let global_path = config_dir.join("settings.json");
        let mut settings = load_settings_file(&global_path);

        // Layer 2: project-local settings file (merged on top)
        if let Some(proj) = project_dir {
            let project_path = proj.join(".piku").join("settings.json");
            let project_settings = load_settings_file(&project_path);
            merge_settings(&mut settings, &project_settings);
        }

        // Layer 3: CLI overrides (highest precedence)
        if let Some(p) = cli_provider {
            settings.provider = Some(p.to_string());
        }
        if let Some(m) = cli_model {
            settings.model = Some(m.to_string());
        }

        PikuConfig {
            provider: settings.provider,
            model: settings.model,
            max_turns: settings.max_turns,
            config_dir,
        }
    }

    /// Sessions directory.
    #[must_use]
    pub fn sessions_dir(&self) -> PathBuf {
        self.config_dir.join("sessions")
    }

    /// Traces directory.
    #[must_use]
    pub fn traces_dir(&self) -> PathBuf {
        self.config_dir.join("traces")
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn global_config_dir() -> PathBuf {
    let base = std::env::var("XDG_CONFIG_HOME").map_or_else(
        |_| {
            std::env::var("HOME").map_or_else(
                |_| PathBuf::from(".config"),
                |h| PathBuf::from(h).join(".config"),
            )
        },
        PathBuf::from,
    );
    base.join("piku")
}

fn load_settings_file(path: &Path) -> SettingsFile {
    match std::fs::read_to_string(path) {
        Ok(content) => serde_json::from_str(&content).unwrap_or_else(|e| {
            eprintln!("[piku] warning: failed to parse {}: {e}", path.display());
            SettingsFile::default()
        }),
        Err(_) => SettingsFile::default(),
    }
}

fn merge_settings(base: &mut SettingsFile, overlay: &SettingsFile) {
    if overlay.provider.is_some() {
        base.provider.clone_from(&overlay.provider);
    }
    if overlay.model.is_some() {
        base.model.clone_from(&overlay.model);
    }
    if overlay.max_turns.is_some() {
        base.max_turns = overlay.max_turns;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_empty_config() {
        let cfg = PikuConfig::load(None, None, None);
        assert!(cfg.provider.is_none());
        assert!(cfg.model.is_none());
    }

    #[test]
    fn cli_overrides_file() {
        let cfg = PikuConfig::load(Some("ollama"), Some("gemma4"), None);
        assert_eq!(cfg.provider.as_deref(), Some("ollama"));
        assert_eq!(cfg.model.as_deref(), Some("gemma4"));
    }

    #[test]
    fn project_overrides_global() {
        let dir = tempfile::tempdir().unwrap();
        let piku_dir = dir.path().join(".piku");
        std::fs::create_dir_all(&piku_dir).unwrap();
        std::fs::write(
            piku_dir.join("settings.json"),
            r#"{"provider": "groq", "model": "llama-3"}"#,
        )
        .unwrap();

        let cfg = PikuConfig::load(None, None, Some(dir.path()));
        assert_eq!(cfg.provider.as_deref(), Some("groq"));
        assert_eq!(cfg.model.as_deref(), Some("llama-3"));
    }

    #[test]
    fn cli_beats_project() {
        let dir = tempfile::tempdir().unwrap();
        let piku_dir = dir.path().join(".piku");
        std::fs::create_dir_all(&piku_dir).unwrap();
        std::fs::write(piku_dir.join("settings.json"), r#"{"provider": "groq"}"#).unwrap();

        let cfg = PikuConfig::load(Some("anthropic"), None, Some(dir.path()));
        assert_eq!(cfg.provider.as_deref(), Some("anthropic"));
    }

    #[test]
    fn sessions_and_traces_dirs() {
        let cfg = PikuConfig::load(None, None, None);
        assert!(cfg.sessions_dir().ends_with("piku/sessions"));
        assert!(cfg.traces_dir().ends_with("piku/traces"));
    }
}
