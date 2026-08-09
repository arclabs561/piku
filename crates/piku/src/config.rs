/// Unified Piku settings loaded from global and project files plus CLI overrides.
///
/// Precedence: CLI flags > project settings > global settings > defaults.
/// Provider-specific environment variables are resolved separately by the
/// runtime's provider resolver.
///
/// Config file: `$XDG_CONFIG_HOME/piku/settings.toml`, falling back to
/// `~/.config/piku/settings.toml` (user-global).
/// Project-local overrides: `.piku/settings.toml` (merged on top).
use std::path::{Path, PathBuf};

const PROVIDER_ENV_KEYS: &[&str] = &[
    "OPENROUTER_API_KEY",
    "OPENROUTER_BASE_URL",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_BASE_URL",
    "GROQ_API_KEY",
    "GROQ_BASE_URL",
    "OLLAMA_HOST",
    "PIKU_BASE_URL",
    "PIKU_API_KEY",
    "PIKU_MODEL",
];

// ---------------------------------------------------------------------------
// Provider-specific config block
// ---------------------------------------------------------------------------

/// Per-provider configuration block (e.g. `[openrouter]` in settings.toml).
/// Each provider uses a subset of these fields:
///   openrouter/anthropic/groq: `api_key`, `base_url`, model
///   ollama: host, model
///   custom: `base_url`, `api_key`, model
#[derive(Debug, Clone, Default, serde::Deserialize)]
pub struct ProviderConfig {
    pub api_key: Option<String>,
    pub base_url: Option<String>,
    pub model: Option<String>,
    pub host: Option<String>,
}

// ---------------------------------------------------------------------------
// Settings file schema
// ---------------------------------------------------------------------------

/// On-disk settings file shape (`settings.toml`).
#[derive(Debug, Clone, Default, serde::Deserialize)]
pub struct SettingsFile {
    /// Default provider name (openrouter, anthropic, groq, ollama, custom).
    pub provider: Option<String>,
    /// Default model override.
    pub model: Option<String>,
    /// Maximum turns per agent turn (overrides the runtime default).
    pub max_turns: Option<u32>,
    /// Tool names or patterns to auto-allow without prompting.
    /// Supports: exact match (`"bash"`), glob prefix (`"bash(git *)"`)
    /// matching the same syntax as hook `if` conditions.
    #[serde(default)]
    pub allow: Vec<String>,
    /// Tool names to always deny.
    #[serde(default)]
    pub deny: Vec<String>,
    /// Provider-specific config blocks.
    #[serde(default)]
    pub openrouter: Option<ProviderConfig>,
    #[serde(default)]
    pub anthropic: Option<ProviderConfig>,
    #[serde(default)]
    pub groq: Option<ProviderConfig>,
    #[serde(default)]
    pub ollama: Option<ProviderConfig>,
    #[serde(default)]
    pub custom: Option<ProviderConfig>,
}

// ---------------------------------------------------------------------------
// Resolved config
// ---------------------------------------------------------------------------

/// Fully resolved configuration after merging global/project files and CLI.
#[derive(Debug, Clone)]
pub struct PikuConfig {
    /// Provider name override (from CLI > file).
    pub provider: Option<String>,
    /// Model name override (from CLI > file).
    pub model: Option<String>,
    /// Max turns per agent turn.
    pub max_turns: Option<u32>,
    /// Tool names/patterns to auto-allow (global + project merged).
    pub allow: Vec<String>,
    /// Tool names to always deny (global + project merged).
    pub deny: Vec<String>,
    /// Per-provider config blocks (global + project merged).
    pub provider_configs: ProviderConfigMap,
    /// Resolved user-global config dir (`XDG_CONFIG_HOME` or `~/.config` fallback).
    pub config_dir: PathBuf,
}

/// Holds provider config blocks after merging layers.
#[derive(Debug, Clone, Default)]
pub struct ProviderConfigMap {
    pub openrouter: Option<ProviderConfig>,
    pub anthropic: Option<ProviderConfig>,
    pub groq: Option<ProviderConfig>,
    pub ollama: Option<ProviderConfig>,
    pub custom: Option<ProviderConfig>,
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
        let global_path = config_dir.join("settings.toml");
        let mut settings = load_settings_file(&global_path);

        // Layer 2: project-local settings file (merged on top)
        if let Some(proj) = project_dir {
            let project_path = proj.join(".piku").join("settings.toml");
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
            allow: settings.allow,
            deny: settings.deny,
            provider_configs: ProviderConfigMap {
                openrouter: settings.openrouter,
                anthropic: settings.anthropic,
                groq: settings.groq,
                ollama: settings.ollama,
                custom: settings.custom,
            },
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

    /// Durable semantic run records directory.
    #[must_use]
    pub fn runs_dir(&self) -> PathBuf {
        self.config_dir.join("runs")
    }

    /// Durable parent-child links for spawned agent runs.
    #[must_use]
    pub fn agent_links_dir(&self) -> PathBuf {
        self.config_dir.join("agent-links")
    }

    /// Check if a tool call is pre-allowed by the config allowlist.
    /// Returns `Some(true)` if allowed, `Some(false)` if denied, `None` if no rule matches.
    #[must_use]
    pub fn check_permission_rule(
        &self,
        tool_name: &str,
        params: &serde_json::Value,
    ) -> Option<bool> {
        // Deny rules take precedence.
        for pattern in &self.deny {
            if matches_tool_pattern(pattern, tool_name, params) {
                return Some(false);
            }
        }
        for pattern in &self.allow {
            if matches_tool_pattern(pattern, tool_name, params) {
                return Some(true);
            }
        }
        None
    }
}

/// Match a tool permission pattern against a tool name and its params.
/// Supports: exact tool name (`"bash"`), tool with arg glob (`"bash(git *)"`)
/// using the same syntax as hook `if` conditions.
#[must_use]
pub fn matches_tool_pattern(pattern: &str, tool_name: &str, params: &serde_json::Value) -> bool {
    // Parse `ToolName(glob)` syntax
    if let Some(paren_start) = pattern.find('(') {
        if let Some(paren_end) = pattern.rfind(')') {
            let pat_tool = &pattern[..paren_start];
            if pat_tool != tool_name {
                return false;
            }
            let glob = &pattern[paren_start + 1..paren_end];
            let primary_arg = match tool_name {
                "bash" => params.get("command").and_then(|v| v.as_str()),
                "read_file" | "write_file" | "edit_file" => {
                    params.get("path").and_then(|v| v.as_str())
                }
                "glob" | "grep" => params.get("pattern").and_then(|v| v.as_str()),
                _ => None,
            };
            return primary_arg.is_some_and(|arg| glob_match(glob, arg));
        }
    }
    // Exact tool name match
    pattern == tool_name
}

/// Simple glob: `*` at start, end, or both. `*` alone matches all.
fn glob_match(pattern: &str, value: &str) -> bool {
    if pattern == "*" {
        return true;
    }
    let starts = pattern.starts_with('*');
    let ends = pattern.ends_with('*');
    match (starts, ends) {
        (true, true) => {
            let inner = &pattern[1..pattern.len() - 1];
            value.contains(inner)
        }
        (false, true) => value.starts_with(&pattern[..pattern.len() - 1]),
        (true, false) => value.ends_with(&pattern[1..]),
        (false, false) => value == pattern,
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

/// Load only recognized provider variables from the nearest `.env` when the
/// launching environment did not already define them.
///
/// This deliberately does not source the complete file: unrelated project
/// secrets must not become ambient authority for Piku or its tools.
pub fn load_provider_dotenv(project_dir: Option<&Path>) -> Result<usize, dotenvy::Error> {
    let Some(path) = nearest_dotenv(project_dir) else {
        return Ok(0);
    };
    let updates = provider_dotenv_updates(&path, |key| std::env::var_os(key).is_some())?;
    let count = updates.len();
    for (key, value) in updates {
        std::env::set_var(key, value);
    }
    Ok(count)
}

fn nearest_dotenv(project_dir: Option<&Path>) -> Option<PathBuf> {
    project_dir?
        .ancestors()
        .map(|dir| dir.join(".env"))
        .find(|path| path.is_file())
}

fn provider_dotenv_updates(
    path: &Path,
    is_set: impl Fn(&str) -> bool,
) -> Result<Vec<(String, String)>, dotenvy::Error> {
    let mut updates = Vec::new();
    for entry in dotenvy::from_path_iter(path)? {
        let (key, value) = entry?;
        if PROVIDER_ENV_KEYS.contains(&key.as_str()) && !is_set(&key) {
            updates.push((key, value));
        }
    }
    Ok(updates)
}

fn load_settings_file(path: &Path) -> SettingsFile {
    match std::fs::read_to_string(path) {
        Ok(content) => toml::from_str(&content).unwrap_or_else(|e| {
            tracing::warn!(path = %path.display(), error = %e, "settings file could not be parsed");
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
    // Allow/deny: append (project rules extend global rules).
    base.allow.extend(overlay.allow.iter().cloned());
    base.deny.extend(overlay.deny.iter().cloned());

    // Provider config blocks: overlay wins when set.
    if overlay.openrouter.is_some() {
        base.openrouter.clone_from(&overlay.openrouter);
    }
    if overlay.anthropic.is_some() {
        base.anthropic.clone_from(&overlay.anthropic);
    }
    if overlay.groq.is_some() {
        base.groq.clone_from(&overlay.groq);
    }
    if overlay.ollama.is_some() {
        base.ollama.clone_from(&overlay.ollama);
    }
    if overlay.custom.is_some() {
        base.custom.clone_from(&overlay.custom);
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
        let dir = tempfile::tempdir().unwrap();
        // Isolate from real ~/.config/piku/settings.toml
        std::env::set_var("XDG_CONFIG_HOME", dir.path());
        let cfg = PikuConfig::load(None, None, None);
        assert!(cfg.provider.is_none());
        assert!(cfg.model.is_none());
        std::env::remove_var("XDG_CONFIG_HOME");
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
            piku_dir.join("settings.toml"),
            r#"provider = "groq"
model = "llama-3""#,
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
        std::fs::write(piku_dir.join("settings.toml"), r#"provider = "groq""#).unwrap();

        let cfg = PikuConfig::load(Some("anthropic"), None, Some(dir.path()));
        assert_eq!(cfg.provider.as_deref(), Some("anthropic"));
    }

    #[test]
    fn sessions_traces_and_runs_dirs() {
        let cfg = PikuConfig::load(None, None, None);
        assert!(cfg.sessions_dir().ends_with("piku/sessions"));
        assert!(cfg.traces_dir().ends_with("piku/traces"));
        assert!(cfg.runs_dir().ends_with("piku/runs"));
        assert!(cfg.agent_links_dir().ends_with("piku/agent-links"));
    }

    #[test]
    fn permission_allow_exact() {
        let dir = tempfile::tempdir().unwrap();
        let piku_dir = dir.path().join(".piku");
        std::fs::create_dir_all(&piku_dir).unwrap();
        std::fs::write(
            piku_dir.join("settings.toml"),
            r#"allow = ["bash", "read_file"]"#,
        )
        .unwrap();

        let cfg = PikuConfig::load(None, None, Some(dir.path()));
        assert_eq!(
            cfg.check_permission_rule("bash", &serde_json::json!({})),
            Some(true)
        );
        assert_eq!(
            cfg.check_permission_rule("read_file", &serde_json::json!({})),
            Some(true)
        );
        assert_eq!(
            cfg.check_permission_rule("write_file", &serde_json::json!({})),
            None
        );
    }

    #[test]
    fn permission_allow_glob() {
        let dir = tempfile::tempdir().unwrap();
        let piku_dir = dir.path().join(".piku");
        std::fs::create_dir_all(&piku_dir).unwrap();
        std::fs::write(piku_dir.join("settings.toml"), r#"allow = ["bash(git *)"]"#).unwrap();

        let cfg = PikuConfig::load(None, None, Some(dir.path()));
        let git_push = serde_json::json!({"command": "git push origin main"});
        let rm_rf = serde_json::json!({"command": "rm -rf /"});
        assert_eq!(cfg.check_permission_rule("bash", &git_push), Some(true));
        assert_eq!(cfg.check_permission_rule("bash", &rm_rf), None);
    }

    #[test]
    fn permission_deny_overrides_allow() {
        let dir = tempfile::tempdir().unwrap();
        let piku_dir = dir.path().join(".piku");
        std::fs::create_dir_all(&piku_dir).unwrap();
        std::fs::write(
            piku_dir.join("settings.toml"),
            r#"allow = ["bash"]
deny = ["bash(rm *)"]"#,
        )
        .unwrap();

        let cfg = PikuConfig::load(None, None, Some(dir.path()));
        let git = serde_json::json!({"command": "git status"});
        let rm = serde_json::json!({"command": "rm -rf /"});
        assert_eq!(cfg.check_permission_rule("bash", &git), Some(true));
        assert_eq!(cfg.check_permission_rule("bash", &rm), Some(false));
    }

    #[test]
    fn provider_config_block_parsed() {
        let dir = tempfile::tempdir().unwrap();
        let piku_dir = dir.path().join(".piku");
        std::fs::create_dir_all(&piku_dir).unwrap();
        std::fs::write(
            piku_dir.join("settings.toml"),
            r#"provider = "openrouter"

[openrouter]
api_key = "sk-or-v1-test"
base_url = "https://openrouter.ai/api/v1"
model = "anthropic/claude-sonnet-4.6"

[ollama]
host = "http://localhost:11434"
model = "llama3.1""#,
        )
        .unwrap();

        let cfg = PikuConfig::load(None, None, Some(dir.path()));
        assert_eq!(cfg.provider.as_deref(), Some("openrouter"));

        let or = cfg.provider_configs.openrouter.unwrap();
        assert_eq!(or.api_key.as_deref(), Some("sk-or-v1-test"));
        assert_eq!(or.base_url.as_deref(), Some("https://openrouter.ai/api/v1"));
        assert_eq!(or.model.as_deref(), Some("anthropic/claude-sonnet-4.6"));

        let ol = cfg.provider_configs.ollama.unwrap();
        assert_eq!(ol.host.as_deref(), Some("http://localhost:11434"));
        assert_eq!(ol.model.as_deref(), Some("llama3.1"));
    }

    #[test]
    fn provider_dotenv_import_is_allowlisted_and_non_overriding() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(".env");
        std::fs::write(
            &path,
            "OPENROUTER_API_KEY=file-key\nOPENROUTER_BASE_URL=https://example.test/v1\nDATABASE_PASSWORD=do-not-import\n",
        )
        .unwrap();

        let updates = provider_dotenv_updates(&path, |key| key == "OPENROUTER_BASE_URL").unwrap();
        assert_eq!(
            updates,
            vec![("OPENROUTER_API_KEY".to_string(), "file-key".to_string())]
        );
    }

    #[test]
    fn nearest_dotenv_prefers_the_project_over_its_parent() {
        let dir = tempfile::tempdir().unwrap();
        let project = dir.path().join("project");
        std::fs::create_dir(&project).unwrap();
        std::fs::write(dir.path().join(".env"), "OPENROUTER_API_KEY=parent\n").unwrap();
        std::fs::write(project.join(".env"), "OPENROUTER_API_KEY=project\n").unwrap();

        assert_eq!(nearest_dotenv(Some(&project)), Some(project.join(".env")));
    }
}
