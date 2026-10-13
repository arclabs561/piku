/// Unified Piku settings loaded from global and project files plus CLI overrides.
///
/// Provider authority: CLI flags > operator-global settings > defaults.
/// Project settings may restrict permissions and budgets; reviewed allow rules
/// require a separate root- and policy-bound operator approval.
/// Provider-specific environment variables are resolved separately by the
/// runtime's provider resolver.
///
/// Config file: `$XDG_CONFIG_HOME/piku/settings.toml`, falling back to
/// `~/.config/piku/settings.toml` (user-global).
/// Project-local overrides: `.piku/settings.toml` (merged on top).
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

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

/// A named foreground participant in an interleaved room.
#[derive(Debug, Clone, Default, serde::Deserialize)]
pub struct ActorConfig {
    pub label: Option<String>,
    pub provider: Option<String>,
    pub model: Option<String>,
    pub instructions: Option<String>,
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
    /// Named actor profiles, configured as `[actors.researcher]`.
    #[serde(default)]
    pub actors: BTreeMap<String, ActorConfig>,
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
    /// Operator-global and explicitly reviewed project auto-allow rules.
    pub allow: Vec<String>,
    /// Tool names to always deny (global + project merged).
    pub deny: Vec<String>,
    pub actors: BTreeMap<String, ActorConfig>,
    /// Operator-global provider blocks, currently parsed without activation.
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
    /// Load operator settings, project restrictions/reviewed grants, and CLI overrides.
    #[must_use]
    pub fn load(
        cli_provider: Option<&str>,
        cli_model: Option<&str>,
        project_dir: Option<&Path>,
    ) -> Self {
        Self::load_in(cli_provider, cli_model, project_dir, global_config_dir())
    }

    fn load_in(
        cli_provider: Option<&str>,
        cli_model: Option<&str>,
        project_dir: Option<&Path>,
        config_dir: PathBuf,
    ) -> Self {
        // Layer 1: global settings file
        let global_path = config_dir.join("settings.toml");
        let mut settings = load_settings_file(&global_path);

        // Layer 2: project-local settings file (merged on top)
        if let Some(proj) = project_dir {
            let project_path = proj.join(".piku").join("settings.toml");
            let project_settings = load_settings_file(&project_path);
            merge_project_settings(
                &mut settings,
                &project_settings,
                crate::project_permissions::approved(&config_dir, proj, &project_settings),
            );
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
            actors: settings.actors,
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

    #[must_use]
    pub fn actor(&self, name: &str) -> Option<&ActorConfig> {
        self.actors.get(name)
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
            if matches_allow_tool_pattern(pattern, tool_name, params) {
                return Some(true);
            }
        }
        None
    }
}

/// Match a deny rule against a tool name and its params.
/// Supports: exact tool name (`"bash"`), tool with arg glob (`"bash(git *)"`)
/// with stars only at the edges. Scoped bash denies fail closed when command
/// structure involves shell controls or expansion; use
/// `matches_allow_tool_pattern` when considering an auto-grant.
#[must_use]
pub fn matches_tool_pattern(pattern: &str, tool_name: &str, params: &serde_json::Value) -> bool {
    if tool_name == "bash" && pattern.starts_with("bash(") && shell_command_uncertain(params) {
        return true;
    }
    // Parse `ToolName(glob)` syntax
    if let Some(paren_start) = pattern.find('(') {
        if let Some(paren_end) = pattern.rfind(')') {
            if paren_end != pattern.len() - 1 {
                return false;
            }
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

pub(crate) fn global_config_dir() -> PathBuf {
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

pub(crate) fn load_settings_file(path: &Path) -> SettingsFile {
    match std::fs::read_to_string(path) {
        Ok(content) => toml::from_str(&content).unwrap_or_else(|_| {
            tracing::warn!(path = %path.display(), "settings file could not be parsed (invalid TOML)");
            SettingsFile::default()
        }),
        Err(_) => SettingsFile::default(),
    }
}

fn merge_project_settings(base: &mut SettingsFile, overlay: &SettingsFile, approved: bool) {
    if let Some(limit) = overlay.max_turns {
        base.max_turns = Some(
            limit.min(
                base.max_turns
                    .unwrap_or(piku_runtime::agent_loop::DEFAULT_MAX_TURNS),
            ),
        );
    }
    base.deny.extend(overlay.deny.iter().cloned());
    if approved {
        base.allow.extend(overlay.allow.iter().cloned());
    }
    for (name, project_actor) in &overlay.actors {
        let actor = base.actors.entry(name.clone()).or_default();
        if project_actor.label.is_some() {
            actor.label.clone_from(&project_actor.label);
        }
        if project_actor.instructions.is_some() {
            actor.instructions.clone_from(&project_actor.instructions);
        }
    }
    // Provider connections, models and actor routing are operator authority.
}

/// Auto-grants for scoped shell commands require a simple command. Explicit
/// whole-tool grants remain broad and are displayed as such during review.
#[must_use]
pub fn matches_allow_tool_pattern(
    pattern: &str,
    tool_name: &str,
    params: &serde_json::Value,
) -> bool {
    if tool_name == "bash" && pattern.contains('(') && shell_command_uncertain(params) {
        return false;
    }
    matches_tool_pattern(pattern, tool_name, params)
}

fn shell_command_uncertain(params: &serde_json::Value) -> bool {
    params
        .get("command")
        .and_then(serde_json::Value::as_str)
        .is_none_or(|command| {
            command.chars().any(|c| {
                matches!(
                    c,
                    ';' | '&'
                        | '|'
                        | '$'
                        | '`'
                        | '\n'
                        | '\r'
                        | '<'
                        | '>'
                        | '('
                        | ')'
                        | '\\'
                        | '\''
                        | '"'
                        | '*'
                        | '?'
                        | '['
                        | ']'
                        | '{'
                        | '}'
                        | '~'
                        | '#'
                )
            })
        })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn isolated_load(
        provider: Option<&str>,
        model: Option<&str>,
        project: Option<&Path>,
    ) -> PikuConfig {
        let config = tempfile::tempdir().unwrap();
        PikuConfig::load_in(provider, model, project, config.path().to_path_buf())
    }

    fn approved_load(project: &Path) -> PikuConfig {
        let config = tempfile::tempdir().unwrap();
        let settings = load_settings_file(&project.join(".piku/settings.toml"));
        std::fs::write(
            config.path().join("approved-permissions.json"),
            serde_json::to_vec(&BTreeMap::from([(
                project.canonicalize().unwrap(),
                crate::project_permissions::digest(&settings),
            )]))
            .unwrap(),
        )
        .unwrap();
        PikuConfig::load_in(None, None, Some(project), config.path().to_path_buf())
    }

    #[test]
    fn load_empty_config() {
        let cfg = isolated_load(None, None, None);
        assert!(cfg.provider.is_none());
        assert!(cfg.model.is_none());
    }

    #[test]
    fn cli_overrides_file() {
        let cfg = isolated_load(Some("ollama"), Some("gemma4"), None);
        assert_eq!(cfg.provider.as_deref(), Some("ollama"));
        assert_eq!(cfg.model.as_deref(), Some("gemma4"));
    }

    #[test]
    fn project_cannot_select_provider_or_model() {
        let dir = tempfile::tempdir().unwrap();
        let piku_dir = dir.path().join(".piku");
        std::fs::create_dir_all(&piku_dir).unwrap();
        std::fs::write(
            piku_dir.join("settings.toml"),
            r#"provider = "groq"
model = "llama-3""#,
        )
        .unwrap();

        let cfg = isolated_load(None, None, Some(dir.path()));
        assert!(cfg.provider.is_none());
        assert!(cfg.model.is_none());
    }

    #[test]
    fn cli_beats_project() {
        let dir = tempfile::tempdir().unwrap();
        let piku_dir = dir.path().join(".piku");
        std::fs::create_dir_all(&piku_dir).unwrap();
        std::fs::write(piku_dir.join("settings.toml"), r#"provider = "groq""#).unwrap();

        let cfg = isolated_load(Some("anthropic"), None, Some(dir.path()));
        assert_eq!(cfg.provider.as_deref(), Some("anthropic"));
    }

    #[test]
    fn sessions_traces_and_runs_dirs() {
        let cfg = isolated_load(None, None, None);
        assert!(cfg.sessions_dir().ends_with("sessions"));
        assert!(cfg.traces_dir().ends_with("traces"));
        assert!(cfg.runs_dir().ends_with("runs"));
        assert!(cfg.agent_links_dir().ends_with("agent-links"));
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

        let cfg = approved_load(dir.path());
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

        let cfg = approved_load(dir.path());
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

        let cfg = approved_load(dir.path());
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

        let cfg = PikuConfig::load_in(None, None, None, piku_dir);
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
    fn project_restrictions_apply_without_approval_and_cannot_raise_budget() {
        let mut operator = SettingsFile {
            max_turns: Some(5),
            allow: vec!["bash".into()],
            ..SettingsFile::default()
        };
        let project = SettingsFile {
            max_turns: Some(50),
            allow: vec!["write_file".into()],
            deny: vec!["bash".into()],
            ..SettingsFile::default()
        };
        merge_project_settings(&mut operator, &project, false);
        assert_eq!(operator.max_turns, Some(5));
        assert_eq!(operator.allow, ["bash"]);
        assert_eq!(operator.deny, ["bash"]);
        let mut defaults = SettingsFile::default();
        merge_project_settings(&mut defaults, &project, false);
        assert_eq!(
            defaults.max_turns,
            Some(piku_runtime::agent_loop::DEFAULT_MAX_TURNS)
        );
        let restrictive = SettingsFile {
            max_turns: Some(2),
            ..SettingsFile::default()
        };
        merge_project_settings(&mut operator, &restrictive, false);
        assert_eq!(operator.max_turns, Some(2));
    }

    #[test]
    fn project_actor_task_data_preserves_operator_connection() {
        let mut operator = SettingsFile::default();
        operator.actors.insert(
            "reviewer".into(),
            ActorConfig {
                provider: Some("custom".into()),
                model: Some("operator-model".into()),
                ..ActorConfig::default()
            },
        );
        let actor = ActorConfig {
            label: Some("Reviewer".into()),
            instructions: Some("Review tests".into()),
            provider: Some("groq".into()),
            model: Some("project-model".into()),
        };
        let project = SettingsFile {
            actors: BTreeMap::from([("reviewer".into(), actor.clone()), ("new".into(), actor)]),
            ..SettingsFile::default()
        };
        merge_project_settings(&mut operator, &project, true);
        let existing = &operator.actors["reviewer"];
        assert_eq!(existing.provider.as_deref(), Some("custom"));
        assert_eq!(existing.model.as_deref(), Some("operator-model"));
        assert_eq!(existing.instructions.as_deref(), Some("Review tests"));
        assert!(operator.actors["new"].provider.is_none());
        assert!(operator.actors["new"].model.is_none());
    }

    #[test]
    fn shell_scoped_grants_reject_compounds_and_denies_fail_closed() {
        for command in [
            "git status; rm -rf /",
            "git status && echo x",
            "git $(echo status)",
            "git `echo status`",
            "git status\nrm x",
            "git 'status'",
            "git status > x",
        ] {
            let params = serde_json::json!({"command": command});
            assert!(
                !matches_allow_tool_pattern("bash(git *)", "bash", &params),
                "{command}"
            );
            assert!(
                matches_tool_pattern("bash(rm *)", "bash", &params),
                "{command}"
            );
        }
        assert!(matches_allow_tool_pattern(
            "bash(git status)",
            "bash",
            &serde_json::json!({"command":"git status"})
        ));
        assert!(!matches_allow_tool_pattern(
            "bash(git * status)",
            "bash",
            &serde_json::json!({"command":"git foo status"})
        ));
        assert!(!matches_tool_pattern(
            "bash(git *)suffix",
            "bash",
            &serde_json::json!({"command":"git status"})
        ));
    }

    #[test]
    fn loading_project_policy_preserves_operator_routes_and_denials() {
        let config = tempfile::tempdir().unwrap();
        let project = tempfile::tempdir().unwrap();
        std::fs::create_dir(project.path().join(".piku")).unwrap();
        std::fs::write(config.path().join("settings.toml"), "provider = \"custom\"\nmodel = \"operator-model\"\nallow = [\"bash\"]\ndeny = [\"bash(rm *)\"]\n[custom]\nbase_url = \"http://operator.test\"\n").unwrap();
        let path = project.path().join(".piku/settings.toml");
        std::fs::write(&path, "provider = \"groq\"\nmodel = \"repo-model\"\nallow = [\"write_file\"]\ndeny = [\"edit_file\"]\n[custom]\nbase_url = \"http://repo.test\"\n").unwrap();
        let cfg = PikuConfig::load_in(
            None,
            None,
            Some(project.path()),
            config.path().to_path_buf(),
        );
        assert_eq!(cfg.provider.as_deref(), Some("custom"));
        assert_eq!(cfg.model.as_deref(), Some("operator-model"));
        assert_eq!(
            cfg.provider_configs
                .custom
                .as_ref()
                .unwrap()
                .base_url
                .as_deref(),
            Some("http://operator.test")
        );
        assert_eq!(
            cfg.check_permission_rule("write_file", &serde_json::json!({})),
            None
        );
        assert_eq!(
            cfg.check_permission_rule("edit_file", &serde_json::json!({})),
            Some(false)
        );
        assert_eq!(
            cfg.check_permission_rule("bash", &serde_json::json!({"command":"rm file"})),
            Some(false)
        );
        let settings = load_settings_file(&path);
        std::fs::write(
            config.path().join("approved-permissions.json"),
            serde_json::to_vec(&BTreeMap::from([(
                project.path().canonicalize().unwrap(),
                crate::project_permissions::digest(&settings),
            )]))
            .unwrap(),
        )
        .unwrap();
        let cfg = PikuConfig::load_in(
            None,
            None,
            Some(project.path()),
            config.path().to_path_buf(),
        );
        assert_eq!(
            cfg.check_permission_rule("write_file", &serde_json::json!({})),
            Some(true)
        );
        assert_eq!(
            cfg.check_permission_rule("bash", &serde_json::json!({"command":"rm file"})),
            Some(false)
        );
        std::fs::write(
            path,
            "allow = [\"write_file\", \"edit_file\"]\ndeny = [\"edit_file\"]\n",
        )
        .unwrap();
        let cfg = PikuConfig::load_in(
            None,
            None,
            Some(project.path()),
            config.path().to_path_buf(),
        );
        assert_eq!(
            cfg.check_permission_rule("write_file", &serde_json::json!({})),
            None
        );
        assert_eq!(
            cfg.check_permission_rule("edit_file", &serde_json::json!({})),
            Some(false)
        );
    }
}
