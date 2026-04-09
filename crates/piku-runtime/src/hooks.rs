/// Lifecycle hooks -- user-defined shell commands that fire at key points.
///
/// Hooks are configured in `.piku/hooks.json` or via `HookRegistry::register`.
/// Each hook receives JSON context on stdin and can block actions via exit code 2.
///
/// Supported events:
/// - `PreToolUse`: before a tool call executes (can block or modify)
/// - `PostToolUse`: after a tool call succeeds
/// - `SessionStart`: when a session begins
/// - `Stop`: after a turn completes (notifications, logging, cleanup)
///
/// Inspired by Claude Code's hooks system (code.claude.com/docs/en/hooks).
use std::path::{Path, PathBuf};
use std::process::Command;

// ---------------------------------------------------------------------------
// Configuration types
// ---------------------------------------------------------------------------

/// A single hook handler (shell command).
#[derive(Debug, Clone, serde::Deserialize)]
pub struct HookHandler {
    /// Shell command to execute. Receives JSON on stdin.
    pub command: String,
    /// Optional: only fire when tool args match this pattern.
    /// Uses the same syntax as permission rules: `Bash(git *)`, `Edit(*.rs)`.
    #[serde(rename = "if")]
    pub if_condition: Option<String>,
    /// Run asynchronously (don't wait for result).
    #[serde(default)]
    pub r#async: bool,
    /// Timeout in seconds (default 10).
    #[serde(default = "default_timeout")]
    pub timeout: u64,
}

fn default_timeout() -> u64 {
    10
}

/// A matcher + hooks pair for a specific event.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct HookEntry {
    /// Pattern to match tool names. Supports exact match (`"Bash"`),
    /// pipe-delimited alternation (`"Edit|Write"`), or wildcard (`"*"`).
    /// `None` matches all tools.
    pub matcher: Option<String>,
    /// Hook handlers to run when matched.
    pub hooks: Vec<HookHandler>,
}

/// Top-level hooks configuration (from `.piku/hooks.json`).
#[derive(Debug, Clone, Default, serde::Deserialize)]
pub struct HookConfig {
    #[serde(rename = "PreToolUse", default)]
    pub pre_tool_use: Vec<HookEntry>,
    #[serde(rename = "PostToolUse", default)]
    pub post_tool_use: Vec<HookEntry>,
    #[serde(rename = "SessionStart", default)]
    pub session_start: Vec<HookEntry>,
    #[serde(rename = "Stop", default)]
    pub stop: Vec<HookEntry>,
}

// ---------------------------------------------------------------------------
// Hook events
// ---------------------------------------------------------------------------

/// What a `PreToolUse` hook can decide.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HookDecision {
    /// Allow the tool call (default).
    Allow,
    /// Block the tool call with a reason.
    Deny(String),
}

/// Result of running hooks for an event.
#[derive(Debug, Clone)]
pub struct HookResult {
    pub decision: HookDecision,
    /// Additional context to inject (from hook stdout).
    pub context: Option<String>,
}

impl Default for HookResult {
    fn default() -> Self {
        Self {
            decision: HookDecision::Allow,
            context: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------

/// Manages hook configuration and execution.
#[derive(Debug, Clone, Default)]
pub struct HookRegistry {
    config: HookConfig,
    project_dir: Option<PathBuf>,
}

impl HookRegistry {
    /// Load hooks from global `~/.config/piku/hooks.json` merged with
    /// project-local `.piku/hooks.json`. Project hooks are appended (run after global).
    #[must_use]
    pub fn load(project_dir: &Path) -> Self {
        // Layer 1: global hooks
        let global_dir = global_config_dir();
        let mut config = load_hooks_file(&global_dir.join("hooks.json"));

        // Layer 2: project-local hooks (appended)
        let project_config = load_hooks_file(&project_dir.join(".piku").join("hooks.json"));
        merge_hook_config(&mut config, project_config);

        Self {
            config,
            project_dir: Some(project_dir.to_path_buf()),
        }
    }

    /// Load hooks from a single project directory only (no global merge).
    /// Used in tests where global config should not interfere.
    #[cfg(test)]
    #[must_use]
    pub fn load_project_only(project_dir: &Path) -> Self {
        let config = load_hooks_file(&project_dir.join(".piku").join("hooks.json"));
        Self {
            config,
            project_dir: Some(project_dir.to_path_buf()),
        }
    }

    /// Human-readable summary of configured hooks.
    #[must_use]
    pub fn summary(&self) -> String {
        use std::fmt::Write;
        let mut out = String::new();
        let events = [
            ("PreToolUse", &self.config.pre_tool_use),
            ("PostToolUse", &self.config.post_tool_use),
            ("SessionStart", &self.config.session_start),
            ("Stop", &self.config.stop),
        ];
        for (name, entries) in events {
            for entry in entries {
                let matcher = entry.matcher.as_deref().unwrap_or("*");
                for handler in &entry.hooks {
                    let cmd: String = handler.command.chars().take(50).collect();
                    let flags = if handler.r#async { " (async)" } else { "" };
                    let _ = writeln!(out, "{name} [{matcher}]: {cmd}{flags}");
                }
            }
        }
        if out.is_empty() {
            out.push_str("(none)");
        }
        out
    }

    /// Check if any hooks are configured.
    #[must_use]
    pub fn has_hooks(&self) -> bool {
        !self.config.pre_tool_use.is_empty()
            || !self.config.post_tool_use.is_empty()
            || !self.config.session_start.is_empty()
            || !self.config.stop.is_empty()
    }

    /// Run `PreToolUse` hooks. Returns the decision (allow/deny).
    #[must_use]
    pub fn run_pre_tool_use(
        &self,
        tool_name: &str,
        tool_input: &serde_json::Value,
        session_id: &str,
        cwd: &Path,
    ) -> HookResult {
        let entries = &self.config.pre_tool_use;
        if entries.is_empty() {
            return HookResult::default();
        }

        let input = serde_json::json!({
            "hook_event_name": "PreToolUse",
            "tool_name": tool_name,
            "tool_input": tool_input,
            "session_id": session_id,
            "cwd": cwd.display().to_string(),
        });

        for entry in entries {
            if !matches_tool(entry.matcher.as_deref(), tool_name) {
                continue;
            }
            for handler in &entry.hooks {
                if !check_if_condition(handler.if_condition.as_deref(), tool_name, tool_input) {
                    continue;
                }
                match run_hook_command(
                    &handler.command,
                    &input,
                    handler.timeout,
                    self.project_dir.as_deref(),
                ) {
                    HookCommandResult::Success(stdout) => {
                        // Parse JSON output for decision
                        if let Some(decision) = parse_pre_tool_decision(&stdout) {
                            if decision.decision != HookDecision::Allow {
                                return decision;
                            }
                        }
                    }
                    HookCommandResult::Blocked(stderr) => {
                        return HookResult {
                            decision: HookDecision::Deny(if stderr.is_empty() {
                                format!("hook blocked {tool_name}")
                            } else {
                                stderr
                            }),
                            context: None,
                        };
                    }
                    HookCommandResult::Error(msg) => {
                        eprintln!("[piku] hook error: {msg}");
                        // Non-blocking error -- continue
                    }
                }
            }
        }

        HookResult::default()
    }

    /// Run `PostToolUse` hooks (fire-and-forget for sync, ignore for async).
    pub fn run_post_tool_use(
        &self,
        tool_name: &str,
        tool_input: &serde_json::Value,
        tool_output: &str,
        is_error: bool,
        session_id: &str,
        cwd: &Path,
    ) {
        let entries = &self.config.post_tool_use;
        if entries.is_empty() {
            return;
        }

        let input = serde_json::json!({
            "hook_event_name": "PostToolUse",
            "tool_name": tool_name,
            "tool_input": tool_input,
            "tool_response": {
                "output": tool_output,
                "is_error": is_error,
            },
            "session_id": session_id,
            "cwd": cwd.display().to_string(),
        });

        for entry in entries {
            if !matches_tool(entry.matcher.as_deref(), tool_name) {
                continue;
            }
            for handler in &entry.hooks {
                if !check_if_condition(handler.if_condition.as_deref(), tool_name, tool_input) {
                    continue;
                }
                if handler.r#async {
                    // Fire and forget
                    let cmd = handler.command.clone();
                    let input_str = input.to_string();
                    let project_dir = self.project_dir.clone();
                    std::thread::spawn(move || {
                        let _ = run_hook_command_raw(&cmd, &input_str, 30, project_dir.as_deref());
                    });
                } else if let HookCommandResult::Error(msg) = run_hook_command(
                    &handler.command,
                    &input,
                    handler.timeout,
                    self.project_dir.as_deref(),
                ) {
                    eprintln!("[piku] post-tool hook error: {msg}");
                }
            }
        }
    }

    /// Run `SessionStart` hooks. Returns additional context to inject.
    #[must_use]
    pub fn run_session_start(&self, session_id: &str, cwd: &Path) -> Option<String> {
        let entries = &self.config.session_start;
        if entries.is_empty() {
            return None;
        }

        let input = serde_json::json!({
            "hook_event_name": "SessionStart",
            "session_id": session_id,
            "cwd": cwd.display().to_string(),
        });

        let mut context_parts = Vec::new();
        for entry in entries {
            for handler in &entry.hooks {
                match run_hook_command(
                    &handler.command,
                    &input,
                    handler.timeout,
                    self.project_dir.as_deref(),
                ) {
                    HookCommandResult::Success(stdout) if !stdout.trim().is_empty() => {
                        context_parts.push(stdout.trim().to_string());
                    }
                    HookCommandResult::Error(msg) => {
                        eprintln!("[piku] session-start hook error: {msg}");
                    }
                    _ => {}
                }
            }
        }

        if context_parts.is_empty() {
            None
        } else {
            Some(context_parts.join("\n\n"))
        }
    }

    /// Run `Stop` hooks after a turn completes. Fire-and-forget (errors logged).
    pub fn run_stop(
        &self,
        session_id: &str,
        cwd: &Path,
        iterations: u32,
        stop_reason: &str,
        usage: &piku_api::TokenUsage,
        duration_ms: u64,
    ) {
        let entries = &self.config.stop;
        if entries.is_empty() {
            return;
        }

        let input = serde_json::json!({
            "hook_event_name": "Stop",
            "session_id": session_id,
            "cwd": cwd.display().to_string(),
            "iterations": iterations,
            "stop_reason": stop_reason,
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "total_tokens": usage.input_tokens + usage.output_tokens,
            "duration_ms": duration_ms,
        });

        for entry in entries {
            for handler in &entry.hooks {
                if handler.r#async {
                    let cmd = handler.command.clone();
                    let input_str = input.to_string();
                    let project_dir = self.project_dir.clone();
                    std::thread::spawn(move || {
                        let _ = run_hook_command_raw(&cmd, &input_str, 30, project_dir.as_deref());
                    });
                } else if let HookCommandResult::Error(msg) = run_hook_command(
                    &handler.command,
                    &input,
                    handler.timeout,
                    self.project_dir.as_deref(),
                ) {
                    eprintln!("[piku] stop hook error: {msg}");
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

enum HookCommandResult {
    /// Exit 0 with stdout.
    Success(String),
    /// Exit 2 (blocking) with stderr.
    Blocked(String),
    /// Any other error.
    Error(String),
}

fn run_hook_command(
    command: &str,
    input: &serde_json::Value,
    timeout_secs: u64,
    project_dir: Option<&Path>,
) -> HookCommandResult {
    run_hook_command_raw(command, &input.to_string(), timeout_secs, project_dir)
}

fn run_hook_command_raw(
    command: &str,
    input_json: &str,
    timeout_secs: u64,
    project_dir: Option<&Path>,
) -> HookCommandResult {
    use std::io::Write;
    use std::process::Stdio;

    let mut cmd = Command::new("sh");
    cmd.arg("-c").arg(command);
    cmd.stdin(Stdio::piped());
    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::piped());

    if let Some(dir) = project_dir {
        cmd.env("PIKU_PROJECT_DIR", dir);
    }

    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => return HookCommandResult::Error(format!("spawn failed: {e}")),
    };

    // Write JSON to stdin then close it so the child sees EOF.
    if let Some(mut stdin) = child.stdin.take() {
        let _ = stdin.write_all(input_json.as_bytes());
    }

    // Poll with timeout instead of blocking indefinitely.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(timeout_secs);
    loop {
        match child.try_wait() {
            Ok(Some(_status)) => break,
            Ok(None) => {
                if std::time::Instant::now() >= deadline {
                    let _ = child.kill();
                    let _ = child.wait(); // reap
                    return HookCommandResult::Error(format!(
                        "hook timed out after {timeout_secs}s: {command}"
                    ));
                }
                std::thread::sleep(std::time::Duration::from_millis(50));
            }
            Err(e) => return HookCommandResult::Error(format!("wait failed: {e}")),
        }
    }

    let output = match child.wait_with_output() {
        Ok(o) => o,
        Err(e) => return HookCommandResult::Error(format!("wait failed: {e}")),
    };

    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();

    match output.status.code() {
        Some(0) => HookCommandResult::Success(stdout),
        Some(2) => HookCommandResult::Blocked(stderr),
        Some(code) => HookCommandResult::Error(format!("exit code {code}: {stderr}")),
        None => HookCommandResult::Error("killed by signal".to_string()),
    }
}

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

fn load_hooks_file(path: &Path) -> HookConfig {
    match std::fs::read_to_string(path) {
        Ok(content) => match serde_json::from_str(&content) {
            Ok(cfg) => cfg,
            Err(e) => {
                eprintln!("[piku] warning: failed to parse {}: {e}", path.display());
                HookConfig::default()
            }
        },
        Err(_) => HookConfig::default(),
    }
}

fn merge_hook_config(base: &mut HookConfig, overlay: HookConfig) {
    base.pre_tool_use.extend(overlay.pre_tool_use);
    base.post_tool_use.extend(overlay.post_tool_use);
    base.session_start.extend(overlay.session_start);
    base.stop.extend(overlay.stop);
}

/// Check if a tool name matches a matcher pattern.
/// Supports: exact match, pipe-delimited alternation (`"Edit|Write"`), wildcard (`"*"`).
fn matches_tool(matcher: Option<&str>, tool_name: &str) -> bool {
    match matcher {
        None => true, // No matcher = match all
        Some(pattern) => pattern
            .split('|')
            .any(|p| p.trim() == tool_name || p.trim() == "*"),
    }
}

/// Check if the `if` condition matches.
/// Format: `ToolName(pattern)` where pattern is matched against the first string arg.
fn check_if_condition(
    condition: Option<&str>,
    tool_name: &str,
    tool_input: &serde_json::Value,
) -> bool {
    let Some(cond) = condition else {
        return true; // No condition = always match
    };

    // Parse `ToolName(pattern)`
    let Some(paren_start) = cond.find('(') else {
        return cond == tool_name; // No parens = just match tool name
    };
    let Some(paren_end) = cond.rfind(')') else {
        return false;
    };

    let cond_tool = &cond[..paren_start];
    if cond_tool != tool_name {
        return false;
    }

    let pattern = &cond[paren_start + 1..paren_end];

    // Match pattern against the primary tool argument
    let primary_arg = match tool_name {
        "bash" => tool_input.get("command").and_then(|v| v.as_str()),
        "read_file" | "write_file" | "edit_file" => tool_input.get("path").and_then(|v| v.as_str()),
        "glob" | "grep" => tool_input.get("pattern").and_then(|v| v.as_str()),
        _ => None,
    };

    let Some(arg) = primary_arg else {
        return false;
    };

    // Glob matching: `*` wildcard at start, end, or both.
    if pattern == "*" {
        return true;
    }
    let starts_star = pattern.starts_with('*');
    let ends_star = pattern.ends_with('*');
    match (starts_star, ends_star) {
        (true, true) => {
            // *contains*
            let inner = &pattern[1..pattern.len() - 1];
            arg.contains(inner)
        }
        (false, true) => {
            // prefix*
            let prefix = &pattern[..pattern.len() - 1];
            arg.starts_with(prefix)
        }
        (true, false) => {
            // *suffix
            let suffix = &pattern[1..];
            arg.ends_with(suffix)
        }
        (false, false) => arg == pattern,
    }
}

/// Parse `PreToolUse` hook JSON output for a decision.
fn parse_pre_tool_decision(stdout: &str) -> Option<HookResult> {
    let json: serde_json::Value = serde_json::from_str(stdout.trim()).ok()?;

    let hook_output = json.get("hookSpecificOutput")?;
    let decision = hook_output
        .get("permissionDecision")
        .and_then(|v| v.as_str())?;

    let reason = hook_output
        .get("permissionDecisionReason")
        .and_then(|v| v.as_str())
        .unwrap_or("blocked by hook")
        .to_string();

    let context = hook_output
        .get("additionalContext")
        .and_then(|v| v.as_str())
        .map(String::from);

    match decision {
        "deny" => Some(HookResult {
            decision: HookDecision::Deny(reason),
            context,
        }),
        "allow" => Some(HookResult {
            decision: HookDecision::Allow,
            context,
        }),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_tool_no_matcher() {
        assert!(matches_tool(None, "bash"));
        assert!(matches_tool(None, "read_file"));
    }

    #[test]
    fn matches_tool_exact() {
        assert!(matches_tool(Some("Bash"), "Bash"));
        assert!(!matches_tool(Some("Bash"), "Edit"));
    }

    #[test]
    fn matches_tool_alternation() {
        assert!(matches_tool(Some("Edit|Write"), "Edit"));
        assert!(matches_tool(Some("Edit|Write"), "Write"));
        assert!(!matches_tool(Some("Edit|Write"), "Bash"));
    }

    #[test]
    fn matches_tool_wildcard() {
        assert!(matches_tool(Some("*"), "anything"));
    }

    #[test]
    fn check_if_no_condition() {
        assert!(check_if_condition(None, "bash", &serde_json::json!({})));
    }

    #[test]
    fn check_if_tool_name_only() {
        assert!(check_if_condition(
            Some("bash"),
            "bash",
            &serde_json::json!({})
        ));
        assert!(!check_if_condition(
            Some("edit"),
            "bash",
            &serde_json::json!({})
        ));
    }

    #[test]
    fn check_if_bash_pattern() {
        let input = serde_json::json!({"command": "git push origin main"});
        assert!(check_if_condition(Some("bash(git *)"), "bash", &input));
        assert!(!check_if_condition(Some("bash(npm *)"), "bash", &input));
    }

    #[test]
    fn check_if_file_pattern() {
        let input = serde_json::json!({"path": "src/main.rs"});
        assert!(check_if_condition(
            Some("edit_file(*.rs)"),
            "edit_file",
            &input
        ));
        assert!(!check_if_condition(
            Some("edit_file(*.ts)"),
            "edit_file",
            &input
        ));
    }

    #[test]
    fn parse_deny_decision() {
        let stdout = r#"{"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":"deny","permissionDecisionReason":"dangerous command"}}"#;
        let result = parse_pre_tool_decision(stdout).unwrap();
        assert_eq!(
            result.decision,
            HookDecision::Deny("dangerous command".to_string())
        );
    }

    #[test]
    fn parse_allow_decision() {
        let stdout =
            r#"{"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":"allow"}}"#;
        let result = parse_pre_tool_decision(stdout).unwrap();
        assert_eq!(result.decision, HookDecision::Allow);
    }

    #[test]
    fn parse_non_json_returns_none() {
        assert!(parse_pre_tool_decision("not json").is_none());
        assert!(parse_pre_tool_decision("").is_none());
    }

    #[test]
    fn load_empty_config() {
        let dir = tempfile::tempdir().unwrap();
        let registry = HookRegistry::load_project_only(dir.path());
        assert!(!registry.has_hooks());
    }

    #[test]
    fn load_config_from_file() {
        let dir = tempfile::tempdir().unwrap();
        let hooks_dir = dir.path().join(".piku");
        std::fs::create_dir_all(&hooks_dir).unwrap();
        std::fs::write(
            hooks_dir.join("hooks.json"),
            r#"{
                "PreToolUse": [{
                    "matcher": "bash",
                    "hooks": [{"command": "echo ok"}]
                }]
            }"#,
        )
        .unwrap();

        let registry = HookRegistry::load_project_only(dir.path());
        assert!(registry.has_hooks());
        assert_eq!(registry.config.pre_tool_use.len(), 1);
    }

    #[test]
    fn run_session_start_captures_stdout() {
        let dir = tempfile::tempdir().unwrap();
        let hooks_dir = dir.path().join(".piku");
        std::fs::create_dir_all(&hooks_dir).unwrap();
        std::fs::write(
            hooks_dir.join("hooks.json"),
            r#"{
                "SessionStart": [{
                    "hooks": [{"command": "echo 'hello from hook'"}]
                }]
            }"#,
        )
        .unwrap();

        let registry = HookRegistry::load_project_only(dir.path());
        let context = registry.run_session_start("test-session", dir.path());
        assert!(context.is_some());
        assert!(context.unwrap().contains("hello from hook"));
    }

    #[test]
    fn pre_tool_use_exit2_blocks() {
        let dir = tempfile::tempdir().unwrap();
        let hooks_dir = dir.path().join(".piku");
        std::fs::create_dir_all(&hooks_dir).unwrap();
        std::fs::write(
            hooks_dir.join("hooks.json"),
            r#"{
                "PreToolUse": [{
                    "matcher": "bash",
                    "hooks": [{"command": "echo 'blocked by policy' >&2; exit 2"}]
                }]
            }"#,
        )
        .unwrap();

        let registry = HookRegistry::load_project_only(dir.path());
        let result = registry.run_pre_tool_use(
            "bash",
            &serde_json::json!({"command": "rm -rf /"}),
            "test-session",
            dir.path(),
        );
        assert_eq!(
            result.decision,
            HookDecision::Deny("blocked by policy".to_string())
        );
    }

    #[test]
    fn pre_tool_use_unmatched_tool_allows() {
        let dir = tempfile::tempdir().unwrap();
        let hooks_dir = dir.path().join(".piku");
        std::fs::create_dir_all(&hooks_dir).unwrap();
        std::fs::write(
            hooks_dir.join("hooks.json"),
            r#"{
                "PreToolUse": [{
                    "matcher": "bash",
                    "hooks": [{"command": "exit 2"}]
                }]
            }"#,
        )
        .unwrap();

        let registry = HookRegistry::load_project_only(dir.path());
        let result = registry.run_pre_tool_use(
            "read_file",
            &serde_json::json!({"path": "test.rs"}),
            "test-session",
            dir.path(),
        );
        assert_eq!(result.decision, HookDecision::Allow);
    }

    #[test]
    fn stop_hook_fires() {
        let dir = tempfile::tempdir().unwrap();
        let hooks_dir = dir.path().join(".piku");
        std::fs::create_dir_all(&hooks_dir).unwrap();
        let marker = dir.path().join("stop_fired.txt");
        let cmd = format!("cat > {}", marker.display());
        let json = serde_json::json!({
            "Stop": [{
                "hooks": [{"command": cmd}]
            }]
        });
        std::fs::write(hooks_dir.join("hooks.json"), json.to_string()).unwrap();

        let registry = HookRegistry::load_project_only(dir.path());
        registry.run_stop(
            "test-session",
            dir.path(),
            3,
            "end_turn",
            &piku_api::TokenUsage::default(),
            1500,
        );
        assert!(marker.exists(), "stop hook should write marker file");
        let content = std::fs::read_to_string(&marker).unwrap();
        assert!(content.contains("\"hook_event_name\":\"Stop\""));
        assert!(content.contains("\"iterations\":3"));
    }

    #[test]
    fn hook_timeout_kills_slow_command() {
        let dir = tempfile::tempdir().unwrap();
        let hooks_dir = dir.path().join(".piku");
        std::fs::create_dir_all(&hooks_dir).unwrap();
        std::fs::write(
            hooks_dir.join("hooks.json"),
            r#"{
                "SessionStart": [{
                    "hooks": [{"command": "sleep 60", "timeout": 1}]
                }]
            }"#,
        )
        .unwrap();

        let registry = HookRegistry::load_project_only(dir.path());
        let start = std::time::Instant::now();
        let context = registry.run_session_start("test-session", dir.path());
        let elapsed = start.elapsed();
        // Should return within ~2s (1s timeout + polling overhead), not 60s.
        assert!(
            elapsed.as_secs() < 5,
            "hook should have been killed by timeout"
        );
        assert!(
            context.is_none(),
            "timed-out hook should not produce context"
        );
    }

    #[test]
    fn pre_tool_use_additional_context() {
        let stdout_json = r#"{"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":"allow","additionalContext":"remember to be careful"}}"#;
        let result = parse_pre_tool_decision(stdout_json).unwrap();
        assert_eq!(result.decision, HookDecision::Allow);
        assert_eq!(result.context.as_deref(), Some("remember to be careful"));
    }

    #[test]
    fn merge_hook_configs() {
        let mut base = HookConfig {
            pre_tool_use: vec![HookEntry {
                matcher: Some("Bash".to_string()),
                hooks: vec![HookHandler {
                    command: "echo global".to_string(),
                    if_condition: None,
                    r#async: false,
                    timeout: 10,
                }],
            }],
            ..Default::default()
        };
        let overlay = HookConfig {
            pre_tool_use: vec![HookEntry {
                matcher: Some("Edit".to_string()),
                hooks: vec![HookHandler {
                    command: "echo project".to_string(),
                    if_condition: None,
                    r#async: false,
                    timeout: 10,
                }],
            }],
            stop: vec![HookEntry {
                matcher: None,
                hooks: vec![HookHandler {
                    command: "echo stop".to_string(),
                    if_condition: None,
                    r#async: false,
                    timeout: 10,
                }],
            }],
            ..Default::default()
        };
        merge_hook_config(&mut base, overlay);
        assert_eq!(base.pre_tool_use.len(), 2);
        assert_eq!(base.stop.len(), 1);
    }

    #[test]
    fn check_if_contains_pattern() {
        let input = serde_json::json!({"command": "git push --force origin main"});
        assert!(check_if_condition(Some("bash(*--force*)"), "bash", &input));
        assert!(!check_if_condition(
            Some("bash(*--delete*)"),
            "bash",
            &input
        ));
    }
}
