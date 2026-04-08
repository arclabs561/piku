/// Agent definitions — built-in and custom (loaded from `.piku/agents/*.md`).
///
/// Each agent has a type name, a system prompt, and optional tool
/// restrictions.  The agent loop resolves these when the model calls
/// `spawn_agent` with a matching `subagent_type`.
///
/// Custom agents are defined as markdown files with YAML-ish frontmatter:
///
/// ```markdown
/// ---
/// name: reviewer
/// description: Review code changes for correctness and style.
/// disallowed_tools: [write_file, edit_file]
/// max_turns: 25
/// ---
///
/// You are a code reviewer. Read the diff and report issues...
/// ```

use std::path::Path;

// ---------------------------------------------------------------------------
// Definition types
// ---------------------------------------------------------------------------

/// A built-in agent definition with `&'static` string references.
#[derive(Debug, Clone)]
pub struct AgentDef {
    pub agent_type: &'static str,
    pub when_to_use: &'static str,
    pub system_prompt: &'static str,
    pub disallowed_tools: &'static [&'static str],
    /// When non-empty, only these tools are available (`disallowed_tools` ignored).
    pub allowed_tools: &'static [&'static str],
    pub max_turns: Option<u32>,
}

/// A custom agent definition loaded from a `.md` file (owns its strings).
#[derive(Debug, Clone)]
pub struct CustomAgentDef {
    pub agent_type: String,
    pub when_to_use: String,
    pub system_prompt: String,
    pub disallowed_tools: Vec<String>,
    pub allowed_tools: Vec<String>,
    pub max_turns: Option<u32>,
    /// Source file path (for debugging).
    pub source_path: String,
}

/// Unified view over both built-in and custom agents.
#[derive(Debug, Clone)]
pub enum AnyAgentDef {
    BuiltIn(&'static AgentDef),
    Custom(CustomAgentDef),
}

impl AnyAgentDef {
    #[must_use]
    pub fn agent_type(&self) -> &str {
        match self {
            Self::BuiltIn(d) => d.agent_type,
            Self::Custom(d) => &d.agent_type,
        }
    }

    #[must_use]
    pub fn when_to_use(&self) -> &str {
        match self {
            Self::BuiltIn(d) => d.when_to_use,
            Self::Custom(d) => &d.when_to_use,
        }
    }

    #[must_use]
    pub fn system_prompt(&self) -> &str {
        match self {
            Self::BuiltIn(d) => d.system_prompt,
            Self::Custom(d) => &d.system_prompt,
        }
    }

    #[must_use]
    pub fn max_turns(&self) -> Option<u32> {
        match self {
            Self::BuiltIn(d) => d.max_turns,
            Self::Custom(d) => d.max_turns,
        }
    }

    /// Check if a tool is allowed for this agent.
    /// Returns true if the tool should be included in the subagent's tool set.
    #[must_use]
    pub fn is_tool_allowed(&self, tool_name: &str) -> bool {
        match self {
            Self::BuiltIn(d) => {
                if !d.allowed_tools.is_empty() {
                    return d.allowed_tools.contains(&tool_name);
                }
                !d.disallowed_tools.contains(&tool_name)
            }
            Self::Custom(d) => {
                if !d.allowed_tools.is_empty() {
                    return d.allowed_tools.iter().any(|t| t == tool_name);
                }
                !d.disallowed_tools.iter().any(|t| t == tool_name)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Built-in registry
// ---------------------------------------------------------------------------

static BUILT_INS: &[AgentDef] = &[VERIFICATION_AGENT, EXPLORER_AGENT];

/// Return all built-in agent definitions.
#[must_use]
pub fn all_built_ins() -> &'static [AgentDef] {
    BUILT_INS
}

/// Look up a built-in agent by type name.
#[must_use]
pub fn find_built_in(agent_type: &str) -> Option<&'static AgentDef> {
    BUILT_INS.iter().find(|a| a.agent_type == agent_type)
}

// ---------------------------------------------------------------------------
// Custom agent loading
// ---------------------------------------------------------------------------

/// Load custom agent definitions from `.piku/agents/` in the given directory.
/// Returns an empty vec if the directory doesn't exist.
pub fn load_custom_agents(project_root: &Path) -> Vec<CustomAgentDef> {
    let agents_dir = project_root.join(".piku").join("agents");
    let entries = match std::fs::read_dir(&agents_dir) {
        Ok(e) => e,
        Err(_) => return Vec::new(),
    };

    let mut agents = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("md") {
            continue;
        }
        match parse_agent_markdown(&path) {
            Ok(def) => agents.push(def),
            Err(e) => {
                eprintln!("[piku] warning: failed to parse agent {}: {e}", path.display());
            }
        }
    }
    agents
}

/// Find an agent by type name across built-ins and custom agents.
#[must_use]
pub fn find_agent<'a>(
    agent_type: &str,
    custom_agents: &'a [CustomAgentDef],
) -> Option<AnyAgentDef> {
    // Custom agents override built-ins with the same name.
    if let Some(custom) = custom_agents.iter().find(|a| a.agent_type == agent_type) {
        return Some(AnyAgentDef::Custom(custom.clone()));
    }
    find_built_in(agent_type).map(AnyAgentDef::BuiltIn)
}

/// Build the agent listing prompt including both built-in and custom agents.
#[must_use]
pub fn agent_listing_prompt_with_custom(custom_agents: &[CustomAgentDef]) -> String {
    let mut out = String::from(
        "# Available agents\n\n\
         You can delegate tasks to specialized agents using the `spawn_agent` tool.\n\
         Pass `subagent_type` to use a named agent.\n\n",
    );
    for a in BUILT_INS {
        out.push_str("- **");
        out.push_str(a.agent_type);
        out.push_str("**: ");
        out.push_str(a.when_to_use);
        out.push('\n');
    }
    for a in custom_agents {
        out.push_str("- **");
        out.push_str(&a.agent_type);
        out.push_str("**: ");
        out.push_str(&a.when_to_use);
        out.push('\n');
    }
    out.push_str(
        "\nOmit `subagent_type` to spawn a general-purpose agent \
         with no specialized system prompt.",
    );
    out
}

/// Build the agent listing (built-ins only, for backward compat).
#[must_use]
pub fn agent_listing_prompt() -> String {
    agent_listing_prompt_with_custom(&[])
}

// ---------------------------------------------------------------------------
// Frontmatter parser (simple, no YAML dep)
// ---------------------------------------------------------------------------

fn parse_agent_markdown(path: &Path) -> Result<CustomAgentDef, String> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("read error: {e}"))?;

    let (frontmatter, body) = split_frontmatter(&content)
        .ok_or_else(|| "missing --- frontmatter delimiters".to_string())?;

    let name = extract_field(&frontmatter, "name")
        .ok_or_else(|| "missing required 'name' field".to_string())?;
    let description = extract_field(&frontmatter, "description")
        .ok_or_else(|| "missing required 'description' field".to_string())?;

    let disallowed_tools = extract_list_field(&frontmatter, "disallowed_tools");
    let allowed_tools = extract_list_field(&frontmatter, "allowed_tools");
    let max_turns = extract_field(&frontmatter, "max_turns")
        .and_then(|v| v.parse::<u32>().ok());

    Ok(CustomAgentDef {
        agent_type: name,
        when_to_use: description,
        system_prompt: body.trim().to_string(),
        disallowed_tools,
        allowed_tools,
        max_turns,
        source_path: path.display().to_string(),
    })
}

/// Split `---\nfrontmatter\n---\nbody` into (frontmatter, body).
/// The closing `---` must be on its own line (prevents confusion with
/// `---` inside code blocks in the body).
fn split_frontmatter(content: &str) -> Option<(String, String)> {
    let trimmed = content.trim_start();
    if !trimmed.starts_with("---") {
        return None;
    }
    // Skip the opening "---" and one newline
    let rest = &trimmed[3..];
    let rest = rest.strip_prefix('\n').or_else(|| rest.strip_prefix("\r\n"))?;

    // Find closing "---" that sits on its own line (handles both \n and \r\n)
    let end = rest
        .match_indices("---")
        .find(|(pos, _)| {
            // Must be preceded by \n (or \r\n) — i.e., on its own line
            if *pos == 0 {
                // "---" at the very start of rest (empty frontmatter case)
                let after = &rest[3..];
                return after.is_empty() || after.starts_with('\n') || after.starts_with("\r\n");
            }
            let before = &rest[..* pos];
            let on_own_line = before.ends_with('\n');
            if !on_own_line {
                return false;
            }
            let after = &rest[pos + 3..];
            after.is_empty() || after.starts_with('\n') || after.starts_with("\r\n")
        })
        .map(|(pos, _)| pos)?;

    // Frontmatter is everything before the closing "---" line.
    // Strip trailing \r\n or \n from frontmatter.
    let frontmatter = rest[..end].trim_end_matches(['\r', '\n']).to_string();
    let body_start = end + 3; // skip "---"
    let body = if body_start < rest.len() {
        // Skip the newline(s) after closing ---
        let remaining = &rest[body_start..];
        let remaining = remaining
            .strip_prefix("\r\n")
            .or_else(|| remaining.strip_prefix('\n'))
            .unwrap_or(remaining);
        remaining.to_string()
    } else {
        String::new()
    };
    Some((frontmatter, body))
}

/// Extract a simple `key: value` field from frontmatter text.
fn extract_field(frontmatter: &str, key: &str) -> Option<String> {
    let prefix = format!("{key}:");
    for line in frontmatter.lines() {
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix(&prefix) {
            let val = rest.trim().trim_matches('"').trim_matches('\'');
            if !val.is_empty() {
                return Some(val.to_string());
            }
        }
    }
    None
}

/// Extract a `key: [a, b, c]` list field from frontmatter text.
fn extract_list_field(frontmatter: &str, key: &str) -> Vec<String> {
    let prefix = format!("{key}:");
    for line in frontmatter.lines() {
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix(&prefix) {
            let rest = rest.trim();
            // Parse [item1, item2, ...] format
            if let Some(inner) = rest.strip_prefix('[').and_then(|r| r.strip_suffix(']')) {
                return inner
                    .split(',')
                    .map(|s| s.trim().trim_matches('"').trim_matches('\'').to_string())
                    .filter(|s| !s.is_empty())
                    .collect();
            }
        }
    }
    Vec::new()
}

// ---------------------------------------------------------------------------
// Verification agent
// ---------------------------------------------------------------------------

const VERIFICATION_SYSTEM_PROMPT: &str = "\
You are a verification specialist. Your job is not to confirm the implementation works — \
it is to try to break it.

You have two documented failure patterns:
1. **Verification avoidance**: reading code, narrating what you *would* test, writing PASS, moving on.
2. **Seduced by the first 80%**: seeing a polished UI or passing tests and not probing the remaining 20%.

Your entire value is in finding the last 20%. The caller may spot-check your commands by \
re-running them. If a PASS step has no command output, your report gets rejected.

=== CRITICAL: DO NOT MODIFY THE PROJECT ===
- Do NOT create, modify, or delete any files in the project directory.
- You MAY write ephemeral test scripts to /tmp or $TMPDIR.
- Do NOT run git write operations.

=== REQUIRED STEPS ===
1. Read PIKU.md / README for build and test commands.
2. Run the build. A broken build is automatic FAIL.
3. Run the test suite. Failing tests are automatic FAIL.
4. Run linters/type-checkers if configured.
5. Apply the change-type-specific strategy below.

=== CHANGE-TYPE STRATEGIES ===
- **Code edits**: build → tests → run the changed code directly → try to break it with bad input.
- **CLI changes**: run with representative inputs → test --help → test edge inputs (empty, long, unicode).
- **Bug fixes**: reproduce original bug → verify fix → check related code for side effects.
- **Refactoring**: existing tests MUST pass unchanged → diff public API (no new/removed exports).

=== RECOGNIZE YOUR OWN RATIONALIZATIONS ===
- \"The code looks correct based on my reading\" — reading is not verification. Run it.
- \"The implementer's tests already pass\" — the implementer is an LLM. Verify independently.
- \"This is probably fine\" — probably is not verified. Run it.
- \"This would take too long\" — not your call.
If you catch yourself writing an explanation instead of a command, stop. Run the command.

=== OUTPUT FORMAT ===
Every check MUST follow this structure:

### Check: [what you're verifying]
**Command run:**
  [exact command executed]
**Output observed:**
  [copy-paste actual output — truncate if very long but keep the relevant part]
**Result: PASS** or **Result: FAIL** (Expected vs Actual)

End with exactly one of:
VERDICT: PASS
VERDICT: FAIL
VERDICT: PARTIAL
(PARTIAL only for environmental limitations — missing tool, server won't start.)";

pub const VERIFICATION_AGENT: AgentDef = AgentDef {
    agent_type: "verification",
    when_to_use: "Verify that implementation work is correct before reporting completion. \
                  Invoke after non-trivial changes (3+ file edits, API changes, bug fixes). \
                  Pass the original task, files changed, and approach taken. \
                  Returns PASS / FAIL / PARTIAL verdict with evidence.",
    system_prompt: VERIFICATION_SYSTEM_PROMPT,
    disallowed_tools: &["spawn_agent", "write_file", "edit_file"],
    allowed_tools: &[],
    max_turns: Some(30),
};

// ---------------------------------------------------------------------------
// Explorer agent
// ---------------------------------------------------------------------------

const EXPLORER_SYSTEM_PROMPT: &str = "\
You are a codebase explorer. Your job is to research and understand — not to modify anything.

Given a question or area to explore, you will:
1. Search the codebase thoroughly using read_file, grep, glob, and list_dir.
2. Trace call paths, find definitions, and understand data flows.
3. Report findings clearly: file paths, line numbers, patterns observed.

You do NOT write any files or run bash commands that have side effects. \
You are a read-only research agent. Be thorough — if you are unsure, \
keep searching rather than guessing.";

pub const EXPLORER_AGENT: AgentDef = AgentDef {
    agent_type: "explorer",
    when_to_use: "Explore and research the codebase without making changes. \
                  Use for 'where is X implemented?', 'how does Y work?', \
                  'find all usages of Z'. Returns a structured research report.",
    system_prompt: EXPLORER_SYSTEM_PROMPT,
    disallowed_tools: &["spawn_agent", "write_file", "edit_file", "bash"],
    allowed_tools: &[],
    max_turns: Some(15),
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn find_built_in_works() {
        assert!(find_built_in("verification").is_some());
        assert!(find_built_in("explorer").is_some());
        assert!(find_built_in("nonexistent").is_none());
    }

    #[test]
    fn split_frontmatter_basic() {
        let content = "---\nname: test\n---\nbody text";
        let (fm, body) = split_frontmatter(content).unwrap();
        assert_eq!(fm, "name: test");
        assert!(body.contains("body text"));
    }

    #[test]
    fn split_frontmatter_ignores_inline_dashes() {
        let content = "---\nname: test\n---\nbody with --- inside it\nand ---more--- dashes";
        let (fm, body) = split_frontmatter(content).unwrap();
        assert_eq!(fm, "name: test");
        assert!(body.contains("body with --- inside"));
    }

    #[test]
    fn extract_field_basic() {
        let fm = "name: reviewer\ndescription: Review code";
        assert_eq!(extract_field(fm, "name"), Some("reviewer".to_string()));
        assert_eq!(extract_field(fm, "description"), Some("Review code".to_string()));
        assert_eq!(extract_field(fm, "missing"), None);
    }

    #[test]
    fn extract_list_field_basic() {
        let fm = "disallowed_tools: [write_file, edit_file, bash]";
        let list = extract_list_field(fm, "disallowed_tools");
        assert_eq!(list, vec!["write_file", "edit_file", "bash"]);
    }

    #[test]
    fn extract_list_field_quoted() {
        let fm = r#"allowed_tools: ["read_file", "grep", "glob"]"#;
        let list = extract_list_field(fm, "allowed_tools");
        assert_eq!(list, vec!["read_file", "grep", "glob"]);
    }

    #[test]
    fn parse_agent_markdown_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let agent_dir = dir.path().join(".piku").join("agents");
        fs::create_dir_all(&agent_dir).unwrap();

        let agent_md = agent_dir.join("reviewer.md");
        fs::write(&agent_md, "\
---
name: reviewer
description: Review code for correctness
disallowed_tools: [write_file, edit_file]
max_turns: 25
---

You are a code reviewer. Check the diff carefully.
").unwrap();

        let agents = load_custom_agents(dir.path());
        assert_eq!(agents.len(), 1);
        assert_eq!(agents[0].agent_type, "reviewer");
        assert_eq!(agents[0].when_to_use, "Review code for correctness");
        assert_eq!(agents[0].disallowed_tools, vec!["write_file", "edit_file"]);
        assert_eq!(agents[0].max_turns, Some(25));
        assert!(agents[0].system_prompt.contains("code reviewer"));
    }

    #[test]
    fn load_custom_agents_no_dir() {
        let dir = tempfile::tempdir().unwrap();
        let agents = load_custom_agents(dir.path());
        assert!(agents.is_empty());
    }

    #[test]
    fn find_agent_custom_overrides_builtin() {
        let custom = vec![CustomAgentDef {
            agent_type: "verification".to_string(),
            when_to_use: "custom verifier".to_string(),
            system_prompt: "custom prompt".to_string(),
            disallowed_tools: vec![],
            allowed_tools: vec![],
            max_turns: Some(10),
            source_path: "test.md".to_string(),
        }];
        let found = find_agent("verification", &custom).unwrap();
        assert_eq!(found.when_to_use(), "custom verifier");
        assert_eq!(found.max_turns(), Some(10));
    }

    #[test]
    fn any_agent_def_tool_filtering() {
        let def = AnyAgentDef::BuiltIn(&EXPLORER_AGENT);
        assert!(def.is_tool_allowed("read_file"));
        assert!(!def.is_tool_allowed("bash"));
        assert!(!def.is_tool_allowed("write_file"));
    }

    #[test]
    fn any_agent_def_allowlist_precedence() {
        let custom = CustomAgentDef {
            agent_type: "narrow".to_string(),
            when_to_use: "test".to_string(),
            system_prompt: "test".to_string(),
            disallowed_tools: vec!["bash".to_string()], // should be ignored
            allowed_tools: vec!["read_file".to_string(), "grep".to_string()],
            max_turns: None,
            source_path: "test.md".to_string(),
        };
        let def = AnyAgentDef::Custom(custom);
        assert!(def.is_tool_allowed("read_file"));
        assert!(def.is_tool_allowed("grep"));
        assert!(!def.is_tool_allowed("bash"));
        assert!(!def.is_tool_allowed("write_file"));
    }

    #[test]
    fn agent_listing_includes_custom() {
        let custom = vec![CustomAgentDef {
            agent_type: "reviewer".to_string(),
            when_to_use: "Review code".to_string(),
            system_prompt: String::new(),
            disallowed_tools: vec![],
            allowed_tools: vec![],
            max_turns: None,
            source_path: String::new(),
        }];
        let listing = agent_listing_prompt_with_custom(&custom);
        assert!(listing.contains("reviewer"));
        assert!(listing.contains("Review code"));
        assert!(listing.contains("verification"));
    }

    // --- Frontmatter edge cases ---

    #[test]
    fn frontmatter_crlf_line_endings() {
        let content = "---\r\nname: test\r\n---\r\nbody with crlf";
        let (fm, body) = split_frontmatter(content).unwrap();
        assert_eq!(fm.trim(), "name: test");
        assert!(body.contains("body with crlf"));
    }

    #[test]
    fn frontmatter_crlf_closing_delimiter() {
        let content = "---\r\nname: test\r\n---\r\nbody";
        let (fm, body) = split_frontmatter(content).unwrap();
        assert!(fm.contains("name: test"));
        assert!(body.contains("body"));
    }

    #[test]
    fn frontmatter_empty_returns_some() {
        let content = "---\n---\nbody text";
        let result = split_frontmatter(content);
        assert!(result.is_some());
        let (fm, body) = result.unwrap();
        assert!(fm.is_empty());
        assert!(body.contains("body text"));
    }

    #[test]
    fn frontmatter_empty_body() {
        let content = "---\nname: minimal\ndescription: test\n---\n";
        let (fm, body) = split_frontmatter(content).unwrap();
        assert!(fm.contains("name: minimal"));
        assert!(body.trim().is_empty());
    }

    #[test]
    fn frontmatter_no_closing_delimiter_returns_none() {
        let content = "---\nname: broken\nno closing delimiter";
        assert!(split_frontmatter(content).is_none());
    }

    #[test]
    fn frontmatter_not_at_start_returns_none() {
        let content = "some text\n---\nname: test\n---\nbody";
        // The leading text means it doesn't start with ---
        assert!(split_frontmatter(content).is_none());
    }

    #[test]
    fn parse_agent_with_all_fields() {
        let dir = tempfile::tempdir().unwrap();
        let agent_dir = dir.path().join(".piku").join("agents");
        fs::create_dir_all(&agent_dir).unwrap();

        fs::write(agent_dir.join("full.md"), "\
---
name: full-agent
description: An agent with all fields
allowed_tools: [read_file, grep]
max_turns: 10
---

You are a test agent.
").unwrap();

        let agents = load_custom_agents(dir.path());
        assert_eq!(agents.len(), 1);
        let a = &agents[0];
        assert_eq!(a.agent_type, "full-agent");
        assert_eq!(a.allowed_tools, vec!["read_file", "grep"]);
        assert_eq!(a.max_turns, Some(10));

        let def = AnyAgentDef::Custom(a.clone());
        assert!(def.is_tool_allowed("read_file"));
        assert!(def.is_tool_allowed("grep"));
        assert!(!def.is_tool_allowed("bash"));
    }

    #[test]
    fn parse_agent_missing_description_skipped() {
        let dir = tempfile::tempdir().unwrap();
        let agent_dir = dir.path().join(".piku").join("agents");
        fs::create_dir_all(&agent_dir).unwrap();

        fs::write(agent_dir.join("bad.md"), "\
---
name: no-desc
---

body
").unwrap();

        let agents = load_custom_agents(dir.path());
        assert!(agents.is_empty()); // should be skipped
    }

    #[test]
    fn non_md_files_ignored() {
        let dir = tempfile::tempdir().unwrap();
        let agent_dir = dir.path().join(".piku").join("agents");
        fs::create_dir_all(&agent_dir).unwrap();

        fs::write(agent_dir.join("notes.txt"), "not an agent").unwrap();
        fs::write(agent_dir.join("config.yaml"), "also not").unwrap();

        let agents = load_custom_agents(dir.path());
        assert!(agents.is_empty());
    }
}
