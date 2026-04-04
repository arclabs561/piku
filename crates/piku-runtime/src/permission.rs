use piku_tools::Destructiveness;

/// Outcome of a permission check.
#[derive(Debug, Clone)]
pub enum PermissionOutcome {
    Allow,
    Deny { reason: String },
}

/// Callback for interactive permission prompts.
/// Implement this to wire in the TUI confirmation UI.
pub trait PermissionPrompter: Send + Sync {
    fn decide(&self, req: &PermissionRequest) -> PermissionOutcome;
}

#[derive(Debug, Clone)]
pub struct PermissionRequest {
    pub tool_name: String,
    pub params: serde_json::Value,
    pub destructiveness: Destructiveness,
    /// Human-readable description of what the tool call will do.
    pub description: String,
}

/// Always-allow prompter (used in single-shot / non-interactive mode).
pub struct AllowAll;

impl PermissionPrompter for AllowAll {
    fn decide(&self, _req: &PermissionRequest) -> PermissionOutcome {
        PermissionOutcome::Allow
    }
}

/// Always-deny prompter (used in tests).
#[allow(dead_code)]
pub struct DenyAll;

impl PermissionPrompter for DenyAll {
    fn decide(&self, req: &PermissionRequest) -> PermissionOutcome {
        PermissionOutcome::Deny {
            reason: format!("DenyAll: {} is not permitted", req.tool_name),
        }
    }
}

/// Evaluate whether a tool call requires permission and what kind.
///
/// Tier 1: static heuristics (free).
/// Tier 2: for `Likely`, call the prompter (which in TUI mode will run the
///         cheap classifier; in single-shot mode is `AllowAll`).
/// Tier 3: for `Definite`, always call the prompter.
pub fn check_permission(
    tool_name: &str,
    params: &serde_json::Value,
    prompter: &dyn PermissionPrompter,
) -> PermissionOutcome {
    let destructiveness = piku_tools::tool_destructiveness(tool_name, params);

    match destructiveness {
        Destructiveness::Safe => PermissionOutcome::Allow,
        Destructiveness::Likely | Destructiveness::Definite => {
            let description = describe_tool_call(tool_name, params);
            prompter.decide(&PermissionRequest {
                tool_name: tool_name.to_string(),
                params: params.clone(),
                destructiveness,
                description,
            })
        }
    }
}

/// Build a short human-readable description of a tool call for display.
fn describe_tool_call(tool_name: &str, params: &serde_json::Value) -> String {
    match tool_name {
        "bash" => {
            let cmd = params
                .get("command")
                .and_then(|v| v.as_str())
                .unwrap_or("?");
            let desc = params.get("description").and_then(|v| v.as_str());
            if let Some(d) = desc {
                format!("bash: {d} — `{cmd}`")
            } else {
                format!("bash: `{cmd}`")
            }
        }
        "write_file" => {
            let path = params.get("path").and_then(|v| v.as_str()).unwrap_or("?");
            format!("write_file: {path}")
        }
        "edit_file" => {
            let path = params.get("path").and_then(|v| v.as_str()).unwrap_or("?");
            format!("edit_file: {path}")
        }
        _ => format!("{tool_name}: {params}"),
    }
}
