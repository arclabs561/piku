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

    /// Configured policy that outranks classification and session state.
    ///
    /// `decide` is only reached by calls classified `Likely` or `Definite`,
    /// and a prompter may have already been told "allow all" for this turn.
    /// Neither is a reason to ignore a rule the operator wrote down, so a
    /// configured denial is asked for first and separately. Returning `Some`
    /// blocks the call whatever its destructiveness. Default: no configured
    /// denials.
    fn denies(&self, tool_name: &str, params: &serde_json::Value) -> Option<String> {
        let _ = (tool_name, params);
        None
    }
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

/// Evaluate whether a tool call requires permission.
///
/// Static tool heuristics allow `Safe` calls directly, before configuration
/// rules. Both `Likely` and `Definite` calls go to the configured prompter. In
/// the TUI, a prior per-turn allow-all wins; otherwise deny rules precede allow
/// rules and prompting. Writable launch turns use `AllowAll`.
pub fn check_permission(
    tool_name: &str,
    params: &serde_json::Value,
    prompter: &dyn PermissionPrompter,
) -> PermissionOutcome {
    // Policy first. Classification decides whether a call needs a human, and
    // a per-turn allow-all decides whether that human is asked again; neither
    // is a reason to run something the operator configured as denied. Asking
    // after either would let a `Safe` classification, or one earlier "allow
    // all", outrank a written rule.
    if let Some(reason) = prompter.denies(tool_name, params) {
        return PermissionOutcome::Deny { reason };
    }

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
#[cfg(test)]
mod authority_tests {
    use super::*;

    /// A prompter with one configured denial and an allow-all disposition,
    /// which is the shape a TUI has after the user says "allow all" once.
    struct DenyOneAllowRest;

    impl PermissionPrompter for DenyOneAllowRest {
        fn decide(&self, _req: &PermissionRequest) -> PermissionOutcome {
            PermissionOutcome::Allow
        }

        fn denies(&self, tool_name: &str, _params: &serde_json::Value) -> Option<String> {
            (tool_name == "write_file").then(|| "denied by settings.json rule".to_string())
        }
    }

    #[test]
    fn a_configured_denial_outranks_a_safe_classification() {
        // Creating a new file classifies as Safe, which used to return Allow
        // before any rule was consulted, so a written deny could not stop it.
        let outcome = check_permission(
            "write_file",
            &serde_json::json!({"path": "new.txt", "content": "x"}),
            &DenyOneAllowRest,
        );
        assert!(
            matches!(outcome, PermissionOutcome::Deny { .. }),
            "configured deny was bypassed: {outcome:?}"
        );
    }

    #[test]
    fn a_configured_denial_outranks_a_per_turn_allow_all() {
        // `decide` returns Allow unconditionally here, standing in for a turn
        // the user already approved wholesale. The rule still wins.
        let outcome = check_permission(
            "write_file",
            &serde_json::json!({"path": "any.txt", "content": "x"}),
            &DenyOneAllowRest,
        );
        assert!(matches!(outcome, PermissionOutcome::Deny { .. }));
    }

    #[test]
    fn an_unmatched_call_is_unaffected() {
        let outcome = check_permission(
            "read_file",
            &serde_json::json!({"path": "a.txt"}),
            &DenyOneAllowRest,
        );
        assert!(matches!(outcome, PermissionOutcome::Allow));
    }
}
