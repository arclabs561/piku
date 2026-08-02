//! Structured decisions for the interactive terminal playground.

use std::time::Duration;

use serde_json::Value;

use super::{Action, SpecialKey};

/// The next physical terminal operation selected by the user-agent.
#[derive(Debug, Clone)]
pub enum PlaygroundDecision {
    Act(Action),
    Quit,
}

/// Parse an LLM response into one bounded terminal operation.
///
/// The legacy `send` variant remains supported so existing personas and
/// providers can participate while the playground learns keyboard-level use.
#[must_use]
pub fn parse_decision(value: &Value) -> PlaygroundDecision {
    let action = &value["next_action"];
    match action["type"].as_str() {
        Some("send") => PlaygroundDecision::Act(Action::Submit(
            action["message"].as_str().unwrap_or("continue").to_string(),
        )),
        Some("type") => PlaygroundDecision::Act(Action::TypeString {
            text: action["text"].as_str().unwrap_or_default().to_string(),
            delay_ms: action["delay_ms"].as_u64().unwrap_or(0).min(250),
        }),
        Some("key") => parse_key(action["key"].as_str())
            .map(|key| PlaygroundDecision::Act(Action::Key(key)))
            .unwrap_or(PlaygroundDecision::Quit),
        Some("observe") => PlaygroundDecision::Act(Action::Observe),
        Some("wait") => PlaygroundDecision::Act(Action::Wait(Duration::from_millis(
            action["ms"].as_u64().unwrap_or(250).clamp(10, 5_000),
        ))),
        _ => PlaygroundDecision::Quit,
    }
}

fn parse_key(value: Option<&str>) -> Option<SpecialKey> {
    match value {
        Some("enter") => Some(SpecialKey::Enter),
        Some("tab") => Some(SpecialKey::Tab),
        Some("escape") => Some(SpecialKey::Escape),
        Some("backspace") => Some(SpecialKey::Backspace),
        Some("delete") => Some(SpecialKey::Delete),
        Some("arrow_up") => Some(SpecialKey::ArrowUp),
        Some("arrow_down") => Some(SpecialKey::ArrowDown),
        Some("arrow_left") => Some(SpecialKey::ArrowLeft),
        Some("arrow_right") => Some(SpecialKey::ArrowRight),
        Some("home") => Some(SpecialKey::Home),
        Some("end") => Some(SpecialKey::End),
        Some("ctrl_c") => Some(SpecialKey::CtrlC),
        Some("ctrl_d") => Some(SpecialKey::CtrlD),
        Some("ctrl_l") => Some(SpecialKey::CtrlL),
        Some("ctrl_a") => Some(SpecialKey::CtrlA),
        Some("ctrl_e") => Some(SpecialKey::CtrlE),
        Some("ctrl_w") => Some(SpecialKey::CtrlW),
        Some("ctrl_u") => Some(SpecialKey::CtrlU),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_legacy_send_as_submit() {
        let decision = parse_decision(&serde_json::json!({
            "next_action": {"type": "send", "message": "inspect src"}
        }));
        assert!(
            matches!(decision, PlaygroundDecision::Act(Action::Submit(text)) if text == "inspect src")
        );
    }

    #[test]
    fn bounds_wait_duration() {
        let decision = parse_decision(&serde_json::json!({
            "next_action": {"type": "wait", "ms": 99_999}
        }));
        assert!(
            matches!(decision, PlaygroundDecision::Act(Action::Wait(wait)) if wait == Duration::from_secs(5))
        );
    }

    #[test]
    fn rejects_unknown_key() {
        let decision = parse_decision(&serde_json::json!({
            "next_action": {"type": "key", "key": "meta_space"}
        }));
        assert!(matches!(decision, PlaygroundDecision::Quit));
    }
}
