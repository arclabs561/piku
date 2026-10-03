//! Surface-neutral projections for durable operator interaction receipts.

use std::fmt::Write;
use std::path::{Component, Path};

use piku_runtime::{RunContentRef as ContentRef, RunEvent, RunEventEnvelope};
use serde::Serialize;

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ReceiptView {
    pub short: String,
    pub canonical: String,
    pub event_sequence: u64,
    pub command: String,
    pub action_id: Option<String>,
    pub result: Option<ContentRef>,
    pub is_error: bool,
}

#[must_use]
pub fn receipts(events: &[RunEventEnvelope]) -> Vec<ReceiptView> {
    let mut projected = Vec::new();
    for event in events {
        let RunEvent::OperatorCommand {
            command,
            action_id,
            result,
            is_error,
        } = &event.event
        else {
            continue;
        };
        projected.push(ReceiptView {
            short: format!("##{}", projected.len() + 1),
            canonical: format!("ref:run:{}:receipt:{}", event.session_id, event.sequence),
            event_sequence: event.sequence,
            command: command.clone(),
            action_id: action_id.clone(),
            result: result.clone(),
            is_error: *is_error,
        });
    }
    projected
}

#[must_use]
pub fn select(receipts: Vec<ReceiptView>, reference: Option<&str>) -> Option<Vec<ReceiptView>> {
    let Some(reference) = reference else {
        return Some(receipts);
    };
    receipts
        .into_iter()
        .find(|receipt| reference == receipt.short || reference == receipt.canonical)
        .map(|receipt| vec![receipt])
}

pub fn raw_result(
    events: &[RunEventEnvelope],
    record_path: &Path,
    reference: &str,
) -> std::io::Result<String> {
    let receipt = select(receipts(events), Some(reference))
        .and_then(|mut receipts| receipts.pop())
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::NotFound, "unknown receipt"))?;
    let content = receipt.result.ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "receipt has no retained result",
        )
    })?;
    read_content(&content, record_path, "receipt result")
}

#[must_use]
pub fn render_receipts_text(receipts: &[ReceiptView]) -> String {
    let mut output = String::new();
    for receipt in receipts {
        let action = receipt.action_id.as_deref().unwrap_or("local command");
        let state = if receipt.is_error {
            "failed"
        } else {
            "completed"
        };
        let _ = write!(
            output,
            "{} · {} · {} · {}",
            receipt.short, receipt.command, action, state
        );
        if let Some(result) = &receipt.result {
            let _ = write!(output, " · {}", content_preview(result));
        } else {
            output.push_str(" · result not retained");
        }
        let _ = writeln!(output, " · {}", receipt.canonical);
    }
    output
}

/// Render one selected receipt for terminal reopening. Inline result text is
/// deliberately expanded here; the activity list remains a compact index.
#[must_use]
pub fn render_receipt_text(receipt: &ReceiptView) -> String {
    let action = receipt.action_id.as_deref().unwrap_or("local command");
    let state = if receipt.is_error {
        "failed"
    } else {
        "completed"
    };
    let mut output = format!(
        "{} · {} · {} · {} · {}\n",
        receipt.short, receipt.command, action, state, receipt.canonical
    );
    match &receipt.result {
        Some(ContentRef::Inline { text }) => {
            output.push_str("result:\n");
            output.push_str(text);
            if !text.ends_with('\n') {
                output.push('\n');
            }
        }
        Some(other) => {
            let _ = writeln!(output, "result: {}", content_preview(other));
        }
        None => output.push_str("result not retained\n"),
    }
    output
}

fn read_content(content: &ContentRef, record_path: &Path, label: &str) -> std::io::Result<String> {
    match content {
        ContentRef::Inline { text } => Ok(text.clone()),
        ContentRef::Unavailable { reason } => Err(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            format!("{label} is unavailable: {reason}"),
        )),
        ContentRef::Artifact(artifact) => {
            if artifact.relative_path.is_absolute()
                || artifact
                    .relative_path
                    .components()
                    .any(|component| !matches!(component, Component::Normal(_)))
            {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "artifact path escapes the run directory",
                ));
            }
            let parent = record_path.parent().ok_or_else(|| {
                std::io::Error::new(std::io::ErrorKind::InvalidInput, "run record has no parent")
            })?;
            std::fs::read_to_string(parent.join(&artifact.relative_path))
        }
    }
}

fn content_preview(content: &ContentRef) -> String {
    match content {
        ContentRef::Inline { text } => text
            .lines()
            .next()
            .unwrap_or("")
            .chars()
            .take(120)
            .collect(),
        ContentRef::Artifact(artifact) => {
            format!(
                "artifact {} ({} bytes)",
                artifact.relative_path.display(),
                artifact.bytes
            )
        }
        ContentRef::Unavailable { reason } => format!("unavailable: {reason}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use piku_runtime::{RunEventScope, RUN_RECORD_SCHEMA_VERSION};

    #[test]
    fn projects_receipts_without_creating_cells() {
        let events = [RunEventEnvelope {
            schema_version: RUN_RECORD_SCHEMA_VERSION,
            sequence: 7,
            recorded_at_ms: 0,
            session_id: "session-1".to_string(),
            scope: RunEventScope::Run,
            event: RunEvent::OperatorCommand {
                command: "/status".to_string(),
                action_id: Some("tui.status".to_string()),
                result: Some(ContentRef::Inline {
                    text: "Status: ready".to_string(),
                }),
                is_error: false,
            },
        }];
        let projected = receipts(&events);
        assert_eq!(projected[0].short, "##1");
        assert_eq!(projected[0].canonical, "ref:run:session-1:receipt:7");
        assert!(render_receipts_text(&projected).contains("/status"));
        assert!(render_receipt_text(&projected[0]).contains("Status: ready"));
        assert_eq!(
            raw_result(&events, Path::new("run.jsonl"), "##1").unwrap(),
            "Status: ready"
        );
    }
}
