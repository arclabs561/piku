//! Text and JSON helpers built on the runtime's surface-neutral cells.

use std::fmt::Write;
use std::path::{Component, Path};

use piku_runtime::{project_cells, CellView, RunContentRef as ContentRef, RunEventEnvelope};

#[must_use]
pub fn cells(events: &[RunEventEnvelope]) -> Vec<CellView> {
    project_cells(events)
}

#[must_use]
pub fn select(cells: Vec<CellView>, reference: Option<&str>) -> Option<Vec<CellView>> {
    let Some(reference) = reference else {
        return Some(cells);
    };
    let cell = cells.into_iter().find(|cell| {
        reference == cell.id.short
            || reference == cell.id.canonical
            || legacy_cell_alias(reference, &cell.id.short)
            || legacy_canonical_alias(reference, &cell.id.canonical)
    })?;
    Some(vec![cell])
}

fn legacy_cell_alias(reference: &str, short: &str) -> bool {
    reference
        .strip_prefix("@c")
        .zip(short.strip_prefix("@@"))
        .is_some_and(|(legacy, current)| legacy == current)
}

fn legacy_canonical_alias(reference: &str, canonical: &str) -> bool {
    canonical
        .strip_prefix("ref:")
        .is_some_and(|legacy| legacy == reference)
}

/// Read one cell's durable output exactly enough for terminal piping or a
/// pager. Native terminal passthrough deliberately returns an error instead of
/// inventing bytes that were never captured.
pub fn raw_output(
    events: &[RunEventEnvelope],
    record_path: &Path,
    reference: &str,
) -> std::io::Result<String> {
    let cell = select(cells(events), Some(reference))
        .and_then(|mut cells| cells.pop())
        .ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::NotFound, "unknown cell reference")
        })?;
    let content = cell
        .items
        .iter()
        .find_map(|item| item.content.as_ref())
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::NotFound, "cell has no output"))?;
    match content {
        ContentRef::Inline { text } => Ok(text.clone()),
        ContentRef::Unavailable { reason } => Err(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            format!("cell output is unavailable: {reason}"),
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

#[must_use]
pub fn render_text(events: &[RunEventEnvelope], selected: Option<&str>) -> String {
    let Some(cells) = select(cells(events), selected) else {
        return String::new();
    };
    render_cells_text(&cells)
}

/// Render a previously selected structured cell projection for a terminal.
/// Keeping selection and rendering separate lets all surfaces share an action
/// result without treating terminal text as the canonical representation.
#[must_use]
pub fn render_cells_text(cells: &[CellView]) -> String {
    let mut output = String::new();
    for cell in cells {
        let _ = writeln!(
            output,
            "{} · {} · {:?} · {}",
            cell.id.short, cell.actor.label, cell.status, cell.id.canonical
        );
        if let Some(input) = &cell.input {
            let _ = writeln!(output, "  prompt: {}", content_preview(input));
        }
        for item in &cell.items {
            let _ = write!(output, "  @e{} {}", item.event_sequence, item.kind);
            if let Some(content) = &item.content {
                let _ = write!(output, " · {}", content_preview(content));
            }
            if let Some(detail) = &item.detail {
                let _ = write!(output, " · {detail}");
            }
            output.push('\n');
        }
    }
    output
}

fn content_preview(content: &ContentRef) -> String {
    match content {
        ContentRef::Inline { text } => text
            .lines()
            .next()
            .unwrap_or("")
            .chars()
            .take(160)
            .collect(),
        ContentRef::Artifact(artifact) => format!(
            "artifact {} ({} bytes)",
            artifact.relative_path.display(),
            artifact.bytes
        ),
        ContentRef::Unavailable { reason } => format!("unavailable: {reason}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use piku_runtime::{RunContentRef, RunEvent, RunEventEnvelope, RUN_RECORD_SCHEMA_VERSION};

    #[test]
    fn selection_rejects_unknown_reference() {
        let events = [RunEventEnvelope {
            schema_version: RUN_RECORD_SCHEMA_VERSION,
            sequence: 9,
            recorded_at_ms: 0,
            session_id: "session-1".to_string(),
            scope: piku_runtime::RunEventScope::Turn {
                turn_id: "turn-1".to_string(),
            },
            event: RunEvent::TurnStarted {
                provider: None,
                model: "model".to_string(),
                input: RunContentRef::Inline {
                    text: "prompt".to_string(),
                },
            },
        }];
        assert!(select(cells(&events), Some("@@2")).is_none());
        assert_eq!(select(cells(&events), Some("@@1")).unwrap().len(), 1);
        assert_eq!(select(cells(&events), Some("@c1")).unwrap().len(), 1);
        assert_eq!(
            select(cells(&events), Some("run:session-1:cell:9"))
                .unwrap()
                .len(),
            1
        );
    }

    #[test]
    fn raw_output_refuses_unavailable_terminal_passthrough() {
        let events = [RunEventEnvelope {
            schema_version: RUN_RECORD_SCHEMA_VERSION,
            sequence: 0,
            recorded_at_ms: 0,
            session_id: "session-1".to_string(),
            scope: piku_runtime::RunEventScope::Run,
            event: RunEvent::ShellCommand {
                command: "less log.txt".to_string(),
                cwd: std::path::PathBuf::from("/workspace"),
                exit_code: Some(0),
                output: RunContentRef::Unavailable {
                    reason: "native terminal passthrough".to_string(),
                },
            },
        }];
        let error = raw_output(&events, Path::new("run.jsonl"), "@@1").unwrap_err();
        assert_eq!(error.kind(), std::io::ErrorKind::Unsupported);
    }
}
