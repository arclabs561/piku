//! Surface-neutral, inspectable projections of durable run records.
//!
//! A cell is not another mutable transcript format. It is a stable view over
//! one turn and the events that occurred within it, so every terminal, web,
//! and export surface can point at the same retained evidence.

use serde::{Deserialize, Serialize};

use std::collections::HashMap;

use crate::{RunContentRef as ContentRef, RunEvent, RunEventEnvelope};

/// A durable address for an inspectable run cell.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CellId {
    /// Portable reference which remains meaningful outside one terminal view.
    pub canonical: String,
    /// Readable `@@N` alias assigned by sequence order within the loaded run.
    pub short: String,
}

/// Completion state derived from terminal events, never guessed from display.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CellStatus {
    Running,
    Complete,
    Failed,
    Interrupted,
}

/// An addressable event inside a cell.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CellItem {
    pub event_sequence: u64,
    pub kind: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<ContentRef>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

/// The recorded producer of a cell. This is provider/model provenance, not a
/// claim that differently named turns share one mutable conversation state.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CellActor {
    /// Stable within a run record and safe to use for grouping/filtering.
    pub id: String,
    /// Human-readable producer label.
    pub label: String,
}

/// One agent turn, ready for a surface to render without inventing state.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CellView {
    pub id: CellId,
    pub turn_id: String,
    pub actor: CellActor,
    pub started_at_ms: u64,
    pub status: CellStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input: Option<ContentRef>,
    pub items: Vec<CellItem>,
}

/// Project ordered run events into inspectable cells.
#[must_use]
pub fn project_cells(events: &[RunEventEnvelope]) -> Vec<CellView> {
    let session_id = events
        .first()
        .map_or("unknown", |event| event.session_id.as_str());
    let mut cells = Vec::new();
    let mut cell_index_by_turn = HashMap::new();
    let actor_turns = events
        .iter()
        .filter_map(|envelope| match &envelope.event {
            RunEvent::ActorTurn {
                target_turn_id,
                actor,
            } => Some((target_turn_id.as_str(), actor)),
            _ => None,
        })
        .collect::<HashMap<_, _>>();

    for envelope in events {
        if matches!(envelope.event, RunEvent::OperatorCommand { .. }) {
            // Operator interactions remain addressable in the audit stream,
            // but navigation must not recursively manufacture output cells.
            continue;
        }
        if let RunEvent::ShellCommand {
            command,
            cwd,
            exit_code,
            output,
        } = &envelope.event
        {
            let cell_number = cells.len() + 1;
            cells.push(shell_command_cell(
                session_id,
                cell_number,
                envelope,
                command,
                cwd,
                *exit_code,
                output,
            ));
            continue;
        }
        let turn_id = envelope.scope.turn_id();
        let RunEvent::TurnStarted {
            provider,
            model,
            input,
        } = &envelope.event
        else {
            if let Some(turn_id) = turn_id {
                if let Some(&cell_index) = cell_index_by_turn.get(turn_id) {
                    if let Some(cell) = cells.get_mut(cell_index) {
                        apply_event(cell, envelope);
                    }
                }
            }
            continue;
        };

        let turn_id = turn_id.unwrap_or("unknown");
        let cell_number = cells.len() + 1;
        let cell_index = cells.len();
        cells.push(CellView {
            id: CellId {
                canonical: format!("ref:run:{session_id}:cell:{}", envelope.sequence),
                short: format!("@@{cell_number}"),
            },
            turn_id: turn_id.to_string(),
            actor: actor_turns.get(turn_id).map_or_else(
                || agent_actor(provider.as_deref(), model),
                |actor| CellActor {
                    id: actor.id.clone(),
                    label: actor.label.clone(),
                },
            ),
            started_at_ms: envelope.recorded_at_ms,
            status: CellStatus::Running,
            input: Some(input.clone()),
            items: Vec::new(),
        });
        cell_index_by_turn.insert(turn_id.to_string(), cell_index);
    }
    cells
}

fn shell_command_cell(
    session_id: &str,
    cell_number: usize,
    envelope: &RunEventEnvelope,
    command: &str,
    cwd: &std::path::Path,
    exit_code: Option<i32>,
    output: &ContentRef,
) -> CellView {
    CellView {
        id: CellId {
            canonical: format!("ref:run:{session_id}:cell:{}", envelope.sequence),
            short: format!("@@{cell_number}"),
        },
        turn_id: "shell".to_string(),
        actor: CellActor {
            id: "shell".to_string(),
            label: "shell".to_string(),
        },
        started_at_ms: envelope.recorded_at_ms,
        status: match exit_code {
            Some(0) => CellStatus::Complete,
            Some(_) => CellStatus::Failed,
            None => CellStatus::Interrupted,
        },
        input: Some(ContentRef::Inline {
            text: command.to_string(),
        }),
        items: vec![CellItem {
            event_sequence: envelope.sequence,
            kind: "shell_command".to_string(),
            content: Some(output.clone()),
            detail: Some(format!("cwd={} exit={exit_code:?}", cwd.display())),
        }],
    }
}

fn agent_actor(provider: Option<&str>, model: &str) -> CellActor {
    let provider = provider.unwrap_or("unknown");
    CellActor {
        id: format!("agent:{provider}:{model}"),
        label: format!("{provider}/{model}"),
    }
}

fn apply_event(cell: &mut CellView, envelope: &RunEventEnvelope) {
    let (kind, content, detail) = match &envelope.event {
        RunEvent::AssistantMessage { content } => {
            ("assistant_message", Some(content.clone()), None)
        }
        RunEvent::ToolStarted { name, .. } => ("tool_started", None, Some(name.clone())),
        RunEvent::ToolCompleted {
            result, is_error, ..
        } => (
            if *is_error {
                "tool_error"
            } else {
                "tool_result"
            },
            Some(result.clone()),
            None,
        ),
        RunEvent::PermissionDecision { decision, .. } => {
            ("permission_decision", None, Some(format!("{decision:?}")))
        }
        RunEvent::TurnCompleted { .. } => {
            cell.status = CellStatus::Complete;
            ("turn_completed", None, None)
        }
        RunEvent::TurnFailed { message, .. } => {
            cell.status = CellStatus::Failed;
            ("turn_failed", None, Some(message.clone()))
        }
        RunEvent::TurnCancelled { reason } => {
            cell.status = CellStatus::Interrupted;
            ("turn_cancelled", None, Some(reason.clone()))
        }
        RunEvent::Warning { message } => ("warning", None, Some(message.clone())),
        _ => return,
    };
    cell.items.push(CellItem {
        event_sequence: envelope.sequence,
        kind: kind.to_string(),
        content,
        detail,
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{RunEventScope as EventScope, RUN_RECORD_SCHEMA_VERSION};

    fn event(sequence: u64, event: RunEvent) -> RunEventEnvelope {
        RunEventEnvelope {
            schema_version: RUN_RECORD_SCHEMA_VERSION,
            sequence,
            recorded_at_ms: sequence,
            session_id: "session-1".to_string(),
            scope: EventScope::Turn {
                turn_id: "turn-1".to_string(),
            },
            event,
        }
    }

    #[test]
    fn projects_completed_turn_with_stable_reference() {
        let cells = project_cells(&[
            event(
                4,
                RunEvent::TurnStarted {
                    provider: None,
                    model: "model".to_string(),
                    input: ContentRef::Inline {
                        text: "inspect this".to_string(),
                    },
                },
            ),
            event(
                5,
                RunEvent::AssistantMessage {
                    content: ContentRef::Inline {
                        text: "done".to_string(),
                    },
                },
            ),
            event(
                6,
                RunEvent::TurnCompleted {
                    usage: None,
                    stop_reason: None,
                },
            ),
        ]);

        assert_eq!(cells.len(), 1);
        assert_eq!(cells[0].id.canonical, "ref:run:session-1:cell:4");
        assert_eq!(cells[0].id.short, "@@1");
        assert_eq!(cells[0].actor.id, "agent:unknown:model");
        assert_eq!(cells[0].status, CellStatus::Complete);
        assert_eq!(cells[0].items[0].kind, "assistant_message");
    }

    #[test]
    fn keeps_operator_commands_out_of_output_cells() {
        let mut command = event(
            4,
            RunEvent::OperatorCommand {
                command: "/cells".to_string(),
                action_id: None,
                result: None,
                is_error: false,
            },
        );
        command.scope = EventScope::Run;

        let cells = project_cells(&[command]);

        assert!(cells.is_empty());
    }

    #[test]
    fn preserves_partial_output_when_turn_is_cancelled() {
        let cells = project_cells(&[
            event(
                0,
                RunEvent::TurnStarted {
                    provider: None,
                    model: "model".to_string(),
                    input: ContentRef::Inline {
                        text: "work".to_string(),
                    },
                },
            ),
            event(
                1,
                RunEvent::AssistantMessage {
                    content: ContentRef::Inline {
                        text: "partial".to_string(),
                    },
                },
            ),
            event(
                2,
                RunEvent::TurnCancelled {
                    reason: "escape".to_string(),
                },
            ),
        ]);
        assert_eq!(cells[0].status, CellStatus::Interrupted);
        assert_eq!(cells[0].items.len(), 2);
    }

    #[test]
    fn preserves_actor_identity_when_turns_are_interleaved() {
        let mut first = event(
            0,
            RunEvent::TurnStarted {
                provider: Some("anthropic".to_string()),
                model: "claude".to_string(),
                input: ContentRef::Inline {
                    text: "investigate".to_string(),
                },
            },
        );
        first.scope = EventScope::Turn {
            turn_id: "researcher-1".to_string(),
        };
        let mut second = event(
            3,
            RunEvent::TurnStarted {
                provider: Some("openrouter".to_string()),
                model: "qwen".to_string(),
                input: ContentRef::Inline {
                    text: "challenge it".to_string(),
                },
            },
        );
        second.scope = EventScope::Turn {
            turn_id: "critic-1".to_string(),
        };

        let cells = project_cells(&[first, second]);

        assert_eq!(cells.len(), 2);
        assert_eq!(cells[0].actor.label, "anthropic/claude");
        assert_eq!(cells[1].actor.label, "openrouter/qwen");
        assert_eq!(cells[0].turn_id, "researcher-1");
        assert_eq!(cells[1].turn_id, "critic-1");
    }

    #[test]
    fn retains_interleaved_events_in_their_started_turns() {
        let mut a_start = event(
            0,
            RunEvent::TurnStarted {
                provider: None,
                model: "model-a".to_string(),
                input: ContentRef::Inline {
                    text: "A start".to_string(),
                },
            },
        );
        a_start.scope = EventScope::Turn {
            turn_id: "A".to_string(),
        };
        let mut b_start = event(
            1,
            RunEvent::TurnStarted {
                provider: None,
                model: "model-b".to_string(),
                input: ContentRef::Inline {
                    text: "B start".to_string(),
                },
            },
        );
        b_start.scope = EventScope::Turn {
            turn_id: "B".to_string(),
        };
        let mut a_message = event(
            2,
            RunEvent::AssistantMessage {
                content: ContentRef::Inline {
                    text: "A message".to_string(),
                },
            },
        );
        a_message.scope = EventScope::Turn {
            turn_id: "A".to_string(),
        };
        let mut b_tool = event(
            3,
            RunEvent::ToolCompleted {
                tool_call_id: "tool-B".to_string(),
                result: ContentRef::Inline {
                    text: "B tool".to_string(),
                },
                is_error: false,
                effects: Vec::new(),
                verification: None,
            },
        );
        b_tool.scope = EventScope::Turn {
            turn_id: "B".to_string(),
        };
        let mut a_cancel = event(
            4,
            RunEvent::TurnCancelled {
                reason: "cancelled".to_string(),
            },
        );
        a_cancel.scope = EventScope::Turn {
            turn_id: "A".to_string(),
        };
        let mut b_complete = event(
            5,
            RunEvent::TurnCompleted {
                usage: None,
                stop_reason: None,
            },
        );
        b_complete.scope = EventScope::Turn {
            turn_id: "B".to_string(),
        };

        let cells = project_cells(&[a_start, b_start, a_message, b_tool, a_cancel, b_complete]);

        assert_eq!(cells.len(), 2);
        assert_eq!(cells[0].turn_id, "A");
        assert_eq!(cells[0].id.short, "@@1");
        assert_eq!(cells[0].status, CellStatus::Interrupted);
        assert_eq!(
            cells[0]
                .items
                .iter()
                .map(|item| item.kind.as_str())
                .collect::<Vec<_>>(),
            ["assistant_message", "turn_cancelled"]
        );
        assert_eq!(cells[1].turn_id, "B");
        assert_eq!(cells[1].id.short, "@@2");
        assert_eq!(cells[1].status, CellStatus::Complete);
        assert_eq!(
            cells[1]
                .items
                .iter()
                .map(|item| item.kind.as_str())
                .collect::<Vec<_>>(),
            ["tool_result", "turn_completed"]
        );
    }
}
