//! Shared, deterministic workbench actions.
//!
//! A surface may have its own input grammar, but it must not reimplement the
//! meaning of an action or fabricate a separate result. This module is kept
//! deliberately small while the action catalog is established.

use std::io;
use std::path::Path;

use piku_runtime::RunEventEnvelope;

/// A typed, read-only action currently shared by the terminal and CLI.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Action {
    InspectCells {
        open_in_pager: bool,
    },
    InspectCell {
        reference: Option<String>,
        open_in_pager: bool,
    },
    InspectReceipts {
        open_in_pager: bool,
    },
    InspectReceipt {
        reference: Option<String>,
        open_in_pager: bool,
    },
}

impl Action {
    #[must_use]
    pub const fn id(&self) -> &'static str {
        match self {
            Self::InspectCells { .. } => "inspect.cells",
            Self::InspectCell { .. } => "inspect.cell",
            Self::InspectReceipts { .. } => "inspect.receipts",
            Self::InspectReceipt { .. } => "inspect.receipt",
        }
    }
}

/// A surface-neutral action result. Renderers choose how to display each
/// block; `ExactOutput` is suitable for a pager or pipe and must not be
/// replaced by a preview. `Cells` intentionally retains the structured
/// projection so a web response, a JSON CLI result, and terminal text all use
/// the same evidence.
#[derive(Debug, Clone, PartialEq)]
pub struct ActionReport {
    pub action_id: &'static str,
    pub blocks: Vec<ActionBlock>,
}

impl ActionReport {
    pub fn require_cells(self) -> io::Result<Vec<piku_runtime::CellView>> {
        match self.blocks.as_slice() {
            [ActionBlock::Cells(cells)] => Ok(cells.clone()),
            _ => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "action did not produce one cell block",
            )),
        }
    }

    pub fn require_exact_output(self) -> io::Result<String> {
        match self.blocks.as_slice() {
            [ActionBlock::ExactOutput(text)] => Ok(text.clone()),
            _ => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "action did not produce one exact-output block",
            )),
        }
    }

    pub fn require_receipts(self) -> io::Result<Vec<crate::receipt_view::ReceiptView>> {
        match self.blocks.as_slice() {
            [ActionBlock::Receipts(receipts)] => Ok(receipts.clone()),
            _ => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "action did not produce one receipt block",
            )),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ActionBlock {
    Cells(Vec<piku_runtime::CellView>),
    Receipts(Vec<crate::receipt_view::ReceiptView>),
    ExactOutput(String),
}

/// Parse the small part of the TUI slash grammar that has a shared action
/// implementation. `Ok(None)` leaves unrelated TUI commands to their local
/// parser.
pub fn parse_tui_slash(input: &str) -> Result<Option<Action>, &'static str> {
    let input = input.trim();
    if let Some(argument) = input.strip_prefix("/cells") {
        if argument.is_empty() {
            return Ok(Some(Action::InspectCells {
                open_in_pager: false,
            }));
        }
        if argument == " --pager" {
            return Ok(Some(Action::InspectCells {
                open_in_pager: true,
            }));
        }
        if argument.starts_with(char::is_whitespace) {
            return Err("usage: /cells [--pager]");
        }
    }
    if let Some(argument) = input.strip_prefix("/activity") {
        if argument.is_empty() {
            return Ok(Some(Action::InspectReceipts {
                open_in_pager: false,
            }));
        }
        if argument == " --pager" {
            return Ok(Some(Action::InspectReceipts {
                open_in_pager: true,
            }));
        }
        if argument.starts_with(char::is_whitespace) {
            return Err("usage: /activity [--pager]");
        }
    }
    if let Some(argument) = input.strip_prefix("/receipt") {
        if !argument.is_empty() && !argument.starts_with(char::is_whitespace) {
            return Ok(None);
        }
        let mut parts = argument.split_whitespace();
        return match (parts.next(), parts.next(), parts.next()) {
            (None, None, None) => Ok(Some(Action::InspectReceipt {
                reference: None,
                open_in_pager: false,
            })),
            (Some("--pager"), None, None) => Err("usage: /receipt ##N [--pager]"),
            (Some(reference), None, None) => Ok(Some(Action::InspectReceipt {
                reference: Some(reference.to_string()),
                open_in_pager: false,
            })),
            (Some(reference), Some("--pager"), None) => Ok(Some(Action::InspectReceipt {
                reference: Some(reference.to_string()),
                open_in_pager: true,
            })),
            _ => Err("usage: /receipt ##N [--pager]"),
        };
    }
    let Some(argument) = input.strip_prefix("/cell") else {
        return Ok(None);
    };
    if !argument.is_empty() && !argument.starts_with(char::is_whitespace) {
        return Ok(None);
    }

    let mut parts = argument.split_whitespace();
    match (parts.next(), parts.next(), parts.next()) {
        (None, None, None) => Ok(Some(Action::InspectCell {
            reference: None,
            open_in_pager: false,
        })),
        (Some("--pager"), None, None) => Err("usage: /cell @@N [--pager]"),
        (Some(reference), None, None) => Ok(Some(Action::InspectCell {
            reference: Some(reference.to_string()),
            open_in_pager: false,
        })),
        (Some(reference), Some("--pager"), None) => Ok(Some(Action::InspectCell {
            reference: Some(reference.to_string()),
            open_in_pager: true,
        })),
        _ => Err("usage: /cell @@N [--pager]"),
    }
}

/// Execute a shared inspection action against one durable run record.
pub fn execute(
    action: &Action,
    events: &[RunEventEnvelope],
    record_path: &Path,
) -> io::Result<ActionReport> {
    match action {
        Action::InspectCells { .. } => Ok(ActionReport {
            action_id: action.id(),
            blocks: vec![ActionBlock::Cells(crate::cell_view::cells(events))],
        }),
        Action::InspectCell {
            reference,
            open_in_pager,
        } => {
            if *open_in_pager {
                let reference = reference.as_deref().ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidInput, "usage: /cell @@N --pager")
                })?;
                return Ok(ActionReport {
                    action_id: action.id(),
                    blocks: vec![ActionBlock::ExactOutput(crate::cell_view::raw_output(
                        events,
                        record_path,
                        reference,
                    )?)],
                });
            }
            Ok(ActionReport {
                action_id: action.id(),
                blocks: vec![ActionBlock::Cells(
                    crate::cell_view::select(crate::cell_view::cells(events), reference.as_deref())
                        .unwrap_or_default(),
                )],
            })
        }
        Action::InspectReceipts { .. } => Ok(ActionReport {
            action_id: action.id(),
            blocks: vec![ActionBlock::Receipts(crate::receipt_view::receipts(events))],
        }),
        Action::InspectReceipt {
            reference,
            open_in_pager,
        } => {
            if *open_in_pager {
                let reference = reference.as_deref().ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidInput, "usage: /receipt ##N --pager")
                })?;
                return Ok(ActionReport {
                    action_id: action.id(),
                    blocks: vec![ActionBlock::ExactOutput(crate::receipt_view::raw_result(
                        events,
                        record_path,
                        reference,
                    )?)],
                });
            }
            Ok(ActionReport {
                action_id: action.id(),
                blocks: vec![ActionBlock::Receipts(
                    crate::receipt_view::select(
                        crate::receipt_view::receipts(events),
                        reference.as_deref(),
                    )
                    .unwrap_or_default(),
                )],
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use piku_runtime::{
        RunContentRef, RunEvent, RunEventEnvelope, RunEventScope, RUN_RECORD_SCHEMA_VERSION,
    };

    #[test]
    fn parses_inspection_actions_without_claiming_other_slash_commands() {
        assert_eq!(
            parse_tui_slash("/cells"),
            Ok(Some(Action::InspectCells {
                open_in_pager: false,
            }))
        );
        assert_eq!(
            parse_tui_slash("/cells --pager"),
            Ok(Some(Action::InspectCells {
                open_in_pager: true,
            }))
        );
        assert_eq!(
            parse_tui_slash("/cell @@12 --pager"),
            Ok(Some(Action::InspectCell {
                reference: Some("@@12".to_string()),
                open_in_pager: true,
            }))
        );
        assert_eq!(
            parse_tui_slash("/activity"),
            Ok(Some(Action::InspectReceipts {
                open_in_pager: false,
            }))
        );
        assert_eq!(
            parse_tui_slash("/receipt ##3 --pager"),
            Ok(Some(Action::InspectReceipt {
                reference: Some("##3".to_string()),
                open_in_pager: true,
            }))
        );
        assert_eq!(parse_tui_slash("/actor primary"), Ok(None));
        assert_eq!(parse_tui_slash("/cellular"), Ok(None));
        assert!(parse_tui_slash("/cells extra").is_err());
        assert!(parse_tui_slash("/cell --pager").is_err());
        assert!(parse_tui_slash("/activity extra").is_err());
        assert!(parse_tui_slash("/receipt --pager").is_err());
    }

    #[test]
    fn inspection_executor_keeps_cells_and_exact_output_distinct() {
        let events = [RunEventEnvelope {
            schema_version: RUN_RECORD_SCHEMA_VERSION,
            sequence: 0,
            recorded_at_ms: 0,
            session_id: "session-1".to_string(),
            scope: RunEventScope::Run,
            event: RunEvent::ShellCommand {
                command: "printf hello".to_string(),
                cwd: std::path::PathBuf::from("/workspace"),
                exit_code: Some(0),
                output: RunContentRef::Inline {
                    text: "hello\n".to_string(),
                },
            },
        }];
        let path = Path::new("/workspace/run.jsonl");

        let cells = execute(
            &Action::InspectCells {
                open_in_pager: false,
            },
            &events,
            path,
        )
        .unwrap()
        .require_cells()
        .unwrap();
        assert_eq!(cells.len(), 1);
        assert_eq!(cells[0].id.short, "@@1");
        let rendered = crate::cell_view::render_cells_text(&cells);
        assert!(rendered.contains("@@1"));
        assert!(rendered.contains("shell_command"));

        let exact = execute(
            &Action::InspectCell {
                reference: Some("@@1".to_string()),
                open_in_pager: true,
            },
            &events,
            path,
        )
        .unwrap()
        .require_exact_output()
        .unwrap();
        assert_eq!(exact, "hello\n");
    }
}
