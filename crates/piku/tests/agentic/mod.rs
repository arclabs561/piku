/// Reusable infrastructure for agentic user testing.
///
/// Module hierarchy:
///   types     — Action, SpecialKey, ScreenSnapshot, Bug, Severity, Finding
///   terminal  — TerminalObserver (persistent VT100 parser)
///   pty       — PtyHandle (raw byte-level PTY I/O)
///   workspace — WorkspaceObserver (filesystem side-effect detection)
///   memory    — ConversationMemory (cross-turn context)
///   checks    — deterministic_checks (cursor, prompt, footer, corruption)
pub mod checks;
pub mod memory;
pub mod pty;
pub mod terminal;
pub mod types;
pub mod workspace;

pub use checks::deterministic_checks;
pub use memory::{ConversationMemory, TurnSummary};
pub use pty::PtyHandle;
pub use terminal::TerminalObserver;
pub use types::*;
pub use workspace::{WorkspaceDiff, WorkspaceObserver};
