//! Deterministic quality audit for durable run records.
//!
//! These measurements describe evidence Piku retained. They are deliberately
//! a vector rather than a single score: collapsing lifecycle completeness,
//! context disclosure, permissions, and artifact availability into one number
//! would make the easiest dimension a target for metric gaming.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::run_record::{
    ContentChange, ContentRef, EventScope, RunDisposition, RunEvent, RunEventEnvelope, ToolEffect,
    UsageRecord, VerificationRecord, VerificationStatus,
};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunAudit {
    pub event_count: usize,
    pub turn_count: usize,
    pub completed_turn_count: usize,
    pub context_build_count: usize,
    pub compaction_count: usize,
    pub tool_calls_started: usize,
    pub tool_calls_completed: usize,
    pub tool_calls_with_permission_decision: usize,
    pub tool_effect_count: usize,
    pub files_created: usize,
    pub files_modified: usize,
    pub file_writes_unchanged: usize,
    pub file_writes_unknown: usize,
    pub verification_count: usize,
    pub verification_passed: usize,
    pub verification_failed: usize,
    pub verification_indeterminate: usize,
    pub user_disposition_count: usize,
    pub latest_user_disposition: Option<RunDisposition>,
    pub context: ContextAudit,
    pub content: ContentAudit,
    pub usage: UsageRecord,
    pub findings: Vec<RunAuditFinding>,
}

impl RunAudit {
    #[must_use]
    pub fn is_structurally_complete(&self) -> bool {
        self.findings
            .iter()
            .all(|finding| finding.severity != AuditSeverity::Error)
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContextAudit {
    pub system_sections_selected: usize,
    pub system_sections_excluded: usize,
    pub messages_selected: usize,
    pub messages_excluded: usize,
    pub tools_selected: usize,
    pub tools_excluded: usize,
    pub empty_disposition_reasons: usize,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContentAudit {
    pub inline_items: usize,
    pub inline_bytes: u64,
    pub artifact_items: usize,
    pub artifact_bytes: u64,
    pub unavailable_items: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunAuditFinding {
    pub severity: AuditSeverity,
    pub code: String,
    pub message: String,
    pub sequences: Vec<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuditSeverity {
    Error,
    Warning,
}

#[derive(Default)]
struct TurnState {
    started: Option<u64>,
    completed: Option<u64>,
    context_builds: usize,
}

struct ToolState<'a> {
    turn_id: &'a str,
    started: u64,
    permission: Option<u64>,
    completed: Option<u64>,
}

/// Audit a record without consulting a model or presentation surface.
#[must_use]
pub fn audit_run_record(events: &[RunEventEnvelope]) -> RunAudit {
    let mut auditor = Auditor::new(events.len());
    for (index, envelope) in events.iter().enumerate() {
        auditor.observe(index, envelope);
    }
    auditor.finish()
}

struct Auditor<'a> {
    audit: RunAudit,
    turns: BTreeMap<&'a str, TurnState>,
    tools: BTreeMap<&'a str, ToolState<'a>>,
}

impl<'a> Auditor<'a> {
    fn new(event_count: usize) -> Self {
        Self {
            audit: RunAudit {
                event_count,
                turn_count: 0,
                completed_turn_count: 0,
                context_build_count: 0,
                compaction_count: 0,
                tool_calls_started: 0,
                tool_calls_completed: 0,
                tool_calls_with_permission_decision: 0,
                tool_effect_count: 0,
                files_created: 0,
                files_modified: 0,
                file_writes_unchanged: 0,
                file_writes_unknown: 0,
                verification_count: 0,
                verification_passed: 0,
                verification_failed: 0,
                verification_indeterminate: 0,
                user_disposition_count: 0,
                latest_user_disposition: None,
                context: ContextAudit::default(),
                content: ContentAudit::default(),
                usage: UsageRecord::default(),
                findings: Vec::new(),
            },
            turns: BTreeMap::new(),
            tools: BTreeMap::new(),
        }
    }

    fn observe(&mut self, index: usize, envelope: &'a RunEventEnvelope) {
        self.audit_order(index, envelope);
        let turn_id = match (&envelope.scope, &envelope.event) {
            (EventScope::Run, RunEvent::UserDisposition { disposition, note }) => {
                self.audit.user_disposition_count += 1;
                self.audit.latest_user_disposition = Some(*disposition);
                audit_content(&mut self.audit.content, note);
                return;
            }
            (EventScope::Run, _) => {
                self.add_finding(
                    AuditSeverity::Error,
                    "turn_event_has_run_scope",
                    format!("{} requires turn scope", event_name(&envelope.event)),
                    vec![envelope.sequence],
                );
                return;
            }
            (EventScope::Turn { .. }, RunEvent::UserDisposition { .. }) => {
                self.add_finding(
                    AuditSeverity::Error,
                    "run_event_has_turn_scope",
                    "user_disposition requires run scope".to_string(),
                    vec![envelope.sequence],
                );
                return;
            }
            (EventScope::Turn { turn_id }, _) => turn_id.as_str(),
        };
        self.audit_turn_boundary(turn_id, envelope);
        match &envelope.event {
            RunEvent::TurnStarted { input, .. } => self.on_turn_started(turn_id, envelope, input),
            RunEvent::ContextBuilt { manifest } => self.on_context(turn_id, manifest),
            RunEvent::CompactionApplied { summary, .. } => {
                self.audit.compaction_count += 1;
                audit_content(&mut self.audit.content, summary);
            }
            RunEvent::AssistantMessage { content } => {
                audit_content(&mut self.audit.content, content);
            }
            RunEvent::ToolStarted { tool_call_id, .. } => {
                self.on_tool_started(turn_id, envelope, tool_call_id);
            }
            RunEvent::PermissionDecision { tool_call_id, .. } => {
                self.on_permission(turn_id, envelope, tool_call_id);
            }
            RunEvent::ToolCompleted {
                tool_call_id,
                result,
                is_error: _,
                effects,
                verification,
            } => {
                audit_content(&mut self.audit.content, result);
                self.on_tool_completed(
                    turn_id,
                    envelope,
                    tool_call_id,
                    effects,
                    verification.as_ref(),
                );
            }
            RunEvent::TurnCompleted { usage, .. } => {
                self.on_turn_completed(turn_id, envelope, usage);
            }
            RunEvent::Warning { .. } => {}
            RunEvent::UserDisposition { .. } => unreachable!("scope checked above"),
        }
    }

    fn audit_order(&mut self, index: usize, envelope: &RunEventEnvelope) {
        let expected = u64::try_from(index).unwrap_or(u64::MAX);
        if envelope.sequence != expected {
            self.add_finding(
                AuditSeverity::Error,
                "sequence_gap",
                format!(
                    "event at index {index} has sequence {}, expected {expected}",
                    envelope.sequence
                ),
                vec![envelope.sequence],
            );
        }
    }

    fn audit_turn_boundary(&mut self, turn_id: &'a str, envelope: &'a RunEventEnvelope) {
        let turn = self.turns.entry(turn_id).or_default();
        let missing_start =
            !matches!(envelope.event, RunEvent::TurnStarted { .. }) && turn.started.is_none();
        let completed = turn.completed;
        if missing_start {
            self.add_finding(
                AuditSeverity::Error,
                "event_before_turn_start",
                format!(
                    "{} appears before its turn starts",
                    event_name(&envelope.event)
                ),
                vec![envelope.sequence],
            );
        }
        if let Some(completed) = completed {
            self.add_finding(
                AuditSeverity::Error,
                "event_after_turn_completion",
                format!("event appears after turn completed at sequence {completed}"),
                vec![completed, envelope.sequence],
            );
        }
    }

    fn on_turn_started(
        &mut self,
        turn_id: &'a str,
        envelope: &'a RunEventEnvelope,
        input: &ContentRef,
    ) {
        let previous = self
            .turns
            .entry(turn_id)
            .or_default()
            .started
            .replace(envelope.sequence);
        if let Some(previous) = previous {
            self.add_finding(
                AuditSeverity::Error,
                "duplicate_turn_start",
                format!("turn {turn_id} starts more than once"),
                vec![previous, envelope.sequence],
            );
        }
        audit_content(&mut self.audit.content, input);
    }

    fn on_context(&mut self, turn_id: &'a str, manifest: &crate::ContextManifest) {
        self.turns.entry(turn_id).or_default().context_builds += 1;
        self.audit.context_build_count += 1;
        for section in &manifest.system_sections {
            count_disposition(
                section.selected,
                &section.reason,
                &mut self.audit.context.system_sections_selected,
                &mut self.audit.context.system_sections_excluded,
                &mut self.audit.context.empty_disposition_reasons,
            );
        }
        for message in &manifest.messages {
            count_disposition(
                message.selected,
                &message.reason,
                &mut self.audit.context.messages_selected,
                &mut self.audit.context.messages_excluded,
                &mut self.audit.context.empty_disposition_reasons,
            );
        }
        for tool in &manifest.tools {
            count_disposition(
                tool.selected,
                &tool.reason,
                &mut self.audit.context.tools_selected,
                &mut self.audit.context.tools_excluded,
                &mut self.audit.context.empty_disposition_reasons,
            );
        }
    }

    fn on_tool_started(
        &mut self,
        turn_id: &'a str,
        envelope: &'a RunEventEnvelope,
        tool_call_id: &'a str,
    ) {
        self.audit.tool_calls_started += 1;
        let previous = self.tools.insert(
            tool_call_id,
            ToolState {
                turn_id,
                started: envelope.sequence,
                permission: None,
                completed: None,
            },
        );
        if let Some(previous) = previous {
            self.add_finding(
                AuditSeverity::Error,
                "duplicate_tool_start",
                format!("tool call {tool_call_id} starts more than once"),
                vec![previous.started, envelope.sequence],
            );
        }
    }

    fn on_permission(&mut self, turn_id: &str, envelope: &RunEventEnvelope, tool_call_id: &str) {
        let Some(tool) = self.tools.get_mut(tool_call_id) else {
            self.add_finding(
                AuditSeverity::Error,
                "permission_without_tool_start",
                format!("permission references unknown tool call {tool_call_id}"),
                vec![envelope.sequence],
            );
            return;
        };
        let started = tool.started;
        let crosses_turns = tool.turn_id != turn_id;
        let previous = tool.permission.replace(envelope.sequence);
        if crosses_turns {
            self.add_finding(
                AuditSeverity::Error,
                "tool_crosses_turns",
                format!("permission for tool call {tool_call_id} is recorded in another turn"),
                vec![started, envelope.sequence],
            );
        }
        if let Some(previous) = previous {
            self.add_finding(
                AuditSeverity::Error,
                "duplicate_permission_decision",
                format!("tool call {tool_call_id} has multiple permission decisions"),
                vec![previous, envelope.sequence],
            );
        } else {
            self.audit.tool_calls_with_permission_decision += 1;
        }
    }

    fn on_tool_completed(
        &mut self,
        turn_id: &str,
        envelope: &RunEventEnvelope,
        tool_call_id: &str,
        effects: &[ToolEffect],
        verification: Option<&VerificationRecord>,
    ) {
        let Some(tool) = self.tools.get_mut(tool_call_id) else {
            self.add_finding(
                AuditSeverity::Error,
                "tool_completion_without_start",
                format!("result references unknown tool call {tool_call_id}"),
                vec![envelope.sequence],
            );
            return;
        };
        let started = tool.started;
        let crosses_turns = tool.turn_id != turn_id;
        let previous = tool.completed.replace(envelope.sequence);
        if crosses_turns {
            self.add_finding(
                AuditSeverity::Error,
                "tool_crosses_turns",
                format!("result for tool call {tool_call_id} is recorded in another turn"),
                vec![started, envelope.sequence],
            );
        }
        if let Some(previous) = previous {
            self.add_finding(
                AuditSeverity::Error,
                "duplicate_tool_completion",
                format!("tool call {tool_call_id} completes more than once"),
                vec![previous, envelope.sequence],
            );
        } else {
            self.audit.tool_calls_completed += 1;
        }
        self.audit.tool_effect_count += effects.len();
        for effect in effects {
            let (path, content_change) = match effect {
                ToolEffect::FileWrite {
                    path,
                    content_change,
                } => (path, content_change),
            };
            match content_change {
                ContentChange::Created => self.audit.files_created += 1,
                ContentChange::Modified => self.audit.files_modified += 1,
                ContentChange::Unchanged => self.audit.file_writes_unchanged += 1,
                ContentChange::Unknown => self.audit.file_writes_unknown += 1,
            }
            if path.as_os_str().is_empty() {
                self.add_finding(
                    AuditSeverity::Error,
                    "empty_effect_path",
                    format!("tool call {tool_call_id} reports an empty effect path"),
                    vec![envelope.sequence],
                );
            }
        }
        if let Some(verification) = verification {
            self.audit.verification_count += 1;
            match verification.status {
                VerificationStatus::Passed => self.audit.verification_passed += 1,
                VerificationStatus::Failed { .. } => self.audit.verification_failed += 1,
                VerificationStatus::Indeterminate { .. } => {
                    self.audit.verification_indeterminate += 1;
                }
            }
            if verification.description.trim().is_empty() {
                self.add_finding(
                    AuditSeverity::Warning,
                    "empty_verification_description",
                    format!("tool call {tool_call_id} records verification without a description"),
                    vec![envelope.sequence],
                );
            }
        }
    }

    fn on_turn_completed(
        &mut self,
        turn_id: &'a str,
        envelope: &'a RunEventEnvelope,
        usage: &UsageRecord,
    ) {
        let previous = self
            .turns
            .entry(turn_id)
            .or_default()
            .completed
            .replace(envelope.sequence);
        if let Some(previous) = previous {
            self.add_finding(
                AuditSeverity::Error,
                "duplicate_turn_completion",
                format!("turn {turn_id} completes more than once"),
                vec![previous, envelope.sequence],
            );
        } else {
            self.audit.completed_turn_count += 1;
        }
        self.audit.usage.input_tokens = self
            .audit
            .usage
            .input_tokens
            .saturating_add(usage.input_tokens);
        self.audit.usage.output_tokens = self
            .audit
            .usage
            .output_tokens
            .saturating_add(usage.output_tokens);
    }

    fn finish(mut self) -> RunAudit {
        self.audit.turn_count = self.turns.len();
        for (turn_id, turn) in std::mem::take(&mut self.turns) {
            if turn.started.is_none() {
                continue;
            }
            if turn.context_builds == 0 {
                self.add_finding(
                    AuditSeverity::Error,
                    "turn_without_context_manifest",
                    format!("turn {turn_id} records no provider context manifest"),
                    turn.started.into_iter().collect(),
                );
            }
            if turn.completed.is_none() {
                self.add_finding(
                    AuditSeverity::Error,
                    "incomplete_turn",
                    format!("turn {turn_id} never completes"),
                    turn.started.into_iter().collect(),
                );
            }
        }
        for (tool_call_id, tool) in std::mem::take(&mut self.tools) {
            if tool.completed.is_none() {
                self.add_finding(
                    AuditSeverity::Error,
                    "incomplete_tool_call",
                    format!("tool call {tool_call_id} never completes"),
                    vec![tool.started],
                );
            }
        }
        let empty_reasons = self.audit.context.empty_disposition_reasons;
        if empty_reasons > 0 {
            self.add_finding(
                AuditSeverity::Warning,
                "empty_context_disposition_reason",
                format!("{empty_reasons} context items have no selection or exclusion reason"),
                Vec::new(),
            );
        }
        self.audit
    }

    fn add_finding(
        &mut self,
        severity: AuditSeverity,
        code: &str,
        message: String,
        sequences: Vec<u64>,
    ) {
        finding(&mut self.audit, severity, code, message, sequences);
    }
}

fn count_disposition(
    selected: bool,
    reason: &str,
    selected_count: &mut usize,
    excluded_count: &mut usize,
    empty_reasons: &mut usize,
) {
    if selected {
        *selected_count += 1;
    } else {
        *excluded_count += 1;
    }
    if reason.trim().is_empty() {
        *empty_reasons += 1;
    }
}

fn audit_content(audit: &mut ContentAudit, content: &ContentRef) {
    match content {
        ContentRef::Inline { text } => {
            audit.inline_items += 1;
            audit.inline_bytes = audit.inline_bytes.saturating_add(text.len() as u64);
        }
        ContentRef::Artifact(artifact) => {
            audit.artifact_items += 1;
            audit.artifact_bytes = audit.artifact_bytes.saturating_add(artifact.bytes);
        }
        ContentRef::Unavailable { .. } => audit.unavailable_items += 1,
    }
}

fn finding(
    audit: &mut RunAudit,
    severity: AuditSeverity,
    code: &str,
    message: String,
    sequences: Vec<u64>,
) {
    audit.findings.push(RunAuditFinding {
        severity,
        code: code.to_string(),
        message,
        sequences,
    });
}

fn event_name(event: &RunEvent) -> &'static str {
    match event {
        RunEvent::TurnStarted { .. } => "turn_started",
        RunEvent::ContextBuilt { .. } => "context_built",
        RunEvent::CompactionApplied { .. } => "compaction_applied",
        RunEvent::AssistantMessage { .. } => "assistant_message",
        RunEvent::ToolStarted { .. } => "tool_started",
        RunEvent::PermissionDecision { .. } => "permission_decision",
        RunEvent::ToolCompleted { .. } => "tool_completed",
        RunEvent::TurnCompleted { .. } => "turn_completed",
        RunEvent::Warning { .. } => "warning",
        RunEvent::UserDisposition { .. } => "user_disposition",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::run_record::{ContextManifest, PermissionDecision, RUN_RECORD_SCHEMA_VERSION};

    fn envelope(sequence: u64, event: RunEvent) -> RunEventEnvelope {
        RunEventEnvelope {
            schema_version: RUN_RECORD_SCHEMA_VERSION,
            sequence,
            recorded_at_ms: sequence,
            session_id: "session-1".to_string(),
            scope: EventScope::Turn {
                turn_id: "turn-0".to_string(),
            },
            event,
        }
    }

    fn inline(text: &str) -> ContentRef {
        ContentRef::Inline {
            text: text.to_string(),
        }
    }

    fn run_envelope(sequence: u64, event: RunEvent) -> RunEventEnvelope {
        RunEventEnvelope {
            schema_version: RUN_RECORD_SCHEMA_VERSION,
            sequence,
            recorded_at_ms: sequence,
            session_id: "session-1".to_string(),
            scope: EventScope::Run,
            event,
        }
    }

    #[test]
    fn audits_complete_run_and_keeps_dimensions_separate() {
        let events = vec![
            envelope(
                0,
                RunEvent::TurnStarted {
                    provider: Some("test".to_string()),
                    model: "model".to_string(),
                    input: inline("inspect"),
                },
            ),
            envelope(
                1,
                RunEvent::ContextBuilt {
                    manifest: ContextManifest {
                        model: "model".to_string(),
                        context_window_tokens: 100,
                        estimated_input_tokens: 10,
                        system_sections: Vec::new(),
                        messages: Vec::new(),
                        tools: Vec::new(),
                    },
                },
            ),
            envelope(
                2,
                RunEvent::ToolStarted {
                    tool_call_id: "tool-1".to_string(),
                    name: "read_file".to_string(),
                    arguments: serde_json::json!({"path": "README.md"}),
                },
            ),
            envelope(
                3,
                RunEvent::PermissionDecision {
                    tool_call_id: "tool-1".to_string(),
                    decision: PermissionDecision::Allowed,
                },
            ),
            envelope(
                4,
                RunEvent::ToolCompleted {
                    tool_call_id: "tool-1".to_string(),
                    result: inline("contents"),
                    is_error: false,
                    effects: Vec::new(),
                    verification: None,
                },
            ),
            envelope(
                5,
                RunEvent::TurnCompleted {
                    usage: UsageRecord {
                        input_tokens: 12,
                        output_tokens: 4,
                    },
                    stop_reason: Some("end_turn".to_string()),
                },
            ),
        ];

        let audit = audit_run_record(&events);

        assert!(audit.is_structurally_complete());
        assert_eq!(audit.turn_count, 1);
        assert_eq!(audit.tool_calls_started, 1);
        assert_eq!(audit.tool_calls_completed, 1);
        assert_eq!(audit.tool_calls_with_permission_decision, 1);
        assert_eq!(audit.content.inline_items, 2);
        assert_eq!(audit.usage.input_tokens, 12);
        assert!(audit.findings.is_empty());
    }

    #[test]
    fn run_disposition_does_not_create_a_synthetic_turn() {
        let audit = audit_run_record(&[run_envelope(
            0,
            RunEvent::UserDisposition {
                disposition: RunDisposition::NeedsWork,
                note: inline("verify the browser projection"),
            },
        )]);

        assert!(audit.is_structurally_complete());
        assert_eq!(audit.turn_count, 0);
        assert_eq!(audit.user_disposition_count, 1);
        assert_eq!(
            audit.latest_user_disposition,
            Some(RunDisposition::NeedsWork)
        );
        assert_eq!(audit.content.inline_items, 1);
    }

    #[test]
    fn audits_effects_and_verification_as_separate_evidence_dimensions() {
        let mut events = vec![
            envelope(
                0,
                RunEvent::TurnStarted {
                    provider: Some("test".to_string()),
                    model: "model".to_string(),
                    input: inline("change and verify"),
                },
            ),
            envelope(
                1,
                RunEvent::ContextBuilt {
                    manifest: ContextManifest {
                        model: "model".to_string(),
                        context_window_tokens: 100,
                        estimated_input_tokens: 10,
                        system_sections: Vec::new(),
                        messages: Vec::new(),
                        tools: Vec::new(),
                    },
                },
            ),
        ];
        let statuses = [
            VerificationStatus::Passed,
            VerificationStatus::Failed { exit_code: Some(2) },
            VerificationStatus::Indeterminate {
                reason: crate::run_record::VerificationIndeterminate::TimedOut,
            },
        ];
        for (index, status) in statuses.into_iter().enumerate() {
            let sequence = 2 + u64::try_from(index).unwrap() * 2;
            let tool_call_id = format!("tool-{index}");
            events.push(envelope(
                sequence,
                RunEvent::ToolStarted {
                    tool_call_id: tool_call_id.clone(),
                    name: "bash".to_string(),
                    arguments: serde_json::json!({"purpose": "verification"}),
                },
            ));
            events.push(envelope(
                sequence + 1,
                RunEvent::ToolCompleted {
                    tool_call_id,
                    result: inline("observed result"),
                    is_error: !matches!(status, VerificationStatus::Passed),
                    effects: (index == 1)
                        .then(|| ToolEffect::FileWrite {
                            path: "partial.txt".into(),
                            content_change: ContentChange::Created,
                        })
                        .into_iter()
                        .collect(),
                    verification: Some(VerificationRecord {
                        description: format!("check {index}"),
                        status,
                    }),
                },
            ));
        }
        events.push(envelope(
            8,
            RunEvent::TurnCompleted {
                usage: UsageRecord::default(),
                stop_reason: Some("end_turn".to_string()),
            },
        ));

        let audit = audit_run_record(&events);

        assert!(audit.is_structurally_complete());
        assert_eq!(audit.tool_effect_count, 1);
        assert_eq!(audit.files_created, 1);
        assert_eq!(audit.verification_count, 3);
        assert_eq!(audit.verification_passed, 1);
        assert_eq!(audit.verification_failed, 1);
        assert_eq!(audit.verification_indeterminate, 1);
    }

    #[test]
    fn identifies_missing_context_and_unfinished_tool() {
        let events = vec![
            envelope(
                0,
                RunEvent::TurnStarted {
                    provider: None,
                    model: "model".to_string(),
                    input: inline("inspect"),
                },
            ),
            envelope(
                1,
                RunEvent::ToolStarted {
                    tool_call_id: "tool-1".to_string(),
                    name: "read_file".to_string(),
                    arguments: serde_json::json!({}),
                },
            ),
        ];

        let audit = audit_run_record(&events);
        let codes = audit
            .findings
            .iter()
            .map(|finding| finding.code.as_str())
            .collect::<Vec<_>>();

        assert!(!audit.is_structurally_complete());
        assert!(codes.contains(&"turn_without_context_manifest"));
        assert!(codes.contains(&"incomplete_turn"));
        assert!(codes.contains(&"incomplete_tool_call"));
    }

    #[test]
    fn does_not_treat_permission_coverage_as_a_quality_score() {
        let events = vec![
            envelope(
                0,
                RunEvent::TurnStarted {
                    provider: None,
                    model: "model".to_string(),
                    input: inline("inspect"),
                },
            ),
            envelope(
                1,
                RunEvent::ContextBuilt {
                    manifest: ContextManifest {
                        model: "model".to_string(),
                        context_window_tokens: 100,
                        estimated_input_tokens: 10,
                        system_sections: Vec::new(),
                        messages: Vec::new(),
                        tools: Vec::new(),
                    },
                },
            ),
            envelope(
                2,
                RunEvent::ToolStarted {
                    tool_call_id: "tool-1".to_string(),
                    name: "unknown".to_string(),
                    arguments: serde_json::json!({}),
                },
            ),
            envelope(
                3,
                RunEvent::ToolCompleted {
                    tool_call_id: "tool-1".to_string(),
                    result: inline("not available"),
                    is_error: true,
                    effects: Vec::new(),
                    verification: None,
                },
            ),
            envelope(
                4,
                RunEvent::TurnCompleted {
                    usage: UsageRecord::default(),
                    stop_reason: Some("end_turn".to_string()),
                },
            ),
        ];

        let audit = audit_run_record(&events);

        assert!(audit.is_structurally_complete());
        assert_eq!(audit.tool_calls_with_permission_decision, 0);
        assert!(audit.findings.is_empty());
    }
}
