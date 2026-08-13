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
    #[serde(default)]
    pub failed_turn_count: usize,
    #[serde(default)]
    pub cancelled_turn_count: usize,
    pub context_build_count: usize,
    pub compaction_count: usize,
    pub tool_calls_started: usize,
    pub tool_calls_completed: usize,
    pub tool_calls_with_permission_decision: usize,
    #[serde(default)]
    pub authority_lease_count: usize,
    #[serde(default)]
    pub workspace_write_granted: usize,
    #[serde(default)]
    pub workspace_write_denied: usize,
    #[serde(default)]
    pub workspace_write_revoked: usize,
    pub tool_effect_count: usize,
    #[serde(default)]
    pub unattributed_effect_count: usize,
    pub files_created: usize,
    pub files_modified: usize,
    #[serde(default)]
    pub files_deleted: usize,
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
    started_input_sha256: Option<crate::Sha256Digest>,
    started_input_bytes: Option<usize>,
    started_input_digest_unavailable: bool,
    completed: Option<u64>,
    context_builds: usize,
    authority_lease: Option<u64>,
}

struct ToolState<'a> {
    turn_id: &'a str,
    started: u64,
    permission: Option<u64>,
    completed: Option<u64>,
}

#[derive(Clone, Copy)]
enum TurnOutcome {
    Completed,
    Failed,
    Cancelled,
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
                failed_turn_count: 0,
                cancelled_turn_count: 0,
                context_build_count: 0,
                compaction_count: 0,
                tool_calls_started: 0,
                tool_calls_completed: 0,
                tool_calls_with_permission_decision: 0,
                authority_lease_count: 0,
                workspace_write_granted: 0,
                workspace_write_denied: 0,
                workspace_write_revoked: 0,
                tool_effect_count: 0,
                unattributed_effect_count: 0,
                files_created: 0,
                files_modified: 0,
                files_deleted: 0,
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
        self.observe_turn_event(turn_id, envelope);
    }

    fn observe_turn_event(&mut self, turn_id: &'a str, envelope: &'a RunEventEnvelope) {
        match &envelope.event {
            RunEvent::TurnStarted { input, .. } => self.on_turn_started(turn_id, envelope, input),
            RunEvent::ContextBuilt { manifest } => self.on_context(turn_id, manifest),
            RunEvent::RequestContextResolved { .. } => {
                self.on_request_context(turn_id, envelope);
            }
            RunEvent::ContextUnavailable { reason } => {
                self.on_context_unavailable(turn_id, envelope, reason);
            }
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
            RunEvent::AuthorityLease {
                authority,
                expires_at_ms,
                issued_at_ms,
                turn_deadline_ms,
                cwd,
                tool_profile,
                outcome,
                ..
            } => {
                self.on_authority_lease(
                    turn_id,
                    envelope,
                    *authority,
                    *issued_at_ms,
                    *expires_at_ms,
                    *turn_deadline_ms,
                    cwd,
                    tool_profile,
                    outcome,
                );
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
                self.on_turn_terminal(turn_id, envelope, TurnOutcome::Completed, usage.as_ref());
            }
            RunEvent::TurnFailed { .. } => {
                self.on_turn_terminal(turn_id, envelope, TurnOutcome::Failed, None);
            }
            RunEvent::TurnCancelled { .. } => {
                self.on_turn_terminal(turn_id, envelope, TurnOutcome::Cancelled, None);
            }
            RunEvent::ContextSourcesResolved { .. }
            | RunEvent::Warning { .. }
            | RunEvent::ChildRunRef { .. } => {}
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
        let turn = self.turns.entry(turn_id).or_default();
        match input {
            ContentRef::Inline { text } => {
                turn.started_input_sha256 = Some(crate::Sha256Digest::of_bytes(text.as_bytes()));
                turn.started_input_bytes = Some(text.len());
            }
            ContentRef::Artifact(artifact) => {
                turn.started_input_bytes = usize::try_from(artifact.bytes).ok();
                turn.started_input_digest_unavailable = true;
            }
            ContentRef::Unavailable { .. } => {}
        }
        audit_content(&mut self.audit.content, input);
    }

    fn on_request_context(&mut self, turn_id: &'a str, envelope: &'a RunEventEnvelope) {
        let RunEvent::RequestContextResolved {
            context,
            composed_input_sha256,
            composed_input_bytes,
            ..
        } = &envelope.event
        else {
            unreachable!("request context handler requires request context event")
        };
        audit_content(&mut self.audit.content, context);
        let (bytes_mismatch, digest_mismatch, digest_unavailable) = {
            let turn = self.turns.entry(turn_id).or_default();
            (
                turn.started_input_bytes
                    .is_some_and(|bytes| bytes != *composed_input_bytes),
                turn.started_input_sha256
                    .as_ref()
                    .is_some_and(|digest| digest != composed_input_sha256),
                turn.started_input_digest_unavailable,
            )
        };
        if bytes_mismatch {
            self.add_finding(
                AuditSeverity::Error,
                "request_context_input_bytes_mismatch",
                format!(
                    "turn {turn_id} request context byte count does not match its recorded input"
                ),
                vec![envelope.sequence],
            );
        }
        if digest_mismatch {
            self.add_finding(
                AuditSeverity::Error,
                "request_context_input_digest_mismatch",
                format!("turn {turn_id} request context digest does not match its recorded input"),
                vec![envelope.sequence],
            );
        }
        if digest_unavailable {
            self.add_finding(
                AuditSeverity::Error,
                "request_context_input_digest_unverifiable",
                format!(
                    "turn {turn_id} request context digest cannot be verified against its artifact input"
                ),
                vec![envelope.sequence],
            );
        }
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

    fn on_context_unavailable(
        &mut self,
        turn_id: &'a str,
        envelope: &'a RunEventEnvelope,
        reason: &str,
    ) {
        self.turns.entry(turn_id).or_default().context_builds += 1;
        if reason.trim().is_empty() {
            self.add_finding(
                AuditSeverity::Warning,
                "empty_context_unavailable_reason",
                format!("turn {turn_id} marks context unavailable without a reason"),
                vec![envelope.sequence],
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

    #[allow(clippy::too_many_arguments)]
    fn on_authority_lease(
        &mut self,
        turn_id: &'a str,
        envelope: &RunEventEnvelope,
        authority: crate::run_record::TurnAuthority,
        issued_at_ms: u64,
        expires_at_ms: u64,
        turn_deadline_ms: u64,
        cwd: &std::path::Path,
        tool_profile: &str,
        outcome: &crate::run_record::AuthorityLeaseOutcome,
    ) {
        use crate::run_record::{AuthorityLeaseOutcome, TurnAuthority};

        self.audit.authority_lease_count += 1;
        let previous = self
            .turns
            .entry(turn_id)
            .or_default()
            .authority_lease
            .replace(envelope.sequence);
        if let Some(previous) = previous {
            self.add_finding(
                AuditSeverity::Error,
                "duplicate_authority_lease",
                format!("turn {turn_id} records more than one authority lease"),
                vec![previous, envelope.sequence],
            );
        }
        if authority == TurnAuthority::WorkspaceWrite {
            match outcome {
                AuthorityLeaseOutcome::Granted => self.audit.workspace_write_granted += 1,
                AuthorityLeaseOutcome::Denied { .. } => self.audit.workspace_write_denied += 1,
                AuthorityLeaseOutcome::Revoked { .. } => self.audit.workspace_write_revoked += 1,
            }
        }
        if issued_at_ms > expires_at_ms || issued_at_ms > turn_deadline_ms {
            self.add_finding(
                AuditSeverity::Error,
                "invalid_authority_deadline",
                "authority lease deadline precedes issuance".to_string(),
                vec![envelope.sequence],
            );
        }
        if !cwd.is_absolute() {
            self.add_finding(
                AuditSeverity::Error,
                "invalid_authority_cwd",
                "authority lease working directory must be absolute".to_string(),
                vec![envelope.sequence],
            );
        }
        if tool_profile.trim().is_empty() {
            self.add_finding(
                AuditSeverity::Error,
                "empty_authority_tool_profile",
                "authority lease has an empty tool profile".to_string(),
                vec![envelope.sequence],
            );
        }
        match outcome {
            AuthorityLeaseOutcome::Denied { reason }
            | AuthorityLeaseOutcome::Revoked { reason }
                if reason.trim().is_empty() =>
            {
                self.add_finding(
                    AuditSeverity::Error,
                    "empty_authority_outcome_reason",
                    "denied or revoked authority lease requires a reason".to_string(),
                    vec![envelope.sequence],
                );
            }
            _ => {}
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
            match effect {
                ToolEffect::FileWrite {
                    path,
                    content_change,
                } => {
                    match content_change {
                        ContentChange::Created => self.audit.files_created += 1,
                        ContentChange::Modified => self.audit.files_modified += 1,
                        ContentChange::Unchanged => self.audit.file_writes_unchanged += 1,
                        ContentChange::Unknown => self.audit.file_writes_unknown += 1,
                    }
                    self.audit_effect_path(tool_call_id, envelope.sequence, path);
                }
                ToolEffect::FileDelete { path } => {
                    self.audit.files_deleted += 1;
                    self.audit_effect_path(tool_call_id, envelope.sequence, path);
                }
                ToolEffect::ShellCommand { .. } => {}
                ToolEffect::Unattributed { reason, .. } => {
                    self.audit.unattributed_effect_count += 1;
                    if reason.trim().is_empty() {
                        self.add_finding(
                            AuditSeverity::Error,
                            "empty_unattributed_effect_reason",
                            format!(
                                "tool call {tool_call_id} reports an unattributed effect without a reason"
                            ),
                            vec![envelope.sequence],
                        );
                    }
                }
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

    fn audit_effect_path(&mut self, tool_call_id: &str, sequence: u64, path: &std::path::Path) {
        if path.as_os_str().is_empty() {
            self.add_finding(
                AuditSeverity::Error,
                "empty_effect_path",
                format!("tool call {tool_call_id} reports an empty effect path"),
                vec![sequence],
            );
        }
    }

    fn on_turn_terminal(
        &mut self,
        turn_id: &'a str,
        envelope: &'a RunEventEnvelope,
        outcome: TurnOutcome,
        usage: Option<&UsageRecord>,
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
            match outcome {
                TurnOutcome::Completed => self.audit.completed_turn_count += 1,
                TurnOutcome::Failed => self.audit.failed_turn_count += 1,
                TurnOutcome::Cancelled => self.audit.cancelled_turn_count += 1,
            }
        }
        if let Some(usage) = usage {
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
        RunEvent::ContextSourcesResolved { .. } => "context_sources_resolved",
        RunEvent::RequestContextResolved { .. } => "request_context_resolved",
        RunEvent::ContextUnavailable { .. } => "context_unavailable",
        RunEvent::CompactionApplied { .. } => "compaction_applied",
        RunEvent::AssistantMessage { .. } => "assistant_message",
        RunEvent::ToolStarted { .. } => "tool_started",
        RunEvent::PermissionDecision { .. } => "permission_decision",
        RunEvent::AuthorityLease { .. } => "authority_lease",
        RunEvent::ToolCompleted { .. } => "tool_completed",
        RunEvent::TurnCompleted { .. } => "turn_completed",
        RunEvent::TurnFailed { .. } => "turn_failed",
        RunEvent::TurnCancelled { .. } => "turn_cancelled",
        RunEvent::Warning { .. } => "warning",
        RunEvent::UserDisposition { .. } => "user_disposition",
        RunEvent::ChildRunRef { .. } => "child_run_ref",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::run_record::{
        ArtifactRef, ContextManifest, ContextSourceSummary, PermissionDecision,
        RUN_RECORD_SCHEMA_VERSION,
    };
    use crate::{Sha256Digest, SourceReference, Trust};

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
                    usage: Some(UsageRecord {
                        input_tokens: 12,
                        output_tokens: 4,
                    }),
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
    fn resolved_sources_are_not_counted_as_runtime_context_builds() {
        let events = vec![
            envelope(
                0,
                RunEvent::TurnStarted {
                    provider: Some("test".into()),
                    model: "model".into(),
                    input: inline("inspect"),
                },
            ),
            envelope(
                1,
                RunEvent::ContextSourcesResolved {
                    sources: vec![ContextSourceSummary {
                        id: "note-1".into(),
                        sources: vec![SourceReference {
                            reference: "surface:scratch/object:note-1".into(),
                            sha256: Sha256Digest::of_bytes(b"note"),
                        }],
                        output_sha256: Sha256Digest::of_bytes(b"note"),
                        byte_size: 4,
                        trust: Trust::UntrustedEvidence,
                    }],
                },
            ),
            envelope(
                2,
                RunEvent::TurnCompleted {
                    usage: None,
                    stop_reason: Some("complete".into()),
                },
            ),
        ];

        let audit = audit_run_record(&events);
        assert_eq!(audit.context_build_count, 0);
        assert_eq!(audit.event_count, 3);
    }

    #[test]
    fn request_context_digest_and_size_are_checked_against_turn_input() {
        let input = "exact composed request";
        let events = vec![
            envelope(
                0,
                RunEvent::TurnStarted {
                    provider: Some("codex".into()),
                    model: "model".into(),
                    input: inline(input),
                },
            ),
            envelope(
                1,
                RunEvent::RequestContextResolved {
                    context: inline(""),
                    sources: Vec::new(),
                    history_messages: 0,
                    composed_input_sha256: Sha256Digest::of_bytes(b"different request"),
                    composed_input_bytes: input.len() + 1,
                },
            ),
            envelope(
                2,
                RunEvent::TurnCompleted {
                    usage: None,
                    stop_reason: Some("complete".into()),
                },
            ),
        ];

        let audit = audit_run_record(&events);
        let codes = audit
            .findings
            .iter()
            .map(|finding| finding.code.as_str())
            .collect::<Vec<_>>();
        assert!(codes.contains(&"request_context_input_bytes_mismatch"));
        assert!(codes.contains(&"request_context_input_digest_mismatch"));
        assert_eq!(audit.content.inline_items, 2);
    }

    #[test]
    fn materialized_turn_input_cannot_bypass_request_context_digest_verification() {
        let input_bytes = 16 * 1024 + 1;
        let events = vec![
            envelope(
                0,
                RunEvent::TurnStarted {
                    provider: Some("codex".into()),
                    model: "model".into(),
                    input: ContentRef::Artifact(ArtifactRef {
                        relative_path: "artifacts/00000000-input.txt".into(),
                        media_type: "text/plain; charset=utf-8".into(),
                        bytes: input_bytes as u64,
                    }),
                },
            ),
            envelope(
                1,
                RunEvent::RequestContextResolved {
                    context: inline(""),
                    sources: Vec::new(),
                    history_messages: 0,
                    composed_input_sha256: Sha256Digest::of_bytes(&vec![b'x'; input_bytes]),
                    composed_input_bytes: input_bytes,
                },
            ),
        ];

        let audit = audit_run_record(&events);

        assert!(!audit.is_structurally_complete());
        assert!(audit
            .findings
            .iter()
            .any(|finding| { finding.code == "request_context_input_digest_unverifiable" }));
        assert!(!audit
            .findings
            .iter()
            .any(|finding| { finding.code == "request_context_input_bytes_mismatch" }));
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
                    effects: match index {
                        1 => vec![ToolEffect::FileWrite {
                            path: "partial.txt".into(),
                            content_change: ContentChange::Created,
                        }],
                        2 => vec![ToolEffect::FileDelete {
                            path: "obsolete.txt".into(),
                        }],
                        _ => Vec::new(),
                    },
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
                usage: Some(UsageRecord::default()),
                stop_reason: Some("end_turn".to_string()),
            },
        ));

        let audit = audit_run_record(&events);

        assert!(audit.is_structurally_complete());
        assert_eq!(audit.tool_effect_count, 2);
        assert_eq!(audit.files_created, 1);
        assert_eq!(audit.files_deleted, 1);
        assert_eq!(audit.verification_count, 3);
        assert_eq!(audit.verification_passed, 1);
        assert_eq!(audit.verification_failed, 1);
        assert_eq!(audit.verification_indeterminate, 1);
    }

    #[test]
    fn rejects_empty_paths_for_writes_and_deletes() {
        let mut events = vec![envelope(
            0,
            RunEvent::TurnStarted {
                provider: Some("test".into()),
                model: "model".into(),
                input: inline("mutate files"),
            },
        )];
        for (index, effect) in [
            ToolEffect::FileWrite {
                path: "".into(),
                content_change: ContentChange::Modified,
            },
            ToolEffect::FileDelete { path: "".into() },
        ]
        .into_iter()
        .enumerate()
        {
            let sequence = 1 + u64::try_from(index).unwrap() * 2;
            let tool_call_id = format!("tool-{index}");
            events.push(envelope(
                sequence,
                RunEvent::ToolStarted {
                    tool_call_id: tool_call_id.clone(),
                    name: "filesystem".into(),
                    arguments: serde_json::json!({}),
                },
            ));
            events.push(envelope(
                sequence + 1,
                RunEvent::ToolCompleted {
                    tool_call_id,
                    result: inline("done"),
                    is_error: false,
                    effects: vec![effect],
                    verification: None,
                },
            ));
        }
        events.push(envelope(
            5,
            RunEvent::TurnCompleted {
                usage: None,
                stop_reason: Some("complete".into()),
            },
        ));

        let audit = audit_run_record(&events);

        assert!(!audit.is_structurally_complete());
        assert_eq!(audit.files_modified, 1);
        assert_eq!(audit.files_deleted, 1);
        assert_eq!(
            audit
                .findings
                .iter()
                .filter(|finding| finding.code == "empty_effect_path")
                .count(),
            2
        );
    }

    #[test]
    fn audits_authority_and_unattributed_effects_without_inventing_file_writes() {
        let events = vec![
            envelope(
                0,
                RunEvent::TurnStarted {
                    provider: Some("codex".into()),
                    model: "model".into(),
                    input: inline("change one file"),
                },
            ),
            envelope(
                1,
                RunEvent::ContextBuilt {
                    manifest: ContextManifest {
                        model: "model".into(),
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
                RunEvent::AuthorityLease {
                    authority: crate::run_record::TurnAuthority::WorkspaceWrite,
                    scope_digest: Sha256Digest::of_bytes(b"scope"),
                    issued_at_ms: 10,
                    expires_at_ms: 20,
                    turn_deadline_ms: 30,
                    cwd: "/workspace".into(),
                    network_access: false,
                    tool_profile: "codex_workspace_write".into(),
                    outcome: crate::run_record::AuthorityLeaseOutcome::Granted,
                },
            ),
            envelope(
                3,
                RunEvent::ToolStarted {
                    tool_call_id: "shell-1".into(),
                    name: "shell".into(),
                    arguments: serde_json::json!({"command": "build"}),
                },
            ),
            envelope(
                4,
                RunEvent::ToolCompleted {
                    tool_call_id: "shell-1".into(),
                    result: inline("done"),
                    is_error: false,
                    effects: vec![ToolEffect::Unattributed {
                        category: crate::run_record::EffectCategory::FileSystem,
                        reason: "shell execution does not report changed paths".into(),
                    }],
                    verification: None,
                },
            ),
            envelope(
                5,
                RunEvent::TurnCompleted {
                    usage: None,
                    stop_reason: Some("complete".into()),
                },
            ),
        ];

        let audit = audit_run_record(&events);

        assert!(audit.is_structurally_complete());
        assert_eq!(audit.authority_lease_count, 1);
        assert_eq!(audit.workspace_write_granted, 1);
        assert_eq!(audit.workspace_write_denied, 0);
        assert_eq!(audit.workspace_write_revoked, 0);
        assert_eq!(audit.unattributed_effect_count, 1);
        assert_eq!(audit.files_created, 0);
        assert_eq!(audit.files_modified, 0);
        assert_eq!(audit.file_writes_unknown, 0);
    }

    #[test]
    fn rejects_duplicate_or_malformed_authority_evidence() {
        let authority = |issued_at_ms, outcome| RunEvent::AuthorityLease {
            authority: crate::run_record::TurnAuthority::WorkspaceWrite,
            scope_digest: Sha256Digest::of_bytes(b"scope"),
            issued_at_ms,
            expires_at_ms: 20,
            turn_deadline_ms: 30,
            cwd: "".into(),
            network_access: false,
            tool_profile: String::new(),
            outcome,
        };
        let events = vec![
            envelope(
                0,
                RunEvent::TurnStarted {
                    provider: Some("codex".into()),
                    model: "model".into(),
                    input: inline("change"),
                },
            ),
            envelope(
                1,
                authority(
                    40,
                    crate::run_record::AuthorityLeaseOutcome::Denied {
                        reason: String::new(),
                    },
                ),
            ),
            envelope(
                2,
                authority(10, crate::run_record::AuthorityLeaseOutcome::Granted),
            ),
        ];

        let audit = audit_run_record(&events);
        let codes = audit
            .findings
            .iter()
            .map(|finding| finding.code.as_str())
            .collect::<Vec<_>>();

        assert!(codes.contains(&"invalid_authority_deadline"));
        assert!(codes.contains(&"invalid_authority_cwd"));
        assert!(codes.contains(&"empty_authority_tool_profile"));
        assert!(codes.contains(&"empty_authority_outcome_reason"));
        assert!(codes.contains(&"duplicate_authority_lease"));
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
                    usage: Some(UsageRecord::default()),
                    stop_reason: Some("end_turn".to_string()),
                },
            ),
        ];

        let audit = audit_run_record(&events);

        assert!(audit.is_structurally_complete());
        assert_eq!(audit.tool_calls_with_permission_decision, 0);
        assert!(audit.findings.is_empty());
    }

    #[test]
    fn distinguishes_failed_and_cancelled_turns_from_completed_turns() {
        let failed = audit_run_record(&[
            envelope(
                0,
                RunEvent::TurnStarted {
                    provider: Some("codex".to_string()),
                    model: "resolved by Codex".to_string(),
                    input: inline("inspect"),
                },
            ),
            envelope(
                1,
                RunEvent::ContextUnavailable {
                    reason: "native executor owns context assembly".to_string(),
                },
            ),
            envelope(
                2,
                RunEvent::TurnFailed {
                    class: "executor".to_string(),
                    message: "turn rejected".to_string(),
                },
            ),
        ]);

        assert!(failed.is_structurally_complete());
        assert_eq!(failed.completed_turn_count, 0);
        assert_eq!(failed.failed_turn_count, 1);
        assert_eq!(failed.cancelled_turn_count, 0);
        assert_eq!(failed.usage, UsageRecord::default());

        let cancelled = audit_run_record(&[
            envelope(
                0,
                RunEvent::TurnStarted {
                    provider: Some("codex".to_string()),
                    model: "resolved by Codex".to_string(),
                    input: inline("inspect"),
                },
            ),
            envelope(
                1,
                RunEvent::ContextUnavailable {
                    reason: "native executor owns context assembly".to_string(),
                },
            ),
            envelope(
                2,
                RunEvent::TurnCancelled {
                    reason: "browser disconnected".to_string(),
                },
            ),
        ]);

        assert!(cancelled.is_structurally_complete());
        assert_eq!(cancelled.completed_turn_count, 0);
        assert_eq!(cancelled.failed_turn_count, 0);
        assert_eq!(cancelled.cancelled_turn_count, 1);
    }

    #[test]
    fn completed_turn_can_report_usage_as_unavailable_without_inventing_zeroes() {
        let audit = audit_run_record(&[
            envelope(
                0,
                RunEvent::TurnStarted {
                    provider: Some("codex".to_string()),
                    model: "resolved by Codex".to_string(),
                    input: inline("inspect"),
                },
            ),
            envelope(
                1,
                RunEvent::ContextUnavailable {
                    reason: "native executor owns context assembly".to_string(),
                },
            ),
            envelope(
                2,
                RunEvent::TurnCompleted {
                    usage: None,
                    stop_reason: Some("completed".to_string()),
                },
            ),
        ]);

        assert!(audit.is_structurally_complete());
        assert_eq!(audit.completed_turn_count, 1);
        assert_eq!(audit.usage, UsageRecord::default());
    }
}
