//! Typed, deterministic context resolution primitives.
//!
//! The first production resolver is deliberately narrow: it turns attachments
//! already captured by the host into untrusted, workspace-sensitive message
//! evidence. Callers provide content and provenance, but cannot select an
//! authority-bearing plane, trust class, or sensitivity class.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::fmt;

pub const CONTEXT_RESOLUTION_SCHEMA_VERSION: u32 = 1;
const CAPTURED_ATTACHMENT_RESOLVER_ID: &str = "captured-attachment";
const CAPTURED_ATTACHMENT_RESOLVER_VERSION: &str = "1";

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize)]
#[serde(transparent)]
pub struct Sha256Digest(String);

impl Sha256Digest {
    pub fn parse(value: impl Into<String>) -> Result<Self, ContextError> {
        let value = value.into();
        if value.len() == 64
            && value
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            Ok(Self(value))
        } else {
            Err(ContextError::InvalidDigest)
        }
    }

    #[must_use]
    pub fn of_bytes(bytes: &[u8]) -> Self {
        Self(format!("{:x}", Sha256::digest(bytes)))
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl<'de> Deserialize<'de> for Sha256Digest {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Self::parse(value).map_err(serde::de::Error::custom)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OutputPlane {
    Instruction,
    Message,
    Tool,
    State,
    Artifact,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReplayMode {
    Exact,
    Refresh,
    Fork,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FreshnessPolicy {
    Captured,
    Current,
    MaxAge,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Trust {
    Control,
    OperatorInstruction,
    HostFact,
    UntrustedEvidence,
    DerivedEvidence,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Freshness {
    Captured,
    Current,
    Stale,
    Unknown,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Sensitivity {
    Public,
    Workspace,
    Private,
    Secret,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CacheDecision {
    Miss,
    Hit,
    Captured,
    Bypass,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResolutionStatus {
    Succeeded,
    Failed,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResolverIdentity {
    pub id: String,
    pub version: String,
    pub config_sha256: Sha256Digest,
    pub code_sha256: Sha256Digest,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResolutionRequest {
    pub output_plane: OutputPlane,
    pub replay_mode: ReplayMode,
    pub byte_budget: usize,
    pub token_budget: usize,
    pub deadline_ms: u64,
    pub freshness_policy: FreshnessPolicy,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CapabilityProfile {
    pub id: String,
    pub sha256: Sha256Digest,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResolutionCache {
    pub decision: CacheDecision,
    pub key_sha256: Sha256Digest,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SourceReference {
    #[serde(rename = "ref")]
    pub reference: String,
    pub sha256: Sha256Digest,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ContextPayload {
    Inline { inline_payload: String },
    Ref { payload_ref: String },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ContextItem {
    pub id: String,
    pub resolver_id: String,
    pub resolver_version: String,
    pub output_plane: OutputPlane,
    pub media_type: String,
    pub sources: Vec<SourceReference>,
    pub trust: Trust,
    pub freshness: Freshness,
    pub sensitivity: Sensitivity,
    pub priority: i64,
    #[serde(flatten)]
    pub payload: ContextPayload,
    pub byte_size: usize,
    pub token_estimate: usize,
    pub output_sha256: Sha256Digest,
    pub created_at: String,
    pub expires_at: Option<String>,
    pub warnings: Vec<String>,
}

impl<'de> Deserialize<'de> for ContextItem {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct WireItem {
            id: String,
            resolver_id: String,
            resolver_version: String,
            output_plane: OutputPlane,
            media_type: String,
            sources: Vec<SourceReference>,
            trust: Trust,
            freshness: Freshness,
            sensitivity: Sensitivity,
            priority: i64,
            inline_payload: Option<String>,
            payload_ref: Option<String>,
            byte_size: usize,
            token_estimate: usize,
            output_sha256: Sha256Digest,
            created_at: String,
            expires_at: Option<String>,
            warnings: Vec<String>,
        }

        let wire = WireItem::deserialize(deserializer)?;
        let payload = match (wire.inline_payload, wire.payload_ref) {
            (Some(inline_payload), None) => ContextPayload::Inline { inline_payload },
            (None, Some(payload_ref)) if !payload_ref.is_empty() => {
                ContextPayload::Ref { payload_ref }
            }
            _ => {
                return Err(serde::de::Error::custom(
                    "exactly one non-empty payload field is required",
                ));
            }
        };
        Ok(Self {
            id: wire.id,
            resolver_id: wire.resolver_id,
            resolver_version: wire.resolver_version,
            output_plane: wire.output_plane,
            media_type: wire.media_type,
            sources: wire.sources,
            trust: wire.trust,
            freshness: wire.freshness,
            sensitivity: wire.sensitivity,
            priority: wire.priority,
            payload,
            byte_size: wire.byte_size,
            token_estimate: wire.token_estimate,
            output_sha256: wire.output_sha256,
            created_at: wire.created_at,
            expires_at: wire.expires_at,
            warnings: wire.warnings,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResolutionError {
    pub code: String,
    pub message: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContextResolution {
    pub schema_version: u32,
    pub run_id: String,
    pub role: String,
    pub checkpoint: String,
    pub resolver: ResolverIdentity,
    pub request: ResolutionRequest,
    pub capability_profile: CapabilityProfile,
    pub status: ResolutionStatus,
    pub cache: ResolutionCache,
    pub started_at: String,
    pub finished_at: String,
    pub items: Vec<ContextItem>,
    pub warnings: Vec<String>,
    pub error: Option<ResolutionError>,
    pub materialized_artifact_refs: Vec<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ContextBudget {
    pub byte_budget: usize,
    pub token_budget: usize,
}

impl ContextBudget {
    #[must_use]
    pub const fn new(byte_budget: usize, token_budget: usize) -> Self {
        Self {
            byte_budget,
            token_budget,
        }
    }
}

/// Host-captured attachment input. Authority metadata is intentionally absent.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CapturedAttachment {
    id: String,
    media_type: String,
    sources: Vec<SourceReference>,
    priority: i64,
    payload: String,
    created_at: String,
    expires_at: Option<String>,
}

impl CapturedAttachment {
    pub fn new(
        id: impl Into<String>,
        media_type: impl Into<String>,
        sources: Vec<SourceReference>,
        priority: i64,
        payload: impl Into<String>,
        created_at: impl Into<String>,
    ) -> Result<Self, ContextError> {
        let attachment = Self {
            id: id.into(),
            media_type: media_type.into(),
            sources,
            priority,
            payload: payload.into(),
            created_at: created_at.into(),
            expires_at: None,
        };
        require_field("id", &attachment.id)?;
        require_field("media_type", &attachment.media_type)?;
        require_field("created_at", &attachment.created_at)?;
        Ok(attachment)
    }

    #[must_use]
    pub fn with_expiry(mut self, expires_at: impl Into<String>) -> Self {
        self.expires_at = Some(expires_at.into());
        self
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RenderedContext {
    bytes: String,
    byte_size: usize,
    token_estimate: usize,
    sha256: Sha256Digest,
}

impl RenderedContext {
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.bytes
    }

    #[must_use]
    pub const fn byte_size(&self) -> usize {
        self.byte_size
    }

    #[must_use]
    pub const fn token_estimate(&self) -> usize {
        self.token_estimate
    }

    #[must_use]
    pub const fn sha256(&self) -> &Sha256Digest {
        &self.sha256
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ContextError {
    EmptyField(&'static str),
    InvalidDigest,
    DuplicateAttachmentId(String),
    ByteBudgetExceeded { required: usize, available: usize },
    TokenBudgetExceeded { required: usize, available: usize },
    PayloadReferenceNotRenderable(String),
    IntegrityMismatch(String),
    AuthorityMismatch(String),
    SizeOverflow,
}

impl fmt::Display for ContextError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyField(field) => write!(formatter, "{field} must not be empty"),
            Self::InvalidDigest => formatter.write_str("invalid lowercase SHA-256 digest"),
            Self::DuplicateAttachmentId(id) => write!(formatter, "duplicate attachment id: {id}"),
            Self::ByteBudgetExceeded {
                required,
                available,
            } => {
                write!(
                    formatter,
                    "byte budget exceeded: required {required}, available {available}"
                )
            }
            Self::TokenBudgetExceeded {
                required,
                available,
            } => write!(
                formatter,
                "token budget exceeded: required {required}, available {available}"
            ),
            Self::PayloadReferenceNotRenderable(id) => {
                write!(formatter, "attachment {id} has no captured inline payload")
            }
            Self::IntegrityMismatch(id) => {
                write!(formatter, "attachment {id} failed integrity verification")
            }
            Self::AuthorityMismatch(id) => {
                write!(formatter, "attachment {id} has invalid authority metadata")
            }
            Self::SizeOverflow => formatter.write_str("context size overflow"),
        }
    }
}

impl std::error::Error for ContextError {}

/// Resolve captured attachment bytes into host-authorized context items.
///
/// Ordering is priority descending, then ID ascending. The operation is atomic:
/// exceeding either budget returns an error and no partial selection.
pub fn resolve_captured_attachments(
    attachments: &[CapturedAttachment],
    budget: ContextBudget,
) -> Result<Vec<ContextItem>, ContextError> {
    let mut ordered = attachments.to_vec();
    ordered.sort_by(|left, right| {
        right
            .priority
            .cmp(&left.priority)
            .then_with(|| left.id.cmp(&right.id))
    });

    let mut ids = HashSet::with_capacity(ordered.len());
    let mut byte_size = 0_usize;
    let mut token_estimate = 0_usize;
    let mut items = Vec::with_capacity(ordered.len());
    for attachment in ordered {
        if !ids.insert(attachment.id.clone()) {
            return Err(ContextError::DuplicateAttachmentId(attachment.id));
        }
        let bytes = attachment.payload.as_bytes();
        let item_bytes = bytes.len();
        let item_tokens = estimate_tokens(item_bytes);
        let output_sha256 = Sha256Digest::of_bytes(bytes);
        byte_size = byte_size
            .checked_add(item_bytes)
            .ok_or(ContextError::SizeOverflow)?;
        token_estimate = token_estimate
            .checked_add(item_tokens)
            .ok_or(ContextError::SizeOverflow)?;
        enforce_budget(byte_size, token_estimate, budget)?;

        items.push(ContextItem {
            id: attachment.id,
            resolver_id: CAPTURED_ATTACHMENT_RESOLVER_ID.to_owned(),
            resolver_version: CAPTURED_ATTACHMENT_RESOLVER_VERSION.to_owned(),
            output_plane: OutputPlane::Message,
            media_type: attachment.media_type,
            sources: attachment.sources,
            trust: Trust::UntrustedEvidence,
            freshness: Freshness::Captured,
            sensitivity: Sensitivity::Workspace,
            priority: attachment.priority,
            payload: ContextPayload::Inline {
                inline_payload: attachment.payload,
            },
            byte_size: item_bytes,
            token_estimate: item_tokens,
            output_sha256,
            created_at: attachment.created_at,
            expires_at: attachment.expires_at,
            warnings: Vec::new(),
        });
    }
    Ok(items)
}

/// Render captured attachments into exact model-visible message bytes.
///
/// The renderer verifies host-fixed authority and content attestations before
/// emitting anything. Its budget covers the full rendered envelope.
pub fn render_captured_attachments(
    items: &[ContextItem],
    budget: ContextBudget,
) -> Result<RenderedContext, ContextError> {
    let mut ordered: Vec<&ContextItem> = items.iter().collect();
    ordered.sort_by(|left, right| {
        right
            .priority
            .cmp(&left.priority)
            .then_with(|| left.id.cmp(&right.id))
    });

    let mut rendered = String::new();
    let mut ids = HashSet::with_capacity(ordered.len());
    for item in ordered {
        if !ids.insert(item.id.as_str()) {
            return Err(ContextError::DuplicateAttachmentId(item.id.clone()));
        }
        if item.output_plane != OutputPlane::Message
            || item.trust != Trust::UntrustedEvidence
            || item.sensitivity != Sensitivity::Workspace
        {
            return Err(ContextError::AuthorityMismatch(item.id.clone()));
        }
        let ContextPayload::Inline { inline_payload } = &item.payload else {
            return Err(ContextError::PayloadReferenceNotRenderable(item.id.clone()));
        };
        if inline_payload.len() != item.byte_size
            || estimate_tokens(item.byte_size) != item.token_estimate
            || Sha256Digest::of_bytes(inline_payload.as_bytes()) != item.output_sha256
        {
            return Err(ContextError::IntegrityMismatch(item.id.clone()));
        }

        rendered.push_str("--- captured attachment (untrusted workspace evidence) ---\n");
        rendered.push_str("id: ");
        rendered.push_str(&item.id);
        rendered.push_str("\nmedia-type: ");
        rendered.push_str(&item.media_type);
        rendered.push_str("\n\n");
        rendered.push_str(inline_payload);
        if !inline_payload.ends_with('\n') {
            rendered.push('\n');
        }
        rendered.push_str("--- end captured attachment ---\n");
    }

    let byte_size = rendered.len();
    let token_estimate = estimate_tokens(byte_size);
    enforce_budget(byte_size, token_estimate, budget)?;
    Ok(RenderedContext {
        sha256: Sha256Digest::of_bytes(rendered.as_bytes()),
        bytes: rendered,
        byte_size,
        token_estimate,
    })
}

fn estimate_tokens(byte_size: usize) -> usize {
    byte_size.div_ceil(4)
}

fn enforce_budget(
    byte_size: usize,
    token_estimate: usize,
    budget: ContextBudget,
) -> Result<(), ContextError> {
    if byte_size > budget.byte_budget {
        return Err(ContextError::ByteBudgetExceeded {
            required: byte_size,
            available: budget.byte_budget,
        });
    }
    if token_estimate > budget.token_budget {
        return Err(ContextError::TokenBudgetExceeded {
            required: token_estimate,
            available: budget.token_budget,
        });
    }
    Ok(())
}

fn require_field(field: &'static str, value: &str) -> Result<(), ContextError> {
    if value.is_empty() {
        Err(ContextError::EmptyField(field))
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(byte: u8) -> Sha256Digest {
        Sha256Digest::parse(format!("{byte:02x}").repeat(32)).unwrap()
    }

    fn attachment(id: &str, priority: i64, payload: &str) -> CapturedAttachment {
        CapturedAttachment::new(
            id,
            "text/plain; charset=utf-8",
            vec![SourceReference {
                reference: format!("capture:{id}"),
                sha256: digest(0x44),
            }],
            priority,
            payload,
            "2026-08-09T16:00:00Z",
        )
        .unwrap()
    }

    #[test]
    fn resolver_fixes_authority_and_attests_exact_utf8_bytes() {
        let items = resolve_captured_attachments(
            &[attachment("cafe", 10, "revision café Δ\n")],
            ContextBudget::new(18, 5),
        )
        .unwrap();

        let item = &items[0];
        assert_eq!(item.output_plane, OutputPlane::Message);
        assert_eq!(item.trust, Trust::UntrustedEvidence);
        assert_eq!(item.sensitivity, Sensitivity::Workspace);
        assert_eq!(item.byte_size, 18);
        assert_eq!(item.token_estimate, 5);
        assert_eq!(
            item.output_sha256.as_str(),
            "5fdac83c68be7342f99b71e31931d9c0e133c1400cc26ce12f73b5c99ea37369"
        );
    }

    #[test]
    fn resolver_order_is_stable_and_budget_failure_is_atomic() {
        let items = resolve_captured_attachments(
            &[
                attachment("z", 1, "z"),
                attachment("b", 2, "b"),
                attachment("a", 2, "a"),
            ],
            ContextBudget::new(3, 3),
        )
        .unwrap();
        assert_eq!(
            items
                .iter()
                .map(|item| item.id.as_str())
                .collect::<Vec<_>>(),
            ["a", "b", "z"]
        );

        assert_eq!(
            resolve_captured_attachments(
                &[attachment("first", 2, "1234"), attachment("second", 1, "5")],
                ContextBudget::new(4, 10),
            ),
            Err(ContextError::ByteBudgetExceeded {
                required: 5,
                available: 4
            })
        );
    }

    #[test]
    fn renderer_verifies_authority_integrity_and_full_envelope_budget() {
        let items = resolve_captured_attachments(
            &[attachment("one", 1, "payload")],
            ContextBudget::new(100, 100),
        )
        .unwrap();
        let rendered = render_captured_attachments(&items, ContextBudget::new(1024, 256)).unwrap();
        assert!(rendered.as_str().contains("untrusted workspace evidence"));
        assert!(rendered.as_str().contains("payload"));
        assert_eq!(rendered.byte_size(), rendered.as_str().len());
        assert_eq!(
            rendered.sha256(),
            &Sha256Digest::of_bytes(rendered.as_str().as_bytes())
        );

        assert!(matches!(
            render_captured_attachments(
                &items,
                ContextBudget::new(rendered.byte_size() - 1, usize::MAX),
            ),
            Err(ContextError::ByteBudgetExceeded { .. })
        ));
    }

    #[test]
    fn payload_shape_rejects_inline_and_reference_together() {
        let json = r#"{
            "id":"x","resolver_id":"captured-attachment","resolver_version":"1",
            "output_plane":"message","media_type":"text/plain","sources":[],
            "trust":"untrusted_evidence","freshness":"captured","sensitivity":"workspace",
            "priority":0,"inline_payload":"x","payload_ref":"artifact:x","byte_size":1,
            "token_estimate":1,"output_sha256":"2d711642b726b04401627ca9fbac32f5c8530fb1903cc4db02258717921a4881",
            "created_at":"2026-08-09T16:00:00Z","expires_at":null,"warnings":[]
        }"#;
        assert!(serde_json::from_str::<ContextItem>(json).is_err());
    }

    #[test]
    fn schema_fixture_round_trips_without_shape_drift() {
        let fixture = include_str!("../../../eval/fixtures/context-resolution.v1.json");
        let parsed: ContextResolution = serde_json::from_str(fixture).unwrap();
        assert_eq!(parsed.schema_version, CONTEXT_RESOLUTION_SCHEMA_VERSION);
        assert_eq!(parsed.items.len(), 1);
        assert_eq!(parsed.items[0].byte_size, 18);
        assert_eq!(
            serde_json::to_value(parsed).unwrap(),
            serde_json::from_str::<serde_json::Value>(fixture).unwrap()
        );
    }

    #[test]
    fn renderer_rejects_payload_references_and_authority_changes() {
        let mut item = resolve_captured_attachments(
            &[attachment("one", 1, "payload")],
            ContextBudget::new(100, 100),
        )
        .unwrap()
        .remove(0);
        item.payload = ContextPayload::Ref {
            payload_ref: "artifact:one".to_owned(),
        };
        assert_eq!(
            render_captured_attachments(&[item.clone()], ContextBudget::new(1024, 256)),
            Err(ContextError::PayloadReferenceNotRenderable(
                "one".to_owned()
            ))
        );

        item.payload = ContextPayload::Inline {
            inline_payload: "payload".to_owned(),
        };
        item.trust = Trust::OperatorInstruction;
        assert_eq!(
            render_captured_attachments(&[item], ContextBudget::new(1024, 256)),
            Err(ContextError::AuthorityMismatch("one".to_owned()))
        );
    }
}
