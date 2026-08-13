//! Single-use authority leases for write-capable web turns.

use std::collections::HashMap;
use std::fmt;
use std::path::{Component, Path, PathBuf};
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::Serialize;
use sha2::{Digest, Sha256};
use thiserror::Error;

const NONCE_BYTES: usize = 16;
const MAX_IDENTIFIER_BYTES: usize = 256;
const MAX_PROFILE_BYTES: usize = 4 * 1024;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Authority {
    #[default]
    ReadOnly,
    WorkspaceWrite,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct LeaseNonce([u8; NONCE_BYTES]);

impl fmt::Debug for LeaseNonce {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("LeaseNonce([REDACTED])")
    }
}

impl LeaseNonce {
    pub(crate) fn expose_token(self) -> String {
        const HEX: &[u8; 16] = b"0123456789abcdef";
        let mut token = String::with_capacity(NONCE_BYTES * 2);
        for byte in self.0 {
            token.push(HEX[usize::from(byte >> 4)] as char);
            token.push(HEX[usize::from(byte & 0x0f)] as char);
        }
        token
    }

    pub(crate) fn parse(token: &str) -> Result<Self, LeaseError> {
        if token.len() != NONCE_BYTES * 2 || !token.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            return Err(LeaseError::InvalidNonceSyntax);
        }
        let mut bytes = [0; NONCE_BYTES];
        for (output, pair) in bytes.iter_mut().zip(token.as_bytes().chunks_exact(2)) {
            *output = (hex_nibble(pair[0])? << 4) | hex_nibble(pair[1])?;
        }
        Ok(Self(bytes))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(crate) struct LeaseScope {
    pub authority: Authority,
    pub workspace_root: PathBuf,
    pub executor: String,
    pub thread_id: String,
    pub turn_id: String,
    pub prompt_digest: [u8; 32],
    pub start_deadline_ms: u64,
    pub expires_at_ms: u64,
    pub working_directory: PathBuf,
    pub environment_digest: [u8; 32],
    pub network_enabled: bool,
    pub tool_profile: String,
}

impl LeaseScope {
    pub(crate) fn validate(&self) -> Result<(), LeaseError> {
        bounded_identifier("executor", &self.executor, MAX_IDENTIFIER_BYTES)?;
        bounded_identifier("thread_id", &self.thread_id, MAX_IDENTIFIER_BYTES)?;
        bounded_identifier("turn_id", &self.turn_id, MAX_IDENTIFIER_BYTES)?;
        bounded_identifier("tool_profile", &self.tool_profile, MAX_PROFILE_BYTES)?;
        canonical_absolute_path("workspace_root", &self.workspace_root)?;
        canonical_absolute_path("working_directory", &self.working_directory)?;
        if !self.working_directory.starts_with(&self.workspace_root) {
            return Err(LeaseError::InvalidScope(
                "working_directory must be within workspace_root",
            ));
        }
        if self.expires_at_ms < self.start_deadline_ms {
            return Err(LeaseError::InvalidScope(
                "expires_at must not precede start_deadline",
            ));
        }
        Ok(())
    }

    pub(crate) fn digest(&self) -> Result<[u8; 32], LeaseError> {
        self.validate()?;
        let mut canonical = CanonicalDigest::new();
        canonical.field(b"piku.write-lease.v1");
        canonical.field(&[match self.authority {
            Authority::ReadOnly => 0,
            Authority::WorkspaceWrite => 1,
        }]);
        canonical.path(&self.workspace_root)?;
        canonical.text(&self.executor);
        canonical.text(&self.thread_id);
        canonical.text(&self.turn_id);
        canonical.field(&self.prompt_digest);
        canonical.field(&self.start_deadline_ms.to_be_bytes());
        canonical.field(&self.expires_at_ms.to_be_bytes());
        canonical.path(&self.working_directory)?;
        canonical.field(&self.environment_digest);
        canonical.field(&[u8::from(self.network_enabled)]);
        canonical.text(&self.tool_profile);
        Ok(canonical.finish())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(crate) struct LeaseSummary {
    pub authority: Authority,
    pub scope_digest: [u8; 32],
    pub issued_at_ms: u64,
    pub start_deadline_ms: u64,
    pub expires_at_ms: u64,
}

struct Lease {
    scope_digest: [u8; 32],
    summary: LeaseSummary,
}

#[derive(Default)]
pub(crate) struct WriteLeaseStore {
    leases: Mutex<HashMap<LeaseNonce, Lease>>,
}

impl WriteLeaseStore {
    pub(crate) fn issue(
        &self,
        scope: &LeaseScope,
    ) -> Result<(LeaseNonce, LeaseSummary), LeaseError> {
        let scope_digest = scope.digest()?;
        let summary = LeaseSummary {
            authority: scope.authority,
            scope_digest,
            issued_at_ms: unix_millis(SystemTime::now())?,
            start_deadline_ms: scope.start_deadline_ms,
            expires_at_ms: scope.expires_at_ms,
        };
        let mut leases = self.leases.lock().map_err(|_| LeaseError::Unavailable)?;
        let nonce = loop {
            let mut bytes = [0; NONCE_BYTES];
            getrandom::fill(&mut bytes).map_err(|_| LeaseError::RandomnessUnavailable)?;
            let nonce = LeaseNonce(bytes);
            if !leases.contains_key(&nonce) {
                break nonce;
            }
        };
        leases.insert(
            nonce,
            Lease {
                scope_digest,
                summary: summary.clone(),
            },
        );
        Ok((nonce, summary))
    }

    /// Consumes before validating. A bad digest or late request permanently revokes the lease.
    pub(crate) fn consume(
        &self,
        nonce: LeaseNonce,
        scope: &LeaseScope,
        now: SystemTime,
    ) -> Result<LeaseSummary, LeaseError> {
        let lease = self
            .leases
            .lock()
            .map_err(|_| LeaseError::Unavailable)?
            .remove(&nonce)
            .ok_or(LeaseError::UnknownOrConsumed)?;
        let supplied_digest = scope.digest().map_err(|_| LeaseError::ScopeMismatch)?;
        if !constant_time_eq(&lease.scope_digest, &supplied_digest) {
            return Err(LeaseError::ScopeMismatch);
        }
        let now_ms = unix_millis(now)?;
        if now_ms > lease.summary.expires_at_ms {
            return Err(LeaseError::Expired);
        }
        if now_ms > lease.summary.start_deadline_ms {
            return Err(LeaseError::StartDeadlineElapsed);
        }
        Ok(lease.summary)
    }

    #[allow(dead_code)] // Explicit revocation route is a separate operator-control slice.
    pub(crate) fn revoke(&self, nonce: LeaseNonce) -> Result<bool, LeaseError> {
        Ok(self
            .leases
            .lock()
            .map_err(|_| LeaseError::Unavailable)?
            .remove(&nonce)
            .is_some())
    }
}

#[derive(Debug, Error, PartialEq, Eq)]
pub(crate) enum LeaseError {
    #[error("invalid lease nonce syntax")]
    InvalidNonceSyntax,
    #[error("invalid lease scope: {0}")]
    InvalidScope(&'static str),
    #[error("lease nonce is unknown or already consumed")]
    UnknownOrConsumed,
    #[error("lease scope does not match")]
    ScopeMismatch,
    #[error("lease start deadline elapsed")]
    StartDeadlineElapsed,
    #[error("lease expired")]
    Expired,
    #[error("system clock predates the Unix epoch")]
    InvalidClock,
    #[error("secure randomness unavailable")]
    RandomnessUnavailable,
    #[error("lease store unavailable")]
    Unavailable,
}

fn hex_nibble(byte: u8) -> Result<u8, LeaseError> {
    match byte {
        b'0'..=b'9' => Ok(byte - b'0'),
        b'a'..=b'f' => Ok(byte - b'a' + 10),
        b'A'..=b'F' => Ok(byte - b'A' + 10),
        _ => Err(LeaseError::InvalidNonceSyntax),
    }
}

fn bounded_identifier(
    field: &'static str,
    value: &str,
    max_bytes: usize,
) -> Result<(), LeaseError> {
    if value.is_empty() || value.len() > max_bytes || value.chars().any(char::is_control) {
        return Err(LeaseError::InvalidScope(field));
    }
    Ok(())
}

fn canonical_absolute_path(field: &'static str, path: &Path) -> Result<(), LeaseError> {
    if !path.is_absolute()
        || path
            .components()
            .any(|component| matches!(component, Component::CurDir | Component::ParentDir))
    {
        return Err(LeaseError::InvalidScope(field));
    }
    if !matches!(path.canonicalize(), Ok(canonical) if canonical == path) {
        return Err(LeaseError::InvalidScope(field));
    }
    Ok(())
}

fn unix_millis(time: SystemTime) -> Result<u64, LeaseError> {
    let millis = time
        .duration_since(UNIX_EPOCH)
        .map_err(|_| LeaseError::InvalidClock)?
        .as_millis();
    u64::try_from(millis).map_err(|_| LeaseError::InvalidClock)
}

fn constant_time_eq(left: &[u8; 32], right: &[u8; 32]) -> bool {
    left.iter()
        .zip(right)
        .fold(0_u8, |difference, (left, right)| {
            difference | (left ^ right)
        })
        == 0
}

struct CanonicalDigest(Sha256);

impl CanonicalDigest {
    fn new() -> Self {
        Self(Sha256::new())
    }

    fn field(&mut self, value: &[u8]) {
        self.0.update((value.len() as u64).to_be_bytes());
        self.0.update(value);
    }

    fn text(&mut self, value: &str) {
        self.field(value.as_bytes());
    }

    fn path(&mut self, value: &Path) -> Result<(), LeaseError> {
        self.text(
            value
                .to_str()
                .ok_or(LeaseError::InvalidScope("paths must contain valid UTF-8"))?,
        );
        Ok(())
    }

    fn finish(self) -> [u8; 32] {
        self.0.finalize().into()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use std::sync::{Arc, Barrier};
    use std::thread;
    use std::time::Duration;

    use super::*;

    const NOW_MS: u64 = 1_000_000;

    fn scope() -> LeaseScope {
        let workspace_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .canonicalize()
            .unwrap();
        LeaseScope {
            authority: Authority::WorkspaceWrite,
            working_directory: workspace_root.join("src").canonicalize().unwrap(),
            workspace_root,
            executor: "codex".into(),
            thread_id: "thread-1".into(),
            turn_id: "turn-1".into(),
            prompt_digest: Sha256::digest(b"prompt").into(),
            start_deadline_ms: NOW_MS + 1_000,
            expires_at_ms: NOW_MS + 5_000,
            environment_digest: Sha256::digest(b"environment").into(),
            network_enabled: false,
            tool_profile: "default".into(),
        }
    }

    fn now(ms: u64) -> SystemTime {
        UNIX_EPOCH + Duration::from_millis(ms)
    }

    #[test]
    fn authority_defaults_to_read_only() {
        assert_eq!(Authority::default(), Authority::ReadOnly);
    }

    #[test]
    fn default_store_contains_no_lease() {
        let store = WriteLeaseStore::default();
        let nonce = LeaseNonce([0x5a; NONCE_BYTES]);

        assert_eq!(
            store.consume(nonce, &scope(), now(NOW_MS)),
            Err(LeaseError::UnknownOrConsumed)
        );
        assert!(!store.revoke(nonce).unwrap());
    }

    #[test]
    fn digest_is_stable() {
        assert_eq!(scope().digest().unwrap(), scope().digest().unwrap());
    }

    #[test]
    fn every_bound_field_change_mismatches_and_consumes() {
        let base = scope();
        let parent = base.workspace_root.parent().unwrap().to_path_buf();
        let variants: Vec<LeaseScope> = vec![
            LeaseScope {
                authority: Authority::ReadOnly,
                ..base.clone()
            },
            LeaseScope {
                workspace_root: parent.clone(),
                working_directory: parent,
                ..base.clone()
            },
            LeaseScope {
                executor: "other".into(),
                ..base.clone()
            },
            LeaseScope {
                thread_id: "thread-2".into(),
                ..base.clone()
            },
            LeaseScope {
                turn_id: "turn-2".into(),
                ..base.clone()
            },
            LeaseScope {
                prompt_digest: [9; 32],
                ..base.clone()
            },
            LeaseScope {
                start_deadline_ms: base.start_deadline_ms + 1,
                ..base.clone()
            },
            LeaseScope {
                expires_at_ms: base.expires_at_ms + 1,
                ..base.clone()
            },
            LeaseScope {
                working_directory: base.workspace_root.clone(),
                ..base.clone()
            },
            LeaseScope {
                environment_digest: [8; 32],
                ..base.clone()
            },
            LeaseScope {
                network_enabled: true,
                ..base.clone()
            },
            LeaseScope {
                tool_profile: "restricted".into(),
                ..base.clone()
            },
        ];

        for changed in variants {
            let store = WriteLeaseStore::default();
            let (nonce, _) = store.issue(&base).unwrap();
            assert_eq!(
                store.consume(nonce, &changed, now(NOW_MS)),
                Err(LeaseError::ScopeMismatch)
            );
            assert_eq!(
                store.consume(nonce, &base, now(NOW_MS)),
                Err(LeaseError::UnknownOrConsumed)
            );
        }
    }

    #[test]
    fn elapsed_deadline_consumes_lease() {
        let store = WriteLeaseStore::default();
        let lease_scope = scope();
        let (nonce, _) = store.issue(&lease_scope).unwrap();
        assert_eq!(
            store.consume(nonce, &lease_scope, now(lease_scope.start_deadline_ms + 1)),
            Err(LeaseError::StartDeadlineElapsed)
        );
        assert_eq!(
            store.consume(nonce, &lease_scope, now(NOW_MS)),
            Err(LeaseError::UnknownOrConsumed)
        );
    }

    #[test]
    fn expired_lease_is_rejected() {
        let store = WriteLeaseStore::default();
        let mut lease_scope = scope();
        lease_scope.start_deadline_ms = lease_scope.expires_at_ms;
        let (nonce, _) = store.issue(&lease_scope).unwrap();
        assert_eq!(
            store.consume(nonce, &lease_scope, now(lease_scope.expires_at_ms + 1)),
            Err(LeaseError::Expired)
        );
    }

    #[test]
    fn sequential_replay_fails() {
        let store = WriteLeaseStore::default();
        let lease_scope = scope();
        let (nonce, _) = store.issue(&lease_scope).unwrap();
        assert!(store.consume(nonce, &lease_scope, now(NOW_MS)).is_ok());
        assert_eq!(
            store.consume(nonce, &lease_scope, now(NOW_MS)),
            Err(LeaseError::UnknownOrConsumed)
        );
    }

    #[test]
    fn concurrent_replay_has_exactly_one_winner() {
        let store = Arc::new(WriteLeaseStore::default());
        let lease_scope = Arc::new(scope());
        let (nonce, _) = store.issue(&lease_scope).unwrap();
        let barrier = Arc::new(Barrier::new(8));
        let handles: Vec<_> = (0..8)
            .map(|_| {
                let store = Arc::clone(&store);
                let lease_scope = Arc::clone(&lease_scope);
                let barrier = Arc::clone(&barrier);
                thread::spawn(move || {
                    barrier.wait();
                    store.consume(nonce, &lease_scope, now(NOW_MS))
                })
            })
            .collect();
        let successes = handles
            .into_iter()
            .flat_map(|handle| handle.join().unwrap())
            .count();
        assert_eq!(successes, 1);
    }

    #[test]
    fn nonces_are_unique() {
        let store = WriteLeaseStore::default();
        let lease_scope = scope();
        let nonces: HashSet<_> = (0..1_024)
            .map(|_| store.issue(&lease_scope).unwrap().0)
            .collect();
        assert_eq!(nonces.len(), 1_024);
    }

    #[test]
    fn nonce_token_has_bounded_syntax_and_round_trips() {
        let store = WriteLeaseStore::default();
        let (nonce, _) = store.issue(&scope()).unwrap();
        let token = nonce.expose_token();
        assert_eq!(token.len(), 32);
        assert!(token.bytes().all(|byte| byte.is_ascii_hexdigit()));
        assert_eq!(LeaseNonce::parse(&token), Ok(nonce));
        assert_eq!(
            LeaseNonce::parse("short"),
            Err(LeaseError::InvalidNonceSyntax)
        );
        assert_eq!(
            LeaseNonce::parse("gggggggggggggggggggggggggggggggg"),
            Err(LeaseError::InvalidNonceSyntax)
        );
    }

    #[test]
    fn serialized_summary_excludes_nonce_prompt_and_environment_values() {
        let store = WriteLeaseStore::default();
        let (_, summary) = store.issue(&scope()).unwrap();
        let encoded = serde_json::to_string(&summary).unwrap();
        assert!(!encoded.contains("nonce"));
        assert!(!encoded.contains("prompt"));
        assert!(!encoded.contains("environment"));
    }
}
