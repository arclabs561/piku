use std::path::Path;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::Deserialize;
use sha2::{Digest, Sha256};

const SCHEMA: &str = "piku.codex-write-attestation.v1";
const MAX_AGE: Duration = Duration::from_hours(168);
const REQUIRED_GATES: &[&str] = &[
    "initialized",
    "thread_started",
    "thread_resumed",
    "turn_completed",
    "command_write_inside",
    "file_change_inside",
    "sibling_write_denied",
    "network_denied",
    "elevation_denied",
    "native_lifecycle_observed",
];

#[derive(Debug, Deserialize)]
struct Attestation {
    schema: String,
    piku_version: String,
    codex_version: String,
    host_os: String,
    host_arch: String,
    launch_policy_sha256: String,
    probed_at_unix_ms: u64,
    passed_gates: Vec<String>,
}

/// Proof that the current Codex runtime passed every required write gate.
///
/// The private field prevents callers from manufacturing write authority
/// without passing `verify`.
#[derive(Debug)]
pub(super) struct VerifiedWriteAttestation {
    _verified: (),
}

pub(super) fn verify(
    path: &Path,
    launch_policy: &[u8],
    codex_version: &str,
    now: SystemTime,
) -> Result<VerifiedWriteAttestation, &'static str> {
    let bytes = std::fs::read(path).map_err(|_| "workspace-write attestation is unavailable")?;
    let attestation: Attestation =
        serde_json::from_slice(&bytes).map_err(|_| "workspace-write attestation is invalid")?;
    if attestation.schema != SCHEMA
        || attestation.piku_version != env!("CARGO_PKG_VERSION")
        || attestation.host_os != std::env::consts::OS
        || attestation.host_arch != std::env::consts::ARCH
        || attestation.codex_version != codex_version
    {
        return Err("workspace-write attestation does not match this runtime");
    }
    let digest = format!("{:x}", Sha256::digest(launch_policy));
    if attestation.launch_policy_sha256 != digest {
        return Err("workspace-write attestation does not match the launch policy");
    }
    let now_ms = now
        .duration_since(UNIX_EPOCH)
        .map_err(|_| "system clock is before the Unix epoch")?
        .as_millis();
    let probed_at_ms = u128::from(attestation.probed_at_unix_ms);
    if probed_at_ms > now_ms {
        return Err("workspace-write attestation is dated in the future");
    }
    let age_ms = now_ms - probed_at_ms;
    if age_ms > MAX_AGE.as_millis() {
        return Err("workspace-write attestation is stale");
    }
    let mut passed = attestation
        .passed_gates
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>();
    passed.sort_unstable();
    let mut required = REQUIRED_GATES.to_vec();
    required.sort_unstable();
    if passed != required {
        return Err("workspace-write attestation has incomplete gates");
    }
    Ok(VerifiedWriteAttestation { _verified: () })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(policy: &[u8], now_ms: u64) -> serde_json::Value {
        serde_json::json!({
            "schema": SCHEMA,
            "piku_version": env!("CARGO_PKG_VERSION"),
            "codex_version": "codex-cli test",
            "host_os": std::env::consts::OS,
            "host_arch": std::env::consts::ARCH,
            "launch_policy_sha256": format!("{:x}", Sha256::digest(policy)),
            "probed_at_unix_ms": now_ms,
            "passed_gates": REQUIRED_GATES
        })
    }

    #[test]
    fn accepts_only_a_current_complete_matching_attestation() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("attestation.json");
        let policy = b"policy";
        let now = UNIX_EPOCH + Duration::from_secs(10_000);
        std::fs::write(
            &path,
            serde_json::to_vec(&fixture(policy, 10_000_000)).unwrap(),
        )
        .unwrap();
        assert!(verify(&path, policy, "codex-cli test", now).is_ok());

        let mut incomplete = fixture(policy, 10_000_000);
        incomplete["passed_gates"] = serde_json::json!(["initialized"]);
        std::fs::write(&path, serde_json::to_vec(&incomplete).unwrap()).unwrap();
        assert_eq!(
            verify(&path, policy, "codex-cli test", now).unwrap_err(),
            "workspace-write attestation has incomplete gates"
        );
    }

    #[test]
    fn rejects_stale_policy_and_runtime_mismatches() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("attestation.json");
        let now = UNIX_EPOCH + Duration::from_hours(192);
        std::fs::write(&path, serde_json::to_vec(&fixture(b"policy", 0)).unwrap()).unwrap();
        assert_eq!(
            verify(&path, b"other", "codex-cli test", now).unwrap_err(),
            "workspace-write attestation does not match the launch policy"
        );
        assert_eq!(
            verify(&path, b"policy", "codex-cli test", now).unwrap_err(),
            "workspace-write attestation is stale"
        );
        assert_eq!(
            verify(&path, b"policy", "other version", UNIX_EPOCH).unwrap_err(),
            "workspace-write attestation does not match this runtime"
        );

        std::fs::write(&path, serde_json::to_vec(&fixture(b"policy", 1)).unwrap()).unwrap();
        assert_eq!(
            verify(&path, b"policy", "codex-cli test", UNIX_EPOCH).unwrap_err(),
            "workspace-write attestation is dated in the future"
        );
    }
}
