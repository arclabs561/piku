use std::io::Read as _;
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt as _;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::Deserialize;
use sha2::{Digest, Sha256};

const SCHEMA: &str = "piku.codex-write-attestation.v3";
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
    "native_lifecycle_observed",
];

#[derive(Debug, Deserialize)]
struct Attestation {
    schema: String,
    piku_version: String,
    codex_version: String,
    codex_executable_path: PathBuf,
    codex_executable_sha256: String,
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
    executable: PathBuf,
    executable_sha256: String,
}

impl VerifiedWriteAttestation {
    pub(super) fn executable(&self) -> Result<&Path, &'static str> {
        if file_sha256(&self.executable)? != self.executable_sha256 {
            return Err("attested Codex executable changed after verification");
        }
        Ok(&self.executable)
    }
}

pub(super) fn verify<F>(
    path: &Path,
    launch_policy: &[u8],
    now: SystemTime,
    version_probe: F,
) -> Result<VerifiedWriteAttestation, &'static str>
where
    F: FnOnce(&Path) -> Result<String, &'static str>,
{
    let bytes = std::fs::read(path).map_err(|_| "workspace-write attestation is unavailable")?;
    let attestation: Attestation =
        serde_json::from_slice(&bytes).map_err(|_| "workspace-write attestation is invalid")?;
    if attestation.schema != SCHEMA
        || attestation.piku_version != env!("CARGO_PKG_VERSION")
        || attestation.host_os != std::env::consts::OS
        || attestation.host_arch != std::env::consts::ARCH
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
    if !attestation.codex_executable_path.is_absolute() {
        return Err("attested Codex executable path is not absolute");
    }
    let executable = attestation
        .codex_executable_path
        .canonicalize()
        .map_err(|_| "attested Codex executable is unavailable")?;
    if executable != attestation.codex_executable_path {
        return Err("attested Codex executable path is not canonical");
    }
    let metadata = executable
        .metadata()
        .map_err(|_| "attested Codex executable is unavailable")?;
    if !metadata.is_file() {
        return Err("attested Codex executable is not a regular file");
    }
    #[cfg(unix)]
    if metadata.permissions().mode() & 0o111 == 0 {
        return Err("attested Codex executable is not executable");
    }
    if file_sha256(&executable)? != attestation.codex_executable_sha256 {
        return Err("attested Codex executable digest does not match");
    }
    if version_probe(&executable)? != attestation.codex_version {
        return Err("workspace-write attestation does not match this runtime");
    }
    Ok(VerifiedWriteAttestation {
        executable,
        executable_sha256: attestation.codex_executable_sha256,
    })
}

fn file_sha256(path: &Path) -> Result<String, &'static str> {
    let mut file =
        std::fs::File::open(path).map_err(|_| "attested Codex executable is unavailable")?;
    let mut digest = Sha256::new();
    let mut buffer = vec![0_u8; 64 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .map_err(|_| "cannot read attested Codex executable")?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(policy: &[u8], now_ms: u64, executable: &Path) -> serde_json::Value {
        serde_json::json!({
            "schema": SCHEMA,
            "piku_version": env!("CARGO_PKG_VERSION"),
            "codex_version": "codex-cli test",
            "codex_executable_path": executable,
            "codex_executable_sha256": file_sha256(executable).unwrap(),
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
        let executable = std::env::current_exe().unwrap().canonicalize().unwrap();
        let now = UNIX_EPOCH + Duration::from_secs(10_000);
        std::fs::write(
            &path,
            serde_json::to_vec(&fixture(policy, 10_000_000, &executable)).unwrap(),
        )
        .unwrap();
        assert!(verify(&path, policy, now, |_| Ok("codex-cli test".to_string())).is_ok());

        let mut incomplete = fixture(policy, 10_000_000, &executable);
        incomplete["passed_gates"] = serde_json::json!(["initialized"]);
        std::fs::write(&path, serde_json::to_vec(&incomplete).unwrap()).unwrap();
        assert_eq!(
            verify(&path, policy, now, |_| Ok("codex-cli test".to_string())).unwrap_err(),
            "workspace-write attestation has incomplete gates"
        );
    }

    #[test]
    fn rejects_stale_policy_and_runtime_mismatches() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("attestation.json");
        let executable = std::env::current_exe().unwrap().canonicalize().unwrap();
        let now = UNIX_EPOCH + Duration::from_hours(192);
        std::fs::write(
            &path,
            serde_json::to_vec(&fixture(b"policy", 0, &executable)).unwrap(),
        )
        .unwrap();
        assert_eq!(
            verify(&path, b"other", now, |_| Ok("codex-cli test".to_string())).unwrap_err(),
            "workspace-write attestation does not match the launch policy"
        );
        assert_eq!(
            verify(&path, b"policy", now, |_| Ok("codex-cli test".to_string())).unwrap_err(),
            "workspace-write attestation is stale"
        );
        assert_eq!(
            verify(&path, b"policy", UNIX_EPOCH, |_| Ok(
                "other version".to_string()
            ))
            .unwrap_err(),
            "workspace-write attestation does not match this runtime"
        );

        std::fs::write(
            &path,
            serde_json::to_vec(&fixture(b"policy", 1, &executable)).unwrap(),
        )
        .unwrap();
        assert_eq!(
            verify(&path, b"policy", UNIX_EPOCH, |_| Ok(
                "codex-cli test".to_string()
            ))
            .unwrap_err(),
            "workspace-write attestation is dated in the future"
        );
    }

    #[test]
    fn verified_authority_rejects_executable_replacement() {
        let directory = tempfile::tempdir().unwrap();
        let executable = directory.path().join("codex-payload");
        std::fs::copy(std::env::current_exe().unwrap(), &executable).unwrap();
        let executable = executable.canonicalize().unwrap();
        let path = directory.path().join("attestation.json");
        let policy = b"policy";
        std::fs::write(
            &path,
            serde_json::to_vec(&fixture(policy, 10_000_000, &executable)).unwrap(),
        )
        .unwrap();
        let authority = verify(
            &path,
            policy,
            UNIX_EPOCH + Duration::from_secs(10_000),
            |_| Ok("codex-cli test".to_string()),
        )
        .unwrap();

        std::fs::write(&executable, b"replaced").unwrap();
        assert_eq!(
            authority.executable().unwrap_err(),
            "attested Codex executable changed after verification"
        );
    }
}
