//! Offline subprocess checks for the explicit subscription route.
use std::os::unix::fs::PermissionsExt;
use std::process::Command;

fn command(root: &std::path::Path) -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_piku"));
    command
        .current_dir(root)
        .env("XDG_CONFIG_HOME", root.join("config"))
        .env("PIKU_NO_DOTENV", "1")
        .env_remove("OPENROUTER_API_KEY")
        .env_remove("ANTHROPIC_API_KEY")
        .env_remove("GROQ_API_KEY")
        .env_remove("PIKU_BASE_URL")
        .env_remove("PIKU_RESTARTED");
    command
}

#[test]
fn empty_auth_status_is_metadata_only_and_funding_is_explicit() {
    let root = tempfile::tempdir().unwrap();
    let output = command(root.path())
        .args(["auth", "chatgpt", "status"])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let text = String::from_utf8(output.stdout).unwrap();
    assert!(text.contains("No ChatGPT accounts stored"));
    assert!(text.contains("No account selected"));
    assert!(text.contains("allowance and credits unknown"));
    assert!(text.contains("no hard output-token cap"));
    assert!(text.contains("no authentication or quota probe"));
    assert!(!root
        .path()
        .join("config/piku/auth/chatgpt/selected.json")
        .exists());
}

#[test]
fn explicit_chatgpt_does_not_fall_back_to_configured_custom_provider() {
    let root = tempfile::tempdir().unwrap();
    let output = command(root.path())
        .env("PIKU_BASE_URL", "http://127.0.0.1:1/v1")
        .env("PIKU_MODEL", "fallback-model")
        .args([
            "--provider",
            "chatgpt",
            "--model",
            "selected-model",
            "-p",
            "test",
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let error = String::from_utf8(output.stderr).unwrap();
    assert!(error.contains("no ChatGPT account selected"), "{error}");
    assert!(!error.contains("using custom provider"), "{error}");
}

#[test]
fn auth_select_requires_existing_account_and_does_not_write_selection() {
    let root = tempfile::tempdir().unwrap();
    let output = command(root.path())
        .args([
            "auth",
            "chatgpt",
            "select",
            "00000000-0000-4000-8000-000000000000",
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(!root
        .path()
        .join("config/piku/auth/chatgpt/selected.json")
        .exists());
}

#[test]
fn model_catalog_requires_explicit_selected_account() {
    let root = tempfile::tempdir().unwrap();
    let output = command(root.path())
        .args(["auth", "chatgpt", "models"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("no ChatGPT account selected"));
}

struct PendingLoginCleanup<'a>(&'a std::path::Path);
impl Drop for PendingLoginCleanup<'_> {
    fn drop(&mut self) {
        let _ = command(self.0)
            .args(["auth", "chatgpt", "cancel-login"])
            .output();
        for _ in 0..100 {
            if let Ok(output) = command(self.0).args(["auth", "chatgpt", "status"]).output() {
                let status = String::from_utf8_lossy(&output.stdout);
                if !status.contains("phase=waiting_for_browser")
                    && !status.contains("phase=signing_in")
                {
                    break;
                }
            }
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
    }
}

#[test]
#[serial_test::serial]
fn captured_login_returns_local_bootstrap_promptly() {
    let root = tempfile::tempdir().unwrap();
    let _cleanup = PendingLoginCleanup(root.path());
    let started = std::time::Instant::now();
    let output = command(root.path())
        .args(["auth", "chatgpt", "login"])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(started.elapsed() < std::time::Duration::from_secs(3));
    let text = String::from_utf8(output.stdout).unwrap();
    assert!(text.contains("http://127.0.0.1:"), "{text}");
    assert!(!text.contains("id_token_hint"));
    assert!(!text.contains("https://auth.openai.com"));
    assert!(
        !root
            .path()
            .join("config/piku/auth/chatgpt/host.json")
            .exists(),
        "opening bootstrap must precede provider setup"
    );
}

#[test]
#[serial_test::serial]
fn captured_returning_login_cannot_disclose_retained_identity_hint() {
    let root = tempfile::tempdir().unwrap();
    let _cleanup = PendingLoginCleanup(root.path());
    let directory = root.path().join("config/piku/auth/chatgpt");
    std::fs::create_dir_all(&directory).unwrap();
    std::fs::set_permissions(&directory, std::fs::Permissions::from_mode(0o700)).unwrap();
    let id = "00000000-0000-4000-8000-000000000000";
    let hint = "synthetic-retained-id-token-must-not-leak";
    let registration = directory.join(format!("registration-{id}.json"));
    let body = serde_json::json!({
        "version": 1, "id": id, "client_id": "synthetic-client", "host_id": "synthetic-host",
        "identity": {"issuer":"https://synthetic.invalid", "subject":"synthetic", "email":null},
        "tokens": {"access_token":"synthetic-access", "refresh_token":null,
            "id_token":hint, "nonce":"synthetic", "scopes":[], "expires_at":0}
    })
    .to_string();
    std::fs::write(&registration, &body).unwrap();
    std::fs::set_permissions(&registration, std::fs::Permissions::from_mode(0o600)).unwrap();
    let output = command(root.path())
        .args(["auth", "chatgpt", "login", "--registration", id])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(text.contains("http://127.0.0.1:"));
    assert!(!text.contains(hint));
    assert!(!text.contains("id_token_hint"));
    assert!(!text.contains("synthetic-access"));
    assert_eq!(std::fs::read_to_string(registration).unwrap(), body);
    assert!(!directory.join("host.json").exists());
    assert!(!directory.join("selected.json").exists());
}

#[test]
#[serial_test::serial]
fn captured_login_bootstrap_status_cancel_and_single_flight() {
    let root = tempfile::tempdir().unwrap();
    let cleanup = PendingLoginCleanup(root.path());
    let output = command(root.path())
        .args(["auth", "chatgpt", "login"])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let text = String::from_utf8(output.stdout).unwrap();
    let url = text
        .lines()
        .find(|line| line.starts_with("http://127.0.0.1:"))
        .unwrap();
    let client = reqwest::blocking::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .timeout(std::time::Duration::from_secs(2))
        .build()
        .unwrap();
    let response = client.get(url).send().unwrap();
    assert!(response.status().is_success());
    assert_eq!(response.headers().get("cache-control").unwrap(), "no-store");
    assert_eq!(
        response.headers().get("referrer-policy").unwrap(),
        "no-referrer"
    );
    let page = response.text().unwrap();
    assert!(page.contains("Continue"));
    assert!(!page.contains("id_token_hint"));
    assert!(!page.contains("auth.openai.com"));
    let second = command(root.path())
        .args(["auth", "chatgpt", "login"])
        .output()
        .unwrap();
    assert!(
        !second.status.success(),
        "concurrent login must not create another active helper"
    );
    let status = command(root.path())
        .args(["auth", "chatgpt", "status"])
        .output()
        .unwrap();
    assert!(String::from_utf8_lossy(&status.stdout).contains("phase=waiting_for_browser"));
    assert!(!root
        .path()
        .join("config/piku/auth/chatgpt/host.json")
        .exists());
    drop(cleanup);
    let status = command(root.path())
        .args(["auth", "chatgpt", "status"])
        .output()
        .unwrap();
    assert!(String::from_utf8_lossy(&status.stdout).contains("phase=cancelled"));
    assert!(
        client.get(url).send().is_err(),
        "cancel must close the bootstrap listener"
    );
}

#[test]
#[serial_test::serial]
fn captured_login_setup_error_returns_without_url() {
    let root = tempfile::tempdir().unwrap();
    let output = command(root.path())
        .args([
            "auth",
            "chatgpt",
            "login",
            "--registration",
            "00000000-0000-4000-8000-000000000000",
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
    let error = String::from_utf8_lossy(&output.stderr);
    assert!(error.contains("ChatGPT"), "{error}");
    assert!(!error.contains("id_token_hint"));
    assert!(!root
        .path()
        .join("config/piku/auth/chatgpt/host.json")
        .exists());
    let status = command(root.path())
        .args(["auth", "chatgpt", "status"])
        .output()
        .unwrap();
    assert!(!String::from_utf8_lossy(&status.stdout).contains("phase=waiting_for_browser"));
}
