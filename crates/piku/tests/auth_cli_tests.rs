//! Offline subprocess checks for the explicit subscription route.
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

#[test]
fn captured_login_refuses_before_accessing_auth_store() {
    let root = tempfile::tempdir().unwrap();
    for args in [
        vec!["auth", "chatgpt", "login"],
        vec![
            "auth",
            "chatgpt",
            "login",
            "--registration",
            "00000000-0000-4000-8000-000000000000",
        ],
    ] {
        let output = command(root.path()).args(args).output().unwrap();
        assert!(!output.status.success());
        assert!(
            output.stdout.is_empty(),
            "captured login must not emit authorization URLs"
        );
        let error = String::from_utf8(output.stderr).unwrap();
        assert!(error.contains("requires an attended terminal"), "{error}");
        assert!(
            !root.path().join("config").exists(),
            "refusal must precede auth storage and discovery"
        );
    }
}

#[test]
fn captured_returning_login_cannot_disclose_retained_identity_hint() {
    let root = tempfile::tempdir().unwrap();
    let directory = root.path().join("config/piku/auth/chatgpt");
    std::fs::create_dir_all(&directory).unwrap();
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
    let output = command(root.path())
        .args(["auth", "chatgpt", "login", "--registration", id])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
    let error = String::from_utf8(output.stderr).unwrap();
    assert!(error.contains("requires an attended terminal"), "{error}");
    assert!(!error.contains(hint));
    assert!(!error.contains("id_token_hint"));
    assert_eq!(std::fs::read_to_string(registration).unwrap(), body);
    assert_eq!(
        std::fs::read_dir(directory).unwrap().count(),
        1,
        "refusal must not create host, lock, or selection state"
    );
}
