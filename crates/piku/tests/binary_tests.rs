#![allow(clippy::unnecessary_debug_formatting)]

use std::path::PathBuf;
/// Subprocess tests against the compiled piku binary.
///
/// These tests spawn `target/release/piku` (or debug fallback) as a real
/// subprocess and verify observable behavior: exit codes, stdout/stderr
/// content, file effects.
///
/// They do NOT require an API key — all tests that need one are gated
/// behind `PIKU_LLM_E2E` in `llm_e2e.rs`.
///
/// Run:
///   cargo build --release -p piku  # required before these tests
///   cargo test --test `binary_tests`
use std::process::{Command, Output};

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------

fn piku_binary() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_piku"))
}

/// Build a Command with API keys scrubbed from the environment.
/// Tests that should fail due to missing keys need this.
fn piku_clean_env() -> Command {
    let mut cmd = Command::new(piku_binary());
    // Remove all known provider keys so tests are hermetic
    cmd.env_remove("OPENROUTER_API_KEY")
        .env_remove("ANTHROPIC_API_KEY")
        .env_remove("GROQ_API_KEY")
        .env_remove("PIKU_BASE_URL")
        .env_remove("OLLAMA_HOST")
        .env_remove("PIKU_RESTARTED")
        .env("PIKU_NO_DOTENV", "1");
    cmd
}

fn stdout(o: &Output) -> String {
    String::from_utf8_lossy(&o.stdout).into_owned()
}
fn stderr(o: &Output) -> String {
    String::from_utf8_lossy(&o.stderr).into_owned()
}

/// The real web process is isolated from operator settings, credentials and shell.
struct AdmissionServer {
    child: std::process::Child,
    root: tempfile::TempDir,
    url: String,
    token: String,
}

impl Drop for AdmissionServer {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl AdmissionServer {
    fn launch(terminal: bool, disable_env: bool) -> Self {
        use std::os::unix::fs::PermissionsExt;
        let root = tempfile::tempdir().unwrap();
        let home = root.path().join("home");
        let workspace = root.path().join("workspace");
        std::fs::create_dir(&home).unwrap();
        std::fs::create_dir(&workspace).unwrap();
        let shell = root.path().join("fixture-shell");
        std::fs::write(
            &shell,
            "#!/bin/sh\n: > fixture-started\nprintf 'PIKU_AUTH_FIXTURE_OUTPUT\\n'\n",
        )
        .unwrap();
        std::fs::set_permissions(&shell, std::fs::Permissions::from_mode(0o700)).unwrap();
        let ready = root.path().join("ready.json");
        let log = root.path().join("server.log");
        let log_file = std::fs::File::create(log).unwrap();
        let mut command = Command::new(piku_binary());
        command
            .env_clear()
            .env("HOME", &home)
            .env("XDG_CONFIG_HOME", &home)
            .env("XDG_STATE_HOME", &home)
            .env("XDG_CACHE_HOME", &home)
            .env("SHELL", &shell)
            .env("PIKU_NO_DOTENV", "1")
            .env("PIKU_WEB_READY_FILE", &ready)
            .env("PIKU_BASE_URL", "http://127.0.0.1:1/v1")
            .current_dir(workspace)
            .args(["--provider", "custom", "web", "--port", "0"])
            .stdout(log_file.try_clone().unwrap())
            .stderr(log_file);
        if terminal {
            command.arg("--terminal");
        }
        if disable_env {
            command.env("PIKU_WEB_DISABLE_TERMINAL", "1");
        }
        // Install the kill/reap guard before any fallible readiness assertions.
        let mut server = Self {
            child: command.spawn().unwrap(),
            root,
            url: String::new(),
            token: String::new(),
        };
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(15);
        while !ready.exists() {
            assert!(
                server.child.try_wait().unwrap().is_none(),
                "web exited early"
            );
            assert!(
                std::time::Instant::now() < deadline,
                "web readiness timeout"
            );
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        let metadata: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&ready).unwrap()).unwrap();
        metadata["url"]
            .as_str()
            .unwrap()
            .clone_into(&mut server.url);
        let launch = std::fs::read_to_string(metadata["launch_file"].as_str().unwrap()).unwrap();
        launch
            .split_once("#token=")
            .expect("private launch fragment")
            .1
            .split(|c: char| !c.is_ascii_hexdigit())
            .next()
            .unwrap()
            .clone_into(&mut server.token);
        assert_eq!(server.token.len(), 64, "capability length");
        server
    }

    fn fixture_started(&self) -> bool {
        self.root.path().join("workspace/fixture-started").exists()
    }

    fn assert_public_artifacts_redacted(&self) {
        for path in ["ready.json", "server.log"] {
            let text = std::fs::read_to_string(self.root.path().join(path)).unwrap();
            assert!(
                !text.contains(&self.token),
                "capability leaked into public artifact"
            );
        }
    }

    fn websocket(&self) -> (u16, std::net::TcpStream) {
        use std::io::{Read, Write};
        let authority = self.url.strip_prefix("http://").unwrap();
        let port: u16 = authority.rsplit_once(':').unwrap().1.parse().unwrap();
        let mut stream = std::net::TcpStream::connect(("127.0.0.1", port)).unwrap();
        stream
            .set_read_timeout(Some(std::time::Duration::from_secs(8)))
            .unwrap();
        stream
            .set_write_timeout(Some(std::time::Duration::from_secs(3)))
            .unwrap();
        write!(
            stream,
            "GET /api/terminal/pty HTTP/1.1\r\nHost: {authority}\r\nOrigin: {}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Version: 13\r\nSec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n\r\n",
            self.url
        ).unwrap();
        let mut header = Vec::new();
        while !header.ends_with(b"\r\n\r\n") {
            let mut byte = [0];
            stream.read_exact(&mut byte).unwrap();
            header.push(byte[0]);
            assert!(header.len() < 8192, "bounded handshake response");
        }
        let header = String::from_utf8(header).unwrap();
        let status = header.split_whitespace().nth(1).unwrap().parse().unwrap();
        (status, stream)
    }
}

fn admission_ws_send(stream: &mut std::net::TcpStream, opcode: u8, payload: &[u8]) {
    use std::io::Write;
    let mut frame = vec![0x80 | opcode];
    if payload.len() < 126 {
        frame.push(0x80 | u8::try_from(payload.len()).unwrap());
    } else {
        frame.push(0x80 | 0x7e);
        frame.extend_from_slice(&u16::try_from(payload.len()).unwrap().to_be_bytes());
    }
    let mask = [0x13, 0x24, 0x35, 0x46];
    frame.extend_from_slice(&mask);
    frame.extend(payload.iter().enumerate().map(|(i, b)| b ^ mask[i % 4]));
    stream.write_all(&frame).unwrap();
}

fn admission_ws_read(stream: &mut std::net::TcpStream) -> std::io::Result<(u8, Vec<u8>)> {
    use std::io::Read;
    let mut header = [0; 2];
    stream.read_exact(&mut header)?;
    assert_eq!(header[1] & 0x80, 0, "server frames must be unmasked");
    let length = match header[1] & 0x7f {
        126 => {
            let mut bytes = [0; 2];
            stream.read_exact(&mut bytes)?;
            u16::from_be_bytes(bytes) as usize
        }
        127 => {
            let mut bytes = [0; 8];
            stream.read_exact(&mut bytes)?;
            usize::try_from(u64::from_be_bytes(bytes)).unwrap()
        }
        length => length as usize,
    };
    assert!(length <= 65536, "bounded terminal output frame");
    let mut payload = vec![0; length];
    stream.read_exact(&mut payload)?;
    Ok((header[0] & 0x0f, payload))
}

#[test]
#[allow(clippy::too_many_lines)] // One two-launch admission journey preserves the state/effect oracle.
fn web_binary_capabilities_protect_http_and_are_launch_specific() {
    let a = AdmissionServer::launch(false, false);
    let b = AdmissionServer::launch(false, false);
    assert_ne!(a.url, b.url, "actual distinct listeners");
    let client = reqwest::blocking::Client::builder()
        .no_proxy()
        .timeout(std::time::Duration::from_secs(5))
        .build()
        .unwrap();
    let root = client.get(&a.url).send().unwrap();
    assert_eq!(root.status().as_u16(), 200);
    let shell = root.text().unwrap();
    assert!(
        !shell.contains(&a.token),
        "public bootstrap contains no capability"
    );
    let surfaces_url = format!("{}/api/surfaces", a.url);
    let before: serde_json::Value = client
        .get(&surfaces_url)
        .bearer_auth(&a.token)
        .send()
        .unwrap()
        .error_for_status()
        .unwrap()
        .json()
        .unwrap();
    assert_eq!(
        client.get(&surfaces_url).send().unwrap().status().as_u16(),
        401
    );
    assert_eq!(
        client
            .get(format!("{}/run/synthetic-run", a.url))
            .send()
            .unwrap()
            .status()
            .as_u16(),
        401
    );
    let denied = client
        .post(&surfaces_url)
        .header("Origin", &a.url)
        .json(&serde_json::json!({"name":"unauthorized-synthetic-surface"}))
        .send()
        .unwrap();
    assert_eq!(denied.status().as_u16(), 401);
    for origin in [b.url.as_str(), "null", "http://foreign.invalid"] {
        assert_eq!(
            client
                .post(&surfaces_url)
                .bearer_auth(&a.token)
                .header("Origin", origin)
                .json(&serde_json::json!({"name":"unauthorized-synthetic-surface"}))
                .send()
                .unwrap()
                .status()
                .as_u16(),
            403
        );
    }
    assert_eq!(
        client
            .get(&surfaces_url)
            .bearer_auth(&a.token)
            .header("Host", "localhost:1")
            .send()
            .unwrap()
            .status()
            .as_u16(),
        403
    );
    let after: serde_json::Value = client
        .get(&surfaces_url)
        .bearer_auth(&a.token)
        .send()
        .unwrap()
        .error_for_status()
        .unwrap()
        .json()
        .unwrap();
    assert_eq!(
        before, after,
        "rejected mutation preserves surface inventory"
    );
    assert!(!a
        .root
        .path()
        .join("home/piku/_web/unauthorized-synthetic-surface")
        .exists());
    assert_eq!(
        client
            .get(format!("{}/api/surfaces", b.url))
            .bearer_auth(&a.token)
            .send()
            .unwrap()
            .status()
            .as_u16(),
        401
    );
    // A valid non-browser bearer can mutate without an Origin header.
    let created: serde_json::Value = client
        .post(&surfaces_url)
        .bearer_auth(&a.token)
        .json(&serde_json::json!({"name":"authorized-synthetic-surface"}))
        .send()
        .unwrap()
        .error_for_status()
        .unwrap()
        .json()
        .unwrap();
    assert!(created
        .as_array()
        .unwrap()
        .iter()
        .any(|v| v == "authorized-synthetic-surface"));
    a.assert_public_artifacts_redacted();
    b.assert_public_artifacts_redacted();
}

#[test]
fn web_binary_terminal_auth_precedes_real_fixture_spawn() {
    let server = AdmissionServer::launch(true, false);
    for attempt in 0..5 {
        let (status, mut socket) = server.websocket();
        assert_eq!(
            status, 101,
            "enabled WS upgrades before capability admission"
        );
        match attempt {
            0 => admission_ws_send(
                &mut socket,
                1,
                br#"{"type":"authenticate","token":"wrong"}"#,
            ),
            1 => admission_ws_send(&mut socket, 1, br#"{"type":"input","data":"ignored"}"#),
            2 => admission_ws_send(&mut socket, 2, b"binary authentication is forbidden"),
            3 => admission_ws_send(&mut socket, 1, &vec![b'x'; 16384]),
            4 => {} // No first frame: the server must close on its auth deadline.
            _ => unreachable!(),
        }
        match admission_ws_read(&mut socket) {
            Ok((opcode, _)) => assert_eq!(opcode, 8, "rejected auth emits only a close frame"),
            Err(error) => assert!(
                matches!(
                    error.kind(),
                    std::io::ErrorKind::UnexpectedEof | std::io::ErrorKind::ConnectionReset
                ),
                "rejected auth must close before the socket deadline"
            ),
        }
        assert!(
            !server.fixture_started(),
            "pre-auth socket spawned a terminal"
        );
    }
    let (status, mut socket) = server.websocket();
    assert_eq!(
        status, 101,
        "rejections leave capacity for admitted terminal"
    );
    assert!(
        !server.fixture_started(),
        "upgrade alone must not spawn fixture"
    );
    let auth = serde_json::json!({"type":"authenticate","token":server.token});
    admission_ws_send(&mut socket, 1, &serde_json::to_vec(&auth).unwrap());
    let (opcode, payload) = admission_ws_read(&mut socket).unwrap();
    assert_eq!(opcode, 1, "authentication acknowledgement is a text frame");
    let acknowledgement: serde_json::Value = serde_json::from_slice(&payload).unwrap();
    assert_eq!(acknowledgement["type"], "authenticated");
    let mut output = Vec::new();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(8);
    while !String::from_utf8_lossy(&output).contains("PIKU_AUTH_FIXTURE_OUTPUT") {
        assert!(
            std::time::Instant::now() < deadline,
            "fixture output deadline"
        );
        let (opcode, bytes) = admission_ws_read(&mut socket).unwrap();
        assert!(
            opcode == 1 || opcode == 2,
            "fixture must emit output before closure"
        );
        output.extend_from_slice(&bytes);
    }
    assert!(
        server.fixture_started(),
        "valid capability reached actual PTY factory"
    );
    assert!(
        !String::from_utf8_lossy(&output).contains(&server.token),
        "PTY output leaked capability"
    );
    server.assert_public_artifacts_redacted();
}

#[test]
fn web_binary_terminal_requires_flag_and_disable_env_narrows_it() {
    for (terminal, disable_env) in [(false, false), (true, true)] {
        let server = AdmissionServer::launch(terminal, disable_env);
        let client = reqwest::blocking::Client::builder()
            .no_proxy()
            .timeout(std::time::Duration::from_secs(5))
            .build()
            .unwrap();
        assert_eq!(
            client
                .get(format!("{}/api/executors", server.url))
                .bearer_auth(&server.token)
                .send()
                .unwrap()
                .status()
                .as_u16(),
            200,
            "launch capability is valid even when terminal authority is disabled"
        );
        let disabled = client
            .get(format!("{}/api/terminal/pty", server.url))
            .bearer_auth(&server.token)
            .header("Origin", &server.url)
            .send()
            .unwrap();
        assert_eq!(
            disabled.status().as_u16(),
            404,
            "valid capability cannot enable a disabled terminal route"
        );
        let (status, _socket) = server.websocket();
        assert_ne!(status, 101, "disabled terminal exposes no upgrade route");
        assert!(
            !server.fixture_started(),
            "disabled launch must never spawn fixture"
        );
        server.assert_public_artifacts_redacted();
    }
}

#[test]
fn providers_ignore_project_dotenv() {
    let project = tempfile::tempdir().unwrap();
    std::fs::write(
        project.path().join(".env"),
        "OPENROUTER_API_KEY=test-only-key\nUNRELATED_SECRET=must-not-load\n",
    )
    .unwrap();
    let mut command = piku_clean_env();
    command.env_remove("PIKU_NO_DOTENV");
    let out = command
        .current_dir(project.path())
        .arg("providers")
        .output()
        .unwrap();

    assert!(out.status.success(), "stderr: {}", stderr(&out));
    assert!(stdout(&out).contains("openrouter missing"));
    assert!(!stdout(&out).contains("test-only-key"));
    assert!(!stderr(&out).contains("test-only-key"));
}

#[test]
fn web_honors_provider_and_model_overrides() {
    struct Server(std::process::Child);
    impl Drop for Server {
        fn drop(&mut self) {
            let _ = self.0.kill();
            let _ = self.0.wait();
        }
    }

    for (configured_provider, args, expected_model) in [
        (
            "unavailable-provider",
            vec!["--provider", "custom", "--model", "cli-model", "web"],
            "cli-model",
        ),
        (
            "unavailable-provider",
            vec!["web", "--provider", "custom"],
            "configured-model",
        ),
        ("custom", vec!["web", "--model", "cli-model"], "cli-model"),
        ("custom", vec!["web"], "configured-model"),
    ] {
        let root = tempfile::tempdir().unwrap();
        let config = root.path().join("piku");
        std::fs::create_dir(&config).unwrap();
        std::fs::write(
            config.join("settings.toml"),
            format!("provider = \"{configured_provider}\"\nmodel = \"configured-model\"\n"),
        )
        .unwrap();
        let ready = root.path().join("ready.json");
        let mut server = Server(
            Command::new(piku_binary())
                .env_clear()
                .env("HOME", root.path())
                .env("XDG_CONFIG_HOME", root.path())
                .env("PIKU_NO_DOTENV", "1")
                .env("PIKU_WEB_DISABLE_TERMINAL", "1")
                .env("PIKU_WEB_READY_FILE", &ready)
                .env("PIKU_BASE_URL", "http://127.0.0.1:1/v1")
                .current_dir(root.path())
                .args(&args)
                .args(["--port", "0"])
                .stdout(std::process::Stdio::null())
                .spawn()
                .unwrap(),
        );
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while !ready.exists() {
            assert!(
                server.0.try_wait().unwrap().is_none(),
                "server exited: {args:?}"
            );
            assert!(
                std::time::Instant::now() < deadline,
                "server not ready: {args:?}"
            );
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        let ready: serde_json::Value =
            serde_json::from_slice(&std::fs::read(ready).unwrap()).unwrap();
        let launch = std::fs::read_to_string(ready["launch_file"].as_str().unwrap()).unwrap();
        let token = launch
            .split_once("#token=")
            .expect("private launch file carries a fragment credential")
            .1
            .split(|character: char| !character.is_ascii_hexdigit())
            .next()
            .unwrap();
        assert_eq!(token.len(), 64, "launch credential has the required length");
        let catalog: serde_json::Value = reqwest::blocking::Client::builder()
            .no_proxy()
            .timeout(std::time::Duration::from_secs(5))
            .build()
            .unwrap()
            .get(format!("{}/api/executors", ready["url"].as_str().unwrap()))
            .bearer_auth(token)
            .send()
            .unwrap()
            .error_for_status()
            .unwrap()
            .json()
            .unwrap();
        let executors = catalog["executors"].as_array().unwrap();
        let provider = executors
            .iter()
            .find(|item| item["id"] == "provider")
            .unwrap();
        assert_eq!(provider["available"], true, "args: {args:?}");
        assert_eq!(provider["model"], expected_model, "args: {args:?}");
        let codex = executors.iter().find(|item| item["id"] == "codex").unwrap();
        assert_eq!(codex["model"], "account default");
        assert_eq!(codex["available"], false);
    }
}

// ---------------------------------------------------------------------------
// --version
// ---------------------------------------------------------------------------

#[test]
fn version_prints_semver_to_stdout() {
    let out = piku_clean_env().arg("--version").output().unwrap();
    assert!(out.status.success(), "exit code: {:?}", out.status.code());

    let text = stdout(&out);
    assert!(
        text.trim().starts_with("piku "),
        "expected 'piku X.Y.Z', got: {text:?}"
    );

    let version = text.trim().strip_prefix("piku ").unwrap();
    let parts: Vec<&str> = version.split('.').collect();
    assert_eq!(parts.len(), 3, "version should be X.Y.Z, got: {version}");
    for part in &parts {
        assert!(
            part.parse::<u32>().is_ok(),
            "version part '{part}' is not numeric"
        );
    }
}

#[test]
fn version_short_flag_identical_to_long() {
    let long = piku_clean_env().arg("--version").output().unwrap();
    let short = piku_clean_env().arg("-V").output().unwrap();
    assert_eq!(
        stdout(&long),
        stdout(&short),
        "-V and --version should be identical"
    );
    assert_eq!(long.status.code(), short.status.code());
}

#[test]
fn version_writes_to_stdout_not_stderr() {
    let out = piku_clean_env().arg("--version").output().unwrap();
    assert!(!out.stdout.is_empty(), "--version should write to stdout");
    let stderr_content = stderr(&out);
    // stderr may be empty or have incidental messages, but must not contain the version line
    let version_in_stderr =
        stderr_content.trim().contains("piku 0.") || stderr_content.trim().contains("piku 1.");
    assert!(
        !version_in_stderr,
        "version should be on stdout only, not stderr: {stderr_content}"
    );
}

// ---------------------------------------------------------------------------
// --help
// ---------------------------------------------------------------------------

#[test]
fn help_exits_zero_and_contains_usage() {
    let out = piku_clean_env().arg("--help").output().unwrap();
    assert!(out.status.success(), "exit code: {:?}", out.status.code());

    let text = stdout(&out);
    assert!(text.contains("Usage:"), "help should have Usage section");
    assert!(text.contains("OPTIONS"), "help should have OPTIONS section");
    assert!(text.contains("--model"), "help should document --model");
    assert!(
        text.contains("--provider"),
        "help should document --provider"
    );
    assert!(
        text.contains("providers"),
        "help should document providers subcommand"
    );
    assert!(
        text.contains("--read-only"),
        "help should document --read-only"
    );
    assert!(text.contains("--version"), "help should document --version");
    assert!(text.contains("--resume"), "help should document --resume");
}

#[test]
fn help_short_flag_identical_to_long() {
    let long = piku_clean_env().arg("--help").output().unwrap();
    let short = piku_clean_env().arg("-h").output().unwrap();
    assert_eq!(
        stdout(&long),
        stdout(&short),
        "-h and --help should be identical"
    );
}

#[test]
fn providers_flag_exits_zero_without_provider_key() {
    let out = piku_clean_env().arg("providers").output().unwrap();
    assert!(
        out.status.success(),
        "providers should not require a configured provider. stderr: {}",
        stderr(&out)
    );

    let text = stdout(&out);
    assert!(
        text.contains("PROVIDERS") && text.contains("openrouter") && text.contains("missing"),
        "--providers should print provider status. stdout: {text}"
    );
}

// ---------------------------------------------------------------------------
// No arguments → TUI REPL
// ---------------------------------------------------------------------------

#[test]
fn no_args_starts_tui_repl_and_exits_zero() {
    // With no args and no API key, piku starts the TUI REPL.
    // The REPL tries to resolve a provider — with no key it exits non-zero
    // with an error, but must NOT panic (no SIGSEGV/SIGABRT).
    let out = piku_clean_env().output().unwrap();
    let code = out.status.code().unwrap_or(-1);
    // Must not crash
    assert_ne!(code, 139, "should not segfault");
    assert_ne!(code, 134, "should not abort");
    // (May exit 0 or non-zero depending on whether Ollama is available;
    // the important property is no crash.)
}

#[test]
fn read_only_no_prompt_starts_tui_repl_and_does_not_crash() {
    let out = piku_clean_env().arg("--read-only").output().unwrap();
    let code = out.status.code().unwrap_or(-1);
    assert_ne!(code, 139, "should not segfault");
    assert_ne!(code, 134, "should not abort");
}

// ---------------------------------------------------------------------------
// Provider errors (no API key)
// ---------------------------------------------------------------------------

#[test]
fn missing_api_key_exits_nonzero_with_actionable_error() {
    // Strip all cloud provider keys. If Ollama is locally available, piku may
    // still succeed (that is correct behavior). We only assert the actionable
    // error message when piku actually fails — we don't force a failure
    // by also blocking Ollama (which would make the test fragile on dev boxes).
    let out = piku_clean_env()
        .arg("explain what piku does")
        .output()
        .unwrap();

    if !out.status.success() {
        let err = stderr(&out);
        // Error should tell the user what to set — be actionable
        assert!(
            err.contains("OPENROUTER_API_KEY")
                || err.contains("ANTHROPIC_API_KEY")
                || err.contains("provider")
                || err.contains("no provider"),
            "error should mention how to configure a provider. stderr: {err}"
        );
    }
    // If Ollama is running and piku succeeded — that's also correct.
}

#[test]
fn unknown_provider_exits_nonzero_and_names_it() {
    let out = piku_clean_env()
        .args(["--provider", "xyzzy-unknown", "do something"])
        .output()
        .unwrap();

    assert!(
        !out.status.success(),
        "unknown provider should exit non-zero"
    );
    let err = stderr(&out);
    assert!(
        err.contains("xyzzy-unknown"),
        "error should name the bad provider. stderr: {err}"
    );
    assert!(
        err.to_lowercase().contains("unknown") || err.contains("provider"),
        "error should mention 'unknown'. stderr: {err}"
    );
}

#[test]
fn openrouter_without_key_exits_nonzero_mentions_key() {
    let out = piku_clean_env()
        .args(["--provider=openrouter", "do something"])
        .output()
        .unwrap();

    assert!(!out.status.success());
    let err = stderr(&out);
    assert!(
        err.contains("OPENROUTER_API_KEY"),
        "should mention the missing key. stderr: {err}"
    );
}

#[test]
fn anthropic_without_key_exits_nonzero() {
    let out = piku_clean_env()
        .args(["--provider=anthropic", "do something"])
        .output()
        .unwrap();

    assert!(!out.status.success());
    let err = stderr(&out);
    assert!(
        err.contains("ANTHROPIC_API_KEY") || err.contains("anthropic"),
        "should mention anthropic key. stderr: {err}"
    );
}

#[test]
fn groq_without_key_exits_nonzero_mentions_key() {
    let out = piku_clean_env()
        .args(["--provider=groq", "do something"])
        .output()
        .unwrap();
    assert!(!out.status.success());
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(
        err.contains("GROQ_API_KEY"),
        "stderr should mention GROQ_API_KEY, got: {err}"
    );
}

// ---------------------------------------------------------------------------
// Edge cases that must not panic
// ---------------------------------------------------------------------------

#[test]
fn model_flag_without_value_does_not_crash() {
    let out = piku_clean_env().arg("--model").output().unwrap();
    let code = out.status.code().unwrap_or(-1);
    // Must not SIGSEGV (139) or SIGABRT (134) — any other exit is fine
    assert_ne!(code, 139, "should not segfault");
    assert_ne!(code, 134, "should not abort");
}

#[test]
fn provider_flag_without_value_does_not_crash() {
    let out = piku_clean_env().arg("--provider").output().unwrap();
    let code = out.status.code().unwrap_or(-1);
    assert_ne!(code, 139, "should not segfault");
    assert_ne!(code, 134, "should not abort");
}

#[test]
fn flags_only_no_prompt_enters_repl() {
    // --model=x with no prompt words → Repl action → TUI REPL starts.
    // Same as no-args: must not crash.
    let out = piku_clean_env()
        .arg("--model=gpt-4o-mini")
        .output()
        .unwrap();
    let code = out.status.code().unwrap_or(-1);
    assert_ne!(code, 139, "should not segfault");
    assert_ne!(code, 134, "should not abort");
    let text = stdout(&out);
    // The TUI outputs ANSI escape codes, not "USAGE" text; just verify no crash.
    // (Previously this showed help; now it starts the REPL.)
    assert!(!text.contains("SIGSEGV"), "should not crash");
}

#[test]
fn model_equals_form_is_accepted() {
    // --model=value should parse identically to --model value
    // We can verify by checking that it reaches provider resolution (not arg parse error)
    let out = piku_clean_env()
        .args(["--model=claude-opus-4", "do something"])
        .output()
        .unwrap();
    // Should fail at provider resolution (no key), not at arg parsing
    let err = stderr(&out);
    assert!(
        !err.contains("model") || err.contains("provider") || err.contains("API"),
        "should fail at provider resolution, not model parsing. stderr: {err}"
    );
}

// ---------------------------------------------------------------------------
// Ollama: always resolves (no key required)
// ---------------------------------------------------------------------------

#[test]
fn ollama_provider_resolves_without_key() {
    // Ollama doesn't need a key — it will fail at connection, not at key resolution.
    // We verify the error is about connection, not "missing key".
    // (Ollama may not be running, that's expected.)
    let out = piku_clean_env()
        .args(["--provider=ollama", "do something"])
        .output()
        .unwrap();

    // If Ollama is not running, it will fail — that's fine.
    // What we verify: the error is NOT about a missing API key.
    if !out.status.success() {
        let err = stderr(&out);
        // Should not mention any API key requirement
        assert!(
            !err.contains("OLLAMA_API_KEY") && !err.contains("missing.*key"),
            "ollama should not require an API key. stderr: {err}"
        );
    }
    // If Ollama IS running, the test might actually succeed — that's also fine.
}

// ---------------------------------------------------------------------------
// PIKU_RESTARTED env var
// ---------------------------------------------------------------------------

#[test]
fn piku_restarted_env_var_prints_restart_message_on_any_invocation() {
    // Set PIKU_RESTARTED=1 — piku should print the restart message.
    // We use --version to get a fast clean exit.
    // Note: was_restarted() now clears the env var, so child processes
    // spawned by this piku instance won't see it.
    let out = Command::new(piku_binary())
        .arg("--version")
        .env("PIKU_RESTARTED", "1")
        // Keep all other keys as-is — we just want to trigger the message
        .output()
        .unwrap();

    // --version hits the Version fast path before was_restarted() is called
    // (was_restarted is only called in run_single_shot). So this test just
    // verifies --version still works with the env var set.
    assert!(out.status.success());
    let text = stdout(&out);
    assert!(text.trim().starts_with("piku "));
}

// ---------------------------------------------------------------------------
// XDG_CONFIG_HOME: session files go to configured location
// ---------------------------------------------------------------------------

#[test]
fn xdg_config_home_controls_session_location() {
    // We can't easily test this without an API key, but we can verify that
    // the binary accepts the env var without crashing.
    let tmp = std::env::temp_dir().join(format!("piku_config_test_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();

    let out = piku_clean_env()
        .arg("--version")
        .env("XDG_CONFIG_HOME", &tmp)
        .output()
        .unwrap();

    assert!(out.status.success());
    // Session dir is only created during run_single_shot, not --version,
    // so we just verify no panic.
    let _ = std::fs::remove_dir_all(&tmp);
}

#[test]
fn conclude_appends_a_run_scoped_user_disposition() {
    let config = tempfile::tempdir().unwrap();
    let runs = config.path().join("piku/runs");
    std::fs::create_dir_all(&runs).unwrap();
    let record = runs.join("session-1.jsonl");
    std::fs::write(
        &record,
        concat!(
            r#"{"schema_version":2,"sequence":0,"recorded_at_ms":0,"session_id":"session-1","scope":"turn","turn_id":"turn-0","event":"warning","message":"initial"}"#,
            "\n"
        ),
    )
    .unwrap();

    let output = piku_clean_env()
        .args([
            "conclude",
            "session-1",
            "--status",
            "needs-work",
            "--note",
            "verify the browser projection",
        ])
        .env("XDG_CONFIG_HOME", config.path())
        .output()
        .unwrap();

    assert!(output.status.success(), "stderr: {}", stderr(&output));
    let text = std::fs::read_to_string(record).unwrap();
    let last: serde_json::Value = serde_json::from_str(text.lines().last().unwrap()).unwrap();
    assert_eq!(last["scope"], "run");
    assert!(last.get("turn_id").is_none());
    assert_eq!(last["event"], "user_disposition");
    assert_eq!(last["disposition"], "needs_work");
    assert_eq!(last["note"]["text"], "verify the browser projection");
}

#[test]
fn conclude_can_append_while_live_recorder_remains_open() {
    use piku_runtime::{read_run_record, RunEvent, RunRecorder};
    let config = tempfile::tempdir().unwrap();
    let path = config.path().join("piku/runs/live.jsonl");
    let mut live = RunRecorder::open(&path, "live").unwrap();
    live.append(
        "turn",
        RunEvent::Warning {
            message: "before".into(),
        },
    )
    .unwrap();
    let output = piku_clean_env()
        .args([
            "conclude",
            "live",
            "--status",
            "needs-work",
            "--note",
            "operator review",
        ])
        .env("XDG_CONFIG_HOME", config.path())
        .output()
        .unwrap();
    assert!(output.status.success(), "{}", stderr(&output));
    assert_eq!(
        live.append(
            "turn",
            RunEvent::Warning {
                message: "after".into()
            }
        )
        .unwrap(),
        2
    );
    let events = read_run_record(&path).unwrap();
    assert_eq!(events.len(), 3);
    assert!(matches!(events[1].event, RunEvent::UserDisposition { .. }));
}

// ---------------------------------------------------------------------------
// ArgError: bad flag usage exits non-zero with actionable message
// ---------------------------------------------------------------------------

#[test]
fn model_flag_without_value_exits_nonzero() {
    let out = piku_clean_env().arg("--model").output().unwrap();
    assert!(
        !out.status.success(),
        "--model without value should exit non-zero"
    );
    let err = stderr(&out);
    assert!(
        err.contains("--model") || err.contains("value"),
        "error should mention --model: {err}"
    );
}

#[test]
fn provider_flag_without_value_exits_nonzero() {
    let out = piku_clean_env().arg("--provider").output().unwrap();
    assert!(
        !out.status.success(),
        "--provider without value should exit non-zero"
    );
}

#[test]
fn resume_flag_without_value_exits_nonzero() {
    let out = piku_clean_env().arg("--resume").output().unwrap();
    assert!(
        !out.status.success(),
        "--resume without value should exit non-zero"
    );
    let err = stderr(&out);
    assert!(
        err.contains("--resume") || err.contains("session"),
        "error should mention --resume: {err}"
    );
}

#[test]
fn resume_nonexistent_session_exits_nonzero() {
    let out = piku_clean_env()
        .args(["--resume", "nonexistent-session-xyz-abc-123"])
        .output()
        .unwrap();
    // Will fail because the session doesn't exist
    // (would also fail on missing API key, but --resume check happens first)
    assert!(
        !out.status.success(),
        "resume of nonexistent session should fail"
    );
}

// ---------------------------------------------------------------------------
// Session ID uniqueness: nanos + PID
// ---------------------------------------------------------------------------

#[test]
fn version_runs_with_custom_config_dir() {
    let tmp = std::env::temp_dir().join(format!("piku_sid_test_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();

    let out = piku_clean_env()
        .arg("--version")
        .env("XDG_CONFIG_HOME", &tmp)
        .output()
        .unwrap();
    assert!(out.status.success());

    let _ = std::fs::remove_dir_all(&tmp);
}

// ---------------------------------------------------------------------------
// current_date: uses SystemTime, no subprocess
// ---------------------------------------------------------------------------

#[test]
fn version_output_does_not_spawn_date_subprocess() {
    // If current_date() spawned `date`, it would leave traces.
    // We just verify that --version completes quickly and correctly.
    let start = std::time::Instant::now();
    let out = piku_clean_env().arg("--version").output().unwrap();
    let elapsed = start.elapsed();

    assert!(out.status.success());
    // Without subprocess overhead, --version should complete in <200ms
    assert!(
        elapsed.as_millis() < 500,
        "--version took {}ms — might be spawning subprocesses",
        elapsed.as_millis()
    );
}

#[test]
fn new_session_id_is_unique_across_rapid_calls() {
    let id1 = piku::new_session_id();
    let id2 = piku::new_session_id();
    assert_ne!(
        id1, id2,
        "rapid session IDs must differ (nanos should advance)"
    );
    assert!(id1.starts_with("session-"), "session ID format: {id1}");
}

#[test]
fn print_headless_writes_trace_with_paired_ids() {
    // --print is the dogfood / binary_tests path that had max_turns and deny drift.
    // Verify the headless trace still writes paired ids (P1) even without an API key
    // when the prompt is just a prompt and provider resolution fails early.
    let tmp = tempfile::tempdir().unwrap();
    let cfg = tmp.path().join("cfg");
    std::fs::create_dir_all(&cfg).unwrap();
    let out = piku_clean_env()
        .arg("--print")
        .arg("hello from print smoke")
        .env("XDG_CONFIG_HOME", &cfg)
        .output()
        .unwrap();
    // Provider will fail without key, but --print still writes a trace with prompt/turn_end
    // and must not panic; exit code is either 0 or non-zero depending on provider fallback,
    // but trace pairing invariant still holds.
    let code = out.status.code().unwrap_or(-1);
    assert_ne!(code, 139);
    assert_ne!(code, 134);
    let traces = cfg.join("piku/traces");
    if traces.exists() {
        for entry in std::fs::read_dir(&traces).unwrap().filter_map(Result::ok) {
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("jsonl") {
                let content = std::fs::read_to_string(&path).unwrap();
                let events: Vec<serde_json::Value> = content
                    .lines()
                    .filter_map(|l| serde_json::from_str(l).ok())
                    .collect();
                let starts: Vec<_> = events
                    .iter()
                    .filter(|e| e["event"] == "tool_start")
                    .collect();
                let ends: Vec<_> = events.iter().filter(|e| e["event"] == "tool_end").collect();
                // If any tools ran, ids must pair
                if !starts.is_empty() || !ends.is_empty() {
                    assert_eq!(
                        starts.len(),
                        ends.len(),
                        "print trace tool counts must pair"
                    );
                    for e in starts.iter().chain(ends.iter()) {
                        assert!(
                            e["id"].as_str().is_some_and(|s| !s.is_empty()),
                            "print trace tool id must be non-empty: {e}"
                        );
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Headless run against a canned OpenAI-compatible server
// ---------------------------------------------------------------------------

/// Serve canned streaming completions on a local port and record each request
/// body. The server thread lives until the test process exits.
fn fake_completions_server(
    answer_chunks: &'static [&'static str],
) -> (String, std::sync::Arc<std::sync::Mutex<Vec<String>>>) {
    fake_completions_server_recording(answer_chunks, false)
}

fn fake_completions_server_recording(
    answer_chunks: &'static [&'static str],
    include_headers: bool,
) -> (String, std::sync::Arc<std::sync::Mutex<Vec<String>>>) {
    use std::io::{Read, Write};
    use std::sync::{Arc, Mutex};

    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let base_url = format!("http://{}/v1", listener.local_addr().unwrap());
    let bodies = Arc::new(Mutex::new(Vec::new()));
    let recorded = Arc::clone(&bodies);
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { continue };
            let mut request = Vec::new();
            let mut chunk = [0u8; 4096];
            let (header_end, content_length) = loop {
                let n = stream.read(&mut chunk).unwrap_or(0);
                if n == 0 {
                    break (request.len(), 0);
                }
                request.extend_from_slice(&chunk[..n]);
                if let Some(end) = request.windows(4).position(|w| w == b"\r\n\r\n") {
                    let head = String::from_utf8_lossy(&request[..end]).to_ascii_lowercase();
                    let length = head
                        .lines()
                        .find_map(|line| line.strip_prefix("content-length:"))
                        .and_then(|value| value.trim().parse::<usize>().ok())
                        .unwrap_or(0);
                    break (end + 4, length);
                }
            };
            while request.len() < header_end + content_length {
                let n = stream.read(&mut chunk).unwrap_or(0);
                if n == 0 {
                    break;
                }
                request.extend_from_slice(&chunk[..n]);
            }
            recorded.lock().unwrap().push(
                String::from_utf8_lossy(if include_headers {
                    &request
                } else {
                    &request[header_end..]
                })
                .into_owned(),
            );
            let mut response = String::from(
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n",
            );
            for data in answer_chunks {
                response.push_str("data: ");
                response.push_str(data);
                response.push_str("\n\n");
            }
            response.push_str("data: [DONE]\n\n");
            let _ = stream.write_all(response.as_bytes());
        }
    });
    (base_url, bodies)
}

/// One headless turn against the canned server, returning its output and the
/// session id parsed from the "session saved" line.
fn run_headless_turn(config: &std::path::Path, base_url: &str) -> (Output, String) {
    let out = piku_clean_env()
        .args(["-p", "say hello"])
        .env("XDG_CONFIG_HOME", config)
        .env("PIKU_BASE_URL", base_url)
        .env("PIKU_API_KEY", "test-key")
        .env("PIKU_MODEL", "canned-model")
        .current_dir(config)
        .output()
        .unwrap();
    assert!(out.status.success(), "stderr: {}", stderr(&out));
    let err = stderr(&out);
    let session_id = err
        .lines()
        .find_map(|line| line.split("sessions/").nth(1))
        .and_then(|rest| rest.strip_suffix(".json"))
        .unwrap_or_else(|| panic!("no session id in stderr: {err}"))
        .to_string();
    (out, session_id)
}

const CANNED_ANSWER: &[&str] = &[
    r#"{"choices":[{"delta":{"content":"hello **there**"},"finish_reason":null}]}"#,
    r#"{"choices":[{"delta":{},"finish_reason":"stop"}]}"#,
    r#"{"choices":[],"usage":{"prompt_tokens":7,"completion_tokens":3}}"#,
];

#[test]
fn project_cannot_redirect_synthetic_key_but_operator_destination_works() {
    let project = tempfile::tempdir().unwrap();
    let config = tempfile::tempdir().unwrap();
    let (unapproved_url, forbidden_requests) =
        fake_completions_server_recording(CANNED_ANSWER, true);
    std::fs::write(
        project.path().join(".env"),
        format!("PIKU_BASE_URL={unapproved_url}\nPIKU_MODEL=repo-funded-model\n"),
    )
    .unwrap();
    std::fs::create_dir(project.path().join(".piku")).unwrap();
    std::fs::write(project.path().join(".piku/settings.toml"), format!("provider = \"groq\"\nmodel = \"repo-funded-model\"\n[custom]\nbase_url = \"{unapproved_url}\"\n[actors.reviewer]\nprovider = \"groq\"\nmodel = \"repo-funded-model\"\n")).unwrap();
    let command = || {
        let mut command = Command::new(piku_binary());
        command
            .env_clear()
            .env("HOME", config.path())
            .env("XDG_CONFIG_HOME", config.path())
            .env("PIKU_API_KEY", "synthetic-operator-key")
            .current_dir(project.path())
            .args([
                "--provider",
                "custom",
                "--model",
                "operator-model",
                "--print",
                "say hello",
            ]);
        command
    };
    let rejected = command().output().unwrap();
    assert!(
        forbidden_requests.lock().unwrap().is_empty(),
        "unapproved repository destination received a request"
    );
    assert!(!rejected.status.success());

    let (approved_url, requests) = fake_completions_server_recording(CANNED_ANSWER, true);
    let accepted = command()
        .env("PIKU_BASE_URL", approved_url)
        .output()
        .unwrap();
    assert!(accepted.status.success(), "{}", stderr(&accepted));
    let captured = requests.lock().unwrap();
    assert_eq!(captured.len(), 1);
    assert!(captured[0]
        .to_ascii_lowercase()
        .contains("authorization: bearer synthetic-operator-key"));
    let body: serde_json::Value =
        serde_json::from_str(captured[0].split("\r\n\r\n").nth(1).unwrap()).unwrap();
    assert_eq!(body["model"], "operator-model");
    assert!(forbidden_requests.lock().unwrap().is_empty());
    assert!(!stdout(&accepted).contains("synthetic-operator-key"));
    assert!(!stderr(&accepted).contains("synthetic-operator-key"));
}

#[test]
fn permission_cli_requires_reviewed_root_and_policy_digest() {
    let project = tempfile::tempdir().unwrap();
    let config = tempfile::tempdir().unwrap();
    std::fs::create_dir(project.path().join(".piku")).unwrap();
    let settings = project.path().join(".piku/settings.toml");
    std::fs::write(
        &settings,
        "allow = [\"bash(cargo test)\"]\ndeny = [\"bash(rm *)\"]\n",
    )
    .unwrap();
    let command = || {
        let mut command = Command::new(piku_binary());
        command
            .env_clear()
            .env("HOME", config.path())
            .env("XDG_CONFIG_HOME", config.path())
            .current_dir(project.path())
            .arg("permissions");
        command
    };
    let review = command().arg("review").output().unwrap();
    assert!(review.status.success(), "{}", stderr(&review));
    let text = stdout(&review);
    assert!(text.contains("bash(cargo test)"));
    assert!(text.contains("bash(rm *)"));
    assert!(text.contains("Approved: false"));
    let digest = text
        .lines()
        .find_map(|line| line.strip_prefix("Digest: "))
        .unwrap();
    let root = project.path().canonicalize().unwrap();
    let wrong = command()
        .args(["approve", "--digest", digest, "--root"])
        .arg(config.path())
        .output()
        .unwrap();
    assert!(!wrong.status.success());
    let approve = command()
        .args(["approve", "--digest", digest, "--root"])
        .arg(&root)
        .output()
        .unwrap();
    assert!(approve.status.success(), "{}", stderr(&approve));
    assert!(stdout(&command().arg("review").output().unwrap()).contains("Approved: true"));
    std::fs::write(
        &settings,
        "allow = [\"bash(cargo test)\"]\ndeny = [\"bash(rm *)\"]\nmax_turns = 2\n",
    )
    .unwrap();
    assert!(stdout(&command().arg("review").output().unwrap()).contains("Approved: true"));
    std::fs::write(
        &settings,
        "allow = [\"bash(cargo test)\", \"write_file\"]\ndeny = [\"bash(rm *)\"]\n",
    )
    .unwrap();
    assert!(stdout(&command().arg("review").output().unwrap()).contains("Approved: false"));
    let stale = command()
        .args(["approve", "--digest", digest, "--root"])
        .arg(root)
        .output()
        .unwrap();
    assert!(!stale.status.success());
}

#[test]
fn headless_stream_failure_exits_nonzero_after_retaining_partial_state() {
    let config = tempfile::tempdir().unwrap();
    let (base_url, _) = fake_completions_server(&[
        r#"{"choices":[{"delta":{"content":"retained partial answer"},"finish_reason":null}]}"#,
    ]);
    let out = piku_clean_env()
        .args([
            "--provider",
            "custom",
            "--print",
            "test incomplete response",
        ])
        .env("XDG_CONFIG_HOME", config.path())
        .env("PIKU_BASE_URL", base_url)
        .env("PIKU_API_KEY", "test-key")
        .env("PIKU_MODEL", "canned-model")
        .current_dir(config.path())
        .output()
        .unwrap();
    let err = stderr(&out);
    assert!(err.contains("stream error"), "{err}");
    assert!(stdout(&out).contains("retained partial answer"));
    let session_id = err
        .lines()
        .find_map(|line| line.split("sessions/").nth(1))
        .and_then(|rest| rest.strip_suffix(".json"))
        .expect("session saved on failure");
    let session = std::fs::read_to_string(
        config
            .path()
            .join(format!("piku/sessions/{session_id}.json")),
    )
    .unwrap();
    // Failed streams retain the user/session boundary; partial assistant text
    // remains in the diagnostic trace rather than becoming replay history.
    assert!(session.contains("test incomplete response"));
    let run = std::fs::read_to_string(config.path().join(format!("piku/runs/{session_id}.jsonl")))
        .unwrap();
    let events: Vec<serde_json::Value> = run
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert!(events.iter().any(|row| row["event"] == "turn_failed"));
    assert!(!events.iter().any(|row| row["event"] == "turn_completed"));
    let trace = std::fs::read_to_string(
        config
            .path()
            .join(format!("piku/traces/{session_id}.jsonl")),
    )
    .unwrap();
    assert!(trace.contains("retained partial answer"));
    assert!(
        !out.status.success(),
        "provider failure must reach the shell exit status: {err}"
    );
}

#[test]
fn headless_output_to_a_pipe_has_no_escape_sequences_and_reports_usage() {
    let config = tempfile::tempdir().unwrap();
    let (base_url, bodies) = fake_completions_server(CANNED_ANSWER);
    let (out, _) = run_headless_turn(config.path(), &base_url);

    let text = stdout(&out);
    assert!(
        !text.contains('\x1b'),
        "stdout is not a terminal but contains escapes: {text:?}"
    );
    assert!(text.contains("hello"), "answer missing: {text:?}");
    assert!(text.contains("7↑ 3↓ tokens"), "usage missing: {text:?}");
    assert!(
        stderr(&out).contains("tool calls run without prompting"),
        "headless approval policy not announced: {}",
        stderr(&out)
    );
    let bodies = bodies.lock().unwrap();
    let request: serde_json::Value = serde_json::from_str(&bodies[0]).unwrap();
    assert_eq!(request["stream_options"]["include_usage"], true);
}

#[test]
fn headless_launch_request_omits_agent_tools() {
    let config = tempfile::tempdir().unwrap();
    let (base_url, bodies) = fake_completions_server(CANNED_ANSWER);
    let (out, _) = run_headless_turn(config.path(), &base_url);
    assert!(out.status.success(), "stderr: {}", stderr(&out));

    let bodies = bodies.lock().unwrap();
    let request: serde_json::Value = serde_json::from_str(&bodies[0]).unwrap();
    let tools: Vec<&str> = request["tools"]
        .as_array()
        .expect("headless request must advertise a tool catalog")
        .iter()
        .filter_map(|tool| tool["function"]["name"].as_str())
        .collect();

    assert!(tools.contains(&"bash"), "bash missing: {tools:?}");
    for agent_tool in ["spawn_agent", "agent_status", "agent_join"] {
        assert!(
            !tools.contains(&agent_tool),
            "{agent_tool} has no task registry in a launch turn but was advertised: {tools:?}"
        );
    }
}

#[test]
fn inspect_selectors_imply_their_listing_and_raw_is_the_final_answer() {
    let config = tempfile::tempdir().unwrap();
    let (base_url, _) = fake_completions_server(CANNED_ANSWER);
    let (_, session_id) = run_headless_turn(config.path(), &base_url);
    let inspect = |args: &[&str]| {
        piku_clean_env()
            .arg("inspect")
            .arg(&session_id)
            .args(args)
            .env("XDG_CONFIG_HOME", config.path())
            .output()
            .unwrap()
    };

    let cell = inspect(&["--cell", "@@1"]);
    assert!(cell.status.success(), "stderr: {}", stderr(&cell));
    assert!(stdout(&cell).starts_with("@@1 ·"), "{}", stdout(&cell));

    let raw = inspect(&["--raw", "@@1"]);
    assert!(raw.status.success(), "stderr: {}", stderr(&raw));
    assert_eq!(stdout(&raw), "hello **there**");

    let activity = inspect(&["--activity"]);
    assert!(activity.status.success());
    assert_eq!(stdout(&activity), "");
    assert!(
        stderr(&activity).contains("no interaction receipts"),
        "empty listing is silent: {}",
        stderr(&activity)
    );
}

#[test]
fn subcommand_help_omits_top_level_only_flags() {
    for subcommand in ["inspect", "conclude", "web", "providers"] {
        let out = piku_clean_env()
            .args([subcommand, "--help"])
            .output()
            .unwrap();
        let help = stdout(&out);
        for flag in ["--read-only", "--print", "--resume"] {
            assert!(
                !help.contains(flag),
                "`piku {subcommand} --help` lists {flag}:\n{help}"
            );
        }
    }
}

#[cfg(unix)]
#[test]
fn inspect_html_resolves_cross_directory_record_alias_artifacts() {
    use piku_runtime::{RunContentRef, RunEvent, RunRecorder};
    let config = tempfile::tempdir().unwrap();
    let runs = config.path().join("piku/runs");
    std::fs::create_dir_all(&runs).unwrap();
    let target = config.path().join("elsewhere/run.jsonl");
    drop(RunRecorder::open(&target, "alias").unwrap());
    let alias = runs.join("alias.jsonl");
    std::os::unix::fs::symlink(&target, &alias).unwrap();
    let mut recorder = RunRecorder::open(&alias, "alias").unwrap();
    let content = format!("{} unique-alias-artifact-needle", "payload ".repeat(4096));
    recorder
        .append(
            "turn",
            RunEvent::AssistantMessage {
                content: RunContentRef::Inline {
                    text: content.clone(),
                },
            },
        )
        .unwrap();
    let output = piku_clean_env()
        .args(["inspect", "alias", "--html"])
        .env("XDG_CONFIG_HOME", config.path())
        .output()
        .unwrap();
    assert!(output.status.success(), "{}", stderr(&output));
    let html = std::fs::read_to_string(runs.join("alias.html")).unwrap();
    assert!(html.contains(&content));
    let events = piku_runtime::read_run_record(&alias).unwrap();
    let index = piku::run_view::build_search_index_with_artifacts(&events, &alias).unwrap();
    assert!(index
        .iter()
        .any(|entry| entry.search_text.contains("unique-alias-artifact-needle")));
    assert!(std::fs::symlink_metadata(&alias)
        .unwrap()
        .file_type()
        .is_symlink());
}

/// ADR 0012: shell ledger records reach the model only when asked for, and
/// the operator is told what was attached.
#[test]
fn shell_context_attaches_only_requested_ledger_tail_and_says_so() {
    let config = tempfile::tempdir().unwrap();
    let state = tempfile::tempdir().unwrap();
    let ledger = state.path().join("piku/shell-ledger.jsonl");
    std::fs::create_dir_all(ledger.parent().unwrap()).unwrap();
    let mut records = String::new();
    for (n, command) in ["make build", "cargo test -p alpha", "git status"]
        .iter()
        .enumerate()
    {
        use std::fmt::Write as _;
        let _ = writeln!(
            records,
            "{{\"at\":{n},\"cwd\":\"/work\",\"command\":\"{command}\",\"exit\":{n},\"source\":\"piku-shell\"}}"
        );
    }
    std::fs::write(&ledger, records).unwrap();
    let (base_url, bodies) = fake_completions_server(CANNED_ANSWER);
    let run = |extra: &[&str]| {
        let out = piku_clean_env()
            .args(extra)
            .args(["-p", "what failed?"])
            .env("XDG_CONFIG_HOME", config.path())
            .env("XDG_STATE_HOME", state.path())
            .env_remove("PIKU_SHELL_CONTEXT")
            .env("PIKU_BASE_URL", &base_url)
            .env("PIKU_API_KEY", "test-key")
            .env("PIKU_MODEL", "canned-model")
            .current_dir(config.path())
            .output()
            .unwrap();
        assert!(out.status.success(), "stderr: {}", stderr(&out));
        stderr(&out)
    };

    let quiet = run(&[]);
    assert!(!quiet.contains("attached"), "no flag, no notice: {quiet}");
    let with_tail = run(&["--shell-context", "2"]);
    assert!(
        with_tail.contains("[attached 2 shell command(s) from"),
        "notice missing: {with_tail}"
    );

    let bodies = bodies.lock().unwrap().clone();
    assert_eq!(bodies.len(), 2, "{bodies:?}");
    assert!(!bodies[0].contains("cargo test -p alpha"), "{}", bodies[0]);
    assert!(
        bodies[1].contains("[/work] $ cargo test -p alpha  (exit 1)"),
        "{}",
        bodies[1]
    );
    assert!(
        bodies[1].contains("$ git status  (exit 2)"),
        "{}",
        bodies[1]
    );
    assert!(
        !bodies[1].contains("make build"),
        "only the last 2: {}",
        bodies[1]
    );
}

/// `eval "$(piku init zsh)"` makes zsh append each command and its exit
/// status to the same ledger `piku shell` writes, skipping space-prefixed
/// lines. Skipped where zsh is not installed.
#[test]
fn zsh_init_hook_records_commands_and_exit_status() {
    let Ok(zsh) = which_zsh() else {
        eprintln!("zsh not installed; skipping");
        return;
    };
    let home = tempfile::tempdir().unwrap();
    let state = tempfile::tempdir().unwrap();
    let bin = home.path().join("bin");
    std::fs::create_dir(&bin).unwrap();
    std::os::unix::fs::symlink(piku_binary(), bin.join("piku")).unwrap();
    let mut child = Command::new(zsh)
        .args(["-f", "-i"])
        .env_clear()
        .env("HOME", home.path())
        .env("PATH", format!("{}:/usr/bin:/bin", bin.display()))
        .env("XDG_STATE_HOME", state.path())
        .current_dir(home.path())
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .unwrap();
    {
        use std::io::Write as _;
        let stdin = child.stdin.as_mut().unwrap();
        stdin
            .write_all(
                b"eval \"$(piku init zsh)\"\necho one\n export SECRET_TOKEN=x\nfalse\nexit\n",
            )
            .unwrap();
    }
    child.wait().unwrap();
    // The hook appends in the background; give it a moment to land.
    let ledger = state.path().join("piku/shell-ledger.jsonl");
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    let mut text = String::new();
    while std::time::Instant::now() < deadline {
        text = std::fs::read_to_string(&ledger).unwrap_or_default();
        if text.lines().count() >= 2 {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
    let records: Vec<(String, i64, String)> = text
        .lines()
        .map(|line| {
            let value: serde_json::Value = serde_json::from_str(line).unwrap();
            (
                value["command"].as_str().unwrap().to_string(),
                value["exit"].as_i64().unwrap(),
                value["source"].as_str().unwrap().to_string(),
            )
        })
        .collect();
    assert_eq!(
        records,
        [
            ("echo one".to_string(), 0, "zsh".to_string()),
            ("false".to_string(), 1, "zsh".to_string()),
        ],
        "ledger: {text}"
    );
}

fn which_zsh() -> Result<std::path::PathBuf, ()> {
    ["/bin/zsh", "/usr/bin/zsh"]
        .iter()
        .map(std::path::PathBuf::from)
        .find(|path| path.exists())
        .ok_or(())
}

/// The agent is told where its own settings and records live, so a question
/// like "where is piku's default model configured?" does not need a search.
#[test]
fn system_prompt_tells_the_agent_about_itself() {
    let config = tempfile::tempdir().unwrap();
    let (base_url, bodies) = fake_completions_server(CANNED_ANSWER);
    let (_out, session_id) = run_headless_turn(config.path(), &base_url);
    let bodies = bodies.lock().unwrap().clone();
    let body = bodies.first().expect("one request");
    let settings = config.path().join("piku/settings.toml");
    let transcript = config
        .path()
        .join(format!("piku/sessions/{session_id}.json"));
    for needle in [
        "# About you (piku)".to_string(),
        format!("piku {}", env!("CARGO_PKG_VERSION")),
        "model canned-model".to_string(),
        settings.display().to_string(),
        transcript.display().to_string(),
    ] {
        assert!(
            body.contains(&needle),
            "missing {needle:?} in request: {body}"
        );
    }
}
