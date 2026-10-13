//! Synthetic operator journey through the real CLI and persisted run evidence.
//! Protocol/command scaffolding derives from `run_recovery_eval::terminal_workflow`.

use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;
use std::process::{Command, Output};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde_json::{json, Value};

const AMBIGUOUS: &str = "Prepare settings.json for the demo. Ask before choosing port or debug behavior; preserve unrelated files.";
const CORRECTION: &str = "By demo I mean port=9090 and debug=false, not a public debug server. Preserve service=fixture and all other files. Read before editing.";
const RESUME: &str = "The previous response was interrupted. Inspect the retained settings.json, run git diff --check as verification, and report remaining uncertainty; do not rewrite files.";
const ORIGINAL: &str = "{\"port\":8080,\"debug\":true,\"service\":\"fixture\"}\n";
const EXPECTED: &str = "{\"port\":9090,\"debug\":false,\"service\":\"fixture\"}\n";
const NEIGHBOR: &str = "operator-owned note: preserve exactly\n";
const REQUEST_BYTES: usize = 65_536;

fn bounded_output(command: &mut Command) -> Output {
    let mut stdout = tempfile::tempfile().unwrap();
    let mut stderr = tempfile::tempfile().unwrap();
    let mut child = command
        .stdout(stdout.try_clone().unwrap())
        .stderr(stderr.try_clone().unwrap())
        .spawn()
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(45);
    let status = loop {
        if let Some(status) = child.try_wait().unwrap() {
            break status;
        }
        if Instant::now() >= deadline {
            child.kill().unwrap();
            child.wait().unwrap();
            panic!("command exceeded 45 seconds: {command:?}");
        }
        std::thread::sleep(Duration::from_millis(10));
    };
    stdout.seek(SeekFrom::Start(0)).unwrap();
    stderr.seek(SeekFrom::Start(0)).unwrap();
    let mut output = Output {
        status,
        stdout: Vec::new(),
        stderr: Vec::new(),
    };
    stdout.read_to_end(&mut output.stdout).unwrap();
    stderr.read_to_end(&mut output.stderr).unwrap();
    output
}

fn checked(command: &mut Command) -> Output {
    let output = bounded_output(command);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    output
}

fn cli(config: &Path, workspace: &Path, url: &str) -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_piku"));
    command
        .env_clear()
        .env("PATH", std::env::var_os("PATH").unwrap())
        .env("HOME", config)
        .env("XDG_CONFIG_HOME", config)
        .env("PIKU_NO_DOTENV", "1")
        .env("PIKU_BASE_URL", url)
        .env("PIKU_API_KEY", "synthetic-test-key")
        .env("PIKU_MODEL", "canned-model")
        .current_dir(workspace);
    command
}

fn git(workspace: &Path, args: &[&str]) -> String {
    String::from_utf8(
        checked(
            Command::new("git")
                .current_dir(workspace)
                .env("GIT_CONFIG_NOSYSTEM", "1")
                .env("GIT_CONFIG_GLOBAL", "/dev/null")
                .args(args),
        )
        .stdout,
    )
    .unwrap()
}

fn scripted_server(
    responses: Vec<(Value, Option<&'static str>)>,
) -> (String, Arc<Mutex<Vec<Value>>>, std::thread::JoinHandle<()>) {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    let url = format!("http://{}/v1", listener.local_addr().unwrap());
    let requests = Arc::new(Mutex::new(Vec::new()));
    let captured = Arc::clone(&requests);
    let handle = std::thread::spawn(move || {
        for (delta, stop) in responses {
            let deadline = Instant::now() + Duration::from_secs(30);
            let mut stream = loop {
                match listener.accept() {
                    Ok((stream, _)) => break stream,
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        assert!(Instant::now() < deadline, "scripted request never arrived");
                        std::thread::sleep(Duration::from_millis(10));
                    }
                    Err(error) => panic!("accept: {error}"),
                }
            };
            stream.set_nonblocking(false).unwrap();
            stream
                .set_read_timeout(Some(Duration::from_secs(10)))
                .unwrap();
            stream
                .set_write_timeout(Some(Duration::from_secs(10)))
                .unwrap();
            let mut request = Vec::new();
            let mut chunk = [0u8; 4096];
            let (header_end, length) = loop {
                let n = stream.read(&mut chunk).unwrap();
                assert_ne!(n, 0, "incomplete HTTP headers");
                request.extend_from_slice(&chunk[..n]);
                if let Some(end) = request.windows(4).position(|w| w == b"\r\n\r\n") {
                    let headers = String::from_utf8_lossy(&request[..end]).to_ascii_lowercase();
                    let length = headers
                        .lines()
                        .find_map(|line| line.strip_prefix("content-length:"))
                        .unwrap()
                        .trim()
                        .parse::<usize>()
                        .unwrap();
                    assert!(length <= REQUEST_BYTES, "frozen full-request byte ceiling");
                    break (end + 4, length);
                }
            };
            while request.len() < header_end + length {
                let n = stream.read(&mut chunk).unwrap();
                assert_ne!(n, 0, "incomplete HTTP body");
                request.extend_from_slice(&chunk[..n]);
            }
            captured
                .lock()
                .unwrap()
                .push(serde_json::from_slice(&request[header_end..header_end + length]).unwrap());
            write!(stream, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\ndata: {delta}\n\n").unwrap();
            if let Some(stop) = stop {
                let finish = json!({"choices":[{"delta":{},"finish_reason":stop}]});
                write!(stream, "data: {finish}\n\ndata: [DONE]\n\n").unwrap();
            }
            // None deliberately closes before a terminal frame. This is a
            // stream interruption with orderly CLI persistence, not a crash.
        }
    });
    (url, requests, handle)
}

fn tool(id: &str, name: &str, arguments: &Value) -> Value {
    json!({"choices":[{"delta":{"tool_calls":[{"index":0,"id":id,"type":"function","function":{"name":name,"arguments":arguments.to_string()}}]},"finish_reason":null}]})
}

fn text(content: &str) -> Value {
    json!({"choices":[{"delta":{"content":content},"finish_reason":null}]})
}

fn workspace_accepted(workspace: &Path) -> bool {
    // Host observations are independent of the scripted provider's answer,
    // tool success, and the agent's whitespace-only verification command.
    std::fs::read(workspace.join("settings.json")).unwrap() == EXPECTED.as_bytes()
        && serde_json::from_slice::<Value>(&std::fs::read(workspace.join("settings.json")).unwrap())
            .unwrap()
            == json!({"port":9090,"debug":false,"service":"fixture"})
        && std::fs::read(workspace.join("notes.txt")).unwrap() == NEIGHBOR.as_bytes()
        && git(workspace, &["status", "--porcelain"]) == " M settings.json\n"
}

fn stage(
    config: &Path,
    ledger: &Path,
    output: &Output,
    started: Instant,
    assertions: impl FnOnce(),
) {
    let record = super::test_helpers::build_live_ledger_record(
        super::test_helpers::EvaluationSurface::Cli,
        config,
        "operator-journey",
        "fixture",
        "canned-model",
        output.status.success(),
        started.elapsed(),
    );
    super::test_helpers::complete_scenario_with_ledger(Some(ledger), record, assertions);
}

fn events(path: &Path) -> Vec<Value> {
    std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect()
}

fn exact_user_messages(request: &Value) -> Vec<&str> {
    request["messages"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|message| message["role"] == "user")
        .filter_map(|message| message["content"].as_str())
        .collect()
}

fn scripted_responses() -> Vec<(Value, Option<&'static str>)> {
    let read = |id| tool(id, "read_file", &json!({"path":"settings.json"}));
    vec![
        (read("ambiguous-read"), Some("tool_calls")),
        (
            text("Which port and debug mode should the demo use?"),
            Some("stop"),
        ),
        (read("correction-read"), Some("tool_calls")),
        (
            tool(
                "correction-edit",
                "edit_file",
                &json!({"path":"settings.json","old_string":ORIGINAL.trim_end(),"new_string":EXPECTED.trim_end()}),
            ),
            Some("tool_calls"),
        ),
        (
            text("The edit was applied; verification is still pending."),
            None,
        ),
        (read("resume-read"), Some("tool_calls")),
        (
            tool(
                "resume-check",
                "bash",
                &json!({"command":"git diff --check","purpose":"verification","timeout_ms":1000}),
            ),
            Some("tool_calls"),
        ),
        (
            text("Whitespace verification passed. This is not a deployment or runtime check."),
            Some("stop"),
        ),
    ]
}

fn assert_reopened(run_path: &Path, session_id: &str, captured: &[Value]) {
    assert_eq!(captured.len(), 8, "frozen request budget");
    assert_eq!(exact_user_messages(&captured[0]), [AMBIGUOUS]);
    assert_eq!(exact_user_messages(&captured[2]), [AMBIGUOUS, CORRECTION]);
    assert_eq!(
        exact_user_messages(&captured[5]),
        [AMBIGUOUS, CORRECTION, RESUME]
    );
    let prior_results: Vec<_> = captured[5]["messages"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|message| message["role"] == "tool")
        .filter_map(|message| message["tool_call_id"].as_str())
        .collect();
    assert_eq!(
        prior_results,
        ["ambiguous-read", "correction-read", "correction-edit"]
    );
    let reopened = piku_runtime::read_run_record(run_path).unwrap();
    let audit = piku_runtime::audit_run_record(&reopened);
    assert!(audit.is_structurally_complete(), "{audit:?}");
    let retained = events(run_path);
    for (sequence, event) in retained.iter().enumerate() {
        assert_eq!(event["sequence"], sequence);
        assert_eq!(event["session_id"], session_id);
    }
    let contexts: Vec<_> = retained
        .iter()
        .filter(|event| event["event"] == "context_built")
        .collect();
    assert_eq!(
        contexts.len(),
        8,
        "each received iteration needs inspectable context"
    );
    assert!(contexts
        .iter()
        .all(|event| event["manifest"]["estimated_input_tokens"]
            .as_u64()
            .is_some_and(|tokens| tokens > 0)));
    let terminal: Vec<_> = retained
        .iter()
        .filter_map(|event| match event["event"].as_str() {
            Some(kind @ ("turn_completed" | "turn_failed")) => Some(kind),
            _ => None,
        })
        .collect();
    assert_eq!(
        terminal,
        ["turn_completed", "turn_failed", "turn_completed"]
    );
    let starts: Vec<_> = retained
        .iter()
        .filter(|event| event["event"] == "tool_started")
        .collect();
    assert_eq!(
        starts
            .iter()
            .map(|event| event["name"].as_str().unwrap())
            .collect::<Vec<_>>(),
        ["read_file", "read_file", "edit_file", "read_file", "bash"]
    );
    for event in starts {
        let ends: Vec<_> = retained
            .iter()
            .filter(|end| {
                end["event"] == "tool_completed" && end["tool_call_id"] == event["tool_call_id"]
            })
            .collect();
        assert_eq!(ends.len(), 1);
        assert_eq!(ends[0]["is_error"], false);
    }
    let checks: Vec<_> = retained
        .iter()
        .filter(|event| {
            event["event"] == "tool_completed" && event["tool_call_id"] == "resume-check"
        })
        .collect();
    assert_eq!(checks[0]["verification"]["status"], "passed");
}

fn export_evidence(
    config: &Path,
    run_path: &Path,
    session: &Path,
    ledger: &Path,
    workspace: &Path,
    requests: &Arc<Mutex<Vec<Value>>>,
) {
    if let Some(destination) = std::env::var_os("PIKU_OPERATOR_EVIDENCE_DIR") {
        let destination = std::path::PathBuf::from(destination);
        std::fs::create_dir_all(&destination).unwrap();
        for entry in walkdir::WalkDir::new(config) {
            let entry = entry.unwrap();
            let target = destination
                .join("config")
                .join(entry.path().strip_prefix(config).unwrap());
            if entry.file_type().is_dir() {
                std::fs::create_dir_all(target).unwrap();
            } else {
                std::fs::copy(entry.path(), target).unwrap();
            }
        }
        std::fs::write(
            destination.join("requests.json"),
            serde_json::to_vec_pretty(&*requests.lock().unwrap()).unwrap(),
        )
        .unwrap();
        std::fs::copy(run_path, destination.join("run.jsonl")).unwrap();
        std::fs::copy(session, destination.join("session.json")).unwrap();
        std::fs::copy(ledger, destination.join("ledger.jsonl")).unwrap();
        std::fs::copy(
            workspace.join("settings.json"),
            destination.join("settings.json"),
        )
        .unwrap();
        std::fs::copy(workspace.join("notes.txt"), destination.join("notes.txt")).unwrap();
        let request_bytes: Vec<_> = requests
            .lock()
            .unwrap()
            .iter()
            .map(|request| serde_json::to_vec(request).unwrap().len())
            .collect();
        std::fs::write(destination.join("manifest.json"), serde_json::to_vec_pretty(&json!({
            "contract":"ambiguity-correction-stream-interruption-resume-v1",
            "provider":"fixture", "model":"canned-model", "live_calls":false,
            "source_config_root":config,"relocated_config_root":destination.join("config"),
            "request_count":8, "request_bytes":request_bytes, "request_byte_ceiling":REQUEST_BYTES,
            "process_deadline_seconds":45, "accept_deadline_seconds":30,
            "expected_terminal_events":["turn_completed","turn_failed","turn_completed"],
            "expected_workspace":{"settings.json":EXPECTED,"notes.txt":NEIGHBOR},
            "limits":["scripted plumbing only","orderly failed-stream persistence","no abrupt crash or PTY cancellation","usage tokens unreported by fixture"]
        })).unwrap()).unwrap();
    }
}

fn prepare_fixture(workspace: &Path, config: &Path) {
    std::fs::create_dir_all(workspace).unwrap();
    std::fs::create_dir_all(config).unwrap();
    std::fs::write(workspace.join("settings.json"), ORIGINAL).unwrap();
    std::fs::write(workspace.join("notes.txt"), NEIGHBOR).unwrap();
    git(workspace, &["init", "-q"]);
    git(workspace, &["add", "settings.json", "notes.txt"]);
    git(
        workspace,
        &[
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "-qm",
            "fixture",
        ],
    );
}

#[test]
fn dogfood_ambiguity_correction_interruption_resume() {
    let fixture = tempfile::tempdir().unwrap();
    let workspace = fixture.path().join("workspace");
    let config = fixture.path().join("config");
    prepare_fixture(&workspace, &config);

    let responses = scripted_responses();
    let (url, requests, server) = scripted_server(responses);
    let ledger = fixture.path().join("ledger.jsonl");
    let started = Instant::now();
    let first = checked(cli(&config, &workspace, &url).args(["-p", AMBIGUOUS]));
    let session = std::fs::read_dir(config.join("piku/sessions"))
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .find(|path| path.extension().is_some_and(|ext| ext == "json"))
        .unwrap();
    let session_id = session.file_stem().unwrap().to_str().unwrap();
    let run_path = config.join("piku/runs").join(format!("{session_id}.jsonl"));
    stage(&config, &ledger, &first, started, || {
        assert_eq!(
            std::fs::read(workspace.join("settings.json")).unwrap(),
            ORIGINAL.as_bytes()
        );
        assert_eq!(
            std::fs::read(workspace.join("notes.txt")).unwrap(),
            NEIGHBOR.as_bytes()
        );
        assert!(git(&workspace, &["status", "--porcelain"]).is_empty());
    });
    let original_record = std::fs::read_to_string(&run_path).unwrap();

    let started = Instant::now();
    let interrupted = bounded_output(
        cli(&config, &workspace, &url).args(["--resume", session_id, "-p", CORRECTION]),
    );
    stage(&config, &ledger, &interrupted, started, || {
        assert!(
            !interrupted.status.success(),
            "interruption must not claim completed execution"
        );
        assert!(
            workspace_accepted(&workspace),
            "partial effects must survive a failed stream"
        );
        let retained = events(&run_path);
        assert_eq!(
            retained
                .iter()
                .filter(|event| event["event"] == "turn_failed")
                .count(),
            1
        );
        assert_eq!(
            retained
                .iter()
                .filter(|event| event["event"] == "turn_completed")
                .count(),
            1
        );
        assert!(std::fs::read_to_string(&run_path)
            .unwrap()
            .starts_with(&original_record));
        let reloaded = piku_runtime::Session::load(&session).unwrap();
        assert_eq!(reloaded.id, session_id);
        assert!(reloaded.messages.iter().any(|message| message.blocks.iter().any(|block| matches!(block, piku_runtime::ContentBlock::ToolUse { id, .. } if id == "correction-edit"))));
    });
    let interrupted_record = std::fs::read_to_string(&run_path).unwrap();

    let started = Instant::now();
    let resumed =
        checked(cli(&config, &workspace, &url).args(["--resume", session_id, "-p", RESUME]));
    server.join().unwrap();
    stage(&config, &ledger, &resumed, started, || {
        assert!(workspace_accepted(&workspace));
        assert!(std::fs::read_to_string(&run_path)
            .unwrap()
            .starts_with(&interrupted_record));
        assert_reopened(&run_path, session_id, &requests.lock().unwrap());
    });
    let stages = events(&ledger);
    assert_eq!(
        stages
            .iter()
            .map(|event| event["run_status"].as_str().unwrap())
            .collect::<Vec<_>>(),
        ["completed", "inconclusive", "completed"]
    );
    assert_eq!(
        stages[1]["result"], "failure",
        "resume cannot rewrite failed execution"
    );
    for record in &stages {
        super::test_helpers::validate_evaluation_envelope(record).unwrap();
    }

    export_evidence(&config, &run_path, &session, &ledger, &workspace, &requests);
}
