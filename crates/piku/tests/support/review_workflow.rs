//! Shared deterministic CLI and retained-work review fixtures.

use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;
use std::process::{Command, Output};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde_json::{json, Value};

pub(crate) fn checked(command: &mut Command) -> Output {
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
    assert!(
        output.status.success(),
        "{command:?}\n{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    output
}

pub(crate) fn cli(config: &Path, workspace: &Path, url: &str) -> Command {
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

pub(crate) fn git(workspace: &Path, args: &[&str]) -> String {
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

pub(crate) fn scripted_server(
    file: &Path,
    resume_only: bool,
) -> (String, Arc<Mutex<Vec<Value>>>, std::thread::JoinHandle<()>) {
    let tool = |id: &str, debug: bool| json!({"choices":[{"delta":{"tool_calls":[{"index":0,"id":id,"type":"function","function":{"name":"write_file","arguments":json!({"path":file,"content":format!("{{\"port\":9090,\"debug\":{debug},\"service\":\"fixture\"}}\n")}).to_string()}}]},"finish_reason":null}]});
    let final_answer =
        json!({"choices":[{"delta":{"content":"Configuration complete."},"finish_reason":null}]});
    let responses = [
        (tool("partial-write", true), "tool_calls"),
        (final_answer.clone(), "stop"),
        (tool("corrected-write", false), "tool_calls"),
        (final_answer, "stop"),
    ];
    scripted_response_server(
        responses
            .into_iter()
            .skip(if resume_only { 2 } else { 0 })
            .collect(),
    )
}

/// A partial SSE response with stop="interrupt" stays open until cancellation.
pub(crate) fn scripted_response_server(
    responses: Vec<(Value, &'static str)>,
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
            if stop == "interrupt" {
                write!(stream, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\ndata: {delta}\n\n").unwrap();
                // The application must close its cancelled request. A timeout
                // advances this bounded fixture, but cannot satisfy the test's
                // independent durable cancellation predicate.
                let _ = stream.read(&mut chunk);
                continue;
            }
            let finish = json!({"choices":[{"delta":{},"finish_reason":stop}]});
            write!(stream, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\ndata: {delta}\n\ndata: {finish}\n\ndata: [DONE]\n\n").unwrap();
        }
    });
    (url, requests, handle)
}

// This oracle reads the actual workspace independently of model text and
// tool status. Wrong types, extra keys, unrelated edits, or a dirty parent fail.
pub(crate) fn workspace_accepted(workspace: &Path) -> bool {
    let actual: Value =
        serde_json::from_slice(&std::fs::read(workspace.join("settings.json")).unwrap()).unwrap();
    actual == json!({"port":9090,"debug":false,"service":"fixture"})
        && git(workspace, &["status", "--porcelain"]) == " M settings.json\n"
}

pub(crate) fn assert_review_history(
    record_path: &Path,
    rejected: &str,
    session: &str,
    note: &str,
    acceptance: &str,
) {
    let final_record = std::fs::read_to_string(record_path).unwrap();
    assert!(
        final_record.starts_with(rejected),
        "review history rewritten"
    );
    let reopened = piku_runtime::read_run_record(record_path).unwrap();
    let audit = piku_runtime::audit_run_record(&reopened);
    assert!(audit.is_structurally_complete(), "{audit:?}");
    assert_eq!(audit.turn_count, 2);
    assert_eq!(audit.completed_turn_count, 2);
    let events: Vec<Value> = final_record
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    for (sequence, event) in events.iter().enumerate() {
        assert_eq!(event["sequence"], sequence);
        assert_eq!(event["session_id"], session);
    }
    let dispositions: Vec<_> = events
        .iter()
        .filter(|event| event["event"] == "user_disposition")
        .collect();
    assert_eq!(dispositions.len(), 2);
    assert_eq!(dispositions[0]["disposition"], "needs_work");
    assert_eq!(dispositions[0]["note"]["text"], note);
    assert_eq!(dispositions[1]["disposition"], "accepted");
    assert_eq!(dispositions[1]["note"]["text"], acceptance);
    assert_eq!(
        events
            .iter()
            .filter(|event| event["event"] == "turn_completed")
            .count(),
        2
    );
    assert_eq!(
        events
            .iter()
            .filter(|event| event["event"] == "tool_completed" && event["is_error"] == false)
            .count(),
        2
    );
}

pub(crate) fn prepare_worktree(repo: &Path, retained: &Path) -> &'static str {
    git(repo, &["init", "-q"]);
    let original = "{\"port\":8080,\"debug\":true,\"service\":\"fixture\"}\n";
    std::fs::write(repo.join("settings.json"), original).unwrap();
    git(repo, &["add", "settings.json"]);
    git(
        repo,
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
    git(
        repo,
        &[
            "worktree",
            "add",
            "-qb",
            "recovery-fixture",
            retained.to_str().unwrap(),
        ],
    );
    original
}
