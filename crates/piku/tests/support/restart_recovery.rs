//! Operational checkpoint recovery through a real executable replacement.

use super::{piku_binary, review_workflow, smoke_tempdir, wait_ready, Pty};
use std::process::Command;
use std::time::{Duration, SystemTime};

use piku_runtime::{ConversationMessage, RunEvent, RunRecorder, Session};
use serde_json::{json, Value};
use serial_test::serial;
use sha2::{Digest, Sha256};

#[test]
#[serial]
#[ignore = "PTY acceptance: run isolated via scripts/ci.sh pty"]
#[allow(clippy::too_many_lines)]
fn real_successor_recovers_same_session_without_replaying_resume_prompt() {
    use std::os::unix::fs::MetadataExt;

    // Reuse the retained-evidence convention of the neighboring terminal test.
    let fixture = if let Ok(destination) = std::env::var("PIKU_TERMINAL_ACCEPTANCE_EVIDENCE") {
        std::fs::create_dir_all(&destination).unwrap();
        tempfile::tempdir_in(destination).unwrap()
    } else {
        smoke_tempdir()
    };
    let root = fixture.path().to_path_buf();
    let _guard = if std::env::var_os("PIKU_TERMINAL_ACCEPTANCE_EVIDENCE").is_some() {
        eprintln!("retained restart fixture: {}", fixture.keep().display());
        None
    } else {
        Some(fixture)
    };
    let config = root.join("config");
    let workspace = root.join("workspace");
    let target = root.join("approved-target");
    let candidate = target.join("release/piku");
    let executable = root.join("executing-piku");
    std::fs::create_dir_all(&workspace).unwrap();
    std::fs::create_dir_all(candidate.parent().unwrap()).unwrap();
    let binary = piku_binary();
    let binary_bytes = std::fs::read(&binary).unwrap();
    let binary_hash = format!("{:x}", Sha256::digest(&binary_bytes));
    std::fs::copy(&binary, &executable).unwrap();
    std::fs::copy(&binary, &candidate).unwrap();
    let earlier = SystemTime::UNIX_EPOCH + Duration::from_secs(1_700_000_000);
    let newer = earlier + Duration::from_secs(60);
    std::fs::File::open(&executable)
        .unwrap()
        .set_modified(earlier)
        .unwrap();
    std::fs::File::open(&candidate)
        .unwrap()
        .set_modified(newer)
        .unwrap();
    let before_inode = std::fs::metadata(&executable).unwrap().ino();

    let id = "session-restart-operational-full-id";
    let session_path = config.join("piku/sessions").join(format!("{id}.json"));
    let record_path = config.join("piku/runs").join(format!("{id}.jsonl"));
    let mut session = Session::new(id.into());
    session.record_provider("custom", "canned-model");
    session.push(ConversationMessage::user("Retained checkpoint history."));
    session.save(&session_path).unwrap();
    RunRecorder::open(&record_path, id)
        .unwrap()
        .append_run(RunEvent::UserDisposition {
            disposition: piku_runtime::run_record::RunDisposition::NeedsWork,
            note: piku_runtime::run_record::ContentRef::Inline {
                text: "Retain this original record prefix.".into(),
            },
        })
        .unwrap();
    let prefix = std::fs::read(&record_path).unwrap();
    std::fs::write(root.join("record-prefix.jsonl"), &prefix).unwrap();

    let text =
        |content: &str| json!({"choices":[{"delta":{"content":content},"finish_reason":null}]});
    let (url, requests, server) = review_workflow::scripted_response_server(vec![
        (text("ORIGINAL_RESUME_COMPLETED"), "stop"),
        // The successor's normal writable exit extracts memory separately.
        (text("[]"), "stop"),
    ]);
    let prompt = "Complete this original resume prompt exactly once.";
    let config_text = config.to_str().unwrap();
    let ceiling = root.to_str().unwrap();
    let nonexistent_git = root.join("no-git-directory");
    let environment = [
        ("HOME", config_text),
        ("XDG_CONFIG_HOME", config_text),
        ("PIKU_NO_DOTENV", "1"),
        ("PIKU_API_KEY", "synthetic-test-key"),
        ("PIKU_MODEL", "canned-model"),
        ("PIKU_BASE_URL", url.as_str()),
        ("GIT_CEILING_DIRECTORIES", ceiling),
        ("GIT_DIR", nonexistent_git.to_str().unwrap()),
    ];
    let args = [
        "--provider",
        "custom",
        "--model",
        "canned-model",
        "--resume",
        id,
        "--self-update-root",
        target.to_str().unwrap(),
        prompt,
    ];
    let mut command = Command::new(&executable);
    command.args(args);
    let mut pty = Pty::spawn_command_with_env(command, &workspace, &environment);
    assert!(
        pty.wait_for("restarted with new binary", Duration::from_secs(12)),
        "successor banner missing:\n{}",
        pty.captured()
    );
    let ready = wait_ready(&mut pty);
    assert!(ready.shows("restarted with new binary"));
    // The helper's initial marker alone cannot satisfy this execution witness:
    // startup admission, atomic replacement, and a post-restart TUI all agree.
    assert!(pty.captured().contains("approved newer binary detected"));
    assert!(pty.captured().contains(id));
    assert_ne!(std::fs::metadata(&executable).unwrap().ino(), before_inode);
    pty.exit_cleanly();
    assert!(
        pty.eof,
        "successor did not exit normally:\n{}",
        pty.captured()
    );
    assert!(matches!(
        pty._proc.status(),
        Some(nix::sys::wait::WaitStatus::Exited(_, 0))
    ));
    std::fs::write(root.join("terminal.txt"), pty.captured()).unwrap();
    drop(pty);
    server.join().unwrap();

    let requests = requests.lock().unwrap();
    std::fs::write(
        root.join("requests.json"),
        serde_json::to_vec_pretty(&*requests).unwrap(),
    )
    .unwrap();
    assert_eq!(requests.len(), 2, "one launch turn plus exit extraction");
    assert_eq!(
        requests[0]["messages"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|message| message["role"] == "user")
            // Compat combines adjacent user messages into one content array.
            .flat_map(|message| message["content"].as_array().unwrap())
            .filter(|block| block["text"] == prompt)
            .count(),
        1
    );
    assert!(
        requests[1].get("tools").is_none(),
        "second call must be extraction"
    );
    assert_eq!(
        requests[1]["messages"]
            .as_array()
            .unwrap()
            .iter()
            .find(|message| message["role"] == "user")
            .unwrap()["content"],
        format!("User: Retained checkpoint history.\nUser: {prompt}\nAssistant: ORIGINAL_RESUME_COMPLETED\n")
    );
    let saved = Session::load(&session_path).unwrap();
    assert_eq!(saved.id, id);
    assert_eq!(saved.messages.len(), 3);
    let saved_json = serde_json::to_value(&saved).unwrap();
    assert_eq!(
        saved_json["messages"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|message| message["role"] == "user" && message["blocks"][0]["text"] == prompt)
            .count(),
        1
    );
    let final_record = std::fs::read(&record_path).unwrap();
    assert!(final_record.starts_with(&prefix));
    assert!(final_record.len() > prefix.len());
    let record = piku_runtime::read_run_record(&record_path).unwrap();
    let audit = piku_runtime::audit_run_record(&record);
    assert!(audit.is_structurally_complete(), "{audit:?}");
    assert_eq!(audit.turn_count, 1);
    assert_eq!(audit.completed_turn_count, 1);
    let events: Vec<Value> = String::from_utf8(final_record)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    for (sequence, event) in events.iter().enumerate() {
        assert_eq!(event["sequence"], sequence);
        assert_eq!(event["session_id"], id);
    }
    let started: Vec<_> = events
        .iter()
        .filter(|event| event["event"] == "turn_started")
        .collect();
    assert_eq!(started.len(), 1);
    assert_eq!(started[0]["input"]["text"], prompt);
    assert_eq!(std::fs::read(&binary).unwrap(), binary_bytes);
    assert_eq!(std::fs::read(&candidate).unwrap(), binary_bytes);
    assert_eq!(std::fs::read(&executable).unwrap(), binary_bytes);
    std::fs::write(
        root.join("readback.json"),
        serde_json::to_vec_pretty(&json!({
            "binary":binary,"binary_sha256":binary_hash,"argv":args,
            "session_id":id,"requests":*requests,"record_events":events.len(),
            "turns":audit.turn_count,"completed_turns":audit.completed_turn_count,
            "record_prefix_preserved":true,"successor_exit":0
        }))
        .unwrap(),
    )
    .unwrap();
}
