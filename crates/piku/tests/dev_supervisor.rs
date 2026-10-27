//! End-to-end `piku dev` protocol tests.
//!
//! The supervisor is a normal process, so these run without a TTY: the child
//! and the build command are shell scripts. They prove the restart protocol,
//! the receipt trail, tolerance of a failed build, and the crash-guard revert.

use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

const TIMEOUT: Duration = Duration::from_secs(10);

fn piku_binary() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_piku"))
}

fn script(dir: &Path, name: &str, body: &str) -> PathBuf {
    let path = dir.join(name);
    fs::write(&path, body).expect("write script");
    let mut permissions = fs::metadata(&path).expect("script metadata").permissions();
    permissions.set_mode(0o755);
    fs::set_permissions(&path, permissions).expect("script permissions");
    path
}

fn lines(path: &Path) -> usize {
    fs::read_to_string(path).map_or(0, |text| text.lines().count())
}

fn wait_for_lines(path: &Path, want: usize) -> bool {
    let start = Instant::now();
    while start.elapsed() < TIMEOUT {
        if lines(path) >= want {
            return true;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    false
}

fn wait_for_text(path: &Path, needle: &str) -> bool {
    let start = Instant::now();
    while start.elapsed() < TIMEOUT {
        if fs::read_to_string(path).is_ok_and(|text| text.contains(needle)) {
            return true;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    false
}

fn stop(child: &mut Child) {
    let _ = Command::new("kill")
        .arg("-TERM")
        .arg(child.id().to_string())
        .status();
    let _ = child.wait();
}

struct Fixture {
    dir: tempfile::TempDir,
    watch: PathBuf,
    state: PathBuf,
    target: PathBuf,
    starts: PathBuf,
    source: PathBuf,
}

impl Fixture {
    fn new() -> Self {
        let dir = tempfile::tempdir().expect("tempdir");
        let watch = dir.path().join("workspace");
        let state = dir.path().join("state");
        let target = dir.path().join("target");
        let source = watch.join("crates/piku/src/lib.rs");
        fs::create_dir_all(source.parent().expect("source parent")).expect("source dir");
        fs::create_dir_all(target.join("debug")).expect("target dir");
        fs::write(&source, "one").expect("write source");
        let starts = dir.path().join("starts");
        Self {
            dir,
            watch,
            state,
            target,
            starts,
            source,
        }
    }

    fn supervisor(&self, build: &Path) -> Child {
        Command::new(piku_binary())
            .arg("dev")
            .env("PIKU_DEV_WATCH_ROOT", &self.watch)
            .env("PIKU_DEV_STATE_DIR", &self.state)
            .env("PIKU_DEV_BUILD", build)
            .env("CARGO_TARGET_DIR", &self.target)
            .env("STARTS", &self.starts)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn supervisor")
    }

    fn touch_source(&self, text: &str) {
        fs::write(&self.source, text).expect("touch source");
    }

    fn receipts(&self) -> String {
        fs::read_to_string(self.state.join("reloads.jsonl")).unwrap_or_default()
    }
}

#[test]
fn reload_restarts_the_child_on_the_same_session() {
    let fixture = Fixture::new();
    // Like the real TUI: on reload, save the draft naming the session, exit 75.
    script(
        &fixture.target.join("debug"),
        "piku",
        "#!/bin/sh\necho \"start ${PIKU_SESSION_ID:-none} ${PIKU_RESTARTED:-0}\" >> \"$STARTS\"\ntrap 'printf %s \"{\\\"session_id\\\":\\\"sess-1\\\",\\\"text\\\":\\\"\\\"}\" > \"$PIKU_DEV_STATE_DIR/draft.json\"; exit 75' USR1\nwhile :; do sleep 0.05; done\n",
    );
    let build = script(fixture.dir.path(), "build.sh", "#!/bin/sh\nexit 0\n");

    let mut supervisor = fixture.supervisor(&build);
    assert!(wait_for_lines(&fixture.starts, 1), "first child start");
    fixture.touch_source("two");
    assert!(
        wait_for_lines(&fixture.starts, 2),
        "the supervisor must restart the child after a rebuild"
    );
    stop(&mut supervisor);

    let starts = fs::read_to_string(&fixture.starts).unwrap_or_default();
    let starts: Vec<&str> = starts.lines().collect();
    assert_eq!(
        starts[..2],
        ["start none 0", "start sess-1 1"],
        "the restarted child must resume the saved session"
    );
    let receipts = fixture.receipts();
    assert!(receipts.contains("\"kind\":\"reload\""), "{receipts}");
    assert!(!receipts.contains("build-failed"), "{receipts}");
}

#[test]
fn failed_build_keeps_the_running_child() {
    let fixture = Fixture::new();
    script(
        &fixture.target.join("debug"),
        "piku",
        "#!/bin/sh\necho start >> \"$STARTS\"\ntrap 'exit 75' USR1\nwhile :; do sleep 0.05; done\n",
    );
    let build = script(
        fixture.dir.path(),
        "build.sh",
        "#!/bin/sh\nif [ -f \"$BUILD_FAIL\" ]; then exit 1; fi\nexit 0\n",
    );
    let fail_marker = fixture.dir.path().join("fail");

    let mut supervisor = Command::new(piku_binary())
        .arg("dev")
        .env("PIKU_DEV_WATCH_ROOT", &fixture.watch)
        .env("PIKU_DEV_STATE_DIR", &fixture.state)
        .env("PIKU_DEV_BUILD", &build)
        .env("CARGO_TARGET_DIR", &fixture.target)
        .env("STARTS", &fixture.starts)
        .env("BUILD_FAIL", &fail_marker)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn supervisor");

    assert!(wait_for_lines(&fixture.starts, 1), "first child start");
    // Fail only the next build; the initial build above had to succeed.
    fs::write(&fail_marker, "").expect("fail marker");
    fixture.touch_source("two");
    assert!(
        wait_for_text(&fixture.state.join("status"), "build failed"),
        "status must report the failed build"
    );
    std::thread::sleep(Duration::from_millis(500));
    assert_eq!(
        lines(&fixture.starts),
        1,
        "a failed build must not restart the child"
    );

    fs::remove_file(&fail_marker).expect("clear fail marker");
    fixture.touch_source("three");
    assert!(
        wait_for_lines(&fixture.starts, 2),
        "a later successful build must restart the child"
    );
    stop(&mut supervisor);

    let receipts = fixture.receipts();
    assert!(receipts.contains("build-failed"), "{receipts}");
    assert!(receipts.contains("\"kind\":\"reload\""), "{receipts}");
}

#[test]
fn repeated_fast_crashes_revert_to_the_last_good_binary() {
    let fixture = Fixture::new();
    script(
        &fixture.target.join("debug"),
        "piku",
        "#!/bin/sh\necho \"$0\" >> \"$STARTS\"\nif [ \"$(wc -l < \"$STARTS\")\" -ge 3 ]; then exit 0; fi\nexit 1\n",
    );
    let build = script(fixture.dir.path(), "build.sh", "#!/bin/sh\nexit 0\n");

    let mut supervisor = fixture.supervisor(&build);
    assert!(wait_for_lines(&fixture.starts, 3), "three starts");
    let status = supervisor.wait().expect("supervisor exit");
    assert!(status.success(), "supervisor should exit cleanly");

    let starts = fs::read_to_string(&fixture.starts).expect("starts");
    let paths: Vec<&str> = starts.lines().collect();
    assert_eq!(paths.len(), 3);
    assert!(
        paths[2].ends_with("last-good-piku"),
        "third start must use the reverted binary, got {}",
        paths[2]
    );
    assert!(
        fixture.receipts().contains("\"kind\":\"crash\""),
        "a revert must be recorded"
    );
}
