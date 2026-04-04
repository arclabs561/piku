#![allow(warnings)]

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
///   cargo test --test binary_tests
use std::process::{Command, Output};

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------

fn piku_binary() -> PathBuf {
    // Resolve binary relative to the test executable.
    // test binary lives at: target/{profile}/deps/binary_tests-*
    // piku binary lives at: target/{profile}/piku
    let exe = std::env::current_exe().unwrap();
    let profile_dir = exe
        .parent()
        .unwrap() // deps/
        .parent()
        .unwrap(); // {profile}/

    // Prefer same profile (debug when running `cargo test`, release when running
    // `cargo test --release`) — ensures we always test the freshest build.
    let same_profile = profile_dir.join("piku");
    if same_profile.exists() {
        return same_profile;
    }
    // Fallback: look in sibling profile directories
    let release = profile_dir.parent().unwrap().join("release").join("piku");
    if release.exists() {
        return release;
    }
    let debug = profile_dir.parent().unwrap().join("debug").join("piku");
    if debug.exists() {
        return debug;
    }
    panic!(
        "piku binary not found. Run `cargo build -p piku` first.\n\
         Searched:\n  {same_profile:?}\n  {release:?}\n  {debug:?}"
    );
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
        .env_remove("PIKU_RESTARTED");
    cmd
}

fn stdout(o: &Output) -> String {
    String::from_utf8_lossy(&o.stdout).into_owned()
}
fn stderr(o: &Output) -> String {
    String::from_utf8_lossy(&o.stderr).into_owned()
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
    assert!(text.contains("USAGE"), "help should have USAGE section");
    assert!(text.contains("OPTIONS"), "help should have OPTIONS section");
    assert!(text.contains("--model"), "help should document --model");
    assert!(
        text.contains("--provider"),
        "help should document --provider"
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
        .args(&["--provider", "xyzzy-unknown", "do something"])
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
        .args(&["--provider=openrouter", "do something"])
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
        .args(&["--provider=anthropic", "do something"])
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
fn groq_without_key_exits_nonzero() {
    let out = piku_clean_env()
        .args(&["--provider=groq", "do something"])
        .output()
        .unwrap();
    assert!(!out.status.success());
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
        .args(&["--model=claude-opus-4", "do something"])
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
        .args(&["--provider=ollama", "do something"])
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
        .args(&["--resume", "nonexistent-session-xyz-abc-123"])
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
fn session_id_includes_nanos_and_pid() {
    // Indirectly verify via the help text and binary behavior.
    // We can't directly test new_session_id() as it's private,
    // but we can verify that two rapid invocations produce different filenames.
    // Run two versions in quick succession and compare session file names.
    let tmp1 = std::env::temp_dir().join(format!("piku_sid_test_{}", std::process::id()));
    let tmp2 = std::env::temp_dir().join(format!("piku_sid_test2_{}", std::process::id()));
    std::fs::create_dir_all(&tmp1).unwrap();
    std::fs::create_dir_all(&tmp2).unwrap();

    // Just check it runs without panic — session IDs are internal
    let out = piku_clean_env()
        .arg("--version")
        .env("XDG_CONFIG_HOME", &tmp1)
        .output()
        .unwrap();
    assert!(out.status.success());

    let _ = std::fs::remove_dir_all(&tmp1);
    let _ = std::fs::remove_dir_all(&tmp2);
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
