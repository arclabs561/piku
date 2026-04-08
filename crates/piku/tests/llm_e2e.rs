/// Real end-to-end tests: spawns the compiled piku binary against a real LLM API.
///
/// GATING: All tests require `PIKU_LLM_E2E=1` to run. Without it they skip.
///
/// SETUP:
///   cargo build --release -p piku
///   export OPENROUTER_API_KEY=sk-or-...   # or `ANTHROPIC_API_KEY` / `GROQ_API_KEY`
///   `PIKU_LLM_E2E=1` cargo test --test `llm_e2e` -- --nocapture
///
/// DESIGN PRINCIPLES:
///   1. Assert on filesystem side-effects, not LLM prose (deterministic)
///   2. Use cheap/fast models (gpt-4o-mini, haiku, llama-8b) — <10s per test
///   3. One tool call per test where possible (no multi-turn complexity)
///   4. Prompt is unambiguous — the task either succeeded or it didn't
///   5. 90s wall-clock timeout in the assertion, but fast models finish <15s
use std::path::PathBuf;
use std::process::{Command, Stdio};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn is_llm_e2e_enabled() -> bool {
    std::env::var("PIKU_LLM_E2E")
        .map(|v| v == "1" || v == "true")
        .unwrap_or(false)
}

fn piku_binary() -> PathBuf {
    let exe = std::env::current_exe().unwrap();
    let profile_dir = exe.parent().unwrap().parent().unwrap();
    // Prefer same-profile binary (freshest build)
    let same = profile_dir.join("piku");
    if same.exists() {
        return same;
    }
    let release = profile_dir.parent().unwrap().join("release").join("piku");
    if release.exists() {
        return release;
    }
    let debug = profile_dir.parent().unwrap().join("debug").join("piku");
    if debug.exists() {
        return debug;
    }
    panic!("piku binary not found — run `cargo build -p piku` first");
}

/// Returns true if env var is set AND non-empty.
fn has_key(var: &str) -> bool {
    std::env::var(var).map(|v| !v.is_empty()).unwrap_or(false)
}

/// Choose the cheapest/fastest available provider and model for tool-use tests.
///
/// Priority: `OpenRouter` (gpt-4o-mini has reliable tool use), then Anthropic, then Groq.
/// Groq's llama models sometimes generate malformed tool calls — de-prioritized.
fn detect_provider() -> Option<(&'static str, &'static str, &'static str)> {
    // Returns (provider_name, env_var, model)
    if has_key("OPENROUTER_API_KEY") {
        return Some(("openrouter", "OPENROUTER_API_KEY", "openai/gpt-4o-mini"));
    }
    if has_key("ANTHROPIC_API_KEY") {
        return Some(("anthropic", "ANTHROPIC_API_KEY", "claude-haiku-4-5"));
    }
    if has_key("GROQ_API_KEY") {
        // moonshotai/kimi-k2-instruct has better tool use than llama-8b
        return Some(("groq", "GROQ_API_KEY", "moonshotai/kimi-k2-instruct"));
    }
    None
}

fn tempdir() -> PathBuf {
    let base = std::env::temp_dir().join(format!(
        "piku_llm_e2e_{}_{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.subsec_nanos())
            .unwrap_or(0),
        std::process::id(),
    ));
    std::fs::create_dir_all(&base).unwrap();
    base
}

/// Run piku with a prompt against a controlled directory.
/// Returns (stdout, stderr, `exit_success`).
fn run_piku(
    prompt: &str,
    provider: &str,
    model: &str,
    key_var: &str,
    working_dir: &std::path::Path,
    config_dir: &std::path::Path,
) -> (String, String, bool) {
    let api_key = std::env::var(key_var).unwrap();

    let output = Command::new(piku_binary())
        .arg("--provider")
        .arg(provider)
        .arg("--model")
        .arg(model)
        .arg(prompt)
        .env_clear()
        .env(key_var, &api_key)
        .env("PATH", std::env::var("PATH").unwrap_or_default())
        .env("HOME", std::env::var("HOME").unwrap_or_default())
        .env("XDG_CONFIG_HOME", config_dir)
        .current_dir(working_dir)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .expect("failed to spawn piku");

    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    eprintln!("=== piku stdout ===\n{stdout}");
    eprintln!("=== piku stderr ===\n{stderr}");

    (stdout, stderr, output.status.success())
}

// ---------------------------------------------------------------------------
// Test 1: piku adds a doc comment to a Rust function
// ---------------------------------------------------------------------------

/// The simplest possible deterministic task:
/// - File has a function with a unique sentinel comment above it
/// - Task: add a doc comment to the function
/// - Success: file contains `///` on the line before `pub fn add`
/// - The exact comment text is irrelevant — structure is what we assert
#[test]
fn piku_adds_doc_comment_to_rust_function() {
    if !is_llm_e2e_enabled() {
        eprintln!("Skipping: PIKU_LLM_E2E not set");
        return;
    }
    let Some((provider, key_var, model)) = detect_provider() else {
        eprintln!("Skipping: no API key found");
        return;
    };

    let dir = tempdir();
    let config = dir.join("config");
    let src_dir = dir.join("src");
    std::fs::create_dir_all(&src_dir).unwrap();

    let target = src_dir.join("lib.rs");
    let original = r"// PIKU_TEST_MARKER: add a doc comment to the add function below
pub fn add(a: i32, b: i32) -> i32 {
    a + b
}
";
    std::fs::write(&target, original).unwrap();

    let prompt = format!(
        "In the file {path}, the function `add` has a marker comment above it. \
         Add a Rust doc comment (using `///`) directly above the line `pub fn add`. \
         The comment should briefly explain what the function does. \
         Do not change anything else.",
        path = target.display()
    );

    let (stdout, stderr, success) = run_piku(&prompt, provider, model, key_var, &dir, &config);

    assert!(success, "piku should exit 0. stderr: {stderr}");

    let final_content = std::fs::read_to_string(&target).unwrap();

    // Primary: file was modified
    assert_ne!(
        final_content, original,
        "file should have been modified. stdout: {stdout}"
    );

    // Structural: a `///` doc comment exists in proximity to `pub fn add`
    // (within 3 lines, allowing for blank lines that models sometimes insert)
    let lines: Vec<&str> = final_content.lines().collect();
    let add_line_idx = lines
        .iter()
        .position(|l| l.trim().starts_with("pub fn add"));
    assert!(
        add_line_idx.is_some(),
        "pub fn add should still be in file: {final_content}"
    );

    let add_idx = add_line_idx.unwrap();
    // Look back up to 3 lines for a doc comment (models sometimes add blank lines)
    let search_start = add_idx.saturating_sub(3);
    let has_doc_nearby = lines[search_start..add_idx]
        .iter()
        .any(|l| l.trim().starts_with("///"));
    assert!(
        has_doc_nearby,
        "there should be a `///` doc comment within 3 lines before `pub fn add`.\n\
         Final content:\n{final_content}"
    );

    // Preservation: function body unchanged
    assert!(
        final_content.contains("a + b"),
        "function body should be preserved: {final_content}"
    );

    // Session was saved
    let session_dir = config.join("piku").join("sessions");
    let session_count = std::fs::read_dir(&session_dir)
        .map(|rd| rd.filter_map(std::result::Result::ok).count())
        .unwrap_or(0);
    assert!(session_count > 0, "piku should have saved a session file");
}

// ---------------------------------------------------------------------------
// Test 2: piku creates a new file when asked
// ---------------------------------------------------------------------------

#[test]
fn piku_creates_new_file_with_content() {
    if !is_llm_e2e_enabled() {
        eprintln!("Skipping: PIKU_LLM_E2E not set");
        return;
    }
    let Some((provider, key_var, model)) = detect_provider() else {
        eprintln!("Skipping: no API key found");
        return;
    };

    let dir = tempdir();
    let config = dir.join("config");
    let target = dir.join("hello.txt");

    let prompt = format!(
        "Create a new file at {path} containing exactly the text: \
         PIKU_CREATED_FILE_MARKER\n\
         Do not add anything else to the file.",
        path = target.display()
    );

    let (_, stderr, success) = run_piku(&prompt, provider, model, key_var, &dir, &config);
    assert!(success, "piku should exit 0. stderr: {stderr}");

    assert!(target.exists(), "piku should have created the file");
    let content = std::fs::read_to_string(&target).unwrap();
    assert!(
        content.contains("PIKU_CREATED_FILE_MARKER"),
        "file should contain the marker. content: {content:?}"
    );
}

// ---------------------------------------------------------------------------
// Test 3: piku reads a file and answers a question about it
// ---------------------------------------------------------------------------

/// Verifies the `read_file` tool works end-to-end with a real LLM.
/// We can't assert the exact answer but we can assert:
/// - piku produced non-empty output
/// - The output contains words from the file (demonstrates it was read)
#[test]
fn piku_reads_file_and_references_content() {
    if !is_llm_e2e_enabled() {
        eprintln!("Skipping: PIKU_LLM_E2E not set");
        return;
    }
    let Some((provider, key_var, model)) = detect_provider() else {
        eprintln!("Skipping: no API key found");
        return;
    };

    let dir = tempdir();
    let config = dir.join("config");
    let target = dir.join("manifest.txt");

    // File with a unique sentinel that the model must reference
    let unique_token = "XYLOPHONE_PURPLE_42";
    std::fs::write(
        &target,
        format!("project_id: {unique_token}\nversion: 1.0\n"),
    )
    .unwrap();

    let prompt = format!(
        "Read the file at {path} and tell me the project_id value.",
        path = target.display()
    );

    let (stdout, stderr, success) = run_piku(&prompt, provider, model, key_var, &dir, &config);
    assert!(success, "piku should exit 0. stderr: {stderr}");

    // The response must mention the unique token
    assert!(
        stdout.contains(unique_token),
        "piku's response should contain '{unique_token}' (from the file). stdout: {stdout}"
    );
}

// ---------------------------------------------------------------------------
// Test 4: piku handles bash tool (runs a real command)
// ---------------------------------------------------------------------------

#[test]
fn piku_runs_bash_and_reports_output() {
    if !is_llm_e2e_enabled() {
        eprintln!("Skipping: PIKU_LLM_E2E not set");
        return;
    }
    let Some((provider, key_var, model)) = detect_provider() else {
        eprintln!("Skipping: no API key found");
        return;
    };

    let dir = tempdir();
    let config = dir.join("config");

    // Ask for a deterministic bash operation
    let sentinel_file = dir.join("sentinel.txt");
    let prompt = format!(
        "Run the bash command: echo PIKU_BASH_WORKED > {path}\n\
         Then confirm what you wrote to the file.",
        path = sentinel_file.display()
    );

    let (_, stderr, success) = run_piku(&prompt, provider, model, key_var, &dir, &config);
    assert!(success, "piku should exit 0. stderr: {stderr}");

    // The file should exist and have the sentinel
    assert!(sentinel_file.exists(), "bash should have created the file");
    let content = std::fs::read_to_string(&sentinel_file).unwrap();
    assert!(
        content.contains("PIKU_BASH_WORKED"),
        "file should contain sentinel. content: {content:?}"
    );
}

// ---------------------------------------------------------------------------
// Test 5: piku self-describes using glob + read
// ---------------------------------------------------------------------------

/// Tests the full agentic loop: glob finds files, read reads one.
/// Uses the actual piku source tree as the codebase.
#[test]
fn piku_explores_own_codebase_with_glob_and_read() {
    if !is_llm_e2e_enabled() {
        eprintln!("Skipping: PIKU_LLM_E2E not set");
        return;
    }
    let Some((provider, key_var, model)) = detect_provider() else {
        eprintln!("Skipping: no API key found");
        return;
    };

    // Run from the piku workspace root so it can find its own source
    let workspace_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf();

    let config = std::env::temp_dir().join(format!("piku_e2e_config_{}", std::process::id()));

    let prompt = "Use glob to find all .rs files in the crates/ directory \
                  (pattern: crates/**/*.rs, path: .). \
                  Then read crates/piku/src/lib.rs and tell me what modules it exports.";

    let (stdout, stderr, success) =
        run_piku(prompt, provider, model, key_var, &workspace_root, &config);

    let _ = std::fs::remove_dir_all(&config);

    assert!(success, "piku should exit 0. stderr: {stderr}");

    // The response should mention modules from lib.rs
    // lib.rs exports `self_update` and `cli`
    assert!(
        stdout.contains("self_update") || stdout.contains("cli"),
        "response should mention modules from lib.rs. stdout: {stdout}"
    );
}
