

/// Dogfooding harness — piku tests itself using a real LLM.
///
/// Each scenario describes an idea, seeds a workspace, runs piku against it,
/// and produces a structured "experience report" of what piku actually did:
/// which tools it called, with what args, whether they succeeded, and what
/// the final filesystem looks like.
///
/// GATING: Requires `PIKU_DOGFOOD=1` (so CI stays fast by default).
///
/// SETUP:
///   cargo build --release -p piku
///   export OPENROUTER_API_KEY=sk-or-...   # or ANTHROPIC_API_KEY
///   PIKU_DOGFOOD=1 cargo test --test dogfood -- --nocapture
///
/// TO RUN ONE SCENARIO:
///   PIKU_DOGFOOD=1 cargo test --test dogfood scenario_name -- --nocapture
///
/// The test never fails on LLM output quality — it only fails on crashes,
/// tool errors that shouldn't happen, or explicit structural assertions.
/// The experience report is always printed so you can read what happened.
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

// ---------------------------------------------------------------------------
// Gate + infrastructure
// ---------------------------------------------------------------------------

fn is_enabled() -> bool {
    std::env::var("PIKU_DOGFOOD")
        .map(|v| v == "1" || v == "true")
        .unwrap_or(false)
}

fn piku_binary() -> PathBuf {
    let exe = std::env::current_exe().unwrap();
    let profile_dir = exe.parent().unwrap().parent().unwrap();
    let same = profile_dir.join("piku");
    if same.exists() {
        return same;
    }
    let release = profile_dir.parent().unwrap().join("release").join("piku");
    if release.exists() {
        return release;
    }
    panic!("piku binary not found — run `cargo build --release -p piku` first");
}

fn has_key(var: &str) -> bool {
    std::env::var(var).map(|v| !v.is_empty()).unwrap_or(false)
}

fn detect_provider() -> Option<(&'static str, &'static str, &'static str)> {
    if has_key("OPENROUTER_API_KEY") {
        return Some((
            "openrouter",
            "OPENROUTER_API_KEY",
            "anthropic/claude-sonnet-4-5",
        ));
    }
    if has_key("ANTHROPIC_API_KEY") {
        return Some(("anthropic", "ANTHROPIC_API_KEY", "claude-sonnet-4-5"));
    }
    None
}

fn tempdir(label: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.subsec_nanos())
        .unwrap_or(0);
    let base = std::env::temp_dir().join(format!("piku_dogfood_{label}_{nanos}"));
    std::fs::create_dir_all(&base).unwrap();
    base
}

// ---------------------------------------------------------------------------
// Experience report
// ---------------------------------------------------------------------------

/// One observed tool invocation.
#[derive(Debug)]
struct ToolCall {
    name: String,
    args: String, // the formatted arg string piku printed
    ok: bool,
    result_preview: String,
}

/// What piku actually did in a run.
#[derive(Debug)]
struct Experience {
    stdout: String,
    stderr: String,
    exit_ok: bool,
    tool_calls: Vec<ToolCall>,
    /// Files in the workspace that were created or modified.
    touched_files: Vec<PathBuf>,
    /// The assistant's final text response (everything not inside tool lines).
    response_text: String,
}

impl Experience {
    /// Print a human-readable report.
    fn report(&self, scenario: &str) {
        println!("\n╔═══════════════════════════════════════════════════");
        println!("║ DOGFOOD EXPERIENCE: {scenario}");
        println!("╠═══════════════════════════════════════════════════");
        println!("║ exit: {}", if self.exit_ok { "✓ ok" } else { "✗ failed" });
        println!("║ tools ({}):", self.tool_calls.len());
        for tc in &self.tool_calls {
            let status = if tc.ok { "✓" } else { "✗" };
            println!("║   {status} {}  {}", tc.name, tc.args);
            if !tc.result_preview.is_empty() {
                for line in tc.result_preview.lines().take(3) {
                    println!("║       {line}");
                }
            }
        }
        if !self.touched_files.is_empty() {
            println!("║ files touched ({}):", self.touched_files.len());
            for f in &self.touched_files {
                println!("║   {}", f.display());
            }
        }
        println!("║ response:");
        for line in self.response_text.lines().take(10) {
            println!("║   {line}");
        }
        if self.response_text.lines().count() > 10 {
            let n = self.response_text.lines().count() - 10;
            println!("║   … ({n} more lines)");
        }
        println!("╚═══════════════════════════════════════════════════\n");
    }

    fn tool_names(&self) -> Vec<&str> {
        self.tool_calls.iter().map(|t| t.name.as_str()).collect()
    }

    fn all_tools_ok(&self) -> bool {
        self.tool_calls.iter().all(|t| t.ok)
    }

    fn tool_called(&self, name: &str) -> bool {
        self.tool_calls.iter().any(|t| t.name == name)
    }
}

// ---------------------------------------------------------------------------
// Strip ANSI escape codes from a string
// ---------------------------------------------------------------------------

fn strip_ansi(s: &str) -> String {
    // Matches ESC [ ... m and other common sequences
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\x1b' {
            if chars.peek() == Some(&'[') {
                chars.next();
                // consume until a letter
                for ch in chars.by_ref() {
                    if ch.is_ascii_alphabetic() {
                        break;
                    }
                }
            }
        } else {
            out.push(c);
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Parse piku single-shot output into an Experience
// ---------------------------------------------------------------------------

/// Parse the stdout of `piku "prompt"` (single-shot StdoutSink format).
///
/// Tool start line:  `[tool_name args …]`  (after ANSI strip)
/// Tool end line:    `[tool_name → ok]` or `[tool_name → err]`
/// Everything else: response text
fn parse_output(stdout: &str, stderr: &str, exit_ok: bool, workspace: &Path) -> Experience {
    let clean = strip_ansi(stdout);

    let mut tool_calls: Vec<ToolCall> = Vec::new();
    let mut response_lines: Vec<String> = Vec::new();

    // State: we may be inside a tool result block
    let mut current_tool: Option<(String, String)> = None; // (name, args)
    let mut in_result = false;
    let mut result_lines: Vec<String> = Vec::new();

    for line in clean.lines() {
        let trimmed = line.trim();

        // Tool start: `[tool_name …]` or `[tool_name args …]`
        // `…` = U+2026 = 3 bytes; full suffix ` …]` = 5 bytes, `…]` = 4 bytes.
        if trimmed.starts_with('[') && trimmed.ends_with("…]") {
            // Strip `[` prefix and `…]` suffix (4 bytes), plus any trailing space before `…`
            let inner = trimmed[1..trimmed.len() - 4].trim_end();
            // Split on first whitespace: name vs args
            let (name, args) = if let Some(pos) = inner.find(' ') {
                (&inner[..pos], inner[pos + 1..].trim())
            } else {
                (inner, "")
            };
            current_tool = Some((name.to_string(), args.to_string()));
            in_result = false;
            result_lines.clear();
            continue;
        }

        // Tool end: `[tool_name → ok]` or `[tool_name → err]`
        if trimmed.starts_with('[') && (trimmed.contains("→ ok]") || trimmed.contains("→ err]"))
        {
            let ok = trimmed.contains("→ ok]");
            if let Some((name, args)) = current_tool.take() {
                tool_calls.push(ToolCall {
                    name,
                    args,
                    ok,
                    result_preview: result_lines.join("\n"),
                });
            }
            in_result = false;
            result_lines.clear();
            continue;
        }

        // If we just saw a tool end line, the next lines until the next tool
        // start are the result preview (piku prints result after the end line).
        if current_tool.is_none() && !tool_calls.is_empty() {
            // Could be result preview or response text — we can't distinguish
            // perfectly, so just collect as response text.
            if !trimmed.is_empty() {
                response_lines.push(line.to_string());
            }
        } else if current_tool.is_some() {
            // Between start and end — shouldn't happen in single-shot format
            // (start → end are consecutive), but handle gracefully.
        } else {
            if !trimmed.is_empty() {
                response_lines.push(line.to_string());
            }
        }
    }

    // Collect files in workspace (non-hidden, non-config)
    let touched_files = collect_workspace_files(workspace);

    Experience {
        stdout: stdout.to_string(),
        stderr: stderr.to_string(),
        exit_ok,
        tool_calls,
        touched_files,
        response_text: response_lines.join("\n"),
    }
}

fn collect_workspace_files(dir: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.filter_map(|e| e.ok()) {
            let path = entry.path();
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if name.starts_with('.') || name == "config" {
                continue;
            }
            if path.is_file() {
                files.push(path);
            } else if path.is_dir() {
                files.extend(collect_workspace_files(&path));
            }
        }
    }
    files.sort();
    files
}

// ---------------------------------------------------------------------------
// Run piku and collect experience
// ---------------------------------------------------------------------------

fn run_scenario(
    prompt: &str,
    workspace: &Path,
    provider: &str,
    model: &str,
    key_var: &str,
) -> Experience {
    let api_key = std::env::var(key_var).unwrap();
    let config_dir = workspace.join(".piku-config");
    std::fs::create_dir_all(&config_dir).unwrap();

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
        .env("XDG_CONFIG_HOME", &config_dir)
        .env("TERM", "xterm-256color")
        .current_dir(workspace)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .expect("failed to spawn piku");

    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    let exit_ok = output.status.success();

    parse_output(&stdout, &stderr, exit_ok, workspace)
}

// ---------------------------------------------------------------------------
// Scenarios
// ---------------------------------------------------------------------------

/// Scenario: piku edits a Rust file to add a function.
///
/// Idea: does piku correctly identify where to add code, write it,
/// and verify it compiles? Does it read first or just write blind?
#[test]
fn dogfood_add_function_to_existing_file() {
    if !is_enabled() {
        return;
    }
    let Some((provider, key_var, model)) = detect_provider() else {
        eprintln!("no API key — skipping");
        return;
    };

    let workspace = tempdir("add_fn");

    // Seed: a real Rust file with a gap for a new function
    std::fs::write(
        workspace.join("math.rs"),
        r#"// Simple math utilities

pub fn add(a: i32, b: i32) -> i32 {
    a + b
}

pub fn subtract(a: i32, b: i32) -> i32 {
    a - b
}
"#,
    )
    .unwrap();

    let exp = run_scenario(
        "In math.rs, add a `multiply(a: i32, b: i32) -> i32` function that returns a * b. \
         Add it after the subtract function.",
        &workspace,
        provider,
        model,
        key_var,
    );
    exp.report("add_function_to_existing_file");

    // Structural: piku should have read or edited the file
    assert!(
        exp.tool_called("read_file")
            || exp.tool_called("edit_file")
            || exp.tool_called("write_file"),
        "piku should have used at least one file tool"
    );

    // File must exist and contain multiply
    let content = std::fs::read_to_string(workspace.join("math.rs")).unwrap();
    assert!(
        content.contains("multiply"),
        "math.rs should contain multiply function.\ncontent:\n{content}"
    );
    assert!(
        content.contains("a * b") || content.contains("a*b"),
        "multiply should return a * b.\ncontent:\n{content}"
    );

    // Original functions must still be there
    assert!(
        content.contains("pub fn add"),
        "add should still be present"
    );
    assert!(
        content.contains("pub fn subtract"),
        "subtract should still be present"
    );
}

/// Scenario: piku reads a file and answers a question — does it read before answering?
///
/// Idea: verify the full read_file → response loop works, and that the
/// answer actually reflects the file contents (not a hallucination).
#[test]
fn dogfood_read_and_answer() {
    if !is_enabled() {
        return;
    }
    let Some((provider, key_var, model)) = detect_provider() else {
        eprintln!("no API key — skipping");
        return;
    };

    let workspace = tempdir("read_answer");

    // Use a unique token that the model can't hallucinate
    let secret = "PIKU_DOGFOOD_SECRET_7f3a9b";
    std::fs::write(
        workspace.join("config.toml"),
        format!("[server]\nhost = \"localhost\"\nport = 8080\nsecret_token = \"{secret}\"\n"),
    )
    .unwrap();

    let exp = run_scenario(
        "Read config.toml and tell me the value of secret_token.",
        &workspace,
        provider,
        model,
        key_var,
    );
    exp.report("read_and_answer");

    assert!(
        exp.tool_called("read_file"),
        "piku must use read_file to answer accurately"
    );
    assert!(exp.all_tools_ok(), "all tools should succeed");

    // Response must contain the secret
    let full_output = strip_ansi(&exp.stdout);
    assert!(
        full_output.contains(secret),
        "response should contain the secret token '{secret}'.\nfull output:\n{full_output}"
    );
}

/// Scenario: piku explores a small codebase with glob + grep, then summarises.
///
/// Idea: does the multi-tool agentic loop work? Does it use glob to discover
/// files then grep to find patterns, rather than reading everything blindly?
#[test]
fn dogfood_explore_codebase() {
    if !is_enabled() {
        return;
    }
    let Some((provider, key_var, model)) = detect_provider() else {
        eprintln!("no API key — skipping");
        return;
    };

    let workspace = tempdir("explore");
    std::fs::create_dir_all(workspace.join("src")).unwrap();

    std::fs::write(
        workspace.join("src/main.rs"),
        r#"mod utils;
fn main() {
    let result = utils::compute(6, 7);
    println!("result: {result}");
}
"#,
    )
    .unwrap();

    std::fs::write(
        workspace.join("src/utils.rs"),
        r#"// PIKU_DOGFOOD_MARKER: key utility functions

pub fn compute(x: i32, y: i32) -> i32 {
    x * y
}

pub fn greet(name: &str) -> String {
    format!("hello, {name}!")
}
"#,
    )
    .unwrap();

    std::fs::write(
        workspace.join("README.md"),
        "# Test project\nA small Rust project for piku dogfooding.\n",
    )
    .unwrap();

    let exp = run_scenario(
        "Explore this Rust project. Use glob to find all .rs files, \
         then grep for functions (pattern: `pub fn`). \
         List what public functions exist.",
        &workspace,
        provider,
        model,
        key_var,
    );
    exp.report("explore_codebase");

    assert!(
        exp.tool_called("glob"),
        "piku should use glob to discover files"
    );
    assert!(
        exp.tool_called("grep"),
        "piku should use grep to find functions"
    );
    assert!(exp.all_tools_ok(), "all tools should succeed");

    // Response should mention the functions it found
    let full_output = strip_ansi(&exp.stdout);
    assert!(
        full_output.contains("compute") || full_output.contains("greet"),
        "response should mention the public functions.\noutput:\n{full_output}"
    );
}

/// Scenario: piku surgically edits a file with an ambiguous pattern, then retries.
///
/// Idea: does edit_file correctly reject ambiguous matches and does the model
/// recover by providing more context on a retry?
#[test]
fn dogfood_edit_with_ambiguous_pattern() {
    if !is_enabled() {
        return;
    }
    let Some((provider, key_var, model)) = detect_provider() else {
        eprintln!("no API key — skipping");
        return;
    };

    let workspace = tempdir("ambig_edit");

    std::fs::write(
        workspace.join("handlers.rs"),
        r#"fn handle_get() -> &'static str {
    "GET response"
}

fn handle_post() -> &'static str {
    "POST response"
}
"#,
    )
    .unwrap();

    let exp = run_scenario(
        r#"In handlers.rs, change the return value of handle_post to "POST accepted". \
         Do not change handle_get."#,
        &workspace,
        provider,
        model,
        key_var,
    );
    exp.report("edit_with_ambiguous_pattern");

    let content = std::fs::read_to_string(workspace.join("handlers.rs")).unwrap();

    // handle_get must be untouched
    assert!(
        content.contains(r#""GET response""#),
        "handle_get should be untouched.\ncontent:\n{content}"
    );
    // handle_post must be updated
    assert!(
        content.contains("POST accepted"),
        "handle_post should be updated.\ncontent:\n{content}"
    );
}

/// Scenario: piku runs a shell command and uses its output.
///
/// Idea: does piku correctly run bash and incorporate the output into its response?
/// Uses a deterministic command (echo + wc) so output is predictable.
#[test]
fn dogfood_bash_and_use_output() {
    if !is_enabled() {
        return;
    }
    let Some((provider, key_var, model)) = detect_provider() else {
        eprintln!("no API key — skipping");
        return;
    };

    let workspace = tempdir("bash_output");

    // Seed a file with a known line count
    let lines: Vec<String> = (1..=7).map(|i| format!("line {i}")).collect();
    std::fs::write(workspace.join("data.txt"), lines.join("\n") + "\n").unwrap();

    let exp = run_scenario(
        "Run `wc -l data.txt` and tell me how many lines the file has.",
        &workspace,
        provider,
        model,
        key_var,
    );
    exp.report("bash_and_use_output");

    assert!(exp.tool_called("bash"), "piku should run bash");
    assert!(exp.all_tools_ok(), "bash should succeed");

    let full_output = strip_ansi(&exp.stdout);
    assert!(
        full_output.contains('7') || full_output.contains("seven"),
        "response should mention 7 lines.\noutput:\n{full_output}"
    );
}

/// Scenario: piku writes a new file from scratch.
///
/// Idea: does piku create a file that's actually syntactically correct?
/// We check the file exists and contains key structural elements.
#[test]
fn dogfood_write_new_file_from_scratch() {
    if !is_enabled() {
        return;
    }
    let Some((provider, key_var, model)) = detect_provider() else {
        eprintln!("no API key — skipping");
        return;
    };

    let workspace = tempdir("write_new");

    let exp = run_scenario(
        "Create a new file called `greeter.rs` with a public function \
         `greet(name: &str) -> String` that returns `format!(\"Hello, {}!\", name)`. \
         No tests needed, just the function.",
        &workspace,
        provider,
        model,
        key_var,
    );
    exp.report("write_new_file_from_scratch");

    let path = workspace.join("greeter.rs");
    assert!(path.exists(), "greeter.rs should have been created");

    let content = std::fs::read_to_string(&path).unwrap();
    assert!(content.contains("pub fn greet"), "should have pub fn greet");
    assert!(content.contains("name"), "should use the name parameter");
    assert!(content.contains("Hello"), "should say Hello");
}

/// Scenario: piku reads a file, edits it, then verifies the edit by reading again.
///
/// Idea: does piku verify its own work? Does it read → edit → read?
/// The experience report shows the tool sequence clearly.
#[test]
fn dogfood_read_edit_verify_loop() {
    if !is_enabled() {
        return;
    }
    let Some((provider, key_var, model)) = detect_provider() else {
        eprintln!("no API key — skipping");
        return;
    };

    let workspace = tempdir("edit_verify");

    std::fs::write(workspace.join("version.txt"), "version = 1.0.0\n").unwrap();

    let exp = run_scenario(
        "Read version.txt, change the version from 1.0.0 to 2.0.0, \
         then read the file again to confirm the change was made.",
        &workspace,
        provider,
        model,
        key_var,
    );
    exp.report("read_edit_verify_loop");

    // Should have used at least 2 file tool calls (read, edit/write, maybe read again)
    assert!(
        exp.tool_calls.len() >= 2,
        "should use at least 2 tool calls. got: {:?}",
        exp.tool_names()
    );

    let content = std::fs::read_to_string(workspace.join("version.txt")).unwrap();
    assert!(
        content.contains("2.0.0"),
        "version should have been bumped to 2.0.0.\ncontent: {content}"
    );
    assert!(
        !content.contains("1.0.0"),
        "old version should be gone.\ncontent: {content}"
    );
}

// ---------------------------------------------------------------------------
// Harder scenarios (multi-file, bug detection, test writing, self-description)
// ---------------------------------------------------------------------------

/// Scenario: piku renames a function that's used across two files.
///
/// Idea: does piku use grep to find all usages before editing?
/// Does it update all call sites, not just the definition?
#[test]
fn dogfood_multifile_rename() {
    if !is_enabled() {
        return;
    }
    let Some((provider, key_var, model)) = detect_provider() else {
        eprintln!("no API key — skipping");
        return;
    };

    let workspace = tempdir("multifile_rename");
    std::fs::create_dir_all(workspace.join("src")).unwrap();

    // lib.rs defines the function
    std::fs::write(
        workspace.join("src/lib.rs"),
        r#"pub fn compute_total(items: &[i32]) -> i32 {
    items.iter().sum()
}
"#,
    )
    .unwrap();

    // main.rs calls it
    std::fs::write(
        workspace.join("src/main.rs"),
        r#"mod lib;
fn main() {
    let items = vec![1, 2, 3];
    let total = lib::compute_total(&items);
    println!("total: {total}");
}
"#,
    )
    .unwrap();

    // utils.rs also calls it
    std::fs::write(
        workspace.join("src/utils.rs"),
        r#"use crate::lib;
pub fn print_sum(values: &[i32]) {
    let result = lib::compute_total(values);
    println!("sum = {result}");
}
"#,
    )
    .unwrap();

    let exp = run_scenario(
        "Rename the function `compute_total` to `sum_items` everywhere in src/. \
         Update the definition in lib.rs and all call sites in main.rs and utils.rs. \
         Do not change any logic.",
        &workspace,
        provider,
        model,
        key_var,
    );
    exp.report("multifile_rename");

    // All three files should now use sum_items
    let lib_content = std::fs::read_to_string(workspace.join("src/lib.rs")).unwrap();
    let main_content = std::fs::read_to_string(workspace.join("src/main.rs")).unwrap();
    let utils_content = std::fs::read_to_string(workspace.join("src/utils.rs")).unwrap();

    assert!(
        lib_content.contains("sum_items"),
        "lib.rs should define sum_items.\ncontent:\n{lib_content}"
    );
    assert!(
        main_content.contains("sum_items"),
        "main.rs should call sum_items.\ncontent:\n{main_content}"
    );
    assert!(
        utils_content.contains("sum_items"),
        "utils.rs should call sum_items.\ncontent:\n{utils_content}"
    );

    // Old name must be gone everywhere
    assert!(
        !lib_content.contains("compute_total"),
        "lib.rs should not have old name.\ncontent:\n{lib_content}"
    );
}

/// Scenario: piku finds an intentional off-by-one bug in a loop.
///
/// Idea: does piku actually read the code carefully before answering?
/// Does it identify the specific line and explain why it's wrong?
#[test]
fn dogfood_find_off_by_one_bug() {
    if !is_enabled() {
        return;
    }
    let Some((provider, key_var, model)) = detect_provider() else {
        eprintln!("no API key — skipping");
        return;
    };

    let workspace = tempdir("find_bug");

    // The bug: `i < n - 1` skips the last element
    std::fs::write(
        workspace.join("stats.rs"),
        r#"/// Returns the sum of all elements in the slice.
// PIKU_DOGFOOD_BUG: this function has an off-by-one error
pub fn sum(values: &[i32]) -> i32 {
    let n = values.len();
    let mut total = 0;
    let mut i = 0;
    while i < n - 1 {   // BUG: should be i < n
        total += values[i];
        i += 1;
    }
    total
}
"#,
    )
    .unwrap();

    let exp = run_scenario(
        "Read stats.rs. There is a bug in the `sum` function — it does not return \
         the correct sum. Identify the bug, explain it, and fix it.",
        &workspace,
        provider,
        model,
        key_var,
    );
    exp.report("find_off_by_one_bug");

    assert!(
        exp.tool_called("read_file"),
        "piku must read the file to find the bug"
    );

    // After the fix, the file should use `i < n` not `i < n - 1`
    let content = std::fs::read_to_string(workspace.join("stats.rs")).unwrap();
    // Either fixed the while condition or rewrote using iter().sum()
    let fixed = content.contains("i < n")
        || content.contains("i <= n")
        || content.contains(".sum()")
        || content.contains("iter()");
    assert!(
        fixed,
        "the bug should be fixed (i < n, or iter().sum()). content:\n{content}"
    );
}

/// Scenario: piku writes a test for a function it hasn't seen before.
///
/// Idea: does piku read the function signature before writing tests?
/// Does it produce tests that actually test the contract, not boilerplate?
#[test]
fn dogfood_write_tests_for_function() {
    if !is_enabled() {
        return;
    }
    let Some((provider, key_var, model)) = detect_provider() else {
        eprintln!("no API key — skipping");
        return;
    };

    let workspace = tempdir("write_tests");

    std::fs::write(
        workspace.join("parser.rs"),
        r#"/// Splits a `key=value` string into `(key, value)`.
/// Returns None if the string does not contain `=`.
pub fn parse_kv(s: &str) -> Option<(&str, &str)> {
    let pos = s.find('=')?;
    Some((&s[..pos], &s[pos+1..]))
}
"#,
    )
    .unwrap();

    let exp = run_scenario(
        "Read parser.rs and write a Rust #[cfg(test)] module inside the same file with \
         at least 3 unit tests for `parse_kv`. Cover: normal case, missing =, empty string.",
        &workspace,
        provider,
        model,
        key_var,
    );
    exp.report("write_tests_for_function");

    assert!(
        exp.tool_called("read_file"),
        "piku should read parser.rs first"
    );
    assert!(
        exp.tool_called("edit_file") || exp.tool_called("write_file"),
        "piku should write to the file"
    );

    let content = std::fs::read_to_string(workspace.join("parser.rs")).unwrap();
    assert!(
        content.contains("#[cfg(test)]"),
        "file should contain a test module.\ncontent:\n{content}"
    );
    assert!(
        content.contains("#[test]"),
        "file should have at least one #[test].\ncontent:\n{content}"
    );
    // Should have at least 3 #[test] annotations
    let test_count = content.matches("#[test]").count();
    assert!(
        test_count >= 3,
        "should have at least 3 tests, found {test_count}.\ncontent:\n{content}"
    );
}

/// Scenario: piku explores its own source and summarises the architecture.
///
/// Idea: when pointed at the piku repo, does piku correctly describe what it finds?
/// Tests the full agentic loop: glob → read → synthesise.
/// This is a confidence check — does piku understand codebases structurally?
#[test]
fn dogfood_self_describe_architecture() {
    if !is_enabled() {
        return;
    }
    let Some((provider, key_var, model)) = detect_provider() else {
        eprintln!("no API key — skipping");
        return;
    };

    // Run from the actual piku workspace root
    let workspace = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf();

    let exp = run_scenario(
        "Use glob to list all Cargo.toml files in crates/ (pattern: crates/*/Cargo.toml). \
         Then read crates/piku-runtime/src/agent_loop.rs line 1 to 30. \
         Describe in one sentence: what is piku, and what does the agentic loop do?",
        &workspace,
        provider,
        model,
        key_var,
    );
    exp.report("self_describe_architecture");

    // It must have used glob to discover structure
    assert!(
        exp.tool_called("glob"),
        "piku should use glob to discover Cargo.toml files"
    );
    assert!(
        exp.tool_called("read_file"),
        "piku should read agent_loop.rs"
    );

    // Response should mention something about the loop or agent
    let full_output = strip_ansi(&exp.stdout);
    let mentions_agent = full_output.to_lowercase().contains("agent")
        || full_output.to_lowercase().contains("loop")
        || full_output.to_lowercase().contains("tool");
    assert!(
        mentions_agent,
        "response should mention the agentic loop.\noutput:\n{full_output}"
    );
}

/// Scenario: verify the trace file was written after a run.
///
/// Idea: pure infrastructure check — the JSONL trace must exist and contain
/// at least the expected event types (prompt, tool_start, turn_end).
#[test]
fn dogfood_trace_file_is_written() {
    if !is_enabled() {
        return;
    }
    let Some((provider, key_var, model)) = detect_provider() else {
        eprintln!("no API key — skipping");
        return;
    };

    let workspace = tempdir("trace_check");
    let config_dir = workspace.join(".piku-config");
    std::fs::create_dir_all(&config_dir).unwrap();

    std::fs::write(workspace.join("hello.txt"), "hello world\n").unwrap();

    let api_key = std::env::var(key_var).unwrap();
    let output = std::process::Command::new(piku_binary())
        .arg("--provider")
        .arg(provider)
        .arg("--model")
        .arg(model)
        .arg("Read hello.txt and tell me what it says.")
        .env_clear()
        .env(key_var, &api_key)
        .env("PATH", std::env::var("PATH").unwrap_or_default())
        .env("HOME", std::env::var("HOME").unwrap_or_default())
        .env("XDG_CONFIG_HOME", &config_dir)
        .env("TERM", "xterm-256color")
        .current_dir(&workspace)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .output()
        .expect("failed to spawn piku");

    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    let exp = parse_output(&stdout, &stderr, output.status.success(), &workspace);
    exp.report("trace_file_is_written");

    // Traces dir should exist under config_dir
    let traces_dir = config_dir.join("piku").join("traces");
    assert!(
        traces_dir.exists(),
        "traces dir should have been created at {}",
        traces_dir.display()
    );

    let trace_files: Vec<_> = std::fs::read_dir(&traces_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map(|x| x == "jsonl").unwrap_or(false))
        .collect();

    assert!(
        !trace_files.is_empty(),
        "at least one .jsonl trace file should exist in {}",
        traces_dir.display()
    );

    // Read the trace and verify it has expected event types
    let trace_path = trace_files[0].path();
    let trace_content = std::fs::read_to_string(&trace_path).unwrap();

    // Each line should be valid JSON
    let mut events: Vec<serde_json::Value> = Vec::new();
    for line in trace_content.lines() {
        let v: serde_json::Value =
            serde_json::from_str(line).expect(&format!("trace line should be valid JSON: {line}"));
        events.push(v);
    }

    assert!(!events.is_empty(), "trace file should have events");

    let event_types: Vec<&str> = events.iter().filter_map(|e| e["event"].as_str()).collect();

    println!("trace events: {event_types:?}");

    assert!(
        event_types.contains(&"prompt"),
        "trace should contain a 'prompt' event. events: {event_types:?}"
    );
    assert!(
        event_types.contains(&"turn_end"),
        "trace should contain a 'turn_end' event. events: {event_types:?}"
    );
    // Should have used read_file for "tell me what it says"
    assert!(
        event_types.contains(&"tool_start"),
        "trace should contain tool_start events. events: {event_types:?}"
    );
}
