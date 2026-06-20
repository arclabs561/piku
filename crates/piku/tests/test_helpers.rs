// Shared test helpers.
#![allow(dead_code)]

use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Duration;

/// Strip ANSI escape sequences from a string for plain-text assertions.
#[must_use]
pub fn strip_ansi(s: &str) -> String {
    let mut out = String::new();
    let mut chars = s.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '\x1b' {
            if chars.peek() == Some(&'[') {
                chars.next();
                for c in chars.by_ref() {
                    if c.is_ascii_alphabetic() {
                        break;
                    }
                }
            } else if chars.peek() == Some(&']') {
                // OSC sequence: skip until BEL or ST
                chars.next();
                for c in chars.by_ref() {
                    if c == '\x07' || c == '\\' {
                        break;
                    }
                }
            }
        } else {
            out.push(ch);
        }
    }
    out
}

fn current_test_name() -> String {
    std::thread::current()
        .name()
        .and_then(|name| name.rsplit("::").next())
        .unwrap_or("unknown")
        .to_string()
}

pub fn latest_trace_path(config_dir: &Path) -> Option<PathBuf> {
    let traces_dir = config_dir.join("piku").join("traces");
    let mut paths: Vec<PathBuf> = std::fs::read_dir(traces_dir)
        .ok()?
        .filter_map(std::result::Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "jsonl"))
        .collect();
    paths.sort();
    paths.pop()
}

pub fn append_live_ledger(
    suite: &str,
    provider: &str,
    model: &str,
    config_dir: &Path,
    exit_ok: bool,
    duration: Duration,
) {
    let Ok(ledger_path) = std::env::var("PIKU_LIVE_LEDGER") else {
        return;
    };

    let trace_path = latest_trace_path(config_dir);
    let mut input_tokens = 0;
    let mut output_tokens = 0;
    let mut iterations = 0;
    let mut tool_starts = 0;
    let mut tool_ends = 0;
    let mut failed_tools = 0;
    let mut permission_denied = 0;

    if let Some(path) = &trace_path {
        if let Ok(content) = std::fs::read_to_string(path) {
            for line in content.lines() {
                let Ok(event) = serde_json::from_str::<serde_json::Value>(line) else {
                    continue;
                };
                match event["event"].as_str() {
                    Some("tool_start") => tool_starts += 1,
                    Some("tool_end") => {
                        tool_ends += 1;
                        if event["ok"].as_bool() == Some(false) {
                            failed_tools += 1;
                        }
                    }
                    Some("permission_denied") => permission_denied += 1,
                    Some("turn_end") => {
                        input_tokens = event["input_tokens"].as_u64().unwrap_or(0);
                        output_tokens = event["output_tokens"].as_u64().unwrap_or(0);
                        iterations = event["iterations"].as_u64().unwrap_or(0);
                    }
                    _ => {}
                }
            }
        }
    }

    let failure_class = if exit_ok {
        "none"
    } else if permission_denied > 0 {
        "permission_denied"
    } else if failed_tools > 0 {
        "tool_failure"
    } else {
        "unknown_failure"
    };

    let record = serde_json::json!({
        "suite": suite,
        "test": current_test_name(),
        "provider": provider,
        "model": model,
        "result": if exit_ok { "success" } else { "failure" },
        "failure_class": failure_class,
        "trace_path": trace_path.as_ref().map(|path| path.display().to_string()),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "iterations": iterations,
        "tool_starts": tool_starts,
        "tool_ends": tool_ends,
        "failed_tools": failed_tools,
        "permission_denied": permission_denied,
        "duration_ms": duration.as_millis(),
    });

    let path = Path::new(&ledger_path);
    if let Some(parent) = path.parent() {
        if let Err(err) = std::fs::create_dir_all(parent) {
            eprintln!(
                "failed to create live ledger dir {}: {err}",
                parent.display()
            );
            return;
        }
    }
    match std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
    {
        Ok(mut file) => {
            if let Err(err) = writeln!(file, "{record}") {
                eprintln!("failed to write live ledger {}: {err}", path.display());
            }
        }
        Err(err) => eprintln!("failed to open live ledger {}: {err}", path.display()),
    }
}
