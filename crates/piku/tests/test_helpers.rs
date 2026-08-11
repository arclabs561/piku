// Shared test helpers.
#![allow(dead_code)]

use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Duration;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EvaluationSurface {
    Cli,
    Tui,
}

impl EvaluationSurface {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Cli => "cli",
            Self::Tui => "tui",
        }
    }
}

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
    append_live_ledger_for_surface(
        EvaluationSurface::Cli,
        suite,
        provider,
        model,
        config_dir,
        exit_ok,
        duration,
    );
}

pub fn append_live_ledger_for_surface(
    surface: EvaluationSurface,
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

    let record = build_live_ledger_record(
        surface, config_dir, suite, provider, model, exit_ok, duration,
    );
    if let Err(error) = validate_evaluation_envelope(&record) {
        eprintln!("refusing to append invalid evaluation envelope: {error}");
        return;
    }
    append_json_line(Path::new(&ledger_path), &record);
}

#[must_use]
pub fn build_live_ledger_record(
    surface: EvaluationSurface,
    config_dir: &Path,
    suite: &str,
    provider: &str,
    model: &str,
    exit_ok: bool,
    duration: Duration,
) -> serde_json::Value {
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
    let test_name = current_test_name();
    let run_id = format!(
        "{}-{}-{}",
        surface.as_str(),
        test_name,
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos()
    );
    let artifact_refs: Vec<String> = trace_path
        .as_ref()
        .map(|path| vec![path.display().to_string()])
        .unwrap_or_default();

    serde_json::json!({
        "schema_version": 1,
        "run_id": run_id,
        "scenario_id": test_name,
        "surface": surface.as_str(),
        "subject_surface": serde_json::Value::Null,
        "perspective": suite,
        "subject_model": model,
        "explorer_model": serde_json::Value::Null,
        "judge_model": serde_json::Value::Null,
        "task_contract": current_test_name(),
        "record_kind": "run",
        "stage_id": "result",
        "run_status": if exit_ok { "completed" } else { "inconclusive" },
        "product_verdict": serde_json::Value::Null,
        "finding_count": serde_json::Value::Null,
        "evidence_ids": [],
        "artifact_refs": artifact_refs,
        "followups": [],
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
    })
}

pub fn validate_evaluation_envelope(record: &serde_json::Value) -> Result<(), String> {
    let object = record
        .as_object()
        .ok_or_else(|| "envelope must be a JSON object".to_string())?;
    let required = [
        "schema_version",
        "run_id",
        "scenario_id",
        "surface",
        "perspective",
        "task_contract",
        "record_kind",
        "stage_id",
        "run_status",
        "failure_class",
        "product_verdict",
        "finding_count",
        "evidence_ids",
        "artifact_refs",
        "followups",
        "duration_ms",
    ];
    for field in required {
        if !object.contains_key(field) {
            return Err(format!("missing required field {field}"));
        }
    }
    if object["schema_version"].as_u64() != Some(1) {
        return Err("schema_version must equal 1".to_string());
    }
    for field in [
        "run_id",
        "scenario_id",
        "perspective",
        "task_contract",
        "stage_id",
        "failure_class",
    ] {
        if object[field].as_str().is_none_or(str::is_empty) {
            return Err(format!("{field} must be a non-empty string"));
        }
    }
    require_enum(object, "surface", &["cli", "tui", "web"])?;
    require_enum(object, "record_kind", &["run", "stage"])?;
    require_enum(
        object,
        "run_status",
        &[
            "completed",
            "product_failure",
            "harness_failure",
            "infrastructure_failure",
            "timeout",
            "inconclusive",
        ],
    )?;
    if !matches!(
        object["product_verdict"].as_str(),
        None | Some("supported" | "partial" | "not_supported")
    ) || (!object["product_verdict"].is_null() && !object["product_verdict"].is_string())
    {
        return Err("product_verdict has an invalid value".to_string());
    }
    if !(object["finding_count"].is_null() || object["finding_count"].as_u64().is_some()) {
        return Err("finding_count must be null or a non-negative integer".to_string());
    }
    require_string_array(object, "evidence_ids")?;
    require_string_array(object, "artifact_refs")?;
    let followups = object["followups"]
        .as_array()
        .ok_or_else(|| "followups must be an array".to_string())?;
    for followup in followups {
        validate_followup(followup)?;
    }
    if object["duration_ms"].as_u64().is_none() {
        return Err("duration_ms must be a non-negative integer".to_string());
    }
    Ok(())
}

fn require_string_array(
    object: &serde_json::Map<String, serde_json::Value>,
    field: &str,
) -> Result<(), String> {
    let values = object[field]
        .as_array()
        .ok_or_else(|| format!("{field} must be an array"))?;
    if values
        .iter()
        .any(|value| value.as_str().is_none_or(str::is_empty))
    {
        return Err(format!("{field} must contain only non-empty strings"));
    }
    Ok(())
}

fn validate_followup(followup: &serde_json::Value) -> Result<(), String> {
    let object = followup
        .as_object()
        .ok_or_else(|| "followup must be an object".to_string())?;
    let expected = [
        "kind",
        "priority",
        "title",
        "rationale",
        "perspective",
        "evidence_ids",
    ];
    if object.len() != expected.len() || expected.iter().any(|field| !object.contains_key(*field)) {
        return Err("followup fields do not match the envelope schema".to_string());
    }
    require_enum(object, "kind", &["todo", "idea", "retest"])?;
    require_enum(object, "priority", &["high", "medium", "low"])?;
    for field in ["title", "rationale"] {
        if object[field].as_str().is_none_or(str::is_empty) {
            return Err(format!("followup {field} must be a non-empty string"));
        }
    }
    if !(object["perspective"].is_null() || object["perspective"].is_string()) {
        return Err("followup perspective must be null or a string".to_string());
    }
    require_string_array(object, "evidence_ids")
}

fn require_enum(
    object: &serde_json::Map<String, serde_json::Value>,
    field: &str,
    allowed: &[&str],
) -> Result<(), String> {
    let value = object[field]
        .as_str()
        .ok_or_else(|| format!("{field} must be a string"))?;
    if allowed.contains(&value) {
        Ok(())
    } else {
        Err(format!("{field} has invalid value {value}"))
    }
}

fn append_json_line(path: &Path, record: &serde_json::Value) {
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

pub fn assert_trace_tokens_bounded(trace_events: &[serde_json::Value]) {
    // Mirrors eval lens: blowout without compaction signal is a harness miss.
    let mut max_input: u64 = 0;
    let mut has_compaction = false;
    for e in trace_events {
        if e["event"] == "turn_end" {
            max_input = max_input.max(e["input_tokens"].as_u64().unwrap_or(0));
        }
        if e["event"] == "compaction_applied" {
            has_compaction = true;
        }
    }
    if max_input > 1_000_000 && !has_compaction {
        panic!("token blowout: input_tokens={max_input} without compaction_applied — cap or compact must fire (see live transcript 2006149↑)");
    }
}

#[cfg(test)]
mod ledger_token_tests {
    use super::*;

    #[test]
    fn blowout_without_compaction_fails() {
        let events = vec![
            serde_json::json!({"event":"turn_end","input_tokens":2_006_149_u64,"iterations":20_u64}),
        ];
        let r = std::panic::catch_unwind(|| assert_trace_tokens_bounded(&events));
        assert!(r.is_err(), "should panic on blowout without compaction");
    }

    #[test]
    fn blowout_with_compaction_passes() {
        let events = vec![
            serde_json::json!({"event":"compaction_applied","before_messages":20_u64,"after_messages":10_u64}),
            serde_json::json!({"event":"turn_end","input_tokens":2_006_149_u64,"iterations":20_u64}),
        ];
        assert_trace_tokens_bounded(&events);
    }

    #[test]
    fn modest_tokens_pass_without_compaction() {
        let events = vec![
            serde_json::json!({"event":"turn_end","input_tokens":80_000_u64,"iterations":2_u64}),
        ];
        assert_trace_tokens_bounded(&events);
    }
}
