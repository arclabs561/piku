#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::items_after_statements,
    clippy::format_push_string,
    clippy::match_same_arms
)]

pub mod cli;
pub mod config;
pub mod input_helper;
pub mod markdown;
pub mod run_view;
/// Public library surface — used by integration tests and main.rs.
pub mod self_update;
pub mod trace;
pub mod tui_repl;

use std::env;

/// Crate version, from `Cargo.toml`. Shared by the binary, `--version`, and the
/// interactive `/version` command so all three report the same string.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

// ---------------------------------------------------------------------------
// Shared utilities
// ---------------------------------------------------------------------------

#[must_use]
pub fn read_only_system_prompt_section() -> String {
    "# Read-only mode\n\n\
- Inspect files only. Do not edit files, write files, run shell commands, spawn agents, or change memory.\n\
- If a requested task requires changes, describe the exact change instead of making it.\n\
- Do not claim that you changed, fixed, wrote, or ran anything."
        .to_string()
}

pub fn sessions_dir() -> anyhow::Result<std::path::PathBuf> {
    let base = env::var("XDG_CONFIG_HOME").map_or_else(
        |_| {
            env::var("HOME").map_or_else(
                |_| std::path::PathBuf::from(".config"),
                |h| std::path::PathBuf::from(h).join(".config"),
            )
        },
        std::path::PathBuf::from,
    );
    let path = base.join("piku").join("sessions");
    std::fs::create_dir_all(&path)?;
    Ok(path)
}

#[must_use]
pub fn new_session_id() -> String {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    let pid = std::process::id();
    format!("session-{nanos}-{pid}")
}

#[must_use]
pub fn current_date() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());

    let days = secs / 86400;
    let mut year = 1970u32;
    let mut remaining_days = u32::try_from(days).unwrap_or(u32::MAX);
    loop {
        let days_in_year = if is_leap_year(year) { 366 } else { 365 };
        if remaining_days < days_in_year {
            break;
        }
        remaining_days -= days_in_year;
        year += 1;
    }

    let month_days: [u32; 12] = [
        31,
        if is_leap_year(year) { 29 } else { 28 },
        31,
        30,
        31,
        30,
        31,
        31,
        30,
        31,
        30,
        31,
    ];
    let mut month = 1u32;
    for &md in &month_days {
        if remaining_days < md {
            break;
        }
        remaining_days -= md;
        month += 1;
    }
    let day = remaining_days + 1;

    format!("{year:04}-{month:02}-{day:02}")
}

fn is_leap_year(y: u32) -> bool {
    y.is_multiple_of(4) && !y.is_multiple_of(100) || y.is_multiple_of(400)
}

// ---------------------------------------------------------------------------
// Tool input formatting — shared by all sinks
// ---------------------------------------------------------------------------

/// Format the key argument(s) of a tool call for display.
/// Returns a short, human-readable string (no colour codes) or empty string
/// if there is nothing useful to show.
pub fn format_tool_input(tool_name: &str, input: &serde_json::Value) -> String {
    let get_str = |key: &str| -> Option<&str> { input.get(key)?.as_str() };

    match tool_name {
        "read_file" => {
            let path = shorten_path(get_str("path").unwrap_or(""));
            let start = input.get("start_line").and_then(serde_json::Value::as_u64);
            let end = input.get("end_line").and_then(serde_json::Value::as_u64);
            match (start, end) {
                (Some(s), Some(e)) => format!("{path}:{s}-{e}"),
                (Some(s), None) => format!("{path}:{s}-"),
                _ => path.clone(),
            }
        }
        "write_file" | "list_dir" => shorten_path(get_str("path").unwrap_or("")).clone(),
        "edit_file" => {
            let path = shorten_path(get_str("path").unwrap_or(""));
            path.clone()
        }
        "bash" => {
            let cmd = get_str("command").unwrap_or("");
            let line_count = cmd.lines().count();
            let first = cmd.lines().next().unwrap_or("");
            if line_count > 1 {
                // Multiline: show first line + line count
                format!("{} (+{} lines)", truncate_arg(first, 50), line_count - 1)
            } else {
                truncate_arg(first, 72)
            }
        }
        "glob" | "grep" => {
            let pattern = get_str("pattern").unwrap_or("");
            let path = get_str("path").unwrap_or("");
            if path.is_empty() {
                truncate_arg(pattern, 60)
            } else {
                format!("{} in {}", truncate_arg(pattern, 40), shorten_path(path))
            }
        }
        _ => {
            // Generic: show first string-valued key
            if let Some(obj) = input.as_object() {
                for (_, v) in obj {
                    if let Some(s) = v.as_str() {
                        return truncate_arg(s, 60);
                    }
                }
            }
            String::new()
        }
    }
}

/// Shorten a file path for display. If the path has more than 3 components,
/// show …/`last_two_components`. Keeps paths readable in tool headers.
fn shorten_path(path: &str) -> String {
    let parts: Vec<&str> = path.split('/').filter(|s| !s.is_empty()).collect();
    if parts.len() <= 3 {
        return path.to_string();
    }
    format!("…/{}", parts[parts.len() - 2..].join("/"))
}

/// Format a duration compactly: 45s, 2m 30s, 1h 5m 0s.
/// Matches Claude Code's formatDuration pattern.
#[must_use]
pub fn fmt_duration(secs: u64) -> String {
    if secs < 60 {
        format!("{secs}s")
    } else if secs < 3600 {
        let m = secs / 60;
        let s = secs % 60;
        if s == 0 {
            format!("{m}m")
        } else {
            format!("{m}m {s:02}s")
        }
    } else {
        let h = secs / 3600;
        let m = (secs % 3600) / 60;
        let s = secs % 60;
        format!("{h}h {m:02}m {s:02}s")
    }
}

/// Try to pretty-print a JSON string. Returns the original if not valid JSON
/// or too large. Cap at 10k chars to avoid pathological inputs (Claude Code pattern).
#[must_use]
pub fn try_pretty_json(s: &str) -> String {
    const MAX_JSON_LEN: usize = 10_000;
    if s.len() > MAX_JSON_LEN {
        return s.to_string();
    }
    // Only attempt if it looks like JSON
    let trimmed = s.trim();
    if !trimmed.starts_with('{') && !trimmed.starts_with('[') {
        return s.to_string();
    }
    match serde_json::from_str::<serde_json::Value>(trimmed) {
        Ok(val) => serde_json::to_string_pretty(&val).unwrap_or_else(|_| s.to_string()),
        Err(_) => s.to_string(),
    }
}

/// Longest prefix of `s` that is at most `max` bytes and ends on a character
/// boundary.
///
/// `&s[..max]` panics when `max` lands inside a multi-byte character, and
/// everything truncated here is model or tool output, so an em-dash, a CJK
/// character, or an emoji at the wrong offset was a crash rather than a
/// display glitch. Bytes stay the unit because these are size guards, not
/// column budgets.
#[must_use]
pub fn truncate_on_char_boundary(s: &str, max: usize) -> &str {
    if s.len() <= max {
        return s;
    }
    let mut end = max;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

fn truncate_arg(s: &str, max: usize) -> String {
    let first_line = s.lines().next().unwrap_or(s);
    if first_line.len() <= max {
        first_line.to_string()
    } else {
        format!("{}…", truncate_on_char_boundary(first_line, max))
    }
}
#[cfg(test)]
mod truncation_tests {
    use super::truncate_on_char_boundary;

    #[test]
    fn a_multibyte_character_at_the_limit_does_not_panic() {
        // Every site that truncates here handles model or tool output, so the
        // character at the cut is routinely an em-dash, a CJK character, or an
        // emoji. `&s[..max]` panicked on each of them.
        for filler in ["—", "日", "🦀", "é"] {
            for max in 1..12 {
                let text = "ab".to_string() + &filler.repeat(6);
                let cut = truncate_on_char_boundary(&text, max);
                assert!(cut.len() <= max, "{filler} at {max}: {cut:?}");
                assert!(text.starts_with(cut), "{filler} at {max}: {cut:?}");
            }
        }
    }

    #[test]
    fn a_short_string_is_returned_whole() {
        assert_eq!(truncate_on_char_boundary("short", 200), "short");
        assert_eq!(truncate_on_char_boundary("", 10), "");
    }

    #[test]
    fn a_cut_inside_the_first_character_yields_nothing_rather_than_panicking() {
        // The first character is four bytes and the budget is one, so there is
        // no valid prefix. Returning empty is correct; panicking is not.
        assert_eq!(truncate_on_char_boundary("🦀tail", 1), "");
    }
}
