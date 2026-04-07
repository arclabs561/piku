pub mod cli;
pub mod input_helper;
pub mod markdown;
pub mod repl;
/// Public library surface — used by integration tests and main.rs.
pub mod self_update;
pub mod trace;
pub mod tui_repl;

use std::env;

// ---------------------------------------------------------------------------
// Shared utilities (used by both main.rs and repl.rs)
// ---------------------------------------------------------------------------

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
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let pid = std::process::id();
    format!("session-{nanos}-{pid}")
}

#[must_use]
pub fn current_date() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

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
                _ => path.to_string(),
            }
        }
        "write_file" | "list_dir" => shorten_path(get_str("path").unwrap_or("")).to_string(),
        "edit_file" => {
            let path = shorten_path(get_str("path").unwrap_or(""));
            path.to_string()
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
                for (_, v) in obj.iter().take(1) {
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
/// show …/last_two_components. Keeps paths readable in tool headers.
fn shorten_path(path: &str) -> String {
    let parts: Vec<&str> = path.split('/').filter(|s| !s.is_empty()).collect();
    if parts.len() <= 3 {
        return path.to_string();
    }
    format!("…/{}", parts[parts.len() - 2..].join("/"))
}

fn truncate_arg(s: &str, max: usize) -> String {
    let first_line = s.lines().next().unwrap_or(s);
    if first_line.len() <= max {
        first_line.to_string()
    } else {
        format!("{}…", &first_line[..max])
    }
}
