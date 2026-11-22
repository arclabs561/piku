//! Shell command ledger: one JSON line per command, shared by `piku shell` and
//! `piku <prompt> --shell-context`.
//!
//! The file lives at `${XDG_STATE_HOME:-$HOME/.local/state}/piku/shell-ledger.jsonl`
//! and is created owner-only. It records command lines, never output. A line
//! typed with a leading space is not recorded, and credential-looking values
//! are redacted before writing; the redaction narrows exposure, it does not
//! remove it.

use std::fs::{self, OpenOptions};
use std::io::Write;
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LedgerEntry {
    /// Seconds since the Unix epoch.
    pub at: u64,
    /// When the command started, in microseconds. Background writers can
    /// append out of order, so readers sort by this; older records lack it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub at_us: Option<u64>,
    pub cwd: String,
    pub command: String,
    pub exit: i32,
    /// Which writer recorded the line, e.g. `piku-shell`.
    pub source: String,
}

/// zsh integration printed by `piku init zsh`. `preexec` keeps the line as
/// typed (so a leading space still marks it private); `precmd` reads `$?`
/// first, then appends in the background so the prompt never waits on piku.
pub const ZSH_HOOK: &str = r#"# piku shell ledger: records command lines (never output) for
# `piku --shell-context`. Lines typed with a leading space are skipped.
zmodload -F zsh/datetime p:EPOCHREALTIME 2>/dev/null
typeset -g _piku_ledger_cmd="" _piku_ledger_cwd="" _piku_ledger_us=""
_piku_ledger_preexec() {
  _piku_ledger_cmd="$1"; _piku_ledger_cwd="$PWD"
  _piku_ledger_us="${${EPOCHREALTIME/./}[1,16]}"
}
_piku_ledger_precmd() {
  local piku_exit=$?
  [[ -n "$_piku_ledger_cmd" ]] || return 0
  local -a piku_started=()
  [[ -n "$_piku_ledger_us" ]] && piku_started=(--started-us "$_piku_ledger_us")
  command piku ledger-append --exit "$piku_exit" --cwd "$_piku_ledger_cwd" \
    "${piku_started[@]}" -- "$_piku_ledger_cmd" &!
  _piku_ledger_cmd=""
}
autoload -Uz add-zsh-hook
add-zsh-hook preexec _piku_ledger_preexec
add-zsh-hook precmd _piku_ledger_precmd
"#;

/// Records read back for `--shell-context` stop at this many bytes from the
/// end of the file, so an old, large ledger cannot slow a launch.
const TAIL_BYTES: u64 = 256 * 1024;

#[must_use]
pub fn ledger_path() -> Option<PathBuf> {
    let base = match std::env::var_os("XDG_STATE_HOME") {
        Some(dir) if !dir.is_empty() => PathBuf::from(dir),
        _ => {
            let home = std::env::var_os("HOME").filter(|home| !home.is_empty())?;
            PathBuf::from(home).join(".local/state")
        }
    };
    Some(base.join("piku/shell-ledger.jsonl"))
}

/// Build the record for a command that just ran, or `None` when the line must
/// not be recorded (blank, or typed with a leading space).
#[must_use]
pub fn entry(command: &str, cwd: &Path, exit: i32, source: &str) -> Option<LedgerEntry> {
    if command.starts_with(' ') || command.trim().is_empty() {
        return None;
    }
    let at_us = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |elapsed| {
            u64::try_from(elapsed.as_micros()).unwrap_or(u64::MAX)
        });
    Some(LedgerEntry {
        at: at_us / 1_000_000,
        at_us: Some(at_us),
        cwd: cwd.to_string_lossy().into_owned(),
        command: redact(command.trim()),
        exit,
        source: source.to_string(),
    })
}

pub fn append(path: &Path, entry: &LedgerEntry) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|error| format!("create {}: {error}", parent.display()))?;
    }
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .mode(0o600)
        .open(path)
        .map_err(|error| format!("open {}: {error}", path.display()))?;
    // One write per record: with O_APPEND a single small write lands whole,
    // while writeln! issues the record and the newline separately, and the zsh
    // hook runs appenders concurrently in the background.
    let mut line = serde_json::to_string(entry).map_err(|error| error.to_string())?;
    line.push('\n');
    file.write_all(line.as_bytes())
        .map_err(|error| format!("write {}: {error}", path.display()))
}

/// The last `limit` records, oldest first. Unparseable lines are skipped.
pub fn recent(path: &Path, limit: usize) -> Result<Vec<LedgerEntry>, String> {
    use std::io::{Read, Seek, SeekFrom};
    let mut file = match fs::File::open(path) {
        Ok(file) => file,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) => return Err(format!("open {}: {error}", path.display())),
    };
    let len = file.metadata().map_or(0, |meta| meta.len());
    let start = len.saturating_sub(TAIL_BYTES);
    file.seek(SeekFrom::Start(start))
        .map_err(|error| format!("read {}: {error}", path.display()))?;
    let mut text = String::new();
    file.read_to_string(&mut text)
        .map_err(|error| format!("read {}: {error}", path.display()))?;
    // A tail that starts mid-file starts mid-line; drop that partial line.
    let body = if start > 0 {
        text.split_once('\n').map_or("", |(_, rest)| rest)
    } else {
        &text
    };
    let mut entries: Vec<LedgerEntry> = body
        .lines()
        .filter_map(|line| serde_json::from_str(line).ok())
        .collect();
    entries.sort_by_key(|entry| entry.at_us.unwrap_or(entry.at.saturating_mul(1_000_000)));
    let skip = entries.len().saturating_sub(limit);
    Ok(entries.split_off(skip))
}

/// Replace the value of credential-looking `NAME=value` words and
/// `--name value` / `--name=value` flags with `<redacted>`.
#[must_use]
pub fn redact(command: &str) -> String {
    let mut out: Vec<String> = Vec::new();
    let mut redact_next = false;
    for word in command.split(' ') {
        if redact_next && !word.is_empty() {
            out.push("<redacted>".to_string());
            redact_next = false;
            continue;
        }
        let flag = word.trim_start_matches('-');
        let is_flag = flag.len() < word.len();
        if let Some((name, _)) = word.split_once('=') {
            if sensitive(name.trim_start_matches('-')) {
                out.push(format!("{name}=<redacted>"));
                continue;
            }
        } else if is_flag && sensitive(flag) {
            redact_next = true;
        }
        out.push(word.to_string());
    }
    out.join(" ")
}

fn sensitive(name: &str) -> bool {
    let name = name.to_ascii_lowercase();
    ["token", "secret", "pass", "key", "auth"]
        .iter()
        .any(|needle| name.contains(needle))
}

/// Render records as model context. Plain lines so the model and the operator
/// read the same thing.
#[must_use]
pub fn render(entries: &[LedgerEntry]) -> String {
    let mut text =
        String::from("Recent shell commands (oldest first, from the piku shell ledger):\n");
    for entry in entries {
        text.push_str(&format!(
            "[{}] $ {}  (exit {})\n",
            entry.cwd, entry.command, entry.exit
        ));
    }
    text
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn leading_space_and_blank_lines_are_not_recorded() {
        let cwd = Path::new("/tmp");
        assert!(entry(" export SECRET=x", cwd, 0, "t").is_none());
        assert!(entry("   ", cwd, 0, "t").is_none());
        assert_eq!(entry("ls -la", cwd, 2, "t").unwrap().command, "ls -la");
    }

    #[test]
    fn redact_hides_credential_assignments_and_flag_values_only() {
        assert_eq!(
            redact("GITHUB_TOKEN=ghp_abc gh api user"),
            "GITHUB_TOKEN=<redacted> gh api user"
        );
        assert_eq!(
            redact("mysql --password hunter2 -u root"),
            "mysql --password <redacted> -u root"
        );
        assert_eq!(redact("curl --api-key=k1 x"), "curl --api-key=<redacted> x");
        assert_eq!(
            redact("cargo test --release FOO=1"),
            "cargo test --release FOO=1"
        );
    }

    #[test]
    fn recent_orders_by_start_time_not_append_order() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("shell-ledger.jsonl");
        for (command, ms) in [
            ("second", 2_000_000),
            ("first", 1_000_001),
            ("third", 3_000_000),
        ] {
            let mut record = entry(command, Path::new("/w"), 0, "t").unwrap();
            record.at_us = Some(ms);
            append(&path, &record).unwrap();
        }
        let order: Vec<String> = recent(&path, 3)
            .unwrap()
            .into_iter()
            .map(|e| e.command)
            .collect();
        assert_eq!(order, ["first", "second", "third"]);
    }

    #[test]
    fn concurrent_appends_never_share_a_line() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("shell-ledger.jsonl");
        let threads: Vec<_> = (0..8)
            .map(|t| {
                let path = path.clone();
                std::thread::spawn(move || {
                    for n in 0..50 {
                        let record =
                            entry(&format!("echo {t}-{n}"), Path::new("/w"), 0, "t").unwrap();
                        append(&path, &record).unwrap();
                    }
                })
            })
            .collect();
        for thread in threads {
            thread.join().unwrap();
        }
        let text = fs::read_to_string(&path).unwrap();
        assert_eq!(text.lines().count(), 400);
        for line in text.lines() {
            serde_json::from_str::<LedgerEntry>(line).unwrap_or_else(|e| panic!("{e}: {line}"));
        }
    }

    #[test]
    fn append_is_owner_only_and_recent_returns_the_tail_in_order() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("state/piku/shell-ledger.jsonl");
        for n in 0..5 {
            let record = entry(&format!("echo {n}"), dir.path(), n, "t").unwrap();
            append(&path, &record).unwrap();
        }
        fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap()
            .write_all(b"not json\n")
            .unwrap();
        let mode = fs::metadata(&path).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o600);
        let tail = recent(&path, 2).unwrap();
        let commands: Vec<&str> = tail.iter().map(|e| e.command.as_str()).collect();
        assert_eq!(commands, ["echo 3", "echo 4"]);
        assert_eq!(tail[1].exit, 4);
        assert!(
            recent(&dir.path().join("missing.jsonl"), 3)
                .unwrap()
                .is_empty()
        );
    }
}
