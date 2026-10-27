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
typeset -g _piku_ledger_cmd="" _piku_ledger_cwd=""
_piku_ledger_preexec() { _piku_ledger_cmd="$1"; _piku_ledger_cwd="$PWD"; }
_piku_ledger_precmd() {
  local piku_exit=$?
  [[ -n "$_piku_ledger_cmd" ]] || return 0
  command piku ledger-append --exit "$piku_exit" --cwd "$_piku_ledger_cwd" -- "$_piku_ledger_cmd" &!
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
    let at = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_secs());
    Some(LedgerEntry {
        at,
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
    let line = serde_json::to_string(entry).map_err(|error| error.to_string())?;
    writeln!(file, "{line}").map_err(|error| format!("write {}: {error}", path.display()))
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
