//! Versioned session state for the self-exec upgrade (docs/live-update.md).
//!
//! The blob holds version, working directory, line buffer, the cursor as a char
//! index, and the job table. Version 2 adds the jobs; children survive `exec`,
//! so their pids, process groups, commands, and running/stopped state must
//! survive with them. It is written with a private temporary file and renamed
//! into place, so a reader never sees a partial state.

use serde::{Deserialize, Serialize};
use std::fs::{self, OpenOptions};
use std::io::{self, Read, Write};
use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
use std::path::{Path, PathBuf};

/// Bumped whenever the blob shape changes incompatibly.
pub const VERSION: u32 = 2;

const MAX_BLOB_BYTES: u64 = 1024 * 1024;
const MAX_JOBS: usize = 1024;
const MAX_JOB_COMMAND: usize = 4096;

/// The exact stdout a candidate must produce for `--self-check <blob>`: the
/// blob version it read, and nothing else that matters. An exit status alone
/// proves nothing, because any program can exit 0.
pub fn self_check_reply() -> String {
    format!("ok {VERSION}")
}

/// Whether `output` is the reply a passing `--self-check` produces. Trailing or
/// leading whitespace from the candidate's own newline is ignored; any other
/// word, an empty reply, or a different version is a failed check.
pub fn confirms_blob(output: &str) -> bool {
    output.trim() == self_check_reply()
}

/// Running or stopped, as the blob spells it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum JobState {
    Running,
    Stopped,
}

/// One job as it crosses `exec`: the shell's job number is not part of the
/// identity, but the process group, its members, and the command are.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JobRecord {
    pub pgid: i32,
    pub pids: Vec<i32>,
    pub command: String,
    pub state: JobState,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct State {
    pub version: u32,
    pub cwd: String,
    pub buffer: String,
    /// Char index into `buffer`, not a byte offset.
    pub cursor: usize,
    /// Jobs that were alive when the blob was written.
    pub jobs: Vec<JobRecord>,
}

impl State {
    /// Captures the current directory, a line, and the live job table.
    pub fn capture(buffer: String, cursor: usize, jobs: Vec<JobRecord>) -> Result<Self, String> {
        let cwd = std::env::current_dir()
            .map_err(|error| format!("current directory: {error}"))?
            .to_string_lossy()
            .into_owned();
        let state = Self {
            version: VERSION,
            cwd,
            buffer,
            cursor,
            jobs,
        };
        state.validate()?;
        Ok(state)
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.version != VERSION {
            return Err(format!(
                "unsupported state version {} (this build writes {VERSION})",
                self.version
            ));
        }
        if self.cwd.is_empty() || !Path::new(&self.cwd).is_absolute() {
            return Err("saved cwd is not an absolute path".into());
        }
        if self.buffer.len() as u64 > MAX_BLOB_BYTES {
            return Err("saved buffer is too large".into());
        }
        let chars = self.buffer.chars().count();
        if self.cursor > chars {
            return Err(format!(
                "saved cursor {} is past the end of the {chars}-char buffer",
                self.cursor
            ));
        }
        if self.jobs.len() > MAX_JOBS {
            return Err(format!("saved job table has more than {MAX_JOBS} jobs"));
        }
        for job in &self.jobs {
            if job.pgid <= 0 {
                return Err("saved job has a non-positive process group".into());
            }
            if job.pids.is_empty() {
                return Err(format!("saved job {} has no processes", job.pgid));
            }
            if job.pids.iter().any(|pid| *pid <= 0) {
                return Err(format!("saved job {} names an invalid pid", job.pgid));
            }
            if job.command.len() > MAX_JOB_COMMAND {
                return Err(format!("saved job {} has an oversized command", job.pgid));
            }
        }
        Ok(())
    }
}

/// Default blob location: the session's runtime dir, else `/tmp`.
pub fn blob_path() -> PathBuf {
    let dir = std::env::var_os("XDG_RUNTIME_DIR")
        .filter(|dir| !dir.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/tmp"));
    dir.join(format!("piku-shell-state-{}.json", std::process::id()))
}

/// Writes `state` atomically: temp file with mode 0600, fsync, rename.
pub fn write(path: &Path, state: &State) -> Result<(), String> {
    state.validate()?;
    let directory = match path.parent() {
        Some(parent) if !parent.as_os_str().is_empty() => parent,
        _ => Path::new("."),
    };
    let temporary = directory.join(format!(
        ".piku-shell-state-{}-{}.tmp",
        std::process::id(),
        monotonic_nanos()
    ));
    let bytes = serde_json::to_vec(state).map_err(|error| format!("encode state: {error}"))?;

    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(&temporary)
        .map_err(|error| format!("create {}: {error}", temporary.display()))?;
    let written = file
        .write_all(&bytes)
        .and_then(|()| file.write_all(b"\n"))
        .and_then(|()| file.sync_all());
    drop(file);
    if let Err(error) = written {
        let _ = fs::remove_file(&temporary);
        return Err(format!("write {}: {error}", temporary.display()));
    }
    if let Err(error) = fs::rename(&temporary, path) {
        let _ = fs::remove_file(&temporary);
        return Err(format!("rename to {}: {error}", path.display()));
    }
    // Persist the directory entry where the filesystem supports it.
    if let Ok(directory) = fs::File::open(directory) {
        let _ = directory.sync_all();
    }
    Ok(())
}

/// Reads and validates a blob without removing it (`--self-check`).
pub fn load(path: &Path) -> Result<State, String> {
    let bytes = read_bounded(path)?;
    let state = parse(&bytes, path)?;
    state.validate()?;
    Ok(state)
}

fn read_bounded(path: &Path) -> Result<Vec<u8>, String> {
    let file = fs::File::open(path).map_err(|error| format!("open {}: {error}", path.display()))?;
    let mut bytes = Vec::new();
    file.take(MAX_BLOB_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| format!("read {}: {error}", path.display()))?;
    if bytes.len() as u64 > MAX_BLOB_BYTES {
        return Err(format!(
            "{} is larger than {MAX_BLOB_BYTES} bytes",
            path.display()
        ));
    }
    Ok(bytes)
}

fn parse(bytes: &[u8], path: &Path) -> Result<State, String> {
    serde_json::from_slice(bytes).map_err(|error| format!("parse {}: {error}", path.display()))
}

/// Replaces this process with the current executable, handing it the blob.
/// Returns only on failure.
pub fn exec_self(path: &Path) -> io::Error {
    let executable = match std::env::current_exe() {
        Ok(executable) => executable,
        Err(error) => return error,
    };
    exec_state(path, &executable)
}

/// Replaces this process with `program`, which must be given the blob with
/// `--restore`. Returns only on failure.
pub fn exec_state(path: &Path, program: &Path) -> io::Error {
    use std::os::unix::process::CommandExt;
    // exec() keeps the pid, session, fds, and terminal ownership: this is a
    // replacement, not a child process.
    std::process::Command::new(program)
        .arg("--restore")
        .arg(path)
        .exec()
}

/// Replaces the running executable with `candidate` and keeps the previous
/// binary as `<exe>.old` for manual rollback. The install is a temp file in the
/// executable's directory renamed over the target, so a reader never sees a
/// partially written binary.
///
/// The candidate must already have passed `--self-check`, which means it exited
/// 0 *and* its stdout was exactly `ok <VERSION>` (see `self_check_reply`): an
/// exit status alone would let `:upgrade /usr/bin/true` replace the shell.
pub fn install_binary(candidate: &Path) -> Result<PathBuf, String> {
    let current = std::env::current_exe()
        .map_err(|error| format!("current executable: {error}"))?
        .canonicalize()
        .map_err(|error| format!("current executable: {error}"))?;
    let candidate = candidate
        .canonicalize()
        .map_err(|error| format!("{}: {error}", candidate.display()))?;
    let metadata =
        fs::metadata(&candidate).map_err(|error| format!("{}: {error}", candidate.display()))?;
    if !metadata.is_file() {
        return Err(format!("{}: not a file", candidate.display()));
    }
    if metadata.permissions().mode() & 0o111 == 0 {
        return Err(format!("{}: not executable", candidate.display()));
    }
    if candidate == current {
        return Ok(current);
    }
    let directory = current
        .parent()
        .ok_or_else(|| format!("{}: no parent directory", current.display()))?;
    let backup = with_suffix(&current, ".old");
    fs::copy(&current, &backup).map_err(|error| format!("keep {}: {error}", backup.display()))?;

    let temporary = directory.join(format!(
        ".piku-shell-install-{}-{}.tmp",
        std::process::id(),
        monotonic_nanos()
    ));
    let installed = fs::copy(&candidate, &temporary)
        .map_err(|error| format!("copy {}: {error}", candidate.display()))
        .and_then(|_| {
            fs::rename(&temporary, &current)
                .map_err(|error| format!("rename onto {}: {error}", current.display()))
        });
    if let Err(error) = installed {
        let _ = fs::remove_file(&temporary);
        return Err(error);
    }
    // Persist the directory entry where the filesystem supports it.
    if let Ok(directory) = fs::File::open(directory) {
        let _ = directory.sync_all();
    }
    Ok(current)
}

/// Appends `suffix` to the whole file name, so `piku-shell` becomes
/// `piku-shell.old` rather than `piku.shell.old` losing its extension.
fn with_suffix(path: &Path, suffix: &str) -> PathBuf {
    let mut name = path.as_os_str().to_os_string();
    name.push(suffix);
    PathBuf::from(name)
}

fn monotonic_nanos() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_nanos())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;

    fn state(buffer: &str, cursor: usize) -> State {
        State {
            version: VERSION,
            cwd: "/tmp".into(),
            buffer: buffer.into(),
            cursor,
            jobs: Vec::new(),
        }
    }

    fn job(command: &str, state: JobState) -> JobRecord {
        JobRecord {
            pgid: 1234,
            pids: vec![1234],
            command: command.into(),
            state,
        }
    }

    #[test]
    fn only_the_exact_ok_reply_confirms_a_blob() {
        assert_eq!(self_check_reply(), format!("ok {VERSION}"));
        assert!(confirms_blob(&self_check_reply()));
        assert!(confirms_blob(&format!("  ok {VERSION}\n")));
        // An exit-0 program that says nothing has not checked anything.
        assert!(!confirms_blob(""));
        assert!(!confirms_blob("\n"));
        assert!(!confirms_blob("ok"));
        assert!(!confirms_blob(&format!("ok {}", VERSION + 1)));
        assert!(!confirms_blob(&format!("ok {VERSION}\nextra")));
        assert!(!confirms_blob("piku shell: restored"));
    }

    #[test]
    fn write_is_private_and_load_keeps_the_blob() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("state.json");
        let original = state("echo héllo", 5);
        write(&path, &original).unwrap();

        let mode = fs::metadata(&path).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o600, "state blob must not be group/world readable");
        let loaded = load(&path).unwrap();
        assert_eq!(loaded.cwd, original.cwd);
        assert_eq!(loaded.buffer, original.buffer);
        assert_eq!(loaded.cursor, original.cursor);
        assert!(path.exists(), "load must not consume the blob");
    }

    #[test]
    fn load_leaves_a_valid_blob_and_rejects_a_wrong_version() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("state.json");

        write(&path, &state("echo", 4)).unwrap();
        assert_eq!(load(&path).unwrap().buffer, "echo");
        assert!(path.exists(), "load must not consume the blob");

        let mut wrong_version = state("echo", 0);
        wrong_version.version = VERSION + 1;
        fs::write(&path, serde_json::to_vec(&wrong_version).unwrap()).unwrap();
        assert!(load(&path).is_err());
    }

    #[test]
    fn validation_rejects_impossible_state() {
        let mut past_end = state("abc", 4);
        assert!(past_end.validate().is_err());
        past_end.cursor = 2;
        assert!(past_end.validate().is_ok());

        let mut relative = state("", 0);
        relative.cwd = "relative/path".into();
        assert!(relative.validate().is_err());

        let bad_version = State {
            version: VERSION + 1,
            ..state("", 0)
        };
        assert!(bad_version.validate().is_err());
    }

    #[test]
    fn job_table_survives_a_write_load_round_trip() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("state.json");
        let mut state = state("echo jobs", 9);
        state.jobs = vec![
            job("sleep 30", JobState::Running),
            JobRecord {
                pgid: 5678,
                pids: vec![5678, 5679],
                command: "cat | tr a-z A-Z".into(),
                state: JobState::Stopped,
            },
        ];
        write(&path, &state).unwrap();

        let loaded = load(&path).unwrap();
        assert_eq!(loaded.jobs, state.jobs);
        // The state is plain JSON with the documented field names.
        let text = fs::read_to_string(&path).unwrap();
        assert!(text.contains("\"jobs\""));
        assert!(text.contains("\"pgid\""));
        assert!(text.contains("\"state\":\"stopped\""));
    }

    #[test]
    fn validation_rejects_impossible_job_tables() {
        let mut bad_pgid = state("", 0);
        bad_pgid.jobs = vec![JobRecord {
            pgid: 0,
            ..job("job", JobState::Running)
        }];
        assert!(bad_pgid.validate().is_err());

        let mut no_pids = state("", 0);
        no_pids.jobs = vec![JobRecord {
            pids: Vec::new(),
            ..job("job", JobState::Running)
        }];
        assert!(no_pids.validate().is_err());

        let mut bad_pid = state("", 0);
        bad_pid.jobs = vec![JobRecord {
            pids: vec![-1],
            ..job("job", JobState::Running)
        }];
        assert!(bad_pid.validate().is_err());

        let mut too_many = state("", 0);
        too_many.jobs = vec![job("job", JobState::Running); MAX_JOBS + 1];
        assert!(too_many.validate().is_err());
    }

    #[test]
    fn a_job_record_rejects_unknown_fields() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("state.json");
        let mut state = state("", 0);
        state.jobs = vec![job("sleep 1", JobState::Running)];
        let mut value = serde_json::to_value(&state).unwrap();
        value["jobs"][0]["surprise"] = serde_json::json!(true);
        fs::write(&path, serde_json::to_vec(&value).unwrap()).unwrap();
        assert!(load(&path).is_err());
    }

    #[test]
    fn write_refuses_invalid_state_without_creating_a_file() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("state.json");
        let bad = State {
            version: VERSION + 1,
            ..state("", 0)
        };
        assert!(write(&path, &bad).is_err());
        assert!(!path.exists());
        assert_eq!(
            fs::read_dir(directory.path()).unwrap().count(),
            0,
            "no temporary file may be left behind"
        );
    }
}
