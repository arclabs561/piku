//! piku shell: the interactive shell absorbed from the `wish` project.
//!
//! Line editing, completion, highlighting, history, prompt, job control, and
//! session restore are owned here. Command lines are parsed, expanded, and run
//! by `syntax`, `expand`, `exec`, and `builtins`, without delegating to `sh`.
//!
//! Absorbed from the standalone `wish` shell prototype on 2026-10-26. The
//! self-exec upgrade path is not available here: piku owns the binary it
//! ships, so `piku shell` refuses `:upgrade` and Ctrl-X Ctrl-U rather than
//! re-execing an arbitrary file.

mod builtins;
mod complete;
mod editor;
mod exec;
mod expand;
mod handoff;
mod highlight;
mod history;
mod input;
mod jobs;
pub mod ledger;
mod signals;
mod syntax;
mod term;

use std::ffi::OsString;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::process::{Command as Process, Stdio};
use std::time::{Duration, Instant};

use builtins::Shell;
use editor::LineEditor;
use history::History;
use input::{Input, Key};

/// Names the shell answers itself, offered by Tab completion.
const COMPLETIONS: [&str; 12] = [
    "cd", "pwd", "exit", ":q", ":upgrade", "jobs", "fg", "bg", "wait", "kill", "export", "unset",
];

/// Runs the standalone `piku-shell` binary, returning its exit code.
#[must_use]
pub fn run_cli() -> i32 {
    match run(std::env::args_os().skip(1), "piku-shell", true) {
        Ok(code) => code,
        Err(error) => {
            eprintln!("piku-shell: {error}");
            1
        }
    }
}

/// Runs `piku shell`: the arguments after the subcommand, with the self-exec
/// upgrade refused because piku owns its own binary.
pub fn run_as_subcommand<I>(args: I) -> Result<i32, String>
where
    I: IntoIterator<Item = OsString>,
{
    run(args, "piku shell", false)
}

/// Runs the shell over `args` (the words after the program name).
///
/// `allow_upgrade` enables the wish self-exec path; it is off under piku.
pub fn run<I>(args: I, program: &str, allow_upgrade: bool) -> Result<i32, String>
where
    I: IntoIterator<Item = OsString>,
{
    match parse_args(args, program)? {
        Mode::SelfCheck { path } => {
            // `load` re-validates the blob, so a successful read means the
            // reply is exactly what a candidate must print.
            handoff::load(&path)?;
            println!("{}", handoff::self_check_reply());
            Ok(0)
        }
        Mode::Interactive { restore } => repl(restore, allow_upgrade),
    }
}

enum Mode {
    Interactive { restore: Option<PathBuf> },
    SelfCheck { path: PathBuf },
}

fn parse_args<I>(arguments: I, program: &str) -> Result<Mode, String>
where
    I: IntoIterator<Item = OsString>,
{
    let mut args = arguments.into_iter();
    let flag = args
        .next()
        .map(|flag| flag.into_string())
        .transpose()
        .map_err(|_| usage(program))?;
    let path = args.next().map(PathBuf::from);
    if args.next().is_some() {
        return Err(usage(program));
    }
    match (flag.as_deref(), path) {
        (None, None) => Ok(Mode::Interactive { restore: None }),
        (Some("--restore"), Some(path)) => Ok(Mode::Interactive {
            restore: Some(path),
        }),
        (Some("--self-check"), Some(path)) => Ok(Mode::SelfCheck { path }),
        _ => Err(usage(program)),
    }
}

fn usage(program: &str) -> String {
    format!("usage: {program} [--restore FILE | --self-check FILE]")
}

fn repl(restore: Option<PathBuf>, allow_upgrade: bool) -> Result<i32, String> {
    if !term::has_tty() {
        return Err("standard input is not a terminal".into());
    }
    let mut terminal = term::Terminal::new().map_err(|error| format!("terminal setup: {error}"))?;
    term::install_panic_hook();
    jobs::init().map_err(|error| format!("job control: {error}"))?;
    signals::install().map_err(|error| format!("signal setup: {error}"))?;

    let mut editor = LineEditor::new("", 0);
    let mut shell = Shell::new();
    let mut history = History::load();
    let mut input = Input::new();
    let mut exit_code = 0;
    if let Some(path) = restore {
        restore_session(&path, &mut editor, &mut shell);
    }

    terminal
        .enter_raw()
        .map_err(|error| format!("raw mode: {error}"))?;
    // Only repaint when something the prompt shows could have changed.
    let mut dirty = true;
    loop {
        // The self-pipe is the only place a signal becomes ordinary code.
        let received = signals::drain();
        if let Some(code) = terminating_code(&received) {
            exit_code = code;
            break;
        }
        if received.contains(&libc::SIGWINCH) {
            dirty = true;
        }
        let notices = shell.jobs.sweep();
        if !notices.is_empty() {
            for notice in notices {
                let line = format!("\r\x1b[K{}\r\n", notice.render());
                term::write(line.as_bytes()).map_err(|error| format!("terminal: {error}"))?;
            }
            dirty = true;
        }
        if dirty {
            editor
                .render(&prompt(&shell))
                .map_err(|error| format!("render: {error}"))?;
            dirty = false;
        }
        let key = match input.read_key() {
            Ok(Some(key)) => key,
            Ok(None) => break,
            Err(error) => return Err(format!("read input: {error}")),
        };
        if key == Key::Idle {
            continue;
        }
        dirty = true;
        match key {
            Key::Char(character) => {
                editor.insert(character);
                history.reset();
            }
            Key::Backspace => {
                editor.backspace();
                history.reset();
            }
            Key::Left => editor.left(),
            Key::Right => editor.right(),
            Key::Home => editor.home(),
            Key::End => editor.end(),
            Key::Up => {
                if let Some(line) = history.previous(&editor.text()) {
                    editor.set(&line, line.chars().count());
                }
            }
            Key::Down => {
                if let Some(line) = history.next() {
                    editor.set(&line, line.chars().count());
                }
            }
            Key::Clear | Key::Interrupt => {
                editor.clear();
                history.reset();
            }
            Key::Complete => {
                if let Err(error) = complete::tab(&mut editor, &COMPLETIONS) {
                    warn(&format!("completion: {error}"));
                }
                history.reset();
            }
            Key::Upgrade => {
                if let Err(error) = upgrade(
                    editor.text(),
                    editor.cursor(),
                    None,
                    &mut shell,
                    &mut terminal,
                    allow_upgrade,
                ) {
                    warn(&format!("upgrade: {error}"));
                }
            }
            Key::Eof => {
                if editor.is_empty() {
                    break;
                }
            }
            Key::Esc | Key::Idle | Key::Other => {}
            Key::Enter => {
                let line = editor.text();
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    editor.clear();
                    continue;
                }
                if let Some(candidate) = upgrade_target(trimmed) {
                    // The typed `:upgrade` line is a request, not a line to
                    // keep: only Ctrl-X Ctrl-U preserves the buffer. Otherwise
                    // the new image would show the same request and repeat it.
                    editor.clear();
                    history.reset();
                    if let Err(error) = upgrade(
                        String::new(),
                        0,
                        candidate.as_deref(),
                        &mut shell,
                        &mut terminal,
                        allow_upgrade,
                    ) {
                        warn(&format!("upgrade: {error}"));
                    }
                    continue;
                }
                if trimmed == ":q" {
                    if !shell.allow_exit() {
                        warn("there are running or stopped jobs; exit again to leave them");
                        editor.clear();
                        history.reset();
                        continue;
                    }
                    break;
                }
                // A bare `exit` gets the same one-warning gate here so the
                // prompt survives the first request; the builtin enforces it
                // for `exit N` too.
                if trimmed == "exit" && !shell.allow_exit() {
                    warn("there are running or stopped jobs; exit again to leave them");
                    editor.clear();
                    history.reset();
                    continue;
                }
                // `exit` is not a command to revisit; keeping it out of history
                // also preserves the old recall behavior for `exit`.
                if trimmed != "exit"
                    && let Err(error) = history.append(&line)
                {
                    warn(&format!("history: {error}"));
                }
                let started_in = std::env::current_dir().unwrap_or_default();
                let flow = exec::execute(&line, &mut shell, &mut terminal);
                if trimmed != "exit"
                    && let (Some(path), Some(record)) = (
                        ledger::ledger_path(),
                        ledger::entry(&line, &started_in, shell.status(), "piku-shell"),
                    )
                    && let Err(error) = ledger::append(&path, &record)
                {
                    warn(&format!("shell ledger: {error}"));
                }
                match flow {
                    Ok(exec::Flow::Continue) => {}
                    Ok(exec::Flow::Exit(code)) => {
                        exit_code = code;
                        break;
                    }
                    Err(error) => warn(&error),
                }
                editor.clear();
                history.reset();
            }
        }
    }
    terminal
        .restore()
        .map_err(|error| format!("restore terminal: {error}"))?;
    term::write(b"\r\n").map_err(|error| format!("terminal: {error}"))?;
    Ok(exit_code)
}

/// `128 + signal` for SIGTERM or SIGHUP, the exit status a signal teardown
/// reports.
fn terminating_code(signals: &[i32]) -> Option<i32> {
    signals
        .iter()
        .find(|signal| matches!(**signal, libc::SIGTERM | libc::SIGHUP))
        .map(|signal| 128 + *signal)
}

/// `Some(None)` for a bare `:upgrade`, `Some(Some(path))` with a candidate, and
/// `None` for any other line. The candidate must be absolute.
fn upgrade_target(line: &str) -> Option<Option<PathBuf>> {
    let rest = line.strip_prefix(":upgrade")?;
    if !rest.is_empty() && !rest.starts_with(char::is_whitespace) {
        return None;
    }
    let rest = rest.trim();
    if rest.is_empty() {
        return Some(None);
    }
    Some(Some(PathBuf::from(rest)))
}

/// Writes the blob, optionally installs and self-checks a candidate, then
/// execs. Only returns when the upgrade failed; the caller keeps the line.
fn upgrade(
    rest: String,
    cursor: usize,
    candidate: Option<&Path>,
    shell: &mut Shell,
    terminal: &mut term::Terminal,
    allow_upgrade: bool,
) -> Result<(), String> {
    if !allow_upgrade {
        return Err("upgrade is managed by piku; update piku to update this shell".into());
    }
    let state = handoff::State::capture(rest, cursor, shell.jobs.records())?;
    let path = handoff::blob_path();
    handoff::write(&path, &state)?;

    let mut installed = None;
    let executable = match candidate {
        Some(candidate) => {
            if !candidate.is_absolute() {
                let _ = fs::remove_file(&path);
                return Err(format!(
                    "{}: the candidate must be an absolute path",
                    candidate.display()
                ));
            }
            if let Err(error) = self_check(candidate, &path) {
                let _ = fs::remove_file(&path);
                return Err(error);
            }
            match handoff::install_binary(candidate) {
                Ok(installed_path) => {
                    installed = Some(installed_path.clone());
                    Some(installed_path)
                }
                Err(error) => {
                    let _ = fs::remove_file(&path);
                    return Err(error);
                }
            }
        }
        None => None,
    };

    // Leave raw mode before exec: the replacement image captures the terminal
    // attributes at startup, and it must see the caller's cooked settings
    // rather than the raw settings this image installed.
    terminal
        .restore()
        .map_err(|error| format!("restore terminal for upgrade: {error}"))?;
    let error = match &executable {
        Some(executable) => handoff::exec_state(&path, executable),
        None => handoff::exec_self(&path),
    };
    // exec() only returns on failure.
    let _ = fs::remove_file(&path);
    terminal
        .enter_raw()
        .map_err(|error| format!("raw mode after failed upgrade: {error}"))?;
    match installed {
        Some(installed) => warn(&format!(
            "upgrade failed: {error}; {} is installed for the next start and {}.old is the previous binary",
            installed.display(),
            installed.display()
        )),
        None => warn(&format!("upgrade failed: {error}")),
    }
    Ok(())
}

/// Runs a candidate binary with `--self-check <blob>` and aborts the update
/// unless it exits 0 *and* prints exactly `ok <blob-version>` on stdout. An
/// exit status alone would let any program (`/usr/bin/true`) be installed over
/// the shell, so the reply is the only evidence that the candidate read this
/// blob. The wait is bounded; a candidate that hangs is killed.
fn self_check(candidate: &Path, blob: &Path) -> Result<(), String> {
    const LIMIT: Duration = Duration::from_secs(5);
    let mut child = Process::new(candidate)
        .arg("--self-check")
        .arg(blob)
        .stdin(Stdio::null())
        // The reply is read below; its output must not reach the prompt, and it
        // would be lost at exec anyway.
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|error| format!("self-check {}: {error}", candidate.display()))?;
    let mut reply = child.stdout.take();
    let deadline = Instant::now() + LIMIT;
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                let printed = reply.as_mut().map(read_reply).unwrap_or_default();
                if !status.success() {
                    return Err(format!(
                        "{}: self-check failed ({status}); the running binary is unchanged",
                        candidate.display()
                    ));
                }
                if !handoff::confirms_blob(&printed) {
                    return Err(format!(
                        "{}: self-check did not report `{}`; the running binary is unchanged",
                        candidate.display(),
                        handoff::self_check_reply()
                    ));
                }
                return Ok(());
            }
            Ok(None) => {}
            Err(error) => {
                return Err(format!(
                    "self-check {}: {error}; the running binary is unchanged",
                    candidate.display()
                ));
            }
        }
        if Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            return Err(format!(
                "{}: self-check timed out; the running binary is unchanged",
                candidate.display()
            ));
        }
        std::thread::sleep(Duration::from_millis(20));
    }
}

/// Reads up to 256 bytes of the exited candidate's reply without blocking past
/// a short bound. The candidate itself is gone, but a process it forked could
/// still hold the pipe open; a plain `read_to_end` would hang the upgrade.
fn read_reply(handle: &mut std::process::ChildStdout) -> String {
    use std::os::unix::io::AsRawFd;

    const MAX_REPLY: usize = 256;
    const DRAIN: Duration = Duration::from_millis(200);
    let descriptor = handle.as_raw_fd();
    let deadline = Instant::now() + DRAIN;
    let mut bytes = Vec::new();
    while bytes.len() < MAX_REPLY {
        let mut ready = libc::pollfd {
            fd: descriptor,
            events: libc::POLLIN,
            revents: 0,
        };
        // SAFETY: poll inspects one initialized descriptor and only blocks for
        // the bounded timeout; it touches no other memory.
        let polled = unsafe { libc::poll(&mut ready, 1, 20) };
        if polled == 0 {
            if Instant::now() >= deadline {
                break;
            }
            continue;
        }
        if polled < 0 {
            if io::Error::last_os_error().kind() == io::ErrorKind::Interrupted {
                continue;
            }
            break;
        }
        let mut chunk = [0u8; 64];
        // SAFETY: poll reported the owned pipe readable, so this read returns
        // available bytes (or EOF) without blocking; the buffer is a live
        // local.
        let count = unsafe { libc::read(descriptor, chunk.as_mut_ptr().cast(), chunk.len()) };
        if count <= 0 {
            break;
        }
        bytes.extend_from_slice(&chunk[..count as usize]);
    }
    String::from_utf8_lossy(&bytes).into_owned()
}

fn restore_session(path: &Path, editor: &mut LineEditor, shell: &mut Shell) {
    let state = match handoff::load(path) {
        Ok(state) => state,
        Err(error) => {
            eprintln!("piku shell: could not restore session: {error}");
            return;
        }
    };
    let _ = fs::remove_file(path);
    if let Err(error) = std::env::set_current_dir(&state.cwd) {
        eprintln!(
            "piku shell: cannot restore directory {}: {error}; the line is kept",
            state.cwd
        );
    }
    *editor = LineEditor::new(&state.buffer, state.cursor);
    // Children survive the exec, so the table must too. Entries whose
    // processes are gone were terminated by the swap and are reported once.
    let missing = shell.jobs.restore(state.jobs);
    for record in missing {
        let line = format!(
            "piku shell: job {} ({}) did not survive the upgrade\r\n",
            record.pgid, record.command
        );
        let _ = term::write(line.as_bytes());
    }
    // The replacement process must own the terminal before it reads a key.
    let _ = jobs::take_terminal();
    if std::env::var_os("PIKU_SHELL_RESTORE_MARKER").is_some() {
        eprintln!("piku shell: restored");
    }
}

/// The shell's logical directory with `$HOME` abbreviated to `~`, then ` % `.
fn prompt(shell: &Shell) -> String {
    format!("{} % ", abbreviate_home(&shell.working_directory()))
}

fn abbreviate_home(cwd: &str) -> String {
    let Some(home) = std::env::var_os("HOME").filter(|home| !home.is_empty()) else {
        return cwd.to_string();
    };
    let home = home.to_string_lossy();
    if cwd == home {
        return "~".to_string();
    }
    match cwd.strip_prefix(home.as_ref()) {
        Some(rest) if rest.starts_with('/') => format!("~{rest}"),
        _ => cwd.to_string(),
    }
}

/// Reports a problem on its own line without leaving the prompt line mangled.
fn warn(message: &str) {
    let _ = term::write(b"\r\n");
    eprintln!("piku shell: {message}");
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;

    /// A candidate script for the self-check protocol.
    fn candidate(directory: &Path, body: &str) -> PathBuf {
        let path = directory.join("candidate");
        fs::write(&path, format!("#!/bin/sh\n{body}\n")).unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(0o755)).unwrap();
        path
    }

    #[test]
    fn self_check_requires_exit_zero_and_the_exact_reply() {
        let directory = tempfile::tempdir().unwrap();
        // The scripts ignore the blob path, so its contents are irrelevant here.
        let blob = directory.path().join("state.json");
        fs::write(&blob, b"{}").unwrap();
        let reply = handoff::self_check_reply();

        let good = candidate(directory.path(), &format!("printf '%s\\n' '{reply}'"));
        assert!(self_check(&good, &blob).is_ok());

        // Exits 0 without reading anything: the finding this guards.
        assert!(self_check(Path::new("/usr/bin/true"), &blob).is_err());
        let silent = candidate(directory.path(), "exit 0");
        assert!(self_check(&silent, &blob).is_err());
        let wrong = candidate(directory.path(), "printf '%s\\n' 'ok 99'");
        assert!(self_check(&wrong, &blob).is_err());
        let failing = candidate(
            directory.path(),
            &format!("printf '%s\\n' '{reply}'; exit 1"),
        );
        assert!(
            self_check(&failing, &blob).is_err(),
            "a nonzero exit must fail even with the right reply"
        );

        // A forked child keeps stdout open after the candidate exits; the reply
        // read must be bounded instead of waiting for EOF.
        let held_open = candidate(
            directory.path(),
            &format!("sleep 5 & printf '%s\\n' '{reply}'"),
        );
        let started = Instant::now();
        assert!(self_check(&held_open, &blob).is_ok());
        assert!(
            started.elapsed() < Duration::from_secs(2),
            "the reply read waited for a pipe another process held open"
        );
    }
}
