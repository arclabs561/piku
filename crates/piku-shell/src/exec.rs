//! Native execution: expansion, redirection, pipelines, and job control.
//!
//! Nothing here goes through `sh`. The shell expands the parsed words itself,
//! runs builtins in process, resolves external commands against `PATH`, and
//! connects pipeline stages with pipes. Each job gets its own process group;
//! the shell owns the terminal, hands it to a foreground job, and takes it back
//! before the prompt returns. A trailing `&` leaves the job running and prints
//! `[n] pid` instead of waiting.

use std::fs::{File, OpenOptions};
use std::io::{self, Write};
use std::os::unix::fs::PermissionsExt;
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Child, Command as Process, Stdio};

use crate::builtins::{self, Shell};
use crate::expand;
use crate::jobs::{self, JobState, Pid, Waited};
use crate::signals;
use crate::syntax::{self, Command, Connector, Pipeline, Program, RedirectKind};
use crate::term::{self, Terminal};

/// Whether the shell keeps reading lines after a command line finishes.
pub enum Flow {
    Continue,
    Exit(i32),
}

/// What one pipeline or line left behind.
struct Outcome {
    status: i32,
    /// Set when the shell must exit with this code.
    exit: Option<i32>,
    /// A `piku shell: ...` message already described this failure.
    reported: bool,
}

impl Outcome {
    fn success() -> Self {
        Self {
            status: 0,
            exit: None,
            reported: false,
        }
    }

    fn failure(status: i32) -> Self {
        Self {
            status,
            exit: None,
            reported: false,
        }
    }

    fn reported(status: i32) -> Self {
        Self {
            status,
            exit: None,
            reported: true,
        }
    }
}

/// Parses and runs one line. `Err` is printed with the `piku shell: ` prefix by the
/// caller; a parse error runs nothing at all.
pub fn execute(line: &str, shell: &mut Shell, terminal: &mut Terminal) -> Result<Flow, String> {
    let program = match syntax::parse(line) {
        Ok(program) => program,
        Err(error) => {
            shell.set_status(2);
            return Err(error);
        }
    };
    if program.items.is_empty() {
        return Ok(Flow::Continue);
    }

    // Finish the prompt line before any command writes anything.
    term::write(b"\r\n").map_err(terminal_error)?;
    terminal.restore().map_err(terminal_error)?;

    let outcome = run_program(&program, shell);

    terminal.enter_raw().map_err(terminal_error)?;
    // A requested exit is the shell's own status, not a command failure.
    if outcome.exit.is_none() && outcome.status != 0 && !outcome.reported {
        let message = format!("piku shell: exit {}\r\n", outcome.status);
        term::write(message.as_bytes()).map_err(terminal_error)?;
    }
    Ok(match outcome.exit {
        Some(code) => Flow::Exit(code),
        None => Flow::Continue,
    })
}

fn terminal_error(error: io::Error) -> String {
    format!("terminal: {error}")
}

fn run_program(program: &Program, shell: &mut Shell) -> Outcome {
    report_notices(shell.jobs.sweep());
    let mut last = Outcome::success();
    for item in &program.items {
        let skip = match item.connector {
            Connector::First | Connector::Sequence => false,
            Connector::And => shell.status() != 0,
            Connector::Or => shell.status() == 0,
        };
        if skip {
            continue;
        }
        let outcome = run_pipeline(&item.pipeline, shell, item.background);
        shell.set_status(outcome.status);
        if outcome.exit.is_some() {
            return outcome;
        }
        last = outcome;
    }
    last
}

/// Prints job notices that arrived before this line ran.
fn report_notices(notices: Vec<jobs::Notice>) {
    for notice in notices {
        let line = format!("{}\r\n", notice.render());
        let _ = term::write(line.as_bytes());
    }
}

/// One command with its words expanded and its targets resolved.
struct Expanded {
    argv: Vec<String>,
    redirects: Vec<Redirection>,
}

struct Redirection {
    kind: RedirectKind,
    path: PathBuf,
}

fn run_pipeline(pipeline: &Pipeline, shell: &mut Shell, background: bool) -> Outcome {
    let mut commands = Vec::with_capacity(pipeline.commands.len());
    for command in &pipeline.commands {
        match expand_command(command, shell) {
            Ok(expanded) => commands.push(expanded),
            Err(error) => return report(&error, 1),
        }
    }
    if commands.len() == 1 {
        let expanded = commands.remove(0);
        return run_simple(expanded, shell, background);
    }
    run_children(commands, shell, background)
}

fn expand_command(command: &Command, shell: &Shell) -> Result<Expanded, String> {
    let argv = expand::words(&command.words, shell.variables(), shell.status());
    let mut redirects = Vec::with_capacity(command.redirects.len());
    for redirect in &command.redirects {
        let targets = expand::words(
            std::slice::from_ref(&redirect.target),
            shell.variables(),
            shell.status(),
        );
        match targets.as_slice() {
            [path] => redirects.push(Redirection {
                kind: redirect.kind,
                path: PathBuf::from(path),
            }),
            [] => return Err("redirection needs a file name".into()),
            _ => return Err("redirection target expanded to several words".into()),
        }
    }
    Ok(Expanded { argv, redirects })
}

fn run_simple(expanded: Expanded, shell: &mut Shell, background: bool) -> Outcome {
    let Expanded {
        mut argv,
        redirects,
    } = expanded;

    // Leading `NAME=value` words set variables; this subset has no temporary
    // environment, so they persist like `export` would.
    let mut assigned = 0;
    while let Some((name, value)) = argv.get(assigned).and_then(|word| assignment(word)) {
        shell.set_var(name, value);
        assigned += 1;
    }
    argv.drain(..assigned);

    if argv.is_empty() {
        return match touch(&redirects) {
            Ok(()) => Outcome::success(),
            Err(error) => report(&error, 1),
        };
    }
    if builtins::lookup(&argv[0]) {
        // A builtin changes this process, so it cannot become a separate job.
        if background {
            return report(
                &format!("{}: background builtins are not implemented", argv[0]),
                1,
            );
        }
        return run_builtin(&argv, &redirects, shell);
    }
    run_external(vec![Expanded { argv, redirects }], shell, background)
}

/// Splits `NAME=value` into its parts when the name is a valid identifier.
fn assignment(word: &str) -> Option<(&str, &str)> {
    let (name, value) = word.split_once('=')?;
    builtins::is_identifier(name).then_some((name, value))
}

fn run_builtin(argv: &[String], redirects: &[Redirection], shell: &mut Shell) -> Outcome {
    let mut out: Box<dyn Write> = match builtin_output(redirects) {
        Ok(out) => out,
        Err(error) => return report(&error, 1),
    };
    // A builtin runs in this process while the shell is still the terminal's
    // foreground group, so a default SIGINT disposition would end the session
    // on `Ctrl-C`. The guard records the interrupt, the job builtins stop
    // waiting because of it, and dropping it restores what the prompt expects.
    let interrupt = match signals::InterruptGuard::new() {
        Ok(guard) => Some(guard),
        Err(error) => {
            eprintln!("piku shell: signals: {error}");
            None
        }
    };
    let result = builtins::run(&argv[0], &argv[1..], shell, out.as_mut());
    let _ = out.flush();
    drop(interrupt);

    let reported = result.error.is_some() || result.reported;
    if let Some(error) = result.error {
        eprintln!("piku shell: {error}");
    }
    Outcome {
        status: result.exit.unwrap_or(result.status),
        exit: result.exit,
        reported,
    }
}

/// The writer a builtin should print to, with read redirections validated.
fn builtin_output(redirects: &[Redirection]) -> Result<Box<dyn Write>, String> {
    let mut out: Box<dyn Write> = Box::new(io::stdout());
    for redirect in redirects {
        match redirect.kind {
            RedirectKind::Read => {
                File::open(&redirect.path)
                    .map_err(|error| format!("{}: {error}", redirect.path.display()))?;
            }
            RedirectKind::Write => {
                let file = File::create(&redirect.path)
                    .map_err(|error| format!("{}: {error}", redirect.path.display()))?;
                out = Box::new(file);
            }
            RedirectKind::Append => {
                let file = OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&redirect.path)
                    .map_err(|error| format!("{}: {error}", redirect.path.display()))?;
                out = Box::new(file);
            }
        }
    }
    Ok(out)
}

/// Applies the redirections of a command that has no words: the files are
/// created or truncated, and nothing is written.
fn touch(redirects: &[Redirection]) -> Result<(), String> {
    for redirect in redirects {
        match redirect.kind {
            RedirectKind::Read => {
                File::open(&redirect.path)
                    .map_err(|error| format!("{}: {error}", redirect.path.display()))?;
            }
            RedirectKind::Write => {
                File::create(&redirect.path)
                    .map_err(|error| format!("{}: {error}", redirect.path.display()))?;
            }
            RedirectKind::Append => {
                OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&redirect.path)
                    .map_err(|error| format!("{}: {error}", redirect.path.display()))?;
            }
        };
    }
    Ok(())
}

fn open_redirects(redirects: &[Redirection]) -> Result<(Option<Stdio>, Option<Stdio>), String> {
    let mut stdin = None;
    let mut stdout = None;
    for redirect in redirects {
        let error = |error: io::Error| format!("{}: {error}", redirect.path.display());
        match redirect.kind {
            RedirectKind::Read => {
                stdin = Some(Stdio::from(File::open(&redirect.path).map_err(error)?));
            }
            RedirectKind::Write => {
                stdout = Some(Stdio::from(File::create(&redirect.path).map_err(error)?));
            }
            RedirectKind::Append => {
                let file = OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&redirect.path)
                    .map_err(error)?;
                stdout = Some(Stdio::from(file));
            }
        }
    }
    Ok((stdin, stdout))
}

/// Spawns one command. The first stage of a job passes `group: None` and
/// becomes the group leader; later stages join that group.
///
/// Both the child and the parent call `setpgid`, which closes the race where a
/// signal arrives before one side has run: whichever call wins, the group is
/// the same and the loser fails harmlessly.
fn spawn(
    argv: &[String],
    stdin: Option<Stdio>,
    stdout: Option<Stdio>,
    shell: &Shell,
    group: Option<Pid>,
) -> Result<Child, Outcome> {
    let name = &argv[0];
    let Some(program) = resolve(name, shell.var("PATH")) else {
        return Err(report(&format!("{name}: command not found"), 127));
    };
    let mut process = Process::new(&program);
    process.arg0(name.as_str());
    process.args(&argv[1..]);
    process.env_clear().envs(shell.variables());
    if let Some(stdin) = stdin {
        process.stdin(stdin);
    }
    if let Some(stdout) = stdout {
        process.stdout(stdout);
    }
    let target = group.unwrap_or(0);
    // SAFETY: the closure runs in the forked child between fork and exec. It
    // only calls async-signal-safe functions (`setpgid`, `signal`) and turns
    // the job-control signals back to their defaults, which the shell itself
    // ignores so that Ctrl-Z and Ctrl-C reach the job instead of the shell.
    unsafe {
        process.pre_exec(move || {
            if libc::setpgid(0, target) != 0 {
                return Err(io::Error::last_os_error());
            }
            for signal in [
                libc::SIGINT,
                libc::SIGQUIT,
                libc::SIGTSTP,
                libc::SIGTTIN,
                libc::SIGTTOU,
                libc::SIGCHLD,
                libc::SIGTERM,
                libc::SIGHUP,
                libc::SIGPIPE,
            ] {
                libc::signal(signal, libc::SIG_DFL);
            }
            Ok(())
        });
    }
    let child = process.spawn().map_err(|error| {
        let status = if error.kind() == io::ErrorKind::NotFound {
            127
        } else {
            126
        };
        report(&format!("{name}: {error}"), status)
    })?;
    let pid = child.id() as Pid;
    // SAFETY: setpgid on a child this process just spawned; the child also did
    // it before exec, so EACCES here means the group is already correct.
    let _ = unsafe { libc::setpgid(pid, group.unwrap_or(pid)) };
    Ok(child)
}

/// Runs every stage as a child, with each stdout piped into the next stdin.
/// The job's process group is the first stage's pid.
fn run_children(commands: Vec<Expanded>, shell: &mut Shell, background: bool) -> Outcome {
    let count = commands.len();
    let mut children: Vec<Child> = Vec::with_capacity(count);
    let mut next_stdin: Option<Stdio> = None;
    let mut failure: Option<Outcome> = None;
    let mut group: Option<Pid> = None;

    for (index, command) in commands.iter().enumerate() {
        let last = index + 1 == count;
        if command.argv.is_empty() {
            failure = Some(report("a pipeline needs a command", 1));
            break;
        }
        if assignment(&command.argv[0]).is_some() {
            failure = Some(report(
                "assignments in a pipeline are not implemented yet",
                1,
            ));
            break;
        }
        if builtins::lookup(&command.argv[0]) {
            failure = Some(report(
                &format!(
                    "{}: builtins in a pipeline are not implemented yet",
                    command.argv[0]
                ),
                1,
            ));
            break;
        }
        if !last
            && command
                .redirects
                .iter()
                .any(|redirect| matches!(redirect.kind, RedirectKind::Write | RedirectKind::Append))
        {
            failure = Some(report(
                "redirection is not implemented for a non-final pipeline stage",
                1,
            ));
            break;
        }
        let (stdin, stdout) = match open_redirects(&command.redirects) {
            Ok(stdio) => stdio,
            Err(error) => {
                failure = Some(report(&error, 1));
                break;
            }
        };
        // A stage with its own stdin redirect does not read the previous
        // stage: close this end of the pipe so the writer cannot block on a
        // reader that never arrives.
        let stdin = match stdin {
            Some(stdin) => {
                next_stdin = None;
                Some(stdin)
            }
            None => next_stdin.take(),
        };
        let stdout = if last { stdout } else { Some(Stdio::piped()) };
        let mut child = match spawn(&command.argv, stdin, stdout, shell, group) {
            Ok(child) => child,
            Err(outcome) => {
                failure = Some(outcome);
                break;
            }
        };
        if group.is_none() {
            group = Some(child.id() as Pid);
        }
        if !last {
            match child.stdout.take() {
                Some(handle) => next_stdin = Some(Stdio::from(handle)),
                None => {
                    children.push(child);
                    failure = Some(report("pipeline: no output to connect", 1));
                    break;
                }
            }
        }
        children.push(child);
    }

    // Drop any unconsumed pipe end before waiting: if a later stage never
    // started, an earlier stage must see a closed pipe instead of blocking
    // forever on a reader that will never read.
    drop(next_stdin);

    let Some(pgid) = group.filter(|_| !children.is_empty()) else {
        return failure.unwrap_or_else(|| report("pipeline: nothing ran", 1));
    };
    let pids: Vec<Pid> = children.iter().map(|child| child.id() as Pid).collect();
    let command = describe(&commands);
    let id = shell.jobs.add(pgid, pids, command, JobState::Running);

    if let Some(outcome) = failure {
        // A later stage was refused after this job's first stage was already
        // spawned. Waiting for it here would ignore the terminal's signals and
        // hold the prompt for the child's lifetime, and a stopped child would
        // then be removed as if it had finished. Kill the group, reap it, and
        // forget it instead: the caller restores the terminal and the signal
        // state on its way back to the prompt. The interrupt guard only keeps a
        // stray `Ctrl-C` during this cleanup from ending the session.
        let _interrupt = signals::InterruptGuard::new();
        shell.jobs.terminate(id);
        return outcome;
    }
    if background {
        return start_background(shell, id, pgid);
    }
    wait_foreground(shell, id, pgid)
}

/// One external command or pipeline, run as one job.
fn run_external(commands: Vec<Expanded>, shell: &mut Shell, background: bool) -> Outcome {
    if commands.len() > 1 {
        return run_children(commands, shell, background);
    }
    let Some(command) = commands.into_iter().next() else {
        return Outcome::success();
    };
    let (stdin, stdout) = match open_redirects(&command.redirects) {
        Ok(stdio) => stdio,
        Err(error) => return report(&error, 1),
    };
    let child = match spawn(&command.argv, stdin, stdout, shell, None) {
        Ok(child) => child,
        Err(outcome) => return outcome,
    };
    let pgid = child.id() as Pid;
    let id = shell.jobs.add(
        pgid,
        vec![pgid],
        describe(std::slice::from_ref(&command)),
        JobState::Running,
    );
    if background {
        return start_background(shell, id, pgid);
    }
    wait_foreground(shell, id, pgid)
}

/// Announces a background job and returns to the prompt without waiting.
fn start_background(_shell: &Shell, id: usize, pgid: Pid) -> Outcome {
    let line = format!("[{id}] {pgid}\r\n");
    if let Err(error) = term::write(line.as_bytes()) {
        return report(&format!("job {id}: {error}"), 1);
    }
    Outcome::success()
}

/// Waits for a foreground job with the terminal handed over, then takes the
/// terminal back on every path.
fn wait_foreground(shell: &mut Shell, id: usize, pgid: Pid) -> Outcome {
    // Spawn first: the child must inherit the default SIGINT/SIGQUIT handling.
    // Only then does the shell ignore those signals, so Ctrl-C in the terminal
    // reaches the child and its foreground group without killing the shell.
    let ignored = IgnoredSignals::new();
    let terminal = match jobs::Foreground::new(pgid) {
        Ok(terminal) => Some(terminal),
        Err(error) => {
            eprintln!("piku shell: terminal: {error}");
            None
        }
    };
    let waited = shell.jobs.wait(id);
    drop(terminal);
    drop(ignored);
    match waited {
        Waited::Exited(status) => {
            shell.jobs.remove(id);
            finish(status)
        }
        Waited::Signaled(signal) => {
            shell.jobs.remove(id);
            eprintln!("piku shell: terminated by signal {signal}");
            Outcome::reported(128 + signal)
        }
        Waited::Stopped(status) => {
            shell.jobs.mark_stopped(id);
            if let Some(notice) = shell.jobs.stop_notice(id) {
                let line = format!("{}\r\n", notice.render());
                let _ = term::write(line.as_bytes());
            }
            Outcome::reported(status)
        }
        // Not reachable through `IgnoredSignals`, but the job is untouched by
        // an interrupt and must stay in the table if it ever is.
        Waited::Interrupted(code) => Outcome::reported(code),
        Waited::Terminated(code) => Outcome {
            status: code,
            exit: Some(code),
            reported: true,
        },
    }
}

/// The command line a job shows in `jobs`, built from the expanded words.
fn describe(commands: &[Expanded]) -> String {
    commands
        .iter()
        .map(|command| command.argv.join(" "))
        .collect::<Vec<_>>()
        .join(" | ")
}

/// Turns an already-decoded pipeline status into an outcome. The wait layer
/// reports an exit code or a signal separately, so this must not reinterpret a
/// raw `waitpid` word.
fn finish(status: i32) -> Outcome {
    if status == 0 {
        Outcome::success()
    } else {
        Outcome::failure(status)
    }
}

fn report(message: &str, status: i32) -> Outcome {
    eprintln!("piku shell: {message}");
    // Not `reported`: a failed external command or redirection should still
    // end the line with the same `piku shell: exit N` line as before, so the
    // operator sees the status the shell recorded.
    Outcome::failure(status)
}

/// Finds an executable for `name`, using the shell's own `PATH`.
fn resolve(name: &str, path: Option<&str>) -> Option<PathBuf> {
    if name.contains('/') {
        return Some(PathBuf::from(name));
    }
    let path = path.unwrap_or("/usr/bin:/bin");
    for directory in path.split(':') {
        let directory = if directory.is_empty() { "." } else { directory };
        let candidate = Path::new(directory).join(name);
        if is_executable(&candidate) {
            return Some(candidate);
        }
    }
    None
}

fn is_executable(path: &Path) -> bool {
    std::fs::metadata(path)
        .is_ok_and(|metadata| metadata.is_file() && metadata.permissions().mode() & 0o111 != 0)
}

/// Temporarily ignores SIGINT and SIGQUIT for the shell process itself.
struct IgnoredSignals {
    interrupt: libc::sighandler_t,
    quit: libc::sighandler_t,
}

impl IgnoredSignals {
    fn new() -> Self {
        // SAFETY: signal() installs a process-wide disposition; SIG_IGN is a
        // valid handler and the returned previous handler is saved.
        let interrupt = unsafe { libc::signal(libc::SIGINT, libc::SIG_IGN) };
        // SAFETY: same contract as above for SIGQUIT.
        let quit = unsafe { libc::signal(libc::SIGQUIT, libc::SIG_IGN) };
        Self { interrupt, quit }
    }
}

impl Drop for IgnoredSignals {
    fn drop(&mut self) {
        // SAFETY: restores exactly the handlers recorded by new().
        unsafe {
            libc::signal(libc::SIGINT, self.interrupt);
            libc::signal(libc::SIGQUIT, self.quit);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn assignments_need_a_valid_name() {
        assert_eq!(assignment("A=1"), Some(("A", "1")));
        assert_eq!(assignment("A="), Some(("A", "")));
        assert_eq!(assignment("_X=a=b"), Some(("_X", "a=b")));
        assert_eq!(assignment("1A=x"), None);
        assert_eq!(assignment("=x"), None);
        assert_eq!(assignment("A"), None);
    }

    #[test]
    fn resolve_searches_path_entries_in_order() {
        let first = TempDir::new().unwrap();
        let second = TempDir::new().unwrap();
        for directory in [first.path(), second.path()] {
            let tool = directory.join("wish-test-tool");
            fs::write(&tool, b"#!/bin/sh\n").unwrap();
            let mut permissions = fs::metadata(&tool).unwrap().permissions();
            permissions.set_mode(0o755);
            fs::set_permissions(&tool, permissions).unwrap();
        }
        let path = format!("{}:{}", first.path().display(), second.path().display());
        assert_eq!(
            resolve("wish-test-tool", Some(&path)),
            Some(first.path().join("wish-test-tool"))
        );
        assert_eq!(resolve("not-there", Some(&path)), None);
        // A command with a slash is used as written.
        assert_eq!(
            resolve("./local", Some(&path)),
            Some(PathBuf::from("./local"))
        );
    }

    #[test]
    fn resolve_skips_files_that_are_not_executable() {
        let directory = TempDir::new().unwrap();
        let plain = directory.path().join("plain");
        fs::write(&plain, b"data").unwrap();
        let mut permissions = fs::metadata(&plain).unwrap().permissions();
        permissions.set_mode(0o644);
        fs::set_permissions(&plain, permissions).unwrap();
        assert_eq!(
            resolve("plain", Some(directory.path().to_str().unwrap())),
            None
        );

        let empty = directory.path().join("empty");
        fs::create_dir(&empty).unwrap();
        assert_eq!(
            resolve("empty", Some(directory.path().to_str().unwrap())),
            None,
            "a directory is not a command"
        );
    }
}
