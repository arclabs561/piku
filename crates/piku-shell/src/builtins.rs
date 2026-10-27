//! Builtins that must run in this process, plus the shell's variable table.
//!
//! `cd` can only change the shell's own directory, and `export`/`unset`/`exit`
//! can only change its own state, so they never become child processes. The job
//! builtins (`jobs`, `fg`, `bg`, `wait`, `kill`) own the job table and the
//! terminal handover, so they cannot be children either.
//!
//! Every variable the shell holds is exported to children: there is no
//! shell-local scope yet, so `export NAME` (without a value) is a no-op and a
//! bare `NAME=value` also reaches children.

use std::collections::BTreeMap;
use std::io::Write;
use std::path::Path;

use crate::jobs::{self, JobTable, Waited};

pub use crate::syntax::is_identifier;

/// Shell state that survives across command lines.
pub struct Shell {
    vars: BTreeMap<String, String>,
    status: i32,
    /// Live and stopped jobs; the executor and the job builtins share it.
    pub jobs: JobTable,
    /// A first `exit` with live jobs warned; the next one goes through.
    exit_warned: bool,
}

impl Shell {
    /// Starts from the inherited environment, which is therefore exported.
    pub fn new() -> Self {
        let mut vars: BTreeMap<String, String> = std::env::vars_os()
            .filter_map(|(name, value)| Some((name.into_string().ok()?, value.into_string().ok()?)))
            .collect();
        // Keep an inherited `PWD` only when it still names this process's
        // directory: a fresh PTY or `--restore` changes the latter.
        if let Ok(cwd) = std::env::current_dir() {
            let keep = vars
                .get("PWD")
                .filter(|pwd| pwd.starts_with('/'))
                .is_some_and(|pwd| same_directory(pwd, &cwd));
            if !keep {
                vars.insert("PWD".to_string(), cwd.to_string_lossy().into_owned());
            }
        }
        Self {
            vars,
            status: 0,
            jobs: JobTable::new(),
            exit_warned: false,
        }
    }

    /// The directory to show: the logical `PWD` when it is absolute, else the
    /// physical directory. `cd` keeps `PWD` current, so this is cheap enough
    /// to call for every prompt render.
    pub fn working_directory(&self) -> String {
        if let Some(pwd) = self.var("PWD")
            && pwd.starts_with('/')
        {
            return pwd.to_string();
        }
        std::env::current_dir().map_or_else(
            |_| "?".to_string(),
            |path| path.to_string_lossy().into_owned(),
        )
    }

    /// Exit status of the last command, as `$?` reports it.
    pub fn status(&self) -> i32 {
        self.status
    }

    pub fn set_status(&mut self, status: i32) {
        self.status = status;
    }

    pub fn var(&self, name: &str) -> Option<&str> {
        self.vars.get(name).map(String::as_str)
    }

    pub fn variables(&self) -> &BTreeMap<String, String> {
        &self.vars
    }

    pub fn set_var(&mut self, name: &str, value: &str) {
        self.vars.insert(name.to_string(), value.to_string());
    }

    pub fn unset_var(&mut self, name: &str) {
        self.vars.remove(name);
    }

    /// Whether the shell may exit now. With live jobs the first request warns;
    /// a second one proceeds and leaves the jobs running, as asked.
    pub fn allow_exit(&mut self) -> bool {
        if self.jobs.is_empty() || self.exit_warned {
            return true;
        }
        self.exit_warned = true;
        false
    }
}

/// What a builtin leaves behind for the executor.
pub struct BuiltinResult {
    pub status: i32,
    /// Set when the shell must exit with this code.
    pub exit: Option<i32>,
    /// Already formatted for `piku shell: {error}`.
    pub error: Option<String>,
    /// The builtin already explained this status (a stopped `fg`, say), so the
    /// executor should not add a `piku shell: exit N` line.
    pub reported: bool,
}

impl BuiltinResult {
    fn success() -> Self {
        Self {
            status: 0,
            exit: None,
            error: None,
            reported: false,
        }
    }

    fn failed(error: String) -> Self {
        Self {
            status: 1,
            exit: None,
            error: Some(error),
            reported: false,
        }
    }

    fn stop(code: i32) -> Self {
        Self {
            status: code,
            exit: Some(code),
            error: None,
            reported: false,
        }
    }
}

pub fn lookup(name: &str) -> bool {
    matches!(
        name,
        "cd" | "pwd" | "export" | "unset" | "exit" | ":" | "jobs" | "fg" | "bg" | "wait" | "kill"
    )
}

/// Runs `name` with `args`, writing any output to `out`.
pub fn run(name: &str, args: &[String], shell: &mut Shell, out: &mut dyn Write) -> BuiltinResult {
    match name {
        "cd" => simple(cd(args, shell, out)),
        "pwd" => simple(pwd(shell, out)),
        "export" => simple(export(args, shell)),
        "unset" => simple(unset(args, shell)),
        "exit" => exit(args, shell),
        ":" => BuiltinResult::success(),
        "jobs" => simple(jobs(shell, out)),
        "fg" => foreground(args, shell, out),
        "bg" => background(args, shell, out),
        "wait" => wait(args, shell),
        "kill" => kill(args, shell),
        _ => BuiltinResult::failed(format!("{name}: not a builtin")),
    }
}

fn simple(result: Result<(), String>) -> BuiltinResult {
    match result {
        Ok(()) => BuiltinResult::success(),
        Err(error) => BuiltinResult::failed(error),
    }
}

fn cd(args: &[String], shell: &mut Shell, out: &mut dyn Write) -> Result<(), String> {
    if args.len() > 1 {
        return Err("cd: too many arguments".into());
    }
    let mut report = false;
    let target = match args.first() {
        Some(argument) if argument == "-" => {
            report = true;
            shell
                .var("OLDPWD")
                .map(ToString::to_string)
                .ok_or_else(|| "cd: OLDPWD is not set".to_string())?
        }
        Some(argument) => argument.clone(),
        None => shell
            .var("HOME")
            .map(ToString::to_string)
            .ok_or_else(|| "cd: HOME is not set".to_string())?,
    };
    if target.is_empty() {
        return Err("cd: empty directory name".into());
    }
    let before = std::env::current_dir().map_err(|error| format!("cd: {error}"))?;
    let old_pwd = shell
        .var("PWD")
        .filter(|pwd| pwd.starts_with('/') && same_directory(pwd, &before))
        .map(ToString::to_string);
    let old_pwd = old_pwd.unwrap_or_else(|| before.to_string_lossy().into_owned());
    let new_pwd = resolve_logical(&old_pwd, &target)?;
    std::env::set_current_dir(&new_pwd).map_err(|error| format!("cd: {target}: {error}"))?;
    shell.set_var("OLDPWD", &old_pwd);
    shell.set_var("PWD", &new_pwd);
    if report {
        writeln!(out, "{new_pwd}").map_err(|error| format!("cd: {error}"))?;
    }
    Ok(())
}

/// Prints the logical `PWD` when it still names this process's directory, and
/// the physical directory otherwise. This is the POSIX `pwd -L` default.
fn pwd(shell: &Shell, out: &mut dyn Write) -> Result<(), String> {
    let physical = std::env::current_dir().map_err(|error| format!("pwd: {error}"))?;
    let printed = match shell.var("PWD") {
        Some(pwd) if pwd.starts_with('/') && same_directory(pwd, &physical) => pwd.to_string(),
        _ => physical.to_string_lossy().into_owned(),
    };
    writeln!(out, "{printed}").map_err(|error| format!("pwd: {error}"))
}

/// Resolves `target` against the logical `old` path the way POSIX `cd -L`
/// does: `.` and empty components are dropped, `..` removes the previous
/// logical component, and every other component must exist and be a
/// directory. The returned path keeps the spelling the user gave, so a
/// symlinked directory stays spelled as such.
fn resolve_logical(old: &str, target: &str) -> Result<String, String> {
    let mut path = if target.starts_with('/') {
        String::from("/")
    } else {
        let trimmed = old.trim_end_matches('/');
        if trimmed.is_empty() {
            String::from("/")
        } else {
            trimmed.to_string()
        }
    };

    for component in target.split('/') {
        match component {
            "" | "." => {}
            ".." => {
                if let Some(index) = path.rfind('/') {
                    // Never move above the root: index 0 is the leading slash.
                    path.truncate(if index == 0 { 1 } else { index });
                }
            }
            name => {
                let candidate = if path == "/" {
                    format!("/{name}")
                } else {
                    format!("{path}/{name}")
                };
                let metadata = std::fs::metadata(&candidate)
                    .map_err(|error| format!("cd: {target}: {error}"))?;
                if !metadata.is_dir() {
                    return Err(format!("cd: {target}: not a directory"));
                }
                path = candidate;
            }
        }
    }
    Ok(path)
}

fn same_directory(logical: &str, physical: &Path) -> bool {
    match (
        std::fs::canonicalize(logical),
        std::fs::canonicalize(physical),
    ) {
        (Ok(logical), Ok(physical)) => logical == physical,
        _ => false,
    }
}

fn export(args: &[String], shell: &mut Shell) -> Result<(), String> {
    for argument in args {
        match argument.split_once('=') {
            Some((name, value)) if is_identifier(name) => shell.set_var(name, value),
            Some((name, _)) => {
                return Err(format!("export: {name}: not a valid identifier"));
            }
            // Every variable is already exported, so this only validates.
            None if is_identifier(argument) => {}
            None => return Err(format!("export: {argument}: not a valid identifier")),
        }
    }
    Ok(())
}

fn unset(args: &[String], shell: &mut Shell) -> Result<(), String> {
    for argument in args {
        if !is_identifier(argument) {
            return Err(format!("unset: {argument}: not a valid identifier"));
        }
        shell.unset_var(argument);
    }
    Ok(())
}

fn exit(args: &[String], shell: &mut Shell) -> BuiltinResult {
    if !shell.allow_exit() {
        return BuiltinResult::failed(
            "exit: there are running or stopped jobs; exit again to leave them".into(),
        );
    }
    let Some(argument) = args.first() else {
        // No operand: keep the status of the previous command, as `sh` does.
        return BuiltinResult::stop(shell.status());
    };
    match argument.parse::<i32>() {
        Ok(code) => BuiltinResult::stop(code & 0xff),
        Err(_) => BuiltinResult {
            status: 2,
            exit: Some(2),
            error: Some(format!("exit: {argument}: numeric argument required")),
            reported: false,
        },
    }
}

fn jobs(shell: &Shell, out: &mut dyn Write) -> Result<(), String> {
    for line in shell.jobs.list() {
        writeln!(out, "{line}").map_err(|error| format!("jobs: {error}"))?;
    }
    Ok(())
}

/// Resolves the job argument, defaulting to the current job.
fn job_id(args: &[String], shell: &Shell, name: &str) -> Result<usize, String> {
    match args.first() {
        Some(argument) => shell
            .jobs
            .resolve(argument)
            .ok_or_else(|| format!("{name}: {argument}: no such job")),
        None => shell
            .jobs
            .resolve("%+")
            .ok_or_else(|| format!("{name}: there is no current job")),
    }
}

fn foreground(args: &[String], shell: &mut Shell, out: &mut dyn Write) -> BuiltinResult {
    if args.len() > 1 {
        return BuiltinResult::failed("fg: too many arguments".into());
    }
    let id = match job_id(args, shell, "fg") {
        Ok(id) => id,
        Err(error) => return BuiltinResult::failed(error),
    };
    let Some(job) = shell.jobs.get(id) else {
        return BuiltinResult::failed(format!("fg: job {id} is gone"));
    };
    let command = job.command.clone();
    let pgid = job.pgid;
    // The command goes on the terminal so the transcript shows what resumed.
    if let Err(error) = writeln!(out, "{command}").and_then(|()| out.flush()) {
        return BuiltinResult::failed(format!("fg: {error}"));
    }
    // This guard takes the terminal back on every path, including a stop and
    // a panic in the wait.
    let terminal = match jobs::Foreground::new(pgid) {
        Ok(terminal) => terminal,
        Err(error) => return BuiltinResult::failed(format!("fg: terminal: {error}")),
    };
    if let Err(error) = shell.jobs.continue_job(id) {
        drop(terminal);
        return BuiltinResult::failed(format!("fg: {error}"));
    }
    let waited = shell.jobs.wait(id);
    drop(terminal);
    match waited {
        Waited::Exited(status) => {
            shell.jobs.remove(id);
            BuiltinResult {
                status,
                exit: None,
                error: None,
                reported: false,
            }
        }
        Waited::Signaled(signal) => {
            shell.jobs.remove(id);
            BuiltinResult {
                status: 128 + signal,
                exit: None,
                error: None,
                reported: false,
            }
        }
        Waited::Stopped(status) => {
            shell.jobs.mark_stopped(id);
            if let Some(notice) = shell.jobs.stop_notice(id)
                && let Err(error) = writeln!(out, "{}", notice.render())
            {
                return BuiltinResult::failed(format!("fg: {error}"));
            }
            BuiltinResult {
                status,
                exit: None,
                error: None,
                reported: true,
            }
        }
        // An interrupt leaves the job alone: it is still running or stopped
        // and stays listed for the next `fg`, `wait`, or `kill`.
        Waited::Interrupted(code) => BuiltinResult {
            status: code,
            exit: None,
            error: None,
            reported: true,
        },
        Waited::Terminated(code) => BuiltinResult {
            status: code,
            exit: Some(code),
            error: None,
            reported: false,
        },
    }
}

fn background(args: &[String], shell: &mut Shell, out: &mut dyn Write) -> BuiltinResult {
    if args.len() > 1 {
        return BuiltinResult::failed("bg: too many arguments".into());
    }
    let id = match job_id(args, shell, "bg") {
        Ok(id) => id,
        Err(error) => return BuiltinResult::failed(error),
    };
    let Some(job) = shell.jobs.get(id) else {
        return BuiltinResult::failed(format!("bg: job {id} is gone"));
    };
    let command = job.command.clone();
    if let Err(error) = shell.jobs.continue_job(id) {
        return BuiltinResult::failed(format!("bg: {error}"));
    }
    if let Err(error) = writeln!(out, "[{id}] {command} &") {
        return BuiltinResult::failed(format!("bg: {error}"));
    }
    BuiltinResult::success()
}

fn wait(args: &[String], shell: &mut Shell) -> BuiltinResult {
    if args.len() > 1 {
        return BuiltinResult::failed("wait: at most one job is supported".into());
    }
    let targets = match args.first() {
        Some(_) => match job_id(args, shell, "wait") {
            Ok(id) => vec![id],
            Err(error) => return BuiltinResult::failed(error),
        },
        None => shell.jobs.ids(),
    };
    let mut status = 0;
    let mut interrupted = false;
    for id in targets {
        match shell.jobs.wait(id) {
            Waited::Exited(code) => {
                shell.jobs.remove(id);
                status = code;
            }
            Waited::Signaled(signal) => {
                shell.jobs.remove(id);
                status = 128 + signal;
            }
            Waited::Stopped(code) => {
                shell.jobs.mark_stopped(id);
                status = code;
            }
            Waited::Interrupted(code) => {
                // Ctrl-C aborts the wait, not the job: keep the entry so the
                // operator can `fg`, `wait` again, or `kill` it.
                status = code;
                interrupted = true;
                break;
            }
            Waited::Terminated(code) => {
                return BuiltinResult {
                    status: code,
                    exit: Some(code),
                    error: None,
                    reported: false,
                };
            }
        }
    }
    BuiltinResult {
        status,
        exit: None,
        error: None,
        reported: interrupted,
    }
}

fn kill(args: &[String], shell: &mut Shell) -> BuiltinResult {
    let mut signal = libc::SIGTERM;
    let mut start = 0;
    if let Some(first) = args.first()
        && let Some(name) = first.strip_prefix('-')
        && !name.is_empty()
    {
        match parse_signal(name) {
            Some(parsed) => {
                signal = parsed;
                start = 1;
            }
            None => return BuiltinResult::failed(format!("kill: {first}: unknown signal")),
        }
    }
    let targets = &args[start..];
    if targets.is_empty() {
        return BuiltinResult::failed("kill: usage: kill [-SIGNAL] %n|pid".into());
    }
    let mut status = 0;
    let mut failure = None;
    for target in targets {
        let result = if let Some(spec) = target.strip_prefix('%') {
            match shell.jobs.resolve(&format!("%{spec}")) {
                Some(id) => shell
                    .jobs
                    .get(id)
                    .map_or(Ok(()), |job| signal_group(job.pgid, signal)),
                None => Err(format!("kill: {target}: no such job")),
            }
        } else if let Ok(pid) = target.parse::<i32>() {
            if pid <= 0 {
                Err(format!("kill: {target}: invalid pid"))
            } else {
                signal_pid(pid, signal)
            }
        } else {
            Err(format!("kill: {target}: not a job spec or pid"))
        };
        if let Err(error) = result {
            status = 1;
            failure = Some(error);
        }
    }
    BuiltinResult {
        status,
        exit: None,
        error: failure,
        reported: false,
    }
}

fn signal_group(pgid: libc::pid_t, signal: i32) -> Result<(), String> {
    if pgid <= 0 {
        return Err("kill: job has no process group".into());
    }
    // SAFETY: a signal to a process group this shell created.
    if unsafe { libc::kill(-pgid, signal) } == 0 {
        Ok(())
    } else {
        Err(format!("kill: {pgid}: {}", std::io::Error::last_os_error()))
    }
}

fn signal_pid(pid: libc::pid_t, signal: i32) -> Result<(), String> {
    // SAFETY: a signal to one pid; the kernel validates it.
    if unsafe { libc::kill(pid, signal) } == 0 {
        Ok(())
    } else {
        Err(format!("kill: {pid}: {}", std::io::Error::last_os_error()))
    }
}

fn parse_signal(text: &str) -> Option<i32> {
    let name = text
        .strip_prefix("SIG")
        .unwrap_or(text)
        .to_ascii_uppercase();
    if let Ok(number) = name.parse::<i32>() {
        return (number > 0).then_some(number);
    }
    Some(match name.as_str() {
        "HUP" => libc::SIGHUP,
        "INT" => libc::SIGINT,
        "QUIT" => libc::SIGQUIT,
        "KILL" => libc::SIGKILL,
        "TERM" => libc::SIGTERM,
        "STOP" => libc::SIGSTOP,
        "CONT" => libc::SIGCONT,
        "TSTP" => libc::SIGTSTP,
        "USR1" => libc::SIGUSR1,
        "USR2" => libc::SIGUSR2,
        "WINCH" => libc::SIGWINCH,
        "PIPE" => libc::SIGPIPE,
        "ALRM" => libc::SIGALRM,
        "CHLD" => libc::SIGCHLD,
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shell_with(pairs: &[(&str, &str)]) -> Shell {
        let mut shell = Shell::new();
        for (name, value) in pairs {
            shell.set_var(name, value);
        }
        shell
    }

    #[test]
    fn variables_are_stored_and_unset() {
        let mut shell = shell_with(&[("NAME", "wish")]);
        assert_eq!(shell.var("NAME"), Some("wish"));
        assert_eq!(shell.var("MISSING"), None);

        let mut out = Vec::new();
        let result = run("unset", &["NAME".to_string()], &mut shell, &mut out);
        assert_eq!(result.status, 0);
        assert_eq!(shell.var("NAME"), None);

        let result = run("unset", &["a-b".to_string()], &mut shell, &mut out);
        assert!(result.error.is_some());
    }

    #[test]
    fn export_sets_values_and_rejects_bad_names() {
        let mut shell = Shell::new();
        let mut out = Vec::new();
        let result = run(
            "export",
            &["WISH_TEST=one two".to_string(), "PATH".to_string()],
            &mut shell,
            &mut out,
        );
        assert_eq!(result.status, 0, "{:?}", result.error);
        assert_eq!(shell.var("WISH_TEST"), Some("one two"));

        let result = run("export", &["1BAD=x".to_string()], &mut shell, &mut out);
        assert!(result.error.is_some());
    }

    #[test]
    fn exit_parses_a_status_and_stops() {
        let mut shell = Shell::new();
        let mut out = Vec::new();

        let result = run("exit", &[], &mut shell, &mut out);
        assert_eq!(result.exit, Some(0));
        assert_eq!(result.error, None);

        shell.set_status(7);
        let result = run("exit", &[], &mut shell, &mut out);
        assert_eq!(result.exit, Some(7), "exit keeps the previous status");

        let result = run("exit", &["7".to_string()], &mut shell, &mut out);
        assert_eq!(result.exit, Some(7));

        let result = run("exit", &["999".to_string()], &mut shell, &mut out);
        assert_eq!(result.exit, Some(999 & 0xff));

        let result = run("exit", &["soon".to_string()], &mut shell, &mut out);
        assert_eq!(result.exit, Some(2));
        assert!(result.error.is_some());
    }

    #[test]
    fn pwd_starts_at_the_process_directory() {
        let shell = Shell::new();
        let cwd = std::env::current_dir().unwrap();
        let pwd = shell.var("PWD").expect("PWD is set");
        assert_eq!(
            std::fs::canonicalize(pwd).unwrap(),
            std::fs::canonicalize(&cwd).unwrap()
        );
    }

    #[test]
    fn pwd_writes_to_the_given_stream() {
        let mut shell = Shell::new();
        let mut out = Vec::new();
        let result = run("pwd", &[], &mut shell, &mut out);
        assert_eq!(result.status, 0);
        let printed = String::from_utf8(out).unwrap();
        let cwd = std::env::current_dir().unwrap();
        assert_eq!(
            std::fs::canonicalize(printed.trim_end()).unwrap(),
            std::fs::canonicalize(&cwd).unwrap()
        );
    }

    #[test]
    fn logical_resolution_walks_dot_dot_logically() {
        use std::fs;

        let directory = tempfile::tempdir().unwrap();
        let root = directory.path();
        fs::create_dir_all(root.join("a/b")).unwrap();
        fs::create_dir_all(root.join("x")).unwrap();
        let root_text = root.to_str().unwrap();

        assert_eq!(resolve_logical(root_text, "a/b/../..").unwrap(), root_text);
        assert_eq!(
            resolve_logical(root_text, "a/../x").unwrap(),
            format!("{root_text}/x")
        );
        assert_eq!(resolve_logical("/", "/").unwrap(), "/");
        // A component before `..` must exist, as in a POSIX logical cd.
        assert!(resolve_logical(root_text, "missing/..").is_err());
    }

    #[test]
    fn colon_and_unknown_names() {
        let mut shell = Shell::new();
        let mut out = Vec::new();
        assert_eq!(run(":", &[], &mut shell, &mut out).status, 0);
        assert!(run("nope", &[], &mut shell, &mut out).error.is_some());
        assert!(lookup("cd"));
        assert!(lookup(":"));
        assert!(lookup("jobs"));
        assert!(lookup("kill"));
        assert!(!lookup("echo"));
    }

    #[test]
    fn exit_warns_once_while_jobs_live() {
        let mut shell = Shell::new();
        let mut out = Vec::new();
        shell
            .jobs
            .add(4242, vec![4242], "sleep 60".into(), jobs::JobState::Running);

        let first = run("exit", &[], &mut shell, &mut out);
        assert_eq!(first.exit, None, "the first exit must not leave");
        assert!(
            first
                .error
                .as_deref()
                .is_some_and(|error| error.contains("exit again")),
            "the warning must say how to proceed: {:?}",
            first.error
        );

        let second = run("exit", &[], &mut shell, &mut out);
        assert_eq!(second.exit, Some(0), "the second exit proceeds");
    }

    #[test]
    fn jobs_lists_each_job_once_with_its_state() {
        let mut shell = Shell::new();
        let mut out = Vec::new();
        shell.jobs.add(
            4242,
            vec![4242],
            "make build".into(),
            jobs::JobState::Running,
        );
        shell
            .jobs
            .add(4243, vec![4243], "sleep 30".into(), jobs::JobState::Stopped);
        assert_eq!(run("jobs", &[], &mut shell, &mut out).status, 0);
        let text = String::from_utf8(out).unwrap();
        assert!(text.contains("[1]"));
        assert!(text.contains("[2]+"));
        assert!(text.contains("Running"));
        assert!(text.contains("Stopped"));
        assert!(text.contains("make build"));
    }

    #[test]
    fn kill_parses_signals_and_requires_a_target() {
        assert_eq!(parse_signal("TERM"), Some(libc::SIGTERM));
        assert_eq!(parse_signal("SIGKILL"), Some(libc::SIGKILL));
        assert_eq!(parse_signal("9"), Some(9));
        assert_eq!(parse_signal("nope"), None);

        let mut shell = Shell::new();
        let mut out = Vec::new();
        let result = run("kill", &[], &mut shell, &mut out);
        assert_eq!(result.status, 1);
        assert!(result.error.is_some());

        let result = run(
            "kill",
            &["-BOGUS".to_string(), "%1".to_string()],
            &mut shell,
            &mut out,
        );
        assert!(
            result
                .error
                .is_some_and(|error| error.contains("unknown signal"))
        );

        let result = run("kill", &["%7".to_string()], &mut shell, &mut out);
        assert!(
            result
                .error
                .is_some_and(|error| error.contains("no such job"))
        );
    }
}
