//! Job control: process groups, terminal ownership, and the job table.
//!
//! Every foreign command runs in its own process group. The shell owns the
//! terminal while it edits a line, hands it to a foreground job with
//! `tcsetpgrp`, and takes it back on every path before it reads another key.
//! A shell started in the background never claims the terminal: only the
//! process group the terminal already considers foreground may take it.
//!
//! Children are reaped with a `waitpid(-1, WNOHANG)` sweep rather than a
//! `SIGCHLD` handler: one thread, one table, and no signal can be missed
//! between a fork and a handler installation. A stopped or finished job is
//! announced exactly once.
//!
//! The `unsafe` blocks here are the process-group, waiting, and terminal
//! wrappers only; each carries the safety argument for its call.

use std::io;
use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};

use crate::handoff::{JobRecord, JobState as SavedJobState};
use crate::signals;

pub type Pid = libc::pid_t;

/// The shell's own process group, captured once at startup.
static SHELL_PGID: AtomicI32 = AtomicI32::new(0);
/// Whether this image may use the terminal at all. False when it started in
/// the background, where taking `tcsetpgrp` would steal the terminal from the
/// job that is entitled to it.
static TERMINAL_OWNED: AtomicBool = AtomicBool::new(false);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JobState {
    Running,
    Stopped,
}

impl JobState {
    pub fn text(self) -> &'static str {
        match self {
            Self::Running => "Running",
            Self::Stopped => "Stopped",
        }
    }

    fn saved(self) -> SavedJobState {
        match self {
            Self::Running => SavedJobState::Running,
            Self::Stopped => SavedJobState::Stopped,
        }
    }

    fn restored(state: SavedJobState) -> Self {
        match state {
            SavedJobState::Running => Self::Running,
            SavedJobState::Stopped => Self::Stopped,
        }
    }
}

/// How one process in a job ended, or that it is currently stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Ended {
    Exited(i32),
    Signaled(i32),
    Stopped(i32),
}

/// One job: a pipeline of processes sharing a process group.
#[derive(Debug, Clone)]
pub struct Job {
    pub id: usize,
    pub pgid: Pid,
    pub pids: Vec<Pid>,
    pub command: String,
    pub state: JobState,
    /// Per-stage outcome, in pipeline order; `None` while still running.
    ended: Vec<Option<Ended>>,
    /// A stop has already been announced, so the sweep stays quiet until the
    /// job runs and stops again.
    announced_stop: bool,
}

impl Job {
    fn finished(&self) -> bool {
        self.ended
            .iter()
            .all(|ended| matches!(ended, Some(Ended::Exited(_)) | Some(Ended::Signaled(_))))
    }

    /// The last stage's outcome, which is the pipeline's status.
    fn last_stage(&self) -> Option<Ended> {
        self.ended.last().copied().flatten()
    }

    fn is_stopped(&self) -> bool {
        self.ended
            .iter()
            .any(|ended| matches!(ended, Some(Ended::Stopped(_))))
    }

    fn stop_signal(&self) -> i32 {
        self.ended
            .iter()
            .rev()
            .find_map(|ended| match ended {
                Some(Ended::Stopped(signal)) => Some(*signal),
                _ => None,
            })
            .unwrap_or(libc::SIGTSTP)
    }
}

/// A job event the shell should print above the next prompt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Notice {
    pub id: usize,
    pub marker: char,
    pub state: String,
    pub command: String,
}

impl Notice {
    pub fn render(&self) -> String {
        format!(
            "[{}]{}  {:<24}{}",
            self.id, self.marker, self.state, self.command
        )
    }
}

/// What waiting for a job produced.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Waited {
    /// The job finished; this is its pipeline status.
    Exited(i32),
    /// The last pipeline stage died from this signal.
    Signaled(i32),
    /// The job stopped; this is 128 plus the stop signal.
    Stopped(i32),
    /// The operator interrupted the wait with SIGINT or SIGQUIT; this is 128
    /// plus the interrupt signal and the job is untouched.
    Interrupted(i32),
    /// The shell was asked to exit by SIGTERM or SIGHUP while waiting.
    Terminated(i32),
}

/// The table of live jobs, in start order.
#[derive(Debug, Default)]
pub struct JobTable {
    jobs: Vec<Job>,
    next_id: usize,
    current: Option<usize>,
    previous: Option<usize>,
}

impl JobTable {
    pub fn new() -> Self {
        Self {
            jobs: Vec::new(),
            next_id: 1,
            current: None,
            previous: None,
        }
    }

    /// Registers a spawned job and returns its job number.
    pub fn add(&mut self, pgid: Pid, pids: Vec<Pid>, command: String, state: JobState) -> usize {
        let id = self.next_id;
        self.next_id += 1;
        let ended = vec![None; pids.len()];
        self.jobs.push(Job {
            id,
            pgid,
            pids,
            command,
            state,
            ended,
            announced_stop: false,
        });
        self.previous = self.current;
        self.current = Some(id);
        id
    }

    pub fn get(&self, id: usize) -> Option<&Job> {
        self.jobs.iter().find(|job| job.id == id)
    }

    /// Every job number, in start order.
    pub fn ids(&self) -> Vec<usize> {
        self.jobs.iter().map(|job| job.id).collect()
    }

    pub fn is_empty(&self) -> bool {
        self.jobs.is_empty()
    }

    pub fn remove(&mut self, id: usize) {
        self.jobs.retain(|job| job.id != id);
        if self.current == Some(id) {
            self.current = self.previous.take();
        } else if self.previous == Some(id) {
            self.previous = None;
        }
    }

    fn marker(&self, id: usize) -> char {
        if self.current == Some(id) {
            '+'
        } else if self.previous == Some(id) {
            '-'
        } else {
            ' '
        }
    }

    /// Resolves `%n`, `%+`/`%%`, `%-`, or `%?substring` to a job number.
    pub fn resolve(&self, spec: &str) -> Option<usize> {
        let rest = spec.strip_prefix('%')?;
        match rest {
            "" | "+" | "%" => self.current,
            "-" => self.previous,
            _ => match rest.strip_prefix('?') {
                Some(query) => self
                    .jobs
                    .iter()
                    .rev()
                    .find(|job| job.command.contains(query))
                    .map(|job| job.id),
                None => {
                    let id = rest.parse::<usize>().ok()?;
                    self.get(id).map(|job| job.id)
                }
            },
        }
    }

    /// One `jobs` line per job.
    pub fn list(&self) -> Vec<String> {
        self.jobs
            .iter()
            .map(|job| {
                format!(
                    "[{}]{}  {:<24}{}",
                    job.id,
                    self.marker(job.id),
                    job.state.text(),
                    job.command
                )
            })
            .collect()
    }

    /// The serializable half of the table, in start order.
    pub fn records(&self) -> Vec<JobRecord> {
        self.jobs
            .iter()
            .map(|job| JobRecord {
                pgid: job.pgid,
                pids: job.pids.clone(),
                command: job.command.clone(),
                state: job.state.saved(),
            })
            .collect()
    }

    /// Re-registers the jobs that survived an exec and returns the records
    /// whose processes are gone. A restored stop was already reported by the
    /// previous image, so the sweep stays quiet about it.
    ///
    /// Every recorded pid must still be a live member of the recorded process
    /// group. `kill(pid, 0)` alone cannot tell a surviving job from a pid that
    /// was recycled after the exec, so a zombie is reaped first (it is dead)
    /// and the group of each live pid is checked. A residual window remains: a
    /// pid recycled into a process that happens to sit in a group with the same
    /// id would still be adopted. The blob is private (`0600`) and written
    /// immediately before the exec, and the only consequence is a stale table
    /// entry that `wait` reports as gone.
    pub fn restore(&mut self, records: Vec<JobRecord>) -> Vec<JobRecord> {
        let mut missing = Vec::new();
        for record in records {
            if record
                .pids
                .iter()
                .all(|pid| job_process_survived(*pid, record.pgid))
            {
                let state = JobState::restored(record.state);
                self.add(record.pgid, record.pids, record.command, state);
                if state == JobState::Stopped
                    && let Some(job) = self.jobs.last_mut()
                {
                    job.announced_stop = true;
                }
            } else {
                missing.push(record);
            }
        }
        missing
    }

    /// Kills a job the executor spawned but cannot use, reaps every member, and
    /// forgets it. Used when a later pipeline stage is refused: the started
    /// stage has no reader and no owner, and leaving it running would keep a
    /// process alive that the prompt never waited for.
    pub fn terminate(&mut self, id: usize) {
        let Some(job) = self.get(id) else {
            return;
        };
        let pgid = job.pgid;
        let pids = job.pids.clone();
        self.remove(id);
        if pgid > 0 {
            // SAFETY: SIGKILL to a process group this shell created. ESRCH
            // means the group is already gone.
            unsafe {
                libc::kill(-pgid, libc::SIGKILL);
            }
        }
        for pid in pids {
            loop {
                let mut status = 0;
                // SAFETY: blocking wait for one of this shell's own children;
                // SIGKILL makes it prompt, and `status` is a live local.
                let waited = unsafe { libc::waitpid(pid, &mut status, 0) };
                if waited != -1 {
                    break;
                }
                if io::Error::last_os_error().kind() != io::ErrorKind::Interrupted {
                    // ECHILD: already reaped, so there is nothing to collect.
                    break;
                }
            }
        }
    }

    /// Reaps every child that changed state without blocking, and reports the
    /// jobs that finished or stopped. Each transition is reported once.
    pub fn sweep(&mut self) -> Vec<Notice> {
        let mut reaped = false;
        loop {
            let mut status = 0;
            // SAFETY: waitpid(-1, ..., WNOHANG) inspects this process's own
            // children, writes one status into a live local, and never blocks.
            let pid = unsafe { libc::waitpid(-1, &mut status, libc::WNOHANG | libc::WUNTRACED) };
            if pid <= 0 {
                break;
            }
            reaped = true;
            self.record(pid, status);
        }
        if !reaped {
            return Vec::new();
        }
        self.collect_notices()
    }

    /// Waits for one job, with `WUNTRACED` so a stop returns to the prompt.
    ///
    /// `SIGINT`/`SIGQUIT` cannot be ignored here the way a foreground external
    /// job's wait ignores them: the shell owns the terminal while a builtin
    /// waits, so `Ctrl-C` must end the wait and leave the job running instead of
    /// ending the session. The guard in `exec` installs the handler; this loop
    /// turns the recorded signal into `Waited::Interrupted`.
    pub fn wait(&mut self, id: usize) -> Waited {
        let Some(index) = self.jobs.iter().position(|job| job.id == id) else {
            return Waited::Exited(127);
        };
        let pids = self.jobs[index].pids.clone();
        for pid in pids {
            if self.ended_of(pid).is_some() {
                continue;
            }
            loop {
                if let Some(signal) = signals::interrupted() {
                    return Waited::Interrupted(128 + signal);
                }
                if let Some(signal) = signals::terminating() {
                    return Waited::Terminated(128 + signal);
                }
                let mut status = 0;
                // SAFETY: blocking wait for a child this shell spawned;
                // `status` is a live local and WUNTRACED reports a stop.
                let waited = unsafe { libc::waitpid(pid, &mut status, libc::WUNTRACED) };
                if waited == pid {
                    self.record(pid, status);
                    break;
                }
                if waited == -1 {
                    let error = io::Error::last_os_error();
                    if error.kind() == io::ErrorKind::Interrupted {
                        // The top of the loop reports a recorded interrupt;
                        // anything else simply retries.
                        continue;
                    }
                    // ECHILD means someone already reaped it; treat the stage
                    // as finished instead of hanging on a pid that is gone.
                    self.mark_finished(pid);
                    break;
                }
            }
        }
        let Some(job) = self.jobs.iter().find(|job| job.id == id) else {
            return Waited::Exited(127);
        };
        if job.is_stopped() {
            return Waited::Stopped(128 + job.stop_signal());
        }
        match job.last_stage() {
            Some(Ended::Signaled(signal)) => Waited::Signaled(signal),
            Some(Ended::Stopped(signal)) => Waited::Stopped(128 + signal),
            Some(Ended::Exited(code)) => Waited::Exited(code),
            None => Waited::Exited(0),
        }
    }

    /// Resumes a stopped job in its own group and marks it running.
    pub fn continue_job(&mut self, id: usize) -> io::Result<()> {
        let Some(pgid) = self.get(id).map(|job| job.pgid) else {
            return Ok(());
        };
        // SAFETY: SIGCONT to a process group this shell created. ESRCH means
        // the job already died and the sweep will report it.
        if unsafe { libc::kill(-pgid, libc::SIGCONT) } != 0 {
            let error = io::Error::last_os_error();
            if error.raw_os_error() != Some(libc::ESRCH) {
                return Err(error);
            }
        }
        if let Some(job) = self.jobs.iter_mut().find(|job| job.id == id) {
            job.state = JobState::Running;
            job.announced_stop = false;
            // A recorded stop is stale once the group runs again.
            for ended in &mut job.ended {
                if matches!(ended, Some(Ended::Stopped(_))) {
                    *ended = None;
                }
            }
        }
        Ok(())
    }

    /// Marks a job stopped (after `fg` or a foreground wait saw a stop).
    pub fn mark_stopped(&mut self, id: usize) {
        if let Some(job) = self.jobs.iter_mut().find(|job| job.id == id) {
            job.state = JobState::Stopped;
            job.announced_stop = true;
        }
    }

    /// The `Stopped` notice printed when a foreground job stops.
    pub fn stop_notice(&self, id: usize) -> Option<Notice> {
        let job = self.get(id)?;
        Some(Notice {
            id,
            marker: self.marker(id),
            state: "Stopped".into(),
            command: job.command.clone(),
        })
    }

    fn ended_of(&self, pid: Pid) -> Option<Ended> {
        self.jobs.iter().find_map(|job| {
            match job.pids.iter().position(|candidate| *candidate == pid) {
                Some(index) => job.ended[index],
                None => None,
            }
        })
    }

    fn mark_finished(&mut self, pid: Pid) {
        for job in &mut self.jobs {
            if let Some(index) = job.pids.iter().position(|candidate| *candidate == pid)
                && job.ended[index].is_none()
            {
                job.ended[index] = Some(Ended::Exited(0));
            }
        }
    }

    /// Folds one `waitpid` status into the job that owns `pid`.
    fn record(&mut self, pid: Pid, status: i32) {
        for job in &mut self.jobs {
            let Some(index) = job.pids.iter().position(|candidate| *candidate == pid) else {
                continue;
            };
            if libc::WIFSTOPPED(status) {
                job.ended[index] = Some(Ended::Stopped(libc::WSTOPSIG(status)));
                job.state = JobState::Stopped;
            } else if libc::WIFCONTINUED(status) {
                job.state = JobState::Running;
                job.announced_stop = false;
            } else if libc::WIFSIGNALED(status) {
                job.ended[index] = Some(Ended::Signaled(libc::WTERMSIG(status)));
            } else if libc::WIFEXITED(status) {
                job.ended[index] = Some(Ended::Exited(libc::WEXITSTATUS(status)));
            }
            return;
        }
    }

    fn collect_notices(&mut self) -> Vec<Notice> {
        let mut notices = Vec::new();
        let mut index = 0;
        while index < self.jobs.len() {
            let job = &self.jobs[index];
            let notice = if job.finished() {
                Some(Notice {
                    id: job.id,
                    marker: self.marker(job.id),
                    state: outcome_text(job.last_stage()),
                    command: job.command.clone(),
                })
            } else if job.is_stopped() && !job.announced_stop {
                Some(Notice {
                    id: job.id,
                    marker: self.marker(job.id),
                    state: "Stopped".into(),
                    command: job.command.clone(),
                })
            } else {
                None
            };
            if let Some(notice) = notice {
                let id = notice.id;
                let finished = self.jobs[index].finished();
                notices.push(notice);
                if finished {
                    self.remove(id);
                    // `remove` shifts the next job into this index.
                    continue;
                }
                self.jobs[index].announced_stop = true;
            }
            index += 1;
        }
        notices
    }
}

fn outcome_text(outcome: Option<Ended>) -> String {
    match outcome {
        Some(Ended::Exited(0)) | None => "Done".into(),
        Some(Ended::Exited(code)) => format!("Exit {code}"),
        Some(Ended::Signaled(signal)) => format!("Terminated ({signal})"),
        Some(Ended::Stopped(_)) => "Stopped".into(),
    }
}

/// Whether one recorded job process is still the process the previous image
/// spawned: a live member of the recorded process group.
fn job_process_survived(pid: Pid, pgid: Pid) -> bool {
    if pid <= 0 || pgid <= 0 {
        return false;
    }
    let mut status = 0;
    // SAFETY: waitpid with WNOHANG|WUNTRACED inspects this process's own
    // children, never blocks, and writes one status into a live local. A
    // zombie is reaped here, before anything could mistake it for a live job.
    let waited = unsafe { libc::waitpid(pid, &mut status, libc::WNOHANG | libc::WUNTRACED) };
    if waited == pid {
        // A stop is a live process in the recorded group; an exit or a fatal
        // signal means the job died with the previous image.
        return libc::WIFSTOPPED(status) && same_group(pid, pgid);
    }
    // 0 means still running; -1 is ECHILD (already reaped, or a recycled pid
    // that is not our child). The group check settles both.
    same_group(pid, pgid)
}

/// Whether `pid` is a live member of process group `pgid`.
fn same_group(pid: Pid, pgid: Pid) -> bool {
    // SAFETY: getpgid only reads the group of `pid`; -1 means it is gone.
    (unsafe { libc::getpgid(pid) }) == pgid
}

/// Whether this process may claim the terminal: only the group the terminal
/// already treats as foreground. Anything else is a background shell, and
/// `tcsetpgrp` from there would take the terminal away from the job that owns
/// it.
fn may_claim_terminal(tty_foreground: Pid, shell_group: Pid) -> bool {
    tty_foreground > 0 && shell_group > 0 && tty_foreground == shell_group
}

/// The terminal's foreground process group for stdin, or -1 without a terminal.
fn tty_foreground() -> Pid {
    // SAFETY: tcgetpgrp only reads the terminal's foreground group.
    unsafe { libc::tcgetpgrp(libc::STDIN_FILENO) }
}

/// Prepares the shell for job control.
///
/// `SIGCHLD` is reset before anything can be spawned: a parent that ignored it
/// leaves the disposition set across `exec`, and then every `waitpid` fails with
/// `ECHILD` so a foreground command would report success the instant it
/// started. Terminal ownership is claimed only when this process already
/// belongs to the terminal's foreground group; a shell started in the
/// background reports that and leaves the terminal alone.
pub fn init() -> io::Result<()> {
    // SAFETY: SIG_DFL is the disposition the shell needs for its own children,
    // and it is installed before any child exists.
    unsafe {
        libc::signal(libc::SIGCHLD, libc::SIG_DFL);
    }
    if !crate::term::has_tty() {
        return Ok(());
    }
    let pid = unsafe { libc::getpid() };
    // Decide before moving to a new group: the question is whether the group
    // this process inherits is the one the terminal made foreground.
    // SAFETY: getpgrp reports this process's group and has no preconditions.
    if !may_claim_terminal(tty_foreground(), unsafe { libc::getpgrp() }) {
        // Leave SIGTTOU/SIGTTIN at their defaults: a later read or terminal
        // change then stops this process with SIGTTIN/SIGTTOU, which is the
        // safe outcome for a background shell. Taking the terminal here would
        // steal it from the job that owns it.
        eprintln!(
            "piku shell: not the terminal's foreground process group; not taking the terminal"
        );
        return Ok(());
    }
    for signal in [libc::SIGTTOU, libc::SIGTTIN, libc::SIGTSTP] {
        // SAFETY: installs a process-wide disposition; SIG_IGN is valid and
        // the shell must never stop or read from a terminal it does not own.
        unsafe {
            libc::signal(signal, libc::SIG_IGN);
        }
    }
    // A shell that is not a process-group leader takes its own group so that
    // a foreground job can be given a different one.
    // SAFETY: setpgid(0, 0) only acts on this process; failure leaves the
    // inherited group, which still works.
    if unsafe { libc::getpgrp() } != pid {
        unsafe {
            libc::setpgid(0, 0);
        }
    }
    // SAFETY: getpgrp reports this process's group and has no preconditions.
    SHELL_PGID.store(unsafe { libc::getpgrp() }, Ordering::SeqCst);
    TERMINAL_OWNED.store(true, Ordering::SeqCst);
    take_terminal()
}

pub fn shell_process_group() -> Pid {
    SHELL_PGID.load(Ordering::SeqCst)
}

/// Hands the terminal to `pgid` while SIGTTOU is ignored, as a shell must. A
/// shell that never owned the terminal leaves the foreground job alone.
pub fn give_terminal(pgid: Pid) -> io::Result<()> {
    if !TERMINAL_OWNED.load(Ordering::SeqCst) {
        return Ok(());
    }
    set_terminal_group(pgid)
}

/// Takes the terminal back for the shell.
pub fn take_terminal() -> io::Result<()> {
    if !TERMINAL_OWNED.load(Ordering::SeqCst) {
        return Ok(());
    }
    let pgid = shell_process_group();
    if pgid == 0 {
        return Ok(());
    }
    set_terminal_group(pgid)
}

fn set_terminal_group(pgid: Pid) -> io::Result<()> {
    loop {
        // SAFETY: tcsetpgrp changes the terminal's foreground group on stdin;
        // SIGTTOU is ignored so a background shell may still reclaim it.
        if unsafe { libc::tcsetpgrp(libc::STDIN_FILENO, pgid) } == 0 {
            return Ok(());
        }
        let error = io::Error::last_os_error();
        if error.kind() == io::ErrorKind::Interrupted {
            continue;
        }
        // Without a terminal there is nothing to hand over.
        if error.raw_os_error() == Some(libc::ENOTTY) {
            return Ok(());
        }
        return Err(error);
    }
}

/// Terminal ownership for the life of one foreground job. Dropping it restores
/// the shell's group on every path, including a stop and a panic.
pub struct Foreground {
    _private: (),
}

impl Foreground {
    pub fn new(pgid: Pid) -> io::Result<Self> {
        give_terminal(pgid)?;
        Ok(Self { _private: () })
    }
}

impl Drop for Foreground {
    fn drop(&mut self) {
        let _ = take_terminal();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table_with(pids: &[Pid], command: &str) -> (JobTable, usize) {
        let mut table = JobTable::new();
        let id = table.add(pids[0], pids.to_vec(), command.into(), JobState::Running);
        (table, id)
    }

    #[test]
    fn job_numbers_start_at_one_and_marks_track_current_and_previous() {
        let mut table = JobTable::new();
        let first = table.add(100, vec![100], "sleep 1".into(), JobState::Running);
        let second = table.add(200, vec![200], "sleep 2".into(), JobState::Running);
        assert_eq!(first, 1);
        assert_eq!(second, 2);
        assert_eq!(table.marker(first), '-');
        assert_eq!(table.marker(second), '+');

        table.remove(second);
        assert_eq!(table.marker(first), '+');
        assert_eq!(table.list().len(), 1);
    }

    #[test]
    fn specs_resolve_the_default_previous_number_and_substring() {
        let mut table = JobTable::new();
        let first = table.add(100, vec![100], "make build".into(), JobState::Running);
        let second = table.add(200, vec![200], "sleep 30".into(), JobState::Stopped);

        assert_eq!(table.resolve("%+"), Some(second));
        assert_eq!(table.resolve("%%"), Some(second));
        assert_eq!(table.resolve("%"), Some(second));
        assert_eq!(table.resolve("%-"), Some(first));
        assert_eq!(table.resolve("%1"), Some(first));
        assert_eq!(table.resolve("%?make"), Some(first));
        assert_eq!(table.resolve("%?nothing"), None);
        assert_eq!(table.resolve("%9"), None);
        assert_eq!(table.resolve("1"), None, "a bare number is not a job spec");
    }

    #[test]
    fn a_finished_job_is_reported_once_and_removed() {
        let mut table = JobTable::new();
        let id = table.add(100, vec![100], "sleep 1".into(), JobState::Running);
        // A zero wait status is "exited with 0" on both Linux and macOS.
        table.record(100, 0);
        let notices = table.collect_notices();
        assert_eq!(notices.len(), 1);
        assert_eq!(notices[0].id, id);
        assert_eq!(notices[0].state, "Done");
        assert!(table.is_empty());
        assert!(table.collect_notices().is_empty(), "reported only once");
    }

    #[test]
    fn a_stop_is_reported_once_until_the_job_runs_again() {
        let mut table = JobTable::new();
        let id = table.add(100, vec![100], "vim".into(), JobState::Running);
        let stopped = libc::SIGTSTP << 8 | 0x7f;
        table.record(100, stopped);
        assert_eq!(table.collect_notices().len(), 1);
        assert!(table.collect_notices().is_empty());
        assert_eq!(table.get(id).map(|job| job.state), Some(JobState::Stopped));
        assert_eq!(
            table.stop_notice(id).map(|n| n.state),
            Some("Stopped".into())
        );
    }

    #[test]
    fn records_round_trip_through_the_handoff_shape() {
        let (mut table, first) = table_with(&[100, 101], "cat | tr a-z A-Z");
        let second = table.add(200, vec![200], "sleep 30".into(), JobState::Stopped);
        let records = table.records();
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].pgid, 100);
        assert_eq!(records[0].pids, vec![100, 101]);
        assert_eq!(records[1].state, SavedJobState::Stopped);

        // A pid far above any real one is never live, so both records come
        // back as missing.
        let dead = 2_000_000;
        let missing = table.restore(vec![
            JobRecord {
                pgid: 100,
                pids: vec![dead],
                command: "cat | tr a-z A-Z".into(),
                state: SavedJobState::Running,
            },
            JobRecord {
                pgid: 200,
                pids: vec![dead],
                command: "sleep 30".into(),
                state: SavedJobState::Stopped,
            },
        ]);
        assert_eq!(missing.len(), 2);
        assert_eq!(table.resolve("%1"), Some(first));
        assert!(
            table.get(second).is_some(),
            "restore adds, it does not reset"
        );
        assert_eq!(table.list().len(), 2);
    }

    #[test]
    fn restore_adopts_a_live_group_and_drops_a_dead_or_recycled_pid() {
        use std::os::unix::process::CommandExt;

        let mut table = JobTable::new();
        // A real child in its own process group, like a job the previous image
        // spawned and the exec preserved.
        let mut child = std::process::Command::new("/bin/sleep")
            .arg("30")
            .process_group(0)
            .spawn()
            .expect("spawn a job");
        let pgid = child.id() as Pid;
        let live = JobRecord {
            pgid,
            pids: vec![pgid],
            command: "sleep 30".into(),
            state: SavedJobState::Running,
        };

        assert!(table.restore(vec![live.clone()]).is_empty());
        let id = table.resolve("%1").expect("the restored job");
        assert_eq!(table.get(id).map(|job| job.pgid), Some(pgid));

        // The same live pid in a group that is not its own is a recycled pid,
        // not this job, so the record is dropped rather than adopted.
        let recycled = JobRecord {
            pgid: pgid + 1,
            ..live.clone()
        };
        assert_eq!(table.restore(vec![recycled]).len(), 1);
        assert_eq!(table.list().len(), 1, "a mismatched record adds nothing");

        // Once reaped, the pid no longer names a job.
        let _ = child.kill();
        let _ = child.wait();
        assert_eq!(table.restore(vec![live]).len(), 1);
        assert_eq!(table.list().len(), 1);
    }

    #[test]
    fn only_the_foreground_group_may_claim_the_terminal() {
        assert!(may_claim_terminal(42, 42));
        assert!(
            !may_claim_terminal(42, 43),
            "a background shell must not take the terminal"
        );
        assert!(!may_claim_terminal(-1, 42), "no controlling terminal");
        assert!(!may_claim_terminal(0, 0), "group 0 never names a shell");

        // The test runner's stdin is not a terminal, so the real probe reports
        // the same "do not claim" answer.
        if !crate::term::has_tty() {
            assert!(tty_foreground() < 0, "no terminal has no foreground group");
            // SAFETY: getpgrp reports this process's group.
            assert!(!may_claim_terminal(tty_foreground(), unsafe {
                libc::getpgrp()
            }));
        }
    }
}
