//! Async-signal-safe signal delivery through a self-pipe.
//!
//! The handlers do exactly two things: store the signal number in an atomic and
//! write one byte to a non-blocking pipe. Everything else — restoring the
//! terminal, exiting, redrawing — happens in the main loop after it drains the
//! pipe, where ordinary code is safe to run.
//!
//! `SIGTERM` and `SIGHUP` ask the shell to leave with `128 + signal`;
//! `SIGWINCH` asks for a redraw. `Ctrl-Z` is not handled here: it stays a
//! terminal signal for the foreground job (see `jobs.rs`).
//!
//! `SIGINT` and `SIGQUIT` are installed only for the duration of in-process
//! work (`InterruptGuard`): a builtin runs while the shell is still the
//! terminal's foreground group, so without the guard `Ctrl-C` would kill the
//! session instead of aborting the builtin or `wait`.

use std::io;
use std::sync::atomic::{AtomicI32, Ordering};

/// Read end of the self-pipe, or -1 before `install`.
static PIPE_READ: AtomicI32 = AtomicI32::new(-1);
/// Write end of the self-pipe, read by the handler.
static PIPE_WRITE: AtomicI32 = AtomicI32::new(-1);
/// The signal that asked the shell to exit, or 0.
static TERMINATING: AtomicI32 = AtomicI32::new(0);
/// The last `SIGINT`/`SIGQUIT` that arrived while an `InterruptGuard` was
/// installed, or 0. Read by the wait loop, which cannot drain the pipe without
/// stealing the main loop's event.
static INTERRUPTED: AtomicI32 = AtomicI32::new(0);

/// The handler: atomics and one non-blocking `write` only.
extern "C" fn handler(signal: libc::c_int) {
    if matches!(signal, libc::SIGTERM | libc::SIGHUP) {
        TERMINATING.store(signal, Ordering::SeqCst);
    }
    notify(signal);
}

/// Writes one byte to the self-pipe. The only async-signal-safe way to wake the
/// main loop.
fn notify(signal: libc::c_int) {
    let descriptor = PIPE_WRITE.load(Ordering::Relaxed);
    if descriptor >= 0 {
        let byte = [signal as u8];
        // SAFETY: `write` is async-signal-safe, the buffer is a live local, and
        // the pipe is non-blocking. A full pipe drops a notification, never a
        // consistency guarantee.
        unsafe {
            libc::write(descriptor, byte.as_ptr().cast(), 1);
        }
    }
}

/// The handler for the signals that abort in-process work.
extern "C" fn interrupt_handler(signal: libc::c_int) {
    INTERRUPTED.store(signal, Ordering::SeqCst);
    notify(signal);
}

/// The interrupt that arrived while the shell ran something in process, as a
/// signal number, or `None`. Stays set until the executor clears it.
pub fn interrupted() -> Option<i32> {
    match INTERRUPTED.load(Ordering::SeqCst) {
        0 => None,
        signal => Some(signal),
    }
}

pub fn clear_interrupt() {
    INTERRUPTED.store(0, Ordering::SeqCst);
}

/// Makes `SIGINT` and `SIGQUIT` abort in-process work instead of the shell.
///
/// Installed around a builtin: the shell still owns the terminal and is its
/// foreground group, so the terminal delivers `Ctrl-C` here and a default
/// disposition would end the session. The handler only records the signal, so a
/// blocked `waitpid` in `wait`/`fg` returns `EINTR`, notices the flag, and
/// returns to the prompt. Dropping the guard restores the exact previous
/// dispositions (which may themselves be `SIG_IGN`, inherited from a parent).
pub struct InterruptGuard {
    saved: [(libc::c_int, libc::sigaction); 2],
}

impl InterruptGuard {
    pub fn new() -> io::Result<Self> {
        clear_interrupt();
        // SAFETY: `libc::sigaction` is a plain C struct of integers and a
        // pointer; all-zero is a valid placeholder that every successful
        // `sigaction` call below overwrites with the previous action.
        let mut saved = [(0, unsafe { std::mem::zeroed() }); 2];
        for (index, signal) in [libc::SIGINT, libc::SIGQUIT].into_iter().enumerate() {
            // SAFETY: same placeholder contract as above.
            let mut action: libc::sigaction = unsafe { std::mem::zeroed() };
            action.sa_sigaction = interrupt_handler as *const () as libc::sighandler_t;
            // No SA_RESTART: a blocked `waitpid` must return `EINTR`.
            action.sa_flags = 0;
            // SAFETY: `action.sa_mask` is zeroed memory of the right type and
            // `sigemptyset` only initializes it.
            unsafe {
                libc::sigemptyset(&mut action.sa_mask);
            }
            // SAFETY: installs `interrupt_handler`, which only stores an atomic
            // and writes to the self-pipe, and records the previous action in
            // the live slot `saved[index].1`.
            if unsafe { libc::sigaction(signal, &action, &mut saved[index].1) } != 0 {
                let error = io::Error::last_os_error();
                for (installed, previous) in saved.iter().take(index) {
                    // SAFETY: each entry was filled by a successful sigaction.
                    unsafe {
                        libc::sigaction(*installed, previous, std::ptr::null_mut());
                    }
                }
                clear_interrupt();
                return Err(error);
            }
            saved[index].0 = signal;
        }
        Ok(Self { saved })
    }
}

impl Drop for InterruptGuard {
    fn drop(&mut self) {
        for (signal, previous) in self.saved.iter() {
            if *signal == 0 {
                continue;
            }
            // SAFETY: `previous` is the action a successful `sigaction` call
            // recorded for `signal` in `new`.
            unsafe {
                libc::sigaction(*signal, previous, std::ptr::null_mut());
            }
        }
        clear_interrupt();
    }
}

/// Installs the handlers and creates the self-pipe. Idempotent.
pub fn install() -> io::Result<()> {
    if PIPE_READ.load(Ordering::SeqCst) >= 0 {
        return Ok(());
    }
    let mut descriptors = [0 as libc::c_int; 2];
    // SAFETY: `pipe` writes two descriptors into the array.
    if unsafe { libc::pipe(descriptors.as_mut_ptr()) } != 0 {
        return Err(io::Error::last_os_error());
    }
    for descriptor in descriptors {
        set_nonblocking(descriptor)?;
        set_cloexec(descriptor)?;
    }
    PIPE_READ.store(descriptors[0], Ordering::SeqCst);
    PIPE_WRITE.store(descriptors[1], Ordering::SeqCst);

    for signal in [libc::SIGTERM, libc::SIGHUP, libc::SIGWINCH] {
        // SAFETY: installs `handler`, which only touches atomics and the pipe;
        // the mask is empty and no SA_RESTART requests a retry, so a blocked
        // read or wait returns EINTR and the main loop can notice the pipe.
        unsafe {
            let mut action: libc::sigaction = std::mem::zeroed();
            action.sa_sigaction = handler as *const () as libc::sighandler_t;
            action.sa_flags = 0;
            libc::sigemptyset(&mut action.sa_mask);
            if libc::sigaction(signal, &action, std::ptr::null_mut()) != 0 {
                return Err(io::Error::last_os_error());
            }
        }
    }
    Ok(())
}

fn set_nonblocking(descriptor: libc::c_int) -> io::Result<()> {
    // SAFETY: fcntl reads and then writes flags on an owned descriptor.
    let flags = unsafe { libc::fcntl(descriptor, libc::F_GETFL) };
    if flags == -1 {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: same descriptor, flags read a moment ago.
    if unsafe { libc::fcntl(descriptor, libc::F_SETFL, flags | libc::O_NONBLOCK) } == -1 {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}

fn set_cloexec(descriptor: libc::c_int) -> io::Result<()> {
    // SAFETY: fcntl reads and then writes flags on an owned descriptor.
    let flags = unsafe { libc::fcntl(descriptor, libc::F_GETFD) };
    if flags == -1 {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: same descriptor; FD_CLOEXEC keeps the pipe out of children and
    // out of the replacement image, which installs its own.
    if unsafe { libc::fcntl(descriptor, libc::F_SETFD, flags | libc::FD_CLOEXEC) } == -1 {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}

/// Drains the self-pipe and returns the signals received since the last call.
pub fn drain() -> Vec<i32> {
    let descriptor = PIPE_READ.load(Ordering::SeqCst);
    if descriptor < 0 {
        return Vec::new();
    }
    let mut signals = Vec::new();
    loop {
        let mut byte = 0u8;
        // SAFETY: non-blocking read from the self-pipe into one live byte.
        let count = unsafe { libc::read(descriptor, (&raw mut byte).cast(), 1) };
        if count == 1 {
            signals.push(i32::from(byte));
            continue;
        }
        if count == -1 && io::Error::last_os_error().kind() == io::ErrorKind::Interrupted {
            continue;
        }
        break;
    }
    signals
}

/// The terminating signal, if SIGTERM or SIGHUP arrived. Readable from a wait
/// loop, which cannot drain the pipe without stealing the main loop's event.
pub fn terminating() -> Option<i32> {
    match TERMINATING.load(Ordering::SeqCst) {
        0 => None,
        signal => Some(signal),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_handler_byte_drains_once() {
        // Use the pipe directly so the test does not deliver real signals to
        // the test runner.
        install().expect("install the self-pipe");
        let descriptor = PIPE_WRITE.load(Ordering::SeqCst);
        let byte = [libc::SIGWINCH as u8];
        // SAFETY: writes to the pipe this module owns.
        let written = unsafe { libc::write(descriptor, byte.as_ptr().cast(), 1) };
        assert_eq!(written, 1);
        assert_eq!(drain(), vec![libc::SIGWINCH]);
        assert!(drain().is_empty(), "the pipe must be drained");
    }

    #[test]
    fn the_interrupt_guard_records_a_signal_and_restores_the_disposition() {
        use std::time::{Duration, Instant};

        clear_interrupt();
        // SAFETY: SIG_IGN is a valid disposition and the test puts the
        // harness's own disposition back at the end.
        let previous = unsafe { libc::signal(libc::SIGINT, libc::SIG_IGN) };
        {
            let _guard = InterruptGuard::new().expect("install the interrupt guard");
            // SAFETY: targets this thread only, so no other test sees it.
            unsafe {
                libc::pthread_kill(libc::pthread_self(), libc::SIGINT);
            }
            let deadline = Instant::now() + Duration::from_secs(1);
            while interrupted().is_none() {
                assert!(
                    Instant::now() < deadline,
                    "the interrupt handler never recorded SIGINT"
                );
                std::thread::sleep(Duration::from_millis(1));
            }
            assert_eq!(interrupted(), Some(libc::SIGINT));
        }
        assert_eq!(interrupted(), None, "the guard clears the interrupt");

        // The previous disposition is back: the inherited ignore swallows the
        // signal instead of recording it.
        // SAFETY: targets this thread only.
        unsafe {
            libc::pthread_kill(libc::pthread_self(), libc::SIGINT);
        }
        assert_eq!(interrupted(), None);
        // SAFETY: restores the disposition the harness had on entry.
        unsafe {
            libc::signal(libc::SIGINT, previous);
        }
    }
}
