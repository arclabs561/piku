//! Terminal modes and byte output.
//!
//! The shell owns the terminal while it reads a line, so it has to switch
//! between raw mode (one keystroke at a time, no echo) and the caller's cooked
//! mode (for child commands). The cooked attributes are captured once at
//! startup and restored on every exit path, including panics.

use std::io::{self, Write};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

/// The attributes to return to. `None` before the first `Terminal` exists.
static COOKED: Mutex<Option<libc::termios>> = Mutex::new(None);
/// Whether the terminal is currently in raw mode.
static RAW_ACTIVE: AtomicBool = AtomicBool::new(false);

pub fn has_tty() -> bool {
    // SAFETY: isatty only inspects the descriptor.
    unsafe { libc::isatty(libc::STDIN_FILENO) == 1 }
}

/// Reads the current terminal attributes for `fd`.
fn get(fd: libc::c_int) -> io::Result<libc::termios> {
    // SAFETY: tcgetattr writes into the supplied termios on success only; the
    // value starts zeroed and is owned by this function.
    let mut attributes: libc::termios = unsafe { std::mem::zeroed() };
    // SAFETY: `attributes` is a valid, writable termios.
    if unsafe { libc::tcgetattr(fd, &mut attributes) } == -1 {
        return Err(io::Error::last_os_error());
    }
    Ok(attributes)
}

/// Applies terminal attributes to `fd` immediately, without flushing input.
fn set(fd: libc::c_int, attributes: &libc::termios) -> io::Result<()> {
    loop {
        // SAFETY: TCSANOW applies the supplied, valid termios in place.
        if unsafe { libc::tcsetattr(fd, libc::TCSANOW, attributes) } == 0 {
            return Ok(());
        }
        let error = io::Error::last_os_error();
        if error.kind() == io::ErrorKind::Interrupted {
            continue;
        }
        return Err(error);
    }
}

/// Restores the cooked attributes recorded at startup, if raw mode is active.
fn restore_saved() -> io::Result<()> {
    let Some(cooked) = COOKED.lock().ok().and_then(|slot| *slot) else {
        return Ok(());
    };
    set(libc::STDIN_FILENO, &cooked)
}

/// Restores the terminal before printing a panic, in addition to the usual
/// hook. `Drop` already covers unwinding, but the hook runs first and keeps the
/// panic message readable.
pub fn install_panic_hook() {
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        if RAW_ACTIVE.load(Ordering::SeqCst) && restore_saved().is_ok() {
            RAW_ACTIVE.store(false, Ordering::SeqCst);
        }
        previous(info);
    }));
}

/// Owns the cooked terminal attributes and the raw-mode flag.
pub struct Terminal {
    cooked: libc::termios,
}

impl Terminal {
    /// Captures the caller's terminal state. Requires a terminal on stdin.
    pub fn new() -> io::Result<Self> {
        let cooked = get(libc::STDIN_FILENO)?;
        if let Ok(mut slot) = COOKED.lock() {
            *slot = Some(cooked);
        }
        Ok(Self { cooked })
    }

    /// Switches stdin to raw mode (`cfmakeraw`, applied immediately).
    pub fn enter_raw(&mut self) -> io::Result<()> {
        let mut raw = self.cooked;
        // SAFETY: cfmakeraw initializes every field it changes in place.
        unsafe { libc::cfmakeraw(&mut raw) };
        set(libc::STDIN_FILENO, &raw)?;
        RAW_ACTIVE.store(true, Ordering::SeqCst);
        Ok(())
    }

    /// Returns to the cooked attributes. Safe to call repeatedly. The active
    /// flag is cleared only after the kernel accepts the change, so a failed
    /// restore can still be retried by `Drop` or the panic hook.
    pub fn restore(&mut self) -> io::Result<()> {
        if !RAW_ACTIVE.load(Ordering::SeqCst) {
            return Ok(());
        }
        set(libc::STDIN_FILENO, &self.cooked)?;
        RAW_ACTIVE.store(false, Ordering::SeqCst);
        Ok(())
    }
}

impl Drop for Terminal {
    fn drop(&mut self) {
        let _ = self.restore();
    }
}

/// Writes bytes to stdout and flushes them. Every write is flushed because the
/// shell must never leave a prompt or a prompt edit sitting in a userspace
/// buffer, especially across `exec`.
pub fn write(bytes: &[u8]) -> io::Result<()> {
    let mut out = io::stdout().lock();
    out.write_all(bytes)?;
    out.flush()
}
