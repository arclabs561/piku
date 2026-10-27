//! Decodes terminal bytes into editing keys.
//!
//! Input is read one byte at a time and nothing is read ahead of the key being
//! handled. That is deliberate: `:upgrade` replaces the process with `exec`,
//! and any byte already read into userspace would be lost across the swap.

use std::io;

/// Keys the line editor understands. Everything else is discarded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Key {
    Char(char),
    Enter,
    Backspace,
    Left,
    Right,
    Home,
    End,
    Up,
    Down,
    /// Ctrl-U: clear the whole line.
    Clear,
    /// Tab: complete the current word.
    Complete,
    /// Ctrl-C: clear the line, never exit.
    Interrupt,
    /// Ctrl-D: exit only when the line is empty (decided by the caller).
    Eof,
    /// Ctrl-X Ctrl-U: replace the process, keeping the current line.
    Upgrade,
    /// No key arrived within the poll window; the caller checks signals and
    /// the job table before waiting again.
    Idle,
    Esc,
    Other,
}

/// How long to wait for the rest of an escape sequence after ESC.
const ESCAPE_TIMEOUT_MS: i32 = 30;
/// How long the prompt wait for a key lasts before the caller gets a chance to
/// sweep jobs and drain the signal pipe. Short enough to notice a resize or a
/// terminated job promptly, long enough to cost nothing while idle.
const IDLE_TIMEOUT_MS: i32 = 200;

pub struct Input;

impl Input {
    pub fn new() -> Self {
        Self
    }

    /// Reads one key, or `Idle` when nothing arrived within the poll window.
    /// `None` means stdin reached end of file.
    pub fn read_key(&mut self) -> io::Result<Option<Key>> {
        if !readable(IDLE_TIMEOUT_MS)? {
            return Ok(Some(Key::Idle));
        }
        let Some(byte) = read_byte()? else {
            return Ok(None);
        };
        Ok(Some(match byte {
            0x1b => escape()?,
            b'\t' => Key::Complete,
            b'\r' | b'\n' => Key::Enter,
            0x7f | 0x08 => Key::Backspace,
            0x01 => Key::Home,
            0x05 => Key::End,
            0x15 => Key::Clear,
            0x03 => Key::Interrupt,
            0x04 => Key::Eof,
            0x18 => {
                // Ctrl-X is a prefix. Wait briefly so a lone Ctrl-X cannot
                // swallow the next keystroke.
                if !readable(ESCAPE_TIMEOUT_MS)? {
                    Key::Other
                } else {
                    match read_byte()? {
                        Some(0x15) => Key::Upgrade,
                        Some(0x03) => Key::Interrupt,
                        Some(0x04) => Key::Eof,
                        _ => Key::Other,
                    }
                }
            }
            0x00..=0x1f => Key::Other,
            first => utf8_char(first)?,
        }))
    }
}

/// Reads one byte, retrying on EINTR. `None` is end of file.
fn read_byte() -> io::Result<Option<u8>> {
    loop {
        let mut byte = 0u8;
        // SAFETY: read writes at most one byte into `byte`, a live local.
        let count = unsafe { libc::read(libc::STDIN_FILENO, (&raw mut byte).cast(), 1) };
        if count == 1 {
            return Ok(Some(byte));
        }
        if count == 0 {
            return Ok(None);
        }
        let error = io::Error::last_os_error();
        if error.kind() == io::ErrorKind::Interrupted {
            continue;
        }
        return Err(error);
    }
}

/// Waits up to `timeout_ms` for another byte on stdin. A signal makes poll
/// return `EINTR`; reporting "not ready" lets the caller drain the self-pipe
/// instead of blocking through the notification.
fn readable(timeout_ms: i32) -> io::Result<bool> {
    let mut descriptor = libc::pollfd {
        fd: libc::STDIN_FILENO,
        events: libc::POLLIN,
        revents: 0,
    };
    // SAFETY: poll reads and writes exactly one initialized pollfd.
    let ready = unsafe { libc::poll(&mut descriptor, 1, timeout_ms) };
    if ready >= 0 {
        return Ok(ready > 0);
    }
    let error = io::Error::last_os_error();
    if error.kind() == io::ErrorKind::Interrupted {
        return Ok(false);
    }
    Err(error)
}

/// Decodes a CSI or SS3 sequence after ESC. A lone ESC is discarded (the slice
/// has no Escape binding), and an unrecognized sequence is discarded too.
fn escape() -> io::Result<Key> {
    if !readable(ESCAPE_TIMEOUT_MS)? {
        return Ok(Key::Esc);
    }
    let Some(next) = read_byte()? else {
        return Ok(Key::Esc);
    };
    match next {
        b'[' => csi(),
        b'O' => {
            // Application cursor keys: ESC O A..D/H/F.
            Ok(match read_byte()? {
                Some(b'A') => Key::Up,
                Some(b'B') => Key::Down,
                Some(b'C') => Key::Right,
                Some(b'D') => Key::Left,
                Some(b'H') => Key::Home,
                Some(b'F') => Key::End,
                _ => Key::Other,
            })
        }
        _ => Ok(Key::Esc),
    }
}

/// Consumes a CSI sequence (parameters then one final byte) and maps it.
fn csi() -> io::Result<Key> {
    let mut parameters = Vec::with_capacity(4);
    loop {
        let Some(byte) = read_byte()? else {
            return Ok(Key::Other);
        };
        parameters.push(byte);
        if parameters.len() > 16 {
            return Ok(Key::Other);
        }
        // 0x40..=0x7e terminates a CSI sequence.
        if (0x40..=0x7e).contains(&byte) {
            break;
        }
    }
    Ok(match parameters.as_slice() {
        b"A" => Key::Up,
        b"B" => Key::Down,
        b"C" => Key::Right,
        b"D" => Key::Left,
        b"H" | b"1~" | b"7~" => Key::Home,
        b"F" | b"4~" | b"8~" => Key::End,
        _ => Key::Other,
    })
}

/// Reads the continuation bytes of one UTF-8 character. Invalid input is
/// dropped rather than inserted.
fn utf8_char(first: u8) -> io::Result<Key> {
    let len = match first {
        0x20..=0x7e => return Ok(Key::Char(first as char)),
        0xc2..=0xdf => 2,
        0xe0..=0xef => 3,
        0xf0..=0xf4 => 4,
        _ => return Ok(Key::Other),
    };
    let mut bytes = [0u8; 4];
    bytes[0] = first;
    for slot in bytes.iter_mut().take(len).skip(1) {
        match read_byte()? {
            Some(byte) => *slot = byte,
            None => return Ok(Key::Other),
        }
    }
    Ok(std::str::from_utf8(&bytes[..len])
        .ok()
        .and_then(|text| text.chars().next())
        .map_or(Key::Other, Key::Char))
}
