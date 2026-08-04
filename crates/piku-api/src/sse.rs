/// Minimal SSE line parser.
///
/// Yields complete `(event_type, data)` pairs from a stream of raw bytes.
/// Handles the `event:` and `data:` fields; ignores `id:` and `retry:`.
#[derive(Debug, Default)]
pub struct SseParser {
    event_type: Option<String>,
    data_buf: String,
}

/// Decodes response bytes into text across chunk boundaries.
///
/// `reqwest` chunks at transport boundaries, which can land mid-codepoint.
/// Decoding each chunk on its own with `from_utf8_lossy` turned the split
/// character into U+FFFD, and because that is valid UTF-8 inside a JSON
/// string the payload still parsed: the corruption reached the terminal, the
/// session file, and the next request's history without any error. Any
/// non-ASCII model output was exposed. The trailing incomplete sequence is
/// carried into the next chunk instead.
#[derive(Debug, Default)]
pub struct Utf8Stream {
    tail: Vec<u8>,
}

impl Utf8Stream {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Decode one chunk, holding back any incomplete trailing character.
    ///
    /// Only a truncated sequence at the very end is held; corrupt bytes in the
    /// middle become U+FFFD and decoding continues, so one bad byte cannot
    /// stall every token behind it.
    pub fn push(&mut self, chunk: &[u8]) -> String {
        self.tail.extend_from_slice(chunk);
        let mut out = String::new();
        loop {
            match std::str::from_utf8(&self.tail) {
                Ok(text) => {
                    out.push_str(text);
                    self.tail.clear();
                    return out;
                }
                Err(error) => {
                    let valid = error.valid_up_to();
                    out.push_str(&String::from_utf8_lossy(&self.tail[..valid]));
                    // `Some` means corrupt: replace it and keep going. `None`
                    // means truncated, so the rest arrives in the next chunk.
                    let Some(bad) = error.error_len() else {
                        self.tail.drain(..valid);
                        return out;
                    };
                    out.push('\u{FFFD}');
                    self.tail.drain(..valid + bad);
                }
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct SseEvent {
    pub event_type: Option<String>,
    pub data: String,
}

impl SseParser {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed a single line from the HTTP response body.
    /// Returns `Some(SseEvent)` when a complete event is ready (blank line dispatch).
    pub fn feed_line(&mut self, line: &str) -> Option<SseEvent> {
        let line = line.trim_end_matches('\r');

        if line.is_empty() {
            // blank line = dispatch
            if self.data_buf.is_empty() {
                return None;
            }
            let data = std::mem::take(&mut self.data_buf);
            // strip trailing newline from data
            let data = data.strip_suffix('\n').unwrap_or(&data).to_string();
            let event = SseEvent {
                event_type: self.event_type.take(),
                data,
            };
            return Some(event);
        }

        if let Some(value) = line.strip_prefix("event:") {
            self.event_type = Some(value.trim_start().to_string());
        } else if let Some(value) = line.strip_prefix("data:") {
            self.data_buf.push_str(value.trim_start());
            self.data_buf.push('\n');
        }
        // ignore id:, retry:, comments (:)

        None
    }

    /// Flush any buffered data as a final event. Call this after the stream
    /// ends to handle the case where the last event lacks the trailing blank
    /// line — common when TCP connections close mid-terminator (Cloudflare,
    /// proxy truncation) and would otherwise silently drop `data: [DONE]`.
    pub fn finish(&mut self) -> Option<SseEvent> {
        if self.data_buf.is_empty() {
            return None;
        }
        let data = std::mem::take(&mut self.data_buf);
        let data = data.strip_suffix('\n').unwrap_or(&data).to_string();
        Some(SseEvent {
            event_type: self.event_type.take(),
            data,
        })
    }
}
