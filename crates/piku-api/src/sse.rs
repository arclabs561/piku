/// Minimal SSE line parser.
///
/// Yields complete `(event_type, data)` pairs from a stream of raw bytes.
/// Handles the `event:` and `data:` fields; ignores `id:` and `retry:`.
#[derive(Debug, Default)]
pub struct SseParser {
    event_type: Option<String>,
    data_buf: String,
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
}
