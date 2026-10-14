use std::collections::HashMap;
use std::pin::Pin;

use futures_util::{Stream, StreamExt};
use serde::Deserialize;

use crate::error::ApiError;
use crate::provider::Provider;
use crate::sse::SseParser;
use crate::types::{Event, MessageRequest, StopReason, TokenUsage};

const DEFAULT_BASE_URL: &str = "https://api.anthropic.com";
const ANTHROPIC_VERSION: &str = "2023-06-01";
const ANTHROPIC_BETA: &str = "prompt-caching-2024-07-31";

#[derive(Clone)]
pub struct AnthropicProvider {
    api_key: String,
    base_url: String,
    client: reqwest::Client,
}

impl AnthropicProvider {
    #[must_use]
    pub fn new(api_key: String) -> Self {
        Self::with_base_url(api_key, DEFAULT_BASE_URL.to_string())
    }

    #[must_use]
    pub fn with_base_url(api_key: String, base_url: String) -> Self {
        Self {
            api_key,
            base_url,
            client: reqwest::Client::new(),
        }
    }

    pub fn from_env() -> Result<Self, ApiError> {
        let api_key = std::env::var("ANTHROPIC_API_KEY").map_err(|_| ApiError::MissingApiKey)?;
        let base_url =
            std::env::var("ANTHROPIC_BASE_URL").unwrap_or_else(|_| DEFAULT_BASE_URL.to_string());
        Ok(Self::with_base_url(api_key, base_url))
    }
}

impl Provider for AnthropicProvider {
    fn name(&self) -> &'static str {
        "anthropic"
    }

    fn boxed_clone(&self) -> Box<dyn Provider + Send + Sync + 'static> {
        Box::new(self.clone())
    }

    fn stream_message(
        &self,
        request: MessageRequest,
    ) -> Pin<Box<dyn Stream<Item = Result<Event, ApiError>> + Send + '_>> {
        let url = format!("{}/v1/messages", self.base_url);
        let api_key = self.api_key.clone();
        let client = self.client.clone();

        Box::pin(async_stream::try_stream! {
            if request.messages.iter().flat_map(|message| &message.content).any(|block| matches!(block, crate::RequestContent::Replay { .. })) {
                Err(ApiError::Provider("Responses replay cannot be sent to Anthropic".into()))?;
            }
            let resp = client
                .post(&url)
                .header("x-api-key", &api_key)
                .header("anthropic-version", ANTHROPIC_VERSION)
                .header("anthropic-beta", ANTHROPIC_BETA)
                .header("content-type", "application/json")
                .json(&request)
                .send()
                .await?;

            if !resp.status().is_success() {
                let status = resp.status().as_u16();
                let body = resp.text().await.unwrap_or_default();
                Err(ApiError::Http { status, body })?;
                return;
            }

            let mut stream = resp.bytes_stream();
            let mut parser = SseParser::new();
            let mut usage = AnthropicUsage::default();
            let mut decoder = crate::sse::Utf8Stream::new();
            let mut line_buf = String::new();

            while let Some(chunk) = stream.next().await {
                let bytes = chunk?;
                // Decoded across chunk boundaries: a character split by the
                // transport must not become U+FFFD.
                let text = decoder.push(&bytes);

                for ch in text.chars() {
                    if ch == '\n' {
                        if let Some(sse) = parser.feed_line(&line_buf) {
                            let events = parse_anthropic_sse(sse.event_type.as_deref(), &sse.data, &mut usage)?;
                            for event in events {
                                yield event;
                            }
                        }
                        line_buf.clear();
                    } else {
                        line_buf.push(ch);
                    }
                }
            }

            // flush any remaining line (no trailing newline)
            if !line_buf.is_empty() {
                if let Some(sse) = parser.feed_line(&line_buf) {
                    let events = parse_anthropic_sse(sse.event_type.as_deref(), &sse.data, &mut usage)?;
                    for event in events {
                        yield event;
                    }
                }
            }
            // Flush any buffered data pending blank-line dispatch.
            if let Some(sse) = parser.finish() {
                let events = parse_anthropic_sse(sse.event_type.as_deref(), &sse.data, &mut usage)?;
                for event in events {
                    yield event;
                }
            }
        })
    }
}

// ---------------------------------------------------------------------------
// Anthropic SSE event parsing
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct RawStreamEvent {
    #[serde(rename = "type")]
    event_type: String,
    #[serde(flatten)]
    rest: HashMap<String, serde_json::Value>,
}

/// Anthropic reports cumulative counters, while runtime usage events add.
/// State belongs to one response stream, never the provider or its clones.
#[derive(Default)]
struct AnthropicUsage {
    counters: [u32; 4],
}

impl AnthropicUsage {
    fn observe(&mut self, value: &serde_json::Value) -> Option<TokenUsage> {
        let object = value.as_object()?;
        let mut next = self.counters;
        let mut supplied_counter = false;
        for (index, field) in [
            "input_tokens",
            "output_tokens",
            "cache_creation_input_tokens",
            "cache_read_input_tokens",
        ]
        .into_iter()
        .enumerate()
        {
            // Absent counters provide no new evidence. Reject an entire
            // malformed/regressive snapshot without advancing the baseline,
            // rather than inventing a reset, a negative delta or extra usage.
            if let Some(value) = object.get(field) {
                let count = u32::try_from(value.as_u64()?).ok()?;
                if count < self.counters[index] {
                    return None;
                }
                next[index] = count;
                supplied_counter = true;
            }
        }
        let delta = std::array::from_fn::<_, 4, _>(|index| next[index] - self.counters[index]);
        self.counters = next;
        if !supplied_counter {
            return None;
        }
        Some(TokenUsage {
            input_tokens: delta[0],
            output_tokens: delta[1],
            cache_creation_input_tokens: delta[2],
            cache_read_input_tokens: delta[3],
        })
    }
}

/// Public re-export for tests only.
#[cfg(test)]
pub fn parse_anthropic_sse_pub(
    event_type: Option<&str>,
    data: &str,
) -> Result<Vec<Event>, ApiError> {
    parse_anthropic_sse(event_type, data, &mut AnthropicUsage::default())
}

fn parse_anthropic_sse(
    event_type: Option<&str>,
    data: &str,
    usage: &mut AnthropicUsage,
) -> Result<Vec<Event>, ApiError> {
    if data == "[DONE]" {
        return Ok(vec![]);
    }

    let raw: RawStreamEvent = serde_json::from_str(data)
        .map_err(|e| ApiError::SseParse(format!("bad JSON: {e}: {data}")))?;

    let et = event_type.unwrap_or(&raw.event_type);

    match et {
        "content_block_start" => Ok(anthropic_content_block_start(&raw)),
        "content_block_delta" => Ok(anthropic_content_block_delta(&raw)),
        "content_block_stop" => Ok(vec![Event::ToolUseEnd {
            id: anthropic_index_id(&raw),
        }]),
        "message_delta" => Ok(anthropic_message_delta(&raw, usage)),
        "message_start" => Ok(anthropic_message_start(&raw, usage)),
        _ => Ok(vec![]),
    }
}

fn anthropic_content_block_start(raw: &RawStreamEvent) -> Vec<Event> {
    let Some(block) = raw.rest.get("content_block") else {
        return vec![];
    };
    if block.get("type").and_then(|v| v.as_str()) != Some("tool_use") {
        return vec![];
    }

    let id = block
        .get("id")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let name = block
        .get("name")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    vec![Event::ToolUseStart { id, name }]
}

fn anthropic_content_block_delta(raw: &RawStreamEvent) -> Vec<Event> {
    let Some(delta) = raw.rest.get("delta") else {
        return vec![];
    };
    let delta_type = delta.get("type").and_then(|v| v.as_str()).unwrap_or("");

    match delta_type {
        "text_delta" => anthropic_text_delta(delta),
        "input_json_delta" => anthropic_input_json_delta(raw, delta),
        _ => vec![],
    }
}

fn anthropic_text_delta(delta: &serde_json::Value) -> Vec<Event> {
    let text = delta
        .get("text")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    if text.is_empty() {
        vec![]
    } else {
        vec![Event::TextDelta { text }]
    }
}

fn anthropic_input_json_delta(raw: &RawStreamEvent, delta: &serde_json::Value) -> Vec<Event> {
    let partial_json = delta
        .get("partial_json")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    // We need the tool use id — it's in the outer index, not the delta.
    // We use a placeholder; the runtime correlates by stream position.
    let id = anthropic_index_id(raw);
    vec![Event::ToolUseDelta { id, partial_json }]
}

fn anthropic_message_delta(raw: &RawStreamEvent, usage: &mut AnthropicUsage) -> Vec<Event> {
    let mut events = Vec::new();
    if let Some(reason) = raw
        .rest
        .get("delta")
        .and_then(|delta| delta.get("stop_reason"))
        .and_then(|v| v.as_str())
    {
        events.push(Event::MessageStop {
            stop_reason: StopReason::from_wire_str(reason),
        });
    }

    // usage may also be in message_delta
    if let Some(usage_val) = raw.rest.get("usage") {
        if let Some(usage) = usage.observe(usage_val) {
            events.push(Event::UsageDelta { usage });
        }
    }

    events
}

fn anthropic_message_start(raw: &RawStreamEvent, usage: &mut AnthropicUsage) -> Vec<Event> {
    // usage is in message.usage
    if let Some(msg) = raw.rest.get("message") {
        if let Some(usage_val) = msg.get("usage") {
            if let Some(usage) = usage.observe(usage_val) {
                return vec![Event::UsageDelta { usage }];
            }
        }
    }
    vec![]
}

fn anthropic_index_id(raw: &RawStreamEvent) -> String {
    raw.rest
        .get("index")
        .and_then(serde_json::Value::as_u64)
        .map(|i| format!("__idx_{i}"))
        .unwrap_or_default()
}
