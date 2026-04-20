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
            let mut line_buf = String::new();

            while let Some(chunk) = stream.next().await {
                let bytes = chunk?;
                let text = String::from_utf8_lossy(&bytes);

                for ch in text.chars() {
                    if ch == '\n' {
                        if let Some(sse) = parser.feed_line(&line_buf) {
                            let events = parse_anthropic_sse(sse.event_type.as_deref(), &sse.data)?;
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
                    let events = parse_anthropic_sse(sse.event_type.as_deref(), &sse.data)?;
                    for event in events {
                        yield event;
                    }
                }
            }
            // Flush any buffered data pending blank-line dispatch.
            if let Some(sse) = parser.finish() {
                let events = parse_anthropic_sse(sse.event_type.as_deref(), &sse.data)?;
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

/// Public re-export for tests only.
#[cfg(test)]
pub fn parse_anthropic_sse_pub(
    event_type: Option<&str>,
    data: &str,
) -> Result<Vec<Event>, ApiError> {
    parse_anthropic_sse(event_type, data)
}

#[allow(clippy::too_many_lines)]
fn parse_anthropic_sse(event_type: Option<&str>, data: &str) -> Result<Vec<Event>, ApiError> {
    if data == "[DONE]" {
        return Ok(vec![]);
    }

    let raw: RawStreamEvent = serde_json::from_str(data)
        .map_err(|e| ApiError::SseParse(format!("bad JSON: {e}: {data}")))?;

    let et = event_type.unwrap_or(&raw.event_type);

    match et {
        "content_block_start" => {
            let Some(block) = raw.rest.get("content_block") else {
                return Ok(vec![]);
            };
            if block.get("type").and_then(|v| v.as_str()) == Some("tool_use") {
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
                return Ok(vec![Event::ToolUseStart { id, name }]);
            }
            Ok(vec![])
        }

        "content_block_delta" => {
            let Some(delta) = raw.rest.get("delta") else {
                return Ok(vec![]);
            };
            let delta_type = delta.get("type").and_then(|v| v.as_str()).unwrap_or("");

            match delta_type {
                "text_delta" => {
                    let text = delta
                        .get("text")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    if text.is_empty() {
                        Ok(vec![])
                    } else {
                        Ok(vec![Event::TextDelta { text }])
                    }
                }
                "input_json_delta" => {
                    let partial_json = delta
                        .get("partial_json")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    // We need the tool use id — it's in the outer index, not the delta.
                    // We use a placeholder; the runtime correlates by stream position.
                    let id = raw
                        .rest
                        .get("index")
                        .and_then(serde_json::Value::as_u64)
                        .map(|i| format!("__idx_{i}"))
                        .unwrap_or_default();
                    Ok(vec![Event::ToolUseDelta { id, partial_json }])
                }
                _ => Ok(vec![]),
            }
        }

        "content_block_stop" => {
            let id = raw
                .rest
                .get("index")
                .and_then(serde_json::Value::as_u64)
                .map(|i| format!("__idx_{i}"))
                .unwrap_or_default();
            Ok(vec![Event::ToolUseEnd { id }])
        }

        "message_delta" => {
            let Some(delta) = raw.rest.get("delta") else {
                return Ok(vec![Event::MessageStop {
                    stop_reason: StopReason::EndTurn,
                }]);
            };
            let stop_reason = delta
                .get("stop_reason")
                .and_then(|v| v.as_str())
                .map_or(StopReason::EndTurn, StopReason::from_wire_str);

            let mut events = vec![Event::MessageStop { stop_reason }];

            // usage may also be in message_delta
            if let Some(usage_val) = raw.rest.get("usage") {
                if let Ok(usage) = serde_json::from_value::<TokenUsage>(usage_val.clone()) {
                    events.push(Event::UsageDelta { usage });
                }
            }

            Ok(events)
        }

        "message_start" => {
            // usage is in message.usage
            if let Some(msg) = raw.rest.get("message") {
                if let Some(usage_val) = msg.get("usage") {
                    if let Ok(usage) = serde_json::from_value::<TokenUsage>(usage_val.clone()) {
                        return Ok(vec![Event::UsageDelta { usage }]);
                    }
                }
            }
            Ok(vec![])
        }

        _ => Ok(vec![]),
    }
}
