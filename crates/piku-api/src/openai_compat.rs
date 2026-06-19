/// Generic `OpenAI`-compatible streaming provider.
///
/// Handles the full SSE → Event pipeline for any endpoint that speaks the
/// `OpenAI` chat completions wire format (streaming).  Per-provider concerns
/// (base URL, auth headers, extra headers) are supplied at construction time.
use std::pin::Pin;

use futures_util::{Stream, StreamExt};
use serde::Deserialize;

use crate::error::ApiError;
use crate::provider::Provider;
use crate::sse::SseParser;
use crate::types::{Event, MessageRequest, RequestContent, StopReason, TokenUsage, ToolDefinition};

/// A single extra HTTP header (name, value).
#[derive(Debug, Clone)]
pub struct Header {
    pub name: String,
    pub value: String,
}

impl Header {
    pub fn new(name: impl Into<String>, value: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            value: value.into(),
        }
    }
}

/// Auth strategy for the provider.
#[derive(Debug, Clone)]
pub enum Auth {
    /// `Authorization: Bearer <token>`
    Bearer(String),
    /// `x-api-key: <key>`
    ApiKey(String),
    /// No auth (e.g. local Ollama).
    None,
}

#[derive(Clone)]
pub struct OpenAiCompatProvider {
    pub name: String,
    pub base_url: String,
    pub auth: Auth,
    pub extra_headers: Vec<Header>,
    client: reqwest::Client,
}

impl OpenAiCompatProvider {
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        base_url: impl Into<String>,
        auth: Auth,
        extra_headers: Vec<Header>,
    ) -> Self {
        Self {
            name: name.into(),
            base_url: base_url.into(),
            auth,
            extra_headers,
            client: reqwest::Client::new(),
        }
    }
}

impl Provider for OpenAiCompatProvider {
    fn name(&self) -> &str {
        &self.name
    }

    fn boxed_clone(&self) -> Box<dyn Provider + Send + Sync + 'static> {
        let cloned: OpenAiCompatProvider = OpenAiCompatProvider::clone(self);
        Box::new(cloned)
    }

    fn stream_message(
        &self,
        request: MessageRequest,
    ) -> Pin<Box<dyn Stream<Item = Result<Event, ApiError>> + Send + '_>> {
        let url = format!("{}/chat/completions", self.base_url);
        let auth = self.auth.clone();
        let extra_headers = self.extra_headers.clone();
        let client = self.client.clone();
        let body = build_openai_body(&request);

        Box::pin(async_stream::try_stream! {
            let mut req = client
                .post(&url)
                .header("content-type", "application/json");

            // auth
            req = match &auth {
                Auth::Bearer(token) => req.header("Authorization", format!("Bearer {token}")),
                Auth::ApiKey(key) => req.header("x-api-key", key),
                Auth::None => req,
            };

            // extra headers
            for h in &extra_headers {
                req = req.header(&h.name, &h.value);
            }

            let resp = req.json(&body).send().await?;

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
                            let events = parse_openai_sse(&sse.data)?;
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

            // Flush remaining line (no terminating newline).
            if !line_buf.is_empty() {
                if let Some(sse) = parser.feed_line(&line_buf) {
                    let events = parse_openai_sse(&sse.data)?;
                    for event in events {
                        yield event;
                    }
                }
            }
            // Flush any buffered data left pending a blank-line dispatch.
            // Otherwise a stream that closes without a trailing blank line
            // silently drops its final event (often `[DONE]`).
            if let Some(sse) = parser.finish() {
                let events = parse_openai_sse(&sse.data)?;
                for event in events {
                    yield event;
                }
            }
        })
    }
}

// ---------------------------------------------------------------------------
// Build `OpenAI` chat completions request body
// ---------------------------------------------------------------------------

#[must_use]
pub fn build_openai_body(request: &MessageRequest) -> serde_json::Value {
    let mut messages: Vec<serde_json::Value> = Vec::new();

    if let Some(system_message) = openai_system_message(request) {
        messages.push(system_message);
    }

    for msg in &request.messages {
        messages.extend(openai_messages_for_request(msg));
    }

    let mut body = serde_json::json!({
        "model": request.model,
        "messages": messages,
        "max_tokens": request.max_tokens,
        "stream": true,
    });

    if let Some(tools) = &request.tools {
        if !tools.is_empty() {
            body["tools"] = tools_to_oai(tools);
            body["tool_choice"] = serde_json::json!("auto");
        }
    }

    body
}

fn openai_system_message(request: &MessageRequest) -> Option<serde_json::Value> {
    let system_blocks = request.system.as_ref()?;
    // OpenAI takes system as a plain string message: flatten all blocks.
    let content: String = system_blocks
        .iter()
        .map(|b| b.text.as_str())
        .collect::<Vec<_>>()
        .join("\n\n");
    if content.is_empty() {
        None
    } else {
        Some(serde_json::json!({ "role": "system", "content": content }))
    }
}

fn openai_messages_for_request(msg: &crate::types::RequestMessage) -> Vec<serde_json::Value> {
    if msg
        .content
        .iter()
        .any(|c| matches!(c, RequestContent::ToolResult { .. }))
    {
        return openai_tool_result_messages(msg);
    }

    if msg
        .content
        .iter()
        .any(|c| matches!(c, RequestContent::ToolUse { .. }))
    {
        return vec![openai_tool_use_message(msg)];
    }

    vec![openai_regular_message(msg)]
}

fn openai_tool_result_messages(msg: &crate::types::RequestMessage) -> Vec<serde_json::Value> {
    // Emit one "tool" role message per ToolResult block.
    // Any accompanying Text blocks (from coalesced messages after a
    // ReplaceAndExec restart) are emitted separately as a "user" message
    // AFTER the tool results, so the model sees the follow-up prompt.
    let mut messages: Vec<serde_json::Value> = msg
        .content
        .iter()
        .filter_map(openai_tool_result_message)
        .collect();

    let text_blocks: Vec<&str> = msg.content.iter().filter_map(request_text).collect();
    if !text_blocks.is_empty() {
        let combined = text_blocks.join("\n");
        messages.push(serde_json::json!({ "role": "user", "content": combined }));
    }

    messages
}

fn openai_tool_result_message(block: &RequestContent) -> Option<serde_json::Value> {
    let RequestContent::ToolResult {
        tool_use_id,
        content,
        is_error,
    } = block
    else {
        return None;
    };
    let content_str = if is_error.unwrap_or(false) {
        format!("Error: {content}")
    } else {
        content.clone()
    };
    Some(serde_json::json!({
        "role": "tool",
        "tool_call_id": tool_use_id,
        "content": content_str,
    }))
}

fn openai_tool_use_message(msg: &crate::types::RequestMessage) -> serde_json::Value {
    let text: String = msg
        .content
        .iter()
        .filter_map(request_text)
        .collect::<Vec<_>>()
        .join("");

    let tool_calls: Vec<serde_json::Value> =
        msg.content.iter().filter_map(openai_tool_call).collect();

    let mut obj = serde_json::json!({
        "role": "assistant",
        "tool_calls": tool_calls,
    });
    if !text.is_empty() {
        obj["content"] = serde_json::Value::String(text);
    }
    obj
}

fn openai_tool_call(block: &RequestContent) -> Option<serde_json::Value> {
    let RequestContent::ToolUse { id, name, input } = block else {
        return None;
    };
    Some(serde_json::json!({
        "id": id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": input.to_string(),
        }
    }))
}

fn openai_regular_message(msg: &crate::types::RequestMessage) -> serde_json::Value {
    let content = if msg.content.len() == 1 {
        if let RequestContent::Text { text } = &msg.content[0] {
            serde_json::Value::String(text.clone())
        } else {
            openai_content_array(&msg.content)
        }
    } else {
        openai_content_array(&msg.content)
    };

    serde_json::json!({ "role": msg.role, "content": content })
}

fn openai_content_array(content: &[RequestContent]) -> serde_json::Value {
    serde_json::Value::Array(content.iter().map(request_content_to_oai).collect())
}

fn request_text(block: &RequestContent) -> Option<&str> {
    if let RequestContent::Text { text } = block {
        Some(text.as_str())
    } else {
        None
    }
}

fn request_content_to_oai(block: &RequestContent) -> serde_json::Value {
    match block {
        RequestContent::Text { text } => serde_json::json!({ "type": "text", "text": text }),
        RequestContent::ToolResult {
            tool_use_id,
            content,
            is_error,
        } => serde_json::json!({
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "content": content,
            "is_error": is_error.unwrap_or(false),
        }),
        RequestContent::ToolUse { id, name, input } => serde_json::json!({
            "type": "tool_use", "id": id, "name": name, "input": input,
        }),
    }
}

fn tools_to_oai(tools: &[ToolDefinition]) -> serde_json::Value {
    serde_json::Value::Array(
        tools
            .iter()
            .map(|t| {
                serde_json::json!({
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.input_schema,
                    }
                })
            })
            .collect(),
    )
}

// ---------------------------------------------------------------------------
// Parse OpenAI-compatible SSE chunk → Vec<Event>
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct OaiChunk {
    choices: Vec<OaiChoice>,
    #[serde(default)]
    usage: Option<OaiUsage>,
}

#[derive(Debug, Deserialize)]
struct OaiChoice {
    delta: OaiDelta,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize, Default)]
struct OaiDelta {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<OaiToolCallDelta>>,
}

#[derive(Debug, Deserialize)]
struct OaiToolCallDelta {
    index: u32,
    id: Option<String>,
    function: Option<OaiFunctionDelta>,
}

#[derive(Debug, Deserialize)]
struct OaiFunctionDelta {
    name: Option<String>,
    arguments: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OaiUsage {
    prompt_tokens: Option<u32>,
    completion_tokens: Option<u32>,
}

pub fn parse_openai_sse(data: &str) -> Result<Vec<Event>, ApiError> {
    if data == "[DONE]" {
        return Ok(vec![]);
    }

    let chunk: OaiChunk = serde_json::from_str(data)
        .map_err(|e| ApiError::SseParse(format!("bad JSON: {e}: {data}")))?;

    let mut events = Vec::new();

    for choice in &chunk.choices {
        if let Some(text) = &choice.delta.content {
            if !text.is_empty() {
                events.push(Event::TextDelta { text: text.clone() });
            }
        }

        if let Some(tool_calls) = &choice.delta.tool_calls {
            for tc in tool_calls {
                if let Some(func) = &tc.function {
                    // New tool call: has a name
                    if let Some(name) = &func.name {
                        let id = tc
                            .id
                            .clone()
                            .unwrap_or_else(|| format!("__tc_{}", tc.index));
                        events.push(Event::ToolUseStart {
                            id: id.clone(),
                            name: name.clone(),
                        });
                    }
                    // Argument delta
                    if let Some(args) = &func.arguments {
                        if !args.is_empty() {
                            let id = tc
                                .id
                                .clone()
                                .unwrap_or_else(|| format!("__tc_{}", tc.index));
                            events.push(Event::ToolUseDelta {
                                id,
                                partial_json: args.clone(),
                            });
                        }
                    }
                }
            }
        }

        if let Some(reason) = &choice.finish_reason {
            let stop_reason = match reason.as_str() {
                "stop" => StopReason::EndTurn,
                "tool_calls" => StopReason::ToolUse,
                "length" => StopReason::MaxTokens,
                other => StopReason::Other(other.to_string()),
            };
            events.push(Event::MessageStop { stop_reason });
        }
    }

    if let Some(usage) = &chunk.usage {
        events.push(Event::UsageDelta {
            usage: TokenUsage {
                input_tokens: usage.prompt_tokens.unwrap_or(0),
                output_tokens: usage.completion_tokens.unwrap_or(0),
                cache_creation_input_tokens: 0,
                cache_read_input_tokens: 0,
            },
        });
    }

    Ok(events)
}
