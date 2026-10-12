//! Account-bound, stateless Responses transport for `ChatGPT` plan usage.
use std::collections::{BTreeMap, BTreeSet};
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use futures_util::{Stream, StreamExt};
use serde_json::{json, Value};

use crate::provider::AccessTokenSource;
use crate::replay::{ReplayEnvelope, ReplayRoute, ResponseContent, ResponsesOutputItem};
use crate::sse::{SseParser, Utf8Stream};
use crate::{ApiError, Event, MessageRequest, Provider, RequestContent, StopReason, TokenUsage};

const ENDPOINT: &str = "https://api.openai.com/v1/responses";
const MAX_EVENT_BYTES: usize = 8 * 1024 * 1024;

#[derive(Clone)]
pub struct ResponsesProvider {
    source: Arc<dyn AccessTokenSource>,
    registration: String,
    client: reqwest::Client,
    endpoint: String,
}

/// A model visible to the selected `ChatGPT` registration, in server order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatgptModel {
    pub slug: String,
    pub display_name: String,
}

impl ResponsesProvider {
    /// Fetch the selected registration's catalog; never infer a model from a
    /// plan name or use the generic API's unrelated data-array response shape.
    pub async fn models(&self) -> Result<Vec<ChatgptModel>, ApiError> {
        if self.source.binding() != self.registration {
            return Err(protocol("credential registration changed"));
        }
        let token = self.source.token().await?;
        let url = self
            .endpoint
            .strip_suffix("/responses")
            .ok_or_else(|| protocol("invalid Responses endpoint"))?;
        let response = self
            .client
            .get(format!("{url}/models"))
            .bearer_auth(token.expose_secret())
            .send()
            .await
            .map_err(|_| protocol("model catalog request failed"))?;
        let request_id = response
            .headers()
            .get("x-request-id")
            .and_then(|v| v.to_str().ok())
            .map(|value| redacted_identifier(value, token.expose_secret()))
            .unwrap_or_default();
        let response = check_response(response, token.expose_secret(), &request_id).await?;
        let bytes = bounded_body(response, 1024 * 1024).await?;
        let value: Value =
            serde_json::from_slice(&bytes).map_err(|_| protocol("invalid model catalog JSON"))?;
        let models = value
            .get("models")
            .and_then(Value::as_array)
            .ok_or_else(|| protocol("model catalog lacks models array"))?;
        models
            .iter()
            .filter(|model| model.get("visibility").and_then(Value::as_str) == Some("list"))
            .map(|model| {
                let slug = model
                    .get("slug")
                    .and_then(Value::as_str)
                    .filter(|s| !s.is_empty())
                    .ok_or_else(|| protocol("visible model lacks slug"))?;
                let display_name = model
                    .get("display_name")
                    .and_then(Value::as_str)
                    .ok_or_else(|| protocol("visible model lacks display name"))?;
                Ok(ChatgptModel {
                    slug: slug.to_owned(),
                    display_name: display_name.to_owned(),
                })
            })
            .collect()
    }
    /// Validate route, replay and request constraints before credential or catalog access.
    pub fn validate_request(&self, request: &MessageRequest) -> Result<(), ApiError> {
        if self.source.binding() != self.registration {
            return Err(protocol("credential registration changed"));
        }
        build_body(request, &self.registration).map(|_| ())
    }

    pub fn new(source: Arc<dyn AccessTokenSource>) -> Result<Self, ApiError> {
        if source.binding().is_empty() {
            return Err(protocol("empty credential registration binding"));
        }
        let client = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .connect_timeout(Duration::from_secs(15))
            .timeout(Duration::from_secs(300))
            .build()?;
        Ok(Self {
            registration: source.binding().to_owned(),
            source,
            client,
            endpoint: ENDPOINT.to_owned(),
        })
    }
}

impl Provider for ResponsesProvider {
    fn name(&self) -> &'static str {
        "chatgpt"
    }
    fn boxed_clone(&self) -> Box<dyn Provider + Send + Sync + 'static> {
        Box::new(self.clone())
    }
    fn stream_message(
        &self,
        request: MessageRequest,
    ) -> Pin<Box<dyn Stream<Item = Result<Event, ApiError>> + Send + '_>> {
        Box::pin(async_stream::try_stream! {
            let body = build_body(&request, &self.registration)?;
            if self.source.binding() != self.registration {
                Err(protocol("credential registration changed"))?;
            }
            let token = self.source.token().await?;
            let response = self.client.post(&self.endpoint)
                .bearer_auth(token.expose_secret()).json(&body).send().await
                .map_err(|_| protocol("Responses transport request failed"))?;
            let request_id = response.headers().get("x-request-id")
                .and_then(|value| value.to_str().ok())
                .map(|value| redacted_identifier(value, token.expose_secret())).unwrap_or_default();
            let response = check_response(response, token.expose_secret(), &request_id).await?;
            let advertised = tool_names(&request);
            let mut state = ResponseState::default();
            let mut decoder = Utf8Stream::new();
            let mut parser = SseParser::new();
            let mut pending = String::new();
            let mut event_bytes = 0usize;
            let mut chunks = response.bytes_stream();
            while let Some(chunk) = chunks.next().await {
                let chunk = chunk.map_err(|_| protocol("Responses stream interrupted"))?;
                pending.push_str(&decoder.push(&chunk));
                if pending.len() > MAX_EVENT_BYTES { Err(protocol("Responses event exceeds limit"))?; }
                while let Some(end) = pending.find('\n') {
                    let line: String = pending.drain(..=end).collect();
                    event_bytes += line.len();
                    if event_bytes > MAX_EVENT_BYTES { Err(protocol("Responses event exceeds limit"))?; }
                    if let Some(event) = parser.feed_line(line.trim_end_matches('\n')) {
                        event_bytes = 0;
                        if let Some(text) = state.accept(&event.data, &request_id, token.expose_secret())? {
                            yield Event::TextDelta { text };
                        }
                    }
                }
            }
            if !pending.is_empty() { parser.feed_line(&pending); }
            if let Some(event) = parser.finish() {
                if let Some(text) = state.accept(&event.data, &request_id, token.expose_secret())? {
                    yield Event::TextDelta { text };
                }
            }
            // Wait through EOF: trailing errors or duplicate terminal events
            // must not authorize tools from an earlier completed frame.
            let completed = state.completed.take().ok_or(ApiError::UnexpectedStreamEnd)?;
            let output: Vec<ResponsesOutputItem> = serde_json::from_value(state.items.reconcile(completed.get("output"))?)
                .map_err(|_| protocol("unsupported or malformed Responses output"))?;
            let envelope = ReplayEnvelope { version: 1, route: ReplayRoute::ChatgptResponses, registration: self.registration.clone(), model: request.model.clone(), output };
            validate_replay(&envelope, &self.registration, &request.model, Some(&advertised))?;
            let usage = normalize_usage(completed.get("usage"))?;
            let final_text: String = envelope.output.iter().filter_map(|item| {
                if let ResponsesOutputItem::Message { content, .. } = item { Some(content) } else { None }
            }).flatten().map(|content| match content {
                ResponseContent::OutputText { text, .. } => text.as_str(),
                ResponseContent::Refusal { refusal, .. } => refusal.as_str(),
            }).collect();
            if state.text_seen && state.streamed_text != final_text {
                Err(protocol("streamed text disagrees with completed Responses output"))?;
            }
            if !state.text_seen && !final_text.is_empty() {
                yield Event::TextDelta { text: final_text };
            }
            let mut has_calls = false;
            for item in &envelope.output {
                if let ResponsesOutputItem::FunctionCall { call_id, name, arguments, .. } = item {
                    has_calls = true;
                    yield Event::ToolUseStart { id: call_id.clone(), name: name.clone() };
                    yield Event::ToolUseDelta { id: call_id.clone(), partial_json: arguments.clone() };
                    yield Event::ToolUseEnd { id: call_id.clone() };
                }
            }
            yield Event::Replay { envelope };
            yield Event::UsageDelta { usage };
            yield Event::MessageStop { stop_reason: if has_calls { StopReason::ToolUse } else { StopReason::EndTurn } };
        })
    }
}

fn protocol(message: &str) -> ApiError {
    ApiError::Provider(message.to_owned())
}
fn redacted_identifier(value: &str, secret: &str) -> String {
    value
        .replace(secret, "[REDACTED]")
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.' | '[' | ']'))
        .take(128)
        .collect()
}
fn diagnostic(value: &Value, request_id: &str, secret: &str) -> String {
    let error = value
        .get("error")
        .or_else(|| value.pointer("/response/error"))
        .or_else(|| (value.get("type").and_then(Value::as_str) == Some("error")).then_some(value));
    let shape = if error.is_some() {
        "error"
    } else if value.get("detail").is_some() {
        "detail"
    } else {
        "other"
    };
    let field = |name| {
        error
            .and_then(|e| e.get(name))
            .and_then(Value::as_str)
            .map(|value| redacted_identifier(value, secret))
            .unwrap_or_default()
    };
    json!({"shape":shape,"code":field("code"),"param":field("param"),"request_id":request_id})
        .to_string()
        .replace(secret, "[REDACTED]")
}
fn tool_names(request: &MessageRequest) -> BTreeSet<&str> {
    request
        .tools
        .iter()
        .flatten()
        .map(|tool| tool.name.as_str())
        .collect()
}

fn validate_replay(
    envelope: &ReplayEnvelope,
    registration: &str,
    model: &str,
    tools: Option<&BTreeSet<&str>>,
) -> Result<(), ApiError> {
    if envelope.version != 1
        || envelope.route != ReplayRoute::ChatgptResponses
        || envelope.registration != registration
        || envelope.model != model
    {
        return Err(protocol("Responses replay binding/model mismatch"));
    }
    let mut calls = BTreeSet::new();
    for item in &envelope.output {
        match item {
            ResponsesOutputItem::FunctionCall {
                call_id,
                name,
                namespace,
                arguments,
                extra,
                ..
            } => {
                if extra
                    .get("status")
                    .is_some_and(|status| !status.is_null() && status.as_str() != Some("completed"))
                {
                    return Err(protocol("Responses function item is not completed"));
                }
                if namespace.as_deref() != Some("piku")
                    || tools.is_some_and(|tools| !tools.contains(name.as_str()))
                    || call_id.is_empty()
                    || !calls.insert(call_id)
                {
                    return Err(protocol(
                        "unadvertised or duplicate Responses function call",
                    ));
                }
                if !serde_json::from_str::<Value>(arguments).is_ok_and(|v| v.is_object()) {
                    return Err(protocol(
                        "Responses function arguments must be a JSON object",
                    ));
                }
            }
            ResponsesOutputItem::Message { role, status, .. }
                if role != "assistant" || status != "completed" =>
            {
                return Err(protocol("invalid replay message role/status"))
            }
            _ => {}
        }
    }
    Ok(())
}

fn build_body(request: &MessageRequest, registration: &str) -> Result<Value, ApiError> {
    if request.max_tokens != 0 {
        return Err(protocol(
            "ChatGPT plan usage cannot enforce max_tokens; explicitly select uncapped output",
        ));
    }
    let mut input = Vec::new();
    for message in &request.messages {
        let replays: Vec<_> = message
            .content
            .iter()
            .filter_map(|block| {
                if let RequestContent::Replay { envelope } = block {
                    Some(envelope)
                } else {
                    None
                }
            })
            .collect();
        if !replays.is_empty() {
            if replays.len() != 1
                || message.role != "assistant"
                || message
                    .content
                    .iter()
                    .any(|block| matches!(block, RequestContent::ToolResult { .. }))
            {
                return Err(protocol("invalid mixed Responses replay message"));
            }
            let replay = replays[0];
            validate_replay(replay, registration, &request.model, None)?;
            for item in &replay.output {
                input.push(serde_json::to_value(item)?);
            }
            continue;
        }
        if !matches!(message.role.as_str(), "user" | "assistant" | "developer") {
            return Err(protocol("unsupported Responses message role"));
        }
        for block in &message.content {
            match block {
                RequestContent::Text { text } => {
                    input.push(json!({"role":message.role,"content":text}));
                }
                RequestContent::ToolResult {
                    tool_use_id,
                    content,
                    ..
                } => input.push(
                    json!({"type":"function_call_output","call_id":tool_use_id,"output":content}),
                ),
                RequestContent::ToolUse {
                    id,
                    name,
                    input: arguments,
                } => {
                    input.push(json!({"type":"function_call","call_id":id,"name":name,"namespace":"piku","arguments":arguments.to_string()}));
                }
                RequestContent::Replay { .. } => unreachable!("handled above"),
            }
        }
    }
    let definitions: Vec<_> = request.tools.iter().flatten().map(|tool| json!({"type":"function","name":tool.name,"description":tool.description,"parameters":tool.input_schema,"strict":false})).collect();
    let mut body = json!({"model":request.model,"input":input,"store":false,"stream":true});
    if !definitions.is_empty() {
        body["tools"] = json!([{"type":"namespace","name":"piku","description":"Local Piku tools","tools":definitions}]);
    }
    if let Some(system) = &request.system {
        body["instructions"] = Value::String(
            system
                .iter()
                .map(|block| block.text.as_str())
                .collect::<Vec<_>>()
                .join("\n\n"),
        );
    }
    Ok(body)
}

#[derive(Default)]
struct ResponseState {
    completed: Option<Value>,
    text_seen: bool,
    streamed_text: String,
    items: OutputItems,
    response_id: Option<String>,
}
impl ResponseState {
    fn accept(
        &mut self,
        data: &str,
        request_id: &str,
        secret: &str,
    ) -> Result<Option<String>, ApiError> {
        let event: Value =
            serde_json::from_str(data).map_err(|_| protocol("malformed Responses event"))?;
        let kind = event
            .get("type")
            .and_then(Value::as_str)
            .ok_or_else(|| protocol("Responses event lacks type"))?;
        if matches!(
            kind,
            "response.created" | "response.in_progress" | "response.queued" | "response.completed"
        ) {
            let id = event.pointer("/response/id").and_then(Value::as_str);
            if self.response_id.as_deref().is_some_and(|observed| {
                (id.is_some() || kind == "response.completed") && Some(observed) != id
            }) {
                return Err(protocol("Responses lifecycle identity changed"));
            }
            if let Some(id) = id {
                if id.is_empty() {
                    return Err(protocol("empty Responses lifecycle identity"));
                }
                self.response_id = Some(id.to_owned());
            }
        }
        match kind {
            "response.failed" | "response.incomplete" | "error" => Err(protocol(&format!(
                "{kind}: {}",
                diagnostic(&event, request_id, secret)
            ))),
            "response.completed" => {
                if self.completed.is_some() {
                    return Err(protocol("duplicate response.completed"));
                }
                let response = event
                    .get("response")
                    .ok_or_else(|| protocol("completed event lacks response"))?;
                if response.get("status").and_then(Value::as_str) != Some("completed") {
                    return Err(protocol("response.completed has invalid status"));
                }
                self.completed = Some(response.clone());
                Ok(None)
            }
            _ if self.completed.is_some() => Err(protocol("event after response.completed")),
            "response.output_text.delta" | "response.refusal.delta" => {
                let text = event
                    .get("delta")
                    .and_then(Value::as_str)
                    .ok_or_else(|| protocol("text delta missing"))?;
                if self.streamed_text.len() + text.len() > MAX_EVENT_BYTES {
                    return Err(protocol("Responses text exceeds limit"));
                }
                self.text_seen = true;
                self.streamed_text.push_str(text);
                Ok(Some(text.to_owned()))
            }
            "response.output_item.added"
            | "response.output_item.done"
            | "response.function_call_arguments.delta"
            | "response.function_call_arguments.done" => {
                self.items.observe(kind, &event)?;
                Ok(None)
            }
            "response.created"
            | "response.in_progress"
            | "response.queued"
            | "response.content_part.added"
            | "response.content_part.done"
            | "response.output_text.done"
            | "response.refusal.done"
            | "response.reasoning_summary_part.added"
            | "response.reasoning_summary_part.done"
            | "response.reasoning_summary_text.delta"
            | "response.reasoning_summary_text.done"
            | "response.reasoning_text.delta"
            | "response.reasoning_text.done" => Ok(None),
            _ => Err(protocol("unsupported Responses stream event")),
        }
    }
}
fn normalize_usage(value: Option<&Value>) -> Result<TokenUsage, ApiError> {
    let value = value.ok_or_else(|| protocol("completed response lacks usage"))?;
    let count = |name| {
        value
            .get(name)
            .and_then(Value::as_u64)
            .and_then(|n| u32::try_from(n).ok())
            .ok_or_else(|| protocol("invalid Responses token count"))
    };
    let input_tokens = count("input_tokens")?;
    let output_tokens = count("output_tokens")?;
    input_tokens
        .checked_add(output_tokens)
        .ok_or_else(|| protocol("Responses total token count overflow"))?;
    let cached = value
        .pointer("/input_tokens_details/cached_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let cache_read_input_tokens =
        u32::try_from(cached).map_err(|_| protocol("invalid cached token count"))?;
    if cache_read_input_tokens > input_tokens {
        return Err(protocol("cached token subset exceeds input total"));
    }
    Ok(TokenUsage {
        input_tokens,
        output_tokens,
        cache_read_input_tokens,
        cache_creation_input_tokens: 0,
    })
}

#[cfg(test)]
#[path = "responses_tests.rs"]
mod tests;

async fn bounded_body(response: reqwest::Response, limit: usize) -> Result<Vec<u8>, ApiError> {
    let mut chunks = response.bytes_stream();
    let mut bytes = Vec::new();
    while let Some(chunk) = chunks.next().await {
        let chunk = chunk.map_err(|_| protocol("Responses body interrupted"))?;
        if bytes.len() + chunk.len() > limit {
            return Err(protocol("Responses body exceeds limit"));
        }
        bytes.extend_from_slice(&chunk);
    }
    Ok(bytes)
}
async fn check_response(
    response: reqwest::Response,
    secret: &str,
    request_id: &str,
) -> Result<reqwest::Response, ApiError> {
    let status = response.status();
    if status.is_success() {
        return Ok(response);
    }
    let bytes = bounded_body(response, 16 * 1024).await.unwrap_or_default();
    let value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    Err(ApiError::Http {
        status: status.as_u16(),
        body: diagnostic(&value, request_id, secret),
    })
}

/// Item-done frames carry complete output even when the final response omits it.
/// Keep them indexed until terminal validation and EOF authorize projection.
#[derive(Default)]
struct OutputItems {
    added: BTreeMap<u64, Value>,
    done: BTreeMap<u64, Value>,
    arguments: BTreeMap<u64, String>,
    arguments_done: BTreeMap<u64, String>,
    retained_bytes: usize,
}

impl OutputItems {
    fn observe(&mut self, kind: &str, event: &Value) -> Result<(), ApiError> {
        if kind == "response.output_item.added" {
            let index = output_index(event)?;
            let item = event
                .get("item")
                .ok_or_else(|| protocol("added item missing"))?;
            let id = item
                .get("id")
                .and_then(Value::as_str)
                .filter(|id| !id.is_empty())
                .ok_or_else(|| protocol("added item lacks identity"))?;
            if !matches!(
                item.get("type").and_then(Value::as_str),
                Some("message" | "reasoning" | "function_call")
            ) {
                return Err(protocol("unsupported added output item"));
            }
            if self.added.contains_key(&index)
                || self
                    .added
                    .values()
                    .any(|other| other.get("id").and_then(Value::as_str) == Some(id))
            {
                return Err(protocol("duplicate output index or identity"));
            }
            self.retain_entry(index, item.to_string().len())?;
            self.added.insert(index, item.clone());
            return Ok(());
        }
        let index = self.resolve(event)?;
        if self.done.contains_key(&index) {
            return Err(protocol("event after output item done"));
        }
        if kind == "response.output_item.done" {
            let item = event
                .get("item")
                .ok_or_else(|| protocol("done item missing"))?;
            self.check_item(index, item)?;
            self.retain_entry(index, item.to_string().len())?;
            serde_json::from_value::<ResponsesOutputItem>(item.clone())
                .map_err(|_| protocol("unsupported or malformed done output item"))?;
            self.done.insert(index, item.clone());
        } else {
            if self.added[&index].get("type").and_then(Value::as_str) != Some("function_call") {
                return Err(protocol("arguments event for non-function item"));
            }
            if self.arguments_done.contains_key(&index) {
                return Err(protocol("arguments event after arguments done"));
            }
            if kind == "response.function_call_arguments.delta" {
                let delta = event
                    .get("delta")
                    .and_then(Value::as_str)
                    .ok_or_else(|| protocol("arguments delta missing"))?;
                if self.arguments.contains_key(&index) {
                    self.retain(delta.len())?;
                } else {
                    self.retain_entry(index, delta.len())?;
                }
                self.arguments.entry(index).or_default().push_str(delta);
            } else {
                let args = event
                    .get("arguments")
                    .and_then(Value::as_str)
                    .ok_or_else(|| protocol("done arguments missing"))?;
                if self
                    .arguments
                    .get(&index)
                    .is_some_and(|observed| observed != args)
                {
                    return Err(protocol("done arguments disagree with deltas"));
                }
                self.retain_entry(index, args.len())?;
                self.arguments_done.insert(index, args.to_owned());
            }
        }
        Ok(())
    }

    // Charge every retained copy, not just each incoming frame. Serialized
    // item fields and map/string delimiters keep many small entries bounded.
    fn retain_entry(&mut self, index: u64, payload_bytes: usize) -> Result<(), ApiError> {
        let bytes = payload_bytes
            .checked_add(index.to_string().len() + 5)
            .ok_or_else(|| protocol("aggregate output assembly exceeds limit"))?;
        self.retain(bytes)
    }

    fn retain(&mut self, bytes: usize) -> Result<(), ApiError> {
        let total = self
            .retained_bytes
            .checked_add(bytes)
            .filter(|total| *total <= MAX_EVENT_BYTES)
            .ok_or_else(|| protocol("aggregate output assembly exceeds limit"))?;
        self.retained_bytes = total;
        Ok(())
    }

    fn resolve(&self, event: &Value) -> Result<u64, ApiError> {
        let index = if event.get("output_index").is_some() {
            output_index(event)?
        } else {
            let id = event
                .get("item_id")
                .and_then(Value::as_str)
                .ok_or_else(|| protocol("output event lacks index or item identity"))?;
            *self
                .added
                .iter()
                .find(|(_, item)| item.get("id").and_then(Value::as_str) == Some(id))
                .map(|(index, _)| index)
                .ok_or_else(|| protocol("orphan output event"))?
        };
        let item = self
            .added
            .get(&index)
            .ok_or_else(|| protocol("orphan output index"))?;
        if event
            .get("item_id")
            .is_some_and(|id| item.get("id") != Some(id))
        {
            return Err(protocol("output index and identity disagree"));
        }
        Ok(index)
    }

    fn check_item(&self, index: u64, item: &Value) -> Result<(), ApiError> {
        let added = &self.added[&index];
        for field in ["id", "type", "call_id", "name", "namespace"] {
            if added
                .get(field)
                .is_some_and(|value| !value.is_null() && item.get(field) != Some(value))
            {
                return Err(protocol("output item identity changed"));
            }
        }
        if item
            .get("status")
            .is_some_and(|status| !status.is_null() && status.as_str() != Some("completed"))
        {
            return Err(protocol("output item is not completed"));
        }
        for args in [self.arguments.get(&index), self.arguments_done.get(&index)]
            .into_iter()
            .flatten()
        {
            if item.get("arguments").and_then(Value::as_str) != Some(args) {
                return Err(protocol("output item arguments disagree with stream"));
            }
        }
        Ok(())
    }

    fn reconcile(&self, output: Option<&Value>) -> Result<Value, ApiError> {
        let output = output
            .and_then(Value::as_array)
            .ok_or_else(|| protocol("completed response lacks output array"))?;
        if output.is_empty() {
            let mut items = Vec::new();
            for (expected, index) in self.added.keys().enumerate() {
                if *index != expected as u64 {
                    return Err(protocol("incomplete output item indexes"));
                }
                items.push(
                    self.done
                        .get(index)
                        .ok_or_else(|| protocol("output item never completed"))?
                        .clone(),
                );
            }
            return Ok(Value::Array(items));
        }
        for index in self.added.keys() {
            let index_usize =
                usize::try_from(*index).map_err(|_| protocol("invalid output index"))?;
            let item = output
                .get(index_usize)
                .ok_or_else(|| protocol("terminal output omits tracked item"))?;
            self.check_item(*index, item)?;
            if self.done.get(index).is_some_and(|done| done != item) {
                return Err(protocol("terminal output disagrees with done item"));
            }
        }
        Ok(Value::Array(output.clone()))
    }
}
fn output_index(event: &Value) -> Result<u64, ApiError> {
    event
        .get("output_index")
        .and_then(Value::as_u64)
        .ok_or_else(|| protocol("invalid output index"))
}
