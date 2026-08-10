use std::pin::Pin;
use std::sync::Arc;

use futures_util::stream::{self, Stream};
use futures_util::StreamExt;
use piku_api::{ApiError, Event, MessageRequest, Provider, StopReason, TokenUsage};
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::sync::Mutex;

const PROTOCOL: &str = "page.propose.v1";
const MAX_REQUEST_BYTES: usize = 1024 * 1024;
const MAX_RESPONSE_BYTES: usize = 1024 * 1024;

#[derive(Clone)]
pub(super) struct PageBroker {
    io: Arc<Mutex<tokio::fs::File>>,
}

#[derive(Serialize)]
struct RequestFrame<'a> {
    protocol: &'static str,
    request_id: &'a str,
    request: &'a MessageRequest,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ResponseFrame {
    protocol: String,
    request_id: String,
    ok: bool,
    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    usage: Option<TokenUsage>,
    #[serde(default)]
    error: Option<String>,
}

impl PageBroker {
    pub(super) fn is_configured() -> bool {
        std::env::var_os("PIKU_PAGE_BROKER_FD").is_some()
    }

    pub(super) fn configured_model() -> String {
        std::env::var("PIKU_PAGE_BROKER_MODEL").unwrap_or_else(|_| "page-broker".into())
    }

    pub(super) async fn from_env() -> Result<Option<Self>, ApiError> {
        let Some(fd) = std::env::var_os("PIKU_PAGE_BROKER_FD") else {
            return Ok(None);
        };
        let fd = fd
            .to_str()
            .and_then(|value| value.parse::<i32>().ok())
            .ok_or_else(|| provider_error("invalid file descriptor"))?;
        Self::from_fd(fd).await.map(Some)
    }

    pub(super) async fn from_fd(fd: i32) -> Result<Self, ApiError> {
        if fd < 0 {
            return Err(provider_error("invalid file descriptor"));
        }
        let io = tokio::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(format!("/dev/fd/{fd}"))
            .await
            .map_err(|_| provider_error("cannot open broker channel"))?;
        Ok(Self {
            io: Arc::new(Mutex::new(io)),
        })
    }

    async fn exchange(&self, request: MessageRequest) -> Result<Vec<Event>, ApiError> {
        let request_id = crate::new_session_id();
        let mut bytes = encode_request(&request_id, &request)?;
        bytes.push(b'\n');

        let mut io = self.io.lock().await;
        io.write_all(&bytes)
            .await
            .map_err(|_| provider_error("broker write failed"))?;
        io.flush()
            .await
            .map_err(|_| provider_error("broker write failed"))?;

        let mut response = Vec::new();
        loop {
            let byte = io
                .read_u8()
                .await
                .map_err(|_| provider_error("broker read failed"))?;
            if byte == b'\n' {
                break;
            }
            if response.len() == MAX_RESPONSE_BYTES {
                return Err(provider_error("broker response too large"));
            }
            response.push(byte);
        }
        decode_response(&request_id, &response)
    }
}

impl Provider for PageBroker {
    fn stream_message(
        &self,
        request: MessageRequest,
    ) -> Pin<Box<dyn Stream<Item = Result<Event, ApiError>> + Send + '_>> {
        Box::pin(
            stream::once(async move { self.exchange(request).await }).flat_map(
                |result| match result {
                    Ok(events) => stream::iter(events.into_iter().map(Ok)).left_stream(),
                    Err(error) => stream::once(async move { Err(error) }).right_stream(),
                },
            ),
        )
    }

    fn name(&self) -> &'static str {
        "page-broker"
    }

    fn boxed_clone(&self) -> Box<dyn Provider + Send + Sync + 'static> {
        Box::new(self.clone())
    }
}

fn encode_request(request_id: &str, request: &MessageRequest) -> Result<Vec<u8>, ApiError> {
    if request
        .tools
        .as_ref()
        .is_some_and(|tools| !tools.is_empty())
    {
        return Err(provider_error("page broker does not allow tools"));
    }
    let bytes = serde_json::to_vec(&RequestFrame {
        protocol: PROTOCOL,
        request_id,
        request,
    })
    .map_err(|_| provider_error("cannot encode broker request"))?;
    if bytes.len() > MAX_REQUEST_BYTES {
        return Err(provider_error("page broker request too large"));
    }
    Ok(bytes)
}

fn decode_response(request_id: &str, bytes: &[u8]) -> Result<Vec<Event>, ApiError> {
    let response: ResponseFrame =
        serde_json::from_slice(bytes).map_err(|_| provider_error("invalid broker response"))?;
    if response.protocol != PROTOCOL {
        return Err(provider_error("broker protocol mismatch"));
    }
    if response.request_id != request_id {
        return Err(provider_error("broker request ID mismatch"));
    }
    match (response.ok, response.text, response.usage, response.error) {
        (true, Some(text), Some(usage), None) => Ok(vec![
            Event::TextDelta { text },
            Event::UsageDelta { usage },
            Event::MessageStop {
                stop_reason: StopReason::EndTurn,
            },
        ]),
        (false, None, None, Some(error)) => {
            let _ = error;
            Err(provider_error("broker rejected request"))
        }
        _ => Err(provider_error("invalid broker response")),
    }
}

fn provider_error(message: &str) -> ApiError {
    ApiError::Provider(format!("page broker: {message}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use piku_api::{RequestMessage, ToolDefinition};

    fn request() -> MessageRequest {
        MessageRequest {
            model: "test-model".into(),
            max_tokens: 16,
            messages: vec![RequestMessage {
                role: "user".into(),
                content: Vec::new(),
            }],
            system: None,
            tools: None,
            stream: true,
        }
    }

    #[test]
    fn rejects_tools_before_encoding() {
        let mut request = request();
        request.tools = Some(vec![ToolDefinition {
            name: "forbidden".into(),
            description: String::new(),
            input_schema: serde_json::json!({}),
        }]);
        let error = encode_request("request-1", &request).unwrap_err();
        assert_eq!(
            error.to_string(),
            "provider error: page broker: page broker does not allow tools"
        );
    }

    #[test]
    fn rejects_oversize_request() {
        let mut request = request();
        request.model = "x".repeat(MAX_REQUEST_BYTES);
        let error = encode_request("request-1", &request).unwrap_err();
        assert_eq!(
            error.to_string(),
            "provider error: page broker: page broker request too large"
        );
    }

    #[test]
    fn rejects_mismatched_response_identity() {
        let response = serde_json::json!({
            "protocol": PROTOCOL,
            "request_id": "other",
            "ok": true,
            "text": "source",
            "usage": {"input_tokens": 1, "output_tokens": 2}
        });
        let error =
            decode_response("request-1", &serde_json::to_vec(&response).unwrap()).unwrap_err();
        assert_eq!(
            error.to_string(),
            "provider error: page broker: broker request ID mismatch"
        );
    }

    #[test]
    fn converts_success_to_provider_events() {
        let response = serde_json::json!({
            "protocol": PROTOCOL,
            "request_id": "request-1",
            "ok": true,
            "text": "source",
            "usage": {"input_tokens": 3, "output_tokens": 5}
        });
        let events = decode_response("request-1", &serde_json::to_vec(&response).unwrap()).unwrap();
        assert!(matches!(&events[0], Event::TextDelta { text } if text == "source"));
        assert!(
            matches!(&events[1], Event::UsageDelta { usage } if usage.input_tokens == 3 && usage.output_tokens == 5)
        );
        assert!(matches!(
            &events[2],
            Event::MessageStop {
                stop_reason: StopReason::EndTurn
            }
        ));
    }
}
