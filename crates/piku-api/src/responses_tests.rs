//! Synthetic protocol fixtures derived from Responses function-calling and
//! stateless-reasoning schemas; no credentials or live model calls.
use super::*;
use crate::{AccessToken, RequestMessage, SystemBlock, ToolDefinition};
use std::sync::atomic::{AtomicUsize, Ordering};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

const SECRET: &str = "syntheticbearercredentialsyntheticbearercredentialsyntheticbearercredentialsyntheticbearercredentialsyntheticbearercredentialsyntheticbearercredentialsyntheticbearercredentialsyntheticbearercredentialsyntheticbearercredential";
struct Credentials {
    calls: AtomicUsize,
}
impl AccessTokenSource for Credentials {
    fn token(
        &self,
    ) -> Pin<Box<dyn std::future::Future<Output = Result<AccessToken, ApiError>> + Send + '_>> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Box::pin(async { Ok(AccessToken::new(SECRET.into())) })
    }
    fn binding(&self) -> &'static str {
        "registration-fixture"
    }
}
fn request() -> MessageRequest {
    MessageRequest {
        model: "fixture-model".into(), max_tokens: 0, stream: true,
        system: Some(vec![SystemBlock::text("fixture instructions".into())]),
        messages: vec![RequestMessage { role:"user".into(),content:vec![RequestContent::Text {text:"fixture input".into()}] }],
        tools: Some(["read_file", "write_file"].into_iter().map(|name| ToolDefinition {name:name.into(),description:"fixture".into(),input_schema:json!({"type":"object","properties":{"path":{"type":"string"}},"required":["path"]})}).collect()),
    }
}
fn message(text: &str) -> Value {
    json!({"type":"message","id":"msg-fixture","role":"assistant","status":"completed","phase":"final_answer","content":[{"type":"output_text","text":text,"annotations":[]}]})
}
fn completion(output: Value) -> Value {
    let mut event = json!({"type":"response.completed","response":{"id":"resp-fixture","status":"completed","usage":{"input_tokens":100,"output_tokens":20,"total_tokens":120,"input_tokens_details":{"cached_tokens":40},"output_tokens_details":{"reasoning_tokens":10}}}});
    event["response"]["output"] = output;
    event
}
fn frames(events: &[Value]) -> String {
    use std::fmt::Write;
    events.iter().fold(String::new(), |mut text, event| {
        write!(
            &mut text,
            "event: {}\ndata: {event}\n\n",
            event["type"].as_str().unwrap()
        )
        .unwrap();
        text
    })
}

async fn server(responses: Vec<(u16, String)>) -> (String, tokio::task::JoinHandle<Vec<Value>>) {
    server_with_request_id(responses, "req-fixture").await
}

async fn server_with_request_id(
    responses: Vec<(u16, String)>,
    request_id: &'static str,
) -> (String, tokio::task::JoinHandle<Vec<Value>>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}/v1/responses", listener.local_addr().unwrap());
    let handle = tokio::spawn(async move {
        let mut bodies = Vec::new();
        for (status, response) in responses {
            let (mut stream, _) = tokio::time::timeout(Duration::from_secs(5), listener.accept())
                .await
                .unwrap()
                .unwrap();
            let mut bytes = Vec::new();
            let mut buffer = [0u8; 1024];
            let (end, length) = loop {
                let n = tokio::time::timeout(Duration::from_secs(3), stream.read(&mut buffer))
                    .await
                    .unwrap()
                    .unwrap();
                assert_ne!(n, 0);
                bytes.extend_from_slice(&buffer[..n]);
                if let Some(end) = bytes.windows(4).position(|w| w == b"\r\n\r\n") {
                    let headers = String::from_utf8_lossy(&bytes[..end]).to_ascii_lowercase();
                    assert!(
                        headers.starts_with("post /v1/responses ")
                            || headers.starts_with("get /v1/models ")
                    );
                    assert!(headers.contains(&format!("authorization: bearer {SECRET}")));
                    let length = headers
                        .lines()
                        .find_map(|line| line.strip_prefix("content-length:"))
                        .unwrap_or("0")
                        .trim()
                        .parse::<usize>()
                        .unwrap();
                    break (end + 4, length);
                }
            };
            while bytes.len() < end + length {
                let n = tokio::time::timeout(Duration::from_secs(3), stream.read(&mut buffer))
                    .await
                    .unwrap()
                    .unwrap();
                assert_ne!(n, 0);
                bytes.extend_from_slice(&buffer[..n]);
            }
            bodies.push(if length == 0 {
                Value::Null
            } else {
                serde_json::from_slice(&bytes[end..end + length]).unwrap()
            });
            stream.write_all(format!("HTTP/1.1 {status} Fixture\r\nContent-Type: text/event-stream\r\nx-request-id: {request_id}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",response.len()).as_bytes()).await.unwrap();
            for chunk in response.as_bytes().chunks(7) {
                stream.write_all(chunk).await.unwrap();
                tokio::task::yield_now().await;
            }
            stream.shutdown().await.unwrap();
        }
        bodies
    });
    (url, handle)
}
fn provider(url: String) -> (ResponsesProvider, Arc<Credentials>) {
    let source = Arc::new(Credentials {
        calls: AtomicUsize::new(0),
    });
    let mut provider = ResponsesProvider::new(source.clone()).unwrap();
    provider.endpoint = url;
    (provider, source)
}
async fn collect(
    provider: &ResponsesProvider,
    request: MessageRequest,
) -> Vec<Result<Event, ApiError>> {
    tokio::time::timeout(
        Duration::from_secs(5),
        provider.stream_message(request).collect(),
    )
    .await
    .unwrap()
}

fn resumed_request(replay: ReplayEnvelope) -> MessageRequest {
    let mut next = request();
    next.messages.push(RequestMessage {
        role: "assistant".into(),
        content: vec![
            RequestContent::Text {
                text: "duplicate projection".into(),
            },
            RequestContent::ToolUse {
                id: "call-read".into(),
                name: "read_file".into(),
                input: json!({"path":"a"}),
            },
            RequestContent::Replay { envelope: replay },
        ],
    });
    next.messages.push(RequestMessage {
        role: "user".into(),
        content: vec![
            RequestContent::ToolResult {
                tool_use_id: "call-read".into(),
                content: "a contents".into(),
                is_error: Some(false),
            },
            RequestContent::ToolResult {
                tool_use_id: "call-write".into(),
                content: "written".into(),
                is_error: Some(false),
            },
            RequestContent::Text {
                text: "continue".into(),
            },
        ],
    });
    next
}

fn assert_request_shape(body: &Value) {
    assert_eq!(body["store"], false);
    assert_eq!(body["stream"], true);
    assert_eq!(body["instructions"], "fixture instructions");
    assert_eq!(body["tools"][0]["type"], "namespace");
    assert_eq!(body["tools"][0]["name"], "piku");
    assert_eq!(body["tools"][0]["tools"][0]["type"], "function");
    for key in [
        "max_tokens",
        "max_output_tokens",
        "previous_response_id",
        "temperature",
        "defer_loading",
    ] {
        assert!(body.get(key).is_none());
    }
}

#[tokio::test]
async fn real_http_interleaved_calls_replay_and_tool_results() {
    let output = json!([
        {"type":"reasoning","id":"rs-fixture","summary":[],"encrypted_content":"opaque-encrypted-fixture"},
        {"type":"message","id":"commentary","role":"assistant","status":"completed","phase":"commentary","content":[{"type":"output_text","text":"hello α","annotations":[]}]},
        {"type":"function_call","id":"item-read","call_id":"call-read","namespace":"piku","name":"read_file","arguments":"{\"path\":\"a\"}","status":"completed"},
        {"type":"function_call","id":"item-write","call_id":"call-write","namespace":"piku","name":"write_file","arguments":"{\"path\":\"b\"}","status":"completed"}
    ]);
    let events = [
        json!({"type":"response.output_text.delta","delta":"hello α"}),
        json!({"type":"response.output_item.added","output_index":2,"item":{"type":"function_call","id":"item-read","call_id":"call-read","name":"read_file","namespace":"piku","arguments":""}}),
        json!({"type":"response.function_call_arguments.delta","item_id":"item-read","delta":"{\"path\":"}),
        json!({"type":"response.output_item.added","output_index":3,"item":{"type":"function_call","id":"item-write","call_id":"call-write","name":"write_file","namespace":"piku","arguments":""}}),
        json!({"type":"response.function_call_arguments.delta","item_id":"item-write","delta":"{\"path\":\"b\"}"}),
        json!({"type":"response.function_call_arguments.delta","item_id":"item-read","delta":"\"a\"}"}),
        completion(output.clone()),
    ];
    let (url, server) = server(vec![
        (200, frames(&events)),
        (200, frames(&[completion(json!([message("done")]))])),
    ])
    .await;
    let (provider, source) = provider(url);
    let events: Vec<_> = collect(&provider, request())
        .await
        .into_iter()
        .map(Result::unwrap)
        .collect();
    assert!(matches!(&events[0],Event::TextDelta{text} if text=="hello α"));
    let ids: Vec<_> = events
        .iter()
        .filter_map(|event| {
            if let Event::ToolUseStart { id, .. } = event {
                Some(id.as_str())
            } else {
                None
            }
        })
        .collect();
    assert_eq!(ids, ["call-read", "call-write"]);
    let replay = events
        .iter()
        .find_map(|event| {
            if let Event::Replay { envelope } = event {
                Some(envelope.clone())
            } else {
                None
            }
        })
        .unwrap();
    assert_eq!(serde_json::to_value(&replay.output).unwrap(), output);
    assert!(!format!("{replay:?}").contains("opaque"));
    let usage: Vec<_> = events
        .iter()
        .filter_map(|event| {
            if let Event::UsageDelta { usage } = event {
                Some(usage)
            } else {
                None
            }
        })
        .collect();
    assert_eq!(usage.len(), 1);
    assert_eq!(usage[0].total_tokens(), 120);
    assert_eq!(usage[0].cache_read_input_tokens, 40);
    assert_eq!(
        events
            .iter()
            .filter(|e| matches!(
                e,
                Event::MessageStop {
                    stop_reason: StopReason::ToolUse
                }
            ))
            .count(),
        1
    );
    let next = resumed_request(replay);
    assert!(collect(&provider, next).await.iter().all(Result::is_ok));
    let bodies = server.await.unwrap();
    assert_eq!(bodies.len(), 2);
    assert_eq!(source.calls.load(Ordering::SeqCst), 2);
    assert_request_shape(&bodies[0]);
    let input = bodies[1]["input"].as_array().unwrap();
    assert_eq!(&input[1..5], output.as_array().unwrap());
    assert_eq!(
        input[5],
        json!({"type":"function_call_output","call_id":"call-read","output":"a contents"})
    );
    assert_eq!(input[6]["call_id"], "call-write");
    assert_eq!(input[7]["content"], "continue");
    assert_eq!(input.len(), 8);
}

#[tokio::test]
async fn failures_after_text_never_release_tools_or_completion() {
    let call = json!({"type":"function_call","id":"item","call_id":"call","name":"read_file","namespace":"piku","arguments":"{}"});
    let done = completion(json!([call]));
    let cases = vec![
        vec![],
        vec![
            json!({"type":"response.failed","response":{"error":{"code":"subscription_sharing_usage_limit_exceeded","message":SECRET}}}),
        ],
        vec![
            json!({"type":"response.incomplete","response":{"incomplete_details":{"reason":"max_output_tokens"}}}),
        ],
        vec![done.clone(), done.clone()],
        vec![
            done,
            json!({"type":"error","code":"late_failure","param":"tools[0].type","message":SECRET,"sequence_number":9}),
        ],
        vec![completion(
            json!([{"type":"unknown_future_tool","secret":SECRET}]),
        )],
        vec![completion(
            json!([{"type":"function_call","id":"i","call_id":"c","name":"read_file","namespace":"foreign","arguments":"{}"}]),
        )],
    ];
    for mut case in cases {
        case.insert(
            0,
            json!({"type":"response.output_text.delta","delta":"partial"}),
        );
        let (url, server) = server(vec![(200, frames(&case))]).await;
        let (provider, _) = provider(url);
        let events = collect(&provider, request()).await;
        assert!(matches!(&events[0],Ok(Event::TextDelta{text}) if text=="partial"));
        assert!(events.iter().any(Result::is_err));
        assert!(!events.iter().any(|e| matches!(
            e,
            Ok(Event::ToolUseStart { .. }
                | Event::Replay { .. }
                | Event::UsageDelta { .. }
                | Event::MessageStop { .. })
        )));
        for error in events.into_iter().filter_map(Result::err) {
            assert!(!error.to_string().contains(SECRET));
        }
        server.await.unwrap();
    }
}

#[tokio::test]
async fn http_error_is_bounded_redacted_and_keeps_request_id() {
    let (url,server)=server(vec![(429,json!({"error":{"code":"subscription_sharing_usage_limit_exceeded","param":"model","message":SECRET}}).to_string())]).await;
    let (provider, _) = provider(url);
    let events = collect(&provider, request()).await;
    assert_eq!(events.len(), 1);
    let error = events[0].as_ref().unwrap_err();
    assert!(matches!(error, ApiError::Http { status: 429, .. }));
    let text = error.to_string();
    assert!(text.contains("subscription_sharing_usage_limit_exceeded"));
    assert!(text.contains("req-fixture"));
    assert!(!text.contains(SECRET));
    assert!(text.len() < 512);
    server.await.unwrap();
    assert_eq!(
        format!("{:?}", AccessToken::new(SECRET.into())),
        "AccessToken([REDACTED])"
    );
}

fn envelope() -> ReplayEnvelope {
    ReplayEnvelope {
        version: 1,
        route: ReplayRoute::ChatgptResponses,
        registration: "registration-fixture".into(),
        model: "fixture-model".into(),
        output: serde_json::from_value(json!([message("prior")])).unwrap(),
    }
}
#[tokio::test]
async fn mismatched_replay_and_caps_fail_before_credentials_or_http() {
    let (provider, source) = provider("http://127.0.0.1:9/v1/responses".into());
    for field in ["registration", "model", "version", "role", "cap"] {
        let mut request = request();
        let mut replay = envelope();
        match field {
            "registration" => replay.registration = "different".into(),
            "model" => replay.model = "different".into(),
            "version" => replay.version = 2,
            "cap" => request.max_tokens = 1,
            _ => {}
        }
        request.messages.push(RequestMessage {
            role: if field == "role" { "user" } else { "assistant" }.into(),
            content: vec![RequestContent::Replay { envelope: replay }],
        });
        let events = collect(&provider, request).await;
        assert_eq!(events.len(), 1);
        assert!(events[0].is_err());
    }
    assert_eq!(source.calls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn legacy_providers_reject_replay_without_sending_http() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let providers: Vec<Box<dyn Provider>> = vec![
        Box::new(crate::AnthropicProvider::with_base_url(
            "synthetic".into(),
            url.clone(),
        )),
        Box::new(crate::OpenAiCompatProvider::new(
            "legacy",
            url,
            crate::openai_compat::Auth::None,
            vec![],
        )),
    ];
    for provider in providers {
        let mut request = request();
        request.messages.push(RequestMessage {
            role: "assistant".into(),
            content: vec![RequestContent::Replay {
                envelope: envelope(),
            }],
        });
        let events: Vec<_> = provider.stream_message(request).collect().await;
        assert_eq!(events.len(), 1);
        assert!(
            matches!(&events[0],Err(ApiError::Provider(message)) if message.contains("replay"))
        );
    }
    assert!(
        tokio::time::timeout(Duration::from_millis(50), listener.accept())
            .await
            .is_err()
    );
}

#[tokio::test]
async fn catalog_uses_selected_registration_and_official_shape() {
    let (url,server)=server(vec![(200,json!({"models":[{"slug":"second","display_name":"Second","visibility":"list"},{"slug":"hidden","display_name":"Hidden","visibility":"hidden"},{"slug":"first","display_name":"First","visibility":"list"}]}).to_string()),(200,json!({"data":[{"id":"wrong-protocol"}]}).to_string())]).await;
    let (provider, source) = provider(url);
    let models = provider.models().await.unwrap();
    assert_eq!(
        models,
        vec![
            ChatgptModel {
                slug: "second".into(),
                display_name: "Second".into()
            },
            ChatgptModel {
                slug: "first".into(),
                display_name: "First".into()
            }
        ]
    );
    assert!(provider.models().await.is_err());
    assert_eq!(source.calls.load(Ordering::SeqCst), 2);
    assert_eq!(server.await.unwrap(), vec![Value::Null, Value::Null]);
}

#[tokio::test]
async fn dropping_stream_cancels_pending_http_body() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}/v1/responses", listener.local_addr().unwrap());
    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let mut request = Vec::new();
        let mut bytes = [0u8; 4096];
        let (end, length) = loop {
            let n = socket.read(&mut bytes).await.unwrap();
            assert_ne!(n, 0);
            request.extend_from_slice(&bytes[..n]);
            if let Some(end) = request.windows(4).position(|w| w == b"\r\n\r\n") {
                let header = String::from_utf8_lossy(&request[..end]).to_ascii_lowercase();
                let length = header
                    .lines()
                    .find_map(|line| line.strip_prefix("content-length:"))
                    .unwrap()
                    .trim()
                    .parse::<usize>()
                    .unwrap();
                break (end + 4, length);
            }
        };
        while request.len() < end + length {
            let n = socket.read(&mut bytes).await.unwrap();
            assert_ne!(n, 0);
            request.extend_from_slice(&bytes[..n]);
        }
        socket
            .write_all(
                b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n",
            )
            .await
            .unwrap();
        socket
            .write_all(
                frames(&[json!({"type":"response.output_text.delta","delta":"partial"})])
                    .as_bytes(),
            )
            .await
            .unwrap();
        tokio::time::timeout(Duration::from_secs(2), socket.read(&mut bytes))
            .await
            .unwrap()
            .unwrap()
    });
    let (provider, source) = provider(url);
    let mut stream = provider.stream_message(request());
    assert!(
        matches!(tokio::time::timeout(Duration::from_secs(2),stream.next()).await.unwrap(),Some(Ok(Event::TextDelta{text})) if text=="partial")
    );
    drop(stream);
    assert_eq!(server.await.unwrap(), 0);
    assert_eq!(source.calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn narrowed_catalog_replays_history_but_rejects_new_removed_tool() {
    let old_call = json!({"type":"function_call","id":"old-item","call_id":"old-call","name":"write_file","namespace":"piku","arguments":"{\"path\":\"old\"}"});
    let mut replay = envelope();
    replay.output = serde_json::from_value(json!([old_call.clone()])).unwrap();
    let mut request = request();
    request
        .tools
        .as_mut()
        .unwrap()
        .retain(|tool| tool.name == "read_file");
    request.messages.push(RequestMessage {
        role: "assistant".into(),
        content: vec![RequestContent::Replay { envelope: replay }],
    });
    request.messages.push(RequestMessage {
        role: "user".into(),
        content: vec![RequestContent::ToolResult {
            tool_use_id: "old-call".into(),
            content: "prior write".into(),
            is_error: Some(false),
        }],
    });
    let (url, server) = server(vec![(
        200,
        frames(&[completion(json!([old_call.clone()]))]),
    )])
    .await;
    let (provider, _) = provider(url);
    let events = collect(&provider, request).await;
    assert_eq!(events.len(), 1);
    assert!(events[0].is_err());
    // No new call projection is delivered for execution, even though the
    // immutable historical item is valid input under the narrowed catalog.
    let bodies = server.await.unwrap();
    assert_eq!(bodies[0]["input"][1], old_call);
    assert_eq!(bodies[0]["tools"][0]["tools"].as_array().unwrap().len(), 1);
}

#[tokio::test]
async fn authoritative_text_and_refusal_fallback_and_projection_agreement() {
    let mut refusal = message("unused");
    refusal["content"] = json!([{"type":"refusal","refusal":"cannot do that"}]);
    let cases = vec![
        (
            vec![completion(json!([message("final only")]))],
            true,
            "final only",
        ),
        (
            vec![completion(json!([refusal.clone()]))],
            true,
            "cannot do that",
        ),
        (
            vec![
                json!({"type":"response.refusal.delta","delta":"cannot do that"}),
                completion(json!([refusal])),
            ],
            true,
            "cannot do that",
        ),
        (
            vec![
                json!({"type":"response.output_text.delta","output_index":0,"content_index":0,"delta":"first"}),
                json!({"type":"response.output_text.delta","output_index":1,"content_index":0,"delta":"second"}),
                completion(json!([message("first"), message("second")])),
            ],
            true,
            "firstsecond",
        ),
        // Interleaved text from different items cannot silently become a
        // successful transcript whose ordering differs from retained output.
        (
            vec![
                json!({"type":"response.output_text.delta","output_index":1,"content_index":0,"delta":"second"}),
                json!({"type":"response.output_text.delta","output_index":0,"content_index":0,"delta":"first"}),
                completion(json!([message("first"), message("second")])),
            ],
            false,
            "secondfirst",
        ),
    ];
    for (frames_to_send, success, expected_text) in cases {
        let (url, server) = server(vec![(200, frames(&frames_to_send))]).await;
        let (provider, _) = provider(url);
        let events = collect(&provider, request()).await;
        let text: String = events
            .iter()
            .filter_map(|event| {
                if let Ok(Event::TextDelta { text }) = event {
                    Some(text.as_str())
                } else {
                    None
                }
            })
            .collect();
        assert_eq!(text, expected_text);
        assert_eq!(events.iter().all(Result::is_ok), success);
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, Ok(Event::MessageStop { .. })))
                .count(),
            usize::from(success)
        );
        if !success {
            assert!(!events.iter().any(|event| matches!(
                event,
                Ok(Event::Replay { .. } | Event::ToolUseStart { .. })
            )));
        }
        server.await.unwrap();
    }
}

#[tokio::test]
async fn official_top_level_error_preserves_code_param_and_redacts_message() {
    let (url, server) = server(vec![(200, frames(&[json!({"type":"error","code":"usage_limit","param":"tools[0].type","message":SECRET,"sequence_number":1})]))]).await;
    let (provider, _) = provider(url);
    let events = collect(&provider, request()).await;
    assert_eq!(events.len(), 1);
    let error = events[0].as_ref().unwrap_err().to_string();
    assert!(error.contains("usage_limit"));
    assert!(error.contains("tools[0].type"));
    assert!(error.contains("req-fixture"));
    assert!(!error.contains(SECRET));
    server.await.unwrap();
}

#[tokio::test]
async fn long_credential_echo_is_redacted_before_identifier_truncation() {
    let responses = vec![
        (
            401,
            json!({"error":{"code":SECRET,"param":SECRET}}).to_string(),
        ),
        (
            200,
            frames(&[
                json!({"type":"error","code":SECRET,"param":SECRET,"message":"private","sequence_number":1}),
            ]),
        ),
        (
            401,
            json!({"error":{"code":SECRET,"param":SECRET}}).to_string(),
        ),
    ];
    let (url, server) = server_with_request_id(responses, SECRET).await;
    let (provider, _) = provider(url);
    for _ in 0..2 {
        let events = collect(&provider, request()).await;
        assert_eq!(events.len(), 1);
        let diagnostic = events[0].as_ref().unwrap_err().to_string();
        assert!(
            !diagnostic.contains(&SECRET[..32]),
            "credential prefix leaked"
        );
    }
    let diagnostic = provider.models().await.unwrap_err().to_string();
    assert!(
        !diagnostic.contains(&SECRET[..32]),
        "catalog credential prefix leaked"
    );
    server.await.unwrap();
}

#[tokio::test]
async fn incomplete_function_items_never_release_executable_events() {
    for status in ["incomplete", "in_progress"] {
        let output = json!([{"type":"function_call","id":"item","call_id":"call","name":"read_file","namespace":"piku","arguments":"{}","status":status}]);
        let (url, server) = server(vec![(200, frames(&[completion(output)]))]).await;
        let (provider, _) = provider(url);
        let events = collect(&provider, request()).await;
        assert!(
            events.iter().any(Result::is_err),
            "noncompleted item accepted"
        );
        assert!(!events.iter().any(|event| matches!(
            event,
            Ok(Event::ToolUseStart { .. }
                | Event::Replay { .. }
                | Event::UsageDelta { .. }
                | Event::MessageStop { .. })
        )));
        server.await.unwrap();
    }
}

#[test]
fn public_preflight_rejects_mismatched_replay_without_credentials() {
    let (provider, source) = provider("http://127.0.0.1:1/v1/responses".into());
    assert!(provider.validate_request(&request()).is_ok());
    for field in ["registration", "model", "version", "cap"] {
        let mut replay = envelope();
        let mut request = request();
        match field {
            "registration" => replay.registration = "other-registration".into(),
            "model" => replay.model = "other-model".into(),
            "version" => replay.version = 99,
            _ => request.max_tokens = 1,
        }
        request.messages.push(RequestMessage {
            role: "assistant".into(),
            content: vec![RequestContent::Replay { envelope: replay }],
        });
        assert!(provider.validate_request(&request).is_err());
    }
    assert_eq!(source.calls.load(Ordering::SeqCst), 0);
}
