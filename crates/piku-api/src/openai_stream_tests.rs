//! Captured `OpenRouter` terminal/usage shape with synthetic tool identity.
use super::*;
use serde_json::{json, Value};
use std::fmt::Write as _;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

async fn collect_wire(rows: Vec<Value>) -> Vec<Result<Event, ApiError>> {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let provider = OpenAiCompatProvider::new(
        "fixture",
        format!("http://{}", listener.local_addr().unwrap()),
        Auth::None,
        vec![],
    );
    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let mut bytes = Vec::new();
        let mut buffer = [0; 4096];
        loop {
            let n = socket.read(&mut buffer).await.unwrap();
            assert_ne!(n, 0);
            bytes.extend_from_slice(&buffer[..n]);
            if let Some(end) = bytes.windows(4).position(|w| w == b"\r\n\r\n") {
                let headers = String::from_utf8_lossy(&bytes[..end]).to_ascii_lowercase();
                let length: usize = headers
                    .lines()
                    .find_map(|line| line.strip_prefix("content-length:"))
                    .unwrap()
                    .trim()
                    .parse()
                    .unwrap();
                if bytes.len() >= end + 4 + length {
                    break;
                }
            }
        }
        let mut body = String::new();
        for row in rows {
            write!(body, "data: {row}\n\n").unwrap();
        }
        body.push_str("data: [DONE]\n\n");
        socket.write_all(format!("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",body.len()).as_bytes()).await.unwrap();
        for chunk in body.as_bytes().chunks(13) {
            if socket.write_all(chunk).await.is_err() {
                break;
            }
        }
    });
    let request = MessageRequest {
        model: "fixture-model".into(),
        max_tokens: 128,
        stream: true,
        system: None,
        messages: vec![],
        tools: None,
    };
    let events = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        provider.stream_message(request).collect(),
    )
    .await
    .unwrap();
    server.await.unwrap();
    events
}

#[tokio::test]
async fn captured_duplicate_terminal_preserves_usage_and_real_call_identity() {
    let rows = serde_json::from_str(include_str!("openai_stream_fixture.json")).unwrap();
    let events = collect_wire(rows).await;
    assert!(events.iter().all(Result::is_ok));
    let events: Vec<_> = events.into_iter().map(Result::unwrap).collect();
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
    let mut arguments = String::new();
    for event in &events {
        if let Event::ToolUseDelta { id, partial_json } = event {
            assert_eq!(id, "synthetic-call");
            arguments.push_str(partial_json);
        }
    }
    assert_eq!(
        serde_json::from_str::<Value>(&arguments).unwrap(),
        json!({"path":"version.txt"})
    );
    let usages: Vec<_> = events
        .iter()
        .filter_map(|e| {
            if let Event::UsageDelta { usage } = e {
                Some(usage)
            } else {
                None
            }
        })
        .collect();
    assert_eq!(usages.len(), 1);
    assert_eq!(usages[0].input_tokens, 394);
    assert_eq!(usages[0].output_tokens, 52);
}

#[tokio::test]
async fn conflicting_terminal_late_payload_and_multiple_choices_fail() {
    let stop = json!({"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]});
    for rows in [
        vec![
            stop.clone(),
            json!({"choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":1}}),
            json!({"choices":[{"index":0,"delta":{"content":"late after usage"},"finish_reason":null}]}),
        ],
        vec![
            stop.clone(),
            json!({"choices":[],"error":{"message":"synthetic late failure"}}),
        ],
        vec![
            stop.clone(),
            json!({"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}),
        ],
        vec![
            stop.clone(),
            json!({"choices":[{"index":0,"delta":{"content":"late"},"finish_reason":"stop"}]}),
        ],
        vec![json!({"choices":[{"index":1,"delta":{},"finish_reason":"stop"}]})],
        vec![
            json!({"choices":[{"index":0,"delta":{},"finish_reason":"stop"},{"index":1,"delta":{},"finish_reason":"stop"}]}),
        ],
    ] {
        assert!(collect_wire(rows).await.iter().any(Result::is_err));
    }
}

#[tokio::test]
async fn interleaved_nonarrival_indexes_keep_arguments_with_their_calls() {
    let rows = vec![
        json!({"choices":[{"index":0,"delta":{"tool_calls":[{"index":4,"id":"four","function":{"name":"read_file","arguments":"{"}},{"index":2,"id":"two","function":{"name":"read_file","arguments":"{"}}]},"finish_reason":null}]}),
        json!({"choices":[{"index":0,"delta":{"tool_calls":[{"index":2,"function":{"arguments":"\"path\":\"two\"}"}},{"index":4,"function":{"arguments":"\"path\":\"four\"}"}}]},"finish_reason":null}]}),
        json!({"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}),
    ];
    let events = collect_wire(rows).await;
    let mut calls = std::collections::BTreeMap::<String, String>::new();
    for event in events {
        if let Event::ToolUseDelta { id, partial_json } = event.unwrap() {
            calls.entry(id).or_default().push_str(&partial_json);
        }
    }
    assert_eq!(calls.get("four").unwrap(), "{\"path\":\"four\"}");
    assert_eq!(calls.get("two").unwrap(), "{\"path\":\"two\"}");
    assert_eq!(calls.len(), 2);
}
