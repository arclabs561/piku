//! Captured `OpenRouter` terminal/usage shape with synthetic tool identity.
use super::*;
use crate::{InferenceAttemptReceipt, InferenceObservation};
use serde_json::{json, Value};
use std::fmt::Write as _;
use std::sync::Mutex;
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

async fn collect_wire(rows: Vec<Value>) -> Vec<Result<Event, ApiError>> {
    let (provider, server) = wire_provider(rows, 200, false, 0).await;
    let events = collect(&provider).await;
    server.await.unwrap();
    events
}

async fn wire_provider(
    rows: Vec<Value>,
    status: u16,
    hold_open: bool,
    length_offset: usize,
) -> (OpenAiCompatProvider, tokio::task::JoinHandle<()>) {
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
                    assert!(headers.starts_with("post /chat/completions "));
                    let request: Value =
                        serde_json::from_slice(&bytes[end + 4..end + 4 + length]).unwrap();
                    assert_eq!(request["model"], "fixture-model");
                    assert_eq!(request["stream_options"]["include_usage"], true);
                    break;
                }
            }
        }
        let mut body = String::new();
        for row in rows {
            write!(body, "data: {row}\n\n").unwrap();
        }
        body.push_str("data: [DONE]\n\n");
        let length = if hold_open {
            String::new()
        } else {
            format!("Content-Length: {}\r\n", body.len() + length_offset)
        };
        socket.write_all(format!("HTTP/1.1 {status} Fixture\r\nContent-Type: text/event-stream\r\n{length}Connection: close\r\n\r\n").as_bytes()).await.unwrap();
        for chunk in body.as_bytes().chunks(13) {
            if socket.write_all(chunk).await.is_err() {
                break;
            }
        }
        if hold_open {
            match tokio::time::timeout(Duration::from_secs(3), socket.read(&mut buffer))
                .await
                .unwrap()
            {
                Ok(0) => {}
                Err(error) if error.kind() == std::io::ErrorKind::ConnectionReset => {}
                result => panic!("dropped body did not disconnect: {result:?}"),
            }
        }
    });
    (provider, server)
}

fn request() -> MessageRequest {
    MessageRequest {
        model: "fixture-model".into(),
        max_tokens: 128,
        stream: true,
        system: None,
        messages: vec![],
        tools: None,
    }
}

async fn collect(provider: &dyn Provider) -> Vec<Result<Event, ApiError>> {
    tokio::time::timeout(
        Duration::from_secs(5),
        provider.stream_message(request()).collect(),
    )
    .await
    .unwrap()
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

type ObservationLog = Arc<Mutex<Vec<InferenceObservation>>>;

fn observer() -> (Arc<InferenceObserver>, ObservationLog) {
    let log = Arc::new(Mutex::new(Vec::new()));
    let captured = Arc::clone(&log);
    (
        Arc::new(move |event| captured.lock().unwrap().push(event)),
        log,
    )
}

fn receipt(log: &ObservationLog) -> InferenceAttemptReceipt {
    let events = log.lock().unwrap();
    assert_eq!(events.len(), 2);
    let InferenceObservation::Started {
        attempt_id,
        started_at,
    } = &events[0]
    else {
        panic!("missing send start")
    };
    let InferenceObservation::Finished(receipt) = &events[1] else {
        panic!("missing send finish")
    };
    assert_eq!(*attempt_id, receipt.attempt_id);
    assert_eq!(*started_at, receipt.started_at);
    assert!(receipt.finished_at >= receipt.started_at);
    let debug = format!("{events:?}");
    for excluded in [
        "fixture-model",
        "synthetic-call",
        "version.txt",
        "private payload",
    ] {
        assert!(!debug.contains(excluded));
    }
    receipt.clone()
}

fn terminal() -> Value {
    json!({"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]})
}

fn usage() -> Value {
    json!({"choices":[],"usage":{"prompt_tokens":3,"completion_tokens":1,"total_tokens":4,"prompt_tokens_details":{"cached_tokens":2},"completion_tokens_details":{"reasoning_tokens":1}}})
}

fn expected_usage() -> InferenceReportedUsage {
    InferenceReportedUsage {
        input_tokens: 3,
        output_tokens: 1,
        cached_input_tokens: Some(2),
        reasoning_output_tokens: Some(1),
    }
}

#[tokio::test]
async fn observed_clone_and_boxed_clone_preserve_real_transport_and_aliases() {
    let (callback, log) = observer();
    // The public Responses alias accepts the same Arc without adapting events.
    let old_name: Arc<crate::responses::ResponsesObserver> = Arc::clone(&callback);
    let (provider, server) = wire_provider(vec![terminal(), usage()], 200, false, 0).await;
    let observed = provider.observed_clone(old_name).unwrap();
    let events = collect(observed.boxed_clone().as_ref()).await;
    assert!(events.iter().all(Result::is_ok));
    server.await.unwrap();
    let first = receipt(&log);
    assert_eq!(first.outcome, InferenceAttemptOutcome::Completed);
    assert!(first.transport_complete);
    assert_eq!(first.reported_usage, Some(expected_usage()));

    let (callback, second_log) = observer();
    let (provider, server) = wire_provider(vec![terminal(), usage()], 200, false, 0).await;
    assert!(collect(&provider.with_observer(callback))
        .await
        .iter()
        .all(Result::is_ok));
    server.await.unwrap();
    assert_ne!(first.attempt_id, receipt(&second_log).attempt_id);
}

#[tokio::test]
async fn trailing_protocol_and_http_body_errors_keep_usage_incomplete() {
    for (rows, status, offset, failure) in [
        (
            vec![
                terminal(),
                usage(),
                json!({"choices":[],"error":{"message":"private payload"}}),
            ],
            200,
            0,
            InferenceAttemptFailure::Protocol,
        ),
        (
            vec![terminal(), usage()],
            200,
            1,
            InferenceAttemptFailure::Stream,
        ),
        (vec![], 429, 0, InferenceAttemptFailure::Http),
    ] {
        let (callback, log) = observer();
        let (provider, server) = wire_provider(rows, status, false, offset).await;
        let events = collect(&provider.with_observer(callback)).await;
        assert!(events.iter().any(Result::is_err));
        server.await.unwrap();
        let receipt = receipt(&log);
        assert_eq!(receipt.outcome, InferenceAttemptOutcome::Failed(failure));
        assert!(!receipt.transport_complete);
        assert_eq!(receipt.reported_usage, (status == 200).then(expected_usage));
    }
}

#[tokio::test]
async fn missing_or_invalid_observation_usage_preserves_legacy_task_events() {
    for observed_usage in [
        None,
        Some(json!({"prompt_tokens":3})),
        Some(
            json!({"prompt_tokens":3,"completion_tokens":1,"prompt_tokens_details":{"cached_tokens":"bad"}}),
        ),
        Some(
            json!({"prompt_tokens":3,"completion_tokens":1,"completion_tokens_details":{"reasoning_tokens":2}}),
        ),
        Some(json!({"prompt_tokens":3,"completion_tokens":1,"total_tokens":5})),
    ] {
        let mut rows = vec![terminal()];
        let has_usage = observed_usage.is_some();
        if let Some(usage) = observed_usage {
            rows.push(json!({"choices":[],"usage":usage}));
        }
        let (callback, log) = observer();
        let (provider, server) = wire_provider(rows, 200, false, 0).await;
        let events = collect(&provider.with_observer(callback)).await;
        assert!(events.iter().all(Result::is_ok));
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, Ok(Event::MessageStop { .. })))
                .count(),
            1
        );
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, Ok(Event::UsageDelta { .. })))
                .count(),
            usize::from(has_usage)
        );
        server.await.unwrap();
        let receipt = receipt(&log);
        assert_eq!(receipt.outcome, InferenceAttemptOutcome::Completed);
        assert!(receipt.transport_complete);
        assert_eq!(receipt.reported_usage, None);
    }
}

#[tokio::test]
async fn clean_eof_without_terminal_is_only_an_observation_failure() {
    let (callback, log) = observer();
    let (provider, server) = wire_provider(vec![usage()], 200, false, 0).await;
    let events = collect(&provider.with_observer(callback)).await;
    assert!(events.iter().all(Result::is_ok));
    assert!(!events
        .iter()
        .any(|event| matches!(event, Ok(Event::MessageStop { .. }))));
    server.await.unwrap();
    let receipt = receipt(&log);
    assert_eq!(
        receipt.outcome,
        InferenceAttemptOutcome::Failed(InferenceAttemptFailure::Protocol)
    );
    assert!(receipt.transport_complete);
    assert_eq!(receipt.reported_usage, Some(expected_usage()));
}

#[tokio::test]
async fn dropped_and_timed_out_bodies_record_actual_connection_cancellation() {
    for timed_out in [false, true] {
        let rows = vec![
            json!({"choices":[{"index":0,"delta":{"content":"private payload"},"finish_reason":null}]}),
            terminal(),
            usage(),
        ];
        let (callback, log) = observer();
        let (provider, server) = wire_provider(rows, 200, true, 0).await;
        let provider = provider.with_observer(callback);
        let mut stream = provider.stream_message(request());
        assert!(matches!(
            tokio::time::timeout(Duration::from_secs(3), stream.next())
                .await
                .unwrap(),
            Some(Ok(Event::TextDelta { .. }))
        ));
        if timed_out {
            assert!(
                tokio::time::timeout(Duration::from_millis(50), stream.collect::<Vec<_>>())
                    .await
                    .is_err()
            );
        } else {
            drop(stream);
        }
        server.await.unwrap();
        let receipt = receipt(&log);
        assert_eq!(receipt.outcome, InferenceAttemptOutcome::Dropped);
        assert!(!receipt.transport_complete);
        assert_eq!(receipt.reported_usage, timed_out.then(expected_usage));
    }
}

#[tokio::test]
async fn unpolled_and_replay_rejected_compat_requests_emit_no_send() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let (callback, log) = observer();
    let provider = OpenAiCompatProvider::new(
        "fixture",
        format!("http://{}", listener.local_addr().unwrap()),
        Auth::None,
        vec![],
    )
    .with_observer(callback);
    drop(provider.stream_message(request()));
    let mut rejected = request();
    rejected.messages.push(crate::RequestMessage {
        role: "assistant".into(),
        content: vec![RequestContent::Replay {
            envelope: crate::ReplayEnvelope {
                version: 1,
                route: crate::ReplayRoute::ChatgptResponses,
                registration: "fixture-registration".into(),
                model: "fixture-model".into(),
                output: vec![],
            },
        }],
    });
    let events: Vec<_> = provider.stream_message(rejected).collect().await;
    assert_eq!(events.len(), 1);
    assert!(events[0].is_err());
    assert!(log.lock().unwrap().is_empty());
    assert!(
        tokio::time::timeout(Duration::from_millis(50), listener.accept())
            .await
            .is_err()
    );
}

#[tokio::test]
async fn compatible_failed_send_reports_unknown_usage() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    drop(listener);
    let (callback, log) = observer();
    let provider =
        OpenAiCompatProvider::new("fixture", url, Auth::None, vec![]).with_observer(callback);
    let events = collect(&provider).await;
    assert_eq!(events.len(), 1);
    assert!(matches!(&events[0], Err(ApiError::Request(_))));
    let receipt = receipt(&log);
    assert_eq!(
        receipt.outcome,
        InferenceAttemptOutcome::Failed(InferenceAttemptFailure::Send)
    );
    assert_eq!(receipt.reported_usage, None);
    assert!(!receipt.transport_complete);
}
