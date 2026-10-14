#![allow(clippy::too_many_lines)]

use super::*;
use crate::{
    run_turn_with_registry, AllowAll, RecordingSink, RunEvent, RunHandle, RunRecorder, Session,
    TaskRegistry,
};
use piku_api::{InferenceObservation, OpenAiCompatProvider};
use serde_json::{json, Value};
use std::sync::Mutex;
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

type Receipts = Arc<Mutex<Vec<(CausalBinding, InferenceObservation)>>>;

fn observation(receipts: &Receipts) -> RequestObservation {
    let receipts = Arc::clone(receipts);
    RequestObservation {
        factory: Arc::new(move |_, binding| {
            let receipts = Arc::clone(&receipts);
            Arc::new(move |event| receipts.lock().unwrap().push((binding.clone(), event)))
        }),
        binding: CausalBinding::default(),
    }
}

async fn read_request(socket: &mut tokio::net::TcpStream) -> Value {
    let mut bytes = Vec::new();
    let (start, length) = loop {
        let mut buffer = [0; 4096];
        let count = socket.read(&mut buffer).await.unwrap();
        assert!(count > 0);
        bytes.extend_from_slice(&buffer[..count]);
        assert!(bytes.len() < 1_000_000);
        if let Some(end) = bytes.windows(4).position(|window| window == b"\r\n\r\n") {
            let headers = String::from_utf8_lossy(&bytes[..end]).to_ascii_lowercase();
            let length = headers
                .lines()
                .find_map(|line| line.strip_prefix("content-length:"))
                .unwrap()
                .trim()
                .parse::<usize>()
                .unwrap();
            break (end + 4, length);
        }
    };
    while bytes.len() < start + length {
        let mut buffer = [0; 4096];
        let count = socket.read(&mut buffer).await.unwrap();
        assert!(count > 0);
        bytes.extend_from_slice(&buffer[..count]);
    }
    serde_json::from_slice(&bytes[start..start + length]).unwrap()
}

async fn reply(socket: &mut tokio::net::TcpStream, chunk: Value) {
    let body = format!("data: {chunk}\n\ndata: [DONE]\n\n");
    let response = format!("HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}", body.len());
    socket.write_all(response.as_bytes()).await.unwrap();
}

#[tokio::test]
async fn actual_recorded_child_finishes_after_later_parent_turn_with_immutable_binding() {
    tokio::time::timeout(Duration::from_secs(10), tokio::task::LocalSet::new().run_until(async {
        let directory = tempfile::tempdir().unwrap();
        let run_path = directory.path().join("runs/session-100-1.jsonl");
        let registry = TaskRegistry::with_persistence_run_path("session-100-1", directory.path().join("sessions"), directory.path().join("runs"), directory.path().join("links"), Some(run_path.clone()));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}/v1", listener.local_addr().unwrap());
        let child_seen = Arc::new(tokio::sync::Notify::new());
        let release_child = Arc::new(tokio::sync::Notify::new());
        let seen = Arc::clone(&child_seen);
        let release = Arc::clone(&release_child);
        let server = tokio::spawn(async move {
            let mut handlers = tokio::task::JoinSet::new();
            let parent_calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
            for _ in 0..4 {
                let (mut socket, _) = listener.accept().await.unwrap();
                let seen = Arc::clone(&seen);
                let release = Arc::clone(&release);
                let parent_calls = Arc::clone(&parent_calls);
                handlers.spawn(async move {
                    let request = read_request(&mut socket).await;
                    let first_user = request["messages"].as_array().unwrap().iter().find(|message| message["role"] == "user").unwrap();
                    let content = first_user["content"].to_string();
                    if content.contains("child-fixture") {
                        seen.notify_one();
                        release.notified().await;
                        reply(&mut socket, json!({"choices":[{"index":0,"delta":{"content":"child finished"},"finish_reason":"stop"}]})).await;
                    } else if parent_calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst) == 0 {
                        let arguments = json!({"name":"child","task":"child-fixture","background":true,"max_turns":1}).to_string();
                        reply(&mut socket, json!({"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"spawn","type":"function","function":{"name":"spawn_agent","arguments":arguments}}]},"finish_reason":"tool_calls"}]})).await;
                    } else {
                        if parent_calls.load(std::sync::atomic::Ordering::SeqCst) == 2 { seen.notified().await; }
                        reply(&mut socket, json!({"choices":[{"index":0,"delta":{"content":"parent finished"},"finish_reason":"stop"}]})).await;
                    }
                });
            }
            while let Some(result) = handlers.join_next().await { result.unwrap(); }
        });
        let provider = OpenAiCompatProvider::new("custom", url, piku_api::openai_compat::Auth::None, vec![]);
        let receipts: Receipts = Arc::new(Mutex::new(Vec::new()));
        let mut sink = ObservationSink { inner: crate::task::DevNullSink, observation: Some(observation(&receipts)) };
        let mut run = RunHandle::open(Session::new("session-100-1".into()), &run_path).unwrap();
        for number in 1..=2 {
            let mut turn = run.begin_turn(&mut sink, format!("turn-{number:032x}"));
            let (session, sink) = turn.parts();
            // Nonroot depth skips ambient memory recall; all test acquisitions
            // are confined to the explicitly supplied localhost provider.
            let result = run_turn_with_registry("parent-fixture", session, &provider, "fixture", &[], piku_tools::all_tool_definitions(), &AllowAll, sink, Some(2), None, &registry, 1, &[], None, None).await;
            assert!(result.stream_error.is_none());
            turn.finish().unwrap();
        }
        let child = registry.running().pop().unwrap();
        // Parent's later turn finished while the actual child's POST is open.
        {
        let before = receipts.lock().unwrap();
        assert_eq!(before.iter().filter(|(_, event)| matches!(event, InferenceObservation::Started { .. })).count(), 4);
        assert_eq!(before.iter().filter(|(_, event)| matches!(event, InferenceObservation::Finished(_))).count(), 3);
        }
        release_child.notify_one();
        assert_eq!(registry.wait_for(&child.id).await.unwrap().status, crate::TaskStatus::Done);
        server.await.unwrap();
        let child_evidence = registry.evidence(&child.id).unwrap();
        let parent_events = crate::read_run_record(&run_path).unwrap();
        let child_events = crate::read_run_record(&child_evidence.run_record_path).unwrap();
        let receipts = receipts.lock().unwrap();
        for (binding, event) in receipts.iter() {
            assert_eq!(binding.purpose, RequestPurpose::AgentTurn);
            let events = if binding.task_id.is_some() {
                assert_eq!(binding.task_id.as_deref(), Some(child.id.0.as_str()));
                assert_eq!(binding.run_id.as_deref(), Some(child_evidence.child_session_id.as_str()));
                assert_eq!(binding.parent_origin, Some(CausalOrigin { run_id: Some("session-100-1".into()), turn_id: Some(format!("turn-{:032x}",1)) }));
                &child_events
            } else { &parent_events };
            let context = events.iter().find(|event| Some(event.sequence) == binding.context_sequence).unwrap();
            assert!(matches!(context.event, RunEvent::ContextBuilt { .. }));
            assert_eq!(context.scope.turn_id(), binding.turn_id.as_deref());
            let id = match event { InferenceObservation::Started { attempt_id, .. } => *attempt_id, InferenceObservation::Finished(receipt) => receipt.attempt_id, _ => panic!("unknown observation") };
            assert!(receipts.iter().filter(|(_, event)| match event { InferenceObservation::Started { attempt_id, .. } => *attempt_id == id, InferenceObservation::Finished(receipt) => receipt.attempt_id == id, _ => false }).all(|(other, _)| other == binding));
        }
    })).await.unwrap();
}

#[tokio::test]
async fn failed_context_append_does_not_reuse_previous_request_sequence() {
    use crate::OutputSink;
    struct FaultSink {
        observation: RequestObservation,
        path: std::path::PathBuf,
        contexts: usize,
    }
    impl OutputSink for FaultSink {
        fn request_observation(&self) -> Option<RequestObservation> {
            Some(self.observation.clone())
        }
        fn on_text(&mut self, _: &str) {}
        fn on_tool_start(&mut self, _: &str, _: &str, _: &Value) {}
        fn on_tool_end(&mut self, _: &str, _: &str, _: &str, _: bool) -> crate::PostToolAction {
            crate::PostToolAction::Continue
        }
        fn on_permission_denied(&mut self, _: &str, _: &str) {}
        fn on_turn_complete(&mut self, _: &piku_api::TokenUsage, _: u32) {}
        fn on_run_event(&mut self, event: &RunEvent) {
            if matches!(event, RunEvent::ContextBuilt { .. }) {
                self.contexts += 1;
                if self.contexts == 2 {
                    std::fs::copy(&self.path, self.path.with_extension("retained.jsonl")).unwrap();
                    std::fs::write(&self.path, b"").unwrap();
                }
            }
        }
    }
    tokio::time::timeout(Duration::from_secs(5), async {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("run.jsonl");
        let receipts: Receipts = Arc::new(Mutex::new(Vec::new()));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let provider = OpenAiCompatProvider::new("custom", format!("http://{}/v1", listener.local_addr().unwrap()), piku_api::openai_compat::Auth::None, vec![]);
        let read_path = directory.path().join("fixture.txt");
        std::fs::write(&read_path, b"owned fixture").unwrap();
        let server = tokio::spawn(async move {
            for number in 0..2 {
                let (mut socket, _) = listener.accept().await.unwrap();
                read_request(&mut socket).await;
                if number == 0 {
                    let arguments = json!({"path":read_path}).to_string();
                    reply(&mut socket, json!({"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"read","type":"function","function":{"name":"read_file","arguments":arguments}}]},"finish_reason":"tool_calls"}]})).await;
                } else { reply(&mut socket, json!({"choices":[{"index":0,"delta":{"content":"done"},"finish_reason":"stop"}]})).await; }
            }
        });
        let mut recorder = RunRecorder::open(&path, "session-100-1").unwrap();
        let mut inner = FaultSink { observation: observation(&receipts), path: path.clone(), contexts: 0 };
        let mut sink = RecordingSink::new(&mut inner, &mut recorder, "turn-00000000000000000000000000000001");
        let mut session = Session::new("session-100-1".into());
        crate::run_turn("fixture", &mut session, &provider, "fixture", &[], piku_tools::read_only_tool_definitions(), &AllowAll, &mut sink, Some(2), None).await;
        server.await.unwrap();
        assert!(sink.take_record_error().is_some());
        let receipts = receipts.lock().unwrap();
        let starts: Vec<_> = receipts.iter().filter(|(_, event)| matches!(event, InferenceObservation::Started { .. })).collect();
        assert_eq!(starts.len(), 2);
        assert!(starts[0].0.context_sequence.is_some());
        assert_eq!(starts[1].0.context_sequence, None);
        assert!(receipts.iter().skip(2).all(|(binding, _)| binding.context_sequence.is_none()));
        crate::read_run_record(path.with_extension("retained.jsonl")).unwrap();
    }).await.unwrap();
}

#[tokio::test]
async fn actual_periodic_memory_pipeline_retains_trigger_without_foreground_estimate() {
    tokio::time::timeout(Duration::from_secs(5), async {
        let directory = tempfile::tempdir().unwrap();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}/v1", listener.local_addr().unwrap());
        let received = Arc::new(tokio::sync::Notify::new());
        let release = Arc::new(tokio::sync::Notify::new());
        let server_received = Arc::clone(&received);
        let server_release = Arc::clone(&release);
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            read_request(&mut socket).await;
            server_received.notify_one();
            server_release.notified().await;
            reply(&mut socket, json!({"choices":[{"index":0,"delta":{"content":"NONE"},"finish_reason":"stop"}]})).await;
        });
        let receipts: Receipts = Arc::new(Mutex::new(Vec::new()));
        let mut original = observation(&receipts);
        original.binding.run_id = Some("session-100-1".into());
        original.binding.turn_id = Some(format!("turn-{:032x}", 1));
        original.binding.context_sequence = Some(41);
        let periodic = original.periodic_memory();
        let provider = OpenAiCompatProvider::new("custom", url.clone(), piku_api::openai_compat::Auth::None, vec![]);
        let observed = periodic.observed_clone(&provider).unwrap();
        let config = crate::EmbedConfig { backend: crate::EmbedBackend::Ollama, base_url: url, model: "fixture".into(), api_key: None };
        let path = directory.path().join("memories.json");
        let task = tokio::spawn(async move { crate::embed_memory::extract_and_persist("fixture", observed.as_ref(), "fixture", &path, &config).await });
        received.notified().await;
        original.binding.turn_id = Some(format!("turn-{:032x}", 2));
        release.notify_one();
        assert_eq!(task.await.unwrap().unwrap(), 0);
        server.await.unwrap();
        let receipts = receipts.lock().unwrap();
        assert_eq!(receipts.len(), 2);
        assert!(receipts.iter().all(|(binding, _)| binding.purpose == RequestPurpose::PeriodicMemory && binding.turn_id == Some(format!("turn-{:032x}",1)) && binding.context_sequence.is_none()));
        assert!(matches!(receipts[1].1, InferenceObservation::Finished(ref receipt) if receipt.transport_complete));
    }).await.unwrap();
}
