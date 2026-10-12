use super::*;
use piku_api::{ApiError, Event, MessageRequest, Provider, StopReason, TokenUsage};
use std::pin::Pin;

#[derive(Clone, Copy, Debug)]
enum Ending {
    Complete,
    StopSequence,
    Error,
    MaxTokens,
    Eof,
    LateError,
    DuplicateStop,
    ConflictingStop,
}

struct Scripted {
    extraction: Ending,
    judge: Ending,
}

impl Provider for Scripted {
    fn name(&self) -> &'static str {
        "memory-stream-fixture"
    }
    fn stream_message(
        &self,
        request: MessageRequest,
    ) -> Pin<Box<dyn futures_util::Stream<Item = Result<Event, ApiError>> + Send + '_>> {
        let judge = request.max_tokens == 32;
        let ending = if judge { self.judge } else { self.extraction };
        let mut events = vec![Ok(Event::TextDelta {
            text: if judge {
                "UPDATE 0"
            } else {
                "- [importance:7] corrected fact {tags: fixture}"
            }
            .into(),
        })];
        match ending {
            Ending::Complete
            | Ending::LateError
            | Ending::DuplicateStop
            | Ending::ConflictingStop => events.push(Ok(Event::MessageStop {
                stop_reason: StopReason::EndTurn,
            })),
            Ending::StopSequence => events.push(Ok(Event::MessageStop {
                stop_reason: StopReason::StopSequence,
            })),
            Ending::MaxTokens => events.push(Ok(Event::MessageStop {
                stop_reason: StopReason::MaxTokens,
            })),
            Ending::Error | Ending::Eof => {}
        }
        if matches!(ending, Ending::DuplicateStop | Ending::ConflictingStop) {
            events.push(Ok(Event::MessageStop {
                stop_reason: if matches!(ending, Ending::DuplicateStop) {
                    StopReason::EndTurn
                } else {
                    StopReason::MaxTokens
                },
            }));
        }
        // Real adapters can emit usage after their completion event.
        events.push(Ok(Event::UsageDelta {
            usage: TokenUsage::default(),
        }));
        if matches!(ending, Ending::Error | Ending::LateError) {
            events.push(Err(ApiError::UnexpectedStreamEnd));
        }
        Box::pin(futures_util::stream::iter(events))
    }
}

async fn memory_embedding_server() -> (String, tokio::task::JoinHandle<()>) {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let server = tokio::spawn(async move {
        loop {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut request = Vec::new();
            let mut buf = [0; 1024];
            loop {
                let n = socket.read(&mut buf).await.unwrap();
                assert!(n > 0);
                request.extend_from_slice(&buf[..n]);
                if let Some(end) = request.windows(4).position(|s| s == b"\r\n\r\n") {
                    let header = String::from_utf8_lossy(&request[..end]);
                    let length: usize = header
                        .lines()
                        .find_map(|l| {
                            l.to_ascii_lowercase()
                                .strip_prefix("content-length: ")
                                .map(|v| v.parse().unwrap())
                        })
                        .unwrap();
                    if request.len() >= end + 4 + length {
                        break;
                    }
                }
            }
            let body = "{\"embeddings\":[[0.8,0.6]]}";
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
            socket.write_all(response.as_bytes()).await.unwrap();
        }
    });
    (url, server)
}

fn seed(path: &Path) {
    MemoryStore::transaction(path, |store| {
        assert_eq!(
            store.insert("original fact".into(), vec![], vec![1.0, 0.0], 7),
            Some(0)
        );
        Ok(())
    })
    .unwrap();
}

#[tokio::test]
async fn incomplete_extraction_cannot_persist_facts() {
    let (base_url, server) = memory_embedding_server().await;
    let config = EmbedConfig {
        backend: EmbedBackend::Ollama,
        base_url,
        model: "fixture".into(),
        api_key: None,
    };
    let mut failures = Vec::new();
    for ending in [
        Ending::Error,
        Ending::MaxTokens,
        Ending::Eof,
        Ending::LateError,
        Ending::DuplicateStop,
        Ending::ConflictingStop,
    ] {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("memory.json");
        seed(&path);
        let before = std::fs::read(&path).unwrap();
        let provider = Scripted {
            extraction: ending,
            judge: Ending::Complete,
        };
        let count = extract_and_persist("conversation", &provider, "fixture", &path, &config)
            .await
            .unwrap();
        if count != 0 || std::fs::read(&path).unwrap() != before {
            failures.push(format!("{ending:?}"));
        }
    }
    server.abort();
    assert!(
        failures.is_empty(),
        "incomplete extraction persisted facts: {failures:?}"
    );
}

#[tokio::test]
async fn incomplete_judge_keeps_original_memory_and_add_fallback() {
    let (base_url, server) = memory_embedding_server().await;
    let config = EmbedConfig {
        backend: EmbedBackend::Ollama,
        base_url,
        model: "fixture".into(),
        api_key: None,
    };
    let mut failures = Vec::new();
    for ending in [
        Ending::Error,
        Ending::MaxTokens,
        Ending::Eof,
        Ending::LateError,
        Ending::DuplicateStop,
        Ending::ConflictingStop,
    ] {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("memory.json");
        seed(&path);
        let provider = Scripted {
            extraction: Ending::Complete,
            judge: ending,
        };
        assert_eq!(
            extract_and_persist("conversation", &provider, "fixture", &path, &config)
                .await
                .unwrap(),
            1
        );
        let store = MemoryStore::try_load(&path).unwrap();
        if store.entries.len() != 2 || !store.entries.iter().all(|entry| entry.is_valid) {
            failures.push(format!("{ending:?}"));
        }
    }
    server.abort();
    assert!(
        failures.is_empty(),
        "incomplete judge superseded original: {failures:?}"
    );
}

#[tokio::test]
async fn completed_extraction_and_judge_can_update_memory() {
    let (base_url, server) = memory_embedding_server().await;
    let config = EmbedConfig {
        backend: EmbedBackend::Ollama,
        base_url,
        model: "fixture".into(),
        api_key: None,
    };
    for ending in [Ending::Complete, Ending::StopSequence] {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("memory.json");
        seed(&path);
        let provider = Scripted {
            extraction: ending,
            judge: ending,
        };
        assert_eq!(
            extract_and_persist("conversation", &provider, "fixture", &path, &config)
                .await
                .unwrap(),
            1
        );
        let store = MemoryStore::try_load(&path).unwrap();
        assert_eq!(store.entries.len(), 2);
        assert!(!store.entries[0].is_valid);
        assert!(store.entries[1].is_valid);
        assert_eq!(store.entries[1].content, "corrected fact");
    }
    server.abort();
}
