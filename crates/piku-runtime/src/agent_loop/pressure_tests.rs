use super::*;
use std::sync::atomic::{AtomicUsize, Ordering};

struct RepeatedProvider {
    requests: AtomicUsize,
    usage: TokenUsage,
}
impl Provider for RepeatedProvider {
    fn name(&self) -> &'static str {
        "scripted-pressure"
    }
    fn stream_message(
        &self,
        request: MessageRequest,
    ) -> std::pin::Pin<
        Box<dyn futures_util::Stream<Item = Result<Event, piku_api::ApiError>> + Send + '_>,
    > {
        assert!(!request.messages.is_empty());
        let index = self.requests.fetch_add(1, Ordering::Relaxed);
        Box::pin(futures_util::stream::iter([
            Ok(Event::TextDelta {
                text: "observed".into(),
            }),
            Ok(Event::UsageDelta {
                usage: self.usage.clone(),
            }),
            Ok(Event::MessageStop {
                stop_reason: if index.is_multiple_of(2) {
                    StopReason::ToolUse
                } else {
                    StopReason::EndTurn
                },
            }),
        ]))
    }
}
#[derive(Default)]
struct PressureSink {
    manifests: Vec<crate::run_record::ContextManifest>,
    pressure: Vec<f32>,
}
impl OutputSink for PressureSink {
    fn on_text(&mut self, _: &str) {}
    fn on_tool_start(&mut self, _: &str, _: &str, _: &serde_json::Value) {}
    fn on_tool_end(&mut self, _: &str, _: &str, _: &str, _: bool) -> PostToolAction {
        PostToolAction::Continue
    }
    fn on_permission_denied(&mut self, _: &str, _: &str) {}
    fn on_turn_complete(&mut self, _: &TokenUsage, _: u32) {}
    fn on_context_pressure(&mut self, pressure: f32) {
        self.pressure.push(pressure);
    }
    fn on_run_event(&mut self, event: &crate::run_record::RunEvent) {
        if let crate::run_record::RunEvent::ContextBuilt { manifest } = event {
            self.manifests.push(manifest.clone());
        }
    }
}

#[tokio::test]
#[allow(clippy::cast_precision_loss)]
async fn pressure_tracks_latest_request_not_repeated_or_cached_usage() {
    // Same selected context under uncached, cache-read, cache-create and absent
    // usage reports. Cache layouts follow TokenUsage fields retained by the
    // Anthropic parser; OAI-compatible prompt usage occupies input_tokens.
    let usages = [
        TokenUsage {
            input_tokens: 150_000,
            ..TokenUsage::default()
        },
        TokenUsage {
            input_tokens: 10,
            cache_read_input_tokens: 149_990,
            ..TokenUsage::default()
        },
        TokenUsage {
            input_tokens: 10,
            cache_creation_input_tokens: 149_990,
            ..TokenUsage::default()
        },
        TokenUsage::default(),
    ];
    let mut previous_pressure = None;
    for usage in usages {
        let provider = RepeatedProvider {
            requests: AtomicUsize::new(0),
            usage: usage.clone(),
        };
        let mut session = Session::new("pressure".into());
        let mut sink = PressureSink::default();
        for turn in 0..2 {
            let result = run_turn_inner(
                "inspect context",
                &mut session,
                &provider,
                "scripted",
                &["stable system".into()],
                vec![],
                &crate::permission::AllowAll,
                &mut sink,
                Some(2),
                None,
                None,
                0,
                &[],
                None,
                None,
            )
            .await;
            assert_eq!(result.result.iterations, 2);
            assert_eq!(result.result.usage.input_tokens, usage.input_tokens * 2);
            assert_eq!(
                result.result.usage.cache_read_input_tokens,
                usage.cache_read_input_tokens * 2
            );
            assert_eq!(sink.manifests.len(), (turn + 1) * 2);
            assert_eq!(sink.pressure.len(), turn + 1);
            let latest = sink.manifests.last().unwrap();
            assert!(latest.estimated_input_tokens > 0);
            assert!(latest.estimated_input_tokens < 100);
            let expected =
                latest.estimated_input_tokens as f32 / latest.context_window_tokens as f32;
            assert!(
                (sink.pressure[turn] - expected).abs() < 1e-7,
                "reported {} but selected request occupies {expected}",
                sink.pressure[turn]
            );
        }
        assert_eq!(provider.requests.load(Ordering::Relaxed), 4);
        if let Some(previous) = previous_pressure {
            assert_eq!(sink.pressure, previous);
        }
        previous_pressure = Some(sink.pressure);
    }
}
