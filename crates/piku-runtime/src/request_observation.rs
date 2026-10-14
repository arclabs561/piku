//! Immutable causal ownership for explicitly requested inference observations.

use std::sync::Arc;

use piku_api::{ApiError, InferenceObserver, Provider};
use serde::{Deserialize, Serialize};

#[cfg(test)]
mod tests;

/// The runtime operation that owns a request, independent of transport outcome.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum RequestPurpose {
    #[default]
    AgentTurn,
    PeriodicMemory,
    ExitMemory,
}

/// The immediate operation that spawned a child; never the child's own record.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct CausalOrigin {
    pub run_id: Option<String>,
    pub turn_id: Option<String>,
}

/// Opaque semantic references only. Missing references are unknown, not inferred
/// from timing, payloads, or whichever foreground turn is currently active.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct CausalBinding {
    pub run_id: Option<String>,
    pub turn_id: Option<String>,
    /// Sequence of a successfully appended `ContextBuilt` in this run/turn.
    pub context_sequence: Option<u64>,
    pub task_id: Option<String>,
    pub parent_origin: Option<CausalOrigin>,
    pub purpose: RequestPurpose,
}

/// The caller owns logging policy and queue; the runtime supplies causal refs.
pub type RequestObserverFactory =
    dyn Fn(&str, CausalBinding) -> Arc<InferenceObserver> + Send + Sync;

/// An owned factory plus snapshot that can cross a child/background task boundary.
#[derive(Clone)]
#[non_exhaustive]
pub struct RequestObservation {
    pub factory: Arc<RequestObserverFactory>,
    pub binding: CausalBinding,
}

impl RequestObservation {
    /// Start with unknown causal references; owners may fill acknowledged refs.
    #[must_use]
    pub fn new(factory: Arc<RequestObserverFactory>) -> Self {
        Self {
            factory,
            binding: CausalBinding::default(),
        }
    }

    /// Replace only the callback on a clone of the selected route. The observer
    /// owns its binding before the lazy transport is polled.
    pub fn observed_clone(&self, provider: &dyn Provider) -> Result<Box<dyn Provider>, ApiError> {
        provider.observed_clone((self.factory)(provider.name(), self.binding.clone()))
    }

    /// Extraction retains its triggering owner, without a foreground estimate.
    #[must_use]
    pub fn periodic_memory(&self) -> Self {
        let mut observation = self.clone();
        observation.binding.purpose = RequestPurpose::PeriodicMemory;
        observation.binding.context_sequence = None;
        observation
    }

    pub(crate) fn child(&self, run_id: &str, task_id: &str) -> Self {
        let parent_origin =
            (self.binding.run_id.is_some() || self.binding.turn_id.is_some()).then(|| {
                CausalOrigin {
                    run_id: self.binding.run_id.clone(),
                    turn_id: self.binding.turn_id.clone(),
                }
            });
        Self {
            factory: Arc::clone(&self.factory),
            binding: CausalBinding {
                run_id: Some(run_id.to_owned()),
                task_id: Some(task_id.to_owned()),
                parent_origin,
                ..CausalBinding::default()
            },
        }
    }
}

/// Private child composition preserves the public `DevNullSink` constructor.
pub(crate) struct ObservationSink<T> {
    pub(crate) inner: T,
    pub(crate) observation: Option<RequestObservation>,
}

impl<T: crate::OutputSink> crate::OutputSink for ObservationSink<T> {
    fn request_observation(&self) -> Option<RequestObservation> {
        self.observation.clone()
    }
    fn on_text(&mut self, text: &str) {
        self.inner.on_text(text);
    }
    fn on_tool_start(&mut self, name: &str, id: &str, input: &serde_json::Value) {
        self.inner.on_tool_start(name, id, input);
    }
    fn on_tool_end(
        &mut self,
        name: &str,
        id: &str,
        result: &str,
        is_error: bool,
    ) -> crate::PostToolAction {
        self.inner.on_tool_end(name, id, result, is_error)
    }
    fn on_permission_denied(&mut self, name: &str, reason: &str) {
        self.inner.on_permission_denied(name, reason);
    }
    fn on_turn_complete(&mut self, usage: &piku_api::TokenUsage, iterations: u32) {
        self.inner.on_turn_complete(usage, iterations);
    }
    fn on_interjection(&mut self, text: &str) {
        self.inner.on_interjection(text);
    }
    fn on_context_pressure(&mut self, pressure: f32) {
        self.inner.on_context_pressure(pressure);
    }
    fn on_provider_stream(&mut self, elapsed: u64, blocks: usize, stop: &str) {
        self.inner.on_provider_stream(elapsed, blocks, stop);
    }
    fn on_run_event(&mut self, event: &crate::RunEvent) {
        self.inner.on_run_event(event);
    }
}
