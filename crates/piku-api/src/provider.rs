use std::pin::Pin;
use std::sync::Arc;

use futures_util::Stream;

use crate::error::ApiError;
use crate::observation::InferenceObserver;
use crate::types::{Event, MessageRequest};

/// A provider that can stream messages from an LLM API.
pub trait Provider: Send + Sync {
    /// Stream one response. Successful responses emit exactly one
    /// [`Event::MessageStop`]; usage may follow that event. Missing or repeated
    /// terminal events do not establish valid completion.
    fn stream_message(
        &self,
        request: MessageRequest,
    ) -> Pin<Box<dyn Stream<Item = Result<Event, ApiError>> + Send + '_>>;

    /// Human-readable name for this provider (e.g. "anthropic", "openai").
    fn name(&self) -> &str;

    /// Clone into a boxed provider suitable for background tasks (subagent spawning).
    /// The default implementation panics — only real providers need to implement this.
    fn boxed_clone(&self) -> Box<dyn Provider + Send + Sync + 'static> {
        unimplemented!("boxed_clone not implemented for this provider")
    }

    /// Clone the existing route with an opt-in inference observer. Unsupported
    /// providers refuse before inference; this never selects a fallback route.
    fn observed_clone(
        &self,
        _observer: Arc<InferenceObserver>,
    ) -> Result<Box<dyn Provider>, ApiError> {
        Err(ApiError::Provider(
            "request observation is unsupported by selected provider".into(),
        ))
    }
}

/// A short-lived bearer value. Never serialize or include it in diagnostics.
pub struct AccessToken(String);

impl AccessToken {
    #[must_use]
    pub fn new(secret: String) -> Self {
        Self(secret)
    }

    #[must_use]
    pub fn expose_secret(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Debug for AccessToken {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("AccessToken([REDACTED])")
    }
}

/// Credential lifecycle owner bound to one immutable account registration.
/// Implementations coordinate refresh; transport never discovers accounts.
pub trait AccessTokenSource: Send + Sync {
    fn token(
        &self,
    ) -> Pin<Box<dyn std::future::Future<Output = Result<AccessToken, ApiError>> + Send + '_>>;
    fn binding(&self) -> &str;
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Unsupported;
    impl Provider for Unsupported {
        fn name(&self) -> &'static str {
            "unsupported"
        }
        fn stream_message(
            &self,
            _: MessageRequest,
        ) -> Pin<Box<dyn Stream<Item = Result<Event, ApiError>> + Send + '_>> {
            panic!("unsupported observation must refuse before inference")
        }
    }

    #[test]
    fn observation_defaults_to_refusal_without_cloning_or_inference() {
        let provider: &dyn Provider = &Unsupported;
        assert!(
            matches!(provider.observed_clone(Arc::new(|_| panic!("unexpected receipt"))), Err(ApiError::Provider(message)) if message.contains("unsupported"))
        );
    }
}
