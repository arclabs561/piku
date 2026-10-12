use std::pin::Pin;

use futures_util::Stream;

use crate::error::ApiError;
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
