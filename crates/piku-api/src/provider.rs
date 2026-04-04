use std::pin::Pin;

use futures_util::Stream;

use crate::error::ApiError;
use crate::types::{Event, MessageRequest};

/// A provider that can stream messages from an LLM API.
pub trait Provider: Send + Sync {
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
