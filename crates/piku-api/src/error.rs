use thiserror::Error;

#[derive(Debug, Error)]
pub enum ApiError {
    #[error("missing API key — set ANTHROPIC_API_KEY or configure ~/.config/piku/config.toml")]
    MissingApiKey,

    #[error("HTTP error {status}: {body}")]
    Http { status: u16, body: String },

    #[error("SSE parse error: {0}")]
    SseParse(String),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("request error: {0}")]
    Request(#[from] reqwest::Error),

    #[error("stream ended unexpectedly")]
    UnexpectedStreamEnd,

    #[error("provider error: {0}")]
    Provider(String),
}
