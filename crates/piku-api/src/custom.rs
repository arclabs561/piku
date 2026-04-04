/// Custom OpenAI-compatible server.
///
/// Configured entirely via environment variables:
///   `PIKU_BASE_URL` base URL of the server (required)
///   `PIKU_API_KEY` bearer token (optional — omit for unauthenticated servers)
///   `PIKU_MODEL` default model name (optional)
use crate::error::ApiError;
use crate::openai_compat::{Auth, OpenAiCompatProvider};

pub const ENV_BASE_URL: &str = "PIKU_BASE_URL";
pub const ENV_API_KEY: &str = "PIKU_API_KEY";
pub const ENV_MODEL: &str = "PIKU_MODEL";

pub fn from_env() -> Result<(OpenAiCompatProvider, Option<String>), ApiError> {
    let base_url = std::env::var(ENV_BASE_URL)
        .map_err(|_| ApiError::Provider(format!("missing {ENV_BASE_URL}")))?;

    let auth = match std::env::var(ENV_API_KEY) {
        Ok(key) if !key.is_empty() => Auth::Bearer(key),
        _ => Auth::None,
    };

    let model = std::env::var(ENV_MODEL).ok();

    let provider = OpenAiCompatProvider::new("custom", base_url, auth, vec![]);
    Ok((provider, model))
}

#[must_use]
pub fn is_configured() -> bool {
    std::env::var(ENV_BASE_URL).is_ok()
}
