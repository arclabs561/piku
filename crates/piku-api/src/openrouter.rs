use crate::error::ApiError;
use crate::openai_compat::{Auth, Header, OpenAiCompatProvider};

const BASE_URL: &str = "https://openrouter.ai/api/v1";

#[must_use]
pub fn new(api_key: String) -> OpenAiCompatProvider {
    OpenAiCompatProvider::new(
        "openrouter",
        BASE_URL,
        Auth::Bearer(api_key),
        vec![
            Header::new("HTTP-Referer", "https://github.com/piku"),
            Header::new("X-Title", "piku"),
        ],
    )
}

pub fn from_env() -> Result<OpenAiCompatProvider, ApiError> {
    let key = std::env::var("OPENROUTER_API_KEY")
        .map_err(|_| ApiError::Provider("missing OPENROUTER_API_KEY".to_string()))?;
    let base = std::env::var("OPENROUTER_BASE_URL").unwrap_or_else(|_| BASE_URL.to_string());
    Ok(OpenAiCompatProvider::new(
        "openrouter",
        base,
        Auth::Bearer(key),
        vec![
            Header::new("HTTP-Referer", "https://github.com/piku"),
            Header::new("X-Title", "piku"),
        ],
    ))
}
