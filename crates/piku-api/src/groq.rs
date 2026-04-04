use crate::error::ApiError;
use crate::openai_compat::{Auth, OpenAiCompatProvider};

const BASE_URL: &str = "https://api.groq.com/openai/v1";
pub const DEFAULT_MODEL: &str = "moonshotai/kimi-k2-instruct";

pub fn from_env() -> Result<OpenAiCompatProvider, ApiError> {
    let key = std::env::var("GROQ_API_KEY")
        .map_err(|_| ApiError::Provider("missing GROQ_API_KEY".to_string()))?;
    let base = std::env::var("GROQ_BASE_URL").unwrap_or_else(|_| BASE_URL.to_string());
    Ok(OpenAiCompatProvider::new(
        "groq",
        base,
        Auth::Bearer(key),
        vec![],
    ))
}
