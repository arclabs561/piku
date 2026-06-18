use std::env;

use piku_api::{AnthropicProvider, ApiError, OpenAiCompatProvider, Provider};

pub const DEFAULT_MODEL_OPENROUTER: &str = "anthropic/claude-sonnet-4.6";
pub const DEFAULT_MODEL_ANTHROPIC: &str = "claude-sonnet-4-5";
pub const DEFAULT_MODEL_GROQ: &str = piku_api::groq::DEFAULT_MODEL;
pub const DEFAULT_MODEL_OLLAMA: &str = piku_api::ollama::DEFAULT_MODEL;

/// Provider client selected for a piku run, plus the provider's default model.
pub struct ResolvedProvider {
    pub provider: Box<dyn Provider>,
    pub default_model: String,
}

impl ResolvedProvider {
    pub fn resolve(provider_override: Option<&str>) -> Result<Self, ApiError> {
        match provider_override {
            Some(name) => Self::resolve_named(name),
            None => Self::resolve_opportunistic(),
        }
    }

    pub fn resolve_named(name: &str) -> Result<Self, ApiError> {
        match name {
            "openrouter" | "or" => {
                let p = piku_api::openrouter::from_env()?;
                Ok(Self::compat(p, DEFAULT_MODEL_OPENROUTER))
            }
            "anthropic" => {
                let p = AnthropicProvider::from_env()?;
                Ok(Self {
                    provider: Box::new(p),
                    default_model: DEFAULT_MODEL_ANTHROPIC.into(),
                })
            }
            "groq" => {
                let p = piku_api::groq::from_env()?;
                Ok(Self::compat(p, DEFAULT_MODEL_GROQ))
            }
            "ollama" => {
                let p = piku_api::ollama::from_env();
                Ok(Self::compat(p, DEFAULT_MODEL_OLLAMA))
            }
            "custom" => {
                let (p, model) = piku_api::custom::from_env()?;
                let default_model = model.unwrap_or_else(|| "unknown".to_string());
                Ok(Self::compat(p, &default_model))
            }
            other => Err(ApiError::Provider(format!(
                "unknown provider '{other}' - use: openrouter, anthropic, groq, ollama, custom"
            ))),
        }
    }

    pub fn resolve_opportunistic() -> Result<Self, ApiError> {
        if piku_api::custom::is_configured() {
            let (p, model) = piku_api::custom::from_env()?;
            let default_model = model.unwrap_or_else(|| "unknown".to_string());
            eprintln!("[piku] using custom provider ({default_model})");
            return Ok(Self::compat(p, &default_model));
        }
        if env::var("OPENROUTER_API_KEY").is_ok() {
            let p = piku_api::openrouter::from_env()?;
            return Ok(Self::compat(p, DEFAULT_MODEL_OPENROUTER));
        }
        if env::var("ANTHROPIC_API_KEY").is_ok() {
            let p = AnthropicProvider::from_env()?;
            return Ok(Self {
                provider: Box::new(p),
                default_model: DEFAULT_MODEL_ANTHROPIC.into(),
            });
        }
        if env::var("GROQ_API_KEY").is_ok() {
            let p = piku_api::groq::from_env()?;
            return Ok(Self::compat(p, DEFAULT_MODEL_GROQ));
        }
        if piku_api::ollama::is_available() {
            let p = piku_api::ollama::from_env();
            return Ok(Self::compat(p, DEFAULT_MODEL_OLLAMA));
        }
        Err(ApiError::Provider(
            "no provider available - set one of:\n  \
             OPENROUTER_API_KEY, ANTHROPIC_API_KEY, GROQ_API_KEY,\n  \
             OLLAMA_HOST (or run Ollama on localhost:11434),\n  \
             PIKU_BASE_URL (custom OpenAI-compatible server)"
                .to_string(),
        ))
    }

    #[must_use]
    pub fn compat(p: OpenAiCompatProvider, default_model: &str) -> Self {
        Self {
            provider: Box::new(p),
            default_model: default_model.to_string(),
        }
    }

    #[must_use]
    pub fn name(&self) -> &str {
        self.provider.name()
    }

    #[must_use]
    pub fn as_provider(&self) -> &dyn Provider {
        self.provider.as_ref()
    }
}
