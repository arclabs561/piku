/// Embedding backend protocol.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EmbedBackend {
    /// Ollama native: POST {url}/api/embed, response: `{"embeddings": [[...]]}`
    Ollama,
    /// OpenAI-compatible: POST `{url}/v1/embeddings`, response: `{"data": [{"embedding": [...]}]}`
    /// Works with any provider that implements the OAI-compatible embeddings API.
    OpenAiCompat,
}

/// Resolved embedding configuration.
#[derive(Debug, Clone)]
pub struct EmbedConfig {
    pub backend: EmbedBackend,
    pub base_url: String,
    pub model: String,
    pub api_key: Option<String>,
}

impl EmbedConfig {
    /// Build from environment variables, with fallback chain:
    /// 1. `PIKU_EMBED_URL` with optional `PIKU_EMBED_BACKEND`
    /// 2. `OLLAMA_HOST` set (or default localhost) selects Ollama native
    /// 3. `OPENROUTER_API_KEY` set selects `/v1/embeddings` via `OpenRouter`
    ///
    /// Backend: `PIKU_EMBED_BACKEND=ollama|openai-compat` for custom URLs.
    /// Model: `PIKU_EMBED_MODEL` overrides the default for any backend.
    /// API key: `PIKU_EMBED_API_KEY` → `OPENROUTER_API_KEY` → none.
    #[must_use]
    pub fn from_env() -> Self {
        // Explicit override: user set a custom embedding URL.
        if let Ok(url) = std::env::var("PIKU_EMBED_URL") {
            let model = std::env::var("PIKU_EMBED_MODEL")
                .unwrap_or_else(|_| "text-embedding-3-small".to_string());
            let api_key = std::env::var("PIKU_EMBED_API_KEY")
                .ok()
                .or_else(|| std::env::var("OPENROUTER_API_KEY").ok());
            let backend = embed_backend_from_env().unwrap_or(EmbedBackend::OpenAiCompat);
            return Self {
                backend,
                base_url: url.trim_end_matches('/').to_string(),
                model,
                api_key,
            };
        }

        // Default: Ollama (local, no API key needed)
        // OLLAMA_HOST is set by default in most piku environments.
        let ollama_url = std::env::var("OLLAMA_HOST")
            .unwrap_or_else(|_| piku_api::ollama::DEFAULT_HOST.to_string());
        let embed_model = std::env::var("PIKU_EMBED_MODEL");

        // If OLLAMA_HOST was explicitly set, always use Ollama
        if std::env::var("OLLAMA_HOST").is_ok() {
            return Self {
                backend: EmbedBackend::Ollama,
                base_url: ollama_url,
                model: embed_model.unwrap_or_else(|_| "nomic-embed-text".to_string()),
                api_key: None,
            };
        }

        // If OpenRouter API key is available, use it for embeddings too
        if let Ok(key) = std::env::var("OPENROUTER_API_KEY") {
            let base = std::env::var("OPENROUTER_BASE_URL")
                .unwrap_or_else(|_| "https://openrouter.ai/api".to_string());
            return Self {
                backend: EmbedBackend::OpenAiCompat,
                base_url: base.trim_end_matches('/').to_string(),
                model: embed_model.unwrap_or_else(|_| "openai/text-embedding-3-small".to_string()),
                api_key: Some(key),
            };
        }

        // Last resort: Ollama at default host (will fail at call time if not running)
        Self {
            backend: EmbedBackend::Ollama,
            base_url: ollama_url,
            model: embed_model.unwrap_or_else(|_| "nomic-embed-text".to_string()),
            api_key: None,
        }
    }
}

fn embed_backend_from_env() -> Option<EmbedBackend> {
    let value = std::env::var("PIKU_EMBED_BACKEND").ok()?;
    match value.trim().to_ascii_lowercase().as_str() {
        "ollama" => Some(EmbedBackend::Ollama),
        "openai" | "openai-compat" | "openai_compat" | "openai-compatible" => {
            Some(EmbedBackend::OpenAiCompat)
        }
        _ => None,
    }
}

/// Embed text using the configured backend.
/// Returns a normalized vector (dimension depends on model).
pub async fn embed_text(text: &str, ollama_url: &str, model: &str) -> Result<Vec<f32>, String> {
    // Legacy signature compat: build config from args
    let config = EmbedConfig {
        backend: EmbedBackend::Ollama,
        base_url: ollama_url.to_string(),
        model: model.to_string(),
        api_key: None,
    };
    embed_text_with_config(text, &config).await
}

/// Embed text using an explicit config (supports both Ollama and OpenAI-compat).
pub async fn embed_text_with_config(text: &str, config: &EmbedConfig) -> Result<Vec<f32>, String> {
    let client = reqwest::Client::new();

    let (url, body, auth) = match config.backend {
        EmbedBackend::Ollama => (
            format!("{}/api/embed", config.base_url.trim_end_matches('/')),
            serde_json::json!({
                "model": config.model,
                "input": text,
            }),
            None,
        ),
        EmbedBackend::OpenAiCompat => (
            format!("{}/v1/embeddings", config.base_url.trim_end_matches('/')),
            serde_json::json!({
                "model": config.model,
                "input": text,
            }),
            config.api_key.as_deref(),
        ),
    };

    let mut req = client.post(&url).json(&body);
    if let Some(key) = auth {
        req = req.header("Authorization", format!("Bearer {key}"));
    }

    let resp = req
        .send()
        .await
        .map_err(|e| format!("embed request failed: {e}"))?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        return Err(format!("embed failed ({status}): {body}"));
    }

    let json: serde_json::Value = resp
        .json()
        .await
        .map_err(|e| format!("embed parse failed: {e}"))?;

    // Parse response: Ollama uses "embeddings[0]", OpenAI-compat uses "data[0].embedding"
    let raw_embedding = match config.backend {
        EmbedBackend::Ollama => json
            .get("embeddings")
            .and_then(|e| e.get(0))
            .ok_or_else(|| "no embeddings in Ollama response".to_string())?,
        EmbedBackend::OpenAiCompat => json
            .get("data")
            .and_then(|d| d.get(0))
            .and_then(|d| d.get("embedding"))
            .ok_or_else(|| "no data[0].embedding in response".to_string())?,
    };

    let mut vec: Vec<f32> = serde_json::from_value(raw_embedding.clone())
        .map_err(|e| format!("embedding parse failed: {e}"))?;

    // Normalize to unit length so dot product = cosine similarity.
    super::normalize(&mut vec);

    Ok(vec)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    const EMBED_ENV_KEYS: &[&str] = &[
        "PIKU_EMBED_URL",
        "PIKU_EMBED_BACKEND",
        "PIKU_EMBED_MODEL",
        "PIKU_EMBED_API_KEY",
        "OPENROUTER_API_KEY",
        "OPENROUTER_BASE_URL",
        "OLLAMA_HOST",
    ];

    struct EnvRestore(Vec<(&'static str, Option<String>)>);

    impl EnvRestore {
        fn new() -> Self {
            let saved = EMBED_ENV_KEYS
                .iter()
                .map(|key| (*key, std::env::var(key).ok()))
                .collect();
            for key in EMBED_ENV_KEYS {
                std::env::remove_var(key);
            }
            Self(saved)
        }
    }

    impl Drop for EnvRestore {
        fn drop(&mut self) {
            for (key, value) in &self.0 {
                if let Some(value) = value {
                    std::env::set_var(key, value);
                } else {
                    std::env::remove_var(key);
                }
            }
        }
    }

    fn with_clean_embed_env<T>(f: impl FnOnce() -> T) -> T {
        let _guard = ENV_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let _restore = EnvRestore::new();
        f()
    }

    #[test]
    fn custom_url_defaults_to_openai_compat() {
        with_clean_embed_env(|| {
            std::env::set_var("PIKU_EMBED_URL", "http://localhost:11434/");

            let config = EmbedConfig::from_env();

            assert_eq!(config.backend, EmbedBackend::OpenAiCompat);
            assert_eq!(config.base_url, "http://localhost:11434");
            assert_eq!(config.model, "text-embedding-3-small");
            assert_eq!(config.api_key, None);
        });
    }

    #[test]
    fn custom_url_respects_explicit_ollama_backend() {
        with_clean_embed_env(|| {
            std::env::set_var("PIKU_EMBED_URL", "http://localhost:11434/");
            std::env::set_var("PIKU_EMBED_BACKEND", "ollama");
            std::env::set_var("PIKU_EMBED_MODEL", "nomic-embed-text");

            let config = EmbedConfig::from_env();

            assert_eq!(config.backend, EmbedBackend::Ollama);
            assert_eq!(config.base_url, "http://localhost:11434");
            assert_eq!(config.model, "nomic-embed-text");
            assert_eq!(config.api_key, None);
        });
    }

    #[test]
    fn default_reuses_ollama_host() {
        with_clean_embed_env(|| {
            let config = EmbedConfig::from_env();

            assert_eq!(config.backend, EmbedBackend::Ollama);
            assert_eq!(config.base_url, piku_api::ollama::DEFAULT_HOST);
            assert_eq!(config.model, "nomic-embed-text");
            assert_eq!(config.api_key, None);
        });
    }
}
