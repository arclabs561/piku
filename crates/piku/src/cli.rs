#![allow(clippy::assigning_clones)]

/// CLI argument parsing and provider resolution.
/// Split into lib.rs so integration tests can access these types.
use std::env;

use anyhow::Result;
use piku_api::{AnthropicProvider, OpenAiCompatProvider};
use piku_runtime::Provider;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

pub const DEFAULT_MODEL_OPENROUTER: &str = "anthropic/claude-sonnet-4.6";
pub const DEFAULT_MODEL_ANTHROPIC: &str = "claude-sonnet-4-5";
pub const DEFAULT_MODEL_GROQ: &str = piku_api::groq::DEFAULT_MODEL;
pub const DEFAULT_MODEL_OLLAMA: &str = piku_api::ollama::DEFAULT_MODEL;

// ---------------------------------------------------------------------------
// CLI action
// ---------------------------------------------------------------------------

pub enum CliAction {
    Version,
    Help,
    /// Interactive REPL — no prompt given.
    Repl {
        model: Option<String>,
        provider_override: Option<String>,
    },
    /// Prompt without a prior session.
    SingleShot {
        prompt: String,
        model: Option<String>,
        provider_override: Option<String>,
    },
    /// Resume a previous session by ID, then run prompt.
    Resume {
        session_id: String,
        prompt: Option<String>,
        model: Option<String>,
        provider_override: Option<String>,
    },
    /// Error in argument parsing — surface to user.
    ArgError(String),
}

#[must_use]
pub fn parse_args(args: &[String]) -> CliAction {
    let mut model: Option<String> = None;
    let mut provider_override: Option<String> = None;
    let mut resume_session: Option<String> = None;
    let mut rest: Vec<String> = Vec::new();
    let mut i = 0;

    while i < args.len() {
        match args[i].as_str() {
            "--version" | "-V" => return CliAction::Version,
            "--help" | "-h" => return CliAction::Help,
            "--model" => match args.get(i + 1) {
                Some(v) if !v.starts_with('-') => {
                    model = Some(v.clone());
                    i += 2;
                }
                _ => {
                    return CliAction::ArgError(
                        "--model requires a value (e.g. --model=claude-opus-4)".to_string(),
                    )
                }
            },
            flag if flag.starts_with("--model=") => {
                let val = &flag[8..];
                if val.is_empty() {
                    return CliAction::ArgError("--model= requires a non-empty value".to_string());
                }
                model = Some(val.to_string());
                i += 1;
            }
            "--provider" => match args.get(i + 1) {
                Some(v) if !v.starts_with('-') => {
                    provider_override = Some(v.clone());
                    i += 2;
                }
                _ => {
                    return CliAction::ArgError(
                        "--provider requires a value (e.g. --provider=openrouter)".to_string(),
                    )
                }
            },
            flag if flag.starts_with("--provider=") => {
                let val = &flag[11..];
                if val.is_empty() {
                    return CliAction::ArgError(
                        "--provider= requires a non-empty value".to_string(),
                    );
                }
                provider_override = Some(val.to_string());
                i += 1;
            }
            "--resume" => match args.get(i + 1) {
                Some(v) if !v.starts_with('-') => {
                    resume_session = Some(v.clone());
                    i += 2;
                }
                _ => {
                    return CliAction::ArgError(
                        "--resume requires a session ID (e.g. --resume session-123)".to_string(),
                    )
                }
            },
            flag if flag.starts_with("--resume=") => {
                let val = &flag[9..];
                if val.is_empty() {
                    return CliAction::ArgError(
                        "--resume= requires a non-empty session ID".to_string(),
                    );
                }
                resume_session = Some(val.to_string());
                i += 1;
            }
            other => {
                rest.push(other.to_string());
                i += 1;
            }
        }
    }

    let prompt_str = if rest.is_empty() {
        None
    } else {
        Some(rest.join(" "))
    };

    if let Some(session_id) = resume_session {
        CliAction::Resume {
            session_id,
            prompt: prompt_str,
            model,
            provider_override,
        }
    } else if let Some(prompt) = prompt_str {
        CliAction::SingleShot {
            prompt,
            model,
            provider_override,
        }
    } else {
        // No prompt given → interactive REPL
        CliAction::Repl {
            model,
            provider_override,
        }
    }
}

// ---------------------------------------------------------------------------
// Provider resolution
// ---------------------------------------------------------------------------

/// Resolved provider with its default model name.
pub struct ResolvedProvider {
    pub provider: Box<dyn Provider>,
    pub default_model: String,
}

impl ResolvedProvider {
    pub fn resolve(provider_override: Option<&str>) -> Result<Self> {
        match provider_override {
            Some(name) => Self::resolve_named(name),
            None => Self::resolve_opportunistic(),
        }
    }

    pub fn resolve_named(name: &str) -> Result<Self> {
        match name {
            "openrouter" | "or" => {
                let p = piku_api::openrouter::from_env().map_err(|e| anyhow::anyhow!("{e}"))?;
                Ok(Self::compat(p, DEFAULT_MODEL_OPENROUTER))
            }
            "anthropic" => {
                let p = AnthropicProvider::from_env().map_err(|e| anyhow::anyhow!("{e}"))?;
                Ok(Self {
                    provider: Box::new(p),
                    default_model: DEFAULT_MODEL_ANTHROPIC.into(),
                })
            }
            "groq" => {
                let p = piku_api::groq::from_env().map_err(|e| anyhow::anyhow!("{e}"))?;
                Ok(Self::compat(p, DEFAULT_MODEL_GROQ))
            }
            "ollama" => {
                let p = piku_api::ollama::from_env();
                Ok(Self::compat(p, DEFAULT_MODEL_OLLAMA))
            }
            "custom" => {
                let (p, model) =
                    piku_api::custom::from_env().map_err(|e| anyhow::anyhow!("{e}"))?;
                let default_model = model.unwrap_or_else(|| "unknown".to_string());
                Ok(Self::compat(p, &default_model))
            }
            other => anyhow::bail!(
                "unknown provider '{other}' — use: openrouter, anthropic, groq, ollama, custom"
            ),
        }
    }

    pub fn resolve_opportunistic() -> Result<Self> {
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
            let p = AnthropicProvider::from_env().map_err(|e| anyhow::anyhow!("{e}"))?;
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
        anyhow::bail!(
            "no provider available — set one of:\n  \
             OPENROUTER_API_KEY, ANTHROPIC_API_KEY, GROQ_API_KEY,\n  \
             OLLAMA_HOST (or run Ollama on localhost:11434),\n  \
             PIKU_BASE_URL (custom OpenAI-compatible server)"
        )
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
