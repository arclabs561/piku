use crate::openai_compat::{Auth, OpenAiCompatProvider};

const DEFAULT_HOST: &str = "http://localhost:11434";
pub const DEFAULT_MODEL: &str = "llama3.1";

#[must_use]
pub fn from_env() -> OpenAiCompatProvider {
    // OLLAMA_HOST overrides the default (no key required)
    let host = std::env::var("OLLAMA_HOST").unwrap_or_else(|_| DEFAULT_HOST.to_string());
    let base = format!("{host}/v1");
    OpenAiCompatProvider::new("ollama", base, Auth::None, vec![])
}

/// Check if Ollama appears to be running (best-effort, non-blocking check).
#[must_use]
pub fn is_available() -> bool {
    let host = std::env::var("OLLAMA_HOST").unwrap_or_else(|_| DEFAULT_HOST.to_string());
    // Try a quick TCP connect to the host:port
    let addr = host
        .trim_start_matches("http://")
        .trim_start_matches("https://");
    let addr = if addr.contains(':') {
        addr.to_string()
    } else {
        format!("{addr}:11434")
    };
    std::net::TcpStream::connect_timeout(
        &addr
            .parse()
            .unwrap_or_else(|_| "127.0.0.1:11434".parse().unwrap()),
        std::time::Duration::from_millis(200),
    )
    .is_ok()
}
