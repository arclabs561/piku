use std::sync::OnceLock;

use tracing_subscriber::EnvFilter;

static INITIALIZED: OnceLock<()> = OnceLock::new();

/// Install Piku's process-wide structured diagnostic subscriber.
///
/// Human-facing command output remains on stdout. Operational diagnostics use
/// tracing on stderr and can be filtered with `RUST_LOG`.
pub fn init() {
    INITIALIZED.get_or_init(|| {
        let filter = EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| EnvFilter::new("piku=info,piku_runtime=info"));
        let _ = tracing_subscriber::fmt()
            .with_env_filter(filter)
            .with_target(false)
            .with_thread_ids(false)
            .with_thread_names(false)
            .compact()
            .try_init();
    });
}

#[cfg(test)]
mod tests {
    #[test]
    fn initialization_is_idempotent() {
        super::init();
        super::init();
    }
}
