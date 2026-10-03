use std::sync::OnceLock;

use tracing_subscriber::EnvFilter;

static INITIALIZED: OnceLock<()> = OnceLock::new();
const DEFAULT_FILTER: &str = "piku=warn,piku_runtime=warn";

/// Install Piku's process-wide structured diagnostic subscriber.
///
/// Human-facing command output remains on stdout. Operational diagnostics use
/// tracing on stderr and are quiet by default because the interactive TUI
/// inherits stderr into the conversation viewport. `RUST_LOG` explicitly opts
/// into diagnostic detail.
pub fn init() {
    INITIALIZED.get_or_init(|| {
        let filter =
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(DEFAULT_FILTER));
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

    #[test]
    fn default_filter_keeps_diagnostics_out_of_the_interactive_view() {
        assert_eq!(super::DEFAULT_FILTER, "piku=warn,piku_runtime=warn");
    }
}
