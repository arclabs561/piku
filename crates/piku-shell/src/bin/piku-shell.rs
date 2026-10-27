//! Standalone entry point for the shell that `piku shell` runs in-process.

fn main() {
    std::process::exit(piku_shell::run_cli());
}
