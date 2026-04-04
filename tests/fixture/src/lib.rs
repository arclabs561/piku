pub mod stats;
pub mod parser;
pub mod utils;

/// Process a batch of values and return a summary string.
// TODO: this function is referenced in README.md but not yet implemented.
// A caller who tries to use it will get a compile error.
// Scenario: ask piku to implement it based on the README description.
pub fn process_batch(values: &[i32]) -> String {
    let _ = values; // placeholder — not implemented
    unimplemented!("process_batch: see README for spec")
}
