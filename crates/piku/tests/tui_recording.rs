//! Contract tests for replayable, annotated managed-PTY evidence.

#![allow(dead_code)] // This target compiles the reusable PTY seam as a focused contract.

#[path = "agentic/types.rs"]
mod types;
use types::{Action, ScreenSnapshot};
#[path = "agentic/recording.rs"]
mod recording;

#[test]
fn completed_bundle_distinguishes_raw_input_from_terminal_echo() {
    let directory = tempfile::tempdir().unwrap();
    let (mut recording, paths) =
        recording::TuiRecording::create(directory.path(), 24, 80, "contract", false).unwrap();

    recording.input(b"private command\r").unwrap();
    recording.output(b"private command\r\n").unwrap();
    recording.finish().unwrap();

    let trace = std::fs::read_to_string(paths.trace).unwrap();
    assert!(trace.contains("\"raw_input_events\":\"omitted\""));
    assert!(trace.contains("\"echoed_input_in_output\":\"may_be_present\""));
    assert!(trace.contains("\"disposition\":\"complete\""));

    let cast = std::fs::read_to_string(paths.cast).unwrap();
    assert!(cast.contains("private command"));
}
