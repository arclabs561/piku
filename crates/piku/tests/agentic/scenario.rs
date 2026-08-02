//! Filesystem-backed scenario contracts and deterministic acceptance checks.

use std::io::Read;
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy)]
pub struct Scenario {
    pub id: &'static str,
    pub contexts: &'static [&'static str],
    pub goal: &'static str,
    pub verifications: &'static [Verification],
}

#[derive(Debug, Clone, Copy)]
pub enum Verification {
    FileContains {
        path: &'static str,
        needle: &'static str,
    },
    CommandSucceeds {
        program: &'static str,
        args: &'static [&'static str],
        timeout_secs: u64,
    },
}

impl Verification {
    #[must_use]
    pub fn label(self) -> String {
        match self {
            Self::FileContains { path, needle } => format!("{path} contains {needle:?}"),
            Self::CommandSucceeds { program, args, .. } => {
                format!("{} {}", program, args.join(" "))
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct VerificationResult {
    pub label: String,
    pub passed: bool,
    pub evidence: String,
}

pub const FEATURE_LINE_NUMBERS: Scenario = Scenario {
    id: "feature-line-numbers",
    contexts: &[
        "feature-development",
        "existing-rust-repository",
        "multi-file-change",
        "tests-present",
        "write-permissions",
    ],
    goal: "Return 1-based line numbers from both search functions, format run output as N:line, and keep the fixture test suite passing.",
    verifications: &[
        Verification::FileContains {
            path: "src/lib.rs",
            needle: "Vec<(usize, &'a str)>",
        },
        Verification::FileContains {
            path: "src/lib.rs",
            needle: ".enumerate()",
        },
        Verification::CommandSucceeds {
            program: "cargo",
            args: &["test", "--quiet"],
            timeout_secs: 30,
        },
    ],
};

#[must_use]
pub fn for_persona(persona: &str) -> Option<Scenario> {
    match persona {
        "feature_implementer" => Some(FEATURE_LINE_NUMBERS),
        _ => None,
    }
}

#[must_use]
pub fn verify(scenario: Scenario, workspace: &Path) -> Vec<VerificationResult> {
    scenario
        .verifications
        .iter()
        .map(|verification| match verification {
            Verification::FileContains { path, needle } => {
                let full_path = workspace.join(path);
                match std::fs::read_to_string(&full_path) {
                    Ok(content) => VerificationResult {
                        label: format!("{path} contains {needle:?}"),
                        passed: content.contains(needle),
                        evidence: if content.contains(needle) {
                            "required text present".to_string()
                        } else {
                            "required text absent".to_string()
                        },
                    },
                    Err(error) => VerificationResult {
                        label: format!("{path} contains {needle:?}"),
                        passed: false,
                        evidence: format!("could not read file: {error}"),
                    },
                }
            }
            Verification::CommandSucceeds {
                program,
                args,
                timeout_secs,
            } => run_bounded_command(workspace, program, args, *timeout_secs),
        })
        .collect()
}

fn run_bounded_command(
    workspace: &Path,
    program: &str,
    args: &[&str],
    timeout_secs: u64,
) -> VerificationResult {
    let label = format!("{} {}", program, args.join(" "));
    let mut child = match Command::new(program)
        .args(args)
        .current_dir(workspace)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(child) => child,
        Err(error) => {
            return VerificationResult {
                label,
                passed: false,
                evidence: format!("could not start command: {error}"),
            };
        }
    };

    let deadline = Instant::now() + Duration::from_secs(timeout_secs);
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                let mut output = String::new();
                if let Some(mut stdout) = child.stdout.take() {
                    let _ = stdout.read_to_string(&mut output);
                }
                if let Some(mut stderr) = child.stderr.take() {
                    let _ = stderr.read_to_string(&mut output);
                }
                return VerificationResult {
                    label,
                    passed: status.success(),
                    evidence: bounded_evidence(&output),
                };
            }
            Ok(None) if Instant::now() < deadline => {
                std::thread::sleep(Duration::from_millis(25));
            }
            Ok(None) => {
                let _ = child.kill();
                let _ = child.wait();
                return VerificationResult {
                    label,
                    passed: false,
                    evidence: format!("timed out after {timeout_secs}s"),
                };
            }
            Err(error) => {
                let _ = child.kill();
                let _ = child.wait();
                return VerificationResult {
                    label,
                    passed: false,
                    evidence: format!("could not poll command: {error}"),
                };
            }
        }
    }
}

fn bounded_evidence(output: &str) -> String {
    const MAX_CHARS: usize = 2_000;
    if output.chars().count() <= MAX_CHARS {
        return output.to_string();
    }
    let prefix: String = output.chars().take(MAX_CHARS / 2).collect();
    let suffix: String = output
        .chars()
        .rev()
        .take(MAX_CHARS / 2)
        .collect::<String>()
        .chars()
        .rev()
        .collect();
    format!("{prefix}\n… [verification output bounded] …\n{suffix}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn feature_scenario_has_goal_context_and_executable_oracle() {
        let scenario = for_persona("feature_implementer").unwrap();
        assert!(!scenario.goal.is_empty());
        assert!(scenario.contexts.contains(&"feature-development"));
        assert!(scenario
            .verifications
            .iter()
            .any(|verification| matches!(verification, Verification::CommandSucceeds { .. })));
    }

    #[test]
    fn file_verification_reports_pass_and_fail() {
        let directory = tempfile::tempdir().unwrap();
        std::fs::write(directory.path().join("result.txt"), "line numbers").unwrap();
        let scenario = Scenario {
            id: "test",
            contexts: &["test"],
            goal: "test",
            verifications: &[
                Verification::FileContains {
                    path: "result.txt",
                    needle: "line",
                },
                Verification::FileContains {
                    path: "result.txt",
                    needle: "missing",
                },
            ],
        };

        let results = verify(scenario, directory.path());
        assert!(results[0].passed);
        assert!(!results[1].passed);
    }
}
