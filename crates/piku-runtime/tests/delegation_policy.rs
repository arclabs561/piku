use futures_util::Stream;
use piku_api::{ApiError, Event, MessageRequest, Provider, StopReason};
use piku_runtime::{
    run_turn_with_registry, PermissionOutcome, PermissionPrompter, PermissionRequest, Session,
    TaskRegistry,
};
use std::{path::PathBuf, pin::Pin};

struct DenyWrites;
impl PermissionPrompter for DenyWrites {
    fn delegated_policy(&self) -> Option<std::sync::Arc<dyn PermissionPrompter>> {
        Some(std::sync::Arc::new(Self))
    }
    fn decide(&self, _: &PermissionRequest) -> PermissionOutcome {
        PermissionOutcome::Allow
    }
    fn denies(&self, name: &str, _: &serde_json::Value) -> Option<String> {
        (name == "write_file").then(|| "configured write denial".into())
    }
}
struct DelegatingProvider {
    child: bool,
    path: PathBuf,
}
impl Provider for DelegatingProvider {
    fn name(&self) -> &'static str {
        "delegation-fixture"
    }
    fn boxed_clone(&self) -> Box<dyn Provider + Send + Sync + 'static> {
        Box::new(Self {
            child: true,
            path: self.path.clone(),
        })
    }
    fn stream_message(
        &self,
        _: MessageRequest,
    ) -> Pin<Box<dyn Stream<Item = Result<Event, ApiError>> + Send + '_>> {
        let (name, params) = if self.child {
            (
                "write_file",
                serde_json::json!({"path":self.path,"content":"denied write executed"}),
            )
        } else {
            (
                "spawn_agent",
                serde_json::json!({"task":"write fixture","max_turns":1}),
            )
        };
        Box::pin(futures_util::stream::iter(vec![
            Ok(Event::ToolUseStart {
                id: "call".into(),
                name: name.into(),
            }),
            Ok(Event::ToolUseDelta {
                id: "call".into(),
                partial_json: params.to_string(),
            }),
            Ok(Event::ToolUseEnd { id: "call".into() }),
            Ok(Event::MessageStop {
                stop_reason: StopReason::ToolUse,
            }),
        ]))
    }
}
#[tokio::test]
async fn delegated_child_retains_configured_parent_denial() {
    tokio::task::LocalSet::new()
        .run_until(async {
            for persistent in [false, true] {
                let dir = tempfile::tempdir().unwrap();
                let path = dir.path().join("denied.txt");
                let registry = if persistent {
                    TaskRegistry::with_persistence(
                        "parent",
                        dir.path().join("sessions"),
                        dir.path().join("runs"),
                        dir.path().join("links"),
                    )
                } else {
                    TaskRegistry::new()
                };
                let mut session = Session::new("parent".into());
                let mut sink = piku_runtime::task::DevNullSink;
                run_turn_with_registry(
                    "delegate",
                    &mut session,
                    &DelegatingProvider {
                        child: false,
                        path: path.clone(),
                    },
                    "fixture",
                    &[],
                    piku_tools::all_tool_definitions(),
                    &DenyWrites,
                    &mut sink,
                    Some(1),
                    None,
                    &registry,
                    0,
                    &[],
                    None,
                    None,
                )
                .await;
                let entries = registry.all();
                assert_eq!(entries.len(), 1, "child must actually launch");
                tokio::time::timeout(
                    std::time::Duration::from_secs(5),
                    registry.wait_for(&entries[0].id),
                )
                .await
                .unwrap()
                .unwrap();
                assert!(
                    !path.exists(),
                    "child bypassed parent configured denial (persistent={persistent})"
                );
            }
        })
        .await;
}

struct Unsnapshotable;
impl PermissionPrompter for Unsnapshotable {
    fn decide(&self, _: &PermissionRequest) -> PermissionOutcome {
        PermissionOutcome::Allow
    }
}

#[tokio::test]
async fn unknown_policy_refuses_spawn_and_explicit_allow_all_still_runs() {
    tokio::task::LocalSet::new()
        .run_until(async {
            for (policy, should_spawn) in [
                (&Unsnapshotable as &dyn PermissionPrompter, false),
                (&piku_runtime::AllowAll as &dyn PermissionPrompter, true),
            ] {
                let dir = tempfile::tempdir().unwrap();
                let path = dir.path().join("allowed.txt");
                let registry = TaskRegistry::new();
                let mut session = Session::new("parent".into());
                run_turn_with_registry(
                    "delegate",
                    &mut session,
                    &DelegatingProvider {
                        child: false,
                        path: path.clone(),
                    },
                    "fixture",
                    &[],
                    piku_tools::all_tool_definitions(),
                    policy,
                    &mut piku_runtime::task::DevNullSink,
                    Some(1),
                    None,
                    &registry,
                    0,
                    &[],
                    None,
                    None,
                )
                .await;
                assert_eq!(registry.all().len(), usize::from(should_spawn));
                if should_spawn {
                    let id = registry.all()[0].id.clone();
                    tokio::time::timeout(std::time::Duration::from_secs(5), registry.wait_for(&id))
                        .await
                        .unwrap()
                        .unwrap();
                    assert_eq!(
                        std::fs::read_to_string(path).unwrap(),
                        "denied write executed"
                    );
                } else {
                    assert!(!path.exists());
                    assert!(serde_json::to_string(&session)
                        .unwrap()
                        .contains("parent permission policy cannot be delegated"));
                }
            }
        })
        .await;
}

#[derive(Clone)]
struct DenyContextRead {
    path: PathBuf,
    deny_here: bool,
    deny_child: bool,
}
impl PermissionPrompter for DenyContextRead {
    fn delegated_policy(&self) -> Option<std::sync::Arc<dyn PermissionPrompter>> {
        Some(std::sync::Arc::new(Self {
            path: self.path.clone(),
            deny_here: self.deny_child,
            deny_child: self.deny_child,
        }))
    }
    fn decide(&self, _: &PermissionRequest) -> PermissionOutcome {
        PermissionOutcome::Allow
    }
    fn denies(&self, name: &str, params: &serde_json::Value) -> Option<String> {
        (self.deny_here
            && name == "read_file"
            && params["path"] == self.path.to_string_lossy().as_ref())
        .then(|| "configured attachment denial".into())
    }
}
struct ContextProvider {
    child: bool,
    path: PathBuf,
    captured: std::sync::Arc<std::sync::Mutex<Vec<String>>>,
}
impl Provider for ContextProvider {
    fn name(&self) -> &'static str {
        "context-fixture"
    }
    fn boxed_clone(&self) -> Box<dyn Provider + Send + Sync + 'static> {
        Box::new(Self {
            child: true,
            path: self.path.clone(),
            captured: self.captured.clone(),
        })
    }
    fn stream_message(
        &self,
        request: MessageRequest,
    ) -> Pin<Box<dyn Stream<Item = Result<Event, ApiError>> + Send + '_>> {
        let events = if self.child {
            self.captured
                .lock()
                .unwrap()
                .push(serde_json::to_string(&request).unwrap());
            vec![
                Event::TextDelta {
                    text: "done".into(),
                },
                Event::MessageStop {
                    stop_reason: StopReason::EndTurn,
                },
            ]
        } else {
            vec![Event::ToolUseStart {id:"context".into(),name:"spawn_agent".into()},
                Event::ToolUseDelta {id:"context".into(), partial_json:serde_json::json!({"task":"inspect attachment","context_files":[self.path],"max_turns":1}).to_string()},
                Event::ToolUseEnd {id:"context".into()}, Event::MessageStop {stop_reason:StopReason::ToolUse}]
        };
        Box::pin(futures_util::stream::iter(events.into_iter().map(Ok)))
    }
}
#[tokio::test]
async fn denied_context_attachment_never_reaches_child_model() {
    tokio::task::LocalSet::new()
        .run_until(async {
            let dir = tempfile::tempdir_in(std::env::current_dir().unwrap()).unwrap();
            let path = dir.path().join("attachment.txt");
            std::fs::write(&path, "CONTEXT_DENIAL_FIXTURE_MARKER").unwrap();
            // Exercise parent and child-only restrictions independently.
            for (deny_here, deny_child) in [(false, false), (true, false), (false, true)] {
                let denied = deny_here || deny_child;
                let captured = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
                let policy = DenyContextRead {
                    path: path.clone(),
                    deny_here,
                    deny_child,
                };
                let prompter: &dyn PermissionPrompter = if denied {
                    &policy
                } else {
                    &piku_runtime::AllowAll
                };
                let registry = TaskRegistry::new();
                let mut session = Session::new("attachment-parent".into());
                run_turn_with_registry(
                    "delegate",
                    &mut session,
                    &ContextProvider {
                        child: false,
                        path: path.clone(),
                        captured: captured.clone(),
                    },
                    "fixture",
                    &[],
                    piku_tools::all_tool_definitions(),
                    prompter,
                    &mut piku_runtime::task::DevNullSink,
                    Some(1),
                    None,
                    &registry,
                    0,
                    &[],
                    None,
                    None,
                )
                .await;
                for entry in registry.all() {
                    tokio::time::timeout(
                        std::time::Duration::from_secs(5),
                        registry.wait_for(&entry.id),
                    )
                    .await
                    .unwrap()
                    .unwrap();
                }
                let requests = captured.lock().unwrap();
                if denied {
                    assert!(
                        !requests
                            .iter()
                            .any(|request| request.contains("CONTEXT_DENIAL_FIXTURE_MARKER")),
                        "denied attachment bytes reached child provider"
                    );
                    assert!(
                        requests.is_empty(),
                        "denied required context must refuse spawn"
                    );
                    assert!(registry.all().is_empty());
                    assert!(serde_json::to_string(&session)
                        .unwrap()
                        .contains("configured attachment denial"));
                } else {
                    assert_eq!(requests.len(), 1);
                    assert!(requests[0].contains("CONTEXT_DENIAL_FIXTURE_MARKER"));
                }
            }
        })
        .await;
}
