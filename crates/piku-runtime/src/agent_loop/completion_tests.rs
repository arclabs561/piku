use super::*;

struct ChildProvider {
    tool: bool,
    stop: StopReason,
    write_path: Option<std::path::PathBuf>,
}

impl Provider for ChildProvider {
    fn name(&self) -> &'static str {
        "scripted-completion"
    }
    fn stream_message(
        &self,
        _: MessageRequest,
    ) -> std::pin::Pin<
        Box<dyn futures_util::Stream<Item = Result<Event, piku_api::ApiError>> + Send + '_>,
    > {
        let events = if self.tool {
            vec![
                Event::ToolUseStart {
                    id: "read".into(),
                    name: if self.write_path.is_some() {
                        "write_file"
                    } else {
                        "read_file"
                    }
                    .into(),
                },
                Event::ToolUseDelta {
                    id: "read".into(),
                    partial_json: self.write_path.as_ref().map_or_else(
                        || r#"{"path":"Cargo.toml"}"#.into(),
                        |path| {
                            serde_json::json!({"path": path, "content": "retained work"})
                                .to_string()
                        },
                    ),
                },
                Event::ToolUseEnd { id: "read".into() },
                Event::MessageStop {
                    stop_reason: StopReason::ToolUse,
                },
            ]
        } else {
            vec![
                Event::TextDelta {
                    text: "finished".into(),
                },
                Event::MessageStop {
                    stop_reason: self.stop.clone(),
                },
            ]
        };
        Box::pin(futures_util::stream::iter(events.into_iter().map(Ok)))
    }
}

async fn child_at_cap(
    tool: bool,
    stop: StopReason,
) -> (
    crate::task::TaskEntry,
    Vec<crate::run_record::RunEventEnvelope>,
) {
    let dir = tempfile::tempdir().unwrap();
    let registry = TaskRegistry::with_persistence(
        "parent",
        dir.path().join("sessions"),
        dir.path().join("runs"),
        dir.path().join("links"),
    );
    let id = registry.register("child".into(), "read and summarize".into(), 1, None);
    run_subagent_task(
        id.clone(),
        "read and summarize".into(),
        Box::new(ChildProvider {
            tool,
            stop,
            write_path: None,
        }),
        "scripted".into(),
        vec![],
        piku_tools::all_tool_definitions(),
        1,
        registry.clone(),
        1,
        None,
        None,
        vec![],
        None,
        std::sync::Arc::new(crate::permission::AllowAll),
    )
    .await;
    let entry = registry.status(&id).unwrap();
    let evidence = entry.evidence.as_ref().unwrap();
    let resumed = Session::load(&evidence.session_path).unwrap();
    assert!(!resumed.messages.is_empty());
    let (joined, is_error) = execute_agent_join(
        &serde_json::json!({"task_id": id.to_string(), "timeout_secs": 1}),
        &registry,
    )
    .await;
    assert!(!is_error);
    assert!(joined.contains(&format!("task {id} {}", entry.status)));
    assert!(joined.contains(entry.output.as_deref().unwrap()));
    assert!(joined.contains(&evidence.run_record_path.display().to_string()));
    assert!(joined.contains(&evidence.session_path.display().to_string()));
    assert!(joined.contains(&evidence.link_path.display().to_string()));
    let events = crate::run_record::read_run_record(&evidence.run_record_path).unwrap();
    (entry, events)
}

#[tokio::test]
async fn capped_child_requesting_tools_is_incomplete() {
    let (entry, events) = child_at_cap(true, StopReason::EndTurn).await;
    assert_eq!(entry.status, crate::task::TaskStatus::Failed);
    assert_eq!(entry.turns_used, 1);
    assert!(entry.output.unwrap().contains("max_turns"));
    assert!(events.iter().any(|e| matches!(&e.event, crate::run_record::RunEvent::TurnCompleted { stop_reason: Some(reason), .. } if reason == "max_turns")));
}

#[tokio::test]
async fn final_answer_at_cap_is_successful_end_turn() {
    let (entry, events) = child_at_cap(false, StopReason::EndTurn).await;
    assert_eq!(entry.status, crate::task::TaskStatus::Done);
    assert_eq!(entry.output.as_deref(), Some("finished"));
    assert!(events.iter().any(|e| matches!(&e.event, crate::run_record::RunEvent::TurnCompleted { stop_reason: Some(reason), .. } if reason == "end_turn")));
}

#[tokio::test]
async fn provider_token_limit_is_incomplete() {
    let (entry, events) = child_at_cap(false, StopReason::MaxTokens).await;
    assert_eq!(entry.status, crate::task::TaskStatus::Failed);
    assert_eq!(entry.turns_used, 1);
    assert!(entry.output.unwrap().contains("max_tokens"));
    assert!(events.iter().any(|e| matches!(&e.event, crate::run_record::RunEvent::TurnCompleted { stop_reason: Some(reason), .. } if reason == "max_tokens")));
}

#[tokio::test]
async fn exhausted_child_join_reports_retained_work_recovery() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("work.txt");
    let registry = TaskRegistry::new();
    let id = registry.register(
        "writer".into(),
        "write and summarize".into(),
        1,
        Some(dir.path().to_path_buf()),
    );
    run_subagent_task(
        id.clone(),
        "write and summarize".into(),
        Box::new(ChildProvider {
            tool: true,
            stop: StopReason::ToolUse,
            write_path: Some(path.clone()),
        }),
        "scripted".into(),
        vec![],
        piku_tools::all_tool_definitions(),
        1,
        registry.clone(),
        1,
        Some(dir.path().to_path_buf()),
        Some("retained-test-branch".into()),
        vec![],
        None,
        std::sync::Arc::new(crate::permission::AllowAll),
    )
    .await;
    assert_eq!(std::fs::read_to_string(path).unwrap(), "retained work");
    let (joined, is_error) = execute_agent_join(
        &serde_json::json!({"task_id": id.to_string(), "timeout_secs": 1}),
        &registry,
    )
    .await;
    assert!(!is_error);
    assert!(joined.contains("failed"));
    assert!(joined.contains("max_turns"));
    assert!(joined.contains("retained or cleanup incomplete"));
    assert!(joined.contains(&format!("task {id}")));
    assert!(joined.contains(&dir.path().display().to_string()));
    assert!(joined.contains("retained-test-branch"));
}

#[tokio::test]
async fn setup_failure_join_reports_retained_work_recovery() {
    let dir = tempfile::tempdir().unwrap();
    let repo = dir.path().join("repo");
    let worktree = dir.path().join("worktree");
    std::fs::create_dir(&repo).unwrap();
    let git = |args: &[&str]| {
        let output = std::process::Command::new("git")
            .args(args)
            .current_dir(&repo)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
    };
    git(&["init", "--quiet"]);
    git(&[
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "-c",
        "commit.gpgsign=false",
        "commit",
        "--allow-empty",
        "-m",
        "fixture",
    ]);
    git(&[
        "worktree",
        "add",
        "-b",
        "setup-recovery",
        worktree.to_str().unwrap(),
    ]);
    // Model a SubagentStart hook's write before child evidence setup runs.
    let retained = worktree.join("hook-created.txt");
    std::fs::write(&retained, "hook work survives setup failure").unwrap();

    for (failure_dir, cause) in [
        ("links", "could not persist subagent link"),
        ("runs", "could not open subagent run record"),
    ] {
        let evidence_root = dir.path().join(failure_dir);
        std::fs::create_dir(&evidence_root).unwrap();
        std::fs::write(evidence_root.join(failure_dir), "not a directory").unwrap();
        let registry = TaskRegistry::with_persistence(
            "parent",
            evidence_root.join("sessions"),
            evidence_root.join("runs"),
            evidence_root.join("links"),
        );
        let id = registry.register(
            "child".into(),
            "inspect work".into(),
            1,
            Some(worktree.clone()),
        );
        run_subagent_task(
            id.clone(),
            "inspect work".into(),
            Box::new(ChildProvider {
                tool: false,
                stop: StopReason::EndTurn,
                write_path: None,
            }),
            "scripted".into(),
            vec![],
            piku_tools::all_tool_definitions(),
            1,
            registry.clone(),
            1,
            Some(worktree.clone()),
            Some("setup-recovery".into()),
            vec![],
            None,
            std::sync::Arc::new(crate::permission::AllowAll),
        )
        .await;
        assert_eq!(
            std::fs::read_to_string(&retained).unwrap(),
            "hook work survives setup failure"
        );
        let (joined, is_error) = execute_agent_join(
            &serde_json::json!({"task_id": id.to_string(), "timeout_secs": 1}),
            &registry,
        )
        .await;
        assert!(!is_error);
        assert!(joined.contains(&format!("task {id} failed")), "{joined}");
        assert!(joined.contains(cause), "{joined}");
        assert!(
            joined.contains("retained or cleanup incomplete"),
            "{joined}"
        );
        assert!(joined.contains(&worktree.display().to_string()), "{joined}");
        assert!(joined.contains("Branch: setup-recovery"), "{joined}");
    }
}

struct RecordingFaultProvider {
    write_path: std::path::PathBuf,
    requested: std::sync::atomic::AtomicBool,
}

impl Provider for RecordingFaultProvider {
    fn name(&self) -> &'static str {
        "scripted-recording-fault"
    }

    fn stream_message(
        &self,
        _: MessageRequest,
    ) -> std::pin::Pin<
        Box<dyn futures_util::Stream<Item = Result<Event, piku_api::ApiError>> + Send + '_>,
    > {
        if self
            .requested
            .swap(true, std::sync::atomic::Ordering::SeqCst)
        {
            // Exceed the recorder's real inline boundary after the file effect.
            Box::pin(futures_util::stream::iter([
                Ok(Event::TextDelta {
                    text: "x".repeat(crate::run_record::RUN_INLINE_CONTENT_LIMIT_BYTES + 1),
                }),
                Ok(Event::MessageStop {
                    stop_reason: StopReason::EndTurn,
                }),
            ]))
        } else {
            Box::pin(futures_util::stream::iter([
                Ok(Event::ToolUseStart {
                    id: "write".into(),
                    name: "write_file".into(),
                }),
                Ok(Event::ToolUseDelta {
                    id: "write".into(),
                    partial_json:
                        serde_json::json!({"path": self.write_path, "content": "retained work"})
                            .to_string(),
                }),
                Ok(Event::ToolUseEnd { id: "write".into() }),
                Ok(Event::MessageStop {
                    stop_reason: StopReason::ToolUse,
                }),
            ]))
        }
    }
}

async fn child_with_recording_fault(
    root: &std::path::Path,
    block_session: bool,
) -> (TaskRegistry, crate::task::AgentTaskId) {
    let registry = TaskRegistry::with_persistence(
        "parent",
        root.join("sessions"),
        root.join("runs"),
        root.join("links"),
    );
    let id = registry.register("writer".into(), "write and summarize".into(), 1, None);
    let evidence = registry.evidence(&id).unwrap();
    std::fs::create_dir_all(evidence.run_record_path.parent().unwrap()).unwrap();
    // Same artifact obstruction used by the real PTY recording-failure test.
    std::fs::write(
        evidence.run_record_path.with_extension("artifacts"),
        "blocked",
    )
    .unwrap();
    if block_session {
        std::fs::write(root.join("sessions"), "blocked").unwrap();
    }
    run_subagent_task(
        id.clone(),
        "write and summarize".into(),
        Box::new(RecordingFaultProvider {
            write_path: root.join("work.txt"),
            requested: std::sync::atomic::AtomicBool::new(false),
        }),
        "scripted".into(),
        vec![],
        piku_tools::all_tool_definitions()
            .into_iter()
            .filter(|tool| tool.name == "write_file")
            .collect(),
        2,
        registry.clone(),
        1,
        None,
        None,
        vec![],
        None,
        std::sync::Arc::new(crate::permission::AllowAll),
    )
    .await;
    assert_eq!(
        std::fs::read_to_string(root.join("work.txt")).unwrap(),
        "retained work"
    );
    let records = crate::run_record::read_run_record(&evidence.run_record_path).unwrap();
    let audit = crate::run_audit::audit_run_record(&records);
    assert_eq!(audit.tool_calls_completed, 1);
    assert_eq!(audit.files_created, 1);
    assert_eq!(audit.completed_turn_count, 0);
    assert!(!audit.is_structurally_complete());
    (registry, id)
}

#[tokio::test]
async fn recording_failure_preserves_child_session_for_manual_resume() {
    let root = tempfile::tempdir().unwrap();
    let (registry, id) = child_with_recording_fault(root.path(), false).await;
    let entry = registry.status(&id).unwrap();
    assert_eq!(entry.status, crate::task::TaskStatus::Failed);
    assert_eq!(entry.turns_used, 2);
    let evidence = entry.evidence.as_ref().unwrap();
    let (joined, is_error) = execute_agent_join(
        &serde_json::json!({"task_id": id.to_string(), "timeout_secs": 1}),
        &registry,
    )
    .await;
    assert!(!is_error);
    assert!(joined.contains("run record failed"), "{joined}");
    assert!(joined.contains(&evidence.session_path.display().to_string()));
    let mut resumed = Session::load(&evidence.session_path)
        .expect("a recording failure must not suppress the independent session snapshot");
    assert!(resumed
        .messages
        .iter()
        .flat_map(|message| &message.blocks)
        .any(|block| matches!(block, ContentBlock::ToolUse { name, .. } if name == "write_file")));
    let prior_content: Vec<_> = resumed
        .messages
        .iter()
        .map(|message| serde_json::to_value((&message.role, &message.blocks)).unwrap())
        .collect();
    let result = run_turn(
        "review the retained write",
        &mut resumed,
        &ChildProvider {
            tool: false,
            stop: StopReason::EndTurn,
            write_path: None,
        },
        "scripted",
        &[],
        vec![],
        &crate::permission::AllowAll,
        &mut crate::task::DevNullSink,
        Some(1),
        None,
    )
    .await;
    assert!(result.stream_error.is_none());
    assert_eq!(resumed.messages.len(), prior_content.len() + 2);
    // Request construction may refresh importance scores; retained content is
    // the recovery invariant, rather than those derived selection weights.
    for (message, expected) in resumed.messages.iter().zip(prior_content) {
        assert!(
            serde_json::to_value((&message.role, &message.blocks)).unwrap() == expected,
            "manual resume changed retained message content"
        );
    }
    resumed.save(&evidence.session_path).unwrap();
    assert_eq!(
        Session::load(&evidence.session_path)
            .unwrap()
            .messages
            .len(),
        resumed.messages.len()
    );
    assert_eq!(
        std::fs::read_to_string(root.path().join("work.txt")).unwrap(),
        "retained work"
    );
    assert_eq!(
        registry.status(&id).unwrap().status,
        crate::task::TaskStatus::Failed
    );
}

#[tokio::test]
async fn recording_and_session_failures_are_both_reported_by_child_join() {
    let root = tempfile::tempdir().unwrap();
    let (registry, id) = child_with_recording_fault(root.path(), true).await;
    let (joined, is_error) = execute_agent_join(
        &serde_json::json!({"task_id": id.to_string(), "timeout_secs": 1}),
        &registry,
    )
    .await;
    assert!(!is_error);
    assert!(joined.contains("run record failed"), "{joined}");
    assert!(joined.contains("session save failed"), "{joined}");
    assert_eq!(
        registry.status(&id).unwrap().status,
        crate::task::TaskStatus::Failed
    );
}
