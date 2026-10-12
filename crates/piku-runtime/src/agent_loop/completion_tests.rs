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
