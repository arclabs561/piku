use super::*;

#[tokio::test]
async fn unknown_task_join_returns_named_error_without_default_wait() {
    let registry = TaskRegistry::new();
    let result = tokio::time::timeout(
        Duration::from_millis(100),
        crate::agent_loop::execute_agent_join(
            &serde_json::json!({"task_id": "missing-task"}),
            &registry,
        ),
    )
    .await
    .expect("unknown task must not wait for the 300-second default");
    assert!(result.1);
    assert!(
        result.0.contains("unknown task: missing-task"),
        "{}",
        result.0
    );
    assert!(registry.inner().waiters.is_empty());
}

#[test]
fn unknown_task_receiver_closes_without_registering_waiter() {
    let registry = TaskRegistry::new();
    let mut rx = registry.wait_for(&AgentTaskId("missing".into()));
    assert!(matches!(
        rx.try_recv(),
        Err(oneshot::error::TryRecvError::Closed)
    ));
    assert!(registry.inner().waiters.is_empty());
}

#[tokio::test]
async fn waiter_registration_prunes_cancelled_receivers_across_tasks() {
    let registry = TaskRegistry::new();
    let first = registry.register("first".into(), "work".into(), 1, None);
    let second = registry.register("second".into(), "work".into(), 1, None);
    let live = registry.wait_for(&first);
    drop(registry.wait_for(&first));
    drop(registry.wait_for(&second));
    let other_live = registry.wait_for(&first);
    {
        let inner = registry.inner();
        assert_eq!(inner.waiters.len(), 1);
        assert_eq!(inner.waiters[&first].len(), 2);
    }
    registry.complete(&first, "finished", 3);
    for rx in [live, other_live] {
        let entry = rx.await.unwrap();
        assert_eq!(entry.status, TaskStatus::Done);
        assert_eq!(entry.output.as_deref(), Some("finished"));
        assert_eq!(entry.turns_used, 3);
    }
    assert!(registry.inner().waiters.is_empty());
    let terminal = registry.wait_for(&first).await.unwrap();
    assert_eq!(terminal.status, TaskStatus::Done);
    assert!(registry.inner().waiters.is_empty());
}

#[tokio::test]
async fn repeated_join_timeouts_keep_only_the_latest_closed_waiter() {
    let registry = TaskRegistry::new();
    let id = registry.register("running".into(), "work".into(), 1, None);
    for _ in 0..32 {
        let (message, error) = crate::agent_loop::execute_agent_join(
            &serde_json::json!({"task_id": id.to_string(), "timeout_secs": 0}),
            &registry,
        )
        .await;
        assert!(error);
        assert!(message.contains("timed out after 0s"));
        assert_eq!(registry.inner().waiters[&id].len(), 1);
    }
    let rx = registry.wait_for(&id);
    assert_eq!(registry.inner().waiters[&id].len(), 1);
    registry.fail(&id, "stopped");
    assert_eq!(rx.await.unwrap().status, TaskStatus::Failed);
    assert!(registry.inner().waiters.is_empty());
    let (joined, error) = crate::agent_loop::execute_agent_join(
        &serde_json::json!({"task_id": id.to_string()}),
        &registry,
    )
    .await;
    assert!(!error);
    assert!(joined.contains("failed"));
    assert!(joined.contains("stopped"));
}

#[tokio::test]
async fn running_and_already_terminal_joins_return_completed_output() {
    let registry = TaskRegistry::new();
    let id = registry.register("running".into(), "work".into(), 1, None);
    let params = serde_json::json!({"task_id": id.to_string(), "timeout_secs": 1});
    let (result, ()) = tokio::join!(
        crate::agent_loop::execute_agent_join(&params, &registry),
        async {
            tokio::task::yield_now().await;
            registry.complete(&id, "verified output", 2);
        }
    );
    assert!(!result.1);
    assert!(result.0.contains(&format!("task {id} done")));
    assert!(result.0.contains("verified output"));
    let repeated = crate::agent_loop::execute_agent_join(&params, &registry).await;
    assert_eq!(repeated, result);
    assert!(registry.inner().waiters.is_empty());
}
