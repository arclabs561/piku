//! Explicit, metadata-only transport diagnostics for one terminal launch.
//!
//! A cutoff records what this writer saw, not proof that background producers
//! settled or that reported usage equals billing. Callbacks never wait for IO.

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{mpsc, Arc};
use std::time::{Duration, Instant};

use piku_api::observation::{InferenceObservation, InferenceObserver};
use serde_json::{json, Value};

const CAPACITY: usize = 256;
const FINISH_TIMEOUT: Duration = Duration::from_secs(2);

/// Resolve the normal selected route once, then opt into its supported observer.
pub fn resolve(
    provider: Option<&str>,
    log: Option<&RequestLog>,
) -> anyhow::Result<piku_runtime::ResolvedProvider> {
    let resolved = piku_runtime::ResolvedProvider::resolve(provider)?;
    if let Some(log) = log {
        log.observe(resolved)
    } else {
        Ok(resolved)
    }
}

struct State {
    accepting: AtomicBool,
    callbacks: AtomicU64,
    active: AtomicU64,
    full: AtomicU64,
    closed: AtomicU64,
    rejected: AtomicU64,
    enqueued: AtomicU64,
    drained: AtomicU64,
    failed: AtomicBool,
    origin: Instant,
    namespace: String,
}

struct Event {
    provider: String,
    binding: Option<Value>,
    observation: InferenceObservation,
}

fn observer(
    events: mpsc::SyncSender<Event>,
    state: Arc<State>,
    provider: String,
    binding: Option<Value>,
) -> Arc<InferenceObserver> {
    Arc::new(move |observation| {
        state.callbacks.fetch_add(1, Ordering::SeqCst);
        if state.accepting.load(Ordering::SeqCst) {
            match &observation {
                InferenceObservation::Started { .. } => {
                    state.active.fetch_add(1, Ordering::SeqCst);
                }
                InferenceObservation::Finished(_) => {
                    state.active.fetch_sub(1, Ordering::SeqCst);
                }
                _ => {}
            }
            match events.try_send(Event {
                provider: provider.clone(),
                binding: binding.clone(),
                observation,
            }) {
                Ok(()) => {
                    state.enqueued.fetch_add(1, Ordering::SeqCst);
                }
                Err(mpsc::TrySendError::Full(_)) => {
                    state.full.fetch_add(1, Ordering::SeqCst);
                }
                Err(mpsc::TrySendError::Disconnected(_)) => {
                    state.closed.fetch_add(1, Ordering::SeqCst);
                }
            }
        } else {
            state.rejected.fetch_add(1, Ordering::SeqCst);
        }
        state.callbacks.fetch_sub(1, Ordering::SeqCst);
    })
}

// Admit bounded opaque forms used by current constructors. Other imported text
// remains unknown; matching this grammar does not verify an ID's authorship.
fn decimal(value: &str) -> bool {
    !value.is_empty()
        && value.bytes().all(|byte| byte.is_ascii_digit())
        && value.parse::<u128>().is_ok()
}

fn hexadecimal(value: &str) -> bool {
    !value.is_empty()
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        && u128::from_str_radix(value, 16).is_ok()
}

fn task_identity(value: &str) -> bool {
    let Some(parts) = value.strip_prefix("agent-") else {
        return false;
    };
    let parts: Vec<_> = parts.split('-').collect();
    parts.len() == 3 && decimal(parts[0]) && hexadecimal(parts[1]) && hexadecimal(parts[2])
}

fn run_identity(value: &str) -> bool {
    if let Some(task) = value.strip_prefix("subagent-") {
        return task_identity(task);
    }
    let Some(parts) = value.strip_prefix("session-") else {
        return false;
    };
    let parts: Vec<_> = parts.split('-').collect();
    parts.len() == 2 && decimal(parts[0]) && decimal(parts[1])
}

fn turn_identity(value: &str) -> bool {
    let Some(turn) = value.strip_prefix("turn-") else {
        return false;
    };
    if turn.len() == 32 && hexadecimal(turn) {
        return true;
    }
    // The private child runner currently emits turn-<depth>-0.
    let Some(depth) = turn.strip_suffix("-0") else {
        return false;
    };
    decimal(depth)
}

fn safe_identity(value: Option<String>, validate: fn(&str) -> bool) -> Option<String> {
    value.filter(|value| value.len() <= 128 && validate(value))
}

fn binding_value(binding: piku_runtime::CausalBinding) -> Value {
    use piku_runtime::RequestPurpose;
    let run_id = safe_identity(binding.run_id, run_identity);
    let turn_id = safe_identity(binding.turn_id, turn_identity);
    // A sequence without a known run and turn cannot identify a ContextBuilt.
    let context_sequence = if run_id.is_some() && turn_id.is_some() {
        binding.context_sequence
    } else {
        None
    };
    let parent_origin = binding.parent_origin.and_then(|parent| {
        let run_id = safe_identity(parent.run_id, run_identity);
        let turn_id = safe_identity(parent.turn_id, turn_identity);
        (run_id.is_some() || turn_id.is_some()).then(|| json!({"run_id":run_id,"turn_id":turn_id}))
    });
    let purpose = match binding.purpose {
        RequestPurpose::AgentTurn => "agent_turn",
        RequestPurpose::PeriodicMemory => "periodic_memory",
        RequestPurpose::ExitMemory => "exit_memory",
        _ => "unknown",
    };
    json!({"run_id":run_id,"turn_id":turn_id,"context_sequence":context_sequence,
        "task_id":safe_identity(binding.task_id, task_identity),"parent_origin":parent_origin,"purpose":purpose})
}

enum Control {
    Checkpoint(mpsc::SyncSender<Result<(), String>>),
    Finish(mpsc::SyncSender<Result<(), String>>),
}

/// One fresh output and its bounded writer. No project settings enable it.
pub struct RequestLog {
    state: Arc<State>,
    events: mpsc::SyncSender<Event>,
    controls: mpsc::SyncSender<Control>,
    finished: bool,
}

impl RequestLog {
    pub fn create(path: &Path) -> anyhow::Result<Self> {
        let mut nonce = [0; 16];
        getrandom::fill(&mut nonce)
            .map_err(|error| anyhow::anyhow!("request-log namespace: {error}"))?;
        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let file = options
            .open(path)
            .map_err(|error| anyhow::anyhow!("could not create fresh request log: {error}"))?;
        Self::start(
            file,
            format!("launch-{:032x}", u128::from_ne_bytes(nonce)),
            CAPACITY,
        )
    }

    fn start(file: File, namespace: String, capacity: usize) -> anyhow::Result<Self> {
        let (log, receiver, commands) = Self::channels(namespace, capacity);
        let writer_state = Arc::clone(&log.state);
        std::thread::Builder::new()
            .name("piku-request-log".into())
            .spawn(move || write_loop(file, &receiver, &commands, &writer_state))
            .map_err(|error| anyhow::anyhow!("could not start request-log writer: {error}"))?;
        Ok(log)
    }

    fn channels(
        namespace: String,
        capacity: usize,
    ) -> (Self, mpsc::Receiver<Event>, mpsc::Receiver<Control>) {
        let state = Arc::new(State {
            accepting: AtomicBool::new(true),
            callbacks: AtomicU64::new(0),
            active: AtomicU64::new(0),
            full: AtomicU64::new(0),
            closed: AtomicU64::new(0),
            rejected: AtomicU64::new(0),
            enqueued: AtomicU64::new(0),
            drained: AtomicU64::new(0),
            failed: AtomicBool::new(false),
            origin: Instant::now(),
            namespace,
        });
        let (events, receiver) = mpsc::sync_channel(capacity);
        let (controls, commands) = mpsc::sync_channel(1);
        (
            Self {
                state,
                events,
                controls,
                finished: false,
            },
            receiver,
            commands,
        )
    }

    /// Capture this construction's name, including in old actor/background clones.
    #[must_use]
    pub fn observer(&self, provider: &str) -> Arc<InferenceObserver> {
        observer(
            self.events.clone(),
            Arc::clone(&self.state),
            provider.to_owned(),
            None,
        )
    }

    /// Project the existing writer handles into an immutable per-request factory.
    /// It copies only format-filtered opaque refs, never recorder or payload state.
    #[must_use]
    pub fn factory(&self) -> Arc<piku_runtime::RequestObserverFactory> {
        let events = self.events.clone();
        let state = Arc::clone(&self.state);
        Arc::new(move |provider, binding| {
            observer(
                events.clone(),
                Arc::clone(&state),
                provider.to_owned(),
                Some(binding_value(binding)),
            )
        })
    }

    /// Optional terminal-sink context; `RecordingSink` supplies acknowledged refs.
    #[must_use]
    pub fn request_observation(&self) -> piku_runtime::RequestObservation {
        piku_runtime::RequestObservation::new(self.factory())
    }

    /// Refuse unsupported routes before inference, without selecting a fallback.
    pub fn observe(
        &self,
        mut resolved: piku_runtime::ResolvedProvider,
    ) -> anyhow::Result<piku_runtime::ResolvedProvider> {
        resolved.provider = resolved
            .as_provider()
            .observed_clone(self.observer(resolved.name()))?;
        Ok(resolved)
    }

    /// Before exec, persist an explicitly unsettled checkpoint. An exec failure
    /// leaves the same recorder available to the continuing old process.
    pub fn checkpoint(&self) -> anyhow::Result<()> {
        self.command(false)
    }

    /// Exec currently reuses argv, which would collide with the fresh path.
    /// Refuse before binary replacement rather than overwrite or disable logging.
    pub fn prepare_exec(&self) -> anyhow::Result<()> {
        self.checkpoint()?;
        anyhow::bail!("request-log recording prevents self-update: restart explicitly with a fresh request-log path");
    }

    /// Close admission, drain accepted events within a bound, and write a cutoff.
    /// This intentionally does not certify detached producers as settled.
    pub fn finish(mut self) -> anyhow::Result<()> {
        self.state.accepting.store(false, Ordering::SeqCst);
        let result = self.command(true);
        self.finished = true;
        result
    }

    fn command(&self, finish: bool) -> anyhow::Result<()> {
        let (reply, receiver) = mpsc::sync_channel(1);
        let command = if finish {
            Control::Finish(reply)
        } else {
            Control::Checkpoint(reply)
        };
        self.controls.try_send(command).map_err(|_| {
            anyhow::anyhow!("request-log writer unavailable; recording is incomplete")
        })?;
        receiver
            .recv_timeout(FINISH_TIMEOUT)
            .map_err(|_| {
                anyhow::anyhow!("request-log writer cutoff timed out; recording is incomplete")
            })?
            .map_err(anyhow::Error::msg)
    }
}

impl Drop for RequestLog {
    fn drop(&mut self) {
        if !self.finished {
            self.state.accepting.store(false, Ordering::SeqCst);
            let (reply, _) = mpsc::sync_channel(1);
            let _ = self.controls.try_send(Control::Finish(reply));
        }
    }
}

fn offset(state: &State, instant: Instant) -> u64 {
    u64::try_from(instant.saturating_duration_since(state.origin).as_micros()).unwrap_or(u64::MAX)
}

fn event_value(state: &State, event: Event) -> Value {
    use piku_api::observation::{
        InferenceAttemptFailure as Failure, InferenceAttemptOutcome as Outcome,
    };
    let mut value = json!({ "schema": 1, "namespace": state.namespace, "provider": event.provider, "binding":event.binding });
    match event.observation {
        InferenceObservation::Started {
            attempt_id,
            started_at,
        } => {
            value["event"] = json!("started");
            value["attempt_id"] = json!(attempt_id);
            value["started_us"] = json!(offset(state, started_at));
        }
        InferenceObservation::Finished(receipt) => {
            value["event"] = json!("finished");
            value["attempt_id"] = json!(receipt.attempt_id);
            value["started_us"] = json!(offset(state, receipt.started_at));
            value["finished_us"] = json!(offset(state, receipt.finished_at));
            let (outcome, failure) = match receipt.outcome {
                Outcome::Completed => ("completed", None),
                Outcome::Dropped => ("dropped", None),
                Outcome::Failed(kind) => (
                    "failed",
                    Some(match kind {
                        Failure::Send => "send",
                        Failure::Http => "http",
                        Failure::Stream => "stream",
                        Failure::Protocol => "protocol",
                        _ => "unknown",
                    }),
                ),
                _ => ("unknown", None),
            };
            value["outcome"] = json!(outcome);
            value["failure"] = json!(failure);
            value["transport_complete"] = json!(receipt.transport_complete);
            value["reported_usage"] = receipt.reported_usage.map_or(Value::Null, |usage| json!({
                "input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens,
                "cached_input_tokens": usage.cached_input_tokens, "reasoning_output_tokens": usage.reasoning_output_tokens,
            }));
        }
        _ => {
            value["event"] = json!("unknown");
        }
    }
    value
}

fn summary(state: &State, event: &str) -> Value {
    json!({ "schema": 1, "event": event, "namespace": state.namespace,
        "cutoff_us": offset(state, Instant::now()), "active_at_cutoff": state.active.load(Ordering::SeqCst),
        "callbacks_at_cutoff": state.callbacks.load(Ordering::SeqCst),
        "queue_full": state.full.load(Ordering::SeqCst), "queue_closed": state.closed.load(Ordering::SeqCst),
        "rejected_after_cutoff": state.rejected.load(Ordering::SeqCst),
        "queued_at_cutoff": state.enqueued.load(Ordering::SeqCst).saturating_sub(state.drained.load(Ordering::SeqCst)),
        "writer_failed": state.failed.load(Ordering::SeqCst), "producers_unsettled": true })
}

fn write_value(file: &mut File, value: &Value, state: &State) -> bool {
    if state.failed.load(Ordering::SeqCst) {
        return false;
    }
    if serde_json::to_writer(&mut *file, value).is_err() || file.write_all(b"\n").is_err() {
        state.failed.store(true, Ordering::SeqCst);
        return false;
    }
    true
}

fn drain_event(file: &mut File, event: Event, state: &State) {
    write_value(file, &event_value(state, event), state);
    state.drained.fetch_add(1, Ordering::SeqCst);
}

fn write_loop(
    mut file: File,
    events: &mpsc::Receiver<Event>,
    controls: &mpsc::Receiver<Control>,
    state: &State,
) {
    write_value(
        &mut file,
        &json!({ "schema": 1, "event": "launch", "namespace": state.namespace }),
        state,
    );
    loop {
        match controls.try_recv() {
            Ok(command) => {
                let finish = matches!(command, Control::Finish(_));
                // At finish, callbacks already in flight may still enqueue. Bound
                // this wait as well as the outer caller's acknowledgement deadline.
                let deadline = Instant::now() + FINISH_TIMEOUT / 2;
                loop {
                    let mut empty = false;
                    for _ in 0..CAPACITY {
                        if let Ok(event) = events.try_recv() {
                            drain_event(&mut file, event, state);
                        } else {
                            empty = true;
                            break;
                        }
                    }
                    if !finish
                        || (empty
                            && state.callbacks.load(Ordering::SeqCst) == 0
                            && state.enqueued.load(Ordering::SeqCst)
                                == state.drained.load(Ordering::SeqCst))
                        || Instant::now() >= deadline
                    {
                        break;
                    }
                    std::thread::yield_now();
                }
                write_value(
                    &mut file,
                    &summary(state, if finish { "cutoff" } else { "checkpoint" }),
                    state,
                );
                if file.flush().is_err() {
                    state.failed.store(true, Ordering::SeqCst);
                }
                let result = if state.failed.load(Ordering::SeqCst) {
                    Err("request-log write failed; recording is incomplete".into())
                } else if state.full.load(Ordering::SeqCst) > 0
                    || state.closed.load(Ordering::SeqCst) > 0
                    || (finish
                        && (state.callbacks.load(Ordering::SeqCst) > 0
                            || state.enqueued.load(Ordering::SeqCst)
                                != state.drained.load(Ordering::SeqCst)))
                {
                    Err("request-log events were lost or unsettled at cutoff; recording is incomplete".into())
                } else {
                    Ok(())
                };
                let reply = match command {
                    Control::Finish(reply) | Control::Checkpoint(reply) => reply,
                };
                let _ = reply.try_send(result);
                if finish {
                    return;
                }
            }
            Err(mpsc::TryRecvError::Disconnected) => return,
            Err(mpsc::TryRecvError::Empty) => {}
        }
        match events.recv_timeout(Duration::from_millis(10)) {
            Ok(event) => {
                drain_event(&mut file, event, state);
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(mpsc::RecvTimeoutError::Disconnected) => return,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures_util::StreamExt;
    use piku_api::{MessageRequest, OpenAiCompatProvider, RequestContent, RequestMessage};

    fn rows(path: &Path) -> Vec<Value> {
        std::fs::read_to_string(path)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect()
    }

    #[test]
    fn fresh_files_are_private_and_launch_namespaces_are_distinct() {
        let directory = tempfile::tempdir().unwrap();
        let first = directory.path().join("first.jsonl");
        let second = directory.path().join("second.jsonl");
        RequestLog::create(&first).unwrap().finish().unwrap();
        let before = std::fs::read(&first).unwrap();
        assert!(RequestLog::create(&first).is_err());
        assert_eq!(std::fs::read(&first).unwrap(), before);
        RequestLog::create(&second).unwrap().finish().unwrap();
        assert_ne!(rows(&first)[0]["namespace"], rows(&second)[0]["namespace"]);
        assert_eq!(rows(&first).last().unwrap()["event"], "cutoff");
        assert_eq!(rows(&first).last().unwrap()["producers_unsettled"], true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::{symlink, PermissionsExt};
            assert_eq!(
                std::fs::metadata(&first).unwrap().permissions().mode() & 0o777,
                0o600
            );
            let alias = directory.path().join("alias.jsonl");
            symlink(&first, &alias).unwrap();
            assert!(RequestLog::create(&alias).is_err());
            assert_eq!(std::fs::read(&first).unwrap(), before);
        }
    }

    #[test]
    fn checkpoint_refuses_exec_and_preserves_provider_names_in_late_clones() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("checkpoint.jsonl");
        let log = RequestLog::create(&path).unwrap();
        let old_actor = log.observer("old-provider");
        let next_actor = log.observer("next-provider");
        next_actor(InferenceObservation::Started {
            attempt_id: 1,
            started_at: Instant::now(),
        });
        assert!(log
            .prepare_exec()
            .unwrap_err()
            .to_string()
            .contains("restart explicitly"));
        let checkpoint = rows(&path).pop().unwrap();
        assert_eq!(checkpoint["event"], "checkpoint");
        assert_eq!(checkpoint["producers_unsettled"], true);
        // A failed/refused exec does not disable the continuing launch's sink.
        old_actor(InferenceObservation::Started {
            attempt_id: 2,
            started_at: Instant::now(),
        });
        log.finish().unwrap();
        let records = rows(&path);
        let old = records.iter().find(|row| row["attempt_id"] == 2).unwrap();
        assert_eq!(old["provider"], "old-provider");
        assert_eq!(records.last().unwrap()["active_at_cutoff"], 2);
        assert_eq!(records.last().unwrap()["queued_at_cutoff"], 0);
    }

    #[test]
    fn unsupported_observation_refuses_before_inference() {
        struct Unsupported;
        impl piku_api::Provider for Unsupported {
            fn name(&self) -> &'static str {
                "unsupported"
            }
            fn stream_message(
                &self,
                _: MessageRequest,
            ) -> std::pin::Pin<
                Box<
                    dyn futures_util::Stream<Item = Result<piku_api::Event, piku_api::ApiError>>
                        + Send
                        + '_,
                >,
            > {
                panic!("unsupported observation must refuse before inference")
            }
        }
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("refused.jsonl");
        let log = RequestLog::create(&path).unwrap();
        let resolved = piku_runtime::ResolvedProvider {
            provider: Box::new(Unsupported),
            default_model: "unused".into(),
        };
        assert!(log
            .observe(resolved)
            .err()
            .unwrap()
            .to_string()
            .contains("unsupported"));
        log.finish().unwrap();
        assert_eq!(rows(&path).len(), 2);
        assert_eq!(rows(&path)[1]["active_at_cutoff"], 0);
    }

    #[test]
    fn real_callback_reports_full_and_closed_queues_without_waiting() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("loss.jsonl");
        let file = File::create(&path).unwrap();
        // Hold the consumer until the callback fills the actual bounded queue.
        let (log, receiver, commands) = RequestLog::channels("test-loss".into(), 1);
        let observer = log.observer("custom");
        let before = Instant::now();
        for attempt_id in 1..=3 {
            observer(InferenceObservation::Started {
                attempt_id,
                started_at: Instant::now(),
            });
        }
        assert!(before.elapsed() < Duration::from_secs(1));
        let state = Arc::clone(&log.state);
        let writer = std::thread::spawn(move || write_loop(file, &receiver, &commands, &state));
        assert!(log.finish().unwrap_err().to_string().contains("lost"));
        writer.join().unwrap();
        let cutoff = rows(&path).pop().unwrap();
        assert_eq!(cutoff["queue_full"], 2);
        assert_eq!(cutoff["active_at_cutoff"], 3);
        // A retained provider callback after cutoff remains callable but rejected.
        observer(InferenceObservation::Started {
            attempt_id: 4,
            started_at: Instant::now(),
        });
        let (closed, receiver, commands) = RequestLog::channels("test-closed".into(), 1);
        drop(receiver);
        drop(commands);
        closed.observer("custom")(InferenceObservation::Started {
            attempt_id: 1,
            started_at: Instant::now(),
        });
        assert_eq!(closed.state.closed.load(Ordering::SeqCst), 1);
        assert!(closed.finish().is_err());
    }

    #[test]
    fn write_failure_and_unresponsive_writer_are_recording_errors() {
        let file = tempfile::NamedTempFile::new().unwrap();
        let read_only = File::open(file.path()).unwrap();
        let log = RequestLog::start(read_only, "test-io".into(), CAPACITY).unwrap();
        assert!(log
            .finish()
            .unwrap_err()
            .to_string()
            .contains("write failed"));
        let (unresponsive, receiver, commands) = RequestLog::channels("test-timeout".into(), 1);
        let before = Instant::now();
        assert!(unresponsive
            .finish()
            .unwrap_err()
            .to_string()
            .contains("timed out"));
        assert!(before.elapsed() < FINISH_TIMEOUT + Duration::from_secs(1));
        drop((receiver, commands));
    }

    async fn transport(body: String) -> (String, tokio::task::JoinHandle<String>) {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut bytes = Vec::new();
            let header_end;
            loop {
                let mut buffer = [0; 1024];
                let count = socket.read(&mut buffer).await.unwrap();
                assert!(count > 0);
                bytes.extend_from_slice(&buffer[..count]);
                if let Some(index) = bytes.windows(4).position(|slice| slice == b"\r\n\r\n") {
                    header_end = index + 4;
                    break;
                }
            }
            let headers = String::from_utf8(bytes[..header_end].to_vec()).unwrap();
            let length = headers
                .lines()
                .find_map(|line| {
                    line.to_ascii_lowercase()
                        .strip_prefix("content-length:")
                        .map(|value| value.trim().parse::<usize>().unwrap())
                })
                .unwrap();
            while bytes.len() < header_end + length {
                let mut buffer = [0; 1024];
                let count = socket.read(&mut buffer).await.unwrap();
                assert!(count > 0);
                bytes.extend_from_slice(&buffer[..count]);
            }
            let response = format!("HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}", body.len());
            socket.write_all(response.as_bytes()).await.unwrap();
            String::from_utf8(bytes).unwrap()
        });
        (
            format!("http://{address}/private-endpoint-sentinel"),
            server,
        )
    }

    #[tokio::test]
    async fn actual_transport_preserves_usage_outcomes_and_excludes_payloads() {
        for (label, body, dropped, expected_outcome) in [
            ("reported", "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":17,\"completion_tokens\":3,\"prompt_tokens_details\":{\"cached_tokens\":7}}}\n\ndata: [DONE]\n\n", false, "completed"),
            ("unknown", "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n", false, "completed"),
            ("protocol", "data: invalid-response-body-sentinel\n\n", false, "failed"),
            ("dropped", "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"private-response-sentinel\"}}]}\n\n", true, "dropped"),
        ] {
            let directory = tempfile::tempdir().unwrap();
            let path = directory.path().join("log.jsonl");
            let log = RequestLog::create(&path).unwrap();
            let (url, server) = transport(body.into()).await;
            let provider = OpenAiCompatProvider::new("custom", url, piku_api::openai_compat::Auth::Bearer("private-auth-sentinel".into()), vec![]);
            let resolved = log.observe(piku_runtime::ResolvedProvider::compat(provider, "private-model-sentinel")).unwrap();
            let clone = if label == "unknown" {
                // Keep the legacy provider-bound API explicitly unbound.
                resolved.as_provider().boxed_clone()
            } else {
                let mut binding = piku_runtime::CausalBinding::default();
                binding.run_id = Some("session-123456789-7".into());
                binding.turn_id = Some("turn-abcdef0123456789abcdef0123456789".into());
                binding.context_sequence = Some(37);
                binding.task_id = Some("agent-7-abc-123def".into());
                let mut parent = piku_runtime::CausalOrigin::default();
                parent.run_id = Some("session-987654321-7".into());
                parent.turn_id = Some("turn-1-0".into());
                binding.parent_origin = Some(parent);
                if label == "protocol" {
                    binding.run_id = Some("private-run-sentinel".into());
                    binding.turn_id = Some("private-turn-sentinel".into());
                    binding.task_id = Some("private-task-sentinel".into());
                    let mut parent = piku_runtime::CausalOrigin::default();
                    parent.run_id = Some("private-parent-sentinel".into());
                    parent.turn_id = Some("private-parent-turn-sentinel".into());
                    binding.parent_origin = Some(parent);
                }
                let callback = (log.factory())(resolved.name(), binding.clone());
                // Caller advances before polling; the callback already owns
                // both receipt halves' safe metadata and route name.
                binding.run_id = Some("private-later-run-sentinel".into());
                binding.turn_id = Some("private-later-turn-sentinel".into());
                assert_eq!(binding.run_id.as_deref(), Some("private-later-run-sentinel"));
                assert_eq!(binding.turn_id.as_deref(), Some("private-later-turn-sentinel"));
                resolved.as_provider().observed_clone(callback).unwrap()
            };
            let request = MessageRequest {
                model: "private-model-sentinel".into(), max_tokens: 10,
                messages: vec![RequestMessage { role: "user".into(), content: vec![RequestContent::Text { text: "private-prompt-sentinel".into() }] }],
                system: None, tools: None, stream: true,
            };
            let mut stream = clone.stream_message(request);
            if dropped {
                assert!(tokio::time::timeout(Duration::from_secs(5), stream.next()).await.unwrap().unwrap().is_ok());
            } else {
                tokio::time::timeout(Duration::from_secs(5), async { while stream.next().await.is_some() {} }).await.unwrap();
            }
            drop(stream);
            let received = tokio::time::timeout(Duration::from_secs(5), server).await.unwrap().unwrap();
            assert!(received.contains("private-prompt-sentinel"));
            assert!(received.contains("private-auth-sentinel"));
            log.finish().unwrap();
            let records = rows(&path);
            let started = &records[1];
            let finished = &records[2];
            assert_eq!(started["event"], "started");
            assert_eq!(finished["event"], "finished");
            assert_eq!(started["attempt_id"], finished["attempt_id"]);
            assert_eq!(started["namespace"], finished["namespace"]);
            assert_eq!(started["binding"], finished["binding"]);
            if label == "unknown" {
                assert!(started["binding"].is_null());
            } else if label == "protocol" {
                assert_eq!(started["binding"], json!({"run_id":null,"turn_id":null,
                    "context_sequence":null,"task_id":null,"parent_origin":null,"purpose":"agent_turn"}));
            } else {
                assert_eq!(started["binding"], json!({"run_id":"session-123456789-7",
                    "turn_id":"turn-abcdef0123456789abcdef0123456789","context_sequence":37,
                    "task_id":"agent-7-abc-123def","parent_origin":{"run_id":"session-987654321-7",
                    "turn_id":"turn-1-0"},"purpose":"agent_turn"}));
            }
            assert_eq!(finished["provider"], "custom");
            assert!(started["attempt_id"].as_u64().unwrap() > 0);
            assert!(finished["finished_us"].as_u64().unwrap() >= finished["started_us"].as_u64().unwrap());
            assert_eq!(finished["outcome"], expected_outcome, "{label}");
            assert_eq!(finished["transport_complete"], !dropped && label != "protocol");
            if label == "reported" {
                assert_eq!(finished["reported_usage"], json!({"input_tokens":17,"output_tokens":3,"cached_input_tokens":7,"reasoning_output_tokens":null}));
            } else { assert!(finished["reported_usage"].is_null()); }
            if label == "protocol" { assert_eq!(finished["failure"], "protocol"); }
            assert_eq!(records.last().unwrap()["active_at_cutoff"], 0);
            let text = std::fs::read_to_string(path).unwrap();
            assert!(!text.contains("private-"));
            assert!(!text.contains("invalid-response-body-sentinel"));
        }
    }
}
