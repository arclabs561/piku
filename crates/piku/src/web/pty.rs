use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use futures_util::{SinkExt, StreamExt};
use portable_pty::{native_pty_system, CommandBuilder, MasterPty, PtySize};
use serde::Deserialize;
use tokio::sync::{mpsc, OwnedSemaphorePermit};

use super::AppState;

const MAX_INPUT_BYTES: usize = 64 * 1024;
const MAX_COLS: u16 = 500;
const MAX_ROWS: u16 = 200;
const OUTPUT_QUEUE_CHUNKS: usize = 64;

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum PtyControl {
    Resize { cols: u16, rows: u16 },
}

struct OpenPty {
    master: Arc<Mutex<Box<dyn MasterPty + Send>>>,
    reader: Box<dyn Read + Send>,
    writer: Arc<Mutex<Box<dyn Write + Send>>>,
    child: Box<dyn portable_pty::Child + Send + Sync>,
}

pub(super) async fn terminal_pty_handler(
    ws: WebSocketUpgrade,
    State(state): State<Arc<AppState>>,
) -> Response {
    let Ok(permit) = Arc::clone(&state.terminal_slots).try_acquire_owned() else {
        return (
            StatusCode::TOO_MANY_REQUESTS,
            "terminal session limit reached",
        )
            .into_response();
    };
    let root = state.workspace_root.as_ref().clone();
    let terminal_id = crate::new_session_id();
    tracing::info!(
        terminal_id,
        kind = "human_terminal",
        shell = %configured_shell().display(),
        "terminal opened"
    );
    ws.on_upgrade(move |socket| run_terminal(socket, root, permit, terminal_id))
        .into_response()
}

async fn run_terminal(
    mut socket: WebSocket,
    root: PathBuf,
    _permit: OwnedSemaphorePermit,
    terminal_id: String,
) {
    let started = Instant::now();
    let opened = match tokio::task::spawn_blocking(move || open_pty(&root)).await {
        Ok(Ok(opened)) => opened,
        Ok(Err(error)) => {
            tracing::error!(terminal_id, kind = "human_terminal", %error, "terminal failed");
            let _ = socket
                .send(Message::text(format!("piku: terminal failed: {error}\r\n")))
                .await;
            return;
        }
        Err(error) => {
            tracing::error!(terminal_id, kind = "human_terminal", worker_error = %error, "terminal worker failed");
            let _ = socket
                .send(Message::text(format!(
                    "piku: terminal worker failed: {error}\r\n"
                )))
                .await;
            return;
        }
    };

    let (mut socket_tx, mut socket_rx) = socket.split();
    let (output_tx, mut output_rx) = mpsc::channel::<Vec<u8>>(OUTPUT_QUEUE_CHUNKS);
    let mut reader = opened.reader;
    std::thread::spawn(move || {
        let mut buffer = [0_u8; 8192];
        loop {
            match reader.read(&mut buffer) {
                Ok(0) | Err(_) => break,
                Ok(read) => {
                    if output_tx.blocking_send(buffer[..read].to_vec()).is_err() {
                        break;
                    }
                }
            }
        }
    });

    let mut output_task = tokio::spawn(async move {
        while let Some(bytes) = output_rx.recv().await {
            if socket_tx.send(Message::binary(bytes)).await.is_err() {
                break;
            }
        }
    });

    let writer = Arc::clone(&opened.writer);
    let master = Arc::clone(&opened.master);
    let input_task = async move {
        while let Some(Ok(message)) = socket_rx.next().await {
            match message {
                Message::Binary(bytes) if bytes.len() <= MAX_INPUT_BYTES => {
                    let write_result = writer
                        .lock()
                        .map_err(|_| ())
                        .and_then(|mut writer| writer.write_all(&bytes).map_err(|_| ()));
                    if write_result.is_err() {
                        break;
                    }
                }
                Message::Text(text) => {
                    let Ok(PtyControl::Resize { cols, rows }) = serde_json::from_str(&text) else {
                        continue;
                    };
                    if cols == 0 || rows == 0 || cols > MAX_COLS || rows > MAX_ROWS {
                        continue;
                    }
                    let _ = master.lock().ok().and_then(|master| {
                        master
                            .resize(PtySize {
                                rows,
                                cols,
                                pixel_width: 0,
                                pixel_height: 0,
                            })
                            .ok()
                    });
                }
                Message::Close(_) => break,
                Message::Ping(_) | Message::Pong(_) | Message::Binary(_) => {}
            }
        }
    };

    tokio::select! {
        () = input_task => {}
        _ = &mut output_task => {}
    }

    let mut killer = opened.child.clone_killer();
    let _ = killer.kill();
    drop(opened.writer);
    if !output_task.is_finished() {
        output_task.abort();
    }
    let mut child = opened.child;
    let _ = tokio::task::spawn_blocking(move || child.wait()).await;
    tracing::info!(
        terminal_id,
        kind = "human_terminal",
        elapsed_seconds = started.elapsed().as_secs_f32(),
        "terminal closed"
    );
}

fn open_pty(root: &Path) -> anyhow::Result<OpenPty> {
    let pair = native_pty_system().openpty(PtySize {
        rows: 24,
        cols: 80,
        pixel_width: 0,
        pixel_height: 0,
    })?;
    let shell = configured_shell();
    let mut command = CommandBuilder::new(&shell);
    command.arg("-l");
    command.cwd(root);
    command.env("TERM", "xterm-256color");
    command.env("COLORTERM", "truecolor");
    command.env("PIKU_WEB_TERMINAL", "1");
    let child = pair.slave.spawn_command(command)?;
    drop(pair.slave);
    let reader = pair.master.try_clone_reader()?;
    let writer = pair.master.take_writer()?;
    Ok(OpenPty {
        master: Arc::new(Mutex::new(pair.master)),
        reader,
        writer: Arc::new(Mutex::new(writer)),
        child,
    })
}

fn configured_shell() -> PathBuf {
    std::env::var_os("SHELL")
        .map(PathBuf::from)
        .filter(|path| path.is_absolute() && path.is_file())
        .unwrap_or_else(|| {
            ["/bin/zsh", "/bin/sh"]
                .into_iter()
                .map(PathBuf::from)
                .find(|path| path.is_file())
                .unwrap_or_else(|| PathBuf::from("/bin/sh"))
        })
}

#[cfg(test)]
mod tests {
    use super::{configured_shell, PtyControl, MAX_COLS, MAX_ROWS};

    #[test]
    fn resize_control_is_typed_and_bounded() {
        let control: PtyControl = serde_json::from_str(r#"{"type":"resize","cols":120,"rows":40}"#)
            .expect("resize control parses");
        let PtyControl::Resize { cols, rows } = control;
        assert_eq!((cols, rows), (120, 40));
        assert!(cols <= MAX_COLS);
        assert!(rows <= MAX_ROWS);
    }

    #[test]
    fn configured_shell_is_an_absolute_file() {
        let shell = configured_shell();
        assert!(shell.is_absolute());
        assert!(shell.is_file());
    }
}
