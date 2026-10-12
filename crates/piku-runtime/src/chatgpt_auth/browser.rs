//! A short-lived browser bootstrap, separate from the credential refresh lock.
use super::{
    now, open_protected, read_protected, write_protected, AccountInfo, AuthError, AuthManager,
    StoreLock,
};
use serde::{Deserialize, Serialize};
use std::{fmt, time::Duration};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio::net::TcpStream;
use uuid::Uuid;

const LIFETIME: Duration = Duration::from_secs(180);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LoginPhase {
    WaitingForBrowser,
    SigningIn,
    Completed,
    Cancelled,
    Expired,
    Failed,
}
impl LoginPhase {
    fn active(self) -> bool {
        matches!(self, Self::WaitingForBrowser | Self::SigningIn)
    }
}

/// Safe progress metadata; never contains callback state or authorization URLs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoginStatus {
    pub attempt_id: String,
    pub phase: LoginPhase,
    pub expires_at: u64,
    pub registration_id: Option<String>,
}
#[derive(Serialize, Deserialize)]
struct Cancellation {
    attempt_id: String,
}

/// Owns a single bounded attempt. Dropping it cancels the attempt and unlocks it.
pub struct BrowserLogin {
    manager: AuthManager,
    listener: Option<TcpListener>,
    bootstrap_url: String,
    form_nonce: String,
    registration: Option<String>,
    request_consent: bool,
    status: LoginStatus,
    pub(super) deadline: tokio::time::Instant,
    _lock: StoreLock,
}
impl fmt::Debug for BrowserLogin {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BrowserLogin")
            .field("status", &self.status)
            .finish_non_exhaustive()
    }
}
impl Drop for BrowserLogin {
    fn drop(&mut self) {
        if self.status.phase.active() {
            self.status.phase = LoginPhase::Cancelled;
            let _ = self.persist();
        }
    }
}

impl AuthManager {
    /// Prepare a local handoff without contacting the identity provider.
    pub async fn prepare_browser_login(
        &self,
        registration: Option<String>,
        request_consent: bool,
    ) -> Result<BrowserLogin, AuthError> {
        let file = open_protected(&self.owner.directory.join(".login.lock"), true)?;
        file.try_lock().map_err(|_| AuthError::RetryLater)?;
        let lock = StoreLock(file);
        if let Some(id) = &registration {
            self.read(id)?;
        }
        let listener = TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0))
            .await
            .map_err(|_| AuthError::RetryLater)?;
        let bootstrap_url = format!(
            "http://127.0.0.1:{}/login/{}",
            listener
                .local_addr()
                .map_err(|_| AuthError::RetryLater)?
                .port(),
            Uuid::new_v4()
        );
        let attempt = BrowserLogin {
            manager: self.clone(),
            listener: Some(listener),
            bootstrap_url,
            form_nonce: Uuid::new_v4().to_string(),
            registration,
            request_consent,
            status: LoginStatus {
                attempt_id: Uuid::new_v4().to_string(),
                phase: LoginPhase::WaitingForBrowser,
                expires_at: now()?.saturating_add(LIFETIME.as_secs()),
                registration_id: None,
            },
            deadline: tokio::time::Instant::now() + LIFETIME,
            _lock: lock,
        };
        attempt.persist()?;
        Ok(attempt)
    }

    /// Read progress without touching credentials, selecting an account, or doing network IO.
    pub fn login_status(&self) -> Result<Option<LoginStatus>, AuthError> {
        let path = self.owner.directory.join("login-status.json");
        if !path.try_exists().map_err(|_| AuthError::Storage)? {
            return Ok(None);
        }
        let mut status: LoginStatus = read_protected(&path)?;
        if status.phase.active() && now()? >= status.expires_at {
            status.phase = LoginPhase::Expired;
        }
        Ok(Some(status))
    }

    /// Request cancellation of the currently observed attempt only.
    pub fn cancel_login(&self) -> Result<bool, AuthError> {
        let Some(status) = self.login_status()? else {
            return Ok(false);
        };
        if !status.phase.active() {
            return Ok(false);
        }
        write_protected(
            &self.owner.directory.join("login-cancel.json"),
            &Cancellation {
                attempt_id: status.attempt_id,
            },
        )?;
        Ok(true)
    }
}

impl BrowserLogin {
    /// A local, short-lived capability URL. It contains no OAuth state or identity hint.
    #[must_use]
    pub fn bootstrap_url(&self) -> &str {
        &self.bootstrap_url
    }
    pub async fn run(mut self) -> Result<AccountInfo, AuthError> {
        let manager = self.manager.clone();
        let id = self.status.attempt_id.clone();
        let deadline = self.deadline;
        let result = tokio::select! {
            biased;
            error = wait_cancel(&manager, &id) => Err(error),
            () = tokio::time::sleep_until(deadline) => Err(AuthError::TimedOut),
            result = self.serve() => result,
        };
        self.status.registration_id = result
            .as_ref()
            .ok()
            .map(|info| info.registration_id.clone());
        self.status.phase = match &result {
            Ok(_) => LoginPhase::Completed,
            Err(AuthError::Cancelled) => LoginPhase::Cancelled,
            Err(AuthError::TimedOut) => LoginPhase::Expired,
            Err(_) => LoginPhase::Failed,
        };
        self.persist()?;
        result
    }
    fn persist(&self) -> Result<(), AuthError> {
        write_protected(
            &self.manager.owner.directory.join("login-status.json"),
            &self.status,
        )
    }
    async fn serve(&mut self) -> Result<AccountInfo, AuthError> {
        let expected = reqwest::Url::parse(&self.bootstrap_url).map_err(|_| AuthError::Protocol)?;
        let listener = self.listener.take().ok_or(AuthError::Protocol)?;
        loop {
            let (mut socket, _) = listener.accept().await.map_err(|_| AuthError::RetryLater)?;
            let Ok(Ok(request)) =
                tokio::time::timeout(Duration::from_secs(3), read_request(&mut socket)).await
            else {
                let _ = reply(&mut socket, "400 Bad Request", "", "Invalid request.").await;
                continue;
            };
            if request.host != expected.authority() || request.path != expected.path() {
                reply(&mut socket, "404 Not Found", "", "Not found.").await?;
                continue;
            }
            if request.method == "GET" {
                let body = format!("<!doctype html><meta charset=\"utf-8\"><title>Sign in to piku</title><h1>Sign in to piku</h1><p>Continue to ChatGPT to authorize this sign-in. No account is selected automatically.</p><form method=\"post\" action=\"{}\"><input type=\"hidden\" name=\"confirm\" value=\"{}\"><button type=\"submit\">Continue</button></form>", expected.path(), self.form_nonce);
                reply(&mut socket, "200 OK", "", &body).await?;
                continue;
            }
            if request.method != "POST"
                || request.origin.as_deref()
                    != Some(expected.origin().ascii_serialization().as_str())
                || request.content_type.as_deref() != Some("application/x-www-form-urlencoded")
                || request.body != format!("confirm={}", self.form_nonce)
            {
                reply(&mut socket, "403 Forbidden", "", "Confirmation required.").await?;
                continue;
            }
            // Consume the capability before starting OAuth; replays cannot initiate another attempt.
            drop(listener);
            self.status.phase = LoginPhase::SigningIn;
            self.persist()?;
            let pending = self
                .manager
                .begin_login_with_hint(self.registration.as_deref(), self.request_consent, false)
                .await?;
            let header = format!("Location: {}\r\n", pending.authorization_url());
            reply(
                &mut socket,
                "303 See Other",
                &header,
                "Continue in your browser.",
            )
            .await?;
            return self.manager.complete_login(pending).await;
        }
    }
}

async fn wait_cancel(manager: &AuthManager, id: &str) -> AuthError {
    let path = manager.owner.directory.join("login-cancel.json");
    loop {
        match path.try_exists() {
            Ok(true) => match read_protected::<Cancellation>(&path) {
                Ok(cancel) if cancel.attempt_id == id => return AuthError::Cancelled,
                Ok(_) => {}
                Err(error) => return error,
            },
            Ok(false) => {}
            Err(_) => return AuthError::Storage,
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}
struct Request {
    method: String,
    path: String,
    host: String,
    origin: Option<String>,
    content_type: Option<String>,
    body: String,
}
async fn read_request(socket: &mut TcpStream) -> Result<Request, AuthError> {
    let mut bytes = Vec::new();
    while !bytes.ends_with(b"\r\n\r\n") {
        if bytes.len() >= 8192 {
            return Err(AuthError::InvalidCallback);
        }
        bytes.push(
            socket
                .read_u8()
                .await
                .map_err(|_| AuthError::InvalidCallback)?,
        );
    }
    let text = std::str::from_utf8(&bytes).map_err(|_| AuthError::InvalidCallback)?;
    let mut lines = text.lines();
    let mut first = lines
        .next()
        .ok_or(AuthError::InvalidCallback)?
        .split_whitespace();
    let method = first.next().ok_or(AuthError::InvalidCallback)?.to_owned();
    let path = first.next().ok_or(AuthError::InvalidCallback)?.to_owned();
    if first.next() != Some("HTTP/1.1") || first.next().is_some() {
        return Err(AuthError::InvalidCallback);
    }
    let mut headers = std::collections::HashMap::new();
    for line in lines.filter(|line| !line.is_empty()) {
        let (key, value) = line.split_once(':').ok_or(AuthError::InvalidCallback)?;
        if headers
            .insert(key.to_ascii_lowercase(), value.trim().to_owned())
            .is_some()
        {
            return Err(AuthError::InvalidCallback);
        }
    }
    if headers.contains_key("transfer-encoding") {
        return Err(AuthError::InvalidCallback);
    }
    let length = headers
        .get("content-length")
        .map(|v| v.parse::<usize>())
        .transpose()
        .map_err(|_| AuthError::InvalidCallback)?
        .unwrap_or(0);
    if length > 1024 {
        return Err(AuthError::InvalidCallback);
    }
    let mut body = vec![0; length];
    socket
        .read_exact(&mut body)
        .await
        .map_err(|_| AuthError::InvalidCallback)?;
    Ok(Request {
        method,
        path,
        host: headers.remove("host").ok_or(AuthError::InvalidCallback)?,
        origin: headers.remove("origin"),
        content_type: headers.remove("content-type"),
        body: String::from_utf8(body).map_err(|_| AuthError::InvalidCallback)?,
    })
}
async fn reply(
    socket: &mut TcpStream,
    status: &str,
    headers: &str,
    body: &str,
) -> Result<(), AuthError> {
    let response = format!("HTTP/1.1 {status}\r\n{headers}Content-Type: text/html; charset=utf-8\r\nCache-Control: no-store\r\nReferrer-Policy: no-referrer\r\nX-Frame-Options: DENY\r\nContent-Security-Policy: default-src 'none'; form-action 'self' https://auth.openai.com; frame-ancestors 'none'\r\nConnection: close\r\nContent-Length: {}\r\n\r\n{body}", body.len());
    socket
        .write_all(response.as_bytes())
        .await
        .map_err(|_| AuthError::RetryLater)?;
    socket.shutdown().await.map_err(|_| AuthError::RetryLater)
}
