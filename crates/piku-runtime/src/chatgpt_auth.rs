//! Account-bound Sign in with `ChatGPT` credentials. Tokens never enter configuration.
use std::fmt;
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use openidconnect::core::{
    CoreAuthenticationFlow, CoreClient, CoreProviderMetadata, CoreTokenResponse,
};
use openidconnect::{
    AccessTokenHash, AuthUrl, AuthorizationCode, ClientId, CsrfToken, IssuerUrl, Nonce,
    OAuth2TokenResponse, PkceCodeChallenge, PkceCodeVerifier, RedirectUrl, RefreshToken, Scope,
    TokenResponse, TokenUrl,
};
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use uuid::Uuid;

mod browser;
pub use browser::{BrowserLogin, LoginPhase, LoginStatus};

const DIRECT_SCOPE: &str = "chatgpt.tokens.use.direct";
const RESOURCE: &str = "https://api.openai.com/v1";
const CALLBACK_PATH: &str = "/auth/callback";
const REFRESH_MARGIN: u64 = 60;

/// User-level credential root; project settings cannot redirect it.
pub fn default_directory() -> Result<PathBuf, AuthError> {
    let base = std::env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".config")))
        .filter(|path| path.is_absolute())
        .ok_or(AuthError::Storage)?;
    Ok(base.join("piku/auth/chatgpt"))
}

/// Sanitized outcomes; upstream bodies and credential-bearing URLs are never included.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuthError {
    Storage,
    UnknownRegistration,
    InvalidCallback,
    Declined,
    TimedOut,
    Cancelled,
    InvalidIdentity,
    PlanPermissionMissing,
    SignInRequired,
    InvalidClient,
    RetryLater,
    Protocol,
    ChangedRegistration,
}
impl fmt::Display for AuthError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Storage => "protected ChatGPT credential storage unavailable",
            Self::UnknownRegistration => "unknown ChatGPT registration",
            Self::InvalidCallback => "invalid ChatGPT authorization callback",
            Self::Declined => "ChatGPT sign-in was declined",
            Self::TimedOut => "ChatGPT sign-in timed out",
            Self::Cancelled => "ChatGPT sign-in was cancelled",
            Self::InvalidIdentity => "ChatGPT identity validation failed",
            Self::PlanPermissionMissing => "ChatGPT plan permission is not granted",
            Self::SignInRequired => "ChatGPT sign-in is required",
            Self::InvalidClient => "ChatGPT client registration is invalid",
            Self::RetryLater => "ChatGPT authentication temporarily unavailable; retry later",
            Self::Protocol => "invalid ChatGPT authentication response",
            Self::ChangedRegistration => "ChatGPT registration changed during sign-in; retry",
        })
    }
}
impl std::error::Error for AuthError {}

/// Explicit secret handoff to transport only. No serialization or revealing Debug.
pub struct CredentialToken(String);
impl CredentialToken {
    #[must_use]
    pub fn expose_secret(&self) -> &str {
        &self.0
    }
}
impl fmt::Debug for CredentialToken {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("CredentialToken([redacted])")
    }
}

/// Private account-picker data, never a token-bearing diagnostic object.
#[derive(Clone)]
pub struct AccountInfo {
    pub registration_id: String,
    pub client_id: String,
    pub issuer: Option<String>,
    pub subject: Option<String>,
    pub email: Option<String>,
    pub signed_in: bool,
    pub plan_enabled: bool,
    pub expires_at: Option<u64>,
}
impl fmt::Debug for AccountInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AccountInfo")
            .field("registration_id", &self.registration_id)
            .field("signed_in", &self.signed_in)
            .field("plan_enabled", &self.plan_enabled)
            .finish_non_exhaustive()
    }
}

#[derive(Serialize, Deserialize, Clone, PartialEq, Eq)]
struct Identity {
    issuer: String,
    subject: String,
    email: Option<String>,
}
#[derive(Serialize, Deserialize, Clone, PartialEq, Eq)]
struct Tokens {
    access_token: String,
    refresh_token: Option<String>,
    id_token: String,
    nonce: String,
    scopes: Vec<String>,
    expires_at: u64,
}
#[derive(Serialize, Deserialize, Clone, PartialEq, Eq)]
struct Registration {
    version: u32,
    id: String,
    client_id: String,
    host_id: String,
    identity: Option<Identity>,
    tokens: Option<Tokens>,
}
impl Registration {
    fn info(&self) -> AccountInfo {
        AccountInfo {
            registration_id: self.id.clone(),
            client_id: self.client_id.clone(),
            issuer: self.identity.as_ref().map(|i| i.issuer.clone()),
            subject: self.identity.as_ref().map(|i| i.subject.clone()),
            email: self.identity.as_ref().and_then(|i| i.email.clone()),
            signed_in: self.identity.is_some() && self.tokens.is_some(),
            plan_enabled: self
                .tokens
                .as_ref()
                .is_some_and(|t| t.scopes.iter().any(|s| s == DIRECT_SCOPE)),
            expires_at: self.tokens.as_ref().map(|t| t.expires_at),
        }
    }
}
#[derive(Serialize, Deserialize)]
struct Host {
    id: String,
}
#[derive(Serialize, Deserialize)]
struct Selected {
    id: Option<String>,
}

#[derive(Clone)]
struct Endpoints {
    issuer: String,
    authorize: String,
    token: String,
}
impl Default for Endpoints {
    fn default() -> Self {
        Self {
            issuer: "https://auth.openai.com".into(),
            authorize: "https://auth.openai.com/api/accounts/authorize".into(),
            token: "https://auth.openai.com/api/accounts/oauth/token".into(),
        }
    }
}
struct Owner {
    directory: PathBuf,
    http: reqwest::Client,
    endpoints: Endpoints,
    gate: tokio::sync::Mutex<()>,
}

/// Clones share coordination; every token lease selects an immutable registration.
#[derive(Clone)]
pub struct AuthManager {
    owner: Arc<Owner>,
}
impl fmt::Debug for AuthManager {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("AuthManager([protected])")
    }
}

/// An attempt owns its listener and fresh PKCE/state/nonce. Dropping it cancels login.
pub struct PendingLogin {
    owner: Arc<Owner>,
    listener: TcpListener,
    authorization_url: String,
    redirect_uri: String,
    state: CsrfToken,
    nonce: Nonce,
    verifier: PkceCodeVerifier,
    metadata: CoreProviderMetadata,
    registration: Registration,
    expected: Option<Registration>,
}
impl PendingLogin {
    /// For explicit browser handoff only; returning sign-in can contain an ID-token hint.
    #[must_use]
    pub fn authorization_url(&self) -> &str {
        &self.authorization_url
    }
}
impl fmt::Debug for PendingLogin {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("PendingLogin([redacted])")
    }
}

#[derive(Clone)]
pub struct BoundAuth {
    manager: AuthManager,
    id: String,
}
impl fmt::Debug for BoundAuth {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BoundAuth")
            .field("registration_id", &self.id)
            .finish_non_exhaustive()
    }
}
#[derive(Debug, Clone, Copy)]
pub struct LogoutOutcome {
    pub remote_revocation_confirmed: bool,
}

impl AuthManager {
    pub fn open(directory: PathBuf) -> Result<Self, AuthError> {
        Self::with_endpoints(directory, Endpoints::default())
    }
    fn with_endpoints(
        directory: impl AsRef<Path>,
        endpoints: Endpoints,
    ) -> Result<Self, AuthError> {
        let directory = directory.as_ref();
        protected_directory(directory)?;
        let directory = directory.canonicalize().map_err(|_| AuthError::Storage)?;
        let http = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .connect_timeout(Duration::from_secs(5))
            .timeout(Duration::from_secs(15))
            .build()
            .map_err(|_| AuthError::Protocol)?;
        Ok(Self {
            owner: Arc::new(Owner {
                directory,
                endpoints,
                http,
                gate: tokio::sync::Mutex::new(()),
            }),
        })
    }
    async fn lock(&self) -> Result<(tokio::sync::MutexGuard<'_, ()>, StoreLock), AuthError> {
        let guard = tokio::time::timeout(Duration::from_secs(40), self.owner.gate.lock())
            .await
            .map_err(|_| AuthError::RetryLater)?;
        let lock = StoreLock::acquire(&self.owner.directory.join(".lock")).await?;
        Ok((guard, lock))
    }
    fn path(&self, id: &str) -> Result<PathBuf, AuthError> {
        let id = Uuid::parse_str(id).map_err(|_| AuthError::UnknownRegistration)?;
        Ok(self.owner.directory.join(format!("registration-{id}.json")))
    }
    fn read(&self, id: &str) -> Result<Registration, AuthError> {
        let record: Registration = read_protected(&self.path(id)?)?;
        if record.version != 1 || record.id != id || record.client_id == "dynamic_agent_client" {
            return Err(AuthError::Storage);
        }
        Ok(record)
    }
    fn write(&self, record: &Registration) -> Result<(), AuthError> {
        write_protected(&self.path(&record.id)?, record)
    }
    fn host(&self) -> Result<String, AuthError> {
        let path = self.owner.directory.join("host.json");
        let host: Host = if path.try_exists().map_err(|_| AuthError::Storage)? {
            read_protected(&path)?
        } else {
            let host = Host {
                id: format!("urn:uuid:{}", Uuid::new_v4()),
            };
            write_protected(&path, &host)?;
            host
        };
        let id = host
            .id
            .strip_prefix("urn:uuid:")
            .ok_or(AuthError::Storage)?;
        if Uuid::parse_str(id)
            .map_err(|_| AuthError::Storage)?
            .get_version_num()
            != 4
        {
            return Err(AuthError::Storage);
        }
        Ok(host.id)
    }
    async fn metadata(&self) -> Result<CoreProviderMetadata, AuthError> {
        CoreProviderMetadata::discover_async(
            IssuerUrl::new(self.owner.endpoints.issuer.clone()).map_err(|_| AuthError::Protocol)?,
            &self.owner.http,
        )
        .await
        .map_err(|_| AuthError::RetryLater)
    }
    pub async fn list(&self) -> Result<Vec<AccountInfo>, AuthError> {
        let _lock = self.lock().await?;
        let mut accounts = Vec::new();
        for entry in std::fs::read_dir(&self.owner.directory).map_err(|_| AuthError::Storage)? {
            let name = entry.map_err(|_| AuthError::Storage)?.file_name();
            let name = name.to_string_lossy();
            if let Some(id) = name
                .strip_prefix("registration-")
                .and_then(|s| s.strip_suffix(".json"))
            {
                accounts.push(self.read(id)?.info());
            }
        }
        accounts.sort_by(|a, b| a.registration_id.cmp(&b.registration_id));
        Ok(accounts)
    }
    pub async fn select(&self, id: &str) -> Result<BoundAuth, AuthError> {
        let _lock = self.lock().await?;
        let record = self.read(id)?;
        if record.identity.is_none() || record.tokens.is_none() {
            return Err(AuthError::SignInRequired);
        }
        if !record.info().plan_enabled {
            return Err(AuthError::PlanPermissionMissing);
        }
        write_protected(
            &self.owner.directory.join("selected.json"),
            &Selected {
                id: Some(id.into()),
            },
        )?;
        Ok(BoundAuth {
            manager: self.clone(),
            id: id.into(),
        })
    }
    /// Snapshot the selected registration; subsequent selection never retargets this handle.
    pub fn selected(&self) -> Result<Option<BoundAuth>, AuthError> {
        let Some(id) = self.selected_registration_id()? else {
            return Ok(None);
        };
        let record = self.read(&id)?;
        if record.identity.is_none() || record.tokens.is_none() {
            return Err(AuthError::SignInRequired);
        }
        if !record.info().plan_enabled {
            return Err(AuthError::PlanPermissionMissing);
        }
        Ok(Some(BoundAuth {
            manager: self.clone(),
            id,
        }))
    }
    pub fn selected_registration_id(&self) -> Result<Option<String>, AuthError> {
        let path = self.owner.directory.join("selected.json");
        if !path.try_exists().map_err(|_| AuthError::Storage)? {
            return Ok(None);
        }
        let selected: Selected = read_protected(&path)?;
        if let Some(id) = &selected.id {
            self.path(id)?;
        }
        Ok(selected.id)
    }
    pub async fn begin_login(
        &self,
        id: Option<&str>,
        request_consent: bool,
    ) -> Result<PendingLogin, AuthError> {
        self.begin_login_with_hint(id, request_consent, true).await
    }
    async fn begin_login_with_hint(
        &self,
        id: Option<&str>,
        request_consent: bool,
        include_identity_hint: bool,
    ) -> Result<PendingLogin, AuthError> {
        let (registration, expected) = {
            let _lock = self.lock().await?;
            let host_id = self.host()?;
            let expected = id.map(|id| self.read(id)).transpose()?;
            let registration = expected.clone().unwrap_or_else(|| Registration {
                version: 1,
                id: Uuid::new_v4().to_string(),
                client_id: "dynamic_agent_client".into(),
                host_id: host_id.clone(),
                identity: None,
                tokens: None,
            });
            if registration.host_id != host_id {
                return Err(AuthError::Storage);
            }
            (registration, expected)
        };
        let listener = TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0))
            .await
            .map_err(|_| AuthError::RetryLater)?;
        let redirect_uri = format!(
            "http://127.0.0.1:{}{CALLBACK_PATH}",
            listener
                .local_addr()
                .map_err(|_| AuthError::RetryLater)?
                .port()
        );
        let metadata = self.metadata().await?;
        let client = CoreClient::from_provider_metadata(
            metadata.clone(),
            ClientId::new(registration.client_id.clone()),
            None,
        )
        .set_auth_uri(
            AuthUrl::new(self.owner.endpoints.authorize.clone())
                .map_err(|_| AuthError::Protocol)?,
        )
        .set_redirect_uri(RedirectUrl::new(redirect_uri.clone()).map_err(|_| AuthError::Protocol)?);
        let (challenge, verifier) = PkceCodeChallenge::new_random_sha256();
        let mut request = client
            .authorize_url(
                CoreAuthenticationFlow::AuthorizationCode,
                CsrfToken::new_random,
                Nonce::new_random,
            )
            .set_pkce_challenge(challenge)
            .add_extra_param("resource", RESOURCE)
            .add_extra_param("ext_agent_host_id", registration.host_id.clone());
        for scope in [
            "profile",
            "email",
            "offline_access",
            "resource.invoke",
            DIRECT_SCOPE,
        ] {
            request = request.add_scope(Scope::new(scope.into()));
        }
        if registration.client_id == "dynamic_agent_client" {
            request = request.add_extra_param("agent_name_hint", "piku");
        }
        if include_identity_hint {
            if let Some(tokens) = &registration.tokens {
                request = request.add_extra_param("id_token_hint", tokens.id_token.clone());
            }
        }
        if request_consent {
            request = request.add_extra_param("prompt", "consent");
        }
        let (url, state, nonce) = request.url();
        Ok(PendingLogin {
            owner: self.owner.clone(),
            listener,
            authorization_url: url.to_string(),
            redirect_uri,
            state,
            nonce,
            verifier,
            metadata,
            registration,
            expected,
        })
    }
    pub async fn complete_login(
        &self,
        mut pending: PendingLogin,
    ) -> Result<AccountInfo, AuthError> {
        if !Arc::ptr_eq(&self.owner, &pending.owner) {
            return Err(AuthError::InvalidCallback);
        }
        let callback = tokio::time::timeout(Duration::from_secs(180), receive_callback(&pending))
            .await
            .map_err(|_| AuthError::TimedOut)??;
        let _lock = self.lock().await?;
        if let Some(expected) = &pending.expected {
            if self.read(&expected.id)? != *expected {
                return Err(AuthError::ChangedRegistration);
            }
        }
        if pending.registration.client_id == "dynamic_agent_client" {
            pending.registration.client_id =
                callback.client_id.ok_or(AuthError::InvalidCallback)?;
            // Retain the issued client for a fresh code attempt even if exchange fails.
            self.write(&pending.registration)?;
        } else if callback
            .client_id
            .is_some_and(|id| id != pending.registration.client_id)
        {
            return Err(AuthError::InvalidCallback);
        }
        let client = CoreClient::from_provider_metadata(
            pending.metadata,
            ClientId::new(pending.registration.client_id.clone()),
            None,
        )
        .set_token_uri(
            TokenUrl::new(self.owner.endpoints.token.clone()).map_err(|_| AuthError::Protocol)?,
        )
        .set_redirect_uri(RedirectUrl::new(pending.redirect_uri).map_err(|_| AuthError::Protocol)?);
        let response = client
            .exchange_code(AuthorizationCode::new(callback.code))
            .set_pkce_verifier(pending.verifier)
            .add_extra_param("resource", RESOURCE)
            .request_async(&self.owner.http)
            .await
            .map_err(token_error)?;
        let id_token = response.id_token().ok_or(AuthError::InvalidIdentity)?;
        let verifier = client.id_token_verifier();
        let claims = id_token
            .claims(&verifier, &pending.nonce)
            .map_err(|_| AuthError::InvalidIdentity)?;
        verify_access_hash(&response, &verifier, claims.access_token_hash())?;
        let identity = Identity {
            issuer: claims.issuer().as_str().into(),
            subject: claims.subject().as_str().into(),
            email: claims.email().map(|s| s.as_str().into()),
        };
        if identity.subject.is_empty()
            || pending
                .registration
                .identity
                .as_ref()
                .is_some_and(|old| old.issuer != identity.issuer || old.subject != identity.subject)
        {
            return Err(AuthError::InvalidIdentity);
        }
        let tokens = token_set(&response, pending.nonce.secret().clone(), None)?;
        pending.registration.identity = Some(identity);
        pending.registration.tokens = Some(tokens);
        self.write(&pending.registration)?;
        Ok(pending.registration.info())
    }
    pub async fn logout(&self, id: &str) -> Result<LogoutOutcome, AuthError> {
        let _lock = self.lock().await?;
        let mut record = self.read(id)?;
        let confirmed = if let Some(refresh) = record
            .tokens
            .as_ref()
            .and_then(|t| t.refresh_token.as_deref())
        {
            self.revoke(&record.client_id, refresh).await
        } else {
            false
        };
        record.tokens = None;
        self.write(&record)?;
        if self.selected_registration_id()?.as_deref() == Some(id) {
            write_protected(
                &self.owner.directory.join("selected.json"),
                &Selected { id: None },
            )?;
        }
        Ok(LogoutOutcome {
            remote_revocation_confirmed: confirmed,
        })
    }
    async fn revoke(&self, client_id: &str, refresh: &str) -> bool {
        let url = format!(
            "{}/.well-known/openid-configuration",
            self.owner.endpoints.issuer
        );
        let Ok(response) = self.owner.http.get(url).send().await else {
            return false;
        };
        if !response.status().is_success() {
            return false;
        }
        let Ok(document) = response.json::<serde_json::Value>().await else {
            return false;
        };
        let Some(endpoint) = document
            .get("revocation_endpoint")
            .and_then(serde_json::Value::as_str)
        else {
            return false;
        };
        let (Ok(endpoint), Ok(issuer)) = (
            reqwest::Url::parse(endpoint),
            reqwest::Url::parse(&self.owner.endpoints.issuer),
        ) else {
            return false;
        };
        if endpoint.origin() != issuer.origin() {
            return false;
        }
        for attempt in 0..2 {
            let result = self
                .owner
                .http
                .post(endpoint.clone())
                .form(&[
                    ("token", refresh),
                    ("token_type_hint", "refresh_token"),
                    ("client_id", client_id),
                ])
                .send()
                .await;
            match result {
                Ok(response) if response.status() == reqwest::StatusCode::OK => return true,
                Ok(response) if !response.status().is_server_error() => return false,
                _ if attempt == 0 => tokio::time::sleep(Duration::from_millis(100)).await,
                _ => return false,
            }
        }
        false
    }
}

impl BoundAuth {
    #[must_use]
    pub fn binding(&self) -> &str {
        &self.id
    }
    pub async fn access_token(&self) -> Result<CredentialToken, AuthError> {
        let _lock = self.manager.lock().await?;
        let mut record = self.manager.read(&self.id)?;
        let old = record.tokens.as_ref().ok_or(AuthError::SignInRequired)?;
        if !record.info().plan_enabled {
            return Err(AuthError::PlanPermissionMissing);
        }
        if old.expires_at > now()?.saturating_add(REFRESH_MARGIN) {
            return Ok(CredentialToken(old.access_token.clone()));
        }
        let refresh = old
            .refresh_token
            .as_ref()
            .ok_or(AuthError::SignInRequired)?;
        let metadata = self.manager.metadata().await?;
        let client = CoreClient::from_provider_metadata(
            metadata,
            ClientId::new(record.client_id.clone()),
            None,
        )
        .set_token_uri(
            TokenUrl::new(self.manager.owner.endpoints.token.clone())
                .map_err(|_| AuthError::Protocol)?,
        );
        let result = client
            .exchange_refresh_token(&RefreshToken::new(refresh.clone()))
            .add_extra_param("resource", RESOURCE)
            .request_async(&self.manager.owner.http)
            .await;
        let response = match result {
            Ok(response) => response,
            Err(error) => {
                let error = token_error(error);
                if error == AuthError::SignInRequired {
                    record.tokens = None;
                    self.manager.write(&record)?;
                }
                return Err(error);
            }
        };
        if let Some(id_token) = response.id_token() {
            let verifier = client.id_token_verifier();
            let claims = id_token
                .claims(&verifier, |nonce: Option<&Nonce>| {
                    if nonce.is_some_and(|value| value.secret() != &old.nonce) {
                        Err("refresh nonce mismatch".into())
                    } else {
                        Ok(())
                    }
                })
                .map_err(|_| AuthError::InvalidIdentity)?;
            verify_access_hash(&response, &verifier, claims.access_token_hash())?;
            let identity = record.identity.as_ref().ok_or(AuthError::InvalidIdentity)?;
            if claims.subject().as_str() != identity.subject
                || claims.issuer().as_str() != identity.issuer
            {
                return Err(AuthError::InvalidIdentity);
            }
        }
        let tokens = token_set(&response, old.nonce.clone(), Some(old))?;
        let token = CredentialToken(tokens.access_token.clone());
        record.tokens = Some(tokens);
        self.manager.write(&record)?;
        if !record.info().plan_enabled {
            return Err(AuthError::PlanPermissionMissing);
        }
        Ok(token)
    }
}

fn token_error<E: std::error::Error + 'static>(
    error: openidconnect::RequestTokenError<
        E,
        openidconnect::StandardErrorResponse<openidconnect::core::CoreErrorResponseType>,
    >,
) -> AuthError {
    match error {
        openidconnect::RequestTokenError::ServerResponse(response) => {
            match response.error().to_string().as_str() {
                "invalid_grant"
                | "invalid_refresh_token"
                | "token_expired"
                | "refresh_token_expired"
                | "refresh_token_invalidated"
                | "refresh_token_reused" => AuthError::SignInRequired,
                "invalid_client" => AuthError::InvalidClient,
                _ => AuthError::RetryLater,
            }
        }
        openidconnect::RequestTokenError::Request(_) => AuthError::RetryLater,
        _ => AuthError::Protocol,
    }
}
fn verify_access_hash(
    response: &CoreTokenResponse,
    verifier: &openidconnect::core::CoreIdTokenVerifier<'_>,
    expected: Option<&AccessTokenHash>,
) -> Result<(), AuthError> {
    if let Some(expected) = expected {
        let id_token = response.id_token().ok_or(AuthError::InvalidIdentity)?;
        let actual = AccessTokenHash::from_token(
            response.access_token(),
            id_token
                .signing_alg()
                .map_err(|_| AuthError::InvalidIdentity)?,
            id_token
                .signing_key(verifier)
                .map_err(|_| AuthError::InvalidIdentity)?,
        )
        .map_err(|_| AuthError::InvalidIdentity)?;
        if actual != *expected {
            return Err(AuthError::InvalidIdentity);
        }
    }
    Ok(())
}
fn token_set(
    response: &CoreTokenResponse,
    nonce: String,
    old: Option<&Tokens>,
) -> Result<Tokens, AuthError> {
    if response.access_token().secret().is_empty()
        || response.token_type() != &openidconnect::core::CoreTokenType::Bearer
    {
        return Err(AuthError::Protocol);
    }
    let expires_at = now()?
        .checked_add(response.expires_in().ok_or(AuthError::Protocol)?.as_secs())
        .ok_or(AuthError::Protocol)?;
    let scopes = response
        .scopes()
        .map(|s| s.iter().map(|s| s.as_str().into()).collect())
        .or_else(|| old.map(|t| t.scopes.clone()))
        .ok_or(AuthError::Protocol)?;
    Ok(Tokens {
        access_token: response.access_token().secret().clone(),
        refresh_token: response
            .refresh_token()
            .map(|r| r.secret().clone())
            .or_else(|| old.and_then(|t| t.refresh_token.clone())),
        id_token: response
            .id_token()
            .map(ToString::to_string)
            .or_else(|| old.map(|t| t.id_token.clone()))
            .ok_or(AuthError::InvalidIdentity)?,
        nonce,
        scopes,
        expires_at,
    })
}
fn now() -> Result<u64, AuthError> {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|v| v.as_secs())
        .map_err(|_| AuthError::Protocol)
}

struct Callback {
    code: String,
    client_id: Option<String>,
}
async fn receive_callback(pending: &PendingLogin) -> Result<Callback, AuthError> {
    let (mut socket, peer) = pending
        .listener
        .accept()
        .await
        .map_err(|_| AuthError::RetryLater)?;
    if !peer.ip().is_loopback() {
        return Err(AuthError::InvalidCallback);
    }
    let mut bytes = Vec::new();
    while !bytes.ends_with(b"\r\n\r\n") {
        if bytes.len() >= 16_384 {
            return Err(AuthError::InvalidCallback);
        }
        let byte = tokio::time::timeout(Duration::from_secs(5), socket.read_u8())
            .await
            .map_err(|_| AuthError::TimedOut)?
            .map_err(|_| AuthError::InvalidCallback)?;
        bytes.push(byte);
    }
    let request = std::str::from_utf8(&bytes).map_err(|_| AuthError::InvalidCallback)?;
    let first = request.lines().next().ok_or(AuthError::InvalidCallback)?;
    let mut parts = first.split_whitespace();
    if parts.next() != Some("GET") {
        return Err(AuthError::InvalidCallback);
    }
    let target = parts.next().ok_or(AuthError::InvalidCallback)?;
    if !target.starts_with("/auth/callback?")
        || parts.next() != Some("HTTP/1.1")
        || parts.next().is_some()
    {
        return Err(AuthError::InvalidCallback);
    }
    let expected = reqwest::Url::parse(&pending.redirect_uri).map_err(|_| AuthError::Protocol)?;
    let url = expected
        .join(target)
        .map_err(|_| AuthError::InvalidCallback)?;
    if url.origin() != expected.origin() || url.path() != CALLBACK_PATH {
        return Err(AuthError::InvalidCallback);
    }
    let hosts: Vec<_> = request
        .lines()
        .skip(1)
        .filter_map(|line| line.split_once(':'))
        .filter(|(name, _)| name.eq_ignore_ascii_case("host"))
        .map(|(_, value)| value.trim())
        .collect();
    let authority = format!(
        "127.0.0.1:{}",
        expected.port().ok_or(AuthError::InvalidCallback)?
    );
    if hosts != [authority.as_str()] {
        return Err(AuthError::InvalidCallback);
    }
    let mut query = std::collections::HashMap::new();
    for (key, value) in url.query_pairs() {
        if query.insert(key.into_owned(), value.into_owned()).is_some() {
            return Err(AuthError::InvalidCallback);
        }
    }
    if query.get("state") != Some(pending.state.secret()) {
        return Err(AuthError::InvalidCallback);
    }
    if query.contains_key("error") {
        return Err(AuthError::Declined);
    }
    let code = query
        .remove("code")
        .filter(|s| !s.is_empty())
        .ok_or(AuthError::InvalidCallback)?;
    let client_id = query.remove("client_id");
    if client_id.as_ref().is_some_and(|id| {
        id.is_empty()
            || id == "dynamic_agent_client"
            || id.len() > 256
            || id.chars().any(char::is_whitespace)
    }) {
        return Err(AuthError::InvalidCallback);
    }
    socket.write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 51\r\nContent-Type: text/plain\r\nConnection: close\r\n\r\nCallback received. Return to piku to check sign-in.").await.map_err(|_| AuthError::RetryLater)?;
    Ok(Callback { code, client_id })
}

struct StoreLock(File);
impl StoreLock {
    async fn acquire(path: &Path) -> Result<Self, AuthError> {
        let file = open_protected(path, true)?;
        let deadline = tokio::time::Instant::now() + Duration::from_secs(35);
        loop {
            match file.try_lock() {
                Ok(()) => return Ok(Self(file)),
                Err(std::fs::TryLockError::WouldBlock)
                    if tokio::time::Instant::now() < deadline =>
                {
                    tokio::time::sleep(Duration::from_millis(25)).await;
                }
                _ => return Err(AuthError::RetryLater),
            }
        }
    }
}
impl Drop for StoreLock {
    fn drop(&mut self) {
        let _ = self.0.unlock();
    }
}

fn protected_directory(path: &Path) -> Result<(), AuthError> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::{DirBuilderExt, PermissionsExt};
        let mut builder = std::fs::DirBuilder::new();
        builder.recursive(true).mode(0o700);
        builder.create(path).map_err(|_| AuthError::Storage)?;
        let metadata = std::fs::symlink_metadata(path).map_err(|_| AuthError::Storage)?;
        if !metadata.is_dir() || metadata.permissions().mode() & 0o077 != 0 {
            return Err(AuthError::Storage);
        }
        Ok(())
    }
    #[cfg(not(unix))]
    {
        let _ = path;
        Err(AuthError::Storage)
    }
}
fn open_protected(path: &Path, create: bool) -> Result<File, AuthError> {
    let mut options = OpenOptions::new();
    options
        .read(true)
        .write(create)
        .create(create)
        .truncate(false);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600).custom_flags(nix::libc::O_NOFOLLOW);
    }
    let file = options.open(path).map_err(|_| AuthError::Storage)?;
    let metadata = file.metadata().map_err(|_| AuthError::Storage)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if metadata.permissions().mode() & 0o077 != 0 {
            return Err(AuthError::Storage);
        }
    }
    if !metadata.is_file() || metadata.len() > 1_048_576 {
        return Err(AuthError::Storage);
    }
    Ok(file)
}
fn read_protected<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T, AuthError> {
    let mut data = Vec::new();
    open_protected(path, false)?
        .take(1_048_577)
        .read_to_end(&mut data)
        .map_err(|_| AuthError::Storage)?;
    serde_json::from_slice(&data).map_err(|_| AuthError::Storage)
}
fn write_protected<T: Serialize>(path: &Path, value: &T) -> Result<(), AuthError> {
    let parent = path.parent().ok_or(AuthError::Storage)?;
    let stage = parent.join(format!(".stage-{}", Uuid::new_v4()));
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut file = options.open(&stage).map_err(|_| AuthError::Storage)?;
    let result = (|| {
        let data = serde_json::to_vec(value).map_err(|_| AuthError::Storage)?;
        file.write_all(&data).map_err(|_| AuthError::Storage)?;
        file.sync_all().map_err(|_| AuthError::Storage)?;
        std::fs::rename(&stage, path).map_err(|_| AuthError::Storage)
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(stage);
    }
    result
}

impl piku_api::AccessTokenSource for BoundAuth {
    fn token(
        &self,
    ) -> std::pin::Pin<
        Box<
            dyn std::future::Future<Output = Result<piku_api::AccessToken, piku_api::ApiError>>
                + Send
                + '_,
        >,
    > {
        Box::pin(async move {
            let token = self
                .access_token()
                .await
                .map_err(|error| piku_api::ApiError::Provider(error.to_string()))?;
            Ok(piku_api::AccessToken::new(token.0))
        })
    }
    fn binding(&self) -> &str {
        &self.id
    }
}

#[cfg(test)]
mod tests;
