use super::*;
use openidconnect::core::{
    CoreIdToken, CoreIdTokenClaims, CoreJwsSigningAlgorithm, CoreRsaPrivateSigningKey,
};
use openidconnect::{
    Audience, EmptyAdditionalClaims, EndUserEmail, PrivateSigningKey, StandardClaims,
    SubjectIdentifier,
};
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};

// Generated solely for this loopback fixture. Never used for a real identity.
const SYNTHETIC_KEY: &str = r"-----BEGIN RSA PRIVATE KEY-----
MIIEpQIBAAKCAQEAtuc4se42a+H1JbfBn7wPsMD65gOfLOzJHbjSZst6hkHgpBed
qQIHIvzTFn7SY62AMf1uoJ2To5Qe7ZwoNs57iR+FHHkhyLN4mbpLTb+TqoA43nQU
ORvNspUEigbZZTR8rJim1p+rGHotOhaQkC5SO1eD6wGrMTd5uL7rNlldr9uswDTS
URfOx0QgJ7kmCj+qGTeqkCQCySLSFCBMfkfWcj/RsXP99IForzGcEnCAlT978T4s
r5t2kwLtg2bLeqmM+ev4s83ENLoixG7dTLac9zPbQEI82eoTFC3aImnQ4e2Ra7j1
wGd9r7DhGEPHLnw9ZHfkgB0DWyPyYo+awAmzKwIDAQABAoIBAQCKKoL+jmeX+Xo5
wuVGaEcmC7G8vfGpezm19R8jvgRRlURAUQKtYDXJ7Vj8PlNqLu+PU9YVCNQ1lrgC
YJtUQ4i4nhHs32hw93pcDonCMr026GONT/zS4zjw9XcXWXMx64Ne0nX93WHfPLQb
lF4iPIur/kFN7tXGeRS/RaHjTYDjTXNgZexW8w5eDtmQgFRfKuFi/VN0h4XZrrM4
v3I3AtwK2mAwfgV/vRSCfXFDlyZRtrYl8B/KRaXjC0E6qjvXJ/z1FH9tRZhZnra+
mHBuEbHJtETdk4QGA/d59bJdN9JPbU+kDAcVuo1Rc0KwhRnQWrVpcNl4zsyl2ZwG
9lPzej5JAoGBAO7IWlzf/QRqTF232LrRE3BfrDdPc4XtgJ1m1o5VEmHawutwku5L
dZ1pNADByHn8euW1/P+rcS8NAGtKUJ23SU15uLRxrQftaz0VoRHlsSZyy7Mm9BO4
uw18n8O9rYarLWdHGJX2RoHgUHrBy+h0WZEeUDYWIYRBklJd9bC5G481AoGBAMQX
ZmW3rm+d71kousZUD4IVQ5TgDIjT5R0YSwE19ceWalM08DEHqr0PgmdwhzYOiCOD
0/X6uyN5E5D8kCJEHRzQPqkuf6qXuMwPGiopzSgmG8SUL8h/JU2SkGr8cODxEC9u
XyGUJUfNiF7p0l2gOzSPTWtrpBF6DuCYcBjLYqTfAoGAH4jIHkHaeU3Mfgf8kFdy
OMH5zWgpuoMuHBtrtGpZviCU42D/lobSppIxMvRklG+aj2GkYuNyVCNkbcbntJxz
Cs9LbFEi58wcAHotxl5OMeSxMByx2TTpRxOxSblwBDvxCBWkkpFuqFJGbp/PXQiY
l4ioUSxFCuFaAJS2ELuxfw0CgYEApQ7BXauQhFmfyB/1n83QRAIcIeu6AU7Dk4sI
T5uZKAiftSxhZhP7DytH7EQXPFRdGe/0ciA8d+6JnshJlt5Y1xEfPY9rXTBbGEE6
F/WZwErbosvNsOn9f194euXFKkMgCZzRLNWjz93dBW+yaKbgTyy00VAJCcEBVuJZ
sTU58mcCgYEA2/Fb77+oHmvhTOu9oRSYdYSpDyD34fmZtKJvs/+FCYCJAyY2jInz
SLnSwXA7J738w91loy7ne4WwJOoffH3PFVOPKfuU0tY4BLFZBTbA6881agv2FWkm
9uYjQ+eMFhb+aByfn+BUzkVWARO8Xm16wn1JZJVtB8O6AmTJN8Arwrw=
-----END RSA PRIVATE KEY-----
";

#[derive(Clone, Copy, PartialEq, Eq)]
enum Fault {
    None,
    Nonce,
    Audience,
    Issuer,
    Signature,
    Expired,
    Subject,
    Scope,
    InvalidGrant,
    Transient,
}
struct MockState {
    fault: Fault,
    nonce: String,
    short_expiry: bool,
    forms: Vec<HashMap<String, String>>,
}
struct Mock {
    url: String,
    state: Arc<std::sync::Mutex<MockState>>,
    refreshes: Arc<AtomicUsize>,
    requests: Arc<AtomicUsize>,
    task: tokio::task::JoinHandle<()>,
}
impl Drop for Mock {
    fn drop(&mut self) {
        self.task.abort();
    }
}
impl Mock {
    // Keep the fixture HTTP routes together; validation uses the real OIDC client.
    #[allow(clippy::too_many_lines)]
    async fn start() -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let state = Arc::new(std::sync::Mutex::new(MockState {
            fault: Fault::None,
            nonce: String::new(),
            short_expiry: false,
            forms: Vec::new(),
        }));
        let refreshes = Arc::new(AtomicUsize::new(0));
        let requests = Arc::new(AtomicUsize::new(0));
        let task = tokio::spawn({
            let url = url.clone();
            let state = state.clone();
            let refreshes = refreshes.clone();
            let requests = requests.clone();
            let key = CoreRsaPrivateSigningKey::from_pem(SYNTHETIC_KEY, None).unwrap();
            async move {
                loop {
                    let (mut socket, _) = listener.accept().await.unwrap();
                    requests.fetch_add(1, Ordering::SeqCst);
                    let mut bytes = Vec::new();
                    let mut buffer = [0; 4096];
                    let header_end = loop {
                        let count = socket.read(&mut buffer).await.unwrap();
                        assert!(count > 0);
                        bytes.extend_from_slice(&buffer[..count]);
                        if let Some(end) = bytes.windows(4).position(|v| v == b"\r\n\r\n") {
                            break end;
                        }
                    };
                    let header = String::from_utf8(bytes[..header_end].to_vec()).unwrap();
                    let length = header
                        .lines()
                        .filter_map(|line| line.split_once(':'))
                        .find(|(name, _)| name.eq_ignore_ascii_case("content-length"))
                        .map_or(0, |(_, value)| value.trim().parse::<usize>().unwrap());
                    while bytes.len() < header_end + 4 + length {
                        let count = socket.read(&mut buffer).await.unwrap();
                        assert!(count > 0);
                        bytes.extend_from_slice(&buffer[..count]);
                    }
                    let route = header
                        .lines()
                        .next()
                        .unwrap()
                        .split_whitespace()
                        .nth(1)
                        .unwrap();
                    let (status, body) = if route == "/.well-known/openid-configuration" {
                        (200, serde_json::json!({"issuer":url,"authorization_endpoint":format!("{url}/authorize"),"token_endpoint":format!("{url}/token"),"jwks_uri":format!("{url}/jwks"),"revocation_endpoint":format!("{url}/revoke"),"response_types_supported":["code"],"subject_types_supported":["public"],"id_token_signing_alg_values_supported":["RS256"]}).to_string())
                    } else if route == "/jwks" {
                        (
                            200,
                            serde_json::json!({"keys":[key.as_verification_key()]}).to_string(),
                        )
                    } else {
                        let raw = String::from_utf8(bytes[header_end + 4..].to_vec()).unwrap();
                        let mut parsed = reqwest::Url::parse("http://fixture/").unwrap();
                        parsed.set_query(Some(&raw));
                        let form: HashMap<String, String> = parsed
                            .query_pairs()
                            .map(|(k, v)| (k.into_owned(), v.into_owned()))
                            .collect();
                        let refresh = form
                            .get("grant_type")
                            .is_some_and(|value| value == "refresh_token");
                        let (fault, nonce, short_expiry) = {
                            let mut state = state.lock().unwrap();
                            state.forms.push(form.clone());
                            (state.fault, state.nonce.clone(), state.short_expiry)
                        };
                        if refresh {
                            refreshes.fetch_add(1, Ordering::SeqCst);
                            tokio::time::sleep(Duration::from_millis(50)).await;
                        }
                        if fault == Fault::Transient {
                            (503, "{\"error\":\"temporarily_unavailable\",\"error_description\":\"synthetic-secret-must-not-leak\"}".into())
                        } else if fault == Fault::InvalidGrant {
                            (400, "{\"error\":\"invalid_grant\"}".into())
                        } else if route == "/revoke" {
                            (200, String::new())
                        } else {
                            assert_eq!(form.get("resource").map(String::as_str), Some(RESOURCE));
                            assert_ne!(form["client_id"], "dynamic_agent_client");
                            let access = openidconnect::AccessToken::new(
                                if refresh {
                                    "synthetic-access-rotated"
                                } else {
                                    "synthetic-access-initial"
                                }
                                .into(),
                            );
                            let claims = CoreIdTokenClaims::new(
                                IssuerUrl::new(if fault == Fault::Issuer {
                                    "https://wrong.invalid".into()
                                } else {
                                    url.clone()
                                })
                                .unwrap(),
                                vec![Audience::new(if fault == Fault::Audience {
                                    "wrong-client".into()
                                } else {
                                    form["client_id"].clone()
                                })],
                                if fault == Fault::Expired {
                                    (SystemTime::now() - Duration::from_secs(600)).into()
                                } else {
                                    (SystemTime::now() + Duration::from_secs(3600)).into()
                                },
                                SystemTime::now().into(),
                                StandardClaims::new(SubjectIdentifier::new(
                                    if fault == Fault::Subject {
                                        "another-subject"
                                    } else {
                                        "fixture-subject"
                                    }
                                    .into(),
                                ))
                                .set_email(Some(
                                    EndUserEmail::new("fixture@example.invalid".into()),
                                )),
                                EmptyAdditionalClaims {},
                            )
                            .set_nonce(Some(Nonce::new(
                                if fault == Fault::Nonce {
                                    "wrong-nonce".into()
                                } else {
                                    nonce
                                },
                            )));
                            let mut id_token = CoreIdToken::new(
                                claims,
                                &key,
                                CoreJwsSigningAlgorithm::RsaSsaPkcs1V15Sha256,
                                Some(&access),
                                None,
                            )
                            .unwrap()
                            .to_string();
                            if fault == Fault::Signature {
                                let start = id_token.rfind('.').unwrap() + 1;
                                let replacement = if &id_token[start..=start] == "a" {
                                    "b"
                                } else {
                                    "a"
                                };
                                id_token.replace_range(start..=start, replacement);
                            }
                            (200,serde_json::json!({"access_token":access.secret(),"refresh_token":if refresh { "synthetic-refresh-rotated" } else { "synthetic-refresh-initial" },"id_token":id_token,"token_type":"Bearer","expires_in":if short_expiry && !refresh {10} else {3600},"scope":if fault==Fault::Scope {"openid profile"} else {"openid profile email offline_access resource.invoke chatgpt.tokens.use.direct"}}).to_string())
                        }
                    };
                    let response = format!("HTTP/1.1 {status} Fixture\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",body.len());
                    let _ = socket.write_all(response.as_bytes()).await;
                }
            }
        });
        Self {
            url,
            state,
            refreshes,
            requests,
            task,
        }
    }
    fn manager(&self, path: &Path) -> AuthManager {
        AuthManager::with_endpoints(
            path.join("auth"),
            Endpoints {
                issuer: self.url.clone(),
                authorize: format!("{}/authorize", self.url),
                token: format!("{}/token", self.url),
            },
        )
        .unwrap()
    }
    fn prepare(&self, pending: &PendingLogin) {
        self.state.lock().unwrap().nonce = pending.nonce.secret().clone();
    }
}
async fn callback(
    manager: &AuthManager,
    pending: PendingLogin,
    state: Option<&str>,
    client: Option<&str>,
) -> Result<AccountInfo, AuthError> {
    let mut url = reqwest::Url::parse(&pending.redirect_uri).unwrap();
    url.query_pairs_mut()
        .append_pair("state", state.unwrap_or(pending.state.secret()))
        .append_pair("code", "synthetic-code");
    if let Some(client) = client {
        url.query_pairs_mut().append_pair("client_id", client);
    }
    let browser = async {
        let _ = reqwest::Client::new().get(url).send().await;
    };
    let (result, ()) = tokio::join!(manager.complete_login(pending), browser);
    result
}
async fn login(mock: &Mock, manager: &AuthManager, client: &str) -> AccountInfo {
    let pending = manager.begin_login(None, false).await.unwrap();
    mock.prepare(&pending);
    callback(manager, pending, None, Some(client))
        .await
        .unwrap()
}

#[tokio::test]
async fn real_oidc_login_binds_issued_client_pkce_and_host() {
    let mock = Mock::start().await;
    let directory = tempfile::tempdir().unwrap();
    let manager = mock.manager(directory.path());
    let pending = manager.begin_login(None, false).await.unwrap();
    mock.prepare(&pending);
    let url = reqwest::Url::parse(pending.authorization_url()).unwrap();
    let query: HashMap<_, _> = url
        .query_pairs()
        .map(|(k, v)| (k.into_owned(), v.into_owned()))
        .collect();
    assert_eq!(query["client_id"], "dynamic_agent_client");
    assert_eq!(query["agent_name_hint"], "piku");
    assert_eq!(query["code_challenge_method"], "S256");
    let info = callback(&manager, pending, None, Some("issued-one"))
        .await
        .unwrap();
    assert!(info.plan_enabled);
    assert_eq!(info.subject.as_deref(), Some("fixture-subject"));
    let form = mock.state.lock().unwrap().forms[0].clone();
    assert_eq!(form["client_id"], "issued-one");
    assert_eq!(form["redirect_uri"], query["redirect_uri"]);
    assert_eq!(
        PkceCodeChallenge::from_code_verifier_sha256(&PkceCodeVerifier::new(
            form["code_verifier"].clone()
        ))
        .as_str(),
        query["code_challenge"]
    );
    assert!(manager.selected().unwrap().is_none());
    let bound = manager.select(&info.registration_id).await.unwrap();
    let reopened = mock.manager(directory.path());
    assert_eq!(
        reopened.selected().unwrap().unwrap().binding(),
        bound.binding()
    );
    let again = reopened
        .begin_login(Some(bound.binding()), false)
        .await
        .unwrap();
    let next = reqwest::Url::parse(again.authorization_url()).unwrap();
    let query2: HashMap<_, _> = next
        .query_pairs()
        .map(|(k, v)| (k.into_owned(), v.into_owned()))
        .collect();
    assert_eq!(query2["client_id"], "issued-one");
    assert_eq!(query2["ext_agent_host_id"], query["ext_agent_host_id"]);
    assert!(!query2.contains_key("agent_name_hint"));
    assert_ne!(query2["nonce"], query["nonce"]);
    let token = bound.access_token().await.unwrap();
    assert_eq!(token.expose_secret(), "synthetic-access-initial");
    let debug = format!("{manager:?} {again:?} {info:?} {bound:?} {token:?}");
    assert!(!debug.contains("synthetic-access"));
    assert!(!debug.contains("fixture@example"));
    assert!(!debug.contains("eyJ"));
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        assert_eq!(
            std::fs::metadata(&manager.owner.directory)
                .unwrap()
                .permissions()
                .mode()
                & 0o777,
            0o700
        );
        for entry in std::fs::read_dir(&manager.owner.directory).unwrap() {
            assert_eq!(
                entry.unwrap().metadata().unwrap().permissions().mode() & 0o777,
                0o600
            );
        }
    }
}

#[tokio::test]
async fn oidc_rejects_wrong_state_nonce_audience_issuer_signature_and_expiry() {
    let mock = Mock::start().await;
    let directory = tempfile::tempdir().unwrap();
    let manager = mock.manager(directory.path());
    let pending = manager.begin_login(None, false).await.unwrap();
    assert_eq!(
        callback(&manager, pending, Some("wrong-state"), Some("issued-one"))
            .await
            .unwrap_err(),
        AuthError::InvalidCallback
    );
    assert!(mock.state.lock().unwrap().forms.is_empty());
    for fault in [
        Fault::Nonce,
        Fault::Audience,
        Fault::Issuer,
        Fault::Signature,
        Fault::Expired,
    ] {
        mock.state.lock().unwrap().fault = fault;
        let pending = manager.begin_login(None, false).await.unwrap();
        mock.prepare(&pending);
        assert_eq!(
            callback(&manager, pending, None, Some("issued-one"))
                .await
                .unwrap_err(),
            AuthError::InvalidIdentity
        );
    }
    assert!(manager
        .list()
        .await
        .unwrap()
        .iter()
        .all(|info| !info.signed_in));
    assert!(manager.selected().unwrap().is_none());
}

#[tokio::test]
async fn returning_identity_and_client_cannot_replace_selected_account() {
    let mock = Mock::start().await;
    let directory = tempfile::tempdir().unwrap();
    let manager = mock.manager(directory.path());
    let info = login(&mock, &manager, "issued-one").await;
    manager.select(&info.registration_id).await.unwrap();
    let before = std::fs::read(manager.path(&info.registration_id).unwrap()).unwrap();
    let pending = manager
        .begin_login(Some(&info.registration_id), false)
        .await
        .unwrap();
    mock.prepare(&pending);
    assert_eq!(
        callback(&manager, pending, None, Some("different-client"))
            .await
            .unwrap_err(),
        AuthError::InvalidCallback
    );
    mock.state.lock().unwrap().fault = Fault::Subject;
    let pending = manager
        .begin_login(Some(&info.registration_id), false)
        .await
        .unwrap();
    mock.prepare(&pending);
    assert_eq!(
        callback(&manager, pending, None, None).await.unwrap_err(),
        AuthError::InvalidIdentity
    );
    assert_eq!(
        std::fs::read(manager.path(&info.registration_id).unwrap()).unwrap(),
        before
    );
}

#[tokio::test]
async fn missing_scope_retains_identity_but_never_activates_or_issues_token() {
    let mock = Mock::start().await;
    mock.state.lock().unwrap().fault = Fault::Scope;
    let directory = tempfile::tempdir().unwrap();
    let manager = mock.manager(directory.path());
    let info = login(&mock, &manager, "issued-one").await;
    assert!(info.signed_in);
    assert!(!info.plan_enabled);
    assert_eq!(
        manager.select(&info.registration_id).await.unwrap_err(),
        AuthError::PlanPermissionMissing
    );
    assert!(manager.selected().unwrap().is_none());
    let pending = manager
        .begin_login(Some(&info.registration_id), true)
        .await
        .unwrap();
    let url = reqwest::Url::parse(pending.authorization_url()).unwrap();
    assert!(url
        .query_pairs()
        .any(|(k, v)| k == "prompt" && v == "consent"));
}

#[tokio::test]
async fn clones_and_independent_managers_refresh_once_and_rotate_together() {
    let mock = Mock::start().await;
    mock.state.lock().unwrap().short_expiry = true;
    let directory = tempfile::tempdir().unwrap();
    let manager = mock.manager(directory.path());
    let info = login(&mock, &manager, "issued-one").await;
    let one = manager.select(&info.registration_id).await.unwrap();
    let two = one.clone();
    let independent = mock.manager(directory.path()).selected().unwrap().unwrap();
    let (first, second, third) = tokio::join!(
        one.access_token(),
        two.access_token(),
        independent.access_token()
    );
    for token in [first, second, third] {
        assert_eq!(token.unwrap().expose_secret(), "synthetic-access-rotated");
    }
    assert_eq!(mock.refreshes.load(Ordering::SeqCst), 1);
    let record = manager.read(&info.registration_id).unwrap();
    let tokens = record.tokens.unwrap();
    assert_eq!(
        tokens.refresh_token.as_deref(),
        Some("synthetic-refresh-rotated")
    );
    assert!(tokens.expires_at > now().unwrap() + REFRESH_MARGIN);
}

#[tokio::test]
async fn refresh_transient_preserves_credentials_terminal_clears_only_tokens() {
    let mock = Mock::start().await;
    mock.state.lock().unwrap().short_expiry = true;
    let directory = tempfile::tempdir().unwrap();
    let manager = mock.manager(directory.path());
    let info = login(&mock, &manager, "issued-one").await;
    let bound = manager.select(&info.registration_id).await.unwrap();
    let before = std::fs::read(manager.path(&info.registration_id).unwrap()).unwrap();
    mock.state.lock().unwrap().fault = Fault::Transient;
    let error = bound.access_token().await.unwrap_err();
    assert_eq!(error, AuthError::RetryLater);
    assert!(!error.to_string().contains("synthetic-secret"));
    assert_eq!(
        std::fs::read(manager.path(&info.registration_id).unwrap()).unwrap(),
        before
    );
    mock.state.lock().unwrap().fault = Fault::InvalidGrant;
    assert_eq!(
        bound.access_token().await.unwrap_err(),
        AuthError::SignInRequired
    );
    let record = manager.read(&info.registration_id).unwrap();
    assert!(record.tokens.is_none());
    assert!(record.identity.is_some());
    assert_eq!(record.client_id, "issued-one");
}

#[tokio::test]
async fn switching_and_logout_do_not_retarget_bound_clones() {
    let mock = Mock::start().await;
    let directory = tempfile::tempdir().unwrap();
    let manager = mock.manager(directory.path());
    let first = login(&mock, &manager, "issued-one").await;
    let bound = manager.select(&first.registration_id).await.unwrap();
    let second = login(&mock, &manager, "issued-two").await;
    assert_eq!(first.email, second.email);
    assert_ne!(first.registration_id, second.registration_id);
    manager.select(&second.registration_id).await.unwrap();
    assert_eq!(bound.binding(), first.registration_id);
    let outcome = manager.logout(&second.registration_id).await.unwrap();
    assert!(outcome.remote_revocation_confirmed);
    assert!(manager.selected().unwrap().is_none());
    assert!(manager
        .read(&second.registration_id)
        .unwrap()
        .tokens
        .is_none());
    assert!(bound.access_token().await.is_ok());
    let pending = manager
        .begin_login(Some(&second.registration_id), false)
        .await
        .unwrap();
    assert!(!reqwest::Url::parse(pending.authorization_url())
        .unwrap()
        .query_pairs()
        .any(|(k, _)| k == "id_token_hint"));
}

#[tokio::test]
async fn revoked_scope_is_atomically_saved_and_blocks_further_tokens() {
    let mock = Mock::start().await;
    mock.state.lock().unwrap().short_expiry = true;
    let directory = tempfile::tempdir().unwrap();
    let manager = mock.manager(directory.path());
    let info = login(&mock, &manager, "issued-one").await;
    let bound = manager.select(&info.registration_id).await.unwrap();
    mock.state.lock().unwrap().fault = Fault::Scope;
    assert_eq!(
        bound.access_token().await.unwrap_err(),
        AuthError::PlanPermissionMissing
    );
    assert_eq!(
        bound.access_token().await.unwrap_err(),
        AuthError::PlanPermissionMissing
    );
    assert_eq!(mock.refreshes.load(Ordering::SeqCst), 1);
    assert!(
        !manager
            .read(&info.registration_id)
            .unwrap()
            .info()
            .plan_enabled
    );
}

// Invoked only by the process-boundary test below. Environment carries fixture
// paths and a loopback origin, never credentials.
#[tokio::test]
async fn refresh_subprocess_helper() {
    let Some(directory) = std::env::var_os("PIKU_AUTH_FIXTURE_DIRECTORY") else {
        return;
    };
    let url = std::env::var("PIKU_AUTH_FIXTURE_ORIGIN").unwrap();
    assert!(url.starts_with("http://127.0.0.1:"));
    let manager = AuthManager::with_endpoints(
        PathBuf::from(directory),
        Endpoints {
            issuer: url.clone(),
            authorize: format!("{url}/authorize"),
            token: format!("{url}/token"),
        },
    )
    .unwrap();
    assert_eq!(
        manager
            .selected()
            .unwrap()
            .unwrap()
            .access_token()
            .await
            .unwrap()
            .expose_secret(),
        "synthetic-access-rotated"
    );
}

#[tokio::test]
async fn separate_process_refresh_reloads_rotated_registration() {
    let mock = Mock::start().await;
    mock.state.lock().unwrap().short_expiry = true;
    let directory = tempfile::tempdir().unwrap();
    let manager = mock.manager(directory.path());
    let info = login(&mock, &manager, "issued-one").await;
    let bound = manager.select(&info.registration_id).await.unwrap();
    let mut child = tokio::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "chatgpt_auth::tests::refresh_subprocess_helper"])
        .env("PIKU_AUTH_FIXTURE_DIRECTORY", &manager.owner.directory)
        .env("PIKU_AUTH_FIXTURE_ORIGIN", &mock.url)
        .kill_on_drop(true)
        .spawn()
        .unwrap();
    // The child owns the file lock and its refresh HTTP request is in flight.
    tokio::time::timeout(Duration::from_secs(10), async {
        while mock.refreshes.load(Ordering::SeqCst) == 0 {
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
    })
    .await
    .unwrap();
    let token = bound.access_token().await.unwrap();
    assert_eq!(token.expose_secret(), "synthetic-access-rotated");
    assert!(tokio::time::timeout(Duration::from_secs(10), child.wait())
        .await
        .unwrap()
        .unwrap()
        .success());
    assert_eq!(mock.refreshes.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn invalid_refresh_identity_preserves_last_valid_credentials() {
    for fault in [
        Fault::Nonce,
        Fault::Audience,
        Fault::Issuer,
        Fault::Signature,
        Fault::Expired,
        Fault::Subject,
    ] {
        let mock = Mock::start().await;
        mock.state.lock().unwrap().short_expiry = true;
        let directory = tempfile::tempdir().unwrap();
        let manager = mock.manager(directory.path());
        let info = login(&mock, &manager, "issued-one").await;
        let bound = manager.select(&info.registration_id).await.unwrap();
        let path = manager.path(&info.registration_id).unwrap();
        let before = std::fs::read(&path).unwrap();
        mock.state.lock().unwrap().fault = fault;
        assert_eq!(
            bound.access_token().await.unwrap_err(),
            AuthError::InvalidIdentity
        );
        assert_eq!(std::fs::read(path).unwrap(), before);
    }
}

#[tokio::test]
async fn failed_first_exchange_retains_issued_client_for_retry() {
    let mock = Mock::start().await;
    let directory = tempfile::tempdir().unwrap();
    let manager = mock.manager(directory.path());
    let pending = manager.begin_login(None, false).await.unwrap();
    mock.prepare(&pending);
    mock.state.lock().unwrap().fault = Fault::InvalidGrant;
    assert_eq!(
        callback(&manager, pending, None, Some("issued-one"))
            .await
            .unwrap_err(),
        AuthError::SignInRequired
    );
    let accounts = manager.list().await.unwrap();
    assert_eq!(accounts.len(), 1);
    assert!(!accounts[0].signed_in);
    let retry = manager
        .begin_login(Some(&accounts[0].registration_id), false)
        .await
        .unwrap();
    let url = reqwest::Url::parse(retry.authorization_url()).unwrap();
    let query: HashMap<_, _> = url.query_pairs().collect();
    assert_eq!(query["client_id"], "issued-one");
    assert!(!query.contains_key("agent_name_hint"));
    assert!(!query.contains_key("id_token_hint"));
}

#[cfg(unix)]
#[tokio::test]
async fn permissive_or_symlinked_credentials_fail_closed() {
    use std::os::unix::fs::{symlink, PermissionsExt};
    let mock = Mock::start().await;
    let directory = tempfile::tempdir().unwrap();
    let manager = mock.manager(directory.path());
    let info = login(&mock, &manager, "issued-one").await;
    let bound = manager.select(&info.registration_id).await.unwrap();
    let path = manager.path(&info.registration_id).unwrap();
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644)).unwrap();
    assert!(bound.access_token().await.is_err());
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).unwrap();
    let moved = directory.path().join("fixture-target");
    std::fs::rename(&path, &moved).unwrap();
    symlink(&moved, &path).unwrap();
    assert!(bound.access_token().await.is_err());
    assert!(std::fs::symlink_metadata(path)
        .unwrap()
        .file_type()
        .is_symlink());
}

mod browser;
