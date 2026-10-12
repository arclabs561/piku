use super::*;

fn browser_client() -> reqwest::Client {
    reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .unwrap()
}
async fn form(client: &reqwest::Client, url: &str) -> String {
    let response = client.get(url).send().await.unwrap();
    assert_eq!(response.status(), 200);
    assert_eq!(response.headers()["cache-control"], "no-store");
    assert_eq!(response.headers()["referrer-policy"], "no-referrer");
    let text = response.text().await.unwrap();
    assert!(!text.contains("id_token_hint"));
    text.split("name=\"confirm\" value=\"")
        .nth(1)
        .unwrap()
        .split('"')
        .next()
        .unwrap()
        .into()
}
async fn continue_login(client: &reqwest::Client, url: &str, nonce: &str) -> reqwest::Response {
    client
        .post(url)
        .header(
            "origin",
            reqwest::Url::parse(url)
                .unwrap()
                .origin()
                .ascii_serialization(),
        )
        .form(&[("confirm", nonce)])
        .send()
        .await
        .unwrap()
}

#[tokio::test]
async fn bootstrap_waits_for_confirmed_post_and_finishes_native_oidc() {
    let mock = Mock::start().await;
    let directory = tempfile::tempdir().unwrap();
    let manager = mock.manager(directory.path());
    let attempt = manager.prepare_browser_login(None, false).await.unwrap();
    let url = attempt.bootstrap_url().to_owned();
    assert!(!format!("{attempt:?}").contains(&url));
    assert_eq!(mock.requests.load(Ordering::SeqCst), 0);
    assert!(manager.prepare_browser_login(None, false).await.is_err());
    let runner = tokio::spawn(attempt.run());
    let client = browser_client();
    let nonce = form(&client, &url).await;
    assert_eq!(mock.requests.load(Ordering::SeqCst), 0);
    assert!(!manager.owner.directory.join("host.json").exists());
    assert_eq!(
        manager.login_status().unwrap().unwrap().phase,
        LoginPhase::WaitingForBrowser
    );
    let rejected = continue_login(&client, &url, "wrong").await;
    assert_eq!(rejected.status(), 403);
    let missing_origin = client
        .post(&url)
        .form(&[("confirm", &nonce)])
        .send()
        .await
        .unwrap();
    assert_eq!(missing_origin.status(), 403);
    assert_eq!(mock.requests.load(Ordering::SeqCst), 0);
    let redirect = continue_login(&client, &url, &nonce).await;
    assert_eq!(redirect.status(), 303);
    let authorize = reqwest::Url::parse(redirect.headers()["location"].to_str().unwrap()).unwrap();
    let query: HashMap<_, _> = authorize
        .query_pairs()
        .map(|(k, v)| (k.into_owned(), v.into_owned()))
        .collect();
    assert!(!query.contains_key("id_token_hint"));
    mock.state.lock().unwrap().nonce = query["nonce"].clone();
    let mut callback = reqwest::Url::parse(&query["redirect_uri"]).unwrap();
    callback
        .query_pairs_mut()
        .append_pair("state", &query["state"])
        .append_pair("code", "synthetic-code")
        .append_pair("client_id", "issued-browser");
    assert_eq!(client.get(callback).send().await.unwrap().status(), 200);
    let info = runner.await.unwrap().unwrap();
    assert!(info.signed_in && info.plan_enabled);
    assert!(manager.selected().unwrap().is_none());
    let status = manager.login_status().unwrap().unwrap();
    assert_eq!(status.phase, LoginPhase::Completed);
    assert_eq!(
        status.registration_id.as_deref(),
        Some(info.registration_id.as_str())
    );
    assert!(!manager.cancel_login().unwrap());
    assert!(client.get(url).send().await.is_err());
}

#[tokio::test]
async fn returning_bootstrap_omits_hint_and_does_not_replace_selection_or_block_refresh() {
    let mock = Mock::start().await;
    mock.state.lock().unwrap().short_expiry = true;
    let directory = tempfile::tempdir().unwrap();
    let manager = mock.manager(directory.path());
    let info = login(&mock, &manager, "issued-returning").await;
    let bound = manager.select(&info.registration_id).await.unwrap();
    let attempt = manager
        .prepare_browser_login(Some(info.registration_id.clone()), false)
        .await
        .unwrap();
    // A waiting browser attempt owns a different lock from token rotation.
    assert_eq!(
        bound.access_token().await.unwrap().expose_secret(),
        "synthetic-access-rotated"
    );
    let url = attempt.bootstrap_url().to_owned();
    let runner = tokio::spawn(attempt.run());
    let client = browser_client();
    let nonce = form(&client, &url).await;
    let redirect = continue_login(&client, &url, &nonce).await;
    let location = redirect.headers()["location"].to_str().unwrap();
    assert!(!location.contains("id_token_hint"));
    assert!(!location.contains("synthetic-access"));
    let url = reqwest::Url::parse(location).unwrap();
    let query: HashMap<_, _> = url.query_pairs().collect();
    assert_eq!(query["client_id"], "issued-returning");
    assert!(!query.contains_key("agent_name_hint"));
    assert!(manager.cancel_login().unwrap());
    assert_eq!(runner.await.unwrap().unwrap_err(), AuthError::Cancelled);
    assert_eq!(
        manager.selected().unwrap().unwrap().binding(),
        info.registration_id
    );
    assert_eq!(
        bound.access_token().await.unwrap().expose_secret(),
        "synthetic-access-rotated"
    );
}

#[tokio::test]
async fn cancellation_expiry_drop_and_retry_keep_attempts_isolated() {
    let mock = Mock::start().await;
    let directory = tempfile::tempdir().unwrap();
    let manager = mock.manager(directory.path());
    let first = manager.prepare_browser_login(None, false).await.unwrap();
    assert!(manager.cancel_login().unwrap());
    assert_eq!(first.run().await.unwrap_err(), AuthError::Cancelled);
    assert_eq!(
        manager.login_status().unwrap().unwrap().phase,
        LoginPhase::Cancelled
    );
    let mut second = manager.prepare_browser_login(None, false).await.unwrap();
    second.deadline = tokio::time::Instant::now();
    assert_eq!(second.run().await.unwrap_err(), AuthError::TimedOut);
    assert_eq!(
        manager.login_status().unwrap().unwrap().phase,
        LoginPhase::Expired
    );
    let third = manager.prepare_browser_login(None, false).await.unwrap();
    let third_id = manager.login_status().unwrap().unwrap().attempt_id;
    drop(third);
    assert_eq!(
        manager.login_status().unwrap().unwrap().phase,
        LoginPhase::Cancelled
    );
    let fourth = manager.prepare_browser_login(None, false).await.unwrap();
    assert_ne!(
        manager.login_status().unwrap().unwrap().attempt_id,
        third_id
    );
    assert_eq!(
        manager.login_status().unwrap().unwrap().phase,
        LoginPhase::WaitingForBrowser
    );
    drop(fourth);
    assert_eq!(mock.requests.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn bootstrap_rejects_cross_origin_and_capability_replay_without_network() {
    let mock = Mock::start().await;
    let directory = tempfile::tempdir().unwrap();
    let manager = mock.manager(directory.path());
    let attempt = manager.prepare_browser_login(None, false).await.unwrap();
    let url = attempt.bootstrap_url().to_owned();
    let runner = tokio::spawn(attempt.run());
    let client = browser_client();
    let nonce = form(&client, &url).await;
    let response = client
        .post(&url)
        .header("origin", "https://attacker.invalid")
        .form(&[("confirm", &nonce)])
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 403);
    let response = client
        .get(&url)
        .header("host", "attacker.invalid")
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 404);
    let wrong_path = format!("{url}-wrong");
    assert_eq!(client.get(wrong_path).send().await.unwrap().status(), 404);
    assert_eq!(mock.requests.load(Ordering::SeqCst), 0);
    assert!(manager.cancel_login().unwrap());
    assert_eq!(runner.await.unwrap().unwrap_err(), AuthError::Cancelled);
    assert!(client.get(url).send().await.is_err());
}

#[tokio::test]
async fn failed_callback_persists_safe_failure_and_releases_single_flight() {
    let mock = Mock::start().await;
    let directory = tempfile::tempdir().unwrap();
    let manager = mock.manager(directory.path());
    let attempt = manager.prepare_browser_login(None, false).await.unwrap();
    let url = attempt.bootstrap_url().to_owned();
    let runner = tokio::spawn(attempt.run());
    let client = browser_client();
    let nonce = form(&client, &url).await;
    let redirect = continue_login(&client, &url, &nonce).await;
    let authorize = reqwest::Url::parse(redirect.headers()["location"].to_str().unwrap()).unwrap();
    let query: HashMap<_, _> = authorize.query_pairs().collect();
    let mut callback = reqwest::Url::parse(&query["redirect_uri"]).unwrap();
    callback
        .query_pairs_mut()
        .append_pair("state", "wrong")
        .append_pair("code", "synthetic-code")
        .append_pair("client_id", "issued-browser");
    let _ = client.get(callback).send().await;
    assert_eq!(
        runner.await.unwrap().unwrap_err(),
        AuthError::InvalidCallback
    );
    let status = manager.login_status().unwrap().unwrap();
    assert_eq!(status.phase, LoginPhase::Failed);
    assert!(status.registration_id.is_none());
    assert!(manager.list().await.unwrap().is_empty());
    let raw = std::fs::read_to_string(manager.owner.directory.join("login-status.json")).unwrap();
    for forbidden in ["synthetic", "http:", "code", "nonce", "token", "client_id"] {
        assert!(!raw.contains(forbidden));
    }
    drop(manager.prepare_browser_login(None, false).await.unwrap());
}
