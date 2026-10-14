//! Explicit account-bound subscription route; no discovery or paid fallback.
use std::pin::Pin;
use std::sync::Arc;

use futures_util::{Stream, TryStreamExt};
use piku_api::{ApiError, ChatgptModel, Event, MessageRequest, Provider, ResponsesProvider};

use crate::chatgpt_auth::{default_directory, AuthManager};

pub(super) const NOTICE: &str = "ChatGPT subscription funding; allowance and credits unknown; no hard output-token cap for foreground, children, or memory calls";

pub(super) fn selected_provider() -> Result<ResponsesProvider, ApiError> {
    let manager =
        AuthManager::open(default_directory().map_err(auth_error)?).map_err(auth_error)?;
    let account = manager.selected().map_err(auth_error)?.ok_or_else(|| {
        ApiError::Provider("no ChatGPT account selected; use `piku auth chatgpt login`, then `piku auth chatgpt select ACCOUNT_ID`".into())
    })?;
    ResponsesProvider::new(Arc::new(account))
}

fn auth_error(error: crate::chatgpt_auth::AuthError) -> ApiError {
    ApiError::Provider(error.to_string())
}

#[derive(Clone)]
pub(super) struct ChatgptProvider {
    inner: ResponsesProvider,
    catalog: Arc<tokio::sync::OnceCell<Vec<ChatgptModel>>>,
}
impl ChatgptProvider {
    pub(super) fn selected() -> Result<Self, ApiError> {
        Ok(Self {
            inner: selected_provider()?,
            catalog: Arc::new(tokio::sync::OnceCell::new()),
        })
    }
    pub(super) fn selected_with_observer(
        observer: Arc<piku_api::responses::ResponsesObserver>,
    ) -> Result<Self, ApiError> {
        let mut provider = Self::selected()?;
        provider.inner = provider.inner.with_observer(observer);
        Ok(provider)
    }
    async fn validate_model(&self, model: &str) -> Result<(), ApiError> {
        if model.trim().is_empty() {
            return Err(ApiError::Provider("ChatGPT requires an explicit model; run `piku auth chatgpt models`, then pass `--provider chatgpt --model MODEL`".into()));
        }
        let catalog = self.catalog.get_or_try_init(|| self.inner.models()).await?;
        if !catalog.iter().any(|candidate| candidate.slug == model) {
            return Err(ApiError::Provider("selected model is not available in the ChatGPT account catalog; run `piku auth chatgpt models` and select a listed model".into()));
        }
        Ok(())
    }
}
impl Provider for ChatgptProvider {
    fn name(&self) -> &'static str {
        "chatgpt"
    }
    fn boxed_clone(&self) -> Box<dyn Provider + Send + Sync + 'static> {
        Box::new(self.clone())
    }
    fn observed_clone(
        &self,
        observer: Arc<piku_api::InferenceObserver>,
    ) -> Result<Box<dyn Provider>, ApiError> {
        let mut cloned = self.clone();
        cloned.inner = cloned.inner.with_observer(observer);
        Ok(Box::new(cloned))
    }
    fn stream_message(
        &self,
        request: MessageRequest,
    ) -> Pin<Box<dyn Stream<Item = Result<Event, ApiError>> + Send + '_>> {
        Box::pin(
            futures_util::stream::once(async move {
                self.inner.validate_request(&request)?;
                self.validate_model(&request.model).await?;
                Ok::<_, ApiError>(self.inner.stream_message(request))
            })
            .try_flatten(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures_util::StreamExt;

    struct NeverToken;
    impl piku_api::AccessTokenSource for NeverToken {
        fn token(
            &self,
        ) -> Pin<
            Box<
                dyn std::future::Future<Output = Result<piku_api::AccessToken, ApiError>>
                    + Send
                    + '_,
            >,
        > {
            panic!("invalid model must be rejected before credential or inference access")
        }
        fn binding(&self) -> &'static str {
            "test-registration"
        }
    }

    #[tokio::test]
    async fn missing_and_unlisted_models_refuse_before_inference() {
        let observations = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let observed = Arc::clone(&observations);
        let provider = ChatgptProvider {
            inner: ResponsesProvider::new(Arc::new(NeverToken))
                .unwrap()
                .with_observer(Arc::new(move |_| {
                    observed.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                })),
            catalog: Arc::new(tokio::sync::OnceCell::new()),
        };
        assert!(provider
            .validate_model("")
            .await
            .unwrap_err()
            .to_string()
            .contains("explicit model"));
        provider
            .catalog
            .set(vec![ChatgptModel {
                slug: "visible-model".into(),
                display_name: "Visible".into(),
            }])
            .unwrap();
        let cloned = provider.clone();
        assert!(Arc::ptr_eq(&provider.catalog, &cloned.catalog));
        assert!(cloned.validate_model("visible-model").await.is_ok());
        let request = MessageRequest {
            model: "hidden-or-mistyped-model".into(),
            messages: vec![],
            system: None,
            tools: None,
            max_tokens: 0,
            stream: true,
        };
        let mut stream = cloned.stream_message(request);
        assert!(stream
            .next()
            .await
            .unwrap()
            .unwrap_err()
            .to_string()
            .contains("not available"));
        assert!(stream.next().await.is_none());
        assert_eq!(observations.load(std::sync::atomic::Ordering::Relaxed), 0);
    }
    #[tokio::test]
    async fn replay_mismatch_refuses_before_catalog_credentials() {
        for (registration, model) in [
            ("other-registration", "selected-model"),
            ("test-registration", "other-model"),
        ] {
            let provider = ChatgptProvider {
                inner: ResponsesProvider::new(Arc::new(NeverToken)).unwrap(),
                catalog: Arc::new(tokio::sync::OnceCell::new()),
            };
            let request = MessageRequest {
                model: "selected-model".into(),
                max_tokens: 0,
                system: None,
                tools: None,
                stream: true,
                messages: vec![piku_api::RequestMessage {
                    role: "assistant".into(),
                    content: vec![piku_api::RequestContent::Replay {
                        envelope: piku_api::ReplayEnvelope {
                            version: 1,
                            route: piku_api::ReplayRoute::ChatgptResponses,
                            registration: registration.into(),
                            model: model.into(),
                            output: vec![],
                        },
                    }],
                }],
            };
            let mut stream = provider.stream_message(request);
            assert!(stream.next().await.unwrap().is_err());
            assert!(stream.next().await.is_none());
            assert!(provider.catalog.get().is_none());
        }
    }
}
