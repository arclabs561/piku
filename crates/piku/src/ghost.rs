//! Model-backed ghost text: a dim continuation of the draft in the prompt.
//!
//! After a typing pause the editor asks for a suggestion. Three short samples
//! are drawn from a small model, and only the text they agree on is shown, cut
//! back to a word boundary. Samples that diverge early mean the model is
//! unsure, and then nothing is shown. History matches take precedence and cost
//! nothing; this fills in when history has no match.

use std::sync::mpsc::{self, Receiver};
use std::sync::Arc;
use std::time::Duration;

use futures_util::StreamExt as _;
use piku_api::types::{Event, MessageRequest, RequestContent, RequestMessage, SystemBlock};
use piku_api::Provider;

use crate::input_helper::GhostSource;

/// Default suggestion model on `OpenRouter`: fast and about $0.10 per million
/// input tokens.
pub const DEFAULT_OPENROUTER_MODEL: &str = "anthropic/claude-haiku-5.5";
const SAMPLES: usize = 3;
const MAX_TOKENS: u32 = 24;
const MAX_CHARS: usize = 48;
const TIMEOUT: Duration = Duration::from_secs(4);

const SYSTEM: &str = "You predict how a developer will finish the message they are typing to \
a coding agent in a terminal. Reply with only the next few words that continue the draft \
exactly where it stops, with no quotes, no explanation and no repetition of the draft. If \
you cannot tell, reply with nothing.";

/// The suggestion model for this provider: `PIKU_GHOST_MODEL` wins (`off`
/// disables); otherwise `OpenRouter` routes get the default and others none.
#[must_use]
pub fn configured_model(provider: &str) -> Option<String> {
    match std::env::var("PIKU_GHOST_MODEL") {
        Ok(model) if model == "off" || model.is_empty() => None,
        Ok(model) => Some(model),
        Err(_) => (provider == "openrouter").then(|| DEFAULT_OPENROUTER_MODEL.to_string()),
    }
}

/// Suggestions from a model on its own thread, so the editor never waits.
pub struct ModelGhost {
    provider: Arc<dyn Provider + Send + Sync + 'static>,
    model: String,
}

impl ModelGhost {
    #[must_use]
    pub fn new(provider: Box<dyn Provider + Send + Sync + 'static>, model: String) -> Self {
        Self {
            provider: Arc::from(provider),
            model,
        }
    }
}

impl GhostSource for ModelGhost {
    fn request(&mut self, draft: &str) -> Receiver<Option<String>> {
        let (send, receive) = mpsc::channel();
        let provider = Arc::clone(&self.provider);
        let model = self.model.clone();
        let draft = draft.to_string();
        std::thread::spawn(move || {
            let Ok(runtime) = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
            else {
                let _ = send.send(None);
                return;
            };
            let samples = runtime.block_on(async {
                let requests = (0..SAMPLES).map(|_| sample(provider.as_ref(), &model, &draft));
                tokio::time::timeout(TIMEOUT, futures_util::future::join_all(requests))
                    .await
                    .unwrap_or_default()
            });
            let _ = send.send(agreed_continuation(&draft, &samples));
        });
        receive
    }
}

async fn sample(provider: &(dyn Provider + Send + Sync), model: &str, draft: &str) -> String {
    let request = MessageRequest {
        model: model.to_string(),
        max_tokens: MAX_TOKENS,
        messages: vec![RequestMessage {
            role: "user".to_string(),
            content: vec![RequestContent::Text {
                text: format!("Draft so far:\n{draft}"),
            }],
        }],
        system: Some(vec![SystemBlock {
            block_type: "text".to_string(),
            text: SYSTEM.to_string(),
            cache_control: None,
        }]),
        tools: None,
        stream: true,
    };
    let mut text = String::new();
    let mut stream = provider.stream_message(request);
    while let Some(event) = stream.next().await {
        match event {
            Ok(Event::TextDelta { text: delta }) => text.push_str(&delta),
            Ok(_) => {}
            Err(_) => return String::new(),
        }
    }
    text
}

/// The continuation every sample agrees on, or `None` when they disagree.
///
/// Samples are trimmed to their first line and joined to the draft (a sample
/// that echoes the draft is stripped of it). The shared prefix is cut back to
/// the last word boundary unless every sample is identical, and must be at
/// least four visible characters. Fewer than two usable samples is too little
/// evidence to show anything.
#[must_use]
pub fn agreed_continuation(draft: &str, samples: &[String]) -> Option<String> {
    let usable: Vec<String> = samples
        .iter()
        .map(|sample| {
            let line = sample.lines().next().unwrap_or("");
            let line = line.strip_prefix(draft).unwrap_or(line);
            // Keep exactly one separating space when the draft does not end in one.
            let line = line.trim_end();
            if draft.ends_with(' ') {
                line.trim_start().to_string()
            } else if line.starts_with(' ') {
                format!(" {}", line.trim_start())
            } else {
                line.to_string()
            }
        })
        .filter(|line| !line.trim().is_empty())
        .collect();
    if usable.len() < 2 || usable.len() * 2 < samples.len() + 1 {
        return None;
    }
    let first = &usable[0];
    let shared = usable.iter().skip(1).fold(first.len(), |len, other| {
        first
            .char_indices()
            .zip(other.chars())
            .take_while(|((_, a), b)| a == b)
            .last()
            .map_or(0, |((index, c), _)| (index + c.len_utf8()).min(len))
    });
    let mut text = &first[..shared];
    let identical = usable.iter().all(|other| other == first);
    if !identical {
        // Cut back to a word boundary so a half-agreed word is not shown.
        text = text.rfind(' ').map_or("", |space| &text[..space]);
    }
    let text: String = text.chars().take(MAX_CHARS).collect();
    // A lone short word such as "the" is noise, not help.
    (text.trim().chars().count() >= 4).then_some(text)
}

#[cfg(test)]
mod tests {
    use super::agreed_continuation;

    fn samples(items: &[&str]) -> Vec<String> {
        items.iter().map(|s| (*s).to_string()).collect()
    }

    #[test]
    fn identical_samples_are_shown_whole() {
        assert_eq!(
            agreed_continuation("run the ", &samples(&["tests", "tests", "tests"])).as_deref(),
            Some("tests")
        );
    }

    #[test]
    fn shared_prefix_is_cut_back_to_a_word_boundary() {
        let got = agreed_continuation(
            "fix the",
            &samples(&[
                " failing test in parser",
                " failing test in lexer",
                " failing tests",
            ]),
        );
        assert_eq!(got.as_deref(), Some(" failing"));
    }

    #[test]
    fn early_disagreement_shows_nothing() {
        assert_eq!(
            agreed_continuation("add ", &samples(&["logging", "a test", "docs"])),
            None
        );
    }

    #[test]
    fn one_usable_sample_is_not_enough_evidence() {
        assert_eq!(
            agreed_continuation("add ", &samples(&["logging to the parser", "", ""])),
            None
        );
    }

    #[test]
    fn echoed_draft_and_extra_lines_are_dropped() {
        assert_eq!(
            agreed_continuation(
                "explain ",
                &samples(&["explain this error\nmore", "this error", "this error"])
            )
            .as_deref(),
            Some("this error")
        );
    }

    #[test]
    fn a_lone_short_word_is_not_shown() {
        assert_eq!(
            agreed_continuation("add a test for ", &samples(&["the", "the", "the"])),
            None
        );
    }

    #[test]
    fn long_agreement_is_capped() {
        let long = "word ".repeat(30);
        let got = agreed_continuation("x ", &samples(&[&long, &long, &long])).unwrap();
        assert!(got.chars().count() <= super::MAX_CHARS);
    }
}
