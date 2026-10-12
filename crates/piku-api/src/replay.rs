//! Provider output retained for stateless continuation, not presentation text.
use std::collections::BTreeMap;
use std::fmt;

use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReplayRoute {
    ChatgptResponses,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReplayEnvelope {
    pub version: u32,
    pub route: ReplayRoute,
    pub registration: String,
    pub model: String,
    pub output: Vec<ResponsesOutputItem>,
}

impl fmt::Debug for ReplayEnvelope {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ReplayEnvelope")
            .field("version", &self.version)
            .field("route", &self.route)
            .field("output_items", &self.output.len())
            .finish_non_exhaustive()
    }
}

/// Known replayable output variants. Additional fields survive round trips;
/// unknown item kinds are rejected instead of silently losing continuation.
#[derive(Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponsesOutputItem {
    Message {
        id: String,
        role: String,
        status: String,
        content: Vec<ResponseContent>,
        #[serde(skip_serializing_if = "Option::is_none")]
        phase: Option<String>,
        #[serde(flatten)]
        extra: BTreeMap<String, Value>,
    },
    Reasoning {
        id: String,
        summary: Vec<ReasoningSummary>,
        #[serde(skip_serializing_if = "Option::is_none")]
        encrypted_content: Option<String>,
        #[serde(flatten)]
        extra: BTreeMap<String, Value>,
    },
    FunctionCall {
        #[serde(skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        call_id: String,
        name: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        namespace: Option<String>,
        arguments: String,
        #[serde(flatten)]
        extra: BTreeMap<String, Value>,
    },
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseContent {
    OutputText {
        text: String,
        annotations: Vec<Value>,
        #[serde(flatten)]
        extra: BTreeMap<String, Value>,
    },
    Refusal {
        refusal: String,
        #[serde(flatten)]
        extra: BTreeMap<String, Value>,
    },
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ReasoningSummary {
    SummaryText {
        text: String,
        #[serde(flatten)]
        extra: BTreeMap<String, Value>,
    },
}
