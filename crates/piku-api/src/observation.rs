//! Opt-in metadata-only observations of local inference POST attempts.
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use crate::ApiError;

static NEXT_ATTEMPT_ID: AtomicU64 = AtomicU64::new(1);

/// Trusted synchronous observation callback. It must not block or panic.
/// Receipts contain no request, credential, account, model or response content.
pub type InferenceObserver = dyn Fn(InferenceObservation) + Send + Sync;

/// One local POST send and its eventual transport/protocol disposition.
/// IDs and monotonic timestamps are meaningful only within this process.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum InferenceObservation {
    Started {
        attempt_id: u64,
        started_at: Instant,
    },
    Finished(InferenceAttemptReceipt),
}

#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct InferenceAttemptReceipt {
    pub attempt_id: u64,
    pub started_at: Instant,
    pub finished_at: Instant,
    pub outcome: InferenceAttemptOutcome,
    /// Valid reported counters, independent of replay/output acceptance.
    /// The first valid sample is retained; duplicate or cumulative samples are
    /// not summed. This does not certify a final billable total.
    /// Missing or malformed usage is unknown. On incomplete transport these
    /// counters are only available reported evidence, not a complete total.
    pub reported_usage: Option<InferenceReportedUsage>,
    /// Whether the successful HTTP response body reached clean EOF.
    pub transport_complete: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum InferenceAttemptOutcome {
    /// Clean transport and accepted protocol output; not task success.
    Completed,
    Failed(InferenceAttemptFailure),
    /// Includes consumer drop, timeout and cancellation, without inferring which.
    Dropped,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum InferenceAttemptFailure {
    Send,
    Http,
    Stream,
    Protocol,
}

/// Counters actually present in valid reported usage. Absent detail stays unknown.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct InferenceReportedUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub cached_input_tokens: Option<u32>,
    pub reasoning_output_tokens: Option<u32>,
}

pub(crate) struct AttemptObservation<'a> {
    observer: Option<&'a Arc<InferenceObserver>>,
    receipt: Option<InferenceAttemptReceipt>,
}

impl<'a> AttemptObservation<'a> {
    pub(crate) fn enabled(&self) -> bool {
        self.receipt.is_some()
    }

    pub(crate) fn start(observer: Option<&'a Arc<InferenceObserver>>) -> Self {
        let receipt = observer.map(|observer| {
            let attempt_id = NEXT_ATTEMPT_ID
                .try_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
                .expect("Inference attempt IDs exhausted");
            let started_at = Instant::now();
            observer(InferenceObservation::Started {
                attempt_id,
                started_at,
            });
            InferenceAttemptReceipt {
                attempt_id,
                started_at,
                finished_at: started_at,
                outcome: InferenceAttemptOutcome::Dropped,
                reported_usage: None,
                transport_complete: false,
            }
        });
        Self { observer, receipt }
    }

    pub(crate) fn finish(&mut self, outcome: InferenceAttemptOutcome) {
        if let Some(mut receipt) = self.receipt.take() {
            receipt.finished_at = Instant::now();
            receipt.outcome = outcome;
            if let Some(observer) = self.observer {
                observer(InferenceObservation::Finished(receipt));
            }
        }
    }

    pub(crate) fn check<T>(
        &mut self,
        result: Result<T, ApiError>,
        failure: InferenceAttemptFailure,
    ) -> Result<T, ApiError> {
        if result.is_err() {
            self.finish(InferenceAttemptOutcome::Failed(failure));
        }
        result
    }

    pub(crate) fn usage(&mut self, usage: Option<InferenceReportedUsage>) {
        if let Some(receipt) = self.receipt.as_mut() {
            if receipt.reported_usage.is_none() {
                receipt.reported_usage = usage;
            }
        }
    }

    pub(crate) fn transport_complete(&mut self) {
        if let Some(receipt) = self.receipt.as_mut() {
            receipt.transport_complete = true;
        }
    }
}

impl Drop for AttemptObservation<'_> {
    fn drop(&mut self) {
        self.finish(InferenceAttemptOutcome::Dropped);
    }
}
