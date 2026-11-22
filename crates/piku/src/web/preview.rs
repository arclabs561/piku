//! Single-use capability tickets for sandboxed HTML previews.
//!
//! The workspace shell runs under a strict script policy, so previews cannot be
//! `srcdoc` documents (they inherit the parent policy). Instead the
//! authenticated shell deposits preview HTML here and receives an unguessable,
//! short-lived ticket; the iframe navigates to `/preview/<ticket>` without the
//! launch credential, and the response carries its own sandboxing policy.
use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// Preview documents run their own inline scripts inside an opaque origin
/// (`sandbox allow-scripts`) and may not fetch, frame, or submit anywhere.
pub(super) const PREVIEW_CSP: &str = "default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; img-src data: blob:; font-src data:; connect-src 'none'; media-src data: blob:; frame-src 'none'; form-action 'none'; base-uri 'none'; frame-ancestors 'self'; sandbox allow-scripts";
pub(super) const MAX_PREVIEW_BYTES: usize = 16 * 1024 * 1024;
const PREVIEW_TTL: Duration = Duration::from_secs(60);
const MAX_PENDING_PREVIEWS: usize = 32;
const MAX_PENDING_BYTES: usize = 64 * 1024 * 1024;

struct Pending {
    html: String,
    expires_at: Instant,
}

#[derive(Default)]
pub(super) struct PreviewStore {
    pending: Mutex<HashMap<String, Pending>>,
}

impl PreviewStore {
    /// Stores `html` and returns a fresh 256-bit hex ticket. Expired entries are
    /// purged first; if the store is still over its bounds the oldest pending
    /// previews (typically for frames already discarded) are evicted.
    pub(super) fn insert(&self, html: String, now: Instant) -> anyhow::Result<String> {
        anyhow::ensure!(html.len() <= MAX_PREVIEW_BYTES, "preview too large");
        let mut bytes = [0_u8; 32];
        getrandom::fill(&mut bytes).map_err(|_| anyhow::anyhow!("preview entropy unavailable"))?;
        let mut ticket = String::with_capacity(64);
        for byte in bytes {
            use std::fmt::Write as _;
            write!(&mut ticket, "{byte:02x}").expect("writing into String cannot fail");
        }
        let mut pending = self.pending.lock().expect("preview store lock");
        pending.retain(|_, entry| entry.expires_at > now);
        let mut held: usize = pending.values().map(|entry| entry.html.len()).sum();
        while pending.len() >= MAX_PENDING_PREVIEWS || held + html.len() > MAX_PENDING_BYTES {
            let Some(oldest) = pending
                .iter()
                .min_by_key(|(_, entry)| entry.expires_at)
                .map(|(ticket, _)| ticket.clone())
            else {
                break;
            };
            if let Some(evicted) = pending.remove(&oldest) {
                held -= evicted.html.len();
            }
        }
        pending.insert(
            ticket.clone(),
            Pending {
                html,
                expires_at: now + PREVIEW_TTL,
            },
        );
        Ok(ticket)
    }

    /// Redeems a ticket exactly once. Unknown, reused, and expired tickets all
    /// return `None`; the entry is removed whether or not it was still valid.
    pub(super) fn take(&self, ticket: &str, now: Instant) -> Option<String> {
        let entry = self
            .pending
            .lock()
            .expect("preview store lock")
            .remove(ticket)?;
        (entry.expires_at > now).then_some(entry.html)
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.pending.lock().unwrap().len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tickets_expire_and_redeem_once() {
        let store = PreviewStore::default();
        let start = Instant::now();
        let fresh = store.insert("fresh".into(), start).unwrap();
        let stale = store.insert("stale".into(), start).unwrap();
        assert_ne!(fresh, stale);
        assert_eq!(fresh.len(), 64);
        assert_eq!(store.take(&fresh, start).as_deref(), Some("fresh"));
        assert_eq!(store.take(&fresh, start), None, "reuse is rejected");
        assert_eq!(store.take(&stale, start + PREVIEW_TTL), None, "expired");
        assert_eq!(store.len(), 0, "expired lookups still remove the entry");
        assert_eq!(store.take("unknown", start), None);
    }

    #[test]
    fn pending_previews_are_bounded_by_count_and_bytes() {
        let store = PreviewStore::default();
        let start = Instant::now();
        let first = store.insert("first".into(), start).unwrap();
        for index in 1..=MAX_PENDING_PREVIEWS {
            let later = start + Duration::from_millis(u64::try_from(index).unwrap());
            store.insert(format!("p{index}"), later).unwrap();
        }
        assert_eq!(store.len(), MAX_PENDING_PREVIEWS);
        assert_eq!(store.take(&first, start), None, "oldest was evicted");

        let big = "x".repeat(MAX_PREVIEW_BYTES);
        let mut tickets = Vec::new();
        for index in 0..8 {
            let at = start + Duration::from_secs(1) + Duration::from_millis(index);
            tickets.push(store.insert(big.clone(), at).unwrap());
        }
        let held: usize = store
            .pending
            .lock()
            .unwrap()
            .values()
            .map(|entry| entry.html.len())
            .sum();
        assert!(held <= MAX_PENDING_BYTES);
        assert!(store.take(tickets.last().unwrap(), start).is_some());
        assert!(store
            .insert("x".repeat(MAX_PREVIEW_BYTES + 1), start)
            .is_err());
    }
}
