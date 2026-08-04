//! One bounded recursive review of playground evidence and primary attestations.

use std::collections::HashSet;

use super::{
    safe_truncate, CritiqueEntry, LlmClient, ObserverClaimRecord, Persona, ReviewClaimRecord,
};

pub struct RecursiveReview {
    pub judge_observations: Vec<String>,
    pub piku_observations: Vec<String>,
    pub verdict: String,
    /// Typed counter-attestations retained only after complete validation.
    pub claim_assessments: Vec<ObserverClaimRecord>,
    /// Reasons the observer response was rejected as one invalid attestation.
    pub invalid_reasons: Vec<String>,
    /// Why the observation is or is not usable. Only a `valid` status makes
    /// `claim_assessments` evidence about the primary review.
    pub status: &'static str,
}

const SYSTEM: &str = r#"You are the final observer in a terminal-agent evaluation.
You receive evidence from a real piku PTY session plus typed primary-review
attestations. Do not invent events. Distinguish what terminal/workspace evidence
shows from what either judge merely asserted. You must assess every primary
claim exactly once; only use its listed claim ID and only cite supplied turns.

Respond with only this JSON object:
{
  "judge_observations": ["specific assessment of the primary judge"],
  "piku_observations": ["specific assessment of piku behavior"],
  "verdict": "one concise evidence-calibrated conclusion",
  "claim_assessments": [{"target_claim_id": "user-bug-1-1", "disposition": "SUPPORTED|RETRACTED|INCONCLUSIVE", "reason": "string", "evidence_turns": [1]}]
}

This is the final observer. Do not request another judge or recurse further."#;

pub fn observe(
    llm: &LlmClient,
    persona: &Persona,
    entries: &[CritiqueEntry],
    primary_claims: &[ReviewClaimRecord],
) -> RecursiveReview {
    let mut evidence = format!(
        "PERSONA: {} — {}\nPRIMARY CLAIMS:\n",
        persona.name, persona.description,
    );
    for claim in primary_claims {
        evidence.push_str(&format!(
            "CLAIM [{}] verdict={} cites {:?}: {}\n",
            claim.id,
            claim.verdict,
            claim.evidence_turns,
            safe_truncate(&claim.rationale, 500),
        ));
    }
    evidence.push_str("\nTURNS:\n");
    const MAX_TURNS: usize = 12;
    let start = entries.len().saturating_sub(MAX_TURNS);
    if start > 0 {
        evidence.push_str("Earlier turns omitted to keep this recursive review bounded.\n");
    }

    for (index, entry) in entries.iter().enumerate().skip(start) {
        evidence.push_str(&format!(
            "\nTURN {}\nACTION: {}\nVIEWPORT:\n{}\nWORKSPACE: {}\n",
            index + 1,
            entry.action_desc,
            safe_truncate(&entry.screen_text, 1_500),
            safe_truncate(&entry.workspace_diff, 1_500),
        ));
        for observation in &entry.observations {
            evidence.push_str(&format!("AGENT OBSERVATION: {observation}\n"));
        }
        for bug in &entry.bugs {
            evidence.push_str(&format!(
                "AGENT BUG [{}]: {}\n",
                bug.severity, bug.description
            ));
        }
        for finding in &entry.deterministic_findings {
            evidence.push_str(&format!(
                "DETERMINISTIC [{}]: {}\n",
                finding.severity, finding.description
            ));
        }
    }

    let outcome = llm.call_json(SYSTEM, &evidence);
    let Some(parsed) = outcome.value() else {
        return RecursiveReview {
            judge_observations: Vec::new(),
            piku_observations: Vec::new(),
            verdict: format!(
                "recursive observer unavailable ({}): {}",
                outcome.status(),
                safe_truncate(outcome.detail(), 200)
            ),
            claim_assessments: Vec::new(),
            invalid_reasons: Vec::new(),
            status: outcome.status(),
        };
    };
    let strings = |field: &str| {
        parsed[field]
            .as_array()
            .map(|items| {
                items
                    .iter()
                    .filter_map(|item| item.as_str().map(str::to_owned))
                    .collect()
            })
            .unwrap_or_default()
    };
    let primary_ids = primary_claims
        .iter()
        .map(|claim| claim.id.clone())
        .collect();
    let (claim_assessments, invalid_reasons) =
        match validate_claim_assessments(parsed, entries.len(), &primary_ids) {
            Ok(claims) => (claims, Vec::new()),
            Err(reasons) => (Vec::new(), reasons),
        };
    let status = if invalid_reasons.is_empty() {
        outcome.status()
    } else {
        "invalid"
    };
    RecursiveReview {
        judge_observations: strings("judge_observations"),
        piku_observations: strings("piku_observations"),
        verdict: parsed["verdict"]
            .as_str()
            .unwrap_or("recursive observer returned no verdict")
            .to_owned(),
        claim_assessments,
        invalid_reasons,
        status,
    }
}

/// Validate the observer's complete, source-preserving assessment set.
///
/// The observer is one attestation. Any unknown, duplicate, uncited, or
/// omitted primary claim invalidates the whole response, so no plausible subset
/// can silently influence a later handoff.
fn validate_claim_assessments(
    review: &serde_json::Value,
    entry_count: usize,
    primary_ids: &HashSet<String>,
) -> Result<Vec<ObserverClaimRecord>, Vec<String>> {
    let mut records = Vec::new();
    let mut rejected = Vec::new();
    let mut seen_ids = HashSet::new();
    for assessment in review["claim_assessments"].as_array().into_iter().flatten() {
        let target = assessment["target_claim_id"].as_str();
        let known = target.is_some_and(|id| primary_ids.contains(id));
        let duplicate = target.is_some_and(|id| !seen_ids.insert(id.to_string()));
        let valid_disposition = matches!(
            assessment["disposition"].as_str(),
            Some("SUPPORTED" | "RETRACTED" | "INCONCLUSIVE")
        );
        let cited = cites_only_real_turns(&assessment["evidence_turns"], entry_count);
        if known && !duplicate && valid_disposition && cited {
            records.push(ObserverClaimRecord {
                target_claim_id: target.unwrap_or_default().to_string(),
                disposition: assessment["disposition"]
                    .as_str()
                    .unwrap_or_default()
                    .to_string(),
                rationale: safe_truncate(assessment["reason"].as_str().unwrap_or_default(), 500)
                    .to_string(),
                evidence_turns: assessment["evidence_turns"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .filter_map(serde_json::Value::as_u64)
                    .collect(),
            });
        } else {
            let identity = match target {
                Some(id) if duplicate => format!("duplicate primary claim {id}"),
                Some(id) if !known => format!("unknown primary claim {id}"),
                Some(id) => id.to_string(),
                None => "missing primary claim id".to_string(),
            };
            rejected.push(format!("{identity}: invalid observer assessment"));
        }
    }
    for missing in primary_ids.difference(&seen_ids) {
        rejected.push(format!("primary claim {missing} was not assessed"));
    }
    if rejected.is_empty() {
        Ok(records)
    } else {
        Err(rejected)
    }
}

fn cites_only_real_turns(turns: &serde_json::Value, entry_count: usize) -> bool {
    turns.as_array().is_some_and(|turns| {
        !turns.is_empty()
            && turns.iter().all(|turn| {
                turn.as_u64()
                    .is_some_and(|turn| turn >= 1 && turn <= entry_count as u64)
            })
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn malformed_assessment_invalidates_the_whole_observer_record() {
        let response = serde_json::json!({
            "claim_assessments": [
                {"target_claim_id": "user-bug-1-1", "disposition": "SUPPORTED", "reason": "first", "evidence_turns": [1]},
                {"target_claim_id": "user-bug-1-1", "disposition": "SUPPORTED", "reason": "duplicate", "evidence_turns": [1]},
                {"target_claim_id": "unknown", "disposition": "RETRACTED", "reason": "bad", "evidence_turns": [1]},
                {"target_claim_id": "user-bug-2-1", "disposition": "INCONCLUSIVE", "reason": "uncited", "evidence_turns": []}
            ]
        });
        let primary_ids = HashSet::from([
            "user-bug-1-1".to_string(),
            "user-bug-2-1".to_string(),
            "user-bug-3-1".to_string(),
        ]);

        let rejected = validate_claim_assessments(&response, 1, &primary_ids).unwrap_err();

        assert!(rejected
            .iter()
            .any(|reason| reason.contains("unknown primary claim")));
        assert!(rejected
            .iter()
            .any(|reason| reason.contains("duplicate primary claim")));
        assert!(rejected
            .iter()
            .any(|reason| reason.contains("user-bug-2-1")));
        assert!(rejected
            .iter()
            .any(|reason| reason.contains("user-bug-3-1 was not assessed")));
    }

    #[test]
    fn complete_cited_assessments_are_retained() {
        let response = serde_json::json!({
            "claim_assessments": [
                {"target_claim_id": "user-bug-1-1", "disposition": "SUPPORTED", "reason": "turn one", "evidence_turns": [1]},
                {"target_claim_id": "user-bug-2-1", "disposition": "INCONCLUSIVE", "reason": "turn two", "evidence_turns": [2]}
            ]
        });
        let primary_ids = HashSet::from(["user-bug-1-1".to_string(), "user-bug-2-1".to_string()]);

        let retained = validate_claim_assessments(&response, 2, &primary_ids).unwrap();

        assert_eq!(retained.len(), 2);
        assert_eq!(retained[1].disposition, "INCONCLUSIVE");
    }
}
