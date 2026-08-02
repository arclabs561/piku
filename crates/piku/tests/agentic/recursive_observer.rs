//! One bounded recursive review of playground evidence and the primary judge.

use super::{safe_truncate, CritiqueEntry, LlmClient, Persona};

pub struct RecursiveReview {
    pub judge_observations: Vec<String>,
    pub piku_observations: Vec<String>,
    pub verdict: String,
    pub primary_review_grounded: bool,
}

const SYSTEM: &str = r#"You are the final observer in a terminal-agent evaluation.
You receive evidence from a real piku PTY session plus the primary meta-judge's
review. Do not invent events. Distinguish what the terminal/workspace evidence
shows from what either judge merely asserted.

Respond with only this JSON object:
{
  "judge_observations": ["specific assessment of the primary judge"],
  "piku_observations": ["specific assessment of piku behavior"],
  "verdict": "one concise evidence-calibrated conclusion",
  "primary_review_grounded": true
}

This is the final observer. Do not request another judge or recurse further."#;

pub fn observe(
    llm: &LlmClient,
    persona: &Persona,
    entries: &[CritiqueEntry],
    primary_review: &str,
) -> RecursiveReview {
    let mut evidence = format!(
        "PERSONA: {} — {}\nPRIMARY REVIEW:\n{}\n\nTURNS:\n",
        persona.name,
        persona.description,
        safe_truncate(primary_review, 4_000),
    );
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

    let parsed = llm.call_json(SYSTEM, &evidence);
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
    RecursiveReview {
        judge_observations: strings("judge_observations"),
        piku_observations: strings("piku_observations"),
        verdict: parsed["verdict"]
            .as_str()
            .unwrap_or("recursive observer returned no verdict")
            .to_owned(),
        primary_review_grounded: parsed["primary_review_grounded"].as_bool().unwrap_or(false),
    }
}
