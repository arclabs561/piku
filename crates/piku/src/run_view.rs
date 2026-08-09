//! Read-only projections of Piku's durable semantic run record.

use std::collections::BTreeMap;
use std::fmt::Write;
use std::path::{Component, Path};

use piku_runtime::{audit_run_record, RunContentRef as ContentRef, RunEvent, RunEventEnvelope};
use serde::{Deserialize, Serialize};

/// Surface-neutral evidence corpus used by the browser filter and deterministic
/// projection evaluations. `search_text` is an index, not a display claim.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunSearchEntry {
    pub sequence: u64,
    pub scope: String,
    pub event_kind: String,
    pub preview_text: String,
    pub search_text: String,
    pub full_content_chars: usize,
    pub storage_ref: Option<String>,
    pub byte_count: Option<u64>,
}

/// Build the same artifact-aware search corpus embedded in the HTML workbench.
pub fn build_search_index_with_artifacts(
    events: &[RunEventEnvelope],
    record_path: &Path,
) -> std::io::Result<Vec<RunSearchEntry>> {
    let artifacts = load_artifacts(events, record_path)?;
    build_search_index(events, &artifacts).map_err(std::io::Error::other)
}

#[must_use]
pub fn render_text(events: &[RunEventEnvelope]) -> String {
    let mut output = String::new();
    let session = events.first().map_or("unknown", |event| &event.session_id);
    let audit = audit_run_record(events);
    let status = if audit.is_structurally_complete() {
        "complete"
    } else {
        "incomplete"
    };
    let _ = writeln!(
        output,
        "run {session} · {} events · evidence {status}",
        events.len()
    );
    let _ = writeln!(
        output,
        "{} / {} turns completed · {} failed · {} cancelled · {} / {} tools complete · {} permission decisions",
        audit.completed_turn_count,
        audit.turn_count,
        audit.failed_turn_count,
        audit.cancelled_turn_count,
        audit.tool_calls_completed,
        audit.tool_calls_started,
        audit.tool_calls_with_permission_decision
    );
    let _ = writeln!(
        output,
        "context: {} selected / {} excluded messages · content: {} inline / {} artifact / {} unavailable",
        audit.context.messages_selected,
        audit.context.messages_excluded,
        audit.content.inline_items,
        audit.content.artifact_items,
        audit.content.unavailable_items
    );
    let _ = writeln!(
        output,
        "effects: {} created / {} modified / {} unchanged / {} unknown · verification: {} passed / {} failed / {} indeterminate",
        audit.files_created,
        audit.files_modified,
        audit.file_writes_unchanged,
        audit.file_writes_unknown,
        audit.verification_passed,
        audit.verification_failed,
        audit.verification_indeterminate
    );
    for finding in &audit.findings {
        let _ = writeln!(
            output,
            "audit {:?} {} · {}",
            finding.severity, finding.code, finding.message
        );
    }
    let mut current_scope = "";
    for envelope in events {
        let scope = envelope.scope.turn_id().unwrap_or("run");
        if scope != current_scope {
            current_scope = scope;
            let _ = writeln!(output, "\n{current_scope}");
        }
        match &envelope.event {
            RunEvent::TurnStarted {
                provider,
                model,
                input,
            } => {
                let _ = writeln!(
                    output,
                    "  → user [{} / {model}] {}",
                    provider.as_deref().unwrap_or("unknown"),
                    content_preview(input, 120)
                );
            }
            RunEvent::ContextBuilt { manifest } => {
                let selected = manifest
                    .messages
                    .iter()
                    .filter(|message| message.selected)
                    .count();
                let excluded = manifest.messages.len().saturating_sub(selected);
                let _ = writeln!(
                    output,
                    "  ◇ context {} est. tokens · {selected} selected · {excluded} excluded · {} tools",
                    manifest.estimated_input_tokens,
                    manifest.tools.len()
                );
            }
            RunEvent::ContextSourcesResolved { sources } => {
                let bytes: usize = sources.iter().map(|source| source.byte_size).sum();
                let _ = writeln!(
                    output,
                    "  ◇ {} context sources resolved · {bytes} bytes",
                    sources.len()
                );
                for source in sources {
                    let _ = writeln!(
                        output,
                        "    ↳ {} · {:?} · {} bytes · sha256:{}",
                        source.id,
                        source.trust,
                        source.byte_size,
                        source.output_sha256.as_str()
                    );
                    for reference in &source.sources {
                        let _ = writeln!(
                            output,
                            "      {} · sha256:{}",
                            reference.reference,
                            reference.sha256.as_str()
                        );
                    }
                }
            }
            RunEvent::ContextUnavailable { reason } => {
                let _ = writeln!(output, "  ◇ context unavailable · {reason}");
            }
            RunEvent::CompactionApplied {
                before_messages,
                after_messages,
                masked_tool_results,
                ..
            } => {
                let _ = writeln!(
                    output,
                    "  ↘ compacted {before_messages} → {after_messages} messages · {masked_tool_results} masked results"
                );
            }
            RunEvent::AssistantMessage { content } => {
                let _ = writeln!(output, "  ← assistant {}", content_preview(content, 160));
            }
            RunEvent::ToolStarted {
                name, arguments, ..
            } => {
                let _ = writeln!(output, "  ● {name} {arguments}");
            }
            RunEvent::PermissionDecision { decision, .. } => {
                let _ = writeln!(output, "  ◆ permission {decision:?}");
            }
            RunEvent::ToolCompleted {
                result,
                is_error,
                effects,
                verification,
                ..
            } => {
                let mark = if *is_error { "×" } else { "✓" };
                let _ = writeln!(output, "  {mark} {}", content_preview(result, 120));
                for effect in effects {
                    match effect {
                        piku_runtime::RunToolEffect::FileWrite {
                            path,
                            content_change,
                        } => {
                            let _ = writeln!(
                                output,
                                "    ↳ file {content_change:?} {}",
                                path.display()
                            );
                        }
                        piku_runtime::RunToolEffect::ShellCommand { command, exit_code } => {
                            let _ = writeln!(output, "    ↳ shell exit={exit_code:?} `{command}`");
                        }
                    }
                }
                if let Some(verification) = verification {
                    let _ = writeln!(
                        output,
                        "    ↳ verify {:?} · {}",
                        verification.status, verification.description
                    );
                }
            }
            RunEvent::TurnCompleted { usage, stop_reason } => {
                let accounting = usage.as_ref().map_or_else(
                    || "tokens not reported".to_string(),
                    |usage| format!("{}↑ {}↓", usage.input_tokens, usage.output_tokens),
                );
                let _ = writeln!(
                    output,
                    "  ■ {accounting} · {}",
                    stop_reason.as_deref().unwrap_or("unknown")
                );
            }
            RunEvent::TurnFailed { class, message } => {
                let _ = writeln!(output, "  × failed [{class}] · {message}");
            }
            RunEvent::TurnCancelled { reason } => {
                let _ = writeln!(output, "  ◼ cancelled · {reason}");
            }
            RunEvent::Warning { message } => {
                let _ = writeln!(output, "  ! {message}");
            }
            RunEvent::UserDisposition { disposition, note } => {
                let _ = writeln!(
                    output,
                    "  ◆ user {disposition:?} · {}",
                    content_preview(note, 160)
                );
            }
            RunEvent::ChildRunRef {
                relationship,
                child_session_id,
                task_id,
                run_record_ref,
                ..
            } => {
                let _ = writeln!(
                    output,
                    "  ⇄ {relationship} → child session {child_session_id} (task {task_id}) · run {}",
                    run_record_ref.display()
                );
            }
        }
    }
    output
}

pub fn render_json(events: &[RunEventEnvelope]) -> serde_json::Result<String> {
    serde_json::to_string_pretty(&serde_json::json!({
        "audit": audit_run_record(events),
        "events": events,
    }))
}

pub fn render_html(events: &[RunEventEnvelope]) -> serde_json::Result<String> {
    render_html_document(events, &BTreeMap::new())
}

/// Render a self-contained workbench with referenced text artifacts embedded.
/// Artifact paths are constrained to the run directory before they are read.
pub fn render_html_with_artifacts(
    events: &[RunEventEnvelope],
    record_path: &Path,
) -> std::io::Result<String> {
    let artifacts = load_artifacts(events, record_path)?;
    render_html_document(events, &artifacts).map_err(std::io::Error::other)
}

fn render_html_document(
    events: &[RunEventEnvelope],
    artifacts: &BTreeMap<String, String>,
) -> serde_json::Result<String> {
    let title = events
        .first()
        .map_or("Piku run", |event| event.session_id.as_str());
    let search_index = build_search_index(events, artifacts)?;
    let data = serde_json::to_string(&serde_json::json!({
        "audit": audit_run_record(events),
        "artifacts": artifacts,
        "events": events,
        "search_index": search_index,
    }))?
    .replace('&', "\\u0026")
    .replace('<', "\\u003c")
    .replace('>', "\\u003e")
    .replace('\u{2028}', "\\u2028")
    .replace('\u{2029}', "\\u2029");
    Ok(format!(
        r#"<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; img-src data:">
<title>{}</title>
<style>
:root{{--ink:#171714;--paper:#f2efe5;--muted:#6d6a60;--rule:#b9b4a5;--signal:#c7ff45;--danger:#c74632}}
*{{box-sizing:border-box}}html{{background:var(--ink);color:var(--paper)}}body{{margin:0;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:14px;line-height:1.5}}
header{{position:sticky;top:0;z-index:2;display:grid;grid-template-columns:minmax(16rem,1fr) minmax(15rem,32rem);gap:2rem;align-items:end;padding:1.4rem 2rem 1rem;background:rgba(23,23,20,.94);border-bottom:1px solid #555146;backdrop-filter:blur(10px)}}
h1{{margin:0;font-family:"Iowan Old Style",Georgia,serif;font-size:clamp(1.6rem,4vw,3.6rem);font-weight:500;letter-spacing:-.045em;line-height:.9}}.kicker{{color:var(--signal);font-size:.72rem;letter-spacing:.16em;text-transform:uppercase}}
input{{width:100%;border:0;border-bottom:1px solid var(--rule);padding:.65rem .1rem;background:transparent;color:var(--paper);font:inherit;outline:none}}input:focus{{border-color:var(--signal)}}
main{{display:grid;grid-template-columns:minmax(12rem,19rem) minmax(0,1fr);min-height:calc(100vh - 6rem)}}nav{{position:sticky;top:6.2rem;align-self:start;max-height:calc(100vh - 6.2rem);overflow:auto;padding:1.5rem 1.2rem 3rem 2rem;border-right:1px solid #555146}}
nav button{{display:block;width:100%;margin:0 0 .45rem;border:0;padding:.35rem 0;background:none;color:var(--muted);font:inherit;text-align:left;cursor:pointer}}nav button:hover,nav button:focus{{color:var(--signal);outline:none}}
#workbench{{min-width:0}}#audit{{display:grid;grid-template-columns:repeat(4,minmax(8rem,1fr));gap:1px;margin:2rem clamp(1rem,5vw,6rem) 0;background:#555146;border:1px solid #555146;max-width:78rem}}.metric{{min-height:7rem;padding:1rem;background:#1d1d19}}.metric strong{{display:block;margin-top:.5rem;font-family:"Iowan Old Style",Georgia,serif;font-size:2rem;font-weight:500;line-height:1}}.metric.error strong{{color:#ff7966}}.metric .detail{{color:var(--muted);font-size:.72rem}}#timeline{{padding:2rem clamp(1rem,5vw,6rem) 8rem;max-width:90rem}}article{{display:grid;grid-template-columns:5rem minmax(0,1fr);gap:1.2rem;padding:1.15rem 0;border-top:1px solid #555146;animation:arrive .28s ease-out both;animation-delay:calc(var(--i) * 12ms)}}
.seq{{color:var(--muted);font-size:.72rem}}h2{{margin:0 0 .55rem;font-size:.78rem;letter-spacing:.12em;text-transform:uppercase;color:var(--signal)}}pre{{max-height:28rem;overflow:auto;margin:.4rem 0 0;padding:1rem;background:#22221e;color:#e9e4d7;white-space:pre-wrap;word-break:break-word}}
dl{{display:grid;grid-template-columns:max-content 1fr;gap:.25rem 1rem;margin:.4rem 0}}dt{{color:var(--muted)}}dd{{margin:0}}.summary{{max-width:76ch;margin:.7rem 0;color:#d7d2c6}}details{{margin-top:.6rem}}summary{{cursor:pointer;color:var(--muted)}}.error h2{{color:#ff7966}}.empty{{padding:5rem 0;color:var(--muted)}}
@keyframes arrive{{from{{opacity:0;transform:translateY(5px)}}to{{opacity:1;transform:none}}}}@media(max-width:720px){{header{{grid-template-columns:1fr}}main{{display:block}}nav{{position:static;border-right:0;border-bottom:1px solid #555146;padding:1rem 1.2rem}}#audit{{grid-template-columns:repeat(2,minmax(8rem,1fr))}}article{{grid-template-columns:3rem 1fr}}}}
@media(prefers-reduced-motion:reduce){{article{{animation:none}}}}
</style>
</head>
<body>
<header><div><div class="kicker">durable investigation record</div><h1>{}</h1></div><label><span class="kicker">filter evidence</span><input id="filter" type="search" placeholder="tool, turn, reason, content…" autocomplete="off"></label></header>
<main><nav id="turns" aria-label="Turns"></nav><div id="workbench"><section id="audit" aria-label="Run evidence metrics"></section><section id="timeline" aria-live="polite"></section></div></main>
<script id="run-data" type="application/json">{}</script>
<script>
const documentData=JSON.parse(document.getElementById('run-data').textContent);const events=documentData.events,audit=documentData.audit;const searchBySequence=new Map(documentData.search_index.map(e=>[e.sequence,e]));const timeline=document.getElementById('timeline');const turns=document.getElementById('turns');const filter=document.getElementById('filter');
const text=v=>v==null?'':typeof v==='string'?v:JSON.stringify(v,null,2);const label=e=>e.event.replaceAll('_',' ');const eventText=e=>searchBySequence.get(e.sequence)?.search_text??'';const artifactText=c=>c?.relative_path?documentData.artifacts[c.relative_path]:null;const preview=v=>v&&v.length>320?v.slice(0,320)+'…':v??'';const content=c=>c?.text??(c?.relative_path?preview(artifactText(c))||`${{c.relative_path}} · ${{c.bytes}} bytes`:c?.reason??'');const eventContent=e=>e.input??e.summary??e.content??e.result??e.note;const scopeLabel=e=>e.scope==='run'?'run':e.turn_id;
function eventSummary(e){{switch(e.event){{case'turn_started':return `${{e.provider??'unknown'}} / ${{e.model}} · ${{content(e.input)}}`;case'context_built':{{const m=e.manifest.messages,s=m.filter(x=>x.selected).length;return `${{e.manifest.estimated_input_tokens}} estimated tokens · ${{s}} selected · ${{m.length-s}} excluded · ${{e.manifest.tools.length}} tools`}}case'context_unavailable':return e.reason;case'compaction_applied':return `${{e.before_messages}} → ${{e.after_messages}} messages · ${{e.masked_tool_results}} masked results`;case'assistant_message':return content(e.content);case'tool_started':return `${{e.name}} ${{text(e.arguments)}}`;case'permission_decision':return e.decision;case'tool_completed':{{const facts=[];if(e.effects?.length)facts.push(`${{e.effects.length}} effect${{e.effects.length===1?'':'s'}}`);if(e.verification)facts.push(`verify ${{e.verification.status}}`);return `${{e.is_error?'error':'ok'}} · ${{content(e.result)}}${{facts.length?' · '+facts.join(' · '):''}}`}}case'turn_completed':return `${{e.usage?`${{e.usage.input_tokens}}↑ ${{e.usage.output_tokens}}↓`:'tokens not reported'}} · ${{e.stop_reason??'unknown'}}`;case'turn_failed':return `${{e.class}} · ${{e.message}}`;case'turn_cancelled':return e.reason;case'warning':return e.message;case'user_disposition':return `${{e.disposition}} · ${{content(e.note)}}`;default:return''}}}}
function drawAudit(){{const root=document.getElementById('audit');const cards=[['evidence',audit.findings.some(f=>f.severity==='error')?'incomplete':'complete',`${{audit.findings.length}} findings`],['turns',`${{audit.completed_turn_count}} / ${{audit.turn_count}}`,'completed · '+audit.failed_turn_count+' failed · '+audit.cancelled_turn_count+' cancelled'],['effects',audit.tool_effect_count,`${{audit.files_created}} created · ${{audit.files_modified}} modified`],['verification',audit.verification_count,`${{audit.verification_passed}} passed · ${{audit.verification_failed}} failed · ${{audit.verification_indeterminate}} indeterminate`]];cards.forEach(([name,value,detail])=>{{const d=document.createElement('div');d.className='metric'+(name==='evidence'&&value==='incomplete'?' error':'');const k=document.createElement('div'),v=document.createElement('strong'),p=document.createElement('div');k.className='kicker';k.textContent=name;v.textContent=value;p.className='detail';p.textContent=detail;d.append(k,v,p);root.append(d)}})}}
function draw(q=''){{timeline.replaceChildren();turns.replaceChildren();const seen=new Set();let shown=0;events.forEach((e,i)=>{{if(q&&!eventText(e).includes(q))return;shown++;const scope=scopeLabel(e);if(!seen.has(scope)){{seen.add(scope);const b=document.createElement('button');b.textContent=scope;b.onclick=()=>document.getElementById('event-'+e.sequence)?.scrollIntoView({{behavior:'smooth'}});turns.append(b)}}const a=document.createElement('article');a.id='event-'+e.sequence;a.style.setProperty('--i',i);if(e.is_error)a.className='error';const s=document.createElement('div');s.className='seq';s.textContent=String(e.sequence).padStart(4,'0');const body=document.createElement('div');const h=document.createElement('h2');h.textContent=label(e);body.append(h);const dl=document.createElement('dl');[['scope',scope],['recorded',new Date(e.recorded_at_ms).toLocaleString()]].forEach(([k,v])=>{{const dt=document.createElement('dt'),dd=document.createElement('dd');dt.textContent=k;dd.textContent=v;dl.append(dt,dd)}});body.append(dl);const summary=document.createElement('p');summary.className='summary';summary.textContent=eventSummary(e);body.append(summary);const payload={{...e}};['schema_version','sequence','recorded_at_ms','session_id','scope','turn_id','event'].forEach(k=>delete payload[k]);const d=document.createElement('details');const sum=document.createElement('summary');sum.textContent='inspect payload';const pre=document.createElement('pre');pre.textContent=text(payload);d.append(sum,pre);body.append(d);const resolved=artifactText(eventContent(e));if(resolved!=null){{const ad=document.createElement('details'),as=document.createElement('summary'),ap=document.createElement('pre');as.textContent='inspect full artifact';ap.textContent=resolved;ad.append(as,ap);body.append(ad)}}a.append(s,body);timeline.append(a)}});if(!shown){{const p=document.createElement('p');p.className='empty';p.textContent='No evidence matches this filter.';timeline.append(p)}}}}
filter.addEventListener('input',()=>draw(filter.value.trim().toLowerCase()));document.addEventListener('keydown',e=>{{if(e.key==='/'&&document.activeElement!==filter){{e.preventDefault();filter.focus()}}}});drawAudit();draw();
</script>
</body></html>"#,
        escape_html(title),
        escape_html(title),
        data
    ))
}

fn build_search_index(
    events: &[RunEventEnvelope],
    artifacts: &BTreeMap<String, String>,
) -> serde_json::Result<Vec<RunSearchEntry>> {
    events
        .iter()
        .map(|envelope| {
            let content = event_content(envelope);
            let (full_content, storage_ref, byte_count) = match content {
                Some(ContentRef::Inline { text }) => {
                    (text.as_str(), None, u64::try_from(text.len()).ok())
                }
                Some(ContentRef::Artifact(artifact)) => {
                    let key = artifact.relative_path.display().to_string();
                    (
                        artifacts.get(&key).map_or("", String::as_str),
                        Some(key),
                        Some(artifact.bytes),
                    )
                }
                Some(ContentRef::Unavailable { reason }) => (reason.as_str(), None, None),
                None => ("", None, None),
            };
            let mut search_text = serde_json::to_string(envelope)?.to_lowercase();
            if storage_ref.is_some() && !full_content.is_empty() {
                search_text.push('\n');
                search_text.push_str(&full_content.to_lowercase());
            }
            Ok(RunSearchEntry {
                sequence: envelope.sequence,
                scope: envelope.scope.turn_id().unwrap_or("run").to_string(),
                event_kind: event_kind(&envelope.event).to_string(),
                preview_text: content.map_or_else(
                    || event_kind(&envelope.event).replace('_', " "),
                    |content| content_preview(content, 320),
                ),
                search_text,
                full_content_chars: full_content.chars().count(),
                storage_ref,
                byte_count,
            })
        })
        .collect()
}

fn event_kind(event: &RunEvent) -> &'static str {
    match event {
        RunEvent::TurnStarted { .. } => "turn_started",
        RunEvent::ContextBuilt { .. } => "context_built",
        RunEvent::ContextSourcesResolved { .. } => "context_sources_resolved",
        RunEvent::ContextUnavailable { .. } => "context_unavailable",
        RunEvent::CompactionApplied { .. } => "compaction_applied",
        RunEvent::AssistantMessage { .. } => "assistant_message",
        RunEvent::ToolStarted { .. } => "tool_started",
        RunEvent::PermissionDecision { .. } => "permission_decision",
        RunEvent::ToolCompleted { .. } => "tool_completed",
        RunEvent::TurnCompleted { .. } => "turn_completed",
        RunEvent::TurnFailed { .. } => "turn_failed",
        RunEvent::TurnCancelled { .. } => "turn_cancelled",
        RunEvent::Warning { .. } => "warning",
        RunEvent::UserDisposition { .. } => "user_disposition",
        RunEvent::ChildRunRef { .. } => "child_run_ref",
    }
}

fn load_artifacts(
    events: &[RunEventEnvelope],
    record_path: &Path,
) -> std::io::Result<BTreeMap<String, String>> {
    let base = record_path
        .parent()
        .ok_or_else(|| std::io::Error::other("run record path has no parent"))?;
    let mut artifacts = BTreeMap::new();
    for content in events.iter().filter_map(event_content) {
        let ContentRef::Artifact(artifact) = content else {
            continue;
        };
        if artifact.relative_path.is_absolute()
            || artifact
                .relative_path
                .components()
                .any(|component| !matches!(component, Component::Normal(_)))
        {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "artifact path escapes run directory: {}",
                    artifact.relative_path.display()
                ),
            ));
        }
        let text = std::fs::read_to_string(base.join(&artifact.relative_path))?;
        let actual_bytes = u64::try_from(text.len())
            .map_err(|_| std::io::Error::other("artifact length does not fit in u64"))?;
        if actual_bytes != artifact.bytes {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "artifact size mismatch for {}: recorded {}, found {actual_bytes}",
                    artifact.relative_path.display(),
                    artifact.bytes
                ),
            ));
        }
        artifacts.insert(artifact.relative_path.display().to_string(), text);
    }
    Ok(artifacts)
}

fn event_content(envelope: &RunEventEnvelope) -> Option<&ContentRef> {
    match &envelope.event {
        RunEvent::TurnStarted { input, .. } => Some(input),
        RunEvent::CompactionApplied { summary, .. } => Some(summary),
        RunEvent::AssistantMessage { content } => Some(content),
        RunEvent::ToolCompleted { result, .. } => Some(result),
        RunEvent::ContextBuilt { .. }
        | RunEvent::ContextSourcesResolved { .. }
        | RunEvent::ContextUnavailable { .. }
        | RunEvent::ToolStarted { .. }
        | RunEvent::PermissionDecision { .. }
        | RunEvent::TurnCompleted { .. }
        | RunEvent::TurnFailed { .. }
        | RunEvent::TurnCancelled { .. }
        | RunEvent::Warning { .. }
        | RunEvent::ChildRunRef { .. } => None,
        RunEvent::UserDisposition { note, .. } => Some(note),
    }
}

fn content_preview(content: &ContentRef, limit: usize) -> String {
    let text = match content {
        ContentRef::Inline { text } => text.clone(),
        ContentRef::Artifact(artifact) => format!(
            "artifact {} ({} bytes, {})",
            artifact.relative_path.display(),
            artifact.bytes,
            artifact.media_type
        ),
        ContentRef::Unavailable { reason } => format!("unavailable: {reason}"),
    };
    let normalized = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if normalized.chars().count() <= limit {
        normalized
    } else {
        format!("{}…", normalized.chars().take(limit).collect::<String>())
    }
}

fn escape_html(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

#[cfg(test)]
mod tests {
    use super::*;
    use piku_runtime::{
        ArtifactRef, ContextSourceSummary, RunContentChange, RunContentRef, RunToolEffect,
        Sha256Digest, SourceReference, Trust, UsageRecord, VerificationRecord, VerificationStatus,
        RUN_RECORD_SCHEMA_VERSION,
    };

    fn event(event: RunEvent) -> RunEventEnvelope {
        RunEventEnvelope {
            schema_version: RUN_RECORD_SCHEMA_VERSION,
            sequence: 0,
            recorded_at_ms: 0,
            session_id: "session-1".to_string(),
            scope: piku_runtime::RunEventScope::Turn {
                turn_id: "turn-0".to_string(),
            },
            event,
        }
    }

    fn run_event(event: RunEvent) -> RunEventEnvelope {
        RunEventEnvelope {
            schema_version: RUN_RECORD_SCHEMA_VERSION,
            sequence: 0,
            recorded_at_ms: 0,
            session_id: "session-1".to_string(),
            scope: piku_runtime::RunEventScope::Run,
            event,
        }
    }

    #[test]
    fn text_projection_surfaces_outcome_and_usage() {
        let output = render_text(&[event(RunEvent::TurnCompleted {
            usage: Some(UsageRecord {
                input_tokens: 12,
                output_tokens: 7,
            }),
            stop_reason: Some("end_turn".to_string()),
        })]);
        assert!(output.contains("12↑ 7↓ · end_turn"));
    }

    #[test]
    fn text_projection_surfaces_context_provenance_without_payload() {
        let output = render_text(&[event(RunEvent::ContextSourcesResolved {
            sources: vec![ContextSourceSummary {
                id: "note-1".into(),
                sources: vec![SourceReference {
                    reference: "surface:scratch/object:note-1".into(),
                    sha256: Sha256Digest::of_bytes(b"private source bytes"),
                }],
                output_sha256: Sha256Digest::of_bytes(b"resolved bytes"),
                byte_size: 14,
                trust: Trust::UntrustedEvidence,
            }],
        })]);

        assert!(output.contains("1 context sources resolved · 14 bytes"));
        assert!(output.contains("note-1 · UntrustedEvidence · 14 bytes · sha256:"));
        assert!(output.contains("surface:scratch/object:note-1 · sha256:"));
        assert!(!output.contains("private source bytes"));
        assert!(!output.contains("resolved bytes"));
    }

    #[test]
    fn projections_do_not_invent_usage_or_success_for_other_terminal_states() {
        let output = render_text(&[
            event(RunEvent::TurnCompleted {
                usage: None,
                stop_reason: Some("completed".to_string()),
            }),
            event(RunEvent::TurnFailed {
                class: "executor".to_string(),
                message: "rejected".to_string(),
            }),
            event(RunEvent::TurnCancelled {
                reason: "disconnected".to_string(),
            }),
        ]);

        assert!(output.contains("tokens not reported · completed"));
        assert!(output.contains("failed [executor] · rejected"));
        assert!(output.contains("cancelled · disconnected"));
    }

    #[test]
    fn projections_surface_effect_and_verification_evidence() {
        let completion = event(RunEvent::ToolCompleted {
            tool_call_id: "call-1".to_string(),
            result: RunContentRef::Inline {
                text: "ok".to_string(),
            },
            is_error: false,
            effects: vec![RunToolEffect::FileWrite {
                path: "src/lib.rs".into(),
                content_change: RunContentChange::Modified,
            }],
            verification: Some(VerificationRecord {
                description: "unit tests".to_string(),
                status: VerificationStatus::Passed,
            }),
        });

        let text = render_text(std::slice::from_ref(&completion));
        let html = render_html(&[completion]).unwrap();

        assert!(text.contains("file Modified src/lib.rs"));
        assert!(text.contains("verify Passed · unit tests"));
        assert!(html.contains("verify ${e.verification.status}"));
        assert!(html.contains("tool_effect_count"));
    }

    #[test]
    fn projections_label_user_disposition_as_run_level() {
        let disposition = run_event(RunEvent::UserDisposition {
            disposition: piku_runtime::RunDisposition::Accepted,
            note: RunContentRef::Inline {
                text: "the evidence is sufficient".to_string(),
            },
        });

        let text = render_text(std::slice::from_ref(&disposition));
        let html = render_html(&[disposition]).unwrap();

        assert!(text.contains("\nrun\n"));
        assert!(text.contains("user Accepted · the evidence is sufficient"));
        assert!(html.contains("user_disposition"));
        assert!(html.contains("scopeLabel"));
    }

    #[test]
    fn json_projection_keeps_metrics_beside_raw_events() {
        let json = render_json(&[event(RunEvent::TurnCompleted {
            usage: Some(UsageRecord {
                input_tokens: 12,
                output_tokens: 7,
            }),
            stop_reason: Some("end_turn".to_string()),
        })])
        .unwrap();
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();

        assert_eq!(value["events"].as_array().map(Vec::len), Some(1));
        assert_eq!(value["audit"]["event_count"], 1);
        assert_eq!(value["audit"]["completed_turn_count"], 1);
        assert_eq!(
            value["audit"]["findings"][0]["code"],
            "event_before_turn_start"
        );
    }

    #[test]
    fn html_projection_neutralizes_script_termination_from_recorded_content() {
        let html = render_html(&[event(RunEvent::AssistantMessage {
            content: RunContentRef::Inline {
                text: "</script><script>alert(1)</script>".to_string(),
            },
        })])
        .unwrap();
        assert!(!html.contains("</script><script>alert(1)</script>"));
        assert!(html.contains("\\u003c/script\\u003e"));
        assert!(html.contains("Content-Security-Policy"));
    }

    #[test]
    fn html_projection_embeds_a_verified_run_artifact() {
        let directory = tempfile::tempdir().unwrap();
        let record_path = directory.path().join("session-1.jsonl");
        let artifact_path = directory
            .path()
            .join("session-1.artifacts/00000001-assistant.txt");
        std::fs::create_dir_all(artifact_path.parent().unwrap()).unwrap();
        let artifact_text = "the full load-bearing artifact";
        std::fs::write(&artifact_path, artifact_text).unwrap();
        let event = event(RunEvent::AssistantMessage {
            content: RunContentRef::Artifact(ArtifactRef {
                relative_path: "session-1.artifacts/00000001-assistant.txt".into(),
                media_type: "text/plain; charset=utf-8".to_string(),
                bytes: u64::try_from(artifact_text.len()).unwrap(),
            }),
        });

        let html = render_html_with_artifacts(&[event], &record_path).unwrap();

        assert!(html.contains(artifact_text));
        assert!(html.contains("inspect full artifact"));
    }

    #[test]
    fn html_projection_rejects_an_artifact_that_escapes_the_run_directory() {
        let directory = tempfile::tempdir().unwrap();
        let event = event(RunEvent::AssistantMessage {
            content: RunContentRef::Artifact(ArtifactRef {
                relative_path: "../outside.txt".into(),
                media_type: "text/plain".to_string(),
                bytes: 1,
            }),
        });

        let error =
            render_html_with_artifacts(&[event], &directory.path().join("run.jsonl")).unwrap_err();

        assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
        assert!(error.to_string().contains("escapes run directory"));
    }
}
