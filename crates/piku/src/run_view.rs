//! Read-only projections of Piku's durable semantic run record.

use std::fmt::Write;

use piku_runtime::{RunContentRef as ContentRef, RunEvent, RunEventEnvelope};

#[must_use]
pub fn render_text(events: &[RunEventEnvelope]) -> String {
    let mut output = String::new();
    let session = events.first().map_or("unknown", |event| &event.session_id);
    let _ = writeln!(output, "run {session} · {} events", events.len());
    let mut current_turn = "";
    for envelope in events {
        if envelope.turn_id != current_turn {
            current_turn = &envelope.turn_id;
            let _ = writeln!(output, "\n{current_turn}");
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
                result, is_error, ..
            } => {
                let mark = if *is_error { "×" } else { "✓" };
                let _ = writeln!(output, "  {mark} {}", content_preview(result, 120));
            }
            RunEvent::TurnCompleted { usage, stop_reason } => {
                let _ = writeln!(
                    output,
                    "  ■ {}↑ {}↓ · {}",
                    usage.input_tokens,
                    usage.output_tokens,
                    stop_reason.as_deref().unwrap_or("unknown")
                );
            }
            RunEvent::Warning { message } => {
                let _ = writeln!(output, "  ! {message}");
            }
        }
    }
    output
}

pub fn render_json(events: &[RunEventEnvelope]) -> serde_json::Result<String> {
    serde_json::to_string_pretty(events)
}

pub fn render_html(events: &[RunEventEnvelope]) -> serde_json::Result<String> {
    let title = events
        .first()
        .map_or("Piku run", |event| event.session_id.as_str());
    let data = serde_json::to_string(events)?
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
#timeline{{padding:2rem clamp(1rem,5vw,6rem) 8rem;max-width:90rem}}article{{display:grid;grid-template-columns:5rem minmax(0,1fr);gap:1.2rem;padding:1.15rem 0;border-top:1px solid #555146;animation:arrive .28s ease-out both;animation-delay:calc(var(--i) * 12ms)}}
.seq{{color:var(--muted);font-size:.72rem}}h2{{margin:0 0 .55rem;font-size:.78rem;letter-spacing:.12em;text-transform:uppercase;color:var(--signal)}}pre{{max-height:28rem;overflow:auto;margin:.4rem 0 0;padding:1rem;background:#22221e;color:#e9e4d7;white-space:pre-wrap;word-break:break-word}}
dl{{display:grid;grid-template-columns:max-content 1fr;gap:.25rem 1rem;margin:.4rem 0}}dt{{color:var(--muted)}}dd{{margin:0}}.summary{{max-width:76ch;margin:.7rem 0;color:#d7d2c6}}details{{margin-top:.6rem}}summary{{cursor:pointer;color:var(--muted)}}.error h2{{color:#ff7966}}.empty{{padding:5rem 0;color:var(--muted)}}
@keyframes arrive{{from{{opacity:0;transform:translateY(5px)}}to{{opacity:1;transform:none}}}}@media(max-width:720px){{header{{grid-template-columns:1fr}}main{{display:block}}nav{{position:static;border-right:0;border-bottom:1px solid #555146;padding:1rem 1.2rem}}article{{grid-template-columns:3rem 1fr}}}}
@media(prefers-reduced-motion:reduce){{article{{animation:none}}}}
</style>
</head>
<body>
<header><div><div class="kicker">durable investigation record</div><h1>{}</h1></div><label><span class="kicker">filter evidence</span><input id="filter" type="search" placeholder="tool, turn, reason, content…" autocomplete="off"></label></header>
<main><nav id="turns" aria-label="Turns"></nav><section id="timeline" aria-live="polite"></section></main>
<script id="run-data" type="application/json">{}</script>
<script>
const events=JSON.parse(document.getElementById('run-data').textContent);const timeline=document.getElementById('timeline');const turns=document.getElementById('turns');const filter=document.getElementById('filter');
const text=v=>v==null?'':typeof v==='string'?v:JSON.stringify(v,null,2);const label=e=>e.event.replaceAll('_',' ');const eventText=e=>JSON.stringify(e).toLowerCase();const content=c=>c?.text??(c?.relative_path?`${{c.relative_path}} · ${{c.bytes}} bytes`:c?.reason??'');
function eventSummary(e){{switch(e.event){{case'turn_started':return `${{e.provider??'unknown'}} / ${{e.model}} · ${{content(e.input)}}`;case'context_built':{{const m=e.manifest.messages,s=m.filter(x=>x.selected).length;return `${{e.manifest.estimated_input_tokens}} estimated tokens · ${{s}} selected · ${{m.length-s}} excluded · ${{e.manifest.tools.length}} tools`}}case'compaction_applied':return `${{e.before_messages}} → ${{e.after_messages}} messages · ${{e.masked_tool_results}} masked results`;case'assistant_message':return content(e.content);case'tool_started':return `${{e.name}} ${{text(e.arguments)}}`;case'permission_decision':return e.decision;case'tool_completed':return `${{e.is_error?'error':'ok'}} · ${{content(e.result)}}`;case'turn_completed':return `${{e.usage.input_tokens}}↑ ${{e.usage.output_tokens}}↓ · ${{e.stop_reason??'unknown'}}`;case'warning':return e.message;default:return''}}}}
function draw(q=''){{timeline.replaceChildren();turns.replaceChildren();const seen=new Set();let shown=0;events.forEach((e,i)=>{{if(q&&!eventText(e).includes(q))return;shown++;if(!seen.has(e.turn_id)){{seen.add(e.turn_id);const b=document.createElement('button');b.textContent=e.turn_id;b.onclick=()=>document.getElementById('event-'+e.sequence)?.scrollIntoView({{behavior:'smooth'}});turns.append(b)}}const a=document.createElement('article');a.id='event-'+e.sequence;a.style.setProperty('--i',i);if(e.is_error)a.className='error';const s=document.createElement('div');s.className='seq';s.textContent=String(e.sequence).padStart(4,'0');const body=document.createElement('div');const h=document.createElement('h2');h.textContent=label(e);body.append(h);const dl=document.createElement('dl');[['turn',e.turn_id],['recorded',new Date(e.recorded_at_ms).toLocaleString()]].forEach(([k,v])=>{{const dt=document.createElement('dt'),dd=document.createElement('dd');dt.textContent=k;dd.textContent=v;dl.append(dt,dd)}});body.append(dl);const summary=document.createElement('p');summary.className='summary';summary.textContent=eventSummary(e);body.append(summary);const payload={{...e}};['schema_version','sequence','recorded_at_ms','session_id','turn_id','event'].forEach(k=>delete payload[k]);const d=document.createElement('details');const sum=document.createElement('summary');sum.textContent='inspect payload';const pre=document.createElement('pre');pre.textContent=text(payload);d.append(sum,pre);body.append(d);a.append(s,body);timeline.append(a)}});if(!shown){{const p=document.createElement('p');p.className='empty';p.textContent='No evidence matches this filter.';timeline.append(p)}}}}
filter.addEventListener('input',()=>draw(filter.value.trim().toLowerCase()));document.addEventListener('keydown',e=>{{if(e.key==='/'&&document.activeElement!==filter){{e.preventDefault();filter.focus()}}}});draw();
</script>
</body></html>"#,
        escape_html(title),
        escape_html(title),
        data
    ))
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
    use piku_runtime::{RunContentRef, UsageRecord, RUN_RECORD_SCHEMA_VERSION};

    fn event(event: RunEvent) -> RunEventEnvelope {
        RunEventEnvelope {
            schema_version: RUN_RECORD_SCHEMA_VERSION,
            sequence: 0,
            recorded_at_ms: 0,
            session_id: "session-1".to_string(),
            turn_id: "turn-0".to_string(),
            event,
        }
    }

    #[test]
    fn text_projection_surfaces_outcome_and_usage() {
        let output = render_text(&[event(RunEvent::TurnCompleted {
            usage: UsageRecord {
                input_tokens: 12,
                output_tokens: 7,
            },
            stop_reason: Some("end_turn".to_string()),
        })]);
        assert!(output.contains("12↑ 7↓ · end_turn"));
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
}
