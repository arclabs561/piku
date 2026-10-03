# Design: harness-owned browser recording

Status: proposed

## Problem

Piku's TUI evaluator already records a standard asciicast and a Piku-owned
JSONL sidecar at the PTY boundary. It therefore captures the interaction even
when a judge times out, forgets a useful observation, or fails while producing
its report. The web evaluator does not have the equivalent for model-driven
Codex journeys. Its deterministic Playwright tests can retain a video, but the
more informative browser-MCP journeys currently retain only Codex events and
agent-requested screenshots.

Asking a judge to start and stop a browser recording is the wrong boundary. It
spends its action budget, is easy to omit, makes the artifact depend on prompt
obedience, and loses the recording on exactly the evaluator failures worth
reviewing. Recording must be a property of the harness, not of agent behavior.

## Context

The current path is:

```text
managed evaluator → parallel-agent-eval → Codex exec → stdio Playwright MCP → browser
```

Codex has an explicit browser-tool allowlist and its JSON event stream is
already retained and bounded. The installed Playwright MCP supports video,
chapter, and action-overlay operations behind its `devtools` capability, but
does not expose a reliable automatic-recording configuration in the installed
version. The harness already defaults human review output to
`~/Library/Caches/piku/eval-review`, outside a CloudDocs-backed checkout.

## Non-goals

- Do not ask Codex to start, stop, chapter, or otherwise manage recording.
- Do not make video a source of product truth. It is review material alongside
  structured browser events, screenshots, predicates, and reports.
- Do not make model-driven runs a release or CI gate merely because they now
  produce richer artifacts.
- Do not stream recordings to a hosted service or inject their contents into a
  judge prompt. They remain local operator artifacts.
- Do not replace the TUI asciicast or CLI transcript formats with browser media.
  A shared manifest may index them, but each surface keeps its native replay.

## Options considered

### Prompt-managed browser video

The evaluator receives `browser_start_video`, `browser_video_chapter`, and
`browser_stop_video` in its allowlist. This was prototyped and rejected. The
model did call the tools, proving capability wiring, but that proof also showed
the defect: recording was an agent choice rather than a lifecycle guarantee.

### Browser-specific automatic configuration

Some newer Playwright MCP documentation describes automatic video settings.
The installed MCP (`0.0.78`) exposes interactive video tools but not a stable
automatic save-video contract. Depending on undocumented configuration would
make a completed evaluation falsely claim an artifact that may never exist.
Deferred until a pinned MCP version exposes and tests that contract.

### Harness-owned transparent MCP recorder

Run a small local stdio proxy between Codex and Playwright MCP. It forwards
Codex's allowed tool traffic unchanged, captures a screenshot after each
state-changing action, and writes a sidecar from the observed protocol stream.
Chosen.

## Chosen approach

Add `browser-recorder-proxy.mjs` and launch it as the configured Playwright MCP
command for model-driven evaluators only:

```text
Codex ⇄ JSON-RPC stdio ⇄ Piku recorder proxy ⇄ JSON-RPC stdio ⇄ Playwright MCP
                                             └─ local review bundle
```

The proxy starts the underlying server with `--caps=devtools`, the ordinary
loopback origin restriction, and a run- and role-specific local output
directory. Its `tools/list` response exposes only the existing product-browser
allowlist to Codex. Recorder tools never appear in Codex's schema or event
budget.

The installed Playwright MCP's interactive screencast tool produced a zero-byte
file in a real managed journey and could hang finalization. It is therefore not
used for evaluator review assets. Instead, after a state-changing browser tool
has completed, the proxy holds that response briefly, takes one internal
`browser_take_screenshot`, then returns the original response unchanged. This
keeps a frame paired with the completed action without giving Codex a recorder
choice or exposing recorder tools.

On browser close, Codex stdin EOF, timeout, ordinary exit, or signal, the
finalizer turns the captured PNG frames into one local H.264 MP4 with bounded
`ffmpeg`. It verifies the MP4 is a non-empty regular file below its output
directory and writes a footer. A hard kill or render failure produces an
explicitly incomplete bundle, never a fabricated successful recording.

## Recorder protocol and artifacts

The proxy is a framed JSON-RPC relay, not a text filter. It keeps two private
request-ID namespaces: Codex IDs pass through unchanged; recorder IDs use a
reserved generated prefix and their responses are consumed by the proxy. It
must forward initialization, cancellation, progress, notifications, tool
requests, and errors without assuming every message is a tool call. Tool-list
filtering occurs only on the response to `tools/list`.

Each model-driven role receives a local directory:

```text
~/Library/Caches/piku/eval-review/web-<run>/<role>/
  frame-001.png            # proxy-captured post-action frame
  frame-002.png
  journey.mp4              # local H.264 action-frame replay for Safari
  journey.mp4.ffconcat     # ffmpeg input order and frame duration
  recorder.jsonl           # protocol timeline and recorder lifecycle
  events.jsonl             # Codex event stream
  screenshots/             # agent-requested evidence
  evidence.json            # validated judge report, if produced
  manifest.json            # role outcome and content hashes
```

`journey.mp4` is the review recording, not a preview-only substitute. It is
created from the retained action frames by a bounded local `ffmpeg` invocation;
absence or rendering failure is recorded but cannot change the evaluation
outcome. The frames remain available for frame-by-frame review. The run-level
manifest links roles and their finalization state. Small control-plane manifests
may remain under `.artifacts`; media and human-review copies must be written
directly to the local review root rather than copied out of CloudDocs later.

`recorder.jsonl` contains a header with schema, recording strategy, paths
relative to the role root, and input-redaction policy. Its events are:
`recording_started`, `tool_started`, `frame_captured`, `frame_unavailable`,
`recording_stop_requested`, and `recording_finalized`. Each has a monotonic
elapsed timestamp, public tool name, success state, and bounded error
classification. Browser fill values, model output, headers, and raw JSON-RPC
bodies are not copied into this sidecar by default.

## Lifecycle and failure semantics

| Condition | Judge result | Recorder disposition |
| --- | --- | --- |
| normal judge completion | normal validator outcome | `complete` with media hash |
| invalid report | `invalid_report` | recording remains complete if finalized |
| agent timeout/budget stop | existing evaluator failure class | finalize before process-group escalation |
| tool/server failure | existing failure class | `unavailable` or `partial`, named cause |
| proxy protocol violation | harness failure | fail closed; preserve only verified partial files |
| forced kill / crash | harness or signal outcome | no footer, `incomplete` manifest state |

Video availability never upgrades a run to success and video absence never
erases validated browser evidence. Conversely, a completed evaluation must not
claim a review recording unless the finalizer wrote a footer and verified the
media hash.

## Implementation plan

1. Extract the existing local review-root helpers from `run-managed-eval.mjs`
   into a small shared module. Add pure path-confinement tests for run, role,
   and media paths. Reversible.
2. Implement a line/framing-safe MCP relay with fixture child transports. Test
   initialization forwarding, reserved IDs, `tools/list` filtering,
   cancellation, malformed messages, EOF, and a recorder-stop timeout. No
   browser dependency in these tests.
3. Add the recorder's immutable JSONL schema and finalization verifier. Test
   absent, escaping, symlinked, zero-byte, and complete media cases. Reversible
   until the parallel runner adopts it.
4. Make `runCodex` launch the proxy for browser judges, passing only role-local
   paths and loopback origin. Keep Codex's visible tool allowlist unchanged.
5. Move browser-generated screenshots and role review artifacts directly into
   the local review bundle; retain compact control manifests and hashes in the
   repository artifact directory.
6. Add optional WebM-to-MP4 conversion and a `review-manifest.json` that marks
   conversion availability without changing verdicts.
7. Run one deliberately timed-out and one normal parallel judge evaluation.
   Inspect the actual media, sidecar, completion footer, and cleanup state.

## Decision gates

- A timeout after the first browser action must still produce either a verified
  finalized recording or an explicit incomplete recorder state.
- Codex's effective tool schema and counted action budget must not contain a
  recorder-only tool.
- Every finalized video path must resolve beneath the role-local review root;
  an escaping or symlinked target fails the harness.
- A video-stop failure must not hang cleanup longer than its configured bound.
- An evaluator receiving a malicious product string must not be able to alter
  recorder configuration, output path, or finalization policy.
- A normal run must yield one video start, one finalization attempt, and no
  more than one finalized media record per role.

## Open questions

- Does the proxy need semantic chapter cards initially, or is the JSONL action
  timeline sufficient until an operator-facing chapter renderer exists?
- Should MP4 conversion be default on macOS only, or an explicit export step on
  all platforms?
- How much browser-action detail can the sidecar retain before local privacy
  concerns outweigh review utility?

---
Decided: 2026-08-14 | Session: Codex
