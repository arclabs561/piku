# Design: managed page executor boundary

Status: implementing

## Problem

Managed browser evaluation deliberately starts Piku without provider or Codex
credentials. That makes the product server safe to inspect, but it also makes a
real page-change journey impossible: page requests currently resolve a provider
inside the web process.

Passing credentials into that process is not acceptable. Piku exposes a real
human terminal, and a login shell is ambient user authority. A synthetic home
and isolated working directory improve hygiene but do not confine a same-user
process to that directory.

## Chosen approach

The managed evaluator owns one anonymous duplex channel between the web child
and a parent-side, tool-free page proposal broker. The web process receives no
credential value, credential path, socket address, shell, or general executor
catalog through this channel. The broker receives only a bounded model request
for the page renderer and may call one fixed provider endpoint. It cannot read
workspace files, execute commands, operate a browser, persist page state, or
apply its own response.

The first protocol method is `page.propose.v1`. Requests and responses are
newline-delimited JSON, correlated by an unpredictable request ID, bounded on
both sides, and rejected on unknown protocol, tools, oversized content,
mismatched identity, malformed usage, or trailing authority-bearing fields.
The existing Rust agent loop consumes the broker as a `Provider`, so session
history, run records, streamed activity, and host validation remain on the same
path as an in-process provider.

Piku remains the only mutation authority. It validates complete-document
creation or exact unique source patches, compares the source inspected by the
request with the current saved source, persists atomically, and emits the saved
snapshot and verification evidence. Broker prose is never executable.

Managed modes have two capability profiles:

- trusted deterministic E2E may exercise the real terminal against a seeded
  artifact-local workspace;
- model-driven `single`, `parallel`, and `focus-pair` evaluation does not
  register the PTY route and does not render a terminal start control.

## Options considered

### Put credentials back in the web server

Rejected. Environment filtering is not a security boundary once the process
can start an unrestricted shell. A fake `HOME` also does not prevent absolute
filesystem reads.

### Run Codex directly as the page renderer

Deferred. A read-only Codex sandbox still grants filesystem reads, while page
HTML is untrusted input. This would combine untrusted content, private host
access, and provider egress. Revisit only when Codex can be confined to a
page-only dynamic tool contract and unrelated files are demonstrably
unreachable.

### Use a named Unix socket or loopback HTTP service

Deferred. Both require discoverability, authentication, stale-endpoint
cleanup, and confused-deputy defenses that an inherited anonymous channel does
not. Revisit if brokers must reconnect, outlive the managed server, or serve
multiple clients.

### Let the evaluator invoke the model and inject the result

Rejected as an acceptance path. It would test the judge's mutation path, not
the product's, and would make evaluator output authoritative for product state.
It remains useful only as an offline oracle or deterministic fixture generator.

## Non-goals

- Do not expose chat, workspace arrangement, repository mutation, or terminal
  control through the page broker.
- Do not make the broker a general Codex, MCP, shell, filesystem, or browser
  executor.
- Do not claim the ordinary human terminal is sandboxed. It remains ambient
  operator authority outside model-driven managed evaluation.
- Do not move source validation or persistence out of the Piku host.
- Do not silently fall back between broker, provider, fixture, and Codex
  execution; identity and failure must remain visible.

## Implementation sequence

1. Run managed Piku from a seeded artifact-local workspace and remove the PTY
   route from model-driven capability profiles. Implemented in `f166bac`.
2. Add the parent-owned `page.propose.v1` channel and optional Rust provider
   adapter. Keep the old in-process provider path for ordinary launches.
3. Bind request ID, surface, target, instruction digest, base-source digest,
   deadline, and output limit. Recheck the base digest immediately before save.
4. Record broker lifecycle, model identity, request/response digests, and stable
   failure class in the managed lifecycle binding without recording prompts or
   credentials.
5. Add one deterministic protocol proof and one opt-in live page journey. The
   live result is inconclusive on provider or harness failure, never a product
   failure by substitution.

## Decision gates

- Stop if the inherited channel is reachable from any model-controlled child.
- Stop if the broker requires filesystem, shell, browser, MCP, or arbitrary
  egress authority.
- Reject a proposal if source or target identity changed while inference ran.
- Reject the live acceptance claim until page creation and revision both
  survive reload with the same recorded target and source lineage.
- Revisit a long-lived sidecar only after a real reconnect or multi-client
  requirement appears.

---

Decided: 2026-08-10 | Session: 019fe1d4
