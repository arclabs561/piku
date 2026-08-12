# Design: Codex execution backend

Status: accepted; read-only chat checkpoint implemented

## Problem

The spatial web workspace currently presents chat and change cards as agentic
objects, but executes them through Piku's provider-neutral model loop. The
Playwright evaluators, meanwhile, run as Codex. This creates two meanings of
"agent": the product is judged by Codex while the product itself may fail on a
separate OpenRouter credential. It also leaves Codex-native threads, approvals,
tool events, sandboxing, and resumability unavailable to the workspace that is
intended to improve on Codex's presentation.

Provider configuration has a separate plumbing gap. The runtime resolver reads
inherited environment variables, while Piku parses provider blocks that it does
not pass to the resolver. The main binary does not load a project or ancestor
`.env`. A key can therefore exist on disk without being available to a server
process. Secret values must not be copied into workspace records or logs to
paper over this mismatch.

## Context

Piku owns the spatial board, object identity, artifact source, evidence record,
and capability boundaries. A generated page cannot control host chrome or a
terminal. Each chat card needs self-contained editable history, optional
context, rerun lineage, and resume.

Codex app-server is the supported interface for deep product integrations. It
offers thread start/resume/fork, turn-level model/personality/sandbox settings,
streamed item events, approvals, and experimental dynamic tools. `codex exec`
is the supported non-interactive pipeline interface. The TypeScript and Python
SDKs wrap Codex for application and automation use, but would introduce another
runtime into this Rust service.

## Options considered

### Keep the direct-provider loop as the only executor

This is portable across OpenRouter, Anthropic, Groq, Ollama, and compatible
servers. It is also the narrowest path for a canvas-only source patch because
Piku controls the exact tool catalog. It is rejected as the sole default: Piku
would need to rebuild Codex's thread, approval, sandbox, tool-event, and coding
behavior while still presenting itself as a Codex replacement.

### Spawn `codex exec` for every turn

This is easy to prototype and already powers bounded evaluators. JSONL exposes
progress and structured output. It is rejected for interactive cards because a
fresh process per turn makes thread lifecycle, steering, cancellation,
approvals, and concurrent event routing unnecessarily indirect. Keep it for
ephemeral judges and CI-style jobs.

### Use a Codex SDK

The SDK gives a higher-level thread API. It is deferred because Piku is a Rust
service and app-server already exposes the underlying JSON-RPC protocol. Adding
Node or Python solely as a bridge would create a second package/runtime
boundary without adding authority or capability.

### Replace Piku's runtime with Codex app-server

This maximizes reuse but is rejected. Codex must not become the authority for
Piku's spatial document, durable evidence schema, page sandbox, or terminal
capabilities. It would also erase Piku's provider-neutral research and local
model path.

### Add app-server as an executor adapter

Chosen. Codex app-server becomes the default executor for coding-oriented chat
and workspace-task cards. Piku remains the workspace and evidence authority and
stores a typed projection of Codex thread, turn, item, approval, and usage
events. The existing provider-neutral loop remains an explicit backend for
local models, provider comparison, deterministic fixtures, and narrow page
generation where its constrained tool contract is materially safer.

## Chosen approach

Add an `Executor` boundary with `codex` and `provider` implementations. New chat
and workspace-task cards default to `codex`; their persisted state includes the
backend name and opaque Codex thread ID. Existing cards migrate to `provider`
so resuming them cannot silently change model, tools, or authority. The card
header always names its backend, model, sandbox, and workspace access.

Speak app-server's versioned JSON-RPC protocol directly from Rust. The first
checkpoint starts one supervised child per active turn, maps each chat card to
one durable thread, streams events into the existing activity timeline, and
kills the child when the client disconnects. This proves persistence and
cancellation before introducing a shared long-lived supervisor. Piku never
places secrets, page source, or unrelated card content into Codex context
implicitly.

Piku stores the Codex-owned state root under a private `0700` application
directory with a reference to the existing Codex authentication. Codex may
populate that root with its own session databases, model and plugin metadata,
system skills, caches, and shell snapshots. Piku does not load the user's Codex
configuration, hooks, MCP servers, or global instructions. The Piku notebook
stores only the opaque thread ID and visible execution metadata. Edited reruns
start a new native thread; an appended turn resumes the existing one.

The app-server child receives an allowlisted environment rather than Piku's
complete process environment. The allowlist contains only process-launch and
TLS plumbing. Both `HOME` and `CODEX_HOME` point at the private Piku Codex root;
the operator home is not inherited because launch wrappers may use it to
rehydrate provider credentials or configuration.
Provider credentials and agent configuration variables are not inherited.
App-server stderr is projected into structured Piku diagnostics so launch
failures name the broken boundary instead of collapsing into a generic exit.

Evaluation jobs use a different contract. They invoke `codex exec` with
`--ephemeral`, `--ignore-user-config`, and `--ignore-rules`, plus an explicit
model, read-only sandbox, reasoning level, output schema, and task prompt. A
noninteractive `never` approval policy permits those predeclared read-only
operations without granting a broader sandbox. A
browser judge receives an explicit project-owned headless Playwright MCP
declaration whose tools are preapproved and restricted to the local Piku
origins; it does not discover the browser through personal Codex config.
The judge process receives only authentication discovery and basic launch/TLS
environment, not provider keys. This is the reproducible "naked Codex" path;
it is intentionally not used for durable interactive threads.

Canvas source changes initially stay on the constrained provider executor.
Move them to Codex only after a Piku-owned dynamic tool or equivalent narrow
adapter can expose `read_page_source`, `apply_page_patch`, and
`verify_page_render` without repository or shell authority. Dynamic tools are
currently experimental, so this is a gate rather than an assumption.

Credential discovery is fixed independently. Inherited environment remains the
first authority. Piku may selectively read only recognized provider variables
from the nearest `.env` when absent, without importing arbitrary entries.
Explicit settings may select provider/model/endpoints, but durable config should
reference a secret source rather than store raw keys. Startup and the web UI
report backend readiness using booleans and stable reasons, never values.

## Tradeoffs

- Codex authentication and local installation become requirements for the
  default coding executor.
- Piku must track app-server protocol versions and process lifecycle.
- Two explicit executors remain, increasing test-matrix size.
- Existing provider cards retain behavior rather than receiving an automatic
  Codex quality upgrade.
- Experimental dynamic tools cannot be a production security boundary.

## Non-goals

- Do not make Codex authoritative for workspace layout, saved page source,
  terminal access, or Piku's evidence ledger.
- Do not load every value from a repository `.env` into the server process.
- Do not hide backend choice or silently fall back from Codex to a paid provider.
- Do not use `codex exec` as the interactive thread transport.
- Do not send user files or cards to either executor unless the user selected
  them as context or the active capability explicitly requires them.

## Authority layers

Piku uses “capability” in the object-capability sense: an unforgeable value
both names authority and permits its use. Linux capabilities are different:
they split privileges historically held by root into per-thread sets. Dropping
their bounding and ambient sets is useful hardening, but it does not confine an
ordinary same-user process to one workspace ([capabilities(7)][linux-caps]).

Keep three layers separate:

1. **Application authority.** Typed attestations and single-use leases decide
   whether Piku may request one write turn. Authority is explicit, attenuated,
   non-ambient, and unavailable after consumption.
2. **Execution containment.** The OS must enforce filesystem, network,
   process, and privilege boundaries. On Linux, `no_new_privs`, Landlock,
   namespaces, and seccomp are complementary: Landlock restrictions stack and
   pass to descendants, while seccomp reduces syscall surface and explicitly
   is not a complete sandbox ([Landlock][landlock], [seccomp][seccomp]). A
   launcher such as bubblewrap assembles these primitives but is only as safe
   as its concrete policy ([bubblewrap][bubblewrap]).
3. **Runtime evidence.** eBPF/LSM systems can observe and sometimes enforce
   actual process, file, and network activity. They are a valuable independent
   witness, not the authority token itself. Tetragon also warns that signal
   termination may occur after an operation; inline denial is stronger
   ([Tetragon enforcement][tetragon]).

Other agent systems reinforce the same division. GitHub documents sandbox and
firewall coverage gaps, Deno warns that subprocess permission escapes its
runtime permission model, and MCP recommends a host broker that retains
credentials and authorizes each tool call ([Copilot firewall][copilot],
[Deno permissions][deno], [MCP client security][mcp]). Piku should therefore
prefer a host-held narrow capability plus a descendant-enforced OS boundary
and independent evidence, rather than treating prompts, policy fields, or
telemetry as enforcement.

[linux-caps]: https://man7.org/linux/man-pages/man7/capabilities.7.html
[landlock]: https://docs.kernel.org/userspace-api/landlock.html
[seccomp]: https://docs.kernel.org/userspace-api/seccomp_filter.html
[bubblewrap]: https://github.com/containers/bubblewrap
[tetragon]: https://tetragon.io/docs/concepts/enforcement/
[copilot]: https://docs.github.com/en/copilot/how-tos/copilot-on-github/customize-copilot/customize-cloud-agent/customize-the-agent-firewall
[deno]: https://docs.deno.com/runtime/fundamentals/security/
[mcp]: https://modelcontextprotocol.io/docs/develop/clients/client-best-practices

## Implementation plan

1. Add a read-only backend readiness endpoint and visible selector. This is
   reversible and must expose no credential values.
2. Fix selective provider credential resolution and add process-versus-file
   regression tests. This is independent of Codex.
3. Define executor-neutral thread/turn/item events and map the current provider
   loop onto them without behavior changes.
4. Add a stdio app-server adapter, initialize handshake, isolated state,
   capability probe, and typed protocol fixtures. **Implemented for one
   supervised process per active turn.**
5. Route new chat cards to Codex read-only threads; persist opaque thread IDs,
   stream items, and test resume/cancel/reload. **Implemented.**
6. Add explicit workspace-write escalation and host-owned approvals. No card
   receives mutation authority merely because its backend is Codex.
7. Evaluate a narrow page dynamic-tool adapter only after the coding path is
   stable and the protocol capability is present.

## Decision gates

- Stop if app-server cannot be supervised and resumed without orphaned children
  or ambiguous thread ownership.
- Stop if a Codex event cannot be projected without parsing presentation text.
- Reject the default switch until a successful chat survives reload, resume,
  cancellation, and identical-output rerun with stable lineage.
- Reject workspace-write until approval, sandbox, cwd, environment, network,
  and terminal boundaries are visible and tested independently.
- Keep page changes on the provider loop until the Codex adapter can expose only
  page-scoped tools and prove unrelated files are unreachable.
- Revisit the SDK choice if direct JSON-RPC compatibility maintenance exceeds
  the cost of a supported sidecar runtime.

## Open questions

- Whether Codex threads should inherit project `AGENTS.md` by default or require
  an explicit per-card context toggle.
- Whether one app-server process should serve all surfaces or one process should
  isolate each workspace.
- Which secret-reference mechanism should back provider settings on macOS.

---
Decided: 2026-08-08 | Session: 019fe1d4
