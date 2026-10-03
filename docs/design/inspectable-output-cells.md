# Design: inspectable output cells

Status: proposed

## Decision

Make Piku's terminal and browser present a run as an ordered set of inspectable
cells. A cell is a projection of an immutable prompt revision and its durable
run events, not a mutable notebook execution unit and not a transcript archive.

The primary outcome is immediate legibility while a run is active: an operator
can stop following a streaming wall of text, select the meaningful unit, expand
its complete result, and resume from a precise point without asking the model to
repeat itself. Durable records make that possible after the fact, but preservation
is a consequence rather than the product's reason for being.

The existing append-only run record remains the semantic source of truth.
`OutputSink`, the TUI, JSONL/CLI output, HTML run view, SSE, and a future editor
adapter are projections of it. Human PTY streams remain separate direct-
manipulation artifacts; an agent cell may link to an explicitly captured PTY
artifact but must never imply that it controls the PTY.

## Why a document-shaped interaction

Notebook interfaces demonstrate a useful interaction property: input, result,
and explanatory material stay adjacent, so a person can work on a problem in
pieces and return to a particular piece rather than reconstructing it from an
ephemeral chat. Jupyter explicitly attaches results to the input that produced
them and supports in-place iterative work. Its command and edit modes also make
navigation distinct from changing a cell.

That is the right inspiration for Piku's attention model, but not its execution
model. Notebook kernels permit a visible cell sequence to diverge from hidden
runtime state after edits, reruns, removals, or out-of-order execution. Research
on notebook lineage identifies that mismatch as a source of errors. Piku avoids
it by never replacing a prior run: editing a prompt produces a new prompt
revision and a new child run with explicit ancestry.

The desired feeling is therefore *an enhanced shell with inspectable pieces*,
not a general-purpose computational notebook.

“Cell” is the implementation/projection term, not mandatory terminal
vocabulary. Default terminal language remains **run**, **command**, **output**,
**effect**, and **job**. A user can adopt a notebook-like presentation, but
Piku must not require them to think of ordinary shell work as notebook cells.

## Terminal sovereignty

The terminal is not a legacy output target for Piku to hide behind a chat
universe. It is the operator's familiar, inspectable working environment:
files, commands, processes, editors, pagers, shell history, and scrollback all
remain available without asking an agent to translate or regenerate them.

Piku therefore acts as a well-behaved terminal process with richer views, not
as the owner of a replacement universe. Its design obligations are:

- **Readable ordinary output.** A person can retain plain streaming and native
  scrollback, or select and expand cells without accepting an opaque renderer.
- **Escape to familiar tools.** Any full result or artifact has a stable ID and
  can be printed, copied, sent to the configured pager, opened in the configured
  editor, or inspected through a composable CLI projection.
- **No private UI-only truth.** Cell identity, run state, branch ancestry,
  effects, and artifact references are queryable outside the TUI and browser.
  The visual layout may be private preference; the work cannot be trapped there.
- **Explicit boundaries.** The agent's tool calls, permissions, cwd, and
  mutations are visible events. Piku does not impersonate the user's shell,
  silently take over a human PTY, or turn terminal bytes into chat context
  without an explicit capture action.
- **Progressive enhancement.** A minimal terminal remains useful when a
  full-screen UI, browser, mouse, rich-media renderer, or custom key profile is
  absent. Enhanced cells add inspection rather than requiring a new worldview.

This is also why the default command surface must be designed alongside the
TUI. A cell view is not successful if it can only be reached through Piku's own
screen; the terminal equivalent must let an operator list runs, select a cell,
read its complete output, and inspect effects with ordinary shell composition.

### Workspace reality strip

Every interactive view needs a small, always-legible workspace status line:

```text
cwd: crates/piku  ·  repo: piku  ·  branch: main  ·  Git: 2 modified, 1 staged
run: R42 running  ·  known effects: 1 write  ·  snapshot: refreshed after tool 7
```

It is a navigation aid, not a claim that Piku owns the checkout. The expanded
workspace view exposes the actual paths, normal `git diff` access, current
branch/HEAD, and the snapshot time. Refresh occurs on explicit request, after a
recorded tool effect, and before a mutation decision; it is never presented as
live when it has not been checked.

Two facts must remain separate in a shared or already-dirty worktree:

- **Known run effects** are causal events recorded from this run's tools.
- **Working-tree delta** is what Git sees relative to an explicit baseline. It
  may include human or concurrent work, so unrecorded paths are labeled
  unattributed rather than assigned to the agent.

The status line makes the shell's reality visible at a glance; it does not
replace Git. Completion cells link directly to the familiar questions: what
files changed, what does the diff say, what is staged, and which check ran?

### Direct shell escape

The direct-shell path is part of the product contract. An operator can run a
normal command without sending it to the model, and its stdout, stderr, exit
status, TTY behavior, pager, and interactive programs remain native terminal
behavior. When Piku temporarily yields the screen to that child process, it
restores its frame afterward without treating the bytes as an agent event.

The current `!command` path proves the need but buffers a child process and
shows only a short preview. Replace that behavior with terminal-inheriting
passthrough for ordinary shell work. Add a distinct, explicit capture action
when a person wants a bounded command result recorded as a cell artifact or
included in model context. The three paths must stay visibly different:

```text
shell passthrough  → native terminal only
capture as evidence → bounded artifact with source and digest
attach as context   → bounded artifact selected for a particular prompt
```

## Shared surface philosophy: the operator's workbench

Piku's CLI, TUI, and web are three views of one local workbench. They must not
compete to become the “real” Piku or teach incompatible habits. The operator's
files, Git checkout, shell commands, and durable run record are real; each
surface makes a different subset easier to act on.

This inherits concrete terminal contracts rather than a vague terminal aesthetic:

| Earlier terminal experience | Contract worth preserving | Piku consequence |
| --- | --- | --- |
| Unix filters and pipes | a tool accepts/produces inspectable data and composes with other tools | CLI emits plain text for people, structured events for programs, stable IDs for selection, and meaningful exit codes |
| REPL history | an experiment is incremental but recoverable | prompt revisions, commands, and runs are addressable; an edit creates a child rather than rewriting the past |
| `less` and a pager | long output is an object to navigate, search, copy, and leave | full artifacts open in the configured pager/editor; previews never pretend to be complete output |
| shell job control | a running thing has identity, visible state, interrupt, resume, and foreground/background ownership | runs show `queued/running/interrupted/completed/failed`, preserve partial output, and make `Esc`/interrupt change state visibly rather than erase it |
| tmux/screen panes and scrollback | layout improves attention but does not own the programs or their data | TUI and web layout are replaceable preferences; a run, artifact, and effect can be queried without reopening the same layout |
| editor modes and keymaps | familiar movement should be adaptable, not universal | Piku binds semantic actions through profiles; the default stays discoverable and a Vim profile remains optional |

The concise philosophy is: **Piku is a local, agent-augmented shell workbench.
It makes the work easier to see, steer, retain, and connect without making the
work cease to be ordinary files, commands, Git state, and processes.**

### CLI: composition and truth

The CLI is Piku's public truth surface. It is usable in a pipe, script, pager,
editor task, or another terminal client. It reports a result through exit code,
human text, and opt-in structured output; its full-content and event-selection
operations work without an interactive renderer. It is where Piku proves that a
cell is not trapped in the TUI, and that Git/workspace status is not decorative.

### TUI: attention without enclosure

The TUI is a terminal-native focus aid: sticky prompt, scrollback, selected
cells, independent output viewport, search, and concise workspace reality
strip. It always offers a direct shell escape, configured pager/editor handoff,
and visible current cwd/repository/run state. Full-screen behavior is optional;
it must restore the terminal honestly and never make the user depend on its
private display cache.

### Evaluation action and observation contract

An evaluator must use the native interaction granularity of the surface. It is
not credible to judge a TUI as a web page with a single opaque “type text”
action, nor to put a remote model in the latency-sensitive keystroke loop.
Every evaluator therefore separates three layers:

| Layer | TUI example | CLI example | Web example |
| --- | --- | --- | --- |
| Raw event | UTF-8 character, `Tab`, `Esc`, `Ctrl-C`, resize, mouse wheel | argv byte stream, signal, stdin chunk | pointer/key/input event, viewport change |
| Semantic action | enter `/cell @@3`, cancel the active turn, scroll to a cell | `piku inspect --raw @@3` | submit a card edit, open a result, cancel a run |
| Observation | parsed VT100 grid, cursor, styles, input row, scroll position, PTY transcript | exit code, stdout/stderr artifact, run record | DOM predicate, screenshot, browser/network event, run record |

The judge chooses semantic actions. A local deterministic driver expands each
one into bounded raw events and records the intermediate observations needed to
explain a transition. This keeps a judge from reacting too slowly to ordinary
redraws while retaining the evidence that a human actually sees.

For text input, the default driver uses a short burst, drains the PTY after
each character, and takes a named screen snapshot at semantic boundaries:
first prefix (`/`, `@`, or `@@`), completed command/reference, completion or
edit-mode transition, submit, cancellation, and resize. A burst may end early
when a configured predicate changes, such as a completion menu appearing or
the prompt entering an error state. The evaluator records both the input events
and the resulting screen delta, so an implementation cannot pass by accepting
the final line while corrupting intermediate cursor, selection, or redraw
behavior.

Time is an observation input, not a reason for busy polling. A local driver can
wait for a bounded duration or a predicate with a deadline, then records
`running`, `settled`, `timed_out`, or `interrupted`. Submission and cancellation
are always split from the observation that proves their result. This is what
makes `Esc` meaningful mid-generation: the evidence must show an active turn,
the interrupt event, preserved partial output, and the later recovered state,
not merely a final “cancelled” label.

The same contract remains deliberately smaller elsewhere: CLI commands are
observed at process and artifact boundaries; web actions are observed after
DOM, rendered-pixel, or network state changes. A shared evaluation packet
names the common claims, while each surface keeps its own action and evidence
vocabulary.

Every surface also produces a **replay timeline**: surface-native raw evidence
plus an append-only annotation sidecar. The timeline is the review object; a
recording is one payload inside it, not a TUI-only feature.

| Surface | Replay payload | Annotation anchors |
| --- | --- | --- |
| CLI | exact argv, bounded stdin metadata, stdout/stderr artifacts, exit/signal and timing | process start/end, output boundaries, run-record references, judge findings |
| TUI | asciicast-compatible PTY output and resize stream | semantic input actions, VT100 screen observations, interrupt/recovery transitions, judge findings |
| Web | Playwright event trace, screenshots, DOM predicates, network observations | browser actions, visual/predicate checkpoints, persisted-object references, judge findings |

Annotations must cite evidence IDs or a raw-artifact reference and have a
monotonic offset from the evaluation start. They are never injected into the
terminal/video payload simply to make playback convenient. That keeps the
underlying replay portable, while a local viewer can scrub the timeline and
show the corresponding judge notes, findings, and retest obligations.

### Web: spatial inspection, not a competing universe

The web surface is a local, capability-scoped projection for things terminals
are weak at: virtualized large histories, rich artifacts, side-by-side forks,
and connected garden views. It consumes the same cell/action/query contract as
the CLI and TUI. Browser layout, tabs, and open panes are personal workspace
state; they cannot become the only location of an output, a diff, an authority
decision, or a way to resume work. Every visible object has a stable identity
that maps back to CLI/TUI operations.

## Terminal character: preserve the joy, repair the friction

The terminal's appeal is not simply that it is text. It is a personally grown
place where a command, its consequence, the current directory, and the next
small experiment share one legible medium. A command can be composed with a
pipe, recalled and altered, sent to a pager or editor, put in the background,
wrapped in an alias, or promoted into a script. The visible transcript provides
rhythm and a sense of local causality: *I ran this here; this happened; now I
can decide what to do.*

That character has real limits. Scrollback is a poor long-term memory: command
and output become separated by volume, earlier text is awkward to target in a
new command, and terminal programs sometimes redraw or destroy what the user
thought was an ordinary transcript. Command syntax and option recall impose a
large learning burden. History, completion, keybindings, clipboard behavior,
terminal emulators, multiplexers, and shell line editors have overlapping but
inconsistent responsibility. Job control is powerful but normally scoped to one
shell, not a whole person's work. These are repair targets, not reasons to
discard the shell.

Piku should preserve the first list and carefully repair the second:

| Keep this terminal pleasure | Repair this terminal frustration |
| --- | --- |
| a command and its result feel like one small experiment | retain a durable, selectable command/run/result relationship without forcing everything into model context |
| output is yours to page, search, copy, pipe, and reopen | give long output an artifact reference and pager handoff instead of clipping it or burying it in scrollback |
| cwd, files, Git, and processes provide concrete orientation | keep workspace reality visible and distinguish Piku-known effects from unrelated dirtiness |
| a familiar shell can be gradually personalized | offer profiles, aliases/actions, and clear help without requiring any one keymap or replacing shell configuration |
| foreground work is immediate and interruptible | make active, waiting, background, and completed runs explicit, including what `Esc` or interrupt will do |
| a transcript is a useful memory of what happened | preserve source-linked cells and garden items while keeping ordinary terminal scrollback ephemeral and unclaimed |

This means Piku must resist two opposite mistakes: a sterile machine-event log
that loses the feeling of working, and a full-screen faux terminal that takes
control away. The aim is an enriched typescript—ordinary shell life, plus
addressable experiments, accurate state, and a path from fleeting output to
chosen knowledge.

### Refinements from terminal and agent research

The terminal’s strongest precedent is not the raw byte stream by itself; it is
the human-visible relationship between an action, its result, and the next
choice. Piku preserves that relationship as typed events, then offers both
forms of each output:

- **exact/raw:** the original bounded artifact for copying, piping, parsing, or
  audit; and
- **rendered/safe:** a presentation that can fold, highlight, summarize routine
  activity, and sanitize control sequences without changing the source.

No renderer may make its normalized text appear to be the raw source. Every
reference carries the producing cwd, timestamp, run/tool identity, exit or
completion status, and relevant input/output/effect references. A relative
path or copied output without this context is an incomplete observation.

Use a progressive-enhancement ladder:

1. **Plain terminal path:** normal shell output, direct commands, native
   pager/editor/job control; Piku is useful even without a full-screen UI.
2. **Semantic overlay:** stable run/output references, workspace reality,
   artifact retrieval, raw/safe copy, and quiet status notifications.
3. **Focused TUI:** selection, folding, hints, independent output viewport,
   branch navigation, and job rail.
4. **Local web workbench:** only where spatial comparison, virtualization, or
   garden connections beat a terminal projection.

Each higher level consumes the same record and can be left without losing
access to the work. This also makes terminal-emulator integration additive:
prompt/command boundary markers such as OSC 133 improve navigation and output
selection when available, but are never Piku's persistence mechanism or a
requirement for a correct CLI/TUI result.

The default chronological stream remains visible because temporal order and
screen position are useful human memory. It is not replaced wholesale by cards.
Runs gain compact boundaries in that stream; selection, folding, search, and
references make a boundary actionable when the operator needs more than the
linear story. A focused view is entered deliberately and returns to the same
place in the stream on exit.

### Output ownership and terminal compatibility

When Piku runs inside an interactive terminal it may emit optional OSC 133
prompt/input/output/end markers and OSC 7 cwd metadata, following terminal
feature detection. This lets capable terminals select a whole command output or
jump between prompts using their native scrollback. Piku never relies on these
markers to reconstruct a record: shell changes, SSH, multiplexers, and
unsupported terminals may remove them.

A full-screen renderer owns stdout exclusively. Background tasks, hooks,
plugins, and debug logging must route through a renderer queue, dedicated log,
or durable artifact; they must never write directly into an active TUI. This is
both a correctness and trust condition—one stray ANSI writer can corrupt the
screen and make the visible transcript unreliable.

Command/output grouping also has a strict attribution boundary. It is reliable
for one foreground process/PTY. Once independent processes write concurrently,
chronological terminal bytes cannot honestly be assigned to a specific command.
Piku records those jobs with separate logs/PTYS and labels mixed or unknown
attribution rather than inventing a tidy block relationship.

## Addressing: point without copying

One address model is enough. Every durable object has a stable canonical ID;
the current surface additionally shows a short, session-scoped reference such
as `@@12` for a cell, `@e12.4` for an event, or `@@12#L40-L58` for a selected
span of an immutable artifact. The short form is resolved against the visible
session and expanded to the canonical identity before it reaches the runtime or
model. A reference that is stale, ambiguous, or outside the current authority
fails visibly rather than guessing.

Selection is the primary way to communicate a reference. Select a cell, result
span, file effect, or garden note and invoke `reference.insert_selected`; the
composer shows a visible attachment chip rather than requiring copy/paste. The
submitted prompt contains a structured reference and an explicit selection
policy (preview, complete artifact, or context attachment). A person can still
type `@@12`, and CLI callers can pass the canonical ID, but neither is required
for normal use.

`@@12` is a transcript handle, not a generic reference sigil: it means the
twelfth visible cell in this run view and must fail outside that scope. A
durable Piku object uses an explicit identity such as
`ref:run:<id>:cell:<sequence>`. File locations are distinct structured values
internally (`uri` plus a half-open range), rendered for people as
`path:line:column..line:column`. Piku must never guess that a file-shaped
string is a cell reference. Temporary hint labels use `%name`; they disappear
after use and are never persisted.

Vim/EasyMotion and browser hint modes are useful only as a speed layer. A
`reference.hint` action overlays short labels on eligible *visible* targets;
typing the label selects, opens, attaches, or forks according to the chosen
action. The labels disappear after use and are never persisted or sent to the
model. They are coordinates, not names. This preserves the satisfying
“letters appear; jump there” interaction without confusing an ephemeral `af`
label with a durable source citation.

The smallest initial surface is therefore:

```text
select object → attach it to the prompt → visible @reference chip → submit
hint visible object → select it → inspect / attach / fork
```

No automatic semantic extraction, named-entity system, or ambient “the thing I
just looked at” heuristic is required for that to work.

## Interleaved agents are a room, not a task tree

A multi-agent conversation is not the same feature as subagent delegation.
Delegation creates a child job with an owner, a result handoff, and normally a
separate completion boundary. An interleaved room instead has one chronological
run in which the person deliberately routes successive turns to named actors:
researcher, critic, implementer, or another configured provider/model. Each
turn remains a first-class cell in the same visible sequence.

The record must attach an actor identity to every turn. The first increment
derives it from the recorded provider/model provenance, so inspection can never
lose who produced a cell. A configurable actor registry is the next increment:
an actor has a durable local name, display label, provider/model profile, and a
separate private working context. The room then shares only explicitly chosen
material: user messages, attached references, and published cell outputs. It
does **not** silently concatenate every assistant response into every actor's
provider history. That separation prevents a critic from inheriting a
researcher's private scratch context merely because the two appear adjacent on
screen.

Routing is a foreground selection, not a prompt incantation: select an actor,
write the next message, and submit. The timeline visibly labels each cell with
that actor; filtering or focusing an actor never reorders the underlying
chronology. An actor can be put in the background only by creating a Piku job,
which is then shown as a separate run rather than masquerading as an
interleaved reply. This keeps three meanings distinct: **room turn**,
**delegated job**, and **shell job**.

The initial UI needs only `actor.select`, `actor.list`, and an actor label on
each cell. Context sharing, role-specific instructions, and concurrent
turn-taking remain explicit follow-on decisions, gated on the actor registry
and a test showing that an actor receives only the room material selected for
it.

### Delegated-job handoffs

When a delegated subagent finishes, its parent room receives a durable
**handoff notice**, not a fabricated assistant reply and not an automatic model
interjection. The notice contains task state, a compact summary, and a stable
child-run reference; the person can inspect or attach that result deliberately.
The parent agent receives it as model context only when a routing policy says
so or when it explicitly joins the task.

Recursive delegation uses the same rule at every edge. A descendant reports to
its immediate parent first. Bubbling to the attended room is opt-in per edge
and coalesces into one summary while that ancestor is busy. This retains the
pleasant “my work came back” feeling without turning a deep task tree into a
noisy, nondeterministic conversation or leaking a descendant's private context
past its parent.

The per-edge delivery choice is intentionally small: `join` (the parent asks
for the full result), `notice` (a compact durable handoff in the parent room),
or `bubble` (forward that compact notice one level upward). A `bubble` chain
may reach the top-level room, but only after every edge on that path opted in;
it carries the child-run reference rather than recursively copying the result.
Failures always produce a durable notice at their direct parent, while upward
failure bubbling follows the same opt-in rule. This is predictable enough to
configure and safe enough for recursive trees.

## Foreground and background

There are three different concepts that must not share one overloaded word:

- **OS shell job:** owned by the invoking shell and controlled by its native
  process group (`Ctrl-Z`, `fg`, `bg`, `jobs`). Piku respects it; it does not
  replace or falsify the shell's job table.
- **Foreground Piku run:** the currently attended cell. It receives streamed
  rendering, interjections, and an explicit interrupt action.
- **Background Piku run:** a Piku-owned durable run that remains visible in a
  compact jobs list and can be reattached. It never writes unexpected streaming
  bytes into the foreground prompt.

Foreground/background is presentation ownership, not authority escalation. A
background run retains its original tool policy and cwd. If it needs permission
or a user answer, it moves to an explicit `waiting_permission` or
`waiting_input` state and notifies the operator; it may not approve itself or
hide a blocking question.

The minimum agent-job actions are `run.background`, `run.foreground`,
`run.interrupt`, `run.wait`, and `run.list`. They share the same stable run ID
and lifecycle state in CLI, TUI, and web. Reattaching renders the retained cell
and subsequent events; it does not ask the model to reconstruct them.

Piku must not promise shell-like detachment until a run supervisor can outlive
the TUI/CLI process and reconnect safely. Today spawned background subagents use
a `DevNullSink` that discards their display output; that is incompatible with
inspectable background work. Before exposing background cells as a primary
operator feature, route their events and complete artifacts into the durable
record, then test exit, reconnect, cancellation, permission wait, and
concurrent-worktree behavior. Until then, `Ctrl-Z` continues to mean native
shell suspension of the Piku process and its foreground work, exactly as a
terminal user expects.

### Notification and attention policy

The jobs rail is quiet by default. A completed background run, failure, or
permission wait changes its state and increments a notification count without
stealing the prompt, moving the viewport, or injecting arbitrary output. The
operator chooses notification thresholds (for example, only after a duration or
only when unfocused) and can `run.wait @r42` at a deliberate boundary.

This follows the shell’s useful habit of surfacing background status near a
prompt instead of interrupting a command stream. A notification is never the
only record: it points to the retained job/run, exact output artifact, exit
state, and next available action.

## Explore and garden are one loop

Piku should support two complementary ways of attending to the same work:

- **Explore** is temporal and generative: ask, run, inspect, interrupt, compare,
  and follow a lead. Its primary object is the live cell and its evidence.
- **Garden** is spatial and curatorial: name a useful result, connect it to
  related evidence and repository artifacts, annotate uncertainty, and return to
  it later. Its primary objects are human-authored notes, claims, and links.

They are lenses over one durable record, not separate products or modes that
copy data between themselves. A person should be able to promote a selected
cell or a precise result span into a garden item with one action. Promotion
creates a new human-authored item that links to the immutable source event and
artifact digest; it does not duplicate the full output, silently convert a
model statement into fact, or mutate the originating cell. The garden item
shows source, status, and revision so it remains possible to distinguish a
useful provisional note from verified evidence.

Conversely, a garden item can reopen its source cell, artifact, or fork point
and start a new exploration child. That round trip is the intended workflow:

```text
question → live cell → inspect result → promote a useful piece → connect/annotate
         ↑                                                        ↓
         └──────────── reopen evidence or branch a fresh inquiry ┘
```

Piku's existing workspace, card, source, and human-conclusion concepts are the
right landing places for this; the cell layer supplies addressable provenance
and immediate inspection. Layout and open-pane state are presentation
preferences, while promoted notes and their source links are durable work.

JupyterLab points in this direction more than classic Notebook: it combines
notebooks, text documents, terminals, file browser, search, an inspector, and
persisted workspace layout. Its centralized command registry and collapsible,
scrollable outputs also support an extensible interaction grammar. But it does
not by itself make a reliable explore-to-curate transition: notebook output can
still be tied to hidden kernel state, while UI workspace state is not knowledge
provenance. Piku should retain JupyterLab's composable work area while making
the source link and lifecycle state explicit.

## Cell contract

### Operator interactions are receipts, not cells

An operator command is retained as one durable interaction receipt: typed input,
semantic action identity, retained result when one was deliberately captured,
and success or failure. This preserves the shell-like “I entered this and it
did that” relationship without treating navigation as substantive work.

Cells remain reserved for agent turns, captured shell commands, background-job
results, and explicit forks. In particular, `/cells` reads the run snapshot
that preceded its own receipt, so it cannot recursively create an empty cell
for the act of listing cells. A later activity/audit projection may show the
receipt and its full result; the cell list stays focused on inspectable work.

Native shell passthrough records its invocation and exit state but leaves its
output unavailable unless the operator chose the explicit capture path. The
captured shell path is a cell because its result is substantive retained work.

Every displayed cell has stable identity and these visible states:

```text
PromptRevision
  ├─ queued | running | completed | interrupted | failed
  ├─ parent cell / fork origin (if any)
  ├─ immutable input snapshot and selected context revision
  ├─ ordered events: assistant deltas, tool calls, results, permissions,
  │  effects, verification, warnings, and compaction boundaries
  └─ complete content references with byte count, digest, and availability
```

A projection may coalesce routine streaming activity into a preview, but it
must preserve a one-action path to the full event and artifact content. It must
not auto-collapse an error, a denied permission, a mutation, a verification
failure, an interruption, or an unavailable artifact. Truncation is always
labeled with its bound and a path to the complete referenced artifact where the
operator is authorized to read it.

The UI distinguishes:

- **selected cell**: the navigation target;
- **expanded cell**: its complete event/result view is shown;
- **focused pane**: prompt, cell list, output viewport, detail, or search;
- **run state**: durable lifecycle state, never inferred from cursor position;
- **context state**: a visible compaction or source-revision boundary.

This is deliberately not the same as model context. A cell can remain fully
inspectable after a provider compacts its active context, and the compaction
event says exactly what changed.

## Editing, interruption, and forks

Editing must feel direct without rewriting history:

| Operator intent | Result |
| --- | --- |
| Change a submitted prompt | open an editor seeded from that prompt; submit creates a child prompt revision and run |
| Continue from a completed or interrupted cell | create a child run with an explicit parent and inherited-context summary |
| Steer a live run | append an attributed interjection event to that run |
| Stop a live run | record `interrupted`; preserve partial output and every completed tool event |
| Revisit a prior branch | select it; never overwrite a sibling branch |

`Esc` is a state machine, not a destructive shortcut. In a modal view it first
returns focus to the prompt or closes the transient overlay. During a running
cell it offers or performs the explicit `run.interrupt` action and immediately
leaves the partial cell inspectable. A second, deliberate action may open
rewind/fork selection, but may never silently erase the run or mutate its
record. This retains the useful “escape mid-generation, then continue from
there” behavior while keeping the fork visible.

## Configurable interaction grammar

Piku defines semantic actions once. Each surface binds them through a profile,
with user-global configuration overridden by project-local configuration where
appropriate. Runtime events never contain keyboard choices.

Minimum actions:

```text
cell.next / cell.previous / cell.expand / cell.collapse_all
output.next / output.previous / output.open_artifact
viewport.up / viewport.down / viewport.left / viewport.right
focus.prompt / focus.cells / focus.output / focus.detail
search.open / search.next / search.previous
reference.hint / reference.insert_selected / reference.open
prompt.edit_fork / prompt.submit / prompt.steer
run.interrupt / run.resume_from_here / run.background / run.foreground / run.wait / run.list
history.show_forks
```

The default profile exposes discoverable, terminal-safe bindings and a command
palette. An optional `vim` profile can bind these actions to a familiar modal
grammar: `j`/`k` select cells, `]o`/`[o` move between output cells, Enter
expands, `/` searches, Shift-j/k and Shift-h/l scroll the viewport, and `gP`
closes previews. This takes inspiration from a common editor workflow but is
not Piku's required keymap; concrete bindings remain replaceable. In particular,
terminal-unreliable chords such as Ctrl-h are never required for baseline use.

Configuration validation rejects duplicate bindings within one focus context,
unknown actions, and unrepresentable terminal sequences. Help renders the
currently active bindings rather than documenting a fixed set.

## Surface behavior

### Terminal

Preserve the sticky prompt and ordinary terminal scrollback. Above it, render a
compact cell list with a selected-cell preview. Expanding a cell enters a
bounded viewport whose output can be scrolled independently; the terminal's
native scrollback remains an escape hatch, not the only way to inspect output.

The full-result action opens the referenced artifact with the configured pager
or prints it safely to stdout in a non-interactive invocation. A shell-native
CLI should provide the same choice in composable form: enumerate cells/events,
select one by stable ID, then emit either a preview, complete text, or structured
events. Exact command names follow existing CLI conventions rather than this
design inventing a parallel command surface.

### Browser

The browser is an alternative projection, not a browser-owned conversation.
It renders the same cells, event IDs, artifact references, branches, and state
markers; it can add virtualization, side-by-side fork comparison, rich media,
and a read-only shareable run view. Live updates use the existing event stream.
Any mutation calls the same runtime action contract and carries the same
capability/authority checks as the TUI.

### Export and external harnesses

HTML or JSON exports are useful snapshots, but they are secondary. Imports from
Codex, Claude, or other harnesses map supported structured events into Piku's
record with source attribution. Their transcript files and UI behavior do not
become Piku's schema.

### Optional Codex execution

Codex has two deliberately different integration roles. The app-server
protocol is the interactive Codex executor: it supports an explicitly selected
thread, streamed turn events, interruption, and Piku's per-turn authority
lease. `codex exec --json` is a bounded non-interactive worker: it is suitable
for a one-shot proposal, annotation, review, or evaluator because its JSONL
stream and schema-constrained final response can be retained as evidence.

They are not interchangeable modes for the same action. A request chooses one
executor before it starts; Piku records the executor, sandbox, input boundary,
and retained event stream. An `exec` worker remains read-only unless the
operator grants a separately modeled write authority. It cannot silently reuse
an interactive Codex thread or inherit an ambient provider secret. LLM output
may propose an action or annotate evidence, but the shared typed action
catalog remains the authority for parsing, validation, and execution.

## Prior art: what to adopt and where to stop

### Pi

Pi is the closest terminal-first precedent. It keeps sessions as local JSONL
trees with stable IDs and parent links, supports navigating back to an entry and
branching, uses an external editor through `$VISUAL`/`$EDITOR`, and exposes
namespaced semantic keybinding IDs that extensions and UI hints share. Its
Escape behavior is also configurable: one Escape interrupts and a second can
open the tree or fork.

Adopt those interaction seams: local addressable session data, edit-to-branch,
semantic action IDs, active-keymap hints, extension render hooks, and ordinary
editor handoff. Pi's `Ctrl+O` is useful evidence that collapsible tool results
matter, but it is a display-wide toggle. Piku needs selected-cell and
selected-result expansion, independent output viewports, and a shell query path
to the full content.

Pi's `!command` streams a direct shell command and adds its result to the next
prompt, while `!!command` omits it from model context; large results are
bounded. That is a good acknowledgement that shell work and model context are
different, but Piku should make the distinction durable and inspectable: viewing
an output must never imply context capture, and capture must retain source,
bound, and operator intent.

### OpenCode and Aider

OpenCode provides a useful separation between its interactive TUI and headless
server/CLI, plus JSON session export. It supports the principle that the agent
surface should be queryable beyond one screen, though an export alone does not
make a live result pleasant to inspect. Aider reinforces the lower-friction
terminal pattern: external-editor prompt composition, vi/emacs input modes,
shell history, and direct Git/diff commands. Neither supplies the complete
cell-and-garden model; Piku should preserve their escape hatches rather than
replace them with a richer closed UI.

## Implementation sequence

1. **Projection contract.** Add a surface-neutral `CellView` derived only from
   existing `RunEventEnvelope` and artifact references. It groups a prompt
   revision with its event range, state, ancestry, salient markers, previews,
   and expandable content handles. Each content handle has exact/raw and
   rendered/safe forms plus cwd, timing, producer, exit state, and provenance.
   Add deterministic tests for partial output, tool failure, cancellation,
   compaction, unavailable artifacts, forks, raw/rendered non-confusion, and
   attribution loss under concurrent writers.
2. **Action/config contract.** Add presentation-only semantic action IDs,
   focus contexts, profile inheritance, user/project precedence, validation,
   and a machine-readable help projection. Do not add key bindings to
   `piku-runtime`.
3. **TUI proof.** Implement cell selection, expansion, independent output
   viewport, search, artifact opening, interrupt, and prompt-edit-to-fork on
   the existing sticky-bottom REPL. Preserve plain streaming and terminal
   scrollback as a no-config fallback. Enforce one TUI stdout writer; route
   asynchronous events through the renderer queue. Add optional OSC 7/133
   markers only when terminal/shell compatibility is established.
4. **CLI projection.** Expose stable cell/event selection and full-content
   output for scripts and pagers, alongside JSONL. It must never require model
   regeneration to reveal already-recorded content.
5. **Web convergence.** Rework the existing SSE/run-view rendering to consume
   `CellView`; add virtualized cells, fork comparison, and responsive bindings.
   Test equivalence with the terminal projection against the same fixture run.
6. **Garden bridge.** Add promote/open-source actions from cells and result
   spans into the existing workspace/note/card concepts. Require source event,
   artifact digest, author, and epistemic status on promotion; test that edits
   to a garden note do not alter the source cell and that reopening creates a
   child run rather than a rerun-in-place.

## Acceptance gates

- A 10,000-line tool result remains readable through a selected cell and full
  artifact action without copying it into model context or rerunning the model.
- Interrupting during assistant streaming leaves a complete partial cell that
  can be inspected and forked; no result is silently discarded.
- Editing an old prompt makes an observable child branch and leaves the parent
  output unchanged.
- The same fixture run produces equivalent cell identity, lifecycle state,
  salient markers, and full-content references in TUI, CLI, and browser tests.
- A custom profile changes bindings but not event sequence, branch structure,
  permission decisions, or artifact authorization.
- Selecting a result span attaches an explicit stable reference to a prompt;
  a temporary hint label never appears in the recorded prompt or durable event.
- Errors, mutations, permission denials, interruption, and compaction remain
  visible in the compact view.
- A promoted garden note carries a stable link to its source cell/event and its
  own author/status; changing either side never silently rewrites the other.
- A pager/editor handoff and an interactive shell command receive a real
  foreground PTY; their control sequences never enter captured output.
- Concurrent output is either independently logged and attributed by producer,
  or visibly marked mixed/unknown. The product never assigns it by line order.
- A full-screen TUI remains intact when a background job, hook, or plugin emits
  an event; all bytes pass through its sole renderer or a separate artifact.
- A background completion/permission wait does not move the foreground prompt
  or viewport; its retained output can be opened by stable run ID.

## Non-goals

- A mutable Python-style kernel or out-of-order executable cells.
- Replacing repository files, Git history, or explicit artifacts with a
  proprietary notebook document.
- Treating human PTY bytes as agent chat or granting the agent control of the
  human terminal.
- Prescribing one person's Vim mappings as Piku's universal interface.
- Claiming an undo, replay, output owner, or Git attribution that the recorded
  evidence cannot establish.

## Evidence

Jupyter's documentation describes a document that combines inputs, outputs,
and explanatory text, attaches results to the generating cell, and separates
command from edit mode. It is the positive interaction precedent. Its persistent
kernel model also shows the relevant danger: the kernel can remain active
independently of the visible browser document. Notebook-lineage research finds
that edits, reordering, and reruns make hidden state diverge from visible cells.

- https://jupyter-notebook.readthedocs.io/en/4.x/notebook.html
- https://jupyterlab.readthedocs.io/en/stable/user/interface.html
- https://jupyterlab.readthedocs.io/en/stable/user/commands.html
- https://pi.dev/docs/latest/sessions
- https://pi.dev/docs/latest/keybindings
- https://pi.dev/docs/latest/settings
- https://opencode.ai/docs/cli/
- https://aider.chat/docs/usage/commands.html
- https://www.nokia.com/bell-labs/unix-history/philosophy.html
- https://www.gnu.org/s/bash/manual/html_node/Job-Control-Basics.html
- https://man7.org/linux/man-pages/man1/less.1.html
- https://man7.org/linux/man-pages/man1/tmux.1.html
- https://www.inf.usi.ch/lanza/PUBS/P/MacI2025a.pdf
- https://arxiv.org/abs/2012.10206
- https://arxiv.org/abs/2603.10664
- https://wezterm.org/shell-integration.html
- https://ghostty.org/docs/features/shell-integration
- https://docs.warp.dev/terminal/blocks/
- https://docs.warp.dev/terminal/blocks/background-blocks/
- https://github.com/anomalyco/opencode/issues/8639
- https://arxiv.org/abs/2012.06981
- https://www.microsoft.com/en-us/research/publication/whats-wrong-with-computational-notebooks/
