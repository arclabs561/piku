# piku

`piku` is a Rust agentic coding harness with a sticky-bottom TUI, tool execution,
session persistence, and local dogfood workflows.

## Layout

```text
.
├── crates/
│   ├── piku/         # CLI + TUI entrypoint
│   ├── piku-api/     # provider clients and streaming/event types
│   ├── piku-runtime/ # agent loop, session, permissions
│   └── piku-tools/   # built-in tools
├── tests/fixture/    # isolated play-dir used by agentic dogfood
└── justfile          # repo entrypoints
```

## Build

```bash
cargo build --workspace
```

## Test

`just check` runs the full gate (fmt, clippy with `-D warnings`, tests, release
build), the exact commands CI runs, defined once in `scripts/ci.sh`:

```bash
just check
```

Just the default test suite (fast, deterministic, no live LLM, no PTY):

```bash
cargo test --workspace
```

The PTY smoke tests (`tui_smoke`) drive the real binary over a pseudo-terminal;
they are isolated into their own stage because they stall under full-workspace
concurrency. `just check` runs them; standalone:

```bash
cargo test --test tui_smoke -- --ignored
```

The end-to-end suites that drive a real LLM (`llm_e2e`, `dogfood`, and the
`agentic_user` personas) are `#[ignore]`d so the default run reports them as
*ignored* rather than silently passing. They are opt-in, need a provider key,
and run per-suite (not `--workspace -- --ignored`, which would also wake the PTY
tests):

```bash
export OPENROUTER_API_KEY=sk-or-...        # or ANTHROPIC_API_KEY / GROQ_API_KEY
cargo test --test llm_e2e -- --ignored
cargo test --test dogfood -- --ignored
just agentic-user                          # one persona; see justfile for more
```

Live model matrix runs are separate from PR CI. Local examples:

```bash
just live
just live-random
just live-dogfood

PIKU_LIVE_PROVIDER=openrouter \
PIKU_LIVE_MODEL=anthropic/claude-sonnet-4-5 \
PIKU_LIVE_KEY_VAR=OPENROUTER_API_KEY \
./scripts/ci.sh live

PIKU_LIVE_LEDGER="$PWD/target/live-ledger/local.jsonl" ./scripts/ci.sh live-random
```

Local live runs write a JSONL ledger under `target/live-ledger/` unless
`PIKU_LIVE_LEDGER` is set. Use an absolute path when overriding it.

To export local PR and issue data for dogfood prompts:

```bash
just github-corpus
just github-prompt
just github-dogfood
PIKU_GITHUB_CORPUS_LIMIT=100 just github-corpus owner/repo
```

## Run

Interactive (TUI REPL):

```bash
cargo run -p piku -- --help
piku "explain src/main.rs"
piku --read-only
```

Headless (run once, print, exit; for scripts and pipelines, like `aider -m` or
`claude -p`):

```bash
piku -p "explain src/main.rs" > explanation.txt
piku --read-only "inspect this repo and suggest the next fix"
```

## Dogfood

Default isolated smoke run:

```bash
just agentic-user
```

Run against a temp copy of this repo's real code:

```bash
just agentic-user-real
```

Use a custom play dir:

```bash
PIKU_AGENTIC_PLAYDIR=/path/to/playdir just agentic-user
```

Full multi-turn mode:

```bash
just agentic-user-full
```

Terminal playground with a viewport-observing user agent, keyboard-level actions,
and an append-only local evidence ledger. A primary meta-judge and one bounded
recursive observer separately review judging quality and piku's behavior; both
reviews become ledger records:

```bash
just playground confident_dev 6
just playground-sample adversarial 8 42
```

Each opt-in run appends turn evidence and its final meta-review to
`target/agentic-findings/playground.jsonl`. Set `PIKU_AGENTIC_LEDGER` to use a
different local path.

The simulated user is driven by the test harness's direct HTTPS client to
OpenRouter; piku itself is the real CLI in a PTY. `just playground` pins
OpenRouter for every role: GPT-5.6 Sol for piku, GPT-5.6 Terra for the simulated
user, and Claude Opus 5 for the primary judge plus recursive observer. Override
the three model arguments in the recipe when comparing a hypothesis. Advanced
use can pin any role with `PIKU_AGENTIC_{USER,PIKU,JUDGE}_PROVIDER=openrouter`
and its matching `*_MODEL`. Every resolved provider/model pair is recorded as a
credential-free `config` entry. The harness reads an optional local `.env` as
dotenv data; it never shell-sources that file or places credentials in a child
process command line.

`just playground-sample` rotates piku and simulated-user models from a small
OpenRouter pool: GPT-5.6 Sol/Terra/Luna, DeepSeek V4 Flash/Pro, MiMo-V2.5, and
Hy3. Claude Opus 5 remains the judge anchor so role variation does not confound
review quality. Its decimal seed and resolved assignment are ledgered, so an
interesting run can be replayed. Permission prompts are also ledgered with the
harness response; the default is one-time approval (`y`). Set
`PIKU_AGENTIC_PERMISSION_RESPONSE=n` for an explicit-denial replay.

The point of the playground is to improve piku, not accumulate model prose.
Every completed run ends with an `improvement_handoff`: deterministic failures
and ungrounded-review detections are **verified findings** to reproduce and
fix; model-reported bugs are **hypotheses** that require reproduction before
changing piku. Judge records cite observed turns, and the recursive observer
can invalidate an ungrounded primary review. The terminal output states the
next action, and the handoff is append-only evidence for the next engineering
session.
It also writes a deterministic JSON development-context packet next to the
ledger (`target/agentic-findings/development-context/`): use that packet, not
free-form judge prose, to select the next piku change and its validation run.
The packet carries a bounded history of prior deterministic findings (including
recurrence), never unverified historical model allegations.

A persona with a filesystem task also carries a scenario contract: a stated
goal plus executable acceptance checks run against the workspace piku edited
(file contents, and a bounded `cargo test`). The contract and its per-check
results are ledgered as `scenario_contract` and in the development-context
packet. A failed acceptance check outranks every screen-level finding and sets
the next action to `fix_piku_for_failed_scenario_acceptance`, because it is
measured against the workspace rather than inferred from the terminal.
Readiness checks say the terminal behaved; acceptance checks say the work
succeeded, and a run that produced plausible prose with a failing workspace is
a failed run.

Every LLM review call reports one of `valid`, `provider_failure`, or
`invalid_json`. Invalid JSON gets one schema-specific repair attempt; a
provider failure returns immediately rather than paying for a retry of a
transport or quota error, and neither falls back to another LLM layer. A
non-valid outcome contributes no observations and is recorded as a harness
finding, so "the judge never ran" is never filed as "the judge found the
review ungrounded" or as a piku defect.

Findings are stamped with the piku revision they were observed against. A run
inherits only the deterministic findings reproduced against the build it is
testing; anything last seen on an older build is named as unreproduced and
treated as closed until a run reproduces it. History that outlives the code it
described otherwise steers each run at problems that may already be fixed.

Runs come in two roles, recorded in the config ledger with the piku revision.
`just playground-control` pins models and seed, so the same control on two
builds is a real before-and-after. `just playground-sample` randomizes to
reach failure shapes a fixed pair never hits. `just playground-paired` runs
one of each: comparing two randomized runs measures the sample, not the
change.
