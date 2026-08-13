set shell := ["bash", "-eu", "-o", "pipefail", "-c"]

# Run the full check gate: the exact commands CI runs. Defined in scripts/ci.sh
# so local and CI can't drift.
check:
    ./scripts/ci.sh all

# Individual gate stages (same source as `just check` and CI).
fmt:
    ./scripts/ci.sh fmt
scripts:
    ./scripts/ci.sh scripts
clippy:
    ./scripts/ci.sh clippy
web:
    ./scripts/ci.sh web
web-bundle:
    ./scripts/ci.sh web-bundle
web-harness:
    ./scripts/ci.sh web-harness

# Test the web workspace against an already-running local Piku server.
web-e2e url="http://127.0.0.1:9090":
    cd crates/piku/web-ui && PIKU_WEB_URL={{url}} npm run test:e2e

# Run a full Codex + Playwright user journey against the rendered workspace.
# The agent manipulates the page, observes screenshots, challenges its own
# findings, and leaves a structured report plus JSONL tool evidence.
web-agent url="http://127.0.0.1:9090":
    cd crates/piku/web-ui && npm run test:agent -- --url {{url}}

# Explicit alias for discoverability in QA workflows.
web-qa-agent url="http://127.0.0.1:9090":
    just web-agent {{url}}

# Run two isolated Codex + Playwright explorers concurrently, then give their
# raw evidence packets to a fresh synthesis process. Each explorer owns a
# temporary Piku surface and browser context; all outcomes enter the shared
# evaluation ledger.
web-agent-parallel url="http://127.0.0.1:9090":
    cd crates/piku/web-ui && PIKU_WEB_URL={{url}} npm run test:agent:parallel

# Run a same-revision blinded/focused pair sequentially. The focus ledger must
# be a private operator-owned file outside this workspace.
web-agent-focus-pair focus pair_ordinal="0" url="http://127.0.0.1:9090":
    cd crates/piku/web-ui && PIKU_WEB_URL={{url}} PIKU_EVAL_FOCUS_EVENTS={{focus}} PIKU_EVAL_PAIR_ORDINAL={{pair_ordinal}} npm run test:agent:focus-pair

# Verify the installed Codex app-server enforces Piku's intended read/write
# sandbox boundary without invoking a model.
codex-sandbox-probe:
    node scripts/codex-app-server-probe.mjs

# Generate the private, versioned browser write attestation with an explicitly
# selected native Codex payload. Wrapper scripts are rejected.
codex-write-attest executable:
    node scripts/codex-app-server-probe.mjs --interactive --executable "{{executable}}" --attestation "${XDG_CONFIG_HOME:-$HOME/.config}/piku/_codex/workspace-write-attestation.json"

# Inspect or mutate the private append-only focus ledger through its single
# authoritative validator. Remaining arguments are passed to the Node CLI.
eval-focus *args:
    node scripts/evaluation-focus-cli.mjs {{args}}

# Summarize shared CLI and web live-evaluation evidence.
eval-summary ledger="target/live-ledger":
    node scripts/evaluation-summary.mjs {{ledger}}

# Fast contract tests for the shared evaluation ledger and web judge harness.
eval-harness-test:
    node --test scripts/evaluation-summary.test.mjs scripts/evaluation-focus.test.mjs scripts/evaluation-focus-cli.test.mjs scripts/codex-app-server-probe.test.mjs
    cd crates/piku/web-ui && npm run test:harness

# Run local live LLM smoke tests and write a ledger under target/live-ledger.
live:
    ./scripts/ci.sh live

# Pick one available local provider/model row and write a ledger.
live-random:
    ./scripts/ci.sh live-random

# Run the report-first live dogfood suite through one random available row.
live-dogfood:
    PIKU_LIVE_SUITE=dogfood ./scripts/ci.sh live-random

# Export GitHub PR and issue artifacts for local dogfood.
github-corpus repo="":
    if [ -n "{{repo}}" ]; then ./scripts/github-corpus.sh "{{repo}}"; else ./scripts/github-corpus.sh; fi

# Build a dogfood prompt seed from the latest exported GitHub corpus.
github-prompt pr="":
    if [ -n "{{pr}}" ]; then ./scripts/github-corpus-prompt.sh "" "{{pr}}"; else ./scripts/github-corpus-prompt.sh; fi

# Run a corpus prompt against a temp repo copy and append a live ledger row.
github-dogfood pr="":
    if [ -n "{{pr}}" ]; then ./scripts/github-corpus-run.sh "{{pr}}"; else ./scripts/github-corpus-run.sh; fi

# Quick, executable agentic-user smoke test.
#
# Usage:
#   just agentic-user
#   just agentic-user cautious_beginner
#   just agentic-user adversarial 3
#
# Optional:
#   PIKU_AGENTIC_PLAYDIR=/path/to/playdir just agentic-user
agentic-user persona="confident_dev" turns="1":
    PIKU_AGENTIC_MAX_TURNS={{turns}} cargo test --test agentic_user -p piku -- agentic_user_{{persona}} --ignored --nocapture

# Same harness, but seed the play dir from this repo's real code.
#
# This is the best default for realistic iteration: the agent works in a temp
# copy of the current repo, so the real tree stays untouched.
agentic-user-real persona="confident_dev" turns="1":
    PLAYDIR=$(mktemp -d) && rsync -a --delete --exclude target --exclude .git ./ "$PLAYDIR/repo/" && PIKU_AGENTIC_SCENARIO=repo PIKU_AGENTIC_MAX_TURNS={{turns}} PIKU_AGENTIC_PLAYDIR="$PLAYDIR/repo" cargo test --test agentic_user -p piku -- agentic_user_{{persona}} --ignored --nocapture

# Full multi-turn run for a persona.
#
# Usage:
#   just agentic-user-full
#   just agentic-user-full confident_dev
agentic-user-full persona="confident_dev":
    PIKU_AGENTIC_FULL=1 cargo test --test agentic_user -p piku -- agentic_user_{{persona}} --ignored --nocapture

# Interactive-terminal playground: OpenRouter models observe piku's VT100
# viewport, choose one keyboard-level action per turn, and append evidence plus
# review records to target/agentic-findings/playground.jsonl.
#
# Usage:
#   just playground
#   just playground adversarial 8
#   just playground confident_dev 4 openai/gpt-5.6-sol openai/gpt-5.6-terra anthropic/claude-opus-5
#   PIKU_AGENTIC_LEDGER=/tmp/piku-playground.jsonl just playground confident_dev 4
playground persona="confident_dev" turns="6" piku_model="openai/gpt-5.6-sol" user_model="openai/gpt-5.6-terra" judge_model="anthropic/claude-opus-5":
    PIKU_AGENTIC_PIKU_PROVIDER=openrouter PIKU_AGENTIC_PIKU_MODEL={{piku_model}} PIKU_AGENTIC_USER_PROVIDER=openrouter PIKU_AGENTIC_USER_MODEL={{user_model}} PIKU_AGENTIC_JUDGE_PROVIDER=openrouter PIKU_AGENTIC_JUDGE_MODEL={{judge_model}} PIKU_AGENTIC_MAX_TURNS={{turns}} cargo test --test agentic_user -p piku -- agentic_user_{{persona}} --ignored --nocapture

# Regression baseline: pinned models, pinned seed, ledgered as run_role=control
# alongside the piku revision. Compare a control to the same control on another
# build. Comparing two randomized runs measures the sample, not the change.
#
# Usage:
#   just playground-control
#   just playground-control adversarial 8
playground-control persona="adversarial" turns="6":
    PIKU_AGENTIC_RUN_ROLE=control PIKU_AGENTIC_MODEL_SELECTION_SEED=1 PIKU_AGENTIC_PIKU_PROVIDER=openrouter PIKU_AGENTIC_PIKU_MODEL=openai/gpt-5.6-terra PIKU_AGENTIC_USER_PROVIDER=openrouter PIKU_AGENTIC_USER_MODEL=openai/gpt-5.6-terra PIKU_AGENTIC_JUDGE_PROVIDER=openrouter PIKU_AGENTIC_JUDGE_MODEL=anthropic/claude-opus-5 PIKU_AGENTIC_MAX_TURNS={{turns}} cargo test --test agentic_user -p piku -- agentic_user_{{persona}} --ignored --nocapture

# One control run for comparison, then one randomized discovery run. Run this
# after a piku change: the control says whether the change moved the baseline,
# the discovery run looks for failure shapes the fixed pair never reaches.
#
# Usage:
#   just playground-paired
#   just playground-paired adversarial 8 42
playground-paired persona="adversarial" turns="6" seed="":
    just playground-control {{persona}} {{turns}}
    PIKU_AGENTIC_RUN_ROLE=discovery just playground-sample {{persona}} {{turns}} {{seed}}

# Sample a reproducible OpenRouter subject/user assignment. The judge stays on
# Claude Opus 5 as a calibration anchor; selected models and seed are recorded
# in the config ledger, so a finding can be replayed exactly.
# Usage: just playground-sample adversarial 8 42
playground-sample persona="adversarial" turns="6" seed="":
    bash -eu -o pipefail -c 'seed="$1"; if [ -z "$seed" ]; then seed="$(date +%s)"; fi; case "$seed" in *[!0-9]*) printf "seed must be decimal: %s\\n" "$seed" >&2; exit 2;; esac; RANDOM="$seed"; piku_models=(openai/gpt-5.6-sol openai/gpt-5.6-terra deepseek/deepseek-v4-flash deepseek/deepseek-v4-pro xiaomi/mimo-v2.5 tencent/hy3); user_models=(openai/gpt-5.6-terra openai/gpt-5.6-luna deepseek/deepseek-v4-flash xiaomi/mimo-v2.5 tencent/hy3); piku_model="${piku_models[RANDOM % ${#piku_models[@]}]}"; user_model="${user_models[RANDOM % ${#user_models[@]}]}"; judge_model="anthropic/claude-opus-5"; printf "playground sample seed=%s piku=%s user=%s judge=%s\\n" "$seed" "$piku_model" "$user_model" "$judge_model"; PIKU_AGENTIC_MODEL_SELECTION_SEED="$seed" PIKU_AGENTIC_PIKU_PROVIDER=openrouter PIKU_AGENTIC_PIKU_MODEL="$piku_model" PIKU_AGENTIC_USER_PROVIDER=openrouter PIKU_AGENTIC_USER_MODEL="$user_model" PIKU_AGENTIC_JUDGE_PROVIDER=openrouter PIKU_AGENTIC_JUDGE_MODEL="$judge_model" PIKU_AGENTIC_MAX_TURNS="$2" cargo test --test agentic_user -p piku -- "agentic_user_$3" --ignored --nocapture' -- '{{seed}}' '{{turns}}' '{{persona}}'

# Run the report-first dogfood suite.
dogfood:
    cargo test --test dogfood -p piku -- --ignored --nocapture
