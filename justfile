set shell := ["bash", "-eu", "-o", "pipefail", "-c"]

# Run the full check gate — the exact commands CI runs (fmt, clippy -D warnings,
# test, build). Defined in scripts/ci.sh so local and CI can't drift.
check:
    ./scripts/ci.sh all

# Individual gate stages (same source as `just check` and CI).
fmt:
    ./scripts/ci.sh fmt
clippy:
    ./scripts/ci.sh clippy

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

# Run the report-first dogfood suite.
dogfood:
    cargo test --test dogfood -p piku -- --ignored --nocapture
