set shell := ["bash", "-eu", "-o", "pipefail", "-c"]

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
    PIKU_AGENTIC_USER=1 PIKU_AGENTIC_MAX_TURNS={{turns}} cargo test --test agentic_user -p piku -- agentic_user_{{persona}} --nocapture

# Same harness, but seed the play dir from this repo's real code.
#
# This is the best default for realistic iteration: the agent works in a temp
# copy of the current repo, so the real tree stays untouched.
agentic-user-real persona="confident_dev" turns="1":
    PLAYDIR=$(mktemp -d) && rsync -a --delete --exclude target --exclude .git ./ "$PLAYDIR/repo/" && PIKU_AGENTIC_USER=1 PIKU_AGENTIC_SCENARIO=repo PIKU_AGENTIC_MAX_TURNS={{turns}} PIKU_AGENTIC_PLAYDIR="$PLAYDIR/repo" cargo test --test agentic_user -p piku -- agentic_user_{{persona}} --nocapture

# Full multi-turn run for a persona.
#
# Usage:
#   just agentic-user-full
#   just agentic-user-full confident_dev
agentic-user-full persona="confident_dev":
    PIKU_AGENTIC_USER=1 PIKU_AGENTIC_FULL=1 cargo test --test agentic_user -p piku -- agentic_user_{{persona}} --nocapture

# Run the report-first dogfood suite.
dogfood:
    PIKU_DOGFOOD=1 cargo test --test dogfood -p piku -- --nocapture
