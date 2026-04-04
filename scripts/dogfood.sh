#!/usr/bin/env bash
# dogfood.sh — run piku's dogfood test suite
#
# USAGE:
#   ./scripts/dogfood.sh [SCENARIO_FILTER]
#
# EXAMPLES:
#   ./scripts/dogfood.sh                        # run all dogfood scenarios
#   ./scripts/dogfood.sh multifile_rename        # run one scenario by name
#   ./scripts/dogfood.sh trace                   # run scenarios matching "trace"
#
# REQUIREMENTS:
#   - ANTHROPIC_API_KEY or OPENROUTER_API_KEY set in env
#   - cargo build --release -p piku completed (or run with --build flag)
#
# FLAGS:
#   --build     Run cargo build --release before the tests
#   --fast      Use the llm_e2e suite (faster, cheaper models) instead of dogfood
#   --help      Print this message
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── Colours ─────────────────────────────────────────────────────────────────
RED="\033[0;31m"; GREEN="\033[0;32m"; YELLOW="\033[0;33m"
CYAN="\033[0;36m"; BOLD="\033[1m"; DIM="\033[2m"; RESET="\033[0m"

# ── Defaults ─────────────────────────────────────────────────────────────────
BUILD=0
FAST=0
FILTER=""

# ── Arg parse ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --build) BUILD=1; shift ;;
        --fast)  FAST=1;  shift ;;
        --help|-h)
            sed -n '2,/^set /p' "$0" | grep '^#' | sed 's/^# \?//'
            exit 0
            ;;
        -*) echo "unknown flag: $1" >&2; exit 1 ;;
        *)  FILTER="$1"; shift ;;
    esac
done

# ── API key check ─────────────────────────────────────────────────────────────
if [[ -z "${ANTHROPIC_API_KEY:-}" && -z "${OPENROUTER_API_KEY:-}" ]]; then
    echo -e "${RED}error:${RESET} ANTHROPIC_API_KEY or OPENROUTER_API_KEY must be set"
    exit 1
fi

echo -e "${BOLD}piku dogfood runner${RESET}  $(date '+%Y-%m-%d %H:%M')"
echo -e "${DIM}repo: $REPO_ROOT${RESET}"
echo ""

# ── Build ─────────────────────────────────────────────────────────────────────
if [[ $BUILD -eq 1 ]]; then
    echo -e "${CYAN}→ building piku (release)…${RESET}"
    cargo build --release -p piku 2>&1 | tail -5
    echo ""
fi

# Verify binary exists
BINARY="$REPO_ROOT/target/release/piku"
if [[ ! -f "$BINARY" ]]; then
    echo -e "${RED}error:${RESET} piku binary not found at $BINARY"
    echo "  Run: cargo build --release -p piku"
    echo "  Or:  ./scripts/dogfood.sh --build"
    exit 1
fi
echo -e "${DIM}binary: $BINARY  ($(stat -f '%z' "$BINARY" 2>/dev/null || stat -c '%s' "$BINARY" 2>/dev/null) bytes)${RESET}"
echo ""

# ── Test suite selection ──────────────────────────────────────────────────────
if [[ $FAST -eq 1 ]]; then
    SUITE="llm_e2e"
    GATE="PIKU_LLM_E2E=1"
    echo -e "${CYAN}suite: llm_e2e (fast/cheap models)${RESET}"
else
    SUITE="dogfood"
    GATE="PIKU_DOGFOOD=1"
    echo -e "${CYAN}suite: dogfood (full experience reports)${RESET}"
fi

# ── Run ───────────────────────────────────────────────────────────────────────
echo ""
START=$(date +%s)

# Build the cargo test command
CMD=(cargo test --test "$SUITE" --release)
if [[ -n "$FILTER" ]]; then
    CMD+=("$FILTER")
fi
CMD+=(-- --nocapture)

echo -e "${BOLD}Running:${RESET} ${GATE} ${CMD[*]}"
echo -e "${DIM}─────────────────────────────────────────────────${RESET}"
echo ""

export "$GATE"
"${CMD[@]}"
STATUS=$?

END=$(date +%s)
ELAPSED=$((END - START))

echo ""
echo -e "${DIM}─────────────────────────────────────────────────${RESET}"
if [[ $STATUS -eq 0 ]]; then
    echo -e "${GREEN}${BOLD}✓ all tests passed${RESET}  (${ELAPSED}s)"
else
    echo -e "${RED}${BOLD}✗ tests failed${RESET}  (${ELAPSED}s, exit $STATUS)"
fi

# ── Trace summary ─────────────────────────────────────────────────────────────
TRACES_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/piku/traces"
if [[ -d "$TRACES_DIR" ]]; then
    TRACE_COUNT=$(find "$TRACES_DIR" -name "*.jsonl" -newer "$BINARY" 2>/dev/null | wc -l | tr -d ' ')
    if [[ "$TRACE_COUNT" -gt 0 ]]; then
        echo ""
        echo -e "${DIM}traces written: $TRACE_COUNT new .jsonl file(s) in $TRACES_DIR${RESET}"
        echo -e "${DIM}inspect with:   cat $TRACES_DIR/session-*.jsonl | jq .${RESET}"
    fi
fi

exit $STATUS
