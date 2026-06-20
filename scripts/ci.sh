#!/usr/bin/env bash
#
# Single source of truth for the check gate.
#
# CI (.github/workflows/ci.yml) runs each stage as its own step so failures are
# easy to locate in the UI; `just check` (or `scripts/ci.sh all`) runs them all
# locally. Either way the underlying commands are defined HERE and only here, so
# local-green and CI-green can't drift apart.
#
# Usage: scripts/ci.sh {fmt|clippy|test|pty|build|live|live-random|all}   (default: all)
set -euo pipefail

# Resolve repo root from this script's location so it works from any cwd.
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

if [[ -n "${CODEX_SANDBOX:-}" ]]; then
  # sccache can fail inside the Codex seatbelt while direct rustc still works.
  export RUSTC_WRAPPER=
fi

fmt() {
  cargo fmt --all -- --check
}

# `-D warnings` makes clippy a real gate, not advisory. The workspace opts into
# clippy::pedantic (Cargo.toml), so this enforces pedantic-clean production code;
# test-only pedantic noise is allowed at the relevant test scope, not here.
clippy() {
  cargo clippy --workspace --all-targets -- -D warnings
}

# Default `cargo test` deliberately does NOT run the `#[ignore]`d suites
# (llm_e2e, dogfood, agentic_user personas) — those need a live provider and are
# opt-in via `--ignored`. They report as "ignored", never as a silent pass.
test_() {
  cargo test --workspace
}

# PTY smoke tests drive the real piku binary over a pseudo-terminal. They are
# `#[ignore]`d so the main `test` stage stays fast and deterministic, then run
# HERE in isolation: alone (no other test binaries competing) their teardown is
# fast and they pass in ~15s, whereas under full-workspace concurrency they
# starve and stall. `#[serial]` keeps them one-at-a-time within the binary.
pty() {
  cargo test --test tui_smoke -p piku -- --ignored
}

build() {
  cargo build --release -p piku
}

live() {
  local suite="${PIKU_LIVE_SUITE:-llm_e2e}"
  cargo test -p piku --test "$suite" -- --ignored --nocapture
}

live_random() {
  local suite="${PIKU_LIVE_SUITE:-llm_e2e}"
  local rows=(
    "openrouter|openai/gpt-4o-mini|OPENROUTER_API_KEY"
    "openrouter|anthropic/claude-sonnet-4-5|OPENROUTER_API_KEY"
    "openrouter|google/gemini-2.5-flash|OPENROUTER_API_KEY"
    "anthropic|claude-haiku-4-5|ANTHROPIC_API_KEY"
    "groq|moonshotai/kimi-k2-instruct|GROQ_API_KEY"
  )
  local available=()
  local provider model key_var row

  for row in "${rows[@]}"; do
    IFS='|' read -r provider model key_var <<<"$row"
    if [[ -n "${!key_var:-}" ]]; then
      available+=("$row")
    fi
  done

  if (( ${#available[@]} == 0 )); then
    printf 'error: set OPENROUTER_API_KEY, ANTHROPIC_API_KEY, or GROQ_API_KEY for live-random\n' >&2
    return 1
  fi

  row="${available[$((RANDOM % ${#available[@]}))]}"
  IFS='|' read -r provider model key_var <<<"$row"
  printf 'live-random: suite=%s provider=%s model=%s\n' "$suite" "$provider" "$model"
  PIKU_LIVE_PROVIDER="$provider" \
    PIKU_LIVE_MODEL="$model" \
    PIKU_LIVE_KEY_VAR="$key_var" \
    PIKU_LIVE_SUITE="$suite" \
    live
}

stage="${1:-all}"
case "$stage" in
  fmt) fmt ;;
  clippy) clippy ;;
  test) test_ ;;
  pty) pty ;;
  build) build ;;
  live) live ;;
  live-random) live_random ;;
  all)
    fmt
    clippy
    test_
    pty
    build
    echo "all checks passed"
    ;;
  *)
    echo "usage: $0 {fmt|clippy|test|pty|build|live|live-random|all}" >&2
    exit 2
    ;;
esac
