#!/usr/bin/env bash
#
# Single source of truth for the check gate.
#
# CI (.github/workflows/ci.yml) runs each stage as its own step so failures are
# easy to locate in the UI; `just check` (or `scripts/ci.sh all`) runs them all
# locally. Either way the underlying commands are defined HERE and only here, so
# local-green and CI-green can't drift apart.
#
# Usage: scripts/ci.sh {fmt|scripts|clippy|test|pty|web-bundle|web-harness|web|build|live|live-random|all}   (default: all)
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

scripts() {
  local script
  for script in scripts/*.sh; do
    bash -n "$script"
  done
  ./scripts/github-corpus-prompt-selftest.sh
  ./scripts/github-corpus-run-selftest.sh
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
#
# They pass in ~15s on an idle machine and starve indefinitely on a busy one,
# so the stage is bounded. Without a bound `just check` hangs rather than
# failing, which is worse than a red gate: the run produces no verdict at all
# and the person waiting cannot tell a stall from slow tests. On timeout the
# message says the machine was busy, because that is what a timeout here means
# and reading it as a piku failure sends someone debugging the wrong thing.
pty() {
  local seconds="${PIKU_PTY_TIMEOUT_SECS:-240}"
  local runner=""
  if command -v timeout >/dev/null 2>&1; then
    runner="timeout"
  elif command -v gtimeout >/dev/null 2>&1; then
    runner="gtimeout"
  fi

  if [[ -z "$runner" ]]; then
    cargo test --test tui_smoke -p piku -- --ignored --test-threads=1
    return
  fi

  local status=0
  # --foreground so the tests keep the terminal they need for a PTY.
  "$runner" --foreground "$seconds" cargo test --test tui_smoke -p piku -- --ignored --test-threads=1 || status=$?
  if (( status == 124 )); then
    printf 'pty: no result after %ss. These tests need an idle machine; rerun scripts/ci.sh pty alone, or raise PIKU_PTY_TIMEOUT_SECS.\n' "$seconds" >&2
  fi
  return "$status"
}

web_bundle() (
  local web_root="$REPO_ROOT/crates/piku/web-ui"
  local generated_root="$REPO_ROOT/crates/piku/src/web"
  local scratch

  if [[ ! -d "$web_root/node_modules" ]]; then
    printf 'web: dependencies are missing; run npm ci in crates/piku/web-ui first\n' >&2
    exit 1
  fi

  scratch="$(mktemp -d "${TMPDIR:-/tmp}/piku-web-check.XXXXXX")"
  trap 'rm -rf "$scratch"' EXIT

  (
    cd "$web_root"
    ./node_modules/.bin/esbuild app.js --bundle --format=iife \
      --platform=browser --target=safari17 --outfile="$scratch/app.js" \
      --loader:.css=css
  )

  local drift=0
  local asset
  for asset in app.js app.css; do
    if ! cmp -s "$scratch/$asset" "$generated_root/$asset"; then
      printf 'web: crates/piku/src/web/%s is stale; run npm run build in crates/piku/web-ui\n' "$asset" >&2
      drift=1
    fi
  done
  (( drift == 0 )) || exit 1
)

web_harness() {
  cd "$REPO_ROOT/crates/piku/web-ui"
  npm run test:harness
}

web() {
  web_bundle
  web_harness
}

build() {
  cargo build --release -p piku
}

default_live_ledger() {
  if [[ -n "${PIKU_LIVE_LEDGER:-}" ]]; then
    return
  fi

  local suite="${PIKU_LIVE_SUITE:-llm_e2e}"
  local provider="${PIKU_LIVE_PROVIDER:-auto}"
  local model="${PIKU_LIVE_MODEL:-auto}"
  local stamp safe
  stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  safe="${suite}-${provider}-${model}-${stamp}"
  safe="${safe//\//_}"
  safe="${safe//:/_}"
  safe="${safe// /_}"

  mkdir -p "$REPO_ROOT/target/live-ledger"
  export PIKU_LIVE_LEDGER="$REPO_ROOT/target/live-ledger/${safe}.jsonl"
}

live() {
  local suite="${PIKU_LIVE_SUITE:-llm_e2e}"
  default_live_ledger
  printf 'live-ledger: %s\n' "$PIKU_LIVE_LEDGER"
  cargo test -p piku --test "$suite" -- --ignored --nocapture --test-threads=1
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
  scripts) scripts ;;
  clippy) clippy ;;
  test) test_ ;;
  pty) pty ;;
  web-bundle) web_bundle ;;
  web-harness) web_harness ;;
  web) web ;;
  build) build ;;
  live) live ;;
  live-random) live_random ;;
  all)
    fmt
    scripts
    clippy
    test_
    pty
    web_bundle
    web_harness
    build
    echo "all checks passed"
    ;;
  *)
    echo "usage: $0 {fmt|scripts|clippy|test|pty|web-bundle|web-harness|web|build|live|live-random|all}" >&2
    exit 2
    ;;
esac
