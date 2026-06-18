#!/usr/bin/env bash
#
# Single source of truth for the check gate.
#
# CI (.github/workflows/ci.yml) runs each stage as its own step so failures are
# easy to locate in the UI; `just check` (or `scripts/ci.sh all`) runs them all
# locally. Either way the underlying commands are defined HERE and only here, so
# local-green and CI-green can't drift apart.
#
# Usage: scripts/ci.sh {fmt|clippy|test|pty|build|all}   (default: all)
set -euo pipefail

# Resolve repo root from this script's location so it works from any cwd.
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

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

stage="${1:-all}"
case "$stage" in
  fmt) fmt ;;
  clippy) clippy ;;
  test) test_ ;;
  pty) pty ;;
  build) build ;;
  all)
    fmt
    clippy
    test_
    pty
    build
    echo "all checks passed"
    ;;
  *)
    echo "usage: $0 {fmt|clippy|test|pty|build|all}" >&2
    exit 2
    ;;
esac
