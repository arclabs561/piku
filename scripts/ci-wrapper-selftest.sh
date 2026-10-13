#!/usr/bin/env bash
# Exercise the real CI entry point with a hermetic Cargo boundary probe.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

cat >"$tmp/cargo" <<'CARGO'
#!/usr/bin/env bash
set -euo pipefail
if [[ "${RUSTC_WRAPPER+x}" != "$EXPECTED_SET" || "${RUSTC_WRAPPER-}" != "$EXPECTED_WRAPPER" ]]; then
  printf 'CI changed the caller-selected compiler wrapper\n' >&2
  exit 97
fi
printf '%s\n' "$1" > "$PROBE_CALL"
exit "${PROBE_EXIT:-0}"
CARGO
chmod +x "$tmp/cargo"

for sandbox in '' seatbelt; do
  for state in configured empty unset; do
    for stage in fmt clippy test build; do
      wrapper_args=(RUSTC_WRAPPER= EXPECTED_SET=x EXPECTED_WRAPPER=)
      case "$state" in
        configured) wrapper_args=(RUSTC_WRAPPER=/fixture/compiler-wrapper EXPECTED_SET=x EXPECTED_WRAPPER=/fixture/compiler-wrapper) ;;
        unset) wrapper_args=(-u RUSTC_WRAPPER EXPECTED_SET= EXPECTED_WRAPPER=) ;;
      esac
      env "${wrapper_args[@]}" CODEX_SANDBOX="$sandbox" PATH="$tmp:$PATH" \
        PROBE_CALL="$tmp/call" bash "$SCRIPT_DIR/ci.sh" "$stage"
      [[ "$(cat "$tmp/call")" == "$stage" ]]
    done
  done
done

# A compiler/Cargo failure must stay visible, rather than retry without a wrapper.
result=0
env RUSTC_WRAPPER=/fixture/compiler-wrapper EXPECTED_SET=x \
  EXPECTED_WRAPPER=/fixture/compiler-wrapper CODEX_SANDBOX=seatbelt \
  PATH="$tmp:$PATH" PROBE_CALL="$tmp/call" PROBE_EXIT=73 \
  bash "$SCRIPT_DIR/ci.sh" build || result=$?
[[ "$result" == 73 ]]
printf 'ci wrapper selftest: 24 routing cases and failure propagation passed\n'
