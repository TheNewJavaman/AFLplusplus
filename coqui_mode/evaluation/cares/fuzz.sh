#!/usr/bin/env bash
#
# Launch coqui mode (coqui_mode) on the c-ares DNS parser fuzz target.
#
# Pass-through: any args to this script are forwarded to afl-fuzz, so you can
# append the usual flags, e.g.:
#
#   ./fuzz.sh               # fuzz forever
#   ./fuzz.sh -V 120        # stop after 120s (useful for benchmarks)
#   ./fuzz.sh -x ./dict/dns.dict
#
# Requires: ./build.sh has been run successfully.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

HARNESS="cares_parse_reply_fuzzer"
CUBIN="$HERE/$HARNESS.cubin"
CPU_BIN="$HERE/${HARNESS}_cpu"
SEEDS="$HERE/seeds"
OUT="$HERE/out"

AFL_FUZZ="${AFL_FUZZ:-/home/gpizarro/cuAFL/afl-fuzz}"

# --- Sanity checks -----------------------------------------------------------

[ -x "$AFL_FUZZ" ]   || { echo "error: $AFL_FUZZ not found or not executable" >&2; exit 1; }
[ -f "$CUBIN" ]      || { echo "error: missing $CUBIN (run ./build.sh first)" >&2; exit 1; }
[ -x "$CPU_BIN" ]    || { echo "error: missing $CPU_BIN (run ./build.sh first)" >&2; exit 1; }
[ -d "$SEEDS" ]      || { echo "error: missing $SEEDS (run ./build.sh first)" >&2; exit 1; }

# --- AFL++ env vars ----------------------------------------------------------

export AFL_COQUI_CUBIN="$CUBIN"
export AFL_I_DONT_CARE_ABOUT_MISSING_CRASHES=1
export AFL_SKIP_CPUFREQ=1
export AFL_SKIP_BIN_CHECK=1
export AFL_NO_UI="${AFL_NO_UI:-1}"

# Auto-select a dict only if the user didn't already pass -x.
extra_args=()
for a in "$@"; do
  if [ "$a" = "-x" ]; then
    has_x=1
    break
  fi
done
if [ -z "${has_x:-}" ] && [ -f "$HERE/dict/dns.dict" ]; then
  extra_args+=(-x "$HERE/dict/dns.dict")
fi

echo "[cares/fuzz] AFL_COQUI_CUBIN=$AFL_COQUI_CUBIN"
echo "[cares/fuzz] launching: $AFL_FUZZ --coqui gpu0 -i $SEEDS -o $OUT ${extra_args[*]:-} $* -- $CPU_BIN"
echo ""

exec "$AFL_FUZZ" \
  --coqui gpu0 \
  -i "$SEEDS" \
  -o "$OUT" \
  "${extra_args[@]}" \
  "$@" \
  -- "$CPU_BIN"
