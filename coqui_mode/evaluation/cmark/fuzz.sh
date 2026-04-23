#!/usr/bin/env bash
# Launch cuAFL afl-fuzz on the cmark target.  Requires ./build.sh first.
#
# Extra args ($@) are passed through to afl-fuzz (before the '--' separator
# breaks that, append them as '-x dict/markdown.dict' etc.).

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

HARNESS="cmark_fuzzer"
CUBIN="$SCRIPT_DIR/${HARNESS}.cubin"
CPU_BIN="$SCRIPT_DIR/${HARNESS}_cpu"
SEEDS="$SCRIPT_DIR/seeds"
OUT="$SCRIPT_DIR/out"

AFL_FUZZ="/home/gpizarro/cuAFL/afl-fuzz"

# Sanity.
for f in "$CUBIN" "$CPU_BIN" "$SEEDS"; do
  if [ ! -e "$f" ]; then
    echo "ERROR: missing $f -- run ./build.sh first." >&2
    exit 1
  fi
done

mkdir -p "$OUT"

export AFL_COQUI_CUBIN="$CUBIN"
export AFL_I_DONT_CARE_ABOUT_MISSING_CRASHES=1
export AFL_SKIP_CPUFREQ=1
export AFL_SKIP_BIN_CHECK=1
export AFL_NO_UI=1

exec "$AFL_FUZZ" --coqui gpu0 -i "$SEEDS" -o "$OUT" "$@" -- "$CPU_BIN"
