#!/usr/bin/env bash
# Launch coqui mode afl-fuzz against the cjson target on GPU device 0.
#
# Prerequisites: ./build.sh has produced cjson_fuzzer.cubin, cjson_fuzzer.conf,
# cjson_fuzzer_cpu, and the seeds/ symlink. Output lands in ./out.
#
# Any extra arguments are forwarded to afl-fuzz (after the target), e.g.
#   ./fuzz.sh -V 300        # 5-minute bench
#   ./fuzz.sh -M main       # explicit fuzzer id

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

AFL_FUZZ="${AFL_FUZZ:-$HERE/../../../afl-fuzz}"
CUBIN="$HERE/cjson_fuzzer.cubin"
CONF="$HERE/cjson_fuzzer.conf"
CPU_BIN="$HERE/cjson_fuzzer_cpu"
SEEDS="$HERE/seeds"
OUTDIR="$HERE/out"

# --- Pre-flight ---------------------------------------------------------
test -x "$AFL_FUZZ" || { echo "[fuzz] missing $AFL_FUZZ — build coqui mode first" >&2; exit 1; }
test -f "$CUBIN"    || { echo "[fuzz] missing $CUBIN — run ./build.sh"     >&2; exit 1; }
test -f "$CONF"     || { echo "[fuzz] missing $CONF — run ./build.sh"      >&2; exit 1; }
test -x "$CPU_BIN"  || { echo "[fuzz] missing $CPU_BIN — run ./build.sh"   >&2; exit 1; }
test -d "$SEEDS"    || { echo "[fuzz] missing $SEEDS — run ./build.sh"     >&2; exit 1; }

# --- AFL env-var setup --------------------------------------------------
# Cubin path (the afl-fuzz driver also looks for a companion file by base
# name, but setting this is explicit and survives cwd changes).
export AFL_COQUI_CUBIN="$CUBIN"

# GPU 0 is the RTX Titan (sm_75). --coqui <gpu0> encodes it too; this
# is the env-var fallback that the runtime also consults.
export AFL_COQUI_DEVICE=${AFL_COQUI_DEVICE:-0}

# Pre-flight bypasses (per coqui mode conventions — host core_pattern + cpufreq
# checks are warnings, not bugs).
export AFL_I_DONT_CARE_ABOUT_MISSING_CRASHES=1
export AFL_SKIP_CPUFREQ=1
export AFL_SKIP_BIN_CHECK=1
export AFL_NO_UI=1

# --- Output dir ---------------------------------------------------------
# Reuse existing out/ when AFL_RESUME=1 is set; otherwise start fresh.
if [ "${AFL_RESUME:-0}" = "1" ] && [ -d "$OUTDIR" ]; then
  echo "[fuzz] resuming existing $OUTDIR"
else
  rm -rf "$OUTDIR"
fi

# --- Dict (optional) ----------------------------------------------------
# fuzz.sh honors any json.dict symlinked into ./dict (see build.sh).
DICT_ARG=()
if [ -d "$HERE/dict" ]; then
  DICT_FILE=$(ls "$HERE/dict"/*.dict 2>/dev/null | head -n1 || true)
  if [ -n "$DICT_FILE" ]; then
    DICT_ARG=(-x "$DICT_FILE")
  fi
fi

echo "==============================================================="
echo "  coqui mode cjson fuzz launch"
echo "  GPU:    device $AFL_COQUI_DEVICE (expect RTX Titan sm_75)"
echo "  cubin:  $CUBIN"
echo "  cpu:    $CPU_BIN"
echo "  seeds:  $SEEDS  ($(ls "$SEEDS" | wc -l) files)"
echo "  out:    $OUTDIR"
[ ${#DICT_ARG[@]} -gt 0 ] && echo "  dict:   ${DICT_ARG[1]}"
echo "==============================================================="

exec "$AFL_FUZZ" --coqui gpu0 \
     -i "$SEEDS" \
     -o "$OUTDIR" \
     "${DICT_ARG[@]}" \
     "$@" \
     -- "$CPU_BIN"
