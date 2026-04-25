#!/usr/bin/env bash
# fuzz.sh — prep the coqui mode bzip2 fuzz workspace + print suggested
# launch commands. Does NOT exec afl-fuzz; user picks Main/Secondary/Coqui.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

AFL_FUZZ="${AFL_FUZZ:-$SCRIPT_DIR/../../../afl-fuzz}"
CUBIN="${SCRIPT_DIR}/bzip2_fuzzer.cubin"
CPU_BIN="${SCRIPT_DIR}/bzip2_fuzzer_cpu"
SEEDS="${SCRIPT_DIR}/seeds"
OUTDIR="${SCRIPT_DIR}/out"
AFL_DEVICE="${AFL_COQUI_DEVICE:-0}"

# Pre-flight
for f in "$CUBIN" "$CPU_BIN" "$SEEDS"; do
  if [ ! -e "$f" ]; then
    echo "ERROR: missing $f -- run ./build.sh first." >&2
    exit 1
  fi
done

# Output dir
if [ "${AFL_RESUME:-0}" = "1" ] && [ -d "$OUTDIR" ]; then
  echo "[fuzz] will reuse existing $OUTDIR (AFL_RESUME=1)"
else
  rm -rf "$OUTDIR"
  mkdir -p "$OUTDIR"
fi

echo "==============================================================="
echo "  coqui mode bzip2 fuzz workspace ready"
echo "  cubin:  $CUBIN"
echo "  cpu:    $CPU_BIN"
echo "  seeds:  $SEEDS  ($(ls "$SEEDS" | wc -l) file(s))"
echo "  out:    $OUTDIR"
echo "==============================================================="

cat <<INNEREOF

# 1. Set environment once (or prefix each command):
export AFL_I_DONT_CARE_ABOUT_MISSING_CRASHES=1 \\
       AFL_SKIP_CPUFREQ=1 \\
       AFL_NO_UI=1 \\
       AFL_COQUI_CUBIN=$CUBIN \\
       AFL_COQUI_DEVICE=$AFL_DEVICE

# 2. Pick ONE instance to launch:

# Main (CPU master):
$AFL_FUZZ -M main -i $SEEDS -o $OUTDIR -- $CPU_BIN

# Secondary (CPU parallel fuzzer; repeat with sec2/sec3/... for more):
$AFL_FUZZ -S sec1 -i $SEEDS -o $OUTDIR -- $CPU_BIN

# Coqui (GPU-backed fuzzer, device $AFL_DEVICE):
$AFL_FUZZ --coqui gpu$AFL_DEVICE -i $SEEDS -o $OUTDIR -- $CPU_BIN

INNEREOF
