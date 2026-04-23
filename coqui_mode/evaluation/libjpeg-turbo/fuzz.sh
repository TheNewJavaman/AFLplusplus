#!/usr/bin/env bash
# fuzz.sh — launch coqui mode afl-fuzz --coqui on the libjpeg-turbo target.
#
# Prereq: run ./build.sh first (produces the cubin + .conf sidecar + _cpu
# binary + seeds/ + dict/).
#
# Extra args are forwarded to afl-fuzz after the target, e.g.:
#   ./fuzz.sh -V 60            # stop after 60 seconds
#   ./fuzz.sh -M main          # run as main sync node
#   AFL_RESUME=1 ./fuzz.sh     # resume an existing out/ dir
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

AFL_FUZZ="/home/gpizarro/cuAFL/afl-fuzz"
HARNESS_NAME="libjpeg_turbo_decompress_fuzzer"
CUBIN="${SCRIPT_DIR}/${HARNESS_NAME}.cubin"
CONF="${SCRIPT_DIR}/${HARNESS_NAME}.conf"
CPU_BIN="${SCRIPT_DIR}/${HARNESS_NAME}_cpu"
SEEDS="${SCRIPT_DIR}/seeds"
OUTDIR="${SCRIPT_DIR}/out"

test -x "${AFL_FUZZ}" || { echo "[fuzz] missing ${AFL_FUZZ} — build coqui mode first" >&2; exit 1; }
test -f "${CUBIN}"    || { echo "[fuzz] missing ${CUBIN} — run ./build.sh"       >&2; exit 1; }
test -f "${CONF}"     || { echo "[fuzz] missing ${CONF} — run ./build.sh"        >&2; exit 1; }
test -x "${CPU_BIN}"  || { echo "[fuzz] missing ${CPU_BIN} — run ./build.sh"     >&2; exit 1; }
test -d "${SEEDS}"    || { echo "[fuzz] missing ${SEEDS} — run ./build.sh"       >&2; exit 1; }

# --- AFL env vars -------------------------------------------------------
# Explicitly point at our cubin (the fuzz driver also infers <elf>.cubin
# from the target path, but setting this is cwd-independent).
export AFL_COQUI_CUBIN="${CUBIN}"

# GPU device index 0 is the RTX Titan (sm_75) per CLAUDE.local.md.
export AFL_COQUI_DEVICE="${AFL_COQUI_DEVICE:-0}"

# Pre-flight bypasses (coqui mode convention for bench/benchmark hosts):
#   MISSING_CRASHES — skip core_pattern warning
#   SKIP_CPUFREQ    — skip cpu governor check
#   SKIP_BIN_CHECK  — we're running an AFL-instrumented binary but afl-fuzz
#                     does GPU-side coverage anyway
#   NO_UI           — line-oriented output for logs / tmux compatibility
export AFL_I_DONT_CARE_ABOUT_MISSING_CRASHES=1
export AFL_SKIP_CPUFREQ=1
export AFL_SKIP_BIN_CHECK=1
export AFL_NO_UI=1

# --- Output dir ---------------------------------------------------------
if [ "${AFL_RESUME:-0}" = "1" ] && [ -d "${OUTDIR}" ]; then
  echo "[fuzz] resuming existing ${OUTDIR}"
else
  rm -rf "${OUTDIR}"
fi

# --- Dict (optional) ----------------------------------------------------
DICT_ARG=()
if [ -d "${SCRIPT_DIR}/dict" ]; then
  DICT_FILE=$(ls "${SCRIPT_DIR}/dict"/*.dict 2>/dev/null | head -n1 || true)
  if [ -n "${DICT_FILE}" ]; then
    DICT_ARG=(-x "${DICT_FILE}")
  fi
fi

echo "==============================================================="
echo "  coqui mode libjpeg-turbo fuzz launch"
echo "  GPU:    device ${AFL_COQUI_DEVICE} (expect RTX Titan sm_75)"
echo "  cubin:  ${CUBIN}"
echo "  cpu:    ${CPU_BIN}"
echo "  seeds:  ${SEEDS}  ($(ls "${SEEDS}" 2>/dev/null | wc -l) files)"
echo "  out:    ${OUTDIR}"
[ ${#DICT_ARG[@]} -gt 0 ] && echo "  dict:   ${DICT_ARG[1]}"
echo "==============================================================="

exec "${AFL_FUZZ}" --coqui gpu0 \
  -i "${SEEDS}" \
  -o "${OUTDIR}" \
  "${DICT_ARG[@]}" \
  "$@" \
  -- "${CPU_BIN}"
