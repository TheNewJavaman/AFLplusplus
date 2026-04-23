#!/usr/bin/env bash
# fuzz.sh — launch cuAFL afl-fuzz --coqui on the libpng target (GPU device 0).
#
# Prerequisite: run ./build.sh first to produce
#   libpng_read_fuzzer.{cubin,conf}, libpng_read_fuzzer_cpu, seeds/, dict/.
#
# Extra args are forwarded to afl-fuzz (after the target), e.g.
#   ./fuzz.sh -V 300            # 5-minute bench
#   ./fuzz.sh -M main           # explicit fuzzer id
#   AFL_RESUME=1 ./fuzz.sh      # resume into existing ./out

set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${HERE}"

AFL_FUZZ="/home/gpizarro/cuAFL/afl-fuzz"
CUBIN="${HERE}/libpng_read_fuzzer.cubin"
CONF="${HERE}/libpng_read_fuzzer.conf"
CPU_BIN="${HERE}/libpng_read_fuzzer_cpu"
SEEDS="${HERE}/seeds"
OUTDIR="${HERE}/out"

# --- Pre-flight ---------------------------------------------------------
test -x "${AFL_FUZZ}" || { echo "[fuzz] missing ${AFL_FUZZ} — build cuAFL first" >&2; exit 1; }
test -f "${CUBIN}"    || { echo "[fuzz] missing ${CUBIN} — run ./build.sh"      >&2; exit 1; }
test -f "${CONF}"     || { echo "[fuzz] missing ${CONF} — run ./build.sh"       >&2; exit 1; }
test -x "${CPU_BIN}"  || { echo "[fuzz] missing ${CPU_BIN} — run ./build.sh"    >&2; exit 1; }
test -d "${SEEDS}"    || { echo "[fuzz] missing ${SEEDS} — run ./build.sh"      >&2; exit 1; }

# --- AFL env-var setup --------------------------------------------------
#   AFL_COQUI_CUBIN  — absolute path to the sm_75 cubin
#   AFL_COQUI_DEVICE — GPU index (matches `--coqui gpu0`)
#   AFL_I_DONT_CARE_ABOUT_MISSING_CRASHES=1 — suppress core-dump warning
#   AFL_SKIP_CPUFREQ=1  — don't gate on CPU freq governor
#   AFL_SKIP_BIN_CHECK=1 — don't re-check the instrumented binary
#   AFL_NO_UI=1         — line-oriented status for logs / tmux-safe
export AFL_COQUI_CUBIN="${CUBIN}"
export AFL_COQUI_DEVICE="${AFL_COQUI_DEVICE:-0}"
export AFL_I_DONT_CARE_ABOUT_MISSING_CRASHES=1
export AFL_SKIP_CPUFREQ=1
export AFL_SKIP_BIN_CHECK=1
export AFL_NO_UI=1

# --- Output dir ---------------------------------------------------------
# Reuse existing out/ when AFL_RESUME=1 is set; otherwise start fresh.
if [[ "${AFL_RESUME:-0}" = "1" && -d "${OUTDIR}" ]]; then
  echo "[fuzz] resuming existing ${OUTDIR}"
else
  rm -rf "${OUTDIR}"
fi

# --- Dict (optional) ----------------------------------------------------
DICT_ARG=()
if [[ -d "${HERE}/dict" ]]; then
  DICT_FILE="$(ls "${HERE}/dict"/*.dict 2>/dev/null | head -n1 || true)"
  if [[ -n "${DICT_FILE}" ]]; then
    DICT_ARG=(-x "${DICT_FILE}")
  fi
fi

echo "==============================================================="
echo "  cuAFL libpng fuzz launch"
echo "  GPU:    device ${AFL_COQUI_DEVICE} (expect RTX Titan sm_75)"
echo "  cubin:  ${CUBIN}"
echo "  cpu:    ${CPU_BIN}"
echo "  seeds:  ${SEEDS}  ($(ls "${SEEDS}" | wc -l) files)"
echo "  out:    ${OUTDIR}"
[[ ${#DICT_ARG[@]} -gt 0 ]] && echo "  dict:   ${DICT_ARG[1]}"
echo "==============================================================="

exec "${AFL_FUZZ}" --coqui gpu0 \
  -i "${SEEDS}" \
  -o "${OUTDIR}" \
  "${DICT_ARG[@]}" \
  "$@" \
  -- "${CPU_BIN}"
