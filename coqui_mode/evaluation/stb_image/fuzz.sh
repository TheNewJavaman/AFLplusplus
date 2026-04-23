#!/usr/bin/env bash
# fuzz.sh — launch coqui mode afl-fuzz --coqui on the stb_image target.
#
# Prereq: run ./build.sh first (produces stb_image_read_fuzzer.{cubin,conf},
# stb_image_read_fuzzer_cpu, seeds/).
#
# Extra args are passed through to afl-fuzz, e.g.:
#   ./fuzz.sh -V 60        # stop after 60 seconds
#   ./fuzz.sh -M main      # run as main sync node
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

AFL_FUZZ="${AFL_FUZZ:-$SCRIPT_DIR/../../../afl-fuzz}"
HARNESS_BASENAME="stb_image_read_fuzzer"

for f in "${HARNESS_BASENAME}.cubin" "${HARNESS_BASENAME}.conf" "${HARNESS_BASENAME}_cpu" seeds; do
  if [[ ! -e "${f}" ]]; then
    echo "ERROR: ${f} not found; run ./build.sh first" >&2
    exit 1
  fi
done

# coqui mode env:
#   AFL_COQUI_CUBIN              — absolute path to the sm_75 cubin
#   AFL_I_DONT_CARE_ABOUT_MISSING_CRASHES=1 — suppress core-dump warning
#   AFL_SKIP_CPUFREQ=1           — don't gate on CPU freq governor
#   AFL_SKIP_BIN_CHECK=1         — don't re-check the instrumented binary
#   AFL_NO_UI=1                  — line-oriented status for logs / tmux-safe
export AFL_COQUI_CUBIN="${SCRIPT_DIR}/${HARNESS_BASENAME}.cubin"
export AFL_I_DONT_CARE_ABOUT_MISSING_CRASHES=1
export AFL_SKIP_CPUFREQ=1
export AFL_SKIP_BIN_CHECK=1
export AFL_NO_UI=1

# Pass a dictionary if build.sh linked one.
DICT_ARG=()
if [[ -e dict/stb_image.dict ]]; then
  DICT_ARG=(-x dict/stb_image.dict)
fi

exec "${AFL_FUZZ}" --coqui gpu0 \
  -i ./seeds \
  -o ./out \
  "${DICT_ARG[@]}" \
  "$@" \
  -- ./"${HARNESS_BASENAME}_cpu"
