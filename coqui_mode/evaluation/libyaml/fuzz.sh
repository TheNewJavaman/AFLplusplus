#!/usr/bin/env bash
# fuzz.sh — launch coqui mode afl-fuzz --coqui on the libyaml target.
#
# Prereq: run ./build.sh first (produces libyaml_parser_fuzzer.{cubin,conf},
#         libyaml_parser_fuzzer_cpu, seeds/).
#
# Extra args are passed through to afl-fuzz, e.g.:
#   ./fuzz.sh -V 60        # stop after 60 seconds
#   ./fuzz.sh -M main      # run as main sync node
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

AFL_FUZZ="/home/gpizarro/cuAFL/afl-fuzz"

for f in libyaml_parser_fuzzer.cubin libyaml_parser_fuzzer.conf libyaml_parser_fuzzer_cpu seeds; do
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
export AFL_COQUI_CUBIN="${SCRIPT_DIR}/libyaml_parser_fuzzer.cubin"
export AFL_I_DONT_CARE_ABOUT_MISSING_CRASHES=1
export AFL_SKIP_CPUFREQ=1
export AFL_SKIP_BIN_CHECK=1
export AFL_NO_UI=1

exec "${AFL_FUZZ}" --coqui gpu0 \
  -i ./seeds \
  -o ./out \
  "$@" \
  -- ./libyaml_parser_fuzzer_cpu
