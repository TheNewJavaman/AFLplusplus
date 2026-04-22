#!/usr/bin/env bash
# benchmark/gpu_havoc_vs_cpu.sh --- interleaved CPU-havoc vs GPU-havoc bench.
#
# Environment:
#   REPO     -- repo root (default: /home/gpizarro/cuAFL)
#   SEEDS    -- seed corpus dir (default: /tmp/cjson-seeds-at; regular files,
#               not nix-store symlinks, else afl-fuzz rejects)
#   DUR      -- per-run duration in seconds (default: 120)
#   N        -- iteration count (default: 20)
#   OUT_ROOT -- output root (default: /tmp/gpu-havoc-bench)
#
# Writes CSV (run, mode, execs_per_sec, edges_found, saved_crashes,
# run_wall_s) to stdout.  If no CPU-instrumented cjson target is present at
# $REPO/cjson_fuzzer_cpu, reports 0s for CPU rows.
set -euo pipefail

REPO="${REPO:-/home/gpizarro/cuAFL}"
SEEDS="${SEEDS:-/tmp/cjson-seeds-at}"
DUR="${DUR:-120}"
N="${N:-20}"
OUT_ROOT="${OUT_ROOT:-/tmp/gpu-havoc-bench}"
mkdir -p "$OUT_ROOT"

CPU_TARGET="$REPO/cjson_fuzzer_cpu"
GPU_TARGET="$REPO/cjson_fuzzer_cpu"          # host ELF used for CPU fsrv under --coqui
GPU_CUBIN="$REPO/cjson_fuzzer.sm_75.cubin"

echo "run,mode,execs_per_sec,edges_found,saved_crashes,run_wall_s"
for ((i = 0; i < N; ++i)); do
  for mode in gpu cpu; do
    OUT="$OUT_ROOT/run-$i-$mode"
    rm -rf "$OUT"
    mkdir -p "$OUT"

    case "$mode" in
      cpu)
        if [ ! -x "$CPU_TARGET" ]; then
          echo "$i,cpu,0,0,0,0"
          continue
        fi
        T0=$(date +%s)
        AFL_I_DONT_CARE_ABOUT_MISSING_CRASHES=1 AFL_SKIP_CPUFREQ=1 \
        AFL_SKIP_BIN_CHECK=1 \
          timeout "$DUR" "$REPO/afl-fuzz" -G 4096 -i "$SEEDS" -o "$OUT" \
          -- "$CPU_TARGET" > "$OUT/stdout.log" 2>&1 || true
        T1=$(date +%s)
        ;;
      gpu)
        T0=$(date +%s)
        AFL_I_DONT_CARE_ABOUT_MISSING_CRASHES=1 AFL_SKIP_CPUFREQ=1 \
        AFL_SKIP_BIN_CHECK=1 AFL_COQUI_CUBIN="$GPU_CUBIN" \
          timeout "$DUR" "$REPO/afl-fuzz" --coqui gpu0 -G 4096 \
          -i "$SEEDS" -o "$OUT" -- "$GPU_TARGET" \
          > "$OUT/stdout.log" 2>&1 || true
        T1=$(date +%s)
        ;;
    esac

    # --coqui creates sync-id subdir "gpu0"; CPU-only creates "default".
    stats="$OUT/default/fuzzer_stats"
    [ -f "$OUT/gpu0/fuzzer_stats" ] && stats="$OUT/gpu0/fuzzer_stats"

    eps=0; edg=0; crs=0
    if [ -f "$stats" ]; then
      eps=$(awk -F': *' '/^execs_per_sec/ {print $2; exit}' "$stats")
      edg=$(awk -F': *' '/^edges_found/   {print $2; exit}' "$stats")
      crs=$(awk -F': *' '/^saved_crashes/ {print $2; exit}' "$stats")
    fi
    echo "$i,$mode,$eps,$edg,$crs,$((T1-T0))"
  done
done
