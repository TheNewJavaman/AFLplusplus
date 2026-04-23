#!/usr/bin/env bash
# Build the coqui mode cjson fuzz target.
#
# Produces in this directory:
#   cjson_fuzzer.cubin      — GPU kernel (compiled by coqui-cc from the
#                              cJSON library sources + harness)
#   cjson_fuzzer.conf       — emitted automatically alongside the cubin
#                              (records stack_size / slab_pool_size / arch)
#   cjson_fuzzer_cpu        — AFL++-instrumented host binary, built via the
#                              coqui nix flake (symlinked from the nix out)
#   seeds/                  — symlink to the upstream cJSON fuzzing/inputs
#                              corpus that the aflplusplus target ships
#   dict/                   — symlink to the upstream cJSON dictionary
#
# Mirrors /home/gpizarro/coqui/nix/targets/cjson.nix exactly for the GPU
# sources, -I, -D and --stack-size. The CPU binary is supplied by the
# `target-cjson-aflplusplus` package from the coqui flake.
#
# Run from this directory:
#   ./build.sh
#
# Reruns are idempotent: existing nix out-links are reused and the cubin
# is regenerated every time (coqui-cc is fast and has no stale-cache mode).

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

# --- Fixed paths --------------------------------------------------------
COQUI_REPO=/home/gpizarro/coqui
COQUI_CC=/usr/local/bin/coqui-cc
ARCH=sm_75
STACK_SIZE=32768
SLAB_POOL_SIZE=0          # cjson.nix does not set a slab pool

CPU_OUT_LINK=/tmp/coqui-cjson-cpu     # nix result link (CPU aflplusplus build)
CPU_BINARY_NAME=cjson_read_fuzzer     # from cpu-target-specs.nix (harnessName)

# --- Pre-flight ---------------------------------------------------------
test -x "$COQUI_CC"          || { echo "[build] missing $COQUI_CC" >&2; exit 1; }
test -d "$COQUI_REPO"        || { echo "[build] missing $COQUI_REPO" >&2; exit 1; }
test -f "$COQUI_REPO/flake.nix" || { echo "[build] $COQUI_REPO is not a flake" >&2; exit 1; }

# --- Step 1: build the CPU (AFL++) binary via nix ------------------------
# target-cjson-aflplusplus ships: harness binary + seeds/ + dict/.
echo "[build] nix build target-cjson-aflplusplus -> $CPU_OUT_LINK"
( cd "$COQUI_REPO" && nix build '.#target-cjson-aflplusplus' \
    --out-link "$CPU_OUT_LINK" )

test -x "$CPU_OUT_LINK/$CPU_BINARY_NAME" \
  || { echo "[build] nix result missing $CPU_BINARY_NAME" >&2; exit 1; }

# Symlink the CPU binary next to our scripts (rename -> cjson_fuzzer_cpu so
# build.sh / fuzz.sh have a single stable name).
ln -sfn "$CPU_OUT_LINK/$CPU_BINARY_NAME" "$HERE/cjson_fuzzer_cpu"

# Symlink seeds/ and dict/ (they are already read-only nix store symlinks
# inside the out path — point at the top-level dirs so afl-fuzz -i works).
ln -sfn "$CPU_OUT_LINK/seeds" "$HERE/seeds"
if [ -d "$CPU_OUT_LINK/dict" ]; then
  ln -sfn "$CPU_OUT_LINK/dict" "$HERE/dict"
fi

# --- Step 2: resolve cJSON library sources --------------------------------
# The aflplusplus build references the cJSON nix-store source; grab the
# directory (v1.7.18 — same fetchFromGitHub hash as cjson.nix) so coqui-cc
# compiles against the exact same sources.
CJSON_SRC=$(nix-store -qR "$CPU_OUT_LINK" \
  | grep -E '/nix/store/[^/]+-source$' \
  | xargs -I{} sh -c '[ -f "{}/cJSON.c" ] && echo "{}"' \
  | head -n1)
test -n "$CJSON_SRC" \
  || { echo "[build] unable to locate cJSON source in nix closure" >&2; exit 1; }
test -f "$CJSON_SRC/cJSON.c" \
  || { echo "[build] $CJSON_SRC missing cJSON.c" >&2; exit 1; }
test -f "$CJSON_SRC/cJSON_Utils.c" \
  || { echo "[build] $CJSON_SRC missing cJSON_Utils.c" >&2; exit 1; }
echo "[build] cJSON sources: $CJSON_SRC"

# --- Step 3: resolve the harness source ---------------------------------
HARNESS="$COQUI_REPO/harness/targets/cjson_read_fuzzer.c"
test -f "$HARNESS" \
  || { echo "[build] missing harness: $HARNESS" >&2; exit 1; }

# --- Step 4: compile the GPU cubin via coqui-cc --------------------------
# Flags mirror nix/targets/cjson.nix: -arch sm_75, --stack-size 32768,
# -I <cjson src>, -D CJSON_HIDE_SYMBOLS, library + harness sources.
# NOTE: coqui-cc does not accept --heap-size or --batch-size.
echo "[build] coqui-cc -> cjson_fuzzer.cubin + .conf"
"$COQUI_CC" \
  -arch "$ARCH" \
  --stack-size "$STACK_SIZE" \
  --slab-pool-size "$SLAB_POOL_SIZE" \
  -I "$CJSON_SRC" \
  -D "CJSON_HIDE_SYMBOLS" \
  "$CJSON_SRC/cJSON.c" \
  "$CJSON_SRC/cJSON_Utils.c" \
  "$HARNESS" \
  -o "$HERE/cjson_fuzzer"

# --- Summary ------------------------------------------------------------
echo
echo "=== coqui mode cjson build complete ==="
ls -la "$HERE/cjson_fuzzer.cubin" "$HERE/cjson_fuzzer.conf" \
       "$HERE/cjson_fuzzer_cpu" "$HERE/seeds" 2>/dev/null || true
[ -L "$HERE/dict" ] && ls -la "$HERE/dict" || true
echo "  conf:"; sed 's/^/    /' "$HERE/cjson_fuzzer.conf"
