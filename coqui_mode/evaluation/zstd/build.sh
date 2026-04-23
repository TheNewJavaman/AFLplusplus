#!/usr/bin/env bash
# build.sh — build the cuAFL zstd fuzz target.
#
# Produces in this directory:
#   zstd_simple_decompress_fuzzer.cubin — GPU kernel (sm_75), compiled by
#                                         coqui-cc from the zstd decompress
#                                         library sources + harness.
#   zstd_simple_decompress_fuzzer.conf  — coqui-cc companion config
#                                         (records stack_size / slab_pool_size /
#                                         arch).
#   zstd_simple_decompress_fuzzer_cpu   — AFL++-instrumented host binary built
#                                         via the coqui nix flake
#                                         (symlinked from the nix out).
#   seeds/                              — symlink to the upstream zstd fuzz
#                                         seed corpus shipped by the
#                                         aflplusplus target.
#   dict/                               — symlink to the upstream zstd
#                                         dictionary.
#
# Mirrors /home/gpizarro/coqui/nix/targets/zstd.nix exactly for the GPU sources,
# -I, -D. The CPU binary is supplied by the `target-zstd-aflplusplus` package
# from the coqui flake.
#
# Run from this directory:
#   ./build.sh
#
# Reruns are idempotent: existing nix out-links are reused and the cubin is
# regenerated every time (coqui-cc is fast and has no stale-cache mode).
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${HERE}"

# --- Fixed paths --------------------------------------------------------
COQUI_REPO="/home/gpizarro/coqui"
COQUI_CC="/usr/local/bin/coqui-cc"
ARCH="sm_75"
STACK_SIZE=32768        # coqui-cc default (zstd.nix does not override)
SLAB_POOL_SIZE=0        # zstd.nix does not set a slab pool

CPU_OUT_LINK="/tmp/cuafl-zstd-cpu"
CPU_BINARY_NAME="zstd_simple_decompress_fuzzer"  # harnessName in cpu-target-specs.nix
HARNESS_NAME="zstd_simple_decompress_fuzzer"     # == CPU_BINARY_NAME for zstd

# --- Pre-flight ---------------------------------------------------------
test -x "${COQUI_CC}"             || { echo "[build] missing ${COQUI_CC}" >&2; exit 1; }
test -d "${COQUI_REPO}"           || { echo "[build] missing ${COQUI_REPO}" >&2; exit 1; }
test -f "${COQUI_REPO}/flake.nix" || { echo "[build] ${COQUI_REPO} is not a flake" >&2; exit 1; }

# --- Step 1: build the CPU (AFL++) binary via nix -----------------------
# target-zstd-aflplusplus ships: harness binary + seeds/ + dict/.
echo "=== [1/4] nix build target-zstd-aflplusplus -> ${CPU_OUT_LINK} ==="
( cd "${COQUI_REPO}" && nix build '.#target-zstd-aflplusplus' \
    --out-link "${CPU_OUT_LINK}" )

if [[ ! -x "${CPU_OUT_LINK}/${CPU_BINARY_NAME}" ]]; then
  echo "[build] nix result missing ${CPU_BINARY_NAME} at ${CPU_OUT_LINK}" >&2
  exit 1
fi

# --- Step 2: resolve zstd source path -----------------------------------
# The facebook/zstd v1.5.6 tarball is the zstd-flavoured `-source` derivation
# in the closure of the CPU build. Pick the one that contains lib/zstd.h.
echo "=== [2/4] Resolve zstd source path ==="
ZSTD_SRC=$(nix-store -qR "${CPU_OUT_LINK}" \
  | grep -E '/nix/store/[^/]+-source$' \
  | xargs -I{} sh -c '[ -f "{}/lib/zstd.h" ] && echo "{}"' \
  | head -n1)

if [[ -z "${ZSTD_SRC}" || ! -f "${ZSTD_SRC}/lib/zstd.h" ]]; then
  echo "[build] unable to locate zstd source (expected lib/zstd.h in a -source derivation)" >&2
  exit 1
fi
echo "  zstd source: ${ZSTD_SRC}"

# --- Step 3: resolve the harness source ---------------------------------
HARNESS="${COQUI_REPO}/harness/targets/${HARNESS_NAME}.c"
if [[ ! -f "${HARNESS}" ]]; then
  echo "[build] missing harness: ${HARNESS}" >&2
  exit 1
fi

# Local device-side stub for `abort()` — see zstd_abort_stub.c.
ABORT_STUB="${HERE}/zstd_abort_stub.c"
if [[ ! -f "${ABORT_STUB}" ]]; then
  echo "[build] missing abort stub: ${ABORT_STUB}" >&2
  exit 1
fi

# --- Step 4: compile the GPU cubin via coqui-cc -------------------------
# Flags mirror /home/gpizarro/coqui/nix/targets/zstd.nix:
#   -I <zstd src>/lib                   — zstd.h, zstd_errors.h
#   -I <zstd src>/lib/common            — error_private.h, mem.h, etc.
#   -I <zstd src>/lib/decompress        — zstd_decompress_internal.h
#   -D ZSTD_NO_INTRINSICS=1             — disable __builtin prefetches etc.
#   -D ZSTD_DISABLE_ASM=1               — no inline asm on GPU
#   -D XXHASH_NAMESPACE=ZSTD_           — avoid xxhash symbol conflicts
#   -D FUZZING_BUILD_MODE_UNSAFE_FOR_PRODUCTION — upstream fuzz hooks
#   -D ZSTD_NO_TRACE=1                  — trace macros off
#   -D ZSTD_DECODER_INTERNAL_BUFFER=4096 — shrink DCtx to ~34KB so it fits
#                                         in the 58KB usable heap.
# Source list matches the nix spec exactly.
echo "=== [3/4] coqui-cc -> ${HARNESS_NAME}.cubin + .conf ==="
"${COQUI_CC}" \
  -arch "${ARCH}" \
  --stack-size "${STACK_SIZE}" \
  --slab-pool-size "${SLAB_POOL_SIZE}" \
  -I "${ZSTD_SRC}/lib" \
  -I "${ZSTD_SRC}/lib/common" \
  -I "${ZSTD_SRC}/lib/decompress" \
  -D ZSTD_NO_INTRINSICS=1 \
  -D ZSTD_DISABLE_ASM=1 \
  -D XXHASH_NAMESPACE=ZSTD_ \
  -D FUZZING_BUILD_MODE_UNSAFE_FOR_PRODUCTION \
  -D ZSTD_NO_TRACE=1 \
  -D ZSTD_DECODER_INTERNAL_BUFFER=4096 \
  "${ZSTD_SRC}/lib/common/debug.c" \
  "${ZSTD_SRC}/lib/common/entropy_common.c" \
  "${ZSTD_SRC}/lib/common/error_private.c" \
  "${ZSTD_SRC}/lib/common/fse_decompress.c" \
  "${ZSTD_SRC}/lib/common/xxhash.c" \
  "${ZSTD_SRC}/lib/common/zstd_common.c" \
  "${ZSTD_SRC}/lib/decompress/huf_decompress.c" \
  "${ZSTD_SRC}/lib/decompress/zstd_ddict.c" \
  "${ZSTD_SRC}/lib/decompress/zstd_decompress.c" \
  "${ZSTD_SRC}/lib/decompress/zstd_decompress_block.c" \
  "${ABORT_STUB}" \
  "${HARNESS}" \
  -o "${HERE}/${HARNESS_NAME}"

if [[ ! -f "${HARNESS_NAME}.cubin" || ! -f "${HARNESS_NAME}.conf" ]]; then
  echo "[build] coqui-cc did not emit ${HARNESS_NAME}.cubin / .conf" >&2
  exit 1
fi

# --- Step 5: link CPU binary + seeds + dict -----------------------------
echo "=== [4/4] Link CPU binary + seeds + dict ==="
ln -sfn "${CPU_OUT_LINK}/${CPU_BINARY_NAME}" "${HERE}/${HARNESS_NAME}_cpu"
ln -sfn "${CPU_OUT_LINK}/seeds"              "${HERE}/seeds"
if [[ -d "${CPU_OUT_LINK}/dict" ]]; then
  ln -sfn "${CPU_OUT_LINK}/dict"              "${HERE}/dict"
fi

# --- Summary ------------------------------------------------------------
echo
echo "=== cuAFL zstd build complete ==="
ls -la "${HERE}/${HARNESS_NAME}.cubin" "${HERE}/${HARNESS_NAME}.conf" \
       "${HERE}/${HARNESS_NAME}_cpu"   "${HERE}/seeds" 2>/dev/null || true
[[ -L "${HERE}/dict" ]] && ls -la "${HERE}/dict" || true
echo "  conf:"; sed 's/^/    /' "${HERE}/${HARNESS_NAME}.conf"
