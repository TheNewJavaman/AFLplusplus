#!/usr/bin/env bash
# build.sh — build bzip2 coqui mode evaluation target (nix-free).
#
# Produces in the current directory:
#   bzip2_fuzzer.cubin — GPU kernel (configurable via ARCH env, default sm_75)
#   bzip2_fuzzer.conf  — companion config emitted by coqui-cc
#   bzip2_fuzzer_cpu   — AFL++ instrumented CPU binary
#   seeds/min.bz2      — minimal bzip2 stream (generated in-place if missing)
#   .build/            — cached upstream source tree (libarchive/bzip2)
#
# No external dependencies beyond afl-clang-fast (from this repo), coqui-cc
# (/usr/local/bin), git, and bzip2 (for generating the seed). Libbzip2 sources
# are cloned from github.com/libarchive/bzip2 at tag bzip2-1.0.8.
#
# Note: the legacy coqui nix spec passes `--heap-size 524288` and
# `--batch-size 32768` to coqui; coqui mode's coqui-cc does NOT accept those
# flags. The coqui mode runtime derives heap at startup and reads batch size
# from AFL_COQUI_BATCH_SIZE.
#
# Overrides:
#   ARCH          GPU compute capability (default sm_75)
#   AFL_CC        path to afl-clang-fast (default: this repo's own afl-clang-fast)
#   COQUI_CC      path to coqui-cc (default /usr/local/bin/coqui-cc)
#   BZ2_CACHE     path to cache the cloned bzip2 sources (default .build/libarchive-bzip2-<short>)

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# --- Config -----------------------------------------------------------------
HARNESS_BASENAME="bzip2_fuzzer"
HARNESS_SRC="${SCRIPT_DIR}/harness.c"
ASSERT_STUB_SRC="${SCRIPT_DIR}/bz2_assert_stub.c"
ARCH="${ARCH:-sm_75}"
STACK_SIZE=32768            # coqui-cc default (bzip2 target does not override)
SLAB_POOL_SIZE=2147483648   # 2 GiB — bzip2 DState + allocations exceed 64KB heap

# libarchive/bzip2 upstream pin (tag bzip2-1.0.8)
BZ2_COMMIT="6a8690fc8d26c815e798c588f796eabe9d684cf0"
BZ2_SHORT="${BZ2_COMMIT:0:8}"
BZ2_URL="https://github.com/libarchive/bzip2.git"

# Tools
AFL_CC="${AFL_CC:-${SCRIPT_DIR}/../../../afl-clang-fast}"
COQUI_CC="${COQUI_CC:-/usr/local/bin/coqui-cc}"
BZ2_CACHE="${BZ2_CACHE:-${SCRIPT_DIR}/.build/libarchive-bzip2-${BZ2_SHORT}}"

# Sanitizers matching legacy coqui `sanitizers.default` (address + full UBSan).
SANITIZE_FLAGS=(
  "-fsanitize=address,array-bounds,bool,builtin,enum,integer-divide-by-zero,null,object-size,return,returns-nonnull-attribute,shift,signed-integer-overflow,unsigned-integer-overflow,unreachable,vla-bound"
  "-fno-sanitize-recover=array-bounds,bool,builtin,enum,integer-divide-by-zero,null,object-size,return,returns-nonnull-attribute,shift,signed-integer-overflow,unreachable,vla-bound"
  "-fno-stack-protector"
)

BZ2_SOURCES=(
  "${BZ2_CACHE}/blocksort.c"
  "${BZ2_CACHE}/huffman.c"
  "${BZ2_CACHE}/crctable.c"
  "${BZ2_CACHE}/randtable.c"
  "${BZ2_CACHE}/compress.c"
  "${BZ2_CACHE}/decompress.c"
  "${BZ2_CACHE}/bzlib.c"
)

# --- Pre-flight -------------------------------------------------------------
[[ -x "$AFL_CC" ]]   || { echo "ERROR: afl-clang-fast not found at $AFL_CC" >&2; exit 1; }
[[ -x "$COQUI_CC" ]] || { echo "ERROR: coqui-cc not found at $COQUI_CC" >&2; exit 1; }
[[ -f "$HARNESS_SRC" ]] || { echo "ERROR: harness missing at $HARNESS_SRC" >&2; exit 1; }
[[ -f "$ASSERT_STUB_SRC" ]] || { echo "ERROR: bz2_assert_stub.c missing at $ASSERT_STUB_SRC" >&2; exit 1; }
command -v git >/dev/null 2>&1 || { echo "ERROR: git not found in PATH" >&2; exit 1; }
command -v bzip2 >/dev/null 2>&1 || { echo "ERROR: bzip2 not found in PATH (needed for seed)" >&2; exit 1; }

# --- [1/4] Fetch upstream bzip2 sources -------------------------------------
echo "=== [1/4] Fetch libarchive/bzip2 at pinned commit ==="
mkdir -p "$(dirname "$BZ2_CACHE")"
if [[ ! -f "${BZ2_CACHE}/bzlib.h" ]]; then
  echo "  git clone $BZ2_URL -> $BZ2_CACHE"
  rm -rf "$BZ2_CACHE"
  git clone --quiet "$BZ2_URL" "$BZ2_CACHE"
  git -C "$BZ2_CACHE" checkout --quiet "$BZ2_COMMIT"
fi
# Integrity check: required .c files must all exist.
for src in "${BZ2_SOURCES[@]}" "${BZ2_CACHE}/bzlib.h"; do
  [[ -f "$src" ]] || { echo "ERROR: missing $src after fetch" >&2; exit 1; }
done
echo "  bzip2 sources: $(wc -c <"${BZ2_CACHE}/bzlib.c") bytes in bzlib.c"

# --- [2/4] Generate minimal seed --------------------------------------------
echo "=== [2/4] Generate minimal bzip2 seed ==="
mkdir -p "${SCRIPT_DIR}/seeds"
if [[ ! -s "${SCRIPT_DIR}/seeds/min.bz2" ]]; then
  printf 'a' | bzip2 -9 >"${SCRIPT_DIR}/seeds/min.bz2"
fi
echo "  seeds/min.bz2: $(wc -c <"${SCRIPT_DIR}/seeds/min.bz2") bytes"

# --- [3/4] Build AFL++ CPU binary -------------------------------------------
echo "=== [3/4] Build AFL++ CPU binary with afl-clang-fast ==="
CPU_OUT="${HARNESS_BASENAME}_cpu"
"$AFL_CC" -O2 -g \
  "${SANITIZE_FLAGS[@]}" \
  -fsanitize=fuzzer \
  -D BZ_NO_STDIO \
  -I "$BZ2_CACHE" \
  "${BZ2_SOURCES[@]}" \
  "$ASSERT_STUB_SRC" \
  "$HARNESS_SRC" \
  -o "$CPU_OUT" \
  -lm

[[ -x "$CPU_OUT" ]] || { echo "ERROR: CPU binary not produced" >&2; exit 1; }

# --- [4/4] Build GPU cubin via coqui-cc -------------------------------------
echo "=== [4/4] Build GPU cubin via coqui-cc (flock-serialized) ==="
# Flags mirror the legacy bzip2 target:
#   -D BZ_NO_STDIO           — disable bzlib's stdio reliance
#   -I <libbzip2 source>     — bzlib.h + bzlib_private.h
#   --slab-pool-size 2 GiB   — bzip2 DState + allocations exceed 64KB heap
# Source list includes the in-tree bz2_assert_stub.c that routes
# bz_internal_error() to __coqui_trap() instead of libc abort()
# (ExternalSymbolGatekeeper rejects `abort` — only __coqui_*/__llvm_* pass).
flock /tmp/coqui-cc.lock "${COQUI_CC}" \
  -arch "${ARCH}" \
  --stack-size "${STACK_SIZE}" \
  --slab-pool-size "${SLAB_POOL_SIZE}" \
  -D BZ_NO_STDIO \
  -I "$BZ2_CACHE" \
  "${BZ2_SOURCES[@]}" \
  "$ASSERT_STUB_SRC" \
  "$HARNESS_SRC" \
  -o "$HARNESS_BASENAME"

[[ -f "${HARNESS_BASENAME}.cubin" && -f "${HARNESS_BASENAME}.conf" ]] \
  || { echo "ERROR: coqui-cc did not emit .cubin/.conf" >&2; exit 1; }

echo
echo "=== Build complete ==="
ls -la "${HARNESS_BASENAME}.cubin" "${HARNESS_BASENAME}.conf" "${CPU_OUT}" seeds
echo "  conf:"; sed 's/^/    /' "${HARNESS_BASENAME}.conf"
