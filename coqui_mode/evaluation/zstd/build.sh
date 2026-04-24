#!/usr/bin/env bash
# build.sh — build zstd coqui mode evaluation target (nix-free).
#
# Produces in the current directory:
#   zstd_simple_decompress_fuzzer.cubin — GPU kernel (configurable via ARCH env, default sm_75)
#   zstd_simple_decompress_fuzzer.conf  — companion config emitted by coqui-cc
#   zstd_simple_decompress_fuzzer_cpu   — AFL++ instrumented CPU binary
#   .build/                             — cached upstream source checkout
#
# No external dependencies beyond afl-clang-fast (from this repo) and
# coqui-cc (installed to /usr/local/bin). The zstd source tree is fetched
# from github.com/facebook/zstd at a pinned commit (v1.5.6).
#
# Overrides:
#   ARCH          GPU compute capability (default sm_75)
#   CLANG_FAST    path to afl-clang-fast (default: this repo's own afl-clang-fast).
#                 NOTE: named CLANG_FAST (no AFL_ prefix) because afl-cc
#                 itself reads $AFL_CC as the underlying compiler to delegate
#                 to (recursive self-call blows past MAX_PARAMS_NUM), and
#                 any unknown AFL_*  env var triggers a 2s sleep per compile
#                 via afl-common's "Mistyped AFL environment variable" check.
#   COQUI_CC      path to coqui-cc (default /usr/local/bin/coqui-cc)
#   ZSTD_SRC      path to a pre-fetched zstd source tree (default .build/<owner-repo-sha>)

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# --- Config -----------------------------------------------------------------
HARNESS_BASENAME="zstd_simple_decompress_fuzzer"
HARNESS_SRC="${SCRIPT_DIR}/harness.c"
ARCH="${ARCH:-sm_75}"
STACK_SIZE=32768        # coqui-cc default (legacy zstd.nix does not override)
SLAB_POOL_SIZE=0        # legacy zstd.nix does not configure a slab pool

# zstd upstream pin (facebook/zstd v1.5.6)
ZSTD_OWNER="facebook"
ZSTD_REPO="zstd"
ZSTD_COMMIT="35016bc1c0b9a2f7121b7ecc312100aad7d9f2ad"   # tag v1.5.6
ZSTD_SHORT="${ZSTD_COMMIT:0:12}"
ZSTD_TARBALL_URL="https://github.com/${ZSTD_OWNER}/${ZSTD_REPO}/archive/${ZSTD_COMMIT}.tar.gz"

# Tools
CLANG_FAST="${CLANG_FAST:-${SCRIPT_DIR}/../../../afl-clang-fast}"
COQUI_CC="${COQUI_CC:-/usr/local/bin/coqui-cc}"
ZSTD_SRC="${ZSTD_SRC:-${SCRIPT_DIR}/.build/${ZSTD_OWNER}-${ZSTD_REPO}-${ZSTD_SHORT}}"

# Sanitizers matching legacy coqui (address + full UBSan incl. object-size).
# zstd uses `sanitizers.default` in cpu-target-specs.nix.
SANITIZE_FLAGS=(
  "-fsanitize=address,array-bounds,bool,builtin,enum,integer-divide-by-zero,null,object-size,return,returns-nonnull-attribute,shift,signed-integer-overflow,unsigned-integer-overflow,unreachable,vla-bound"
  "-fno-sanitize-recover=array-bounds,bool,builtin,enum,integer-divide-by-zero,null,object-size,return,returns-nonnull-attribute,shift,signed-integer-overflow,unreachable,vla-bound"
  "-fno-stack-protector"
)

# --- Pre-flight -------------------------------------------------------------
[[ -x "$CLANG_FAST" ]]     || { echo "ERROR: afl-clang-fast not found at $CLANG_FAST" >&2; exit 1; }
[[ -x "$COQUI_CC" ]]      || { echo "ERROR: coqui-cc not found at $COQUI_CC" >&2; exit 1; }
[[ -f "$HARNESS_SRC" ]]   || { echo "ERROR: harness missing at $HARNESS_SRC" >&2; exit 1; }

# --- [1/3] Fetch upstream zstd at pinned commit -----------------------------
echo "=== [1/3] Fetch ${ZSTD_OWNER}/${ZSTD_REPO} @ ${ZSTD_COMMIT} ==="
mkdir -p "$(dirname "$ZSTD_SRC")"
# Marker file: lib/zstd.h at the root of the extracted tree.
if [[ ! -f "${ZSTD_SRC}/lib/zstd.h" ]]; then
  echo "  fetching ${ZSTD_TARBALL_URL}"
  tmp_tar="$(mktemp -t zstd-src-XXXXXX.tar.gz)"
  trap 'rm -f "$tmp_tar"' EXIT
  curl -fsSL "$ZSTD_TARBALL_URL" -o "$tmp_tar"
  # GitHub archive extracts to <repo>-<full-sha>/, so strip it into ZSTD_SRC.
  rm -rf "$ZSTD_SRC"
  mkdir -p "$ZSTD_SRC"
  tar -xzf "$tmp_tar" --strip-components=1 -C "$ZSTD_SRC"
  rm -f "$tmp_tar"
  trap - EXIT
fi
[[ -f "${ZSTD_SRC}/lib/zstd.h" ]] \
  || { echo "ERROR: ${ZSTD_SRC}/lib/zstd.h missing after fetch" >&2; exit 1; }
echo "  zstd source at: $ZSTD_SRC"

# --- Common compile args (shared between CPU + GPU builds) ------------------
ZSTD_INCLUDES=(
  -I "${ZSTD_SRC}/lib"
  -I "${ZSTD_SRC}/lib/common"
  -I "${ZSTD_SRC}/lib/decompress"
)
ZSTD_DEFINES=(
  -D ZSTD_NO_INTRINSICS=1
  -D ZSTD_DISABLE_ASM=1
  -D XXHASH_NAMESPACE=ZSTD_
  -D FUZZING_BUILD_MODE_UNSAFE_FOR_PRODUCTION
  -D ZSTD_NO_TRACE=1
  -D ZSTD_DECODER_INTERNAL_BUFFER=4096
)
ZSTD_SOURCES=(
  "${ZSTD_SRC}/lib/common/debug.c"
  "${ZSTD_SRC}/lib/common/entropy_common.c"
  "${ZSTD_SRC}/lib/common/error_private.c"
  "${ZSTD_SRC}/lib/common/fse_decompress.c"
  "${ZSTD_SRC}/lib/common/xxhash.c"
  "${ZSTD_SRC}/lib/common/zstd_common.c"
  "${ZSTD_SRC}/lib/decompress/huf_decompress.c"
  "${ZSTD_SRC}/lib/decompress/zstd_ddict.c"
  "${ZSTD_SRC}/lib/decompress/zstd_decompress.c"
  "${ZSTD_SRC}/lib/decompress/zstd_decompress_block.c"
)

# --- [2/3] Build AFL++ CPU binary -------------------------------------------
# Compile zstd library .c files to .o individually, then link with the
# harness. afl-cc expands each -fsanitize=... option into many clang flags,
# so passing 10+ .c files in a single invocation blows past afl-cc's
# MAX_PARAMS_NUM=2048 limit. Per-file compile keeps each argv manageable.
echo "=== [2/3] Build AFL++ CPU binary with afl-clang-fast ==="
CPU_OUT="${HARNESS_BASENAME}_cpu"
OBJ_DIR="${SCRIPT_DIR}/.build/cpu-objs"
mkdir -p "$OBJ_DIR"
CPU_OBJS=()
for src in "${ZSTD_SOURCES[@]}" "$HARNESS_SRC"; do
  obj="${OBJ_DIR}/$(basename "${src}" .c).o"
  echo "  cc -c $(basename "$src")"
  "$CLANG_FAST" -O2 -g -c \
    "${SANITIZE_FLAGS[@]}" \
    "${ZSTD_INCLUDES[@]}" \
    "${ZSTD_DEFINES[@]}" \
    "$src" \
    -o "$obj"
  CPU_OBJS+=("$obj")
done

echo "  link -> $CPU_OUT"
"$CLANG_FAST" -O2 -g \
  "${SANITIZE_FLAGS[@]}" \
  -fsanitize=fuzzer \
  "${CPU_OBJS[@]}" \
  -o "$CPU_OUT"

[[ -x "$CPU_OUT" ]] || { echo "ERROR: CPU binary not produced" >&2; exit 1; }

# --- [3/3] Build GPU cubin via coqui-cc -------------------------------------
# flock: ptxas at -O1 is memory-heavy (tens of GB); serialise with any other
# parallel coqui-cc invocations on this host so we don't OOM.
echo "=== [3/3] Build GPU cubin via coqui-cc (flock /tmp/coqui-cc.lock) ==="
flock /tmp/coqui-cc.lock \
  "$COQUI_CC" \
    -arch "$ARCH" \
    --stack-size "$STACK_SIZE" \
    --slab-pool-size "$SLAB_POOL_SIZE" \
    "${ZSTD_INCLUDES[@]}" \
    "${ZSTD_DEFINES[@]}" \
    "${ZSTD_SOURCES[@]}" \
    "$HARNESS_SRC" \
    -o "$HARNESS_BASENAME"

[[ -f "${HARNESS_BASENAME}.cubin" && -f "${HARNESS_BASENAME}.conf" ]] \
  || { echo "ERROR: coqui-cc did not emit .cubin/.conf" >&2; exit 1; }

echo
echo "=== Build complete ==="
ls -la "${HARNESS_BASENAME}.cubin" "${HARNESS_BASENAME}.conf" "${CPU_OUT}" seeds
echo "  conf:"; sed 's/^/    /' "${HARNESS_BASENAME}.conf"
