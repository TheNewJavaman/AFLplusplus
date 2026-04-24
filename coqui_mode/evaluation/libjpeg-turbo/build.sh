#!/usr/bin/env bash
# build.sh — build libjpeg-turbo coqui mode evaluation target (nix-free).
#
# Produces in the current directory:
#   libjpeg_turbo_decompress_fuzzer.cubin  — GPU kernel (configurable via ARCH env, default sm_75)
#   libjpeg_turbo_decompress_fuzzer.conf   — companion config emitted by coqui-cc
#   libjpeg_turbo_decompress_fuzzer_cpu    — AFL++ instrumented CPU binary
#   .build/                                — cached upstream source tree (git clone at pinned rev)
#   jpeg_include/                          — staged patched header tree + library .c copies
#
# Baseline JPEG decode only.  Arithmetic coding, progressive, multi-scan,
# block smoothing, IDCT scaling/float/fast, color quantization, upsample
# merging, save markers, input smoothing, and 12/16-bit IDCT paths are all
# disabled via the patched jmorecfg.h + custom jconfig.h.  This shrinks
# the NVPTX module and avoids ptxas code-layout pathologies that cause
# runtime CUBIN hangs.
#
# No external dependencies beyond afl-clang-fast (from this repo),
# coqui-cc (installed to /usr/local/bin), curl/git (for fetch), and
# basic POSIX tools (sed, cp).  Upstream source is fetched from
# github.com/libjpeg-turbo/libjpeg-turbo at tag 3.0.4.
#
# Overrides:
#   ARCH          GPU compute capability (default sm_75)
#   AFL_CC        path to afl-clang-fast (default: this repo's own afl-clang-fast)
#   COQUI_CC      path to coqui-cc (default /usr/local/bin/coqui-cc)
#   LIBJPEG_CACHE path to cache the cloned libjpeg-turbo tree
#                 (default .build/libjpeg-turbo-libjpeg-turbo-<short-sha>)

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# --- Config -----------------------------------------------------------------
HARNESS_BASENAME="libjpeg_turbo_decompress_fuzzer"
HARNESS_SRC="${SCRIPT_DIR}/harness.c"
STUBS_SRC="${SCRIPT_DIR}/libjpeg_turbo_stubs.c"
ARCH="${ARCH:-sm_75}"
STACK_SIZE=32768          # coqui-cc default; matches nix spec (no override)
SLAB_POOL_SIZE=0          # libjpeg-turbo.nix does not set a slab pool

# libjpeg-turbo upstream pin (matches coqui's legacy fetch at rev 3.0.4).
LIBJPEG_TAG="3.0.4"
LIBJPEG_COMMIT="${LIBJPEG_TAG}"   # tag doubles as the git ref
LIBJPEG_URL="https://github.com/libjpeg-turbo/libjpeg-turbo.git"

# Tools
AFL_CC="${AFL_CC:-${SCRIPT_DIR}/../../../afl-clang-fast}"
COQUI_CC="${COQUI_CC:-/usr/local/bin/coqui-cc}"
LIBJPEG_CACHE="${LIBJPEG_CACHE:-${SCRIPT_DIR}/.build/libjpeg-turbo-${LIBJPEG_TAG}}"

# Library sources — EXACTLY the list from the legacy libjpeg-turbo nix spec.
# Baseline-only: progressive, multi-scan, smoothing, IDCT scaling/float/fast,
# color quant, upsample merging, save markers, input smoothing all disabled.
LIBJPEG_SRCS=(
  jdapimin.c jdapistd.c jdatasrc.c jdcoefct.c jdcolor.c
  jddctmgr.c jdhuff.c jdinput.c jdmainct.c jdmarker.c
  jdmaster.c jdpostct.c jdsample.c
  jidctint.c jmemmgr.c
  jmemnobs.c jerror.c jutils.c
  jcomapi.c
)

# Sanitizers matching legacy coqui (sanitizers.default = address + full UBSan).
SANITIZE_FLAGS=(
  "-fsanitize=address,array-bounds,bool,builtin,enum,integer-divide-by-zero,null,object-size,return,returns-nonnull-attribute,shift,signed-integer-overflow,unsigned-integer-overflow,unreachable,vla-bound"
  "-fno-sanitize-recover=array-bounds,bool,builtin,enum,integer-divide-by-zero,null,object-size,return,returns-nonnull-attribute,shift,signed-integer-overflow,unreachable,vla-bound"
  "-fno-stack-protector"
)

# --- Pre-flight -------------------------------------------------------------
[[ -x "$AFL_CC"         ]] || { echo "ERROR: afl-clang-fast not found at $AFL_CC"         >&2; exit 1; }
[[ -x "$COQUI_CC"       ]] || { echo "ERROR: coqui-cc not found at $COQUI_CC"             >&2; exit 1; }
[[ -f "$HARNESS_SRC"    ]] || { echo "ERROR: harness missing at $HARNESS_SRC"             >&2; exit 1; }
[[ -f "$STUBS_SRC"      ]] || { echo "ERROR: jsimd stubs missing at $STUBS_SRC"           >&2; exit 1; }

# --- [1/4] Fetch upstream libjpeg-turbo at pinned tag -----------------------
echo "=== [1/4] Fetch libjpeg-turbo ${LIBJPEG_TAG} ==="
mkdir -p "$(dirname "$LIBJPEG_CACHE")"
if [[ ! -f "$LIBJPEG_CACHE/jpeglib.h" || ! -f "$LIBJPEG_CACHE/jmorecfg.h" || ! -f "$LIBJPEG_CACHE/jversion.h.in" ]]; then
  echo "  git clone ${LIBJPEG_URL} -> ${LIBJPEG_CACHE}"
  rm -rf "$LIBJPEG_CACHE"
  git clone --quiet --depth 1 --branch "${LIBJPEG_TAG}" "${LIBJPEG_URL}" "${LIBJPEG_CACHE}"
fi
# Sanity: every library .c file we plan to compile must exist upstream.
for c in "${LIBJPEG_SRCS[@]}"; do
  [[ -f "${LIBJPEG_CACHE}/${c}" ]] || { echo "ERROR: missing ${c} in ${LIBJPEG_CACHE}" >&2; exit 1; }
done
echo "  libjpeg-turbo source: ${LIBJPEG_CACHE}"

# --- [2/4] Stage patched jpeg_include/ tree ---------------------------------
# Replicates the legacy nix spec's preBuild exactly:
#   1. Symlink every upstream .h into jpeg_include/
#   2. Overwrite jmorecfg.h with a sed-patched copy (features disabled)
#   3. Write custom jconfig.h / jconfigint.h
#   4. Generate jversion.h from jversion.h.in template
#   5. COPY (not symlink) each library .c into jpeg_include/ so the quoted
#      `#include "jmorecfg.h"` directive finds the patched copy (same dir
#      as the source file) before the upstream unpatched copy on -I path.
echo "=== [2/4] Stage patched jpeg_include/ tree ==="
STAGE_DIR="${SCRIPT_DIR}/jpeg_include"
rm -rf "${STAGE_DIR}"
mkdir -p "${STAGE_DIR}"

# 1. Link all headers from upstream into stage dir.
for h in "${LIBJPEG_CACHE}"/*.h; do
  ln -sfn "${h}" "${STAGE_DIR}/$(basename "${h}")"
done

# 2. Replace jmorecfg.h with the sed-patched variant (disables heavyweight
#    features to shrink NVPTX module and avoid ptxas code-layout pathologies).
rm -f "${STAGE_DIR}/jmorecfg.h"
sed \
  -e 's/^#define D_MULTISCAN_FILES_SUPPORTED/\/* #undef D_MULTISCAN_FILES_SUPPORTED *\//' \
  -e 's/^#define D_PROGRESSIVE_SUPPORTED/\/* #undef D_PROGRESSIVE_SUPPORTED *\//' \
  -e 's/^#define D_LOSSLESS_SUPPORTED/\/* #undef D_LOSSLESS_SUPPORTED *\//' \
  -e 's/^#define BLOCK_SMOOTHING_SUPPORTED/\/* #undef BLOCK_SMOOTHING_SUPPORTED *\//' \
  -e 's/^#define IDCT_SCALING_SUPPORTED/\/* #undef IDCT_SCALING_SUPPORTED *\//' \
  -e 's/^#define UPSAMPLE_MERGING_SUPPORTED/\/* #undef UPSAMPLE_MERGING_SUPPORTED *\//' \
  -e 's/^#define QUANT_1PASS_SUPPORTED/\/* #undef QUANT_1PASS_SUPPORTED *\//' \
  -e 's/^#define QUANT_2PASS_SUPPORTED/\/* #undef QUANT_2PASS_SUPPORTED *\//' \
  -e 's/^#define DCT_IFAST_SUPPORTED/\/* #undef DCT_IFAST_SUPPORTED *\//' \
  -e 's/^#define DCT_FLOAT_SUPPORTED/\/* #undef DCT_FLOAT_SUPPORTED *\//' \
  -e 's/^#define SAVE_MARKERS_SUPPORTED/\/* #undef SAVE_MARKERS_SUPPORTED *\//' \
  -e 's/^#define INPUT_SMOOTHING_SUPPORTED/\/* #undef INPUT_SMOOTHING_SUPPORTED *\//' \
  "${LIBJPEG_CACHE}/jmorecfg.h" > "${STAGE_DIR}/jmorecfg.h"

# 3. jconfig.h — no arithmetic coding, 8-bit only, no SIMD.
cat > "${STAGE_DIR}/jconfig.h" <<'HEADER'
#define HAVE_PROTOTYPES 1
#define HAVE_UNSIGNED_CHAR 1
#define HAVE_UNSIGNED_SHORT 1
#define HAVE_STDDEF_H 1
#define HAVE_STDLIB_H 1
#define HAVE_LOCALE_H 1
#define NEED_SYS_TYPES_H 1
#define JPEG_LIB_VERSION 62
#define LIBJPEG_TURBO_VERSION 3.0.4
#define LIBJPEG_TURBO_VERSION_NUMBER 3000004
#define BITS_IN_JSAMPLE 8
#define MEM_SRCDST_SUPPORTED 1
#define WITH_SIMD 0
HEADER

# 4. jconfigint.h — internal build config.
cat > "${STAGE_DIR}/jconfigint.h" <<'HEADER'
#define BUILD ""
#define INLINE inline
#define THREAD_LOCAL
#define PACKAGE_NAME "libjpeg-turbo"
#define VERSION "3.0.4"
#define SIZEOF_SIZE_T 8
#define FALLTHROUGH
HEADER

# 5. jversion.h from template (substitute copyright year).
sed 's/@COPYRIGHT_YEAR@/2024/' "${LIBJPEG_CACHE}/jversion.h.in" > "${STAGE_DIR}/jversion.h"

# 6. Copy (not symlink) each library .c into jpeg_include/ so the quoted
#    #include "jmorecfg.h" directive finds the patched copy. If we left
#    the .c files in the upstream source dir, the compiler's "include from
#    source dir first" rule would find the unpatched jmorecfg.h alongside
#    the source file instead of our patched version in -I jpeg_include.
STAGED_SRCS=()
for c in "${LIBJPEG_SRCS[@]}"; do
  cp "${LIBJPEG_CACHE}/${c}" "${STAGE_DIR}/${c}"
  STAGED_SRCS+=("${STAGE_DIR}/${c}")
done

echo "  staged ${STAGE_DIR} (${#LIBJPEG_SRCS[@]} .c files + patched headers)"

# Ensure seeds/ has at least one minimal seed.  Regenerate if missing.
echo "  verifying seeds/ ..."
if [[ ! -d "${SCRIPT_DIR}/seeds" || -z "$(ls -A "${SCRIPT_DIR}/seeds" 2>/dev/null)" ]]; then
  echo "ERROR: seeds/ is empty.  Expected at least one minimal JPEG seed." >&2
  exit 1
fi

# --- [3/4] Build AFL++ CPU binary -------------------------------------------
echo "=== [3/4] Build AFL++ CPU binary with afl-clang-fast ==="
# The CPU build compiles libjpeg-turbo + jsimd stubs + harness.
# Compile each source to .o separately then link — afl-clang-fast's argv
# buffer is small (MAX_PARAMS_NUM=2048) and the heavy -D macro expansion
# afl-cc injects (__AFL_LOOP, __AFL_FUZZ_INIT, __AFL_FUZZ_TESTCASE_BUF, …)
# overflows when combined with 20+ .c sources and a dozen sanitizer flags
# on one command line.
CPU_BUILD_DIR="${SCRIPT_DIR}/.build/cpu-obj"
rm -rf "${CPU_BUILD_DIR}"
mkdir -p "${CPU_BUILD_DIR}"

compile_cpu_obj() {
  local src="$1" out="$2"
  "$AFL_CC" -O2 -g \
    "${SANITIZE_FLAGS[@]}" \
    -I "${STAGE_DIR}" \
    -I "${LIBJPEG_CACHE}" \
    -D NO_GETENV \
    -c "${src}" -o "${out}"
}

CPU_OBJS=()
for c in "${LIBJPEG_SRCS[@]}"; do
  obj="${CPU_BUILD_DIR}/${c%.c}.o"
  compile_cpu_obj "${STAGE_DIR}/${c}" "${obj}"
  CPU_OBJS+=("${obj}")
done
compile_cpu_obj "${STUBS_SRC}"   "${CPU_BUILD_DIR}/libjpeg_turbo_stubs.o"
compile_cpu_obj "${HARNESS_SRC}" "${CPU_BUILD_DIR}/harness.o"
CPU_OBJS+=("${CPU_BUILD_DIR}/libjpeg_turbo_stubs.o" "${CPU_BUILD_DIR}/harness.o")

CPU_OUT="${HARNESS_BASENAME}_cpu"
"$AFL_CC" -O2 -g \
  "${SANITIZE_FLAGS[@]}" \
  -fsanitize=fuzzer \
  "${CPU_OBJS[@]}" \
  -o "${CPU_OUT}" \
  -lm

[[ -x "$CPU_OUT" ]] || { echo "ERROR: CPU binary not produced" >&2; exit 1; }

# --- [4/4] Build GPU cubin via coqui-cc -------------------------------------
# Flags mirror the legacy libjpeg-turbo nix spec:
#   -I jpeg_include          — patched headers (jmorecfg, jconfig, jconfigint, jversion)
#   -I ${LIBJPEG_CACHE}      — upstream headers (jpeglib, etc.)
#   -D NO_GETENV             — strip getenv lookup on GPU (no libc env)
#   sources  = jpeg_include/*.c  (patched copies)
#            + libjpeg_turbo_stubs.c  (SIMD + 12/16-bit no-ops)
#            + harness.c
# jerror.c's snprintf/fprintf/exit/stderr references are rewritten to
# their __coqui_* equivalents by the Libc.cpp pass; no local libc shim
# is needed anymore.
# Sanitizer flags from the nix spec are intentionally omitted — coqui-cc
# does not expose -fsanitize; the coqui mode pass plugin injects ASan/UBSan
# device-side via CoquiPassPlugin.so.  Also, ptxas is memory-heavy at -O1;
# flock serialises the coqui-cc step with other parallel subagents to
# avoid OOMing the box.
echo "=== [4/4] Build GPU cubin via coqui-cc (serialised with flock) ==="
flock /tmp/coqui-cc.lock \
  "${COQUI_CC}" \
    -arch "${ARCH}" \
    --stack-size "${STACK_SIZE}" \
    --slab-pool-size "${SLAB_POOL_SIZE}" \
    -I "${STAGE_DIR}" \
    -I "${LIBJPEG_CACHE}" \
    -D NO_GETENV \
    "${STAGED_SRCS[@]}" \
    "${STUBS_SRC}" \
    "${HARNESS_SRC}" \
    -o "${HARNESS_BASENAME}"

[[ -f "${HARNESS_BASENAME}.cubin" && -f "${HARNESS_BASENAME}.conf" ]] \
  || { echo "ERROR: coqui-cc did not emit .cubin/.conf" >&2; exit 1; }

echo
echo "=== Build complete ==="
ls -la "${HARNESS_BASENAME}.cubin" "${HARNESS_BASENAME}.conf" "${CPU_OUT}" seeds
echo "  conf:"; sed 's/^/    /' "${HARNESS_BASENAME}.conf"
