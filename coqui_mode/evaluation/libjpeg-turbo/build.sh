#!/usr/bin/env bash
# build.sh — build libjpeg-turbo coqui mode evaluation target.
#
# Produces in this directory:
#   libjpeg_turbo_decompress_fuzzer.cubin  — GPU kernel (sm_75)
#   libjpeg_turbo_decompress_fuzzer.conf   — sidecar emitted by coqui-cc
#   libjpeg_turbo_decompress_fuzzer_cpu    — AFL++ instrumented CPU binary
#                                             (symlink into the nix store)
#   seeds/                                 — seed corpus (symlink to nix store)
#   dict/                                  — jpeg marker dictionary (nix store)
#
# Mirrors the `libjpeg-turbo` target in the legacy coqui codebase exactly:
#   - libjpeg-turbo 3.0.4, 20 baseline-only .c files (no arithmetic coding,
#     no progressive, no block smoothing, no IDCT scaling, no color quant,
#     no upsample merging, no save markers, no input smoothing, no 12/16-bit)
#   - patched jmorecfg.h with heavyweight features #undef'd
#   - custom jconfig.h / jconfigint.h
#   - generated jversion.h (from jversion.h.in template)
#   - sources COPIED into jpeg_include/ so #include "jmorecfg.h" finds the
#     patched copy (quote-includes search the source file's directory first
#     before -I paths — see nix/targets/libjpeg-turbo.nix commit history)
#   - -D NO_GETENV
#   - --stack-size 32768 (coqui-cc default, matches nix spec)
#   - --slab-pool-size 0 (nix spec does not set a slab pool)
#
# Note: the nix spec passes `${sanitizers.default}` (-fsanitize=...) to the
# coqui driver, but coqui mode's coqui-cc does not accept -fsanitize flags —
# sanitizers are baked into the coqui mode pass plugin (/usr/local/lib/coqui-cc/
# CoquiPassPlugin.so). This matches the cjson/bzip2 build.sh pattern.
#
# Note: the nix spec stresses that libjpeg-turbo uses setjmp/longjmp in the
# TurboJPEG error-handling path. This is NOT a GPU-only blocker — the
# harness guards all setjmp/longjmp with `#ifndef __COQUI_DEVICE__`, so
# the device build never sees setjmp. The CPU build retains the longjmp
# error handler (oss-fuzz pattern).
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

COQUI_REPO="${COQUI_REPO:?set COQUI_REPO to your legacy coqui checkout path}"
COQUI_CC="/usr/local/bin/coqui-cc"
CPU_OUT_LINK="/tmp/coqui-libjpeg-turbo-cpu"
HARNESS_DIR="${COQUI_REPO}/harness/targets"
ARCH="sm_75"
STACK_SIZE=32768            # coqui-cc default; matches nix spec (no override)
SLAB_POOL_SIZE=0            # libjpeg-turbo.nix does not set a slab pool

HARNESS_NAME="libjpeg_turbo_decompress_fuzzer"

# Library sources — EXACTLY the list from nix/targets/libjpeg-turbo.nix.
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

echo "=== [1/5] Build AFL++ CPU binary via nix ==="
# Idempotent: re-running `nix build` is a no-op when the derivation is
# already realised.
cd "${COQUI_REPO}"
nix build '.#target-libjpeg-turbo-aflplusplus' --out-link "${CPU_OUT_LINK}"
cd "${SCRIPT_DIR}"

if [[ ! -x "${CPU_OUT_LINK}/${HARNESS_NAME}" ]]; then
  echo "ERROR: CPU binary not found at ${CPU_OUT_LINK}/${HARNESS_NAME}" >&2
  exit 1
fi

echo "=== [2/5] Resolve libjpeg-turbo source path ==="
# Find the libjpeg-turbo 3.0.4 source dir in the closure of the CPU build.
# Picking the unique `-source` entry that contains `jpeglib.h` + `jmorecfg.h`.
LIBJPEG_SRC=$(nix-store -qR "${CPU_OUT_LINK}" \
  | grep -E '/nix/store/[^/]+-source$' \
  | xargs -I{} sh -c '[ -f "{}/jpeglib.h" ] && [ -f "{}/jmorecfg.h" ] && [ -f "{}/jversion.h.in" ] && echo "{}"' \
  | head -n1)
if [[ -z "${LIBJPEG_SRC}" || ! -f "${LIBJPEG_SRC}/jpeglib.h" ]]; then
  echo "ERROR: could not resolve libjpeg-turbo source in closure of ${CPU_OUT_LINK}" >&2
  exit 1
fi
echo "  libjpeg-turbo source: ${LIBJPEG_SRC}"

# Sanity: every library .c file we plan to compile must exist upstream.
for c in "${LIBJPEG_SRCS[@]}"; do
  if [[ ! -f "${LIBJPEG_SRC}/${c}" ]]; then
    echo "ERROR: missing ${c} in ${LIBJPEG_SRC}" >&2
    exit 1
  fi
done

echo "=== [3/5] Stage patched jpeg_include/ tree ==="
# Replicate the nix spec's preBuild:
#   1. Symlink every upstream .h into jpeg_include/
#   2. Overwrite jmorecfg.h with a sed-patched copy (features disabled)
#   3. Write custom jconfig.h / jconfigint.h
#   4. Generate jversion.h from jversion.h.in template
#   5. COPY (not symlink) each library .c into jpeg_include/ so the quoted
#      `#include "jmorecfg.h"` directive finds the patched copy (same dir
#      as the source file) before the upstream unpatched copy on -I path.
STAGE_DIR="${SCRIPT_DIR}/jpeg_include"
rm -rf "${STAGE_DIR}"
mkdir -p "${STAGE_DIR}"

# 1. Link all headers from upstream into stage dir.
for h in "${LIBJPEG_SRC}"/*.h; do
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
  "${LIBJPEG_SRC}/jmorecfg.h" > "${STAGE_DIR}/jmorecfg.h"

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
sed 's/@COPYRIGHT_YEAR@/2024/' "${LIBJPEG_SRC}/jversion.h.in" > "${STAGE_DIR}/jversion.h"

# 6. Copy (not symlink) each library .c into jpeg_include/ so the quoted
#    #include "jmorecfg.h" directive finds the patched copy. If we left
#    the .c files in the upstream source dir, the compiler's "include from
#    source dir first" rule would find the unpatched jmorecfg.h alongside
#    the source file instead of our patched version in -I jpeg_include.
for c in "${LIBJPEG_SRCS[@]}"; do
  cp "${LIBJPEG_SRC}/${c}" "${STAGE_DIR}/${c}"
done

echo "  staged ${STAGE_DIR} (${#LIBJPEG_SRCS[@]} .c files + patched headers)"

echo "=== [4/5] Build GPU cubin via coqui-cc ==="
# Flags mirror nix/targets/libjpeg-turbo.nix:
#   -I jpeg_include          — patched headers (jmorecfg, jconfig, jconfigint, jversion)
#   -I ${LIBJPEG_SRC}        — upstream headers (jpeglib, etc.)
#   -D NO_GETENV             — strip getenv lookup on GPU (no libc env)
#   sources  = jpeg_include/*.c  (patched copies)
#            + harness/targets/libjpeg_turbo_stubs.c  (SIMD + 12/16-bit no-ops)
#            + harness/targets/libjpeg_turbo_decompress_fuzzer.c
#            + libjpeg_turbo_libc_stubs.c (local) — coqui mode pass plugin
#              has no snprintf port; jerror.c/format_message calls it
#              only for error strings that GPU never reads.
# Sanitizer flags from the nix spec are intentionally omitted — coqui-cc
# does not expose -fsanitize; the coqui mode pass plugin injects ASan/UBSan
# device-side via CoquiPassPlugin.so.
STAGED_SRCS=()
for c in "${LIBJPEG_SRCS[@]}"; do
  STAGED_SRCS+=("${STAGE_DIR}/${c}")
done

"${COQUI_CC}" \
  -arch "${ARCH}" \
  --stack-size "${STACK_SIZE}" \
  --slab-pool-size "${SLAB_POOL_SIZE}" \
  -I "${STAGE_DIR}" \
  -I "${LIBJPEG_SRC}" \
  -D NO_GETENV \
  "${STAGED_SRCS[@]}" \
  "${HARNESS_DIR}/libjpeg_turbo_stubs.c" \
  "${HARNESS_DIR}/${HARNESS_NAME}.c" \
  "${SCRIPT_DIR}/libjpeg_turbo_libc_stubs.c" \
  -o "${HARNESS_NAME}"

if [[ ! -f "${HARNESS_NAME}.cubin" || ! -f "${HARNESS_NAME}.conf" ]]; then
  echo "ERROR: coqui-cc did not emit ${HARNESS_NAME}.cubin / .conf" >&2
  exit 1
fi

echo "=== [5/5] Link CPU binary + seeds + dict ==="
ln -sfn "${CPU_OUT_LINK}/${HARNESS_NAME}" "${HARNESS_NAME}_cpu"
ln -sfn "${CPU_OUT_LINK}/seeds" seeds
if [[ -d "${CPU_OUT_LINK}/dict" ]]; then
  ln -sfn "${CPU_OUT_LINK}/dict" dict
fi

echo
echo "=== Build complete ==="
ls -la "${HARNESS_NAME}.cubin" "${HARNESS_NAME}.conf" "${HARNESS_NAME}_cpu" seeds 2>/dev/null || true
[ -L dict ] && ls -la dict
echo "  conf:"; sed 's/^/    /' "${HARNESS_NAME}.conf"
