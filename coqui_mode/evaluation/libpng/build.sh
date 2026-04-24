#!/usr/bin/env bash
# build.sh — build libpng coqui mode evaluation target.
#
# Produces in the current directory:
#   libpng_read_fuzzer.cubin      — GPU kernel (sm_75, compiled by coqui-cc)
#   libpng_read_fuzzer.conf       — companion config emitted by coqui-cc
#   libpng_read_fuzzer_cpu        — AFL++ instrumented CPU binary (symlink)
#   libpng_include/               — patched pnglibconf.h + libpng header symlinks
#   seeds/                        — initial seed corpus (symlink)
#   dict/                         — fuzzing dictionary (symlink)
#
# Mirrors the `libpng` target in the legacy coqui codebase — specifically:
#   * Patched pnglibconf.h (strip PNG_SETJMP_SUPPORTED + PNG_SIMPLIFIED_*,
#     add PNG_DISABLE_ADLER32_CHECK_SUPPORTED) via sed + echo.
#   * Symlink libpng's *.h into libpng_include/ so -I picks them up next to
#     the patched pnglibconf.h.
#   * Compile zlib + libpng sources + harness with -D PNG_NO_STDIO -D PNG_NO_SETJMP.
#   * --stack-size 32768 (libpng+zlib call chains overflow the 8KB default).
#   * --slab-pool-size 2 GiB (chunk allocations exceed the per-thread heap).
#
# Notes on coqui mode vs coqui nix differences:
#   * coqui-cc does NOT accept --heap-size, --batch-size, --ignore-signal=,
#     or -fsanitize=. The coqui mode runtime derives heap at startup and reads
#     batch size from AFL_COQUI_BATCH_SIZE.
#   * Sanitizers on the GPU build are not selected via -fsanitize= at
#     compile time. Instead:
#       - ASan is instrumented by coqui_mode/passes/Asan.cpp (outlined
#         fast-path helpers + heap shadow; enabled by default for every
#         cubin this driver builds).
#       - UBSan is NOT yet ported on the GPU side (no Ubsan.cpp pass).
#         CPU crash verification via afl-fuzz catches the UBSan-class
#         bugs post-hoc using the AFL++-instrumented CPU binary.
#       - CFI, MSan, TSan: not applicable for this fuzz surface.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# --- Fixed paths --------------------------------------------------------
COQUI_REPO="${COQUI_REPO:?set COQUI_REPO to your legacy coqui checkout path}"
COQUI_CC="/usr/local/bin/coqui-cc"
CPU_OUT_LINK="/tmp/coqui-libpng-cpu"
HARNESS_DIR="${COQUI_REPO}/harness/targets"
ARCH="${ARCH:-sm_75}"
STACK_SIZE=32768          # libpng+zlib call chains exceed 8KB hardware stack
SLAB_POOL_SIZE=2147483648 # 2 GiB — matches libpng.nix

CPU_BINARY_NAME="libpng_read_fuzzer"

# --- Pre-flight ---------------------------------------------------------
test -x "${COQUI_CC}"             || { echo "[build] missing ${COQUI_CC}" >&2; exit 1; }
test -d "${COQUI_REPO}"           || { echo "[build] missing ${COQUI_REPO}" >&2; exit 1; }
test -f "${COQUI_REPO}/flake.nix" || { echo "[build] ${COQUI_REPO} is not a flake" >&2; exit 1; }
test -f "${HARNESS_DIR}/libpng_read_fuzzer.c" \
  || { echo "[build] missing harness at ${HARNESS_DIR}/libpng_read_fuzzer.c" >&2; exit 1; }

echo "=== [1/5] Build AFL++ CPU binary via nix ==="
# target-libpng-aflplusplus ships: harness binary + seeds/ + dict/.
# Idempotent: `nix build` is a no-op if the derivation is already realised.
( cd "${COQUI_REPO}" && nix build '.#target-libpng-aflplusplus' \
    --out-link "${CPU_OUT_LINK}" )

if [[ ! -x "${CPU_OUT_LINK}/${CPU_BINARY_NAME}" ]]; then
  echo "[build] ERROR: CPU binary not found at ${CPU_OUT_LINK}/${CPU_BINARY_NAME}" >&2
  exit 1
fi

echo "=== [2/5] Resolve libpng + zlib source paths ==="
# Both are `-source` nix derivations in the closure. We identify them by
# the presence of distinctive files so we don't hardcode store hashes.
LIBPNG_SRC=""
ZLIB_SRC=""
for src in $(nix-store -qR "${CPU_OUT_LINK}" | grep -E -- '-source$'); do
  if [[ -z "${LIBPNG_SRC}" && -f "${src}/png.c" && -d "${src}/scripts" ]]; then
    LIBPNG_SRC="${src}"
  fi
  if [[ -z "${ZLIB_SRC}" && -f "${src}/adler32.c" && -f "${src}/deflate.c" ]]; then
    ZLIB_SRC="${src}"
  fi
done

if [[ -z "${LIBPNG_SRC}" ]]; then
  echo "[build] ERROR: could not resolve libpng source (expected png.c in /nix/store/*-source)" >&2
  exit 1
fi
if [[ -z "${ZLIB_SRC}" ]]; then
  echo "[build] ERROR: could not resolve zlib source (expected adler32.c + deflate.c in /nix/store/*-source)" >&2
  exit 1
fi
echo "  libpng src: ${LIBPNG_SRC}"
echo "  zlib src:   ${ZLIB_SRC}"

test -f "${LIBPNG_SRC}/scripts/pnglibconf.h.prebuilt" \
  || { echo "[build] ERROR: ${LIBPNG_SRC} missing scripts/pnglibconf.h.prebuilt" >&2; exit 1; }

echo "=== [3/5] Patch pnglibconf.h and stage libpng headers ==="
# Use the prebuilt pnglibconf.h shipped with libpng, but strip several
# feature-support macros that the prebuilt header unconditionally defines
# (overriding -D PNG_NO_SETJMP / -D PNG_NO_STDIO from the command line):
#
#   PNG_SETJMP_SUPPORTED      — stripped so pngconf.h does NOT #include
#                               <setjmp.h>; coqui-cc's pass aborts on _setjmp.
#   PNG_SIMPLIFIED_{READ,WRITE}_*  — the simplified-API functions
#                               (png_safe_error/png_safe_execute) use
#                               setjmp/longjmp directly under their own guard.
#   PNG_CONSOLE_IO_SUPPORTED  — coqui mode specific: gates the fprintf(stderr, ...)
#                               branches in pngerror.c. coqui mode's coqui-cc
#                               runtime.bc does NOT stub fprintf, so
#                               ExternalSymbolGatekeeper aborts ("unresolved
#                               external 'fprintf' — Used by: png_app_error").
#                               The coqui project's own runtime has a stub;
#                               coqui mode does not. Stripping CONSOLE_IO_SUPPORTED
#                               forces pngerror.c down the "assume nothing"
#                               path that only calls the user error_fn.
#   PNG_STDIO_SUPPORTED       — coqui mode specific: gates fread/fwrite usage in
#                               pngrio.c/pngwio.c. Same reason — no stdio
#                               stubs in coqui mode's device runtime. Our harness
#                               uses png_set_read_fn/png_set_write_fn, so the
#                               default stdio io-ptr paths are unused anyway.
#   PNG_FLOATING_ARITHMETIC_SUPPORTED — coqui mode specific: gates pow()/floor()
#                               calls in png.c gamma computation. The NVPTX
#                               backend cannot select llvm.pow.f64 without
#                               a math-runtime transform; coqui's build has
#                               MathTransform.cpp that rewrites intrinsics to
#                               __coqui_pow et al., coqui mode does not. Stripping
#                               this macro forces libpng's fixed-point gamma
#                               fallback (png_log8bit + png_exp8bit +
#                               png_muldiv) which uses only integer ops.
#                               PNG_FLOATING_POINT_SUPPORTED is kept so the
#                               public API (png_set_gamma(double,double))
#                               still accepts FP args from the harness; only
#                               the internal gamma table computation switches
#                               to fixed-point.
#
# Then append PNG_DISABLE_ADLER32_CHECK_SUPPORTED so the fuzzer can explore
# zlib streams without needing valid checksums (matches libpng.nix).
rm -rf libpng_include
mkdir -p libpng_include
sed -e '/^#define PNG_SETJMP_SUPPORTED$/d' \
    -e '/^#define PNG_SIMPLIFIED_.*$/d' \
    -e '/^#define PNG_CONSOLE_IO_SUPPORTED$/d' \
    -e '/^#define PNG_STDIO_SUPPORTED$/d' \
    -e '/^#define PNG_FLOATING_ARITHMETIC_SUPPORTED$/d' \
    "${LIBPNG_SRC}/scripts/pnglibconf.h.prebuilt" > libpng_include/pnglibconf.h
echo '#define PNG_DISABLE_ADLER32_CHECK_SUPPORTED' >> libpng_include/pnglibconf.h

# Symlink the libpng public headers so `#include "png.h"` resolves alongside
# the patched pnglibconf.h inside libpng_include/.
for h in "${LIBPNG_SRC}"/*.h; do
  ln -sf "${h}" "libpng_include/$(basename "${h}")"
done

echo "=== [4/5] Build GPU cubin via coqui-cc ==="
# Flags mirror the `libpng` target in the legacy coqui codebase:
#   -D PNG_NO_STDIO  — disable stdio-based PNG I/O (we use callbacks)
#   -D PNG_NO_SETJMP — route libpng errors through PNG_ABORT()->abort()->__coqui_abort
#   -I libpng_include -I ${zlib_src}
#   --stack-size 32768  — deep call chains (png_read_image -> ... -> inflate_fast)
#   --slab-pool-size 2 GiB — per-chunk allocations exceed the 64KB per-thread heap
#
# NOTE: coqui-cc does NOT accept --heap-size, --batch-size, --ignore-signal=,
# or -fsanitize= flags (see libpng.nix for comparison).
#
# Extra coqui mode-specific stub: png_abort_stub.c
#   libpng's internal error path ends in PNG_ABORT() (= abort() by default,
#   pngpriv.h:589), and its png_safe_* paths call abort() directly.
#   coqui's nix build papers over this with `--ignore-signal=abort` in its
#   pass plugin; coqui mode's ExternalSymbolGatekeeper does NOT accept `abort`
#   ("unresolved external 'abort' — Used by: png_chunk_error"). The stub
#   provides abort() -> __coqui_trap() so llvm-link has no unresolved
#   `abort` symbol. Same pattern as coqui_mode/evaluation/bzip2/bz2_assert_stub.c.
#
# Source list is identical to libpng.nix: all zlib object files needed for
# inflate + deflate, plus libpng's read + write paths (harness exercises both).
"${COQUI_CC}" \
  -arch "${ARCH}" \
  --stack-size "${STACK_SIZE}" \
  --slab-pool-size "${SLAB_POOL_SIZE}" \
  -I libpng_include \
  -I "${ZLIB_SRC}" \
  -D PNG_NO_STDIO \
  -D PNG_NO_SETJMP \
  "${ZLIB_SRC}/adler32.c" \
  "${ZLIB_SRC}/compress.c" \
  "${ZLIB_SRC}/crc32.c" \
  "${ZLIB_SRC}/deflate.c" \
  "${ZLIB_SRC}/infback.c" \
  "${ZLIB_SRC}/inffast.c" \
  "${ZLIB_SRC}/inflate.c" \
  "${ZLIB_SRC}/inftrees.c" \
  "${ZLIB_SRC}/trees.c" \
  "${ZLIB_SRC}/uncompr.c" \
  "${ZLIB_SRC}/zutil.c" \
  "${LIBPNG_SRC}/png.c" \
  "${LIBPNG_SRC}/pngerror.c" \
  "${LIBPNG_SRC}/pngget.c" \
  "${LIBPNG_SRC}/pngmem.c" \
  "${LIBPNG_SRC}/pngpread.c" \
  "${LIBPNG_SRC}/pngread.c" \
  "${LIBPNG_SRC}/pngrio.c" \
  "${LIBPNG_SRC}/pngrtran.c" \
  "${LIBPNG_SRC}/pngrutil.c" \
  "${LIBPNG_SRC}/pngset.c" \
  "${LIBPNG_SRC}/pngtrans.c" \
  "${LIBPNG_SRC}/pngwio.c" \
  "${LIBPNG_SRC}/pngwrite.c" \
  "${LIBPNG_SRC}/pngwtran.c" \
  "${LIBPNG_SRC}/pngwutil.c" \
  "${SCRIPT_DIR}/png_abort_stub.c" \
  "${HARNESS_DIR}/libpng_read_fuzzer.c" \
  -o libpng_read_fuzzer

if [[ ! -f libpng_read_fuzzer.cubin || ! -f libpng_read_fuzzer.conf ]]; then
  echo "[build] ERROR: coqui-cc did not emit libpng_read_fuzzer.cubin / .conf" >&2
  exit 1
fi

echo "=== [5/5] Link CPU binary + seeds + dict ==="
ln -sfn "${CPU_OUT_LINK}/${CPU_BINARY_NAME}" libpng_read_fuzzer_cpu
ln -sfn "${CPU_OUT_LINK}/seeds" seeds
if [[ -d "${CPU_OUT_LINK}/dict" ]]; then
  ln -sfn "${CPU_OUT_LINK}/dict" dict
fi

echo
echo "=== libpng build complete ==="
ls -la libpng_read_fuzzer.cubin libpng_read_fuzzer.conf libpng_read_fuzzer_cpu seeds 2>/dev/null || true
[[ -L dict ]] && ls -la dict || true
echo "  conf:"; sed 's/^/    /' libpng_read_fuzzer.conf
