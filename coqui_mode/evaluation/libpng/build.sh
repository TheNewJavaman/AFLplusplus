#!/usr/bin/env bash
# build.sh — build libpng coqui mode evaluation target (nix-free).
#
# Produces in the current directory:
#   libpng_read_fuzzer.cubin — GPU kernel (configurable via ARCH env, default sm_75)
#   libpng_read_fuzzer.conf  — companion config emitted by coqui-cc
#   libpng_read_fuzzer_cpu   — AFL++ instrumented CPU binary (libFuzzer-compatible)
#   libpng_include/          — patched pnglibconf.h + libpng header symlinks
#   .build/                  — cached upstream libpng + zlib sources
#
# No external dependencies beyond afl-clang-fast (from this repo) and
# coqui-cc (installed to /usr/local/bin). libpng v1.6.43 and zlib v1.3.1
# are fetched from GitHub release tarballs at pinned commits.
#
# Overrides (set as env vars before running build.sh):
#   ARCH            GPU compute capability (default sm_75)
#   AFL_CC          path to afl-clang-fast wrapper (default: this repo's own).
#                   build.sh unsets this after reading so afl-cc itself does
#                   not treat it as a backend-clang override.
#   COQUI_CC        path to coqui-cc (default /usr/local/bin/coqui-cc)
#   COQUI_LLC_OPT   extra llc flags forwarded to coqui-cc. If llc stalls on
#                   libpng's large module (~44k instrumented accesses) on
#                   your host, set this to "-O1" before running build.sh.
#                   Default unset (llc uses -O2).
#
# GPU vs CPU build differences:
#   GPU build defines PNG_NO_STDIO + PNG_NO_SETJMP so libpng errors route
#   through PNG_ABORT() -> abort(), which the Libc.cpp pass rewrites to
#   __coqui_abort() -> __coqui_trap().
#   CPU build defines PNG_NO_STDIO only; setjmp is kept so libpng longjmps
#   back to the harness on parse errors, keeping the persistent-mode loop
#   alive for AFL++.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# --- Config -----------------------------------------------------------------
HARNESS_BASENAME="libpng_read_fuzzer"
HARNESS_SRC="${SCRIPT_DIR}/harness.c"
ARCH="${ARCH:-sm_75}"
STACK_SIZE=32768          # libpng+zlib call chains exceed 8KB hardware stack
SLAB_POOL_SIZE=2147483648 # 2 GiB — matches libpng.nix

# Upstream pins (matches coqui's nix fetchFromGitHub revs).
LIBPNG_VERSION="1.6.43"
LIBPNG_URL="https://github.com/pnggroup/libpng/archive/refs/tags/v${LIBPNG_VERSION}.tar.gz"
ZLIB_VERSION="1.3.1"
ZLIB_URL="https://github.com/madler/zlib/archive/refs/tags/v${ZLIB_VERSION}.tar.gz"

BUILD_DIR="${SCRIPT_DIR}/.build"
LIBPNG_SRC="${BUILD_DIR}/libpng-${LIBPNG_VERSION}"
ZLIB_SRC="${BUILD_DIR}/zlib-${ZLIB_VERSION}"

# Tools. AFL_CC is the wrapper; unset the env var of the same name below so
# afl-cc doesn't interpret it as an override of its internal backend clang
# (that would cause infinite recursion: afl-cc execs $AFL_CC, which is
# another afl-cc, which reads $AFL_CC again, etc.).
AFL_CC_BIN="${AFL_CC:-${SCRIPT_DIR}/../../../afl-clang-fast}"
unset AFL_CC
COQUI_CC="${COQUI_CC:-/usr/local/bin/coqui-cc}"

# Users hitting llc register-allocator stalls can export COQUI_LLC_OPT="-O1"
# before invoking build.sh. coqui-cc will forward it to its llc step.
export COQUI_LLC_OPT="${COQUI_LLC_OPT:-}"

# Sanitizers matching legacy coqui cpu-target-specs.nix (sanitizers.default).
SANITIZE_FLAGS=(
  "-fsanitize=address,array-bounds,bool,builtin,enum,integer-divide-by-zero,null,object-size,return,returns-nonnull-attribute,shift,signed-integer-overflow,unsigned-integer-overflow,unreachable,vla-bound"
  "-fno-sanitize-recover=array-bounds,bool,builtin,enum,integer-divide-by-zero,null,object-size,return,returns-nonnull-attribute,shift,signed-integer-overflow,unreachable,vla-bound"
  "-fno-stack-protector"
)

# libpng + zlib source lists (matches libpng.nix and cpu-target-specs.nix).
ZLIB_SOURCES=(
  adler32.c compress.c crc32.c deflate.c infback.c inffast.c
  inflate.c inftrees.c trees.c uncompr.c zutil.c
)
LIBPNG_SOURCES=(
  png.c pngerror.c pngget.c pngmem.c pngpread.c pngread.c
  pngrio.c pngrtran.c pngrutil.c pngset.c pngtrans.c pngwio.c
  pngwrite.c pngwtran.c pngwutil.c
)

# --- Pre-flight -------------------------------------------------------------
[[ -x "$AFL_CC_BIN" ]]   || { echo "ERROR: afl-clang-fast not found at $AFL_CC_BIN" >&2; exit 1; }
[[ -x "$COQUI_CC" ]] || { echo "ERROR: coqui-cc not found at $COQUI_CC" >&2; exit 1; }
[[ -f "$HARNESS_SRC" ]] || { echo "ERROR: harness missing at $HARNESS_SRC" >&2; exit 1; }

# --- [1/4] Fetch upstream libpng + zlib sources -----------------------------
echo "=== [1/4] Fetch libpng ${LIBPNG_VERSION} + zlib ${ZLIB_VERSION} ==="
mkdir -p "$BUILD_DIR"

fetch_tarball() {
  local url="$1" out_dir="$2" marker="$3"
  if [[ -f "$out_dir/$marker" ]]; then
    echo "  cached: $out_dir"
    return 0
  fi
  echo "  curl $url"
  rm -rf "$out_dir"
  curl -fsSL "$url" | tar -xz -C "$BUILD_DIR"
  [[ -f "$out_dir/$marker" ]] \
    || { echo "ERROR: expected $out_dir/$marker after extraction" >&2; exit 1; }
}

fetch_tarball "$LIBPNG_URL" "$LIBPNG_SRC" "png.c"
fetch_tarball "$ZLIB_URL"   "$ZLIB_SRC"   "adler32.c"

# --- [2/4] Stage patched pnglibconf.h + libpng headers ----------------------
echo "=== [2/4] Patch pnglibconf.h and stage libpng headers ==="
# The prebuilt pnglibconf.h ships with several feature-support macros
# unconditionally defined; we strip different subsets for the CPU vs GPU
# builds and append PNG_DISABLE_ADLER32_CHECK_SUPPORTED so the fuzzer can
# explore zlib streams without needing valid checksums (matches libpng.nix).
#
# Two staged header dirs:
#   libpng_include/     — CPU build: keeps PNG_SETJMP_SUPPORTED (harness's
#                         setjmp path needs png_jmpbuf), keeps stdio +
#                         console_io + fp arithmetic (host libc has them).
#                         Only strips PNG_SIMPLIFIED_{READ,WRITE}_* so the
#                         simplified API (unused) isn't linked.
#   libpng_include_gpu/ — GPU build: strips everything libpng.nix strips,
#                         matching the legacy GPU build:
#                           PNG_SETJMP_SUPPORTED — pngconf.h would
#                             #include <setjmp.h>, coqui-cc's pass aborts
#                             on _setjmp.
#                           PNG_SIMPLIFIED_{READ,WRITE}_* — simplified-API
#                             functions use setjmp/longjmp directly.
#                           PNG_CONSOLE_IO_SUPPORTED — gates fprintf in
#                             pngerror.c; ExternalSymbolGatekeeper rejects
#                             unresolved fprintf.
#                           PNG_STDIO_SUPPORTED — gates fread/fwrite in
#                             pngrio.c/pngwio.c; harness uses callbacks.
#                           PNG_FLOATING_ARITHMETIC_SUPPORTED — gates
#                             pow/floor in png.c gamma code; NVPTX can't
#                             select llvm.pow.f64. Stripping forces the
#                             fixed-point gamma fallback. PNG_FLOATING_
#                             POINT_SUPPORTED stays so the public API
#                             (png_set_gamma(double,double)) still takes FP.
rm -rf libpng_include libpng_include_gpu
mkdir -p libpng_include libpng_include_gpu

# CPU: only strip the simplified-API macros and append ADLER32 bypass.
# The CPU setjmp path in the harness depends on PNG_SETJMP_SUPPORTED.
sed -e '/^#define PNG_SIMPLIFIED_.*$/d' \
    "${LIBPNG_SRC}/scripts/pnglibconf.h.prebuilt" > libpng_include/pnglibconf.h
echo '#define PNG_DISABLE_ADLER32_CHECK_SUPPORTED' >> libpng_include/pnglibconf.h

# GPU: strip setjmp, simplified-API, stdio/console-io, and fp-arithmetic.
sed -e '/^#define PNG_SETJMP_SUPPORTED$/d' \
    -e '/^#define PNG_SIMPLIFIED_.*$/d' \
    -e '/^#define PNG_CONSOLE_IO_SUPPORTED$/d' \
    -e '/^#define PNG_STDIO_SUPPORTED$/d' \
    -e '/^#define PNG_FLOATING_ARITHMETIC_SUPPORTED$/d' \
    "${LIBPNG_SRC}/scripts/pnglibconf.h.prebuilt" > libpng_include_gpu/pnglibconf.h
echo '#define PNG_DISABLE_ADLER32_CHECK_SUPPORTED' >> libpng_include_gpu/pnglibconf.h

# Symlink libpng's public headers alongside the patched pnglibconf.h in
# both include dirs so `#include "png.h"` resolves correctly.
for h in "${LIBPNG_SRC}"/*.h; do
  ln -sf "$h" "libpng_include/$(basename "$h")"
  ln -sf "$h" "libpng_include_gpu/$(basename "$h")"
done

# --- [3/4] Build AFL++ CPU binary via afl-clang-fast ------------------------
echo "=== [3/4] Build AFL++ CPU binary with afl-clang-fast ==="
CPU_OUT="${HARNESS_BASENAME}_cpu"
CPU_OBJ_DIR="${BUILD_DIR}/cpu_obj"
mkdir -p "$CPU_OBJ_DIR"

# Defines: PNG_NO_STDIO only (keep setjmp on CPU for longjmp-based recovery
# — matches cpu-target-specs.nix libpng entry). -lstdc++ is preserved for
# parity with the legacy spec's isCxx = true even though the harness is
# pure C (harmless for a C program).
#
# We compile each translation unit to an object file individually because
# afl-clang-fast prepends its own pass-plugin flags per-input-file and
# exceeds MAX_PARAMS_NUM (2048) when passing 27 sources at once.
#
# NOTE: -fsanitize=fuzzer is NOT used for per-file compile. afl-cc handles
# -fsanitize=fuzzer by swapping in libAFLDriver.a, and while that only
# matters at link time, the interaction with -fsanitize=address,... in the
# same compile command causes afl-cc to duplicate its -fpass-plugin flags
# multiple times, blowing past MAX_PARAMS_NUM (2048) even for a single
# translation unit. Compile with sanitizers only, link with fuzzer driver.
CPU_CFLAGS=(
  -O2 -g
  "${SANITIZE_FLAGS[@]}"
  -I libpng_include
  -I "$ZLIB_SRC"
  -D PNG_NO_STDIO
)

compile_cpu_obj() {
  local src="$1" obj="$2"
  "$AFL_CC_BIN" "${CPU_CFLAGS[@]}" -c "$src" -o "$obj"
}

CPU_OBJS=()
compile_cpu_obj "$HARNESS_SRC" "${CPU_OBJ_DIR}/harness.o"
CPU_OBJS+=("${CPU_OBJ_DIR}/harness.o")
for f in "${ZLIB_SOURCES[@]}"; do
  obj="${CPU_OBJ_DIR}/zlib_${f%.c}.o"
  compile_cpu_obj "${ZLIB_SRC}/${f}" "$obj"
  CPU_OBJS+=("$obj")
done
for f in "${LIBPNG_SOURCES[@]}"; do
  obj="${CPU_OBJ_DIR}/libpng_${f%.c}.o"
  compile_cpu_obj "${LIBPNG_SRC}/${f}" "$obj"
  CPU_OBJS+=("$obj")
done

# Link with fuzzer driver + sanitizer runtimes.
"$AFL_CC_BIN" -O2 -g \
  "${SANITIZE_FLAGS[@]}" \
  -fsanitize=fuzzer \
  "${CPU_OBJS[@]}" \
  -o "$CPU_OUT" \
  -lm -lstdc++

[[ -x "$CPU_OUT" ]] || { echo "ERROR: CPU binary not produced" >&2; exit 1; }

# --- [4/4] Build GPU cubin via coqui-cc -------------------------------------
echo "=== [4/4] Build GPU cubin via coqui-cc ==="
# Flags mirror the `libpng` target in the legacy coqui nix spec:
#   -D PNG_NO_STDIO  — disable stdio-based PNG I/O (harness uses callbacks)
#   -D PNG_NO_SETJMP — route libpng errors through PNG_ABORT() -> abort()
#                      -> Libc.cpp rewrite -> __coqui_abort() -> __coqui_trap()
#   -I libpng_include_gpu -I ${zlib_src}
#   --stack-size 32768  — deep call chains overflow 8KB default
#   --slab-pool-size 2 GiB — per-chunk allocations exceed 64KB per-thread heap
#
# NOTE: coqui-cc does NOT accept --heap-size, --batch-size, --ignore-signal=,
# or -fsanitize= flags (see libpng.nix for comparison). The coqui mode
# runtime derives heap at startup and reads batch size from
# AFL_COQUI_BATCH_SIZE at fuzz-time.
#
# libpng's unrecoverable-error path ends in PNG_ABORT() (= abort() by
# default, pngpriv.h). The Libc.cpp pass rewrites `abort` -> `__coqui_abort`
# (coqui_libc.c: __coqui_abort -> __coqui_trap), so no local stub is needed.
#
# flock /tmp/coqui-cc.lock serializes with parallel target builds — ptxas
# at -O1 can consume tens of GB; concurrent builds OOM the host.
GPU_SOURCES=()
for f in "${ZLIB_SOURCES[@]}";   do GPU_SOURCES+=("${ZLIB_SRC}/${f}");   done
for f in "${LIBPNG_SOURCES[@]}"; do GPU_SOURCES+=("${LIBPNG_SRC}/${f}"); done
GPU_SOURCES+=("$HARNESS_SRC")

flock /tmp/coqui-cc.lock \
"$COQUI_CC" \
  -arch "$ARCH" \
  --stack-size "$STACK_SIZE" \
  --slab-pool-size "$SLAB_POOL_SIZE" \
  -I libpng_include_gpu \
  -I "$ZLIB_SRC" \
  -D PNG_NO_STDIO \
  -D PNG_NO_SETJMP \
  "${GPU_SOURCES[@]}" \
  -o "$HARNESS_BASENAME"

[[ -f "${HARNESS_BASENAME}.cubin" && -f "${HARNESS_BASENAME}.conf" ]] \
  || { echo "ERROR: coqui-cc did not emit .cubin/.conf" >&2; exit 1; }

echo
echo "=== Build complete ==="
ls -la "${HARNESS_BASENAME}.cubin" "${HARNESS_BASENAME}.conf" "${CPU_OUT}" seeds
echo "  conf:"; sed 's/^/    /' "${HARNESS_BASENAME}.conf"
