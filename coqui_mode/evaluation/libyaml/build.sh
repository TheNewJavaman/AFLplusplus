#!/usr/bin/env bash
# build.sh — build libyaml coqui mode evaluation target (nix-free).
#
# Produces in the current directory:
#   libyaml_parser_fuzzer.cubin — GPU kernel (configurable via ARCH env, default sm_75)
#   libyaml_parser_fuzzer.conf  — companion config emitted by coqui-cc
#   libyaml_parser_fuzzer_cpu   — AFL++ instrumented CPU binary
#   libyaml_src/                — patched libyaml sources
#   .build/                     — cached upstream source checkout
#
# No external dependencies beyond afl-clang-fast (from this repo) and
# coqui-cc (installed to /usr/local/bin). libyaml is fetched from
# github.com/yaml/libyaml at a pinned tag.
#
# Overrides:
#   ARCH          GPU compute capability (default sm_75)
#   AFL_CC        path to afl-clang-fast (default: this repo's own afl-clang-fast)
#   COQUI_CC      path to coqui-cc (default /usr/local/bin/coqui-cc)
#
# Notes:
#   - libyaml's yaml_private.h is patched to shrink INPUT_RAW_BUFFER_SIZE
#     16384 -> 512 and the INITIAL_{STACK,QUEUE,STRING}_SIZE defaults 16 -> 4
#     so the parser fits in the per-thread 64 KB heap. Matches the legacy
#     coqui libyaml target exactly.
#   - libyaml uses no autotools-generated config.h; version macros
#     (YAML_VERSION_*) are supplied via -D.
#   - libyaml.nix does NOT set --stack-size or --slab-pool-size, so we use
#     coqui-cc defaults (32768 stack; 0 slab pool).
#   - libyaml's api.c/scanner.c use assert() (from <assert.h>), which expands
#     to __assert_fail on NVPTX. The Libc.cpp pass rewrites __assert_fail to
#     __coqui_assert_fail, but we still pass -D NDEBUG to make assert() a
#     no-op so GPU asserts become non-fatal (reachable data-dependent paths);
#     the CPU AFL++ binary still honors them.
#   - strdup/memcpy/memmove/memset call sites are rewritten by the Libc.cpp
#     pass to their __coqui_* equivalents provided by the runtime, so no
#     local libc shim is needed.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# --- Config -----------------------------------------------------------------
HARNESS_BASENAME="libyaml_parser_fuzzer"
HARNESS_SRC="${SCRIPT_DIR}/harness.c"
ARCH="${ARCH:-sm_75}"
THREAD_BUDGET_US=500000    # 500ms: drops trap rate 12.5% → 1.1% (throughput ~neutral)

# libyaml upstream pin (matches legacy coqui fetch at rev 0.2.5).
LIBYAML_OWNER="yaml"
LIBYAML_REPO="libyaml"
LIBYAML_REV="0.2.5"
LIBYAML_SHORT="0.2.5"
LIBYAML_URL="https://github.com/${LIBYAML_OWNER}/${LIBYAML_REPO}/archive/${LIBYAML_REV}.tar.gz"
LIBYAML_CACHE_DIR="${SCRIPT_DIR}/.build/${LIBYAML_OWNER}-${LIBYAML_REPO}-${LIBYAML_SHORT}"
LIBYAML_TARBALL="${SCRIPT_DIR}/.build/libyaml-${LIBYAML_SHORT}.tar.gz"

# Tools
AFL_CC="${AFL_CC:-${SCRIPT_DIR}/../../../afl-clang-fast}"
COQUI_CC="${COQUI_CC:-/usr/local/bin/coqui-cc}"

# Sanitizers matching legacy coqui (sanitizers.default — address + full UBSan).
SANITIZE_FLAGS=(
  "-fsanitize=address,array-bounds,bool,builtin,enum,integer-divide-by-zero,null,object-size,return,returns-nonnull-attribute,shift,signed-integer-overflow,unsigned-integer-overflow,unreachable,vla-bound"
  "-fno-sanitize-recover=array-bounds,bool,builtin,enum,integer-divide-by-zero,null,object-size,return,returns-nonnull-attribute,shift,signed-integer-overflow,unreachable,vla-bound"
  "-fno-stack-protector"
)

# --- Pre-flight -------------------------------------------------------------
[[ -x "$AFL_CC" ]]   || { echo "ERROR: afl-clang-fast not found at $AFL_CC" >&2; exit 1; }
[[ -x "$COQUI_CC" ]] || { echo "ERROR: coqui-cc not found at $COQUI_CC" >&2; exit 1; }
[[ -f "$HARNESS_SRC" ]] || { echo "ERROR: harness missing at $HARNESS_SRC" >&2; exit 1; }

# --- [1/4] Fetch upstream libyaml at pinned tag -----------------------------
echo "=== [1/4] Fetch libyaml at tag ${LIBYAML_REV} ==="
mkdir -p "${SCRIPT_DIR}/.build"
# Marker file: include/yaml.h. If it exists under the cache dir, skip fetch.
if [[ ! -f "${LIBYAML_CACHE_DIR}/include/yaml.h" ]]; then
  echo "  curl ${LIBYAML_URL} -> ${LIBYAML_TARBALL}"
  curl -fsSL "${LIBYAML_URL}" -o "${LIBYAML_TARBALL}"
  rm -rf "${LIBYAML_CACHE_DIR}"
  mkdir -p "${LIBYAML_CACHE_DIR}"
  tar -xzf "${LIBYAML_TARBALL}" -C "${LIBYAML_CACHE_DIR}" --strip-components=1
fi
# Light integrity check.
[[ -f "${LIBYAML_CACHE_DIR}/src/yaml_private.h" && -f "${LIBYAML_CACHE_DIR}/include/yaml.h" ]] \
  || { echo "ERROR: libyaml fetch incomplete at ${LIBYAML_CACHE_DIR}" >&2; exit 1; }
echo "  libyaml source: ${LIBYAML_CACHE_DIR}"

# --- [2/4] Patch libyaml sources --------------------------------------------
echo "=== [2/4] Patch libyaml sources (yaml_private.h buffer shrink) ==="
# Patch yaml_private.h to shrink I/O buffers for the per-thread 64 KB GPU heap.
#   INPUT_RAW_BUFFER_SIZE: 16384 -> 512  (INPUT_BUFFER_SIZE follows as 3x)
#   INITIAL_STACK_SIZE:        16 -> 4
#   INITIAL_QUEUE_SIZE:        16 -> 4
#   INITIAL_STRING_SIZE:       16 -> 4
# Matches the legacy coqui libyaml target exactly.
rm -rf libyaml_src
mkdir -p libyaml_src
for f in api.c reader.c scanner.c parser.c loader.c yaml_private.h; do
  cp "${LIBYAML_CACHE_DIR}/src/${f}" "libyaml_src/${f}"
done
chmod +w libyaml_src/yaml_private.h
sed -i \
  -e 's/^#define INPUT_RAW_BUFFER_SIZE.*/#define INPUT_RAW_BUFFER_SIZE   512/' \
  -e 's/^#define INITIAL_STACK_SIZE.*/#define INITIAL_STACK_SIZE  4/' \
  -e 's/^#define INITIAL_QUEUE_SIZE.*/#define INITIAL_QUEUE_SIZE  4/' \
  -e 's/^#define INITIAL_STRING_SIZE.*/#define INITIAL_STRING_SIZE 4/' \
  libyaml_src/yaml_private.h

# --- [3/4] Build AFL++ CPU binary -------------------------------------------
echo "=== [3/4] Build AFL++ CPU binary with afl-clang-fast ==="
CPU_OUT="${HARNESS_BASENAME}_cpu"
"$AFL_CC" -O2 -g \
  "${SANITIZE_FLAGS[@]}" \
  -fsanitize=fuzzer \
  -I "${LIBYAML_CACHE_DIR}/include" \
  -D YAML_DECLARE_STATIC \
  -D "YAML_VERSION_STRING=\"${LIBYAML_REV}\"" \
  -D YAML_VERSION_MAJOR=0 \
  -D YAML_VERSION_MINOR=2 \
  -D YAML_VERSION_PATCH=5 \
  libyaml_src/api.c \
  libyaml_src/reader.c \
  libyaml_src/scanner.c \
  libyaml_src/parser.c \
  libyaml_src/loader.c \
  "$HARNESS_SRC" \
  -o "$CPU_OUT" \
  -lm

[[ -x "$CPU_OUT" ]] || { echo "ERROR: CPU binary not produced" >&2; exit 1; }

# --- [4/4] Build GPU cubin via coqui-cc -------------------------------------
echo "=== [4/4] Build GPU cubin via coqui-cc (serialized via flock) ==="
# flock serializes coqui-cc invocations across parallel builds.  ptxas at -O1
# is memory-heavy (tens of GB); concurrent builds would OOM the box.
flock /tmp/coqui-cc.lock \
  "$COQUI_CC" \
  -arch "$ARCH" \
  -I "${LIBYAML_CACHE_DIR}/include" \
  -I libyaml_src \
  -D YAML_DECLARE_STATIC \
  -D "YAML_VERSION_STRING=\"${LIBYAML_REV}\"" \
  -D YAML_VERSION_MAJOR=0 \
  -D YAML_VERSION_MINOR=2 \
  -D YAML_VERSION_PATCH=5 \
  -D NDEBUG \
  libyaml_src/api.c \
  libyaml_src/reader.c \
  libyaml_src/scanner.c \
  libyaml_src/parser.c \
  libyaml_src/loader.c \
  "$HARNESS_SRC" \
  -o "$HARNESS_BASENAME"

[[ -f "${HARNESS_BASENAME}.cubin" && -f "${HARNESS_BASENAME}.conf" ]] \
  || { echo "ERROR: coqui-cc did not emit .cubin/.conf" >&2; exit 1; }

echo
echo "=== Build complete ==="
ls -la "${HARNESS_BASENAME}.cubin" "${HARNESS_BASENAME}.conf" "${CPU_OUT}" seeds
echo "  conf:"; sed 's/^/    /' "${HARNESS_BASENAME}.conf"
echo "  runtime: export AFL_COQUI_THREAD_BUDGET_US=${THREAD_BUDGET_US}"
