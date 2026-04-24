#!/usr/bin/env bash
# build.sh — build cmark coqui mode evaluation target (nix-free).
#
# Produces in the current directory:
#   cmark_fuzzer.cubin — GPU kernel (configurable via ARCH env, default sm_75)
#   cmark_fuzzer.conf  — companion config emitted by coqui-cc
#   cmark_fuzzer_cpu   — AFL++ instrumented CPU binary
#   generated/         — cmark config headers synthesized inline (cmark ships
#                        no autotools/CMake step here — headers are required)
#   .build/            — cached upstream source (git clone of cmark at pinned rev)
#
# No external dependencies beyond afl-clang-fast (from this repo) and
# coqui-cc (installed to /usr/local/bin). cmark is cloned from
# github.com/commonmark/cmark at a pinned tag/commit.
#
# Overrides:
#   ARCH          GPU compute capability (default sm_75)
#   AFL_CC        path to afl-clang-fast (default: this repo's own afl-clang-fast)
#   COQUI_CC      path to coqui-cc (default /usr/local/bin/coqui-cc)
#   CMARK_SRC     path to the cmark source tree (default .build/commonmark-cmark-0.31.1)

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# --- Config -----------------------------------------------------------------
HARNESS_BASENAME="cmark_fuzzer"
HARNESS_SRC="${SCRIPT_DIR}/harness.c"
ARCH="${ARCH:-sm_75}"
STACK_SIZE=32768          # cmark's legacy coqui spec default
SLAB_POOL_SIZE=0

# cmark upstream pin (matches legacy nix spec: rev = tag 0.31.1)
CMARK_OWNER="commonmark"
CMARK_REPO="cmark"
CMARK_REV="0.31.1"
CMARK_URL="https://github.com/${CMARK_OWNER}/${CMARK_REPO}.git"

# Tools
AFL_CC="${AFL_CC:-${SCRIPT_DIR}/../../../afl-clang-fast}"
COQUI_CC="${COQUI_CC:-/usr/local/bin/coqui-cc}"
CMARK_SRC="${CMARK_SRC:-${SCRIPT_DIR}/.build/${CMARK_OWNER}-${CMARK_REPO}-${CMARK_REV}}"

# Sanitizers matching legacy coqui cpu-sanitizer-flags.nix `sanitizers.default`.
SANITIZE_FLAGS=(
  "-fsanitize=address,array-bounds,bool,builtin,enum,integer-divide-by-zero,null,object-size,return,returns-nonnull-attribute,shift,signed-integer-overflow,unsigned-integer-overflow,unreachable,vla-bound"
  "-fno-sanitize-recover=array-bounds,bool,builtin,enum,integer-divide-by-zero,null,object-size,return,returns-nonnull-attribute,shift,signed-integer-overflow,unreachable,vla-bound"
)

# cmark sources — matches both the GPU (nix/targets/cmark.nix) and CPU
# (nix/cpu-target-specs.nix → cmark) spec source lists exactly.
CMARK_SOURCES=(
  "${CMARK_SRC}/src/blocks.c"
  "${CMARK_SRC}/src/buffer.c"
  "${CMARK_SRC}/src/cmark.c"
  "${CMARK_SRC}/src/cmark_ctype.c"
  "${CMARK_SRC}/src/commonmark.c"
  "${CMARK_SRC}/src/houdini_href_e.c"
  "${CMARK_SRC}/src/houdini_html_e.c"
  "${CMARK_SRC}/src/houdini_html_u.c"
  "${CMARK_SRC}/src/html.c"
  "${CMARK_SRC}/src/inlines.c"
  "${CMARK_SRC}/src/iterator.c"
  "${CMARK_SRC}/src/latex.c"
  "${CMARK_SRC}/src/man.c"
  "${CMARK_SRC}/src/node.c"
  "${CMARK_SRC}/src/references.c"
  "${CMARK_SRC}/src/render.c"
  "${CMARK_SRC}/src/scanners.c"
  "${CMARK_SRC}/src/utf8.c"
  "${CMARK_SRC}/src/xml.c"
)

# --- Pre-flight -------------------------------------------------------------
[[ -x "$AFL_CC" ]]   || { echo "ERROR: afl-clang-fast not found at $AFL_CC" >&2; exit 1; }
[[ -x "$COQUI_CC" ]] || { echo "ERROR: coqui-cc not found at $COQUI_CC" >&2; exit 1; }
[[ -f "$HARNESS_SRC" ]] || { echo "ERROR: harness missing at $HARNESS_SRC" >&2; exit 1; }

# --- [1/4] Fetch upstream cmark source --------------------------------------
echo "=== [1/4] Fetch cmark source at pinned rev ${CMARK_REV} ==="
mkdir -p "$(dirname "$CMARK_SRC")"
# Cache marker: src/cmark.c — its presence means we've already cloned.
if [[ ! -f "${CMARK_SRC}/src/cmark.c" ]]; then
  rm -rf "$CMARK_SRC"
  echo "  git clone $CMARK_URL -> $CMARK_SRC"
  git clone --quiet "$CMARK_URL" "$CMARK_SRC"
  git -C "$CMARK_SRC" -c advice.detachedHead=false checkout --quiet "$CMARK_REV"
fi
echo "  cmark source: $CMARK_SRC ($(git -C "$CMARK_SRC" rev-parse --short HEAD))"

# --- [2/4] Generate cmark config headers ------------------------------------
echo "=== [2/4] Generate cmark config headers ==="
# cmark's upstream build is CMake; it generates config.h, cmark_export.h,
# and cmark_version.h from templates.  We skip CMake entirely and emit
# minimal equivalents inline (matches the legacy coqui nix preBuild).
mkdir -p "${SCRIPT_DIR}/generated"
cat > "${SCRIPT_DIR}/generated/config.h" << 'HEOF'
#define HAVE_STDBOOL_H 1
#define HAVE___BUILTIN_EXPECT 1
#define HAVE___ATTRIBUTE__ 1
HEOF
cat > "${SCRIPT_DIR}/generated/cmark_export.h" << 'HEOF'
#ifndef CMARK_EXPORT_H
#define CMARK_EXPORT_H
#define CMARK_EXPORT
#define CMARK_NO_EXPORT
#endif
HEOF
cat > "${SCRIPT_DIR}/generated/cmark_version.h" << 'HEOF'
#ifndef CMARK_VERSION_H
#define CMARK_VERSION_H
#define CMARK_VERSION ((0 << 16) | (31 << 8) | 1)
#define CMARK_VERSION_STRING "0.31.1"
#endif
HEOF

# --- [3/4] Build AFL++ CPU binary -------------------------------------------
# NOTE: afl-cc has a 2048-param cap (MAX_PARAMS_NUM in src/afl-cc.c).  The
# full sanitizer list + 19 cmark TUs in one invocation blows past it, so we
# compile each TU to .o first, then link.  Same final binary either way.
# -fsanitize=fuzzer is only passed at link time (afl-cc swaps it for
# libAFLDriver.a at that stage; -c with it would pull AFLDriver into every
# object and bloat the per-compile param count).
echo "=== [3/4] Build AFL++ CPU binary with afl-clang-fast ==="
CPU_OUT="${HARNESS_BASENAME}_cpu"
CPU_OBJ_DIR="${SCRIPT_DIR}/.build/cpu-objs"
rm -rf "$CPU_OBJ_DIR"
mkdir -p "$CPU_OBJ_DIR"

cpu_compile_flags=(
  -O2 -g
  "${SANITIZE_FLAGS[@]}"
  -I "${CMARK_SRC}/src"
  -I "${CMARK_SRC}/src/include"
  -I "${SCRIPT_DIR}/generated"
  -D CMARK_STATIC_DEFINE
)

cpu_objs=()
for src in "${CMARK_SOURCES[@]}" "$HARNESS_SRC"; do
  obj="${CPU_OBJ_DIR}/$(basename "${src%.c}").o"
  "$AFL_CC" "${cpu_compile_flags[@]}" -c "$src" -o "$obj"
  cpu_objs+=("$obj")
done

"$AFL_CC" -O2 -g \
  "${SANITIZE_FLAGS[@]}" \
  -fsanitize=fuzzer \
  "${cpu_objs[@]}" \
  -o "$CPU_OUT" \
  -lm

[[ -x "$CPU_OUT" ]] || { echo "ERROR: CPU binary not produced" >&2; exit 1; }

# --- [4/4] Build GPU cubin via coqui-cc -------------------------------------
echo "=== [4/4] Build GPU cubin via coqui-cc ==="
# flock /tmp/coqui-cc.lock serializes ptxas with other parallel builds
# (ptxas at -O1 can use tens of GB of RAM; parallel invocations OOM the box).
flock /tmp/coqui-cc.lock "$COQUI_CC" \
  -arch "$ARCH" \
  --stack-size "$STACK_SIZE" \
  --slab-pool-size "$SLAB_POOL_SIZE" \
  -I "${CMARK_SRC}/src" \
  -I "${CMARK_SRC}/src/include" \
  -I "${SCRIPT_DIR}/generated" \
  -D CMARK_STATIC_DEFINE \
  -D NDEBUG \
  "${CMARK_SOURCES[@]}" \
  "$HARNESS_SRC" \
  -o "$HARNESS_BASENAME"

[[ -f "${HARNESS_BASENAME}.cubin" && -f "${HARNESS_BASENAME}.conf" ]] \
  || { echo "ERROR: coqui-cc did not emit .cubin/.conf" >&2; exit 1; }

echo
echo "=== Build complete ==="
ls -la "${HARNESS_BASENAME}.cubin" "${HARNESS_BASENAME}.conf" "${CPU_OUT}" seeds
echo "  conf:"; sed 's/^/    /' "${HARNESS_BASENAME}.conf"
