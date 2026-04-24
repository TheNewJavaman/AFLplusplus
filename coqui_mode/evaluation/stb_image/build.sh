#!/usr/bin/env bash
# build.sh — build stb_image coqui mode evaluation target (nix-free).
#
# Produces in the current directory:
#   stb_image_read_fuzzer.cubin — GPU kernel (configurable via ARCH env, default sm_75)
#   stb_image_read_fuzzer.conf  — companion config emitted by coqui-cc
#   stb_image_read_fuzzer_cpu   — AFL++ instrumented CPU binary
#   .build/                     — cached upstream source (stb_image.h)
#
# No external dependencies beyond afl-clang-fast (from this repo) and
# coqui-cc (installed to /usr/local/bin). stb_image.h is fetched from
# github.com/nothings/stb at a pinned commit.
#
# Overrides:
#   ARCH          GPU compute capability (default sm_75)
#   AFL_CC        path to afl-clang-fast (default: this repo's own afl-clang-fast)
#   COQUI_CC      path to coqui-cc (default /usr/local/bin/coqui-cc)
#   STB_CACHE     path to cache the fetched stb_image.h (default .build/stb_image.h)

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# --- Config -----------------------------------------------------------------
HARNESS_BASENAME="stb_image_read_fuzzer"
HARNESS_SRC="${SCRIPT_DIR}/harness.c"
ARCH="${ARCH:-sm_75}"
STACK_SIZE=65536          # multi-format decoder is stack-heavy
SLAB_POOL_SIZE=0

# stb upstream pin (matches coqui's legacy fetch at rev 28d546d5e)
STB_COMMIT="28d546d5eb77d4585506a20480f4de2e706dff4c"
STB_URL="https://raw.githubusercontent.com/nothings/stb/${STB_COMMIT}/stb_image.h"

# Tools
AFL_CC="${AFL_CC:-${SCRIPT_DIR}/../../../afl-clang-fast}"
COQUI_CC="${COQUI_CC:-/usr/local/bin/coqui-cc}"
STB_CACHE="${STB_CACHE:-${SCRIPT_DIR}/.build/stb_image.h}"

# Sanitizers matching legacy coqui (address + UBSan except object-size).
# stb_image has flexible-array member usage in some decoders so we avoid
# object-size. Same set used for the legacy cmark target.
SANITIZE_FLAGS=(
  "-fsanitize=address,array-bounds,bool,builtin,enum,integer-divide-by-zero,null,return,returns-nonnull-attribute,shift,signed-integer-overflow,unsigned-integer-overflow,unreachable,vla-bound"
  "-fno-sanitize-recover=array-bounds,bool,builtin,enum,integer-divide-by-zero,null,return,returns-nonnull-attribute,shift,signed-integer-overflow,unreachable,vla-bound"
  "-fno-stack-protector"
)

# --- Pre-flight -------------------------------------------------------------
[[ -x "$AFL_CC" ]]   || { echo "ERROR: afl-clang-fast not found at $AFL_CC" >&2; exit 1; }
[[ -x "$COQUI_CC" ]] || { echo "ERROR: coqui-cc not found at $COQUI_CC" >&2; exit 1; }
[[ -f "$HARNESS_SRC" ]] || { echo "ERROR: harness missing at $HARNESS_SRC" >&2; exit 1; }

# --- [1/3] Fetch upstream stb_image.h ---------------------------------------
echo "=== [1/3] Fetch stb_image.h at pinned commit ==="
mkdir -p "$(dirname "$STB_CACHE")"
if [[ ! -f "$STB_CACHE" ]]; then
  echo "  curl $STB_URL -> $STB_CACHE"
  curl -fsSL "$STB_URL" -o "$STB_CACHE"
fi
# Light integrity check: commit pin in the URL already pins contents, but
# verify the file is non-empty and has the stb header signature.
if ! grep -q "stb_image" "$STB_CACHE"; then
  echo "ERROR: cached $STB_CACHE doesn't look like stb_image.h" >&2
  rm -f "$STB_CACHE"
  exit 1
fi
echo "  stb_image.h: $(wc -l <"$STB_CACHE") lines, $(wc -c <"$STB_CACHE") bytes"

# --- [2/3] Build AFL++ CPU binary -------------------------------------------
echo "=== [2/3] Build AFL++ CPU binary with afl-clang-fast ==="
CPU_OUT="${HARNESS_BASENAME}_cpu"
"$AFL_CC" -O2 -g \
  "${SANITIZE_FLAGS[@]}" \
  -fsanitize=fuzzer \
  -I "$(dirname "$STB_CACHE")" \
  "$HARNESS_SRC" \
  -o "$CPU_OUT" \
  -lm

[[ -x "$CPU_OUT" ]] || { echo "ERROR: CPU binary not produced" >&2; exit 1; }

# --- [3/3] Build GPU cubin via coqui-cc -------------------------------------
echo "=== [3/3] Build GPU cubin via coqui-cc ==="
# coqui-cc takes the harness + stb_image.h include dir. STBI_ASSERT is
# stubbed because coqui mode's runtime has no __assert_fail. STBI_NO_HDR
# disables the Radiance decoder (uses strtol which isn't in the runtime).
"$COQUI_CC" \
  -arch "$ARCH" \
  --stack-size "$STACK_SIZE" \
  --slab-pool-size "$SLAB_POOL_SIZE" \
  -I "$(dirname "$STB_CACHE")" \
  -D "STBI_ASSERT(x)=" \
  -D STBI_NO_HDR \
  "$HARNESS_SRC" \
  -o "$HARNESS_BASENAME"

[[ -f "${HARNESS_BASENAME}.cubin" && -f "${HARNESS_BASENAME}.conf" ]] \
  || { echo "ERROR: coqui-cc did not emit .cubin/.conf" >&2; exit 1; }

echo
echo "=== Build complete ==="
ls -la "${HARNESS_BASENAME}.cubin" "${HARNESS_BASENAME}.conf" "${CPU_OUT}" seeds dict
echo "  conf:"; sed 's/^/    /' "${HARNESS_BASENAME}.conf"
