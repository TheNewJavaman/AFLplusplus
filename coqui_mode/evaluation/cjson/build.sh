#!/usr/bin/env bash
# build.sh — build cjson coqui mode evaluation target (nix-free).
#
# Produces in the current directory:
#   cjson_fuzzer.cubin — GPU kernel (configurable via ARCH env, default sm_75)
#   cjson_fuzzer.conf  — companion config emitted by coqui-cc
#   cjson_fuzzer_cpu   — AFL++ instrumented CPU binary
#   .build/            — cached upstream source (cJSON at pinned commit)
#
# No external dependencies beyond afl-clang-fast (from this repo) and
# coqui-cc (installed to /usr/local/bin). cJSON sources fetched from
# github.com/DaveGamble/cJSON at a pinned commit (v1.7.18).
#
# Overrides:
#   ARCH          GPU compute capability (default sm_75)
#   AFL_CC        path to afl-clang-fast (default: this repo's own afl-clang-fast)
#   COQUI_CC      path to coqui-cc (default /usr/local/bin/coqui-cc)
#   CJSON_CACHE   path to cache the fetched cJSON source (default .build/cJSON-<short>)

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# --- Config -----------------------------------------------------------------
HARNESS_BASENAME="cjson_fuzzer"
HARNESS_SRC="${SCRIPT_DIR}/harness.c"
ARCH="${ARCH:-sm_75}"
STACK_SIZE=32768           # matches legacy nix cjson.nix
SLAB_POOL_SIZE=0           # cjson does not need a slab pool

# cJSON upstream pin (matches coqui's legacy fetch at v1.7.18)
CJSON_COMMIT="acc76239bee01d8e9c858ae2cab296704e52d916"
CJSON_SHORT="${CJSON_COMMIT:0:7}"
CJSON_URL="https://github.com/DaveGamble/cJSON/archive/${CJSON_COMMIT}.tar.gz"

# Tools
AFL_CC="${AFL_CC:-${SCRIPT_DIR}/../../../afl-clang-fast}"
COQUI_CC="${COQUI_CC:-/usr/local/bin/coqui-cc}"
CJSON_CACHE="${CJSON_CACHE:-${SCRIPT_DIR}/.build/cJSON-${CJSON_SHORT}}"

# Sanitizers matching legacy coqui (sanitizers.default: address + full UBSan).
SANITIZE_FLAGS=(
  "-fsanitize=address,array-bounds,bool,builtin,enum,integer-divide-by-zero,null,object-size,return,returns-nonnull-attribute,shift,signed-integer-overflow,unsigned-integer-overflow,unreachable,vla-bound"
  "-fno-sanitize-recover=array-bounds,bool,builtin,enum,integer-divide-by-zero,null,object-size,return,returns-nonnull-attribute,shift,signed-integer-overflow,unreachable,vla-bound"
)

# --- Pre-flight -------------------------------------------------------------
[[ -x "$AFL_CC" ]]   || { echo "ERROR: afl-clang-fast not found at $AFL_CC" >&2; exit 1; }
[[ -x "$COQUI_CC" ]] || { echo "ERROR: coqui-cc not found at $COQUI_CC" >&2; exit 1; }
[[ -f "$HARNESS_SRC" ]] || { echo "ERROR: harness missing at $HARNESS_SRC" >&2; exit 1; }

# --- [1/3] Fetch upstream cJSON sources -------------------------------------
echo "=== [1/3] Fetch cJSON at pinned commit ${CJSON_SHORT} ==="
mkdir -p "$(dirname "$CJSON_CACHE")"
if [[ ! -f "${CJSON_CACHE}/cJSON.c" || ! -f "${CJSON_CACHE}/cJSON_Utils.c" ]]; then
  TARBALL="${CJSON_CACHE}.tar.gz"
  echo "  curl $CJSON_URL -> $TARBALL"
  curl -fsSL "$CJSON_URL" -o "$TARBALL"
  rm -rf "$CJSON_CACHE"
  mkdir -p "$CJSON_CACHE"
  # Extract with --strip-components=1 so files land directly in $CJSON_CACHE.
  tar -xzf "$TARBALL" -C "$CJSON_CACHE" --strip-components=1
  rm -f "$TARBALL"
fi
for f in cJSON.c cJSON_Utils.c cJSON.h cJSON_Utils.h; do
  [[ -f "${CJSON_CACHE}/${f}" ]] || {
    echo "ERROR: ${CJSON_CACHE}/${f} missing after fetch" >&2; exit 1; }
done
echo "  cJSON.c:       $(wc -l <"${CJSON_CACHE}/cJSON.c") lines"
echo "  cJSON_Utils.c: $(wc -l <"${CJSON_CACHE}/cJSON_Utils.c") lines"

# --- [2/3] Build AFL++ CPU binary -------------------------------------------
echo "=== [2/3] Build AFL++ CPU binary with afl-clang-fast ==="
CPU_OUT="${HARNESS_BASENAME}_cpu"
"$AFL_CC" -O2 -g \
  "${SANITIZE_FLAGS[@]}" \
  -fsanitize=fuzzer \
  -I "$CJSON_CACHE" \
  -D CJSON_HIDE_SYMBOLS \
  "${CJSON_CACHE}/cJSON.c" \
  "${CJSON_CACHE}/cJSON_Utils.c" \
  "$HARNESS_SRC" \
  -o "$CPU_OUT"

[[ -x "$CPU_OUT" ]] || { echo "ERROR: CPU binary not produced" >&2; exit 1; }

# --- [3/3] Build GPU cubin via coqui-cc -------------------------------------
echo "=== [3/3] Build GPU cubin via coqui-cc ==="
# Serialize with other parallel subagents: ptxas -O1 is memory-heavy.
flock /tmp/coqui-cc.lock "$COQUI_CC" \
  -arch "$ARCH" \
  --stack-size "$STACK_SIZE" \
  --slab-pool-size "$SLAB_POOL_SIZE" \
  -I "$CJSON_CACHE" \
  -D CJSON_HIDE_SYMBOLS \
  "${CJSON_CACHE}/cJSON.c" \
  "${CJSON_CACHE}/cJSON_Utils.c" \
  "$HARNESS_SRC" \
  -o "$HARNESS_BASENAME"

[[ -f "${HARNESS_BASENAME}.cubin" && -f "${HARNESS_BASENAME}.conf" ]] \
  || { echo "ERROR: coqui-cc did not emit .cubin/.conf" >&2; exit 1; }

echo
echo "=== Build complete ==="
ls -la "${HARNESS_BASENAME}.cubin" "${HARNESS_BASENAME}.conf" "${CPU_OUT}" seeds
echo "  conf:"; sed 's/^/    /' "${HARNESS_BASENAME}.conf"
