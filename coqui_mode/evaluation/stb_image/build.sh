#!/usr/bin/env bash
# build.sh — build stb_image coqui mode evaluation target.
#
# Produces in the current directory:
#   stb_image_read_fuzzer.cubin — GPU kernel (sm_75)
#   stb_image_read_fuzzer.conf  — companion config emitted by coqui-cc
#   stb_image_read_fuzzer_cpu   — AFL++ instrumented CPU binary (symlink to nix store)
#   seeds/                      — initial seed corpus (symlink to nix store)
#   dict/                       — fuzzing dictionary (symlink to nix store)
#
# Mirrors coqui's nix spec: the `stb_image` target in the legacy coqui codebase
#
# stb_image notes:
#   - stb is a header-only library, so only the harness .c is compiled
#     (STB_IMAGE_IMPLEMENTATION is defined inside stb_image_read_fuzzer.c).
#   - STBI_NO_STDIO / STBI_NO_SIMD / STBI_NO_THREAD_LOCALS are defined in the
#     harness itself; we don't need to pass them as -D.
#   - nix spec uses --stack-size 65536 (double the default) because the
#     multi-format decoder is stack-overflow-prone. Dim cap is 64 in harness.
#   - nix spec sets sanitizer flags via the coqui wrapper; coqui-cc does NOT
#     accept those flags, so we omit them here (they only matter for the CPU
#     binary, which nix builds separately).
#   - stb_image.h calls assert() which expands to __assert_fail on NVPTX.
#     coqui mode's runtime has no __assert_fail stub, and its ExternalSymbolGatekeeper
#     hard-fails. stb_image documents an override: define STBI_ASSERT(x) before
#     the header. We pass `-D STBI_ASSERT(x)=` so asserts become no-ops on GPU.
#     (Nix/coqui doesn't need this because the coqui wrapper provides a stub.)
#   - STBI_NO_HDR: the Radiance HDR decoder calls strtol(), which coqui mode's runtime
#     does not stub. Disabling HDR on GPU is a GPU-only restriction; the CPU
#     AFL++ binary built by nix still enables HDR. Consequence: GPU-driven
#     coverage won't steer toward HDR code paths. Add a runtime strtol stub
#     (ideally in coqui mode runtime.bc) to lift this limitation.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

COQUI_REPO="${COQUI_REPO:?set COQUI_REPO to your legacy coqui checkout path}"
COQUI_CC="/usr/local/bin/coqui-cc"
CPU_OUT_LINK="/tmp/coqui-stb_image-cpu"
HARNESS_DIR="${COQUI_REPO}/harness/targets"
ARCH="sm_75"
STACK_SIZE=65536         # from stb_image.nix (--stack-size 65536)
SLAB_POOL_SIZE=0         # stb_image.nix does not set a slab pool

HARNESS_BASENAME="stb_image_read_fuzzer"

echo "=== [1/4] Build AFL++ CPU binary via nix ==="
# Idempotent: `nix build` is a no-op if the derivation is already realised.
cd "${COQUI_REPO}"
nix build '.#target-stb_image-aflplusplus' --out-link "${CPU_OUT_LINK}"
cd "${SCRIPT_DIR}"

if [[ ! -x "${CPU_OUT_LINK}/${HARNESS_BASENAME}" ]]; then
  echo "ERROR: CPU binary not found at ${CPU_OUT_LINK}/${HARNESS_BASENAME}" >&2
  exit 1
fi

echo "=== [2/4] Resolve stb source path ==="
# stb is fetched via fetchFromGitHub, so the header lives in a `-source`
# derivation inside the CPU build's closure. Look for one that contains
# stb_image.h so we don't accidentally pick up some other -source derivation.
STB_SRC=$(nix-store -qR "${CPU_OUT_LINK}" \
  | grep -E '/nix/store/[^/]+-source$' \
  | xargs -I{} sh -c '[ -f "{}/stb_image.h" ] && echo "{}"' \
  | head -n1)
if [[ -z "${STB_SRC}" || ! -f "${STB_SRC}/stb_image.h" ]]; then
  echo "ERROR: could not locate stb source (expected stb_image.h)" >&2
  exit 1
fi
echo "  stb source: ${STB_SRC}"

HARNESS="${HARNESS_DIR}/${HARNESS_BASENAME}.c"
if [[ ! -f "${HARNESS}" ]]; then
  echo "ERROR: missing harness: ${HARNESS}" >&2
  exit 1
fi

echo "=== [3/4] Build GPU cubin via coqui-cc ==="
# Flags mirror the `stb_image` target in the legacy coqui codebase plus one coqui mode-only
# fix:
#   -arch sm_75                 — required for coqui mode runtime (RTX Titan)
#   --stack-size 65536          — nix spec overrides the 32768 default
#   -I <stb source>             — stb_image.h lives there
#   -D STBI_ASSERT(x)=          — disable stb_image's internal assert() (coqui mode-only;
#                                 nix builds rely on a runtime __assert_fail stub
#                                 that coqui mode doesn't ship)
# The harness .c defines STB_IMAGE_IMPLEMENTATION + STBI_NO_STDIO internally,
# so no extra -D is needed (unlike bzip2/cjson).
"${COQUI_CC}" \
  -arch "${ARCH}" \
  --stack-size "${STACK_SIZE}" \
  --slab-pool-size "${SLAB_POOL_SIZE}" \
  -I "${STB_SRC}" \
  -D "STBI_ASSERT(x)=" \
  -D STBI_NO_HDR \
  "${HARNESS}" \
  -o "${HARNESS_BASENAME}"

if [[ ! -f "${HARNESS_BASENAME}.cubin" || ! -f "${HARNESS_BASENAME}.conf" ]]; then
  echo "ERROR: coqui-cc did not emit ${HARNESS_BASENAME}.cubin / .conf" >&2
  exit 1
fi

echo "=== [4/4] Link CPU binary + seeds + dict ==="
ln -sfn "${CPU_OUT_LINK}/${HARNESS_BASENAME}" "${HARNESS_BASENAME}_cpu"
ln -sfn "${CPU_OUT_LINK}/seeds" seeds
if [[ -d "${CPU_OUT_LINK}/dict" ]]; then
  ln -sfn "${CPU_OUT_LINK}/dict" dict
fi

echo
echo "=== Build complete ==="
ls -la "${HARNESS_BASENAME}.cubin" "${HARNESS_BASENAME}.conf" \
       "${HARNESS_BASENAME}_cpu" seeds 2>/dev/null || true
[[ -L dict ]] && ls -la dict || true
echo "  conf:"; sed 's/^/    /' "${HARNESS_BASENAME}.conf"
