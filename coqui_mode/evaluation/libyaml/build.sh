#!/usr/bin/env bash
# build.sh — build libyaml coqui mode evaluation target.
#
# Produces in the current directory:
#   libyaml_parser_fuzzer.cubin — GPU kernel (sm_75)
#   libyaml_parser_fuzzer.conf  — companion config emitted by coqui-cc
#   libyaml_parser_fuzzer_cpu   — AFL++ instrumented CPU binary (symlink to nix store)
#   seeds/                      — initial seed corpus (symlink to nix store)
#   dict/                       — fuzzing dictionary (symlink to nix store)
#   libyaml_src/                — patched libyaml sources (needed at compile time)
#
# Mirrors coqui's nix spec: /home/gpizarro/coqui/nix/targets/libyaml.nix
#
# Notes:
#   - libyaml.nix patches yaml_private.h to shrink INPUT_RAW_BUFFER_SIZE 16384 -> 512
#     and the INITIAL_{STACK,QUEUE,STRING}_SIZE defaults 16 -> 4 so the parser fits
#     in the per-thread 64 KB heap. We replicate that sed here.
#   - libyaml.nix does NOT set --stack-size or --slab-pool-size, so we use coqui-cc
#     defaults (32768 stack; 0 slab pool).
#   - libyaml's api.c/scanner.c use assert() (from <assert.h>), which expands to
#     __assert_fail on NVPTX. coqui's compiler rewrites that via its LibcTransform
#     pass, but coqui mode's coqui-cc driver only runs the coqui-link + always-inline
#     passes (LibcTransform is not ported), so the ExternalSymbolGatekeeper
#     rejects __assert_fail. We pass -D NDEBUG to make <assert.h> expand assert()
#     to a no-op at preprocess time — matches the stb_image target's
#     `-D STBI_ASSERT(x)=` workaround. GPU asserts become non-fatal;
#     the CPU AFL++ binary still honors them for host-side verification.
#   - Same story for strdup/memcpy/memmove/memset: coqui mode's coqui-cc runtime.bc
#     exports only __coqui_malloc/calloc/free/realloc/memcmp/strlen/strcmp/
#     strncmp/strchr/strtod/trap — no strdup, no memcpy, no memset. libyaml
#     needs all four, so we link a local libyaml_stubs.c that defines them
#     (via __builtin_memcpy for memcpy/memset, manual loop for memmove, and
#     __coqui_malloc + __coqui_strlen + loop for strdup).
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

COQUI_REPO="/home/gpizarro/coqui"
COQUI_CC="/usr/local/bin/coqui-cc"
CPU_OUT_LINK="/tmp/coqui-libyaml-cpu"
HARNESS_DIR="${COQUI_REPO}/harness/targets"
ARCH="sm_75"

echo "=== [1/5] Build AFL++ CPU binary via nix ==="
# Idempotent: `nix build` is a no-op if the derivation is already realised.
cd "${COQUI_REPO}"
nix build '.#target-libyaml-aflplusplus' --out-link "${CPU_OUT_LINK}"
cd "${SCRIPT_DIR}"

if [[ ! -x "${CPU_OUT_LINK}/libyaml_parser_fuzzer" ]]; then
  echo "ERROR: CPU binary not found at ${CPU_OUT_LINK}/libyaml_parser_fuzzer" >&2
  exit 1
fi

echo "=== [2/5] Resolve libyaml source path ==="
# The libyaml fetchFromGitHub tarball is the only `-source` derivation
# in the CPU build closure that has src/yaml_private.h + include/yaml.h.
YAML_SRC=""
while IFS= read -r p; do
  if [[ -f "${p}/src/yaml_private.h" && -f "${p}/include/yaml.h" ]]; then
    YAML_SRC="${p}"
    break
  fi
done < <(nix-store -qR "${CPU_OUT_LINK}" | grep -E -- '-source$')

if [[ -z "${YAML_SRC}" ]]; then
  echo "ERROR: could not resolve libyaml source in CPU build closure" >&2
  exit 1
fi
echo "  libyaml source: ${YAML_SRC}"

echo "=== [3/5] Patch libyaml sources (yaml_private.h buffer shrink) ==="
# Patch yaml_private.h to shrink I/O buffers for the per-thread 64 KB GPU heap.
#   INPUT_RAW_BUFFER_SIZE: 16384 -> 512  (INPUT_BUFFER_SIZE follows as 3x)
#   INITIAL_STACK_SIZE:        16 -> 4
#   INITIAL_QUEUE_SIZE:        16 -> 4
#   INITIAL_STRING_SIZE:       16 -> 4
# Matches /home/gpizarro/coqui/nix/targets/libyaml.nix buildPhase exactly.
rm -rf libyaml_src
mkdir -p libyaml_src
for f in api.c reader.c scanner.c parser.c loader.c yaml_private.h; do
  cp "${YAML_SRC}/src/${f}" "libyaml_src/${f}"
done
chmod +w libyaml_src/yaml_private.h
sed -i \
  -e 's/^#define INPUT_RAW_BUFFER_SIZE.*/#define INPUT_RAW_BUFFER_SIZE   512/' \
  -e 's/^#define INITIAL_STACK_SIZE.*/#define INITIAL_STACK_SIZE  4/' \
  -e 's/^#define INITIAL_QUEUE_SIZE.*/#define INITIAL_QUEUE_SIZE  4/' \
  -e 's/^#define INITIAL_STRING_SIZE.*/#define INITIAL_STRING_SIZE 4/' \
  libyaml_src/yaml_private.h

echo "=== [4/5] Build GPU cubin via coqui-cc ==="
# Flags mirror /home/gpizarro/coqui/nix/targets/libyaml.nix buildPhase:
#   -I ${src}/include           — yaml.h
#   -I libyaml_src              — yaml_private.h + api.c friends
#   -D YAML_DECLARE_STATIC      — internal linkage
#   -D YAML_VERSION_*           — version macros (normally provided by configure)
# Source list matches the nix spec exactly (parser-only; no dumper/emitter/writer).
# Sanitizer flags from nix/sanitizer-flags.nix:default are NOT replicated here —
# coqui-cc (coqui mode build) does not accept -fsanitize; it instruments on its own.
"${COQUI_CC}" \
  -arch "${ARCH}" \
  -I "${YAML_SRC}/include" \
  -I libyaml_src \
  -D YAML_DECLARE_STATIC \
  -D 'YAML_VERSION_STRING="0.2.5"' \
  -D YAML_VERSION_MAJOR=0 \
  -D YAML_VERSION_MINOR=2 \
  -D YAML_VERSION_PATCH=5 \
  -D NDEBUG \
  libyaml_src/api.c \
  libyaml_src/reader.c \
  libyaml_src/scanner.c \
  libyaml_src/parser.c \
  libyaml_src/loader.c \
  "${HARNESS_DIR}/libyaml_parser_fuzzer.c" \
  "${SCRIPT_DIR}/libyaml_stubs.c" \
  -o libyaml_parser_fuzzer

if [[ ! -f libyaml_parser_fuzzer.cubin || ! -f libyaml_parser_fuzzer.conf ]]; then
  echo "ERROR: coqui-cc did not emit libyaml_parser_fuzzer.cubin / libyaml_parser_fuzzer.conf" >&2
  exit 1
fi

echo "=== [5/5] Link CPU binary + seeds + dict ==="
ln -sfn "${CPU_OUT_LINK}/libyaml_parser_fuzzer" libyaml_parser_fuzzer_cpu
ln -sfn "${CPU_OUT_LINK}/seeds" seeds
ln -sfn "${CPU_OUT_LINK}/dict"  dict

echo
echo "=== Build complete ==="
ls -la libyaml_parser_fuzzer.cubin libyaml_parser_fuzzer.conf libyaml_parser_fuzzer_cpu seeds dict
