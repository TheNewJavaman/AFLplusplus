#!/usr/bin/env bash
# build.sh — build bzip2 coqui mode evaluation target.
#
# Produces in the current directory:
#   bzip2_fuzzer.cubin — GPU kernel (sm_75)
#   bzip2_fuzzer.conf  — companion config emitted by coqui-cc
#   bzip2_fuzzer_cpu   — AFL++ instrumented CPU binary (symlink to nix store)
#   seeds/             — initial seed corpus (symlink to nix store)
#   dict/              — fuzzing dictionary (symlink to nix store)
#
# Mirrors coqui's nix spec: /home/gpizarro/coqui/nix/targets/bzip2.nix
#
# Note: the nix spec passes `--heap-size 524288` and `--batch-size 32768` to
# coqui; coqui mode's coqui-cc does NOT accept those flags. The coqui mode runtime
# derives heap at startup and reads batch size from AFL_COQUI_BATCH_SIZE.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

COQUI_REPO="/home/gpizarro/coqui"
COQUI_CC="/usr/local/bin/coqui-cc"
CPU_OUT_LINK="/tmp/coqui-bzip2-cpu"
HARNESS_DIR="${COQUI_REPO}/harness/targets"
ARCH="sm_75"
SLAB_POOL_SIZE=2147483648   # 2 GiB — from bzip2.nix
STACK_SIZE=32768            # coqui-cc default (bzip2.nix does not override)

echo "=== [1/4] Build AFL++ CPU binary via nix ==="
# Idempotent: `nix build` is a no-op if the derivation is already realised.
cd "${COQUI_REPO}"
nix build '.#target-bzip2-aflplusplus' --out-link "${CPU_OUT_LINK}"
cd "${SCRIPT_DIR}"

if [[ ! -x "${CPU_OUT_LINK}/bzip2_decompress_fuzzer" ]]; then
  echo "ERROR: CPU binary not found at ${CPU_OUT_LINK}/bzip2_decompress_fuzzer" >&2
  exit 1
fi

echo "=== [2/4] Resolve libbzip2 source path ==="
# The libarchive bzip2 tarball is the only `-source` derivation in the closure
# of the CPU build. Avoids hardcoding a /nix/store hash.
BZ2_SRC=$(nix-store -qR "${CPU_OUT_LINK}" | grep -E -- '-source$' | head -n1)
if [[ -z "${BZ2_SRC}" || ! -f "${BZ2_SRC}/blocksort.c" ]]; then
  echo "ERROR: could not resolve libbzip2 source (expected blocksort.c in ${BZ2_SRC:-<empty>})" >&2
  exit 1
fi
echo "  libbzip2 source: ${BZ2_SRC}"

echo "=== [3/4] Build GPU cubin via coqui-cc ==="
# Flags mirror /home/gpizarro/coqui/nix/targets/bzip2.nix:
#   -D BZ_NO_STDIO           — disable bzlib's stdio reliance
#   -I <libbzip2 source>     — bzlib.h + bzlib_private.h
#   --slab-pool-size 2 GiB   — bzip2 DState + allocations exceed 64KB heap
# Source list matches the nix spec, except we use a local bz2_assert_stub.c
# that routes bz_internal_error() to __coqui_trap() instead of libc abort()
# (ExternalSymbolGatekeeper rejects `abort` — only __coqui_*/__llvm_* pass).
"${COQUI_CC}" \
  -arch "${ARCH}" \
  --stack-size "${STACK_SIZE}" \
  --slab-pool-size "${SLAB_POOL_SIZE}" \
  -D BZ_NO_STDIO \
  -I "${BZ2_SRC}" \
  "${BZ2_SRC}/blocksort.c" \
  "${BZ2_SRC}/huffman.c" \
  "${BZ2_SRC}/crctable.c" \
  "${BZ2_SRC}/randtable.c" \
  "${BZ2_SRC}/compress.c" \
  "${BZ2_SRC}/decompress.c" \
  "${BZ2_SRC}/bzlib.c" \
  "${SCRIPT_DIR}/bz2_assert_stub.c" \
  "${HARNESS_DIR}/bzip2_decompress_target.c" \
  -o bzip2_fuzzer

if [[ ! -f bzip2_fuzzer.cubin || ! -f bzip2_fuzzer.conf ]]; then
  echo "ERROR: coqui-cc did not emit bzip2_fuzzer.cubin / bzip2_fuzzer.conf" >&2
  exit 1
fi

echo "=== [4/4] Link CPU binary + seeds + dict ==="
ln -sfn "${CPU_OUT_LINK}/bzip2_decompress_fuzzer" bzip2_fuzzer_cpu

# Seeds: AFL reads its input dir with lstat() + S_ISREG, which fails for
# symlinks. The nix `seeds/` dir itself is a real dir, but each seed inside
# it is a symlink into /nix/store (`rf1s3acb...-bzip2-seeds`). Copy the
# resolved files into a local seeds/ directory so AFL's S_ISREG check passes.
rm -rf seeds
mkdir -p seeds
cp -L "${CPU_OUT_LINK}/seeds/"*.seed seeds/
chmod u+rw seeds seeds/*.seed

# Dict is a single file, fine as a symlink directory.
ln -sfn "${CPU_OUT_LINK}/dict" dict

echo
echo "=== Build complete ==="
ls -ld bzip2_fuzzer.cubin bzip2_fuzzer.conf bzip2_fuzzer_cpu seeds dict
echo "  seeds/ contains $(ls seeds | wc -l) files"
