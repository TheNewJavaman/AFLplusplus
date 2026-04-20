#!/usr/bin/env bash
# coqui_mode/build_coqui_support.sh --- build the LLVM pass plugin, compile
# the device runtime to runtime.bc, and install coqui-cc.
#
# Requires: LLVM 18 (clang, llvm-link, opt, llc), CUDA toolkit 13+ (ptxas).

set -euo pipefail

COQUI_DIR="$(cd "$(dirname "$0")" && pwd)"
PREFIX="${PREFIX:-/usr/local}"

echo "[*] Building coqui_mode LLVM pass plugin..."
mkdir -p "$COQUI_DIR/passes/build"
cd "$COQUI_DIR/passes/build"
cmake ..
make -j"$(nproc)"
cd "$COQUI_DIR"

echo "[*] Compiling device runtime to bitcode..."
RUNTIME_FLAGS="--target=nvptx64-nvidia-cuda -O2 -ffreestanding -emit-llvm -c -I $COQUI_DIR/runtime"
mkdir -p "$COQUI_DIR/runtime/build"
for f in coqui_runtime coqui_coverage coqui_sancov coqui_memory coqui_asan coqui_libc; do
    clang $RUNTIME_FLAGS "$COQUI_DIR/runtime/$f.c" -o "$COQUI_DIR/runtime/build/$f.bc"
done

echo "[*] Linking runtime bitcode..."
llvm-link "$COQUI_DIR/runtime/build/coqui_runtime.bc" \
          "$COQUI_DIR/runtime/build/coqui_coverage.bc" \
          "$COQUI_DIR/runtime/build/coqui_sancov.bc" \
          "$COQUI_DIR/runtime/build/coqui_memory.bc" \
          "$COQUI_DIR/runtime/build/coqui_asan.bc" \
          "$COQUI_DIR/runtime/build/coqui_libc.bc" \
          -o "$COQUI_DIR/runtime/build/runtime.bc"

echo "[*] Install..."
install -m 755 "$COQUI_DIR/bin/coqui-cc" "$PREFIX/bin/coqui-cc"
install -d "$PREFIX/lib/coqui-cc"
install -m 644 "$COQUI_DIR/runtime/build/runtime.bc" "$PREFIX/lib/coqui-cc/runtime.bc"
install -m 644 "$COQUI_DIR/passes/build/libCoquiPassPlugin.so" "$PREFIX/lib/coqui-cc/CoquiPassPlugin.so"

echo "[+] coqui-cc installed to $PREFIX/bin/coqui-cc"
echo "[+] Runtime: $PREFIX/lib/coqui-cc/runtime.bc"
echo "[+] Pass plugin: $PREFIX/lib/coqui-cc/CoquiPassPlugin.so"
