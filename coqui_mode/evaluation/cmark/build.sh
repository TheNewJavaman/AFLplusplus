#!/usr/bin/env bash
# Build the coqui mode cmark fuzz target.
#
# Produces, in this directory:
#   cmark_fuzzer.cubin   - GPU kernel built by coqui mode's coqui-cc
#   cmark_fuzzer.conf    - sidecar written by coqui-cc (stack/slab/arch)
#   cmark_fuzzer_cpu     - AFL++-instrumented CPU binary (symlink into nix store)
#   seeds/               - initial corpus (symlink into nix store)
#   dict/                - fuzzer dictionary (symlink into nix store)
#
# This script is idempotent.  Reference: the `cmark` target in the legacy coqui codebase.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

COQUI_REPO="${COQUI_REPO:?set COQUI_REPO to your legacy coqui checkout path}"
COQUI_CC="/usr/local/bin/coqui-cc"
HARNESS="cmark_fuzzer"
ARCH="sm_75"
STACK_SIZE=32768      # coqui-cc default; cmark's nix spec doesn't override
SLAB_POOL_SIZE=0      # cmark's nix spec doesn't set --slab-pool-size

CPU_LINK="/tmp/coqui-cmark-cpu"

echo "=== [1/4] Building CPU (AFL++) target via nix ==="
# Produces $CPU_LINK -> /nix/store/...-aflplusplus-cmark-0.31.1
# containing: cmark_fuzzer, cmark_fuzzer.cmplog, seeds/, dict/
pushd "$COQUI_REPO" >/dev/null
nix build '.#target-cmark-aflplusplus' --out-link "$CPU_LINK"
popd >/dev/null

echo "=== [2/4] Resolving cmark source + harness ==="
# Fetch the pinned cmark tarball the same way the nix spec does.  Identical
# hash => identical /nix/store path across invocations.
CMARK_SRC="$(nix-build --no-out-link -E '
  with import <nixpkgs> {};
  fetchFromGitHub {
    owner = "commonmark";
    repo = "cmark";
    rev = "0.31.1";
    hash = "sha256-+JLw7zCjjozjq1RhRQGFqHj/MTUTq3t7A0V3T2U2PQk=";
  }
')"
echo "cmark source: $CMARK_SRC"

HARNESS_SRC="$COQUI_REPO/harness/targets/cmark_fuzzer.c"
if [ ! -f "$HARNESS_SRC" ]; then
  echo "ERROR: harness not found at $HARNESS_SRC" >&2
  exit 1
fi
echo "harness: $HARNESS_SRC"

echo "=== [3/4] Generating cmark config headers ==="
# Replicates preBuild from the nix spec.
mkdir -p "$SCRIPT_DIR/generated"
cat > "$SCRIPT_DIR/generated/config.h" << 'HEOF'
#define HAVE_STDBOOL_H 1
#define HAVE___BUILTIN_EXPECT 1
#define HAVE___ATTRIBUTE__ 1
HEOF
cat > "$SCRIPT_DIR/generated/cmark_export.h" << 'HEOF'
#ifndef CMARK_EXPORT_H
#define CMARK_EXPORT_H
#define CMARK_EXPORT
#define CMARK_NO_EXPORT
#endif
HEOF
cat > "$SCRIPT_DIR/generated/cmark_version.h" << 'HEOF'
#ifndef CMARK_VERSION_H
#define CMARK_VERSION_H
#define CMARK_VERSION ((0 << 16) | (31 << 8) | 1)
#define CMARK_VERSION_STRING "0.31.1"
#endif
HEOF

echo "=== [4/4] Compiling $HARNESS.cubin with coqui-cc ($ARCH) ==="
# Source list + includes + defines mirror the coqui nix target spec.
"$COQUI_CC" \
  -arch "$ARCH" \
  --stack-size "$STACK_SIZE" \
  --slab-pool-size "$SLAB_POOL_SIZE" \
  -I "$CMARK_SRC/src" \
  -I "$CMARK_SRC/src/include" \
  -I "$SCRIPT_DIR/generated" \
  -D CMARK_STATIC_DEFINE \
  -D NDEBUG \
  "$CMARK_SRC/src/blocks.c" \
  "$CMARK_SRC/src/buffer.c" \
  "$CMARK_SRC/src/cmark.c" \
  "$CMARK_SRC/src/cmark_ctype.c" \
  "$CMARK_SRC/src/commonmark.c" \
  "$CMARK_SRC/src/houdini_href_e.c" \
  "$CMARK_SRC/src/houdini_html_e.c" \
  "$CMARK_SRC/src/houdini_html_u.c" \
  "$CMARK_SRC/src/html.c" \
  "$CMARK_SRC/src/inlines.c" \
  "$CMARK_SRC/src/iterator.c" \
  "$CMARK_SRC/src/latex.c" \
  "$CMARK_SRC/src/man.c" \
  "$CMARK_SRC/src/node.c" \
  "$CMARK_SRC/src/references.c" \
  "$CMARK_SRC/src/render.c" \
  "$CMARK_SRC/src/scanners.c" \
  "$CMARK_SRC/src/utf8.c" \
  "$CMARK_SRC/src/xml.c" \
  "$HARNESS_SRC" \
  -o "$SCRIPT_DIR/$HARNESS"

echo "=== Wiring CPU binary + seeds + dict into $SCRIPT_DIR ==="
ln -sfn "$CPU_LINK/cmark_fuzzer"       "$SCRIPT_DIR/${HARNESS}_cpu"
ln -sfn "$CPU_LINK/seeds"              "$SCRIPT_DIR/seeds"
ln -sfn "$CPU_LINK/dict"               "$SCRIPT_DIR/dict"

echo
echo "=== Artifacts ==="
ls -la "$SCRIPT_DIR/${HARNESS}.cubin" "$SCRIPT_DIR/${HARNESS}.conf" \
       "$SCRIPT_DIR/${HARNESS}_cpu" "$SCRIPT_DIR/seeds" "$SCRIPT_DIR/dict"
echo
echo "=== Build OK.  Run ./fuzz.sh to launch coqui mode. ==="
