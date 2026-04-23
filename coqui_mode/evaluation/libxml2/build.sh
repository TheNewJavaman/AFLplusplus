#!/usr/bin/env bash
# build.sh — build libxml2 coqui mode evaluation target.
#
# Produces in the current directory:
#   libxml2_xml_read_fuzzer.cubin — GPU kernel (sm_75)
#   libxml2_xml_read_fuzzer.conf  — companion config emitted by coqui-cc
#   libxml2_xml_read_fuzzer_cpu   — AFL++ instrumented CPU binary (symlink)
#   seeds/                        — seed corpus symlink into nix store
#   dict/                         — dictionary symlink into nix store
#
# Mirrors coqui's nix spec: the `libxml2` target in the legacy coqui codebase
#
# libxml2 needs generated config headers (config.h, xmlversion.h) and two
# source patches (error.c, xmlstring.c) before compilation. The nix spec
# does this in its buildPhase; we reproduce that inline in a .build/
# directory before calling coqui-cc.
#
# Note: the nix spec passes `--heap-size 131072` and `--batch-size 65536`
# to coqui; coqui mode's coqui-cc does NOT accept those flags. The coqui mode
# runtime derives heap at startup and reads batch size from
# AFL_COQUI_BATCH_SIZE.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

COQUI_REPO="${COQUI_REPO:?set COQUI_REPO to your legacy coqui checkout path}"
COQUI_CC="/usr/local/bin/coqui-cc"
CPU_OUT_LINK="/tmp/coqui-libxml2-cpu"
HARNESS_DIR="${COQUI_REPO}/harness/targets"
ARCH="sm_75"
SLAB_POOL_SIZE=2147483648   # 2 GiB — from libxml2.nix
STACK_SIZE=32768            # libxml2.nix sets --stack-size 32768

BUILD_DIR="${SCRIPT_DIR}/.libxml2_fuzzer.build"

echo "=== [1/5] Build AFL++ CPU binary via nix ==="
# Idempotent: `nix build` is a no-op if the derivation is already realised.
cd "${COQUI_REPO}"
nix build '.#target-libxml2-aflplusplus' --out-link "${CPU_OUT_LINK}"
cd "${SCRIPT_DIR}"

if [[ ! -x "${CPU_OUT_LINK}/libxml2_xml_read_fuzzer" ]]; then
  echo "ERROR: CPU binary not found at ${CPU_OUT_LINK}/libxml2_xml_read_fuzzer" >&2
  exit 1
fi

echo "=== [2/5] Resolve libxml2 source path ==="
# The libxml2 source tarball is the only `-source` derivation in the closure
# of the CPU build (a GitHub fetch, not a plain tarball).
XML2_SRC=$(nix-store -qR "${CPU_OUT_LINK}" | grep -E -- '-source$' | head -n1)
if [[ -z "${XML2_SRC}" || ! -f "${XML2_SRC}/parser.c" ]]; then
  echo "ERROR: could not resolve libxml2 source (expected parser.c in ${XML2_SRC:-<empty>})" >&2
  exit 1
fi
echo "  libxml2 source: ${XML2_SRC}"

echo "=== [3/5] Generate config headers + patch sources ==="
# Nuke and recreate build dir for idempotency.
rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}/include/libxml"

# Copy the libxml2 include tree so we can drop generated xmlversion.h beside it.
cp -r "${XML2_SRC}/include/libxml/." "${BUILD_DIR}/include/libxml/"
# Strip read-only bits inherited from /nix/store so sed/cp can write here.
chmod -R u+w "${BUILD_DIR}"

# Minimal config.h — matches the libxml2.nix buildPhase verbatim.
# Threading disabled; POSIX headers declared so open/read/write/close/dup
# are properly declared (LibcTransform redirects them at IR level).
cat > "${BUILD_DIR}/config.h" <<'HEADER'
#define VERSION "2.13.4"
#define HAVE_STDINT_H 1
#define HAVE_STDLIB_H 1
#define HAVE_STRING_H 1
#define HAVE_ERRNO_H 1
#define HAVE_CTYPE_H 1
#define HAVE_STDARG_H 1
#define HAVE_FCNTL_H 1
#define HAVE_UNISTD_H 1
#define HAVE_SYS_STAT_H 1
HEADER

# Generate xmlversion.h from the template — WITH_THREADS=0, WITH_OUTPUT=1,
# WITH_PUSH=1, WITH_SAX1=1, WITH_XPATH=1, WITH_ISO8859X=1, WITH_TREE=1.
# All other optional features off (matches libxml2.nix).
sed \
  -e 's/@VERSION@/2.13.4/' \
  -e 's/@LIBXML_VERSION_NUMBER@/21304/' \
  -e 's/@LIBXML_VERSION_EXTRA@//' \
  -e 's/@WITH_THREADS@/0/' \
  -e 's/@WITH_THREAD_ALLOC@/0/' \
  -e 's/@WITH_OUTPUT@/1/' \
  -e 's/@WITH_PUSH@/1/' \
  -e 's/@WITH_READER@/0/' \
  -e 's/@WITH_PATTERN@/0/' \
  -e 's/@WITH_WRITER@/0/' \
  -e 's/@WITH_SAX1@/1/' \
  -e 's/@WITH_HTTP@/0/' \
  -e 's/@WITH_VALID@/0/' \
  -e 's/@WITH_HTML@/0/' \
  -e 's/@WITH_C14N@/0/' \
  -e 's/@WITH_CATALOG@/0/' \
  -e 's/@WITH_XPATH@/1/' \
  -e 's/@WITH_XPTR@/0/' \
  -e 's/@WITH_XINCLUDE@/0/' \
  -e 's/@WITH_ICONV@/0/' \
  -e 's/@WITH_ICU@/0/' \
  -e 's/@WITH_ISO8859X@/1/' \
  -e 's/@WITH_DEBUG@/0/' \
  -e 's/@WITH_REGEXPS@/0/' \
  -e 's/@WITH_RELAXNG@/0/' \
  -e 's/@WITH_FTP@/0/' \
  -e 's/@WITH_LEGACY@/0/' \
  -e 's/@WITH_XPTR_LOCS@/0/' \
  -e 's/@WITH_LZMA@/0/' \
  -e 's/@WITH_TREE@/1/' \
  -e 's/@WITH_SCHEMAS@/0/' \
  -e 's/@WITH_SCHEMATRON@/0/' \
  -e 's/@WITH_MODULES@/0/' \
  -e 's/@WITH_ZLIB@/0/' \
  "${XML2_SRC}/include/libxml/xmlversion.h.in" \
  > "${BUILD_DIR}/include/libxml/xmlversion.h"

# Patch error.c — stub xmlCopyError / xmlFormatError / xmlResetLastError /
# xmlRaiseMemoryError to no-ops. These functions either race on shared
# xmlLastError state (no GPU synchronization) or reach vsnprintf through the
# error-reporting call chain and overflow the CUDA hardware call stack.
# See libxml2.nix buildPhase comments for details.
cp "${XML2_SRC}/error.c" "${BUILD_DIR}/error.c"
awk '
  /^xmlCopyError\(.*\{/ { print "xmlCopyError(const xmlError *from, xmlErrorPtr to) {"; print "    (void)from; (void)to; return(0);"; skip_depth=1; next }

  /^xmlFormatError\(/ { in_fmt=1 }
  in_fmt && /^\{/ { print "{"; print "    (void)err; (void)channel; (void)data;"; in_fmt=0; skip_depth=1; next }

  /^xmlResetLastError\(/ { in_reset_last=1 }
  in_reset_last && /^\{/ { print "{"; in_reset_last=0; skip_depth=1; next }

  /^xmlRaiseMemoryError\(/ { in_raise_mem=1 }
  in_raise_mem && /^\{/ { print "{"; print "    (void)schannel; (void)channel; (void)data; (void)domain; (void)error;"; in_raise_mem=0; skip_depth=1; next }

  skip_depth > 0 {
    n = split($0, a, "{") - 1
    m = split($0, b, "}") - 1
    skip_depth += n - m
    if (skip_depth <= 0) { print "}"; skip_depth=0 }
    next
  }
  { print }
' "${BUILD_DIR}/error.c" > "${BUILD_DIR}/error_patched.c"
mv "${BUILD_DIR}/error_patched.c" "${BUILD_DIR}/error.c"

# Patch xmlstring.c — stub xmlStrVASPrintf to return -1 without calling
# __coqui_vsnprintf. The variadic vsnprintf allocates 512+ bytes on the
# hardware call stack and overflows it from the deep parser call chain.
# Callers handle -1 gracefully (fall through with NULL message).
cp "${XML2_SRC}/xmlstring.c" "${BUILD_DIR}/xmlstring.c"
awk '
  /^xmlStrVASPrintf\(.*\{/ {
    print $0
    print "    if (out != NULL) *out = NULL;"
    print "    (void)maxSize; (void)msg; (void)ap;"
    print "    return(-1);"
    skip_depth=1
    next
  }
  skip_depth > 0 {
    n = split($0, a, "{") - 1
    m = split($0, b, "}") - 1
    skip_depth += n - m
    if (skip_depth <= 0) { print "}"; skip_depth=0 }
    next
  }
  { print }
' "${BUILD_DIR}/xmlstring.c" > "${BUILD_DIR}/xmlstring_patched.c"
mv "${BUILD_DIR}/xmlstring_patched.c" "${BUILD_DIR}/xmlstring.c"

echo "=== [4/5] Build GPU cubin via coqui-cc ==="
# Flags / includes / source list mirror the `libxml2` target in the legacy coqui codebase.
# Build dir comes first in the -I list so build/include/libxml/xmlversion.h
# wins over the template in the source tree.
"${COQUI_CC}" \
  -arch "${ARCH}" \
  --stack-size "${STACK_SIZE}" \
  --slab-pool-size "${SLAB_POOL_SIZE}" \
  -I "${BUILD_DIR}/include" \
  -I "${BUILD_DIR}" \
  -I "${XML2_SRC}" \
  -I "${XML2_SRC}/include" \
  -D HAVE_CONFIG_H \
  "${XML2_SRC}/buf.c" \
  "${XML2_SRC}/chvalid.c" \
  "${XML2_SRC}/dict.c" \
  "${XML2_SRC}/encoding.c" \
  "${XML2_SRC}/entities.c" \
  "${BUILD_DIR}/error.c" \
  "${XML2_SRC}/globals.c" \
  "${XML2_SRC}/hash.c" \
  "${XML2_SRC}/list.c" \
  "${XML2_SRC}/parser.c" \
  "${XML2_SRC}/parserInternals.c" \
  "${XML2_SRC}/SAX2.c" \
  "${XML2_SRC}/threads.c" \
  "${XML2_SRC}/tree.c" \
  "${XML2_SRC}/uri.c" \
  "${XML2_SRC}/valid.c" \
  "${XML2_SRC}/xmlIO.c" \
  "${XML2_SRC}/xmlmemory.c" \
  "${XML2_SRC}/xmlsave.c" \
  "${BUILD_DIR}/xmlstring.c" \
  "${XML2_SRC}/xpath.c" \
  "${HARNESS_DIR}/libxml2_stubs.c" \
  "${HARNESS_DIR}/libxml2_xml_read_fuzzer.c" \
  -o libxml2_xml_read_fuzzer

if [[ ! -f libxml2_xml_read_fuzzer.cubin || ! -f libxml2_xml_read_fuzzer.conf ]]; then
  echo "ERROR: coqui-cc did not emit libxml2_xml_read_fuzzer.cubin / .conf" >&2
  exit 1
fi

echo "=== [5/5] Link CPU binary + seeds + dict ==="
ln -sfn "${CPU_OUT_LINK}/libxml2_xml_read_fuzzer" libxml2_xml_read_fuzzer_cpu
ln -sfn "${CPU_OUT_LINK}/seeds" seeds
ln -sfn "${CPU_OUT_LINK}/dict"  dict

echo
echo "=== Build complete ==="
ls -la libxml2_xml_read_fuzzer.cubin libxml2_xml_read_fuzzer.conf \
       libxml2_xml_read_fuzzer_cpu seeds dict
