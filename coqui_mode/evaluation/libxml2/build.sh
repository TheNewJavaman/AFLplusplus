#!/usr/bin/env bash
# build.sh — build libxml2 coqui mode evaluation target (nix-free).
#
# Produces in the current directory:
#   libxml2_xml_read_fuzzer.cubin — GPU kernel (configurable via ARCH env, default sm_75)
#   libxml2_xml_read_fuzzer.conf  — companion config emitted by coqui-cc
#   libxml2_xml_read_fuzzer_cpu   — AFL++ instrumented CPU binary
#   .build/<owner>-<repo>-<sha>/  — cached libxml2 source checkout
#   .libxml2_fuzzer.build/        — generated headers + patched sources
#
# No external dependencies beyond afl-clang-fast (from this repo) and
# coqui-cc (installed to /usr/local/bin). libxml2 is fetched from
# github.com/GNOME/libxml2 at a pinned commit (tag v2.13.4).
#
# Overrides:
#   ARCH            GPU compute capability (default sm_75)
#   AFL_CC          path to afl-clang-fast (default: this repo's own afl-clang-fast)
#   COQUI_CC        path to coqui-cc (default /usr/local/bin/coqui-cc)
#   COQUI_LLC_OPT   llc opt level for the coqui-cc cubin build (default -O1;
#                   libxml2 has ~211k instrumented accesses and llc default
#                   -O2 stalls in register allocation)

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# --- Config -----------------------------------------------------------------
HARNESS_BASENAME="libxml2_xml_read_fuzzer"
HARNESS_SRC="${SCRIPT_DIR}/harness.c"
STUBS_SRC="${SCRIPT_DIR}/libxml2_stubs.c"
ARCH="${ARCH:-sm_75}"
STACK_SIZE=32768             # libxml2.nix sets --stack-size 32768
SLAB_POOL_SIZE=2147483648    # 2 GiB — from libxml2.nix

# libxml2 upstream pin — matches nix/cpu-target-specs.nix (v2.13.4).
XML2_OWNER="GNOME"
XML2_REPO="libxml2"
XML2_REV="v2.13.4"
XML2_SHORT="v2.13.4"
XML2_VERSION="2.13.4"
XML2_URL="https://github.com/${XML2_OWNER}/${XML2_REPO}/archive/refs/tags/${XML2_REV}.tar.gz"

# Tools
AFL_CC="${AFL_CC:-${SCRIPT_DIR}/../../../afl-clang-fast}"
COQUI_CC="${COQUI_CC:-/usr/local/bin/coqui-cc}"
COQUI_LLC_OPT="${COQUI_LLC_OPT:--O1}"

# Cache dirs
FETCH_CACHE_ROOT="${SCRIPT_DIR}/.build"
XML2_SRC="${FETCH_CACHE_ROOT}/${XML2_OWNER}-${XML2_REPO}-${XML2_SHORT}"
BUILD_DIR="${SCRIPT_DIR}/.libxml2_fuzzer.build"

# Sanitizer flags — matches sanitizers.default (libxml2 uses full set per
# cpu-target-specs.nix entry libxml2.sanitizerFlags = sanitizers.default).
SANITIZE_FLAGS=(
  "-fsanitize=address,array-bounds,bool,builtin,enum,integer-divide-by-zero,null,object-size,return,returns-nonnull-attribute,shift,signed-integer-overflow,unsigned-integer-overflow,unreachable,vla-bound"
  "-fno-sanitize-recover=array-bounds,bool,builtin,enum,integer-divide-by-zero,null,object-size,return,returns-nonnull-attribute,shift,signed-integer-overflow,unreachable,vla-bound"
  "-fno-stack-protector"
)

# --- Pre-flight -------------------------------------------------------------
[[ -x "$AFL_CC"    ]]  || { echo "ERROR: afl-clang-fast not found at $AFL_CC" >&2; exit 1; }
[[ -x "$COQUI_CC"  ]]  || { echo "ERROR: coqui-cc not found at $COQUI_CC" >&2; exit 1; }
[[ -f "$HARNESS_SRC" ]] || { echo "ERROR: harness missing at $HARNESS_SRC" >&2; exit 1; }
[[ -f "$STUBS_SRC"   ]] || { echo "ERROR: stubs missing at $STUBS_SRC" >&2; exit 1; }

# --- [1/4] Fetch libxml2 source at pinned tag -------------------------------
echo "=== [1/4] Fetch libxml2 ${XML2_REV} ==="
mkdir -p "${FETCH_CACHE_ROOT}"
# Cache marker: parser.c is the canonical "source is complete" file.
if [[ ! -f "${XML2_SRC}/parser.c" ]]; then
  echo "  curl ${XML2_URL} -> ${XML2_SRC}"
  rm -rf "${XML2_SRC}"
  mkdir -p "${XML2_SRC}"
  curl -fsSL "${XML2_URL}" | tar -xz --strip-components=1 -C "${XML2_SRC}"
  [[ -f "${XML2_SRC}/parser.c" ]] || {
    echo "ERROR: libxml2 tarball did not contain parser.c" >&2
    exit 1
  }
fi
echo "  libxml2 source: ${XML2_SRC}"

# --- [2/4] Generate config headers + patch sources --------------------------
echo "=== [2/4] Generate config headers + patch sources ==="
# Nuke and recreate build dir for idempotency — cheap relative to coqui-cc.
rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}/include/libxml"

# Copy the libxml2 include tree so we can drop generated xmlversion.h beside it.
cp -r "${XML2_SRC}/include/libxml/." "${BUILD_DIR}/include/libxml/"

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
  -e "s/@VERSION@/${XML2_VERSION}/" \
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
# xmlLastError state (no GPU synchronization) or reach vsnprintf through
# the error-reporting call chain and overflow the CUDA hardware call stack.
# See libxml2.nix buildPhase comments for details.
cp "${XML2_SRC}/error.c" "${BUILD_DIR}/error.c"
awk '
  # xmlCopyError: no-op (prevent copying to lastError)
  /^xmlCopyError\(.*\{/ { print "xmlCopyError(const xmlError *from, xmlErrorPtr to) {"; print "    (void)from; (void)to; return(0);"; skip_depth=1; next }

  # xmlFormatError: no-op (prevent dereferencing lastError pointers)
  /^xmlFormatError\(/ { in_fmt=1 }
  in_fmt && /^\{/ { print "{"; print "    (void)err; (void)channel; (void)data;"; in_fmt=0; skip_depth=1; next }

  # xmlResetLastError: no-op (prevent freeing lastError strings)
  /^xmlResetLastError\(/ { in_reset_last=1 }
  in_reset_last && /^\{/ { print "{"; in_reset_last=0; skip_depth=1; next }

  # xmlRaiseMemoryError: no-op (prevent writing to lastError)
  /^xmlRaiseMemoryError\(/ { in_raise_mem=1 }
  in_raise_mem && /^\{/ { print "{"; print "    (void)schannel; (void)channel; (void)data; (void)domain; (void)error;"; in_raise_mem=0; skip_depth=1; next }

  # Depth-aware skip: track brace nesting to find the real closing brace.
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

# --- [3/4] Build AFL++ CPU binary -------------------------------------------
echo "=== [3/4] Build AFL++ CPU binary with afl-clang-fast ==="
# Matches cpu-target-specs.nix libxml2 entry: full default sanitizers,
# HAVE_CONFIG_H, 22-file explicit source list (core libxml2 + harness +
# stubs). Note the CPU-side config.h here has LIBXML_THREAD_ENABLED=1
# (the GPU side disables threads; the nix CPU spec sets it on). But
# given our unified build dir, the GPU-safe config is already generated
# above; the CPU binary runs single-threaded per LLVMFuzzerTestOneInput
# invocation so having threads disabled is also fine. We re-use the
# same config.h and xmlversion.h from BUILD_DIR.
CPU_OUT="${HARNESS_BASENAME}_cpu"
"$AFL_CC" -O2 -g \
  "${SANITIZE_FLAGS[@]}" \
  -fsanitize=fuzzer \
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
  "${XML2_SRC}/error.c" \
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
  "${XML2_SRC}/xmlstring.c" \
  "${XML2_SRC}/xpath.c" \
  "${STUBS_SRC}" \
  "${HARNESS_SRC}" \
  -o "$CPU_OUT"

[[ -x "$CPU_OUT" ]] || { echo "ERROR: CPU binary not produced" >&2; exit 1; }

# --- [4/4] Build GPU cubin via coqui-cc -------------------------------------
echo "=== [4/4] Build GPU cubin via coqui-cc (flock'd) ==="
# Flags / includes / source list mirror the `libxml2` target in the legacy
# coqui codebase. Build dir comes first in the -I list so
# build/include/libxml/xmlversion.h wins over the template in the source
# tree. flock serializes with other parallel target builds — ptxas is
# memory-heavy (~20 GB RSS at -O1 on this module, 211k instrumented
# accesses) and concurrent invocations OOM the box.
COQUI_LLC_OPT="${COQUI_LLC_OPT}" \
flock /tmp/coqui-cc.lock \
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
  "${STUBS_SRC}" \
  "${HARNESS_SRC}" \
  -o "${HARNESS_BASENAME}"

[[ -f "${HARNESS_BASENAME}.cubin" && -f "${HARNESS_BASENAME}.conf" ]] \
  || { echo "ERROR: coqui-cc did not emit .cubin/.conf" >&2; exit 1; }

echo
echo "=== Build complete ==="
ls -la "${HARNESS_BASENAME}.cubin" "${HARNESS_BASENAME}.conf" "${CPU_OUT}" seeds
echo "  conf:"; sed 's/^/    /' "${HARNESS_BASENAME}.conf"
