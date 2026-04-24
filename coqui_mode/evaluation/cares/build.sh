#!/usr/bin/env bash
# build.sh - build c-ares coqui mode evaluation target (nix-free).
#
# Produces in the current directory:
#   cares_parse_reply_fuzzer.cubin  - GPU kernel (configurable via ARCH env, default sm_75)
#   cares_parse_reply_fuzzer.conf   - companion config emitted by coqui-cc
#   cares_parse_reply_fuzzer_cpu    - AFL++ instrumented CPU binary
#   generated/                      - generated config headers (ares_build.h, ares_config.h)
#   .build/                         - cached upstream c-ares source
#
# No external dependencies beyond afl-clang-fast (from this repo) and
# coqui-cc (installed to /usr/local/bin). c-ares sources are fetched
# from github.com/c-ares/c-ares at a pinned tag (v1.34.4).
#
# Overrides:
#   ARCH             GPU compute capability (default sm_75)
#   AFL_CLANG_FAST   path to afl-clang-fast
#                    (default: this repo's own afl-clang-fast)
#   COQUI_CC         path to coqui-cc (default /usr/local/bin/coqui-cc)
#   CARES_CACHE      path to cache the cloned c-ares source
#                    (default .build/c-ares-c-ares-<tag>)
#
# NOTE: don't use AFL_CC as the override name - afl-cc reserves that env
# var to override its underlying clang, and passing afl-clang-fast as
# AFL_CC causes afl-cc to invoke itself recursively until it exhausts
# MAX_PARAMS_NUM.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# --- Config -----------------------------------------------------------------
HARNESS_BASENAME="cares_parse_reply_fuzzer"
HARNESS_SRC="${SCRIPT_DIR}/harness.c"
STUBS_SRC="${SCRIPT_DIR}/cares_stubs.c"
ARCH="${ARCH:-sm_75}"
SLAB_POOL_SIZE=10737418240        # 10 GiB - DNS name decompression can blow up heap

# c-ares upstream pin (matches legacy coqui fetch at v1.34.4)
CARES_TAG="v1.34.4"
CARES_REPO="https://github.com/c-ares/c-ares.git"

# Tools
AFL_CLANG_FAST="${AFL_CLANG_FAST:-${SCRIPT_DIR}/../../../afl-clang-fast}"
COQUI_CC="${COQUI_CC:-/usr/local/bin/coqui-cc}"
CARES_CACHE="${CARES_CACHE:-${SCRIPT_DIR}/.build/c-ares-c-ares-${CARES_TAG}}"

# afl-cc reads AFL_CC from the env to pick its backing clang; if it's set
# to afl-clang-fast itself (a common mistake when callers reuse the name),
# afl-cc recursively re-execs itself.  Always unset before invoking.
unset AFL_CC AFL_CXX
# Silence afl-cc's "Mistyped AFL environment variable: AFL_CLANG_FAST"
# warning - our override name is intentional (see note above).
export AFL_IGNORE_UNKNOWN_ENVS=1

# Sanitizers matching legacy coqui (sanitizers.default: address + full UBSan).
SANITIZE_FLAGS=(
  "-fsanitize=address,array-bounds,bool,builtin,enum,integer-divide-by-zero,null,object-size,return,returns-nonnull-attribute,shift,signed-integer-overflow,unsigned-integer-overflow,unreachable,vla-bound"
  "-fno-sanitize-recover=array-bounds,bool,builtin,enum,integer-divide-by-zero,null,object-size,return,returns-nonnull-attribute,shift,signed-integer-overflow,unreachable,vla-bound"
)

# --- Pre-flight -------------------------------------------------------------
[[ -x "$AFL_CLANG_FAST" ]] || { echo "ERROR: afl-clang-fast not found at $AFL_CLANG_FAST" >&2; exit 1; }
[[ -x "$COQUI_CC" ]] || { echo "ERROR: coqui-cc not found at $COQUI_CC" >&2; exit 1; }
[[ -f "$HARNESS_SRC" ]] || { echo "ERROR: harness missing at $HARNESS_SRC" >&2; exit 1; }
[[ -f "$STUBS_SRC" ]]   || { echo "ERROR: stubs missing at $STUBS_SRC" >&2; exit 1; }

# --- [1/4] Fetch upstream c-ares at pinned tag ------------------------------
echo "=== [1/4] Fetch c-ares at pinned tag ${CARES_TAG} ==="
mkdir -p "$(dirname "$CARES_CACHE")"
# Marker check: presence of include/ares.h implies a successful prior clone.
if [[ ! -f "$CARES_CACHE/include/ares.h" ]]; then
  echo "  clone $CARES_REPO -> $CARES_CACHE"
  rm -rf "$CARES_CACHE"
  git clone --quiet --depth 1 --branch "$CARES_TAG" "$CARES_REPO" "$CARES_CACHE"
fi
[[ -f "$CARES_CACHE/include/ares.h" ]] \
  || { echo "ERROR: c-ares fetch produced no include/ares.h in $CARES_CACHE" >&2; exit 1; }
[[ -f "$CARES_CACHE/src/lib/record/ares_dns_parse.c" ]] \
  || { echo "ERROR: c-ares fetch missing src/lib/record/ares_dns_parse.c" >&2; exit 1; }
echo "  c-ares: $(du -sh "$CARES_CACHE" | cut -f1)"

# --- [2/4] Generate config headers -----------------------------------------
# c-ares normally produces these via autotools/cmake. Generate by hand to
# avoid running configure. Content mirrors legacy coqui nix spec exactly.
echo "=== [2/4] Generate config headers ==="
mkdir -p generated

cat > generated/ares_build.h << 'HEOF'
#ifndef __CARES_BUILD_H
#define __CARES_BUILD_H
#include <sys/types.h>
#include <sys/socket.h>
#define CARES_TYPEOF_ARES_SOCKLEN_T socklen_t
#define CARES_TYPEOF_ARES_SSIZE_T ssize_t
#endif
HEOF

cat > generated/ares_config.h << 'HEOF'
#ifndef ARES_CONFIG_H
#define ARES_CONFIG_H
/* Headers */
#define HAVE_ARPA_INET_H 1
#define HAVE_ARPA_NAMESER_H 1
#define HAVE_ARPA_NAMESER_COMPAT_H 1
#define HAVE_ASSERT_H 1
#define HAVE_ERRNO_H 1
#define HAVE_FCNTL_H 1
#define HAVE_LIMITS_H 1
#define HAVE_NETDB_H 1
#define HAVE_NETINET_IN_H 1
#define HAVE_NETINET_TCP_H 1
#define HAVE_NET_IF_H 1
#define HAVE_POLL_H 1
#define HAVE_SIGNAL_H 1
#define HAVE_STDBOOL_H 1
#define HAVE_STDINT_H 1
#define HAVE_STDLIB_H 1
#define HAVE_STRING_H 1
#define HAVE_STRINGS_H 1
#define HAVE_SYS_IOCTL_H 1
#define HAVE_SYS_PARAM_H 1
/* HAVE_SYS_RANDOM_H intentionally omitted -- disables getrandom(),
 * falls back to rand() which the Coqui runtime provides. */
#define HAVE_SYS_SELECT_H 1
#define HAVE_SYS_SOCKET_H 1
#define HAVE_SYS_STAT_H 1
#define HAVE_SYS_TIME_H 1
#define HAVE_SYS_TYPES_H 1
#define HAVE_SYS_UIO_H 1
#define HAVE_TIME_H 1
#define HAVE_UNISTD_H 1
/* Functions */
#define HAVE_GETTIMEOFDAY 1
#define HAVE_CONNECT 1
#define HAVE_FCNTL 1
#define HAVE_GETENV 1
#define HAVE_GETHOSTNAME 1
#define HAVE_GETNAMEINFO 1
#define HAVE_IF_INDEXTONAME 1
#define HAVE_INET_NET_PTON 1
#define HAVE_INET_NTOP 1
#define HAVE_INET_PTON 1
#define HAVE_IOCTL 1
#define HAVE_POLL 1
#define HAVE_RECV 1
#define HAVE_RECVFROM 1
#define HAVE_SEND 1
#define HAVE_SOCKET 1
#define HAVE_STAT 1
#define HAVE_STRCASECMP 1
#define HAVE_STRDUP 1
#define HAVE_STRNCASECMP 1
#define HAVE_WRITEV 1
#define HAVE_FCNTL_O_NONBLOCK 1
/* Types */
#define HAVE_BOOL_T 1
#define HAVE_STRUCT_IN6_ADDR 1
#define HAVE_STRUCT_SOCKADDR_IN6 1
#define HAVE_STRUCT_SOCKADDR_IN6_SIN6_SCOPE_ID 1
#define HAVE_STRUCT_SOCKADDR_STORAGE 1
#define HAVE_STRUCT_TIMEVAL 1
#define HAVE_AF_INET6 1
#define HAVE_PF_INET6 1
#define HAVE_STRUCT_ADDRINFO 1
/* Socket signatures */
#define RECV_TYPE_ARG1 int
#define RECV_TYPE_ARG2 void *
#define RECV_TYPE_ARG3 size_t
#define RECV_TYPE_ARG4 int
#define RECV_TYPE_RETV ssize_t
#define RECVFROM_TYPE_ARG1 int
#define RECVFROM_TYPE_ARG2 void *
#define RECVFROM_TYPE_ARG3 size_t
#define RECVFROM_TYPE_ARG4 int
#define RECVFROM_TYPE_ARG5 struct sockaddr *
#define RECVFROM_TYPE_ARG6 socklen_t *
#define RECVFROM_TYPE_RETV ssize_t
#define SEND_TYPE_ARG1 int
#define SEND_TYPE_ARG2 const void *
#define SEND_TYPE_ARG3 size_t
#define SEND_TYPE_ARG4 int
#define SEND_TYPE_RETV ssize_t
#endif
HEOF

# --- Source whitelist -------------------------------------------------------
# Exact same whitelist as the legacy nix spec: DNS parsing + dependencies,
# no networking, no resolver, no event loop.
S="${CARES_CACHE}/src/lib"
CARES_SOURCES=(
  "$S/record/ares_dns_parse.c"
  "$S/record/ares_dns_record.c"
  "$S/record/ares_dns_write.c"
  "$S/record/ares_dns_mapping.c"
  "$S/record/ares_dns_name.c"
  "$S/record/ares_dns_multistring.c"
  "$S/str/ares_buf.c"
  "$S/str/ares_str.c"
  "$S/str/ares_strsplit.c"
  "$S/dsa/ares_array.c"
  "$S/dsa/ares_htable.c"
  "$S/dsa/ares_htable_asvp.c"
  "$S/dsa/ares_htable_dict.c"
  "$S/dsa/ares_htable_strvp.c"
  "$S/dsa/ares_htable_szvp.c"
  "$S/dsa/ares_htable_vpstr.c"
  "$S/dsa/ares_htable_vpvp.c"
  "$S/dsa/ares_llist.c"
  "$S/dsa/ares_slist.c"
  "$S/ares_library_init.c"
  "$S/ares_free_string.c"
  "$S/ares_free_hostent.c"
  "$S/ares_data.c"
  "$S/ares_strerror.c"
  "$S/ares_version.c"
  "$S/inet_ntop.c"
  "$S/inet_net_pton.c"
  "$S/ares_getenv.c"
  "$S/util/ares_math.c"
  "$S/util/ares_rand.c"
  "$S/util/ares_timeval.c"
  "$S/util/ares_threads.c"
  "$S/util/ares_uri.c"
)

# Verify each source exists (catches upstream layout changes at the pinned tag).
for src in "${CARES_SOURCES[@]}"; do
  [[ -f "$src" ]] || { echo "ERROR: missing c-ares source $src" >&2; exit 1; }
done

# --- [3/4] Build AFL++ CPU binary -------------------------------------------
# CPU build matches cpu-target-specs.nix cares spec: -DCOQUI_CPU tells the
# stubs file to skip strcasecmp/gettimeofday (libc provides them).
#
# Compile each .c -> .o separately: afl-clang-fast caps total CLI parameter
# count at MAX_PARAMS_NUM (2048 after internal flag rewrites), and feeding
# 34 sources in a single invocation exceeds that limit.
echo "=== [3/4] Build AFL++ CPU binary with afl-clang-fast ==="
CPU_OUT="${HARNESS_BASENAME}_cpu"
OBJ_DIR="${SCRIPT_DIR}/.build/cpu-obj"
rm -rf "$OBJ_DIR"
mkdir -p "$OBJ_DIR"

CPU_CFLAGS=(
  -O2 -g
  "${SANITIZE_FLAGS[@]}"
  -I "${CARES_CACHE}/include"
  -I "${CARES_CACHE}/src/lib"
  -I "${CARES_CACHE}/src/lib/include"
  -I generated
  -D HAVE_CONFIG_H
  -D COQUI_CPU
)

CPU_OBJS=()
for src in "${CARES_SOURCES[@]}" "$STUBS_SRC" "$HARNESS_SRC"; do
  base="$(basename "$src")"
  obj="$OBJ_DIR/${base%.c}.o"
  # Use a path-unique name for library files that share base names across dirs
  # (ares_htable_*.c in dsa/ are unique, but be defensive).
  if [[ -f "$obj" ]]; then
    # Two .c files ended up with the same basename; dedupe by parent dir tag.
    parent="$(basename "$(dirname "$src")")"
    obj="$OBJ_DIR/${parent}-${base%.c}.o"
  fi
  "$AFL_CLANG_FAST" "${CPU_CFLAGS[@]}" -c "$src" -o "$obj"
  CPU_OBJS+=("$obj")
done

# Link with -fsanitize=fuzzer so afl-clang-fast pulls in libAFLDriver.a.
"$AFL_CLANG_FAST" -O2 -g \
  "${SANITIZE_FLAGS[@]}" \
  -fsanitize=fuzzer \
  "${CPU_OBJS[@]}" \
  -o "$CPU_OUT" \
  -lm

[[ -x "$CPU_OUT" ]] || { echo "ERROR: CPU binary not produced" >&2; exit 1; }

# --- [4/4] Build GPU cubin via coqui-cc -------------------------------------
# flock serializes coqui-cc with other parallel build.sh invocations; ptxas
# at -O1 is memory-heavy (tens of GB), and concurrent builds will OOM.
echo "=== [4/4] Build GPU cubin via coqui-cc (flock-serialized) ==="
flock /tmp/coqui-cc.lock "$COQUI_CC" \
  -arch "$ARCH" \
  --slab-pool-size "$SLAB_POOL_SIZE" \
  -I "${CARES_CACHE}/include" \
  -I "${CARES_CACHE}/src/lib" \
  -I "${CARES_CACHE}/src/lib/include" \
  -I generated \
  -D HAVE_CONFIG_H \
  "${CARES_SOURCES[@]}" \
  "$STUBS_SRC" \
  "$HARNESS_SRC" \
  -o "$HARNESS_BASENAME"

[[ -f "${HARNESS_BASENAME}.cubin" && -f "${HARNESS_BASENAME}.conf" ]] \
  || { echo "ERROR: coqui-cc did not emit .cubin/.conf" >&2; exit 1; }

echo
echo "=== Build complete ==="
ls -la "${HARNESS_BASENAME}.cubin" "${HARNESS_BASENAME}.conf" "${CPU_OUT}" seeds
echo "  conf:"; sed 's/^/    /' "${HARNESS_BASENAME}.conf"
