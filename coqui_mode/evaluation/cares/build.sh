#!/usr/bin/env bash
#
# Build the c-ares (DNS parser) fuzz target for coqui mode.
#
# Produces in this directory:
#   cares_parse_reply_fuzzer.cubin  - GPU kernel (for coqui mode coqui_mode)
#   cares_parse_reply_fuzzer.conf   - runtime config (emitted alongside cubin)
#   cares_parse_reply_fuzzer_cpu    - AFL++-instrumented CPU binary (forkserver target)
#   seeds/                          - 74 DNS wire-format seed corpus (symlink to nix store)
#   dict/dns.dict                   - DNS wire dictionary (symlink to nix store)
#
# Sources and flags mirror the `cares` target in the legacy coqui codebase.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

HARNESS="cares_parse_reply_fuzzer"

# --- Prerequisites -----------------------------------------------------------

COQUI_CC="${COQUI_CC:-/usr/local/bin/coqui-cc}"
COQUI_REPO="${COQUI_REPO:?set COQUI_REPO to your legacy coqui checkout path}"
CPU_OUT_LINK="${CPU_OUT_LINK:-/tmp/coqui-cares-cpu}"

command -v "$COQUI_CC" >/dev/null || {
  echo "error: coqui-cc not found at $COQUI_CC" >&2
  exit 1
}
[ -d "$COQUI_REPO" ] || {
  echo "error: coqui repo not found at $COQUI_REPO" >&2
  exit 1
}

# --- Step 1: Build AFL++ CPU binary via nix ----------------------------------
# The CPU fuzzer variant is used by coqui mode as the forkserver target (it provides
# the classified exit path for crash verification). We consume the nix-built
# artifact directly so we inherit oss-fuzz-equivalent build flags + sanitizers.

echo "[cares/build] nix building target-cares-aflplusplus..."
(
  cd "$COQUI_REPO"
  nix build '.#target-cares-aflplusplus' --out-link "$CPU_OUT_LINK"
)

[ -x "$CPU_OUT_LINK/$HARNESS" ] || {
  echo "error: nix build did not produce $CPU_OUT_LINK/$HARNESS" >&2
  exit 1
}

# --- Step 2: Locate c-ares source + harness in the nix store -----------------
# The CPU derivation pulls the source via fetchFromGitHub; walk the closure to
# find its unpacked path so we can feed identical files to coqui-cc.

echo "[cares/build] resolving c-ares source in nix store..."
CPU_STORE="$(readlink -f "$CPU_OUT_LINK")"
CLOSURE="$(nix-store -q --references "$CPU_STORE")"

CARES_SRC=""
HARNESS_SRC=""
for dep in $CLOSURE; do
  if [ -z "$CARES_SRC" ] && [ -f "$dep/include/ares.h" ]; then
    CARES_SRC="$dep"
  fi
  if [ -z "$HARNESS_SRC" ] && [ -f "$dep/cares_parse_reply_fuzzer.c" ]; then
    HARNESS_SRC="$dep"
  fi
done

[ -n "$CARES_SRC" ]   || { echo "error: could not locate c-ares source in nix store closure" >&2; exit 1; }
[ -n "$HARNESS_SRC" ] || { echo "error: could not locate harness-targets in nix store closure" >&2; exit 1; }

echo "[cares/build]   c-ares source: $CARES_SRC"
echo "[cares/build]   harness src:   $HARNESS_SRC"

# --- Step 3: Generate config headers (replicates nix preBuild) ---------------
# c-ares normally produces these via autotools/cmake. The Coqui target writes
# them by hand because we build without running configure.

echo "[cares/build] generating config headers..."
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

# --- Step 4: Invoke coqui-cc to produce the GPU .cubin -----------------------
# Source whitelist + include paths + defines mirror the nix spec exactly. The
# DNS parsing code is self-contained: no networking, no resolver, no event loop.

echo "[cares/build] compiling $HARNESS.cubin with coqui-cc..."
S="$CARES_SRC/src/lib"

"$COQUI_CC" \
  -arch sm_75 \
  --slab-pool-size 10737418240 \
  -I "$CARES_SRC/include" \
  -I "$CARES_SRC/src/lib" \
  -I "$CARES_SRC/src/lib/include" \
  -I generated \
  -D HAVE_CONFIG_H \
  "$S/record/ares_dns_parse.c" \
  "$S/record/ares_dns_record.c" \
  "$S/record/ares_dns_write.c" \
  "$S/record/ares_dns_mapping.c" \
  "$S/record/ares_dns_name.c" \
  "$S/record/ares_dns_multistring.c" \
  "$S/str/ares_buf.c" \
  "$S/str/ares_str.c" \
  "$S/str/ares_strsplit.c" \
  "$S/dsa/ares_array.c" \
  "$S/dsa/ares_htable.c" \
  "$S/dsa/ares_htable_asvp.c" \
  "$S/dsa/ares_htable_dict.c" \
  "$S/dsa/ares_htable_strvp.c" \
  "$S/dsa/ares_htable_szvp.c" \
  "$S/dsa/ares_htable_vpstr.c" \
  "$S/dsa/ares_htable_vpvp.c" \
  "$S/dsa/ares_llist.c" \
  "$S/dsa/ares_slist.c" \
  "$S/ares_library_init.c" \
  "$S/ares_free_string.c" \
  "$S/ares_free_hostent.c" \
  "$S/ares_data.c" \
  "$S/ares_strerror.c" \
  "$S/ares_version.c" \
  "$S/inet_ntop.c" \
  "$S/inet_net_pton.c" \
  "$S/ares_getenv.c" \
  "$S/util/ares_math.c" \
  "$S/util/ares_rand.c" \
  "$S/util/ares_timeval.c" \
  "$S/util/ares_threads.c" \
  "$S/util/ares_uri.c" \
  "$HARNESS_SRC/cares_stubs.c" \
  "$HARNESS_SRC/cares_parse_reply_fuzzer.c" \
  -o "$HARNESS"

[ -f "$HARNESS.cubin" ] || { echo "error: coqui-cc did not produce $HARNESS.cubin" >&2; exit 1; }
[ -f "$HARNESS.conf" ]  || echo "warning: $HARNESS.conf was not emitted (coqui mode may use defaults)" >&2

# --- Step 5: Stage the CPU binary, seeds, and dict ---------------------------

echo "[cares/build] staging CPU binary + seeds + dict..."
ln -sfn "$CPU_OUT_LINK/$HARNESS" "./${HARNESS}_cpu"

# Seeds: the nix derivation already ships a seeds/ dir as symlinks into the
# c-ares source tree. Point at that directory wholesale.
ln -sfn "$CPU_OUT_LINK/seeds" ./seeds

# Dict: same treatment.
ln -sfn "$CPU_OUT_LINK/dict" ./dict

echo "[cares/build] done."
echo ""
echo "  cubin:   $HERE/$HARNESS.cubin"
echo "  conf:    $HERE/$HARNESS.conf"
echo "  cpu:     $HERE/${HARNESS}_cpu -> $CPU_OUT_LINK/$HARNESS"
echo "  seeds:   $HERE/seeds ($(ls -1 "$CPU_OUT_LINK/seeds" | wc -l) files)"
echo "  dict:    $HERE/dict"
echo ""
echo "Run: ./fuzz.sh [-- -V 60 ...]"
