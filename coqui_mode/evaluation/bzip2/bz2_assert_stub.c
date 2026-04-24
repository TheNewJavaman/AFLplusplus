/*
 * bz2_assert_stub.c — stub for BZ_NO_STDIO mode.
 *
 * When BZ_NO_STDIO is defined, bzlib expects the host to provide
 * bz_internal_error() instead of using the default fprintf-based handler.
 *
 * GPU (coqui-cc) build: coqui mode's coqui-cc ExternalSymbolGatekeeper
 * rejects `abort` as an unresolved external — every FATAL path must route
 * through the __coqui_* runtime. So on __NVPTX__ we call __coqui_trap()
 * (PTX `trap; exit;`) directly, same unrecoverable semantics but a symbol
 * the gatekeeper allows.
 *
 * CPU (afl-clang-fast) build: use libc abort() — the upstream behaviour.
 */

#ifdef __NVPTX__
extern void __coqui_trap(void);
#else
#include <stdlib.h>  /* abort() */
#endif

void bz_internal_error(int errcode) {
    (void)errcode;
#ifdef __NVPTX__
    __coqui_trap();
#else
    abort();
#endif
}
