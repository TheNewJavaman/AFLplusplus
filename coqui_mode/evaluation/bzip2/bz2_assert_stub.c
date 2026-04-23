/*
 * bz2_assert_stub.c — coqui mode-compatible stub for BZ_NO_STDIO mode.
 *
 * When BZ_NO_STDIO is defined, bzlib expects the host to provide
 * bz_internal_error() instead of using the default fprintf-based handler.
 *
 * coqui's upstream stub calls abort(), but coqui mode's coqui-cc
 * ExternalSymbolGatekeeper rejects `abort` as an unresolved external:
 * every FATAL path must route through the __coqui_* runtime. So we
 * call __coqui_trap() (PTX `trap; exit;`) directly — same unrecoverable
 * semantics, but a symbol the gatekeeper allows.
 */

extern void __coqui_trap(void);

void bz_internal_error(int errcode) {
    (void)errcode;
    __coqui_trap();
}
