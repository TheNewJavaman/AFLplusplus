/* png_abort_stub.c — coqui mode-compatible stub for libpng's abort() calls.
 *
 * libpng routes its unrecoverable-error path through PNG_ABORT(), which
 * expands to abort() by default (pngpriv.h:589). With -D PNG_NO_SETJMP,
 * every png_error() eventually hits PNG_ABORT().
 *
 * coqui's nix target passes --ignore-signal=abort to its own coqui-cc
 * pass plugin, which accepts `abort` as a known-allowed external symbol.
 * coqui mode's coqui-cc does NOT support --ignore-signal=; its
 * ExternalSymbolGatekeeper fatally rejects any external whose name does
 * not start with __coqui_* / __llvm_* / llvm.*.
 *
 * By providing our own definition of abort() that calls __coqui_trap()
 * (PTX `trap; exit;`), the linked bitcode has no unresolved `abort`
 * symbol and the gatekeeper passes. Same unrecoverable-termination
 * semantics — just a coqui mode-native trap.
 *
 * Same pattern as coqui_mode/evaluation/bzip2/bz2_assert_stub.c.
 */

extern void __coqui_trap(void);

/* Override libc's abort(). The coqui-cc NVPTX build has no libc-abort in
 * scope, so this definition is the only one llvm-link sees. */
__attribute__((noreturn))
void abort(void) {
    __coqui_trap();
    __builtin_unreachable();
}
