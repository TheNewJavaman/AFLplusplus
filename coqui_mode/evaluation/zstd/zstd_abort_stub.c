/* zstd_abort_stub.c — device-side `abort` implementation for the zstd harness.
 *
 * The zstd harness (harness/targets/zstd_simple_decompress_fuzzer.c) calls
 * `abort()` when a decompressed frame's content-size metadata does not match
 * the actual decompressed size. That indicates a genuine bug in the
 * decompressor, so the kernel should treat it as an unrecoverable crash.
 *
 * cuAFL's `coqui-cc` driver does NOT run the upstream coqui LibcTransform
 * pass that rewrites `abort` → `__coqui_abort`, and the bundled runtime.bc
 * provides only `__coqui_trap` (PTX `trap;`) and `__coqui_exit` (PTX `exit;`).
 * Without a user-supplied definition, the CoquiPassPlugin's
 * ExternalSymbolGatekeeper fails with:
 *   LLVM ERROR: unresolved external 'abort' — port the replacement transform
 *     or runtime stub. Used by: __coqui_fuzz_execute
 *
 * Mirrors harness/targets/bz2_assert_stub.c (where `bz_internal_error` calls
 * `abort`): a tiny local stub compiled into the cubin alongside the harness.
 *
 * `__coqui_trap` emits a PTX `trap;` which cuAFL classifies as a signal-11
 * crash — the desired behaviour for a decompression-bug abort.
 */

extern void __coqui_trap(void);

void abort(void) {
    __coqui_trap();
    /* Unreachable: __coqui_trap does not return. The while(1) silences any
       `noreturn`-analysis warnings in the LLVM pipeline. */
    while (1) {}
}
