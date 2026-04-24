/*
 * coqui_runtime.c --- core device-side utilities.
 *
 * Uses clang NVPTX builtins where available:
 *   - __nvvm_read_ptx_sreg_{tid,ntid,ctaid}_x for thread-identity registers
 *   - __builtin_trap for unrecoverable trap (emits PTX trap; exit;)
 *
 * Falls back to inline PTX asm only for `exit;` (single-thread clean
 * termination; no clang builtin exists). This is the sole piece of
 * inline asm in our runtime; rejection passes exempt __coqui_* functions.
 */

#include "coqui_runtime.h"

/* Pointer to per-thread status array, set by kernel entry (FuzzEntry pass). */
__attribute__((visibility("default")))
__attribute__((used))
coqui_status_t *__coqui_status_array;

/* Per-thread statics pool base pointer. Written by the host launcher
 * (afl-fuzz-coqui.c) via cuModuleGetGlobal/cuMemcpyHtoD when
 * StaticGlobals pooled any writable globals; stays NULL otherwise.
 *
 * Defined here so the symbol exists unconditionally — coqui_asan.c reads
 * it from asan_check_global() to resolve per-thread pool-kind descriptor
 * addresses, and that reference must link even for targets where
 * StaticGlobals found nothing to pool. The StaticGlobals pass looks up
 * this symbol with getOrInsert and reuses this definition instead of
 * creating a duplicate. */
__attribute__((visibility("default")))
__attribute__((used))
u8 *__coqui_global_statics_pool_base;

/* Thread identity via clang NVPTX intrinsic builtins (lower to mov.u32 %r, %tid.x etc). */
u32 __coqui_fuzz_tid(void) {
    return (u32)__nvvm_read_ptx_sreg_ctaid_x()
         * (u32)__nvvm_read_ptx_sreg_ntid_x()
         + (u32)__nvvm_read_ptx_sreg_tid_x();
}

/* Unrecoverable abort — PTX trap; exit; */
void __coqui_trap(void) {
    __builtin_trap();
}

/* Clean per-thread exit — PTX exit;. No clang builtin; inline asm. */
void __coqui_exit(void) {
    __asm__ volatile("exit;");
}

/* Weak fallback for __coqui_slab_release_thread so targets built without
 * the slab runtime linked still resolve this symbol. When coqui_slab.c
 * is linked in, its strong definition wins; otherwise this no-op is used. */
__attribute__((weak))
void __coqui_slab_release_thread(void) {
    /* no slab runtime linked — nothing to release */
}

/* Signalled-exit variant. Stamps the trap reason so the host can
 * distinguish OOM / stack overflow / other categorized traps from a
 * generic crash. Uses __coqui_exit() (not __coqui_trap) so the kernel
 * itself doesn't abort — peer threads keep running. Releases any slab
 * allocations this thread holds so the reclaimed ranges become
 * available to other threads. */
void __coqui_trap_with_reason(u8 reason) {
    __coqui_slab_release_thread();
    u32 tid = __coqui_fuzz_tid();
    if (__coqui_status_array) {
        __coqui_status_array[tid].trap_reason = reason;
    }
    /* sys-wide fence so the status write is visible to the host when the
     * thread's writes retire, then clean per-thread exit. */
    __asm__ volatile("membar.sys;");
    __coqui_exit();
    __builtin_unreachable();
}

/* Status writer */
void __coqui_status_set_phase(u32 tid, u8 phase) {
    __coqui_status_array[tid].phase = phase;
}

/* Benign stubs for rarely-used POSIX hooks */
int atexit(void (*fn)(void))                 { (void)fn; return 0; }
void __cxa_atexit(void (*fn)(void *), void *arg, void *dso)
                                              { (void)fn; (void)arg; (void)dso; }
void __cxa_finalize(void *dso)               { (void)dso; }

/* pthread_once — single-threaded per GPU thread */
int pthread_once(int *once_control, void (*init)(void)) {
    if (*once_control == 0) {
        init();
        *once_control = 1;
    }
    return 0;
}
