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
