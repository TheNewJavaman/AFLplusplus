/*
 * coqui_runtime.c --- core device-side utilities.
 *
 * - Thread identity via PTX inline asm
 * - Trap / exit wrappers
 * - Status struct writers
 * - Safe stub implementations of trivial POSIX functions that pass
 *   the LibcReject gatekeeper (atexit, pthread_once)
 */

#include "coqui_runtime.h"

/* Pointer to per-thread status array, set by kernel entry (FuzzEntry pass).
   Declared __device__ so it lives in global memory. */
__attribute__((visibility("default")))
__attribute__((used))
coqui_status_t *__coqui_status_array;

/* PTX primitives */
u32 __coqui_fuzz_tid(void) {
    u32 tid, bdim, bid;
    __asm__ volatile("mov.u32 %0, %%tid.x;"   : "=r"(tid));
    __asm__ volatile("mov.u32 %0, %%ntid.x;"  : "=r"(bdim));
    __asm__ volatile("mov.u32 %0, %%ctaid.x;" : "=r"(bid));
    return bid * bdim + tid;
}

void __coqui_trap(void) {
    __asm__ volatile("trap;");
}

void __coqui_exit(void) {
    __asm__ volatile("exit;");
}

/* Status writers — thread i writes to __coqui_status_array[i] */
void __coqui_status_set_phase(u32 tid, u8 phase) {
    __coqui_status_array[tid].phase = phase;
}

/* Benign stubs for rarely-used POSIX hooks — lets the gatekeeper pass
   targets that declare but don't meaningfully use these. */
int atexit(void (*fn)(void))                 { (void)fn; return 0; }
void __cxa_atexit(void (*fn)(void *), void *arg, void *dso)
                                              { (void)fn; (void)arg; (void)dso; }
void __cxa_finalize(void *dso)               { (void)dso; }

/* pthread_once — single-threaded on GPU, each thread's "once" runs once per
   thread lifetime (kernel invocation). */
int pthread_once(int *once_control, void (*init)(void)) {
    if (*once_control == 0) {
        init();
        *once_control = 1;
    }
    return 0;
}
