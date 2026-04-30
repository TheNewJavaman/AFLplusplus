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

/* Thread identity via clang NVPTX intrinsic builtins (lower to mov.u32 %r, %tid.x etc).
 *
 * `const`: the PTX %tid/%ntid/%ctaid registers are invariant for a given
 * thread, so the result is a function of (implicit) thread state alone —
 * LLVM may CSE / hoist calls across loops. `always_inline`: hot — every
 * tid lookup folds to three mov.u32 + a mul-add. `nothrow`: never throws. */
__attribute__((const, always_inline, nothrow))
u32 __coqui_fuzz_tid(void) {
    return (u32)__nvvm_read_ptx_sreg_ctaid_x()
         * (u32)__nvvm_read_ptx_sreg_ntid_x()
         + (u32)__nvvm_read_ptx_sreg_tid_x();
}

/* Unrecoverable abort — PTX trap; exit;.
 * `cold`: only reached on fatal paths; keep it out of the icache hot
 * stream. `nothrow`: never throws (consistent with all C runtime). */
__attribute__((cold, nothrow))
void __coqui_trap(void) {
    __builtin_trap();
}

/* Clean per-thread exit — PTX exit;. No clang builtin; inline asm.
 * `cold`: only fatal/exit paths reach here. `nothrow`. */
__attribute__((cold, nothrow))
void __coqui_exit(void) {
    __asm__ volatile("exit;");
}

/* Weak fallback for __coqui_slab_release_thread so targets built without
 * the slab runtime linked still resolve this symbol. When coqui_slab.c
 * is linked in, its strong definition wins; otherwise this no-op is used. */
__attribute__((weak, nothrow))
void __coqui_slab_release_thread(void) {
    /* no slab runtime linked — nothing to release */
}

/* Signalled-exit variant. Stamps the trap reason so the host can
 * distinguish OOM / stack overflow / other categorized traps from a
 * generic crash. Uses __coqui_exit() (not __coqui_trap) so the kernel
 * itself doesn't abort — peer threads keep running. Releases any slab
 * allocations this thread holds so the reclaimed ranges become
 * available to other threads.
 *
 * `noinline, cold, nothrow`: rare path, no need to clutter the icache hot
 * stream with the membar.sys + status write; keeping it out-of-line means
 * the OOM / stack-overflow / longjmp-trap call sites compile to a single
 * cold branch + call. */
__attribute__((noinline, cold, nothrow))
void __coqui_trap_with_reason(u8 reason) {
    __coqui_slab_release_thread();
    u32 tid = __coqui_fuzz_tid();
    if (likely(__coqui_status_array != (coqui_status_t *)0)) {
        __coqui_status_array[tid].trap_reason = reason;
    }
    /* sys-wide fence so the status write is visible to the host when the
     * thread's writes retire, then clean per-thread exit. */
    __asm__ volatile("membar.sys;");
    __coqui_exit();
    __builtin_unreachable();
}

/* Stack-canary check (see COQUI_STACK_CANARY in coqui_runtime.h).
 *
 * The pass MemoryLayout emits a call here at every kernel exit. `slot`
 * points at the u64 alloca written with COQUI_STACK_CANARY at kernel entry.
 * If the canary is intact this is a single load + compare + early return;
 * on mismatch we stamp COQUI_TRAP_STACK_OVERFLOW into the status slot and
 * __coqui_exit() the thread without trapping the whole kernel — peer
 * threads keep running, and the host's rerun loop picks up trap_reason=12
 * so it can re-verify this input on the CPU rerun path.
 *
 * `noinline, nothrow`: keep the canary-mismatch trap call out-of-line so
 * every kernel-exit site compiles to a single load+cmp+call (the load and
 * compare can stay inline if the inliner ever sees fit, but the trap arm
 * stays cold). The function never throws. */
__attribute__((noinline, nothrow))
void __coqui_check_stack_canary(const unsigned long *slot) {
    if (unlikely(*slot != COQUI_STACK_CANARY)) {
        __coqui_trap_with_reason(COQUI_TRAP_STACK_OVERFLOW);
        __builtin_unreachable();
    }
}

/* Status writer. `nothrow` — single store, never throws. */
__attribute__((nothrow))
void __coqui_status_set_phase(u32 tid, u8 phase) {
    __coqui_status_array[tid].phase = phase;
}

/* Benign stubs for rarely-used POSIX hooks. `nothrow` on each — these
 * are pure C runtime helpers that exist solely to satisfy linker symbol
 * lookups; they never throw and the bodies are no-ops. */
__attribute__((nothrow))
int atexit(void (*fn)(void))                 { (void)fn; return 0; }
__attribute__((nothrow))
void __cxa_atexit(void (*fn)(void *), void *arg, void *dso)
                                              { (void)fn; (void)arg; (void)dso; }
__attribute__((nothrow))
void __cxa_finalize(void *dso)               { (void)dso; }

/* pthread_once — single-threaded per GPU thread. `nothrow`. */
__attribute__((nothrow))
int pthread_once(int *once_control, void (*init)(void)) {
    if (unlikely(*once_control == 0)) {
        init();
        *once_control = 1;
    }
    return 0;
}

/* Per-thread clock64 budget poison (Exp #51).
 *
 * `__coqui_thread_budget_cycles` is the ceiling, in SM clock cycles, that
 * any single thread may spend on the kernel. Host writes it via
 * cuModuleGetGlobal+HtoD before the first launch. Value 0 disables the
 * mechanism entirely (production-safe default; existing host-side cull
 * timeout still applies as a backstop).
 *
 * `__coqui_thread_budget_start[BATCH]` records each thread's clock64()
 * snapshot at FuzzEntry. The Coverage pass emits a periodic
 * __coqui_check_thread_budget() call (once every N static BBs) along
 * the user code path; only the slow thread trips, peer threads in the
 * same warp/block keep running, and the kernel finishes naturally
 * before the host cull deadline — avoiding the force-reset that loses
 * the entire 8000-input batch.
 *
 * We size the start array to 8192 (BATCH_SIZE_FOR_COVERAGE in
 * Coverage.cpp) — must match. */
__attribute__((visibility("default")))
__attribute__((used))
u64 __coqui_thread_budget_cycles = 0;

__attribute__((visibility("default")))
__attribute__((used))
u64 __coqui_thread_budget_start[8192];

/* Stamp clock64() at kernel entry. `always_inline` — emitted exactly
 * once per thread (FuzzEntry inserts the call after slab-init, before
 * user code). `nothrow`. */
__attribute__((always_inline, nothrow))
void __coqui_thread_budget_init(void) {
    /* Only stamp when budget is enabled — saves a global store per
     * thread on the disabled-default path. */
    if (__coqui_thread_budget_cycles == 0) return;
    u32 tid = __coqui_fuzz_tid();
    __coqui_thread_budget_start[tid] = (u64)__nvvm_read_ptx_sreg_clock64();
}

/* Periodic budget check. Coverage pass calls this every N static BBs.
 *
 * Disabled-default fast path: a single ld.global.u64 of cycles + setp.eq
 * + early-return branch — costs ~5 PTX ops. The branch is uniform across
 * the warp when budget is unset (typical) so warps stay synchronized.
 *
 * `always_inline, nothrow`: aggressive inlining at every call site avoids
 * the per-call stack-frame setup (NVPTX pushes/pops .local, ~10 SASS ops
 * each) that dominates the disabled-default case. The trap path only ever
 * runs once per offending thread so its (cold) inline cost is irrelevant.
 *
 * Note: clock64() on NVPTX is the *SM* clock (advances even when the
 * thread is stalled waiting for a warp slot), not a per-thread counter.
 * `now - start` therefore measures wall-time-on-SM, including time the
 * thread spent suspended. Set AFL_COQUI_THREAD_BUDGET_US large enough
 * that legit work doesn't trip; the goal is to catch true 3-s+ runaways
 * before the host cull deadline does. */
__attribute__((always_inline, nothrow))
void __coqui_check_thread_budget(void) {
    u64 cycles_cap = __coqui_thread_budget_cycles;
    if (likely(cycles_cap == 0)) return;
    u32 tid = __coqui_fuzz_tid();
    u64 start = __coqui_thread_budget_start[tid];
    u64 now = (u64)__nvvm_read_ptx_sreg_clock64();
    if (unlikely(now - start > cycles_cap)) {
        __coqui_trap_with_reason(COQUI_TRAP_THREAD_BUDGET_EXHAUSTED);
        __builtin_unreachable();
    }
}
