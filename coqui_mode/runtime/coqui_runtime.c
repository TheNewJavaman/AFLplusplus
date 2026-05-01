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
#include "coqui_afl_mutate.h"

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
 * We size the start array to 65536 (BATCH_SIZE_FOR_COVERAGE in
 * Coverage.cpp) — must match. */
__attribute__((visibility("default")))
__attribute__((used))
u64 __coqui_thread_budget_cycles = 0;

__attribute__((visibility("default")))
__attribute__((used))
u64 __coqui_thread_budget_start[65536];

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

/* ========================================================================
 * GPU-side mutation support (Phase 2+3: GPU-mutate AFL integration).
 *
 * When AFL_COQUI_GPU_MUTATE=1, the host sends UNMUTATED parents and the
 * GPU performs all mutation. Each thread picks its parent from the compact
 * parent table, copies it to its output slot, applies AFL-exact havoc
 * mutations using per-thread PRNG seeds, then calls the harness.
 * ======================================================================== */

/* Per-parent compact table (sparse H2D: Change 1).
 *
 * Instead of broadcasting each parent to 128 thread slots (128 copies of
 * the same data), the host uploads each parent ONCE into a compact buffer.
 * Per-thread __coqui_parent_idx[tid] maps each thread to its parent index;
 * __coqui_parent_offsets[idx] and __coqui_parent_lens[idx] locate the parent
 * in the compact buffer. The mutation wrapper copies from the parent table
 * into the per-thread output slot before mutating.
 *
 * These are pointer-sized globals; the host allocates backing storage via
 * cuMemAlloc and writes the device addresses here via cuMemcpyHtoD. */
__attribute__((visibility("default")))
__attribute__((used))
u8  *__coqui_parent_bytes;       /* compact parent data (host-allocated) */

__attribute__((visibility("default")))
__attribute__((used))
u32 *__coqui_parent_offsets;     /* offset per parent (host-allocated) */

__attribute__((visibility("default")))
__attribute__((used))
u32 *__coqui_parent_lens;        /* length per parent (host-allocated) */

__attribute__((visibility("default")))
__attribute__((used))
u32 *__coqui_parent_idx;         /* per-thread parent index (host-allocated) */

/* Per-thread PRNG seed pairs — max 65536 threads (matching BATCH_SIZE cap). */
__attribute__((visibility("default")))
__attribute__((used))
u64 __coqui_mutate_prng[65536 * 2];

/* Maximum stacking depth for per-thread PRNG-derived stacking.
 * Each thread computes: steps = 1 + rand_below(stack_max).
 * Host sets this to 4 (early) or 8 (after 10 min). */
__attribute__((visibility("default")))
__attribute__((used))
u32 __coqui_mutate_stack_max;

/* Master enable flag: 0=off (default), 1=on. */
__attribute__((visibility("default")))
__attribute__((used))
u8 __coqui_gpu_mutate_enabled;

/* Number of unique parents in the current batch (for splice mutations). */
__attribute__((visibility("default")))
__attribute__((used))
u32 __coqui_parent_count;

/* Dictionary (extras) device-side table.
 *
 * The host packs user-supplied dictionary tokens (extras) and auto-discovered
 * tokens (a_extras) into a flat byte buffer with offset/len metadata.
 * Auto-extras are appended after regular extras in the same data buffer;
 * __coqui_a_extras_off gives their starting index in offsets/lens.
 *
 * Layout:
 *   data[0..extras_total_bytes]: packed token bytes
 *   offsets[i]: byte offset into data for token i
 *   lens[i]: byte length of token i
 *   i < extras_cnt: regular extras
 *   extras_cnt <= i < extras_cnt + a_extras_cnt: auto-extras
 *
 * Limits: MAX_EXTRAS=4096 tokens, MAX_EXTRAS_BYTES=131072 total bytes.
 * Tokens exceeding these caps are silently dropped by the host upload. */
#define COQUI_MAX_EXTRAS       4096u
#define COQUI_MAX_EXTRAS_BYTES 131072u

__attribute__((visibility("default")))
__attribute__((used))
u8  __coqui_extras_data[COQUI_MAX_EXTRAS_BYTES];

__attribute__((visibility("default")))
__attribute__((used))
u32 __coqui_extras_offsets[COQUI_MAX_EXTRAS];

__attribute__((visibility("default")))
__attribute__((used))
u32 __coqui_extras_lens[COQUI_MAX_EXTRAS];

__attribute__((visibility("default")))
__attribute__((used))
u32 __coqui_extras_cnt;      /* number of regular extras */

__attribute__((visibility("default")))
__attribute__((used))
u32 __coqui_a_extras_cnt;    /* number of auto-extras (starting after extras) */

/* Maximum input size for mutation (host-configured, matches max_input_size). */
__attribute__((visibility("default")))
__attribute__((used))
u32 __coqui_mutate_max_input_size;

/* Apply GPU-side mutation to the per-thread input buffer.
 *
 * Per-parent H2D (Change 1): instead of reading from the per-thread
 * input slot (which would contain a broadcast copy of the parent),
 * read the parent from the compact parent table. The parent table has
 * each unique parent stored once; __coqui_parent_idx[tid] maps this
 * thread to its parent. Copy the parent into the per-thread output
 * slot, mutate in-place, and the harness reads the mutated result
 * from the same slot.
 *
 * Power-schedule mode: each thread derives its own stacking depth
 * from its PRNG: steps = 1 + rand_below(stack_max). This mirrors
 * AFL's havoc loop where each iteration picks a random stacking.
 *
 * `noinline` keeps the mutation body separate for debugging and avoids
 * inflating the fast path when GPU mutation is disabled. Can be switched
 * to `always_inline` later if call overhead matters.
 *
 * The scratch buffer is a local stack alloca (sized to max_input_size).
 * Mutations that grow the buffer (clone, insert) need this as a temp. */
__attribute__((noinline, nothrow))
void __coqui_mutate_input(u32 tid, u8 *buf, u32 *len_ptr, u32 max_len) {
    if (!__coqui_gpu_mutate_enabled) return;

    /* Read parent from compact parent table and copy to per-thread slot. */
    u32 pidx = __coqui_parent_idx[tid];
    u32 poff = __coqui_parent_offsets[pidx];
    u32 plen = __coqui_parent_lens[pidx];
    u8 *parent_ptr = __coqui_parent_bytes + poff;
    memcpy(buf, parent_ptr, plen);
    *len_ptr = plen;

    coqui_mutate_ctx_t ctx;
    ctx.rand_seed[0] = __coqui_mutate_prng[tid * 2];
    ctx.rand_seed[1] = __coqui_mutate_prng[tid * 2 + 1];

    /* Per-thread stacking depth from PRNG, mirroring AFL's havoc loop:
     *   use_stacking = 1 + rand_below(stack_max)
     * stack_max is set by the host (4 early, 8 after 10 min). */
    u32 sm = __coqui_mutate_stack_max;
    if (sm == 0) sm = 4;  /* defensive default */
    u32 steps = 1 + coqui_rand_below(&ctx, sm);

    /* Scratch buffer on the per-thread stack. Cap at 4096 to stay within
     * typical stack budgets; inputs larger than this will still mutate but
     * mutations that need the scratch (clone/insert) will retry-to-havoc
     * or be skipped for the oversized tail. */
    u8 scratch[4096];
    u32 scratch_max = max_len < 4096 ? max_len : 4096;

    u32 new_len = coqui_afl_mutate(&ctx, buf, plen, steps,
                                    scratch, scratch_max);
    if (new_len > 0) {
        *len_ptr = new_len;
    }
}
