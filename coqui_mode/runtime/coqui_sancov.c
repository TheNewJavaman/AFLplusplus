/*
 * coqui_sancov.c --- SanitizerCoverage runtime callbacks for NVPTX.
 *
 * When the target is compiled with clang -fsanitize-coverage=trace-pc-guard,
 * clang emits:
 *   - One uint32_t guard variable per instrumented edge, grouped into
 *     per-TU [N x i32] arrays named __sancov_gen_*.
 *   - At each edge:  __sanitizer_cov_trace_pc_guard(&guard[i])
 *   - Module init:   __sanitizer_cov_trace_pc_guard_init(start, stop)
 *
 * The SancovCount LLVM pass assigns sequential 1-based IDs to every guard
 * element and sets the arrays to constant. So at runtime, *guard == edge_id
 * and guard_init has nothing to do. trace_pc_guard increments the
 * per-thread cov_map_pool slot at index (edge_id - 1).
 *
 * Matches /home/gpizarro/coqui/runtime/coqui_sancov.c, with two simplifying
 * adaptations for cuAFL:
 *   - We get cov_map_pool + cov_map_size via the MemoryLayout-emitted
 *     __coqui_cov_base() (per-thread pointer) / __coqui_cov_map_size()
 *     (per-thread size) instead of a global pool pointer. This reuses the
 *     cuAFL slot pool and keeps the runtime stateless.
 *   - CmpLog / switch callbacks are not included — cuAFL doesn't ship
 *     -fsanitize-coverage=trace-cmp yet.
 */

#include "coqui_runtime.h"

/* ===-------------------------------------------------------------------===
 * trace_pc_guard_init — no-op.
 *
 * Kept as an exported symbol so that any late-bound references emitted by
 * clang (module-constructor style) resolve without error.
 * ===-------------------------------------------------------------------=== */
void __sanitizer_cov_trace_pc_guard_init(unsigned int *start,
                                          unsigned int *stop) {
    (void)start;
    (void)stop;
}

/* ===-------------------------------------------------------------------===
 * trace_pc_guard — per-edge hit counter.
 *
 * Called by the clang-emitted instrumentation on every basic block edge.
 * Guards are pre-populated with sequential IDs by the SancovCount pass, so
 * we use (*guard - 1) as the cov_map index directly.
 *
 * Every increment is a non-atomic u8 store. The per-thread cov_map is
 * thread-local storage (one base per thread, offsets derived from tid at
 * kernel entry), so concurrent access is impossible. Saturates at 0xFF
 * via u8 wrap — matching AFL's behaviour. The coverage_evaluate pass
 * post-processes via the bucket LUT so wrap-around is absorbed.
 * ===-------------------------------------------------------------------=== */
void __sanitizer_cov_trace_pc_guard(unsigned int *guard) {
    unsigned int edge = *guard;
    if (edge == 0) return;      /* 1-based IDs — zero means uninitialised. */

    u8 *cov = __coqui_cov_base();
    if (!cov) return;           /* defensive: pool not yet bound */

    cov[edge - 1]++;
}
