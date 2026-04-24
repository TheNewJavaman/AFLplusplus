/*
 * coqui_trace.c --- per-thread line-trace recorder for oracle mode.
 *
 * Records the sequence of line IDs each thread executes in a contiguous
 * .global buffer the host allocates and binds at module-load. One
 * 65,536-byte stripe per thread (16,384 entries × 4B) — enough for short
 * harness runs whose traces fit in memory while staying under the
 * device-global budget for batch_size=8192 (~512 MB total).
 *
 * Buffer layout (linear in __coqui_trace_buffer):
 *   thread t's entries occupy [t * COQUI_TRACE_MAX_ENTRIES,
 *                              (t+1) * COQUI_TRACE_MAX_ENTRIES)
 *
 * Counter layout (parallel u32 array __coqui_trace_count):
 *   __coqui_trace_count[t] = number of entries thread t has written
 *
 * Overflow policy: silently stop recording for that thread. Overflow
 * itself is a divergence signal — if the host's expected trace fits
 * but the GPU's overflows, semantics diverged.
 *
 * Globals are externally-bound: the host calls cuModuleGetGlobal +
 * cuMemcpyHtoD to write the device-side allocations' pointers into
 * __coqui_trace_buffer / __coqui_trace_count_ptr. When the host has
 * NOT bound them (production runs, or oracle off), both pointers stay
 * NULL and __coqui_trace_line short-circuits — same pattern as the
 * slab pool's NULL-pool fallthrough.
 *
 * The pass LineTrace.cpp emits the calls when -coqui-line-trace is set.
 * When the flag is off, the pass is a no-op and this file's symbols
 * remain unreferenced — DCE in the runtime link prunes the body.
 */

#include "coqui_runtime.h"

/* Per-thread trace stripe size (bytes / entries). Match host-side
 * COQUI_TRACE_BUFFER_BYTES in afl-fuzz-coqui.h. 64 KB / 4 B = 16,384
 * entries — captures 16k traced lines per thread per batch. */
#define COQUI_TRACE_BUFFER_BYTES 65536u
#define COQUI_TRACE_MAX_ENTRIES  (COQUI_TRACE_BUFFER_BYTES / 4u)

/* Host-bound globals. Empty initializers so an oracle-off launch (host
 * never calls cuModuleGetGlobal/HtoD) sees NULL and __coqui_trace_line
 * short-circuits. Visibility/used so DCE doesn't drop them when the
 * pass is a no-op (no callsites of __coqui_trace_line) — the runtime
 * `runtime.bc` link still needs to expose the symbols for the host to
 * resolve, even on cubins built without -coqui-line-trace. */
__attribute__((visibility("default"))) __attribute__((used))
unsigned int *__coqui_trace_buffer;

__attribute__((visibility("default"))) __attribute__((used))
unsigned int *__coqui_trace_count;

void __coqui_trace_line(unsigned int line_id) {
    /* NULL guard: oracle-off launches leave the buffer pointers
     * unset. The pass-emitted call sites still execute (they're
     * unconditional), so we must short-circuit here. Cheap branch on
     * a register-cached load; predicted not-taken when oracle is on. */
    if (!__coqui_trace_buffer || !__coqui_trace_count)
        return;

    u32 tid = __coqui_fuzz_tid();
    unsigned int *buf = __coqui_trace_buffer + (u64)tid * COQUI_TRACE_MAX_ENTRIES;
    unsigned int *cnt = __coqui_trace_count + tid;

    /* Per-thread counter: no atomic needed — only this thread writes
     * to its stripe. Read-modify-write of cnt[0] is safe. */
    unsigned int idx = *cnt;
    if (idx < COQUI_TRACE_MAX_ENTRIES) {
        buf[idx] = line_id;
        *cnt = idx + 1;
    }
    /* idx >= MAX: overflow. Drop silently — the host detects overflow
     * by seeing count[t] == COQUI_TRACE_MAX_ENTRIES and treats it as a
     * divergence signal (the CPU expected trace must also fit). */
}
