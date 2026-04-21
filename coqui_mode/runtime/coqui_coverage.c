/*
 * coqui_coverage.c --- AFL hash coverage bucketing + virgin comparison.
 *
 * Iteration 20 (warp-shared cov_map): the per-warp cov_map is shared by all
 * 32 lanes. Classify + virgin compare therefore run ONCE per warp, on lane 0:
 *   - __coqui_classify_counts_and_sig: lane 0 walks the 64 KB map; all other
 *     lanes idle. The resulting sig is broadcast via shfl.sync.idx so every
 *     lane returns the same u32 (the host reads status[tid].crash_sig per-lane
 *     but the crash-sig dedup hash-set then collapses duplicates).
 *   - __coqui_virgin_compare_and_flag: lane 0 walks the classified map and
 *     performs atom.or into the global virgin_map. Partial warps (ASan-exited
 *     lane) still route through a slow per-thread fallback for correctness.
 *     When novel, every lane in the warp sets its own bit in the novelty
 *     bitmap so the host processes all 32 slots; crash-sig dedup then
 *     collapses them into a single CPU verify call per warp.
 *
 * Bucketing: 256-entry lookup table; 8-byte-stride zero-skip walk.
 * Fidelity: per-warp novelty (not per-input) — allowed per iter 20 policy.
 * Race tolerance: cov_map edge writes are non-atomic within the warp; the
 * AFL bucket classes compress small count differences so most races are
 * invisible to the classifier.
 *
 * Inline asm used for activemask.b32 / mov %laneid — no clang NVPTX
 * builtin exists outside the CUDA header. Shuffles use clang builtins
 * (__nvvm_shfl_sync_bfly_i32 / __nvvm_shfl_sync_idx_i32) which lower to
 * shfl.sync.bfly.b32 / shfl.sync.idx.b32 directly.
 */

#include "coqui_runtime.h"
#include <stdatomic.h>

/* Bucket lookup — matches AFL's count_class_lookup16. */
__attribute__((section(".const"), used))
u8 __coqui_count_class_lookup[256] = {
    [0]   = COQUI_BUCKET_0,
    [1]   = COQUI_BUCKET_1,
    [2]   = COQUI_BUCKET_2,
    [3]   = COQUI_BUCKET_3,
    [4 ... 7]    = COQUI_BUCKET_4_7,
    [8 ... 15]   = COQUI_BUCKET_8_15,
    [16 ... 31]  = COQUI_BUCKET_16_31,
    [32 ... 127] = COQUI_BUCKET_32_127,
    [128 ... 255] = COQUI_BUCKET_128_UP,
};

/* Byte-LUT classify of one u64 (in/out via pointer). */
static inline void __coqui_classify_word(u64 *w) {
    u64 word = *w;
    u8 *bytes = (u8 *)&word;
    bytes[0] = __coqui_count_class_lookup[bytes[0]];
    bytes[1] = __coqui_count_class_lookup[bytes[1]];
    bytes[2] = __coqui_count_class_lookup[bytes[2]];
    bytes[3] = __coqui_count_class_lookup[bytes[3]];
    bytes[4] = __coqui_count_class_lookup[bytes[4]];
    bytes[5] = __coqui_count_class_lookup[bytes[5]];
    bytes[6] = __coqui_count_class_lookup[bytes[6]];
    bytes[7] = __coqui_count_class_lookup[bytes[7]];
    *w = word;
}

/* Cacheline-chunked classify: 8 u64s (64 bytes) per outer iteration. */
void __coqui_classify_counts(u8 *map) {
    u64 *m64 = (u64 *)map;
    const u32 n_chunks = COQUI_COV_MAP_SIZE / 8;

    for (u32 i = 0; i < n_chunks; i += 8) {
        u64 w0 = m64[i+0], w1 = m64[i+1], w2 = m64[i+2], w3 = m64[i+3];
        u64 w4 = m64[i+4], w5 = m64[i+5], w6 = m64[i+6], w7 = m64[i+7];

        if (i + 8 < n_chunks) {
            asm volatile("prefetch.global.L1 [%0];" :: "l"((const void *)&m64[i+8]));
        }

        if ((w0 | w1 | w2 | w3 | w4 | w5 | w6 | w7) == 0) continue;

        if (w0) { __coqui_classify_word(&w0); m64[i+0] = w0; }
        if (w1) { __coqui_classify_word(&w1); m64[i+1] = w1; }
        if (w2) { __coqui_classify_word(&w2); m64[i+2] = w2; }
        if (w3) { __coqui_classify_word(&w3); m64[i+3] = w3; }
        if (w4) { __coqui_classify_word(&w4); m64[i+4] = w4; }
        if (w5) { __coqui_classify_word(&w5); m64[i+5] = w5; }
        if (w6) { __coqui_classify_word(&w6); m64[i+6] = w6; }
        if (w7) { __coqui_classify_word(&w7); m64[i+7] = w7; }
    }
}

/* FNV-1a 32-bit constants. */
#define COQUI_FNV32_OFFSET  0x811c9dc5u
#define COQUI_FNV32_PRIME   0x01000193u

/* -- Warp helpers -- */

static inline u32 __coqui_active_mask(void) {
    u32 m;
    asm volatile("activemask.b32 %0;" : "=r"(m));
    return m;
}

static inline u32 __coqui_laneid(void) {
    u32 l;
    asm volatile("mov.u32 %0, %%laneid;" : "=r"(l));
    return l;
}

/* Broadcast a 32-bit value from `src_lane` to every lane in `mask`. */
static inline u32 __coqui_warp_bcast_u32(u32 mask, u32 v, int src_lane) {
    return (u32)__nvvm_shfl_sync_idx_i32((int)mask, (int)v, src_lane, 0x1f);
}

/* Broadcast a 64-bit value from `src_lane` to every lane in `mask`. */
static inline u64 __coqui_warp_bcast_u64(u32 mask, u64 v, int src_lane) {
    u32 hi = __coqui_warp_bcast_u32(mask, (u32)(v >> 32), src_lane);
    u32 lo = __coqui_warp_bcast_u32(mask, (u32)v, src_lane);
    return ((u64)hi << 32) | lo;
}

/* Butterfly OR across all lanes in `mask`. */
static inline u64 __coqui_warp_or_u64(u32 mask, u64 v) {
    u32 hi = (u32)(v >> 32);
    u32 lo = (u32)v;
    for (int d = 16; d >= 1; d >>= 1) {
        hi |= (u32)__nvvm_shfl_sync_bfly_i32((int)mask, (int)hi, d, 0x1f);
        lo |= (u32)__nvvm_shfl_sync_bfly_i32((int)mask, (int)lo, d, 0x1f);
    }
    return ((u64)hi << 32) | lo;
}

/* Warp-shared one-pass classify + hash. Lane 0 walks the warp's single
 * cov_map (64 KB, shared across 32 lanes). Other lanes idle. The resulting
 * FNV-1a hash is broadcast via shfl.sync.idx so every lane returns the
 * same u32 signature. Partial warps (ASan-exited lanes) fall through to
 * the first surviving lane for the walk. */
u32 __coqui_classify_counts_and_sig(u8 *map) {
    u32 mask = __coqui_active_mask();
    u32 laneid = __coqui_laneid();

    /* Pick the lowest active lane as the walker. For full warps this is 0;
     * for partial warps (ASan-exited earlier lanes) this is whichever lane
     * is lowest in `mask`. shfl.sync.idx requires the src lane be active. */
    int walker = __builtin_ctz(mask);

    u32 h = 0;
    if ((int)laneid == walker) {
        u64 *m64 = (u64 *)map;
        const u32 n_chunks = COQUI_COV_MAP_SIZE / 8;
        h = COQUI_FNV32_OFFSET;

        for (u32 i = 0; i < n_chunks; i += 8) {
            u64 w0 = m64[i+0], w1 = m64[i+1], w2 = m64[i+2], w3 = m64[i+3];
            u64 w4 = m64[i+4], w5 = m64[i+5], w6 = m64[i+6], w7 = m64[i+7];

            if (i + 8 < n_chunks) {
                asm volatile("prefetch.global.L1 [%0];" :: "l"((const void *)&m64[i+8]));
            }

            if ((w0 | w1 | w2 | w3 | w4 | w5 | w6 | w7) == 0) continue;

            if (w0) {
                __coqui_classify_word(&w0);
                m64[i+0] = w0;
                h = (h ^ (i+0))             * COQUI_FNV32_PRIME;
                h = (h ^ (u32)(w0))         * COQUI_FNV32_PRIME;
                h = (h ^ (u32)(w0 >> 32))   * COQUI_FNV32_PRIME;
            }
            if (w1) {
                __coqui_classify_word(&w1);
                m64[i+1] = w1;
                h = (h ^ (i+1))             * COQUI_FNV32_PRIME;
                h = (h ^ (u32)(w1))         * COQUI_FNV32_PRIME;
                h = (h ^ (u32)(w1 >> 32))   * COQUI_FNV32_PRIME;
            }
            if (w2) {
                __coqui_classify_word(&w2);
                m64[i+2] = w2;
                h = (h ^ (i+2))             * COQUI_FNV32_PRIME;
                h = (h ^ (u32)(w2))         * COQUI_FNV32_PRIME;
                h = (h ^ (u32)(w2 >> 32))   * COQUI_FNV32_PRIME;
            }
            if (w3) {
                __coqui_classify_word(&w3);
                m64[i+3] = w3;
                h = (h ^ (i+3))             * COQUI_FNV32_PRIME;
                h = (h ^ (u32)(w3))         * COQUI_FNV32_PRIME;
                h = (h ^ (u32)(w3 >> 32))   * COQUI_FNV32_PRIME;
            }
            if (w4) {
                __coqui_classify_word(&w4);
                m64[i+4] = w4;
                h = (h ^ (i+4))             * COQUI_FNV32_PRIME;
                h = (h ^ (u32)(w4))         * COQUI_FNV32_PRIME;
                h = (h ^ (u32)(w4 >> 32))   * COQUI_FNV32_PRIME;
            }
            if (w5) {
                __coqui_classify_word(&w5);
                m64[i+5] = w5;
                h = (h ^ (i+5))             * COQUI_FNV32_PRIME;
                h = (h ^ (u32)(w5))         * COQUI_FNV32_PRIME;
                h = (h ^ (u32)(w5 >> 32))   * COQUI_FNV32_PRIME;
            }
            if (w6) {
                __coqui_classify_word(&w6);
                m64[i+6] = w6;
                h = (h ^ (i+6))             * COQUI_FNV32_PRIME;
                h = (h ^ (u32)(w6))         * COQUI_FNV32_PRIME;
                h = (h ^ (u32)(w6 >> 32))   * COQUI_FNV32_PRIME;
            }
            if (w7) {
                __coqui_classify_word(&w7);
                m64[i+7] = w7;
                h = (h ^ (i+7))             * COQUI_FNV32_PRIME;
                h = (h ^ (u32)(w7))         * COQUI_FNV32_PRIME;
                h = (h ^ (u32)(w7 >> 32))   * COQUI_FNV32_PRIME;
            }
        }
    }

    /* Broadcast the walker's hash to all lanes. */
    return __coqui_warp_bcast_u32(mask, h, walker);
}

/* Same FNV-1a fold, read-only (no classify). Used by asan_report when the
 * cov_map is partially written mid-execution. This function is called
 * unpredictably — any lane may crash while others are still executing —
 * so we keep it per-thread walk (each caller walks the shared map
 * independently). Under warp-shared cov this still produces the same sig
 * for all same-warp callers since they read the same map. */
u32 __coqui_trace_sig(u8 *map) {
    u64 *m64 = (u64 *)map;
    const u32 n_chunks = COQUI_COV_MAP_SIZE / 8;
    u32 h = COQUI_FNV32_OFFSET;

    for (u32 i = 0; i < n_chunks; i++) {
        u64 word = m64[i];
        if (word == 0) continue;
        h = (h ^ i)                 * COQUI_FNV32_PRIME;
        h = (h ^ (u32)(word))       * COQUI_FNV32_PRIME;
        h = (h ^ (u32)(word >> 32)) * COQUI_FNV32_PRIME;
    }
    return h;
}

/* Warp-shared virgin compare + novelty flag.
 *
 * Under warp-shared cov_map, every lane in a full warp observes the same
 * classified map. So the virgin-compare walk runs ONCE on lane 0 (walker),
 * which performs atom.or on each non-zero word into the global virgin map
 * and tracks whether any novelty was found. The walker then broadcasts
 * novel=1/0 to all lanes; every lane sets its own novelty bit if novel.
 *
 * Partial warps (ASan-exited lane in the same warp) fall through to a
 * per-thread loop — correctness first, speed second. */
void __coqui_virgin_compare_and_flag(u8 *map, u8 *virgin, u32 *novelty_bitmap) {
    _Atomic u64 *v64 = (_Atomic u64 *)virgin;
    u64 *m64 = (u64 *)map;
    const u32 n = COQUI_COV_MAP_SIZE / 8;
    const u32 mask = __coqui_active_mask();
    u32 laneid = __coqui_laneid();

    if (mask == 0xFFFFFFFFu) {
        u32 novel = 0;
        if (laneid == 0) {
            for (u32 i = 0; i < n; i++) {
                u64 mine = m64[i];
                if (mine == 0) continue;
                u64 was = atomic_fetch_or_explicit(&v64[i], mine,
                                                    memory_order_relaxed);
                if (mine & ~was) novel = 1;
            }
        }
        novel = __coqui_warp_bcast_u32(mask, novel, 0);
        if (novel) {
            u32 tid = __coqui_fuzz_tid();
            _Atomic u32 *nov32 = (_Atomic u32 *)&novelty_bitmap[tid >> 5];
            atomic_fetch_or_explicit(nov32, 1u << (tid & 31u),
                                      memory_order_relaxed);
        }
    } else {
        /* Partial warp: fall back to per-thread walk. Because cov_map is
         * warp-shared, multiple lanes redundantly OR the same data into
         * virgin, but atom.or is idempotent. */
        int novel = 0;
        for (u32 i = 0; i < n; i++) {
            u64 mine = m64[i];
            if (mine == 0) continue;
            u64 was = atomic_fetch_or_explicit(&v64[i], mine,
                                                memory_order_relaxed);
            if (mine & ~was) { novel = 1; }
        }
        if (novel) {
            u32 tid = __coqui_fuzz_tid();
            _Atomic u32 *nov32 = (_Atomic u32 *)&novelty_bitmap[tid >> 5];
            atomic_fetch_or_explicit(nov32, 1u << (tid & 31u),
                                      memory_order_relaxed);
        }
    }
}
