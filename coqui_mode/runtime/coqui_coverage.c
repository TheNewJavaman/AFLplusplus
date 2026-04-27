/*
 * coqui_coverage.c --- AFL hash coverage bucketing + virgin comparison.
 *
 * Bucketing: 256-entry lookup table; 8-byte-stride zero-skip walk.
 * Virgin compare: warp-level OR reduction + one atom.or.b64 per warp per
 * word on full warps (32× fewer L2 atomics vs. per-thread). Falls back to
 * per-thread atom.or.b64 on partial warps (empty-slot boundary warp of a
 * partial batch, or any warp with an ASan-triggered lane exit).
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

/* Byte-LUT classify of one u64 (in/out via pointer). Marked static inline
 * so the chunked outer loop can inline it and the compiler keeps `word` in
 * a register. Byte-identical output to the previous per-word code. */
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

/* Cacheline-chunked classify: 8 u64s (64 bytes) per outer iteration. A
 * single OR across the 8 loaded words lets us skip whole cachelines when
 * they're all zero, which is the common case for sparse cov_maps. The
 * compiler is free to coalesce the 8 adjacent 64-bit loads into wider
 * LDG.E.128 instructions. Per-word classify semantics unchanged.
 * `nothrow`. */
__attribute__((nothrow))
void __coqui_classify_counts(u8 *map) {
    u64 *m64 = (u64 *)map;
    const u32 n_chunks = COQUI_COV_MAP_SIZE / 8;     /* 8192 words */
    /* n_chunks is 8192 == multiple of 8 (1024 cachelines). */

    for (u32 i = 0; i < n_chunks; i += 8) {
        u64 w0 = m64[i+0], w1 = m64[i+1], w2 = m64[i+2], w3 = m64[i+3];
        u64 w4 = m64[i+4], w5 = m64[i+5], w6 = m64[i+6], w7 = m64[i+7];

        if (i + 8 < n_chunks) {
            asm volatile("prefetch.global.L1 [%0];" :: "l"((const void *)&m64[i+8]));
        }

        /* cov_map is sparse — most cachelines are zero. Cold-skip is the
         * dominant branch; classification work only fires on populated
         * cachelines. */
        if (likely((w0 | w1 | w2 | w3 | w4 | w5 | w6 | w7) == 0)) continue;

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

/* One-pass classify + hash. Identical classify semantics to the plain
 * variant; folds an FNV-1a hash of the classified words (and their
 * position, so bit-pattern-identical nonzero words at different offsets
 * still diverge) while the word is in a register. Skipping zero words
 * from the hash is safe because the u32 index is always mixed in, so
 * two maps that differ only in which zeros are skipped can't collide. */
__attribute__((nothrow))
u32 __coqui_classify_counts_and_sig(u8 *map) {
    u64 *m64 = (u64 *)map;
    const u32 n_chunks = COQUI_COV_MAP_SIZE / 8;     /* 8192 words */
    u32 h = COQUI_FNV32_OFFSET;
    /* Chunked 8-u64 (64-byte cacheline) walk. FNV semantics preserved:
     * each nonzero word still folds (i, lo, hi) in ascending-index order,
     * so the returned hash is byte-identical to the per-word variant. */

    for (u32 i = 0; i < n_chunks; i += 8) {
        u64 w0 = m64[i+0], w1 = m64[i+1], w2 = m64[i+2], w3 = m64[i+3];
        u64 w4 = m64[i+4], w5 = m64[i+5], w6 = m64[i+6], w7 = m64[i+7];

        if (i + 8 < n_chunks) {
            asm volatile("prefetch.global.L1 [%0];" :: "l"((const void *)&m64[i+8]));
        }

        /* cov_map is sparse — most cachelines are zero. */
        if (likely((w0 | w1 | w2 | w3 | w4 | w5 | w6 | w7) == 0)) continue;

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
    return h;
}

/* Same FNV-1a fold, read-only (no classify). Used by crash paths that
 * fire mid-execution (asan_report) where the cov_map is partially
 * written; the signature groups crashes at the same site.
 *
 * `pure, nothrow`: reads only the cov_map argument's memory and returns
 * a derived value with no side effects. */
__attribute__((pure, nothrow))
u32 __coqui_trace_sig(u8 *map) {
    u64 *m64 = (u64 *)map;
    const u32 n_chunks = COQUI_COV_MAP_SIZE / 8;
    u32 h = COQUI_FNV32_OFFSET;

    for (u32 i = 0; i < n_chunks; i++) {
        u64 word = m64[i];
        /* cov_map is sparse — most words are zero, especially for
         * partial-execution dumps from crash paths (asan_report). */
        if (likely(word == 0)) continue;
        h = (h ^ i)                 * COQUI_FNV32_PRIME;
        h = (h ^ (u32)(word))       * COQUI_FNV32_PRIME;
        h = (h ^ (u32)(word >> 32)) * COQUI_FNV32_PRIME;
    }
    return h;
}

/* -- Warp helpers -- */

static inline u32 __coqui_active_mask(void) {
    u32 m;
    asm volatile("activemask.b32 %0;" : "=r"(m));
    return m;
}

/* Butterfly OR across all lanes in `mask`. Caller must pass mask =
 * 0xFFFFFFFF (full warp); under partial masks, shfl.sync produces
 * undefined values for lanes whose XOR peer is outside the mask. */
static inline u64 __coqui_warp_or_u64(u32 mask, u64 v) {
    u32 hi = (u32)(v >> 32);
    u32 lo = (u32)v;
    for (int d = 16; d >= 1; d >>= 1) {
        hi |= (u32)__nvvm_shfl_sync_bfly_i32((int)mask, (int)hi, d, 0x1f);
        lo |= (u32)__nvvm_shfl_sync_bfly_i32((int)mask, (int)lo, d, 0x1f);
    }
    return ((u64)hi << 32) | lo;
}

/* Broadcast a 64-bit value from `src_lane` to every lane in `mask`. */
static inline u64 __coqui_warp_bcast_u64(u32 mask, u64 v, int src_lane) {
    u32 hi = (u32)__nvvm_shfl_sync_idx_i32((int)mask, (int)(u32)(v >> 32),
                                            src_lane, 0x1f);
    u32 lo = (u32)__nvvm_shfl_sync_idx_i32((int)mask, (int)(u32)v,
                                            src_lane, 0x1f);
    return ((u64)hi << 32) | lo;
}

__attribute__((nothrow))
void __coqui_virgin_compare_and_flag(u8 *map, u8 *virgin, u32 *novelty_bitmap) {
    _Atomic u64 *v64 = (_Atomic u64 *)virgin;
    u64 *m64 = (u64 *)map;
    const u32 n = COQUI_COV_MAP_SIZE / 8;
    const u32 mask = __coqui_active_mask();
    int novel = 0;

    if (likely(mask == 0xFFFFFFFFu)) {
        u32 laneid;
        asm volatile("mov.u32 %0, %%laneid;" : "=r"(laneid));

        for (u32 i = 0; i < n; i++) {
            u64 mine = m64[i];
            u64 warp_mine = __coqui_warp_or_u64(mask, mine);
            /* cov_map is sparse — most words are zero across the warp. */
            if (likely(warp_mine == 0)) continue;

            /* Non-atomic pre-read of virgin[i]. All lanes map to the same
             * address so L1 serves them from one cacheline. Virgin is
             * monotonic (bits only go 0->1), so skipping the atomic when
             * `warp_mine & ~v == 0` is safe: any bit that flips between
             * this read and when we would have done the atomic was claimed
             * by another warp first, which is the correct outcome.
             *
             * In steady-state fuzzing most edges have already been seen,
             * so the skip-the-atomic path is the hot one. */
            u64 v_pre = atomic_load_explicit(&v64[i], memory_order_relaxed);
            if (likely((warp_mine & ~v_pre) == 0)) continue;

            /* Lane 0 does the atomic; broadcast `was` (pre-OR virgin) so
             * every lane can compute its own novelty contribution
             * mine & ~was. Over-reports novelty within a warp when >1
             * lane independently set the same bit — benign, CPU verify
             * rejects false positives. */
            u64 was0 = 0;
            if (laneid == 0) {
                was0 = atomic_fetch_or_explicit(&v64[i], warp_mine,
                                                memory_order_relaxed);
            }
            u64 was = __coqui_warp_bcast_u64(mask, was0, 0);
            if (unlikely(mine & ~was)) { novel = 1; }
        }
    } else {
        /* Partial warp: original per-thread atomic. */
        for (u32 i = 0; i < n; i++) {
            u64 mine = m64[i];
            if (likely(mine == 0)) continue;
            /* Same non-atomic pre-read + skip as full-warp path. */
            u64 v_pre = atomic_load_explicit(&v64[i], memory_order_relaxed);
            if (likely((mine & ~v_pre) == 0)) continue;
            u64 was = atomic_fetch_or_explicit(&v64[i], mine,
                                                memory_order_relaxed);
            if (unlikely(mine & ~was)) { novel = 1; }
        }
    }

    if (unlikely(novel)) {
        u32 tid = __coqui_fuzz_tid();
        _Atomic u32 *nov32 = (_Atomic u32 *)&novelty_bitmap[tid >> 5];
        atomic_fetch_or_explicit(nov32, 1u << (tid & 31u), memory_order_relaxed);
    }
}
