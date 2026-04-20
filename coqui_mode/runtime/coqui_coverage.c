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

void __coqui_classify_counts(u8 *map) {
    u64 *m64 = (u64 *)map;
    const u32 n_chunks = COQUI_COV_MAP_SIZE / 8;

    for (u32 i = 0; i < n_chunks; i++) {
        u64 word = m64[i];
        if (word == 0) continue;

        u8 *bytes = (u8 *)&word;
        bytes[0] = __coqui_count_class_lookup[bytes[0]];
        bytes[1] = __coqui_count_class_lookup[bytes[1]];
        bytes[2] = __coqui_count_class_lookup[bytes[2]];
        bytes[3] = __coqui_count_class_lookup[bytes[3]];
        bytes[4] = __coqui_count_class_lookup[bytes[4]];
        bytes[5] = __coqui_count_class_lookup[bytes[5]];
        bytes[6] = __coqui_count_class_lookup[bytes[6]];
        bytes[7] = __coqui_count_class_lookup[bytes[7]];
        m64[i] = word;
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
u32 __coqui_classify_counts_and_sig(u8 *map) {
    u64 *m64 = (u64 *)map;
    const u32 n_chunks = COQUI_COV_MAP_SIZE / 8;
    u32 h = COQUI_FNV32_OFFSET;

    for (u32 i = 0; i < n_chunks; i++) {
        u64 word = m64[i];
        if (word == 0) continue;

        u8 *bytes = (u8 *)&word;
        bytes[0] = __coqui_count_class_lookup[bytes[0]];
        bytes[1] = __coqui_count_class_lookup[bytes[1]];
        bytes[2] = __coqui_count_class_lookup[bytes[2]];
        bytes[3] = __coqui_count_class_lookup[bytes[3]];
        bytes[4] = __coqui_count_class_lookup[bytes[4]];
        bytes[5] = __coqui_count_class_lookup[bytes[5]];
        bytes[6] = __coqui_count_class_lookup[bytes[6]];
        bytes[7] = __coqui_count_class_lookup[bytes[7]];
        m64[i] = word;

        /* Fold the index then the low and high halves. */
        h = (h ^ i)                 * COQUI_FNV32_PRIME;
        h = (h ^ (u32)(word))       * COQUI_FNV32_PRIME;
        h = (h ^ (u32)(word >> 32)) * COQUI_FNV32_PRIME;
    }
    return h;
}

/* Same FNV-1a fold, read-only (no classify). Used by crash paths that
 * fire mid-execution (asan_report) where the cov_map is partially
 * written; the signature groups crashes at the same site. */
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

void __coqui_virgin_compare_and_flag(u8 *map, u8 *virgin, u32 *novelty_bitmap) {
    _Atomic u64 *v64 = (_Atomic u64 *)virgin;
    u64 *m64 = (u64 *)map;
    const u32 n = COQUI_COV_MAP_SIZE / 8;
    const u32 mask = __coqui_active_mask();
    int novel = 0;

    if (mask == 0xFFFFFFFFu) {
        u32 laneid;
        asm volatile("mov.u32 %0, %%laneid;" : "=r"(laneid));

        for (u32 i = 0; i < n; i++) {
            u64 mine = m64[i];
            u64 warp_mine = __coqui_warp_or_u64(mask, mine);
            if (warp_mine == 0) continue;

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
            if (mine & ~was) { novel = 1; }
        }
    } else {
        /* Partial warp: original per-thread atomic. */
        for (u32 i = 0; i < n; i++) {
            u64 mine = m64[i];
            if (mine == 0) continue;
            u64 was = atomic_fetch_or_explicit(&v64[i], mine,
                                                memory_order_relaxed);
            if (mine & ~was) { novel = 1; }
        }
    }

    if (novel) {
        u32 tid = __coqui_fuzz_tid();
        _Atomic u32 *nov32 = (_Atomic u32 *)&novelty_bitmap[tid >> 5];
        atomic_fetch_or_explicit(nov32, 1u << (tid & 31u), memory_order_relaxed);
    }
}
