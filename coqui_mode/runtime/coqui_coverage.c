/*
 * coqui_coverage.c --- AFL hash coverage bucketing + virgin comparison.
 *
 * Bucketing: 256-entry lookup table; 8-byte-stride zero-skip walk.
 * Virgin compare: C11 atomic_fetch_or lowers to PTX atom.or.b64 on NVPTX.
 * No inline asm.
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

void __coqui_virgin_compare_and_flag(u8 *map, u8 *virgin, u32 *novelty_bitmap) {
    _Atomic u64 *v64 = (_Atomic u64 *)virgin;
    u64 *m64 = (u64 *)map;
    const u32 n = COQUI_COV_MAP_SIZE / 8;

    int novel = 0;
    for (u32 i = 0; i < n; i++) {
        u64 mine = m64[i];
        if (mine == 0) continue;
        u64 was = atomic_fetch_or_explicit(&v64[i], mine, memory_order_relaxed);
        if (mine & ~was) { novel = 1; }
    }

    if (novel) {
        u32 tid = __coqui_fuzz_tid();
        _Atomic u32 *nov32 = (_Atomic u32 *)&novelty_bitmap[tid >> 5];
        atomic_fetch_or_explicit(nov32, 1u << (tid & 31u), memory_order_relaxed);
    }
}
