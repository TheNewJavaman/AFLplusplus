/*
 * coqui_coverage.c --- AFL hash coverage bucketing + virgin comparison.
 *
 * Bucketing: 256-entry lookup table in __constant__ memory; walks the
 *   64 KB coverage map 8 bytes at a time with zero-chunk early exit.
 * Virgin compare: atomic-OR the bucketed map into shared virgin_map;
 *   set the novelty bit if any new bits appeared.
 */

#include "coqui_runtime.h"

/* Bucket lookup table — matches AFL's count_class_lookup16. */
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

/* Bucket the map in place. Walks 8-byte chunks with zero-skip. */
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

/* 64-bit atomic OR — PTX atom.or.b64 on global memory. */
static u64 atom_or_64(u64 *addr, u64 val) {
    u64 old;
    __asm__ volatile("atom.or.b64 %0, [%1], %2;"
                     : "=l"(old) : "l"(addr), "l"(val));
    return old;
}

/* 32-bit atomic OR — for the novelty bitmap. */
static u32 atom_or_32(u32 *addr, u32 val) {
    u32 old;
    __asm__ volatile("atom.or.b32 %0, [%1], %2;"
                     : "=r"(old) : "l"(addr), "r"(val));
    return old;
}

/* Compare the thread's bucketed map against the shared virgin map.
   Set the novelty bit for this thread if any new bits appeared. */
void __coqui_virgin_compare_and_flag(u8 *map, u8 *virgin, u32 *novelty_bitmap) {
    u64 *m64 = (u64 *)map;
    u64 *v64 = (u64 *)virgin;
    const u32 n = COQUI_COV_MAP_SIZE / 8;

    int novel = 0;
    for (u32 i = 0; i < n; i++) {
        u64 mine = m64[i];
        if (mine == 0) continue;
        u64 was = atom_or_64(&v64[i], mine);
        if (mine & ~was) { novel = 1; }
    }

    if (novel) {
        u32 tid = __coqui_fuzz_tid();
        atom_or_32(&novelty_bitmap[tid >> 5], 1u << (tid & 31u));
    }
}
