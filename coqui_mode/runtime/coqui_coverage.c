/*
 * coqui_coverage.c --- single-pass SanitizerCoverage evaluation.
 *
 * Walks the per-thread coverage map in 4-byte words, classifies each hit
 * count via a const-address-space LUT, zeros the word in place (so the
 * next batch starts from a clean map — eliminates the host-side cuMemset
 * that AFL-style maps required), then warp-OR-reduces the classified word
 * and atom.or's one word per warp per offset into the global cov bitmap.
 *
 * Returns 1 if THIS thread contributed a new bit (pre-merge global did not
 * already have it), 0 otherwise. Novelty-bitmap writing is handled by the
 * FuzzEntry-generated IR (see FuzzEntry.cpp).
 *
 * Ported from /home/gpizarro/coqui/runtime/coqui_coverage.c. Deviations:
 *   - cov_map size is passed in at runtime (the caller reads
 *     __coqui_cov_map_size() which returns the pass-materialized constant).
 *     This lets the same runtime bitcode work for different cov_map sizes
 *     across rebuilds.
 *   - trace_sig still produces a u32 FNV-1a hash but now walks cov_map_size
 *     bytes (N edges, dense) instead of 64 KB (sparse).
 */

#include "coqui_runtime.h"

/* ===-------------------------------------------------------------------===
 * AFL hit-count classification — LUT in NVPTX address_space(4) = const mem.
 *
 * The 256-byte table fits in the 64 KB constant cache so every byte-lookup
 * is a fast broadcast read (no global memory traffic).
 * ===-------------------------------------------------------------------=== */

/* clang-format off */
static const u8 __attribute__((address_space(4))) __coqui_classify_lut[256] = {
  /*   0 */   0,
  /*   1 */   1,
  /*   2 */   2,
  /*   3 */   4,
  /* 4-7 */   8,  8,  8,  8,
  /* 8-15 */ 16, 16, 16, 16, 16, 16, 16, 16,
  /* 16-31 */
  32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
  /* 32-127 */
  64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
  64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
  64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
  64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
  64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
  64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
  /* 128-255 */
  128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
  128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
  128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
  128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
  128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
  128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
  128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
  128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
};
/* clang-format on */

/* ===-------------------------------------------------------------------===
 * Atomic / shuffle primitives (inline PTX — matches coqui reference)
 * ===-------------------------------------------------------------------=== */

static inline u32 __coqui_atom_or_u32(u32 *addr, u32 val) {
    u32 old;
    asm volatile("atom.or.b32 %0, [%1], %2;"
                 : "=r"(old)
                 : "l"(addr), "r"(val));
    return old;
}

/* Butterfly OR across the full 32-lane warp (mask must be 0xFFFFFFFF). */
static inline u32 __coqui_shfl_xor_u32_sync(u32 mask, u32 val, u32 delta) {
    u32 result;
    asm volatile("shfl.sync.bfly.b32 %0, %1, %2, 0x1f, %3;"
                 : "=r"(result)
                 : "r"(val), "r"(delta), "r"(mask));
    return result;
}

/* Lane 0 broadcast of a u32. */
static inline u32 __coqui_shfl_bcast_u32_sync(u32 mask, u32 val) {
    u32 result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1f, %2;"
                 : "=r"(result)
                 : "r"(val), "r"(mask));
    return result;
}

static inline u32 __coqui_lane_id(void) {
    u32 lane;
    asm volatile("mov.u32 %0, %%laneid;" : "=r"(lane));
    return lane;
}

static inline u32 __coqui_active_mask(void) {
    u32 m;
    asm volatile("activemask.b32 %0;" : "=r"(m));
    return m;
}

/* ===-------------------------------------------------------------------===
 * Single-pass coverage evaluation — warp-reduced atom.or
 *
 * For each 4-byte word of the thread's cov map:
 *   1. Load the raw u32.
 *   2. If zero, continue (fast skip; most words are zero for cjson/jsmn).
 *   3. Zero the slot in place (next batch starts from clean map — no host
 *      memset needed).
 *   4. Classify each of the 4 bytes via the const-LUT.
 *   5. Warp-OR across all 32 lanes via shfl.sync.bfly.b32 (5 levels).
 *   6. Lane 0 atom.or's the warp-merged classified word into global_cov
 *      at the SAME word offset. Other lanes skip the atomic — one atomic
 *      per warp per word, not 32.
 *   7. Lane 0 broadcasts the pre-merge old value so every lane can compute
 *      its own novelty contribution (classified & ~old) and set found_new.
 *
 * Under the active_mask==0xFFFFFFFF fast path, the warp reduction is
 * correct. Under a partial mask (e.g. a thread exited early via ASan trap),
 * shfl.sync with mask=full-warp on inactive lanes is UB, so we fall back to
 * per-thread atom.or in that case — same pattern as the previous
 * virgin_compare_and_flag.
 *
 * cov_map_size MUST be a multiple of 4.
 * ===-------------------------------------------------------------------=== */

int __coqui_coverage_evaluate(u8 *thread_cov, u8 *global_cov,
                              u64 cov_map_size) {
    int found_new = 0;
    const u64 words = cov_map_size / 4;
    const u32 mask = __coqui_active_mask();
    const u32 laneid = __coqui_lane_id();

    if (mask == 0xFFFFFFFFu) {
        for (u64 w = 0; w < words; w++) {
            u64 base = w * 4;
            u32 *word_ptr = (u32 *)__builtin_assume_aligned(
                thread_cov + base, _Alignof(u32));
            u32 raw = *word_ptr;

            u32 classified = 0;
            if (raw != 0) {
                /* Zero slot in place for next batch (same cache line). */
                *word_ptr = 0;

                classified =
                    (u32)__coqui_classify_lut[(u8)(raw)] |
                    ((u32)__coqui_classify_lut[(u8)(raw >> 8)]  << 8) |
                    ((u32)__coqui_classify_lut[(u8)(raw >> 16)] << 16) |
                    ((u32)__coqui_classify_lut[(u8)(raw >> 24)] << 24);
            }

            /* Warp butterfly OR to merge classified words across all lanes. */
            u32 warp_val = classified;
            for (u32 d = 16; d >= 1; d >>= 1) {
                warp_val |= __coqui_shfl_xor_u32_sync(mask, warp_val, d);
            }

            /* Skip the atomic entirely if nobody in the warp saw coverage. */
            if (warp_val == 0) continue;

            u32 *global_ptr = (u32 *)__builtin_assume_aligned(
                global_cov + base, _Alignof(u32));

            /* Lane 0 atom.or's the merged word; broadcasts the old value. */
            u32 old0 = 0;
            if (laneid == 0) {
                old0 = __coqui_atom_or_u32(global_ptr, warp_val);
            }
            u32 old = __coqui_shfl_bcast_u32_sync(mask, old0);

            /* Novelty: did THIS thread contribute a bit not already in old? */
            if (classified & ~old) found_new = 1;
        }
    } else {
        /* Partial warp: per-thread atom.or. No shuffle. */
        for (u64 w = 0; w < words; w++) {
            u64 base = w * 4;
            u32 *word_ptr = (u32 *)__builtin_assume_aligned(
                thread_cov + base, _Alignof(u32));
            u32 raw = *word_ptr;
            if (raw == 0) continue;
            *word_ptr = 0;

            u32 classified =
                (u32)__coqui_classify_lut[(u8)(raw)] |
                ((u32)__coqui_classify_lut[(u8)(raw >> 8)]  << 8) |
                ((u32)__coqui_classify_lut[(u8)(raw >> 16)] << 16) |
                ((u32)__coqui_classify_lut[(u8)(raw >> 24)] << 24);

            u32 *global_ptr = (u32 *)__builtin_assume_aligned(
                global_cov + base, _Alignof(u32));
            u32 old = __coqui_atom_or_u32(global_ptr, classified);
            if (classified & ~old) found_new = 1;
        }
    }

    return found_new;
}

/* ===-------------------------------------------------------------------===
 * FNV-1a over the (already classified + zeroed) cov map, sized by
 * cov_map_size. Used by asan_report to stamp crash_sig mid-execution.
 *
 * At kernel-exit time, coverage_evaluate has already zeroed the map, so
 * trace_sig returns the FNV offset basis for every clean thread — this is
 * fine because clean threads never consult crash_sig.
 *
 * At ASan trap time, the map is partially populated (whatever raw
 * counts the executed edges produced so far). trace_sig groups crashes at
 * the same parser site to the same signature. Uses the raw bytes — no
 * classification — because we want a deterministic hash of the path that
 * reached the crash, not the post-classified value.
 * ===-------------------------------------------------------------------=== */

#define COQUI_FNV32_OFFSET  0x811c9dc5u
#define COQUI_FNV32_PRIME   0x01000193u

u32 __coqui_trace_sig(u8 *map, u64 cov_map_size) {
    u32 h = COQUI_FNV32_OFFSET;
    const u64 words = cov_map_size / 4;
    u32 *m32 = (u32 *)map;
    for (u64 i = 0; i < words; i++) {
        u32 w = m32[i];
        if (w == 0) continue;
        h = (h ^ (u32)i) * COQUI_FNV32_PRIME;
        h = (h ^ w)      * COQUI_FNV32_PRIME;
    }
    return h;
}
