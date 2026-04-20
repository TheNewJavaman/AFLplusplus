/*
 * coqui_asan.c --- heap-only AddressSanitizer for coqui_mode.
 *
 * Ported from /coqui/runtime/coqui_fuzz_asan.c with adjustments for
 * the new memory layout: shadow is a suffix of the heap region.
 *
 * Layout (from coqui internals spec §4.2):
 *   [usable heap (8/9 of H)] [shadow (1/9 of H)]
 *
 * Where:
 *   __coqui_heap_base()   → start of usable heap
 *   __coqui_shadow_base() → start of shadow (already offset by usable_heap_size)
 *   __coqui_heap_size()   → total heap region size H
 *
 * Shadow encoding (AFL/ASan convention):
 *   0x00       → all 8 heap bytes accessible
 *   0x01-0x07  → first k bytes accessible, rest poisoned
 *   0xFA       → heap red zone
 *   0xFD       → freed heap region (use-after-free detection)
 *   0xFF       → fully poisoned (generic)
 *
 * Error codes (stored in coqui_status_t.asan_error):
 *   1  heap-buffer-overflow
 *   2  use-after-free
 */

#include "coqui_runtime.h"

/* Red-zone poison marker values (match coqui reference implementation) */
#define ASAN_CLEAN           0x00u
#define ASAN_REDZONE_POISON  0xFAu
#define ASAN_FREED           0xFDu

/* Red-zone sizes around every allocation */
#define ASAN_LEFT_REDZONE    16u
#define ASAN_RIGHT_REDZONE   16u

/* Block header size (matches coqui_memory.c: [size:u32][pad:u32]) */
#define BLOCK_HDR_SIZE       8u

/* Error codes */
#define ASAN_ERROR_HEAP_OVERFLOW  1
#define ASAN_ERROR_USE_AFTER_FREE 2

/* Global pool allocator function pointers (set by slab runtime if linked).
   Day-1: always NULL. */
static void *(*g_slab_malloc)(unsigned long) = (void *)0;
static void  (*g_slab_free)(void *)          = (void *)0;

void __coqui_asan_register_slab(void *(*m)(unsigned long), void (*f)(void *)) {
    g_slab_malloc = m;
    g_slab_free   = f;
}

/* ===-------------------------------------------------------------------===
 * Shadow helpers
 * ===-------------------------------------------------------------------=== */

/*
 * asan_poison_range — fill shadow bytes with a given poison value.
 *
 * Uses 8-byte stores where possible (8x fewer stores on NVPTX global mem).
 * heap_off and n are byte offsets/sizes in the usable heap.
 */
static void asan_poison_range(unsigned long heap_off, unsigned long n,
                               u8 value) {
    u8 *shadow = __coqui_shadow_base();
    unsigned long start = heap_off >> 3;
    unsigned long end   = (heap_off + n) >> 3;
    unsigned long cnt   = end - start;

    /* Build a word with `value` replicated in every byte. */
    unsigned long word = value;
    word |= word << 8;
    word |= word << 16;
    word |= word << 32;

    u8 *p = shadow + start;

    /* Align to 8-byte boundary. */
    while (cnt && ((unsigned long)p & 7u)) {
        *p++ = value;
        cnt--;
    }

    /* Bulk 8-byte stores. */
    unsigned long *wp = (unsigned long *)__builtin_assume_aligned(p, 8);
    while (cnt >= 8) {
        *wp++ = word;
        cnt -= 8;
    }

    /* Tail bytes. */
    p = (u8 *)wp;
    while (cnt--)
        *p++ = value;
}

/*
 * asan_unpoison_range — mark shadow bytes accessible (zero fill).
 *
 * Handles partial tail granule: if n is not a multiple of 8, the last
 * shadow byte gets the partial-accessible encoding (= n & 7).
 */
static void asan_unpoison_range(unsigned long heap_off, unsigned long n) {
    u8 *shadow = __coqui_shadow_base();

    unsigned long aligned    = n & ~7UL;
    unsigned long partial    = n & 7u;
    unsigned long start      = heap_off >> 3;
    unsigned long full_end   = (heap_off + aligned) >> 3;
    unsigned long cnt        = full_end - start;

    u8 *p = shadow + start;

    /* Align to 8-byte boundary. */
    while (cnt && ((unsigned long)p & 7u)) {
        *p++ = ASAN_CLEAN;
        cnt--;
    }

    /* Bulk 8-byte stores (value = 0). */
    unsigned long *wp = (unsigned long *)__builtin_assume_aligned(p, 8);
    while (cnt >= 8) {
        *wp++ = 0UL;
        cnt -= 8;
    }

    /* Tail. */
    p = (u8 *)wp;
    while (cnt--)
        *p++ = ASAN_CLEAN;

    /* Partial last granule: first `partial` bytes accessible, rest poisoned. */
    if (partial)
        shadow[full_end] = (u8)partial;
}

/* ===-------------------------------------------------------------------===
 * Error reporting
 * ===-------------------------------------------------------------------=== */

/*
 * Per-thread status array (defined in coqui_runtime.c).
 * Declared at file scope to avoid clang complaints about local externs.
 */
extern coqui_status_t *__coqui_status_array;

static void asan_report(int error_type) {
    u32 tid = __coqui_fuzz_tid();
    __coqui_status_array[tid].asan_error = (u8)error_type;

    /* Stamp a crash signature from the partial coverage map so the host
     * can dedup verify calls by signature. Mid-execution crashes leave the
     * cov_map partially written; threads that hit the same parser site
     * converge to the same partial map ⇒ same signature. */
    __coqui_status_array[tid].crash_sig = __coqui_trace_sig(__coqui_cov_base());

    __coqui_exit();
}

/* ===-------------------------------------------------------------------===
 * Access checker (shared by load and store)
 * ===-------------------------------------------------------------------=== */

/*
 * asan_check_access — returns 1 if the access [ptr, ptr+access_size) is
 * poisoned, writing *out_error_type with the appropriate error code.
 * Returns 0 if the access is clean or outside the heap (not our concern).
 *
 * Iterates over every shadow granule touched by the access, matching the
 * reference implementation's multi-granule loop.
 */
static int asan_check_access(void *ptr, u8 access_size, int *out_error_type) {
    u8  *heap     = __coqui_heap_base();
    u32  heap_sz  = __coqui_heap_size();
    u8  *shadow   = __coqui_shadow_base();

    unsigned long a = (unsigned long)ptr;
    unsigned long heap_start = (unsigned long)heap;

    /* Fast path: pointer is outside the usable heap — not our concern. */
    if (a < heap_start || a + access_size > heap_start + heap_sz)
        return 0;

    unsigned long off     = a - heap_start;
    unsigned long end_off = off + (unsigned long)access_size - 1u;
    unsigned long si      = off >> 3;
    unsigned long si_end  = end_off >> 3;

    for (unsigned long i = si; i <= si_end; i++) {
        u8 s = shadow[i];

        if (s == ASAN_CLEAN)
            continue;   /* fully accessible granule */

        if (s >= 0x80u) {
            /* Fully poisoned — distinguish freed vs red zone. */
            *out_error_type = (s == ASAN_FREED) ? ASAN_ERROR_USE_AFTER_FREE
                                                 : ASAN_ERROR_HEAP_OVERFLOW;
            return 1;
        }

        /* Partial granule: first s bytes accessible.
           Check whether this access touches any byte beyond offset s. */
        unsigned long end_in_granule = (i == si_end) ? (end_off & 7u) + 1u : 8u;
        if (end_in_granule > (unsigned long)s) {
            *out_error_type = ASAN_ERROR_HEAP_OVERFLOW;
            return 1;
        }
    }

    return 0;
}

/* ===-------------------------------------------------------------------===
 * Size-specialized check functions (called by instrumented code)
 * ===-------------------------------------------------------------------=== */

#define CHECK_IMPL(N)                                                   \
    void __coqui_asan_check_load_##N(void *ptr) {                       \
        int err = 0;                                                     \
        if (asan_check_access(ptr, (u8)(N), &err)) asan_report(err);   \
    }                                                                    \
    void __coqui_asan_check_store_##N(void *ptr) {                      \
        int err = 0;                                                     \
        if (asan_check_access(ptr, (u8)(N), &err)) asan_report(err);   \
    }

CHECK_IMPL(1)
CHECK_IMPL(2)
CHECK_IMPL(4)
CHECK_IMPL(8)

/* ===-------------------------------------------------------------------===
 * Allocator wrappers with red zones
 * ===-------------------------------------------------------------------=== */

/*
 * __coqui_asan_malloc — allocate `size` bytes with left and right red zones.
 *
 * Physical layout in the heap:
 *   [left_redzone (16)] [user data (size)] [right_redzone (16)]
 *
 * Shadow state after allocation:
 *   left_redzone  → ASAN_REDZONE_POISON
 *   user data     → ASAN_CLEAN (+ partial tail if size not multiple of 8)
 *   right_redzone → ASAN_REDZONE_POISON
 */
void *__coqui_asan_malloc(unsigned long size) {
    unsigned long padded = ASAN_LEFT_REDZONE + size + ASAN_RIGHT_REDZONE;
    void *raw = __coqui_malloc(padded);
    if (!raw) return (void *)0;

    u8 *user      = (u8 *)raw + ASAN_LEFT_REDZONE;
    u8 *heap_base = __coqui_heap_base();
    unsigned long raw_off = (u8 *)raw - heap_base;

    asan_poison_range(raw_off,                             ASAN_LEFT_REDZONE,
                      ASAN_REDZONE_POISON);
    asan_unpoison_range(raw_off + ASAN_LEFT_REDZONE,       size);
    asan_poison_range(raw_off + ASAN_LEFT_REDZONE + size,  ASAN_RIGHT_REDZONE,
                      ASAN_REDZONE_POISON);

    return (void *)user;
}

/*
 * __coqui_asan_free — free a pointer previously returned by __coqui_asan_malloc.
 *
 * Poisons the entire allocation (left_redzone + user + right_redzone) with
 * ASAN_FREED so that use-after-free is detected.  Bounds-checks block before
 * touching the header to avoid crashing on bad pointers.
 */
void __coqui_asan_free(void *ptr) {
    if (!ptr) return;

    u8 *user  = (u8 *)ptr;
    u8 *raw   = user - ASAN_LEFT_REDZONE;
    u8 *block = raw - BLOCK_HDR_SIZE;

    /* Bounds check: block header must be within the heap region. */
    u8 *heap_lo = __coqui_heap_base();
    u8 *heap_hi = heap_lo + __coqui_heap_size();
    if (block < heap_lo || block >= heap_hi) return;

    /* Read the original total block size from the header (includes BLOCK_HDR_SIZE). */
    u32 full_sz = *(u32 *)block;
    if (full_sz < BLOCK_HDR_SIZE) return;   /* sanity */

    /* Poison the entire payload (left_redzone + user + right_redzone). */
    unsigned long payload  = (unsigned long)(full_sz - BLOCK_HDR_SIZE);
    unsigned long raw_off  = (unsigned long)(raw - heap_lo);
    asan_poison_range(raw_off, payload, ASAN_FREED);

    __coqui_free(raw);
}
