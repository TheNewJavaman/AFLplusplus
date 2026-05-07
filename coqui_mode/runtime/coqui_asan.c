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
#define ASAN_ERROR_HEAP_OVERFLOW    1
#define ASAN_ERROR_USE_AFTER_FREE   2
#define ASAN_ERROR_GLOBAL_OVERFLOW  3
#define ASAN_ERROR_SLAB_OVERFLOW    4

/* Maximum globals we can track. cJSON uses ~10; a TSan-scale target
 * uses a few hundred. 1024 is the same ceiling the reference runtime
 * uses and comfortably covers all evaluation targets. */
#define ASAN_MAX_GLOBALS 1024

/* Maximum live slab allocations tracked at once. Slots are append-only
 * (free poisons entry to a sentinel; index is not recycled), so this is
 * a cumulative-allocation ceiling across the kernel run, not a snapshot
 * of live blocks. 4096 covers every evaluation harness observed so far:
 * bzip2/libxml2/libpng with 2 GiB slab pools routinely cap at ~1-2k
 * allocations per kernel launch. */
#define ASAN_MAX_SLAB_DESCS 4096

/* Slab pool allocator function pointers (set by slab runtime if linked).
 * If the slab runtime is linked in, coqui_slab.c's __coqui_slab_setup()
 * registers its malloc/free pair here. For non-slab targets the pointers
 * stay NULL and the ASan slab paths short-circuit. */
static void *(*g_slab_malloc)(unsigned long) = (void *)0;
static void  (*g_slab_free)(void *)          = (void *)0;

__attribute__((nothrow))
void __coqui_asan_register_slab(void *(*m)(unsigned long), void (*f)(void *)) {
    g_slab_malloc = m;
    g_slab_free   = f;
}

/* Slab pool globals (defined in coqui_slab.c when linked). Declared weak
 * so the ASan runtime links cleanly for targets without a slab runtime;
 * every consumer guards on `__coqui_slab_pool` being non-NULL. */
extern char *__coqui_slab_pool __attribute__((weak));
extern unsigned long __coqui_slab_pool_size __attribute__((weak));
extern char *__coqui_slab_shadow __attribute__((weak));

/* Slab-shadow poison/unpoison (byte-wise; the slab pool rarely needs
 * the asan-heap vector write path because slab allocations are uncommon
 * relative to heap and the extra code size matters more than throughput
 * here). */
static void slab_shadow_poison(unsigned long slab_off, unsigned long n,
                                u8 value) {
    if (unlikely(!__coqui_slab_shadow)) return;
    u8 *shadow = (u8 *)__coqui_slab_shadow;
    unsigned long start = slab_off >> 3;
    unsigned long end   = (slab_off + n) >> 3;
    for (unsigned long i = start; i < end; i++) shadow[i] = value;
}

static void slab_shadow_unpoison(unsigned long slab_off, unsigned long n) {
    if (unlikely(!__coqui_slab_shadow)) return;
    u8 *shadow = (u8 *)__coqui_slab_shadow;
    unsigned long aligned = n & ~7UL;
    unsigned long partial = n & 7u;
    unsigned long start   = slab_off >> 3;
    unsigned long full    = (slab_off + aligned) >> 3;
    for (unsigned long i = start; i < full; i++) shadow[i] = ASAN_CLEAN;
    if (partial) shadow[full] = (u8)partial;
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
 * Global variable red-zone tracking
 *
 * The IR pass (coqui::runAsanGlobals) emits a unified descriptor table
 * listing two kinds of entries:
 *
 *   Non-pool (pool_stride == 0):
 *     Each qualifying user global `T g;` is replaced with a padded struct
 *     `{ T orig; u8 rz[32]; }` so there is a physical gap after the user
 *     bytes. The descriptor's `beg` points at the padded global (= the
 *     absolute address shared by all threads).
 *
 *   Pool (pool_stride > 0):
 *     Writable globals that StaticGlobals pooled into the per-thread slab
 *     have no single absolute address; each thread's slab lives at
 *     `__coqui_global_statics_pool_base + tid * pool_stride` and the
 *     pooled entry sits at `+ pool_offset` within that slab. StaticGlobals
 *     reserves a 32-byte trailing gap between pooled entries so the
 *     descriptor's total_size (= user_size + 32) still covers real poison
 *     bytes. `beg` is NULL for pool entries; asan_check_global() resolves
 *     the per-thread real_beg at check time.
 *
 * A one-shot call to __coqui_asan_register_globals() is injected at the
 * top of __coqui_fuzz_kernel, so the table is populated before any user
 * code runs. Descriptors carry their pool-base-relative form, so all
 * threads write identical values — the per-thread address variation
 * happens inside asan_check_global().
 *
 * We cannot rely on the existing heap shadow: globals live in NVPTX
 * .global memory, outside any per-thread heap. Instead, the outlined
 * fast-path helper now routes out-of-heap accesses to the slow path
 * (instead of returning clean) so we can check the global table here.
 * ===-------------------------------------------------------------------=== */

static struct __coqui_asan_global_desc
    __coqui_asan_globals[ASAN_MAX_GLOBALS];
static unsigned long __coqui_asan_num_globals;

/* Extern ptr bound by the host launcher at module load (see
 * afl-fuzz-coqui.c phase 9). Declared here so asan_check_global() can
 * resolve per-thread pool-entry addresses.
 *
 * This symbol is defined by StaticGlobals.cpp as `extern ptr
 * __coqui_global_statics_pool_base` and written at runtime by the host
 * via cuModuleGetGlobal + cuMemcpyHtoD. For targets with no pooled
 * globals the symbol is still declared (we take its address here only if
 * at least one pool-kind descriptor exists) — the `if (stride)` gate
 * below ensures we never dereference it in the no-pool case. */
extern u8 *__coqui_global_statics_pool_base;

__attribute__((nothrow))
void __coqui_asan_register_globals(const struct __coqui_asan_global_desc *descs,
                                    unsigned long count) {
    /* All GPU threads call this concurrently with identical arguments,
     * so straightforward per-index copies are idempotent — no atomics. */
    unsigned long n = count < ASAN_MAX_GLOBALS ? count : ASAN_MAX_GLOBALS;
    for (unsigned long i = 0; i < n; i++)
        __coqui_asan_globals[i] = descs[i];
    __coqui_asan_num_globals = n;
}

/*
 * asan_check_global — returns 1 if the access [addr, addr+size) overlaps
 * any registered global's red zone, 0 otherwise.
 *
 * We flag two cases:
 *   (a) the access starts inside the red zone [user_end, total_end),
 *   (b) the access starts inside the user region but extends past
 *       user_end (partial overflow).
 *
 * For pool-kind descriptors (pool_stride > 0) the per-thread real_beg
 * is computed on the fly as `pool_base + tid*stride + pool_offset`, so
 * each thread checks against its own slab's pooled-entry locations.
 *
 * Linear scan is fine: typical targets register well under 100 globals,
 * and the slow path only runs on shadow misses / out-of-heap addresses.
 */
static int asan_check_global(unsigned long addr, unsigned long size) {
    unsigned long n      = __coqui_asan_num_globals;
    unsigned long pool   = (unsigned long)__coqui_global_statics_pool_base;
    unsigned long tid    = (unsigned long)__coqui_fuzz_tid();
    for (unsigned long i = 0; i < n; i++) {
        unsigned long stride = __coqui_asan_globals[i].pool_stride;
        unsigned long beg;
        if (stride) {
            /* Pool-kind: resolve this thread's slab-relative location. */
            beg = pool + tid * stride + __coqui_asan_globals[i].pool_offset;
        } else {
            beg = (unsigned long)__coqui_asan_globals[i].beg;
        }
        unsigned long user_end  = beg + __coqui_asan_globals[i].user_size;
        unsigned long total_end = beg + __coqui_asan_globals[i].total_size;

        if (addr >= user_end && addr < total_end)
            return 1;
        if (addr >= beg && addr < user_end && (addr + size) > user_end)
            return 1;
    }
    return 0;
}

/* ===-------------------------------------------------------------------===
 * Slab-pool red-zone tracking
 *
 * The slab pool is a host-allocated region shared by all GPU threads; no
 * per-thread shadow exists for it (unlike the heap, where shadow is a
 * suffix of each thread's heap slice). Instead, every
 * __coqui_asan_slab_malloc appends a descriptor to a kernel-wide table
 * and the slowpath linearly scans the table after global-checks miss.
 *
 * The table is append-only: __coqui_asan_slab_free poisons the `beg`
 * field to NULL (slowpath skips NULL rows). Slot indices are never
 * recycled so lookups remain branch-free and immune to ABA races.
 *
 * Concurrency: __coqui_asan_slab_num_descs is updated via a generic-space
 * atomic add (matches every other coqui device counter). Each thread that
 * grabs a slot writes the payload into its private row — no CAS loop
 * needed because the index is unique per winner.
 * ===-------------------------------------------------------------------=== */

static struct __coqui_asan_slab_desc
    __coqui_asan_slab_descs[ASAN_MAX_SLAB_DESCS];
static unsigned int __coqui_asan_num_slab_descs;

/* atomic add on generic pointer — matches every other coqui counter. */
static unsigned int asan_atom_add_gen(volatile unsigned int *addr,
                                       unsigned int val) {
    unsigned int old;
    __asm__ volatile("atom.add.u32 %0, [%1], %2;"
                     : "=r"(old) : "l"(addr), "r"(val));
    return old;
}

/*
 * asan_slab_register — called from __coqui_asan_slab_malloc. Returns slot
 * index if the descriptor table has room, ASAN_MAX_SLAB_DESCS otherwise
 * (ASan gracefully degrades: a slab alloc beyond the ceiling still
 * succeeds, just without detection).
 */
static unsigned int asan_slab_register(const void *beg,
                                        unsigned long user_size,
                                        unsigned long total_size) {
    unsigned int idx = asan_atom_add_gen(&__coqui_asan_num_slab_descs, 1u);
    if (unlikely(idx >= ASAN_MAX_SLAB_DESCS)) return ASAN_MAX_SLAB_DESCS;
    __coqui_asan_slab_descs[idx].beg        = beg;
    __coqui_asan_slab_descs[idx].user_size  = user_size;
    __coqui_asan_slab_descs[idx].total_size = total_size;
    return idx;
}

/*
 * asan_slab_lookup — find the slab descriptor whose [beg, beg+total_size)
 * contains `addr`. Returns the row index or ASAN_MAX_SLAB_DESCS if not
 * found. Linear scan: typical slab-pool-enabled runs register low-hundreds
 * of descriptors and the slowpath only runs on shadow misses.
 */
static unsigned int asan_slab_lookup(unsigned long addr) {
    unsigned int n = __coqui_asan_num_slab_descs;
    if (n > ASAN_MAX_SLAB_DESCS) n = ASAN_MAX_SLAB_DESCS;
    for (unsigned int i = 0; i < n; i++) {
        unsigned long beg = (unsigned long)__coqui_asan_slab_descs[i].beg;
        if (beg == 0) continue;   /* freed slot */
        unsigned long tend = beg + __coqui_asan_slab_descs[i].total_size;
        if (addr >= beg && addr < tend) return i;
    }
    return ASAN_MAX_SLAB_DESCS;
}

/*
 * asan_check_slab — returns 1 if [addr, addr+size) overlaps any slab
 * descriptor's red zone (equivalently, any byte past user_end within
 * total_size). Matches the two-case logic used by asan_check_global.
 */
static int asan_check_slab(unsigned long addr, unsigned long size) {
    unsigned int n = __coqui_asan_num_slab_descs;
    if (n > ASAN_MAX_SLAB_DESCS) n = ASAN_MAX_SLAB_DESCS;
    for (unsigned int i = 0; i < n; i++) {
        unsigned long beg = (unsigned long)__coqui_asan_slab_descs[i].beg;
        if (beg == 0) continue;  /* freed slot */
        unsigned long user_end  = beg + __coqui_asan_slab_descs[i].user_size;
        unsigned long total_end = beg + __coqui_asan_slab_descs[i].total_size;

        if (addr >= user_end && addr < total_end)
            return 1;
        if (addr >= beg && addr < user_end && (addr + size) > user_end)
            return 1;
    }
    return 0;
}

/* Slab-pool shadow check for tier-2 accesses. Mirrors asan_check_access()
 * structure but runs against __coqui_slab_shadow using a slab-pool-relative
 * offset. Returns 1 on poisoned access, writing *out_error_type with the
 * error code. Returns 0 if the pointer is outside the slab pool. */
static int asan_check_slab_shadow(void *ptr, u8 access_size,
                                   int *out_error_type) {
    if (unlikely(!__coqui_slab_pool || !__coqui_slab_shadow)) return 0;
    unsigned long a = (unsigned long)ptr;
    unsigned long sp = (unsigned long)__coqui_slab_pool;
    if (unlikely(a < sp || a + access_size > sp + __coqui_slab_pool_size)) return 0;

    unsigned long off     = a - sp;
    unsigned long end_off = off + (unsigned long)access_size - 1u;
    unsigned long si      = off >> 3;
    unsigned long si_end  = end_off >> 3;
    u8 *shadow = (u8 *)__coqui_slab_shadow;

    for (unsigned long i = si; i <= si_end; i++) {
        u8 s = shadow[i];
        if (s == ASAN_CLEAN) continue;
        if (s >= 0x80u) {
            *out_error_type = (s == ASAN_FREED) ? ASAN_ERROR_USE_AFTER_FREE
                                                : ASAN_ERROR_SLAB_OVERFLOW;
            return 1;
        }
        unsigned long end_in_granule = (i == si_end) ? (end_off & 7u) + 1u : 8u;
        if (end_in_granule > (unsigned long)s) {
            *out_error_type = ASAN_ERROR_SLAB_OVERFLOW;
            return 1;
        }
    }
    return 0;
}

/* ===-------------------------------------------------------------------===
 * Error reporting
 * ===-------------------------------------------------------------------=== */

/*
 * Per-thread status array (defined in coqui_runtime.c).
 * Declared at file scope to avoid clang complaints about local externs.
 */
extern coqui_status_t *__coqui_status_array;

/* `noinline, cold, nothrow`: only invoked from slow-path arms that already
 * detected poison. Out-of-line keeps the crash-sig fold + status write
 * out of the hot icache. */
__attribute__((noinline, cold, nothrow))
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

    /* Fast path: pointer is outside the usable heap — not our concern.
     * The slow path is reached almost exclusively from out-of-heap accesses
     * (globals / slab); the in-heap shadow miss is comparatively rare. */
    if (unlikely(a < heap_start || a + access_size > heap_start + heap_sz))
        return 0;

    unsigned long off     = a - heap_start;
    unsigned long end_off = off + (unsigned long)access_size - 1u;
    unsigned long si      = off >> 3;
    unsigned long si_end  = end_off >> 3;

    for (unsigned long i = si; i <= si_end; i++) {
        u8 s = shadow[i];

        if (likely(s == ASAN_CLEAN))
            continue;   /* fully accessible granule */

        if (unlikely(s >= 0x80u)) {
            /* Fully poisoned — distinguish freed vs red zone. */
            *out_error_type = (s == ASAN_FREED) ? ASAN_ERROR_USE_AFTER_FREE
                                                 : ASAN_ERROR_HEAP_OVERFLOW;
            return 1;
        }

        /* Partial granule: first s bytes accessible.
           Check whether this access touches any byte beyond offset s. */
        unsigned long end_in_granule = (i == si_end) ? (end_off & 7u) + 1u : 8u;
        if (unlikely(end_in_granule > (unsigned long)s)) {
            *out_error_type = ASAN_ERROR_HEAP_OVERFLOW;
            return 1;
        }
    }

    return 0;
}

/* ===-------------------------------------------------------------------===
 * Size-specialized check functions (called by instrumented code)
 *
 * Split into a FAST PATH (inlined at every call site) and a SLOW PATH
 * (outlined, shared by all call sites of the same size). Pattern follows
 * coqui upstream commits 26c9110 / 61bd5e4 / b5baaad: the common case
 * (access outside heap, or shadow byte == ASAN_CLEAN) is ~5-10 SASS insns
 * inline; the rare case (poisoned / partial granule) falls through to a
 * single outlined slow-path function that runs the full asan_check_access
 * multi-granule walk.
 *
 * Fast path:
 *   1. pointer not in heap → clean (not our concern)
 *   2. first granule shadow byte == 0 AND, if access crosses granule,
 *      second granule shadow byte == 0 → clean
 *   otherwise → fall through to slow path
 *
 * This is CONSERVATIVE: a partial granule (s in 1..7) with a clean in-bounds
 * access triggers a slow-path call even though the full checker would return
 * clean too. That is safe (correctness preserved) and the slow path decides
 * correctly.
 * ===-------------------------------------------------------------------=== */

/* `always_inline, nothrow`: hot fast path. Caller folds this into a
 * straight-line range + shadow byte check. */
static inline __attribute__((always_inline, nothrow))
int asan_fastpath_ok(void *ptr, u8 access_size) {
    u8  *heap        = __coqui_heap_base();
    u32  heap_sz     = __coqui_heap_size();
    unsigned long a          = (unsigned long)ptr;
    unsigned long heap_start = (unsigned long)heap;

    /* Pointer outside the usable heap — not our concern, treat as clean.
     * Inlined fast-path sees both heap and out-of-heap traffic; bias the
     * branch toward "in heap" so the shadow check is the fall-through.
     * NB: this differs from asan_check_access (slow-path) where reaching
     * that body already filtered out most in-heap shadow hits. */
    if (unlikely(a < heap_start || a + access_size > heap_start + heap_sz))
        return 1;

    unsigned long off    = a - heap_start;
    u8 *shadow           = __coqui_shadow_base();
    u8 s                 = shadow[off >> 3];

    /* First granule must be fully clean. */
    if (unlikely(s != ASAN_CLEAN)) return 0;

    /* If the access crosses into a second granule, check that one too. */
    unsigned long end_off = off + (unsigned long)access_size - 1u;
    if (unlikely((end_off >> 3) != (off >> 3))) {
        s = shadow[end_off >> 3];
        if (unlikely(s != ASAN_CLEAN)) return 0;
    }
    return 1;
}

/*
 * Slow-path wrappers. The fast-path helper calls into here when:
 *   - the access is in the heap region AND the shadow byte is nonzero, or
 *   - the access is outside the heap (global / other addrspace(0) data).
 *
 * For in-heap shadow hits, asan_check_access walks the granules and
 * reports heap-buffer-overflow / use-after-free. For out-of-heap
 * accesses, it returns 0 — in that case we fall back to scanning the
 * global-variable descriptor table so OOB into global red zones gets
 * caught.
 */
/* `noinline, cold, nothrow`: outlined slow path; only entered when the
 * fast-path shadow check or range check failed. `cold` lets llc place
 * these at the end of the parent fuzz kernel so the hot fall-through
 * (clean access) is straight-line. `nothrow` matches every other C
 * runtime entry point. */
#define SLOWPATH_IMPL(N)                                                \
    __attribute__((noinline, cold, nothrow))                             \
    void __coqui_asan_slowpath_load_##N(void *ptr) {                     \
        int err = 0;                                                     \
        if (asan_check_access(ptr, (u8)(N), &err)) asan_report(err);   \
        if (asan_check_slab_shadow(ptr, (u8)(N), &err)) asan_report(err); \
        if (asan_check_global((unsigned long)ptr, (u8)(N)))              \
            asan_report(ASAN_ERROR_GLOBAL_OVERFLOW);                     \
        if (asan_check_slab((unsigned long)ptr, (u8)(N)))                \
            asan_report(ASAN_ERROR_SLAB_OVERFLOW);                       \
    }                                                                    \
    __attribute__((noinline, cold, nothrow))                             \
    void __coqui_asan_slowpath_store_##N(void *ptr) {                    \
        int err = 0;                                                     \
        if (asan_check_access(ptr, (u8)(N), &err)) asan_report(err);   \
        if (asan_check_slab_shadow(ptr, (u8)(N), &err)) asan_report(err); \
        if (asan_check_global((unsigned long)ptr, (u8)(N)))              \
            asan_report(ASAN_ERROR_GLOBAL_OVERFLOW);                     \
        if (asan_check_slab((unsigned long)ptr, (u8)(N)))                \
            asan_report(ASAN_ERROR_SLAB_OVERFLOW);                       \
    }

SLOWPATH_IMPL(1)
SLOWPATH_IMPL(2)
SLOWPATH_IMPL(4)
SLOWPATH_IMPL(8)

/* `always_inline, nothrow`: fast-path wrappers around the size-specialized
 * checker. These are the "old" inline entry points retained for any
 * pre-pass call site; the post-pass instrumentation prefers the outlined
 * fast helpers in Asan.cpp. */
#define CHECK_IMPL(N)                                                   \
    __attribute__((always_inline, nothrow))                              \
    void __coqui_asan_check_load_##N(void *ptr) {                       \
        if (asan_fastpath_ok(ptr, (u8)(N))) return;                     \
        __coqui_asan_slowpath_load_##N(ptr);                             \
    }                                                                    \
    __attribute__((always_inline, nothrow))                              \
    void __coqui_asan_check_store_##N(void *ptr) {                      \
        if (asan_fastpath_ok(ptr, (u8)(N))) return;                     \
        __coqui_asan_slowpath_store_##N(ptr);                            \
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
 * Two-tier: per-thread heap first (tier 1), then shared slab pool on
 * exhaustion (tier 2). On full exhaustion of both tiers, stamp
 * trap_reason = COQUI_TRAP_OOM and exit the thread (the host re-runs the
 * input on the CPU forkserver).
 *
 * Physical layout in the heap (both tiers use the same layout):
 *   [left_redzone (16)] [user data (size)] [right_redzone (16)]
 *
 * Shadow state after allocation (tier 1 heap):
 *   left_redzone  → ASAN_REDZONE_POISON
 *   user data     → ASAN_CLEAN (+ partial tail if size not multiple of 8)
 *   right_redzone → ASAN_REDZONE_POISON
 * Tier 2 (slab) uses a separate shadow arena (__coqui_slab_shadow).
 *
 * `nothrow`: all C runtime entry points. The body is too large after slab
 * integration to mark always_inline at every user-malloc call site
 * (poison-range loops + slab tier path), so we leave inlining to LLVM's
 * heuristics and rely on the fast-path ASan helpers (which ARE
 * always_inline) for the per-access common case.
 */
__attribute__((nothrow))
void *__coqui_asan_malloc(unsigned long size) {
    unsigned long padded = ASAN_LEFT_REDZONE + size + ASAN_RIGHT_REDZONE;

    /* Tier 1: per-thread heap. The heap is sized to satisfy the common
     * mid-execution allocation pattern; falling through to the slab tier
     * only happens for targets that exhaust the per-thread budget. */
    void *raw = __coqui_malloc(padded);
    if (likely(raw != (void *)0)) {
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

    /* Tier 2: slab pool (if a slab runtime is linked and has registered a
     * malloc_fn). g_slab_malloc stays NULL for non-slab targets, in which
     * case we proceed directly to the OOM trap. */
    if (unlikely(g_slab_malloc && __coqui_slab_pool && __coqui_slab_pool_size)) {
        void *slab_raw = g_slab_malloc(padded);
        if (likely(slab_raw != (void *)0)) {
            u8 *user = (u8 *)slab_raw + ASAN_LEFT_REDZONE;
            unsigned long slab_off =
                (unsigned long)((u8 *)slab_raw - (u8 *)__coqui_slab_pool);
            slab_shadow_poison(slab_off,
                               ASAN_LEFT_REDZONE,
                               ASAN_REDZONE_POISON);
            slab_shadow_unpoison(slab_off + ASAN_LEFT_REDZONE, size);
            unsigned long aligned = (size + 7UL) & ~7UL;
            slab_shadow_poison(slab_off + ASAN_LEFT_REDZONE + aligned,
                               ASAN_RIGHT_REDZONE,
                               ASAN_REDZONE_POISON);
            return (void *)user;
        }
    }

    /* Both tiers exhausted. Stamp trap_reason = OOM and exit this thread.
     * The host-side post-batch scan finds trap_reason == COQUI_TRAP_OOM
     * and re-runs the input on the CPU forkserver where no GPU memory
     * constraint applies. */
    __coqui_trap_with_reason(COQUI_TRAP_OOM);
    return (void *)0; /* unreachable */
}

/*
 * __coqui_asan_free — free a pointer previously returned by __coqui_asan_malloc.
 *
 * Poisons the entire allocation (left_redzone + user + right_redzone) with
 * ASAN_FREED so that use-after-free is detected.  Bounds-checks block before
 * touching the header to avoid crashing on bad pointers.
 *
 * `nothrow`: all C runtime entry points. Like __coqui_asan_malloc, body is
 * too large to always_inline (slab range check + slab_shadow_poison +
 * heap path). LLVM's inliner picks this up at the call sites where it's
 * profitable.
 */
__attribute__((nothrow))
void __coqui_asan_free(void *ptr) {
    if (unlikely(!ptr)) return;

    u8 *user  = (u8 *)ptr;
    u8 *raw   = user - ASAN_LEFT_REDZONE;

    /* Slab-pool routing: if ptr is in the slab range, free via the slab
     * allocator and poison slab shadow. Block header is [size:u32] 8 bytes
     * before user_raw (matching coqui_slab.c's layout). The slab path is
     * the rare case — most allocations live in the per-thread heap. */
    if (unlikely(g_slab_free && __coqui_slab_pool && __coqui_slab_pool_size)) {
        u8 *sp_lo = (u8 *)__coqui_slab_pool;
        u8 *sp_hi = sp_lo + __coqui_slab_pool_size;
        if (user >= sp_lo && user < sp_hi) {
            unsigned long slab_off = (unsigned long)(raw - sp_lo);
            /* Slab block header sits 8B before `raw` (= SLAB_BLOCK_HDR_SIZE
             * before the leading red zone).  Decode the actual allocation
             * byte count via slab_blk_actual_bytes(): for sub-block allocs
             * this is block_size (= 8 + aligned_user); for multi-slab allocs
             * the header encodes n_slabs*SLAB_SIZE in bits [4..31] with a
             * 0xF low-nibble sentinel — treating it directly as block_size
             * would over-poison by ~16× and corrupt neighbour allocations'
             * shadow bytes.  Clamp to the remaining pool to guard against
             * a corrupted header walking off the end. */
            u8 *slab_hdr = raw - 8;
            u32 blk = (slab_hdr >= sp_lo && slab_hdr < sp_hi)
                        ? *(u32 *)slab_hdr : 0u;
            unsigned long actual = slab_blk_actual_bytes(blk);
            unsigned long payload = (actual > SLAB_BLOCK_HDR_SIZE)
                                      ? (actual - SLAB_BLOCK_HDR_SIZE) : 0u;
            if (unlikely(slab_off + payload > __coqui_slab_pool_size)) {
                unsigned long safe = (__coqui_slab_pool_size > slab_off)
                                       ? (__coqui_slab_pool_size - slab_off) : 0;
                payload = safe & ~7UL;
            }
            if (likely(payload)) slab_shadow_poison(slab_off, payload, ASAN_FREED);
            g_slab_free((void *)raw);
            return;
        }
    }

    u8 *block = raw - BLOCK_HDR_SIZE;

    /* Bounds check: block header must be within the heap region. */
    u8 *heap_lo = __coqui_heap_base();
    u8 *heap_hi = heap_lo + __coqui_heap_size();
    if (unlikely(block < heap_lo || block >= heap_hi)) return;

    /* Read the original total block size from the header (includes BLOCK_HDR_SIZE). */
    u32 full_sz = *(u32 *)block;
    if (unlikely(full_sz < BLOCK_HDR_SIZE)) return;   /* sanity */

    /* Poison the entire payload (left_redzone + user + right_redzone). */
    unsigned long payload  = (unsigned long)(full_sz - BLOCK_HDR_SIZE);
    unsigned long raw_off  = (unsigned long)(raw - heap_lo);
    asan_poison_range(raw_off, payload, ASAN_FREED);

    __coqui_free(raw);
}

/* Internal byte-wise copy/set for the realloc/calloc paths.
 * We do NOT call __coqui_memcpy / __coqui_memset via external decls because
 * those live in coqui_libc.c and would be subject to ASan load/store
 * instrumentation. Keeping these static local mirrors coqui_memory.c:150. */
static void asan_memset_u8(void *dst, int c, unsigned long n) {
    u8 *d = (u8 *)dst;
    for (unsigned long i = 0; i < n; i++) d[i] = (u8)c;
}

static void asan_memcpy_u8(void * __restrict__ dst, const void * __restrict__ src, unsigned long n) {
    u8 *d = (u8 *)dst;
    const u8 *s = (const u8 *)src;
    for (unsigned long i = 0; i < n; i++) d[i] = s[i];
}

/*
 * __coqui_asan_calloc — calloc() with red zones.
 *
 * Computes nmemb * size with overflow-safe saturation (NULL on overflow),
 * delegates to __coqui_asan_malloc so the result participates in the usual
 * cross-tier fall-through (heap → slab → OOM trap), then zero-fills only
 * the user region. Red zones stay poisoned (asan_*malloc already set them
 * to ASAN_REDZONE_POISON for both tiers).
 */
__attribute__((nothrow))
void *__coqui_asan_calloc(unsigned long nmemb, unsigned long size) {
    unsigned long total;
    if (unlikely(__builtin_umull_overflow(nmemb, size, &total)))
        return (void *)0;

    void *ptr = __coqui_asan_malloc(total);
    if (unlikely(!ptr)) return (void *)0;

    /* Zero ONLY the user region — red zones were poisoned in the shadow
     * by __coqui_asan_malloc and must remain untouched. */
    asan_memset_u8(ptr, 0, total);
    return ptr;
}

/*
 * __coqui_asan_realloc — realloc() with cross-tier routing.
 *
 * Algorithm mirrors the reference coqui runtime and the shared-slab-pool
 * design doc (search "ASan-Wrapped Slab Realloc"):
 *
 *   ptr == NULL          → __coqui_asan_malloc(new_size)
 *   new_size == 0        → __coqui_asan_free(ptr); return NULL
 *   ptr in heap range    → read old_size from [raw - BLOCK_HDR_SIZE] u32,
 *                          subtract header + 2*red_zones; malloc + copy + free
 *   ptr in slab range    → read old_size from [raw - 8] u32 (slab block size
 *                          header sits 8B before raw, which is 24B before the
 *                          user pointer), subtract header + 2*red_zones;
 *                          malloc + copy + free
 *   otherwise            → NULL (invalid pointer)
 *
 * No in-place extension: we always malloc-new + memcpy + free-old. This is
 * simpler and lets the new allocation take whichever tier it can —
 * matching the coqui reference semantics for cross-tier reallocs (see
 * ~/coqui/runtime/coqui_fuzz_asan.c and
 * ~/coqui/docs/superpowers/specs/2026-03-25-shared-slab-pool-allocator-design.md).
 *
 * `noinline, nothrow`: large body covering both tiers + memcpy loop;
 * keeping it out-of-line means each user realloc site is a single call.
 * `nothrow` matches all C runtime functions.
 */
__attribute__((noinline, nothrow))
void *__coqui_asan_realloc(void *ptr, unsigned long new_size) {
    if (unlikely(!ptr)) return __coqui_asan_malloc(new_size);
    if (unlikely(new_size == 0)) {
        __coqui_asan_free(ptr);
        return (void *)0;
    }

    u8 *user = (u8 *)ptr;
    u8 *raw  = user - ASAN_LEFT_REDZONE;

    unsigned long old_size;

    /* Slab-pool routing first: the slab block header lives 8B before raw.
     * Match __coqui_asan_free's slab-range check (see line 680). */
    u8 *sp_lo = (u8 *)__coqui_slab_pool;
    u8 *sp_hi = sp_lo + (__coqui_slab_pool ? __coqui_slab_pool_size : 0);
    if (unlikely(__coqui_slab_pool && __coqui_slab_pool_size &&
                 user >= sp_lo && user < sp_hi)) {
        /* Slab block layout handed out by __coqui_asan_slab_malloc /
         * __coqui_asan_malloc's tier-2 fall-through:
         *   [slab hdr: u32] [left_rz (16)] [user] [right_rz (16)]
         * For sub-block allocs: slab hdr = 8 + total_aligned (byte count).
         * For multi-slab allocs: slab hdr encodes n_slabs*SLAB_SIZE in bits
         * [4..31] with 0xF low-nibble sentinel — treating it directly as a
         * byte count would make old_size ~16× too large and cause an OOB
         * read from user memory during the memcpy below.
         * slab_blk_actual_bytes() decodes both cases correctly. */
        u8 *slab_hdr = raw - 8;
        if (unlikely(slab_hdr < sp_lo || slab_hdr >= sp_hi)) return (void *)0;
        u32 blk = *(u32 *)slab_hdr;
        unsigned long actual = slab_blk_actual_bytes(blk);
        if (unlikely(actual < SLAB_BLOCK_HDR_SIZE + ASAN_LEFT_REDZONE + ASAN_RIGHT_REDZONE))
            return (void *)0;
        old_size = actual - SLAB_BLOCK_HDR_SIZE
                   - ASAN_LEFT_REDZONE - ASAN_RIGHT_REDZONE;
    } else {
        /* Per-thread heap: mirror __coqui_asan_free's header read (line 709). */
        u8 *block = raw - BLOCK_HDR_SIZE;
        u8 *heap_lo = __coqui_heap_base();
        u8 *heap_hi = heap_lo + __coqui_heap_size();
        if (unlikely(block < heap_lo || block >= heap_hi)) return (void *)0;
        u32 full_sz = *(u32 *)block;
        if (unlikely(full_sz < BLOCK_HDR_SIZE + ASAN_LEFT_REDZONE + ASAN_RIGHT_REDZONE))
            return (void *)0;
        old_size = (unsigned long)(full_sz - BLOCK_HDR_SIZE)
                   - ASAN_LEFT_REDZONE - ASAN_RIGHT_REDZONE;
    }

    void *new_ptr = __coqui_asan_malloc(new_size);
    if (unlikely(!new_ptr)) return (void *)0;

    unsigned long copy_size = (old_size < new_size) ? old_size : new_size;
    asan_memcpy_u8(new_ptr, ptr, copy_size);
    __coqui_asan_free(ptr);
    return new_ptr;
}

/* ===-------------------------------------------------------------------===
 * Slab-pool ASan wrappers
 *
 * Physical layout handed out to the caller:
 *   [left redzone (16)] [user data (size)] [right redzone (16)]
 * The leading red zone doubles as a descriptor back-pointer: the first
 * 4 bytes store `slot + 1` (zero = untracked) so __coqui_asan_slab_free
 * can invalidate the descriptor without re-scanning the table.
 *
 * If no slab runtime has registered (g_slab_malloc == NULL — the current
 * default in coqui_mode), these return NULL / no-op, matching the
 * behavior of the built-in slab hook stubs.
 * ===-------------------------------------------------------------------=== */

/* Offset to the user pointer is 16B (ASAN_LEFT_REDZONE). The back-pointer
 * lives in the first 4B of the leading red zone.
 *
 * `nothrow`: C runtime entry. No always_inline — function pointer
 * registered with the slab runtime via __coqui_asan_register_slab and
 * called indirectly from __coqui_asan_malloc; an inline body would not
 * survive the indirect call. */
__attribute__((nothrow))
void *__coqui_asan_slab_malloc(unsigned long size) {
    if (unlikely(!g_slab_malloc)) return (void *)0;
    if (unlikely(size == 0)) size = 1;

    /* Align user size up to 8 bytes so right-redzone starts on a boundary. */
    unsigned long user_aligned = (size + 7UL) & ~7UL;
    unsigned long total        = ASAN_LEFT_REDZONE + user_aligned + ASAN_RIGHT_REDZONE;

    u8 *raw = (u8 *)g_slab_malloc(total);
    if (unlikely(!raw)) return (void *)0;

    u8 *user = raw + ASAN_LEFT_REDZONE;

    /* Register descriptor. total_size is measured from user-beg, so
     * [user_end, user_end + right_redzone) is the detectable red zone. */
    unsigned int slot = asan_slab_register((const void *)user,
                                            size,
                                            user_aligned + ASAN_RIGHT_REDZONE);

    /* Stash (slot + 1) in the first 4 bytes of the leading red zone for
     * O(1) teardown. 0 sentinel = "registration overflowed", free will
     * fall back to a linear lookup. */
    unsigned int tag = (slot < ASAN_MAX_SLAB_DESCS) ? (slot + 1u) : 0u;
    *(unsigned int *)raw = tag;

    return (void *)user;
}

__attribute__((nothrow))
void __coqui_asan_slab_free(void *ptr) {
    if (unlikely(!ptr)) return;
    if (unlikely(!g_slab_free)) return;

    u8 *user = (u8 *)ptr;
    u8 *raw  = user - ASAN_LEFT_REDZONE;

    /* Reject pointers that are not in the slab pool: no descriptor exists
     * for them, and calling g_slab_free with a heap/global pointer would
     * corrupt the slab allocator. */
    unsigned int hot_slot = *(unsigned int *)raw;
    if (likely(hot_slot > 0u && hot_slot <= ASAN_MAX_SLAB_DESCS)) {
        /* Fast path: clear the descriptor slot the hot-tag points to, and
         * verify the beg matches (guards against stale tags if the caller
         * hands us a random pointer). */
        unsigned int idx = hot_slot - 1u;
        if (likely(__coqui_asan_slab_descs[idx].beg == (const void *)user)) {
            __coqui_asan_slab_descs[idx].beg        = (void *)0;
            __coqui_asan_slab_descs[idx].user_size  = 0;
            __coqui_asan_slab_descs[idx].total_size = 0;
        }
    } else {
        /* Slow path: tag was 0 (registration overflow) or garbage. Scan
         * the table; if we find the user beg, clear it. */
        unsigned int idx = asan_slab_lookup((unsigned long)user);
        if (idx < ASAN_MAX_SLAB_DESCS &&
            __coqui_asan_slab_descs[idx].beg == (const void *)user) {
            __coqui_asan_slab_descs[idx].beg        = (void *)0;
            __coqui_asan_slab_descs[idx].user_size  = 0;
            __coqui_asan_slab_descs[idx].total_size = 0;
        }
    }

    /* Clear the hot tag so a subsequent use-after-free on this pointer
     * doesn't re-hit the fast path and mis-interpret a stale index. */
    *(unsigned int *)raw = 0u;

    g_slab_free((void *)raw);
}
