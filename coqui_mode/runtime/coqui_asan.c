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
 * Day-1: always NULL — no device-side slab runtime is linked yet in
 * coqui_mode (the host allocates the slab buffer via cuMemAlloc but no
 * kernel-side allocator owns it). When a future slab runtime registers,
 * the ASan wrappers below become active. */
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
 * Global variable red-zone tracking
 *
 * The IR pass (coqui::runAsanGlobals) replaces each qualifying global
 * `T g;` with a padded struct `{ T orig; u8 rz[32]; }` so there is a
 * physical gap after the user bytes. It then emits a descriptor table
 * and a one-shot call to __coqui_asan_register_globals() inside
 * __coqui_fuzz_kernel. The descriptor records where each global's user
 * region ends and where its red zone ends, so we can detect an access
 * that lands in [user_end, total_end).
 *
 * We cannot rely on the existing heap shadow: globals live in NVPTX
 * .global memory, outside any per-thread heap. Instead, the outlined
 * fast-path helper now routes out-of-heap accesses to the slow path
 * (instead of returning clean) so we can check the global table here.
 * ===-------------------------------------------------------------------=== */

static struct __coqui_asan_global_desc
    __coqui_asan_globals[ASAN_MAX_GLOBALS];
static unsigned long __coqui_asan_num_globals;

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
 * Linear scan is fine: typical targets register well under 100 globals,
 * and the slow path only runs on shadow misses / out-of-heap addresses.
 */
static int asan_check_global(unsigned long addr, unsigned long size) {
    unsigned long n = __coqui_asan_num_globals;
    for (unsigned long i = 0; i < n; i++) {
        unsigned long beg       = (unsigned long)__coqui_asan_globals[i].beg;
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
    if (idx >= ASAN_MAX_SLAB_DESCS) return ASAN_MAX_SLAB_DESCS;
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

static inline __attribute__((always_inline))
int asan_fastpath_ok(void *ptr, u8 access_size) {
    u8  *heap        = __coqui_heap_base();
    u32  heap_sz     = __coqui_heap_size();
    unsigned long a          = (unsigned long)ptr;
    unsigned long heap_start = (unsigned long)heap;

    /* Pointer outside the usable heap — not our concern, treat as clean. */
    if (a < heap_start || a + access_size > heap_start + heap_sz)
        return 1;

    unsigned long off    = a - heap_start;
    u8 *shadow           = __coqui_shadow_base();
    u8 s                 = shadow[off >> 3];

    /* First granule must be fully clean. */
    if (s != ASAN_CLEAN) return 0;

    /* If the access crosses into a second granule, check that one too. */
    unsigned long end_off = off + (unsigned long)access_size - 1u;
    if ((end_off >> 3) != (off >> 3)) {
        s = shadow[end_off >> 3];
        if (s != ASAN_CLEAN) return 0;
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
#define SLOWPATH_IMPL(N)                                                \
    __attribute__((noinline))                                            \
    void __coqui_asan_slowpath_load_##N(void *ptr) {                     \
        int err = 0;                                                     \
        if (asan_check_access(ptr, (u8)(N), &err)) asan_report(err);   \
        if (asan_check_global((unsigned long)ptr, (u8)(N)))              \
            asan_report(ASAN_ERROR_GLOBAL_OVERFLOW);                     \
        if (asan_check_slab((unsigned long)ptr, (u8)(N)))                \
            asan_report(ASAN_ERROR_SLAB_OVERFLOW);                       \
    }                                                                    \
    __attribute__((noinline))                                            \
    void __coqui_asan_slowpath_store_##N(void *ptr) {                    \
        int err = 0;                                                     \
        if (asan_check_access(ptr, (u8)(N), &err)) asan_report(err);   \
        if (asan_check_global((unsigned long)ptr, (u8)(N)))              \
            asan_report(ASAN_ERROR_GLOBAL_OVERFLOW);                     \
        if (asan_check_slab((unsigned long)ptr, (u8)(N)))                \
            asan_report(ASAN_ERROR_SLAB_OVERFLOW);                       \
    }

SLOWPATH_IMPL(1)
SLOWPATH_IMPL(2)
SLOWPATH_IMPL(4)
SLOWPATH_IMPL(8)

#define CHECK_IMPL(N)                                                   \
    __attribute__((always_inline))                                       \
    void __coqui_asan_check_load_##N(void *ptr) {                       \
        if (asan_fastpath_ok(ptr, (u8)(N))) return;                     \
        __coqui_asan_slowpath_load_##N(ptr);                             \
    }                                                                    \
    __attribute__((always_inline))                                       \
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
 * lives in the first 4B of the leading red zone. */
void *__coqui_asan_slab_malloc(unsigned long size) {
    if (!g_slab_malloc) return (void *)0;
    if (size == 0) size = 1;

    /* Align user size up to 8 bytes so right-redzone starts on a boundary. */
    unsigned long user_aligned = (size + 7UL) & ~7UL;
    unsigned long total        = ASAN_LEFT_REDZONE + user_aligned + ASAN_RIGHT_REDZONE;

    u8 *raw = (u8 *)g_slab_malloc(total);
    if (!raw) return (void *)0;

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

void __coqui_asan_slab_free(void *ptr) {
    if (!ptr) return;
    if (!g_slab_free) return;

    u8 *user = (u8 *)ptr;
    u8 *raw  = user - ASAN_LEFT_REDZONE;

    /* Reject pointers that are not in the slab pool: no descriptor exists
     * for them, and calling g_slab_free with a heap/global pointer would
     * corrupt the slab allocator. */
    unsigned int hot_slot = *(unsigned int *)raw;
    if (hot_slot > 0u && hot_slot <= ASAN_MAX_SLAB_DESCS) {
        /* Fast path: clear the descriptor slot the hot-tag points to, and
         * verify the beg matches (guards against stale tags if the caller
         * hands us a random pointer). */
        unsigned int idx = hot_slot - 1u;
        if (__coqui_asan_slab_descs[idx].beg == (const void *)user) {
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
