/*
 * coqui_memory.c --- per-thread heap allocator for coqui mode coqui_mode.
 *
 * Freelist-first, bump-fallback allocator over the per-thread heap
 * region (set up by MemoryLayout transform; accessed via __coqui_heap_base).
 *
 * Block layout:
 *   [size: u32][pad: u32][user data...]
 *   size includes the 8-byte header.
 *
 * Per-thread state (lives at start of heap region):
 *   free_head (u64) — absolute pointer to first free block
 *   bump_top  (u32) — current bump position within heap region
 */

#include "coqui_runtime.h"

/* Per-thread control header at the start of the heap region. */
typedef struct heap_hdr {
    void *free_head;   /* singly-linked list of free blocks, largest first */
    u32   bump_top;    /* bump allocator position */
    u32   _pad;
} heap_hdr_t;

#define HEAP_MIN_ALLOC 8u
#define HEAP_HDR_SIZE  sizeof(heap_hdr_t)   /* 16 bytes */
#define BLOCK_HDR_SIZE 8u                   /* [size:u32][pad:u32] */

/* Align up to 8 bytes. `const, always_inline, nothrow` — pure arithmetic. */
__attribute__((const, always_inline, nothrow))
static u32 align8(u32 x) { return (x + 7u) & ~7u; }

/* Get this thread's heap control header. `pure` because it reads only the
 * (per-thread, runtime-stable) heap base; `always_inline` because every
 * heap op opens with this load. `nothrow`. */
__attribute__((pure, always_inline, nothrow))
static heap_hdr_t *heap_hdr(void) {
    return (heap_hdr_t *)__coqui_heap_base();
}

/* Slab pool globals -- defined in coqui_slab.c when linked; weak decls
 * here so __coqui_memory_init can zero the per-thread stack-chain head
 * and __coqui_free_raw can route slab-pool pointers to the slab free
 * path without hard-depending on the slab runtime being linked. When
 * the slab runtime is absent, __coqui_slab_pool stays NULL and both
 * paths short-circuit. */
extern char *__coqui_slab_pool __attribute__((weak));
extern unsigned long __coqui_slab_pool_size __attribute__((weak));
__attribute__((weak)) void __coqui_slab_free(void *ptr);

/* Initialize the heap control header (called once per thread at kernel entry).
   MemoryLayout transform inserts a call to this. `nothrow`. */
__attribute__((nothrow))
void __coqui_memory_init(void) {
    heap_hdr_t *hdr = heap_hdr();
    hdr->free_head = (void *)0;
    hdr->bump_top  = HEAP_HDR_SIZE;

    /* Zero the stack-spill chain head (slab_pool[tid*32+24]). Loaded lazily
     * by __coqui_stack_alloc on the first call from this thread; if no spill
     * happens, the slot stays 0 and the cleanup walk is a no-op. */
    if (__coqui_slab_pool != (char *)0) {
        u32 tid = __coqui_fuzz_tid();
        unsigned long *stack_head = (unsigned long *)(__coqui_slab_pool
                                  + (unsigned long)tid * COQUI_SLAB_THREAD_CTRL_STRIDE
                                  + COQUI_SLAB_STACK_HEAD_OFFSET);
        *stack_head = 0;
    }
}

/* The actual allocator lives in __coqui_malloc_raw / __coqui_free_raw so that
 * the Asan pass's bulk RAUW (__coqui_malloc → __coqui_asan_malloc) does not
 * rewrite calls inside the allocator implementation itself. The asan-internal
 * redirect in the pass rewires __coqui_asan_malloc's call from __coqui_malloc
 * to __coqui_malloc_raw, so all paths terminate at _raw with no recursion.
 *
 * `nothrow`: C runtime entry. Body has both fast path (freelist hit) and
 * bump path; per-thread freelist hit is the hot case but also moderate
 * size, so we keep this out-of-line. The thin __coqui_malloc wrapper
 * below is always_inline. */
__attribute__((nothrow))
void *__coqui_malloc_raw(unsigned long size) {
    if (unlikely(size == 0)) size = 1;
    u32 need = align8((u32)size) + BLOCK_HDR_SIZE;
    if (need < HEAP_MIN_ALLOC + BLOCK_HDR_SIZE) need = HEAP_MIN_ALLOC + BLOCK_HDR_SIZE;

    heap_hdr_t *hdr = heap_hdr();

    /* Try freelist first-fit (8-iter limit). Once a thread has done any
     * alloc/free churn the freelist hit is the steady-state fast path. */
    void **prev_next = &hdr->free_head;
    void *cur = hdr->free_head;
    int iters = 8;
    while (cur && iters-- > 0) {
        u32 blk_sz = *(u32 *)cur;
        if (likely(blk_sz >= need)) {
            *prev_next = *(void **)((u8 *)cur + BLOCK_HDR_SIZE);
            return (u8 *)cur + BLOCK_HDR_SIZE;
        }
        prev_next = (void **)((u8 *)cur + BLOCK_HDR_SIZE);
        cur = *prev_next;
    }

    /* Bump allocate. On exhaustion, return NULL -- callers (asan_malloc,
     * calloc, realloc) handle NULL by falling through to the shared slab
     * pool (if configured) or stamping trap_reason = COQUI_TRAP_OOM. The
     * previous behavior (trap here) blocked slab fall-through entirely. */
    u32 heap_sz = __coqui_heap_size();
    if (unlikely(hdr->bump_top + need > heap_sz)) {
        return (void *)0;
    }

    u8 *block = __coqui_heap_base() + hdr->bump_top;
    *(u32 *)block = need;
    hdr->bump_top += need;
    return block + BLOCK_HDR_SIZE;
}

__attribute__((nothrow))
void __coqui_free_raw(void *ptr) {
    if (unlikely(!ptr)) return;

    /* Route slab-pool pointers to the slab free path. The range check
     * precedes the heap-header dereference so we never scribble over
     * a slab pointer's first 8 bytes treating them as the heap block
     * header. The slab path is the rare case for targets without a
     * slab pool (and even with one, most allocs land in the heap). */
    if (unlikely(__coqui_slab_pool && __coqui_slab_pool_size)) {
        u8 *sp_lo = (u8 *)__coqui_slab_pool;
        u8 *sp_hi = sp_lo + __coqui_slab_pool_size;
        if ((u8 *)ptr >= sp_lo && (u8 *)ptr < sp_hi) {
            __coqui_slab_free(ptr);
            return;
        }
    }

    u8 *block = (u8 *)ptr - BLOCK_HDR_SIZE;

    /* Bounds check: block must be within heap region */
    u8 *heap_lo = __coqui_heap_base();
    u8 *heap_hi = heap_lo + __coqui_heap_size();
    if (unlikely(block < heap_lo || block >= heap_hi)) return;

    heap_hdr_t *hdr = heap_hdr();
    u32 blk_sz = *(u32 *)block;

    /* Size-sorted insert (8-iter limit, largest first) */
    void **prev_next = &hdr->free_head;
    void *cur = hdr->free_head;
    int iters = 8;
    while (cur && iters-- > 0) {
        u32 cur_sz = *(u32 *)cur;
        if (blk_sz >= cur_sz) break;   /* insert before smaller */
        prev_next = (void **)((u8 *)cur + BLOCK_HDR_SIZE);
        cur = *prev_next;
    }

    *(void **)((u8 *)block + BLOCK_HDR_SIZE) = cur;
    *prev_next = block;
}

/* Public allocator names. After the Asan pass's RAUW these have no callers
 * (every __coqui_malloc/__coqui_free site is rewritten to the asan version)
 * and the bodies are dropped from the linked module. They remain here as the
 * source-level entry points users / libc shims call before the pass runs.
 *
 * NOT always_inline: Asan.cpp does RAUW on the Function symbol; if these
 * were inlined into user call sites before the pass ran, the bypass-asan
 * routing would silently elide the asan wrapper. Leave these as plain
 * (linked) wrappers; nothrow only. */
__attribute__((nothrow))
void *__coqui_malloc(unsigned long size) {
    return __coqui_malloc_raw(size);
}

__attribute__((nothrow))
void __coqui_free(void *ptr) {
    __coqui_free_raw(ptr);
}

/* Internal byte-wise memset / memcpy for runtime use. User-facing memcpy/memset
   go through clang's intrinsic lowering, not these. */
static void *memset_u8(void *dst, int c, unsigned long n) {
    u8 *d = (u8 *)dst;
    for (unsigned long i = 0; i < n; i++) d[i] = (u8)c;
    return dst;
}

static void *memcpy_u8(void *dst, const void *src, unsigned long n) {
    u8 *d = (u8 *)dst;
    const u8 *s = (const u8 *)src;
    for (unsigned long i = 0; i < n; i++) d[i] = s[i];
    return dst;
}

__attribute__((nothrow))
void *__coqui_calloc(unsigned long nmemb, unsigned long size) {
    unsigned long total = nmemb * size;
    void *p = __coqui_malloc(total);
    if (likely(p != (void *)0)) memset_u8(p, 0, total);
    return p;
}

__attribute__((nothrow))
void *__coqui_realloc(void *ptr, unsigned long size) {
    if (unlikely(!ptr)) return __coqui_malloc(size);
    if (unlikely(size == 0)) { __coqui_free(ptr); return (void *)0; }

    /* Slab-pool pointers always take the malloc+copy+free path (no in-place
     * shrink). We cannot safely read a slab allocation's original size from
     * the caller's input buffer: the 8-byte header is BEFORE ptr (at ptr-8),
     * and the slab free path handles the range check itself. This is also
     * how the legacy coqui realloc routes cross-tier reallocs (see
     * runtime/coqui_fuzz_asan.c). */
    if (unlikely(__coqui_slab_pool && __coqui_slab_pool_size)) {
        u8 *sp_lo = (u8 *)__coqui_slab_pool;
        u8 *sp_hi = sp_lo + __coqui_slab_pool_size;
        if ((u8 *)ptr >= sp_lo && (u8 *)ptr < sp_hi) {
            /* Slab blocks store [size:u32][next:u64] with user data at
             * block+8 (see coqui_slab.c). We read the stored block size
             * and subtract the 8B header to get the user-visible size. */
            u8 *sb = (u8 *)ptr - 8;
            u32 old_block = *(u32 *)sb;
            u32 old_size = (old_block > 8u) ? (old_block - 8u) : 0u;
            unsigned long copy = (old_size < size) ? old_size : (u32)size;
            void *new_ptr = __coqui_malloc(size);
            if (unlikely(!new_ptr)) return (void *)0;
            memcpy_u8(new_ptr, ptr, copy);
            __coqui_free(ptr);
            return new_ptr;
        }
    }

    u8 *block = (u8 *)ptr - BLOCK_HDR_SIZE;
    u32 old_size = *(u32 *)block - BLOCK_HDR_SIZE;

    if (old_size >= size) return ptr;   /* no-op shrink */

    void *new_ptr = __coqui_malloc(size);
    if (unlikely(!new_ptr)) return (void *)0;
    memcpy_u8(new_ptr, ptr, old_size);
    __coqui_free(ptr);
    return new_ptr;
}
