/*
 * coqui_memory.c --- per-thread heap allocator for cuAFL coqui_mode.
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

/* Align up to 8 bytes. */
__attribute__((always_inline))
static inline u32 align8(u32 x) { return (x + 7u) & ~7u; }

/* Get this thread's heap control header. */
__attribute__((always_inline))
static inline heap_hdr_t *heap_hdr(void) {
    return (heap_hdr_t *)__coqui_heap_base();
}

/* Initialize the heap control header (called once per thread at kernel entry).
   MemoryLayout transform inserts a call to this. */
void __coqui_memory_init(void) {
    heap_hdr_t *hdr = heap_hdr();
    hdr->free_head = (void *)0;
    hdr->bump_top  = HEAP_HDR_SIZE;
}

/* The actual allocator lives in __coqui_malloc_raw / __coqui_free_raw so that
 * the Asan pass's bulk RAUW (__coqui_malloc → __coqui_asan_malloc) does not
 * rewrite calls inside the allocator implementation itself. The asan-internal
 * redirect in the pass rewires __coqui_asan_malloc's call from __coqui_malloc
 * to __coqui_malloc_raw, so all paths terminate at _raw with no recursion. */
void *__coqui_malloc_raw(unsigned long size) {
    if (size == 0) size = 1;
    u32 need = align8((u32)size) + BLOCK_HDR_SIZE;
    if (need < HEAP_MIN_ALLOC + BLOCK_HDR_SIZE) need = HEAP_MIN_ALLOC + BLOCK_HDR_SIZE;

    heap_hdr_t *hdr = heap_hdr();

    /* Try freelist first-fit (8-iter limit) */
    void **prev_next = &hdr->free_head;
    void *cur = hdr->free_head;
    int iters = 8;
    while (cur && iters-- > 0) {
        u32 blk_sz = *(u32 *)cur;
        if (blk_sz >= need) {
            *prev_next = *(void **)((u8 *)cur + BLOCK_HDR_SIZE);
            return (u8 *)cur + BLOCK_HDR_SIZE;
        }
        prev_next = (void **)((u8 *)cur + BLOCK_HDR_SIZE);
        cur = *prev_next;
    }

    /* Bump allocate */
    u32 heap_sz = __coqui_heap_size();
    if (hdr->bump_top + need > heap_sz) {
        __coqui_trap();   /* OOM: trap per spec §4.6 */
        return (void *)0;
    }

    u8 *block = __coqui_heap_base() + hdr->bump_top;
    *(u32 *)block = need;
    hdr->bump_top += need;
    return block + BLOCK_HDR_SIZE;
}

void __coqui_free_raw(void *ptr) {
    if (!ptr) return;

    u8 *block = (u8 *)ptr - BLOCK_HDR_SIZE;

    /* Bounds check: block must be within heap region */
    u8 *heap_lo = __coqui_heap_base();
    u8 *heap_hi = heap_lo + __coqui_heap_size();
    if (block < heap_lo || block >= heap_hi) return;

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
 * source-level entry points users / libc shims call before the pass runs. */
void *__coqui_malloc(unsigned long size) {
    return __coqui_malloc_raw(size);
}

void __coqui_free(void *ptr) {
    __coqui_free_raw(ptr);
}

/* Internal byte-wise memset / memcpy for runtime use. User-facing memcpy/memset
   go through clang's intrinsic lowering, not these. */
__attribute__((always_inline))
static inline void *memset_u8(void *dst, int c, unsigned long n) {
    u8 *d = (u8 *)dst;
    for (unsigned long i = 0; i < n; i++) d[i] = (u8)c;
    return dst;
}

__attribute__((always_inline))
static inline void *memcpy_u8(void *dst, const void *src, unsigned long n) {
    u8 *d = (u8 *)dst;
    const u8 *s = (const u8 *)src;
    for (unsigned long i = 0; i < n; i++) d[i] = s[i];
    return dst;
}

void *__coqui_calloc(unsigned long nmemb, unsigned long size) {
    unsigned long total = nmemb * size;
    void *p = __coqui_malloc(total);
    if (p) memset_u8(p, 0, total);
    return p;
}

void *__coqui_realloc(void *ptr, unsigned long size) {
    if (!ptr) return __coqui_malloc(size);
    if (size == 0) { __coqui_free(ptr); return (void *)0; }

    u8 *block = (u8 *)ptr - BLOCK_HDR_SIZE;
    u32 old_size = *(u32 *)block - BLOCK_HDR_SIZE;

    if (old_size >= size) return ptr;   /* no-op shrink */

    void *new_ptr = __coqui_malloc(size);
    if (!new_ptr) return (void *)0;
    memcpy_u8(new_ptr, ptr, old_size);
    __coqui_free(ptr);
    return new_ptr;
}
