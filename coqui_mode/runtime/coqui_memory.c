/*
 * coqui_memory.c --- per-thread heap allocator for coqui mode coqui_mode.
 *
 * Freelist-first, bump-fallback allocator with O(1) backward + O(N)
 * forward coalescing on free. Per-thread heap region (set up by
 * MemoryLayout transform; accessed via __coqui_heap_base).
 *
 * Block layout:
 *   [size_flags: u32][prev_size: u32][user data...]
 *   - size_flags: total block size including 8-byte header. Always
 *     a multiple of 8 (alloc payload is align-up-to-8, header is 8),
 *     so bit 0 is repurposed as the in-use flag.
 *   - prev_size: size of the physically previous block in this heap
 *     (0 = first block, or block immediately after the heap header).
 *     Lets free coalesce backward in O(1).
 *
 * On free, both adjacent blocks are checked: if free, they're spliced
 * out of the freelist and merged into a single larger free block. The
 * combined block is reinserted size-sorted into the freelist.
 *
 * Per-thread state (lives at start of heap region):
 *   free_head     (u64) — absolute pointer to first free block
 *   bump_top      (u32) — current bump position within heap region
 *   prev_bump_top (u32) — size of last bump-allocated block (for prev_size)
 */

#include "coqui_runtime.h"

/* Per-thread control header at the start of the heap region. */
typedef struct heap_hdr {
    void *free_head;       /* singly-linked list of free blocks, largest first */
    u32   bump_top;        /* bump allocator position */
    u32   prev_bump_size;  /* size of last bump-allocated block (for prev_size on next bump) */
} heap_hdr_t;

#define HEAP_MIN_ALLOC 8u
#define HEAP_HDR_SIZE  sizeof(heap_hdr_t)   /* 16 bytes */
#define BLOCK_HDR_SIZE 8u                   /* [size_flags:u32][prev_size:u32] */
#define BLOCK_IN_USE   1u                   /* bit 0 of size_flags */

/* Block header overlay. We don't dereference this struct directly because
 * legacy callers wrote the size as a bare u32; instead we treat the block
 * pointer as u8* and access size_flags / prev_size via aligned u32 loads. */

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

/* Block-header field accessors. Bit 0 of size_flags holds in_use. */
__attribute__((const, always_inline, nothrow))
static u32 blk_size(u32 size_flags) { return size_flags & ~BLOCK_IN_USE; }

__attribute__((const, always_inline, nothrow))
static u32 blk_in_use(u32 size_flags) { return size_flags & BLOCK_IN_USE; }

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
    hdr->prev_bump_size = 0;

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

/* Remove a specific block from the freelist by walking the chain.
 * Bounded by FL_REMOVE_MAX iterations to avoid pathological loops on
 * a corrupted list. Returns 1 if removed, 0 if not found / capped. */
#define FL_REMOVE_MAX 32

__attribute__((always_inline, nothrow))
static int fl_remove(heap_hdr_t *hdr, u8 *target) {
    void **prev_next = &hdr->free_head;
    void *cur = hdr->free_head;
    int iters = FL_REMOVE_MAX;
    while (cur && iters-- > 0) {
        if ((u8 *)cur == target) {
            *prev_next = *(void **)((u8 *)cur + BLOCK_HDR_SIZE);
            return 1;
        }
        prev_next = (void **)((u8 *)cur + BLOCK_HDR_SIZE);
        cur = *prev_next;
    }
    return 0;
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
        u32 cur_sf = *(u32 *)cur;
        u32 cur_sz = blk_size(cur_sf);
        if (likely(cur_sz >= need)) {
            /* Splice out of freelist. */
            *prev_next = *(void **)((u8 *)cur + BLOCK_HDR_SIZE);
            /* Mark in-use; preserve prev_size (already at offset 4). */
            *(u32 *)cur = cur_sz | BLOCK_IN_USE;
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
    *(u32 *)block = need | BLOCK_IN_USE;            /* size_flags */
    *(u32 *)(block + 4) = hdr->prev_bump_size;       /* prev_size */
    hdr->bump_top += need;
    hdr->prev_bump_size = need;
    return block + BLOCK_HDR_SIZE;
}

/* Insert `block` (already sized via *(u32*)block) into the freelist
 * size-sorted (largest first), 8-iter scan limit. */
__attribute__((always_inline, nothrow))
static void fl_insert_sized(heap_hdr_t *hdr, u8 *block, u32 blk_sz) {
    void **prev_next = &hdr->free_head;
    void *cur = hdr->free_head;
    int iters = 8;
    while (cur && iters-- > 0) {
        u32 cur_sz = blk_size(*(u32 *)cur);
        if (blk_sz >= cur_sz) break;   /* insert before smaller */
        prev_next = (void **)((u8 *)cur + BLOCK_HDR_SIZE);
        cur = *prev_next;
    }
    *(void **)(block + BLOCK_HDR_SIZE) = cur;
    *prev_next = block;
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
    u32 size_flags = *(u32 *)block;
    u32 blk_sz = blk_size(size_flags);

    /* Mark this block free first. */
    *(u32 *)block = blk_sz;

    /* --- Forward coalesce: if next block is free, splice & merge. --- */
    u8 *next = block + blk_sz;
    u8 *bump_end = heap_lo + hdr->bump_top;
    if (next < bump_end) {
        u32 nsf = *(u32 *)next;
        u32 nsz = blk_size(nsf);
        if (!blk_in_use(nsf) && nsz != 0) {
            if (fl_remove(hdr, next)) {
                blk_sz += nsz;
                *(u32 *)block = blk_sz;
            }
        }
    }

    /* --- Backward coalesce: if previous block is free, splice & merge. --- */
    u32 prev_size = *(u32 *)(block + 4);
    if (prev_size > 0 && prev_size <= (u32)(block - heap_lo)) {
        u8 *prev = block - prev_size;
        if (prev >= heap_lo + HEAP_HDR_SIZE) {
            u32 psf = *(u32 *)prev;
            u32 psz = blk_size(psf);
            if (!blk_in_use(psf) && psz == prev_size) {
                if (fl_remove(hdr, prev)) {
                    psz += blk_sz;
                    *(u32 *)prev = psz;
                    block = prev;
                    blk_sz = psz;
                }
            }
        }
    }

    /* --- Update successor's prev_size after coalesce. --- */
    u8 *after = block + blk_sz;
    if (after < bump_end) {
        *(u32 *)(after + 4) = blk_sz;
    } else if (after == bump_end) {
        /* Free block reaches the bump frontier: reclaim by rewinding bump_top.
         * The previous block's size was `*(u32*)(block+4)`; restore that as
         * the new prev_bump_size so the next bump alloc gets the right
         * predecessor link. */
        hdr->bump_top = (u32)(block - heap_lo);
        hdr->prev_bump_size = *(u32 *)(block + 4);
        return;
    }

    /* Add the (possibly merged) block to the freelist. */
    fl_insert_sized(hdr, block, blk_sz);
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
            /* Slab blocks store [hdr:u32][pad:u32][user data...] with user
             * data at ptr (= block+8, see coqui_slab.c).  For sub-block
             * allocs hdr is block_size (= 8 + aligned_user_bytes).  For
             * multi-slab allocs hdr encodes n_slabs*SLAB_SIZE in bits
             * [4..31] with a 0xF low-nibble sentinel; treating it directly
             * as a byte count would make old_size ~16× too large and cause
             * an OOB read during memcpy_u8.  slab_blk_actual_bytes() decodes
             * both cases, returning the total bytes owned from block start. */
            u8 *sb = (u8 *)ptr - 8;
            u32 old_block = *(u32 *)sb;
            unsigned long actual = slab_blk_actual_bytes(old_block);
            unsigned long old_size = (actual > 8u) ? (actual - 8u) : 0u;
            unsigned long copy = (old_size < size) ? old_size : (unsigned long)size;
            void *new_ptr = __coqui_malloc(size);
            if (unlikely(!new_ptr)) return (void *)0;
            memcpy_u8(new_ptr, ptr, copy);
            __coqui_free(ptr);
            return new_ptr;
        }
    }

    u8 *block = (u8 *)ptr - BLOCK_HDR_SIZE;
    /* Mask off the in-use flag when reading total block size. */
    u32 old_size = blk_size(*(u32 *)block) - BLOCK_HDR_SIZE;

    if (old_size >= size) return ptr;   /* no-op shrink */

    void *new_ptr = __coqui_malloc(size);
    if (unlikely(!new_ptr)) return (void *)0;
    memcpy_u8(new_ptr, ptr, old_size);
    __coqui_free(ptr);
    return new_ptr;
}
