/*
 * coqui_stack_spill.c --- per-thread stack-allocation spill into the slab pool.
 *
 * The StackSpill pass replaces oversize allocas with __coqui_stack_alloc(size).
 * Each thread bump-allocates through a chain of carved 4 KB slab pages.
 * Save/restore at function boundaries rewinds the bump pointer (LIFO).
 *
 * No ASan shadow writes: pointers handed out by _alloc are deliberately
 * routed past coqui_asan.c's check helpers. The pass marks loads/stores
 * through these pointers with !nosanitize so ASan.cpp skips them.
 */

#include "coqui_runtime.h"

/* Globals defined in coqui_slab.c. */
extern char         *__coqui_slab_pool;
extern unsigned long __coqui_slab_pool_size;
extern unsigned int *__coqui_slab_next;
extern unsigned int  __coqui_slab_ctrl_slabs;

/* Page header layout (16 bytes at offset 0 of each carved 4 KB page):
 *   [0..7]   prev_page        (u64) — previous page in chain, or 0
 *   [8..11]  virtual_base     (u32) — sum of usable bytes in earlier pages
 *   [12..15] bump_top         (u32) — bytes used in this page (>= HDR_SIZE)
 * Usable space per page: SLAB_SIZE - COQUI_STACK_PAGE_HDR_SIZE = 4080 bytes. */
#define HDR_PREV_OFF   0
#define HDR_VBASE_OFF  8
#define HDR_BUMP_OFF   12

/* Inline PTX global atomic add — matches slab_atom_add_global in coqui_slab.c.
 * Returns old value; the new slab index is old + val. */
static unsigned int stack_atom_add_global(unsigned int *addr, unsigned int val) {
  unsigned int old;
  asm volatile("atom.global.add.u32 %0, [%1], %2;"
               : "=r"(old) : "l"(addr), "r"(val));
  return old;
}

/* Returns a pointer to the u64 stack_chain_head slot for thread `tid`. */
static inline u64 *thread_stack_head_slot(u32 tid) {
  return (u64 *)(void *)(__coqui_slab_pool
                         + (unsigned long)tid * COQUI_SLAB_THREAD_CTRL_STRIDE
                         + COQUI_SLAB_STACK_HEAD_OFFSET);
}

static inline u32 page_get_bump(char *page) {
  return *(u32 *)(void *)(page + HDR_BUMP_OFF);
}
static inline void page_set_bump(char *page, u32 v) {
  *(u32 *)(void *)(page + HDR_BUMP_OFF) = v;
}
static inline u32 page_get_vbase(char *page) {
  return *(u32 *)(void *)(page + HDR_VBASE_OFF);
}
static inline u64 page_get_prev(char *page) {
  return *(u64 *)(void *)(page + HDR_PREV_OFF);
}

/* Carve one fresh slab page from the global pool.
 * `prev_vbase` is the virtual base to record in the new page's header.
 * Returns NULL on pool exhaustion (caller must trap). */
static char *carve_one_page(u32 prev_vbase) {
  /* slab_idx counts within the data partition; absolute index skips
   * past the [0..ctrl_slabs) per-thread control rows.  Matches the
   * heap path's pattern at coqui_slab.c:242-244. */
  unsigned int slab_idx = stack_atom_add_global(__coqui_slab_next, 1u);
  unsigned int abs_slab = slab_idx + __coqui_slab_ctrl_slabs;
  unsigned long byte_off = (unsigned long)abs_slab * SLAB_SIZE;
  if (unlikely(byte_off + SLAB_SIZE > __coqui_slab_pool_size)) {
    return (char *)0;
  }
  char *page = __coqui_slab_pool + byte_off;
  /* HDR_PREV_OFF is written by the caller (__coqui_stack_alloc) before the
   * page becomes reachable via *head — no need to zero it here. */
  *(u32 *)(void *)(page + HDR_VBASE_OFF) = prev_vbase;
  *(u32 *)(void *)(page + HDR_BUMP_OFF)  = COQUI_STACK_PAGE_HDR_SIZE;
  return page;
}

/* Return the current virtual stack depth (absolute byte offset across all
 * pages in the chain).  Zero means no pages allocated yet. */
__attribute__((nothrow))
unsigned int __coqui_stack_save(void) {
  if (unlikely(!__coqui_slab_pool || __coqui_slab_pool_size == 0))
    return 0;
  u32 tid = __coqui_fuzz_tid();
  u64 *head = thread_stack_head_slot(tid);
  if (*head == 0) return 0;
  char *page = (char *)*head;
  /* virtual depth = vbase + (bump - hdr_size) */
  return page_get_vbase(page) + page_get_bump(page) - COQUI_STACK_PAGE_HDR_SIZE;
}

/* Bump-allocate `size` bytes from the per-thread stack chain.
 * Carves a new slab page when the current one is full. */
__attribute__((nothrow))
void *__coqui_stack_alloc(unsigned int size) {
  if (unlikely(!__coqui_slab_pool || __coqui_slab_pool_size == 0)) {
    __coqui_trap_with_reason(COQUI_TRAP_STACK_OVERFLOW);
    return (void *)0;
  }
  /* Round up to 8-byte alignment. */
  size = (size + 7u) & ~7u;
  if (unlikely(size > COQUI_STACK_PAGE_USABLE)) {
    /* Single frame larger than one page: out of scope; trap now so the
     * StackSpill pass's compile-time guard can catch it earlier next time. */
    __coqui_trap_with_reason(COQUI_TRAP_STACK_OVERFLOW);
    return (void *)0;
  }
  u32 tid = __coqui_fuzz_tid();
  u64 *head = thread_stack_head_slot(tid);
  char *page = (char *)*head;
  if (likely(page != (char *)0)) {
    u32 bump = page_get_bump(page);
    if (likely(bump + size <= SLAB_SIZE)) {
      void *p = (void *)(page + bump);
      page_set_bump(page, bump + size);
      return p;
    }
  }
  /* Need a fresh page. Compute the virtual base for the new page. */
  u32 prev_vbase = page
      ? page_get_vbase(page) + COQUI_STACK_PAGE_USABLE
      : 0u;
  char *fresh = carve_one_page(prev_vbase);
  if (unlikely(!fresh)) {
    __coqui_trap_with_reason(COQUI_TRAP_STACK_OVERFLOW);
    return (void *)0;
  }
  /* Link new page at head of chain. */
  *(u64 *)(void *)(fresh + HDR_PREV_OFF) = (u64)page;
  *head = (u64)fresh;
  page_set_bump(fresh, COQUI_STACK_PAGE_HDR_SIZE + size);
  return (void *)(fresh + COQUI_STACK_PAGE_HDR_SIZE);
}

/* Rewind the bump pointer to the previously saved marker.
 * Pages whose vbase is above `saved` are dropped from the chain head;
 * they remain carved (slab_next already moved past them) and are
 * reclaimed at thread exit by __coqui_slab_release_thread. */
__attribute__((nothrow))
void __coqui_stack_restore(unsigned int saved) {
  if (unlikely(!__coqui_slab_pool || __coqui_slab_pool_size == 0))
    return;
  u32 tid = __coqui_fuzz_tid();
  u64 *head = thread_stack_head_slot(tid);
  char *page = (char *)*head;
  while (page) {
    u32 vbase = page_get_vbase(page);
    if (saved >= vbase) {
      /* The saved marker falls in this page (or at its start).
       * Rewind bump to the saved position. */
      page_set_bump(page, saved - vbase + COQUI_STACK_PAGE_HDR_SIZE);
      *head = (u64)page;
      return;
    }
    /* saved is in an earlier page; walk backwards. */
    page = (char *)page_get_prev(page);
  }
  /* Fully unwound. */
  *head = 0;
}
