/*
 * coqui_slab.c --- device-side slab pool allocator.
 *
 * Shared overflow allocator for targets whose per-thread heap is too small.
 * Ported from ~/coqui/runtime/coqui_fuzz_slab.c with renames:
 *   __coqui_fuzz_slab_* → __coqui_slab_*
 *   __coqui_fuzz_tid    → __coqui_fuzz_tid (unchanged, same name in cuAFL)
 *
 * Architecture (per 2026-03-25-shared-slab-pool-allocator-design.md +
 * 2026-04-12-slab-reclamation-design.md):
 *   Three-tier allocation:
 *     1. Block-local shared-memory bump (fast, ~20 cycles atom.shared.add)
 *     2. Global atomic bump (slab_next, uncontended mid-batch)
 *     3. Treiber-stack pop from thread-exit reclaimed ranges
 *
 * Slab layout (4 KB units):
 *   [0..ctrl_slabs):           per-thread 32B control rows + stack head slot
 *   [ctrl_slabs..max_slabs):   block-partitioned bump region (grid*block threads)
 *
 * Ctrl row at slab_pool[tid * 32]:
 *   [0..7]    free_head  -- absolute pointer to first free block
 *   [8..15]   current_heap -- pointer to current bump heap
 *   [16..19]  heap_limit (u32)
 *   [20..23]  bump_top (u32)
 *   [24..31]  reserved
 *
 * Free-stack head at slab_pool[grid*block*32] (single u64 cell inside
 * ctrl_slabs reservation). Push/pop via atom.cas.b64 in generic space.
 *
 * Range header at the start of every multi-slab range handed to a thread:
 *   [0..7]   prev_range (u64)  -- for per-thread heap chain walk at exit
 *   [8..11]  n_slabs (u32)     -- range size in slab units
 *   [12..15] pad
 *
 * Linked when the slab runtime is compiled in. Targets without --slab-pool-size
 * are still safe: __coqui_slab_pool == 0 makes every entry point return NULL /
 * short-circuit.
 */

#include "coqui_runtime.h"

/* ===------------------------------------------------------------------===
 * Slab pool globals -- owned by this file. Set by __coqui_slab_setup,
 * which the host calls indirectly via the kernel-entry init sequence.
 * Bound to device symbols by the host (cuModuleGetGlobal + cuMemcpyHtoD).
 *
 * Declared with default (zero) initializers so targets compiled without a
 * configured slab pool get benign behavior (every tier fails to NULL).
 * ===------------------------------------------------------------------=== */

__attribute__((visibility("default"))) __attribute__((used))
char *__coqui_slab_pool;

__attribute__((visibility("default"))) __attribute__((used))
unsigned long __coqui_slab_pool_size;

__attribute__((visibility("default"))) __attribute__((used))
char *__coqui_slab_shadow;

__attribute__((visibility("default"))) __attribute__((used))
unsigned int *__coqui_slab_next;

/* slabs per block for the fast-path partitioned region. 0 = no partitioning
 * (all allocs go to global bump). Set by host once at setup. */
__attribute__((visibility("default"))) __attribute__((used))
unsigned int __coqui_slab_block_budget;

/* Computed at setup from grid dims: (grid*block*32 + 8 + 4095)/4096.
 * Reserves the per-thread control rows + free-stack-head cell. */
__attribute__((visibility("default"))) __attribute__((used))
unsigned int __coqui_slab_ctrl_slabs;

/* Declared by FuzzEntry pass when slab is enabled. Returns a generic-space
 * pointer that resolves to this block's __shared__ bucket strip (13 * 8B). */
extern char *__coqui_slab_bucket_base(void);

/* Declared by FuzzEntry pass when slab is enabled. Returns a generic-space
 * pointer to this block's shared-memory allocation counter (4B). */
extern unsigned int *__coqui_slab_block_next(void);

/* Coqui ASan function-pointer registration (coqui_asan.c). */
extern void __coqui_asan_register_slab(
    void *(*malloc_fn)(unsigned long),
    void (*free_fn)(void *));

/* Forward declarations (for asan registration hand-off). */
void *__coqui_slab_malloc(unsigned long size);
void __coqui_slab_free(void *ptr);

/* ===------------------------------------------------------------------===
 * Slab constants
 * ===------------------------------------------------------------------=== */

#define SLAB_SIZE             4096u
#define SLAB_N_BUCKETS        13
#define SLAB_MIN_BUCKET_SIZE  16u
#define SLAB_MULTI_SLAB_IDX   15
#define SLAB_BLOCK_HDR_SIZE   4u

/* ===------------------------------------------------------------------===
 * Inline PTX helpers (static, slab-local).
 * Hand-rolled PTX so the helpers don't depend on external intrinsics and
 * execute as expected atom.* / vote.* instructions.
 * ===------------------------------------------------------------------=== */

static unsigned int slab_activemask(void) {
  unsigned int mask;
  asm volatile("activemask.b32 %0;" : "=r"(mask));
  return mask;
}

static unsigned int slab_ballot_sync(unsigned int mask, int pred) {
  unsigned int result;
  asm volatile("{\n\t"
               ".reg .pred p;\n\t"
               "setp.ne.s32 p, %1, 0;\n\t"
               "vote.ballot.sync.b32 %0, p, %2;\n\t"
               "}"
               : "=r"(result) : "r"(pred), "r"(mask));
  return result;
}

static void slab_bar_warp_sync(unsigned int mask) {
  asm volatile("bar.warp.sync %0;" :: "r"(mask));
}

static int slab_atom_exch_gen(volatile int *addr, int val) {
  int old;
  asm volatile("atom.exch.b32 %0, [%1], %2;"
               : "=r"(old) : "l"(addr), "r"(val));
  return old;
}

static unsigned int slab_atom_add_global(unsigned int *addr, unsigned int val) {
  unsigned int old;
  asm volatile("atom.global.add.u32 %0, [%1], %2;"
               : "=r"(old) : "l"(addr), "r"(val));
  return old;
}

static unsigned int slab_lane_id(void) {
  unsigned int id;
  asm volatile("mov.u32 %0, %%laneid;" : "=r"(id));
  return id;
}

static unsigned int slab_atom_cas_gen(volatile unsigned int *addr,
                                     unsigned int compare,
                                     unsigned int val) {
  unsigned int old;
  asm volatile("atom.cas.b32 %0, [%1], %2, %3;"
               : "=r"(old) : "l"(addr), "r"(compare), "r"(val));
  return old;
}

static unsigned long slab_atom_cas_64(volatile unsigned long *addr,
                                     unsigned long compare,
                                     unsigned long val) {
  unsigned long old;
  asm volatile("atom.cas.b64 %0, [%1], %2, %3;"
               : "=l"(old) : "l"(addr), "l"(compare), "l"(val));
  return old;
}

/* Generic-space atomic add. When given a pointer cast from AS3 the hardware
 * resolves the memory space and executes as atom.shared.add (~20 cycles). */
static unsigned int slab_atom_add_gen(volatile unsigned int *addr,
                                      unsigned int val) {
  unsigned int old;
  asm volatile("atom.add.u32 %0, [%1], %2;"
               : "=r"(old) : "l"(addr), "r"(val));
  return old;
}

static unsigned int slab_blockIdx_x(void) {
  unsigned int bid;
  asm volatile("mov.u32 %0, %%ctaid.x;" : "=r"(bid));
  return bid;
}

/* ===------------------------------------------------------------------===
 * Global free stack (Treiber stack) for reclaimed slab ranges.
 * Head lives at slab_pool[grid*block*32], inside the ctrl_slabs reservation.
 * ===------------------------------------------------------------------=== */

static unsigned long *slab_free_stack_head(void) {
  unsigned int bd, gd;
  asm volatile("mov.u32 %0, %%ntid.x;" : "=r"(bd));
  asm volatile("mov.u32 %0, %%nctaid.x;" : "=r"(gd));
  unsigned long offset = (unsigned long)(bd * gd) * 32;
  return (unsigned long *)(void *)(__coqui_slab_pool + offset);
}

static void slab_free_stack_push(char *range) {
  unsigned long *head = slab_free_stack_head();
  unsigned long new_val = (unsigned long)range;
  unsigned long old_head;
  do {
    old_head = *head;
    *(unsigned long *)(void *)range = old_head;  /* next = old head */
  } while (slab_atom_cas_64(head, old_head, new_val) != old_head);
}

static char *slab_free_stack_pop(unsigned int n_needed) {
  unsigned long *head = slab_free_stack_head();
  unsigned long old_head;
  unsigned long next;
  do {
    old_head = *head;
    if (old_head == 0) return (char *)0;
    next = *(unsigned long *)(void *)old_head;  /* read next pointer */
  } while (slab_atom_cas_64(head, old_head, next) != old_head);
  /* Accept range only if big enough; else push back and fail. */
  char *range = (char *)old_head;
  unsigned int range_n = *(unsigned int *)(void *)(range + 8);
  if (range_n >= n_needed) return range;
  slab_free_stack_push(range);
  return (char *)0;
}

/* ===------------------------------------------------------------------===
 * Two-tier slab allocation: block-local fast path + global atomic fallback.
 *
 * Fast path: atom.shared.add on __coqui_slab_block_next() (resolves inside
 * the SM, ~20 cycles). When the block's budget is exhausted, fall through
 * to global contended atom.global.add on __coqui_slab_next; finally try the
 * Treiber stack of thread-exit-reclaimed ranges.
 * ===------------------------------------------------------------------=== */

static char *slab_alloc_slabs(unsigned int n) {
  unsigned int ctrl_slabs = __coqui_slab_ctrl_slabs;
  unsigned int max_slabs = (unsigned int)(__coqui_slab_pool_size / SLAB_SIZE);

  /* Tier 1: block-local bump via shared atomic. */
  if (__coqui_slab_block_budget > 0) {
    unsigned int *bnext = __coqui_slab_block_next();
    unsigned int local_idx = slab_atom_add_gen(bnext, n);
    if (local_idx + n <= __coqui_slab_block_budget) {
      unsigned int bid = slab_blockIdx_x();
      unsigned int abs_slab =
          ctrl_slabs + bid * __coqui_slab_block_budget + local_idx;
      return __coqui_slab_pool + (unsigned long)abs_slab * SLAB_SIZE;
    }
    /* block budget exhausted */
  }

  /* Tier 2: global atomic bump. */
  unsigned int slab_idx = slab_atom_add_global(__coqui_slab_next, n);
  unsigned int abs_slab = slab_idx + ctrl_slabs;
  if (abs_slab + n <= max_slabs)
    return __coqui_slab_pool + (unsigned long)abs_slab * SLAB_SIZE;

  /* Tier 3: try a reclaimed range from the Treiber stack. */
  char *recycled = slab_free_stack_pop(n);
  if (recycled)
    return recycled;
  return (char *)0;
}

/* ===------------------------------------------------------------------===
 * Slab bucket helpers
 * ===------------------------------------------------------------------=== */

/* `const, always_inline, nothrow` — pure size→bucket lookup, hot per
 * slab_malloc call. `const` because the result is a function of the
 * argument alone (no memory access). */
__attribute__((const, always_inline, nothrow))
static unsigned int slab_size_to_bucket(unsigned int size) {
  if (size <= SLAB_MIN_BUCKET_SIZE) return 0;
  int msb = 31 - __builtin_clz(size - 1);
  unsigned int bucket = (unsigned int)(msb - 3);
  return (bucket < SLAB_N_BUCKETS) ? bucket : SLAB_N_BUCKETS;
}

/* ===------------------------------------------------------------------===
 * Slab pool allocator (per-thread overflow region + absolute-pointer freelist).
 *
 * Per-thread control at slab_pool[tid * 32]:
 *   [0-7]    free_head   -- first free block (absolute ptr)
 *   [8-15]   current_heap -- current bump heap (range head)
 *   [16-19]  heap_limit   -- size of current heap
 *   [20-23]  bump_top     -- bump position in current heap
 *   [24-31]  reserved
 * ===------------------------------------------------------------------=== */

#define SLAB_OVERFLOW_INITIAL_SLABS 4    /* 16KB initial */
#define SLAB_OVERFLOW_GROW_SLABS    64   /* 256KB per chain */
#define SLAB_MAX_SINGLE_ALLOC  (1024u * 1024u)  /* 1MB per-allocation cap */

/* `noinline, nothrow`: large body covering bucket lookup + per-thread
 * freelist + bump-allocate + slab-acquire fall-through. Function pointer
 * registered with ASan via __coqui_asan_register_slab so an inline body
 * would not survive the indirect call anyway. */
__attribute__((noinline, nothrow))
void *__coqui_slab_malloc(unsigned long size) {
  if (!__coqui_slab_pool || __coqui_slab_pool_size == 0)
    return (void *)0;
  if (size == 0) size = 1;
  /* Reject absurdly large single allocations. Corrupted chunk length fields
   * (100MB-4GB) cause catastrophic memset stalls on NVPTX. */
  if (size > SLAB_MAX_SINGLE_ALLOC)
    return (void *)0;
  unsigned int needed = (unsigned int)size + SLAB_BLOCK_HDR_SIZE;
  unsigned int bucket = slab_size_to_bucket(needed);

  /* Multi-slab path (>64KB): grab contiguous slabs, leak on free. */
  if (bucket >= SLAB_N_BUCKETS) {
    unsigned int n_slabs = (needed + SLAB_SIZE - 1) / SLAB_SIZE;
    char *block = slab_alloc_slabs(n_slabs);
    if (!block) return (void *)0;
    *(unsigned int *)(void *)block =
        SLAB_MULTI_SLAB_IDX | ((n_slabs * SLAB_SIZE) << 4);
    return (void *)(block + SLAB_BLOCK_HDR_SIZE);
  }

  (void)bucket;

  unsigned long tid = __coqui_fuzz_tid();
  char *tctrl = __coqui_slab_pool + tid * 32;
  char **free_head_ptr = (char **)(void *)tctrl;
  char **heap_ptr = (char **)(void *)(tctrl + 8);
  unsigned int *limit_ptr = (unsigned int *)(void *)(tctrl + 16);
  unsigned int *bump_ptr = (unsigned int *)(void *)(tctrl + 20);

  /* 8-byte align, min 16 (room for 8B next-pointer + 8B size). */
  unsigned int aligned = ((unsigned int)size + 7) & ~7u;
  if (aligned < 8) aligned = 8;
  /* Store allocation size in first 8 bytes (for first-fit matching). */
  unsigned int block_size = 8 + aligned; /* 8B size header + user data */

  /* Try per-thread freelist (first-fit by size, bounded search). */
  {
    char **prev_next = free_head_ptr;
    char *cur = *free_head_ptr;
    int limit = 128;
    while (cur && limit-- > 0) {
      unsigned int blk_sz = *(unsigned int *)(void *)cur;
      if (blk_sz >= block_size) {
        *prev_next = *(char **)(void *)(cur + 8); /* prev->next = cur->next */
        return (void *)(cur + 8);                 /* user area */
      }
      prev_next = (char **)(void *)(cur + 8); /* &cur->next */
      cur = *(char **)(void *)(cur + 8);       /* cur = cur->next */
    }
  }

  /* Bump-allocate from current heap. */
  char *heap = *heap_ptr;
  unsigned int hlimit = *limit_ptr;
  unsigned int btop = *bump_ptr;

  if (!heap) {
    heap = slab_alloc_slabs(SLAB_OVERFLOW_INITIAL_SLABS);
    if (!heap) return (void *)0;
    *(unsigned long *)(void *)heap = 0;  /* prev = NULL */
    *(unsigned int *)(void *)(heap + 8) = SLAB_OVERFLOW_INITIAL_SLABS;
    *heap_ptr = heap;
    hlimit = SLAB_OVERFLOW_INITIAL_SLABS * SLAB_SIZE;
    *limit_ptr = hlimit;
    btop = 16; /* skip range header */
  }

  if (btop + block_size > hlimit) {
    char *old_heap = heap;
    unsigned int grow = SLAB_OVERFLOW_GROW_SLABS;
    unsigned int need = (block_size + SLAB_SIZE - 1) / SLAB_SIZE;
    if (need > grow) grow = need;
    heap = slab_alloc_slabs(grow);
    if (!heap) return (void *)0;
    *(unsigned long *)(void *)heap = (unsigned long)old_heap;
    *(unsigned int *)(void *)(heap + 8) = grow;
    *heap_ptr = heap;
    hlimit = grow * SLAB_SIZE;
    *limit_ptr = hlimit;
    btop = 16; /* skip range header */
  }

  char *block = heap + btop;
  *(unsigned int *)(void *)block = block_size; /* store size for first-fit */
  *bump_ptr = btop + block_size;
  return (void *)(block + 8); /* user area after 8B size header */
}

/* `noinline, nothrow`: same rationale as __coqui_slab_malloc. */
__attribute__((noinline, nothrow))
void __coqui_slab_free(void *ptr) {
  if (!ptr) return;

  /* Push onto per-thread absolute-pointer freelist. */
  unsigned long tid = __coqui_fuzz_tid();
  char *tctrl = __coqui_slab_pool + tid * 32;
  char **free_head_ptr = (char **)(void *)tctrl;
  /* Block starts 8B before ptr (size header preserved for first-fit). */
  char *block = (char *)ptr - 8;
  if (block < __coqui_slab_pool || block >= __coqui_slab_pool + __coqui_slab_pool_size)
    return;

  /* Insert sorted by size (largest first) with a short search to cap
   * worst-case free cost. */
  unsigned int blk_sz = *(unsigned int *)(void *)block;
  char **prev = free_head_ptr;
  char *cur = *free_head_ptr;
  int limit = 8;
  while (cur && limit-- > 0) {
    unsigned int cur_sz = *(unsigned int *)(void *)cur;
    if (blk_sz >= cur_sz) break;
    prev = (char **)(void *)(cur + 8);
    cur = *(char **)(void *)(cur + 8);
  }
  *(char **)(void *)(block + 8) = cur;
  *prev = block;
}

/* ===------------------------------------------------------------------===
 * Slab reclamation at thread exit. Walk the per-thread heap chain and
 * push each slab range onto the global free stack for reuse.
 * ===------------------------------------------------------------------=== */

/* `nothrow`: C runtime entry called from __coqui_trap_with_reason / clean
 * thread exit. Never throws. */
__attribute__((nothrow))
void __coqui_slab_release_thread(void) {
  if (!__coqui_slab_pool || __coqui_slab_pool_size == 0)
    return;
  unsigned long tid = __coqui_fuzz_tid();
  char *tctrl = __coqui_slab_pool + tid * 32;
  char **heap_ptr = (char **)(void *)(tctrl + 8);
  char *heap = *heap_ptr;
  if (!heap)
    return;
  while (heap) {
    char *prev = (char *)(*(unsigned long *)(void *)heap);
    unsigned int n_slabs = *(unsigned int *)(void *)(heap + 8);
    if (n_slabs > 0)
      slab_free_stack_push(heap);
    heap = prev;
  }
  /* Zero control to prevent double-release. */
  *(unsigned long *)(void *)tctrl = 0;        /* free_head */
  *(unsigned long *)(void *)(tctrl + 8) = 0;  /* current_heap */
  *(unsigned int *)(void *)(tctrl + 16) = 0;  /* heap_limit */
  *(unsigned int *)(void *)(tctrl + 20) = 0;  /* bump_top */
}

/* ===------------------------------------------------------------------===
 * Setup and per-block initialization
 * ===------------------------------------------------------------------=== */

/* Called once per kernel launch (idempotent -- every thread runs it).
 * Parameter-free: the slab globals are bound by the host via
 * cuModuleGetGlobal / cuMemcpyHtoD before launch. This only computes
 * ctrl_slabs (needs runtime grid dims) and registers with ASan. `nothrow`. */
__attribute__((nothrow))
void __coqui_slab_setup(void) {
  if (!__coqui_slab_pool || __coqui_slab_pool_size == 0)
    return;
  unsigned int bd, gd;
  asm volatile("mov.u32 %0, %%ntid.x;" : "=r"(bd));
  asm volatile("mov.u32 %0, %%nctaid.x;" : "=r"(gd));
  __coqui_slab_ctrl_slabs =
      (bd * gd * 32u + 8u + (SLAB_SIZE - 1)) / SLAB_SIZE;
  __coqui_asan_register_slab(__coqui_slab_malloc, __coqui_slab_free);
}

/* Called by thread 0 of every block, before the __syncthreads() barrier
 * that precedes per-thread work. Zeros the shared-memory bucket strip
 * and the per-block allocation counter. `nothrow`. */
__attribute__((nothrow))
void __coqui_slab_init_block(void) {
  if (!__coqui_slab_pool || __coqui_slab_pool_size == 0)
    return;
  char *base = __coqui_slab_bucket_base();
  for (int i = 0; i < SLAB_N_BUCKETS; i++) {
    unsigned int off = i * 8;
    *(unsigned int *)(void *)(base + off) = 0xFFFFFFFFu;
    *(unsigned int *)(void *)(base + off + 4) = 0;
  }
  /* Zero the per-block slab allocation counter (shared memory).
   * After __syncthreads(), all threads in the block see this. */
  unsigned int *bnext = __coqui_slab_block_next();
  *bnext = 0;
}
