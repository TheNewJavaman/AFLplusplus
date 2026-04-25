/*
 * coqui_runtime.h --- coqui mode device-side runtime contract.
 *
 * Shared header between the LLVM pass plugin and the GPU runtime .c files.
 * Declares types, constants, and function prototypes used across passes.
 *
 * Compile with: --target=nvptx64-nvidia-cuda -O2 -ffreestanding
 */

#ifndef _COQUI_RUNTIME_H
#define _COQUI_RUNTIME_H

#include <stdint.h>

/* Branch-prediction hints. Lowered by clang/llvm to PTX branch-hint metadata
 * which the PTX assembler uses to lay out the unlikely successor at the
 * bottom of the function body. On NVPTX SIMT execution doesn't "predict" in
 * the CPU sense, but the layout effect is real: the hot path stays in
 * I-cache and the cold path lives in basic blocks the warp PC rarely visits.
 * Complements PR #16's attribute sweep (cold/always_inline/etc.). */
#define likely(x)   __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)

/* Basic types (matching AFL's u8/u32/u64 convention) */
typedef uint8_t  u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;

/* -- Constants -- */

#define COQUI_COV_MAP_SIZE 65536u        /* 64 KB — AFL default, hardcoded */

/* Bucket class bits (same as AFL count_class_lookup16) */
#define COQUI_BUCKET_0   0x00
#define COQUI_BUCKET_1   0x01
#define COQUI_BUCKET_2   0x02
#define COQUI_BUCKET_3   0x04
#define COQUI_BUCKET_4_7 0x08
#define COQUI_BUCKET_8_15   0x10
#define COQUI_BUCKET_16_31  0x20
#define COQUI_BUCKET_32_127 0x40
#define COQUI_BUCKET_128_UP 0x80

/* Phase markers (written to coqui_status_t.phase by kernel logic) */
#define COQUI_PHASE_EMPTY       0
#define COQUI_PHASE_START       1
#define COQUI_PHASE_EXECUTING   2
#define COQUI_PHASE_EXITED      3
#define COQUI_PHASE_BUCKETING   4
#define COQUI_PHASE_VIRGIN_CMP  5
#define COQUI_PHASE_COMPLETE    6
#define COQUI_PHASE_RESERVED    7

/* Trap reason codes (stored in coqui_status_t.trap_reason). Numbering
 * matches legacy coqui so host-side tooling that knows the old values
 * keeps working. 0 = no trap (the status slot's ordinary state). */
#define COQUI_TRAP_NONE          0
#define COQUI_TRAP_SETJMP        1
#define COQUI_TRAP_LONGJMP       2
#define COQUI_TRAP_SIGACTION     3
#define COQUI_TRAP_FORK          4
#define COQUI_TRAP_DLOPEN        5
#define COQUI_TRAP_SYSCALL       6
#define COQUI_TRAP_PTHREAD       7
#define COQUI_TRAP_EXEC          8
#define COQUI_TRAP_PIPE          9
#define COQUI_TRAP_FENV          10
#define COQUI_TRAP_OOM           11
#define COQUI_TRAP_STACK_OVERFLOW 12
#define COQUI_TRAP_DEVIRT        13
#define COQUI_TRAP_EXCEPTION     14
/* SyscallTransform-emitted reasons. KILL covers raise/kill (signal delivery
 * to self/other); FORK covers fork/vfork/clone (process creation); EXEC
 * covers execve/execv/execvp/execlp/system (process replacement). The
 * existing COQUI_TRAP_FORK=4 / COQUI_TRAP_EXEC=8 are kept for backward
 * compatibility with host-side tooling that may have been compiled against
 * the legacy values; new pass-emitted stubs use the SYSCALL_* aliases. */
#define COQUI_TRAP_SYSCALL_KILL  15
#define COQUI_TRAP_SYSCALL_FORK  16
#define COQUI_TRAP_SYSCALL_EXEC  17
#define COQUI_TRAP_UNSUPPORTED   255

/* Stack-canary sentinel. Written once into an alloca at the outermost kernel
 * frame; compared at every kernel exit point. On NVPTX, allocas and call-stack
 * frames share the per-thread .local region, so a runaway recursion that
 * exceeds CU_LIMIT_STACK_SIZE will eventually corrupt this slot. Mismatch ->
 * __coqui_trap_with_reason(COQUI_TRAP_STACK_OVERFLOW).
 *
 * Value is chosen to be a 64-bit constant unlikely to arise naturally
 * (no run of identical bytes, no ASCII text, no common poison pattern). */
#define COQUI_STACK_CANARY 0xCA7A1C0FFEE0DE50ULL

/* Per-thread status reported to the host.
 *
 * Layout is BYTE-IDENTICAL to the host-side coqui_status_t in
 * include/afl-fuzz-coqui.h (same field order, same sizeof=16). The kernel
 * writes via its 16-byte stride; cuMemAlloc on the host uses the host
 * sizeof, so the two MUST agree or writes alias across slots.
 *
 * _reserved1 was split to carry trap_reason (see COQUI_TRAP_*). The lower
 * byte is the reason code; the upper 3 bytes remain reserved / zero. */
typedef struct coqui_status {
    u8  phase;
    u8  signal;        /* POSIX signal number or 0 */
    u8  asan_error;    /* non-zero if ASan check tripped */
    u8  ubsan_fatal;   /* non-zero if a non-recoverable UBSan check fired */
    u32 crash_sig;     /* FNV-1a hash of classified cov_map; 0 if not crashed */
    u32 ubsan_error;   /* UBSan error code (0=none, 1=overflow, 2=div-zero,
                          3=shift, 4=type-mismatch, 5=oob, 6=ptr-overflow,
                          7=unreachable, 8=load-invalid, 9=float-cast,
                          10=implicit-conv, 11=missing-return, 12=vla-bound,
                          13=nonnull-arg, 14=nonnull-return, 15=dynamic-type) */
    u8  trap_reason;   /* COQUI_TRAP_* value, 0 = no trap */
    u8  _reserved_1;
    u8  _reserved_2;
    u8  _reserved_3;
} coqui_status_t;       /* 16 bytes */

/* -- Device-side helpers (implemented in runtime .c files) -- */

/* Thread identity. `const, nothrow`: the underlying PTX %tid/%ntid/%ctaid
 * registers are invariant for a given thread, so the result depends only
 * on (implicit) thread state — LLVM may CSE/hoist calls across loops in
 * any TU that includes this header. */
__attribute__((const, nothrow))
u32 __coqui_fuzz_tid(void);

/* Trap / exit */
void __coqui_trap(void);   /* PTX `trap;` — unrecoverable */
void __coqui_exit(void);   /* PTX `exit;` — this thread exits, kernel continues */

/* Release slab allocations for this thread (see coqui_slab.c). Called from
 * __coqui_trap_with_reason() and at clean thread exit so reclaimed ranges
 * return to the global free stack for reuse by other threads. */
void __coqui_slab_release_thread(void);

/* Signalled-exit variant. Stamps the trap_reason field on the thread's
 * status slot (so the host can distinguish OOM / stack-overflow / ...),
 * releases any slab allocations, then calls __coqui_exit() — NOT trap.
 * The kernel keeps running so peer threads still make progress. */
void __coqui_trap_with_reason(u8 reason);

/* Stack-canary check. Reads the u64 at `slot`, compares against
 * COQUI_STACK_CANARY, and calls __coqui_trap_with_reason(COQUI_TRAP_STACK_OVERFLOW)
 * on mismatch. The pass MemoryLayout emits a call at every kernel exit;
 * the slot is an alloca the pass sets up at kernel entry. Returns normally
 * on match so the kernel can clean up and `ret` as usual. */
void __coqui_check_stack_canary(const unsigned long *slot);

/* Status writing */
void __coqui_status_set_phase(u32 tid, u8 phase);

/* Region accessors (set up by MemoryLayout transform at kernel entry) */
u8  *__coqui_cov_base(void);       /* per-thread coverage map (64 KB) */
u8  *__coqui_heap_base(void);      /* per-thread heap */
u8  *__coqui_shadow_base(void);    /* per-thread ASan shadow */
u32  __coqui_heap_size(void);      /* runtime-configured heap size */

/* Coverage runtime */
extern __attribute__((visibility("default")))
u8 __coqui_count_class_lookup[256];
void __coqui_classify_counts(u8 *map);

/* One-pass variant: classify every byte *and* fold a 32-bit FNV-1a hash
 * over the classified map so the host can dedup crash-verify on it. The
 * separate __coqui_classify_counts is retained for any future caller that
 * does not need the hash; kernel entry uses the combined form. */
u32  __coqui_classify_counts_and_sig(u8 *map);

void __coqui_virgin_compare_and_flag(u8 *map, u8 *virgin, u32 *novelty_bitmap);

/* Dedup hash of partial coverage at crash time (called from asan_report).
 * Uses the same FNV-1a algorithm as the classify_and_sig variant so
 * signatures are comparable across crashed and clean threads. */
u32  __coqui_trace_sig(u8 *map);

/* prev_loc (per-thread, for AFL hash instrumentation) */
u32 *__coqui_prev_loc_ptr(void);

/* Heap allocator (coqui_memory.c) */
void *__coqui_malloc(unsigned long size);
void  __coqui_free(void *ptr);
void *__coqui_calloc(unsigned long n, unsigned long size);
void *__coqui_realloc(void *ptr, unsigned long size);

/* ASan heap instrumentation (coqui_asan.c) */
void *__coqui_asan_malloc(unsigned long size);
void  __coqui_asan_free(void *ptr);
void *__coqui_asan_calloc(unsigned long nmemb, unsigned long size);
void *__coqui_asan_realloc(void *ptr, unsigned long new_size);
void  __coqui_asan_check_load_1(void *ptr);
void  __coqui_asan_check_load_2(void *ptr);
void  __coqui_asan_check_load_4(void *ptr);
void  __coqui_asan_check_load_8(void *ptr);
void  __coqui_asan_check_store_1(void *ptr);
void  __coqui_asan_check_store_2(void *ptr);
void  __coqui_asan_check_store_4(void *ptr);
void  __coqui_asan_check_store_8(void *ptr);

/* ASan global-variable red-zone descriptor (matches IR-emitted table).
 *   beg        : base address of the padded global (inner + outer red zone)
 *                -- or NULL for a pool-kind entry (see pool_stride below).
 *   user_size  : bytes the original (pre-instrumentation) global occupied
 *   total_size : user_size + right red zone in bytes
 *   pool_offset: for pool-kind entries, byte offset into the per-thread
 *                statics pool slab. Ignored when pool_stride == 0.
 *   pool_stride: 0 for legacy (absolute-`beg`) entries; positive for
 *                pool-kind entries emitted when runStaticGlobals pooled
 *                writable globals. When non-zero the runtime computes
 *                real_beg = __coqui_global_statics_pool_base
 *                         + __coqui_fuzz_tid() * pool_stride
 *                         + pool_offset
 *                so the per-thread slab each thread sees is checked
 *                against its own pooled-entry locations.
 * The runtime linearly scans __coqui_asan_globals[] from the slow path to
 * detect out-of-bounds accesses that land inside the red zone. */
struct __coqui_asan_global_desc {
    const void   *beg;
    unsigned long user_size;
    unsigned long total_size;
    unsigned long pool_offset;
    unsigned long pool_stride;
};

/* Invoked once per kernel launch (all threads write the same descriptors,
 * which is idempotent). The IR pass emits the table + this call. */
void __coqui_asan_register_globals(const struct __coqui_asan_global_desc *descs,
                                    unsigned long count);

/* ASan slab-pool red-zone descriptor.
 *
 * One entry per live slab allocation. The runtime maintains a shared
 * bounded table (ASAN_MAX_SLAB_DESCS) that all threads append to
 * atomically in __coqui_asan_slab_malloc and clear in
 * __coqui_asan_slab_free. Layout mirrors __coqui_asan_global_desc so the
 * slowpath lookup reads the same fields:
 *   beg        : base address of the user buffer (past the leading red zone)
 *   user_size  : bytes the caller originally requested
 *   total_size : user_size + right red zone in bytes (from beg)
 * beg == NULL marks a freed/empty slot. Slot allocation is monotonic:
 * __coqui_asan_slab_free poisons the entry but the slot index is not
 * recycled, which keeps the slowpath lookup branch-free. */
struct __coqui_asan_slab_desc {
    const void   *beg;
    unsigned long user_size;
    unsigned long total_size;
};

/* Slab-pool allocator hook registration. A future device-side slab runtime
 * (not yet linked in coqui_mode) calls this once from its setup kernel to
 * hand its raw malloc/free pair to the ASan wrapper. Until then,
 * __coqui_asan_slab_malloc returns NULL and __coqui_asan_slab_free is a
 * no-op — so targets without slab support emit identical PTX. */
void __coqui_asan_register_slab(void *(*m)(unsigned long), void (*f)(void *));

/* ASan-aware slab allocator wrappers. The pass redirects user-level
 * slab allocations to these; they call the registered raw allocator
 * with size + 2*red_zone bytes and return a pointer past the leading
 * red zone. */
void *__coqui_asan_slab_malloc(unsigned long size);
void  __coqui_asan_slab_free(void *ptr);

/* Libc replacements (coqui_libc.c) */
unsigned long __coqui_strlen(const char *s);
int  __coqui_strcmp(const char *a, const char *b);
int  __coqui_strncmp(const char *a, const char *b, unsigned long n);
int  __coqui_memcmp(const void *a, const void *b, unsigned long n);
char *__coqui_strchr(const char *s, int c);
double __coqui_strtod(const char *nptr, char **endptr);

/* -- Slab pool runtime (coqui_slab.c) -- */

/* Slab page size: every allocation unit from the pool is this many bytes. */
#define SLAB_SIZE  4096u

/* Per-thread slab control block at slab_pool[tid * 32]:
 *   [0..7]   free_head        (u64) — heap free-list head (coqui_slab.c)
 *   [8..15]  current_heap     (u64) — current heap range pointer
 *   [16..19] heap_limit       (u32)
 *   [20..23] heap_bump_top    (u32)
 *   [24..31] stack_chain_head (u64) — current stack-spill page (coqui_stack_spill.c)
 *
 * Stack-spill page header (16 bytes, at offset 0 of each carved 4 KB page):
 *   [0..7]   prev_page (u64)            — previous page in chain or 0
 *   [8..11]  page_virtual_base (u32)    — sum of usable bytes in earlier pages
 *   [12..15] page_bump_top (u32)        — bytes used in this page (>= 16)
 * Usable space per page: SLAB_SIZE - 16 = 4080 bytes. */
#define COQUI_SLAB_THREAD_CTRL_STRIDE   32u
#define COQUI_SLAB_STACK_HEAD_OFFSET    24u
#define COQUI_STACK_PAGE_HDR_SIZE       16u
#define COQUI_STACK_PAGE_USABLE         (SLAB_SIZE - COQUI_STACK_PAGE_HDR_SIZE)

/* Stack-spill runtime (coqui_stack_spill.c).
 *
 * Per-thread bump pointer through chained 4 KB slab pages. Used by the
 * StackSpill compile pass to redirect oversize allocas into the slab pool.
 * The marker returned by __coqui_stack_save is a u32 absolute virtual offset
 * (sum of bytes used across all pages in the chain to date). _restore
 * rewinds the bump pointer; pages with vbase above the saved marker
 * are dropped from the chain head and reclaimed at thread exit by
 * __coqui_slab_release_thread. They are not re-bumped on the next _alloc.
 *
 * On thread exit / __coqui_trap_with_reason, __coqui_slab_release_thread
 * walks the stack chain (via slab_pool[tid*32+24]) and pushes pages onto
 * the global Treiber free stack alongside the heap chain. */
unsigned int __coqui_stack_save(void);
void *        __coqui_stack_alloc(unsigned int size);
void          __coqui_stack_restore(unsigned int saved);

/* Parameter-free setup. Slab pool globals (__coqui_slab_pool etc.) are
 * bound by the host via cuModuleGetGlobal + cuMemcpyHtoD before launch;
 * this call just computes the per-launch ctrl_slabs derived from grid
 * dimensions and registers the slab malloc/free pair with ASan. */
void __coqui_slab_setup(void);

/* Called by thread 0 of every block to initialize the shared-memory
 * buckets, followed by a __syncthreads() barrier in the kernel entry. */
void __coqui_slab_init_block(void);

/* Device-side slab allocator (OOM → NULL). Called by __coqui_asan_malloc
 * as a fall-through when the per-thread heap is exhausted. */
void *__coqui_slab_malloc(unsigned long size);
void  __coqui_slab_free(void *ptr);

/* -- Kernel entry (generated by FuzzEntry transform) -- */

/* Renamed user entry (was LLVMFuzzerTestOneInput). Called by kernel per-thread. */
int __coqui_fuzz_execute(const unsigned char *data, unsigned long size);

/* Device-global virgin map (allocated at module load, accessed via cuModuleGetGlobal) */
extern u8 __coqui_virgin_map[COQUI_COV_MAP_SIZE];

/* -- Line-trace runtime (oracle mode; coqui_trace.c) --
 *
 * Per-line execution recorder gated by the `-coqui-line-trace` opt flag.
 * When enabled, the LineTrace pass inserts calls to __coqui_trace_line()
 * at every unique source line; the host allocates a buffer + counter
 * array and binds them via cuModuleGetGlobal + HtoD. Production runs
 * leave the symbols at NULL and the recorder short-circuits.
 *
 * Buffer geometry (must match host afl-fuzz-coqui.h):
 *   trace[t * COQUI_TRACE_MAX_ENTRIES + i] == ID of the i'th line
 *                                              executed by thread t
 *   __coqui_trace_count[t] == entries written by thread t (≤ MAX)
 *
 * On overflow, the recorder stops for that thread and __coqui_trace_count
 * remains pinned at COQUI_TRACE_MAX_ENTRIES — the host treats this as a
 * divergence signal (the CPU expected trace must also fit). */
extern unsigned int *__coqui_trace_buffer;     /* tid-indexed stripe base */
extern unsigned int *__coqui_trace_count;      /* per-thread fill count */
void __coqui_trace_line(unsigned int line_id);

#endif /* _COQUI_RUNTIME_H */
