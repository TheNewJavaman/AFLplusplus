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

/* Per-thread status reported to the host.
 *
 * Layout is BYTE-IDENTICAL to the host-side coqui_status_t in
 * include/afl-fuzz-coqui.h (same field order, same sizeof=16). The kernel
 * writes via its 16-byte stride; cuMemAlloc on the host uses the host
 * sizeof, so the two MUST agree or writes alias across slots. */
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
    u32 _reserved1;
} coqui_status_t;       /* 16 bytes */

/* -- Device-side helpers (implemented in runtime .c files) -- */

/* Thread identity */
u32 __coqui_fuzz_tid(void);

/* Trap / exit */
void __coqui_trap(void);   /* PTX `trap;` — unrecoverable */
void __coqui_exit(void);   /* PTX `exit;` — this thread exits, kernel continues */

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

/* -- Kernel entry (generated by FuzzEntry transform) -- */

/* Renamed user entry (was LLVMFuzzerTestOneInput). Called by kernel per-thread. */
int __coqui_fuzz_execute(const unsigned char *data, unsigned long size);

/* Device-global virgin map (allocated at module load, accessed via cuModuleGetGlobal) */
extern u8 __coqui_virgin_map[COQUI_COV_MAP_SIZE];

#endif /* _COQUI_RUNTIME_H */
