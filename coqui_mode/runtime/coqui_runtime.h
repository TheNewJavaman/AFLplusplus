/*
 * coqui_runtime.h --- cuAFL device-side runtime contract.
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
    u8  ubsan_fatal;   /* non-zero if non-recoverable UBSan (future) */
    u32 crash_sig;     /* FNV-1a hash of classified cov_map; 0 if not crashed */
    u64 _reserved1;
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
u8  *__coqui_touch_base(void);     /* per-thread touched-region summary bitmap (256 B) */
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

/* Sparse variant: classify + sig driven by a per-thread 2048-bit summary
 * bitmap (256 bytes) that Coverage.cpp maintains alongside cov_map
 * increments. Each summary bit covers 256 B (32 u64s) of cov_map. Output
 * is byte-identical to the full-walk variant when the summary is strictly
 * >= the set of touched regions; Coverage emits bit-sets inline with cov
 * increments so a partial trace (ASan mid-BB abort) takes the full-walk
 * crash path via __coqui_trace_sig instead. */
u32  __coqui_classify_counts_and_sig_sparse(u8 *map, u8 *touch);

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
