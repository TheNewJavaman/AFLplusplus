/*
 * afl-fuzz-coqui.h --- cuAFL coqui_mode contract.
 *
 * Defines the interface between core afl-fuzz and the GPU (coqui) backend.
 * A hollow stub implementation in afl-fuzz-coqui.c makes cuAFL compile and
 * run without a real GPU; the real backend will drop in behind this contract.
 *
 * Spec: docs/superpowers/specs/2026-04-18-cuafl-gpu-backend-design.md
 */

#ifndef _HAVE_AFL_FUZZ_COQUI_H
#define _HAVE_AFL_FUZZ_COQUI_H

#include "types.h"
#include "forkserver.h"

struct afl_state;  /* forward */

/* Tunable defaults. Override with env vars at runtime. */
#define COQUI_DEFAULT_BATCH_SIZE 8192
#define COQUI_MAX_INPUT_DEFAULT  4096

/* ------------------------------------------------------------------------
 * Data types
 * ------------------------------------------------------------------------*/

/* Per-input status returned by the GPU kernel (or stub). */
typedef struct coqui_status {
  u8  asan_error;     /* non-zero if ASan tripped */
  u8  ubsan_fatal;    /* non-zero if non-recoverable UBSan fired */
  u8  signal;         /* non-zero = signal number that killed this thread */
  u8  timeout_flag;   /* non-zero if this input exceeded per-thread budget */
  u32 _reserved;      /* padding / future use */
} coqui_status_t;

/* One ping-pong half: packed input bytes + metadata + device mirrors. */
typedef struct coqui_batch {
  /* Host-side pinned buffers (malloc'd in stub; cuMemAllocHost in real GPU). */
  u8             *h_input_bytes;  /* packed inputs, each start 8-aligned */
  u32            *h_offsets;      /* per-slot offset into h_input_bytes */
  u32            *h_input_lens;   /* per-slot length */
  u8             *h_novelty;      /* 1 bit per slot */
  coqui_status_t *h_status;

  /* Device mirrors (NULL in stub; non-NULL in real GPU). */
  void *d_input_bytes;
  void *d_offsets;
  void *d_input_lens;
  void *d_coverage;
  void *d_novelty;
  void *d_virgin;
  void *d_status;

  /* Fill state. */
  u32 n_inputs;    /* slots used so far */
  u32 bytes_used;  /* bytes consumed in h_input_bytes (unaligned cursor) */
} coqui_batch_t;

/* Per-afl-state coqui context. */
typedef struct coqui_ctx {
  coqui_batch_t  ping;
  coqui_batch_t  pong;
  coqui_batch_t *pending;    /* filling now (CPU side) */
  coqui_batch_t *executing;  /* on GPU (or most recent done) */

  u32 batch_size;      /* snapshot of afl->gpu_batch_size */
  u32 max_input_size;  /* snapshot of afl->max_length (with fallback) */
  u32 byte_budget;     /* size of each h_input_bytes buffer */
  u32 map_size;        /* snapshot of afl->fsrv.map_size */

  u64 oversized_count; /* inputs skipped because they alone exceed byte_budget */
  u64 launch_count;    /* batches launched so far */
} coqui_ctx_t;

/* ------------------------------------------------------------------------
 * API surface (called from core afl-fuzz)
 * ------------------------------------------------------------------------*/

/* Initialize GPU context, allocate ping-pong buffers, load cubin.
   `cubin_path` is the trailing positional argument from the CLI.
   Fails fast (FATAL) on any error.
   Stub: allocates host buffers only; cubin_path is stashed but not loaded. */
void coqui_init(struct afl_state *afl, const char *cubin_path);

/* Append one mutated input to the pending batch.
   Returns 0 unconditionally — never signals stage bail-out.
   May trigger an async launch if the pending batch fills.
   Stub: packs but never launches (discards accumulated inputs). */
u8 coqui_submit_input(struct afl_state *afl, u8 *buf, u32 len);

/* Force-launch any partial batch and wait for all pending work to drain.
   Called at stage boundaries so afl->queued_items reflects every mutation
   the stage generated before the stage decides whether it was productive.
   Stub: resets pending batch state, calls nothing. */
void coqui_flush_batch(struct afl_state *afl);

/* Single-input synchronous path (seed calibration, sync-in calibration).
   Populates afl->fsrv.trace_bits and returns an fsrv_run_result_t value
   (u8-valued: FSRV_RUN_OK / CRASH / TMOUT / ERROR / NOINST).
   Stub: writes trivial coverage (trace_bits[0] = 1) and returns FSRV_RUN_OK. */
u8 coqui_calibrate_one(struct afl_state *afl, u8 *buf, u32 len);

/* Free resources and tear down CUDA context on normal exit or SIGINT.
   Stub: free()s host buffers; no CUDA calls. */
void coqui_shutdown(struct afl_state *afl);

#endif /* _HAVE_AFL_FUZZ_COQUI_H */
