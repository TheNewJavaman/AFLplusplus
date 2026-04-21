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

#include <sys/time.h>
#include <pthread.h>

#include "types.h"
#include "forkserver.h"

struct afl_state;  /* forward */

/* Tunable defaults. Override with env vars at runtime. */
#define COQUI_DEFAULT_BATCH_SIZE 8192
#define COQUI_MAX_INPUT_DEFAULT  4096

/* SPSC ring buffer size for producer/consumer batch hand-off.
 * Producer can have up to (COQUI_RING_SIZE - 1) filled slots waiting +
 * 1 slot being filled. COQUI_RING_SIZE=4 gives the producer 3 queued
 * batches of headroom against consumer latency spikes without burning
 * too much pinned memory (each slot has batch_size×max_input_size/4
 * bytes ≈ 8 MB at default settings). */
#define COQUI_RING_SIZE 4

/* ------------------------------------------------------------------------
 * Data types
 * ------------------------------------------------------------------------*/

/* Per-input status returned by the GPU kernel.
 *
 * IMPORTANT: field order + layout MUST match the device-side definition in
 * coqui_mode/runtime/coqui_runtime.h byte-for-byte. The kernel uses its own
 * 16-byte stride; cuMemAlloc on the host uses sizeof of THIS struct; if
 * the two disagree, writes alias across slots and the host reads garbage.
 *
 * Prior versions of this struct were 8 bytes with a reordered field list —
 * a latent bug that caused every non-novel input to appear "crashed" on the
 * host (phase=COMPLETE read as asan_error=6). Fixed in iter5 P1 as part of
 * adding crash_sig for crash-verify deduplication. */
typedef struct coqui_status {
  u8  phase;          /* kernel-side phase marker (PHASE_COMPLETE on success) */
  u8  signal;         /* POSIX signal number or 0 */
  u8  asan_error;     /* non-zero if ASan check tripped */
  u8  ubsan_fatal;    /* non-zero if non-recoverable UBSan fired */
  u32 crash_sig;      /* FNV-1a signature of classified cov_map (for dedup) */
  u64 _reserved1;
} coqui_status_t;     /* 16 bytes — DO NOT CHANGE without updating runtime header */

/* One ping-pong half: packed input bytes + metadata + device mirrors. */
typedef struct coqui_batch {
  /* Host-side pinned buffers (malloc'd in stub; cuMemAllocHost in real GPU). */
  u8             *h_input_bytes;  /* packed inputs, each start 8-aligned */
  u32            *h_offsets;      /* per-slot offset into h_input_bytes */
  u32            *h_input_lens;   /* per-slot length */
  u8             *h_novelty;      /* 1 bit per slot */
  coqui_status_t *h_status;

  /* Device buffers (CUdeviceptr values); 0 in stub, non-zero in real GPU. */
  unsigned long long d_input_bytes;
  unsigned long long d_offsets;
  unsigned long long d_input_lens;
  unsigned long long d_novelty;
  unsigned long long d_status;

  /* Stream + completion event (CUstream / CUevent) */
  void *stream;
  void *completion_event;

  /* Fill state. */
  u32 n_inputs;    /* slots used so far */
  u32 bytes_used;  /* bytes consumed in h_input_bytes (unaligned cursor) */

  /* Adaptive batch-timeout (B1): wall-clock timestamp at cuEventRecord. Used
   * by coqui_await_and_process to compute healthy-batch latency. Per-batch
   * (not per-context) because ping-pong means launch(X) → flip → wait(Y)
   * where Y was launched in the previous flip and X's launch would otherwise
   * clobber a context-level field. 0 = no launch yet. */
  unsigned long long launch_start_us;
} coqui_batch_t;

/* Per-afl-state coqui context. */
typedef struct coqui_ctx {
  /* SPSC ring buffer of batch slots. Producer (AFL main thread) fills
   * producer_current; when full, it pushes producer_current's index onto
   * the ring and acquires a new slot from the free pool. Consumer thread
   * pops slots, runs launch+await+verify sequentially, then returns the
   * slot to the free pool.
   *
   * Ring invariants:
   *   filled = write_idx - read_idx  (modular, via u32 wraparound)
   *   ring is empty when write_idx == read_idx
   *   ring is full when filled == COQUI_RING_SIZE - 1 (one slot always
   *   reserved for the producer-current in-progress batch). */
  coqui_batch_t slots[COQUI_RING_SIZE];
  u32 slot_queue[COQUI_RING_SIZE];  /* slot indices in FIFO order */
  u32 write_idx;                     /* producer cursor (mod 2^32) */
  u32 read_idx;                      /* consumer dequeue cursor (launched) */
  u32 processed_idx;                 /* consumer completion cursor (await+verify done) */
  u32 producer_current;              /* index of slot producer is filling */

  /* Per-ctx pthread state. The same mutex protects all ring fields and
   * guards coarse AFL-state mutations during consumer verify.
   * ring_not_empty / ring_not_full implement the SPSC block/wake; drained
   * is broadcast by the consumer when ring becomes empty (for flush_batch
   * and shutdown). */
  pthread_mutex_t ring_mutex;
  pthread_cond_t  ring_not_empty;
  pthread_cond_t  ring_not_full;
  pthread_cond_t  ring_drained;
  pthread_t consumer_tid;
  u8  consumer_started;
  u8  consumer_should_exit;
  /* Force-reset synchronization. Consumer sets force_reset_in_progress
   * under ring_mutex, broadcasts ring_not_full (waking a blocked producer),
   * and waits for producer_yielded. Producer, seeing the flag at its next
   * push attempt, sets producer_yielded and waits on ring_not_full. */
  u8  force_reset_in_progress;
  u8  producer_yielded;

  /* Coarse AFL-state mutex. Held by the consumer around
   * process_input_via_cpu_fsrv (write_to_testcase + fuzz_run_target +
   * save_if_interesting) to serialize all mutations of afl->queue_buf,
   * afl->virgin_bits, afl->queued_items, afl->saved_crashes, and associated
   * on-disk files. Producer holds it briefly around queue-scheduling reads
   * at the top of each fuzz_one iteration. Havoc body does NOT hold it. */
  pthread_mutex_t afl_state_mutex;

  u32 batch_size;      /* snapshot of afl->gpu_batch_size */
  u32 max_input_size;  /* snapshot of afl->max_length (with fallback) */
  u32 byte_budget;     /* size of each h_input_bytes buffer */
  u32 map_size;        /* snapshot of afl->fsrv.map_size */

  u64 oversized_count; /* inputs skipped because they alone exceed byte_budget */
  u64 launch_count;    /* batches launched so far */
  u64 total_submits;   /* diagnostic: total coqui_submit_input calls */
  u64 crash_dedup_hits;/* crash-verify calls skipped because signature already seen in batch */
  u64 crash_verify_calls; /* crash-verify calls actually issued (for ratio sanity) */

  /* Host-side per-phase timing accumulators (reset at each [coqui-rate]
   * print).  Each batch contributes one sample; dividing by dl
   * (batches-in-window) gives avg us/batch for that phase.
   *
   * t_submit_us : time inside coqui_launch_batch (API submissions — HtoD,
   *               cuLaunchKernel, DtoH, cuEventRecord; all async, so this
   *               is dominated by driver latency not actual work).
   * t_await_us  : time in the cuStreamQuery poll loop inside
   *               coqui_await_and_process — the dominant wall-clock cost,
   *               reflecting actual GPU execution + implicit DtoH.
   * t_verify_us : CPU-side novelty + crash-verify forkserver calls, post-
   *               kernel.  Separates CPU verify cost from GPU wait. */
  u64 t_submit_us;
  u64 t_await_us;
  u64 t_verify_us;

  /* CPU-side speed gate (ported from coqui driver 9d85772). Blocks corpus
   * admission of inputs that verify >10× slower than baseline, preventing
   * pathologically slow inputs from becoming havoc parents. */
  u64 verify_baseline_us;  /* fixed baseline from first 10 verifications */
  u64 verify_baseline_sum; /* accumulator while building baseline */
  u32 verify_baseline_n;   /* number of verifications seen so far (capped 10) */
  u64 slow_skipped;        /* inputs rejected by the speed gate */

  /* Per-second throughput logger state (always on). */
  u8             rate_log_init;
  struct timeval rate_log_t0;
  u64            rate_log_last_launches;
  u64            rate_log_last_submits;

  /* CUDA handles (populated by coqui_init). void* avoids cuda.h dependency. */
  void *cu_ctx;       /* CUcontext */
  void *cu_module;    /* CUmodule */
  void *cu_kernel;    /* CUfunction */
  void *stream_a;     /* CUstream */
  void *stream_b;     /* CUstream */

  /* Device-persistent buffers (CUdeviceptr = u64) */
  unsigned long long d_virgin_map;
  unsigned long long d_global_statics_pool;
  unsigned long long d_slab_pool;

  /* Config from .conf sidecar */
  unsigned int real_stack_size;
  unsigned long long batch_timeout_us;

  /* Adaptive batch-timeout (sub-proposal B1).
   *
   * Healthy batch wall-clock latency ring buffer. A "healthy" batch is one
   * that completed via cuStreamQuery == CUDA_SUCCESS before the cull
   * deadline fired (NOT culled, NOT force-reset). Once we have
   * >= COQUI_LAT_MIN_SAMPLES samples, we set batch_timeout_us to
   * clamp(COQUI_LAT_MULT * P95, COQUI_LAT_FLOOR_US, COQUI_LAT_CEIL_US)
   * so pathological inputs get culled sooner (saving ~8s of the ~9s per
   * bad batch: 3s static timeout + 5s hard-deadline + ~1s force-reset).
   *
   * Env-var override `AFL_COQUI_TIMEOUT_US` remains authoritative:
   * `timeout_env_override` is captured at init and, if non-zero, the
   * adaptive path never touches batch_timeout_us.
   */
  unsigned long long batch_latency_ring[128];  /* us */
  u32                batch_latency_count;      /* total samples pushed */
  u32                batch_latency_head;       /* next write index */
  u8                 timeout_env_override;     /* 1 if AFL_COQUI_TIMEOUT_US set */
} coqui_ctx_t;

/* Adaptive batch-timeout tunables (B1). */
#define COQUI_LAT_RING_SIZE     128
#define COQUI_LAT_MIN_SAMPLES   32
#define COQUI_LAT_P95_PCT       95
#define COQUI_LAT_MULT          3ULL
#define COQUI_LAT_FLOOR_US      500000ULL    /* 500 ms */
#define COQUI_LAT_CEIL_US       3000000ULL   /* 3 s */

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

/* Lock the coqui AFL-state mutex. Called by the producer (AFL main thread)
 * at fuzz_one scheduling boundaries to serialize against the consumer's
 * save_if_interesting / write_to_testcase work. Cheap no-op in non-gpu mode
 * (callers gate on afl->gpu_mode). */
void coqui_state_lock(struct afl_state *afl);
void coqui_state_unlock(struct afl_state *afl);

#endif /* _HAVE_AFL_FUZZ_COQUI_H */
