/*
 * afl-fuzz-coqui.h --- coqui mode coqui_mode contract.
 *
 * Defines the interface between core afl-fuzz and the GPU (coqui) backend.
 * A hollow stub implementation in afl-fuzz-coqui.c makes coqui mode compile and
 * run without a real GPU; the real backend will drop in behind this contract.
 *
 * Spec: docs/superpowers/specs/2026-04-18-coqui-gpu-backend-design.md
 */

#ifndef _HAVE_AFL_FUZZ_COQUI_H
#define _HAVE_AFL_FUZZ_COQUI_H

#include <sys/time.h>

#include "types.h"
#include "forkserver.h"

struct afl_state;  /* forward */

/* Tunable defaults. Override with env vars at runtime. */
#define COQUI_DEFAULT_BATCH_SIZE 32768  /* legacy parity; cap 65536 */
#define COQUI_MAX_INPUT_DEFAULT  4096

/* Dictionary device-global limits (must match coqui_runtime.c). */
#define COQUI_MAX_EXTRAS         4096u
#define COQUI_MAX_EXTRAS_BYTES   131072u  /* 128 KB */

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
  u32 ubsan_error;    /* UBSan error code (0=none, 1=overflow, 2=div-zero,
                         3=shift, 4=type-mismatch, 5=oob, 6=ptr-overflow,
                         7=unreachable, 8=load-invalid, 9=float-cast,
                         10=implicit-conv, 11=missing-return, 12=vla-bound,
                         13=nonnull-arg, 14=nonnull-return, 15=dynamic-type) */
  u8  trap_reason;    /* COQUI_TRAP_* value (0=none, 11=OOM, 12=STACK_OVERFLOW, …) */
  u8  _reserved_1;
  u8  _reserved_2;
  u8  _reserved_3;
} coqui_status_t;     /* 16 bytes — DO NOT CHANGE without updating runtime header */

/* Trap reason codes mirrored from coqui_mode/runtime/coqui_runtime.h. Host-
 * side helpers (afl-fuzz-coqui.c) scan status[i].trap_reason after each
 * batch to find threads that exited via __coqui_trap_with_reason(). */
#define COQUI_TRAP_NONE          0
#define COQUI_TRAP_OOM           11
#define COQUI_TRAP_STACK_OVERFLOW 12
/* Per-thread clock64 budget exhausted (Exp #51): the Coverage pass emits
 * periodic budget checks; a thread that exceeds AFL_COQUI_THREAD_BUDGET_US
 * exits via __coqui_trap_with_reason. Host treats it like OOM/stack —
 * rerun on CPU forkserver. Default OFF (env unset → cycles_cap=0). */
#define COQUI_TRAP_THREAD_BUDGET_EXHAUSTED 19

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
  coqui_batch_t  ping;
  coqui_batch_t  pong;
  coqui_batch_t *pending;    /* filling now (CPU side) */
  coqui_batch_t *executing;  /* on GPU (or most recent done) */

  u32 batch_size;      /* snapshot of afl->coqui_batch_size */
  u32 max_input_size;  /* snapshot of afl->max_length (with fallback) */
  u32 byte_budget;     /* size of each h_input_bytes buffer */
  u32 map_size;        /* snapshot of afl->fsrv.map_size */

  u64 oversized_count; /* inputs skipped because they alone exceed byte_budget */
  u64 launch_count;    /* batches launched so far */
  u64 total_submits;   /* diagnostic: total coqui_submit_input calls */
  u64 crash_dedup_hits;/* crash-verify calls skipped because signature already seen in batch */
  u64 crash_verify_calls; /* crash-verify calls actually issued (for ratio sanity) */

  /* Persistent cross-batch crash-sig dedup set. Layered above the intra-batch
   * 256-slot dedup: for each crashed non-novel slot that SURVIVES intra-batch
   * dedup, probe this set; if seen, skip (persistent_dedup_hits++); if new,
   * insert and verify via CPU forkserver.
   *
   * Rationale: intra-batch dedup catches ~85% of crashes (most recur within
   * one batch). The remaining 15% (~195 unique sigs/batch for cjson) often
   * recur across batches from the same parser failure site — those are the
   * target of this persistent layer.
   *
   * Open-addressing linear probe. u8 used[] tracks slot occupancy so sig==0
   * is a valid key. Table full → insertion fails, but probe returns "not
   * seen" so we still verify (safe/conservative fallback vs. silently
   * dropping potential new crashes). Size: 1M slots × (4B+1B) = 5 MB. */
  u32 *crash_sig_seen;            /* hash-set of sigs, length = crash_sig_seen_cap */
  u8  *crash_sig_seen_used;       /* 1 = slot occupied, 0 = empty */
  u32  crash_sig_seen_cap;        /* capacity (power of 2) */
  u32  crash_sig_seen_count;      /* slots in use (observability) */
  u64  crash_sig_persistent_hits; /* cross-batch dedup hits (telemetry, lifetime) */

  /* Per-[coqui-rate]-window dedup counters (reset at each print, like
   * t_submit_us).  Using window deltas rather than lifetime totals ensures
   * the "avg/batch" figures printed in [coqui-rate] reflect only the current
   * window, not the cumulative run.  This matters most after a force-reset:
   * coqui_init zeroes launch_count but force_reset restores the large
   * crash_sig_persistent_hits value, making (lifetime / new_launch_count)
   * report astronomically wrong per-batch averages until launch_count grows
   * large enough to dilute the ratio.  Window counters reset naturally after
   * each force-reset because coqui_init zero-initialises them, so they are
   * intentionally NOT preserved in coqui_force_reset's save/restore block. */
  u64  crash_dedup_hits_window;       /* intra-batch dedup hits this window */
  u64  crash_verify_calls_window;     /* CPU verify calls this window */
  u64  crash_sig_persist_hits_window; /* cross-batch dedup hits this window */

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

  /* Finer-grained submit-phase breakdown (subsets of t_submit_us). Each
   * measures the wall-clock wrap around the corresponding CUDA driver
   * calls: HtoD memcpy+memset issues, kernel launch, DtoH memcpy issues.
   * Residual (t_submit_us - htod - launch - dtoh) is small-change glue. */
  u64 t_htod_us;
  u64 t_launch_us;
  u64 t_dtoh_us;

  /* CPU mutation time per batch is computed as a RESIDUAL in the
   * [coqui-rate] print: (wall_per_batch - submit - await - verify). */

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
  unsigned long long d_slab_shadow;
  unsigned long long d_slab_next;
  unsigned long long slab_pool_size;
  unsigned int       slab_block_budget;

  /* OOM rerun counters (set by the post-batch trap_reason scan). */
  u64 oom_inputs_found;
  u64 oom_reruns_completed;
  u64 stack_overflow_inputs_found;
  u64 cpu_rerun_crashes;
  /* Per-thread clock64 budget exhaustions (Exp #51). Counts threads that
   * tripped __coqui_check_thread_budget; each is rerun on the CPU
   * forkserver and counted in oom_reruns_completed alongside OOM/stack. */
  u64 thread_budget_inputs_found;

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

  /* GPU-side kernel phase cycles, summed across all threads in the batch.
   * Read from __coqui_kernel_timing[5] after each batch completes,
   * accumulated across the rate-log window. Divide by (dl * batch_size) for
   * average cycles/thread per phase; divide by ~1.5 GHz to convert to µs.
   * Slots: 0=memory_init, 1=fuzz_execute, 2=classify+sig, 3=virgin_compare,
   * 4=total (clk_e - clk_a). */
  unsigned long long d_kernel_timing;  /* CUdeviceptr to the 5-u64 array */
  u64                k_cycles[5];      /* host-side accumulator, reset at print */
  u32                k_batch_count;    /* batches contributing to accumulator */

  /* Oracle-mode line-trace readback (gated by COQUI_ORACLE=1).
   * When the env var COQUI_ORACLE is set at coqui_init time, the host
   * allocates a per-thread trace buffer + counter array and binds them to
   * the cubin's __coqui_trace_buffer / __coqui_trace_count globals via
   * cuModuleGetGlobal + cuMemcpyHtoD (same wiring pattern as the slab
   * pool). The cubin must have been built with `-coqui-line-trace`; if
   * the symbols aren't exported coqui_init logs a warning and continues
   * with oracle disabled (production runs are unaffected).
   *
   * After each kernel completes, we DtoH thread-0's count + buffer and
   * print the recorded sequence to stderr. Comparing against an
   * expected CPU-corpus trace is intentionally a separate follow-up;
   * the buffer being readable from the host is enough to validate the
   * device-side path end-to-end. */
  u8                  oracle_enabled;       /* 1 when COQUI_ORACLE=1 + globals bound */
  unsigned long long  d_trace_buffer;       /* CUdeviceptr — trace stripe pool */
  unsigned long long  d_trace_count;        /* CUdeviceptr — per-thread counters */
  u32                *h_trace_t0_buf;       /* host scratch for thread-0 readback */
  u32                 h_trace_t0_count;     /* most-recent thread-0 count */

  /* GPU-side mutation (Phase 2+3). When AFL_COQUI_GPU_MUTATE=1, each GPU
   * thread applies additional havoc mutations before calling the harness.
   *
   * Power-schedule mode: the host broadcasts each parent to
   * round_up_32(stage_max) consecutive slots. Each thread independently
   * picks stacking depth from its PRNG: steps = 1 + rand_below(stack_max).
   * stack_max is 4 early, 8 after 10 minutes (mirrors AFL's havoc_stack_pow2). */
  u8                  gpu_mutate_enabled;   /* 0=off, 1=on (env-level) */
  u8                  gpu_mutate_active;    /* per-batch: 1 during havoc/splice, 0 otherwise */
  unsigned long long  d_mutate_flag;        /* CUdeviceptr — __coqui_gpu_mutate_enabled */
  unsigned long long  d_mutate_prng;        /* CUdeviceptr — per-thread PRNG seeds */
  size_t              d_mutate_prng_sz;     /* size of PRNG symbol on device */
  u64                *h_mutate_prng;        /* pinned host buffer for PRNG seeds */
  unsigned long long  d_mutate_stack_max;   /* CUdeviceptr — __coqui_mutate_stack_max */

  /* Per-parent H2D parent table (Change 1: sparse H2D).
   *
   * Instead of copying each parent's bytes to 128 thread slots (broadcasting
   * the same data 128x), upload each parent ONCE into a compact parent table.
   * GPU threads read from the table via __coqui_parent_offsets[parent_idx].
   *
   * h_parent_bytes: packed parent data, one copy per unique parent
   * h_parent_offsets: byte offset into parent buffer for each parent
   * h_parent_lens: byte length of each parent
   * h_parent_idx: per-thread mapping — which parent index this thread uses
   * d_parent_*: device-side globals bound at init */
  u8                 *h_parent_bytes;      /* pinned host buffer for packed parents */
  u32                *h_parent_offsets;     /* pinned: offset[parent_i] */
  u32                *h_parent_lens;        /* pinned: len[parent_i] */
  u32                *h_parent_idx;         /* pinned: per-thread parent index */
  unsigned long long  d_parent_bytes;       /* CUdeviceptr — __coqui_parent_bytes */
  unsigned long long  d_parent_offsets;     /* CUdeviceptr — __coqui_parent_offsets */
  unsigned long long  d_parent_lens;        /* CUdeviceptr — __coqui_parent_lens */
  unsigned long long  d_parent_idx;         /* CUdeviceptr — __coqui_parent_idx */
  u32                 parent_count;         /* unique parents packed this batch */
  u32                 parent_bytes_used;    /* bytes consumed in h_parent_bytes */

  /* Device-side parent_count global (for splice: threads need to know how
   * many parents exist in this batch to pick a splice source). */
  unsigned long long  d_parent_count;      /* CUdeviceptr — __coqui_parent_count */

  /* Dictionary (extras) device sync.
   *
   * The host packs AFL's extras[] + a_extras[] into flat device-global arrays
   * (defined in coqui_runtime.c). Sync happens lazily: only when extras_cnt
   * or a_extras_cnt changes between batches. The device-global arrays are
   * fixed-size (COQUI_MAX_EXTRAS entries, COQUI_MAX_EXTRAS_BYTES bytes). */
  unsigned long long  d_extras_data;       /* CUdeviceptr — __coqui_extras_data */
  unsigned long long  d_extras_offsets;    /* CUdeviceptr — __coqui_extras_offsets */
  unsigned long long  d_extras_lens;       /* CUdeviceptr — __coqui_extras_lens */
  unsigned long long  d_extras_cnt;        /* CUdeviceptr — __coqui_extras_cnt */
  unsigned long long  d_a_extras_cnt;      /* CUdeviceptr — __coqui_a_extras_cnt */
  u32                 last_extras_cnt;     /* cached: last synced extras_cnt */
  u32                 last_a_extras_cnt;   /* cached: last synced a_extras_cnt */
} coqui_ctx_t;

/* Per-thread trace stripe geometry. Must match
 * coqui_mode/runtime/coqui_trace.c COQUI_TRACE_BUFFER_BYTES /
 * COQUI_TRACE_MAX_ENTRIES — if the runtime constants change, update
 * here so the host's cuMemAlloc and DtoH sizes track the device's. */
#define COQUI_TRACE_BUFFER_BYTES 65536u
#define COQUI_TRACE_MAX_ENTRIES  (COQUI_TRACE_BUFFER_BYTES / 4u)

/* Adaptive batch-timeout tunables (B1). */
#define COQUI_LAT_RING_SIZE     128
#define COQUI_LAT_MIN_SAMPLES   32
#define COQUI_LAT_P95_PCT       95
#define COQUI_LAT_MULT          3ULL
#define COQUI_LAT_FLOOR_US      200000ULL    /* 200 ms — tightened from 500ms
                                                     so adaptive can hit
                                                     low-latency targets
                                                     (cjson healthy ~50ms) */
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

#endif /* _HAVE_AFL_FUZZ_COQUI_H */
