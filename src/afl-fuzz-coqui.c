/*
 * afl-fuzz-coqui.c --- cuAFL coqui_mode real CUDA implementation.
 *
 * Real CUDA driver API backend for coqui_mode per spec §8.4.
 *
 * Spec: docs/superpowers/specs/2026-04-18-cuafl-gpu-backend-design.md
 */

#include "afl-fuzz.h"
#include "afl-fuzz-coqui.h"
#include "forkserver.h"

#include <stdlib.h>
#include <string.h>

#include <cuda.h>

#define CUCHECK(expr) do {                                                    \
  CUresult _r = (expr);                                                       \
  if (_r != CUDA_SUCCESS) {                                                   \
    const char *_name = NULL;                                                 \
    cuGetErrorName(_r, &_name);                                               \
    FATAL("CUDA error at %s:%d: %s", __FILE__, __LINE__, _name ? _name : "?");\
  }                                                                           \
} while (0)

static unsigned int getenv_u32(const char *name, unsigned int dflt) {
  const char *v = getenv(name);
  if (!v) return dflt;
  return (unsigned int)strtoul(v, NULL, 0);
}

static unsigned long long getenv_u64(const char *name, unsigned long long dflt) {
  const char *v = getenv(name);
  if (!v) return dflt;
  return strtoull(v, NULL, 0);
}

/* Forward declaration of internal helper */
static void alloc_batch_half_cuda(coqui_batch_t *b, coqui_ctx_t *ctx, CUstream stream);

/* ------------------------------------------------------------------------
 * Internal helpers
 * ------------------------------------------------------------------------*/

static void alloc_batch_half_cuda(coqui_batch_t *b, coqui_ctx_t *ctx, CUstream stream) {

  /* Host pinned memory */
  CUCHECK(cuMemHostAlloc((void**)&b->h_input_bytes, ctx->byte_budget, 0));
  CUCHECK(cuMemHostAlloc((void**)&b->h_offsets, ctx->batch_size * 4, 0));
  CUCHECK(cuMemHostAlloc((void**)&b->h_input_lens, ctx->batch_size * 4, 0));
  CUCHECK(cuMemHostAlloc((void**)&b->h_novelty, ctx->batch_size / 8, 0));
  CUCHECK(cuMemHostAlloc((void**)&b->h_status,
          ctx->batch_size * sizeof(coqui_status_t), 0));

  /* Device mirrors (CUdeviceptr) */
  CUdeviceptr p;
  CUCHECK(cuMemAlloc(&p, ctx->byte_budget));
  b->d_input_bytes = (unsigned long long)p;
  CUCHECK(cuMemAlloc(&p, ctx->batch_size * 4));
  b->d_offsets = (unsigned long long)p;
  CUCHECK(cuMemAlloc(&p, ctx->batch_size * 4));
  b->d_input_lens = (unsigned long long)p;
  CUCHECK(cuMemAlloc(&p, ctx->batch_size / 8));
  b->d_novelty = (unsigned long long)p;
  CUCHECK(cuMemAlloc(&p, ctx->batch_size * sizeof(coqui_status_t)));
  b->d_status = (unsigned long long)p;

  b->stream = (void *)stream;
  CUevent ev;
  CUCHECK(cuEventCreate(&ev, CU_EVENT_DEFAULT));
  b->completion_event = (void *)ev;

  b->n_inputs   = 0;
  b->bytes_used = 0;
  b->launch_start_us = 0;  /* adaptive batch-timeout (B1): no launch yet */

}


/* ------------------------------------------------------------------------
 * API implementation
 * ------------------------------------------------------------------------*/

void coqui_init(afl_state_t *afl, const char *cubin_path) {

  coqui_ctx_t *ctx = ck_alloc(sizeof(coqui_ctx_t));

  /* 1. CUDA driver init */
  CUCHECK(cuInit(0));

  int dev_idx = (int)getenv_u32("AFL_COQUI_DEVICE", 0);
  CUdevice dev;
  CUCHECK(cuDeviceGet(&dev, dev_idx));

  CUcontext cuctx;
  /* CUDA 13+ remaps cuCtxCreate → cuCtxCreate_v4(pctx, params, flags, dev).
     Pass NULL for params to get a regular context (documented as valid). */
  CUCHECK(cuCtxCreate_v4(&cuctx, NULL, 0, dev));
  ctx->cu_ctx = (void *)cuctx;

  /* 2. Verify sm_75+ */
  int major, minor;
  CUCHECK(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev));
  CUCHECK(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev));
  if ((major * 10 + minor) < 75) {
    FATAL("coqui requires sm_75+; device is sm_%d%d", major, minor);
  }

  /* 3. Load cubin, resolve kernel, resolve virgin_map global */
  CUmodule mod;
  CUCHECK(cuModuleLoad(&mod, cubin_path));
  ctx->cu_module = (void *)mod;

  CUfunction kernel;
  CUCHECK(cuModuleGetFunction(&kernel, mod, "__coqui_fuzz_kernel"));
  ctx->cu_kernel = (void *)kernel;

  CUdeviceptr d_virgin;
  size_t virgin_sz;
  CUCHECK(cuModuleGetGlobal(&d_virgin, &virgin_sz, mod, "__coqui_virgin_map"));
  if (virgin_sz != 65536) {
    FATAL("__coqui_virgin_map symbol size %zu != 64KB", virgin_sz);
  }
  ctx->d_virgin_map = (unsigned long long)d_virgin;
  CUCHECK(cuMemsetD8(d_virgin, 0, 65536));

  /* 4. Probe the device for its maximum allowed per-thread stack size.
   *    CUDA's cuCtxSetLimit accepts only values that fit within the device's
   *    .local memory budget (per_thread × max-resident-threads × #SMs must
   *    fit in device memory). There's no direct query API; binary-search
   *    downward from the sm_75+ hardware ceiling (512 KB) to find the
   *    largest accepted value. */
  unsigned int total_budget = 524288;   /* sm_75+ hardware ceiling */
  while (total_budget >= 32768) {
    if (cuCtxSetLimit(CU_LIMIT_STACK_SIZE, total_budget) == CUDA_SUCCESS) break;
    total_budget /= 2;
  }
  if (total_budget < 32768) {
    FATAL("device rejected all per-thread stack sizes >= 32 KB");
  }
  size_t actual_set = 0;
  cuCtxGetLimit(&actual_set, CU_LIMIT_STACK_SIZE);
  OKF("coqui per-thread stack budget: %u KB (driver returned %zu)",
      total_budget / 1024, actual_set);

  /* Derive heap and stack regions from the budget.
   *    Layout (per spec §4.2):
   *      [coverage 64 KB][heap H][shadow H/8][real stack S]
   *      where H = (total - 64K - S) * 8/9 and shadow = H/8 */
  unsigned int stack_size = getenv_u32("AFL_COQUI_STACK_SIZE", 32768);
  unsigned int cov = 65536;
  if (stack_size + cov >= total_budget) {
    FATAL("--stack-size %u + 64KB coverage >= total_budget %u",
          stack_size, total_budget);
  }
  unsigned int remaining = total_budget - cov - stack_size;
  unsigned int heap = (remaining * 8) / 9;
  ctx->real_stack_size = stack_size;

  /* 5. Check static stack usage doesn't exceed budget */
  int static_usage;
  CUCHECK(cuFuncGetAttribute(&static_usage,
    CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, kernel));
  if ((unsigned int)static_usage > total_budget - 1024) {
    FATAL("kernel static stack %d B exceeds budget %u B — bump --stack-size",
          static_usage, total_budget);
  }

  /* 6. Batch sizing (with u64 overflow protection from cuAFL T3.6 fixup) */
  ctx->batch_size = 8192;
  ctx->max_input_size = afl->max_length ? afl->max_length : 4096;
  unsigned long long budget64 = ((unsigned long long)ctx->batch_size
                                  * ctx->max_input_size) / 4;
  unsigned long long floor64 = (unsigned long long)ctx->max_input_size * 256;
  if (budget64 < floor64) budget64 = floor64;
  if (budget64 > 0x40000000ULL) budget64 = 0x40000000ULL;  /* MAX_ALLOC cap */
  ctx->byte_budget = (unsigned int)budget64;
  ctx->map_size = 65536;

  /* 7. Create streams */
  CUstream sa, sb;
  CUCHECK(cuStreamCreate(&sa, CU_STREAM_NON_BLOCKING));
  CUCHECK(cuStreamCreate(&sb, CU_STREAM_NON_BLOCKING));
  ctx->stream_a = (void *)sa;
  ctx->stream_b = (void *)sb;

  /* 8. Allocate ping-pong pair */
  alloc_batch_half_cuda(&ctx->ping, ctx, sa);
  alloc_batch_half_cuda(&ctx->pong, ctx, sb);
  ctx->pending   = &ctx->ping;
  ctx->executing = &ctx->pong;

  /* 9. Statics pool — allocate AND bind to the cubin's pool-base symbol.
   *    StaticGlobals pass emits per-thread accesses as `pool_base + tid * size + offset`,
   *    where `pool_base` is the extern global `__coqui_global_statics_pool_base`.
   *    We allocate the backing storage here AND write its device address into
   *    that symbol so the kernel can compute correct per-thread addresses. */
  CUdeviceptr statics_sym;
  size_t statics_sym_sz;
  if (cuModuleGetGlobal(&statics_sym, &statics_sym_sz, mod,
                        "__coqui_statics_per_thread") == CUDA_SUCCESS) {
    unsigned int per_thread = 0;
    CUCHECK(cuMemcpyDtoH(&per_thread, statics_sym, sizeof(unsigned int)));
    if (per_thread > 0) {
      unsigned long long total_pool =
          (unsigned long long)per_thread * ctx->batch_size;
      CUdeviceptr pool;
      CUCHECK(cuMemAlloc(&pool, total_pool));
      CUCHECK(cuMemsetD8(pool, 0, total_pool));
      ctx->d_global_statics_pool = (unsigned long long)pool;

      /* Bind: write `pool` (a CUdeviceptr) into the kernel's
       * `__coqui_global_statics_pool_base` symbol so device-side accesses
       * see the correct base. The symbol is declared as `ptr` (8 bytes). */
      CUdeviceptr pool_base_sym;
      size_t pool_base_sz;
      CUresult br = cuModuleGetGlobal(&pool_base_sym, &pool_base_sz, mod,
                                       "__coqui_global_statics_pool_base");
      if (br != CUDA_SUCCESS) {
        FATAL("StaticGlobals pool allocated but kernel exports no "
              "__coqui_global_statics_pool_base symbol — pass/runtime mismatch");
      }
      if (pool_base_sz != sizeof(CUdeviceptr)) {
        FATAL("__coqui_global_statics_pool_base symbol size %zu != %zu",
              pool_base_sz, sizeof(CUdeviceptr));
      }
      CUCHECK(cuMemcpyHtoD(pool_base_sym, &pool, sizeof(CUdeviceptr)));
      OKF("coqui statics pool: %llu bytes, bound to __coqui_global_statics_pool_base",
          total_pool);
    }
  }

  /* 10. Slab pool (optional) */
  unsigned long long slab_size = getenv_u64("AFL_COQUI_SLAB_SIZE", 0);
  if (slab_size > 0) {
    CUdeviceptr slab;
    CUCHECK(cuMemAlloc(&slab, slab_size));
    CUCHECK(cuMemsetD8(slab, 0, slab_size));
    ctx->d_slab_pool = (unsigned long long)slab;
    /* TODO: invoke __coqui_slab_setup init kernel when slab runtime ported */
  }

  ctx->batch_timeout_us = getenv_u64("AFL_COQUI_TIMEOUT_US", 3000000);
  ctx->timeout_env_override = getenv("AFL_COQUI_TIMEOUT_US") ? 1 : 0;
  ctx->launch_count     = 0;
  ctx->oversized_count  = 0;
  ctx->total_submits    = 0;
  ctx->crash_dedup_hits = 0;
  ctx->crash_verify_calls = 0;
  ctx->rate_log_init    = 0;
  ctx->rate_log_last_launches = 0;
  ctx->rate_log_last_submits  = 0;
  ctx->verify_baseline_us  = 0;
  ctx->verify_baseline_sum = 0;
  ctx->verify_baseline_n   = 0;
  ctx->slow_skipped        = 0;
  /* Adaptive batch-timeout (B1) ring buffer — zero-init. */
  memset(ctx->batch_latency_ring, 0, sizeof(ctx->batch_latency_ring));
  ctx->batch_latency_count  = 0;
  ctx->batch_latency_head   = 0;
  ctx->ping.launch_start_us = 0;
  ctx->pong.launch_start_us = 0;

  afl->coqui = ctx;

  OKF("coqui initialized: sm_%d%d, batch=%u, stack=%u KB, heap=%u KB, total=%u KB",
      major, minor, ctx->batch_size, stack_size/1024, heap/1024, total_budget/1024);

}

/* Translate GPU status fields to an AFL fault code.
 *
 * Note: the device does not currently set a per-thread timeout_flag — if/when
 * it does, add a field to coqui_status_t in both runtime.h and afl-fuzz-coqui.h
 * at matching byte offsets, then check it here. */
static u8 translate_gpu_status(coqui_status_t *s) {
  if (s->asan_error) return FSRV_RUN_CRASH;
  if (s->ubsan_fatal) return FSRV_RUN_CRASH;
  if (s->signal) return FSRV_RUN_CRASH;
  return FSRV_RUN_OK;
}

/* Run a GPU-flagged or GPU-crashed input through the CPU forkserver for
   real trace_bits, then call save_if_interesting.

   CPU-side speed gate (ported from coqui 9d85772): establish a fixed
   baseline verify time from the first 10 CPU verifications. Any later
   input that takes >10× that baseline to verify is rejected from corpus
   admission. Pathologically slow inputs (deep recursion, hash-collision
   storms) are what cause GPU kernel timeouts; blocking them from the
   corpus prevents them from becoming havoc parents that generate even
   slower children. Fixed (not EMA) baseline so the threshold doesn't
   drift up as slow inputs appear. */
static void process_input_via_cpu_fsrv(afl_state_t *afl,
                                        u8 *input, u32 len) {
  u32 new_size = write_to_testcase(afl, (void **)&input, len, 0);
  if (new_size == 0) return;

  u64 t0 = get_cur_time_us();
  u8  cpu_fault = fuzz_run_target(afl, &afl->fsrv, afl->fsrv.exec_tmout);
  u64 verify_us = get_cur_time_us() - t0;

  coqui_ctx_t *ctx = afl->coqui;
  if (ctx->verify_baseline_n < 10) {
    ctx->verify_baseline_sum += verify_us;
    ctx->verify_baseline_n++;
    if (ctx->verify_baseline_n == 10) {
      ctx->verify_baseline_us = ctx->verify_baseline_sum / 10;
      OKF("coqui speed gate: baseline verify_us=%llu (10×: %llu)",
          (unsigned long long)ctx->verify_baseline_us,
          (unsigned long long)(ctx->verify_baseline_us * 10));
    }
  } else if (ctx->verify_baseline_us > 0 &&
             verify_us > ctx->verify_baseline_us * 10) {
    ctx->slow_skipped++;
    return;   /* don't admit to corpus */
  }

  afl->queued_discovered += save_if_interesting(afl, input, len, cpu_fault);
}

/* Forward decl — defined below. Returns 1 if the CUDA context was reset
 * (caller's `ctx` / `b` snapshots are stale; re-fetch from afl->coqui
 * and abandon the current ping-pong operation). Returns 0 on normal
 * completion. */
static int coqui_await_and_process(afl_state_t *afl, coqui_batch_t *b);
static void coqui_force_reset(afl_state_t *afl, const char *cubin_path);

/* Adaptive batch-timeout (B1) helpers.
 *
 * Push one healthy-batch latency into the ring buffer, then — if we have
 * enough samples AND the user did not pin the timeout via AFL_COQUI_TIMEOUT_US
 * — recompute batch_timeout_us = clamp(MULT * P95, FLOOR, CEIL).
 *
 * Only healthy batches (CUDA_SUCCESS before cull deadline) feed this ring.
 * Culled and force-reset batches are excluded so the estimate tracks the
 * true steady-state latency, not the pathological tail. */
static int coqui_u64_cmp(const void *a, const void *b) {
  unsigned long long av = *(const unsigned long long *)a;
  unsigned long long bv = *(const unsigned long long *)b;
  return (av > bv) - (av < bv);
}

static void coqui_push_healthy_latency(coqui_ctx_t *ctx,
                                        unsigned long long latency_us) {
  /* Defensive: huge outliers (e.g., first-batch init cost) could skew P95
   * upward and defeat the whole point. Cap at 10s so a one-off slow start
   * doesn't pin us to the 3s ceiling forever. */
  if (latency_us > 10000000ULL) latency_us = 10000000ULL;

  ctx->batch_latency_ring[ctx->batch_latency_head] = latency_us;
  ctx->batch_latency_head =
      (ctx->batch_latency_head + 1) % COQUI_LAT_RING_SIZE;
  if (ctx->batch_latency_count < COQUI_LAT_RING_SIZE) {
    ctx->batch_latency_count++;
  }

  if (ctx->timeout_env_override) return;  /* user pin wins */
  if (ctx->batch_latency_count < COQUI_LAT_MIN_SAMPLES) return;

  unsigned long long tmp[COQUI_LAT_RING_SIZE];
  u32 n = ctx->batch_latency_count;
  memcpy(tmp, ctx->batch_latency_ring, n * sizeof(unsigned long long));
  qsort(tmp, n, sizeof(unsigned long long), coqui_u64_cmp);

  /* P95: index = floor((n-1) * 0.95). For n=32 -> index 29.
   * Standard nearest-rank on a ring means small-n estimates skew slightly
   * low, but that is safer than skewing high (high = slower cull). */
  u32 idx = (u32)(((unsigned long long)(n - 1) * COQUI_LAT_P95_PCT) / 100ULL);
  unsigned long long p95 = tmp[idx];

  unsigned long long target = p95 * COQUI_LAT_MULT;
  if (target < COQUI_LAT_FLOOR_US) target = COQUI_LAT_FLOOR_US;
  if (target > COQUI_LAT_CEIL_US)  target = COQUI_LAT_CEIL_US;

  ctx->batch_timeout_us = target;
}

static void coqui_launch_batch(afl_state_t *afl, coqui_batch_t *b) {
  coqui_ctx_t *ctx = afl->coqui;
  CUstream s = (CUstream)b->stream;

  /* Submit-phase timing: wrap the HtoD/memset/launch/DtoH/event-record API
   * calls so the [coqui-rate] print can split driver-submission cost from
   * actual GPU wait (measured separately in coqui_await_and_process). */
  struct timeval _tv_submit_t0;
  gettimeofday(&_tv_submit_t0, NULL);
  unsigned long long _submit_t0_us =
      ((unsigned long long)_tv_submit_t0.tv_sec * 1000000ULL) +
      _tv_submit_t0.tv_usec;

  /* H->D — only copy the live prefix of input_bytes. Kernel reads only
   * input_bytes[offsets[tid]..+lens[tid]] and empty-slot threads
   * (lens[tid]==0) short-circuit at FuzzEntry.cpp L112-113 before any
   * read. Stale tail from previous batches is inert. Round up to
   * 8-byte alignment to match the slot cursor maintained by
   * coqui_submit_input (off = (bytes_used + 7) & ~7u). */
  size_t input_bytes_len = (b->bytes_used + 7u) & ~(size_t)7u;
  if (input_bytes_len > 0) {
    CUCHECK(cuMemcpyHtoDAsync((CUdeviceptr)b->d_input_bytes,
                               b->h_input_bytes, input_bytes_len, s));
  }
  CUCHECK(cuMemcpyHtoDAsync((CUdeviceptr)b->d_offsets,
                             b->h_offsets, ctx->batch_size * 4, s));
  CUCHECK(cuMemcpyHtoDAsync((CUdeviceptr)b->d_input_lens,
                             b->h_input_lens, ctx->batch_size * 4, s));
  CUCHECK(cuMemsetD32Async((CUdeviceptr)b->d_novelty, 0,
                            ctx->batch_size / 32, s));
  CUCHECK(cuMemsetD8Async((CUdeviceptr)b->d_status, 0,
                           ctx->batch_size * sizeof(coqui_status_t), s));

  /* Launch */
  void *args[] = {
    (void *)&b->d_input_bytes,
    (void *)&b->d_offsets,
    (void *)&b->d_input_lens,
    (void *)&b->d_novelty,
    (void *)&b->d_status,
  };
  unsigned grid = ctx->batch_size / 128;
  CUCHECK(cuLaunchKernel((CUfunction)ctx->cu_kernel,
                          grid, 1, 1,    /* grid */
                          128, 1, 1,     /* block */
                          0, s, args, NULL));

  /* D->H */
  CUCHECK(cuMemcpyDtoHAsync(b->h_novelty, (CUdeviceptr)b->d_novelty,
                             ctx->batch_size / 8, s));
  CUCHECK(cuMemcpyDtoHAsync(b->h_status, (CUdeviceptr)b->d_status,
                             ctx->batch_size * sizeof(coqui_status_t), s));

  CUCHECK(cuEventRecord((CUevent)b->completion_event, s));
  ctx->launch_count++;

  /* Close out submit-phase timer now that all driver calls for this batch
   * have returned. */
  {
    struct timeval _tv_submit_t1;
    gettimeofday(&_tv_submit_t1, NULL);
    unsigned long long _submit_t1_us =
        ((unsigned long long)_tv_submit_t1.tv_sec * 1000000ULL) +
        _tv_submit_t1.tv_usec;
    ctx->t_submit_us += (_submit_t1_us - _submit_t0_us);
  }

  /* Adaptive batch-timeout (B1): stamp launch time ON THE BATCH so
   * coqui_await_and_process can compute healthy-batch latency. Per-batch
   * (not per-context) because ping-pong means launch(X) → flip → wait(Y),
   * where a context-level field would get clobbered between Y's launch
   * and Y's wait. */
  struct timeval rate_tv;
  gettimeofday(&rate_tv, NULL);
  b->launch_start_us =
      ((unsigned long long)rate_tv.tv_sec * 1000000ULL) + rate_tv.tv_usec;

  /* Throughput diagnostic: one line per batch launch. Supplements AFL's own
   * execs_per_sec (accurate now that coqui_submit_input bumps total_execs)
   * with per-batch visibility — inputs/batch is the key signal that batches
   * are full (cross-fuzz_one accumulation working) vs partial. */
  if (!ctx->rate_log_init) {
    ctx->rate_log_t0 = rate_tv;
    ctx->rate_log_init = 1;
    ctx->rate_log_last_launches = ctx->launch_count;
    ctx->rate_log_last_submits  = ctx->total_submits;
  } else {
    u64 elapsed_us =
      ((u64)(rate_tv.tv_sec - ctx->rate_log_t0.tv_sec) * 1000000ULL) +
      (rate_tv.tv_usec - ctx->rate_log_t0.tv_usec);
    u64 dl = ctx->launch_count - ctx->rate_log_last_launches;
    u64 ds = ctx->total_submits - ctx->rate_log_last_submits;
    /* Throttle: only print if either ≥ 1s elapsed or ≥ 4 batches accumulated,
     * whichever comes first, so we see signal both when batches flow fast and
     * when they're slow due to pathological inputs. */
    if (elapsed_us >= 1000000ULL || dl >= 4) {
      if (elapsed_us > 0) {
        double avg_dedup_per_batch = dl > 0
            ? (double)ctx->crash_dedup_hits / (double)ctx->launch_count
            : 0.0;
        double avg_verify_per_batch = dl > 0
            ? (double)ctx->crash_verify_calls / (double)ctx->launch_count
            : 0.0;
        /* Per-batch phase averages over this window. submit is driver-API
         * time (all async), await is the GPU wait in the poll loop, verify
         * is CPU-side novelty+crash loops after kernel completion. Summing
         * submit+await+verify approximates total host wall time per batch;
         * the gap to 1/batch-rate is AFL's outer stage overhead. */
        double avg_submit_us_batch = dl > 0
            ? (double)ctx->t_submit_us / (double)dl : 0.0;
        double avg_await_us_batch  = dl > 0
            ? (double)ctx->t_await_us  / (double)dl : 0.0;
        double avg_verify_us_batch = dl > 0
            ? (double)ctx->t_verify_us / (double)dl : 0.0;
        fprintf(stderr,
                "[coqui-rate] %.1f batches/s, %llu submits/s "
                "(avg %.0f inputs/batch, %llu slow-skipped, "
                "crash_dedup_hits=%llu avg=%.1f/batch, "
                "crash_verify_calls=%llu avg=%.1f/batch, "
                "submit=%.0fus await=%.0fus verify=%.0fus /batch)\n",
                (double)dl * 1e6 / (double)elapsed_us,
                (unsigned long long)(ds * 1000000ULL / elapsed_us),
                dl > 0 ? (double)ds / (double)dl : 0.0,
                (unsigned long long)ctx->slow_skipped,
                (unsigned long long)ctx->crash_dedup_hits,
                avg_dedup_per_batch,
                (unsigned long long)ctx->crash_verify_calls,
                avg_verify_per_batch,
                avg_submit_us_batch,
                avg_await_us_batch,
                avg_verify_us_batch);
        fflush(stderr);
      }
      ctx->rate_log_last_launches = ctx->launch_count;
      ctx->rate_log_last_submits  = ctx->total_submits;
      ctx->rate_log_t0 = rate_tv;
      /* Reset per-window timing accumulators; next print shows the next
       * window's phase averages. */
      ctx->t_submit_us = 0;
      ctx->t_await_us  = 0;
      ctx->t_verify_us = 0;
    }
  }
}

static int coqui_await_and_process(afl_state_t *afl, coqui_batch_t *b) {
  coqui_ctx_t *ctx = afl->coqui;
  CUstream s = (CUstream)b->stream;

  /* Two-phase bounded wait with seed culling. Coqui's strategy (driver
   * 2248-2256): when a batch times out, the culprit is usually a single
   * pathological input; deprioritize the queue entry it came from so the
   * same mutations aren't retried. Then grant extra time for the kernel to
   * finish naturally (cuCtxSynchronize-style drain) bounded by a hard
   * ceiling that forces a FATAL instead of an open-ended block. */
  struct timeval tv;
  gettimeofday(&tv, NULL);
  unsigned long long start_us =
    ((unsigned long long)tv.tv_sec * 1000000 + tv.tv_usec);
  unsigned long long cull_deadline_us = start_us + ctx->batch_timeout_us;
  /* Hard ceiling: if the kernel hasn't finished 1s after culling, assume an
   * infinite loop and force-reset (kills the kernel). Reset costs ~1s, which
   * is cheaper than waiting longer for a pathological kernel to finish.
   *
   * iter17 tightening: was 5s. Profile evidence (P1 iter6) showed truly-
   * pathological cjson kernels hang 5–22+ seconds, well beyond any plausible
   * grace window; the 5s grace was almost never recovering legitimately-slow
   * batches and was costing ~4s per pathology cycle. 1s grace gives near-done
   * kernels a fair chance to complete while keeping recovery latency low. */
  unsigned long long hard_deadline_us = cull_deadline_us + 1000000ULL;

  int culled = 0;
  unsigned long long wait_end_us = 0;  /* set when cuStreamQuery reports success */
  while (1) {
    CUresult r = cuStreamQuery(s);
    if (r == CUDA_SUCCESS) {
      struct timeval _tv_wait_end;
      gettimeofday(&_tv_wait_end, NULL);
      wait_end_us =
          ((unsigned long long)_tv_wait_end.tv_sec * 1000000ULL) +
          _tv_wait_end.tv_usec;
      ctx->t_await_us += (wait_end_us - start_us);
      break;
    }
    if (r != CUDA_ERROR_NOT_READY) {
      CUCHECK(r);   /* genuine GPU fault */
    }
    gettimeofday(&tv, NULL);
    unsigned long long now =
      ((unsigned long long)tv.tv_sec * 1000000 + tv.tv_usec);

    if (!culled && now >= cull_deadline_us) {
      WARNF("coqui batch timeout after %llu us — disabling queue entry "
            "'%s' and extending wait by 1s",
            ctx->batch_timeout_us,
            afl->queue_cur ? (const char *)afl->queue_cur->fname : "(none)");
      if (afl->queue_cur) {
        /* Hard skip from scheduling. Also set exec_us so the information
         * survives in AFL's stats dump and the weight formula treats the
         * entry as expensive even if it's later re-enabled. disabled=1
         * and exec_us both persist in fastresume.bin (they live in the
         * queue_entry block serialized at shutdown), so restarts keep
         * the cull — no need to rediscover pathological entries every
         * session. */
        afl->queue_cur->disabled = 1;
        afl->queue_cur->exec_us = (u64)ctx->batch_timeout_us;
        afl->queue_cur->fuzz_level += 1000;
      }
      culled = 1;
      continue;
    }

    if (now >= hard_deadline_us) {
      /* Kernel still running after cull + grace period. Force-kill via
       * context destroy and rebuild: we need the kernel dead NOW so the
       * queue entry we culled can't keep blocking throughput. Any pending
       * work in this batch is lost; caller must re-fetch ctx pointers. */
      WARNF("coqui kernel stuck past %llu us — force-resetting CUDA context "
            "to kill runaway kernel. In-flight inputs lost.",
            (hard_deadline_us - start_us));
      char *cubin_path = ck_strdup(afl->coqui_cubin_path);
      coqui_force_reset(afl, cubin_path);
      ck_free(cubin_path);
      return 1;
    }

    usleep(1000);   /* 1ms poll — matches coqui driver */
  }
  (void)ctx;  /* silence unused-after-reset-path warnings */

  /* Adaptive batch-timeout (B1): record latency of this batch ONLY if it
   * completed healthily (CUDA_SUCCESS before the cull deadline fired).
   * Culled and force-reset batches are the pathology we are trying to
   * escape faster — including them in the stat would drive the timeout
   * estimate UP, which is the opposite of the goal. The `culled` flag
   * is the single source of truth here: we break out of the wait loop
   * on CUDA_SUCCESS, and this code runs only after that break. */
  if (!culled && b->launch_start_us > 0) {
    struct timeval done_tv;
    gettimeofday(&done_tv, NULL);
    unsigned long long done_us =
      ((unsigned long long)done_tv.tv_sec * 1000000ULL) + done_tv.tv_usec;
    if (done_us > b->launch_start_us) {
      coqui_push_healthy_latency(ctx, done_us - b->launch_start_us);
    }
  }

  /* Process flagged inputs (novelty bitmap bits set) */
  for (u32 word_i = 0; word_i < ctx->batch_size / 32; word_i++) {
    u32 bits = ((u32 *)b->h_novelty)[word_i];
    while (bits) {
      u32 bit_pos = __builtin_ctz(bits);
      bits &= bits - 1;
      u32 i = word_i * 32 + bit_pos;
      if (b->h_input_lens[i] == 0) continue;

      u8 *input = b->h_input_bytes + b->h_offsets[i];
      u32 len = b->h_input_lens[i];
      process_input_via_cpu_fsrv(afl, input, len);
    }
  }

  /* Also process crashes not flagged as novel.
   * A GPU crash may not set new coverage bits but still needs to be saved.
   *
   * Crash-signature dedup (iter5 P1): each crashed thread stores an FNV-1a
   * hash of its classified cov_map in status[i].crash_sig. Malformed inputs
   * that fail at the same parser site converge to the same signature, so
   * we only invoke the (expensive) CPU forkserver verify on the FIRST slot
   * with each unique signature. Subsequent duplicates increment
   * crash_dedup_hits and are skipped; the FIRST slot (lowest index) is
   * deterministic and independent of iteration order.
   *
   * Hash-set: 256 slots of open-addressing (power of 2 → bitmask probe).
   * Inserting 4085 keys into 256 slots has worst-case degraded probe
   * length but distinct-signatures count is expected to be O(10-50), so
   * in practice the set is sparsely populated. If the set fills (unlikely),
   * we treat any new probe sequence as "seen" to avoid runaway probing and
   * accept the false-positive (dropped verify). */
  coqui_status_t *hs = b->h_status;

  #define COQUI_CRASH_DEDUP_SLOTS 256u
  u32 dedup_slots[COQUI_CRASH_DEDUP_SLOTS];
  u8  dedup_used[COQUI_CRASH_DEDUP_SLOTS];
  memset(dedup_slots, 0, sizeof(dedup_slots));
  memset(dedup_used,  0, sizeof(dedup_used));

  for (u32 i = 0; i < ctx->batch_size; i++) {
    if (b->h_input_lens[i] == 0) continue;
    u8 nov = (b->h_novelty[i / 8] >> (i % 8)) & 1;
    if (nov) continue;   /* already processed above */

    u8 gpu_fault = translate_gpu_status(&hs[i]);
    if (gpu_fault != FSRV_RUN_CRASH && gpu_fault != FSRV_RUN_TMOUT) continue;

    /* Dedup lookup — open-addressing linear probe. sig==0 is a valid key
     * (treated the same as any other). A full table (no empty slot found
     * after batch_size probes) falls through to "seen" to cap work. */
    u32 sig = hs[i].crash_sig;
    u32 idx = sig & (COQUI_CRASH_DEDUP_SLOTS - 1u);
    int seen = 0;
    int inserted = 0;
    for (u32 probe = 0; probe < COQUI_CRASH_DEDUP_SLOTS; probe++) {
      u32 slot = (idx + probe) & (COQUI_CRASH_DEDUP_SLOTS - 1u);
      if (!dedup_used[slot]) {
        dedup_slots[slot] = sig;
        dedup_used[slot]  = 1;
        inserted = 1;
        break;
      }
      if (dedup_slots[slot] == sig) { seen = 1; break; }
    }
    if (!inserted && !seen) {
      /* Table full — treat as seen to stop runaway probing. */
      seen = 1;
    }

    if (seen) {
      ctx->crash_dedup_hits++;
      continue;
    }

    ctx->crash_verify_calls++;
    u8 *input = b->h_input_bytes + b->h_offsets[i];
    u32 len = b->h_input_lens[i];
    process_input_via_cpu_fsrv(afl, input, len);
  }
  #undef COQUI_CRASH_DEDUP_SLOTS

  /* Close out verify-phase timer: wait_end_us was captured at the CUDA_SUCCESS
   * break out of the poll loop; everything between there and here is the
   * post-kernel host work (novelty walk + crash-sig dedup + fsrv verifies). */
  if (wait_end_us > 0) {
    struct timeval _tv_verify_end;
    gettimeofday(&_tv_verify_end, NULL);
    unsigned long long _verify_end_us =
        ((unsigned long long)_tv_verify_end.tv_sec * 1000000ULL) +
        _tv_verify_end.tv_usec;
    if (_verify_end_us > wait_end_us) {
      ctx->t_verify_us += (_verify_end_us - wait_end_us);
    }
  }

  return 0;
}

u8 coqui_submit_input(afl_state_t *afl, u8 *buf, u32 len) {
  coqui_ctx_t *ctx = afl->coqui;
  coqui_batch_t *b = ctx->pending;

  ctx->total_submits++;
  /* Count each submission as one exec — the corresponding increment in
   * afl_fsrv_run_target is suppressed in coqui_mode to avoid double-counting
   * GPU-flagged inputs that also run through CPU verification. */
  afl->fsrv.total_execs++;

  if (len > ctx->byte_budget) {
    ctx->oversized_count++;
    return 0;
  }

  u32 off = (b->bytes_used + 7) & ~7u;
  if (off + len > ctx->byte_budget || b->n_inputs == ctx->batch_size) {
    /* Launch the full batch async */
    coqui_launch_batch(afl, b);

    /* Flip ping-pong */
    coqui_batch_t *tmp = ctx->pending;
    ctx->pending = ctx->executing;
    ctx->executing = tmp;

    b = ctx->pending;
    off = 0;

    /* If the new pending has in-flight work from a previous flip, drain it
       before reusing */
    if (b->n_inputs > 0) {
      if (coqui_await_and_process(afl, b) == 1) {
        /* Context was reset under us; ctx/b are stale. Drop this input —
         * the context is now functional but the old batch pointers are
         * freed. Caller (havoc loop) will retry on the next iteration. */
        return 0;
      }
      b->n_inputs = 0;
      b->bytes_used = 0;
    }
  }

  memcpy(b->h_input_bytes + off, buf, len);
  b->h_offsets[b->n_inputs] = off;
  b->h_input_lens[b->n_inputs] = len;
  b->n_inputs++;
  b->bytes_used = off + len;
  return 0;
}

void coqui_flush_batch(afl_state_t *afl) {
  coqui_ctx_t *ctx = afl->coqui;

  /* Drain the executing batch if it has in-flight work */
  if (ctx->executing->n_inputs > 0) {
    if (coqui_await_and_process(afl, ctx->executing) == 1) return;
    ctx->executing->n_inputs = 0;
    ctx->executing->bytes_used = 0;
  }

  /* Launch + drain pending if partial */
  if (ctx->pending->n_inputs > 0) {
    coqui_launch_batch(afl, ctx->pending);
    if (coqui_await_and_process(afl, ctx->pending) == 1) return;
    ctx->pending->n_inputs = 0;
    ctx->pending->bytes_used = 0;
  }
}

u8 coqui_calibrate_one(afl_state_t *afl, u8 *buf, u32 len) {
  (void)afl; (void)buf; (void)len;
  /* Deprecated under the coexistence model (coqui internals spec §8.10).
     Calibration now flows through AFL's standard fuzz_run_target on the
     CPU forkserver at afl->fsrv. Kept as a no-op for ABI compatibility;
     reserved for future GPU-side calibration optimization. */
  return FSRV_RUN_OK;
}

/* Force teardown + rebuild for the stuck-kernel path. Destroys the CUDA
 * context unconditionally (which kills the runaway kernel, all streams, and
 * all context-bound device memory), frees host pinned memory (process-scoped,
 * so safe post-destroy), then rebuilds via coqui_init. Does NOT call the
 * regular coqui_shutdown because that cuStreamSynchronizes the stuck streams
 * and would block forever. */
static void coqui_force_reset(afl_state_t *afl, const char *cubin_path) {
  if (!afl->coqui) return;
  coqui_ctx_t *ctx = afl->coqui;

  if (ctx->cu_ctx) cuCtxDestroy((CUcontext)ctx->cu_ctx);

  /* Host pinned buffers are independent of context lifecycle. */
  if (ctx->ping.h_input_bytes) cuMemFreeHost(ctx->ping.h_input_bytes);
  if (ctx->ping.h_offsets)     cuMemFreeHost(ctx->ping.h_offsets);
  if (ctx->ping.h_input_lens)  cuMemFreeHost(ctx->ping.h_input_lens);
  if (ctx->ping.h_novelty)     cuMemFreeHost(ctx->ping.h_novelty);
  if (ctx->ping.h_status)      cuMemFreeHost(ctx->ping.h_status);
  if (ctx->pong.h_input_bytes) cuMemFreeHost(ctx->pong.h_input_bytes);
  if (ctx->pong.h_offsets)     cuMemFreeHost(ctx->pong.h_offsets);
  if (ctx->pong.h_input_lens)  cuMemFreeHost(ctx->pong.h_input_lens);
  if (ctx->pong.h_novelty)     cuMemFreeHost(ctx->pong.h_novelty);
  if (ctx->pong.h_status)      cuMemFreeHost(ctx->pong.h_status);

  ck_free(ctx);
  afl->coqui = NULL;

  coqui_init(afl, cubin_path);
}

static void free_batch_half_cuda(coqui_batch_t *b) {
  if (b->h_input_bytes) cuMemFreeHost(b->h_input_bytes);
  if (b->h_offsets)     cuMemFreeHost(b->h_offsets);
  if (b->h_input_lens)  cuMemFreeHost(b->h_input_lens);
  if (b->h_novelty)     cuMemFreeHost(b->h_novelty);
  if (b->h_status)      cuMemFreeHost(b->h_status);

  if (b->d_input_bytes) cuMemFree((CUdeviceptr)b->d_input_bytes);
  if (b->d_offsets)     cuMemFree((CUdeviceptr)b->d_offsets);
  if (b->d_input_lens)  cuMemFree((CUdeviceptr)b->d_input_lens);
  if (b->d_novelty)     cuMemFree((CUdeviceptr)b->d_novelty);
  if (b->d_status)      cuMemFree((CUdeviceptr)b->d_status);

  if (b->completion_event) cuEventDestroy((CUevent)b->completion_event);

  memset(b, 0, sizeof(*b));
}

void coqui_shutdown(afl_state_t *afl) {
  if (!afl->coqui) return;
  coqui_ctx_t *ctx = afl->coqui;

  /* Drain streams */
  if (ctx->stream_a) cuStreamSynchronize((CUstream)ctx->stream_a);
  if (ctx->stream_b) cuStreamSynchronize((CUstream)ctx->stream_b);

  /* Free ping-pong */
  free_batch_half_cuda(&ctx->ping);
  free_batch_half_cuda(&ctx->pong);

  /* Free persistent device buffers */
  if (ctx->d_global_statics_pool)
    cuMemFree((CUdeviceptr)ctx->d_global_statics_pool);
  if (ctx->d_slab_pool)
    cuMemFree((CUdeviceptr)ctx->d_slab_pool);

  /* Destroy streams, unload module, destroy context */
  if (ctx->stream_a) cuStreamDestroy((CUstream)ctx->stream_a);
  if (ctx->stream_b) cuStreamDestroy((CUstream)ctx->stream_b);
  if (ctx->cu_module) cuModuleUnload((CUmodule)ctx->cu_module);
  if (ctx->cu_ctx) cuCtxDestroy((CUcontext)ctx->cu_ctx);

  ck_free(ctx);
  afl->coqui = NULL;
}
