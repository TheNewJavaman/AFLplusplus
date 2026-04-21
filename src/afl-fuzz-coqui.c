/*
 * afl-fuzz-coqui.c --- cuAFL coqui_mode real CUDA implementation.
 *
 * Real CUDA driver API backend for coqui_mode per spec §8.4.
 *
 * Threading (iter25): SPSC producer/consumer split. AFL's main thread is
 * the producer — it runs havoc and packs mutated inputs into a ring of
 * coqui_batch_t slots. A dedicated consumer thread (spawned by coqui_init)
 * pops ready slots, launches the GPU kernel, waits for completion, then
 * runs post-processing (novelty walk, crash dedup, CPU forkserver verify,
 * save_if_interesting). Producer and consumer synchronize on a ring-wide
 * pthread mutex + two condvars. A separate coarse mutex (afl_state_mutex)
 * serializes the consumer's mutations of AFL global queue state against
 * the producer's scheduler-side reads at the top of each fuzz_one.
 *
 * Spec: docs/superpowers/specs/2026-04-18-cuafl-gpu-backend-design.md
 */

#include "afl-fuzz.h"
#include "afl-fuzz-coqui.h"
#include "forkserver.h"

#include <stdlib.h>
#include <string.h>
#include <errno.h>

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

/* Forward declarations */
static void alloc_batch_slot_cuda(coqui_batch_t *b, coqui_ctx_t *ctx, CUstream stream);
static void free_batch_slot_cuda(coqui_batch_t *b);
static void coqui_launch_batch(afl_state_t *afl, coqui_batch_t *b);
static int  coqui_await_and_process(afl_state_t *afl, coqui_batch_t *b);
static void coqui_force_reset(afl_state_t *afl, const char *cubin_path);
static void *coqui_consumer_thread(void *arg);
static void coqui_start_consumer(afl_state_t *afl);
static void coqui_init_internal(afl_state_t *afl, const char *cubin_path, u8 spawn_consumer);

/* ------------------------------------------------------------------------
 * Internal helpers
 * ------------------------------------------------------------------------*/

static void alloc_batch_slot_cuda(coqui_batch_t *b, coqui_ctx_t *ctx, CUstream stream) {

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
  coqui_init_internal(afl, cubin_path, 1);
}

static void coqui_init_internal(afl_state_t *afl, const char *cubin_path,
                                 u8 spawn_consumer) {

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
   *    (See iter-history in prior commits for the binary-search story.) */
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

  unsigned int stack_size = getenv_u32("AFL_COQUI_STACK_SIZE", 32768);
  unsigned int cov = 65536;
  if (stack_size + cov >= total_budget) {
    FATAL("--stack-size %u + 64KB coverage >= total_budget %u",
          stack_size, total_budget);
  }
  unsigned int remaining = total_budget - cov - stack_size;
  unsigned int heap = (remaining * 8) / 9;
  ctx->real_stack_size = stack_size;

  int static_usage;
  CUCHECK(cuFuncGetAttribute(&static_usage,
    CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, kernel));
  if ((unsigned int)static_usage > total_budget - 1024) {
    FATAL("kernel static stack %d B exceeds budget %u B — bump --stack-size",
          static_usage, total_budget);
  }

  /* 6. Batch sizing */
  ctx->batch_size = 8192;
  ctx->max_input_size = afl->max_length ? afl->max_length : 4096;
  unsigned long long budget64 = ((unsigned long long)ctx->batch_size
                                  * ctx->max_input_size) / 4;
  unsigned long long floor64 = (unsigned long long)ctx->max_input_size * 256;
  if (budget64 < floor64) budget64 = floor64;
  if (budget64 > 0x40000000ULL) budget64 = 0x40000000ULL;
  ctx->byte_budget = (unsigned int)budget64;
  ctx->map_size = 65536;

  /* 7. Create streams (one per ring slot, so async launches don't serialize
   *    on each other — a slot's stream persists across reuse so enqueue
   *    ordering is stable). */
  /* N.b. we still keep stream_a/stream_b fields for backward compat with
   * the shutdown path but now the slots own their own streams. */
  CUstream sa, sb;
  CUCHECK(cuStreamCreate(&sa, CU_STREAM_NON_BLOCKING));
  CUCHECK(cuStreamCreate(&sb, CU_STREAM_NON_BLOCKING));
  ctx->stream_a = (void *)sa;
  ctx->stream_b = (void *)sb;

  /* 8. Allocate all ring slots. We rotate stream assignments across slots
   *    to get some parallelism between back-to-back kernel launches. */
  for (u32 i = 0; i < COQUI_RING_SIZE; i++) {
    /* Even slots -> stream_a, odd -> stream_b. With COQUI_RING_SIZE=4 this
     * gives two slots per stream; sequential launches on the same stream
     * serialize naturally which is what we want for consumer-side await
     * ordering. */
    CUstream s = (i & 1) ? sb : sa;
    alloc_batch_slot_cuda(&ctx->slots[i], ctx, s);
  }

  /* Initialize ring state. Slot 0 is the producer's starting slot. */
  ctx->write_idx = 0;
  ctx->read_idx  = 0;
  ctx->processed_idx = 0;
  ctx->producer_current = 0;
  /* Slot-index queue is populated lazily on push. Default zeros are fine. */
  memset(ctx->slot_queue, 0, sizeof(ctx->slot_queue));

  /* Pthread state. */
  pthread_mutex_init(&ctx->ring_mutex, NULL);
  /* afl_state_mutex is recursive: the consumer holds it across
   * process_input_via_cpu_fsrv (which calls fuzz_run_target), and
   * fuzz_run_target in GPU mode ALSO takes it from its own wrapper —
   * that re-entry is fine with a recursive mutex. */
  {
    pthread_mutexattr_t attr;
    pthread_mutexattr_init(&attr);
    pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE);
    pthread_mutex_init(&ctx->afl_state_mutex, &attr);
    pthread_mutexattr_destroy(&attr);
  }
  pthread_cond_init(&ctx->ring_not_empty, NULL);
  pthread_cond_init(&ctx->ring_not_full,  NULL);
  pthread_cond_init(&ctx->ring_drained,   NULL);
  ctx->consumer_started        = 0;
  ctx->consumer_should_exit    = 0;
  ctx->force_reset_in_progress = 0;
  ctx->producer_yielded        = 0;

  /* 9. Statics pool — allocate AND bind to the cubin's pool-base symbol. */
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
  memset(ctx->batch_latency_ring, 0, sizeof(ctx->batch_latency_ring));
  ctx->batch_latency_count  = 0;
  ctx->batch_latency_head   = 0;
  ctx->t_submit_us = 0;
  ctx->t_await_us  = 0;
  ctx->t_verify_us = 0;

  afl->coqui = ctx;

  /* Spawn consumer thread. Must happen AFTER afl->coqui is assigned,
   * since the consumer will dereference it. On force-reset, the caller
   * is ALREADY the consumer thread rebuilding state — in that path we
   * skip the spawn and let the caller continue running its own loop. */
  if (spawn_consumer) {
    coqui_start_consumer(afl);
  } else {
    /* force-reset path: current thread is the consumer. Adopt the new
     * CUDA context onto this thread. (coqui_init leaves the context
     * current on the creating thread; for the spawn path we pop it so
     * the new consumer thread can adopt it. For the in-place path we
     * keep it on the current thread.) */
    /* no-op: cuCtxCreate_v4 above already pushed the new context onto
     * the current thread's stack. */
  }

  OKF("coqui initialized: sm_%d%d, batch=%u, stack=%u KB, heap=%u KB, total=%u KB, ring=%d",
      major, minor, ctx->batch_size, stack_size/1024, heap/1024, total_budget/1024,
      COQUI_RING_SIZE);

}

/* Translate GPU status fields to an AFL fault code. */
static u8 translate_gpu_status(coqui_status_t *s) {
  if (s->asan_error) return FSRV_RUN_CRASH;
  if (s->ubsan_fatal) return FSRV_RUN_CRASH;
  if (s->signal) return FSRV_RUN_CRASH;
  return FSRV_RUN_OK;
}

/* Run a GPU-flagged or GPU-crashed input through the CPU forkserver for
   real trace_bits, then call save_if_interesting.
   CPU-side speed gate inherited. Locks afl_state_mutex around the full
   write_to_testcase + fuzz_run_target + save_if_interesting sequence so
   AFL state (trace_bits, queue_buf, virgin_bits) is seen consistently by
   this call and no other thread mutates it mid-sequence. */
static void process_input_via_cpu_fsrv(afl_state_t *afl,
                                        u8 *input, u32 len) {
  coqui_ctx_t *ctx = afl->coqui;
  pthread_mutex_lock(&ctx->afl_state_mutex);

  u32 new_size = write_to_testcase(afl, (void **)&input, len, 0);
  if (new_size == 0) {
    pthread_mutex_unlock(&ctx->afl_state_mutex);
    return;
  }

  u64 t0 = get_cur_time_us();
  u8  cpu_fault = fuzz_run_target(afl, &afl->fsrv, afl->fsrv.exec_tmout);
  u64 verify_us = get_cur_time_us() - t0;

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
    pthread_mutex_unlock(&ctx->afl_state_mutex);
    return;
  }

  afl->queued_discovered += save_if_interesting(afl, input, len, cpu_fault);
  pthread_mutex_unlock(&ctx->afl_state_mutex);
}

/* Adaptive batch-timeout (B1) helpers — unchanged from pre-threading. */
static int coqui_u64_cmp(const void *a, const void *b) {
  unsigned long long av = *(const unsigned long long *)a;
  unsigned long long bv = *(const unsigned long long *)b;
  return (av > bv) - (av < bv);
}

static void coqui_push_healthy_latency(coqui_ctx_t *ctx,
                                        unsigned long long latency_us) {
  if (latency_us > 10000000ULL) latency_us = 10000000ULL;

  ctx->batch_latency_ring[ctx->batch_latency_head] = latency_us;
  ctx->batch_latency_head =
      (ctx->batch_latency_head + 1) % COQUI_LAT_RING_SIZE;
  if (ctx->batch_latency_count < COQUI_LAT_RING_SIZE) {
    ctx->batch_latency_count++;
  }

  if (ctx->timeout_env_override) return;
  if (ctx->batch_latency_count < COQUI_LAT_MIN_SAMPLES) return;

  unsigned long long tmp[COQUI_LAT_RING_SIZE];
  u32 n = ctx->batch_latency_count;
  memcpy(tmp, ctx->batch_latency_ring, n * sizeof(unsigned long long));
  qsort(tmp, n, sizeof(unsigned long long), coqui_u64_cmp);

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

  struct timeval _tv_submit_t0;
  gettimeofday(&_tv_submit_t0, NULL);
  unsigned long long _submit_t0_us =
      ((unsigned long long)_tv_submit_t0.tv_sec * 1000000ULL) +
      _tv_submit_t0.tv_usec;

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

  void *args[] = {
    (void *)&b->d_input_bytes,
    (void *)&b->d_offsets,
    (void *)&b->d_input_lens,
    (void *)&b->d_novelty,
    (void *)&b->d_status,
  };
  unsigned grid = ctx->batch_size / 128;
  CUCHECK(cuLaunchKernel((CUfunction)ctx->cu_kernel,
                          grid, 1, 1,
                          128, 1, 1,
                          0, s, args, NULL));

  CUCHECK(cuMemcpyDtoHAsync(b->h_novelty, (CUdeviceptr)b->d_novelty,
                             ctx->batch_size / 8, s));
  CUCHECK(cuMemcpyDtoHAsync(b->h_status, (CUdeviceptr)b->d_status,
                             ctx->batch_size * sizeof(coqui_status_t), s));

  CUCHECK(cuEventRecord((CUevent)b->completion_event, s));
  ctx->launch_count++;

  {
    struct timeval _tv_submit_t1;
    gettimeofday(&_tv_submit_t1, NULL);
    unsigned long long _submit_t1_us =
        ((unsigned long long)_tv_submit_t1.tv_sec * 1000000ULL) +
        _tv_submit_t1.tv_usec;
    ctx->t_submit_us += (_submit_t1_us - _submit_t0_us);
  }

  struct timeval rate_tv;
  gettimeofday(&rate_tv, NULL);
  b->launch_start_us =
      ((unsigned long long)rate_tv.tv_sec * 1000000ULL) + rate_tv.tv_usec;

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
    if (elapsed_us >= 1000000ULL || dl >= 4) {
      if (elapsed_us > 0) {
        double avg_dedup_per_batch = dl > 0
            ? (double)ctx->crash_dedup_hits / (double)ctx->launch_count
            : 0.0;
        double avg_verify_per_batch = dl > 0
            ? (double)ctx->crash_verify_calls / (double)ctx->launch_count
            : 0.0;
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
      ctx->t_submit_us = 0;
      ctx->t_await_us  = 0;
      ctx->t_verify_us = 0;
    }
  }
}

/* Await the consumer-owned batch's completion, then run post-processing.
 * Acquires afl_state_mutex around the verify work. Returns 1 if force-reset
 * fired (caller must re-fetch ctx pointers). */
static int coqui_await_and_process(afl_state_t *afl, coqui_batch_t *b) {
  coqui_ctx_t *ctx = afl->coqui;
  CUstream s = (CUstream)b->stream;

  struct timeval tv;
  gettimeofday(&tv, NULL);
  unsigned long long start_us =
    ((unsigned long long)tv.tv_sec * 1000000 + tv.tv_usec);
  unsigned long long cull_deadline_us = start_us + ctx->batch_timeout_us;
  unsigned long long hard_deadline_us = cull_deadline_us + 1000000ULL;

  int culled = 0;
  unsigned long long wait_end_us = 0;
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
      CUCHECK(r);
    }
    gettimeofday(&tv, NULL);
    unsigned long long now =
      ((unsigned long long)tv.tv_sec * 1000000 + tv.tv_usec);

    if (!culled && now >= cull_deadline_us) {
      /* Reading afl->queue_cur from the consumer — snapshot under the state
       * mutex to avoid a torn read vs. the producer updating queue_cur. */
      const char *fname = "(none)";
      pthread_mutex_lock(&ctx->afl_state_mutex);
      if (afl->queue_cur) fname = (const char *)afl->queue_cur->fname;
      WARNF("coqui batch timeout after %llu us — disabling queue entry "
            "'%s' and extending wait by 1s",
            ctx->batch_timeout_us, fname);
      if (afl->queue_cur) {
        afl->queue_cur->disabled = 1;
        afl->queue_cur->exec_us = (u64)ctx->batch_timeout_us;
        afl->queue_cur->fuzz_level += 1000;
      }
      pthread_mutex_unlock(&ctx->afl_state_mutex);
      culled = 1;
      continue;
    }

    if (now >= hard_deadline_us) {
      WARNF("coqui kernel stuck past %llu us — force-resetting CUDA context "
            "to kill runaway kernel. In-flight inputs lost.",
            (hard_deadline_us - start_us));
      char *cubin_path = ck_strdup(afl->coqui_cubin_path);
      coqui_force_reset(afl, cubin_path);
      ck_free(cubin_path);
      return 1;
    }

    usleep(1000);
  }
  (void)ctx;

  if (!culled && b->launch_start_us > 0) {
    struct timeval done_tv;
    gettimeofday(&done_tv, NULL);
    unsigned long long done_us =
      ((unsigned long long)done_tv.tv_sec * 1000000ULL) + done_tv.tv_usec;
    if (done_us > b->launch_start_us) {
      coqui_push_healthy_latency(ctx, done_us - b->launch_start_us);
    }
  }

  /* Post-kernel work. We don't hold afl_state_mutex across the whole verify
   * phase — instead, each process_input_via_cpu_fsrv call takes the lock
   * around its own write_to_testcase + fuzz_run_target + save_if_interesting
   * sequence. This avoids a producer-consumer priority inversion: if the
   * producer is mid-fuzz_one and hits a path that takes the state lock, it
   * doesn't have to wait for a full verify phase to complete. */

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

  coqui_status_t *hs = b->h_status;

  #define COQUI_CRASH_DEDUP_SLOTS 256u
  u32 dedup_slots[COQUI_CRASH_DEDUP_SLOTS];
  u8  dedup_used[COQUI_CRASH_DEDUP_SLOTS];
  memset(dedup_slots, 0, sizeof(dedup_slots));
  memset(dedup_used,  0, sizeof(dedup_used));

  for (u32 i = 0; i < ctx->batch_size; i++) {
    if (b->h_input_lens[i] == 0) continue;
    u8 nov = (b->h_novelty[i / 8] >> (i % 8)) & 1;
    if (nov) continue;

    u8 gpu_fault = translate_gpu_status(&hs[i]);
    if (gpu_fault != FSRV_RUN_CRASH && gpu_fault != FSRV_RUN_TMOUT) continue;

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

/* -----------------------------------------------------------------------
 * Consumer thread
 * -----------------------------------------------------------------------*/

static void *coqui_consumer_thread(void *arg) {
  afl_state_t *afl = (afl_state_t *)arg;
  coqui_ctx_t *ctx = afl->coqui;

  /* The consumer thread is the CUDA caller after coqui_init returns; bind
   * it to the context we built on the main thread. */
  CUCHECK(cuCtxSetCurrent((CUcontext)ctx->cu_ctx));

  /* Pipelined consumer: keep one batch "in flight" (launched but not yet
   * awaited) so the GPU is busy during the CPU-side verify of the previous
   * batch. This mirrors the original ping-pong scheme but with arbitrary
   * ring depth.
   *
   *   iter 0: pop A, launch A. inflight = A.
   *   iter 1: pop B, launch B (async, GPU now running B after A). await A,
   *           verify A, release slot A. inflight = B.
   *   iter 2: pop C, launch C. await B, verify B, release slot B. inflight = C.
   *   ...
   *
   * This keeps a 2-stage pipeline (launch N+1 while verifying N) which
   * ensures the GPU's idle window is only the gap between one kernel
   * finishing and the next being submitted — typically microseconds. */
  u32 inflight_slot_i = 0xFFFFFFFFu;   /* no inflight at boot */

  /* Two cursors: read_idx = slot most recently dequeued for launch,
   *              processed_idx = slot most recently awaited + verified + reset.
   * Producer reserves slots between [processed_idx, write_idx) as busy.
   * The pipeline: launch slot at read_idx, await+verify slot at processed_idx.
   * In steady state, read_idx is always 1 ahead of processed_idx (the
   * inflight). */
  for (;;) {
    u32 slot_i = 0;
    u8 have_next = 0;

    pthread_mutex_lock(&ctx->ring_mutex);

    if (inflight_slot_i != 0xFFFFFFFFu) {
      /* We have an inflight. Opportunistically launch the next one if a
       * slot is available in the ring (i.e., producer has pushed past
       * our read_idx). */
      if (ctx->write_idx != ctx->read_idx) {
        u32 slot_pos = ctx->read_idx % COQUI_RING_SIZE;
        slot_i = ctx->slot_queue[slot_pos];
        have_next = 1;
        ctx->read_idx++;      /* dequeue now; producer won't touch this slot */
      }
    } else {
      /* No inflight. Block until work arrives. */
      while (ctx->write_idx == ctx->read_idx && !ctx->consumer_should_exit) {
        pthread_cond_wait(&ctx->ring_not_empty, &ctx->ring_mutex);
      }
      if (ctx->consumer_should_exit && ctx->write_idx == ctx->read_idx) {
        pthread_mutex_unlock(&ctx->ring_mutex);
        break;
      }
      u32 slot_pos = ctx->read_idx % COQUI_RING_SIZE;
      slot_i = ctx->slot_queue[slot_pos];
      have_next = 1;
      ctx->read_idx++;
    }
    pthread_mutex_unlock(&ctx->ring_mutex);

    /* Launch the next batch NOW (async). */
    if (have_next) {
      coqui_launch_batch(afl, &ctx->slots[slot_i]);
    }

    /* Process the inflight slot (previous batch's await + verify). */
    if (inflight_slot_i != 0xFFFFFFFFu) {
      coqui_batch_t *inflight_b = &ctx->slots[inflight_slot_i];
      int reset_occurred = coqui_await_and_process(afl, inflight_b);
      if (reset_occurred) {
        /* ctx was torn down. */
        ctx = afl->coqui;
        inflight_slot_i = 0xFFFFFFFFu;
        continue;
      }

      inflight_b->n_inputs = 0;
      inflight_b->bytes_used = 0;
      inflight_b->launch_start_us = 0;

      /* Mark this slot as processed so the producer can refill it.
       * processed_idx tracks "fully processed" slots; producer uses it to
       * determine "busy" = [processed_idx, write_idx). Incrementing it
       * releases the slot for refill. */
      pthread_mutex_lock(&ctx->ring_mutex);
      ctx->processed_idx = (ctx->processed_idx + 1);
      pthread_cond_broadcast(&ctx->ring_not_full);
      if (ctx->write_idx == ctx->processed_idx) {
        pthread_cond_broadcast(&ctx->ring_drained);
      }
      pthread_mutex_unlock(&ctx->ring_mutex);

      inflight_slot_i = 0xFFFFFFFFu;
    }

    /* Slide: the one we just launched becomes the new inflight. */
    if (have_next) {
      inflight_slot_i = slot_i;
    }
  }

  /* Drain any still-inflight batch before exiting. */
  if (inflight_slot_i != 0xFFFFFFFFu) {
    coqui_batch_t *inflight_b = &ctx->slots[inflight_slot_i];
    int reset_occurred = coqui_await_and_process(afl, inflight_b);
    if (!reset_occurred) {
      inflight_b->n_inputs = 0;
      inflight_b->bytes_used = 0;
      pthread_mutex_lock(&ctx->ring_mutex);
      ctx->processed_idx++;
      pthread_cond_broadcast(&ctx->ring_drained);
      pthread_mutex_unlock(&ctx->ring_mutex);
    }
  }

  return NULL;
}

static void coqui_start_consumer(afl_state_t *afl) {
  coqui_ctx_t *ctx = afl->coqui;
  if (ctx->consumer_started) return;

  /* Pop the current context off the main thread so the consumer can adopt
   * it via cuCtxSetCurrent. (CUDA 4.0+ supports a context being current on
   * any single thread at a time; before pushing ourselves back, the main
   * thread shouldn't touch CUDA anymore — and in fact our producer path
   * does NO CUDA calls after coqui_init returns.) */
  CUcontext popped = NULL;
  cuCtxPopCurrent(&popped);
  (void)popped;

  int rc = pthread_create(&ctx->consumer_tid, NULL,
                          coqui_consumer_thread, afl);
  if (rc != 0) {
    FATAL("coqui: pthread_create failed: %s", strerror(rc));
  }
  ctx->consumer_started = 1;
}

/* -----------------------------------------------------------------------
 * Producer-side API
 * -----------------------------------------------------------------------*/

/* Push the producer_current slot onto the ring and acquire a fresh slot.
 * Must be called with ring_mutex NOT held. Blocks if ring is full. */
static void producer_push_and_acquire(coqui_ctx_t *ctx) {
  pthread_mutex_lock(&ctx->ring_mutex);

  /* Record the pushed slot in the FIFO queue at write_idx. */
  u32 wpos = ctx->write_idx % COQUI_RING_SIZE;
  ctx->slot_queue[wpos] = ctx->producer_current;
  ctx->write_idx++;
  pthread_cond_signal(&ctx->ring_not_empty);

  /* Acquire a fresh slot. "Busy" slots are ones between processed_idx
   * (not yet fully processed) and write_idx. We need a free slot — one
   * not in that range. Ring is full when (write_idx - processed_idx) ==
   * COQUI_RING_SIZE, meaning all slots are pushed/inflight/launched. */
  while (!ctx->consumer_should_exit) {
    if (ctx->force_reset_in_progress) {
      ctx->producer_yielded = 1;
      pthread_cond_broadcast(&ctx->ring_not_full);
      while (ctx->force_reset_in_progress && !ctx->consumer_should_exit) {
        pthread_cond_wait(&ctx->ring_not_full, &ctx->ring_mutex);
      }
      ctx->producer_yielded = 0;
      break;
    }
    u32 busy_count = ctx->write_idx - ctx->processed_idx;
    /* Reserve one slot for the producer's next in-progress. */
    if (busy_count < COQUI_RING_SIZE) break;
    pthread_cond_wait(&ctx->ring_not_full, &ctx->ring_mutex);
  }

  /* Find a free slot: not in slot_queue[processed_idx..write_idx) (those
   * are busy: either queued, being launched, or being processed). */
  u32 busy[COQUI_RING_SIZE] = {0};
  for (u32 k = ctx->processed_idx; k != ctx->write_idx; k++) {
    u32 sk = ctx->slot_queue[k % COQUI_RING_SIZE];
    if (sk < COQUI_RING_SIZE) busy[sk] = 1;
  }
  u32 new_slot = 0xFFFFFFFFu;
  for (u32 k = 0; k < COQUI_RING_SIZE; k++) {
    if (!busy[k]) { new_slot = k; break; }
  }
  if (new_slot == 0xFFFFFFFFu) {
    new_slot = 0;  /* shouldn't happen given the while-loop above */
  }
  ctx->producer_current = new_slot;

  pthread_mutex_unlock(&ctx->ring_mutex);

  /* Zero out the slot state so the caller fills cleanly. */
  coqui_batch_t *nb = &ctx->slots[new_slot];
  nb->n_inputs = 0;
  nb->bytes_used = 0;
  nb->launch_start_us = 0;
}

u8 coqui_submit_input(afl_state_t *afl, u8 *buf, u32 len) {
  coqui_ctx_t *ctx = afl->coqui;
  coqui_batch_t *b = &ctx->slots[ctx->producer_current];

  ctx->total_submits++;
  afl->fsrv.total_execs++;

  if (len > ctx->byte_budget) {
    ctx->oversized_count++;
    return 0;
  }

  u32 off = (b->bytes_used + 7) & ~7u;
  if (off + len > ctx->byte_budget || b->n_inputs == ctx->batch_size) {
    /* Current slot is full. Hand it off to the consumer and acquire a
     * fresh one. */
    producer_push_and_acquire(ctx);
    /* ctx may have been replaced on force-reset; re-fetch. */
    ctx = afl->coqui;
    b = &ctx->slots[ctx->producer_current];
    off = 0;
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
  coqui_batch_t *b = &ctx->slots[ctx->producer_current];

  /* If the current slot has data, push it. */
  if (b->n_inputs > 0) {
    producer_push_and_acquire(ctx);
    ctx = afl->coqui;
  }

  /* Wait for ring to fully drain: all pushed slots must be BOTH launched
   * (read_idx == write_idx) AND processed (processed_idx == write_idx). */
  pthread_mutex_lock(&ctx->ring_mutex);
  while (ctx->write_idx != ctx->processed_idx && !ctx->consumer_should_exit) {
    pthread_cond_wait(&ctx->ring_drained, &ctx->ring_mutex);
  }
  pthread_mutex_unlock(&ctx->ring_mutex);
}

u8 coqui_calibrate_one(afl_state_t *afl, u8 *buf, u32 len) {
  (void)afl; (void)buf; (void)len;
  return FSRV_RUN_OK;
}

void coqui_state_lock(afl_state_t *afl) {
  if (!afl->coqui) return;
  coqui_ctx_t *ctx = afl->coqui;
  pthread_mutex_lock(&ctx->afl_state_mutex);
}

void coqui_state_unlock(afl_state_t *afl) {
  if (!afl->coqui) return;
  coqui_ctx_t *ctx = afl->coqui;
  pthread_mutex_unlock(&ctx->afl_state_mutex);
}

/* Force teardown + rebuild for the stuck-kernel path. Called ONLY from the
 * consumer thread (inside coqui_await_and_process when the hard deadline
 * expires). We re-init in place without spawning a new consumer thread,
 * and the CURRENT thread (this consumer) continues running against the
 * fresh ctx.
 *
 * Producer synchronization: we set force_reset_in_progress under ring_mutex
 * on the OLD ctx so if the producer happens to be blocked on ring_not_full,
 * it wakes up and yields. For a producer mid-havoc (not blocked), we rely
 * on the fact that host-side pinned memory remains a valid virtual mapping
 * until we cuMemFreeHost it below; the producer may write into the old
 * slot's h_input_bytes but those writes get discarded. After we rebuild
 * ctx and set afl->coqui to the new ctx, the producer's next submit
 * re-fetches afl->coqui (we do this in coqui_submit_input). Any in-flight
 * inputs in the old slot are lost — already the documented semantic. */
static void coqui_force_reset(afl_state_t *afl, const char *cubin_path) {
  if (!afl->coqui) return;
  coqui_ctx_t *ctx = afl->coqui;

  /* Wake any producer blocked on the old ring so it can observe the
   * teardown and retry cleanly on the new ctx. */
  pthread_mutex_lock(&ctx->ring_mutex);
  ctx->force_reset_in_progress = 1;
  ctx->consumer_should_exit = 1;  /* old ctx's consumer (us) will not re-enter the loop */
  pthread_cond_broadcast(&ctx->ring_not_empty);
  pthread_cond_broadcast(&ctx->ring_not_full);
  pthread_cond_broadcast(&ctx->ring_drained);
  pthread_mutex_unlock(&ctx->ring_mutex);

  /* Tear down CUDA. */
  if (ctx->cu_ctx) cuCtxDestroy((CUcontext)ctx->cu_ctx);

  for (u32 i = 0; i < COQUI_RING_SIZE; i++) {
    coqui_batch_t *b = &ctx->slots[i];
    if (b->h_input_bytes) cuMemFreeHost(b->h_input_bytes);
    if (b->h_offsets)     cuMemFreeHost(b->h_offsets);
    if (b->h_input_lens)  cuMemFreeHost(b->h_input_lens);
    if (b->h_novelty)     cuMemFreeHost(b->h_novelty);
    if (b->h_status)      cuMemFreeHost(b->h_status);
  }

  pthread_mutex_destroy(&ctx->ring_mutex);
  pthread_mutex_destroy(&ctx->afl_state_mutex);
  pthread_cond_destroy(&ctx->ring_not_empty);
  pthread_cond_destroy(&ctx->ring_not_full);
  pthread_cond_destroy(&ctx->ring_drained);

  pthread_t old_consumer_tid = ctx->consumer_tid;
  u8 old_consumer_started    = ctx->consumer_started;
  (void)old_consumer_tid; (void)old_consumer_started;

  ck_free(ctx);
  afl->coqui = NULL;

  /* Rebuild in-place; do NOT spawn a new consumer. The CURRENT thread
   * continues as the consumer. The new ctx's consumer_tid/consumer_started
   * are set as if the current thread owns the role. */
  coqui_init_internal(afl, cubin_path, 0);

  /* Record that the current thread IS the consumer. */
  coqui_ctx_t *new_ctx = afl->coqui;
  new_ctx->consumer_tid     = pthread_self();
  new_ctx->consumer_started = 1;
}

static void free_batch_slot_cuda(coqui_batch_t *b) {
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

  /* Signal + join consumer thread. Flush any pending producer slot. */
  if (ctx->consumer_started) {
    /* Push any partial slot the producer was filling. */
    coqui_batch_t *b = &ctx->slots[ctx->producer_current];
    if (b->n_inputs > 0) {
      /* Ring mutex NOT held; push via the producer helper. */
      producer_push_and_acquire(ctx);
      ctx = afl->coqui;
    }

    pthread_mutex_lock(&ctx->ring_mutex);
    ctx->consumer_should_exit = 1;
    pthread_cond_broadcast(&ctx->ring_not_empty);
    pthread_mutex_unlock(&ctx->ring_mutex);

    pthread_join(ctx->consumer_tid, NULL);
    ctx->consumer_started = 0;
  }

  /* Consumer thread may have dropped the CUDA context off its own stack
   * when exiting. Re-adopt on the main (shutdown) thread. */
  if (ctx->cu_ctx) cuCtxSetCurrent((CUcontext)ctx->cu_ctx);

  /* Drain streams */
  if (ctx->stream_a) cuStreamSynchronize((CUstream)ctx->stream_a);
  if (ctx->stream_b) cuStreamSynchronize((CUstream)ctx->stream_b);

  /* Free ring slots */
  for (u32 i = 0; i < COQUI_RING_SIZE; i++) {
    free_batch_slot_cuda(&ctx->slots[i]);
  }

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

  pthread_mutex_destroy(&ctx->ring_mutex);
  pthread_mutex_destroy(&ctx->afl_state_mutex);
  pthread_cond_destroy(&ctx->ring_not_empty);
  pthread_cond_destroy(&ctx->ring_not_full);
  pthread_cond_destroy(&ctx->ring_drained);

  ck_free(ctx);
  afl->coqui = NULL;
}
