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
  unsigned int stack_size = getenv_u32("AFL_COQUI_STACK_SIZE", 16384);
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
  ctx->launch_count     = 0;
  ctx->oversized_count  = 0;

  afl->coqui = ctx;

  OKF("coqui initialized: sm_%d%d, batch=%u, stack=%u KB, heap=%u KB, total=%u KB",
      major, minor, ctx->batch_size, stack_size/1024, heap/1024, total_budget/1024);

}

/* Translate GPU status fields to an AFL fault code. */
static u8 translate_gpu_status(coqui_status_t *s) {
  if (s->asan_error) return FSRV_RUN_CRASH;
  if (s->ubsan_fatal) return FSRV_RUN_CRASH;
  if (s->signal) return FSRV_RUN_CRASH;
  if (s->timeout_flag) return FSRV_RUN_TMOUT;
  return FSRV_RUN_OK;
}

/* Run a GPU-flagged or GPU-crashed input through the CPU forkserver for
   real trace_bits, then call save_if_interesting. */
static void process_input_via_cpu_fsrv(afl_state_t *afl,
                                        u8 *input, u32 len) {
  u32 new_size = write_to_testcase(afl, (void **)&input, len, 0);
  if (new_size == 0) return;

  u8 cpu_fault = fuzz_run_target(afl, &afl->fsrv, afl->fsrv.exec_tmout);
  afl->queued_discovered += save_if_interesting(afl, input, len, cpu_fault);
}

/* Forward decl — defined below */
static void coqui_await_and_process(afl_state_t *afl, coqui_batch_t *b);

static void coqui_launch_batch(afl_state_t *afl, coqui_batch_t *b) {
  coqui_ctx_t *ctx = afl->coqui;
  CUstream s = (CUstream)b->stream;

  /* H->D */
  CUCHECK(cuMemcpyHtoDAsync((CUdeviceptr)b->d_input_bytes,
                             b->h_input_bytes, ctx->byte_budget, s));
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
}

static void coqui_await_and_process(afl_state_t *afl, coqui_batch_t *b) {
  coqui_ctx_t *ctx = afl->coqui;
  CUstream s = (CUstream)b->stream;

  /* Wait with timeout */
  struct timeval tv;
  gettimeofday(&tv, NULL);
  unsigned long long deadline_us =
    ((unsigned long long)tv.tv_sec * 1000000 + tv.tv_usec)
    + ctx->batch_timeout_us;

  while (1) {
    CUresult r = cuStreamQuery(s);
    if (r == CUDA_SUCCESS) break;
    if (r != CUDA_ERROR_NOT_READY) {
      CUCHECK(r);   /* FATAL for unexpected errors */
    }
    gettimeofday(&tv, NULL);
    unsigned long long now =
      ((unsigned long long)tv.tv_sec * 1000000 + tv.tv_usec);
    if (now >= deadline_us) {
      WARNF("coqui batch timeout after %llu us; salvaging completed threads",
            ctx->batch_timeout_us);
      cuCtxSynchronize();  /* best-effort: drain what we can */
      break;
    }
    usleep(100);
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
     A GPU crash may not set new coverage bits but still needs to be saved. */
  for (u32 i = 0; i < ctx->batch_size; i++) {
    if (b->h_input_lens[i] == 0) continue;
    u8 nov = (b->h_novelty[i / 8] >> (i % 8)) & 1;
    if (nov) continue;   /* already processed above */

    u8 gpu_fault = translate_gpu_status(&b->h_status[i]);
    if (gpu_fault == FSRV_RUN_CRASH || gpu_fault == FSRV_RUN_TMOUT) {
      u8 *input = b->h_input_bytes + b->h_offsets[i];
      u32 len = b->h_input_lens[i];
      process_input_via_cpu_fsrv(afl, input, len);
    }
  }
}

u8 coqui_submit_input(afl_state_t *afl, u8 *buf, u32 len) {
  coqui_ctx_t *ctx = afl->coqui;
  coqui_batch_t *b = ctx->pending;

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
      coqui_await_and_process(afl, b);
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
    coqui_await_and_process(afl, ctx->executing);
    ctx->executing->n_inputs = 0;
    ctx->executing->bytes_used = 0;
  }

  /* Launch + drain pending if partial */
  if (ctx->pending->n_inputs > 0) {
    coqui_launch_batch(afl, ctx->pending);
    coqui_await_and_process(afl, ctx->pending);
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
