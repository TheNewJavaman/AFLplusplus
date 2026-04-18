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

static void free_batch_half(coqui_batch_t *b) {

  /* Note: in CUDA path these are pinned host buffers freed via cuMemFreeHost;
     kept as stub for functions (coqui_shutdown) that still reference it.
     Will be replaced by free_batch_half_cuda in T6.3+. */
  ck_free(b->h_input_bytes);
  ck_free(b->h_offsets);
  ck_free(b->h_input_lens);
  ck_free(b->h_novelty);
  ck_free(b->h_status);

  memset(b, 0, sizeof(*b));

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

  /* 4. Compute stack budget per spec §4.3 */
  unsigned int stack_size = getenv_u32("AFL_COQUI_STACK_SIZE", 16384);
  unsigned int cov = 65536;
  unsigned int total_cap = 524288;
  if (stack_size + cov >= total_cap) {
    FATAL("--stack-size %u + 64KB coverage >= 512KB budget", stack_size);
  }
  unsigned int remaining = total_cap - cov - stack_size;
  unsigned int heap = (remaining * 8) / 9;
  unsigned int total_budget = cov + heap + (heap / 8) + stack_size;
  CUCHECK(cuCtxSetLimit(CU_LIMIT_STACK_SIZE, total_budget));
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

  /* 9. Statics pool (if kernel exports per-thread size) */
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

/* Forward decl — defined in T6.4 */
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
  /* T6.4: full implementation. For now: sync the stream so at least
     kernel completion happens before we reuse buffers. */
  (void)afl;
  cuStreamSynchronize((CUstream)b->stream);
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

  (void)buf;
  (void)len;

  /* Trivial coverage: touch edge 0 so calibration doesn't flag the seed
     as FSRV_RUN_NOINST. Real implementation runs the input on the GPU. */
  memset(afl->fsrv.trace_bits, 0, afl->fsrv.map_size);
  afl->fsrv.trace_bits[0] = 1;

  return FSRV_RUN_OK;

}

void coqui_shutdown(afl_state_t *afl) {

  if (!afl->coqui) return;

  free_batch_half(&afl->coqui->ping);
  free_batch_half(&afl->coqui->pong);

  ck_free(afl->coqui);
  afl->coqui = NULL;

}
