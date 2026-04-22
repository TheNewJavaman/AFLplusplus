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
/* Intentionally NOT including "afl-mutations.h": that header emits non-static
 * array/function definitions (interesting_8/16/32, text_array, binary_array,
 * afl_mutate, ...) into every translation unit that includes it. afl-fuzz-one.c
 * already includes it; including it here too produces multiple-definition
 * link errors under LTO. Task 2.4 (coqui_refresh_seed_pool) uses local
 * `extern u32 binary_array[];` declarations at the point of use, per plan
 * line 1613-1616. */

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

/* AFL mutation-array selectors --- defined in src/afl-fuzz-one.c (included via
 * afl-mutations.h in that TU). We reference them by extern to avoid pulling
 * the full afl-mutations.h here (it emits multi-defined globals). */
extern u32 binary_array[];
extern u32 text_array[];
extern u32 mutation_strategy_exploration_binary[];
extern u32 mutation_strategy_exploitation_binary[];
/* The MUT_*_ARRAY_SIZE macros we need inline-define locally to match
 * afl-mutations.h:34,88,290. */
#define COQUI_MUT_STRATEGY_ARRAY_SIZE 256
#define COQUI_MUT_TXT_ARRAY_SIZE      200
#define COQUI_MUT_BIN_ARRAY_SIZE      256

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

  /* slot_info — host pinned + device mirror. u32 per thread. */
  CUCHECK(cuMemHostAlloc((void**)&b->h_slot_info, ctx->batch_size * 4, 0));
  CUCHECK(cuMemAlloc(&p, ctx->batch_size * 4));
  b->d_slot_info = (unsigned long long)p;

  /* reported_slab + metadata — compact output for novel/crash threads.
   * REPORTED_CAP * max_input_size bytes on host pinned and device. */
  size_t slab_bytes = (size_t)COQUI_REPORTED_CAP * ctx->max_input_size;
  CUCHECK(cuMemHostAlloc((void**)&b->h_reported_slab, slab_bytes, 0));
  CUCHECK(cuMemHostAlloc((void**)&b->h_reported_tid,  COQUI_REPORTED_CAP * 4, 0));
  CUCHECK(cuMemHostAlloc((void**)&b->h_reported_lens, COQUI_REPORTED_CAP * 4, 0));
  CUCHECK(cuMemAlloc(&p, slab_bytes));
  b->d_reported_slab = (unsigned long long)p;
  CUCHECK(cuMemAlloc(&p, COQUI_REPORTED_CAP * 4));
  b->d_reported_tid  = (unsigned long long)p;
  CUCHECK(cuMemAlloc(&p, COQUI_REPORTED_CAP * 4));
  b->d_reported_lens = (unsigned long long)p;
  b->h_reported_count = 0;

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

  /* Bind __coqui_kernel_timing[5] — 5 u64 slots accumulating per-phase
   * cycles across all threads in the batch. The kernel atomic-adds its
   * phase deltas to this; host memsets before each launch and DtoH
   * reads afterwards. */
  CUdeviceptr d_timing;
  size_t timing_sz;
  CUCHECK(cuModuleGetGlobal(&d_timing, &timing_sz, mod, "__coqui_kernel_timing"));
  if (timing_sz != 5 * sizeof(unsigned long long)) {
    FATAL("__coqui_kernel_timing symbol size %zu != 40B", timing_sz);
  }
  ctx->d_kernel_timing = (unsigned long long)d_timing;
  CUCHECK(cuMemsetD8(d_timing, 0, timing_sz));
  memset(ctx->k_cycles, 0, sizeof(ctx->k_cycles));
  ctx->k_batch_count = 0;

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

  /* Runtime bitcode was compiled with -DMAX_INPUT_SIZE=4096. Host-side
   * max_length must not exceed this or kernel .local scratch will
   * truncate mutated inputs. */
  if (ctx->max_input_size > 4096) {
    FATAL("coqui_mode: afl->max_length (%u) > MAX_INPUT_SIZE (4096). "
          "Rebuild target cubin with -DMAX_INPUT_SIZE=%u.",
          ctx->max_input_size, ctx->max_input_size);
  }

  /* 7. Create streams */
  CUstream sa, sb;
  CUCHECK(cuStreamCreate(&sa, CU_STREAM_NON_BLOCKING));
  CUCHECK(cuStreamCreate(&sb, CU_STREAM_NON_BLOCKING));
  ctx->stream_a = (void *)sa;
  ctx->stream_b = (void *)sb;

  CUstream sp;
  CUCHECK(cuStreamCreate(&sp, CU_STREAM_NON_BLOCKING));
  ctx->stream_pool = (void *)sp;

  /* 8. Allocate ping-pong pair */
  alloc_batch_half_cuda(&ctx->ping, ctx, sa);
  alloc_batch_half_cuda(&ctx->pong, ctx, sb);

  /* Seed pool: 1 MB per side, capacity 256 slots. Packed (variable-length). */
  ctx->seed_pool_cap_bytes = 1024u * 1024u;
  CUCHECK(cuMemHostAlloc((void**)&ctx->h_seed_pool_bytes_a,   ctx->seed_pool_cap_bytes, 0));
  CUCHECK(cuMemHostAlloc((void**)&ctx->h_seed_pool_bytes_b,   ctx->seed_pool_cap_bytes, 0));
  CUCHECK(cuMemHostAlloc((void**)&ctx->h_seed_pool_offsets_a, 256 * 4, 0));
  CUCHECK(cuMemHostAlloc((void**)&ctx->h_seed_pool_offsets_b, 256 * 4, 0));
  CUCHECK(cuMemHostAlloc((void**)&ctx->h_seed_pool_lens_a,    256 * 4, 0));
  CUCHECK(cuMemHostAlloc((void**)&ctx->h_seed_pool_lens_b,    256 * 4, 0));
  CUCHECK(cuMemHostAlloc((void**)&ctx->h_seed_pool_cumw_a,    256 * 4, 0));
  CUCHECK(cuMemHostAlloc((void**)&ctx->h_seed_pool_cumw_b,    256 * 4, 0));

  {
    CUdeviceptr dp;
    CUCHECK(cuMemAlloc(&dp, ctx->seed_pool_cap_bytes));
    ctx->d_seed_pool_bytes_a = (unsigned long long)dp;
    CUCHECK(cuMemAlloc(&dp, ctx->seed_pool_cap_bytes));
    ctx->d_seed_pool_bytes_b = (unsigned long long)dp;
    CUCHECK(cuMemAlloc(&dp, 256 * 4)); ctx->d_seed_pool_offsets_a = (unsigned long long)dp;
    CUCHECK(cuMemAlloc(&dp, 256 * 4)); ctx->d_seed_pool_offsets_b = (unsigned long long)dp;
    CUCHECK(cuMemAlloc(&dp, 256 * 4)); ctx->d_seed_pool_lens_a    = (unsigned long long)dp;
    CUCHECK(cuMemAlloc(&dp, 256 * 4)); ctx->d_seed_pool_lens_b    = (unsigned long long)dp;
    CUCHECK(cuMemAlloc(&dp, 256 * 4)); ctx->d_seed_pool_cumw_a    = (unsigned long long)dp;
    CUCHECK(cuMemAlloc(&dp, 256 * 4)); ctx->d_seed_pool_cumw_b    = (unsigned long long)dp;
  }

  ctx->seed_pool_active = 0;

  /* Resolve the 21 module-level globals that the GPU havoc runtime reads.
   * All live as extern declarations in coqui_mode/runtime/coqui_runtime.h
   * and are populated either here (once at init) or by coqui_refresh_seed_pool
   * (per queue_cur) or coqui_launch_batch (per batch). */
  #define BIND_SYM(field, name, expected_sz) do {                               \
    CUdeviceptr _sp; size_t _sz;                                                \
    CUCHECK(cuModuleGetGlobal(&_sp, &_sz, mod, name));                          \
    if (_sz != (expected_sz)) FATAL(name " size %zu != %zu", _sz, (size_t)(expected_sz));\
    ctx->field = (unsigned long long)_sp;                                       \
  } while(0)

  BIND_SYM(sym_seed_pool_base,       "__coqui_seed_pool_base",       sizeof(void*));
  BIND_SYM(sym_seed_pool_offsets,    "__coqui_seed_pool_offsets",    sizeof(void*));
  BIND_SYM(sym_seed_pool_lens,       "__coqui_seed_pool_lens",       sizeof(void*));
  BIND_SYM(sym_seed_pool_cumw,       "__coqui_seed_pool_cumw",       sizeof(void*));
  BIND_SYM(sym_seed_pool_count,      "__coqui_seed_pool_count",      4);
  BIND_SYM(sym_seed_pool_cumw_total, "__coqui_seed_pool_cumw_total", 4);
  BIND_SYM(sym_prng_base,            "__coqui_prng_base",            8);
  BIND_SYM(sym_reported_count,       "__coqui_reported_count",       4);
  BIND_SYM(sym_reported_tid,         "__coqui_reported_tid",         sizeof(void*));
  BIND_SYM(sym_reported_lens,        "__coqui_reported_lens",        sizeof(void*));
  BIND_SYM(sym_mutation_array,       "__coqui_mutation_array",       256 * 4);
  BIND_SYM(sym_mutation_array_size,  "__coqui_mutation_array_size",  4);
  BIND_SYM(sym_havoc_stack_pow2,     "__coqui_havoc_stack_pow2",     4);
  BIND_SYM(sym_queue_cycle,          "__coqui_queue_cycle",          4);
  BIND_SYM(sym_run_over10m,          "__coqui_run_over10m",          4);
  BIND_SYM(sym_extras_base,          "__coqui_extras_base",          sizeof(void*));
  BIND_SYM(sym_extras_offsets,       "__coqui_extras_offsets",       sizeof(void*));
  BIND_SYM(sym_extras_lens,          "__coqui_extras_lens",          sizeof(void*));
  BIND_SYM(sym_extras_cnt,           "__coqui_extras_cnt",           4);
  BIND_SYM(sym_a_extras_base,        "__coqui_a_extras_base",        sizeof(void*));
  BIND_SYM(sym_a_extras_offsets,     "__coqui_a_extras_offsets",     sizeof(void*));
  BIND_SYM(sym_a_extras_lens,        "__coqui_a_extras_lens",        sizeof(void*));
  BIND_SYM(sym_a_extras_cnt,         "__coqui_a_extras_cnt",         4);
  #undef BIND_SYM

  /* Initialize uploaded-version counters so first refresh detects the change. */
  ctx->extras_cnt_uploaded = 0;
  ctx->a_extras_cnt_uploaded = 0;
  ctx->mut_array_uploaded = 0;

  if (afl->extras_cnt > 0) {
    /* Compute packed byte total */
    size_t total = 0;
    for (u32 i = 0; i < afl->extras_cnt; ++i) total += afl->extras[i].len;
    CUdeviceptr p_ex; CUCHECK(cuMemAlloc(&p_ex, total ? total : 1));
    ctx->d_extras_base = (unsigned long long)p_ex;

    u32 *off_h = ck_alloc(afl->extras_cnt * 4);
    u32 *len_h = ck_alloc(afl->extras_cnt * 4);
    u8  *pak   = ck_alloc(total ? total : 1);
    size_t cur = 0;
    for (u32 i = 0; i < afl->extras_cnt; ++i) {
      off_h[i] = (u32)cur;
      len_h[i] = (u32)afl->extras[i].len;
      if (afl->extras[i].len > 0) {
        memcpy(pak + cur, afl->extras[i].data, afl->extras[i].len);
      }
      cur += afl->extras[i].len;
    }
    if (total > 0) {
      CUCHECK(cuMemcpyHtoD((CUdeviceptr)ctx->d_extras_base, pak, total));
    }

    CUdeviceptr p_off; CUCHECK(cuMemAlloc(&p_off, afl->extras_cnt * 4));
    CUCHECK(cuMemcpyHtoD(p_off, off_h, afl->extras_cnt * 4));
    ctx->d_extras_offsets = (unsigned long long)p_off;

    CUdeviceptr p_len; CUCHECK(cuMemAlloc(&p_len, afl->extras_cnt * 4));
    CUCHECK(cuMemcpyHtoD(p_len, len_h, afl->extras_cnt * 4));
    ctx->d_extras_lens = (unsigned long long)p_len;

    /* Bind pointer symbols */
    CUCHECK(cuMemcpyHtoD((CUdeviceptr)ctx->sym_extras_base,    &ctx->d_extras_base,    8));
    CUCHECK(cuMemcpyHtoD((CUdeviceptr)ctx->sym_extras_offsets, &ctx->d_extras_offsets, 8));
    CUCHECK(cuMemcpyHtoD((CUdeviceptr)ctx->sym_extras_lens,    &ctx->d_extras_lens,    8));
    u32 cnt = afl->extras_cnt;
    CUCHECK(cuMemcpyHtoD((CUdeviceptr)ctx->sym_extras_cnt, &cnt, 4));
    ctx->extras_cnt_uploaded = cnt;

    ck_free(off_h); ck_free(len_h); ck_free(pak);
  } else {
    u32 zero = 0;
    CUCHECK(cuMemcpyHtoD((CUdeviceptr)ctx->sym_extras_cnt, &zero, 4));
  }

  /* a_extras starts empty; host writes 0 to sym so kernel sees safe state. */
  {
    u32 zero = 0;
    CUCHECK(cuMemcpyHtoD((CUdeviceptr)ctx->sym_a_extras_cnt, &zero, 4));
  }

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

void coqui_refresh_seed_pool(afl_state_t *afl) {
  coqui_ctx_t *ctx = afl->coqui;
  if (!ctx) return;
  /* Flush any pending batch under the OLD seed pool before we flip the
   * pointer symbols. Otherwise pending flag=1 slots (seed_idx=0) would
   * reference the NEW queue_cur's slot 0 when they finally launch. */
  coqui_flush_batch(afl);
  CUstream sp = (CUstream)ctx->stream_pool;
  u8 next = 1 - ctx->seed_pool_active;

  /* Pick destination buffers for the new side. */
  u8  *dst_bytes   = (next == 0) ? ctx->h_seed_pool_bytes_a   : ctx->h_seed_pool_bytes_b;
  u32 *dst_offsets = (next == 0) ? ctx->h_seed_pool_offsets_a : ctx->h_seed_pool_offsets_b;
  u32 *dst_lens    = (next == 0) ? ctx->h_seed_pool_lens_a    : ctx->h_seed_pool_lens_b;
  u32 *dst_cumw    = (next == 0) ? ctx->h_seed_pool_cumw_a    : ctx->h_seed_pool_cumw_b;

  unsigned long long d_bytes   = (next == 0) ? ctx->d_seed_pool_bytes_a   : ctx->d_seed_pool_bytes_b;
  unsigned long long d_offsets = (next == 0) ? ctx->d_seed_pool_offsets_a : ctx->d_seed_pool_offsets_b;
  unsigned long long d_lens    = (next == 0) ? ctx->d_seed_pool_lens_a    : ctx->d_seed_pool_lens_b;
  unsigned long long d_cumw    = (next == 0) ? ctx->d_seed_pool_cumw_a    : ctx->d_seed_pool_cumw_b;

  /* Slot 0: queue_cur (always). Guard against missing/too-large input. */
  u32 cursor = 0;
  u32 count = 0;
  u64 cumw_total = 0;

  u8 *cur_buf = NULL;
  if (afl->queue_cur && afl->queue_cur->len > 0 &&
      afl->queue_cur->len <= ctx->seed_pool_cap_bytes) {
    cur_buf = queue_testcase_get(afl, afl->queue_cur);
  }
  if (cur_buf) {
    memcpy(dst_bytes + cursor, cur_buf, afl->queue_cur->len);
    dst_offsets[0] = cursor;
    dst_lens[0]    = afl->queue_cur->len;
    cursor += afl->queue_cur->len;
    cumw_total += (u64)(afl->queue_cur->weight * 1000.0);
    dst_cumw[0] = (u32)cumw_total;
    count = 1;
  } else {
    /* No usable queue_cur --- produce empty pool; kernel will fall through splice
     * rejection paths and no thread can splice. */
    ctx->h_seed_pool_count_next = 0;
    ctx->h_seed_pool_cumw_total_next = 0;
    ctx->seed_pool_active = next;
    u32 zero = 0;
    CUCHECK(cuMemcpyHtoD((CUdeviceptr)ctx->sym_seed_pool_count,      &zero, 4));
    CUCHECK(cuMemcpyHtoD((CUdeviceptr)ctx->sym_seed_pool_cumw_total, &zero, 4));
    return;
  }

  /* Slots 1..255: weighted-random-with-replacement samples of queue_buf.
   * Build a cumulative weight array (over all enabled queue entries) once,
   * then sample N-1 times. O(Q) build + O(log Q * N) sample. */
  u32 Q = afl->queued_items;
  if (Q > 1 && count < 256) {
    u64 *cdf = ck_alloc(Q * sizeof(u64));
    u64 total = 0;
    for (u32 i = 0; i < Q; ++i) {
      struct queue_entry *q = afl->queue_buf[i];
      u64 w = (q && !q->disabled) ? (u64)(q->weight * 1000.0) : 0;
      total += w;
      cdf[i] = total;
    }
    if (total > 0) {
      for (u32 slot = count; slot < 256; ++slot) {
        u64 r = (u64)rand_below(afl, (u32)(total > 0xFFFFFFFFu ? 0xFFFFFFFFu : (u32)total));
        u32 lo = 0, hi = Q;
        while (lo < hi) {
          u32 m = (lo + hi) >> 1;
          if (cdf[m] <= r) lo = m + 1; else hi = m;
        }
        if (lo >= Q) lo = Q - 1;
        struct queue_entry *q = afl->queue_buf[lo];
        if (!q || q->len == 0) continue;
        if (cursor + q->len > ctx->seed_pool_cap_bytes) break;
        u8 *q_buf = queue_testcase_get(afl, q);
        if (!q_buf) continue;
        memcpy(dst_bytes + cursor, q_buf, q->len);
        dst_offsets[slot] = cursor;
        dst_lens[slot]    = q->len;
        cursor += q->len;
        cumw_total += (u64)(q->weight * 1000.0);
        dst_cumw[slot] = (u32)cumw_total;
        count++;
      }
    }
    ck_free(cdf);
  }

  /* HtoD on stream_pool --- overlaps in-flight kernel on stream_a/b. */
  CUCHECK(cuMemcpyHtoDAsync((CUdeviceptr)d_bytes,   dst_bytes,   cursor,       sp));
  CUCHECK(cuMemcpyHtoDAsync((CUdeviceptr)d_offsets, dst_offsets, count * 4,    sp));
  CUCHECK(cuMemcpyHtoDAsync((CUdeviceptr)d_lens,    dst_lens,    count * 4,    sp));
  CUCHECK(cuMemcpyHtoDAsync((CUdeviceptr)d_cumw,    dst_cumw,    count * 4,    sp));

  /* Fence: subsequent kernel launches on stream_a/b must wait for this upload. */
  CUevent ev;
  CUCHECK(cuEventCreate(&ev, CU_EVENT_DISABLE_TIMING));
  CUCHECK(cuEventRecord(ev, sp));
  CUCHECK(cuStreamWaitEvent((CUstream)ctx->stream_a, ev, 0));
  CUCHECK(cuStreamWaitEvent((CUstream)ctx->stream_b, ev, 0));
  CUCHECK(cuEventDestroy(ev));

  /* Flip pointer symbols + scalar count/cumw_total. */
  CUCHECK(cuMemcpyHtoD((CUdeviceptr)ctx->sym_seed_pool_base,    &d_bytes,   8));
  CUCHECK(cuMemcpyHtoD((CUdeviceptr)ctx->sym_seed_pool_offsets, &d_offsets, 8));
  CUCHECK(cuMemcpyHtoD((CUdeviceptr)ctx->sym_seed_pool_lens,    &d_lens,    8));
  CUCHECK(cuMemcpyHtoD((CUdeviceptr)ctx->sym_seed_pool_cumw,    &d_cumw,    8));
  u32 cnt_u32 = count;
  u32 cumw_u32 = (u32)(cumw_total & 0xFFFFFFFFu);
  CUCHECK(cuMemcpyHtoD((CUdeviceptr)ctx->sym_seed_pool_count,      &cnt_u32,  4));
  CUCHECK(cuMemcpyHtoD((CUdeviceptr)ctx->sym_seed_pool_cumw_total, &cumw_u32, 4));

  ctx->h_seed_pool_count_next = count;
  ctx->h_seed_pool_cumw_total_next = cumw_u32;
  ctx->seed_pool_active = next;

  /* Upload mutation_array if (input_mode, fuzz_mode) changed. */
  if (!ctx->mut_array_uploaded ||
      ctx->mut_array_input_mode != afl->input_mode ||
      ctx->mut_array_fuzz_mode  != afl->fuzz_mode) {
    /* Mutation-array selection matches src/afl-fuzz-one.c:2180-2233. */
    u32 *active;
    u32  size;
    if (afl->input_mode == 1 /* TEXT */) {
      if (afl->fuzz_mode == 0) { active = binary_array; size = COQUI_MUT_BIN_ARRAY_SIZE; }
      else                     { active = text_array;   size = COQUI_MUT_TXT_ARRAY_SIZE; }
    } else if (afl->input_mode == 2 /* BINARY */) {
      if (afl->fuzz_mode == 0) { active = mutation_strategy_exploration_binary;
                                  size   = COQUI_MUT_STRATEGY_ARRAY_SIZE; }
      else                     { active = mutation_strategy_exploitation_binary;
                                  size   = COQUI_MUT_STRATEGY_ARRAY_SIZE; }
    } else {
      /* DEFAULT / generic */
      if (afl->fuzz_mode == 0) { active = binary_array; size = COQUI_MUT_BIN_ARRAY_SIZE; }
      else                     { active = text_array;   size = COQUI_MUT_TXT_ARRAY_SIZE; }
    }
    /* Zero the device-side array first so unused slots (beyond `size`) don't
     * carry stale data if a later mode switch has a smaller array. */
    u32 zero256[256] = {0};
    CUCHECK(cuMemcpyHtoD((CUdeviceptr)ctx->sym_mutation_array, zero256, 256 * 4));
    CUCHECK(cuMemcpyHtoD((CUdeviceptr)ctx->sym_mutation_array, active,  size * 4));
    CUCHECK(cuMemcpyHtoD((CUdeviceptr)ctx->sym_mutation_array_size, &size, 4));
    ctx->mut_array_input_mode = afl->input_mode;
    ctx->mut_array_fuzz_mode  = afl->fuzz_mode;
    ctx->mut_array_uploaded = 1;
  }

  /* Always update havoc_stack_pow2 + queue_cycle + run_over10m (cheap scalars). */
  u32 pow2 = afl->havoc_stack_pow2;
  CUCHECK(cuMemcpyHtoD((CUdeviceptr)ctx->sym_havoc_stack_pow2, &pow2, 4));
  u32 qc = afl->queue_cycle;
  CUCHECK(cuMemcpyHtoD((CUdeviceptr)ctx->sym_queue_cycle, &qc, 4));
  u32 over10m = afl->run_over10m ? 1u : 0u;
  CUCHECK(cuMemcpyHtoD((CUdeviceptr)ctx->sym_run_over10m, &over10m, 4));

  /* Opportunistic a_extras re-upload if cnt changed. */
  if (afl->a_extras_cnt != ctx->a_extras_cnt_uploaded) {
    /* For v1 we update the count but skip the actual bytes upload. a_extras
     * slots remain empty so EXTRA_AUTO_* ops fall through retry. This avoids
     * having to free-and-realloc device buffers every time cmplog finds a
     * new extra; can be tightened in a follow-up if bench shows it matters. */
    ctx->a_extras_cnt_uploaded = afl->a_extras_cnt;
    u32 zero = 0;
    CUCHECK(cuMemcpyHtoD((CUdeviceptr)ctx->sym_a_extras_cnt, &zero, 4));
  }
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

  /* HtoD slot_info — one u32 per thread encoding (flag, seed_idx). */
  CUCHECK(cuMemcpyHtoDAsync((CUdeviceptr)b->d_slot_info,
                             b->h_slot_info, ctx->batch_size * 4, s));

  /* Zero reported_count so compact-write starts fresh. __coqui_reported_count
   * is a u32 counter (kernel atomicAdds on the module global directly), not
   * a pointer slot — so we zero the module global directly, no HtoD rebind. */
  CUCHECK(cuMemsetD32Async((CUdeviceptr)ctx->sym_reported_count, 0, 1, s));

  /* Bind this batch's reported_tid/lens pointers into the module globals so
   * the kernel's compact-report block writes to the correct side. */
  CUCHECK(cuMemcpyHtoDAsync((CUdeviceptr)ctx->sym_reported_tid,
                             &b->d_reported_tid, 8, s));
  CUCHECK(cuMemcpyHtoDAsync((CUdeviceptr)ctx->sym_reported_lens,
                             &b->d_reported_lens, 8, s));

  /* Pick a fresh per-batch PRNG base. AFL's rand_below gives u32; combine
   * two into u64. Thread-local PRNG state = splitmix64(prng_base ^ tid). */
  u64 prng_base = (u64)rand_below(afl, 0xFFFFFFFFu)
                | ((u64)rand_below(afl, 0xFFFFFFFFu) << 32);
  CUCHECK(cuMemcpyHtoDAsync((CUdeviceptr)ctx->sym_prng_base, &prng_base, 8, s));

  /* Launch */
  void *args[] = {
    (void *)&b->d_input_bytes,
    (void *)&b->d_offsets,
    (void *)&b->d_input_lens,
    (void *)&b->d_slot_info,    /* NEW */
    (void *)&b->d_novelty,
    (void *)&b->d_status,
    (void *)&b->d_reported_slab, /* NEW */
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

  /* DtoH reported_count + full slab (we'll only read the populated prefix
   * on the host). Overhead: ~2 MB per batch, acceptable. */
  CUCHECK(cuMemcpyDtoHAsync(&b->h_reported_count,
                             (CUdeviceptr)ctx->sym_reported_count, 4, s));
  CUCHECK(cuMemcpyDtoHAsync(b->h_reported_slab,
                             (CUdeviceptr)b->d_reported_slab,
                             (size_t)COQUI_REPORTED_CAP * ctx->max_input_size, s));
  CUCHECK(cuMemcpyDtoHAsync(b->h_reported_tid,
                             (CUdeviceptr)b->d_reported_tid,
                             COQUI_REPORTED_CAP * 4, s));
  CUCHECK(cuMemcpyDtoHAsync(b->h_reported_lens,
                             (CUdeviceptr)b->d_reported_lens,
                             COQUI_REPORTED_CAP * 4, s));

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

        /* GPU-side kernel phase timing: synchronous DtoH of the 5-u64
         * cumulative cycle counter; subtract prior reading to get cycles
         * consumed during this window; divide by dl*batch_size for avg
         * cycles/thread. Percentages are slot[0..3] / slot[4] — total
         * includes memory_init+exec+classify+virgin, so pct sums to ~100. */
        unsigned long long cur_cycles[5] = {0};
        cuMemcpyDtoH(cur_cycles, (CUdeviceptr)ctx->d_kernel_timing,
                     sizeof(cur_cycles));
        unsigned long long delta_cycles[5];
        for (int i = 0; i < 5; i++) {
          delta_cycles[i] = cur_cycles[i] - ctx->k_cycles[i];
          ctx->k_cycles[i] = cur_cycles[i];  /* store for next window */
        }
        double pct[4] = {0,0,0,0};
        if (delta_cycles[4] > 0) {
          for (int i = 0; i < 4; i++) {
            pct[i] = 100.0 * (double)delta_cycles[i] / (double)delta_cycles[4];
          }
        }

        fprintf(stderr,
                "[coqui-rate] %.1f batches/s, %llu submits/s "
                "(avg %.0f inputs/batch, %llu slow-skipped, "
                "crash_dedup_hits=%llu avg=%.1f/batch, "
                "crash_verify_calls=%llu avg=%.1f/batch, "
                "submit=%.0fus await=%.0fus verify=%.0fus /batch, "
                "kern: init=%.0f%% exec=%.0f%% classify=%.0f%% virgin=%.0f%%)\n",
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
                avg_verify_us_batch,
                pct[0], pct[1], pct[2], pct[3]);
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

  /* Build tid → reported-slot map for flag=1 inputs.
   * reported_count is the authoritative count after DtoH completed;
   * cap at COQUI_REPORTED_CAP in case kernel over-reported (shouldn't). */
  u32 rcount = b->h_reported_count;
  if (rcount > COQUI_REPORTED_CAP) {
    WARNF("coqui reported_count %u > CAP %u — some novel slots will be "
          "unverified this batch", rcount, COQUI_REPORTED_CAP);
    rcount = COQUI_REPORTED_CAP;
  }
  /* Small hash: 1024 buckets linear-probe over up to CAP (512) entries.
   * Load factor ≤ 50% → fast probe. */
  u32 tid2slot_keys[1024];
  u32 tid2slot_vals[1024];
  u8  tid2slot_used[1024];
  memset(tid2slot_used, 0, sizeof(tid2slot_used));
  for (u32 r = 0; r < rcount; ++r) {
    u32 t = b->h_reported_tid[r];
    u32 h = (t * 0x9E3779B1u) & 1023u;
    for (u32 p = 0; p < 1024; ++p) {
      u32 k = (h + p) & 1023u;
      if (!tid2slot_used[k]) {
        tid2slot_keys[k] = t;
        tid2slot_vals[k] = r;
        tid2slot_used[k] = 1;
        break;
      }
    }
  }

  /* Process flagged inputs (novelty bitmap bits set) */
  for (u32 word_i = 0; word_i < ctx->batch_size / 32; word_i++) {
    u32 bits = ((u32 *)b->h_novelty)[word_i];
    while (bits) {
      u32 bit_pos = __builtin_ctz(bits);
      bits &= bits - 1;
      u32 i = word_i * 32 + bit_pos;

      /* Dispatch on slot_info flag to find where the input bytes live. */
      u32 slot_info = b->h_slot_info[i];
      u8  flag      = (u8)(slot_info >> 24);
      u8 *input;
      u32 len;
      if (flag == COQUI_FLAG_HAVOC) {
        /* Look up reported slot for this tid. */
        u32 h = (i * 0x9E3779B1u) & 1023u;
        u32 slot = (u32)-1;
        for (u32 p = 0; p < 1024; ++p) {
          u32 k = (h + p) & 1023u;
          if (!tid2slot_used[k]) break;        /* not present */
          if (tid2slot_keys[k] == i) { slot = tid2slot_vals[k]; break; }
        }
        if (slot == (u32)-1) continue;         /* overflowed slab; skip */
        input = b->h_reported_slab + slot * ctx->max_input_size;
        len   = b->h_reported_lens[slot];
      } else {
        if (b->h_input_lens[i] == 0) continue;
        input = b->h_input_bytes + b->h_offsets[i];
        len   = b->h_input_lens[i];
      }

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
    /* Dispatch on slot_info flag to find where the input bytes live. */
    u32 slot_info = b->h_slot_info[i];
    u8  flag = (u8)(slot_info >> 24);
    u8 *input;
    u32 len;
    if (flag == COQUI_FLAG_HAVOC) {
      u32 h = (i * 0x9E3779B1u) & 1023u;
      u32 slot = (u32)-1;
      for (u32 p = 0; p < 1024; ++p) {
        u32 k = (h + p) & 1023u;
        if (!tid2slot_used[k]) break;
        if (tid2slot_keys[k] == i) { slot = tid2slot_vals[k]; break; }
      }
      if (slot == (u32)-1) continue;
      input = b->h_reported_slab + slot * ctx->max_input_size;
      len   = b->h_reported_lens[slot];
    } else {
      if (b->h_input_lens[i] == 0) continue;
      input = b->h_input_bytes + b->h_offsets[i];
      len   = b->h_input_lens[i];
    }
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

u8 coqui_submit_havoc_slot(afl_state_t *afl, u32 seed_idx, u8 flag) {
  coqui_ctx_t  *ctx = afl->coqui;
  coqui_batch_t *b  = ctx->pending;

  ctx->total_submits++;
  /* Count each submission as one exec, matching coqui_submit_input's convention. */
  afl->fsrv.total_execs++;

  if (b->n_inputs == ctx->batch_size) {
    /* Batch full — launch + flip ping-pong. Same mechanics as coqui_submit_input. */
    coqui_launch_batch(afl, b);
    coqui_batch_t *tmp = ctx->pending;
    ctx->pending = ctx->executing;
    ctx->executing = tmp;
    b = ctx->pending;
    if (b->n_inputs > 0) {
      if (coqui_await_and_process(afl, b) == 1) return 0;   /* force-reset path */
      b->n_inputs = 0;
      b->bytes_used = 0;
    }
  }

  b->h_slot_info[b->n_inputs] = ((u32)flag << 24) | (seed_idx & 0xFFFFFFu);
  /* offsets/lens default to 0 for flag=1 slots; kernel ignores them when
   * flag != COQUI_FLAG_PREMUT. */
  b->h_offsets[b->n_inputs] = 0;
  b->h_input_lens[b->n_inputs] = 0;
  b->n_inputs++;
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

  if (ctx->ping.h_slot_info)     cuMemFreeHost(ctx->ping.h_slot_info);
  if (ctx->ping.h_reported_slab) cuMemFreeHost(ctx->ping.h_reported_slab);
  if (ctx->ping.h_reported_tid)  cuMemFreeHost(ctx->ping.h_reported_tid);
  if (ctx->ping.h_reported_lens) cuMemFreeHost(ctx->ping.h_reported_lens);
  if (ctx->pong.h_slot_info)     cuMemFreeHost(ctx->pong.h_slot_info);
  if (ctx->pong.h_reported_slab) cuMemFreeHost(ctx->pong.h_reported_slab);
  if (ctx->pong.h_reported_tid)  cuMemFreeHost(ctx->pong.h_reported_tid);
  if (ctx->pong.h_reported_lens) cuMemFreeHost(ctx->pong.h_reported_lens);

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

  if (b->h_slot_info)     cuMemFreeHost(b->h_slot_info);
  if (b->h_reported_slab) cuMemFreeHost(b->h_reported_slab);
  if (b->h_reported_tid)  cuMemFreeHost(b->h_reported_tid);
  if (b->h_reported_lens) cuMemFreeHost(b->h_reported_lens);

  if (b->d_slot_info)      cuMemFree((CUdeviceptr)b->d_slot_info);
  if (b->d_reported_slab)  cuMemFree((CUdeviceptr)b->d_reported_slab);
  if (b->d_reported_tid)   cuMemFree((CUdeviceptr)b->d_reported_tid);
  if (b->d_reported_lens)  cuMemFree((CUdeviceptr)b->d_reported_lens);

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
