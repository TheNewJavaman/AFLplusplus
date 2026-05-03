/*
 * afl-fuzz-coqui.c --- coqui mode coqui_mode real CUDA implementation.
 *
 * Real CUDA driver API backend for coqui_mode per spec §8.4.
 *
 * Spec: docs/superpowers/specs/2026-04-18-coqui-gpu-backend-design.md
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
  ctx->d_virgin_map = (unsigned long long)d_virgin;
  ctx->cov_map_size = (u32)virgin_sz;
  CUCHECK(cuMemsetD8(d_virgin, 0, virgin_sz));
  OKF("coqui coverage map: %zu bytes (dynamic, matches edge count)",
      virgin_sz);

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

  /* Per-thread clock64 budget poison. Write the cycle ceiling into
   * __coqui_thread_budget_cycles. The Coverage pass emits a budget check at
   * every Nth static BB; on overrun the runtime calls
   * __coqui_trap_with_reason(THREAD_BUDGET_EXHAUSTED) so the thread exits
   * cleanly instead of stalling the batch's healthy work.
   *
   * Default 200ms. Set AFL_COQUI_THREAD_BUDGET_US=0 to disable.
   *
   * Cycle conversion: CU_DEVICE_ATTRIBUTE_CLOCK_RATE returns the SM base
   * clock (kHz). clock64() on-device ticks at the ACTUAL SM frequency
   * (including boost), so the real timeout varies ±15% with GPU boost state.
   * This is fine — budget poison is a coarse safety net, not a precision
   * timer. There's no CUDA driver API to query the live boost clock; nvidia-
   * smi can report it but the value changes continuously under load. */
  unsigned long long thread_budget_us =
      getenv_u64("AFL_COQUI_THREAD_BUDGET_US", 200000);
  if (thread_budget_us > 0) {
    int clock_khz = 0;
    CUCHECK(cuDeviceGetAttribute(&clock_khz,
                                  CU_DEVICE_ATTRIBUTE_CLOCK_RATE, dev));
    if (clock_khz <= 0) clock_khz = 1500000; /* defensive default */
    /* cycles = us * (kHz / 1000) */
    unsigned long long cycles_per_us =
        (unsigned long long)clock_khz / 1000ULL;
    unsigned long long budget_cycles = thread_budget_us * cycles_per_us;

    CUdeviceptr d_budget;
    size_t d_budget_sz;
    CUCHECK(cuModuleGetGlobal(&d_budget, &d_budget_sz, mod,
                               "__coqui_thread_budget_cycles"));
    if (d_budget_sz != sizeof(unsigned long long)) {
      FATAL("__coqui_thread_budget_cycles symbol size %zu != 8B", d_budget_sz);
    }
    CUCHECK(cuMemcpyHtoD(d_budget, &budget_cycles,
                          sizeof(unsigned long long)));

    /* Zero-init the per-thread start array — kernel writes its own per
     * launch but a clean default avoids any tail-of-batch surprises. */
    CUdeviceptr d_budget_start;
    size_t d_budget_start_sz;
    CUCHECK(cuModuleGetGlobal(&d_budget_start, &d_budget_start_sz, mod,
                               "__coqui_thread_budget_start"));
    CUCHECK(cuMemsetD8(d_budget_start, 0, d_budget_start_sz));
    OKF("coqui per-thread budget: %llu us = %llu cycles "
        "(@ %d kHz; %u-stride BB checks)",
        thread_budget_us, budget_cycles, clock_khz,
        64u);
  }

  /* 4. Probe the device for its maximum allowed per-thread stack size.
   *    CUDA's cuCtxSetLimit accepts values that fit within the device's
   *    .local memory budget (per_thread × max-resident-threads × #SMs must
   *    fit in device memory). There's no direct query API.
   *
   *    The driver immediately backs the requested per-thread limit with
   *    physical pages: cuCtxSetLimit(S) reserves S × n_threads bytes of
   *    device memory where n_threads = SMs × max_threads_per_SM.  On a
   *    24 GB RTX Titan (72 SMs × 1024 thr/SM = 73728 threads) even 256 KB
   *    consumes 18 GB, leaving little room for a 2 GB slab pool.
   *
   *    To compute a safe ceiling, query free device memory before any
   *    allocations, subtract the expected non-stack footprint (slab pool +
   *    slab shadow + input buffers + 2 GB headroom), and divide by the
   *    thread count.  The probe stays within this ceiling so the stack
   *    reservation leaves enough room for all subsequent allocations.
   *
   *    Phase 1: halve from the safe ceiling until the driver accepts. This
   *    bounds the answer to [last_failed/2, last_failed].
   *    Phase 2: linear-step upward from the accepted value in 8 KB
   *    increments while the driver still accepts, to recover the last
   *    bit of budget the halving missed. */

  /* Compute memory-safe ceiling for the probe.  Query free memory and
   * subtract expected non-stack footprint so the stack reservation never
   * crowds out slab pool + input buffers.  Floor at 32 KB; if the device
   * is extremely memory-constrained the subsequent probe loop will FATAL. */
  unsigned int probe_ceiling = 65536;  /* fallback; overwritten below */
  {
    size_t free_bytes = 0, total_bytes = 0;
    CUCHECK(cuMemGetInfo(&free_bytes, &total_bytes));

    int n_sms = 0, max_thr_per_sm = 0;
    CUCHECK(cuDeviceGetAttribute(&n_sms,
        CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev));
    CUCHECK(cuDeviceGetAttribute(&max_thr_per_sm,
        CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR, dev));
    unsigned long long n_threads =
        (unsigned long long)n_sms * (unsigned long long)max_thr_per_sm;

    /* Reserve: slab pool + shadow (1.125× slab_size), two input buffers
     * (MAX_ALLOC cap = 1 GB each), statics (negligible), plus 2 GB headroom.
     * Use AFL_COQUI_SLAB_SIZE if set, otherwise assume no slab. */
    unsigned long long slab_size_probe =
        (unsigned long long)getenv_u64("AFL_COQUI_SLAB_SIZE", 0);
    unsigned long long non_stack_bytes =
        slab_size_probe + slab_size_probe / 8 +  /* slab + shadow */
        2ULL * 0x40000000ULL +                   /* two input ping-pong buffers */
        2ULL * 1024ULL * 1024ULL * 1024ULL;      /* 2 GB headroom */
    unsigned long long stack_bytes_avail =
        free_bytes > non_stack_bytes ? (free_bytes - non_stack_bytes) : 0;
    unsigned long long per_thread_ceiling_bytes =
        n_threads > 0 ? (stack_bytes_avail / n_threads) : 65536ULL;
    /* Round down to 8 KB alignment; floor at 32 KB. */
    if (per_thread_ceiling_bytes < 32768) per_thread_ceiling_bytes = 32768;
    per_thread_ceiling_bytes = (per_thread_ceiling_bytes / 8192) * 8192;
    /* Hard cap: never exceed 512 KB (sm_75+ hardware ceiling). */
    if (per_thread_ceiling_bytes > 524288) per_thread_ceiling_bytes = 524288;

    probe_ceiling = (unsigned int)per_thread_ceiling_bytes;
  }

  unsigned int total_budget = probe_ceiling;
  unsigned int last_failed = total_budget * 2;
  while (total_budget >= 32768) {
    if (cuCtxSetLimit(CU_LIMIT_STACK_SIZE, total_budget) == CUDA_SUCCESS) break;
    last_failed = total_budget;
    total_budget /= 2;
  }
  if (total_budget < 32768) {
    FATAL("device rejected all per-thread stack sizes >= 32 KB");
  }
  /* Phase 2: try +8 KB increments toward last_failed, capped at
   * probe_ceiling so we never exceed the memory-safe bound computed
   * above.  Without the cap, Phase 2 would step past probe_ceiling
   * if Phase 1 accepted it without halving (last_failed = 2*ceiling). */
  while (total_budget + 8192 < last_failed &&
         total_budget + 8192 <= probe_ceiling) {
    unsigned int trial = total_budget + 8192;
    if (cuCtxSetLimit(CU_LIMIT_STACK_SIZE, trial) != CUDA_SUCCESS) break;
    total_budget = trial;
  }
  size_t actual_set = 0;
  cuCtxGetLimit(&actual_set, CU_LIMIT_STACK_SIZE);
  OKF("coqui per-thread stack budget: %u KB (driver returned %zu)",
      total_budget / 1024, actual_set);

  /* Derive heap and stack regions from the budget.
   *    Layout (per spec §4.2):
   *      [coverage 64 KB][heap H][shadow H/8][real stack S]
   *      where H = (total - 64K - S) * 8/9 and shadow = H/8.
   *
   * AFL_COQUI_HEAP_SIZE overrides H if set; the auto-derived value is used
   * otherwise. The override is rounded down to the nearest multiple of 8
   * (shadow = H/8 must be byte-exact) and bounded above by what fits in the
   * remaining budget after cov and stack. Useful for targets where the
   * auto-derived heap is wrong-sized: cmark needs more than the default
   * leaves; libjpeg-turbo benefits from a smaller heap (frees the rest for
   * call-chain headroom). */
  unsigned int stack_size = getenv_u32("AFL_COQUI_STACK_SIZE", 65536);
  unsigned int cov = ctx->cov_map_size;
  if (stack_size + cov >= total_budget) {
    FATAL("--stack-size %u + 64KB coverage >= total_budget %u",
          stack_size, total_budget);
  }
  unsigned int remaining = total_budget - cov - stack_size;
  unsigned int heap = (remaining * 8) / 9;
  unsigned int heap_override = getenv_u32("AFL_COQUI_HEAP_SIZE", 0);
  if (heap_override) {
    heap_override = (heap_override / 8) * 8;
    if (heap_override + heap_override / 8 > remaining) {
      FATAL("AFL_COQUI_HEAP_SIZE=%u + shadow %u > remaining %u (budget %u "
            "- cov 64K - stack %u). Reduce AFL_COQUI_HEAP_SIZE or grow the "
            "per-thread budget by reducing AFL_COQUI_SLAB_SIZE.",
            heap_override, heap_override / 8, remaining, total_budget,
            stack_size);
    }
    heap = heap_override;
  }
  ctx->real_stack_size = stack_size;

  /* 5. Diagnostic: report static stack usage vs budget.
   *
   * CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES is the per-thread local memory for
   * statically-allocated variables and register spills (the PTX ".local"
   * address space, mapped via LDL/STL). On NVPTX this is backed by the
   * same per-thread pool as the call stack — both are bounded by
   * cuCtxSetLimit(CU_LIMIT_STACK_SIZE, total_budget).
   *
   * Empirically, static lmem alone is not a reliable predictor of runtime
   * launch failure: libyaml runs cleanly with only ~41 KB headroom while
   * cmark/libjpeg-turbo fail at runtime for unrelated reasons even with
   * comparable margins.  We keep the diagnostic so operators see lmem and
   * call-chain headroom before launch, but we no longer block startup —
   * the warning suggests recovery actions if the first batch does fail. */
  #define CALL_HEADROOM_BYTES 65536u
  int static_usage;
  CUCHECK(cuFuncGetAttribute(&static_usage,
    CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, kernel));
  OKF("coqui kernel static lmem: %d B (budget %u B, headroom %u B required)",
      static_usage, total_budget, CALL_HEADROOM_BYTES);
  if ((unsigned int)static_usage + CALL_HEADROOM_BYTES > total_budget) {
    long long deficit = (long long)static_usage + (long long)CALL_HEADROOM_BYTES
                        - (long long)total_budget;
    WARNF("kernel static lmem %d B + %u B call-chain headroom is close to "
          "per-thread budget %u B (deficit %lld B); first kernel launch may "
          "fail with CUDA_ERROR_LAUNCH_FAILED — if so, reduce "
          "AFL_COQUI_SLAB_SIZE or shrink kernel via per-target NoInline; "
          "continuing",
          static_usage, CALL_HEADROOM_BYTES, total_budget, deficit);
  }

  /* 6. Batch sizing -- fixed-size per-thread slots.
   *
   * Each GPU thread gets a fixed max_input_size slot so grow mutations
   * (clone, insert -- 8 of 37 havoc cases) can expand into padding.
   * Total device buffer = batch_size * max_input_size per batch half.
   * At defaults (8192 * 4096) this is 32 MB -- well within GPU VRAM. */
  ctx->batch_size = getenv_u32("AFL_COQUI_BATCH_SIZE", COQUI_DEFAULT_BATCH_SIZE);
  /* Kernel grid = batch_size/128 (block-size 128 threads). batch_size must be
   * a positive multiple of 128, capped to keep ping-pong buffers reasonable. */
  if (ctx->batch_size < 128) ctx->batch_size = 128;
  if (ctx->batch_size > 65536) ctx->batch_size = 65536;
  ctx->batch_size = (ctx->batch_size / 128) * 128;
  /* Per-thread slot size for GPU buffers. Defaults to COQUI_MAX_INPUT_DEFAULT
   * (4096). Inputs larger than max_input_size are skipped (oversized_count++).
   * Overridable via AFL_COQUI_MAX_INPUT_SIZE. */
  ctx->max_input_size = getenv_u32("AFL_COQUI_MAX_INPUT_SIZE",
                                    COQUI_MAX_INPUT_DEFAULT);
  if (ctx->max_input_size < 16) ctx->max_input_size = 16;
  /* Fixed-size slots: each thread owns max_input_size bytes. */
  unsigned long long budget64 = (unsigned long long)ctx->batch_size
                                  * ctx->max_input_size;
  if (budget64 > 0x40000000ULL) budget64 = 0x40000000ULL;  /* 1 GB cap */
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

  /* 10. Slab pool (optional). Enabled when AFL_COQUI_SLAB_SIZE is set.
   *     The coqui-cc driver records --slab-pool-size in the .conf sidecar;
   *     host callers that want slab on a target should set the env var to
   *     the same value (or larger). Binds the slab-pool CUdeviceptrs into
   *     the cubin's __coqui_slab_* globals via cuModuleGetGlobal + HtoD so
   *     the device runtime finds them at launch. */
  unsigned long long slab_size = getenv_u64("AFL_COQUI_SLAB_SIZE", 0);
  ctx->slab_pool_size = slab_size;
  if (slab_size > 0) {
    CUdeviceptr slab_pool;
    CUCHECK(cuMemAlloc(&slab_pool, slab_size));
    CUCHECK(cuMemsetD8(slab_pool, 0, slab_size));
    ctx->d_slab_pool = (unsigned long long)slab_pool;

    /* Shadow is 1:8 of the pool. Poison the whole pool to 0xFA so any
     * access to unallocated slab bytes trips the tier-2 shadow check
     * immediately. The slab allocator rewrites shadow bytes to 0x00 on
     * malloc + redzones on either side. */
    unsigned long long shadow_size = slab_size / 8;
    CUdeviceptr slab_shadow;
    CUCHECK(cuMemAlloc(&slab_shadow, shadow_size));
    CUCHECK(cuMemsetD8(slab_shadow, 0xFA, shadow_size));
    ctx->d_slab_shadow = (unsigned long long)slab_shadow;

    /* Global atomic bump counter — u32, reset per batch. */
    CUdeviceptr slab_next;
    CUCHECK(cuMemAlloc(&slab_next, sizeof(unsigned int)));
    CUCHECK(cuMemsetD32(slab_next, 0, 1));
    ctx->d_slab_next = (unsigned long long)slab_next;

    /* Block budget: each block's fast-path partition size in slabs.
     * Compute ctrl_slabs exactly as the device runtime does so the host
     * and device agree on the partition geometry. Then give each block
     * ~half the leftover per-block share so the global bump path still
     * has room when any single block over-allocates. */
    unsigned int max_slabs = (unsigned int)(slab_size / 4096ULL);
    unsigned int grid = ctx->batch_size / 128;
    unsigned int ctrl_slabs =
        (unsigned int)(((unsigned long long)ctx->batch_size * 32ULL + 8ULL
                        + 4095ULL) / 4096ULL);
    unsigned int data_slabs =
        (max_slabs > ctrl_slabs) ? (max_slabs - ctrl_slabs) : 0;
    ctx->slab_block_budget = grid > 0 ? (data_slabs / grid / 2) : 0;

    /* Bind into the cubin's module globals. cuModuleGetGlobal fails if the
     * cubin wasn't compiled against the slab-enabled runtime; treat that
     * as a hard error — running with AFL_COQUI_SLAB_SIZE set against a
     * non-slab cubin would silently disable slab. */
    struct { const char *name; const void *value; size_t size; } slab_binds[] = {
        {"__coqui_slab_pool",         &slab_pool,              sizeof(CUdeviceptr)},
        {"__coqui_slab_pool_size",    &slab_size,              sizeof(unsigned long)},
        {"__coqui_slab_shadow",       &slab_shadow,            sizeof(CUdeviceptr)},
        {"__coqui_slab_next",         &slab_next,              sizeof(CUdeviceptr)},
        {"__coqui_slab_block_budget", &ctx->slab_block_budget, sizeof(unsigned int)},
    };
    for (unsigned i = 0; i < sizeof(slab_binds)/sizeof(slab_binds[0]); i++) {
      CUdeviceptr sym; size_t sym_sz;
      CUresult br = cuModuleGetGlobal(&sym, &sym_sz, mod, slab_binds[i].name);
      if (br != CUDA_SUCCESS) {
        FATAL("slab runtime missing symbol %s (cubin built without slab support?)",
              slab_binds[i].name);
      }
      if (sym_sz != slab_binds[i].size) {
        FATAL("slab symbol %s size %zu != expected %zu",
              slab_binds[i].name, sym_sz, slab_binds[i].size);
      }
      CUCHECK(cuMemcpyHtoD(sym, slab_binds[i].value, slab_binds[i].size));
    }
    OKF("coqui slab pool: %llu bytes, shadow %llu B, block_budget %u slabs",
        slab_size, shadow_size, ctx->slab_block_budget);
  }

  /* Oracle-mode line-trace setup (gated by COQUI_ORACLE=1).
   * When set, allocate per-thread trace buffer + count array and bind
   * them to the cubin's __coqui_trace_buffer / __coqui_trace_count
   * globals. If the cubin wasn't built with -coqui-line-trace,
   * cuModuleGetGlobal returns CUDA_ERROR_NOT_FOUND — log a warning and
   * leave oracle_enabled=0 so production runs against legacy cubins
   * still work without a rebuild. */
  ctx->oracle_enabled    = 0;
  ctx->d_trace_buffer    = 0;
  ctx->d_trace_count     = 0;
  ctx->h_trace_t0_buf    = NULL;
  ctx->h_trace_t0_count  = 0;
  if (getenv("COQUI_ORACLE")) {
    CUdeviceptr trace_buf_sym, trace_count_sym;
    size_t      buf_sym_sz = 0, count_sym_sz = 0;
    CUresult br1 = cuModuleGetGlobal(&trace_buf_sym, &buf_sym_sz, mod,
                                     "__coqui_trace_buffer");
    CUresult br2 = cuModuleGetGlobal(&trace_count_sym, &count_sym_sz, mod,
                                     "__coqui_trace_count");
    if (br1 != CUDA_SUCCESS || br2 != CUDA_SUCCESS) {
      WARNF("COQUI_ORACLE=1 but cubin lacks __coqui_trace_buffer/count "
            "symbols (built without -coqui-line-trace?); oracle disabled");
    } else {
      /* Allocate buffer: bs * COQUI_TRACE_BUFFER_BYTES.
       * Allocate counts: bs * sizeof(u32). */
      unsigned long long buf_bytes =
          (unsigned long long)ctx->batch_size * COQUI_TRACE_BUFFER_BYTES;
      unsigned long long count_bytes =
          (unsigned long long)ctx->batch_size * sizeof(unsigned int);
      CUdeviceptr d_buf, d_cnt;
      CUCHECK(cuMemAlloc(&d_buf, buf_bytes));
      CUCHECK(cuMemAlloc(&d_cnt, count_bytes));
      ctx->d_trace_buffer = (unsigned long long)d_buf;
      ctx->d_trace_count  = (unsigned long long)d_cnt;

      /* Write the device-side pointer values into the cubin's symbols.
       * Each symbol is a `unsigned int *` (8 bytes on the device). */
      if (buf_sym_sz != sizeof(CUdeviceptr) ||
          count_sym_sz != sizeof(CUdeviceptr)) {
        FATAL("__coqui_trace_buffer/count symbol size mismatch "
              "(buf=%zu, count=%zu, expected %zu)",
              buf_sym_sz, count_sym_sz, sizeof(CUdeviceptr));
      }
      CUCHECK(cuMemcpyHtoD(trace_buf_sym, &d_buf, sizeof(CUdeviceptr)));
      CUCHECK(cuMemcpyHtoD(trace_count_sym, &d_cnt, sizeof(CUdeviceptr)));

      /* Host-side scratch for thread-0 readback. */
      ctx->h_trace_t0_buf = ck_alloc(COQUI_TRACE_BUFFER_BYTES);
      ctx->oracle_enabled = 1;
      OKF("coqui oracle: trace buffer %llu B (%u threads × %u B), "
          "counts %llu B; bound to __coqui_trace_buffer/count",
          buf_bytes, ctx->batch_size, COQUI_TRACE_BUFFER_BYTES, count_bytes);
    }
  }

  /* GPU-side mutation (Phase 2+3). When AFL_COQUI_GPU_MUTATE=1, enable
   * per-thread havoc mutations on the GPU before calling the harness.
   * The host seeds per-thread PRNG state before each batch launch.
   *
   * Power-schedule mode: the host broadcasts each parent to
   * round_up_32(stage_max) consecutive slots. Each thread picks its own
   * stacking depth from its PRNG: steps = 1 + rand_below(stack_max).
   * stack_max starts at 4, bumped to 8 after 10 minutes (mirrors AFL). */
  ctx->gpu_mutate_enabled = 0;
  ctx->gpu_mutate_active = 0;
  ctx->d_mutate_flag = 0;
  ctx->d_mutate_prng = 0;
  ctx->d_mutate_prng_sz = 0;
  ctx->h_mutate_prng = NULL;
  ctx->d_mutate_stack_max = 0;
  {
    const char *gm_env = getenv("AFL_COQUI_GPU_MUTATE");
    if (gm_env && atoi(gm_env) == 1) {
      ctx->gpu_mutate_enabled = 1;

      /* Get device symbol pointers for PRNG, enable flag, stack_max, max_input. */
      CUdeviceptr d_prng_sym;
      size_t prng_sz;
      CUCHECK(cuModuleGetGlobal(&d_prng_sym, &prng_sz, mod,
                                 "__coqui_mutate_prng"));
      ctx->d_mutate_prng = (unsigned long long)d_prng_sym;
      ctx->d_mutate_prng_sz = prng_sz;

      /* Store enable-flag device address for per-batch toggling.
       * Flag starts at 0; set to 1 only during havoc/splice batches. */
      CUdeviceptr d_flag;
      size_t flag_sz;
      CUCHECK(cuModuleGetGlobal(&d_flag, &flag_sz, mod,
                                 "__coqui_gpu_mutate_enabled"));
      ctx->d_mutate_flag = (unsigned long long)d_flag;
      u8 zero = 0;
      CUCHECK(cuMemcpyHtoD(d_flag, &zero, 1));

      /* Per-thread stacking depth ceiling. Each GPU thread computes
       * steps = 1 + rand_below(stack_max). Initial value = 4 (matches
       * AFL's havoc_stack_pow2=1 -> stack_max = 1<<(1+1) = 4). Updated
       * to 8 after 10 min by the host pump (see coqui_launch_batch). */
      CUdeviceptr d_stack_max;
      size_t stack_max_sz;
      CUCHECK(cuModuleGetGlobal(&d_stack_max, &stack_max_sz, mod,
                                 "__coqui_mutate_stack_max"));
      ctx->d_mutate_stack_max = (unsigned long long)d_stack_max;
      u32 initial_stack_max = 4;
      CUCHECK(cuMemcpyHtoD(d_stack_max, &initial_stack_max, sizeof(u32)));

      /* Set max_input_size on device so mutation knows the buffer bound. */
      u32 mis = ctx->max_input_size;
      CUdeviceptr d_mis;
      size_t mis_sz;
      CUCHECK(cuModuleGetGlobal(&d_mis, &mis_sz, mod,
                                 "__coqui_mutate_max_input_size"));
      CUCHECK(cuMemcpyHtoD(d_mis, &mis, sizeof(u32)));

      /* Allocate pinned host buffer for per-batch PRNG seeds. */
      CUCHECK(cuMemHostAlloc((void**)&ctx->h_mutate_prng,
                              ctx->batch_size * 2 * sizeof(u64), 0));

      /* Per-parent H2D parent table (Change 1: sparse H2D).
       *
       * Allocate device buffers for the compact parent table and bind
       * their addresses into the cubin's pointer-sized globals. The host
       * packs unique parents into h_parent_bytes before each batch and
       * H2D-copies only the compact data; the GPU mutation wrapper reads
       * from the parent table instead of from per-thread input slots. */
      {
        /* Parent bytes buffer -- same size as one batch half's input buffer
         * (byte_budget). In practice, unique parents << batch_size so the
         * actual H2D will be much smaller. */
        CUdeviceptr dp;
        CUCHECK(cuMemAlloc(&dp, ctx->byte_budget));
        ctx->d_parent_bytes = (unsigned long long)dp;

        /* Bind pointer into cubin global. */
        CUdeviceptr sym; size_t sym_sz;
        CUCHECK(cuModuleGetGlobal(&sym, &sym_sz, mod, "__coqui_parent_bytes"));
        CUCHECK(cuMemcpyHtoD(sym, &dp, sizeof(CUdeviceptr)));

        /* Parent offsets -- one u32 per potential parent (max = batch_size). */
        CUCHECK(cuMemAlloc(&dp, ctx->batch_size * sizeof(u32)));
        ctx->d_parent_offsets = (unsigned long long)dp;
        CUCHECK(cuModuleGetGlobal(&sym, &sym_sz, mod, "__coqui_parent_offsets"));
        CUCHECK(cuMemcpyHtoD(sym, &dp, sizeof(CUdeviceptr)));

        /* Parent lens -- one u32 per potential parent. */
        CUCHECK(cuMemAlloc(&dp, ctx->batch_size * sizeof(u32)));
        ctx->d_parent_lens = (unsigned long long)dp;
        CUCHECK(cuModuleGetGlobal(&sym, &sym_sz, mod, "__coqui_parent_lens"));
        CUCHECK(cuMemcpyHtoD(sym, &dp, sizeof(CUdeviceptr)));

        /* Per-thread parent index -- one u32 per thread. */
        CUCHECK(cuMemAlloc(&dp, ctx->batch_size * sizeof(u32)));
        ctx->d_parent_idx = (unsigned long long)dp;
        CUCHECK(cuModuleGetGlobal(&sym, &sym_sz, mod, "__coqui_parent_idx"));
        CUCHECK(cuMemcpyHtoD(sym, &dp, sizeof(CUdeviceptr)));

        /* Host pinned buffers for parent table staging. */
        CUCHECK(cuMemHostAlloc((void**)&ctx->h_parent_bytes,
                                ctx->byte_budget, 0));
        CUCHECK(cuMemHostAlloc((void**)&ctx->h_parent_offsets,
                                ctx->batch_size * sizeof(u32), 0));
        CUCHECK(cuMemHostAlloc((void**)&ctx->h_parent_lens,
                                ctx->batch_size * sizeof(u32), 0));
        CUCHECK(cuMemHostAlloc((void**)&ctx->h_parent_idx,
                                ctx->batch_size * sizeof(u32), 0));
        ctx->parent_count = 0;
        ctx->parent_bytes_used = 0;
      }

      /* Bind __coqui_parent_count device global (for splice: GPU threads need
       * to know how many parents are in the current batch). */
      {
        CUdeviceptr sym; size_t sym_sz;
        CUCHECK(cuModuleGetGlobal(&sym, &sym_sz, mod, "__coqui_parent_count"));
        ctx->d_parent_count = (unsigned long long)sym;
        u32 zero_pc = 0;
        CUCHECK(cuMemcpyHtoD(sym, &zero_pc, sizeof(u32)));
      }

      /* Dictionary (extras) device-global binding.
       *
       * The device-side arrays are fixed-size (declared in coqui_runtime.c as
       * arrays, not pointers). We just need their addresses to cuMemcpyHtoD
       * packed data before each batch. No cuMemAlloc needed — the globals are
       * baked into the cubin's .bss / .data. */
      {
        CUdeviceptr sym; size_t sym_sz;
        CUCHECK(cuModuleGetGlobal(&sym, &sym_sz, mod, "__coqui_extras_data"));
        ctx->d_extras_data = (unsigned long long)sym;

        CUCHECK(cuModuleGetGlobal(&sym, &sym_sz, mod, "__coqui_extras_offsets"));
        ctx->d_extras_offsets = (unsigned long long)sym;

        CUCHECK(cuModuleGetGlobal(&sym, &sym_sz, mod, "__coqui_extras_lens"));
        ctx->d_extras_lens = (unsigned long long)sym;

        CUCHECK(cuModuleGetGlobal(&sym, &sym_sz, mod, "__coqui_extras_cnt"));
        ctx->d_extras_cnt = (unsigned long long)sym;

        CUCHECK(cuModuleGetGlobal(&sym, &sym_sz, mod, "__coqui_a_extras_cnt"));
        ctx->d_a_extras_cnt = (unsigned long long)sym;

        /* Initialize counts to zero on device. */
        u32 zero = 0;
        CUCHECK(cuMemcpyHtoD((CUdeviceptr)ctx->d_extras_cnt, &zero, sizeof(u32)));
        CUCHECK(cuMemcpyHtoD((CUdeviceptr)ctx->d_a_extras_cnt, &zero, sizeof(u32)));

        ctx->last_extras_cnt = 0;
        ctx->last_a_extras_cnt = 0;
      }

      OKF("coqui GPU mutation: ENABLED (power-schedule, stack_max=4, max_input=%u, "
          "sparse H2D/D2H, dict+splice)",
          mis);
    }
  }

  ctx->batch_timeout_us = getenv_u64("AFL_COQUI_TIMEOUT_US", 3000000);
  ctx->timeout_env_override = getenv("AFL_COQUI_TIMEOUT_US") ? 1 : 0;
  ctx->launch_count     = 0;
  ctx->oversized_count  = 0;
  ctx->total_submits    = 0;
  ctx->crash_dedup_hits = 0;
  ctx->crash_verify_calls = 0;
  ctx->oom_inputs_found = 0;
  ctx->oom_reruns_completed = 0;
  ctx->stack_overflow_inputs_found = 0;
  ctx->cpu_rerun_crashes = 0;
  ctx->thread_budget_inputs_found = 0;

  /* Persistent cross-batch crash-sig dedup set (1M slots = ~5 MB host RAM).
   * Saturates after ~500 s at the observed 2k-new-sigs/s rate for cjson; for
   * longer runs the fall-through-to-verify behavior keeps correctness intact
   * at the cost of losing dedup benefit as the table fills. */
  ctx->crash_sig_seen_cap   = 1u << 20;   /* 1,048,576 */
  ctx->crash_sig_seen_count = 0;
  ctx->crash_sig_persistent_hits = 0;
  ctx->crash_dedup_hits_window       = 0;
  ctx->crash_verify_calls_window     = 0;
  ctx->crash_sig_persist_hits_window = 0;
  ctx->crash_sig_seen       = ck_alloc(ctx->crash_sig_seen_cap * sizeof(u32));
  ctx->crash_sig_seen_used  = ck_alloc(ctx->crash_sig_seen_cap);
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

/* Re-run an input on the host's AFL++ CPU forkserver and route the
 * outcome through save_if_interesting exactly like process_input_via_cpu_fsrv
 * does. The difference from the normal fast path: we don't apply the speed
 * gate here. An input that OOM'd on the GPU has already been identified as
 * memory-pathological; gating it by verify time would double-penalize and
 * the input wouldn't re-run at all on slow targets.
 *
 * Returns the fault code so the caller can increment the appropriate
 * counter (OOM reruns vs real crashes vs timeouts). */
static u8 rerun_gpu_failed_input(afl_state_t *afl, u8 *input, u32 len) {
  u32 new_size = write_to_testcase(afl, (void **)&input, len, 0);
  if (new_size == 0) return FSRV_RUN_OK; /* skipped */

  u8 cpu_fault = fuzz_run_target(afl, &afl->fsrv, afl->fsrv.exec_tmout);
  afl->queued_discovered += save_if_interesting(afl, input, len, cpu_fault);
  return cpu_fault;
}

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

  /* Finer-grained submit-phase timing: split the single "submit" wall
   * into HtoD/launch/DtoH buckets so the [coqui-rate] print can show
   * which driver call is the bottleneck. Overall t_submit_us is still
   * measured (end-of-block - start) for back-compat.
   *
   * The gettimeofday wrappers add ~1 us each (3 gettimeofdays per batch
   * on top of the original 2); negligible vs batch cycle. */
  struct timeval _tv;
#define STAMP_US(var) do {                                                  \
    gettimeofday(&_tv, NULL);                                               \
    (var) = ((unsigned long long)_tv.tv_sec * 1000000ULL) + _tv.tv_usec;    \
  } while (0)

  unsigned long long _submit_t0_us;
  STAMP_US(_submit_t0_us);

  /* HtoD block */
  unsigned long long _htod_t0;
  STAMP_US(_htod_t0);

  /* H->D input data.
   *
   * GPU-mutate path (Change 1: per-parent H2D): instead of broadcasting
   * each parent to all its thread slots (128 MB), upload the compact
   * parent table: unique parent bytes + offsets + lens + per-thread index.
   * Total H2D drops from ~128 MB to ~1 MB (n_parents * avg_len + 3 small
   * metadata arrays).
   *
   * Non-mutate path: copy the used portion of input_bytes as before. */
  if (ctx->gpu_mutate_active && ctx->parent_count > 0) {
    /* Compact parent bytes -- only the packed unique parents. */
    size_t parent_data_len = (ctx->parent_bytes_used + 7u) & ~(size_t)7u;
    if (parent_data_len > 0) {
      CUCHECK(cuMemcpyHtoDAsync((CUdeviceptr)ctx->d_parent_bytes,
                                 ctx->h_parent_bytes, parent_data_len, s));
    }
    /* Parent offsets + lens -- one u32 per unique parent. */
    CUCHECK(cuMemcpyHtoDAsync((CUdeviceptr)ctx->d_parent_offsets,
                               ctx->h_parent_offsets,
                               ctx->parent_count * sizeof(u32), s));
    CUCHECK(cuMemcpyHtoDAsync((CUdeviceptr)ctx->d_parent_lens,
                               ctx->h_parent_lens,
                               ctx->parent_count * sizeof(u32), s));
    /* Per-thread parent index -- one u32 per active thread. */
    CUCHECK(cuMemcpyHtoDAsync((CUdeviceptr)ctx->d_parent_idx,
                               ctx->h_parent_idx,
                               b->n_inputs * sizeof(u32), s));
  } else {
    /* Non-mutate: copy input_bytes as before. */
    size_t input_bytes_len = (b->bytes_used + 7u) & ~(size_t)7u;
    if (input_bytes_len > 0) {
      CUCHECK(cuMemcpyHtoDAsync((CUdeviceptr)b->d_input_bytes,
                                 b->h_input_bytes, input_bytes_len, s));
    }
  }
  CUCHECK(cuMemcpyHtoDAsync((CUdeviceptr)b->d_offsets,
                             b->h_offsets, ctx->batch_size * 4, s));
  CUCHECK(cuMemcpyHtoDAsync((CUdeviceptr)b->d_input_lens,
                             b->h_input_lens, ctx->batch_size * 4, s));
  CUCHECK(cuMemsetD32Async((CUdeviceptr)b->d_novelty, 0,
                            ctx->batch_size / 32, s));
  CUCHECK(cuMemsetD8Async((CUdeviceptr)b->d_status, 0,
                           ctx->batch_size * sizeof(coqui_status_t), s));

  /* Slab per-batch reset. Three memsets:
   *   1. d_slab_next → 0        (global bump counter restart)
   *   2. slab_pool[0 .. ctrl_slabs*4096 + 8]  → 0
   *      Clears per-thread control rows (batch_size*32 bytes) and the
   *      Treiber-stack free-list head (1 u64). The +8 covers the head
   *      cell that lives at slab_pool[bs*32] per the reclamation spec.
   *   3. d_slab_shadow → 0xFA   (poison entire pool shadow; the
   *      allocator rewrites individual granules to 0x00 on malloc).
   * All three use the batch's stream so they interleave naturally with
   * the async HtoD copies above. Runs only when slab is configured
   * (checked by cubin-compile-time symbol presence -> ctx->d_slab_*
   * being non-zero). */
  if (ctx->d_slab_next) {
    /* Initialize slab_next past the per-block tier-1 partition so that
     * tier-2 global-bump allocations don't alias tier-1 block-local slabs.
     *
     * Tier-1 layout (coqui_slab.c:slab_alloc_slabs):
     *     abs_slab = ctrl_slabs + bid * block_budget + local_idx
     * occupies slab indices [ctrl_slabs, ctrl_slabs + grid * block_budget).
     * Tier-2:
     *     abs_slab = slab_idx + ctrl_slabs   (slab_idx from atom on slab_next)
     * If slab_next starts at 0, tier-2's first allocation lands at
     * ctrl_slabs + 0 — overlapping the first block's tier-1 partition,
     * silently aliasing addresses.
     *
     * Legacy coqui (coqui/driver/coqui_fuzz_driver.c:2507) initializes
     * slab_next to grid * block_budget so tier-2 starts where tier-1
     * ends. Match that. */
    unsigned int slab_next_init =
        (ctx->batch_size / 128u) * ctx->slab_block_budget;
    CUCHECK(cuMemsetD32Async((CUdeviceptr)ctx->d_slab_next, slab_next_init, 1, s));
    /* Zero ctrl-header area. bs*32 + 8 per the reclamation design spec. */
    unsigned long long ctrl_bytes =
        (unsigned long long)ctx->batch_size * 32ULL + 8ULL;
    CUCHECK(cuMemsetD8Async((CUdeviceptr)ctx->d_slab_pool, 0, ctrl_bytes, s));
  }
  if (ctx->d_slab_shadow && ctx->slab_pool_size) {
    CUCHECK(cuMemsetD8Async((CUdeviceptr)ctx->d_slab_shadow, 0xFA,
                             ctx->slab_pool_size / 8, s));
  }

  /* Zero per-thread trace counters before each launch.
   * Trace buffer contents stay (they're overwritten by index, not
   * appended), but the count array MUST start at zero so the first
   * call to __coqui_trace_line writes to slot [0]. The buffer itself
   * is left intact — the runtime only ever reads buf[idx] and writes
   * buf[*cnt], so stale bytes past the count are invisible. */
  if (ctx->oracle_enabled) {
    unsigned long long count_bytes =
        (unsigned long long)ctx->batch_size * sizeof(unsigned int);
    CUCHECK(cuMemsetD8Async((CUdeviceptr)ctx->d_trace_count, 0,
                             count_bytes, s));
  }

  /* GPU mutation: toggle the device-side enable flag per batch.
   * During havoc/splice the host sends unmutated parents and the GPU mutates;
   * during deterministic stages the flag is 0 so the GPU runs the input
   * as-is. */
  if (ctx->gpu_mutate_enabled) {
    u8 flag = ctx->gpu_mutate_active;
    CUCHECK(cuMemcpyHtoDAsync((CUdeviceptr)ctx->d_mutate_flag,
                               &flag, 1, s));

    /* Update stack_max based on run_over10m (mirrors AFL's havoc_stack_pow2
     * bump: 4 early, 8 after 10 min). Only HtoD when the value changes. */
    if (flag && ctx->d_mutate_stack_max) {
      u32 new_sm = afl->run_over10m ? 8 : 4;
      /* Cheap: always write the 4-byte value. HtoD on stream is negligible
       * vs the batch size, and avoids tracking a cached value. */
      CUCHECK(cuMemcpyHtoDAsync((CUdeviceptr)ctx->d_mutate_stack_max,
                                 &new_sm, sizeof(u32), s));
    }

    /* PRNG seeding: fill per-thread seed pairs and HtoD before kernel
     * launch so each thread has unique, non-repeating havoc entropy.
     * In power-schedule mode each parent group has n_threads slots; each
     * thread gets a unique seed derived from the host PRNG + thread index. */
    if (flag) {
      for (u32 i = 0; i < b->n_inputs; i++) {
        ctx->h_mutate_prng[i * 2]     = rand_next(afl) ^ (u64)i;
        ctx->h_mutate_prng[i * 2 + 1] = rand_next(afl) ^ ((u64)i << 32);
      }
      CUCHECK(cuMemcpyHtoDAsync((CUdeviceptr)ctx->d_mutate_prng,
                                 ctx->h_mutate_prng,
                                 b->n_inputs * 2 * sizeof(u64), s));

      /* Upload parent_count so splice mutations know how many parents
       * are available in this batch. */
      CUCHECK(cuMemcpyHtoDAsync((CUdeviceptr)ctx->d_parent_count,
                                 &ctx->parent_count, sizeof(u32), s));

      /* Dictionary sync: upload extras/a_extras to device when counts change.
       *
       * Regular extras (user dict, -x flag) and auto-extras (auto-discovered
       * tokens) are packed contiguously in the device-global arrays:
       *   slots [0, extras_cnt)         → regular extras
       *   slots [extras_cnt, extras_cnt+a_extras_cnt) → auto-extras
       *
       * We check if either count changed since the last sync. Since extras
       * are typically small and infrequently updated (a_extras stabilize
       * after the first few minutes), this sync path is cold. */
      if (afl->extras_cnt != ctx->last_extras_cnt ||
          afl->a_extras_cnt != ctx->last_a_extras_cnt) {

        u32 n_extras = afl->extras_cnt < COQUI_MAX_EXTRAS ?
                       afl->extras_cnt : COQUI_MAX_EXTRAS;
        u32 n_a_extras = afl->a_extras_cnt;
        u32 total_slots = n_extras + n_a_extras;
        if (total_slots > COQUI_MAX_EXTRAS) {
          n_a_extras = COQUI_MAX_EXTRAS - n_extras;
          total_slots = COQUI_MAX_EXTRAS;
        }

        /* Pack extras into staging buffers. The data buffer is 128KB which
         * is too large for the stack; heap-allocate it. Offsets/lens are
         * 16KB each — acceptable on the stack. */
        u32 h_offsets[COQUI_MAX_EXTRAS];
        u32 h_lens[COQUI_MAX_EXTRAS];
        u8 *h_data = (u8 *)malloc(COQUI_MAX_EXTRAS_BYTES);
        if (!h_data) { PFATAL("malloc extras staging"); }
        u32 data_offset = 0;

        /* Regular extras */
        for (u32 i = 0; i < n_extras; i++) {
          u32 elen = afl->extras[i].len;
          if (data_offset + elen > COQUI_MAX_EXTRAS_BYTES) {
            n_extras = i;  /* truncate */
            total_slots = n_extras + n_a_extras;
            break;
          }
          h_offsets[i] = data_offset;
          h_lens[i] = elen;
          memcpy(h_data + data_offset, afl->extras[i].data, elen);
          data_offset += elen;
        }

        /* Auto-extras (appended after regular extras) */
        for (u32 i = 0; i < n_a_extras; i++) {
          u32 elen = afl->a_extras[i].len;
          if (data_offset + elen > COQUI_MAX_EXTRAS_BYTES) {
            n_a_extras = i;  /* truncate */
            total_slots = n_extras + n_a_extras;
            break;
          }
          h_offsets[n_extras + i] = data_offset;
          h_lens[n_extras + i] = elen;
          memcpy(h_data + data_offset, afl->a_extras[i].data, elen);
          data_offset += elen;
        }

        /* Upload to device (synchronous — dict changes are rare). */
        if (data_offset > 0) {
          CUCHECK(cuMemcpyHtoDAsync((CUdeviceptr)ctx->d_extras_data,
                                     h_data, data_offset, s));
        }
        if (total_slots > 0) {
          CUCHECK(cuMemcpyHtoDAsync((CUdeviceptr)ctx->d_extras_offsets,
                                     h_offsets, total_slots * sizeof(u32), s));
          CUCHECK(cuMemcpyHtoDAsync((CUdeviceptr)ctx->d_extras_lens,
                                     h_lens, total_slots * sizeof(u32), s));
        }
        CUCHECK(cuMemcpyHtoDAsync((CUdeviceptr)ctx->d_extras_cnt,
                                   &n_extras, sizeof(u32), s));
        CUCHECK(cuMemcpyHtoDAsync((CUdeviceptr)ctx->d_a_extras_cnt,
                                   &n_a_extras, sizeof(u32), s));

        free(h_data);

        ctx->last_extras_cnt = afl->extras_cnt;
        ctx->last_a_extras_cnt = afl->a_extras_cnt;
      }
    }
  }

  unsigned long long _htod_t1;
  STAMP_US(_htod_t1);
  ctx->t_htod_us += (_htod_t1 - _htod_t0);

  /* Launch */
  unsigned long long _launch_t0;
  STAMP_US(_launch_t0);
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
  unsigned long long _launch_t1;
  STAMP_US(_launch_t1);
  ctx->t_launch_us += (_launch_t1 - _launch_t0);

  /* D->H */
  unsigned long long _dtoh_t0;
  STAMP_US(_dtoh_t0);
  CUCHECK(cuMemcpyDtoHAsync(b->h_novelty, (CUdeviceptr)b->d_novelty,
                             ctx->batch_size / 8, s));
  CUCHECK(cuMemcpyDtoHAsync(b->h_status, (CUdeviceptr)b->d_status,
                             ctx->batch_size * sizeof(coqui_status_t), s));

  /* GPU-mutate novel-export (Change 2: sparse D2H).
   *
   * OLD: full D2H of n_inputs * max_input_size (~128 MB) + all lens.
   * NEW: D2H only the lens array here (128 KB). The actual mutated bytes
   * are fetched per-novel-thread in coqui_await_and_process AFTER the
   * kernel completes, using synchronous cuMemcpyDtoH for only the novel
   * threads (~0.1% of batch). This reduces D2H from ~128 MB to ~10 KB.
   *
   * The lens array must be read back in full because we need it for:
   *   (a) the trap_reason scan (needs lens to know valid inputs)
   *   (b) the crash-verify loop (needs lens for non-novel crashed inputs)
   * Both run before we know which threads are novel. */
  if (ctx->gpu_mutate_active) {
    CUCHECK(cuMemcpyDtoHAsync(b->h_input_lens, (CUdeviceptr)b->d_input_lens,
                               ctx->batch_size * sizeof(u32), s));
  }

  CUCHECK(cuEventRecord((CUevent)b->completion_event, s));
  ctx->launch_count++;
  unsigned long long _dtoh_t1;
  STAMP_US(_dtoh_t1);
  ctx->t_dtoh_us += (_dtoh_t1 - _dtoh_t0);

  /* Close out submit-phase timer now that all driver calls for this batch
   * have returned. */
  {
    unsigned long long _submit_t1_us;
    STAMP_US(_submit_t1_us);
    ctx->t_submit_us += (_submit_t1_us - _submit_t0_us);
  }
#undef STAMP_US

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
            ? (double)ctx->crash_dedup_hits_window / (double)dl
            : 0.0;
        double avg_verify_per_batch = dl > 0
            ? (double)ctx->crash_verify_calls_window / (double)dl
            : 0.0;
        /* Per-batch phase averages over this window. submit is driver-API
         * time (all async), await is the GPU wait in the poll loop, verify
         * is CPU-side novelty+crash loops after kernel completion. Summing
         * submit+await+verify approximates total host wall time per batch;
         * the gap to 1/batch-rate is AFL's outer stage overhead. */
        double avg_submit_us_batch = dl > 0
            ? (double)ctx->t_submit_us / (double)dl : 0.0;
        double avg_htod_us_batch = dl > 0
            ? (double)ctx->t_htod_us / (double)dl : 0.0;
        double avg_launch_us_batch = dl > 0
            ? (double)ctx->t_launch_us / (double)dl : 0.0;
        double avg_dtoh_us_batch = dl > 0
            ? (double)ctx->t_dtoh_us / (double)dl : 0.0;
        double avg_await_us_batch  = dl > 0
            ? (double)ctx->t_await_us  / (double)dl : 0.0;
        double avg_verify_us_batch = dl > 0
            ? (double)ctx->t_verify_us / (double)dl : 0.0;
        /* Wall clock per batch from the batch-rate itself; residual is
         * "mutation + AFL overhead" (not inside any measured phase). */
        double wall_per_batch_us = dl > 0
            ? (double)elapsed_us / (double)dl : 0.0;
        double avg_mut_other_us_batch =
            wall_per_batch_us -
            (avg_submit_us_batch + avg_await_us_batch + avg_verify_us_batch);
        if (avg_mut_other_us_batch < 0) avg_mut_other_us_batch = 0;

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

        double avg_persist_per_batch = dl > 0
            ? (double)ctx->crash_sig_persist_hits_window / (double)dl : 0.0;
        fprintf(stderr,
                "[coqui-rate] %.1f batches/s, %llu submits/s "
                "(avg %.0f inputs/batch, %llu slow-skipped, "
                "crash_dedup_hits=%llu avg=%.1f/batch, "
                "crash_verify_calls=%llu avg=%.1f/batch, "
                "persist_dedup=%llu avg=%.1f/batch cap=%u/%u, "
                "thread_budget_traps=%llu, "
                "host_us/batch: wall=%.0f mut_other=%.0f submit=%.0f "
                "[htod=%.0f launch=%.0f dtoh=%.0f] await=%.0f verify=%.0f, "
                "kern: init=%.0f%% exec=%.0f%% classify=%.0f%% virgin=%.0f%%)\n",
                (double)dl * 1e6 / (double)elapsed_us,
                (unsigned long long)(ds * 1000000ULL / elapsed_us),
                dl > 0 ? (double)ds / (double)dl : 0.0,
                (unsigned long long)ctx->slow_skipped,
                (unsigned long long)ctx->crash_dedup_hits,
                avg_dedup_per_batch,
                (unsigned long long)ctx->crash_verify_calls,
                avg_verify_per_batch,
                (unsigned long long)ctx->crash_sig_persistent_hits,
                avg_persist_per_batch,
                ctx->crash_sig_seen_count,
                ctx->crash_sig_seen_cap,
                (unsigned long long)ctx->thread_budget_inputs_found,
                wall_per_batch_us,
                avg_mut_other_us_batch,
                avg_submit_us_batch,
                avg_htod_us_batch,
                avg_launch_us_batch,
                avg_dtoh_us_batch,
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
      ctx->t_htod_us = 0;
      ctx->t_launch_us = 0;
      ctx->t_dtoh_us = 0;
      ctx->t_await_us  = 0;
      ctx->t_verify_us = 0;
      /* Reset per-window dedup counters (parallel to the timing accumulators
       * above).  These are NOT preserved across force-reset (intentional:
       * coqui_init zeros them fresh so the first post-reset window only shows
       * what happened AFTER the reset, not the lifetime accumulation). */
      ctx->crash_dedup_hits_window       = 0;
      ctx->crash_verify_calls_window     = 0;
      ctx->crash_sig_persist_hits_window = 0;
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
  /* Hard ceiling: if the kernel hasn't finished GRACE_US after culling,
   * assume an infinite loop and force-reset (kills the kernel). Reset costs
   * ~1s, which is cheaper than waiting longer for a pathological kernel.
   *
   * History:
   *   iter1   — 5s  grace (initial conservative default)
   *   iter17  — 1s  grace (cjson pathological kernels hang 5-22+ s; 5s
   *                        grace almost never recovered legit-slow batches)
   *   iter18  — 250ms grace. Image-format targets (libjpeg-turbo, stb_image)
   *             have *unbounded* slow paths from adversarial Huffman /
   *             marker-length inputs — the kernel never finishes naturally,
   *             so the grace is pure wait. Healthy batches finish before
   *             cull_deadline anyway and never touch this grace. The 250ms
   *             window covers the rare "kernel near done at cull moment"
   *             case while keeping recovery latency low.
   *
   * Tunable via AFL_COQUI_TIMEOUT_GRACE_US (default 250000). */
  unsigned long long grace_us =
      getenv_u64("AFL_COQUI_TIMEOUT_GRACE_US", 250000ULL);
  unsigned long long hard_deadline_us = cull_deadline_us + grace_us;

  int culled = 0;
  unsigned long long wait_end_us = 0;  /* set when cuStreamQuery reports success */
  while (1) {
    /* Honor Ctrl-C / SIGTERM: if afl->stop_soon is set, abandon this batch
     * and let the caller propagate the signal up to coqui_shutdown. The
     * in-flight kernel may still be running on the GPU; we don't wait for
     * it. coqui_shutdown drains streams with a similar signal-aware bound,
     * so process exit happens within tens of ms instead of blocking for
     * the full cull/hard-deadline cycle (3-4 s). */
    if (afl->stop_soon) return 1;

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

  /* Sparse D2H for GPU-mutated inputs (Change 2).
   *
   * When GPU mutation is active, the mutated bytes live only on the device
   * (the host never sent them -- it sent the compact parent table, and the
   * GPU mutation wrapper populated per-thread output slots). We must D2H
   * the mutated bytes for every thread whose input will be consumed by the
   * host: novel threads, trapped threads, and crashed threads.
   *
   * Scan novelty bitmap + status to identify which threads need bytes,
   * then issue per-thread synchronous D2H copies. At ~0.1% novelty rate,
   * this fetches ~33 threads * ~100 bytes = ~3 KB instead of 128 MB.
   *
   * The synchronous cuMemcpyDtoH calls complete immediately (no need for
   * additional sync). */
  if (ctx->gpu_mutate_active) {
    for (u32 i = 0; i < ctx->batch_size; i++) {
      if (b->h_input_lens[i] == 0) continue;

      /* Check if this thread's bytes are needed:
       * - Novel (novelty bitmap bit set)
       * - Trapped (trap_reason != NONE -- OOM/stack/budget)
       * - Crashed (asan_error || ubsan_fatal || signal) */
      u8 novel = (b->h_novelty[i / 8] >> (i % 8)) & 1;
      u8 trapped = b->h_status[i].trap_reason != COQUI_TRAP_NONE;
      u8 crashed = b->h_status[i].asan_error ||
                   b->h_status[i].ubsan_fatal ||
                   b->h_status[i].signal;
      if (!novel && !trapped && !crashed) continue;

      /* D2H this thread's mutated bytes. */
      u32 off = b->h_offsets[i];
      u32 len = b->h_input_lens[i];
      CUCHECK(cuMemcpyDtoH(b->h_input_bytes + off,
                            (CUdeviceptr)(b->d_input_bytes + off), len));
    }
  }

  /* Scan for inputs that caused the device runtime to call
   * __coqui_trap_with_reason() (OOM or stack overflow). These did NOT
   * record a crash or novelty on the GPU because they exited too
   * early; rerun them on the CPU forkserver where the per-thread
   * memory constraint doesn't apply. This is the single place where
   * the correctness invariant "every input either completes on the
   * GPU with a recorded outcome OR is rerun on the CPU" is enforced
   * -- so the scan runs before the novelty + crash loops so dedup
   * (which is GPU-outcome based) can't suppress an OOM rerun. */
  {
    u64 oom_this_batch = 0;
    u64 oom_reruns_this_batch = 0;
    u64 stack_ovf_this_batch = 0;
    u64 budget_this_batch = 0;
    coqui_status_t *hs_trap = b->h_status;
    for (u32 i = 0; i < ctx->batch_size; i++) {
      if (b->h_input_lens[i] == 0) continue;
      u8 tr = hs_trap[i].trap_reason;
      if (tr == COQUI_TRAP_NONE) continue;

      u8 *input = b->h_input_bytes + b->h_offsets[i];
      u32 len = b->h_input_lens[i];

      if (tr == COQUI_TRAP_OOM) {
        oom_this_batch++;
        u8 fault = rerun_gpu_failed_input(afl, input, len);
        if (fault == FSRV_RUN_CRASH) ctx->cpu_rerun_crashes++;
        oom_reruns_this_batch++;
      } else if (tr == COQUI_TRAP_STACK_OVERFLOW) {
        stack_ovf_this_batch++;
        u8 fault = rerun_gpu_failed_input(afl, input, len);
        if (fault == FSRV_RUN_CRASH) ctx->cpu_rerun_crashes++;
        oom_reruns_this_batch++;
      } else if (tr == COQUI_TRAP_THREAD_BUDGET_EXHAUSTED) {
        /* This thread exceeded AFL_COQUI_THREAD_BUDGET_US on the GPU.
         * NOT a crash — the input simply ran past the per-thread time
         * cap. Rerun on the CPU forkserver where the budget doesn't
         * apply; the CPU will either complete normally (cheap) or
         * also time out (FSRV_RUN_TMOUT — recorded normally). */
        budget_this_batch++;
        u8 fault = rerun_gpu_failed_input(afl, input, len);
        if (fault == FSRV_RUN_CRASH) ctx->cpu_rerun_crashes++;
        oom_reruns_this_batch++;
      }
    }
    ctx->oom_inputs_found += oom_this_batch;
    ctx->stack_overflow_inputs_found += stack_ovf_this_batch;
    ctx->oom_reruns_completed += oom_reruns_this_batch;
    ctx->thread_budget_inputs_found += budget_this_batch;
    /* Debug-build correctness assertion: every trap_reason entry that
     * isn't len==0 must have been rerun. Intentionally not a FATAL in
     * release builds — a logged mismatch is preferable to killing a
     * long fuzz session over a counting bug. */
    if (oom_this_batch + stack_ovf_this_batch + budget_this_batch
        != oom_reruns_this_batch) {
      WARNF("coqui: OOM counter mismatch: %llu found / %llu reran",
            (unsigned long long)(oom_this_batch + stack_ovf_this_batch
                                  + budget_this_batch),
            (unsigned long long)oom_reruns_this_batch);
    }
  }

  /* Oracle-mode line-trace readback.
   * After a healthy batch, copy thread-0's count + trace stripe back to
   * host scratch and print. Thread 0 is sufficient for end-to-end pass
   * validation — every thread runs the same kernel against its own input,
   * so traces differ only by input-driven branches; the user can choose
   * a deterministic seed input (one that flows down a known path on the
   * CPU build) and compare against an expected trace.
   *
   * Comparison-against-expected and full-batch readback are intentionally
   * NOT implemented here: the user can tail the stderr stream while
   * running a known-input through afl-fuzz, capture the recorded
   * sequence, and diff against the host CPU build's expected trace
   * separately. Wiring a flexible seed/expected-trace harness into core
   * afl-fuzz is a separate piece of work; the readback path being
   * functional is enough to validate the device side. */
  if (ctx->oracle_enabled && ctx->h_trace_t0_buf && b->h_input_lens[0] > 0) {
    /* DtoH thread-0 count first, then conditionally the buffer.
     * Synchronous DtoH so the print below sees the values. */
    unsigned int t0_count = 0;
    CUCHECK(cuMemcpyDtoH(&t0_count, (CUdeviceptr)ctx->d_trace_count,
                          sizeof(unsigned int)));
    if (t0_count > COQUI_TRACE_MAX_ENTRIES)
      t0_count = COQUI_TRACE_MAX_ENTRIES;
    ctx->h_trace_t0_count = t0_count;

    if (t0_count > 0) {
      unsigned long long bytes =
          (unsigned long long)t0_count * sizeof(unsigned int);
      CUCHECK(cuMemcpyDtoH(ctx->h_trace_t0_buf,
                            (CUdeviceptr)ctx->d_trace_buffer, bytes));

      /* Print the recorded sequence to stderr. Capped at 64 entries to
       * keep logs readable; full buffer remains in ctx->h_trace_t0_buf
       * for any callers that hook in. Overflow signaled by the count
       * pinning at COQUI_TRACE_MAX_ENTRIES. */
      u32 cap = t0_count < 64 ? t0_count : 64;
      fprintf(stderr, "[coqui-oracle] batch=%llu tid=0 trace_count=%u%s "
                      "(first %u): ",
              (unsigned long long)ctx->launch_count, t0_count,
              t0_count >= COQUI_TRACE_MAX_ENTRIES ? " (OVERFLOW)" : "",
              cap);
      for (u32 j = 0; j < cap; j++)
        fprintf(stderr, "%u ", ctx->h_trace_t0_buf[j]);
      fprintf(stderr, "\n");
    } else {
      fprintf(stderr, "[coqui-oracle] batch=%llu tid=0 trace_count=0 "
                      "(no traced lines reached)\n",
              (unsigned long long)ctx->launch_count);
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

      /* Skip inputs that already ran through the CPU forkserver via
       * the trap_reason path above — running them again would waste
       * cycles and double-save their coverage. */
      if (b->h_status[i].trap_reason != COQUI_TRAP_NONE) continue;

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
    /* OOM / stack-overflow reruns handled by the trap_reason scan. */
    if (hs[i].trap_reason != COQUI_TRAP_NONE) continue;

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
      ctx->crash_dedup_hits_window++;
      continue;
    }

    /* Survived intra-batch dedup — now check the persistent cross-batch set.
     * Open-addressing linear probe on a power-of-2-sized table. `used[]`
     * tracks occupancy so sig==0 is a valid key. On full table (no empty
     * slot after probing the whole capacity), return "not seen" so we still
     * verify — a safe fallback that degrades gracefully from fully-dedup'd
     * to fully-verified as the set saturates. */
    u32 pcap   = ctx->crash_sig_seen_cap;
    u32 pmask  = pcap - 1u;
    u32 pstart = sig & pmask;
    int p_seen = 0;
    int p_inserted = 0;
    for (u32 probe = 0; probe < pcap; probe++) {
      u32 pslot = (pstart + probe) & pmask;
      if (!ctx->crash_sig_seen_used[pslot]) {
        ctx->crash_sig_seen[pslot]      = sig;
        ctx->crash_sig_seen_used[pslot] = 1;
        ctx->crash_sig_seen_count++;
        p_inserted = 1;
        break;
      }
      if (ctx->crash_sig_seen[pslot] == sig) { p_seen = 1; break; }
    }
    (void)p_inserted;  /* telemetry-only hook point */

    if (p_seen) {
      ctx->crash_sig_persistent_hits++;
      ctx->crash_sig_persist_hits_window++;
      continue;
    }

    ctx->crash_verify_calls++;
    ctx->crash_verify_calls_window++;
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

/* Periodic fuzzer_stats writer for coqui-mode.
 *
 * AFL's normal show_stats / write_stats_file path is gated on fuzz_one()
 * completing one mutation pass. On slow-kernel coqui targets (cmark at
 * BATCH=8192 has ~30s per-batch wall) a single fuzz_one round can take
 * minutes, leaving fuzzer_stats stuck at execs_done=100, run_time=0 even
 * though the GPU is pumping ~1M submits/s. Bench scrapers reading that
 * file see no progress and misdiagnose hangs.
 *
 * Tick the file write directly from the submit-input pump every ~10s of
 * wall time. write_stats_file(afl, 0, 0, 0, 0) is the same snapshot call
 * already used at shutdown in src/afl-fuzz.c:3420 — the zero args fall
 * back to last-cached bitmap_cvg/stability/eps. The check fires once per
 * batch flip (after the previous batch's coverage has been merged), so
 * cost is ~120 cheap branches/sec on fast targets, 1 file write per 10s. */
static u64 coqui_last_stats_tick_ms = 0;

u8 coqui_submit_input(afl_state_t *afl, u8 *buf, u32 len) {
  coqui_ctx_t *ctx = afl->coqui;
  coqui_batch_t *b = ctx->pending;

  ctx->total_submits++;
  /* Count each submission as one exec -- the corresponding increment in
   * afl_fsrv_run_target is suppressed in coqui_mode to avoid double-counting
   * GPU-flagged inputs that also run through CPU verification. */
  afl->fsrv.total_execs++;

  if (len > ctx->max_input_size) {
    ctx->oversized_count++;
    return 0;
  }

  /* Fixed-size slots: slot i starts at i * max_input_size. */
  if (b->n_inputs == ctx->batch_size) {
    /* Launch the full batch async */
    coqui_launch_batch(afl, b);

    /* Flip ping-pong */
    coqui_batch_t *tmp = ctx->pending;
    ctx->pending = ctx->executing;
    ctx->executing = tmp;

    b = ctx->pending;

    /* If the new pending has in-flight work from a previous flip, drain it
       before reusing */
    if (b->n_inputs > 0) {
      if (coqui_await_and_process(afl, b) == 1) {
        /* Context was reset under us; ctx/b are stale. Drop this input --
         * the context is now functional but the old batch pointers are
         * freed. Caller (havoc loop) will retry on the next iteration. */
        return 0;
      }
      b->n_inputs = 0;
      b->bytes_used = 0;
      if (ctx->gpu_mutate_enabled) {
        ctx->parent_count = 0;
        ctx->parent_bytes_used = 0;
      }
    }

    /* Periodic fuzzer_stats refresh (see comment above
     * coqui_last_stats_tick_ms). Hook fires at most once per batch flip,
     * just after coverage from the prior batch has been merged. */
    if (likely(!afl->non_instrumented_mode)) {
      u64 now_ms = get_cur_time();
      if (now_ms - coqui_last_stats_tick_ms >= 10000) {
        write_stats_file(afl, 0, 0, 0, 0);
        coqui_last_stats_tick_ms = now_ms;
      }
    }
  }

  /* Power-schedule mode: when gpu_mutate_active, pack the parent into
   * the compact parent table (one copy) and assign n_threads to it via
   * the parent index array. The GPU mutation wrapper reads from the
   * parent table, copies to the per-thread output slot, and mutates.
   * This replaces the old broadcast pattern (128 copies of same data). */
  if (ctx->gpu_mutate_active && afl->stage_max > 0) {
    u32 n_threads = ((afl->stage_max + 31) / 32) * 32;
    /* If n_threads would exceed remaining capacity, cap to what fits
     * (rounded down to multiple of 32 for warp alignment). */
    u32 remaining = ctx->batch_size - b->n_inputs;
    if (n_threads > remaining) {
      n_threads = (remaining / 32) * 32;
    }
    if (n_threads == 0) {
      /* No room even for 32 threads -- trigger a batch launch. The caller
       * (havoc loop) will re-enter with a fresh batch. We treat this as
       * a full batch by filling the single slot normally; the auto-launch
       * at batch_size handles the rest. */
      goto single_slot;
    }

    /* Pack parent into the compact parent table (one copy). */
    u32 pidx = ctx->parent_count;
    u32 poff = ctx->parent_bytes_used;
    memcpy(ctx->h_parent_bytes + poff, buf, len);
    ctx->h_parent_offsets[pidx] = poff;
    ctx->h_parent_lens[pidx] = len;
    ctx->parent_count++;
    ctx->parent_bytes_used = poff + len;

    /* Assign all n_threads to this parent and fill their slot metadata.
     * Per-thread input slots are NOT filled with data -- the GPU mutation
     * wrapper copies from the parent table. But we still need offsets and
     * lens set so the harness/post-kernel reads work correctly. */
    for (u32 t = 0; t < n_threads; t++) {
      u32 slot = b->n_inputs;
      u32 off = slot * ctx->max_input_size;
      b->h_offsets[slot] = off;
      b->h_input_lens[slot] = len;
      ctx->h_parent_idx[slot] = pidx;
      b->n_inputs++;
      b->bytes_used = off + len;
    }
    /* Count all threads as execs (the GPU will run each). We already
     * counted 1 in total_execs above; add the remaining n_threads-1. */
    afl->fsrv.total_execs += (n_threads - 1);
    ctx->total_submits += (n_threads - 1);
    return 0;
  }

single_slot:;
  u32 off = b->n_inputs * ctx->max_input_size;
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
  /* Reset parent table state for the next stage. */
  if (ctx->gpu_mutate_enabled) {
    ctx->parent_count = 0;
    ctx->parent_bytes_used = 0;
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

  /* Preserve persistent crash-sig dedup across CUDA context reset. The set
   * is pure host-side RAM with no CUDA bindings; rebuilding it from scratch
   * after every force-reset would blow away many minutes of dedup learning
   * (this is a recovery path triggered by pathological GPU kernels, which is
   * exactly the moment we most need dedup to suppress repeat crash verifies).
   * Pointers are captured here and swapped back in after coqui_init re-allocs. */
  u32 *saved_seen      = ctx->crash_sig_seen;
  u8  *saved_used      = ctx->crash_sig_seen_used;
  u32  saved_cap       = ctx->crash_sig_seen_cap;
  u32  saved_count     = ctx->crash_sig_seen_count;
  u64  saved_persist   = ctx->crash_sig_persistent_hits;

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
  /* Parent table host pinned buffers (GPU-mutate). Device-side parent
   * buffers are freed by cuCtxDestroy above. */
  if (ctx->h_parent_bytes)   cuMemFreeHost(ctx->h_parent_bytes);
  if (ctx->h_parent_offsets) cuMemFreeHost(ctx->h_parent_offsets);
  if (ctx->h_parent_lens)    cuMemFreeHost(ctx->h_parent_lens);
  if (ctx->h_parent_idx)     cuMemFreeHost(ctx->h_parent_idx);

  ck_free(ctx);
  afl->coqui = NULL;

  coqui_init(afl, cubin_path);

  /* Swap the freshly-allocated (and zero-initialized) sig set for the one
   * carried across the reset.  Free the fresh one to avoid leaking 5 MB. */
  ctx = afl->coqui;
  if (ctx->crash_sig_seen)      ck_free(ctx->crash_sig_seen);
  if (ctx->crash_sig_seen_used) ck_free(ctx->crash_sig_seen_used);
  ctx->crash_sig_seen            = saved_seen;
  ctx->crash_sig_seen_used       = saved_used;
  ctx->crash_sig_seen_cap        = saved_cap;
  ctx->crash_sig_seen_count      = saved_count;
  ctx->crash_sig_persistent_hits = saved_persist;
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

/* Bounded stream drain: poll cuStreamQuery for up to drain_max_us, returning
 * early if the stream completes or if afl->stop_soon is asserted (Ctrl-C).
 * Falls back to abandoning the stream — the CUDA context teardown that
 * follows shutdown reclaims the kernel/memory, but does NOT guarantee the
 * in-flight kernel actually finishes. Acceptable for shutdown: we're tearing
 * down anyway, no observers care about partial-batch output. */
static void coqui_drain_stream_bounded(afl_state_t *afl, CUstream s,
                                       unsigned long long drain_max_us) {
  if (!s) return;
  struct timeval tv;
  gettimeofday(&tv, NULL);
  unsigned long long start_us =
    ((unsigned long long)tv.tv_sec * 1000000ULL) + tv.tv_usec;
  unsigned long long deadline_us = start_us + drain_max_us;
  while (1) {
    if (afl->stop_soon) return;
    CUresult r = cuStreamQuery(s);
    if (r == CUDA_SUCCESS) return;
    if (r != CUDA_ERROR_NOT_READY) return;  /* genuine error or stream gone */
    gettimeofday(&tv, NULL);
    unsigned long long now =
      ((unsigned long long)tv.tv_sec * 1000000ULL) + tv.tv_usec;
    if (now >= deadline_us) return;
    usleep(1000);
  }
}

void coqui_shutdown(afl_state_t *afl) {
  if (!afl->coqui) return;
  coqui_ctx_t *ctx = afl->coqui;

  /* Drain streams with a 2-second bound and stop_soon awareness. On clean
   * shutdown the streams typically complete within tens of ms; on Ctrl-C
   * we exit immediately. The full unbounded cuStreamSynchronize was a
   * SIGTERM trap when a pathological kernel was in flight at the moment
   * the user expected fast exit. */
  coqui_drain_stream_bounded(afl, (CUstream)ctx->stream_a, 2000000ULL);
  coqui_drain_stream_bounded(afl, (CUstream)ctx->stream_b, 2000000ULL);

  /* Free ping-pong */
  free_batch_half_cuda(&ctx->ping);
  free_batch_half_cuda(&ctx->pong);

  /* Free persistent device buffers */
  if (ctx->d_global_statics_pool)
    cuMemFree((CUdeviceptr)ctx->d_global_statics_pool);
  if (ctx->d_slab_pool)
    cuMemFree((CUdeviceptr)ctx->d_slab_pool);
  if (ctx->d_slab_shadow)
    cuMemFree((CUdeviceptr)ctx->d_slab_shadow);
  if (ctx->d_slab_next)
    cuMemFree((CUdeviceptr)ctx->d_slab_next);
  /* Free oracle-mode line-trace buffers. */
  if (ctx->d_trace_buffer)
    cuMemFree((CUdeviceptr)ctx->d_trace_buffer);
  if (ctx->d_trace_count)
    cuMemFree((CUdeviceptr)ctx->d_trace_count);
  if (ctx->h_trace_t0_buf)
    ck_free(ctx->h_trace_t0_buf);
  /* Free parent table buffers (GPU-mutate). */
  if (ctx->d_parent_bytes)
    cuMemFree((CUdeviceptr)ctx->d_parent_bytes);
  if (ctx->d_parent_offsets)
    cuMemFree((CUdeviceptr)ctx->d_parent_offsets);
  if (ctx->d_parent_lens)
    cuMemFree((CUdeviceptr)ctx->d_parent_lens);
  if (ctx->d_parent_idx)
    cuMemFree((CUdeviceptr)ctx->d_parent_idx);
  if (ctx->h_parent_bytes)
    cuMemFreeHost(ctx->h_parent_bytes);
  if (ctx->h_parent_offsets)
    cuMemFreeHost(ctx->h_parent_offsets);
  if (ctx->h_parent_lens)
    cuMemFreeHost(ctx->h_parent_lens);
  if (ctx->h_parent_idx)
    cuMemFreeHost(ctx->h_parent_idx);

  /* Destroy streams, unload module, destroy context */
  if (ctx->stream_a) cuStreamDestroy((CUstream)ctx->stream_a);
  if (ctx->stream_b) cuStreamDestroy((CUstream)ctx->stream_b);
  if (ctx->cu_module) cuModuleUnload((CUmodule)ctx->cu_module);
  if (ctx->cu_ctx) cuCtxDestroy((CUcontext)ctx->cu_ctx);

  if (ctx->crash_sig_seen)      ck_free(ctx->crash_sig_seen);
  if (ctx->crash_sig_seen_used) ck_free(ctx->crash_sig_seen_used);

  ck_free(ctx);
  afl->coqui = NULL;
}
