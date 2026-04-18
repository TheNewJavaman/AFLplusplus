/*
 * afl-fuzz-coqui.c --- cuAFL coqui_mode stub implementation.
 *
 * Hollow stub behind the coqui_mode contract. Lets cuAFL compile and run
 * end-to-end under --coqui without a real GPU backend. Real CUDA-backed
 * implementation drops in behind the same contract in a later phase.
 *
 * Spec: docs/superpowers/specs/2026-04-18-cuafl-gpu-backend-design.md
 */

#include "afl-fuzz.h"
#include "afl-fuzz-coqui.h"
#include "forkserver.h"

#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------------
 * Internal helpers
 * ------------------------------------------------------------------------*/

static void alloc_batch_half(coqui_batch_t *b, u32 batch_size, u32 byte_budget) {

  b->h_input_bytes = (u8  *)ck_alloc(byte_budget);
  b->h_offsets     = (u32 *)ck_alloc(batch_size * sizeof(u32));
  b->h_input_lens  = (u32 *)ck_alloc(batch_size * sizeof(u32));
  b->h_novelty     = (u8  *)ck_alloc((batch_size + 7) / 8);
  b->h_status      = (coqui_status_t *)ck_alloc(batch_size * sizeof(coqui_status_t));

  b->d_input_bytes = NULL;
  b->d_offsets     = NULL;
  b->d_input_lens  = NULL;
  b->d_coverage    = NULL;
  b->d_novelty     = NULL;
  b->d_virgin      = NULL;
  b->d_status      = NULL;

  b->n_inputs     = 0;
  b->bytes_used   = 0;

}

static void free_batch_half(coqui_batch_t *b) {

  ck_free(b->h_input_bytes);
  ck_free(b->h_offsets);
  ck_free(b->h_input_lens);
  ck_free(b->h_novelty);
  ck_free(b->h_status);

  memset(b, 0, sizeof(*b));

}

/* ------------------------------------------------------------------------
 * API implementation (stub)
 * ------------------------------------------------------------------------*/

void coqui_init(afl_state_t *afl, const char *cubin_path) {

  (void)cubin_path;  /* stub: stashed in afl->coqui_cubin_path but not loaded */

  coqui_ctx_t *ctx = (coqui_ctx_t *)ck_alloc(sizeof(coqui_ctx_t));

  ctx->batch_size     = afl->gpu_batch_size
                            ? afl->gpu_batch_size
                            : COQUI_DEFAULT_BATCH_SIZE;
  ctx->max_input_size = afl->max_length
                            ? afl->max_length
                            : COQUI_MAX_INPUT_DEFAULT;
  ctx->map_size       = afl->fsrv.map_size;

  /* Default byte budget: batch_size * max_input_size / 4 (assume 25% fill),
     with a floor of max_input_size * 256 so tiny batches still work. */
  ctx->byte_budget = ctx->batch_size * ctx->max_input_size / 4;
  if (ctx->byte_budget < ctx->max_input_size * 256) {
    ctx->byte_budget = ctx->max_input_size * 256;
  }

  alloc_batch_half(&ctx->ping, ctx->batch_size, ctx->byte_budget);
  alloc_batch_half(&ctx->pong, ctx->batch_size, ctx->byte_budget);

  ctx->pending   = &ctx->ping;
  ctx->executing = &ctx->pong;

  ctx->oversized_count = 0;
  ctx->launch_count    = 0;

  afl->coqui = ctx;

  OKF("coqui_mode stub initialized (batch_size=%u, max_input=%u, budget=%u)",
      ctx->batch_size, ctx->max_input_size, ctx->byte_budget);

}

u8 coqui_submit_input(afl_state_t *afl, u8 *buf, u32 len) {

  coqui_ctx_t   *ctx = afl->coqui;
  coqui_batch_t *b   = ctx->pending;

  /* Reject inputs that alone exceed the byte budget. */
  if (len > ctx->byte_budget) {
    ctx->oversized_count++;
    return 0;
  }

  /* 8-byte-align the next write cursor. */
  u32 off = (b->bytes_used + 7) & ~7u;

  if (off + len > ctx->byte_budget || b->n_inputs == ctx->batch_size) {

    /* Stub launch: silently discard this full batch and flip to the other
       ping-pong half. Real implementation: async cuLaunchKernel on d_*. */
    b->n_inputs   = 0;
    b->bytes_used = 0;

    ctx->launch_count++;

    /* Flip. */
    coqui_batch_t *tmp = ctx->pending;
    ctx->pending   = ctx->executing;
    ctx->executing = tmp;

    b   = ctx->pending;
    off = 0;

  }

  memcpy(b->h_input_bytes + off, buf, len);
  b->h_offsets[b->n_inputs]    = off;
  b->h_input_lens[b->n_inputs] = len;
  b->n_inputs++;
  b->bytes_used = off + len;

  return 0;

}

void coqui_flush_batch(afl_state_t *afl) {

  coqui_ctx_t   *ctx = afl->coqui;
  coqui_batch_t *b   = ctx->pending;

  /* Stub: discard pending + executing contents; no real launch. */
  b->n_inputs             = 0;
  b->bytes_used           = 0;
  ctx->executing->n_inputs   = 0;
  ctx->executing->bytes_used = 0;

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
