/*
   american fuzzy lop++ - divergence scheduler implementation
   ----------------------------------------------------------

   Written by Gabriel Pizarro

   Ordered-trace divergence detection for AFL++. Uses shared memory to
   communicate reference traces and bandit scores between fuzzer and target.
   The target's instrumentation callback (__sanitizer_cov_trace_pc_guard)
   compares each edge against the reference trace in real time and can
   terminate early (_exit(0)) when the bandit scheduler decides the
   divergence is uninteresting.

   See include/divergence.h for the design overview.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at:

     https://www.apache.org/licenses/LICENSE-2.0

 */

#include "afl-fuzz.h"
#include "divergence.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef USEMMAP
  #include <sys/shm.h>
#endif
#include <sys/mman.h>
#include <fcntl.h>

void div_init(afl_state_t *afl) {

  struct divergence_state *ds = &afl->div_state;
  memset(ds, 0, sizeof(*ds));

  ds->bandit_threshold = 0; /* Start with threshold 0 during warmup */

  /* Allocate divergence shared memory */
#ifdef USEMMAP

  snprintf(ds->shm_file_path, sizeof(ds->shm_file_path), "/afl_div_%d_%ld",
           getpid(), random());

  int fd = shm_open(ds->shm_file_path, O_CREAT | O_RDWR | O_EXCL, 0600);
  if (fd == -1) { PFATAL("div_init: shm_open() failed"); }

  if (ftruncate(fd, DIV_SHM_SIZE)) {

    PFATAL("div_init: ftruncate() failed");

  }

  ds->div_shm =
      mmap(0, DIV_SHM_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (ds->div_shm == MAP_FAILED) {

    close(fd);
    shm_unlink(ds->shm_file_path);
    PFATAL("div_init: mmap() failed");

  }

  ds->shm_id = fd;
  setenv(DIV_SHM_ENV_VAR, ds->shm_file_path, 1);

#else

  ds->shm_id = shmget(IPC_PRIVATE, DIV_SHM_SIZE, IPC_CREAT | IPC_EXCL | 0600);
  if (ds->shm_id < 0) { PFATAL("div_init: shmget() failed"); }

  ds->div_shm = shmat(ds->shm_id, NULL, 0);
  if (ds->div_shm == (void *)-1) {

    shmctl(ds->shm_id, IPC_RMID, NULL);
    PFATAL("div_init: shmat() failed");

  }

  /* Mark for auto-removal when all processes detach */
  shmctl(ds->shm_id, IPC_RMID, NULL);

  u8 *shm_str = alloc_printf("%d", ds->shm_id);
  setenv(DIV_SHM_ENV_VAR, shm_str, 1);
  ck_free(shm_str);

#endif

  memset(ds->div_shm, 0, DIV_SHM_SIZE);

  /* Initialize bandit scores directly in shared memory */
  memset(div_bandit_shm(ds), DIV_BANDIT_INITIAL, DIV_BANDIT_SIZE);

  OKF("Divergence scheduler initialized (trace-based, max_trace=%u, "
      "bandit_size=%u)",
      DIV_MAX_TRACE_LEN, DIV_BANDIT_SIZE);

}

void div_deinit(afl_state_t *afl) {

  struct divergence_state *ds = &afl->div_state;

  if (ds->div_shm) {

#ifdef USEMMAP
    munmap(ds->div_shm, DIV_SHM_SIZE);
    if (ds->shm_id >= 0) { close(ds->shm_id); }
    if (ds->shm_file_path[0]) { shm_unlink(ds->shm_file_path); }
#else
    shmdt(ds->div_shm);
#endif
    ds->div_shm = NULL;

  }

  unsetenv(DIV_SHM_ENV_VAR);

}

/* ---- Pre-execution setup ---- */

void div_prepare_execution(afl_state_t *afl) {

  struct divergence_state *ds = &afl->div_state;

  if (!ds->div_shm) return;

  struct div_shm_header *hdr = (struct div_shm_header *)ds->div_shm;

  /* Copy the current seed's reference trace into shared memory.
     Skip the memcpy when the seed hasn't changed — each seed gets thousands
     of mutations, so this avoids ~99.98% of copies. */
  struct queue_entry *q = afl->queue_cur;

  if (q && q->div_trace && q->div_trace_len > 0) {

    if (q->id != ds->last_seed_id || hdr->ref_len == 0) {

      u32 len = q->div_trace_len;
      if (len > DIV_MAX_TRACE_LEN) len = DIV_MAX_TRACE_LEN;
      hdr->ref_len = len;
      memcpy(div_ref_trace(ds), q->div_trace, len * sizeof(u32));
      ds->last_seed_id = q->id;

    }

  } else {

    /* No reference trace available — disable comparison for this exec */
    hdr->ref_len = 0;
    ds->last_seed_id = UINT32_MAX;

  }

  /* Set threshold */
  hdr->bandit_threshold = ds->warmup_done ? ds->bandit_threshold : 0;

  /* Reset per-execution counters */
  hdr->actual_idx = 0;
  hdr->diverged = 0;
  hdr->killed = 0;
  hdr->diverge_pos = 0;
  hdr->diverge_key = 0;

}

/* ---- Per-mutation post-execution ---- */

u8 div_post_execution(afl_state_t *afl) {

  struct divergence_state *ds = &afl->div_state;

  if (!ds->div_shm) return 0;

  struct div_shm_header *hdr = (struct div_shm_header *)ds->div_shm;

  ds->total_mutations++;

  /* Count edges from the actual trace length */
  u32 actual_len = hdr->actual_idx;
  if (actual_len > DIV_MAX_TRACE_LEN) actual_len = DIV_MAX_TRACE_LEN;
  ds->total_edges += actual_len;

  /* Manage warmup → active transition */
  if (!ds->warmup_done && ds->total_mutations >= DIV_WARMUP_EXECS) {

    ds->warmup_done = 1;
    ds->bandit_threshold = DIV_BANDIT_THRESHOLD_DEFAULT;
    OKF("DIV: Warmup complete after %llu mutations, threshold=%u",
        ds->total_mutations, ds->bandit_threshold);

  }

  /* No reference trace — can't determine divergence, pass through */
  if (hdr->ref_len == 0) {

    afl->div_last_diverged = 0;
    return 0;

  }

  if (!hdr->diverged) {

    /* No divergence: mutation followed the exact same path as the seed
       (or a prefix of it). Completely uninteresting. */
    ds->no_divergence_count++;
    afl->div_last_diverged = 0;
    return 1; /* Skip */

  }

  /* Mutation diverged from seed */
  ds->divergence_count++;

  if (hdr->killed) {

    /* Target was killed mid-execution by bandit in the callback */
    ds->divergence_kills++;
    afl->div_last_diverged = 0;

    /* Decay the bandit score further for this key */
    u32 key = hdr->diverge_key & (DIV_BANDIT_SIZE - 1);
    u8 *bandit = div_bandit_shm(ds);
    u32 score = bandit[key];
    bandit[key] =
        (u8)((score * DIV_BANDIT_DECAY_NUM) / DIV_BANDIT_DECAY_DEN);

    return 1; /* Skip */

  }

  /* Diverged and passed: allow this mutation through to coverage checking */
  ds->divergence_passes++;

  u32 key = hdr->diverge_key & (DIV_BANDIT_SIZE - 1);
  afl->div_last_diverge_key = key;
  afl->div_last_diverged = 1;

  return 0; /* Proceed to save_if_interesting */

}

/* ---- Trace capture ---- */

void div_capture_trace(afl_state_t *afl, struct queue_entry *q) {

  struct divergence_state *ds = &afl->div_state;

  if (!ds->div_shm) return;

  struct div_shm_header *hdr = (struct div_shm_header *)ds->div_shm;

  u32 len = hdr->actual_idx;
  if (len > DIV_MAX_TRACE_LEN) len = DIV_MAX_TRACE_LEN;

  /* Free any existing trace */
  if (q->div_trace) {

    free(q->div_trace);
    q->div_trace = NULL;
    q->div_trace_len = 0;

  }

  if (len == 0) return;

  q->div_trace = (u32 *)malloc(len * sizeof(u32));
  if (!q->div_trace) {

    WARNF("div_capture_trace: malloc failed for %u entries", len);
    return;

  }

  memcpy(q->div_trace, div_actual_trace(ds), len * sizeof(u32));
  q->div_trace_len = len;

  if (unlikely(afl->debug)) {

    ACTF("DIV: Captured trace for #%u (len=%u)", q->id, len);

  }

}

/* ---- Bandit reward/decay ---- */

void div_reward_corpus_add(afl_state_t *afl) {

  struct divergence_state *ds = &afl->div_state;

  if (!ds->div_shm || !afl->div_last_diverged) return;

  u32 key = afl->div_last_diverge_key;

  /* Reward: this divergent edge led to a corpus addition */
  ds->divergence_yield++;

  u8 *bandit = div_bandit_shm(ds);
  u32 score = bandit[key];
  score += DIV_BANDIT_REWARD;
  if (score > 255) score = 255;
  bandit[key] = (u8)score;

  if (unlikely(afl->debug)) {

    ACTF("DIV: Rewarded edge key=%u, new_score=%u", key, score);

  }

}

void div_decay_bandit(afl_state_t *afl) {

  struct divergence_state *ds = &afl->div_state;

  if (!ds->div_shm || !afl->div_last_diverged) return;

  u32 key = afl->div_last_diverge_key;

  /* This divergent mutation was allowed through but didn't contribute
     to the corpus. Decay its score. */
  u8 *bandit = div_bandit_shm(ds);
  u32 score = bandit[key];
  bandit[key] =
      (u8)((score * DIV_BANDIT_DECAY_NUM) / DIV_BANDIT_DECAY_DEN);

}

/* ---- Metrics ---- */

void div_write_stats(afl_state_t *afl, FILE *f) {

  struct divergence_state *ds = &afl->div_state;

  double div_rate = ds->total_mutations > 0
                        ? 100.0 * ds->divergence_count / ds->total_mutations
                        : 0.0;
  double yield_rate = ds->divergence_passes > 0
                          ? 100.0 * ds->divergence_yield / ds->divergence_passes
                          : 0.0;
  fprintf(f, "div_total_mutations  : %llu\n", ds->total_mutations);
  fprintf(f, "div_no_divergence    : %llu\n", ds->no_divergence_count);
  fprintf(f, "div_divergence_count : %llu\n", ds->divergence_count);
  fprintf(f, "div_divergence_rate  : %.2f%%\n", div_rate);
  fprintf(f, "div_divergence_kills : %llu\n", ds->divergence_kills);
  fprintf(f, "div_divergence_passes: %llu\n", ds->divergence_passes);
  fprintf(f, "div_divergence_yield : %llu\n", ds->divergence_yield);
  fprintf(f, "div_yield_rate       : %.2f%%\n", yield_rate);
  fprintf(f, "div_bandit_threshold : %u\n", ds->bandit_threshold);
  fprintf(f, "div_warmup_done      : %u\n", ds->warmup_done);

}
