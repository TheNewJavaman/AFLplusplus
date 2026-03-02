/*
   american fuzzy lop++ - divergence scheduler header
   ---------------------------------------------------

   Written by Gabriel Pizarro

   Ordered-trace divergence detection for AFL++. The target process records
   an ordered edge trace in shared memory. At each basic block, the
   instrumentation callback compares the current edge against the seed's
   reference trace. On first divergence, a bandit scheduler decides whether
   to _exit(0) (true mid-execution termination) or let the mutation finish.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at:

     https://www.apache.org/licenses/LICENSE-2.0

 */

#ifndef _HAVE_DIVERGENCE_H
#define _HAVE_DIVERGENCE_H

#include "types.h"
#include "config.h"

/* ---- Shared memory layout ---- */

/* Environment variable name for the divergence shared memory ID/path.
   Set by the fuzzer, read by the target's runtime. */
#define DIV_SHM_ENV_VAR "__AFL_DIV_SHM_ID"

/* Maximum trace length (number of edge IDs). Traces longer than this are
   truncated — comparison stops, execution continues. 64K entries. */
#define DIV_MAX_TRACE_LEN (1u << 16)

/* Size of the bandit score table. 64KB — keyed by edge ID masked to this. */
#define DIV_BANDIT_SIZE (1u << 16)

/* Shared memory header written/read by both fuzzer and target. */
struct div_shm_header {

  u32 ref_len;            /* Reference trace length (fuzzer writes)          */
  u32 actual_idx;         /* Current position in actual trace (target incrs) */
  u32 bandit_threshold;   /* Kill threshold (fuzzer writes)                  */
  u8  diverged;           /* First divergence detected (target writes)       */
  u8  killed;             /* Target was killed by bandit (target writes)     */
  u8  pad[2];             /* Alignment padding                               */
  u32 diverge_pos;        /* Position of first divergence (target writes)    */
  u32 diverge_key;        /* Bandit key for divergence (target writes)       */
  u8  pad2[40];           /* Pad to 64 bytes total                           */

};

/* Total shared memory size:
   64 bytes header + 2 * 64K * 4 bytes traces + 64K bandit = 576KB + 64 */
#define DIV_SHM_SIZE                                            \
  (sizeof(struct div_shm_header) + DIV_MAX_TRACE_LEN * sizeof(u32) * 2 + \
   DIV_BANDIT_SIZE)

/* ---- Bandit parameters ---- */

/* Default bandit threshold. Scores below this cause the mutation to be
   killed (mid-execution _exit(0)). Set to 0 during warmup (allow everything),
   then raised. Lower = more permissive (fewer kills). */
#define DIV_BANDIT_THRESHOLD_DEFAULT 0

/* Number of mutations to run before enabling the bandit scheduler. During
   warmup all divergent mutations are allowed through to collect data. */
#define DIV_WARMUP_EXECS 100000

/* Bandit score initial value. Unseen edge tuples start optimistic. */
#define DIV_BANDIT_INITIAL 255

/* Bandit decay: score = score * NUM / DEN on each divergent mutation that
   was allowed but did NOT contribute to the corpus. With 127/128, a score
   decays from 255 to 8 after ~450 consecutive misses. */
#define DIV_BANDIT_DECAY_NUM 253
#define DIV_BANDIT_DECAY_DEN 254

/* Bandit reward increment when a divergent mutation IS added to corpus. */
#define DIV_BANDIT_REWARD 128

/* ---- Fuzzer-side divergence state ---- */

struct divergence_state {

  /* Shared memory region (owned by fuzzer, mapped into target via env var) */
  u8  *div_shm;          /* Pointer to shared memory region                */
  s32  shm_id;           /* SysV shm ID (non-USEMMAP) or fd (USEMMAP)     */
#ifdef USEMMAP
  char shm_file_path[64]; /* POSIX shm file path                          */
#endif

  /* Bandit scheduler (lives directly in shm, no fuzzer-side copy) */
  u32  bandit_threshold;   /* Current threshold for kill/pass decision      */
  u8   warmup_done;        /* Whether warmup period is over                 */

  /* Cached seed ID to skip redundant ref_trace memcpy */
  u32  last_seed_id;         /* Queue entry ID of last seed used             */

  /* Metrics */
  u64 total_mutations;
  u64 no_divergence_count;
  u64 divergence_count;
  u64 divergence_kills;
  u64 divergence_passes;
  u64 divergence_yield;
  u64 total_edges;           /* Sum of edges hit across all mutations       */

};

/* ---- Convenience accessors for shared memory regions ---- */

static inline u32 *div_ref_trace(struct divergence_state *ds) {

  return (u32 *)(ds->div_shm + sizeof(struct div_shm_header));

}

static inline u32 *div_actual_trace(struct divergence_state *ds) {

  return (u32 *)(ds->div_shm + sizeof(struct div_shm_header) +
                 DIV_MAX_TRACE_LEN * sizeof(u32));

}

static inline u8 *div_bandit_shm(struct divergence_state *ds) {

  return ds->div_shm + sizeof(struct div_shm_header) +
         DIV_MAX_TRACE_LEN * sizeof(u32) * 2;

}

/* ---- API (implemented in afl-fuzz-divergence.c) ---- */

struct afl_state;  /* forward declaration */

/* Initialize divergence subsystem: allocate shared memory and bandit table. */
void div_init(struct afl_state *afl);

/* Tear down divergence subsystem: free shared memory and allocations. */
void div_deinit(struct afl_state *afl);

/* Prepare shared memory before each mutation execution: copy reference trace,
   bandit table, and reset counters. */
void div_prepare_execution(struct afl_state *afl);

/* Process the result of a mutation execution. Reads divergence status from
   shared memory. Returns 1 if the mutation should be skipped (no divergence
   or killed by bandit). Returns 0 if the mutation diverged and was allowed
   through (should proceed to coverage checking). */
u8 div_post_execution(struct afl_state *afl);

/* Capture the actual trace from shared memory into a queue entry's
   div_trace/div_trace_len fields. Called when a mutation is added to corpus. */
void div_capture_trace(struct afl_state *afl, struct queue_entry *q);

/* Called when a mutation is added to the corpus. Rewards the bandit
   for the divergent edge that led to this corpus addition. */
void div_reward_corpus_add(struct afl_state *afl);

/* Called when a divergent mutation was NOT added to the corpus. Decays
   the bandit score for the edge that diverged. */
void div_decay_bandit(struct afl_state *afl);

/* Write divergence metrics to the stats file. */
void div_write_stats(struct afl_state *afl, FILE *f);

#endif /* _HAVE_DIVERGENCE_H */
