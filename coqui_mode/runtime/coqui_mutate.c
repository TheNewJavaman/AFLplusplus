/*
 * coqui_mutate.c --- GPU-side AFL havoc mutation runtime.
 *
 * One case per MUT_* op from include/afl-mutations.h, each a port of its
 * counterpart in src/afl-fuzz-one.c. Operator semantics MUST match CPU
 * havoc at the distribution level (same weights, same stacking); the
 * per-thread PRNG sequence diverges from CPU by design (Q1 in design doc).
 *
 * Compile: --target=nvptx64-nvidia-cuda -O2 -ffreestanding
 * Spec: docs/superpowers/specs/2026-04-21-gpu-havoc-mutations-design.md
 */

#include "coqui_runtime.h"
#include "coqui_mutate.h"

#ifndef MAX_INPUT_SIZE
  #error "MAX_INPUT_SIZE must be defined by the build (-DMAX_INPUT_SIZE=N)"
#endif

/* AFL interesting-value tables, verbatim from include/afl-mutations.h and
 * include/afl-fuzz.h. Duplicated here because the runtime bitcode can't
 * include AFL headers (host-only types). */
static const signed char  interesting_8[]  = {
    -128, -1, 0, 1, 16, 32, 64, 100, 127
};
static const signed short interesting_16[] = {
    -128, -1, 0, 1, 16, 32, 64, 100, 127,
    -32768, -129, 128, 255, 256, 512, 1000, 1024, 4096, 32767
};
static const signed int   interesting_32[] = {
    -128, -1, 0, 1, 16, 32, 64, 100, 127,
    -32768, -129, 128, 255, 256, 512, 1000, 1024, 4096, 32767,
    -2147483648, -100663046, -32769, 32768, 65535, 65536, 100663045, 2139095040, 2147483647
};

#define INTERESTING_8_CNT  (sizeof(interesting_8)  / sizeof(interesting_8[0]))
#define INTERESTING_16_CNT (sizeof(interesting_16) / sizeof(interesting_16[0]))
#define INTERESTING_32_CNT (sizeof(interesting_32) / sizeof(interesting_32[0]))

/* AFL constants used by the ops — values from include/config.h. */
#define ARITH_MAX       35
#define HAVOC_BLK_SMALL 32
#define HAVOC_BLK_MEDIUM 128
#define HAVOC_BLK_LARGE  1500
#define HAVOC_BLK_XL     32768

/* ------------------------------------------------------------------------
 * PRNG: splitmix64 state, Lemire's fast unbiased bounded random.
 * ------------------------------------------------------------------------*/

static inline u64 splitmix64(u64 *state) {
  u64 z = (*state += 0x9e3779b97f4a7c15ULL);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}

/* Lemire's bounded random: returns r in [0, n). No rejection most of the
 * time; one 32×32->64 multiply + a rare rejection loop. */
static inline u32 gpu_rand_below(u64 *prng, u32 n) {
  if (n <= 1) return 0;
  u32 x = (u32)splitmix64(prng);
  u64 m = (u64)x * (u64)n;
  u32 l = (u32)m;
  if (l < n) {
    u32 t = (u32)(-n) % n;
    while (l < t) {
      x = (u32)splitmix64(prng);
      m = (u64)x * (u64)n;
      l = (u32)m;
    }
  }
  return (u32)(m >> 32);
}

/* ------------------------------------------------------------------------
 * Weighted splice partner selection.
 *
 * CPU precomputes prefix-sum weights into __coqui_seed_pool_cumw[].
 * We binary-search for the slot with cumw[slot] first exceeding the draw.
 * On lo == self_idx, return SELF_SAME; caller retries the op.
 * ------------------------------------------------------------------------*/

static inline u32 weighted_splice_pick(u64 *prng, u32 self_idx) {
  u32 n = __coqui_seed_pool_count;
  if (n == 0) return COQUI_SPLICE_SELF_SAME;
  u32 r = gpu_rand_below(prng, __coqui_seed_pool_cumw_total);
  u32 lo = 0, hi = n;
  while (lo < hi) {
    u32 m = (lo + hi) >> 1;
    if (__coqui_seed_pool_cumw[m] <= r) lo = m + 1; else hi = m;
  }
  if (lo >= n) lo = n - 1;   /* defensive: should not happen with correct CDF */
  return (lo == self_idx) ? COQUI_SPLICE_SELF_SAME : lo;
}

/* ------------------------------------------------------------------------
 * MUT_* enum — must match include/afl-mutations.h:46-85 exactly.
 * ------------------------------------------------------------------------*/

enum {
  MUT_FLIPBIT,         MUT_INTERESTING8,    MUT_INTERESTING16,
  MUT_INTERESTING16BE, MUT_INTERESTING32,   MUT_INTERESTING32BE,
  MUT_ARITH8_,         MUT_ARITH8,          MUT_ARITH16_,
  MUT_ARITH16BE_,      MUT_ARITH16,         MUT_ARITH16BE,
  MUT_ARITH32_,        MUT_ARITH32BE_,      MUT_ARITH32,
  MUT_ARITH32BE,       MUT_RAND8,           MUT_CLONE_COPY,
  MUT_CLONE_FIXED,     MUT_OVERWRITE_COPY,  MUT_OVERWRITE_FIXED,
  MUT_BYTEADD,         MUT_BYTESUB,         MUT_FLIP8,
  MUT_SWITCH,          MUT_DEL,             MUT_SHUFFLE,
  MUT_DELONE,          MUT_INSERTONE,       MUT_ASCIINUM,
  MUT_INSERTASCIINUM,  MUT_EXTRA_OVERWRITE, MUT_EXTRA_INSERT,
  MUT_AUTO_EXTRA_OVERWRITE, MUT_AUTO_EXTRA_INSERT,
  MUT_SPLICE_OVERWRITE,     MUT_SPLICE_INSERT,
  MUT_MAX
};

/* ------------------------------------------------------------------------
 * choose_block_len --- AFL helper used by CLONE/OVERWRITE/DEL ops. Port of
 * src/afl-fuzz-one.c choose_block_len.
 * ------------------------------------------------------------------------*/

static inline u32 choose_block_len(u64 *prng, u32 limit) {
  u32 min_value, max_value;
  /* Port of include/afl-mutations.h:1760-1793.
   * rlim = MIN(queue_cycle, 3); if (!run_over10m) rlim = 1 — early in the
   * run we only pick the small-block tier, same as AFL. */
  u32 rlim = __coqui_queue_cycle < 3u ? __coqui_queue_cycle : 3u;
  if (!__coqui_run_over10m) rlim = 1;
  if (rlim == 0) rlim = 1;  /* defensive against queue_cycle=0 early boot */
  u32 r = gpu_rand_below(prng, rlim);
  if (r == 0) { min_value = 1;            max_value = HAVOC_BLK_SMALL;   }
  else if (r == 1) { min_value = HAVOC_BLK_SMALL; max_value = HAVOC_BLK_MEDIUM; }
  else {
    if (gpu_rand_below(prng, 10)) {
      min_value = HAVOC_BLK_MEDIUM;
      max_value = HAVOC_BLK_LARGE;
    } else {
      min_value = HAVOC_BLK_LARGE;
      max_value = HAVOC_BLK_XL;
    }
  }
  if (min_value >= limit) min_value = 1;
  u32 span = (max_value < limit ? max_value : limit) - min_value + 1;
  return min_value + gpu_rand_below(prng, span);
}

/* ------------------------------------------------------------------------
 * Public entry: __coqui_havoc_mutate
 * ------------------------------------------------------------------------*/

u32 __coqui_havoc_mutate(u8 *buf, u32 len, u32 max_len,
                          u32 self_idx, u64 *prng) {
  (void)buf; (void)max_len; (void)self_idx;   /* suppress "unused" until ops land */
  if (len == 0 || max_len == 0) return len;

  u32 stack_max = 1u << (1 + gpu_rand_below(prng, __coqui_havoc_stack_pow2));
  u32 use_stacking = 1 + gpu_rand_below(prng, stack_max);

  for (u32 i = 0; i < use_stacking; ++i) {
    retry_havoc_step:;
    u32 r  = gpu_rand_below(prng, __coqui_mutation_array_size);
    u32 op = __coqui_mutation_array[r];

    switch (op) {
      /* Bucket 1 ops — filled in by task 1.6 */
      case MUT_FLIPBIT: {
        u8 bit = (u8)gpu_rand_below(prng, 8);
        u32 off = gpu_rand_below(prng, len);
        buf[off] ^= 1u << bit;
        break;
      }
      case MUT_INTERESTING8: {
        u32 pos = gpu_rand_below(prng, len);
        u32 item = gpu_rand_below(prng, INTERESTING_8_CNT);
        buf[pos] = (u8)interesting_8[item];
        break;
      }
      case MUT_INTERESTING16: {
        if (len < 2) goto retry_havoc_step;
        u32 pos = gpu_rand_below(prng, len - 1);
        u32 item = gpu_rand_below(prng, INTERESTING_16_CNT);
        u16 v = (u16)interesting_16[item];
        buf[pos]     = (u8)(v & 0xFF);
        buf[pos + 1] = (u8)((v >> 8) & 0xFF);
        break;
      }
      case MUT_INTERESTING16BE: {
        if (len < 2) goto retry_havoc_step;
        u32 pos = gpu_rand_below(prng, len - 1);
        u32 item = gpu_rand_below(prng, INTERESTING_16_CNT);
        u16 v = (u16)interesting_16[item];
        buf[pos]     = (u8)((v >> 8) & 0xFF);
        buf[pos + 1] = (u8)(v & 0xFF);
        break;
      }
      case MUT_INTERESTING32: {
        if (len < 4) goto retry_havoc_step;
        u32 pos = gpu_rand_below(prng, len - 3);
        u32 item = gpu_rand_below(prng, INTERESTING_32_CNT);
        u32 v = (u32)interesting_32[item];
        buf[pos]     = (u8)(v & 0xFF);
        buf[pos + 1] = (u8)((v >> 8)  & 0xFF);
        buf[pos + 2] = (u8)((v >> 16) & 0xFF);
        buf[pos + 3] = (u8)((v >> 24) & 0xFF);
        break;
      }
      case MUT_INTERESTING32BE: {
        if (len < 4) goto retry_havoc_step;
        u32 pos = gpu_rand_below(prng, len - 3);
        u32 item = gpu_rand_below(prng, INTERESTING_32_CNT);
        u32 v = (u32)interesting_32[item];
        buf[pos]     = (u8)((v >> 24) & 0xFF);
        buf[pos + 1] = (u8)((v >> 16) & 0xFF);
        buf[pos + 2] = (u8)((v >> 8)  & 0xFF);
        buf[pos + 3] = (u8)(v & 0xFF);
        break;
      }
      case MUT_ARITH8_: {
        u32 off = gpu_rand_below(prng, len);
        u32 item = 1 + gpu_rand_below(prng, ARITH_MAX);
        buf[off] = (u8)(buf[off] - item);
        break;
      }
      case MUT_ARITH8: {
        u32 off = gpu_rand_below(prng, len);
        u32 item = 1 + gpu_rand_below(prng, ARITH_MAX);
        buf[off] = (u8)(buf[off] + item);
        break;
      }
      case MUT_ARITH16_: case MUT_ARITH16BE_:
      case MUT_ARITH16: case MUT_ARITH16BE:
      case MUT_ARITH32_: case MUT_ARITH32BE_:
      case MUT_ARITH32: case MUT_ARITH32BE:
      case MUT_RAND8:
      case MUT_CLONE_COPY: case MUT_CLONE_FIXED:
      case MUT_OVERWRITE_COPY: case MUT_OVERWRITE_FIXED:
      case MUT_BYTEADD: case MUT_BYTESUB:
      case MUT_FLIP8: case MUT_SWITCH:
      case MUT_DEL: case MUT_SHUFFLE:
      case MUT_DELONE: case MUT_INSERTONE:
        goto retry_havoc_step;  /* placeholder; fill in task 1.6 */

      /* Bucket 2 ops — filled in by task 1.7 */
      case MUT_ASCIINUM:
      case MUT_INSERTASCIINUM:
        goto retry_havoc_step;  /* placeholder; fill in task 1.7 */

      /* Bucket 3 ops — filled in by task 1.8 */
      case MUT_EXTRA_OVERWRITE: case MUT_EXTRA_INSERT:
      case MUT_AUTO_EXTRA_OVERWRITE: case MUT_AUTO_EXTRA_INSERT:
      case MUT_SPLICE_OVERWRITE: case MUT_SPLICE_INSERT:
        goto retry_havoc_step;  /* placeholder; fill in task 1.8 */

      default:
        goto retry_havoc_step;  /* unknown op code — retry */
    }
  }
  return len;
}
