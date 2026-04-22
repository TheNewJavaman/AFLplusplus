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

/* -- Definitions for the extern symbols declared in coqui_runtime.h.
 * These variables live in the cubin's module globals and are bound by
 * the host via cuModuleGetGlobal + cuMemcpyHtoD. All start at null/zero. */

u8  *__coqui_seed_pool_base     = 0;
u32 *__coqui_seed_pool_offsets  = 0;
u32 *__coqui_seed_pool_lens     = 0;
u32 *__coqui_seed_pool_cumw     = 0;
u32  __coqui_seed_pool_count      = 0;
u32  __coqui_seed_pool_cumw_total = 0;

u8  *__coqui_extras_base        = 0;
u32 *__coqui_extras_offsets     = 0;
u32 *__coqui_extras_lens        = 0;
u32  __coqui_extras_cnt           = 0;

u8  *__coqui_a_extras_base      = 0;
u32 *__coqui_a_extras_offsets   = 0;
u32 *__coqui_a_extras_lens      = 0;
u32  __coqui_a_extras_cnt         = 0;

u64  __coqui_prng_base            = 0;

u32  __coqui_reported_count       = 0;
u32 *__coqui_reported_tid       = 0;
u32 *__coqui_reported_lens      = 0;

/* The three __constant__-memory globals need to be defined with the
 * address_space(4) attribute to match the extern declaration. */
__attribute__((address_space(4))) u32 __coqui_mutation_array[256] = {0};
__attribute__((address_space(4))) u32 __coqui_mutation_array_size = 0;
__attribute__((address_space(4))) u32 __coqui_havoc_stack_pow2    = 0;

/* queue_cycle / run_over10m added by choose_block_len fidelity fix. */
u32  __coqui_queue_cycle          = 0;
u32  __coqui_run_over10m          = 0;

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
 * Splice partner selection.
 *
 * Host pre-selects the 256-entry seed pool by weight, so uniform picking
 * within the pool yields the same statistical "weighted subset" AFL gets
 * from weighted queue scheduling. AFL's havoc splice (afl-fuzz-one.c:3371)
 * rejects (tid == current_entry) and (target->len < 4); we mirror that by
 * rejecting self_idx and pool entries shorter than 4 bytes. Bounded retry
 * (32 tries) avoids infinite loops when no valid partner exists.
 * ------------------------------------------------------------------------*/

static inline u32 weighted_splice_pick(u64 *prng, u32 self_idx) {
  u32 n = __coqui_seed_pool_count;
  if (n <= 1) return COQUI_SPLICE_SELF_SAME;
  for (u32 tries = 0; tries < 32; ++tries) {
    u32 p = gpu_rand_below(prng, n);
    if (p == self_idx) continue;
    if (__coqui_seed_pool_lens[p] < 4) continue;
    return p;
  }
  return COQUI_SPLICE_SELF_SAME;
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
  if (len == 0 || max_len == 0) return len;

  u32 stack_max = 1u << (1 + gpu_rand_below(prng, __coqui_havoc_stack_pow2));
  u32 use_stacking = 1 + gpu_rand_below(prng, stack_max);

  for (u32 i = 0; i < use_stacking; ++i) {
    retry_havoc_step:;
    u32 r  = gpu_rand_below(prng, __coqui_mutation_array_size);
    u32 op = __coqui_mutation_array[r];

    switch (op) {
      /* ---------- Bucket 1: 22 ports of src/afl-fuzz-one.c:2320-3050 ----------
       * Semantic invariants (from AFL source):
       *   - "no retry" guards use `break`, not `goto retry_havoc_step`
       *   - MUT_RAND8 XORs with 1..255 (no no-op)
       *   - MUT_BYTEADD/BYTESUB are `++` / `--` (always ±1)
       *   - MUT_SWITCH: guard < 4, do-while distinct, BLOCK swap
       *   - CLONE_FIXED / OVERWRITE_FIXED / INSERTONE use 50/50 random-vs-neighbor fill
       *   - SHUFFLE inner loop uses do-while to reject i==j
       * GPU adaptation: MAX_FILE→max_len is small (4096), so CLONE_* guards
       * are simplified to "any-growth-possible" with clone_len clipping.
       */

      case MUT_FLIPBIT: {
        u8  bit = (u8)gpu_rand_below(prng, 8);
        u32 off = gpu_rand_below(prng, len);
        buf[off] ^= 1u << bit;
        break;
      }

      case MUT_INTERESTING8: {
        u32 item = gpu_rand_below(prng, INTERESTING_8_CNT);
        buf[gpu_rand_below(prng, len)] = (u8)interesting_8[item];
        break;
      }

      case MUT_INTERESTING16: {
        if (len < 2) break;                          /* no retry */
        u32 item = gpu_rand_below(prng, INTERESTING_16_CNT);
        u32 pos = gpu_rand_below(prng, len - 1);
        u16 v = (u16)interesting_16[item];
        buf[pos]     = (u8)(v & 0xFF);
        buf[pos + 1] = (u8)((v >> 8) & 0xFF);
        break;
      }

      case MUT_INTERESTING16BE: {
        if (len < 2) break;                          /* no retry */
        u32 item = gpu_rand_below(prng, INTERESTING_16_CNT);
        u32 pos = gpu_rand_below(prng, len - 1);
        u16 v = (u16)interesting_16[item];
        buf[pos]     = (u8)((v >> 8) & 0xFF);
        buf[pos + 1] = (u8)(v & 0xFF);
        break;
      }

      case MUT_INTERESTING32: {
        if (len < 4) break;                          /* no retry */
        u32 item = gpu_rand_below(prng, INTERESTING_32_CNT);
        u32 pos = gpu_rand_below(prng, len - 3);
        u32 v = (u32)interesting_32[item];
        buf[pos]     = (u8)(v & 0xFF);
        buf[pos + 1] = (u8)((v >> 8)  & 0xFF);
        buf[pos + 2] = (u8)((v >> 16) & 0xFF);
        buf[pos + 3] = (u8)((v >> 24) & 0xFF);
        break;
      }

      case MUT_INTERESTING32BE: {
        if (len < 4) break;                          /* no retry */
        u32 item = gpu_rand_below(prng, INTERESTING_32_CNT);
        u32 pos = gpu_rand_below(prng, len - 3);
        u32 v = (u32)interesting_32[item];
        buf[pos]     = (u8)((v >> 24) & 0xFF);
        buf[pos + 1] = (u8)((v >> 16) & 0xFF);
        buf[pos + 2] = (u8)((v >> 8)  & 0xFF);
        buf[pos + 3] = (u8)(v & 0xFF);
        break;
      }

      case MUT_ARITH8_: {
        u32 pos = gpu_rand_below(prng, len);
        u32 item = 1 + gpu_rand_below(prng, ARITH_MAX);
        buf[pos] = (u8)(buf[pos] - item);
        break;
      }

      case MUT_ARITH8: {
        u32 pos = gpu_rand_below(prng, len);
        u32 item = 1 + gpu_rand_below(prng, ARITH_MAX);
        buf[pos] = (u8)(buf[pos] + item);
        break;
      }

      case MUT_ARITH16_: {
        if (len < 2) break;                          /* no retry */
        u32 pos = gpu_rand_below(prng, len - 1);
        u16 item = (u16)(1 + gpu_rand_below(prng, ARITH_MAX));
        u16 v = (u16)buf[pos] | ((u16)buf[pos + 1] << 8);
        v = v - item;
        buf[pos]     = (u8)(v & 0xFF);
        buf[pos + 1] = (u8)((v >> 8) & 0xFF);
        break;
      }

      case MUT_ARITH16BE_: {
        if (len < 2) break;                          /* no retry */
        u32 pos = gpu_rand_below(prng, len - 1);
        u16 num = (u16)(1 + gpu_rand_below(prng, ARITH_MAX));
        u16 v = ((u16)buf[pos] << 8) | (u16)buf[pos + 1];      /* BE load */
        v = v - num;
        buf[pos]     = (u8)((v >> 8) & 0xFF);                  /* BE store */
        buf[pos + 1] = (u8)(v & 0xFF);
        break;
      }

      case MUT_ARITH16: {
        if (len < 2) break;                          /* no retry */
        u32 pos = gpu_rand_below(prng, len - 1);
        u16 item = (u16)(1 + gpu_rand_below(prng, ARITH_MAX));
        u16 v = (u16)buf[pos] | ((u16)buf[pos + 1] << 8);
        v = v + item;
        buf[pos]     = (u8)(v & 0xFF);
        buf[pos + 1] = (u8)((v >> 8) & 0xFF);
        break;
      }

      case MUT_ARITH16BE: {
        if (len < 2) break;                          /* no retry */
        u32 pos = gpu_rand_below(prng, len - 1);
        u16 num = (u16)(1 + gpu_rand_below(prng, ARITH_MAX));
        u16 v = ((u16)buf[pos] << 8) | (u16)buf[pos + 1];
        v = v + num;
        buf[pos]     = (u8)((v >> 8) & 0xFF);
        buf[pos + 1] = (u8)(v & 0xFF);
        break;
      }

      case MUT_ARITH32_: {
        if (len < 4) break;                          /* no retry */
        u32 pos = gpu_rand_below(prng, len - 3);
        u32 item = 1 + gpu_rand_below(prng, ARITH_MAX);
        u32 v = (u32)buf[pos] | ((u32)buf[pos + 1] << 8)
              | ((u32)buf[pos + 2] << 16) | ((u32)buf[pos + 3] << 24);
        v = v - item;
        buf[pos]     = (u8)(v & 0xFF);
        buf[pos + 1] = (u8)((v >> 8) & 0xFF);
        buf[pos + 2] = (u8)((v >> 16) & 0xFF);
        buf[pos + 3] = (u8)((v >> 24) & 0xFF);
        break;
      }

      case MUT_ARITH32BE_: {
        if (len < 4) break;                          /* no retry */
        u32 pos = gpu_rand_below(prng, len - 3);
        u32 num = 1 + gpu_rand_below(prng, ARITH_MAX);
        u32 v = ((u32)buf[pos] << 24) | ((u32)buf[pos + 1] << 16)
              | ((u32)buf[pos + 2] << 8) | (u32)buf[pos + 3];
        v = v - num;
        buf[pos]     = (u8)((v >> 24) & 0xFF);
        buf[pos + 1] = (u8)((v >> 16) & 0xFF);
        buf[pos + 2] = (u8)((v >> 8) & 0xFF);
        buf[pos + 3] = (u8)(v & 0xFF);
        break;
      }

      case MUT_ARITH32: {
        if (len < 4) break;                          /* no retry */
        u32 pos = gpu_rand_below(prng, len - 3);
        u32 item = 1 + gpu_rand_below(prng, ARITH_MAX);
        u32 v = (u32)buf[pos] | ((u32)buf[pos + 1] << 8)
              | ((u32)buf[pos + 2] << 16) | ((u32)buf[pos + 3] << 24);
        v = v + item;
        buf[pos]     = (u8)(v & 0xFF);
        buf[pos + 1] = (u8)((v >> 8) & 0xFF);
        buf[pos + 2] = (u8)((v >> 16) & 0xFF);
        buf[pos + 3] = (u8)((v >> 24) & 0xFF);
        break;
      }

      case MUT_ARITH32BE: {
        if (len < 4) break;                          /* no retry */
        u32 pos = gpu_rand_below(prng, len - 3);
        u32 num = 1 + gpu_rand_below(prng, ARITH_MAX);
        u32 v = ((u32)buf[pos] << 24) | ((u32)buf[pos + 1] << 16)
              | ((u32)buf[pos + 2] << 8) | (u32)buf[pos + 3];
        v = v + num;
        buf[pos]     = (u8)((v >> 24) & 0xFF);
        buf[pos + 1] = (u8)((v >> 16) & 0xFF);
        buf[pos + 2] = (u8)((v >> 8) & 0xFF);
        buf[pos + 3] = (u8)(v & 0xFF);
        break;
      }

      case MUT_RAND8: {
        /* AFL: out_buf[pos] ^= 1 + rand(255) — XOR with non-zero byte to
         * eliminate no-op. Not random-overwrite. */
        u32 pos = gpu_rand_below(prng, len);
        u32 item = 1 + gpu_rand_below(prng, 255);
        buf[pos] ^= (u8)item;
        break;
      }

      case MUT_FLIP8: {
        /* AFL: out_buf[rand] ^= 0xff (equivalent to ~out_buf[rand]) */
        buf[gpu_rand_below(prng, len)] ^= 0xFF;
        break;
      }

      case MUT_SWITCH: {
        /* AFL: guard < 4 with break; do-while distinct; BLOCK swap of size
         * choose_block_len(MIN(switch_len, to_end)). */
        if (len < 4) break;                          /* no retry */
        u32 switch_from = gpu_rand_below(prng, len);
        u32 switch_to;
        do {
          switch_to = gpu_rand_below(prng, len);
        } while (switch_from == switch_to);
        u32 switch_len, to_end;
        if (switch_from < switch_to) {
          switch_len = switch_to - switch_from;
          to_end = len - switch_to;
        } else {
          switch_len = switch_from - switch_to;
          to_end = len - switch_from;
        }
        u32 limit = switch_len < to_end ? switch_len : to_end;
        switch_len = choose_block_len(prng, limit);
        /* Block swap, byte-by-byte (ranges are non-overlapping by construction). */
        for (u32 i = 0; i < switch_len; ++i) {
          u8 t = buf[switch_from + i];
          buf[switch_from + i] = buf[switch_to + i];
          buf[switch_to + i] = t;
        }
        break;
      }

      case MUT_BYTEADD: {
        /* AFL: out_buf[rand]++ — always +1 */
        buf[gpu_rand_below(prng, len)]++;
        break;
      }

      case MUT_BYTESUB: {
        /* AFL: out_buf[rand]-- — always -1 */
        buf[gpu_rand_below(prng, len)]--;
        break;
      }

      case MUT_CLONE_COPY: {
        /* AFL's guard is `temp_len + HAVOC_BLK_XL < MAX_FILE`; with MAX_FILE
         * ~= 1MB, that's typically true. Our max_len is small (<=4KB), so
         * HAVOC_BLK_XL (32KB) > max_len always — original guard never passes.
         * Adapt: allow cloning whenever any growth is possible; clip clone_len. */
        if (len + 1 < max_len) {
          u32 clone_len = choose_block_len(prng, len);
          if (clone_len > max_len - len - 1) clone_len = max_len - len - 1;
          if (clone_len == 0) break;
          u32 clone_from = gpu_rand_below(prng, len - clone_len + 1);
          u32 clone_to   = gpu_rand_below(prng, len);
          for (u32 i = len; i > clone_to; --i) buf[i - 1 + clone_len] = buf[i - 1];
          /* In-place copy: after the shift above, bytes originally at position
           * s (for s >= clone_to) now live at s + clone_len. Re-address source
           * reads that fall into the shifted region. */
          for (u32 i = 0; i < clone_len; ++i) {
            u32 src = clone_from + i;
            if (src >= clone_to) src += clone_len;
            buf[clone_to + i] = buf[src];
          }
          len += clone_len;
        } else if (len < 8) {
          break;
        } else {
          goto retry_havoc_step;
        }
        break;
      }

      case MUT_CLONE_FIXED: {
        /* Same GPU adaptation as CLONE_COPY. 50/50 strat for fill byte
         * (random vs neighbor-copy), matching AFL. */
        if (len + 1 < max_len) {
          u32 clone_len = choose_block_len(prng, HAVOC_BLK_XL);
          if (clone_len > max_len - len - 1) clone_len = max_len - len - 1;
          if (clone_len == 0) break;
          u32 clone_to = gpu_rand_below(prng, len);
          u32 strat = gpu_rand_below(prng, 2);
          u32 clone_from = clone_to ? clone_to - 1 : 0;
          u8  item = strat ? (u8)gpu_rand_below(prng, 256) : buf[clone_from];
          for (u32 i = len; i > clone_to; --i) buf[i - 1 + clone_len] = buf[i - 1];
          for (u32 i = 0; i < clone_len; ++i)  buf[clone_to + i] = item;
          len += clone_len;
        } else if (len < 8) {
          break;
        } else {
          goto retry_havoc_step;
        }
        break;
      }

      case MUT_OVERWRITE_COPY: {
        if (len < 2) break;                          /* no retry */
        u32 copy_len = choose_block_len(prng, len - 1);
        u32 copy_from, copy_to;
        do {
          copy_from = gpu_rand_below(prng, len - copy_len + 1);
          copy_to   = gpu_rand_below(prng, len - copy_len + 1);
        } while (copy_from == copy_to);
        /* memmove semantics for possibly overlapping ranges. */
        if (copy_to > copy_from) {
          for (u32 i = copy_len; i > 0; --i)
            buf[copy_to + i - 1] = buf[copy_from + i - 1];
        } else {
          for (u32 i = 0; i < copy_len; ++i)
            buf[copy_to + i] = buf[copy_from + i];
        }
        break;
      }

      case MUT_OVERWRITE_FIXED: {
        if (len < 2) break;                          /* no retry */
        u32 copy_len = choose_block_len(prng, len - 1);
        u32 copy_to  = gpu_rand_below(prng, len - copy_len + 1);
        u32 strat = gpu_rand_below(prng, 2);
        u32 copy_from = copy_to ? copy_to - 1 : 0;
        u8 item = strat ? (u8)gpu_rand_below(prng, 256) : buf[copy_from];
        for (u32 i = 0; i < copy_len; ++i) buf[copy_to + i] = item;
        break;
      }

      case MUT_DEL: {
        if (len < 2) break;                          /* no retry */
        u32 del_len = choose_block_len(prng, len - 1);
        u32 del_from = gpu_rand_below(prng, len - del_len + 1);
        for (u32 i = del_from; i + del_len < len; ++i) buf[i] = buf[i + del_len];
        len -= del_len;
        break;
      }

      case MUT_DELONE: {
        if (len < 2) break;                          /* no retry */
        u32 del_len = 1;
        u32 del_from = gpu_rand_below(prng, len - del_len + 1);
        for (u32 i = del_from; i + del_len < len; ++i) buf[i] = buf[i + del_len];
        len -= del_len;
        break;
      }

      case MUT_SHUFFLE: {
        if (len < 4) break;                          /* no retry */
        u32 slen = choose_block_len(prng, len - 1);
        u32 off = gpu_rand_below(prng, len - slen + 1);
        /* AFL Fisher-Yates inner loop uses do-while(i==j) to reject
         * degenerate swaps. */
        for (u32 i = slen - 1; i > 0; --i) {
          u32 j;
          do {
            j = gpu_rand_below(prng, i + 1);
          } while (j == i);
          u8 t = buf[off + i];
          buf[off + i] = buf[off + j];
          buf[off + j] = t;
        }
        break;
      }

      case MUT_INSERTONE: {
        if (len < 2) break;                          /* no retry (AFL's guard) */
        /* GPU-specific overflow guard: also break if no room to grow (AFL
         * can always grow via realloc; we have a fixed max_len). */
        if (len + 1 >= max_len) break;
        u32 clone_to = gpu_rand_below(prng, len);
        u32 strat = gpu_rand_below(prng, 2);
        u32 clone_from = clone_to ? clone_to - 1 : 0;
        u8 item = strat ? (u8)gpu_rand_below(prng, 256) : buf[clone_from];
        for (u32 i = len; i > clone_to; --i) buf[i] = buf[i - 1];
        buf[clone_to] = item;
        len += 1;
        break;
      }

      /* ---------- Bucket 2: port of src/afl-fuzz-one.c:3044-3229 ---------- */

      case MUT_ASCIINUM: {
        /* Find an ASCII decimal run in buf[], parse it, apply one of 8
         * strategies, write back (may grow or shrink).
         * AFL line 3044. "no retry" guard on len < 4. */
        if (len < 4) break;

        u32 off = gpu_rand_below(prng, len);
        u32 off2 = off, cnt = 0;
        /* Scan forward from off for first ASCII digit. */
        while (off2 + cnt < len &&
               !(buf[off2 + cnt] >= '0' && buf[off2 + cnt] <= '9')) {
          ++cnt;
        }
        /* No digit found to the right — wrap: search [0, off) from the start. */
        if (off2 + cnt == len) {
          off2 = 0;
          cnt = 0;
          while (cnt < off &&
                 !(buf[off2 + cnt] >= '0' && buf[off2 + cnt] <= '9')) {
            ++cnt;
          }
          if (cnt == off) {
            /* No digit anywhere in the input. */
            if (len < 8) break;
            goto retry_havoc_step;
          }
        }
        off = off2 + cnt;          /* position of first digit */
        off2 = off + 1;
        while (off2 < len && buf[off2] >= '0' && buf[off2] <= '9') {
          ++off2;
        }
        /* buf[off..off2) is the digit run. Parse it as s64. */
        s64 val = (s64)(buf[off] - '0');
        for (u32 i = off + 1; i < off2; ++i) {
          val = (val * 10) + (s64)(buf[i] - '0');
        }
        /* Negative sign? AFL checks out_buf[off - 1] == '-'. */
        if (off > 0 && buf[off - 1] == '-') val = -val;

        /* Apply one of 8 strategies. */
        u32 strat = gpu_rand_below(prng, 8);
        switch (strat) {
          case 0: val++; break;
          case 1: val--; break;
          case 2: val *= 2; break;
          case 3: val /= 2; break;
          case 4:
            /* AFL: if (val && (u64)val < 0x19999999) val = rand_next % (val*10);
             *      else val = rand_below(256); */
            if (val != 0 && (u64)val < 0x19999999ULL) {
              u64 r = splitmix64(prng);
              val = (s64)(r % ((u64)val * 10ULL));
            } else {
              val = (s64)gpu_rand_below(prng, 256);
            }
            break;
          case 5: val += (s64)gpu_rand_below(prng, 256); break;
          case 6: val -= (s64)gpu_rand_below(prng, 256); break;
          case 7: val = ~val; break;
        }

        /* Convert val to decimal string (signed, up to 20 chars + sign). */
        u8 tmp[21];
        u32 new_len = 0;
        {
          u64 absval;
          int is_neg = 0;
          if (val < 0) { is_neg = 1; absval = (u64)0 - (u64)val; }
          else          { absval = (u64)val; }
          u8 rev[20];
          u32 nrev = 0;
          if (absval == 0) { rev[nrev++] = '0'; }
          else {
            while (absval > 0 && nrev < 20) {
              rev[nrev++] = (u8)('0' + (absval % 10ULL));
              absval /= 10ULL;
            }
          }
          if (is_neg) tmp[new_len++] = '-';
          for (u32 i = 0; i < nrev; ++i) tmp[new_len++] = rev[nrev - 1 - i];
        }

        /* Replace buf[off..off2) with tmp[0..new_len). */
        u32 old_len = off2 - off;
        if (old_len == new_len) {
          for (u32 i = 0; i < new_len; ++i) buf[off + i] = tmp[i];
        } else if (new_len < old_len) {
          u32 shrink = old_len - new_len;
          for (u32 i = 0; i < new_len; ++i) buf[off + i] = tmp[i];
          /* Shift tail left to close the gap. */
          for (u32 i = off + new_len; i + shrink < len; ++i) {
            buf[i] = buf[i + shrink];
          }
          len -= shrink;
        } else {
          u32 grow = new_len - old_len;
          /* GPU-specific: we can't realloc, only grow if room. AFL would
           * succeed here via afl_realloc; we break on overflow. */
          if (len + grow > max_len) break;
          /* Shift tail right to make room. Iterate from buf[len-1] down
           * through buf[off2] so we don't overwrite unread tail bytes. */
          for (u32 i = len; i > off2; --i) buf[i - 1 + grow] = buf[i - 1];
          for (u32 i = 0; i < new_len; ++i) buf[off + i] = tmp[i];
          len += grow;
        }
        break;
      }

      case MUT_INSERTASCIINUM: {
        /* AFL line 3199. Despite the name, this OVERWRITES len bytes at a
         * random position with ASCII digits of a random u64 (plus possibly
         * uninitialized bytes when the number's decimal repr is shorter than
         * the requested write length). We zero-pad to make the "tail" bytes
         * deterministic. */
        u32 ilen = 1 + gpu_rand_below(prng, 8);        /* 1..8 */
        u32 pos  = gpu_rand_below(prng, len);
        if (len < pos + ilen) {
          /* Not enough room to overwrite at this position. */
          if (len < 8) break;
          goto retry_havoc_step;
        }
        u64 val = splitmix64(prng);                    /* random u64 */
        /* Stringify val (unsigned) up to 20 digits. */
        u8 tmpbuf[20];
        u32 n = 0;
        if (val == 0) { tmpbuf[n++] = '0'; }
        else {
          u8 rev[20];
          u32 nrev = 0;
          while (val > 0 && nrev < 20) {
            rev[nrev++] = (u8)('0' + (val % 10ULL));
            val /= 10ULL;
          }
          for (u32 i = 0; i < nrev; ++i) tmpbuf[n++] = rev[nrev - 1 - i];
        }
        /* Overwrite ilen bytes at pos with tmpbuf (zero-pad beyond n). */
        for (u32 i = 0; i < ilen; ++i) {
          buf[pos + i] = (i < n) ? tmpbuf[i] : (u8)0;
        }
        break;
      }

      /* ---------- Bucket 3: port of src/afl-fuzz-one.c:3231-3499 ---------- */

      case MUT_EXTRA_OVERWRITE: {
        /* AFL line 3231. */
        if (__coqui_extras_cnt == 0) goto retry_havoc_step;
        u32 use_extra = gpu_rand_below(prng, __coqui_extras_cnt);
        u32 extra_len = __coqui_extras_lens[use_extra];
        if (extra_len > len) goto retry_havoc_step;
        u32 insert_at = gpu_rand_below(prng, len - extra_len + 1);
        u32 e_off = __coqui_extras_offsets[use_extra];
        for (u32 i = 0; i < extra_len; ++i) {
          buf[insert_at + i] = __coqui_extras_base[e_off + i];
        }
        break;
      }

      case MUT_EXTRA_INSERT: {
        /* AFL line 3254. Adapt MAX_FILE guard to max_len; clip extra_len
         * if it would overflow max_len (conservative: retry if full extra
         * doesn't fit). */
        if (__coqui_extras_cnt == 0) goto retry_havoc_step;
        u32 use_extra = gpu_rand_below(prng, __coqui_extras_cnt);
        u32 extra_len = __coqui_extras_lens[use_extra];
        if (len + extra_len >= max_len) goto retry_havoc_step;
        u32 insert_at = gpu_rand_below(prng, len + 1);
        u32 e_off = __coqui_extras_offsets[use_extra];
        /* Shift tail right by extra_len to make room. */
        for (u32 i = len; i > insert_at; --i) buf[i - 1 + extra_len] = buf[i - 1];
        /* Insert extras bytes. */
        for (u32 i = 0; i < extra_len; ++i) {
          buf[insert_at + i] = __coqui_extras_base[e_off + i];
        }
        len += extra_len;
        break;
      }

      case MUT_AUTO_EXTRA_OVERWRITE: {
        /* AFL line 3300. */
        if (__coqui_a_extras_cnt == 0) goto retry_havoc_step;
        u32 use_extra = gpu_rand_below(prng, __coqui_a_extras_cnt);
        u32 extra_len = __coqui_a_extras_lens[use_extra];
        if (extra_len > len) goto retry_havoc_step;
        u32 insert_at = gpu_rand_below(prng, len - extra_len + 1);
        u32 e_off = __coqui_a_extras_offsets[use_extra];
        for (u32 i = 0; i < extra_len; ++i) {
          buf[insert_at + i] = __coqui_a_extras_base[e_off + i];
        }
        break;
      }

      case MUT_AUTO_EXTRA_INSERT: {
        /* AFL line 3323. Same shape as EXTRA_INSERT but uses a_extras. */
        if (__coqui_a_extras_cnt == 0) goto retry_havoc_step;
        u32 use_extra = gpu_rand_below(prng, __coqui_a_extras_cnt);
        u32 extra_len = __coqui_a_extras_lens[use_extra];
        if (len + extra_len >= max_len) goto retry_havoc_step;
        u32 insert_at = gpu_rand_below(prng, len + 1);
        u32 e_off = __coqui_a_extras_offsets[use_extra];
        for (u32 i = len; i > insert_at; --i) buf[i - 1 + extra_len] = buf[i - 1];
        for (u32 i = 0; i < extra_len; ++i) {
          buf[insert_at + i] = __coqui_a_extras_base[e_off + i];
        }
        len += extra_len;
        break;
      }

      case MUT_SPLICE_OVERWRITE: {
        /* AFL line 3369. Pick a partner (uniform within pool, reject self
         * and len < 4). copy_len = choose_block_len(new_len - 1), clipped to
         * local len. memmove-style copy — but source is seed_pool (separate
         * region), destination is scratch — no aliasing. */
        u32 partner = weighted_splice_pick(prng, self_idx);
        if (partner == COQUI_SPLICE_SELF_SAME) goto retry_havoc_step;
        u32 new_len = __coqui_seed_pool_lens[partner];
        u32 p_off   = __coqui_seed_pool_offsets[partner];
        /* new_len >= 4 guaranteed by partner selection guard. */
        u32 copy_len = choose_block_len(prng, new_len - 1);
        if (copy_len > len) copy_len = len;
        u32 copy_from = gpu_rand_below(prng, new_len - copy_len + 1);
        u32 copy_to   = gpu_rand_below(prng, len - copy_len + 1);
        for (u32 i = 0; i < copy_len; ++i) {
          buf[copy_to + i] = __coqui_seed_pool_base[p_off + copy_from + i];
        }
        break;
      }

      case MUT_SPLICE_INSERT: {
        /* AFL line 3415. Pick partner, clone_len = choose_block_len(new_len),
         * clone_from random in partner, clone_to random in local, grow local.
         * AFL's `temp_len + HAVOC_BLK_XL >= MAX_FILE` guard adapts to GPU:
         * since max_len is small (~4KB), simplify to "can we grow at all?" */
        u32 partner = weighted_splice_pick(prng, self_idx);
        if (partner == COQUI_SPLICE_SELF_SAME) goto retry_havoc_step;
        if (len + 1 >= max_len) goto retry_havoc_step;
        u32 new_len = __coqui_seed_pool_lens[partner];
        u32 p_off   = __coqui_seed_pool_offsets[partner];
        /* new_len >= 4 guaranteed. */
        u32 clone_len = choose_block_len(prng, new_len);
        /* Clip to fit within max_len. */
        if (clone_len > max_len - len - 1) clone_len = max_len - len - 1;
        if (clone_len == 0) goto retry_havoc_step;
        u32 clone_from = gpu_rand_below(prng, new_len - clone_len + 1);
        u32 clone_to   = gpu_rand_below(prng, len + 1);
        /* Shift local tail right. */
        for (u32 i = len; i > clone_to; --i) buf[i - 1 + clone_len] = buf[i - 1];
        /* Insert from partner. Partner is in seed_pool (separate memory),
         * so no aliasing with our scratch buffer. */
        for (u32 i = 0; i < clone_len; ++i) {
          buf[clone_to + i] = __coqui_seed_pool_base[p_off + clone_from + i];
        }
        len += clone_len;
        break;
      }

      default:
        goto retry_havoc_step;  /* unknown op code — retry */
    }
  }
  return len;
}
