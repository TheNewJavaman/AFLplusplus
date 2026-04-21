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
    -2147483648, -100663046, -32769, 32768, 65535, 65536, 100663045, 2147483647
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
