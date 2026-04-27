/* coqui_mutate.c --- GPU-side havoc mutation runtime.
 *
 * Ported from legacy coqui (runtime/coqui_mutate.c) for cuAFL. Each GPU
 * thread independently mutates its input via stacked havoc with a per-thread
 * xorshift64* PRNG. Eliminates host mut_other from the critical path on fast
 * targets and provides 8192-way mutation parallelism.
 *
 * Differences from legacy port:
 *   - DROPS the cmplog (I2S, DICT_INSERT) and seed-pool (SPLICE) mutation
 *     types: cuAFL has no host-side cmplog struct or seed pool ABI yet.
 *     The remaining mutations cover ~80% of legacy havoc behavior.
 *   - Function signature simplified to (input, size, max_size, prng_base).
 *   - Gated at runtime by a __coqui_mutate_enabled global (default 0); the
 *     host writes 1 when AFL_COQUI_GPU_MUTATE is set so disabling is free.
 *
 * Mutation set retained:
 *   FLIP_BIT, INTERESTING_8/16/32, ARITH_8/16/32, RANDOM_BYTE, DELETE,
 *   CLONE_BLOCK, OVERWRITE_BLOCK.
 *
 * No global state — PRNG lives on the stack (→ registers in NVPTX).
 */

#include "coqui_runtime.h"

/* Forward declarations from coqui_libc.c (private to runtime, no header). */
void *__coqui_memcpy(void *dst, const void *src, unsigned long n);
void *__coqui_memmove(void *dst, const void *src, unsigned long n);

/* ===----------------------------------------------------------------------=
 * Runtime gate globals (bound by host via cuModuleGetGlobal).
 *
 * __coqui_mutate_enabled : 1 when AFL_COQUI_GPU_MUTATE was set at startup,
 *                          0 otherwise. The pass emits a runtime branch on
 *                          this value so disabled runs incur only a load+
 *                          branch per kernel invocation.
 * __coqui_mutate_prng_base : 64-bit value mixed with tid to form the
 *                            per-thread PRNG seed. Host re-rolls per batch
 *                            (e.g., from monotonic time) so each batch sees
 *                            a different mutation pattern.
 * ===-------------------------------------------------------------------- */
__attribute__((visibility("default")))
u8 __coqui_mutate_enabled = 0;

__attribute__((visibility("default")))
u64 __coqui_mutate_prng_base = 0xC0FFEE00DECAFBADULL;

/* ===----------------------------------------------------------------------=
 * PRNG (xorshift64*)
 * ===-------------------------------------------------------------------- */

static unsigned long prng_next(unsigned long *state) {
  unsigned long x = *state;
  x ^= x >> 12;
  x ^= x << 25;
  x ^= x >> 27;
  *state = x;
  return x * 0x2545F4914F6CDD1DULL;
}

/* ===----------------------------------------------------------------------=
 * AFL interesting values
 * ===-------------------------------------------------------------------- */

static const signed char interesting_8[] = {-128, -1, 0, 1, 16, 32, 64, 100,
                                             127};

static const short interesting_16[] = {-32768, -129, -128, -1,   0,
                                       1,      127,  128,  255,  256,
                                       512,    1000, 1024, 4096, 32767};

static const int interesting_32[] = {
    (int)-2147483648LL, -100663046, -32769, -32768, -129, -128, -1, 0,
    1, 127, 128, 255, 256, 512, 1000, 1024,
    4096, 32767, 32768, 65535, 65536, 100663045, 2147483647};

#define ARRAY_LEN(a) (sizeof(a) / sizeof((a)[0]))

/* ===----------------------------------------------------------------------=
 * Mutation types (subset of legacy; cmplog/seedpool variants dropped).
 * ===-------------------------------------------------------------------- */

enum {
  MUT_FLIP_BIT,
  MUT_INTERESTING_8,
  MUT_INTERESTING_16,
  MUT_INTERESTING_32,
  MUT_ARITH_8,
  MUT_ARITH_16,
  MUT_ARITH_32,
  MUT_RANDOM_BYTE,
  MUT_DELETE,
  MUT_CLONE_BLOCK,
  MUT_OVERWRITE_BLOCK,
  MUT_COUNT
};

/* ===----------------------------------------------------------------------=
 * Single mutation application
 * ===-------------------------------------------------------------------- */

static void apply_mutation(unsigned char *input, unsigned long *size_ptr,
                           unsigned long max_size, unsigned long *prng) {
  unsigned long size = *size_ptr;
  if (size == 0)
    return;

  unsigned long r = prng_next(prng);
  unsigned int type = r % MUT_COUNT;

  /* Fall back if mutation type is inapplicable. */
  if ((type == MUT_INTERESTING_16 || type == MUT_ARITH_16) && size < 2)
    type = MUT_FLIP_BIT;
  if ((type == MUT_INTERESTING_32 || type == MUT_ARITH_32) && size < 4)
    type = MUT_FLIP_BIT;

  switch (type) {
  case MUT_FLIP_BIT: {
    unsigned long pos = prng_next(prng) % (size * 8);
    input[pos / 8] ^= (unsigned char)(1 << (pos % 8));
    break;
  }
  case MUT_INTERESTING_8: {
    unsigned long pos = prng_next(prng) % size;
    input[pos] = (unsigned char)
        interesting_8[prng_next(prng) % ARRAY_LEN(interesting_8)];
    break;
  }
  case MUT_INTERESTING_16: {
    unsigned long pos = prng_next(prng) % (size - 1);
    short val = interesting_16[prng_next(prng) % ARRAY_LEN(interesting_16)];
    if (prng_next(prng) & 1) {
      input[pos] = (unsigned char)((val >> 8) & 0xFF);
      input[pos + 1] = (unsigned char)(val & 0xFF);
    } else {
      input[pos] = (unsigned char)(val & 0xFF);
      input[pos + 1] = (unsigned char)((val >> 8) & 0xFF);
    }
    break;
  }
  case MUT_INTERESTING_32: {
    unsigned long pos = prng_next(prng) % (size - 3);
    int val = interesting_32[prng_next(prng) % ARRAY_LEN(interesting_32)];
    if (prng_next(prng) & 1) {
      input[pos] = (unsigned char)((val >> 24) & 0xFF);
      input[pos + 1] = (unsigned char)((val >> 16) & 0xFF);
      input[pos + 2] = (unsigned char)((val >> 8) & 0xFF);
      input[pos + 3] = (unsigned char)(val & 0xFF);
    } else {
      input[pos] = (unsigned char)(val & 0xFF);
      input[pos + 1] = (unsigned char)((val >> 8) & 0xFF);
      input[pos + 2] = (unsigned char)((val >> 16) & 0xFF);
      input[pos + 3] = (unsigned char)((val >> 24) & 0xFF);
    }
    break;
  }
  case MUT_ARITH_8: {
    unsigned long pos = prng_next(prng) % size;
    int delta = (int)(prng_next(prng) % 71) - 35;
    input[pos] = (unsigned char)((int)input[pos] + delta);
    break;
  }
  case MUT_ARITH_16: {
    unsigned long pos = prng_next(prng) % (size - 1);
    int delta = (int)(prng_next(prng) % 71) - 35;
    if (prng_next(prng) & 1) {
      short val = (short)((unsigned short)(input[pos] << 8) |
                          (unsigned short)input[pos + 1]);
      val = (short)(val + delta);
      input[pos] = (unsigned char)((val >> 8) & 0xFF);
      input[pos + 1] = (unsigned char)(val & 0xFF);
    } else {
      short val = (short)((unsigned short)input[pos] |
                          ((unsigned short)input[pos + 1] << 8));
      val = (short)(val + delta);
      input[pos] = (unsigned char)(val & 0xFF);
      input[pos + 1] = (unsigned char)((val >> 8) & 0xFF);
    }
    break;
  }
  case MUT_ARITH_32: {
    unsigned long pos = prng_next(prng) % (size - 3);
    int delta = (int)(prng_next(prng) % 71) - 35;
    if (prng_next(prng) & 1) {
      unsigned int val = ((unsigned int)input[pos] << 24) |
                         ((unsigned int)input[pos + 1] << 16) |
                         ((unsigned int)input[pos + 2] << 8) |
                         (unsigned int)input[pos + 3];
      val = (unsigned int)((int)val + delta);
      input[pos] = (unsigned char)((val >> 24) & 0xFF);
      input[pos + 1] = (unsigned char)((val >> 16) & 0xFF);
      input[pos + 2] = (unsigned char)((val >> 8) & 0xFF);
      input[pos + 3] = (unsigned char)(val & 0xFF);
    } else {
      unsigned int val = (unsigned int)input[pos] |
                         ((unsigned int)input[pos + 1] << 8) |
                         ((unsigned int)input[pos + 2] << 16) |
                         ((unsigned int)input[pos + 3] << 24);
      val = (unsigned int)((int)val + delta);
      input[pos] = (unsigned char)(val & 0xFF);
      input[pos + 1] = (unsigned char)((val >> 8) & 0xFF);
      input[pos + 2] = (unsigned char)((val >> 16) & 0xFF);
      input[pos + 3] = (unsigned char)((val >> 24) & 0xFF);
    }
    break;
  }
  case MUT_RANDOM_BYTE: {
    unsigned long pos = prng_next(prng) % size;
    input[pos] = (unsigned char)(prng_next(prng) & 0xFF);
    break;
  }
  case MUT_DELETE: {
    if (size <= 1)
      goto fallback;
    unsigned long del_len = (prng_next(prng) % 16) + 1;
    if (del_len >= size)
      del_len = size - 1;
    unsigned long pos = prng_next(prng) % (size - del_len + 1);
    __coqui_memmove(input + pos, input + pos + del_len,
                    size - pos - del_len);
    *size_ptr = size - del_len;
    return;
  }
  case MUT_CLONE_BLOCK: {
    /* Copy a block from within the input and insert it at another position. */
    if (size < 2)
      goto fallback;
    unsigned long clone_len = (prng_next(prng) % 32) + 1;
    if (clone_len > size)
      clone_len = size;
    if (size + clone_len > max_size)
      clone_len = max_size - size;
    if (clone_len == 0)
      goto fallback;
    unsigned long src = prng_next(prng) % (size - clone_len + 1);
    unsigned long dst = prng_next(prng) % (size + 1);
    unsigned char tmp[32];
    __coqui_memcpy(tmp, input + src, clone_len);
    __coqui_memmove(input + dst + clone_len, input + dst, size - dst);
    __coqui_memcpy(input + dst, tmp, clone_len);
    *size_ptr = size + clone_len;
    return;
  }
  case MUT_OVERWRITE_BLOCK: {
    if (size < 2)
      goto fallback;
    unsigned long block_len = (prng_next(prng) % 32) + 1;
    if (block_len > size)
      block_len = size;
    unsigned long src = prng_next(prng) % (size - block_len + 1);
    unsigned long dst = prng_next(prng) % (size - block_len + 1);
    if (src == dst)
      goto fallback;
    __coqui_memmove(input + dst, input + src, block_len);
    break;
  }
  fallback: {
    /* Guaranteed mutation: flip a random bit. */
    unsigned long bit_pos = prng_next(prng) % (size * 8);
    input[bit_pos >> 3] ^= (unsigned char)(1 << (bit_pos & 7));
    break;
  }
  }
}

/* ===----------------------------------------------------------------------=
 * Public API
 *
 * Returns the post-mutation size (may differ from input `size` after DELETE
 * or CLONE_BLOCK). Mutates `input` in place; caller must guarantee
 * `max_size` bytes of writable buffer.
 *
 * Calls are emitted unconditionally by the FuzzEntry pass; the function
 * itself short-circuits when the runtime gate global is 0, so disabled runs
 * pay only one load + one branch per kernel invocation.
 * ===-------------------------------------------------------------------- */
unsigned long __coqui_mutate_input(unsigned char *input, unsigned long size,
                                   unsigned long max_size) {
  if (__coqui_mutate_enabled == 0 || size == 0 || max_size == 0)
    return size;

  unsigned long tid = (unsigned long)__coqui_fuzz_tid();
  unsigned long prng = __coqui_mutate_prng_base + tid + 1;
  /* Mix once so neighbouring tids don't produce neighbouring streams. */
  (void)prng_next(&prng);

  /* Stacked havoc: 1-4 mutations per call. Lower count preserves input
   * structure better for grammar-sensitive targets (JSON, XML, PNG). */
  unsigned int n = (unsigned int)(prng_next(&prng) % 4) + 1;
  for (unsigned int i = 0; i < n; i++)
    apply_mutation(input, &size, max_size, &prng);

  return size;
}
