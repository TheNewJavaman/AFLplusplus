/* GPU-compilable standalone port of AFL++'s afl_mutate (havoc mutation).
 *
 * Algorithm-identical to include/afl-mutations.h — same PRNG, same mutation
 * types, same probability weights.  Adapted for NVPTX:
 *   - Self-contained (no AFL headers)
 *   - coqui_mutate_ctx_t replaces afl_state_t
 *   - Scratch buffer passed in (no malloc/realloc on GPU)
 *   - No static variables
 *   - Extras (MUT_EXTRA_*, MUT_AUTO_EXTRA_*) use device-global dictionary
 *     tables uploaded from host; fall through to retry if no dict loaded
 *   - Splice (MUT_SPLICE_*) use the per-batch parent table as splice source;
 *     each thread splices from a randomly chosen other parent in the batch
 *   - MUT_ASCIINUM / MUT_INSERTASCIINUM use GPU-safe integer formatting
 *     (no snprintf/strlen on NVPTX)
 *   - isdigit/memmove replaced with inline GPU-safe equivalents
 */

#ifndef COQUI_AFL_MUTATE_H
#define COQUI_AFL_MUTATE_H

#include <stdint.h>
#include <stddef.h>
#include <string.h>

/* ===================================================================
 * Type aliases (match AFL's include/types.h)
 * When included from coqui_runtime.c, u8/u16/u32/u64 are already
 * defined via coqui_runtime.h — but the signed aliases (s8..s64) are
 * not, so we always define those. Guard the unsigned aliases to avoid
 * redefinition errors.
 * =================================================================== */

#ifndef _COQUI_RUNTIME_H
typedef uint8_t  u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;
#endif
typedef int8_t   s8;
typedef int16_t  s16;
typedef int32_t  s32;
typedef int64_t  s64;

/* ===================================================================
 * Compiler hint macros
 * =================================================================== */

#ifndef likely
#define likely(_x)   __builtin_expect(!!(_x), 1)
#endif
#ifndef unlikely
#define unlikely(_x) __builtin_expect(!!(_x), 0)
#endif

/* ===================================================================
 * Utility macros (from include/types.h)
 * =================================================================== */

#ifndef COQUI_MIN
#define COQUI_MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

#define COQUI_SWAP16(_x)                        \
  ({                                            \
    u16 _ret = (_x);                            \
    (u16)((_ret << 8) | (_ret >> 8));           \
  })

#define COQUI_SWAP32(_x)                                                  \
  ({                                                                     \
    u32 _ret = (_x);                                                     \
    (u32)((_ret << 24) | (_ret >> 24) | ((_ret << 8) & 0x00FF0000) |     \
          ((_ret >> 8) & 0x0000FF00));                                   \
  })

#define COQUI_EXTRACT16(_s, _o)      \
  ({                                 \
    u8 *s = (u8 *)(_s) + (_o);       \
    u16 _ret = s[1];                 \
    _ret = (_ret << 8) | s[0];      \
    _ret;                            \
  })

#define COQUI_EXTRACT32(_s, _o)      \
  ({                                 \
    u8 *s = (u8 *)(_s) + (_o);       \
    u32 _ret = s[3];                 \
    _ret = (_ret << 8) | s[2];      \
    _ret = (_ret << 8) | s[1];      \
    _ret = (_ret << 8) | s[0];      \
    _ret;                            \
  })

#define COQUI_INSERT16(_d, _o, _x)   \
  {                                  \
    u8 *d = (u8 *)(_d) + (_o);       \
    u16 x = _x;                     \
    d[0] = x & 0xFF;                \
    x >>= 8;                        \
    d[1] = x & 0xFF;                \
  }

#define COQUI_INSERT32(_d, _o, _x)   \
  {                                  \
    u8 *d = (u8 *)(_d) + (_o);       \
    u32 x = _x;                     \
    d[0] = x & 0xFF;                \
    x >>= 8;                        \
    d[1] = x & 0xFF;                \
    x >>= 8;                        \
    d[2] = x & 0xFF;                \
    x >>= 8;                        \
    d[3] = x & 0xFF;                \
  }

/* ===================================================================
 * Constants (from include/config.h)
 * =================================================================== */

#define COQUI_HAVOC_BLK_SMALL   32U
#define COQUI_HAVOC_BLK_MEDIUM  128U
#define COQUI_HAVOC_BLK_LARGE   1500U
/* On the GPU, per-thread input slots are fixed-size (default 4096 bytes),
 * so HAVOC_BLK_XL must be smaller than the slot to allow grow mutations.
 * AFL upstream uses 32768 because MAX_FILE is 1MB. With 4096-byte slots,
 * 2048 ensures the clone guard `len + BLK_XL < max_len` passes when
 * len < 2048, giving grow ops room to expand from small seeds. */
#define COQUI_HAVOC_BLK_XL      2048U
#define COQUI_ARITH_MAX          35

/* Dictionary + splice device-global limits. Must match coqui_runtime.c. */
#define COQUI_MAX_EXTRAS       4096U
#define COQUI_MAX_EXTRAS_BYTES 131072U

/* ===================================================================
 * Device globals for dictionary (extras) and splice mutations.
 *
 * These are defined in coqui_runtime.c (device-global storage). The
 * mutation function references them to implement MUT_EXTRA_* and
 * MUT_SPLICE_* cases. The host uploads packed token data before each
 * batch (or when the dictionary changes).
 * =================================================================== */

extern u8  __coqui_extras_data[COQUI_MAX_EXTRAS_BYTES];
extern u32 __coqui_extras_offsets[COQUI_MAX_EXTRAS];
extern u32 __coqui_extras_lens[COQUI_MAX_EXTRAS];
extern u32 __coqui_extras_cnt;
extern u32 __coqui_a_extras_cnt;

/* Parent table (defined in coqui_runtime.c, used for splice). */
extern u8  *__coqui_parent_bytes;
extern u32 *__coqui_parent_offsets;
extern u32 *__coqui_parent_lens;
extern u32 *__coqui_parent_idx;
extern u32  __coqui_parent_count;

/* ===================================================================
 * Interesting value arrays (from include/config.h — exact values)
 * =================================================================== */

static const s8 coqui_interesting_8[] = {
    -128, -1, 0, 1, 16, 32, 64, 100, 127
};

static const s16 coqui_interesting_16[] = {
    /* INTERESTING_8 */
    -128, -1, 0, 1, 16, 32, 64, 100, 127,
    /* INTERESTING_16 */
    -32768, -129, 128, 255, 256, 512, 1000, 1024, 4096, 32767
};

static const s32 coqui_interesting_32[] = {
    /* INTERESTING_8 */
    -128, -1, 0, 1, 16, 32, 64, 100, 127,
    /* INTERESTING_16 */
    -32768, -129, 128, 255, 256, 512, 1000, 1024, 4096, 32767,
    /* INTERESTING_32 */
    -2147483648LL, -100663046, -32769, 32768, 65535, 65536,
    100663045, 2139095040, 2147483647
};

/* ===================================================================
 * Mutation enum (identical to include/afl-mutations.h)
 * =================================================================== */

enum {
    /* 00 */ MUT_FLIPBIT,
    /* 01 */ MUT_INTERESTING8,
    /* 02 */ MUT_INTERESTING16,
    /* 03 */ MUT_INTERESTING16BE,
    /* 04 */ MUT_INTERESTING32,
    /* 05 */ MUT_INTERESTING32BE,
    /* 06 */ MUT_ARITH8_,
    /* 07 */ MUT_ARITH8,
    /* 08 */ MUT_ARITH16_,
    /* 09 */ MUT_ARITH16BE_,
    /* 10 */ MUT_ARITH16,
    /* 11 */ MUT_ARITH16BE,
    /* 12 */ MUT_ARITH32_,
    /* 13 */ MUT_ARITH32BE_,
    /* 14 */ MUT_ARITH32,
    /* 15 */ MUT_ARITH32BE,
    /* 16 */ MUT_RAND8,
    /* 17 */ MUT_CLONE_COPY,
    /* 18 */ MUT_CLONE_FIXED,
    /* 19 */ MUT_OVERWRITE_COPY,
    /* 20 */ MUT_OVERWRITE_FIXED,
    /* 21 */ MUT_BYTEADD,
    /* 22 */ MUT_BYTESUB,
    /* 23 */ MUT_FLIP8,
    /* 24 */ MUT_SWITCH,
    /* 25 */ MUT_DEL,
    /* 26 */ MUT_SHUFFLE,
    /* 27 */ MUT_DELONE,
    /* 28 */ MUT_INSERTONE,
    /* 29 */ MUT_ASCIINUM,
    /* 30 */ MUT_INSERTASCIINUM,
    /* 31 */ MUT_EXTRA_OVERWRITE,
    /* 32 */ MUT_EXTRA_INSERT,
    /* 33 */ MUT_AUTO_EXTRA_OVERWRITE,
    /* 34 */ MUT_AUTO_EXTRA_INSERT,
    /* 35 */ MUT_SPLICE_OVERWRITE,
    /* 36 */ MUT_SPLICE_INSERT,
    MUT_MAX
};

/* ===================================================================
 * Mutation strategy arrays (from include/afl-mutations.h — exact copies)
 * Only exploration_binary is needed for GPU (binary mode, exploration).
 * All four arrays are included for algorithm fidelity.
 * =================================================================== */

#define MUT_STRATEGY_ARRAY_SIZE 256

static const u32 coqui_mutation_strategy_exploration_text[MUT_STRATEGY_ARRAY_SIZE] = {
    MUT_FLIPBIT, MUT_FLIPBIT, MUT_FLIPBIT, MUT_FLIPBIT, MUT_FLIPBIT,
    MUT_FLIPBIT,
    MUT_INTERESTING8, MUT_INTERESTING8, MUT_INTERESTING8, MUT_INTERESTING8,
    MUT_INTERESTING8,
    MUT_INTERESTING16, MUT_INTERESTING16, MUT_INTERESTING16, MUT_INTERESTING16,
    MUT_INTERESTING16,
    MUT_INTERESTING16BE, MUT_INTERESTING16BE, MUT_INTERESTING16BE,
    MUT_INTERESTING16BE, MUT_INTERESTING16BE,
    MUT_INTERESTING32, MUT_INTERESTING32, MUT_INTERESTING32, MUT_INTERESTING32,
    MUT_INTERESTING32,
    MUT_INTERESTING32BE, MUT_INTERESTING32BE, MUT_INTERESTING32BE,
    MUT_INTERESTING32BE, MUT_INTERESTING32BE,
    MUT_ARITH8_, MUT_ARITH8_, MUT_ARITH8_, MUT_ARITH8_, MUT_ARITH8_,
    MUT_ARITH8_,
    MUT_ARITH8, MUT_ARITH8, MUT_ARITH8, MUT_ARITH8, MUT_ARITH8, MUT_ARITH8,
    MUT_ARITH16_, MUT_ARITH16_, MUT_ARITH16_, MUT_ARITH16_, MUT_ARITH16_,
    MUT_ARITH16_,
    MUT_ARITH16BE_, MUT_ARITH16BE_, MUT_ARITH16BE_, MUT_ARITH16BE_,
    MUT_ARITH16BE_, MUT_ARITH16BE_,
    MUT_ARITH16, MUT_ARITH16, MUT_ARITH16, MUT_ARITH16, MUT_ARITH16,
    MUT_ARITH16,
    MUT_ARITH16BE, MUT_ARITH16BE, MUT_ARITH16BE, MUT_ARITH16BE, MUT_ARITH16BE,
    MUT_ARITH16BE,
    MUT_ARITH32_, MUT_ARITH32_, MUT_ARITH32_, MUT_ARITH32_, MUT_ARITH32_,
    MUT_ARITH32_,
    MUT_ARITH32BE_, MUT_ARITH32BE_, MUT_ARITH32BE_, MUT_ARITH32BE_,
    MUT_ARITH32BE_, MUT_ARITH32BE_,
    MUT_ARITH32, MUT_ARITH32, MUT_ARITH32, MUT_ARITH32, MUT_ARITH32,
    MUT_ARITH32,
    MUT_ARITH32BE, MUT_ARITH32BE, MUT_ARITH32BE, MUT_ARITH32BE, MUT_ARITH32BE,
    MUT_ARITH32BE,
    MUT_RAND8, MUT_RAND8, MUT_RAND8, MUT_RAND8, MUT_RAND8, MUT_RAND8,
    MUT_CLONE_COPY, MUT_CLONE_COPY, MUT_CLONE_COPY, MUT_CLONE_COPY,
    MUT_CLONE_COPY, MUT_CLONE_COPY, MUT_CLONE_COPY, MUT_CLONE_COPY,
    MUT_CLONE_COPY, MUT_CLONE_COPY, MUT_CLONE_COPY, MUT_CLONE_COPY,
    MUT_CLONE_COPY,
    MUT_CLONE_FIXED, MUT_CLONE_FIXED, MUT_CLONE_FIXED, MUT_CLONE_FIXED,
    MUT_CLONE_FIXED, MUT_CLONE_FIXED, MUT_CLONE_FIXED, MUT_CLONE_FIXED,
    MUT_CLONE_FIXED, MUT_CLONE_FIXED,
    MUT_OVERWRITE_COPY, MUT_OVERWRITE_COPY, MUT_OVERWRITE_COPY,
    MUT_OVERWRITE_COPY, MUT_OVERWRITE_COPY, MUT_OVERWRITE_COPY,
    MUT_OVERWRITE_COPY, MUT_OVERWRITE_COPY, MUT_OVERWRITE_COPY,
    MUT_OVERWRITE_FIXED, MUT_OVERWRITE_FIXED, MUT_OVERWRITE_FIXED,
    MUT_OVERWRITE_FIXED, MUT_OVERWRITE_FIXED, MUT_OVERWRITE_FIXED,
    MUT_OVERWRITE_FIXED, MUT_OVERWRITE_FIXED,
    MUT_BYTEADD, MUT_BYTEADD, MUT_BYTEADD, MUT_BYTEADD, MUT_BYTEADD,
    MUT_BYTEADD,
    MUT_BYTESUB, MUT_BYTESUB, MUT_BYTESUB, MUT_BYTESUB, MUT_BYTESUB,
    MUT_BYTESUB,
    MUT_FLIP8, MUT_FLIP8, MUT_FLIP8, MUT_FLIP8, MUT_FLIP8, MUT_FLIP8,
    MUT_FLIP8, MUT_FLIP8,
    MUT_SWITCH, MUT_SWITCH, MUT_SWITCH, MUT_SWITCH, MUT_SWITCH, MUT_SWITCH,
    MUT_SWITCH, MUT_SWITCH,
    MUT_DEL, MUT_DEL, MUT_DEL, MUT_DEL, MUT_DEL, MUT_DEL, MUT_DEL,
    MUT_SHUFFLE, MUT_SHUFFLE, MUT_SHUFFLE, MUT_SHUFFLE, MUT_SHUFFLE,
    MUT_SHUFFLE, MUT_SHUFFLE, MUT_SHUFFLE,
    MUT_DELONE, MUT_DELONE, MUT_DELONE, MUT_DELONE, MUT_DELONE, MUT_DELONE,
    MUT_DELONE,
    MUT_INSERTONE, MUT_INSERTONE, MUT_INSERTONE, MUT_INSERTONE, MUT_INSERTONE,
    MUT_INSERTONE,
    MUT_ASCIINUM, MUT_ASCIINUM, MUT_ASCIINUM, MUT_ASCIINUM, MUT_ASCIINUM,
    MUT_ASCIINUM, MUT_ASCIINUM, MUT_ASCIINUM,
    MUT_INSERTASCIINUM, MUT_INSERTASCIINUM, MUT_INSERTASCIINUM,
    MUT_INSERTASCIINUM, MUT_INSERTASCIINUM, MUT_INSERTASCIINUM,
    MUT_INSERTASCIINUM, MUT_INSERTASCIINUM,
    MUT_EXTRA_OVERWRITE, MUT_EXTRA_OVERWRITE, MUT_EXTRA_OVERWRITE,
    MUT_EXTRA_OVERWRITE, MUT_EXTRA_OVERWRITE, MUT_EXTRA_OVERWRITE,
    MUT_EXTRA_OVERWRITE, MUT_EXTRA_OVERWRITE,
    MUT_EXTRA_INSERT, MUT_EXTRA_INSERT, MUT_EXTRA_INSERT, MUT_EXTRA_INSERT,
    MUT_EXTRA_INSERT, MUT_EXTRA_INSERT, MUT_EXTRA_INSERT, MUT_EXTRA_INSERT,
    MUT_AUTO_EXTRA_OVERWRITE, MUT_AUTO_EXTRA_OVERWRITE,
    MUT_AUTO_EXTRA_OVERWRITE, MUT_AUTO_EXTRA_OVERWRITE,
    MUT_AUTO_EXTRA_OVERWRITE, MUT_AUTO_EXTRA_OVERWRITE,
    MUT_AUTO_EXTRA_INSERT, MUT_AUTO_EXTRA_INSERT, MUT_AUTO_EXTRA_INSERT,
    MUT_AUTO_EXTRA_INSERT, MUT_AUTO_EXTRA_INSERT, MUT_AUTO_EXTRA_INSERT,
    MUT_SPLICE_OVERWRITE, MUT_SPLICE_OVERWRITE, MUT_SPLICE_OVERWRITE,
    MUT_SPLICE_OVERWRITE, MUT_SPLICE_OVERWRITE, MUT_SPLICE_OVERWRITE,
    MUT_SPLICE_OVERWRITE, MUT_SPLICE_OVERWRITE, MUT_SPLICE_OVERWRITE,
    MUT_SPLICE_INSERT, MUT_SPLICE_INSERT, MUT_SPLICE_INSERT,
    MUT_SPLICE_INSERT, MUT_SPLICE_INSERT, MUT_SPLICE_INSERT,
    MUT_SPLICE_INSERT, MUT_SPLICE_INSERT, MUT_SPLICE_INSERT, MUT_SPLICE_INSERT
};

static const u32 coqui_mutation_strategy_exploration_binary[MUT_STRATEGY_ARRAY_SIZE] = {
    MUT_FLIPBIT, MUT_FLIPBIT, MUT_FLIPBIT, MUT_FLIPBIT, MUT_FLIPBIT,
    MUT_FLIPBIT, MUT_FLIPBIT,
    MUT_INTERESTING8, MUT_INTERESTING8, MUT_INTERESTING8, MUT_INTERESTING8,
    MUT_INTERESTING8, MUT_INTERESTING8,
    MUT_INTERESTING16, MUT_INTERESTING16, MUT_INTERESTING16, MUT_INTERESTING16,
    MUT_INTERESTING16, MUT_INTERESTING16,
    MUT_INTERESTING16BE, MUT_INTERESTING16BE, MUT_INTERESTING16BE,
    MUT_INTERESTING16BE, MUT_INTERESTING16BE, MUT_INTERESTING16BE,
    MUT_INTERESTING32, MUT_INTERESTING32, MUT_INTERESTING32, MUT_INTERESTING32,
    MUT_INTERESTING32, MUT_INTERESTING32,
    MUT_INTERESTING32BE, MUT_INTERESTING32BE, MUT_INTERESTING32BE,
    MUT_INTERESTING32BE, MUT_INTERESTING32BE, MUT_INTERESTING32BE,
    MUT_ARITH8_, MUT_ARITH8_, MUT_ARITH8_, MUT_ARITH8_, MUT_ARITH8_,
    MUT_ARITH8_, MUT_ARITH8_,
    MUT_ARITH8, MUT_ARITH8, MUT_ARITH8, MUT_ARITH8, MUT_ARITH8, MUT_ARITH8,
    MUT_ARITH8,
    MUT_ARITH16_, MUT_ARITH16_, MUT_ARITH16_, MUT_ARITH16_, MUT_ARITH16_,
    MUT_ARITH16_,
    MUT_ARITH16BE_, MUT_ARITH16BE_, MUT_ARITH16BE_, MUT_ARITH16BE_,
    MUT_ARITH16BE_, MUT_ARITH16BE_,
    MUT_ARITH16, MUT_ARITH16, MUT_ARITH16, MUT_ARITH16, MUT_ARITH16,
    MUT_ARITH16,
    MUT_ARITH16BE, MUT_ARITH16BE, MUT_ARITH16BE, MUT_ARITH16BE, MUT_ARITH16BE,
    MUT_ARITH16BE,
    MUT_ARITH32_, MUT_ARITH32_, MUT_ARITH32_, MUT_ARITH32_, MUT_ARITH32_,
    MUT_ARITH32_,
    MUT_ARITH32BE_, MUT_ARITH32BE_, MUT_ARITH32BE_, MUT_ARITH32BE_,
    MUT_ARITH32BE_, MUT_ARITH32BE_,
    MUT_ARITH32, MUT_ARITH32, MUT_ARITH32, MUT_ARITH32, MUT_ARITH32,
    MUT_ARITH32, MUT_ARITH32,
    MUT_ARITH32BE, MUT_ARITH32BE, MUT_ARITH32BE, MUT_ARITH32BE, MUT_ARITH32BE,
    MUT_ARITH32BE,
    MUT_RAND8, MUT_RAND8, MUT_RAND8, MUT_RAND8, MUT_RAND8, MUT_RAND8,
    MUT_CLONE_COPY, MUT_CLONE_COPY, MUT_CLONE_COPY, MUT_CLONE_COPY,
    MUT_CLONE_COPY, MUT_CLONE_COPY, MUT_CLONE_COPY, MUT_CLONE_COPY,
    MUT_CLONE_COPY, MUT_CLONE_COPY, MUT_CLONE_COPY, MUT_CLONE_COPY,
    MUT_CLONE_COPY, MUT_CLONE_COPY,
    MUT_CLONE_FIXED, MUT_CLONE_FIXED, MUT_CLONE_FIXED, MUT_CLONE_FIXED,
    MUT_CLONE_FIXED, MUT_CLONE_FIXED, MUT_CLONE_FIXED, MUT_CLONE_FIXED,
    MUT_CLONE_FIXED,
    MUT_OVERWRITE_COPY, MUT_OVERWRITE_COPY, MUT_OVERWRITE_COPY,
    MUT_OVERWRITE_COPY, MUT_OVERWRITE_COPY, MUT_OVERWRITE_COPY,
    MUT_OVERWRITE_COPY, MUT_OVERWRITE_COPY, MUT_OVERWRITE_COPY,
    MUT_OVERWRITE_FIXED, MUT_OVERWRITE_FIXED, MUT_OVERWRITE_FIXED,
    MUT_OVERWRITE_FIXED, MUT_OVERWRITE_FIXED, MUT_OVERWRITE_FIXED,
    MUT_OVERWRITE_FIXED,
    MUT_BYTEADD, MUT_BYTEADD, MUT_BYTEADD, MUT_BYTEADD, MUT_BYTEADD,
    MUT_BYTEADD, MUT_BYTEADD,
    MUT_BYTESUB, MUT_BYTESUB, MUT_BYTESUB, MUT_BYTESUB, MUT_BYTESUB,
    MUT_BYTESUB, MUT_BYTESUB,
    MUT_FLIP8, MUT_FLIP8, MUT_FLIP8, MUT_FLIP8, MUT_FLIP8, MUT_FLIP8,
    MUT_FLIP8, MUT_FLIP8, MUT_FLIP8, MUT_FLIP8,
    MUT_SWITCH, MUT_SWITCH, MUT_SWITCH, MUT_SWITCH, MUT_SWITCH, MUT_SWITCH,
    MUT_SWITCH,
    MUT_DEL, MUT_DEL, MUT_DEL, MUT_DEL, MUT_DEL, MUT_DEL,
    MUT_SHUFFLE, MUT_SHUFFLE, MUT_SHUFFLE, MUT_SHUFFLE, MUT_SHUFFLE,
    MUT_SHUFFLE,
    MUT_DELONE, MUT_DELONE, MUT_DELONE, MUT_DELONE, MUT_DELONE, MUT_DELONE,
    MUT_INSERTONE, MUT_INSERTONE, MUT_INSERTONE, MUT_INSERTONE, MUT_INSERTONE,
    MUT_INSERTONE,
    MUT_ASCIINUM, MUT_ASCIINUM, MUT_ASCIINUM, MUT_ASCIINUM, MUT_ASCIINUM,
    MUT_ASCIINUM,
    MUT_INSERTASCIINUM, MUT_INSERTASCIINUM, MUT_INSERTASCIINUM,
    MUT_INSERTASCIINUM, MUT_INSERTASCIINUM, MUT_INSERTASCIINUM,
    MUT_EXTRA_OVERWRITE, MUT_EXTRA_OVERWRITE, MUT_EXTRA_OVERWRITE,
    MUT_EXTRA_OVERWRITE, MUT_EXTRA_OVERWRITE, MUT_EXTRA_OVERWRITE,
    MUT_EXTRA_OVERWRITE,
    MUT_EXTRA_INSERT, MUT_EXTRA_INSERT, MUT_EXTRA_INSERT, MUT_EXTRA_INSERT,
    MUT_EXTRA_INSERT, MUT_EXTRA_INSERT, MUT_EXTRA_INSERT,
    MUT_AUTO_EXTRA_OVERWRITE, MUT_AUTO_EXTRA_OVERWRITE,
    MUT_AUTO_EXTRA_OVERWRITE, MUT_AUTO_EXTRA_OVERWRITE,
    MUT_AUTO_EXTRA_OVERWRITE, MUT_AUTO_EXTRA_OVERWRITE,
    MUT_AUTO_EXTRA_INSERT, MUT_AUTO_EXTRA_INSERT, MUT_AUTO_EXTRA_INSERT,
    MUT_AUTO_EXTRA_INSERT, MUT_AUTO_EXTRA_INSERT, MUT_AUTO_EXTRA_INSERT,
    MUT_SPLICE_OVERWRITE, MUT_SPLICE_OVERWRITE, MUT_SPLICE_OVERWRITE,
    MUT_SPLICE_OVERWRITE, MUT_SPLICE_OVERWRITE, MUT_SPLICE_OVERWRITE,
    MUT_SPLICE_OVERWRITE, MUT_SPLICE_OVERWRITE,
    MUT_SPLICE_INSERT, MUT_SPLICE_INSERT, MUT_SPLICE_INSERT,
    MUT_SPLICE_INSERT, MUT_SPLICE_INSERT, MUT_SPLICE_INSERT,
    MUT_SPLICE_INSERT, MUT_SPLICE_INSERT, MUT_SPLICE_INSERT, MUT_SPLICE_INSERT
};

static const u32 coqui_mutation_strategy_exploitation_text[MUT_STRATEGY_ARRAY_SIZE] = {
    MUT_FLIPBIT, MUT_FLIPBIT, MUT_FLIPBIT, MUT_FLIPBIT, MUT_FLIPBIT,
    MUT_FLIPBIT, MUT_FLIPBIT,
    MUT_INTERESTING8, MUT_INTERESTING8, MUT_INTERESTING8, MUT_INTERESTING8,
    MUT_INTERESTING8, MUT_INTERESTING8, MUT_INTERESTING8,
    MUT_INTERESTING16, MUT_INTERESTING16, MUT_INTERESTING16, MUT_INTERESTING16,
    MUT_INTERESTING16, MUT_INTERESTING16, MUT_INTERESTING16,
    MUT_INTERESTING16BE, MUT_INTERESTING16BE, MUT_INTERESTING16BE,
    MUT_INTERESTING16BE, MUT_INTERESTING16BE, MUT_INTERESTING16BE,
    MUT_INTERESTING16BE,
    MUT_INTERESTING32, MUT_INTERESTING32, MUT_INTERESTING32, MUT_INTERESTING32,
    MUT_INTERESTING32, MUT_INTERESTING32, MUT_INTERESTING32, MUT_INTERESTING32,
    MUT_INTERESTING32BE, MUT_INTERESTING32BE, MUT_INTERESTING32BE,
    MUT_INTERESTING32BE, MUT_INTERESTING32BE, MUT_INTERESTING32BE,
    MUT_INTERESTING32BE, MUT_INTERESTING32BE,
    MUT_ARITH8_, MUT_ARITH8_, MUT_ARITH8_, MUT_ARITH8_, MUT_ARITH8_,
    MUT_ARITH8_,
    MUT_ARITH8, MUT_ARITH8, MUT_ARITH8, MUT_ARITH8, MUT_ARITH8, MUT_ARITH8,
    MUT_ARITH8,
    MUT_ARITH16_, MUT_ARITH16_, MUT_ARITH16_, MUT_ARITH16_, MUT_ARITH16_,
    MUT_ARITH16_,
    MUT_ARITH16BE_, MUT_ARITH16BE_, MUT_ARITH16BE_, MUT_ARITH16BE_,
    MUT_ARITH16BE_, MUT_ARITH16BE_, MUT_ARITH16BE_,
    MUT_ARITH16, MUT_ARITH16, MUT_ARITH16, MUT_ARITH16, MUT_ARITH16,
    MUT_ARITH16, MUT_ARITH16,
    MUT_ARITH16BE, MUT_ARITH16BE, MUT_ARITH16BE, MUT_ARITH16BE, MUT_ARITH16BE,
    MUT_ARITH16BE, MUT_ARITH16BE,
    MUT_ARITH32_, MUT_ARITH32_, MUT_ARITH32_, MUT_ARITH32_, MUT_ARITH32_,
    MUT_ARITH32_,
    MUT_ARITH32BE_, MUT_ARITH32BE_, MUT_ARITH32BE_, MUT_ARITH32BE_,
    MUT_ARITH32BE_, MUT_ARITH32BE_,
    MUT_ARITH32, MUT_ARITH32, MUT_ARITH32, MUT_ARITH32, MUT_ARITH32,
    MUT_ARITH32,
    MUT_ARITH32BE, MUT_ARITH32BE, MUT_ARITH32BE, MUT_ARITH32BE, MUT_ARITH32BE,
    MUT_ARITH32BE, MUT_ARITH32BE,
    MUT_RAND8, MUT_RAND8, MUT_RAND8, MUT_RAND8, MUT_RAND8, MUT_RAND8,
    MUT_RAND8,
    MUT_CLONE_COPY, MUT_CLONE_COPY, MUT_CLONE_COPY, MUT_CLONE_COPY,
    MUT_CLONE_COPY, MUT_CLONE_COPY, MUT_CLONE_COPY, MUT_CLONE_COPY,
    MUT_CLONE_COPY,
    MUT_CLONE_FIXED, MUT_CLONE_FIXED, MUT_CLONE_FIXED, MUT_CLONE_FIXED,
    MUT_CLONE_FIXED, MUT_CLONE_FIXED, MUT_CLONE_FIXED, MUT_CLONE_FIXED,
    MUT_OVERWRITE_COPY, MUT_OVERWRITE_COPY, MUT_OVERWRITE_COPY,
    MUT_OVERWRITE_COPY, MUT_OVERWRITE_COPY, MUT_OVERWRITE_COPY,
    MUT_OVERWRITE_FIXED, MUT_OVERWRITE_FIXED, MUT_OVERWRITE_FIXED,
    MUT_OVERWRITE_FIXED, MUT_OVERWRITE_FIXED, MUT_OVERWRITE_FIXED,
    MUT_BYTEADD, MUT_BYTEADD, MUT_BYTEADD, MUT_BYTEADD, MUT_BYTEADD,
    MUT_BYTEADD, MUT_BYTEADD,
    MUT_BYTESUB, MUT_BYTESUB, MUT_BYTESUB, MUT_BYTESUB, MUT_BYTESUB,
    MUT_BYTESUB,
    MUT_FLIP8, MUT_FLIP8, MUT_FLIP8, MUT_FLIP8, MUT_FLIP8, MUT_FLIP8,
    MUT_FLIP8, MUT_FLIP8, MUT_FLIP8,
    MUT_SWITCH, MUT_SWITCH, MUT_SWITCH, MUT_SWITCH, MUT_SWITCH, MUT_SWITCH,
    MUT_SWITCH,
    MUT_DEL, MUT_DEL, MUT_DEL, MUT_DEL, MUT_DEL, MUT_DEL,
    MUT_SHUFFLE, MUT_SHUFFLE, MUT_SHUFFLE, MUT_SHUFFLE, MUT_SHUFFLE,
    MUT_SHUFFLE, MUT_SHUFFLE,
    MUT_DELONE, MUT_DELONE, MUT_DELONE, MUT_DELONE, MUT_DELONE,
    MUT_INSERTONE, MUT_INSERTONE, MUT_INSERTONE, MUT_INSERTONE, MUT_INSERTONE,
    MUT_INSERTONE,
    MUT_ASCIINUM, MUT_ASCIINUM, MUT_ASCIINUM, MUT_ASCIINUM, MUT_ASCIINUM,
    MUT_ASCIINUM,
    MUT_INSERTASCIINUM, MUT_INSERTASCIINUM, MUT_INSERTASCIINUM,
    MUT_INSERTASCIINUM, MUT_INSERTASCIINUM, MUT_INSERTASCIINUM,
    MUT_INSERTASCIINUM, MUT_INSERTASCIINUM,
    MUT_EXTRA_OVERWRITE, MUT_EXTRA_OVERWRITE, MUT_EXTRA_OVERWRITE,
    MUT_EXTRA_OVERWRITE, MUT_EXTRA_OVERWRITE, MUT_EXTRA_OVERWRITE,
    MUT_EXTRA_OVERWRITE,
    MUT_EXTRA_INSERT, MUT_EXTRA_INSERT, MUT_EXTRA_INSERT, MUT_EXTRA_INSERT,
    MUT_EXTRA_INSERT, MUT_EXTRA_INSERT, MUT_EXTRA_INSERT,
    MUT_AUTO_EXTRA_OVERWRITE, MUT_AUTO_EXTRA_OVERWRITE,
    MUT_AUTO_EXTRA_OVERWRITE, MUT_AUTO_EXTRA_OVERWRITE,
    MUT_AUTO_EXTRA_OVERWRITE, MUT_AUTO_EXTRA_OVERWRITE,
    MUT_AUTO_EXTRA_OVERWRITE,
    MUT_AUTO_EXTRA_INSERT, MUT_AUTO_EXTRA_INSERT, MUT_AUTO_EXTRA_INSERT,
    MUT_AUTO_EXTRA_INSERT, MUT_AUTO_EXTRA_INSERT, MUT_AUTO_EXTRA_INSERT,
    MUT_AUTO_EXTRA_INSERT,
    MUT_SPLICE_OVERWRITE, MUT_SPLICE_OVERWRITE, MUT_SPLICE_OVERWRITE,
    MUT_SPLICE_OVERWRITE, MUT_SPLICE_OVERWRITE, MUT_SPLICE_OVERWRITE,
    MUT_SPLICE_OVERWRITE,
    MUT_SPLICE_INSERT, MUT_SPLICE_INSERT, MUT_SPLICE_INSERT,
    MUT_SPLICE_INSERT, MUT_SPLICE_INSERT, MUT_SPLICE_INSERT,
    MUT_SPLICE_INSERT, MUT_SPLICE_INSERT, MUT_SPLICE_INSERT
};

static const u32 coqui_mutation_strategy_exploitation_binary[MUT_STRATEGY_ARRAY_SIZE] = {
    MUT_FLIPBIT, MUT_FLIPBIT, MUT_FLIPBIT, MUT_FLIPBIT, MUT_FLIPBIT,
    MUT_FLIPBIT, MUT_FLIPBIT,
    MUT_INTERESTING8, MUT_INTERESTING8, MUT_INTERESTING8, MUT_INTERESTING8,
    MUT_INTERESTING8, MUT_INTERESTING8, MUT_INTERESTING8,
    MUT_INTERESTING16, MUT_INTERESTING16, MUT_INTERESTING16, MUT_INTERESTING16,
    MUT_INTERESTING16, MUT_INTERESTING16, MUT_INTERESTING16,
    MUT_INTERESTING16BE, MUT_INTERESTING16BE, MUT_INTERESTING16BE,
    MUT_INTERESTING16BE, MUT_INTERESTING16BE, MUT_INTERESTING16BE,
    MUT_INTERESTING16BE,
    MUT_INTERESTING32, MUT_INTERESTING32, MUT_INTERESTING32, MUT_INTERESTING32,
    MUT_INTERESTING32, MUT_INTERESTING32, MUT_INTERESTING32, MUT_INTERESTING32,
    MUT_INTERESTING32BE, MUT_INTERESTING32BE, MUT_INTERESTING32BE,
    MUT_INTERESTING32BE, MUT_INTERESTING32BE, MUT_INTERESTING32BE,
    MUT_INTERESTING32BE, MUT_INTERESTING32BE,
    MUT_ARITH8_, MUT_ARITH8_, MUT_ARITH8_, MUT_ARITH8_, MUT_ARITH8_,
    MUT_ARITH8_,
    MUT_ARITH8, MUT_ARITH8, MUT_ARITH8, MUT_ARITH8, MUT_ARITH8, MUT_ARITH8,
    MUT_ARITH8,
    MUT_ARITH16_, MUT_ARITH16_, MUT_ARITH16_, MUT_ARITH16_, MUT_ARITH16_,
    MUT_ARITH16_,
    MUT_ARITH16BE_, MUT_ARITH16BE_, MUT_ARITH16BE_, MUT_ARITH16BE_,
    MUT_ARITH16BE_, MUT_ARITH16BE_, MUT_ARITH16BE_,
    MUT_ARITH16, MUT_ARITH16, MUT_ARITH16, MUT_ARITH16, MUT_ARITH16,
    MUT_ARITH16, MUT_ARITH16,
    MUT_ARITH16BE, MUT_ARITH16BE, MUT_ARITH16BE, MUT_ARITH16BE, MUT_ARITH16BE,
    MUT_ARITH16BE, MUT_ARITH16BE,
    MUT_ARITH32_, MUT_ARITH32_, MUT_ARITH32_, MUT_ARITH32_, MUT_ARITH32_,
    MUT_ARITH32_,
    MUT_ARITH32BE_, MUT_ARITH32BE_, MUT_ARITH32BE_, MUT_ARITH32BE_,
    MUT_ARITH32BE_, MUT_ARITH32BE_, MUT_ARITH32BE_,
    MUT_ARITH32, MUT_ARITH32, MUT_ARITH32, MUT_ARITH32, MUT_ARITH32,
    MUT_ARITH32, MUT_ARITH32,
    MUT_ARITH32BE, MUT_ARITH32BE, MUT_ARITH32BE, MUT_ARITH32BE, MUT_ARITH32BE,
    MUT_ARITH32BE, MUT_ARITH32BE,
    MUT_RAND8, MUT_RAND8, MUT_RAND8, MUT_RAND8, MUT_RAND8, MUT_RAND8,
    MUT_RAND8,
    MUT_CLONE_COPY, MUT_CLONE_COPY, MUT_CLONE_COPY, MUT_CLONE_COPY,
    MUT_CLONE_COPY, MUT_CLONE_COPY, MUT_CLONE_COPY, MUT_CLONE_COPY,
    MUT_CLONE_COPY,
    MUT_CLONE_FIXED, MUT_CLONE_FIXED, MUT_CLONE_FIXED, MUT_CLONE_FIXED,
    MUT_CLONE_FIXED, MUT_CLONE_FIXED, MUT_CLONE_FIXED, MUT_CLONE_FIXED,
    MUT_OVERWRITE_COPY, MUT_OVERWRITE_COPY, MUT_OVERWRITE_COPY,
    MUT_OVERWRITE_COPY, MUT_OVERWRITE_COPY, MUT_OVERWRITE_COPY,
    MUT_OVERWRITE_FIXED, MUT_OVERWRITE_FIXED, MUT_OVERWRITE_FIXED,
    MUT_OVERWRITE_FIXED, MUT_OVERWRITE_FIXED, MUT_OVERWRITE_FIXED,
    MUT_BYTEADD, MUT_BYTEADD, MUT_BYTEADD, MUT_BYTEADD, MUT_BYTEADD,
    MUT_BYTEADD, MUT_BYTEADD,
    MUT_BYTESUB, MUT_BYTESUB, MUT_BYTESUB, MUT_BYTESUB, MUT_BYTESUB,
    MUT_BYTESUB,
    MUT_FLIP8, MUT_FLIP8, MUT_FLIP8, MUT_FLIP8, MUT_FLIP8, MUT_FLIP8,
    MUT_FLIP8, MUT_FLIP8, MUT_FLIP8, MUT_FLIP8,
    MUT_SWITCH, MUT_SWITCH, MUT_SWITCH, MUT_SWITCH, MUT_SWITCH, MUT_SWITCH,
    MUT_SWITCH,
    MUT_DEL, MUT_DEL, MUT_DEL, MUT_DEL, MUT_DEL, MUT_DEL,
    MUT_SHUFFLE, MUT_SHUFFLE, MUT_SHUFFLE, MUT_SHUFFLE, MUT_SHUFFLE,
    MUT_SHUFFLE,
    MUT_DELONE, MUT_DELONE, MUT_DELONE, MUT_DELONE, MUT_DELONE,
    MUT_INSERTONE, MUT_INSERTONE, MUT_INSERTONE, MUT_INSERTONE, MUT_INSERTONE,
    MUT_INSERTONE,
    MUT_ASCIINUM, MUT_ASCIINUM, MUT_ASCIINUM, MUT_ASCIINUM, MUT_ASCIINUM,
    MUT_INSERTASCIINUM, MUT_INSERTASCIINUM, MUT_INSERTASCIINUM,
    MUT_INSERTASCIINUM, MUT_INSERTASCIINUM, MUT_INSERTASCIINUM,
    MUT_INSERTASCIINUM,
    MUT_EXTRA_OVERWRITE, MUT_EXTRA_OVERWRITE, MUT_EXTRA_OVERWRITE,
    MUT_EXTRA_OVERWRITE, MUT_EXTRA_OVERWRITE, MUT_EXTRA_OVERWRITE,
    MUT_EXTRA_OVERWRITE,
    MUT_EXTRA_INSERT, MUT_EXTRA_INSERT, MUT_EXTRA_INSERT, MUT_EXTRA_INSERT,
    MUT_EXTRA_INSERT, MUT_EXTRA_INSERT, MUT_EXTRA_INSERT,
    MUT_AUTO_EXTRA_OVERWRITE, MUT_AUTO_EXTRA_OVERWRITE,
    MUT_AUTO_EXTRA_OVERWRITE, MUT_AUTO_EXTRA_OVERWRITE,
    MUT_AUTO_EXTRA_OVERWRITE, MUT_AUTO_EXTRA_OVERWRITE,
    MUT_AUTO_EXTRA_OVERWRITE,
    MUT_AUTO_EXTRA_INSERT, MUT_AUTO_EXTRA_INSERT, MUT_AUTO_EXTRA_INSERT,
    MUT_AUTO_EXTRA_INSERT, MUT_AUTO_EXTRA_INSERT, MUT_AUTO_EXTRA_INSERT,
    MUT_AUTO_EXTRA_INSERT,
    MUT_SPLICE_OVERWRITE, MUT_SPLICE_OVERWRITE, MUT_SPLICE_OVERWRITE,
    MUT_SPLICE_OVERWRITE, MUT_SPLICE_OVERWRITE, MUT_SPLICE_OVERWRITE,
    MUT_SPLICE_OVERWRITE,
    MUT_SPLICE_INSERT, MUT_SPLICE_INSERT, MUT_SPLICE_INSERT,
    MUT_SPLICE_INSERT, MUT_SPLICE_INSERT, MUT_SPLICE_INSERT,
    MUT_SPLICE_INSERT, MUT_SPLICE_INSERT, MUT_SPLICE_INSERT
};

/* ===================================================================
 * PRNG state (replaces afl_state_t for mutation purposes)
 * =================================================================== */

typedef struct {
    u64 rand_seed[2];
} coqui_mutate_ctx_t;

/* ===================================================================
 * PRNG: romuDuoJr (identical to src/afl-performance.c:32)
 * =================================================================== */

static inline __attribute__((always_inline))
u64 coqui_rand_next(coqui_mutate_ctx_t *ctx) {
    u64 xp = ctx->rand_seed[0];
    ctx->rand_seed[0] = 15241094284759029579ULL * ctx->rand_seed[1];
    ctx->rand_seed[1] = ctx->rand_seed[1] - xp;
    ctx->rand_seed[1] = (ctx->rand_seed[1] << 27) | (ctx->rand_seed[1] >> 37);
    return xp;
}

/* ===================================================================
 * rand_below: unbiased modulo (identical to include/afl-fuzz.h:1456)
 * NOTE: /dev/urandom reseeding skipped (not available on GPU)
 * =================================================================== */

static inline __attribute__((always_inline))
u32 coqui_rand_below(coqui_mutate_ctx_t *ctx, u32 limit) {
    if (limit <= 1) return 0;

    /* Modulo is biased - we don't want our fuzzing to be biased so let's
       do it right. Same rejection-sampling loop as AFL++. */
    u64 unbiased_rnd;
    do {
        unbiased_rnd = coqui_rand_next(ctx);
    } while (unbiased_rnd >= (UINT64_MAX - (UINT64_MAX % limit)));

    return unbiased_rnd % limit;
}

/* ===================================================================
 * GPU-safe helpers (replace libc functions not available on NVPTX)
 * =================================================================== */

static inline __attribute__((always_inline))
int coqui_isdigit(u8 c) {
    return c >= '0' && c <= '9';
}

/* GPU-safe memmove: handles overlapping regions */
static inline __attribute__((always_inline))
void coqui_memmove(u8 *dst, const u8 *src, u32 n) {
    if (dst < src) {
        for (u32 i = 0; i < n; i++)
            dst[i] = src[i];
    } else if (dst > src) {
        for (u32 i = n; i > 0; i--)
            dst[i - 1] = src[i - 1];
    }
}

/* GPU-safe s64 to decimal string. Returns length written.
 * buf must have room for at least 21 bytes (incl sign, no NUL). */
static inline __attribute__((always_inline))
u32 coqui_s64_to_str(s64 val, u8 *out) {
    u8 tmp[21];
    u32 pos = 0;
    int neg = 0;
    u64 uval;

    if (val < 0) {
        neg = 1;
        /* Handle INT64_MIN carefully: -(INT64_MIN) overflows s64.
           Cast to u64 first, then negate via unsigned arithmetic. */
        uval = (u64)(-(val + 1)) + 1;
    } else {
        uval = (u64)val;
    }

    /* Generate digits in reverse order */
    do {
        tmp[pos++] = '0' + (u8)(uval % 10);
        uval /= 10;
    } while (uval > 0);

    u32 len = 0;
    if (neg) out[len++] = '-';

    /* Reverse the digits into output */
    for (u32 i = pos; i > 0; i--)
        out[len++] = tmp[i - 1];

    return len;
}

/* GPU-safe u64 to decimal string. Returns length written. */
static inline __attribute__((always_inline))
u32 coqui_u64_to_str(u64 val, u8 *out) {
    u8 tmp[21];
    u32 pos = 0;

    do {
        tmp[pos++] = '0' + (u8)(val % 10);
        val /= 10;
    } while (val > 0);

    /* Reverse into output */
    u32 len = 0;
    for (u32 i = pos; i > 0; i--)
        out[len++] = tmp[i - 1];

    return len;
}

/* ===================================================================
 * choose_block_len (from include/afl-fuzz.h:1760)
 *
 * Adapted: uses coqui_rand_below instead of rand_below(afl, ...).
 * queue_cycle/run_over10m not available on GPU; we hardcode rlim=3
 * (mature fuzzer behavior — same as afl when queue_cycle >= 3).
 * =================================================================== */

static inline __attribute__((always_inline))
u32 coqui_choose_block_len(coqui_mutate_ctx_t *ctx, u32 limit) {

    u32 min_value, max_value;

    switch (coqui_rand_below(ctx, 3)) {

        case 0:
            min_value = 1;
            max_value = COQUI_HAVOC_BLK_SMALL;
            break;

        case 1:
            min_value = COQUI_HAVOC_BLK_SMALL;
            max_value = COQUI_HAVOC_BLK_MEDIUM;
            break;

        default:
            if (likely(coqui_rand_below(ctx, 10))) {
                min_value = COQUI_HAVOC_BLK_MEDIUM;
                max_value = COQUI_HAVOC_BLK_LARGE;
            } else {
                min_value = COQUI_HAVOC_BLK_LARGE;
                max_value = COQUI_HAVOC_BLK_XL;
            }

    }

    if (min_value >= limit) { min_value = 1; }

    return min_value + coqui_rand_below(ctx, COQUI_MIN(max_value, limit) - min_value + 1);
}

/* ===================================================================
 * Main mutation function
 *
 * Adapted from include/afl-mutations.h:1801 (afl_mutate).
 *
 * Parameters:
 *   ctx     - PRNG state (replaces afl_state_t)
 *   buf     - input buffer to mutate in-place (must be >= max_len)
 *   len     - current length of input data
 *   steps   - number of mutation steps to apply
 *   scratch - temporary buffer for insert/clone ops (must be >= max_len)
 *   max_len - maximum allowed output length
 *
 * Simplifications vs AFL:
 *   - is_text=false, is_exploration=true hardcoded (binary exploration)
 *     Caller can swap mutation_array pointer for other strategies.
 *   - Dictionary uses device-global arrays (host-uploaded); falls through
 *     to retry when no dictionary is loaded (extras_cnt == 0)
 *   - Splice uses the batch parent table; falls through when <= 1 parent
 *   - scratch buffer replaces static tmp_buf (no malloc on GPU)
 *
 * Returns: new length of buf after mutation, or 0 on error.
 * =================================================================== */

static inline __attribute__((always_inline))
u32 coqui_afl_mutate(coqui_mutate_ctx_t *ctx,
                     u8 *buf, u32 len, u32 steps,
                     u8 *scratch, u32 max_len) {

    if (!buf || !len) { return 0; }

    /* Default: binary exploration strategy. */
    const u32 *mutation_array = coqui_mutation_strategy_exploration_binary;

    for (u32 step = 0; step < steps; ++step) {

    retry_havoc_step: {

        u32 r = coqui_rand_below(ctx, MUT_STRATEGY_ARRAY_SIZE), item;

        switch (mutation_array[r]) {

            case MUT_FLIPBIT: {

                /* Flip a single bit somewhere. Spooky! */
                u8  bit = coqui_rand_below(ctx, 8);
                u32 off = coqui_rand_below(ctx, len);
                buf[off] ^= 1 << bit;

                break;

            }

            case MUT_INTERESTING8: {

                /* Set byte to interesting value. */
                item = coqui_rand_below(ctx, sizeof(coqui_interesting_8));
                buf[coqui_rand_below(ctx, len)] = coqui_interesting_8[item];
                break;

            }

            case MUT_INTERESTING16: {

                /* Set word to interesting value, little endian. */
                if (unlikely(len < 2)) { break; }

                item = coqui_rand_below(ctx, sizeof(coqui_interesting_16) >> 1);
                COQUI_INSERT16(buf, coqui_rand_below(ctx, len - 1),
                              coqui_interesting_16[item]);

                break;

            }

            case MUT_INTERESTING16BE: {

                /* Set word to interesting value, big endian. */
                if (unlikely(len < 2)) { break; }

                item = coqui_rand_below(ctx, sizeof(coqui_interesting_16) >> 1);
                COQUI_INSERT16(buf, coqui_rand_below(ctx, len - 1),
                              COQUI_SWAP16(coqui_interesting_16[item]));

                break;

            }

            case MUT_INTERESTING32: {

                /* Set dword to interesting value, little endian. */
                if (unlikely(len < 4)) { break; }

                item = coqui_rand_below(ctx, sizeof(coqui_interesting_32) >> 2);
                COQUI_INSERT32(buf, coqui_rand_below(ctx, len - 3),
                              coqui_interesting_32[item]);

                break;

            }

            case MUT_INTERESTING32BE: {

                /* Set dword to interesting value, big endian. */
                if (unlikely(len < 4)) { break; }

                item = coqui_rand_below(ctx, sizeof(coqui_interesting_32) >> 2);
                COQUI_INSERT32(buf, coqui_rand_below(ctx, len - 3),
                              COQUI_SWAP32(coqui_interesting_32[item]));

                break;

            }

            case MUT_ARITH8_: {

                /* Randomly subtract from byte. */
                item = 1 + coqui_rand_below(ctx, COQUI_ARITH_MAX);
                buf[coqui_rand_below(ctx, len)] -= item;
                break;

            }

            case MUT_ARITH8: {

                /* Randomly add to byte. */
                item = 1 + coqui_rand_below(ctx, COQUI_ARITH_MAX);
                buf[coqui_rand_below(ctx, len)] += item;
                break;

            }

            case MUT_ARITH16_: {

                /* Randomly subtract from word, little endian. */
                if (unlikely(len < 2)) { break; }

                u32 pos = coqui_rand_below(ctx, len - 1);
                item = 1 + coqui_rand_below(ctx, COQUI_ARITH_MAX);
                COQUI_INSERT16(buf, pos, COQUI_EXTRACT16(buf, pos) - item);

                break;

            }

            case MUT_ARITH16BE_: {

                /* Randomly subtract from word, big endian. */
                if (unlikely(len < 2)) { break; }

                u32 pos = coqui_rand_below(ctx, len - 1);
                item = 1 + coqui_rand_below(ctx, COQUI_ARITH_MAX);
                COQUI_INSERT16(buf, pos,
                    COQUI_SWAP16(COQUI_SWAP16(COQUI_EXTRACT16(buf, pos)) - item));

                break;

            }

            case MUT_ARITH16: {

                /* Randomly add to word, little endian. */
                if (unlikely(len < 2)) { break; }

                u32 pos = coqui_rand_below(ctx, len - 1);
                item = 1 + coqui_rand_below(ctx, COQUI_ARITH_MAX);
                COQUI_INSERT16(buf, pos, COQUI_EXTRACT16(buf, pos) + item);

                break;

            }

            case MUT_ARITH16BE: {

                /* Randomly add to word, big endian. */
                if (unlikely(len < 2)) { break; }

                u32 pos = coqui_rand_below(ctx, len - 1);
                item = 1 + coqui_rand_below(ctx, COQUI_ARITH_MAX);
                COQUI_INSERT16(buf, pos,
                    COQUI_SWAP16(COQUI_SWAP16(COQUI_EXTRACT16(buf, pos)) + item));

                break;

            }

            case MUT_ARITH32_: {

                /* Randomly subtract from dword, little endian. */
                if (unlikely(len < 4)) { break; }

                u32 pos = coqui_rand_below(ctx, len - 3);
                item = 1 + coqui_rand_below(ctx, COQUI_ARITH_MAX);
                COQUI_INSERT32(buf, pos, COQUI_EXTRACT32(buf, pos) - item);

                break;

            }

            case MUT_ARITH32BE_: {

                /* Randomly subtract from dword, big endian. */
                if (unlikely(len < 4)) { break; }

                u32 pos = coqui_rand_below(ctx, len - 3);
                item = 1 + coqui_rand_below(ctx, COQUI_ARITH_MAX);
                COQUI_INSERT32(buf, pos,
                    COQUI_SWAP32(COQUI_SWAP32(COQUI_EXTRACT32(buf, pos)) - item));

                break;

            }

            case MUT_ARITH32: {

                /* Randomly add to dword, little endian. */
                if (unlikely(len < 4)) { break; }

                u32 pos = coqui_rand_below(ctx, len - 3);
                item = 1 + coqui_rand_below(ctx, COQUI_ARITH_MAX);
                COQUI_INSERT32(buf, pos, COQUI_EXTRACT32(buf, pos) + item);

                break;

            }

            case MUT_ARITH32BE: {

                /* Randomly add to dword, big endian. */
                if (unlikely(len < 4)) { break; }

                u32 pos = coqui_rand_below(ctx, len - 3);
                item = 1 + coqui_rand_below(ctx, COQUI_ARITH_MAX);
                COQUI_INSERT32(buf, pos,
                    COQUI_SWAP32(COQUI_SWAP32(COQUI_EXTRACT32(buf, pos) + item)));

                break;

            }

            case MUT_RAND8: {

                /* Just set a random byte to a random value. Because,
                   why not. We use XOR with 1-255 to eliminate the
                   possibility of a no-op. */
                u32 pos = coqui_rand_below(ctx, len);
                item = 1 + coqui_rand_below(ctx, 255);
                buf[pos] ^= item;
                break;

            }

            case MUT_CLONE_COPY: {

                if (likely(len + COQUI_HAVOC_BLK_XL < max_len)) {

                    /* Clone bytes. */
                    u32 clone_len = coqui_choose_block_len(ctx, len);
                    u32 clone_from = coqui_rand_below(ctx, len - clone_len + 1);
                    u32 clone_to = coqui_rand_below(ctx, len);

                    /* Head */
                    memcpy(scratch, buf, clone_to);

                    /* Inserted part */
                    memcpy(scratch + clone_to, buf + clone_from, clone_len);

                    /* Tail */
                    memcpy(scratch + clone_to + clone_len, buf + clone_to,
                           len - clone_to);

                    len += clone_len;
                    memcpy(buf, scratch, len);

                } else if (unlikely(len < 8)) {

                    break;

                } else {

                    goto retry_havoc_step;

                }

                break;

            }

            case MUT_CLONE_FIXED: {

                if (likely(len + COQUI_HAVOC_BLK_XL < max_len)) {

                    /* Insert a block of constant bytes (25%). */
                    u32 clone_len = coqui_choose_block_len(ctx, COQUI_HAVOC_BLK_XL);
                    u32 clone_to = coqui_rand_below(ctx, len);
                    u32 strat = coqui_rand_below(ctx, 2);
                    u32 clone_from = clone_to ? clone_to - 1 : 0;
                    item = strat ? coqui_rand_below(ctx, 256) : buf[clone_from];

                    /* Head */
                    memcpy(scratch, buf, clone_to);

                    /* Inserted part */
                    memset(scratch + clone_to, item, clone_len);

                    /* Tail */
                    memcpy(scratch + clone_to + clone_len, buf + clone_to,
                           len - clone_to);

                    len += clone_len;
                    memcpy(buf, scratch, len);

                } else if (unlikely(len < 8)) {

                    break;

                } else {

                    goto retry_havoc_step;

                }

                break;

            }

            case MUT_OVERWRITE_COPY: {

                /* Overwrite bytes with a randomly selected chunk bytes. */
                if (unlikely(len < 2)) { break; }

                u32 copy_len = coqui_choose_block_len(ctx, len - 1);
                u32 copy_from = coqui_rand_below(ctx, len - copy_len + 1);
                u32 copy_to = coqui_rand_below(ctx, len - copy_len + 1);

                if (likely(copy_from != copy_to)) {
                    coqui_memmove(buf + copy_to, buf + copy_from, copy_len);
                }

                break;

            }

            case MUT_OVERWRITE_FIXED: {

                /* Overwrite bytes with fixed bytes. */
                if (unlikely(len < 2)) { break; }

                u32 copy_len = coqui_choose_block_len(ctx, len - 1);
                u32 copy_to = coqui_rand_below(ctx, len - copy_len + 1);
                u32 strat = coqui_rand_below(ctx, 2);
                u32 copy_from = copy_to ? copy_to - 1 : 0;
                item = strat ? coqui_rand_below(ctx, 256) : buf[copy_from];
                memset(buf + copy_to, item, copy_len);

                break;

            }

            case MUT_BYTEADD: {

                /* Increase byte by 1. */
                buf[coqui_rand_below(ctx, len)]++;
                break;

            }

            case MUT_BYTESUB: {

                /* Decrease byte by 1. */
                buf[coqui_rand_below(ctx, len)]--;
                break;

            }

            case MUT_FLIP8: {

                /* Flip byte. */
                buf[coqui_rand_below(ctx, len)] ^= 0xff;
                break;

            }

            case MUT_SWITCH: {

                if (unlikely(len < 4)) { break; }

                /* Switch bytes. */
                u32 to_end, switch_to, switch_len, switch_from;
                switch_from = coqui_rand_below(ctx, len);
                do {
                    switch_to = coqui_rand_below(ctx, len);
                } while (unlikely(switch_from == switch_to));

                if (switch_from < switch_to) {
                    switch_len = switch_to - switch_from;
                    to_end = len - switch_to;
                } else {
                    switch_len = switch_from - switch_to;
                    to_end = len - switch_from;
                }

                switch_len = coqui_choose_block_len(ctx, COQUI_MIN(switch_len, to_end));

                /* Backup */
                memcpy(scratch, buf + switch_from, switch_len);

                /* Switch 1 */
                memcpy(buf + switch_from, buf + switch_to, switch_len);

                /* Switch 2 */
                memcpy(buf + switch_to, scratch, switch_len);

                break;

            }

            case MUT_DEL: {

                /* Delete bytes. */
                if (unlikely(len < 2)) { break; }

                /* Don't delete too much. */
                u32 del_len = coqui_choose_block_len(ctx, len - 1);
                u32 del_from = coqui_rand_below(ctx, len - del_len + 1);
                coqui_memmove(buf + del_from, buf + del_from + del_len,
                             len - del_from - del_len);
                len -= del_len;

                break;

            }

            case MUT_SHUFFLE: {

                /* Shuffle bytes. */
                if (unlikely(len < 4)) { break; }

                u32 blen = coqui_choose_block_len(ctx, len - 1);
                u32 off = coqui_rand_below(ctx, len - blen + 1);

                for (u32 i = blen - 1; i > 0; i--) {

                    u32 j;
                    do {
                        j = coqui_rand_below(ctx, i + 1);
                    } while (unlikely(i == j));

                    u8 temp = buf[off + i];
                    buf[off + i] = buf[off + j];
                    buf[off + j] = temp;

                }

                break;

            }

            case MUT_DELONE: {

                /* Delete one byte. */
                if (unlikely(len < 2)) { break; }

                u32 del_len = 1;
                u32 del_from = coqui_rand_below(ctx, len - del_len + 1);
                coqui_memmove(buf + del_from, buf + del_from + del_len,
                             len - del_from - del_len);

                len -= del_len;

                break;

            }

            case MUT_INSERTONE: {

                if (unlikely(len < 2)) { break; }

                u32 clone_len = 1;
                if (unlikely(len + clone_len > max_len)) { goto retry_havoc_step; }
                u32 clone_to = coqui_rand_below(ctx, len);
                u32 strat = coqui_rand_below(ctx, 2);
                u32 clone_from = clone_to ? clone_to - 1 : 0;
                item = strat ? coqui_rand_below(ctx, 256) : buf[clone_from];

                /* Head */
                memcpy(scratch, buf, clone_to);

                /* Inserted part */
                memset(scratch + clone_to, item, clone_len);

                /* Tail */
                memcpy(scratch + clone_to + clone_len, buf + clone_to,
                       len - clone_to);

                len += clone_len;
                memcpy(buf, scratch, len);

                break;

            }

            case MUT_ASCIINUM: {

                if (unlikely(len < 4)) { break; }

                u32 off = coqui_rand_below(ctx, len), off2 = off, cnt = 0;

                while (off2 + cnt < len && !coqui_isdigit(buf[off2 + cnt])) {
                    ++cnt;
                }

                /* none found, wrap */
                if (off2 + cnt == len) {

                    off2 = 0;
                    cnt = 0;

                    while (cnt < off && !coqui_isdigit(buf[off2 + cnt])) {
                        ++cnt;
                    }

                    if (cnt == off) {

                        if (len < 8) {
                            break;
                        } else {
                            goto retry_havoc_step;
                        }

                    }

                }

                off = off2 + cnt;
                off2 = off + 1;

                while (off2 < len && coqui_isdigit(buf[off2])) {
                    ++off2;
                }

                s64 val = buf[off] - '0';
                for (u32 i = off + 1; i < off2; ++i) {

                    u8  digit = buf[i] - '0';
                    s64 valx10;

                    if (val > INT64_MAX / 10 ||
                        (valx10 = (val * 10)) > INT64_MAX - digit) {

                        off2 = i;
                        break;

                    }

                    val = valx10 + digit;

                }

                if (off && buf[off - 1] == '-') { val = -val; }

                u32 strat = coqui_rand_below(ctx, 8);
                switch (strat) {

                    case 0:
                        if (val == INT64_MAX) {
                            val /= 10;
                            --off2;
                        }
                        val++;
                        break;
                    case 1:
                        if (val == INT64_MIN) {
                            val /= 10;
                            --off2;
                        }
                        val--;
                        break;
                    case 2:
                        if (val > INT64_MAX / 2 || val < INT64_MIN / 2) {
                            val /= 10;
                            --off2;
                        }
                        val *= 2;
                        break;
                    case 3:
                        val /= 2;
                        break;
                    case 4:
                        if (likely(val && (u64)val < 0x19999999)) {
                            val = (u64)coqui_rand_next(ctx) % (u64)((u64)val * 10);
                        } else {
                            val = coqui_rand_below(ctx, 256);
                        }
                        break;
                    case 5:
                        if (val > INT64_MAX - 256) {
                            val /= 10;
                            --off2;
                        }
                        val += coqui_rand_below(ctx, 256);
                        break;
                    case 6:
                        if (val < INT64_MIN + 256) {
                            val /= 10;
                            --off2;
                        }
                        val -= coqui_rand_below(ctx, 256);
                        break;
                    case 7:
                        val = ~(val);
                        break;

                }

                /* GPU-safe: use coqui_s64_to_str instead of snprintf */
                u8 numbuf[32];
                u32 new_len = coqui_s64_to_str(val, numbuf);
                u32 old_len = off2 - off;

                if (old_len == new_len) {

                    memcpy(buf + off, numbuf, new_len);

                } else {

                    if (unlikely(off + new_len + len - off2 > max_len)) {
                        goto retry_havoc_step;
                    }

                    /* Head */
                    memcpy(scratch, buf, off);

                    /* Inserted part */
                    memcpy(scratch + off, numbuf, new_len);

                    /* Tail */
                    memcpy(scratch + off + new_len, buf + off2, len - off2);

                    len += (new_len - old_len);
                    memcpy(buf, scratch, len);

                }

                break;

            }

            case MUT_INSERTASCIINUM: {

                u32 ins_len = 1 + coqui_rand_below(ctx, 8);
                u32 pos = coqui_rand_below(ctx, len);

                /* Insert ascii number. */
                if (unlikely(len < pos + ins_len)) {

                    if (unlikely(len < 8)) {
                        break;
                    } else {
                        goto retry_havoc_step;
                    }

                }

                u64 rval = coqui_rand_next(ctx);
                u8 numbuf[32];
                u32 val_len = coqui_u64_to_str(rval, numbuf);
                u32 noff;

                if (ins_len > val_len) {
                    ins_len = val_len;
                    noff = 0;
                } else {
                    noff = val_len - ins_len;
                }

                memcpy(buf + pos, numbuf + noff, ins_len);

                break;

            }

            case MUT_EXTRA_OVERWRITE: {

                /* Dictionary overwrite: pick a random extra token and overwrite
                 * a random position in the buffer with it. Semantically identical
                 * to AFL's afl_mutate case MUT_EXTRA_OVERWRITE. */
                if (unlikely(!__coqui_extras_cnt)) { goto retry_havoc_step; }

                u32 use_extra = coqui_rand_below(ctx, __coqui_extras_cnt);
                u32 extra_len = __coqui_extras_lens[use_extra];

                if (unlikely(extra_len > len)) { goto retry_havoc_step; }

                u32 insert_at = coqui_rand_below(ctx, len - extra_len + 1);
                memcpy(buf + insert_at,
                       __coqui_extras_data + __coqui_extras_offsets[use_extra],
                       extra_len);

                break;

            }

            case MUT_EXTRA_INSERT: {

                /* Dictionary insert: pick a random extra token and insert it
                 * at a random position, shifting the tail. */
                if (unlikely(!__coqui_extras_cnt)) { goto retry_havoc_step; }

                u32 use_extra = coqui_rand_below(ctx, __coqui_extras_cnt);
                u32 extra_len = __coqui_extras_lens[use_extra];
                if (unlikely(len + extra_len > max_len)) { goto retry_havoc_step; }

                u32 insert_at = coqui_rand_below(ctx, len + 1);

                /* Shift tail */
                coqui_memmove(buf + insert_at + extra_len,
                              buf + insert_at, len - insert_at);

                /* Insert token */
                memcpy(buf + insert_at,
                       __coqui_extras_data + __coqui_extras_offsets[use_extra],
                       extra_len);
                len += extra_len;

                break;

            }

            case MUT_AUTO_EXTRA_OVERWRITE: {

                /* Auto-extras overwrite. Auto-extras are stored after regular
                 * extras in the same data/offsets/lens arrays. Index offset
                 * is __coqui_extras_cnt. */
                if (unlikely(!__coqui_a_extras_cnt)) { goto retry_havoc_step; }

                u32 use_extra = coqui_rand_below(ctx, __coqui_a_extras_cnt);
                u32 aidx = __coqui_extras_cnt + use_extra;
                u32 extra_len = __coqui_extras_lens[aidx];

                if (unlikely(extra_len > len)) { goto retry_havoc_step; }

                u32 insert_at = coqui_rand_below(ctx, len - extra_len + 1);
                memcpy(buf + insert_at,
                       __coqui_extras_data + __coqui_extras_offsets[aidx],
                       extra_len);

                break;

            }

            case MUT_AUTO_EXTRA_INSERT: {

                /* Auto-extras insert. */
                if (unlikely(!__coqui_a_extras_cnt)) { goto retry_havoc_step; }

                u32 use_extra = coqui_rand_below(ctx, __coqui_a_extras_cnt);
                u32 aidx = __coqui_extras_cnt + use_extra;
                u32 extra_len = __coqui_extras_lens[aidx];
                if (unlikely(len + extra_len > max_len)) { goto retry_havoc_step; }

                u32 insert_at = coqui_rand_below(ctx, len + 1);

                /* Shift tail */
                coqui_memmove(buf + insert_at + extra_len,
                              buf + insert_at, len - insert_at);

                /* Insert token */
                memcpy(buf + insert_at,
                       __coqui_extras_data + __coqui_extras_offsets[aidx],
                       extra_len);
                len += extra_len;

                break;

            }

            case MUT_SPLICE_OVERWRITE: {

                /* Splice overwrite: pick a random parent from the batch's parent
                 * table and copy a random block from it over the current buffer.
                 * Cross-pollination between corpus entries within the same GPU
                 * batch — semantically equivalent to AFL's splice_buf overwrite.
                 * Self-splice (picking our own parent) is valid: the block
                 * copy_from != copy_to so it still shuffles content. */
                u32 n_parents = __coqui_parent_count;
                if (unlikely(n_parents <= 1)) { goto retry_havoc_step; }

                u32 splice_pidx = coqui_rand_below(ctx, n_parents);

                u32 splice_off = __coqui_parent_offsets[splice_pidx];
                u32 splice_len = __coqui_parent_lens[splice_pidx];
                u8 *splice_buf = __coqui_parent_bytes + splice_off;

                if (unlikely(splice_len < 2)) { goto retry_havoc_step; }

                u32 copy_len = coqui_choose_block_len(ctx, splice_len - 1);
                if (copy_len > len) copy_len = len;

                u32 copy_from = coqui_rand_below(ctx, splice_len - copy_len + 1);
                u32 copy_to = coqui_rand_below(ctx, len - copy_len + 1);
                memcpy(buf + copy_to, splice_buf + copy_from, copy_len);

                break;

            }

            case MUT_SPLICE_INSERT: {

                /* Splice insert: pick a random block from another parent and
                 * insert it at a random position in the buffer. Uses scratch
                 * buffer for the splice (same pattern as MUT_CLONE_COPY). */
                u32 n_parents = __coqui_parent_count;
                if (unlikely(n_parents <= 1)) { goto retry_havoc_step; }
                if (unlikely(len + COQUI_HAVOC_BLK_XL > max_len)) { goto retry_havoc_step; }

                u32 splice_pidx = coqui_rand_below(ctx, n_parents);

                u32 splice_off = __coqui_parent_offsets[splice_pidx];
                u32 splice_len = __coqui_parent_lens[splice_pidx];
                u8 *splice_buf = __coqui_parent_bytes + splice_off;

                if (unlikely(splice_len < 1)) { goto retry_havoc_step; }

                u32 clone_len = coqui_choose_block_len(ctx, splice_len);
                if (unlikely(len + clone_len > max_len)) { goto retry_havoc_step; }

                u32 clone_from = coqui_rand_below(ctx, splice_len - clone_len + 1);
                u32 clone_to = coqui_rand_below(ctx, len + 1);

                /* Head */
                memcpy(scratch, buf, clone_to);

                /* Inserted part from splice source */
                memcpy(scratch + clone_to, splice_buf + clone_from, clone_len);

                /* Tail */
                memcpy(scratch + clone_to + clone_len, buf + clone_to, len - clone_to);

                len += clone_len;
                memcpy(buf, scratch, len);

                break;

            }

        }

    }

    }

    return len;

}

#endif /* COQUI_AFL_MUTATE_H */
