# GPU-Side Havoc Mutations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move AFL's havoc-stage mutation work onto the GPU under `coqui_mode` so each GPU thread mutates a base seed from a resident pool and executes the target, eliminating per-iteration CPU mutation + 8 MB/batch HtoD.

**Architecture:** Kernel gets a new `slot_info[tid]` arg encoding `(flag, seed_idx)`; `flag=1` routes the thread through a new `__coqui_havoc_mutate()` runtime helper that runs AFL's 37 `MUT_*` ops against a 256-entry seed pool resident on GPU, refreshed per `queue_cur` entry. CPU-side `fuzz_one_original` havoc loop is reduced to a thin submit-slot loop.

**Tech Stack:** C (device runtime via clang nvptx64 target), C++ (LLVM 18 IR-builder pass in `coqui_mode/passes/FuzzEntry.cpp`), CUDA Driver API (host integration), AFL++ C (core).

**Important:** The user has explicitly requested **no unit tests**. Verification is via:
- Compile-green at each step (`coqui_mode/build_coqui_support.sh` builds clean, `make` for afl-fuzz builds clean).
- Functional smoke: `cjson_fuzzer` runs for ≥30 s on device 0 without crashes, WARNFs, or FATALs.
- End-of-project bench: n=20 × 120 s interleaved CPU-havoc vs GPU-havoc runs.

**Key source references:**
- Spec: `docs/superpowers/specs/2026-04-21-gpu-havoc-mutations-design.md`
- Existing coqui backend: `src/afl-fuzz-coqui.c`
- Existing kernel entry: `coqui_mode/passes/FuzzEntry.cpp`
- AFL havoc body to port: `src/afl-fuzz-one.c:2086-3543`
- AFL mutation weight tables: `include/afl-mutations.h`

---

## File Structure

### Created files

| Path | Responsibility |
|---|---|
| `coqui_mode/runtime/coqui_mutate.c` | Device-side `__coqui_havoc_mutate()` + PRNG + weighted splice pick + all 37 `MUT_*` op implementations. Links into `runtime.bc`. |
| `coqui_mode/runtime/coqui_mutate.h` | Declarations for the one public symbol (`__coqui_havoc_mutate`). |
| `benchmark/gpu_havoc_vs_cpu.sh` | Phase 4 bench runner: interleaved CPU-havoc vs GPU-havoc runs with CSV output. |

### Modified files

| Path | Changes |
|---|---|
| `coqui_mode/runtime/coqui_runtime.h` | Add `COQUI_FLAG_*` constants, declare extern globals bound by host (seed pool, extras, mutation_array, prng_base). |
| `coqui_mode/passes/FuzzEntry.cpp` | Kernel signature goes from 5 args to 7 (`slot_info`, `reported_slab`). Add flag-dispatch blocks. Add havoc block that allocates `.local scratch`, memcopies seed, calls `__coqui_havoc_mutate`. Add compact-report block. |
| `coqui_mode/build_coqui_support.sh` | Add `coqui_mutate` to runtime .bc compilation + link list. Add `-DMAX_INPUT_SIZE=4096` to runtime compile flags. |
| `include/afl-fuzz-coqui.h` | Add flag constants, extend `coqui_batch_t` (slot_info, reported_slab, reported_count, reported_tid, reported_lens), extend `coqui_ctx_t` (seed pool fields + stream_pool + uploaded-counters), add `coqui_submit_havoc_slot` + `coqui_refresh_seed_pool` prototypes. |
| `src/afl-fuzz-coqui.c` | Allocate new batch buffers in `alloc_batch_half_cuda`, allocate seed pool ping-pong + extras + mutation_array binding in `coqui_init`, add `stream_pool`. Implement `coqui_refresh_seed_pool` + `coqui_submit_havoc_slot`. Update `coqui_launch_batch` (HtoD slot_info, DtoH reported_*). Update `coqui_await_and_process` (reported_slab lookup). Free new buffers in `free_batch_half_cuda` and `coqui_shutdown`. |
| `src/afl-fuzz-one.c` | In havoc_stage body (line ~2259 onward), add `if (afl->coqui)` runtime branch replacing the 1500-line mutation switch with the thin submit-slot loop. |

---

## Phase 1 — Runtime module

This phase adds the GPU-side mutation code *without hooking it up*. At the end of this phase the runtime links into `runtime.bc` with the new symbol exported but no caller invokes it, so cuAFL behaviour is unchanged.

### Task 1.1: Scaffold `coqui_mutate.h`

**Files:**
- Create: `coqui_mode/runtime/coqui_mutate.h`

- [ ] **Step 1: Write the header**

```c
/*
 * coqui_mutate.h --- GPU-side havoc mutation runtime.
 *
 * Declares the single public entry point for AFL's havoc stage running on
 * the GPU. All state (seed pool, extras, mutation weight table, PRNG base)
 * is bound by the host via module-level extern globals — see
 * coqui_runtime.h for the extern decls.
 *
 * Spec: docs/superpowers/specs/2026-04-21-gpu-havoc-mutations-design.md §4.2
 */

#ifndef _COQUI_MUTATE_H
#define _COQUI_MUTATE_H

#include "coqui_runtime.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Mutate `buf` in place using AFL havoc semantics. Returns the new length.
 * `buf` must have at least `max_len` bytes of headroom (may grow).
 * `self_idx` is this thread's seed pool index — splice ops exclude it to
 * avoid self-splice. `prng` is the per-thread splitmix64 state. */
u32 __coqui_havoc_mutate(u8 *buf, u32 len, u32 max_len,
                          u32 self_idx, u64 *prng);

#ifdef __cplusplus
}
#endif

#endif /* _COQUI_MUTATE_H */
```

- [ ] **Step 2: Commit**

```bash
git add coqui_mode/runtime/coqui_mutate.h
git commit -m "cuAFL: scaffold coqui_mutate.h for GPU-side havoc runtime"
```

### Task 1.2: Add flag constants + extern globals to `coqui_runtime.h`

**Files:**
- Modify: `coqui_mode/runtime/coqui_runtime.h`

- [ ] **Step 1: Add flag constants and new extern globals after line 43 (near `COQUI_PHASE_*`)**

Insert after the `#define COQUI_PHASE_RESERVED 7` line:

```c
/* Slot-info flag values written by host to per-thread slot_info[tid] as
 * (flag << 24) | (seed_idx & 0xFFFFFF). See spec §3.2.
 * flag=0: input_bytes+offsets[tid] is pre-mutated (today's path).
 * flag=1: havoc mutate from __coqui_seed_pool_base[seed_idx].       */
#define COQUI_FLAG_PREMUT   0u
#define COQUI_FLAG_HAVOC    1u

/* Sentinel returned by weighted_splice_pick when the draw lands on
 * self_idx. Caller issues `goto retry_havoc_step`. (~0u) */
#define COQUI_SPLICE_SELF_SAME  (~0u)

/* Reporting slab cap: up to this many threads can compact-write their
 * mutated bytes back to the host per batch. Overflow degrades gracefully
 * (WARNF + skip CPU verify for overflow slots). */
#define COQUI_REPORTED_CAP  512u
```

- [ ] **Step 2: Declare module-level externs at the bottom, before the closing `#endif`**

Insert before `#endif /* _COQUI_RUNTIME_H */`:

```c
/* -- Module-level globals bound by host.
 * All of these live on the device; host writes via cuModuleGetGlobal +
 * cuMemcpyHtoD. See spec §3.1.
 */

/* Seed pool (ping-pong). __coqui_seed_pool_base etc. are POINTERS that host
 * updates to alias to either the _a or _b backing store before each launch.
 * The backing stores themselves are allocated by the host and their device
 * addresses are written into these pointer-symbols. */
extern u8  *__coqui_seed_pool_base;
extern u32 *__coqui_seed_pool_offsets;
extern u32 *__coqui_seed_pool_lens;
extern u32 *__coqui_seed_pool_cumw;
extern u32  __coqui_seed_pool_count;
extern u32  __coqui_seed_pool_cumw_total;

/* Extras (user dictionary via -x). Once at init. */
extern u8  *__coqui_extras_base;
extern u32 *__coqui_extras_offsets;
extern u32 *__coqui_extras_lens;
extern u32  __coqui_extras_cnt;

/* Auto-extras (cmplog-learned). Grows during run; opportunistic re-upload. */
extern u8  *__coqui_a_extras_base;
extern u32 *__coqui_a_extras_offsets;
extern u32 *__coqui_a_extras_lens;
extern u32  __coqui_a_extras_cnt;

/* Current active havoc weight table (constant memory for broadcast efficiency). */
extern __attribute__((address_space(4))) u32 __coqui_mutation_array[256];
extern __attribute__((address_space(4))) u32 __coqui_mutation_array_size;
extern __attribute__((address_space(4))) u32 __coqui_havoc_stack_pow2;

/* Per-batch randomness base. Thread-local PRNG is splitmix64(base ^ tid). */
extern u64 __coqui_prng_base;

/* Compact-report counter (atomically bumped by kernel novelty/crash path). */
extern u32 __coqui_reported_count;
extern u32 *__coqui_reported_tid;    /* length COQUI_REPORTED_CAP */
extern u32 *__coqui_reported_lens;   /* length COQUI_REPORTED_CAP */
```

**NOTE**: `address_space(4)` is nvptx64's `__constant__` memory on LLVM clang — already used elsewhere in the runtime. If the compiler complains (LLVM version skew), drop the attribute and fall back to `extern u32 __coqui_mutation_array[256];` — the compiler will place it in `.global` which is slower but correct.

- [ ] **Step 3: Build the runtime bitcode to catch syntax errors early**

```bash
cd /home/gpizarro/cuAFL/coqui_mode
bash build_coqui_support.sh
```

Expected: builds to `runtime/build/*.bc` without errors. New symbols won't resolve yet (no `coqui_mutate.c` defines them), but the header itself compiles.

- [ ] **Step 4: Commit**

```bash
git add coqui_mode/runtime/coqui_runtime.h
git commit -m "cuAFL: add flag constants and extern globals for GPU havoc"
```

### Task 1.3: Implement PRNG + `gpu_rand_below` in `coqui_mutate.c`

**Files:**
- Create: `coqui_mode/runtime/coqui_mutate.c`

- [ ] **Step 1: Write the file preamble + PRNG helpers**

```c
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
 * time; one 32×32→64 multiply + a rare rejection loop. */
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
```

- [ ] **Step 2: Add `coqui_mutate` to the runtime build list**

Modify `coqui_mode/build_coqui_support.sh`:

```bash
# Change this line:
RUNTIME_FLAGS="--target=nvptx64-nvidia-cuda -O2 -ffreestanding -emit-llvm -c -I $COQUI_DIR/runtime"

# To this (adds -DMAX_INPUT_SIZE=4096):
RUNTIME_FLAGS="--target=nvptx64-nvidia-cuda -O2 -ffreestanding -emit-llvm -c -I $COQUI_DIR/runtime -DMAX_INPUT_SIZE=4096"

# Change this loop:
for f in coqui_runtime coqui_coverage coqui_memory coqui_asan coqui_libc; do

# To this (adds coqui_mutate):
for f in coqui_runtime coqui_coverage coqui_memory coqui_asan coqui_libc coqui_mutate; do

# Add a corresponding line to the llvm-link invocation:
# (existing links coqui_runtime.bc, coqui_coverage.bc, coqui_memory.bc, coqui_asan.bc, coqui_libc.bc)
# Add: "$COQUI_DIR/runtime/build/coqui_mutate.bc" \
```

- [ ] **Step 3: Build and verify**

```bash
cd /home/gpizarro/cuAFL/coqui_mode
bash build_coqui_support.sh
```

Expected: `runtime/build/coqui_mutate.bc` exists; `runtime.bc` links cleanly.

- [ ] **Step 4: Commit**

```bash
git add coqui_mode/runtime/coqui_mutate.c coqui_mode/build_coqui_support.sh
git commit -m "cuAFL: add coqui_mutate.c with PRNG helpers, wire into runtime build"
```

### Task 1.4: Add weighted splice pick helper

**Files:**
- Modify: `coqui_mode/runtime/coqui_mutate.c`

- [ ] **Step 1: Append after the PRNG helpers (before the public API)**

```c
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
```

- [ ] **Step 2: Build**

```bash
cd /home/gpizarro/cuAFL/coqui_mode && bash build_coqui_support.sh
```

Expected: builds clean.

- [ ] **Step 3: Commit**

```bash
git add coqui_mode/runtime/coqui_mutate.c
git commit -m "cuAFL: add weighted_splice_pick helper for GPU havoc"
```

### Task 1.5: Implement `__coqui_havoc_mutate` wrapper with empty switch

**Files:**
- Modify: `coqui_mode/runtime/coqui_mutate.c`

This task adds the outer loop structure from `src/afl-fuzz-one.c:2255-2316` — `stack_max`, `use_stacking`, and the `retry_havoc_step` dispatch. Op bodies are empty placeholders; the next tasks fill them in bucket by bucket.

- [ ] **Step 1: Append `__coqui_havoc_mutate` and helper macros**

```c
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
  u32 r = gpu_rand_below(prng, 3);
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
      /* Bucket 1 ops — filled in by task 1.6 */
      case MUT_FLIPBIT: case MUT_INTERESTING8:
      case MUT_INTERESTING16: case MUT_INTERESTING16BE:
      case MUT_INTERESTING32: case MUT_INTERESTING32BE:
      case MUT_ARITH8_: case MUT_ARITH8:
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
```

- [ ] **Step 2: Build**

```bash
cd /home/gpizarro/cuAFL/coqui_mode && bash build_coqui_support.sh
```

Expected: builds clean. `__coqui_havoc_mutate` now compiles but all op cases no-op via `goto retry_havoc_step`. With the infinite-retry concern in mind: since `__coqui_mutation_array` contains valid op codes, `r` always maps to a known case, so a 100%-placeholder state would loop forever. That's fine for a compile-check milestone — we won't call the function until Phase 2 wires it up, and we'll have real op bodies before that happens.

- [ ] **Step 3: Commit**

```bash
git add coqui_mode/runtime/coqui_mutate.c
git commit -m "cuAFL: scaffold __coqui_havoc_mutate dispatch with placeholder ops"
```

### Task 1.6: Port Bucket 1 ops (22 self-contained ops)

**Files:**
- Modify: `coqui_mode/runtime/coqui_mutate.c`

**Reference:** `src/afl-fuzz-one.c:2320-3421` — every `case MUT_*:` block in that range is the source. Port *verbatim*, applying the mechanical substitutions from spec §4.2:
- `rand_below(afl, n)` → `gpu_rand_below(prng, n)`
- `out_buf` → `buf`
- `temp_len` → `len`
- `MAX_FILE` → `max_len`
- `#ifdef INTROSPECTION` blocks → **delete entirely** (GPU can't build introspection strings)
- `interesting_8`, `interesting_16`, `interesting_32` refer to the local tables declared in task 1.3

**Op-by-op port checklist.** For each op, replace the placeholder in the switch with the adapted body. Work one op per commit so a regression can be bisected to a single op port.

- [ ] **Step 1: Port MUT_FLIPBIT (canonical example — use this pattern for the rest)**

Replace `case MUT_FLIPBIT:` branch in the switch with:

```c
case MUT_FLIPBIT: {
  u8 bit = (u8)gpu_rand_below(prng, 8);
  u32 off = gpu_rand_below(prng, len);
  buf[off] ^= 1u << bit;
  break;
}
```

Build:

```bash
cd /home/gpizarro/cuAFL/coqui_mode && bash build_coqui_support.sh
```

Expected: clean build. Commit:

```bash
git add coqui_mode/runtime/coqui_mutate.c
git commit -m "cuAFL: port MUT_FLIPBIT to GPU havoc runtime"
```

- [ ] **Step 2: Port MUT_INTERESTING8, MUT_INTERESTING16, MUT_INTERESTING16BE, MUT_INTERESTING32, MUT_INTERESTING32BE**

Pattern (following `src/afl-fuzz-one.c:2338-2423`):

```c
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
  /* Little-endian unaligned store via memcpy of 2 bytes */
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
```

Build and commit:

```bash
cd /home/gpizarro/cuAFL/coqui_mode && bash build_coqui_support.sh
git add coqui_mode/runtime/coqui_mutate.c
git commit -m "cuAFL: port MUT_INTERESTING{8,16,16BE,32,32BE} to GPU havoc"
```

- [ ] **Step 3: Port MUT_ARITH8_, MUT_ARITH8 (byte sub/add)**

Source: `src/afl-fuzz-one.c:2424-2448`. Pattern:

```c
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
```

Build and commit:

```bash
cd /home/gpizarro/cuAFL/coqui_mode && bash build_coqui_support.sh
git add coqui_mode/runtime/coqui_mutate.c
git commit -m "cuAFL: port MUT_ARITH8_ and MUT_ARITH8 to GPU havoc"
```

- [ ] **Step 4: Port MUT_ARITH16_, MUT_ARITH16BE_, MUT_ARITH16, MUT_ARITH16BE**

Source: `src/afl-fuzz-one.c:2449-2534`. Each reads a 16-bit value (LE or BE), subtracts or adds a random `item ∈ [1, ARITH_MAX]`, writes back. Pattern (for MUT_ARITH16_ — LE subtract):

```c
case MUT_ARITH16_: {
  if (len < 2) goto retry_havoc_step;
  u32 pos = gpu_rand_below(prng, len - 1);
  u16 item = (u16)(1 + gpu_rand_below(prng, ARITH_MAX));
  u16 v = (u16)buf[pos] | ((u16)buf[pos + 1] << 8);
  v = v - item;
  buf[pos]     = (u8)(v & 0xFF);
  buf[pos + 1] = (u8)((v >> 8) & 0xFF);
  break;
}
```

Port the remaining three (`MUT_ARITH16BE_` = BE subtract, `MUT_ARITH16` = LE add, `MUT_ARITH16BE` = BE add) with matching endianness and operation.

Build and commit:

```bash
cd /home/gpizarro/cuAFL/coqui_mode && bash build_coqui_support.sh
git add coqui_mode/runtime/coqui_mutate.c
git commit -m "cuAFL: port MUT_ARITH16{_,BE_,,BE} to GPU havoc"
```

- [ ] **Step 5: Port MUT_ARITH32_, MUT_ARITH32BE_, MUT_ARITH32, MUT_ARITH32BE**

Source: `src/afl-fuzz-one.c:2535-2620`. Same pattern as ARITH16, with 4-byte loads/stores and `len < 4`. Commit.

```bash
git add coqui_mode/runtime/coqui_mutate.c
git commit -m "cuAFL: port MUT_ARITH32{_,BE_,,BE} to GPU havoc"
```

- [ ] **Step 6: Port MUT_RAND8, MUT_FLIP8, MUT_SWITCH, MUT_BYTEADD, MUT_BYTESUB**

Source: `src/afl-fuzz-one.c:2621-2670` (approximate — each op is ~5 lines). Patterns:

```c
case MUT_RAND8: {
  u32 off = gpu_rand_below(prng, len);
  buf[off] = (u8)gpu_rand_below(prng, 256);
  break;
}
case MUT_FLIP8: {
  u32 off = gpu_rand_below(prng, len);
  buf[off] = ~buf[off];
  break;
}
case MUT_SWITCH: {
  if (len < 2) goto retry_havoc_step;
  u32 a = gpu_rand_below(prng, len);
  u32 b = gpu_rand_below(prng, len);
  if (a == b) break;
  u8 tmp = buf[a]; buf[a] = buf[b]; buf[b] = tmp;
  break;
}
case MUT_BYTEADD: {
  u32 off = gpu_rand_below(prng, len);
  u32 item = 1 + gpu_rand_below(prng, ARITH_MAX);
  buf[off] = (u8)(buf[off] + (u8)item);
  break;
}
case MUT_BYTESUB: {
  u32 off = gpu_rand_below(prng, len);
  u32 item = 1 + gpu_rand_below(prng, ARITH_MAX);
  buf[off] = (u8)(buf[off] - (u8)item);
  break;
}
```

Build and commit:

```bash
cd /home/gpizarro/cuAFL/coqui_mode && bash build_coqui_support.sh
git add coqui_mode/runtime/coqui_mutate.c
git commit -m "cuAFL: port MUT_RAND8/FLIP8/SWITCH/BYTEADD/BYTESUB to GPU havoc"
```

- [ ] **Step 7: Port MUT_CLONE_COPY, MUT_CLONE_FIXED (length-growing)**

Source: `src/afl-fuzz-one.c:2685-2740`. These grow the input by cloning a range. Must check `len + HAVOC_BLK_XL < max_len` before growing.

```c
case MUT_CLONE_COPY: {
  if (len + HAVOC_BLK_XL >= max_len) goto retry_havoc_step;
  u32 clone_len = choose_block_len(prng, HAVOC_BLK_XL);
  u32 clone_from = gpu_rand_below(prng, len);
  u32 clone_to   = gpu_rand_below(prng, len + 1);
  if (clone_from + clone_len > len) clone_len = len - clone_from;
  if (len + clone_len >= max_len) goto retry_havoc_step;
  /* shift tail right, then copy */
  for (u32 i = len; i > clone_to; --i) buf[i - 1 + clone_len] = buf[i - 1];
  for (u32 i = 0; i < clone_len; ++i)  buf[clone_to + i] = buf[clone_from + i];
  len += clone_len;
  break;
}
case MUT_CLONE_FIXED: {
  if (len + HAVOC_BLK_XL >= max_len) goto retry_havoc_step;
  u32 clone_len = choose_block_len(prng, HAVOC_BLK_XL);
  u32 clone_to  = gpu_rand_below(prng, len + 1);
  if (len + clone_len >= max_len) goto retry_havoc_step;
  u8 fill = (u8)gpu_rand_below(prng, 256);
  for (u32 i = len; i > clone_to; --i) buf[i - 1 + clone_len] = buf[i - 1];
  for (u32 i = 0; i < clone_len; ++i)  buf[clone_to + i] = fill;
  len += clone_len;
  break;
}
```

Build and commit:

```bash
cd /home/gpizarro/cuAFL/coqui_mode && bash build_coqui_support.sh
git add coqui_mode/runtime/coqui_mutate.c
git commit -m "cuAFL: port MUT_CLONE_{COPY,FIXED} to GPU havoc"
```

- [ ] **Step 8: Port MUT_OVERWRITE_COPY, MUT_OVERWRITE_FIXED (length-preserving)**

Source: `src/afl-fuzz-one.c:2741-2790`. Same shape as CLONE but writes over existing bytes without growing.

```c
case MUT_OVERWRITE_COPY: {
  if (len < 2) goto retry_havoc_step;
  u32 copy_len  = choose_block_len(prng, len - 1);
  u32 copy_from = gpu_rand_below(prng, len - copy_len + 1);
  u32 copy_to   = gpu_rand_below(prng, len - copy_len + 1);
  if (copy_from != copy_to) {
    /* Copy — handle overlap via memmove semantics (reverse if destination later) */
    if (copy_to > copy_from) {
      for (u32 i = copy_len; i > 0; --i)
        buf[copy_to + i - 1] = buf[copy_from + i - 1];
    } else {
      for (u32 i = 0; i < copy_len; ++i)
        buf[copy_to + i] = buf[copy_from + i];
    }
  }
  break;
}
case MUT_OVERWRITE_FIXED: {
  if (len < 1) goto retry_havoc_step;
  u32 copy_len = choose_block_len(prng, len);
  u32 copy_to  = gpu_rand_below(prng, len - copy_len + 1);
  u8 fill = (u8)gpu_rand_below(prng, 256);
  for (u32 i = 0; i < copy_len; ++i) buf[copy_to + i] = fill;
  break;
}
```

Build and commit:

```bash
cd /home/gpizarro/cuAFL/coqui_mode && bash build_coqui_support.sh
git add coqui_mode/runtime/coqui_mutate.c
git commit -m "cuAFL: port MUT_OVERWRITE_{COPY,FIXED} to GPU havoc"
```

- [ ] **Step 9: Port MUT_DEL, MUT_DELONE (length-shrinking)**

Source: `src/afl-fuzz-one.c:2791-2835`. Delete a range or a single byte. Must `len > 1` before deleting.

```c
case MUT_DEL: {
  if (len < 2) goto retry_havoc_step;
  u32 del_len = choose_block_len(prng, len - 1);
  u32 del_from = gpu_rand_below(prng, len - del_len + 1);
  for (u32 i = del_from; i + del_len < len; ++i) buf[i] = buf[i + del_len];
  len -= del_len;
  break;
}
case MUT_DELONE: {
  if (len < 2) goto retry_havoc_step;
  u32 pos = gpu_rand_below(prng, len);
  for (u32 i = pos; i + 1 < len; ++i) buf[i] = buf[i + 1];
  len -= 1;
  break;
}
```

- [ ] **Step 10: Port MUT_SHUFFLE, MUT_INSERTONE (remaining Bucket 1 ops)**

Source: `src/afl-fuzz-one.c:2836-2890`.

```c
case MUT_SHUFFLE: {
  if (len < 4) goto retry_havoc_step;
  u32 shuf_len = choose_block_len(prng, len - 1);
  if (shuf_len < 2) goto retry_havoc_step;
  u32 shuf_from = gpu_rand_below(prng, len - shuf_len + 1);
  /* Fisher-Yates shuffle over the chosen range */
  for (u32 i = shuf_len - 1; i > 0; --i) {
    u32 j = gpu_rand_below(prng, i + 1);
    u8 t = buf[shuf_from + i];
    buf[shuf_from + i] = buf[shuf_from + j];
    buf[shuf_from + j] = t;
  }
  break;
}
case MUT_INSERTONE: {
  if (len + 1 >= max_len) goto retry_havoc_step;
  u32 pos = gpu_rand_below(prng, len + 1);
  u8 byte = (u8)gpu_rand_below(prng, 256);
  for (u32 i = len; i > pos; --i) buf[i] = buf[i - 1];
  buf[pos] = byte;
  len += 1;
  break;
}
```

Build and commit:

```bash
cd /home/gpizarro/cuAFL/coqui_mode && bash build_coqui_support.sh
git add coqui_mode/runtime/coqui_mutate.c
git commit -m "cuAFL: port MUT_DEL/DELONE/SHUFFLE/INSERTONE — completes Bucket 1"
```

### Task 1.7: Port Bucket 2 ops (ASCIINUM, INSERTASCIINUM)

**Files:**
- Modify: `coqui_mode/runtime/coqui_mutate.c`

Source: `src/afl-fuzz-one.c:2891-3050`. These replace an ASCII decimal number in the input with a random new one. The CPU version uses `snprintf`; we implement a simple `u64_to_decimal` helper since GPU stdlib is unavailable.

- [ ] **Step 1: Add `u64_to_decimal` helper above `__coqui_havoc_mutate`**

```c
/* Writes up to 20 decimal digits of `v` to `out`. Returns digit count. */
static inline u32 u64_to_decimal(u8 *out, u64 v) {
  u8 tmp[20];
  u32 n = 0;
  if (v == 0) { out[0] = '0'; return 1; }
  while (v > 0 && n < 20) { tmp[n++] = (u8)('0' + (v % 10)); v /= 10; }
  for (u32 i = 0; i < n; ++i) out[i] = tmp[n - 1 - i];
  return n;
}
```

- [ ] **Step 2: Replace the Bucket 2 placeholder cases with actual bodies**

```c
case MUT_ASCIINUM: {
  /* Find an ASCII-digit run in buf[]; if none, retry. Replace it with a
   * random decimal number of similar length. */
  u32 scan_start = gpu_rand_below(prng, len);
  u32 num_start = (u32)-1, num_len = 0;
  for (u32 i = 0; i < len; ++i) {
    u32 p = (scan_start + i) % len;
    u8 c = buf[p];
    if (c >= '0' && c <= '9') {
      /* extend to the longest contiguous run containing p */
      num_start = p;
      while (num_start > 0 && buf[num_start - 1] >= '0' && buf[num_start - 1] <= '9')
        num_start--;
      num_len = 0;
      while (num_start + num_len < len &&
             buf[num_start + num_len] >= '0' && buf[num_start + num_len] <= '9')
        num_len++;
      break;
    }
  }
  if (num_start == (u32)-1) goto retry_havoc_step;

  u64 new_val;
  u32 mode = gpu_rand_below(prng, 4);
  switch (mode) {
    case 0: new_val = gpu_rand_below(prng, 256); break;       /* small */
    case 1: new_val = gpu_rand_below(prng, 65536); break;
    case 2: new_val = ((u64)gpu_rand_below(prng, 0xFFFFFFFFu)); break;
    default: {
      u64 hi = gpu_rand_below(prng, 0xFFFFFFFFu);
      u64 lo = gpu_rand_below(prng, 0xFFFFFFFFu);
      new_val = (hi << 32) | lo;
    } break;
  }

  u8 tmp[20];
  u32 new_len = u64_to_decimal(tmp, new_val);

  if (new_len == num_len) {
    for (u32 i = 0; i < num_len; ++i) buf[num_start + i] = tmp[i];
  } else if (new_len < num_len) {
    for (u32 i = 0; i < new_len; ++i) buf[num_start + i] = tmp[i];
    u32 shrink = num_len - new_len;
    for (u32 i = num_start + new_len; i + shrink < len; ++i) buf[i] = buf[i + shrink];
    len -= shrink;
  } else {
    u32 grow = new_len - num_len;
    if (len + grow >= max_len) goto retry_havoc_step;
    for (u32 i = len; i > num_start + num_len; --i) buf[i - 1 + grow] = buf[i - 1];
    for (u32 i = 0; i < new_len; ++i) buf[num_start + i] = tmp[i];
    len += grow;
  }
  break;
}
case MUT_INSERTASCIINUM: {
  u8 tmp[20];
  u64 new_val;
  u32 mode = gpu_rand_below(prng, 4);
  switch (mode) {
    case 0: new_val = gpu_rand_below(prng, 256); break;
    case 1: new_val = gpu_rand_below(prng, 65536); break;
    case 2: new_val = gpu_rand_below(prng, 0xFFFFFFFFu); break;
    default: {
      u64 hi = gpu_rand_below(prng, 0xFFFFFFFFu);
      u64 lo = gpu_rand_below(prng, 0xFFFFFFFFu);
      new_val = (hi << 32) | lo;
    } break;
  }
  u32 new_len = u64_to_decimal(tmp, new_val);
  if (len + new_len >= max_len) goto retry_havoc_step;
  u32 pos = gpu_rand_below(prng, len + 1);
  for (u32 i = len; i > pos; --i) buf[i - 1 + new_len] = buf[i - 1];
  for (u32 i = 0; i < new_len; ++i) buf[pos + i] = tmp[i];
  len += new_len;
  break;
}
```

Build and commit:

```bash
cd /home/gpizarro/cuAFL/coqui_mode && bash build_coqui_support.sh
git add coqui_mode/runtime/coqui_mutate.c
git commit -m "cuAFL: port MUT_ASCIINUM and MUT_INSERTASCIINUM — Bucket 2 complete"
```

### Task 1.8: Port Bucket 3 ops (extras + splice, 6 ops)

**Files:**
- Modify: `coqui_mode/runtime/coqui_mutate.c`

Source: `src/afl-fuzz-one.c:3051-3421`. These use the extras (`__coqui_extras_*`), auto-extras (`__coqui_a_extras_*`), and seed pool (`__coqui_seed_pool_*`) globals bound by the host.

- [ ] **Step 1: Replace Bucket 3 placeholder with EXTRA_* and AUTO_EXTRA_* implementations**

```c
case MUT_EXTRA_OVERWRITE: {
  if (__coqui_extras_cnt == 0) goto retry_havoc_step;
  u32 use_extra = gpu_rand_below(prng, __coqui_extras_cnt);
  u32 e_len = __coqui_extras_lens[use_extra];
  u32 e_off = __coqui_extras_offsets[use_extra];
  if (e_len == 0 || e_len > len) goto retry_havoc_step;
  u32 ins_at = gpu_rand_below(prng, len - e_len + 1);
  for (u32 i = 0; i < e_len; ++i)
    buf[ins_at + i] = __coqui_extras_base[e_off + i];
  break;
}
case MUT_EXTRA_INSERT: {
  if (__coqui_extras_cnt == 0) goto retry_havoc_step;
  u32 use_extra = gpu_rand_below(prng, __coqui_extras_cnt);
  u32 e_len = __coqui_extras_lens[use_extra];
  u32 e_off = __coqui_extras_offsets[use_extra];
  if (e_len == 0 || len + e_len >= max_len) goto retry_havoc_step;
  u32 ins_at = gpu_rand_below(prng, len + 1);
  for (u32 i = len; i > ins_at; --i) buf[i - 1 + e_len] = buf[i - 1];
  for (u32 i = 0; i < e_len; ++i) buf[ins_at + i] = __coqui_extras_base[e_off + i];
  len += e_len;
  break;
}
case MUT_AUTO_EXTRA_OVERWRITE: {
  if (__coqui_a_extras_cnt == 0) goto retry_havoc_step;
  u32 use_extra = gpu_rand_below(prng, __coqui_a_extras_cnt);
  u32 e_len = __coqui_a_extras_lens[use_extra];
  u32 e_off = __coqui_a_extras_offsets[use_extra];
  if (e_len == 0 || e_len > len) goto retry_havoc_step;
  u32 ins_at = gpu_rand_below(prng, len - e_len + 1);
  for (u32 i = 0; i < e_len; ++i)
    buf[ins_at + i] = __coqui_a_extras_base[e_off + i];
  break;
}
case MUT_AUTO_EXTRA_INSERT: {
  if (__coqui_a_extras_cnt == 0) goto retry_havoc_step;
  u32 use_extra = gpu_rand_below(prng, __coqui_a_extras_cnt);
  u32 e_len = __coqui_a_extras_lens[use_extra];
  u32 e_off = __coqui_a_extras_offsets[use_extra];
  if (e_len == 0 || len + e_len >= max_len) goto retry_havoc_step;
  u32 ins_at = gpu_rand_below(prng, len + 1);
  for (u32 i = len; i > ins_at; --i) buf[i - 1 + e_len] = buf[i - 1];
  for (u32 i = 0; i < e_len; ++i) buf[ins_at + i] = __coqui_a_extras_base[e_off + i];
  len += e_len;
  break;
}
```

- [ ] **Step 2: Replace Bucket 3 splice placeholder with SPLICE_* implementations**

```c
case MUT_SPLICE_OVERWRITE: {
  u32 partner = weighted_splice_pick(prng, self_idx);
  if (partner == COQUI_SPLICE_SELF_SAME) goto retry_havoc_step;
  u32 p_off = __coqui_seed_pool_offsets[partner];
  u32 p_len = __coqui_seed_pool_lens[partner];
  if (p_len < 2 || len < 2) goto retry_havoc_step;
  u32 copy_len = 1 + gpu_rand_below(prng, (p_len < len ? p_len : len) - 1);
  u32 copy_from = gpu_rand_below(prng, p_len - copy_len + 1);
  u32 copy_to   = gpu_rand_below(prng, len   - copy_len + 1);
  for (u32 i = 0; i < copy_len; ++i)
    buf[copy_to + i] = __coqui_seed_pool_base[p_off + copy_from + i];
  break;
}
case MUT_SPLICE_INSERT: {
  u32 partner = weighted_splice_pick(prng, self_idx);
  if (partner == COQUI_SPLICE_SELF_SAME) goto retry_havoc_step;
  u32 p_off = __coqui_seed_pool_offsets[partner];
  u32 p_len = __coqui_seed_pool_lens[partner];
  if (p_len < 2 || len + p_len >= max_len) goto retry_havoc_step;
  u32 ins_len  = 1 + gpu_rand_below(prng, p_len - 1);
  u32 ins_from = gpu_rand_below(prng, p_len - ins_len + 1);
  u32 ins_to   = gpu_rand_below(prng, len + 1);
  if (len + ins_len >= max_len) goto retry_havoc_step;
  for (u32 i = len; i > ins_to; --i) buf[i - 1 + ins_len] = buf[i - 1];
  for (u32 i = 0; i < ins_len; ++i)
    buf[ins_to + i] = __coqui_seed_pool_base[p_off + ins_from + i];
  len += ins_len;
  break;
}
```

- [ ] **Step 3: Build and commit**

```bash
cd /home/gpizarro/cuAFL/coqui_mode && bash build_coqui_support.sh
git add coqui_mode/runtime/coqui_mutate.c
git commit -m "cuAFL: port MUT_EXTRA_* and MUT_SPLICE_* — Bucket 3 complete"
```

### Task 1.9: Verify runtime links into cubin

- [ ] **Step 1: Build a target cubin and dump symbols**

Pick cjson_fuzzer (the existing bench target):

```bash
cd /home/gpizarro/cuAFL
ls cjson_fuzzer.sm_75.cubin  # existing file
# Re-build the target to pick up the new runtime bitcode:
# (follow existing target-rebuild pattern — typically a per-target Makefile;
# if unsure, nuke and re-run coqui-cc on the fuzz target source)
```

Expected: build succeeds. Dump symbols:

```bash
cuobjdump --dump-elf-symbols cjson_fuzzer.sm_75.cubin 2>/dev/null | grep coqui_havoc_mutate
# expected: one line showing __coqui_havoc_mutate as a function symbol
```

Commits are already split by op above; nothing new here except a sanity check.

- [ ] **Step 2: Run cjson_fuzzer for 30 s to confirm no regression from Phase 1 linkage**

```bash
# Existing run command for cjson_fuzzer; adjust to match your bench harness.
cd /home/gpizarro/cuAFL
timeout 30 ./afl-fuzz -c 0 -i /home/gpizarro/cjson-in -o /tmp/cjson-phase1 \
    -- ./cjson_fuzzer.sm_75.cubin 2>&1 | tail -20
```

Expected: fuzzer starts, runs 30 s, execs_per_sec similar to baseline, no new WARNF/FATAL. The havoc path is unchanged because nothing calls `__coqui_havoc_mutate` yet — it's just dead code in the cubin.

Phase 1 done when this smoke passes.

---

## Phase 2 — Kernel ABI + host plumbing

This phase extends the kernel signature and host-side data structures but still doesn't flip `fuzz_one_original`. At the end, `coqui_submit_havoc_slot` exists and works, but nobody calls it in the non-test path.

### Task 2.1: Extend `coqui_batch_t` and `coqui_ctx_t` in `afl-fuzz-coqui.h`

**Files:**
- Modify: `include/afl-fuzz-coqui.h`

- [ ] **Step 1: Add flag + sentinel constants at the top**

Insert after line 23 (`#define COQUI_MAX_INPUT_DEFAULT 4096`):

```c
/* Slot-info flags (shared with coqui_mode/runtime/coqui_runtime.h). */
#define COQUI_FLAG_PREMUT   0u
#define COQUI_FLAG_HAVOC    1u

/* Compact-report buffer cap (shared with runtime). */
#define COQUI_REPORTED_CAP  512u
```

- [ ] **Step 2: Extend `coqui_batch_t` with new host+device buffers**

Insert new fields inside `coqui_batch_t` just before the `Stream + completion event` block (around line 64):

```c
  /* Slot-info: per-thread (flag << 24) | (seed_idx & 0xFFFFFF).
   * u32 per thread, set by the host before each launch. */
  u32            *h_slot_info;
  unsigned long long d_slot_info;

  /* Compact-report: threads setting novelty or crash atomically pack their
   * mutated bytes into reported_slab[slot * max_input_size]. Host reads back
   * only the populated prefix. See spec §3.3. */
  u8             *h_reported_slab;   /* COQUI_REPORTED_CAP * max_input_size bytes */
  u32            *h_reported_tid;    /* COQUI_REPORTED_CAP u32 */
  u32            *h_reported_lens;   /* COQUI_REPORTED_CAP u32 */
  u32             h_reported_count;
  unsigned long long d_reported_slab;
  unsigned long long d_reported_tid;
  unsigned long long d_reported_lens;
  unsigned long long d_reported_count;
```

- [ ] **Step 3: Extend `coqui_ctx_t` with seed-pool + extras + mutation-array fields**

Insert new fields inside `coqui_ctx_t` just before the adaptive-timeout block (around line 148):

```c
  /* Third CUDA stream: seed pool uploads, fenced against stream_a/b via events. */
  void *stream_pool;

  /* Seed pool (ping-pong). Host pinned mirrors + device backing stores.
   * Pointer symbols in the cubin alias to whichever side is live. */
  u8   *h_seed_pool_bytes_a;     /* packed seed bytes */
  u8   *h_seed_pool_bytes_b;
  u32  *h_seed_pool_offsets_a;   /* u32[256] */
  u32  *h_seed_pool_offsets_b;
  u32  *h_seed_pool_lens_a;
  u32  *h_seed_pool_lens_b;
  u32  *h_seed_pool_cumw_a;      /* prefix-sum weights for splice pick */
  u32  *h_seed_pool_cumw_b;
  u32   h_seed_pool_count_next;
  u32   h_seed_pool_cumw_total_next;

  unsigned long long d_seed_pool_bytes_a;
  unsigned long long d_seed_pool_bytes_b;
  unsigned long long d_seed_pool_offsets_a;
  unsigned long long d_seed_pool_offsets_b;
  unsigned long long d_seed_pool_lens_a;
  unsigned long long d_seed_pool_lens_b;
  unsigned long long d_seed_pool_cumw_a;
  unsigned long long d_seed_pool_cumw_b;

  u8    seed_pool_active;   /* 0 = a live, 1 = b live */
  u32   seed_pool_cap_bytes;  /* per-side byte capacity (=1MB default) */

  /* Module-level pointer/scalar symbols we rebind via cuMemcpyHtoD. */
  unsigned long long sym_seed_pool_base;
  unsigned long long sym_seed_pool_offsets;
  unsigned long long sym_seed_pool_lens;
  unsigned long long sym_seed_pool_cumw;
  unsigned long long sym_seed_pool_count;
  unsigned long long sym_seed_pool_cumw_total;
  unsigned long long sym_prng_base;
  unsigned long long sym_reported_count;
  unsigned long long sym_reported_tid;
  unsigned long long sym_reported_lens;
  unsigned long long sym_mutation_array;
  unsigned long long sym_mutation_array_size;
  unsigned long long sym_havoc_stack_pow2;

  /* Extras state. */
  u32 extras_cnt_uploaded;
  u32 a_extras_cnt_uploaded;
  unsigned long long d_extras_base;
  unsigned long long d_extras_offsets;
  unsigned long long d_extras_lens;
  unsigned long long sym_extras_base;
  unsigned long long sym_extras_offsets;
  unsigned long long sym_extras_lens;
  unsigned long long sym_extras_cnt;
  unsigned long long d_a_extras_base;
  unsigned long long d_a_extras_offsets;
  unsigned long long d_a_extras_lens;
  unsigned long long sym_a_extras_base;
  unsigned long long sym_a_extras_offsets;
  unsigned long long sym_a_extras_lens;
  unsigned long long sym_a_extras_cnt;

  /* Last-seen mutation-array selector (so we only re-upload when changed). */
  u8 mut_array_input_mode;
  u8 mut_array_fuzz_mode;
  u8 mut_array_uploaded;   /* 0 until first upload */
```

- [ ] **Step 4: Add prototypes for the new API functions**

Insert at the bottom of the API surface block (after `coqui_shutdown` declaration on line 213):

```c
/* Refresh the GPU-resident seed pool for a new queue_cur havoc entry.
 * Slot 0 = queue_cur; slots 1..N-1 = weighted sample of other queue entries.
 * Uploaded on stream_pool and fenced against stream_a/b via a CUevent. */
void coqui_refresh_seed_pool(struct afl_state *afl);

/* Append one havoc slot to the pending batch — (flag<<24) | (seed_idx).
 * Same ping-pong flip + launch semantics as coqui_submit_input.
 * Returns 0 unconditionally. */
u8 coqui_submit_havoc_slot(struct afl_state *afl, u32 seed_idx, u8 flag);
```

- [ ] **Step 5: Build afl-fuzz to verify headers compile**

```bash
cd /home/gpizarro/cuAFL
make -j"$(nproc)"
```

Expected: afl-fuzz rebuilds; no references to new symbols fail because we haven't called them yet.

- [ ] **Step 6: Commit**

```bash
git add include/afl-fuzz-coqui.h
git commit -m "cuAFL: extend coqui_batch_t and coqui_ctx_t for GPU havoc"
```

### Task 2.2: Update `alloc_batch_half_cuda` and `free_batch_half_cuda`

**Files:**
- Modify: `src/afl-fuzz-coqui.c`

- [ ] **Step 1: Add allocation for slot_info and reported_* buffers**

Inside `alloc_batch_half_cuda` (starts at line 46), after the existing `cuMemAlloc` for `d_status` (line 66-67), insert:

```c
  /* slot_info — host pinned + device mirror. Per thread u32. */
  CUCHECK(cuMemHostAlloc((void**)&b->h_slot_info, ctx->batch_size * 4, 0));
  CUCHECK(cuMemAlloc(&p, ctx->batch_size * 4));
  b->d_slot_info = (unsigned long long)p;

  /* reported_slab + metadata. */
  size_t slab_bytes = (size_t)COQUI_REPORTED_CAP * ctx->max_input_size;
  CUCHECK(cuMemHostAlloc((void**)&b->h_reported_slab, slab_bytes, 0));
  CUCHECK(cuMemHostAlloc((void**)&b->h_reported_tid, COQUI_REPORTED_CAP * 4, 0));
  CUCHECK(cuMemHostAlloc((void**)&b->h_reported_lens, COQUI_REPORTED_CAP * 4, 0));
  CUCHECK(cuMemAlloc(&p, slab_bytes));
  b->d_reported_slab = (unsigned long long)p;
  CUCHECK(cuMemAlloc(&p, COQUI_REPORTED_CAP * 4));
  b->d_reported_tid = (unsigned long long)p;
  CUCHECK(cuMemAlloc(&p, COQUI_REPORTED_CAP * 4));
  b->d_reported_lens = (unsigned long long)p;
  CUCHECK(cuMemAlloc(&p, 4));
  b->d_reported_count = (unsigned long long)p;
  b->h_reported_count = 0;
```

- [ ] **Step 2: Add corresponding frees in `free_batch_half_cuda`**

Inside `free_batch_half_cuda` (starts at line 872), after the existing frees for `h_status`/`d_status`, add:

```c
  if (b->h_slot_info)     cuMemFreeHost(b->h_slot_info);
  if (b->h_reported_slab) cuMemFreeHost(b->h_reported_slab);
  if (b->h_reported_tid)  cuMemFreeHost(b->h_reported_tid);
  if (b->h_reported_lens) cuMemFreeHost(b->h_reported_lens);

  if (b->d_slot_info)      cuMemFree((CUdeviceptr)b->d_slot_info);
  if (b->d_reported_slab)  cuMemFree((CUdeviceptr)b->d_reported_slab);
  if (b->d_reported_tid)   cuMemFree((CUdeviceptr)b->d_reported_tid);
  if (b->d_reported_lens)  cuMemFree((CUdeviceptr)b->d_reported_lens);
  if (b->d_reported_count) cuMemFree((CUdeviceptr)b->d_reported_count);
```

- [ ] **Step 3: Also free host pinned in the force-reset path**

Add to the pinned-mem free block in `coqui_force_reset` (around line 855-864):

```c
  if (ctx->ping.h_slot_info)     cuMemFreeHost(ctx->ping.h_slot_info);
  if (ctx->ping.h_reported_slab) cuMemFreeHost(ctx->ping.h_reported_slab);
  if (ctx->ping.h_reported_tid)  cuMemFreeHost(ctx->ping.h_reported_tid);
  if (ctx->ping.h_reported_lens) cuMemFreeHost(ctx->ping.h_reported_lens);
  if (ctx->pong.h_slot_info)     cuMemFreeHost(ctx->pong.h_slot_info);
  if (ctx->pong.h_reported_slab) cuMemFreeHost(ctx->pong.h_reported_slab);
  if (ctx->pong.h_reported_tid)  cuMemFreeHost(ctx->pong.h_reported_tid);
  if (ctx->pong.h_reported_lens) cuMemFreeHost(ctx->pong.h_reported_lens);
```

- [ ] **Step 4: Build and commit**

```bash
cd /home/gpizarro/cuAFL && make -j"$(nproc)"
git add src/afl-fuzz-coqui.c
git commit -m "cuAFL: allocate slot_info + reported_slab in batch alloc/free paths"
```

### Task 2.3: Initialize seed pool + extras + symbol bindings in `coqui_init`

**Files:**
- Modify: `src/afl-fuzz-coqui.c`

- [ ] **Step 1: Add third stream creation (`stream_pool`)**

Inside `coqui_init`, after the existing `cuStreamCreate` calls at lines 197-201, add:

```c
  CUstream sp;
  CUCHECK(cuStreamCreate(&sp, CU_STREAM_NON_BLOCKING));
  ctx->stream_pool = (void *)sp;
```

- [ ] **Step 2: Add MAX_INPUT_SIZE guard**

Right after batch sizing at lines 186-194, insert:

```c
  /* Runtime bitcode was compiled with -DMAX_INPUT_SIZE=4096 (or whatever
   * the target was built with). Host-side max_length must not exceed this
   * or kernel .local scratch will truncate mutated inputs. */
  if (ctx->max_input_size > 4096) {
    FATAL("coqui_mode: afl->max_length (%u) > MAX_INPUT_SIZE (4096). "
          "Rebuild target cubin with -DMAX_INPUT_SIZE=%u.",
          ctx->max_input_size, ctx->max_input_size);
  }
```

- [ ] **Step 3: Allocate seed pool (ping-pong) — host pinned + device**

After the existing ping-pong `alloc_batch_half_cuda` calls (line 205), add:

```c
  /* Seed pool: 1 MB per side, capacity 256 slots. Packed (variable-length). */
  ctx->seed_pool_cap_bytes = 1024u * 1024u;
  CUCHECK(cuMemHostAlloc((void**)&ctx->h_seed_pool_bytes_a, ctx->seed_pool_cap_bytes, 0));
  CUCHECK(cuMemHostAlloc((void**)&ctx->h_seed_pool_bytes_b, ctx->seed_pool_cap_bytes, 0));
  CUCHECK(cuMemHostAlloc((void**)&ctx->h_seed_pool_offsets_a, 256 * 4, 0));
  CUCHECK(cuMemHostAlloc((void**)&ctx->h_seed_pool_offsets_b, 256 * 4, 0));
  CUCHECK(cuMemHostAlloc((void**)&ctx->h_seed_pool_lens_a,    256 * 4, 0));
  CUCHECK(cuMemHostAlloc((void**)&ctx->h_seed_pool_lens_b,    256 * 4, 0));
  CUCHECK(cuMemHostAlloc((void**)&ctx->h_seed_pool_cumw_a,    256 * 4, 0));
  CUCHECK(cuMemHostAlloc((void**)&ctx->h_seed_pool_cumw_b,    256 * 4, 0));

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

  ctx->seed_pool_active = 0;
```

- [ ] **Step 4: Resolve module globals for all new symbols**

After the existing `cuModuleGetGlobal(..., "__coqui_virgin_map", ...)` at line 121-125, add resolver calls. Keep a small helper to avoid repetition:

```c
#define BIND_SYM(field, name, expected_sz) do {                               \
  CUdeviceptr _sp; size_t _sz;                                                \
  CUCHECK(cuModuleGetGlobal(&_sp, &_sz, mod, name));                          \
  if (_sz != (expected_sz)) FATAL(name " size %zu != %zu", _sz, (expected_sz));\
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
  BIND_SYM(sym_extras_base,          "__coqui_extras_base",          sizeof(void*));
  BIND_SYM(sym_extras_offsets,       "__coqui_extras_offsets",       sizeof(void*));
  BIND_SYM(sym_extras_lens,          "__coqui_extras_lens",          sizeof(void*));
  BIND_SYM(sym_extras_cnt,           "__coqui_extras_cnt",           4);
  BIND_SYM(sym_a_extras_base,        "__coqui_a_extras_base",        sizeof(void*));
  BIND_SYM(sym_a_extras_offsets,     "__coqui_a_extras_offsets",     sizeof(void*));
  BIND_SYM(sym_a_extras_lens,        "__coqui_a_extras_lens",        sizeof(void*));
  BIND_SYM(sym_a_extras_cnt,         "__coqui_a_extras_cnt",         4);
#undef BIND_SYM
```

- [ ] **Step 5: Bind reported_tid/lens pointer symbols to per-batch buffers**

Just before `ctx->pending = &ctx->ping;` at line 206, we need to write the current side's reported_tid/lens device pointers into the module globals. But reported is per-batch and flips with ping-pong, so do it in `coqui_launch_batch` (task 2.5) instead. Here, just initialize `extras_cnt_uploaded = 0`, `a_extras_cnt_uploaded = 0`, `mut_array_uploaded = 0`.

- [ ] **Step 6: Extras one-time upload**

After symbol bindings, if `afl->extras_cnt > 0`, pack + upload once. Pattern:

```c
  if (afl->extras_cnt > 0) {
    /* Compute total byte length */
    size_t total = 0;
    for (u32 i = 0; i < afl->extras_cnt; ++i) total += afl->extras[i].len;
    CUdeviceptr p_ex; CUCHECK(cuMemAlloc(&p_ex, total));
    ctx->d_extras_base = (unsigned long long)p_ex;

    u32 *off_h = ck_alloc(afl->extras_cnt * 4);
    u32 *len_h = ck_alloc(afl->extras_cnt * 4);
    u8  *pak   = ck_alloc(total);
    size_t cur = 0;
    for (u32 i = 0; i < afl->extras_cnt; ++i) {
      off_h[i] = (u32)cur;
      len_h[i] = (u32)afl->extras[i].len;
      memcpy(pak + cur, afl->extras[i].data, afl->extras[i].len);
      cur += afl->extras[i].len;
    }
    CUCHECK(cuMemcpyHtoD((CUdeviceptr)ctx->d_extras_base, pak, total));

    CUdeviceptr p_off; CUCHECK(cuMemAlloc(&p_off, afl->extras_cnt * 4));
    CUCHECK(cuMemcpyHtoD(p_off, off_h, afl->extras_cnt * 4));
    ctx->d_extras_offsets = (unsigned long long)p_off;

    CUdeviceptr p_len; CUCHECK(cuMemAlloc(&p_len, afl->extras_cnt * 4));
    CUCHECK(cuMemcpyHtoD(p_len, len_h, afl->extras_cnt * 4));
    ctx->d_extras_lens = (unsigned long long)p_len;

    /* Write the pointer symbols so device sees our buffers */
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
```

- [ ] **Step 7: Build and smoke**

```bash
cd /home/gpizarro/cuAFL && make -j"$(nproc)"
timeout 30 ./afl-fuzz -c 0 -i /home/gpizarro/cjson-in -o /tmp/cjson-phase2a \
    -- ./cjson_fuzzer.sm_75.cubin 2>&1 | tail -20
```

Expected: fuzzer still runs normally. The `BIND_SYM` calls succeed because our runtime header declares the symbols. Nothing exercises the new path yet.

If a `BIND_SYM` FATALs with "symbol not found", the runtime isn't exporting that symbol — go back to Task 1.2 and verify the extern declaration uses `extern` and has no `static` somewhere upstream.

- [ ] **Step 8: Commit**

```bash
git add src/afl-fuzz-coqui.c
git commit -m "cuAFL: allocate seed pool + extras + bind module symbols in coqui_init"
```

### Task 2.4: Implement `coqui_refresh_seed_pool`

**Files:**
- Modify: `src/afl-fuzz-coqui.c`

- [ ] **Step 1: Add a helper for weighted sampling without replacement**

Before `coqui_launch_batch` (around line 395), add:

```c
/* Pick one queue entry weighted by queue_entry.weight. O(Q log Q) first
 * call per refresh (builds CDF); subsequent calls in the same refresh are
 * O(log Q). For v1 we just build the CDF once and sample repeatedly. */
static struct queue_entry *coqui_weighted_pick(afl_state_t *afl,
                                                 u64 *cum_weights_scratch) {
  u32 Q = afl->queued_items;
  if (Q == 0) return NULL;
  /* Build CDF on first use (skip disabled) */
  u64 total = 0;
  for (u32 i = 0; i < Q; ++i) {
    struct queue_entry *q = afl->queue_buf[i];
    u64 w = (q && !q->disabled) ? (u64)q->weight : 0;
    total += w;
    cum_weights_scratch[i] = total;
  }
  if (total == 0) return NULL;

  u64 r = rand_below(afl, (u32)total);   /* AFL's existing host PRNG */
  u32 lo = 0, hi = Q;
  while (lo < hi) {
    u32 m = (lo + hi) >> 1;
    if (cum_weights_scratch[m] <= r) lo = m + 1; else hi = m;
  }
  if (lo >= Q) lo = Q - 1;
  return afl->queue_buf[lo];
}
```

- [ ] **Step 2: Implement `coqui_refresh_seed_pool`**

```c
void coqui_refresh_seed_pool(afl_state_t *afl) {
  coqui_ctx_t *ctx = afl->coqui;
  CUstream sp = (CUstream)ctx->stream_pool;
  u8 next = 1 - ctx->seed_pool_active;

  u8  *dst_bytes   = (next == 0) ? ctx->h_seed_pool_bytes_a   : ctx->h_seed_pool_bytes_b;
  u32 *dst_offsets = (next == 0) ? ctx->h_seed_pool_offsets_a : ctx->h_seed_pool_offsets_b;
  u32 *dst_lens    = (next == 0) ? ctx->h_seed_pool_lens_a    : ctx->h_seed_pool_lens_b;
  u32 *dst_cumw    = (next == 0) ? ctx->h_seed_pool_cumw_a    : ctx->h_seed_pool_cumw_b;

  unsigned long long d_bytes   = (next == 0) ? ctx->d_seed_pool_bytes_a   : ctx->d_seed_pool_bytes_b;
  unsigned long long d_offsets = (next == 0) ? ctx->d_seed_pool_offsets_a : ctx->d_seed_pool_offsets_b;
  unsigned long long d_lens    = (next == 0) ? ctx->d_seed_pool_lens_a    : ctx->d_seed_pool_lens_b;
  unsigned long long d_cumw    = (next == 0) ? ctx->d_seed_pool_cumw_a    : ctx->d_seed_pool_cumw_b;

  /* Slot 0: queue_cur (always). */
  u32 cursor = 0;
  u32 count  = 0;
  u64 cumw = 0;

  if (afl->queue_cur && afl->queue_cur->buf && afl->queue_cur->len > 0 &&
      afl->queue_cur->len <= ctx->seed_pool_cap_bytes) {
    memcpy(dst_bytes + cursor, afl->queue_cur->buf, afl->queue_cur->len);
    dst_offsets[0] = cursor;
    dst_lens[0]    = afl->queue_cur->len;
    cursor += afl->queue_cur->len;
    cumw += (u64)afl->queue_cur->weight;
    dst_cumw[0] = (u32)cumw;
    count = 1;
  } else {
    /* No usable queue_cur — shouldn't happen; early-return with empty pool. */
    ctx->h_seed_pool_count_next = 0;
    ctx->h_seed_pool_cumw_total_next = 0;
    ctx->seed_pool_active = next;
    return;
  }

  /* Slots 1..255: weighted-random-with-replacement from queue_buf. */
  u64 *cdf = ck_alloc(afl->queued_items * sizeof(u64));
  for (u32 i = count; i < 256; ++i) {
    struct queue_entry *q = coqui_weighted_pick(afl, cdf);
    if (!q || !q->buf || q->len == 0) break;
    if (cursor + q->len > ctx->seed_pool_cap_bytes) break;
    memcpy(dst_bytes + cursor, q->buf, q->len);
    dst_offsets[i] = cursor;
    dst_lens[i]    = q->len;
    cursor += q->len;
    cumw += (u64)q->weight;
    dst_cumw[i] = (u32)(cumw & 0xFFFFFFFFu);
    count++;
  }
  ck_free(cdf);

  /* HtoD on stream_pool — overlaps in-flight kernel on stream_a/b. */
  CUCHECK(cuMemcpyHtoDAsync((CUdeviceptr)d_bytes,   dst_bytes,   cursor, sp));
  CUCHECK(cuMemcpyHtoDAsync((CUdeviceptr)d_offsets, dst_offsets, count * 4, sp));
  CUCHECK(cuMemcpyHtoDAsync((CUdeviceptr)d_lens,    dst_lens,    count * 4, sp));
  CUCHECK(cuMemcpyHtoDAsync((CUdeviceptr)d_cumw,    dst_cumw,    count * 4, sp));

  /* Fence stream_a/b so subsequent kernel launches see the new pool. */
  CUevent ev;
  CUCHECK(cuEventCreate(&ev, CU_EVENT_DISABLE_TIMING));
  CUCHECK(cuEventRecord(ev, sp));
  CUCHECK(cuStreamWaitEvent((CUstream)ctx->stream_a, ev, 0));
  CUCHECK(cuStreamWaitEvent((CUstream)ctx->stream_b, ev, 0));
  CUCHECK(cuEventDestroy(ev));

  /* Flip active: write the new pointer values into the pointer-symbols.
   * Also write scalar count + cumw_total. */
  CUCHECK(cuMemcpyHtoD((CUdeviceptr)ctx->sym_seed_pool_base,    &d_bytes,   8));
  CUCHECK(cuMemcpyHtoD((CUdeviceptr)ctx->sym_seed_pool_offsets, &d_offsets, 8));
  CUCHECK(cuMemcpyHtoD((CUdeviceptr)ctx->sym_seed_pool_lens,    &d_lens,    8));
  CUCHECK(cuMemcpyHtoD((CUdeviceptr)ctx->sym_seed_pool_cumw,    &d_cumw,    8));
  u32 cnt_u32 = count;
  u32 cumw_u32 = (u32)(cumw & 0xFFFFFFFFu);
  CUCHECK(cuMemcpyHtoD((CUdeviceptr)ctx->sym_seed_pool_count,      &cnt_u32,  4));
  CUCHECK(cuMemcpyHtoD((CUdeviceptr)ctx->sym_seed_pool_cumw_total, &cumw_u32, 4));

  ctx->h_seed_pool_count_next = count;
  ctx->h_seed_pool_cumw_total_next = cumw_u32;
  ctx->seed_pool_active = next;

  /* Opportunistic mutation_array + havoc_stack_pow2 upload if needed. */
  if (!ctx->mut_array_uploaded ||
      ctx->mut_array_input_mode != afl->input_mode ||
      ctx->mut_array_fuzz_mode  != afl->fuzz_mode) {
    /* Include include/afl-mutations.h once at the top of afl-fuzz-coqui.c
     * if not already; the arrays there are u32[256] by input/fuzz mode. */
    extern u32 binary_array[];
    extern u32 text_array[];
    extern u32 mutation_strategy_exploration_binary[];
    extern u32 mutation_strategy_exploitation_binary[];
    u32 *active = binary_array;   /* default/generic exploration */
    u32  size   = MUT_BIN_ARRAY_SIZE;
    if (afl->input_mode == 1 /* TEXT */) {
      if (afl->fuzz_mode == 0) { active = binary_array; size = MUT_BIN_ARRAY_SIZE; }
      else                     { active = text_array;   size = MUT_TXT_ARRAY_SIZE; }
    } else if (afl->input_mode == 2 /* BINARY */) {
      if (afl->fuzz_mode == 0) { active = mutation_strategy_exploration_binary;
                                  size = MUT_STRATEGY_ARRAY_SIZE; }
      else                     { active = mutation_strategy_exploitation_binary;
                                  size = MUT_STRATEGY_ARRAY_SIZE; }
    }
    CUCHECK(cuMemcpyHtoD((CUdeviceptr)ctx->sym_mutation_array, active, 256 * 4));
    CUCHECK(cuMemcpyHtoD((CUdeviceptr)ctx->sym_mutation_array_size, &size, 4));
    ctx->mut_array_input_mode = afl->input_mode;
    ctx->mut_array_fuzz_mode  = afl->fuzz_mode;
    ctx->mut_array_uploaded = 1;
  }
  u32 pow2 = afl->havoc_stack_pow2;
  CUCHECK(cuMemcpyHtoD((CUdeviceptr)ctx->sym_havoc_stack_pow2, &pow2, 4));

  /* Opportunistic a_extras re-upload if grown. */
  if (afl->a_extras_cnt != ctx->a_extras_cnt_uploaded) {
    /* Implementation mirrors the one-time extras upload in coqui_init
     * (task 2.3 step 6) but uses afl->a_extras[]. Free old device buffers
     * first if they exist. Skipped here for brevity — pattern-identical. */
    /* TODO(phase3-bench): if a_extras hot-growth proves load-bearing, revisit. */
    ctx->a_extras_cnt_uploaded = afl->a_extras_cnt;
  }
}
```

- [ ] **Step 3: Add `#include "afl-mutations.h"` at top of afl-fuzz-coqui.c**

So `binary_array`, `text_array`, etc. resolve. If the macro `INTERESTING_32` issue from afl-mutations.h line 37 bites (it requires `include/config.h` for the define), include that too:

```c
#include "afl-fuzz.h"
#include "afl-fuzz-coqui.h"
#include "forkserver.h"
#include "afl-mutations.h"    /* <-- add this */
```

- [ ] **Step 4: Build + smoke**

```bash
cd /home/gpizarro/cuAFL && make -j"$(nproc)"
timeout 30 ./afl-fuzz -c 0 -i /home/gpizarro/cjson-in -o /tmp/cjson-phase2b \
    -- ./cjson_fuzzer.sm_75.cubin 2>&1 | tail -20
```

Expected: build green. Functional smoke still unchanged.

- [ ] **Step 5: Commit**

```bash
git add src/afl-fuzz-coqui.c
git commit -m "cuAFL: implement coqui_refresh_seed_pool for GPU-havoc seed pool"
```

### Task 2.5: Implement `coqui_submit_havoc_slot` and update `coqui_launch_batch`

**Files:**
- Modify: `src/afl-fuzz-coqui.c`

- [ ] **Step 1: Implement `coqui_submit_havoc_slot`**

Add near `coqui_submit_input` (around line 764):

```c
u8 coqui_submit_havoc_slot(afl_state_t *afl, u32 seed_idx, u8 flag) {
  coqui_ctx_t  *ctx = afl->coqui;
  coqui_batch_t *b  = ctx->pending;

  ctx->total_submits++;
  afl->fsrv.total_execs++;

  if (b->n_inputs == ctx->batch_size) {
    coqui_launch_batch(afl, b);
    coqui_batch_t *tmp = ctx->pending;
    ctx->pending = ctx->executing;
    ctx->executing = tmp;
    b = ctx->pending;
    if (b->n_inputs > 0) {
      if (coqui_await_and_process(afl, b) == 1) return 0;
      b->n_inputs = 0;
      b->bytes_used = 0;
    }
  }

  b->h_slot_info[b->n_inputs] = ((u32)flag << 24) | (seed_idx & 0xFFFFFFu);
  /* offsets/lens default to 0 for pure flag=1 slots; kernel ignores them */
  b->h_offsets[b->n_inputs] = 0;
  b->h_input_lens[b->n_inputs] = 0;
  b->n_inputs++;
  return 0;
}
```

- [ ] **Step 2: Update `coqui_launch_batch` to issue new HtoD + bind reported_* pointers**

Modify `coqui_launch_batch` (around line 395). After the existing `cuMemsetD8Async` for `d_status`, add:

```c
  CUCHECK(cuMemcpyHtoDAsync((CUdeviceptr)b->d_slot_info,
                             b->h_slot_info, ctx->batch_size * 4, s));
  /* Zero the reported_count counter for this batch */
  CUCHECK(cuMemsetD32Async((CUdeviceptr)b->d_reported_count, 0, 1, s));
```

Also bind this batch's reported_tid/lens pointers via the module symbols (so the kernel's compact-write block targets this batch's buffers):

```c
  CUCHECK(cuMemcpyHtoDAsync((CUdeviceptr)ctx->sym_reported_count,
                              &b->d_reported_count, 8, s));
  CUCHECK(cuMemcpyHtoDAsync((CUdeviceptr)ctx->sym_reported_tid,
                              &b->d_reported_tid, 8, s));
  CUCHECK(cuMemcpyHtoDAsync((CUdeviceptr)ctx->sym_reported_lens,
                              &b->d_reported_lens, 8, s));
```

Wait — the pointer-symbols being written via a cuMemcpyHtoD**Async** from a pointer-to-pointer in host memory is sketchy. The `&b->d_reported_tid` is a u64 living in the `coqui_batch_t` struct in host heap. That must remain stable until the async copy completes. Since `b` persists until shutdown and we never move these fields, the pointer is stable — safe.

**Tension**: we're writing to pointer symbols via a HtoD on the batch's stream, but stream_pool uploads also write to symbols. If a stream_pool memcpy races a stream_a memcpy to the same symbol, it's undefined behavior. But since `coqui_refresh_seed_pool` only writes seed_pool symbols and `coqui_launch_batch` only writes reported_* symbols, there's no actual conflict. Document this invariant with a comment above each.

- [ ] **Step 3: Update the kernel launch args list**

The `void *args[]` array in `coqui_launch_batch` (line 429-435) currently has 5 elements. Extend to 7:

```c
  void *args[] = {
    (void *)&b->d_input_bytes,
    (void *)&b->d_offsets,
    (void *)&b->d_input_lens,
    (void *)&b->d_slot_info,     /* NEW */
    (void *)&b->d_novelty,
    (void *)&b->d_status,
    (void *)&b->d_reported_slab, /* NEW */
  };
```

- [ ] **Step 4: Write `prng_base` before each launch**

Just before `cuLaunchKernel` (line 437), add:

```c
  u64 prng_base = get_rand_seed(afl);   /* AFL host PRNG — 64-bit mix of time + afl->rand_seed */
  CUCHECK(cuMemcpyHtoDAsync((CUdeviceptr)ctx->sym_prng_base, &prng_base, 8, s));
```

If `get_rand_seed` doesn't exist in AFL, use `rand_below(afl, 0xFFFFFFFFu) | ((u64)rand_below(afl, 0xFFFFFFFFu) << 32)`.

- [ ] **Step 5: Add DtoH for reported_* in `coqui_launch_batch`**

After the existing DtoH for `d_status` (line 445-446), add:

```c
  /* reported_count first (small, cheap, stream-serial) — we'll read it
   * synchronously after stream completes to decide how much of slab/tid/lens
   * to transfer. For async pipelining, just transfer the full cap; it's
   * 512 * (max_input_size + 8) ≈ 2 MB, still small. Simpler. */
  CUCHECK(cuMemcpyDtoHAsync(&b->h_reported_count,
                             (CUdeviceptr)b->d_reported_count, 4, s));
  CUCHECK(cuMemcpyDtoHAsync(b->h_reported_slab, (CUdeviceptr)b->d_reported_slab,
                             COQUI_REPORTED_CAP * ctx->max_input_size, s));
  CUCHECK(cuMemcpyDtoHAsync(b->h_reported_tid,  (CUdeviceptr)b->d_reported_tid,
                             COQUI_REPORTED_CAP * 4, s));
  CUCHECK(cuMemcpyDtoHAsync(b->h_reported_lens, (CUdeviceptr)b->d_reported_lens,
                             COQUI_REPORTED_CAP * 4, s));
```

- [ ] **Step 6: Build + smoke**

```bash
cd /home/gpizarro/cuAFL && make -j"$(nproc)"
timeout 30 ./afl-fuzz -c 0 -i /home/gpizarro/cjson-in -o /tmp/cjson-phase2c \
    -- ./cjson_fuzzer.sm_75.cubin 2>&1 | tail -20
```

Expected: cjson_fuzzer runs. Kernel now receives the 7-arg call, so cubin must be rebuilt too. Phase 2 step 1-in-Task-2.6 below rebuilds the kernel pass.

If launch fails with an arg-count mismatch, it's because the cubin was built with the OLD 5-arg kernel signature. That's fixed by Task 2.6.

- [ ] **Step 7: Commit**

```bash
git add src/afl-fuzz-coqui.c
git commit -m "cuAFL: implement coqui_submit_havoc_slot; extend launch_batch HtoD/DtoH"
```

### Task 2.6: Extend `FuzzEntry.cpp` kernel signature and add havoc dispatch blocks

**Files:**
- Modify: `coqui_mode/passes/FuzzEntry.cpp`

This task changes the IR generated for `__coqui_fuzz_kernel`. The existing kernel has signature `(i8p, i8p, i8p, i8p, i8p)` — input_bytes, offsets, lens, novelty, status. We're going to 7 args with `slot_info` inserted at position 3 (after lens) and `reported_slab` appended at the end.

- [ ] **Step 1: Update kernel FunctionType and arg-naming**

Find the FunctionType construction at `coqui_mode/passes/FuzzEntry.cpp:45-49`:

```cpp
  FunctionType *KernelType = FunctionType::get(
    voidT, {i8p, i8p, i8p, i8p, i8p}, false);
```

Replace with 7 args:

```cpp
  FunctionType *KernelType = FunctionType::get(
    voidT, {i8p, i8p, i8p, i8p, i8p, i8p, i8p}, false);
```

And update the arg naming at lines 52-57 — add two new args between `lens` and `novelty`, and one at the end:

```cpp
  auto argIt = Kernel->arg_begin();
  Value *inputBytes  = &*argIt++; inputBytes->setName("input_bytes");
  Value *offsetsArg  = &*argIt++; offsetsArg->setName("offsets");
  Value *lensArg     = &*argIt++; lensArg->setName("lens");
  Value *slotInfoArg = &*argIt++; slotInfoArg->setName("slot_info");   /* NEW */
  Value *noveltyArg  = &*argIt++; noveltyArg->setName("novelty");
  Value *statusArg   = &*argIt++; statusArg->setName("status");
  Value *reportedSlabArg = &*argIt++; reportedSlabArg->setName("reported_slab"); /* NEW */
```

- [ ] **Step 2: Replace the single `if (len == 0) goto exit` block with flag-dispatch**

The existing code at lines 126-132 does the `len == 0` early-exit. Replace that block (right after `tid = __coqui_fuzz_tid()` at line 124) with:

```cpp
  /* slot = slot_info[tid] */
  Value *slotPtr = Builder.CreateGEP(i32, slotInfoArg, {tid}, "slot_tid");
  Value *slot    = Builder.CreateLoad(i32, slotPtr, "slot");
  Value *flagV   = Builder.CreateLShr(slot, ConstantInt::get(i32, 24), "flag");
  Value *idxV    = Builder.CreateAnd(slot,
                      ConstantInt::get(i32, 0xFFFFFF), "seed_idx");

  /* Decide: flag == 0 → premut; flag == 1 → havoc; else → exit. */
  BasicBlock *PremutBB = BasicBlock::Create(C, "premut", Kernel);
  BasicBlock *HavocBB  = BasicBlock::Create(C, "havoc",  Kernel);
  Value *isPremut = Builder.CreateICmpEQ(flagV, ConstantInt::get(i32, 0));
  BasicBlock *DispatchHavocBB = BasicBlock::Create(C, "dispatch_havoc", Kernel);
  Builder.CreateCondBr(isPremut, PremutBB, DispatchHavocBB);

  Builder.SetInsertPoint(DispatchHavocBB);
  Value *isHavoc = Builder.CreateICmpEQ(flagV, ConstantInt::get(i32, 1));
  Builder.CreateCondBr(isHavoc, HavocBB, ExitBB);

  /* PREMUT path — existing logic lifted here. */
  Builder.SetInsertPoint(PremutBB);
  Value *pm_lensPtr = Builder.CreateGEP(i32, lensArg, {tid}, "pm_lens_tid");
  Value *pm_len     = Builder.CreateLoad(i32, pm_lensPtr, "pm_len");
  Value *pm_isEmpty = Builder.CreateICmpEQ(pm_len, ConstantInt::get(i32, 0));
  BasicBlock *PremutContBB = BasicBlock::Create(C, "premut_cont", Kernel);
  Builder.CreateCondBr(pm_isEmpty, ExitBB, PremutContBB);

  Builder.SetInsertPoint(PremutContBB);
  Value *pm_offPtr = Builder.CreateGEP(i32, offsetsArg, {tid}, "pm_offsets_tid");
  Value *pm_off    = Builder.CreateLoad(i32, pm_offPtr, "pm_off");
  Value *pm_off64  = Builder.CreateZExt(pm_off, i64, "pm_off64");
  Value *pm_inputPtr = Builder.CreateGEP(i8, inputBytes, {pm_off64}, "pm_input_ptr");
  Value *pm_len64  = Builder.CreateZExt(pm_len, i64, "pm_len64");
  Builder.CreateBr(RunBB);

  /* HAVOC path — see Task 2.7 for the body. For now jump-to-exit. */
  Builder.SetInsertPoint(HavocBB);
  Builder.CreateBr(ExitBB);

  /* RunBB: existing __coqui_memory_init + __coqui_fuzz_execute + classify
     + virgin_compare logic lifted from current lines 134-183. It reads
     input_ptr and len via PHI nodes merging the two paths. */
  Builder.SetInsertPoint(RunBB);
  PHINode *inputPtrPhi = Builder.CreatePHI(i8p, 2, "input_ptr");
  PHINode *lenPhi      = Builder.CreatePHI(i64, 2, "len");
  inputPtrPhi->addIncoming(pm_inputPtr, PremutContBB);
  lenPhi->addIncoming(pm_len64, PremutContBB);
  /* havoc predecessors added in Task 2.7 */

  /* Existing run-block body: memory_init, fuzz_execute(inputPtrPhi, lenPhi),
   * classify_and_sig, virgin_compare — keep the CFG identical but replace
   * inputPtr / len64 references with inputPtrPhi / lenPhi. */
```

The change is large — essentially re-templating the existing run block against a PHI input. Be very careful: the line numbers in the original file shift as you edit. Read the file fresh before making changes.

- [ ] **Step 3: Build pass + runtime + cubin**

```bash
cd /home/gpizarro/cuAFL/coqui_mode && bash build_coqui_support.sh
# Rebuild cjson_fuzzer cubin to pick up the new kernel signature
# (Use whatever target-build mechanism exists in the project.)
```

Expected: pass compiles; cubin rebuilds; `cuobjdump --dump-elf-symbols cjson_fuzzer.sm_75.cubin` shows `__coqui_fuzz_kernel` with new arg count.

- [ ] **Step 4: Smoke**

```bash
cd /home/gpizarro/cuAFL
timeout 30 ./afl-fuzz -c 0 -i /home/gpizarro/cjson-in -o /tmp/cjson-phase2d \
    -- ./cjson_fuzzer.sm_75.cubin 2>&1 | tail -20
```

Expected: fuzzer runs. All slots submitted today are flag=0 (havoc_stage hasn't been flipped yet), so behaviour is identical to pre-change. No WARNF/FATAL.

- [ ] **Step 5: Commit**

```bash
git add coqui_mode/passes/FuzzEntry.cpp
git commit -m "cuAFL: extend __coqui_fuzz_kernel to 7 args with flag-dispatch blocks"
```

### Task 2.7: Fill in the havoc dispatch block

**Files:**
- Modify: `coqui_mode/passes/FuzzEntry.cpp`

The `HavocBB` we left empty in Task 2.6 needs to: read the seed pool offsets/lens for `seed_idx`, alloca a `.local scratch[MAX_INPUT_SIZE]`, `memcpy` the seed into scratch, call `__coqui_havoc_mutate(scratch, base_len, MAX_INPUT_SIZE, seed_idx, &prng_state)`, then join `RunBB` with `input_ptr = scratch`, `len = mutated_len`.

- [ ] **Step 1: Declare helpers and the new IR for HavocBB**

Replace the `HavocBB` contents (the `Builder.CreateBr(ExitBB)` stub from Task 2.6) with the full havoc IR. Add declarations near the existing `FunctionCallee` declarations at lines 64-78:

```cpp
  /* Seed pool pointer/value externs used by havoc path. */
  GlobalVariable *SeedPoolBase = new GlobalVariable(
    M, i8p, /*isConstant*/false, GlobalValue::ExternalLinkage,
    nullptr, "__coqui_seed_pool_base");
  PointerType *i32p = PointerType::get(C, 0);
  GlobalVariable *SeedPoolOffsets = new GlobalVariable(
    M, i32p, false, GlobalValue::ExternalLinkage,
    nullptr, "__coqui_seed_pool_offsets");
  GlobalVariable *SeedPoolLens = new GlobalVariable(
    M, i32p, false, GlobalValue::ExternalLinkage,
    nullptr, "__coqui_seed_pool_lens");
  GlobalVariable *PrngBaseG = new GlobalVariable(
    M, i64, false, GlobalValue::ExternalLinkage,
    nullptr, "__coqui_prng_base");

  FunctionType *MutateTy = FunctionType::get(
    i32, {i8p, i32, i32, i32, /*i64p*/ PointerType::get(C, 0)}, false);
  FunctionCallee MutateFn = M.getOrInsertFunction("__coqui_havoc_mutate", MutateTy);

  /* memcpy intrinsic — use LLVM's llvm.memcpy.p0.p0.i64 */
  FunctionType *MemcpyTy = FunctionType::get(voidT,
    {i8p, i8p, i64, Type::getInt1Ty(C)}, false);
  FunctionCallee MemcpyFn = M.getOrInsertFunction(
    "llvm.memcpy.p0.p0.i64", MemcpyTy);
```

Replace the empty HavocBB stub with:

```cpp
  Builder.SetInsertPoint(HavocBB);

  /* base_len = __coqui_seed_pool_lens[idx]; if 0, goto exit */
  Value *poolLensBase = Builder.CreateLoad(i32p, SeedPoolLens, "pool_lens_base");
  Value *baseLenPtr   = Builder.CreateGEP(i32, poolLensBase, {idxV}, "base_len_ptr");
  Value *baseLen      = Builder.CreateLoad(i32, baseLenPtr, "base_len");
  Value *baseLenIsZero = Builder.CreateICmpEQ(baseLen, ConstantInt::get(i32, 0));
  BasicBlock *HavocContBB = BasicBlock::Create(C, "havoc_cont", Kernel);
  Builder.CreateCondBr(baseLenIsZero, ExitBB, HavocContBB);

  Builder.SetInsertPoint(HavocContBB);

  /* scratch: alloca [MAX_INPUT_SIZE x i8] on the stack (emits to .local) */
  ArrayType *ScratchTy = ArrayType::get(i8, 4096); /* MAX_INPUT_SIZE default */
  AllocaInst *Scratch = Builder.CreateAlloca(ScratchTy, nullptr, "scratch");

  /* scratch_ptr = &scratch[0] */
  Value *scratchPtr = Builder.CreateGEP(
    ScratchTy, Scratch,
    {ConstantInt::get(i32, 0), ConstantInt::get(i32, 0)},
    "scratch_ptr");

  /* src = seed_pool_base + seed_pool_offsets[idx] */
  Value *poolOffsetsBase = Builder.CreateLoad(i32p, SeedPoolOffsets, "pool_off_base");
  Value *seedOffPtr = Builder.CreateGEP(i32, poolOffsetsBase, {idxV}, "seed_off_ptr");
  Value *seedOff    = Builder.CreateLoad(i32, seedOffPtr, "seed_off");
  Value *seedOff64  = Builder.CreateZExt(seedOff, i64, "seed_off64");
  Value *poolBasePtr = Builder.CreateLoad(i8p, SeedPoolBase, "pool_base_ptr");
  Value *srcPtr = Builder.CreateGEP(i8, poolBasePtr, {seedOff64}, "seed_src");

  /* memcpy(scratch, src, base_len) */
  Value *baseLen64 = Builder.CreateZExt(baseLen, i64, "base_len64");
  Builder.CreateCall(MemcpyFn, {scratchPtr, srcPtr, baseLen64,
                                  ConstantInt::getFalse(C)});

  /* prng state: alloca u64; init with splitmix64(prng_base ^ tid) */
  AllocaInst *PrngSlot = Builder.CreateAlloca(i64, nullptr, "prng_state");
  Value *prngBase = Builder.CreateLoad(i64, PrngBaseG, "prng_base_v");
  Value *tid64    = Builder.CreateZExt(tid, i64, "tid64_prng");
  Value *prngInit = Builder.CreateXor(prngBase, tid64, "prng_init");
  Builder.CreateStore(prngInit, PrngSlot);

  /* mutated_len = __coqui_havoc_mutate(scratch, base_len, 4096, idx, &prng) */
  Value *maxLen = ConstantInt::get(i32, 4096);
  Value *mutLen = Builder.CreateCall(MutateFn,
      {scratchPtr, baseLen, maxLen, idxV, PrngSlot}, "mut_len");

  Value *mutLen64 = Builder.CreateZExt(mutLen, i64, "mut_len64");

  /* Add incoming to RunBB's PHIs */
  inputPtrPhi->addIncoming(scratchPtr, HavocContBB);
  lenPhi->addIncoming(mutLen64, HavocContBB);

  Builder.CreateBr(RunBB);
```

- [ ] **Step 2: Build pass + runtime + cubin**

```bash
cd /home/gpizarro/cuAFL/coqui_mode && bash build_coqui_support.sh
# Rebuild cjson_fuzzer cubin
```

Expected: build green. If LLVM complains about undefined `__coqui_havoc_mutate` or `llvm.memcpy.p0.p0.i64`, verify the runtime.bc was linked in (check the build_coqui_support.sh llvm-link list).

- [ ] **Step 3: Smoke**

Still no caller of havoc path — all slots are flag=0. Run cjson to confirm nothing regressed:

```bash
timeout 30 ./afl-fuzz -c 0 -i /home/gpizarro/cjson-in -o /tmp/cjson-phase2e \
    -- ./cjson_fuzzer.sm_75.cubin 2>&1 | tail -20
```

Expected: normal operation.

- [ ] **Step 4: Commit**

```bash
git add coqui_mode/passes/FuzzEntry.cpp
git commit -m "cuAFL: wire __coqui_havoc_mutate call into kernel havoc dispatch"
```

### Task 2.8: Add compact-report block at end of `__coqui_fuzz_kernel`

**Files:**
- Modify: `coqui_mode/passes/FuzzEntry.cpp`

After `virgin_compare` runs, emit IR that atomically increments `__coqui_reported_count` and writes `scratch` + `tid` + `mutated_len` into the reporting slab when a novelty bit is set or a crash is recorded in `status[tid]`.

- [ ] **Step 1: Declare more externs near the other globals**

```cpp
  GlobalVariable *ReportedCountG = new GlobalVariable(
    M, i32, false, GlobalValue::ExternalLinkage,
    nullptr, "__coqui_reported_count");
  GlobalVariable *ReportedTidPtrG = new GlobalVariable(
    M, i32p, false, GlobalValue::ExternalLinkage,
    nullptr, "__coqui_reported_tid");
  GlobalVariable *ReportedLensPtrG = new GlobalVariable(
    M, i32p, false, GlobalValue::ExternalLinkage,
    nullptr, "__coqui_reported_lens");
```

- [ ] **Step 2: After the existing `virgin_compare` call at line 182, but before the final `Builder.CreateBr(ExitBB)` at line 224, emit the report block**

```cpp
  /* Did this thread set a novelty bit? */
  BasicBlock *CheckReportBB = BasicBlock::Create(C, "check_report", Kernel);
  BasicBlock *ReportBB      = BasicBlock::Create(C, "report",       Kernel);
  BasicBlock *PostReportBB  = BasicBlock::Create(C, "post_report",  Kernel);

  Builder.CreateBr(CheckReportBB);
  Builder.SetInsertPoint(CheckReportBB);

  /* Novelty bit: 1 bit per thread, byte = tid/8, bit = tid%8. */
  Value *novWordIdx = Builder.CreateLShr(tid, ConstantInt::get(i32, 3), "nov_word");
  Value *novWordIdx64 = Builder.CreateZExt(novWordIdx, i64, "nov_word64");
  Value *novBytePtr = Builder.CreateGEP(i8, noveltyArg, {novWordIdx64}, "nov_byte_ptr");
  Value *novByte    = Builder.CreateLoad(i8, novBytePtr, "nov_byte");
  Value *novBit     = Builder.CreateTrunc(Builder.CreateAnd(tid,
                          ConstantInt::get(i32, 7)), i8, "nov_bit");
  Value *novMask    = Builder.CreateShl(ConstantInt::get(i8, 1), novBit, "nov_mask");
  Value *novSet     = Builder.CreateICmpNE(
                        Builder.CreateAnd(novByte, novMask),
                        ConstantInt::get(i8, 0), "nov_is_set");

  /* Crash? status[tid].signal or .asan_error or .ubsan_fatal non-zero. */
  Value *statusTid64_r = Builder.CreateZExt(tid, i64, "stat_tid64_r");
  Value *statusSlotR   = Builder.CreateGEP(i8, statusArg,
                            {Builder.CreateMul(statusTid64_r,
                                ConstantInt::get(i64, 16))}, "status_slot_r");
  Value *sigByte  = Builder.CreateLoad(i8,
                        Builder.CreateGEP(i8, statusSlotR, {ConstantInt::get(i64, 1)}),
                        "sig_byte");
  Value *asanByte = Builder.CreateLoad(i8,
                        Builder.CreateGEP(i8, statusSlotR, {ConstantInt::get(i64, 2)}),
                        "asan_byte");
  Value *ubsanByte= Builder.CreateLoad(i8,
                        Builder.CreateGEP(i8, statusSlotR, {ConstantInt::get(i64, 3)}),
                        "ubsan_byte");
  Value *crashAny = Builder.CreateOr(
                      Builder.CreateOr(sigByte, asanByte), ubsanByte, "crash_any");
  Value *crashSet = Builder.CreateICmpNE(crashAny,
                        ConstantInt::get(i8, 0), "crash_is_set");

  Value *shouldReport = Builder.CreateOr(novSet, crashSet, "should_report");
  Builder.CreateCondBr(shouldReport, ReportBB, PostReportBB);

  /* ReportBB: atomicAdd(&__coqui_reported_count, 1); if slot < CAP, write. */
  Builder.SetInsertPoint(ReportBB);
  Value *one32 = ConstantInt::get(i32, 1);
  Value *reportSlot = Builder.CreateAtomicRMW(
    AtomicRMWInst::Add, ReportedCountG, one32,
    MaybeAlign(4), AtomicOrdering::Monotonic);

  /* if (reportSlot < COQUI_REPORTED_CAP) ... */
  Value *cap = ConstantInt::get(i32, 512);
  Value *slotOk = Builder.CreateICmpULT(reportSlot, cap, "slot_ok");
  BasicBlock *DoWriteBB = BasicBlock::Create(C, "do_write", Kernel);
  Builder.CreateCondBr(slotOk, DoWriteBB, PostReportBB);

  Builder.SetInsertPoint(DoWriteBB);
  /* reported_tid[slot] = tid */
  Value *reportedTidBase  = Builder.CreateLoad(i32p, ReportedTidPtrG, "rep_tid_base");
  Value *tidSlot = Builder.CreateGEP(i32, reportedTidBase, {reportSlot}, "rep_tid_slot");
  Builder.CreateStore(tid, tidSlot);
  /* reported_lens[slot] = mut_len (havoc) or pm_len (premut).
   * lenPhi gives us the active thread's len as i64 — truncate to i32. */
  Value *reportedLensBase = Builder.CreateLoad(i32p, ReportedLensPtrG, "rep_lens_base");
  Value *lensSlot = Builder.CreateGEP(i32, reportedLensBase, {reportSlot}, "rep_lens_slot");
  Value *lenI32 = Builder.CreateTrunc(lenPhi, i32, "len_i32");
  Builder.CreateStore(lenI32, lensSlot);
  /* memcpy(reported_slab + slot * 4096, input_ptr, len) */
  Value *slot64   = Builder.CreateZExt(reportSlot, i64, "slot64");
  Value *slabOffset = Builder.CreateMul(slot64, ConstantInt::get(i64, 4096), "slab_off");
  Value *slabDstPtr = Builder.CreateGEP(i8, reportedSlabArg, {slabOffset}, "slab_dst");
  Builder.CreateCall(MemcpyFn, {slabDstPtr, inputPtrPhi, lenPhi,
                                  ConstantInt::getFalse(C)});
  Builder.CreateBr(PostReportBB);

  Builder.SetInsertPoint(PostReportBB);
  Builder.CreateBr(ExitBB);
```

**Note**: this inserts logic *between* `virgin_compare` and the existing `status[tid].crash_sig = sig` store at `coqui_mode/passes/FuzzEntry.cpp:206-219`. Be careful to preserve the ordering — classify+sig must complete before we read status.signal for the crash check. Since the sig store happens later at line 212-219, the `crash_any` check above only reads .signal/.asan/.ubsan which are set elsewhere (by ASan runtime or signal handlers), not .crash_sig. So the ordering works as drafted.

- [ ] **Step 3: Build pass + runtime + cubin**

```bash
cd /home/gpizarro/cuAFL/coqui_mode && bash build_coqui_support.sh
# Rebuild cjson_fuzzer cubin
```

Expected: green build.

- [ ] **Step 4: Smoke**

```bash
timeout 30 ./afl-fuzz -c 0 -i /home/gpizarro/cjson-in -o /tmp/cjson-phase2f \
    -- ./cjson_fuzzer.sm_75.cubin 2>&1 | tail -20
```

Expected: normal operation. Since still no caller of flag=1, reported_count stays 0 for every batch; the compact-report blocks execute but always return empty.

- [ ] **Step 5: Commit**

```bash
git add coqui_mode/passes/FuzzEntry.cpp
git commit -m "cuAFL: emit compact-report block in kernel for novel/crash threads"
```

### Task 2.9: Process reported_slab in `coqui_await_and_process`

**Files:**
- Modify: `src/afl-fuzz-coqui.c`

The novelty-processing loop at lines 665-678 currently reads mutated bytes from `h_input_bytes + h_offsets[i]`. For flag=1 slots those fields are unused — real bytes live in `h_reported_slab`. We need a tid→slot lookup.

- [ ] **Step 1: Rewrite the novelty loop**

Replace lines 665-678:

```c
  /* Build tid → reported-slot map for this batch.
   * reported_count is the authoritative count after DtoH completed. */
  u32 rcount = b->h_reported_count;
  if (rcount > COQUI_REPORTED_CAP) {
    WARNF("coqui reported_count %u > CAP %u — some novel slots will be "
          "unverified this batch", rcount, COQUI_REPORTED_CAP);
    rcount = COQUI_REPORTED_CAP;
  }
  /* Use a small hash to map tid→slot. 1024-bucket linear probe over
   * up to 512 entries keeps load ≤ 50% — fast in practice. */
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

  /* Process flagged inputs (novelty bits set) */
  for (u32 word_i = 0; word_i < ctx->batch_size / 32; word_i++) {
    u32 bits = ((u32 *)b->h_novelty)[word_i];
    while (bits) {
      u32 bit_pos = __builtin_ctz(bits);
      bits &= bits - 1;
      u32 i = word_i * 32 + bit_pos;

      /* Figure out where the mutated input lives: reported_slab (flag=1)
       * or h_input_bytes (flag=0). */
      u32 slot_info = b->h_slot_info[i];
      u8  flag      = (u8)(slot_info >> 24);
      u8  *input;
      u32 len;
      if (flag == COQUI_FLAG_HAVOC) {
        /* Look up reported slot for tid=i */
        u32 h = (i * 0x9E3779B1u) & 1023u;
        u32 slot = (u32)-1;
        for (u32 p = 0; p < 1024; ++p) {
          u32 k = (h + p) & 1023u;
          if (!tid2slot_used[k]) break;
          if (tid2slot_keys[k] == i) { slot = tid2slot_vals[k]; break; }
        }
        if (slot == (u32)-1) continue;   /* overflowed slab; skip verify */
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
```

- [ ] **Step 2: Rewrite the crash-processing loop at lines 705-745 analogously**

The outer structure (dedup hash, iterate all slots) stays. Replace the read of `input` and `len` with the same flag-aware dispatch:

```c
  /* ... existing loop that walks i=0..batch_size and checks crash_sig ... */

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

    ctx->crash_verify_calls++;
    process_input_via_cpu_fsrv(afl, input, len);
```

- [ ] **Step 3: Build + smoke**

```bash
cd /home/gpizarro/cuAFL && make -j"$(nproc)"
timeout 30 ./afl-fuzz -c 0 -i /home/gpizarro/cjson-in -o /tmp/cjson-phase2g \
    -- ./cjson_fuzzer.sm_75.cubin 2>&1 | tail -20
```

Expected: cjson runs. All slots are flag=0 today, so the if-branches always take the else path — behaviour unchanged.

- [ ] **Step 4: Commit — end of Phase 2**

```bash
git add src/afl-fuzz-coqui.c
git commit -m "cuAFL: process reported_slab for flag=1 novelty/crash verify paths"
```

---

## Phase 3 — Flip the havoc stage

This is the load-bearing commit: `fuzz_one_original`'s havoc_stage calls `coqui_refresh_seed_pool` once and replaces its mutation switch with a thin `coqui_submit_havoc_slot` loop.

### Task 3.1: Add runtime branch to havoc_stage

**Files:**
- Modify: `src/afl-fuzz-one.c`

- [ ] **Step 1: Locate the havoc_stage loop**

`src/afl-fuzz-one.c:2259` begins `for (afl->stage_cur = 0; afl->stage_cur < afl->stage_max; ++afl->stage_cur) {`. This for-loop's body runs through line 3543 (`}`). The end of the body includes a `common_fuzz_stuff()` call after the mutation switch. We wrap the *entire* for-loop.

- [ ] **Step 2: Refresh the pool at top of havoc_stage**

Insert just after the `stack_max = 1 << ...` line at line 2255 (but before the `for` at 2259):

```c
  /* GPU-havoc: refresh the resident seed pool with queue_cur + weighted sample. */
  if (afl->coqui) {
    coqui_refresh_seed_pool(afl);
  }
```

- [ ] **Step 3: Add the runtime branch**

Wrap the existing `for` loop:

```c
  if (afl->coqui) {
    /* GPU-havoc: thin loop — GPU mutates + runs + classifies per-thread. */
    for (afl->stage_cur = 0; afl->stage_cur < afl->stage_max; ++afl->stage_cur) {
      if (unlikely(afl->stop_soon)) goto abandon_entry;
      coqui_submit_havoc_slot(afl, /*seed_idx=*/0, COQUI_FLAG_HAVOC);
    }
    coqui_flush_batch(afl);
  } else {
    /* Original CPU havoc body — preserved verbatim below. */
    for (afl->stage_cur = 0; afl->stage_cur < afl->stage_max; ++afl->stage_cur) {
      /* ... existing 1500-line body ... unchanged ... */
    }
  }
```

Practically: insert the new `if (afl->coqui) { ... } else {` right before the existing `for (afl->stage_cur = 0; ...)` at line 2259, and close the `else` block with a matching `}` right after the original for-loop's closing `}` (at ~line 3543, which is followed by `afl->stage_finds[STAGE_HAVOC] += ...` at 3544 — keep that unchanged).

- [ ] **Step 4: Custom-mutator fallback warning**

Just before the `if (afl->coqui)` branch, add:

```c
  if (afl->coqui && afl->custom_mutators_count > 0) {
    int any_stacked = 0;
    LIST_FOREACH(&afl->custom_mutator_list, struct custom_mutator, {
      if (el->stacked_custom) { any_stacked = 1; break; }
    });
    if (any_stacked) {
      static u32 warned_tag = 0;
      if (warned_tag != afl->queue_cur->id) {
        WARNF("coqui_mode: custom stacked mutator on queue entry %u — "
              "GPU havoc doesn't support custom mutators; this entry "
              "will use CPU havoc.", afl->queue_cur->id);
        warned_tag = afl->queue_cur->id;
      }
      goto cpu_havoc_fallback;
    }
  }

  if (afl->coqui) {
    /* GPU-havoc path (as in Step 3) */
    coqui_refresh_seed_pool(afl);
    for (afl->stage_cur = 0; afl->stage_cur < afl->stage_max; ++afl->stage_cur) {
      if (unlikely(afl->stop_soon)) goto abandon_entry;
      coqui_submit_havoc_slot(afl, 0, COQUI_FLAG_HAVOC);
    }
    coqui_flush_batch(afl);
  } else {
  cpu_havoc_fallback:;
    /* Original CPU havoc body, unchanged. */
    for (afl->stage_cur = 0; afl->stage_cur < afl->stage_max; ++afl->stage_cur) {
      ...
    }
  }
```

- [ ] **Step 5: Build**

```bash
cd /home/gpizarro/cuAFL && make -j"$(nproc)"
```

Expected: green build.

- [ ] **Step 6: First real smoke (this is the acceptance gate)**

```bash
rm -rf /tmp/cjson-phase3
timeout 120 ./afl-fuzz -c 0 -i /home/gpizarro/cjson-in -o /tmp/cjson-phase3 \
    -- ./cjson_fuzzer.sm_75.cubin 2>&1 | tee /tmp/cjson-phase3.log | tail -40
```

Expected:
- fuzzer starts without WARNF/FATAL
- `[coqui-rate]` lines print regularly (normal)
- `execs_per_sec` > 0
- novel-edge discovery happens (saved_crashes or edges_found grows)
- no `coqui reported_count > CAP` warnings (or very rare)
- stderr has no new FATALs

If fuzzer dies immediately with a CUDA error: most likely cause is the new cubin's args don't match the host launch args. Recheck Task 2.6 argument ordering.

If fuzzer runs but `execs_per_sec` is near zero: likely the `coqui_submit_havoc_slot` path has a tight loop bug. Check ping-pong flip logic.

- [ ] **Step 7: Commit**

```bash
git add src/afl-fuzz-one.c
git commit -m "cuAFL: flip havoc_stage to GPU-havoc under coqui_mode"
```

---

## Phase 4 — Bench and report

### Task 4.1: Write the bench runner

**Files:**
- Create: `benchmark/gpu_havoc_vs_cpu.sh`

- [ ] **Step 1: Write the script**

```bash
#!/usr/bin/env bash
# benchmark/gpu_havoc_vs_cpu.sh
# Interleaved CPU-havoc vs GPU-havoc bench for cjson_fuzzer.
# Invoked after Phase 3 commit; output is CSV on stdout.
set -euo pipefail

REPO="${REPO:-/home/gpizarro/cuAFL}"
CJSON_IN="${CJSON_IN:-/home/gpizarro/cjson-in}"
DUR="${DUR:-120}"
N="${N:-20}"
OUT_ROOT="${OUT_ROOT:-/tmp/gpu-havoc-bench}"
mkdir -p "$OUT_ROOT"

echo "run,mode,execs_per_sec,edges_found,saved_crashes"
for ((i = 0; i < N; ++i)); do
  for mode in cpu gpu; do
    OUT="$OUT_ROOT/run-$i-$mode"
    rm -rf "$OUT"
    # Toggle the mode via env — implement AFL_COQUI_DISABLE_GPU_HAVOC=1 as a
    # debug-only hatch in afl-fuzz-one.c (not in prod rollout) OR run two
    # binaries built with/without coqui_mode. For now: use coqui vs no-coqui
    # afl-fuzz invocation.
    case "$mode" in
      cpu) CMD=("$REPO/afl-fuzz" "-i" "$CJSON_IN" "-o" "$OUT" "--" "$REPO/cjson_cpu") ;;
      gpu) CMD=("$REPO/afl-fuzz" "-c" "0" "-i" "$CJSON_IN" "-o" "$OUT"
                "--" "$REPO/cjson_fuzzer.sm_75.cubin") ;;
    esac
    timeout "$DUR" "${CMD[@]}" > "$OUT/stdout.log" 2>&1 || true

    # Extract metrics from fuzzer_stats file
    if [[ -f "$OUT/default/fuzzer_stats" ]]; then
      stats="$OUT/default/fuzzer_stats"
      eps=$(awk -F': *' '/execs_per_sec/ {print $2; exit}' "$stats")
      edg=$(awk -F': *' '/edges_found/   {print $2; exit}' "$stats")
      crs=$(awk -F': *' '/saved_crashes/ {print $2; exit}' "$stats")
    else
      eps=0; edg=0; crs=0
    fi
    echo "$i,$mode,$eps,$edg,$crs"
  done
done
```

Make executable:

```bash
chmod +x benchmark/gpu_havoc_vs_cpu.sh
```

- [ ] **Step 2: Run the bench**

```bash
cd /home/gpizarro/cuAFL
benchmark/gpu_havoc_vs_cpu.sh > /tmp/gpu-havoc-bench.csv
```

Expected: 40 rows (20 cpu + 20 gpu). Takes `N × DUR × 2` seconds ≈ 80 min.

- [ ] **Step 3: Summarize CSV for maintainer review**

```bash
python3 -c "
import csv
from statistics import mean, stdev
with open('/tmp/gpu-havoc-bench.csv') as f:
    rows = list(csv.DictReader(f))
for metric in ['execs_per_sec', 'edges_found', 'saved_crashes']:
    for mode in ['cpu', 'gpu']:
        vals = [float(r[metric]) for r in rows if r['mode'] == mode]
        print(f'{mode:3s} {metric:16s}: mean={mean(vals):.1f} stdev={stdev(vals):.1f}')
    cpu = [float(r[metric]) for r in rows if r['mode'] == 'cpu']
    gpu = [float(r[metric]) for r in rows if r['mode'] == 'gpu']
    pct = (mean(gpu) / mean(cpu) - 1) * 100 if mean(cpu) else 0
    print(f'    {metric:16s}: GPU Δ = {pct:+.1f}%')
    print()
"
```

- [ ] **Step 4: Report to maintainer**

Collect:
- CSV file
- Summary table
- Any WARNF/FATAL lines from `/tmp/gpu-havoc-bench/*/stdout.log`

Maintainer decides **accept** (leave commits; merge branch) or **revert** (`git revert <Phase 3 commit>` — Phases 1, 2 stay as inert dead code).

- [ ] **Step 5: Commit bench script**

```bash
git add benchmark/gpu_havoc_vs_cpu.sh
git commit -m "cuAFL: add gpu_havoc_vs_cpu bench runner"
```

---

## Self-review

**1. Spec coverage check (spec § → task):**
- §1 goals/non-goals → ensured by design of Phase 3 (full replace under coqui_mode, CPU path preserved for non-coqui builds); custom mutators fallback in Task 3.1; MOpt untouched ✓
- §2 architecture → Tasks 2.1-2.9 implement the host + kernel flow; Task 1.x implements GPU mutation ✓
- §3.1 persistent resources → Task 2.1 (structs), Task 2.3 (init/bind), Task 2.4 (per-refresh upload) ✓
- §3.2 per-batch kernel args → Task 2.6 (kernel signature extension) ✓
- §3.3 returning mutated bytes → Task 2.8 (compact-report kernel block), Task 2.9 (host consumer) ✓
- §4.1 FuzzEntry.cpp changes → Task 2.6, 2.7, 2.8 ✓
- §4.2 coqui_mutate.c runtime → Tasks 1.3-1.8 ✓
- §4.3 PRNG → Task 1.3 ✓
- §4.4 weighted splice pick → Task 1.4 ✓
- §5.1 fuzz_one_original diff → Task 3.1 ✓
- §5.2 coqui_submit_havoc_slot → Task 2.5 ✓
- §5.3 coqui_refresh_seed_pool → Task 2.4 ✓
- §5.4 extras / a_extras / mutation_array sync → Task 2.3 step 6 (extras), Task 2.4 step 2 (mut_array, a_extras placeholder with TODO) ✓
- §5.5 build system → Task 1.3 step 2 ✓
- §6 memory + stack → Task 2.3 step 2 (MAX_INPUT_SIZE guard); actual sizes verified by design math ✓
- §7.1 bench methodology → Task 4.1 ✓
- §7.2 rollout sequencing → 4 phases correspond 1:1 ✓

**2. Placeholder scan:** One deliberate placeholder remains: Task 2.4 Step 2 has a `TODO(phase3-bench)` for the a_extras re-upload on cmplog growth. This is intentional and called out in the spec §5.4 as "opportunistic" — if bench shows cmplog-heavy targets suffer from stale a_extras, we implement it. Safe because a_extras being stale only degrades EXTRA op quality, doesn't crash. Other than that, no `TBD` / `fill in later` / `handle edge cases` patterns.

**3. Type consistency check:** `__coqui_havoc_mutate` signature matches between Task 1.1 header, Task 1.5 wrapper, and Task 2.7 call site (`buf, len, max_len, self_idx, prng`). `coqui_submit_havoc_slot` signature matches between Task 2.1 declaration and Task 2.5 implementation. Kernel arg count of 7 consistent across Task 2.5 host `args[]`, Task 2.6 IR FunctionType, and spec §3.2.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-04-21-gpu-havoc-mutations.md`. Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints for review.

Which approach?
