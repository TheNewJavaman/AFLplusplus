# coqui mode GPU Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an AFL++ executor mode that runs inputs on a GPU in batches, selected via `--coqui <sync_id>`. This plan covers the AFL++ integration layer only — the GPU backend internals (LLVM passes, CUDA runtime) are explicitly deferred and shipped as a hollow stub that makes coqui mode compile and run without crashing.

**Architecture:** A new boolean `coqui_mode` joins the existing mode flags on `afl_forkserver_t` (alongside `qemu_mode`, `frida_mode`, `nyx_mode`, `cs_mode`). The seam is `common_fuzz_stuff()` in `src/afl-fuzz-run.c` — under `--coqui`, a branch routes inputs into a packed batch sink rather than invoking a forkserver. `fuzz_run_target()` is also short-circuited (returns trivial coverage) so calibrate/trim/sync paths succeed in gpu_mode without a real executor. Det and cmplog stages in `fuzz_one()` are gated off. A new module `src/afl-fuzz-coqui.{c,h}` defines the contract and holds the hollow stub; the real GPU backend will drop in behind the same contract in a later phase.

**Tech Stack:** C99, AFL++ build system (GNU make, no autotools). `src/afl-fuzz*.c` files are picked up automatically via `AFL_FUZZ_FILES = $(wildcard src/afl-fuzz*.c)` — no GNUmakefile changes needed. No CUDA dependency in this plan (stub is pure C).

**Spec:** `docs/superpowers/specs/2026-04-18-coqui-gpu-backend-design.md`

**User preferences:**
- No tests (user directive: "let's skip tests"). Verification is manual smoke-testing and `does it compile`.
- Atomic commits: one logical change per commit.
- Code doesn't need to work fully — stub is intentionally hollow, real GPU backend deferred to a follow-up brainstorm.

---

## File Structure

**New files:**
- `include/afl-fuzz-coqui.h` — contract types and API declarations.
- `src/afl-fuzz-coqui.c` — hollow stub implementing the contract.

**Modified files:**
- `include/forkserver.h` — add `coqui_mode` boolean.
- `include/afl-fuzz.h` — add `gpu_mode`, `coqui`, `gpu_batch_size` fields.
- `src/afl-fuzz.c` — convert `getopt` to `getopt_long`, add `--coqui` handler, initialize coqui context near `afl_fsrv_init`.
- `src/afl-fuzz-run.c` — branch in `common_fuzz_stuff` (batch sink) and `fuzz_run_target` (trivial-coverage short-circuit).
- `src/afl-forkserver.c` — early return in `afl_fsrv_start` when `coqui_mode` is set.
- `src/afl-fuzz-one.c` — gate det stages and cmplog stage on `afl->gpu_mode`.
- `Changelog.md` — one-line note for the coqui mode fork.

**Not modified:** `GNUmakefile` (wildcard picks up `afl-fuzz-coqui.c`), `Makefile`, sync code, queue code, bitmap code, TUI, stats, power schedules, or any mutation logic.

---

## Phase 1: Plumbing (CLI flag + state fields)

Adds the `--coqui <sync_id>` flag and the state fields it needs. No behavior yet — the flag parses, sets state, and that's it.

### Task 1.1: Add `coqui_mode` boolean to forkserver state

**Files:**
- Modify: `include/forkserver.h:159-163` (add field next to other mode flags)

- [ ] **Step 1: Add the field**

Open `include/forkserver.h`. Find the block of mode booleans near line 155-163. Add `coqui_mode` as a new field:

```c
bool qemu_mode;                       /* if running in qemu mode or not   */

bool use_fauxsrv;                       /* Fauxsrv for non-forking targets? */

bool frida_mode;                     /* if running in frida mode or not   */

bool frida_asan;                     /* if running with asan in frida mode */

bool cs_mode;                      /* if running in CoreSight mode or not */

bool coqui_mode;                      /* if running in coqui (GPU) mode   */
```

- [ ] **Step 2: Verify compilation**

Run: `make -j$(nproc) afl-fuzz 2>&1 | tail -20`
Expected: builds cleanly, no warnings about the new field.

- [ ] **Step 3: Commit**

```bash
git add include/forkserver.h
git commit -m "coqui mode: add coqui_mode field to afl_forkserver_t

Mirrors the existing mode boolean pattern (qemu_mode, frida_mode,
nyx_mode, cs_mode). No behavior attached yet; wiring lands in later
commits.
"
```

---

### Task 1.2: Add coqui state fields to `afl_state_t`

**Files:**
- Modify: `include/afl-fuzz.h` (add forward declaration + state fields)

- [ ] **Step 1: Add forward declaration near the top of the header**

Near the top of `include/afl-fuzz.h` (after the existing includes, before `typedef struct afl_state` or similar), add:

```c
/* Forward declaration — full type in afl-fuzz-coqui.h. */
struct coqui_ctx;
```

- [ ] **Step 2: Add state fields inside the afl_state_t struct**

Find the `afl_state_t` struct definition in `include/afl-fuzz.h`. Near the other boolean flags (e.g., `non_instrumented_mode`, `skip_deterministic`), add:

```c
u8               gpu_mode;         /* 1 if --coqui was given */
struct coqui_ctx *coqui;           /* opaque coqui_mode context, NULL otherwise */
u32              gpu_batch_size;   /* default 8192, tunable */
char            *coqui_cubin_path; /* path to cubin from CLI trailing arg */
```

- [ ] **Step 3: Verify compilation**

Run: `make -j$(nproc) afl-fuzz 2>&1 | tail -20`
Expected: builds cleanly. (Struct forward declaration is fine because we only hold a pointer.)

- [ ] **Step 4: Commit**

```bash
git add include/afl-fuzz.h
git commit -m "coqui mode: add gpu_mode + coqui context pointer to afl_state_t

Forward-declares struct coqui_ctx so afl-fuzz.h stays free of CUDA
dependencies. The full type ships in afl-fuzz-coqui.h in a later commit.
"
```

---

### Task 1.3: Convert `getopt` to `getopt_long` and add `--coqui` handler

AFL++ uses plain `getopt()` at `src/afl-fuzz.c:719`. We convert to `getopt_long()` — a drop-in superset that accepts the same short-option string plus a long-options table. All existing short flags continue to work identically.

**Files:**
- Modify: `src/afl-fuzz.c` (replace getopt with getopt_long, add long_options table, add case handler)

- [ ] **Step 1: Add `<getopt.h>` include if not already present**

Check the top of `src/afl-fuzz.c` for `#include <getopt.h>`. If absent, add it alongside the other system headers. (On most systems `getopt.h` is already pulled in transitively, but adding it is safe.)

- [ ] **Step 2: Add long-options table and enum sentinel**

Just above the `while ((opt = getopt(...))` block at line 719, add:

```c
enum {
  LONGOPT_COQUI = 256,   /* > CHAR_MAX so it doesn't collide with short opts */
};

static struct option afl_fuzz_long_options[] = {
  {"coqui", required_argument, NULL, LONGOPT_COQUI},
  {NULL,    0,                 NULL, 0}
};
```

- [ ] **Step 3: Replace `getopt` with `getopt_long`**

Change line 719-722 from:

```c
  while ((opt = getopt(
              argc, argv,
              "+a:Ab:B:c:CdDe:E:f:F:g:G:hi:I:K:l:L:m:M:nNo:Op:P:QRs:S:t:T:"
              "uUV:w:WXx:YzZ")) > 0) {
```

to:

```c
  while ((opt = getopt_long(
              argc, argv,
              "+a:Ab:B:c:CdDe:E:f:F:g:G:hi:I:K:l:L:m:M:nNo:Op:P:QRs:S:t:T:"
              "uUV:w:WXx:YzZ",
              afl_fuzz_long_options, NULL)) > 0) {
```

- [ ] **Step 4: Add the `LONGOPT_COQUI` case in the switch**

Add a new case in the switch statement, placed alphabetically-reasonably near the `case 'S':` block at line 1009. The implementation mirrors `-S` and adds coqui-specific setup:

```c
      case LONGOPT_COQUI:                               /* GPU (coqui) sync id */

        if (afl->non_instrumented_mode) {
          FATAL("--coqui is not supported in non-instrumented mode");
        }

        if (afl->fsrv.cs_mode) {
          FATAL("--coqui is not supported in ARM CoreSight mode");
        }

        if (afl->fsrv.qemu_mode || afl->fsrv.frida_mode || afl->fsrv.nyx_mode) {
          FATAL("--coqui is mutually exclusive with -Q/-O/-U/-X");
        }

        if (afl->sync_id) {
          FATAL("Multiple -S/-M/--coqui options not supported");
        }

        if (optarg && *optarg == '-') {
          FATAL(
              "argument for --coqui started with a dash '-', which is "
              "used for options");
        }

        afl->sync_id            = ck_strdup(optarg);
        afl->is_secondary_node  = 1;
        afl->gpu_mode           = 1;
        afl->fsrv.coqui_mode    = 1;
        afl->skip_deterministic = 1;  /* det stages gated off under --coqui */

        break;
```

- [ ] **Step 5: Add `--coqui` to the `--help` output**

Find the `usage()` function in `src/afl-fuzz.c` (search for `Usage:` or `-S id`). Near the `-S id` help line, add:

```c
      "  --coqui id    - GPU secondary mode (coqui_mode), sync id analogous to -S.\n"
      "                  Requires a cubin target produced by coqui-cc.\n"
```

- [ ] **Step 6: Verify compilation and that `--coqui bogus` is parsed**

Run: `make -j$(nproc) afl-fuzz 2>&1 | tail -20`
Expected: builds cleanly.

Run: `./afl-fuzz --coqui 2>&1 | head -5`
Expected: complains about missing argument (because `required_argument` is set), not "unknown option".

Run: `./afl-fuzz --coqui gpu0 -i /tmp -o /tmp 2>&1 | head -10`
Expected: fails for another reason (missing target, missing seeds dir) but NOT for unknown flag. The flag was accepted.

- [ ] **Step 7: Commit**

```bash
git add src/afl-fuzz.c
git commit -m "coqui mode: add --coqui <sync_id> long-only flag

Converts getopt() to getopt_long() (drop-in superset; all existing
short flags unchanged). Adds --coqui handler modeled on -S:
sets sync_id, gpu_mode, fsrv.coqui_mode, skip_deterministic.

-G (max input length) is untouched — coqui mode reads afl->max_length
directly for batch buffer sizing.
"
```

---

## Phase 2: Contract + Stub

Defines the coqui_mode API in a header and provides a hollow implementation that lets coqui mode run end-to-end without crashing.

### Task 2.1: Write `include/afl-fuzz-coqui.h`

**Files:**
- Create: `include/afl-fuzz-coqui.h`

- [ ] **Step 1: Create the header with full type definitions and API**

Create `include/afl-fuzz-coqui.h` with the following content:

```c
/*
 * afl-fuzz-coqui.h --- coqui mode coqui_mode contract.
 *
 * Defines the interface between core afl-fuzz and the GPU (coqui) backend.
 * A hollow stub implementation in afl-fuzz-coqui.c makes coqui mode compile and
 * run without a real GPU; the real backend will drop in behind this contract.
 *
 * Spec: docs/superpowers/specs/2026-04-18-coqui-gpu-backend-design.md
 */

#ifndef _HAVE_AFL_FUZZ_COQUI_H
#define _HAVE_AFL_FUZZ_COQUI_H

#include "types.h"
#include "forkserver.h"

struct afl_state;  /* forward */

/* Tunable defaults. Override with env vars at runtime. */
#define COQUI_DEFAULT_BATCH_SIZE 8192
#define COQUI_MAX_INPUT_DEFAULT  4096

/* ------------------------------------------------------------------------
 * Data types
 * ------------------------------------------------------------------------*/

/* Per-input status returned by the GPU kernel (or stub). */
typedef struct coqui_status {
  u8  asan_error;     /* non-zero if ASan tripped */
  u8  ubsan_fatal;    /* non-zero if non-recoverable UBSan fired */
  u8  signal;         /* non-zero = signal number that killed this thread */
  u8  timeout_flag;   /* non-zero if this input exceeded per-thread budget */
  u32 _reserved;      /* padding / future use */
} coqui_status_t;

/* One ping-pong half: packed input bytes + metadata + device mirrors. */
typedef struct coqui_batch {
  /* Host-side pinned buffers (malloc'd in stub; cuMemAllocHost in real GPU). */
  u8             *h_input_bytes;  /* packed inputs, each start 8-aligned */
  u32            *h_offsets;      /* per-slot offset into h_input_bytes */
  u32            *h_input_lens;   /* per-slot length */
  u8             *h_novelty;      /* 1 bit per slot */
  coqui_status_t *h_status;

  /* Device mirrors (NULL in stub; non-NULL in real GPU). */
  void *d_input_bytes;
  void *d_offsets;
  void *d_input_lens;
  void *d_coverage;
  void *d_novelty;
  void *d_virgin;
  void *d_status;

  /* Fill state. */
  u32 n_inputs;    /* slots used so far */
  u32 bytes_used;  /* bytes consumed in h_input_bytes (unaligned cursor) */
} coqui_batch_t;

/* Per-afl-state coqui context. */
typedef struct coqui_ctx {
  coqui_batch_t  ping;
  coqui_batch_t  pong;
  coqui_batch_t *pending;    /* filling now (CPU side) */
  coqui_batch_t *executing;  /* on GPU (or most recent done) */

  u32 batch_size;      /* snapshot of afl->gpu_batch_size */
  u32 max_input_size;  /* snapshot of afl->max_length (with fallback) */
  u32 byte_budget;     /* size of each h_input_bytes buffer */
  u32 map_size;        /* snapshot of afl->fsrv.map_size */

  u64 oversized_count; /* inputs skipped because they alone exceed byte_budget */
  u64 launch_count;    /* batches launched so far */
} coqui_ctx_t;

/* ------------------------------------------------------------------------
 * API surface (called from core afl-fuzz)
 * ------------------------------------------------------------------------*/

/* Initialize GPU context, allocate ping-pong buffers, load cubin.
   `cubin_path` is the trailing positional argument from the CLI (afl->argv[0]
   once the forkserver argv is parsed).
   Fails fast (FATAL) on any error.
   Stub: allocates host buffers only; cubin_path is stashed but not loaded. */
void coqui_init(struct afl_state *afl, const char *cubin_path);

/* Append one mutated input to the pending batch.
   Returns 0 unconditionally — never signals stage bail-out.
   May trigger an async launch if the pending batch fills.
   Stub: packs but never launches (discards accumulated inputs). */
u8 coqui_submit_input(struct afl_state *afl, u8 *buf, u32 len);

/* Force-launch any partial batch and wait for all pending work to drain.
   Called at stage boundaries so afl->queued_items reflects every mutation
   the stage generated before the stage decides whether it was productive.
   Stub: resets pending batch state, calls nothing. */
void coqui_flush_batch(struct afl_state *afl);

/* Single-input synchronous path (seed calibration, sync-in calibration).
   Populates afl->fsrv.trace_bits and returns an fsrv_run_result_t value
   (u8-valued: FSRV_RUN_OK / CRASH / TMOUT / ERROR / NOINST).
   Stub: writes trivial coverage (trace_bits[0] = 1) and returns FSRV_RUN_OK. */
u8 coqui_calibrate_one(struct afl_state *afl, u8 *buf, u32 len);

/* Free resources and tear down CUDA context on normal exit or SIGINT.
   Stub: free()s host buffers; no CUDA calls. */
void coqui_shutdown(struct afl_state *afl);

#endif /* _HAVE_AFL_FUZZ_COQUI_H */
```

- [ ] **Step 2: Verify compilation**

The header isn't included anywhere yet, so `make` should still pass. Run to confirm:

Run: `make -j$(nproc) afl-fuzz 2>&1 | tail -5`
Expected: builds cleanly.

- [ ] **Step 3: Commit**

```bash
git add include/afl-fuzz-coqui.h
git commit -m "coqui mode: define coqui_mode contract in afl-fuzz-coqui.h

Types (coqui_status_t, coqui_batch_t, coqui_ctx_t) and 5-function API
(coqui_init, coqui_submit_input, coqui_flush_batch, coqui_calibrate_one,
coqui_shutdown). Matches the spec at
docs/superpowers/specs/2026-04-18-coqui-gpu-backend-design.md.

Forward-declares struct afl_state so the header stays free of broader
afl-fuzz.h dependencies.
"
```

---

### Task 2.2: Write hollow stub `src/afl-fuzz-coqui.c`

**Files:**
- Create: `src/afl-fuzz-coqui.c`

- [ ] **Step 1: Create the stub file**

Create `src/afl-fuzz-coqui.c` with a hollow implementation that makes coqui mode compile and run but does no real execution:

```c
/*
 * afl-fuzz-coqui.c --- coqui mode coqui_mode stub implementation.
 *
 * Hollow stub behind the coqui_mode contract. Lets coqui mode compile and run
 * end-to-end under --coqui without a real GPU backend. Real CUDA-backed
 * implementation drops in behind the same contract in a later phase.
 *
 * Spec: docs/superpowers/specs/2026-04-18-coqui-gpu-backend-design.md
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
```

- [ ] **Step 2: Verify compilation**

The GNUmakefile's `AFL_FUZZ_FILES = $(wildcard src/afl-fuzz*.c)` picks up the new file automatically.

Run: `make -j$(nproc) afl-fuzz 2>&1 | tail -20`
Expected: builds cleanly. The new `.c` file is compiled but no one calls its functions yet.

- [ ] **Step 3: Commit**

```bash
git add src/afl-fuzz-coqui.c
git commit -m "coqui mode: hollow stub for coqui_mode

Implements the 5-function contract from afl-fuzz-coqui.h. Allocates
ping-pong host buffers, packs inputs with 8-byte alignment, discards
accumulated inputs on 'launch'. coqui_calibrate_one writes trivial
coverage so AFL++ calibration doesn't flag seeds as non-instrumented.

Picked up automatically by GNUmakefile's afl-fuzz*.c wildcard; no
build-system changes needed.
"
```

---

## Phase 3: Core wiring

Connects `--coqui` to actual behavior: forkserver short-circuit, fuzz_run_target short-circuit, common_fuzz_stuff batch branch, stage gating, and initialization hookup.

### Task 3.1: `afl_fsrv_start` short-circuits for coqui_mode

**Files:**
- Modify: `src/afl-forkserver.c` (add early return at the top of `afl_fsrv_start`)

- [ ] **Step 1: Locate `afl_fsrv_start`**

In `src/afl-forkserver.c`, find the function definition of `afl_fsrv_start(afl_forkserver_t *fsrv, ...)`. The nyx_mode short-circuit at line 160 is the pattern to follow.

- [ ] **Step 2: Add the coqui_mode early return near the top of `afl_fsrv_start`**

Immediately after the opening brace of `afl_fsrv_start` and any initial checks, add:

```c
  if (fsrv->coqui_mode) {

    /* coqui_mode owns execution via the afl-fuzz-coqui.c path.
       The forkserver is never launched in this mode; coqui_init() was
       called earlier from afl-fuzz.c main(). */
    return;

  }
```

Pick a spot consistent with the existing nyx_mode check — immediately before the fork/exec pipe setup.

- [ ] **Step 3: Verify compilation**

Run: `make -j$(nproc) afl-fuzz 2>&1 | tail -10`
Expected: builds cleanly.

- [ ] **Step 4: Commit**

```bash
git add src/afl-forkserver.c
git commit -m "coqui mode: short-circuit afl_fsrv_start in coqui_mode

Returns early — no fork+exec, no pipes. coqui_init() in the new
afl-fuzz-coqui.c path owns execution for --coqui instances. Pattern
mirrors the existing nyx_mode check.
"
```

---

### Task 3.2: `fuzz_run_target` short-circuits with trivial coverage in gpu_mode

Calibration, trim, and sync-in paths all route through `fuzz_run_target`. A single short-circuit at that function's entry makes them all behave consistently under `--coqui` — trace_bits gets trivial coverage, return FSRV_RUN_OK.

**Files:**
- Modify: `src/afl-fuzz-run.c:51` (add gpu_mode branch at the top of `fuzz_run_target`)

- [ ] **Step 1: Add the gpu_mode branch**

In `src/afl-fuzz-run.c`, find `fsrv_run_result_t fuzz_run_target(...)` at line 51. Just after the opening brace, add:

```c
  if (unlikely(afl->gpu_mode)) {

    /* coqui_mode: no real executor at this layer. Populate trace_bits with
       trivial coverage (edge 0 hit) so calibrate/trim/sync paths don't
       flag the input as FSRV_RUN_NOINST. common_fuzz_stuff() handles the
       actual batched execution via coqui_submit_input(). */
    memset(afl->fsrv.trace_bits, 0, afl->fsrv.map_size);
    afl->fsrv.trace_bits[0] = 1;
    return FSRV_RUN_OK;

  }
```

- [ ] **Step 2: Verify compilation**

Run: `make -j$(nproc) afl-fuzz 2>&1 | tail -10`
Expected: builds cleanly.

- [ ] **Step 3: Commit**

```bash
git add src/afl-fuzz-run.c
git commit -m "coqui mode: short-circuit fuzz_run_target in gpu_mode

Calibrate, trim, and sync-in paths all go through fuzz_run_target.
A single gpu_mode branch at the top populates trace_bits with trivial
coverage and returns OK, so those paths don't fight the real batched
backend in common_fuzz_stuff.

Real GPU backend will replace this short-circuit with a single-input
GPU launch (coqui_calibrate_one) in a later phase.
"
```

---

### Task 3.3: `common_fuzz_stuff` routes to coqui batch sink in gpu_mode

**Files:**
- Modify: `src/afl-fuzz-run.c:1419` (add gpu_mode branch at the top of `common_fuzz_stuff`)

- [ ] **Step 1: Include the coqui header**

At the top of `src/afl-fuzz-run.c`, add:

```c
#include "afl-fuzz-coqui.h"
```

- [ ] **Step 2: Add the gpu_mode branch**

Inside `common_fuzz_stuff` at `src/afl-fuzz-run.c:1419`, just after the opening brace and before the `u8 fault;` declaration, add:

```c
  if (unlikely(afl->gpu_mode)) {

    return coqui_submit_input(afl, out_buf, len);

  }
```

- [ ] **Step 3: Verify compilation**

Run: `make -j$(nproc) afl-fuzz 2>&1 | tail -10`
Expected: builds cleanly.

- [ ] **Step 4: Commit**

```bash
git add src/afl-fuzz-run.c
git commit -m "coqui mode: common_fuzz_stuff routes to coqui batch sink in gpu_mode

Load-bearing change: every havoc/splice mutation generated by fuzz_one
calls common_fuzz_stuff, which now routes through coqui_submit_input
for --coqui instances. Stock AFL++ path is untouched (cold-path
if-branch only).
"
```

---

### Task 3.4: Gate det stages in `fuzz_one`

The deterministic stages (bitflip, arithmetic, interest, extras) are already gated by `afl->skip_deterministic`. We set that flag when parsing `--coqui` (Task 1.3). That covers most det code automatically. This task hardens the gating at the specific sites that might not fall under `skip_deterministic`.

**Files:**
- Modify: `src/afl-fuzz-one.c` (audit and reinforce gating at ~3-5 sites)

- [ ] **Step 1: Find det-stage entry points**

Run: `grep -n "skip_deterministic" src/afl-fuzz-one.c`

Expected output shows the existing checks. Most look like:
```c
if (!afl->skip_deterministic && ...) {
```

- [ ] **Step 2: Audit each site and confirm coverage**

Review each occurrence. Most det-stage bodies are already guarded by `!afl->skip_deterministic`, and because Task 1.3 sets `afl->skip_deterministic = 1` when `--coqui` is parsed, those guards already block det code in gpu_mode.

**No additional code change needed if every det-stage entry is behind `skip_deterministic`.** Confirm by looking at each grep hit and reading the surrounding block.

- [ ] **Step 3: If any det-stage site is NOT behind `skip_deterministic`, add the guard**

For each unprotected site, add `&& !afl->gpu_mode` to its condition, or wrap the block:

```c
if (!afl->gpu_mode) {
  /* det work */
}
```

In practice this is a no-op change for P1-P4 if all sites are already properly gated. If found, use a single commit for the audit.

- [ ] **Step 4: Verify compilation**

Run: `make -j$(nproc) afl-fuzz 2>&1 | tail -10`
Expected: builds cleanly.

- [ ] **Step 5: Commit (only if any change was made)**

```bash
git add src/afl-fuzz-one.c
git commit -m "coqui mode: audit det-stage gating under gpu_mode

Most det sites are already gated by skip_deterministic (set when
--coqui is parsed). This commit adds explicit gpu_mode guards to any
det-related site that was not already covered.
"
```

If no change was made, skip the commit and note in the next task's message that audit was clean.

---

### Task 3.5: Gate cmplog stage in `fuzz_one`

CmpLog requires a cmplog-instrumented target binary (via `-c`). Under `--coqui`, the target is a cubin, not a cmplog ELF. The cmplog stage must be skipped.

**Files:**
- Modify: `src/afl-fuzz-one.c` (find cmplog stage entry, add gpu_mode guard)

- [ ] **Step 1: Find the cmplog stage entry in fuzz_one**

Run: `grep -n "cmplog\|shm.cmplog_mode" src/afl-fuzz-one.c | head -10`

Expected: find the branch that invokes the cmplog stage, typically gated by `if (afl->shm.cmplog_mode && ...)`.

- [ ] **Step 2: Add the gpu_mode guard**

Modify the branch to also require `!afl->gpu_mode`:

```c
if (afl->shm.cmplog_mode && !afl->gpu_mode && ...) {
  /* cmplog stage body */
}
```

If there are multiple invocation sites (e.g., one for shape-classifying colorization, one for the main cmplog run), gate each.

- [ ] **Step 3: Verify compilation**

Run: `make -j$(nproc) afl-fuzz 2>&1 | tail -10`
Expected: builds cleanly.

- [ ] **Step 4: Commit**

```bash
git add src/afl-fuzz-one.c
git commit -m "coqui mode: skip cmplog stage in gpu_mode

cmplog needs a cmplog-instrumented ELF passed via -c. Under --coqui the
target is a cubin, so the stage is gated off. Findings from broker-side
cmplog still reach --coqui instances via AFL++'s normal sync mechanism.
"
```

---

### Task 3.6: Initialize coqui context from `main()`

**Files:**
- Modify: `src/afl-fuzz.c` (call `coqui_init` near `afl_fsrv_init`, before the dry-run/fuzz-loop setup)

- [ ] **Step 1: Include the coqui header at the top of src/afl-fuzz.c**

Add to the include block:

```c
#include "afl-fuzz-coqui.h"
```

- [ ] **Step 2: Stash the cubin path from the trailing argv**

The first positional argument after `--` is the target (cubin under --coqui, ELF otherwise). AFL++ already stores this in `afl->argv[0]` or similar after parsing. Find the line where the trailing arg is captured — typically just after the getopt loop. Add alongside existing argv handling:

```c
  if (afl->gpu_mode) {
    if (optind >= argc) {
      FATAL("--coqui requires a target cubin path after '--'");
    }
    afl->coqui_cubin_path = ck_strdup(argv[optind]);
  }
```

Place this immediately after the `while ((opt = getopt_long(...)))` loop closes, before further argument processing.

- [ ] **Step 3: Call `coqui_init` in the appropriate spot**

Near the `afl_fsrv_init(&afl->fsrv);` call at `src/afl-fuzz.c:661`, add a follow-up branch:

```c
  afl_fsrv_init(&afl->fsrv);

  if (afl->gpu_mode) {
    coqui_init(afl, afl->coqui_cubin_path);
  }
```

`coqui_init` must be called **before** any `afl_fsrv_start` call because the fsrv shortcut at Task 3.1 assumes coqui is ready.

- [ ] **Step 4: Call `coqui_shutdown` on exit**

Find the cleanup/exit path in `main()` (search for `afl_fsrv_kill`, `destroy_queue`, or similar). Add:

```c
  if (afl->gpu_mode) {
    coqui_shutdown(afl);
  }
```

near those other teardown calls.

- [ ] **Step 5: Verify compilation**

Run: `make -j$(nproc) afl-fuzz 2>&1 | tail -10`
Expected: builds cleanly.

- [ ] **Step 6: Commit**

```bash
git add src/afl-fuzz.c
git commit -m "coqui mode: wire coqui_init/shutdown into afl-fuzz main()

Initializes the coqui context after afl_fsrv_init and before any
fsrv_start call (which short-circuits in coqui_mode). Cubin path is
taken from the trailing positional argument after '--'.
Teardown mirrors the existing fsrv cleanup.
"
```

---

### Task 3.7: Call `coqui_flush_batch` at stage boundaries

Per spec §5.4, a flush is needed at the end of each havoc/splice stage (and at `fuzz_one` exit) so `afl->queued_items` reflects every mutation before the stage decides whether it was productive.

**Files:**
- Modify: `src/afl-fuzz-one.c` (add flush calls at stage-end points)

- [ ] **Step 1: Locate the end of the havoc and splice stages**

Search for the `havoc_stage:` label and the `splicing_stage:` label in `src/afl-fuzz-one.c`, and find where each stage's main loop ends (typically just before an `abandon_entry:` or a label transition).

Run: `grep -n "havoc_stage:\|splicing_stage:\|abandon_entry:" src/afl-fuzz-one.c`

- [ ] **Step 2: Add flush calls at the end of each stage and at fuzz_one exit**

At the closing curly brace of the havoc stage's main for-loop, and likewise at the end of the splicing stage, add:

```c
  if (unlikely(afl->gpu_mode)) { coqui_flush_batch(afl); }
```

Also add the same line immediately before `abandon_entry:` label body (so the partial batch drains before returning to the caller).

- [ ] **Step 3: Include the coqui header at the top of the file**

Add:

```c
#include "afl-fuzz-coqui.h"
```

if not already present.

- [ ] **Step 4: Verify compilation**

Run: `make -j$(nproc) afl-fuzz 2>&1 | tail -10`
Expected: builds cleanly.

- [ ] **Step 5: Commit**

```bash
git add src/afl-fuzz-one.c
git commit -m "coqui mode: flush pending batch at stage boundaries

Ensures afl->queued_items reflects every havoc/splice mutation before
the stage-end productivity check runs. Also flushes at fuzz_one exit
so the queue-entry / mutation association stays coherent in stats.

Stub flush is a no-op today; real GPU backend drains the async kernel.
"
```

---

## Phase 4: Smoke test

Manual validation that coqui mode compiles and runs end-to-end under `--coqui` without crashing.

### Task 4.1: Build coqui mode from scratch

- [ ] **Step 1: Clean build**

```bash
make clean
make -j$(nproc) 2>&1 | tail -20
```

Expected: builds all tools cleanly (afl-fuzz, afl-showmap, afl-tmin, etc.). No warnings introduced by our diffs.

- [ ] **Step 2: Verify the binary accepts `--coqui`**

```bash
./afl-fuzz --help 2>&1 | grep -A2 coqui
```

Expected: `--coqui id` line appears in help output.

---

### Task 4.2: Run coqui mode with `--coqui` on a simple target

- [ ] **Step 1: Prepare a target and seed**

Use AFL++'s bundled test-instr program. Build it with afl-clang-fast:

```bash
./afl-clang-fast test-instr.c -o /tmp/test-instr
mkdir -p /tmp/seeds && echo -n "hello" > /tmp/seeds/seed0
rm -rf /tmp/out
```

- [ ] **Step 2: Run coqui mode in --coqui mode against the target**

The stub doesn't care what the target is (it doesn't execute anything). We're just checking that afl-fuzz runs without crashing.

```bash
timeout 15 ./afl-fuzz --coqui gpu0 -i /tmp/seeds -o /tmp/out -- /tmp/test-instr 2>&1 | tail -30
```

Expected:
- Startup banner prints.
- `coqui_mode stub initialized (batch_size=8192, max_input=..., budget=...)` line appears.
- Seed calibration completes without FATAL.
- Fuzz loop starts; TUI/stats appear.
- Process exits cleanly after 15s via timeout (not SIGABRT, SIGSEGV, or FATAL).

If it crashes or FATALs, debug before moving on. Likely causes:
- `afl->max_length == 0` at init → check Task 3.6 ordering (coqui_init must see valid max_length).
- Forkserver tried to start → Task 3.1 short-circuit not wired.
- fuzz_run_target hit afl_fsrv_run_target → Task 3.2 short-circuit missing.

- [ ] **Step 3: Inspect out/gpu0/ to verify the sync-dir layout**

```bash
ls -la /tmp/out/gpu0/
```

Expected: standard AFL++ out-dir layout (`queue/`, `crashes/`, `hangs/`, `fuzzer_stats`). The stub never flags inputs, so `queue/` only contains the seed(s); that's correct.

No commit for this task — it's a validation pass.

---

### Task 4.3: Run heterogeneous mesh (broker + coqui mode)

- [ ] **Step 1: Launch the broker in the background**

```bash
rm -rf /tmp/out
./afl-clang-fast test-instr.c -o /tmp/test-instr
./afl-fuzz -M main -i /tmp/seeds -o /tmp/out -- /tmp/test-instr &
BROKER_PID=$!
sleep 3
```

- [ ] **Step 2: Launch coqui mode as the GPU secondary**

In a separate shell or with `&`:

```bash
timeout 20 ./afl-fuzz --coqui gpu0 -i /tmp/seeds -o /tmp/out -- /tmp/test-instr 2>&1 | tail -20
```

Expected:
- Both processes start without FATAL.
- `out/main/` and `out/gpu0/` directories exist.
- No crashes from either process during the 20s run.

- [ ] **Step 3: Clean up**

```bash
kill $BROKER_PID 2>/dev/null
wait
```

No commit for this task — it's a validation pass.

---

## Phase 5: Docs

### Task 5.1: Add a Changelog entry

**Files:**
- Modify: `Changelog.md` (add an entry for the coqui mode fork)

- [ ] **Step 1: Add entry at the top of Changelog.md**

Open `Changelog.md` and add an entry above the existing 4.40c entry:

```markdown
### Version ++coqui mode (dev):
  - Added `--coqui <sync_id>` flag: runs afl-fuzz as a GPU secondary
    (coqui_mode). Currently ships a hollow stub; real GPU backend lands
    in a follow-up phase. See
    docs/superpowers/specs/2026-04-18-coqui-gpu-backend-design.md.

### Version ++4.40c (release):
```

- [ ] **Step 2: Commit**

```bash
git add Changelog.md
git commit -m "coqui mode: Changelog entry for --coqui stub

Documents the new flag and points at the design spec. Real GPU backend
work is deferred to a follow-up brainstorm."
```

---

### Task 5.2: Final build + smoke-test on the finished tree

- [ ] **Step 1: Clean build**

```bash
make clean
make -j$(nproc) 2>&1 | tail -5
```

Expected: builds cleanly on a fresh tree with all Phase 1-3 changes merged.

- [ ] **Step 2: Full CLI sanity check**

```bash
./afl-fuzz -h 2>&1 | grep -A1 coqui
./afl-fuzz --coqui gpu0 -i /tmp/seeds -o /tmp/out -- /tmp/test-instr &
sleep 10 && kill $!
```

Expected: `--coqui` appears in help; fuzzer starts and runs for 10s without error.

No commit for this task — it's a final validation pass.

---

## Self-Review

**Spec coverage check** (spec → plan mapping):

| Spec section | Plan task(s) |
|--------------|--------------|
| §1.4 New code surface | P1 all + P2 + P3 |
| §2.1 `--coqui` long-only flag | T1.3 |
| §2.2 max_length plumbing | T2.2 (reads `afl->max_length` in `coqui_init`) |
| §2.4 Mutual exclusions | T1.3 case handler |
| §2.5 Lifecycle | T3.1 fsrv short-circuit, T3.2 fuzz_run_target short-circuit, T3.6 init hookup |
| §3.1 forkserver.h field | T1.1 |
| §3.2 afl-fuzz.h fields | T1.2 |
| §3.3 afl-fuzz.c flag parsing | T1.3 |
| §3.4 fuzz_one gating | T3.4 (dets) + T3.5 (cmplog) |
| §3.5 common_fuzz_stuff branch | T3.3 |
| §3.6 afl_fsrv_start short-circuit | T3.1 |
| §3.7 Calibration routing | T3.2 (via fuzz_run_target short-circuit) |
| §3.8 New files | T2.1 + T2.2 |
| §4 Contract types + API | T2.1 |
| §4.6 Stub backend | T2.2 |
| §5.4 Stage-boundary flush | T3.7 |
| §6 Error handling | T2.2 (oversized rejection); real-GPU errors deferred per §9 |
| §7 Phased rollout | P1–P5 mapping matches exactly |

**Placeholder scan:** No "TBD", "TODO", or "add appropriate handling" markers in the plan. Each code step contains the literal code to write; each verify step contains the exact command and expected output.

**Type consistency check:**
- `coqui_ctx`, `coqui_batch_t`, `coqui_status_t` — defined in T2.1, used consistently in T2.2.
- `coqui_init/submit_input/flush_batch/calibrate_one/shutdown` — signatures match between T2.1 declaration and T2.2 implementation.
- `afl->gpu_mode`, `afl->coqui`, `afl->gpu_batch_size`, `afl->coqui_cubin_path` — declared in T1.2, used in T1.3/T2.2/T3.2/T3.3/T3.6/T3.7.
- `afl->fsrv.coqui_mode` — declared in T1.1, used in T1.3/T3.1.

All types and identifiers referenced across tasks trace back to a defining task.

---

Plan complete and saved to `docs/superpowers/plans/2026-04-18-coqui-gpu-backend.md`. Two execution options:

**1. Subagent-Driven (recommended)** — dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
