# GPU-side Havoc Mutations under `coqui_mode`

**Status:** draft — design approved 2026-04-21, pending written-spec review.
**Related specs:** `2026-04-18-cuafl-gpu-backend-design.md` (coqui_mode CUDA backend).

---

## 1. Goals and non-goals

### Goals (v1)

- Move AFL's havoc-stage mutation work onto the GPU under `coqui_mode`. CPU no longer pre-mutates inputs for havoc batches.
- **Statistical fidelity** with CPU havoc: same `mutation_array[]` weights for all four `(input_mode, fuzz_mode)` combinations, same stacking distribution (`1 << (1 + rand(havoc_stack_pow2))`), same operator semantics for all 37 `MUT_*` ops including extras (`EXTRA_*`, `AUTO_EXTRA_*`) and splice (`SPLICE_*`). *Not* bit-exact PRNG replay — each GPU thread has its own PRNG.
- **Sparse GPU data**: one resident weighted seed pool (N=256) per ping-pong side, refreshed per `queue_cur`. Per-batch HtoD drops to `slot_info[]` (~32 KB) plus a ~1 MB pool upload amortized across the whole stage.
- **Full replace** — no `AFL_COQUI_CPU_HAVOC` escape hatch. Accept/revert decision deferred to maintainer based on bench results.

### Non-goals (v1)

- Moving *deterministic* stages (bit-flip, arith, interest, extras-det, cmplog/redqueen) onto the GPU. Those stay on CPU; their pre-mutated inputs continue to enter the batch via the existing `coqui_submit_input()` path with `flag=0`.
- Custom mutators (`afl_custom_havoc_mutation`). CPU-only; if a target defines a stacked custom mutator we fall back to CPU havoc for that `queue_cur` with a one-shot `WARNF`.
- MOpt havoc (`fuzz_one`'s second entry point at `src/afl-fuzz-one.c:5240`). CPU-only for v1.
- Bit-exact AFL PRNG replay — explicitly ruled out (would require host-side PRNG pre-serialization).
- Frameshift mutation tracking (`fs_save`, `src/afl-fuzz-one.c:2273`). CPU-only; GPU threads don't touch it.

---

## 2. Architecture overview

### End-to-end data flow for one havoc stage on `queue_cur`

```
CPU                                              GPU
──────────────────────────────────────────────── ────────────────────────────────
fuzz_one_original enters havoc_stage
  │
  ▼
coqui_refresh_seed_pool(queue_cur)               d_seed_pool_{a,b}[≤ 1 MB]
  • picks queue_cur → slot 0                     d_seed_offsets_{a,b}[256]
  • weighted-samples up to 255 others             d_seed_lens_{a,b}[256]
  • HtoD on stream_pool (new 3rd stream)          d_seed_cumw_{a,b}[256]
  • fences against stream_{a,b} via CUevent
  │
  ▼ (inside fuzz_one's for (stage_cur...) loop)
coqui_submit_havoc_slot(seed_idx=0, flag=1)
  • writes 1 slot: h_slot_info[n] = (1<<24)|0
  • when n == batch_size, triggers ping-pong flip
  │
  ▼ (every batch_size submits)
coqui_launch_batch:                             __coqui_fuzz_kernel(tid):
  HtoD slot_info (32 KB)                         unpack slot_info[tid] → (flag, idx)
  launch kernel                                  if flag == 1:
  DtoH novelty, status                             u8 scratch[MAX_INPUT_SIZE] (.local)
  DtoH reported_count, reported_slab[:count]       memcpy(scratch, seed_pool+offsets[idx], lens[idx])
                                                   prng = splitmix64(prng_base ^ tid)
                                                   mutated_len = __coqui_havoc_mutate(
                                                                    scratch, base_len, MAX_INPUT_SIZE,
                                                                    idx, &prng)
                                                   input_ptr = scratch;  len = mutated_len
                                                 else (flag == 0):
                                                   input_ptr = input_bytes + offsets[tid]
                                                   len = lens[tid]
                                                 memory_init; fuzz_execute; classify; virgin_compare
                                                 if (novel || crash): compact-write scratch to reported_slab
  │
  ▼
process novelty bits → CPU forkserver verify    (kernel done)
  using h_reported_slab for mutated bytes
  │
  ▼
next stage_cur iteration
```

### Key invariants

- **CPU still owns `fuzz_one_original`'s outer loop** `for (stage_cur=0; stage_cur<stage_max; ++stage_cur)`. Per-iteration work shrinks to one ~20-line function call.
- **Three CUDA streams** after this change: `stream_a`, `stream_b` (existing batch ping-pong), `stream_pool` (new, seed-pool uploads; fenced against a/b via CUevent).
- **Mixed-flag batches are supported**: `flag=0` (pre-mutated from CPU deterministic stages) and `flag=1` (havoc) coexist in one batch; the kernel entry dispatches on the flag.
- **HtoD paths** unchanged in shape:
  - Per-`queue_cur`: seed-pool bytes + metadata on `stream_pool`.
  - Per-batch: `slot_info[]` on the batch's stream (and populated prefix of `input_bytes` when flag=0 slots exist).
- **DtoH paths**: `novelty` + `status` (unchanged) plus the new `reported_count` and compact `reported_slab` prefix.

---

## 3. GPU data model and kernel ABI

### 3.1 Persistent resources (module-level globals)

Bound by host via `cuModuleGetGlobal` + `cuMemcpyHtoD`, matching the existing pattern for `__coqui_virgin_map` and `__coqui_global_statics_pool_base` at `src/afl-fuzz-coqui.c:119, 231`.

| Symbol | Type | Size | Written when |
|---|---|---|---|
| `__coqui_seed_pool_a_base`, `__coqui_seed_pool_b_base` | `u8*` (packed) | ≤ 1 MB × 2 | per `queue_cur` havoc entry |
| `__coqui_seed_pool_offsets_a/b` | `u32[256]` | 1 KB × 2 | same |
| `__coqui_seed_pool_lens_a/b` | `u32[256]` | 1 KB × 2 | same |
| `__coqui_seed_pool_cumw_a/b` | `u32[256]` (prefix-sum weights for splice) | 1 KB × 2 | same |
| `__coqui_seed_pool_count` | `u32` | 4 B | same (may be < 256 for small corpora) |
| `__coqui_seed_pool_cumw_total` | `u32` | 4 B | same |
| `__coqui_seed_pool_base` | `u8*` (pointer — aliases to a or b) | 8 B | before each launch — picks active pool side |
| `__coqui_seed_pool_offsets` / `_lens` / `_cumw` | `u32*` (pointer — aliases) | 8 B × 3 | same |
| `__coqui_extras_base` | `u8*` | ≤ 64 KB bytes | once at init |
| `__coqui_extras_offsets`, `__coqui_extras_lens`, `__coqui_extras_cnt` | metadata | few KB | once at init |
| `__coqui_a_extras_*` | same shape as extras | ≤ 64 KB | opportunistic re-upload when `afl->a_extras_cnt` changes |
| `__coqui_mutation_array` | `__constant__ u32[256]` | 1 KB | when `(input_mode, fuzz_mode)` changes |
| `__coqui_mutation_array_size` | `__constant__ u32` | 4 B | same |
| `__coqui_havoc_stack_pow2` | `__constant__ u32` | 4 B | always-update per stage refresh (cheap) |
| `__coqui_prng_base` | `u64` | 8 B | before each launch |

### 3.2 Per-batch kernel arguments

Extends the kernel from 5 args to 7. Keeps the existing `input_bytes` / `offsets` / `lens` path for `flag=0` slots; adds `slot_info` for flag routing and `reported_slab` for compact output.

Flag constants (defined in `coqui_mode/runtime/coqui_runtime.h`, mirrored in `src/afl-fuzz-coqui.h`):

```c
#define COQUI_FLAG_PREMUT   0u   /* pre-mutated bytes in input_bytes (today's path) */
#define COQUI_FLAG_HAVOC    1u   /* GPU-side havoc mutation from seed pool */
```

```c
__coqui_fuzz_kernel(
  u8*  input_bytes,    // existing: pre-mutated bytes for flag=0 slots
  u32* offsets,        // existing: into input_bytes (flag=0 only)
  u32* lens,           // existing: length for flag=0 (flag=1 ignores)
  u32* slot_info,      // NEW: packed (flag:8 | seed_idx:24) per thread
  u8*  novelty,        // existing: 1 bit per thread
  u8*  status,         // existing: coqui_status_t per thread
  u8*  reported_slab   // NEW: compact DtoH buffer for mutated bytes on novel/crash
)
```

Packing `(flag, seed_idx)` into one `u32` keeps per-batch HtoD at 32 KB (vs 40 KB for separate arrays); 24 bits for seed index leaves headroom for future N > 256.

### 3.3 Returning mutated bytes to the host

A thread's mutated input lives in its `.local` scratch and is *not* host-addressable. But `process_input_via_cpu_fsrv()` at `src/afl-fuzz-coqui.c:310` needs the actual bytes to feed the CPU forkserver. Novelty rate is typically ≤ 1% of a batch, so a compact write-on-report scheme:

```cuda
// end of __coqui_fuzz_kernel, after virgin_compare
if (novelty_bit_set || crash_flag_set) {
  u32 slot = atomicAdd(&__coqui_reported_count, 1);
  if (slot < REPORTED_CAP) {
    u8* dst = reported_slab + slot * MAX_INPUT_SIZE;
    memcpy(dst, scratch, mutated_len);
    __coqui_reported_tid[slot]  = tid;
    __coqui_reported_lens[slot] = mutated_len;
  }
}
```

- `REPORTED_CAP = 512`. 512 × 4 KB = 2 MB reporting slab per ping-pong side.
- Host-side: after DtoH of `novelty` / `status`, also DtoH `__coqui_reported_count` + the populated prefix of `reported_slab` / `reported_tid` / `reported_lens`.
- Process-novelty loop at `src/afl-fuzz-coqui.c:665-678` gains a tid→slot lookup, reads mutated bytes from `h_reported_slab + slot * MAX_INPUT_SIZE`.
- On overflow (`reported_count ≥ REPORTED_CAP`): `WARNF`, skip CPU verify for the overflow slots — matches the graceful-degradation pattern already used for the crash-dedup hash table at `src/afl-fuzz-coqui.c:730-733`.

---

## 4. Kernel changes and runtime helper

### 4.1 `FuzzEntry.cpp` — two new basic blocks, one new call

Current kernel body at `coqui_mode/passes/FuzzEntry.cpp:113-227` is `entry → run → exit`. The change adds a flag-dispatch block and a mutation path:

```
entry:
  tid = __coqui_fuzz_tid()
  slot = slot_info[tid]                           # NEW
  flag = slot >> 24;  idx = slot & 0xFFFFFF        # NEW
  if (flag == 0) goto premut                      # NEW
  if (flag == 1) goto havoc                       # NEW
  goto exit                                       # unknown flag → no-op

premut:                                           # existing path, unchanged
  len = lens[tid];  if (len == 0) goto exit
  input_ptr = input_bytes + offsets[tid]
  goto run

havoc:                                            # NEW
  base_len = __coqui_seed_pool_lens[idx]
  if (base_len == 0) goto exit
  u8 scratch[MAX_INPUT_SIZE]                      # .local, compile-time sized
  memcpy(scratch, __coqui_seed_pool_base + __coqui_seed_pool_offsets[idx], base_len)
  prng = splitmix64(__coqui_prng_base ^ tid)
  mutated_len = __coqui_havoc_mutate(scratch, base_len, MAX_INPUT_SIZE, idx, &prng)
  input_ptr = scratch;  len = mutated_len
  goto run

run:                                              # existing path, unchanged
  memory_init
  fuzz_execute(input_ptr, len)
  classify_and_sig
  virgin_compare
  # NEW: compact-report block from §3.3

exit: ret
```

`MAX_INPUT_SIZE` is a compile-time macro via `-DMAX_INPUT_SIZE=4096`. `src/afl-fuzz-coqui.c:coqui_init()` gains a FATAL guard: `if (afl->max_length > MAX_INPUT_SIZE)`.

### 4.2 `coqui_mode/runtime/coqui_mutate.c` — new runtime module

Single entry point plus internal helpers:

```c
/* Returns new length. Mutates in place; may grow/shrink up to max_len. */
u32 __coqui_havoc_mutate(u8 *buf, u32 len, u32 max_len, u32 self_idx, u64 *prng);
```

Near-verbatim port of `src/afl-fuzz-one.c:2259-3543`:

```c
u32 __coqui_havoc_mutate(u8 *buf, u32 len, u32 max_len, u32 self_idx, u64 *prng) {
  u32 stack_max = 1u << (1 + gpu_rand_below(prng, __coqui_havoc_stack_pow2));
  u32 use_stacking = 1 + gpu_rand_below(prng, stack_max);
  for (u32 i = 0; i < use_stacking; ++i) {
    retry_havoc_step:;
    u32 r = gpu_rand_below(prng, __coqui_mutation_array_size);
    switch (__coqui_mutation_array[r]) {
      case MUT_FLIPBIT:       /* port of afl-fuzz-one.c case MUT_FLIPBIT */ break;
      case MUT_INTERESTING8:  /* ... */ break;
      /* ... all 37 ops, one case each, lifted verbatim from afl-fuzz-one.c ... */
      case MUT_SPLICE_OVERWRITE: {
        u32 partner = weighted_splice_pick(prng, self_idx);
        if (partner == SELF_SAME) goto retry_havoc_step;
        /* overwrite bytes from seed_pool[partner] */
      } break;
    }
  }
  return len;
}
```

Mechanical substitutions from host code:
- `rand_below(afl, n)` → `gpu_rand_below(prng, n)`
- `out_buf` → `buf`
- `temp_len` → local `len`
- Extras ops read `__coqui_extras_*` and `__coqui_a_extras_*` globals
- Splice ops read `__coqui_seed_pool_*` globals

### 4.3 PRNG — `splitmix64`

- 64-bit state fits in one register; lower register pressure than `xoshiro128+` at 8192 in-flight threads.
- Thread seeding: `prng_state = splitmix64(prng_base ^ tid)` — decorrelated streams per thread even with a fixed `prng_base`.
- `gpu_rand_below()` uses Lemire's fast unbiased bounded random (`__umul64hi`-style 32×32→64 multiply; lowers to one `MUL.WIDE.U32` on Turing). Rejection loop rarely fires.

### 4.4 Weighted splice pick

CPU precomputes `__coqui_seed_pool_cumw` (prefix sum of `queue_entry.weight` for pool slots). Kernel does an 8-step binary search:

```c
#define COQUI_SPLICE_SELF_SAME  (~0u)

static inline u32 weighted_splice_pick(u64 *prng, u32 self_idx) {
  u32 r = gpu_rand_below(prng, __coqui_seed_pool_cumw_total);
  u32 lo = 0, hi = __coqui_seed_pool_count;
  while (lo < hi) {
    u32 m = (lo + hi) >> 1;
    if (__coqui_seed_pool_cumw[m] <= r) lo = m + 1; else hi = m;
  }
  return (lo == self_idx) ? COQUI_SPLICE_SELF_SAME : lo;
}
```

On `COQUI_SPLICE_SELF_SAME` the caller does `goto retry_havoc_step` — same retry mechanism as CPU AFL uses for unusable ops (splice partner too short, zero-length input). No infinite-loop risk because the retry re-rolls `r` against the full `mutation_array[]` and most ops are non-splice. Edge case — pool with a single entry (self only) — is avoided by construction: `coqui_refresh_seed_pool()` always samples ≥ 1 non-self slot when the corpus has > 1 entry; when the corpus has exactly 1 entry, we skip splice ops entirely by setting `__coqui_seed_pool_count = 1` and letting the retry fall through (splice ops will always roll `SELF_SAME` and retry to some non-splice op from `mutation_array[]`).

---

## 5. CPU-side changes

### 5.1 `fuzz_one_original` — havoc_stage body replacement

The existing havoc body at `src/afl-fuzz-one.c:2259-3543` (≈1500 lines) is the whole code path being offloaded. The diff is a runtime branch at the top of the stage loop:

```c
if (afl->coqui) {
  /* GPU-havoc: thin stage loop */
  for (afl->stage_cur = 0; afl->stage_cur < afl->stage_max; ++afl->stage_cur) {
    if (unlikely(afl->stop_soon)) goto abandon_entry;
    coqui_submit_havoc_slot(afl, /*seed_idx=*/0, /*flag=*/COQUI_FLAG_HAVOC);
  }
  coqui_flush_batch(afl);   /* drain trailing partial batch */
} else {
  /* original CPU havoc body — unchanged, ≈1500 lines preserved for non-coqui builds */
  for (afl->stage_cur = 0; afl->stage_cur < afl->stage_max; ++afl->stage_cur) {
    /* ... existing stack_max / use_stacking / switch(mutation_array[r]) ... */
  }
}
```

- CPU-havoc branch fully preserved for non-coqui `afl-fuzz` builds.
- `stage_finds[STAGE_HAVOC]` / `stage_cycles[STAGE_HAVOC]` tally moves into `coqui_await_and_process()`, which already knows when a batch finishes and which slots found novelty.
- Splice-cycle wrapper (`splice_cycle > 0` path) stays on CPU: constructs the spliced base, uploads it as the new `seed_pool[0]` via `coqui_refresh_seed_pool()`, then re-enters the same GPU-havoc loop. No duplicated logic.
- Custom mutators: if `afl->custom_mutators_count > 0` and any mutator has `el->stacked_custom` set, fall back to the CPU path for that `queue_cur` with a one-shot `WARNF`.

### 5.2 `coqui_submit_havoc_slot()` — the per-iteration CPU cost

```c
u8 coqui_submit_havoc_slot(afl_state_t *afl, u32 seed_idx, u8 flag) {
  coqui_ctx_t  *ctx = afl->coqui;
  coqui_batch_t *b  = ctx->pending;

  ctx->total_submits++;
  afl->fsrv.total_execs++;

  if (b->n_inputs == ctx->batch_size) {
    coqui_launch_batch(afl, b);
    /* ping-pong flip — same mechanics as coqui_submit_input */
    coqui_batch_t *tmp = ctx->pending;
    ctx->pending = ctx->executing;
    ctx->executing = tmp;
    b = ctx->pending;
    if (b->n_inputs > 0) {
      if (coqui_await_and_process(afl, b) == 1) return 0;   /* force-reset path */
      b->n_inputs = 0;  b->bytes_used = 0;
    }
  }

  b->h_slot_info[b->n_inputs] = (flag << 24) | (seed_idx & 0xFFFFFFu);
  b->n_inputs++;
  return 0;
}
```

~20 lines, one atomic increment, one conditional, one indexed store. Orders of magnitude cheaper than the mutation switch it replaces.

### 5.3 `coqui_refresh_seed_pool()` — once per havoc_stage entry

Called at the top of `havoc_stage:` before the submit loop. Steps:

1. Pick next ping-pong side (`1 - ctx->seed_pool_active`).
2. Build pool in host-pinned memory:
   - Slot 0: `afl->queue_cur` (always present, never sampled).
   - Slots 1..N-1: N-1 weighted-random-with-replacement picks over the queue using `queue_entry.weight` (computed by AFL's existing `compute_weight()` in `afl-fuzz-queue.c`). Skip disabled entries. Compute `cumw[i]` as prefix sum.
3. `cuMemcpyHtoDAsync` bytes + offsets + lens + cumw onto `stream_pool`.
4. Fence: `cuEventRecord` on `stream_pool`; `cuStreamWaitEvent` on `stream_a` and `stream_b`.
5. Flip `__coqui_seed_pool_base` / `_offsets` / `_lens` / `_cumw` pointer symbols to point at the new side via 4× 8 B `cuMemcpyHtoD`.
6. `ctx->seed_pool_active = next_side`.

Pool can be < 256 slots when the corpus is small or the packed pool bytes would exceed 1 MB.

### 5.4 Resource sync table

| What | When | Cost |
|---|---|---|
| `__coqui_extras_*` | Once at `coqui_init()` (skip if `afl->extras_cnt == 0`) | ≤ 64 KB HtoD |
| `__coqui_a_extras_*` | In `coqui_refresh_seed_pool()` if `afl->a_extras_cnt != ctx->a_extras_uploaded_cnt` | ≤ 64 KB HtoD (rare) |
| `__coqui_mutation_array` | In `coqui_refresh_seed_pool()` if `(input_mode, fuzz_mode)` changed | 1 KB `cuMemcpyHtoDAsync` to `__constant__` |
| `__coqui_havoc_stack_pow2` | In `coqui_refresh_seed_pool()` (always-update) | 4 B |
| `__coqui_prng_base` | In `coqui_launch_batch()` per batch, sourced from `get_rand_seed()` | 8 B |

### 5.5 Build-system changes

- `coqui_mode/runtime/coqui_mutate.c` (new, ~500-800 LOC, one case per `MUT_*`) added to the cubin build in `coqui_mode/build_coqui_support.sh`; one line added to `coqui_mode/runtime/CMakeLists.txt` (if that's the driver; otherwise whatever file drives the existing runtime .c compilation).
- Target cubin compilation gets `-DMAX_INPUT_SIZE=4096`.
- `src/afl-fuzz-coqui.c:coqui_init()` adds: `if (afl->max_length > MAX_INPUT_SIZE) FATAL(...)` with guidance to bump via `-DMAX_INPUT_SIZE=N` at target compile.
- `coqui_mode/passes/FuzzEntry.cpp`: one new kernel arg (`slot_info`), two new basic blocks (`premut`, `havoc`), one new call site (`__coqui_havoc_mutate`), compact-report block at end. **No changes** to `StaticGlobals`, `Asan`, `Coverage`, `Libc`, `Heap`, `MemoryLayout`, other passes.

---

## 6. Memory and stack-budget math

### 6.1 Per-thread `.local` budget

On a device granting the 512 KB ceiling (TITAN RTX sm_75 measured):

| Region | Size | Source |
|---|---|---|
| cov_map | 64 KB | `cov = 65536` (`src/afl-fuzz-coqui.c:167`) |
| heap | ~370 KB | `(remaining × 8) / 9` |
| shadow | ~47 KB | `heap / 8` |
| real_stack | 32 KB | `AFL_COQUI_STACK_SIZE` default |
| **`scratch[MAX_INPUT_SIZE]` (NEW)** | **4 KB** | compile-time |
| PRNG state | 8 B | one `u64` register |
| **Total per-thread** | **~513 KB** | fits 512 KB ceiling after trim |

`scratch[]` is compile-time-sized, so it gets counted in static stack frame and caught by the existing `cuFuncGetAttribute(CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES)` check at `src/afl-fuzz-coqui.c:178-183`. No silent overflow risk; existing `AFL_COQUI_STACK_SIZE` escape hatch lets users bump the budget per target.

Minimum sustainable `AFL_COQUI_STACK_SIZE`: 8 KB. Documented in init-time error message.

### 6.2 Device global memory additions

| Allocation | Size |
|---|---|
| Seed pool A (bytes + off + lens + cumw) | ~1 MB |
| Seed pool B | ~1 MB |
| Extras dictionary | ≤ 64 KB |
| a_extras slot | 64 KB cap |
| Reported-slab (×2 ping-pong) | 2 × 2 MB |
| `d_slot_info` (×2 ping-pong) | 2 × 32 KB |
| reported tid/lens/count | few KB |
| **Subtotal new** | **~7 MB** (vs 24 GB available) |

### 6.3 Per-batch HtoD volume (the win)

| | Pre-change | Post-change (pure-havoc batch) |
|---|---|---|
| `input_bytes` (populated prefix — line 414) | up to **8 MB** | **0 B** |
| `offsets` / `lens` | 64 KB | 0 (kernel skips read when all flag=1) |
| `slot_info` | — | 32 KB |
| **Per-batch HtoD** | **≈ 8 MB** | **≈ 32 KB** |
| Per-`queue_cur` pool upload (amortized) | — | ~1 MB × (1 / stage_max) ≈ negligible |

**≈ 250× reduction** for pure-havoc batches. Mixed batches (some flag=0 from deterministic stages) save proportionally.

### 6.4 Per-batch DtoH volume

| | Today | After |
|---|---|---|
| `novelty` | 1 KB | 1 KB |
| `status` | 128 KB | 128 KB |
| `reported_slab` (compact) | — | avg ~320 KB (1% novelty × 4 KB × 8192) |
| `reported_count` / `_tid` / `_lens` | — | few KB |
| **Per-batch DtoH** | **≈ 130 KB** | **≈ 450 KB avg** |

~3.5× DtoH growth; small relative to PCIe bandwidth (avg 9 MB/s at 20 batch/s). One cost to watch during bench.

---

## 7. Bench and rollout

### 7.1 Bench methodology

Per project memory: cjson n ≥ 20 runs, 120 s+ windows, device 0 (RTX TITAN sm_75).

| Metric | Target | Source |
|---|---|---|
| **execs/sec** | ≥ 20% improvement vs CPU-havoc baseline | `fuzzer_stats: execs_per_sec` |
| **edges/sec** | within −15% of baseline (lower bound matters) | `edges_found / wall_clock` |
| **crash count** | ≥ baseline | `fuzzer_stats: saved_crashes` |
| **stability** | 0 force-resets from mutation logic, 0 new `WARNF` / `FATAL` | stderr scan |

Bench runner: extend `benchmark/` with a script interleaving CPU-havoc and GPU-havoc 120 s runs, dumping per-run CSV.

Fidelity is validated *by the bench itself* — if GPU havoc's operator mix diverges from CPU's, coverage growth falls out of the −15% band and the bench fails.

### 7.2 Rollout sequencing

Land in four separate commits on the `cuAFL` branch; do not merge to `main` until the bench accept/revert decision:

1. **Runtime module** — add `coqui_mode/runtime/coqui_mutate.c` (all 37 `MUT_*` ports + PRNG + weighted splice pick). Compiles into the cubin but has no callers yet. cjson behavior unchanged.
2. **Kernel ABI + pool plumbing** — `FuzzEntry.cpp` adds `slot_info` arg, flag-dispatch blocks, compact-report path; `src/afl-fuzz-coqui.c` adds seed-pool refresh, `coqui_submit_havoc_slot`, reported-slab handling. All existing flag=0 behavior preserved. cjson behavior unchanged.
3. **`fuzz_one_original` GPU-havoc branch** — the runtime `if (afl->coqui)` switch replacing the havoc body. This is the flip point.
4. **Bench** — n = 20 × 120 s interleaved CPU-havoc vs GPU-havoc. Results reported to maintainer. **Maintainer** calls accept/revert.

Regression at step 4 → revert commit 3 only; commits 1-2 are functionally inert and can stay on the branch.

### 7.3 Open risks (watch during bench)

| Risk | Signal | Mitigation |
|---|---|---|
| `MAX_INPUT_SIZE=4096` too small for some targets | init-time FATAL on targets with larger max_length | recompile target with `-DMAX_INPUT_SIZE=N` |
| DtoH of reported_slab bottlenecks | `[coqui-rate]` `verify=` rises sharply | v2: DtoH sized by actual `reported_count` instead of `REPORTED_CAP` |
| Splice weighted sampling biases coverage | edges/sec diverges > 15% from baseline | switch to uniform-random sampler (~5 lines) |
| `a_extras` grows faster than refresh | cmplog-learned dict stale during long runs | shorten refresh cadence |
| `__coqui_prng_base` periodicity correlates mutations across batches | coverage plateaus with oscillation | source `prng_base` from `/dev/urandom` per launch |
