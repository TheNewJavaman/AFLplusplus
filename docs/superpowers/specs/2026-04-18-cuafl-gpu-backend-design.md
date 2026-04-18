# cuAFL: AFL++ with a GPU Execution Backend — Design Spec

**Status:** Draft
**Date:** 2026-04-18
**Scope:** AFL++ integration layer only. The coqui GPU backend internals (LLVM passes, CUDA runtime, compiler) are intentionally deferred to a follow-up brainstorm.

## 1. Overview

cuAFL is an AFL++ fork that adds a new executor mode, `coqui_mode`, selected via a new long-only flag `--coqui <sync_id>`. A cuAFL instance running under `--coqui` behaves as a specialized AFL++ secondary: it shares a sync directory with `-M main` and `-S <id>` peers, writes findings into its own `out/<sync_id>/queue/`, and receives propagated inputs via AFL++'s normal `sync_fuzzers()` mesh. The distinguishing feature is that it executes mutated inputs on an NVIDIA GPU in batches of 8K rather than one at a time through a forkserver.

Every AFL++ subsystem above the execution layer — mutations, corpus management, virgin_bits tracking, power schedules, the TUI, stats, and sync — runs unmodified. The integration seam is the function `common_fuzz_stuff()` in `src/afl-fuzz-run.c`: under `--coqui`, a new branch routes inputs into a packed batch buffer instead of invoking a forkserver. When a batch completes on the GPU, each input flagged as potentially novel is handed back to `save_if_interesting()` with its coverage populated in `trace_bits`. AFL++ decides local novelty; the broker decides canonical novelty; sync propagates findings in both directions.

### 1.1 Deployment topology

```
                                  sync dir (out/)
                                       │
         ┌─────────────────────────────┼─────────────────────────────┐
         │                             │                             │
   ┌─────▼─────┐                ┌──────▼──────┐                ┌─────▼─────┐
   │  -M main  │                │  -S cpu0    │                │ --coqui   │
   │           │                │  (optional) │                │   gpu0    │
   │ CPU fsrv  │                │ CPU fsrv    │                │ CUDA batch│
   │           │                │             │                │           │
   │ dets +    │                │ havoc +     │                │ havoc +   │
   │ cmplog +  │◄───── sync ───►│ splice      │◄──── sync ────►│ splice    │
   │ havoc +   │                │ (+ASan, ...)│                │ (8K/batch)│
   │ splice    │                │             │                │           │
   └───────────┘                └─────────────┘                └───────────┘
                                                                     │
                                                                     ▼
                                                              ┌─────────────┐
                                                              │  NVIDIA GPU │
                                                              │  kernel     │
                                                              │  (cubin)    │
                                                              └─────────────┘
```

### 1.2 Division of responsibility

- **Broker (`-M`)**: authoritative corpus; dets and cmplog (insight-bound work); canonical novelty arbitrator.
- **CPU secondaries (`-S`, optional)**: additional CPU variants — sanitizer builds, alternate instrumentation. Unmodified AFL++.
- **GPU secondary (`--coqui`)**: pure havoc + splice throughput specialist. Speculates on GPU-local coverage. Proposes flagged inputs to the sync dir. Det/cmplog stages are gated off.

### 1.3 Zero-modification subsystems

`src/afl-fuzz-queue.c`, `src/afl-fuzz-bitmap.c`, `src/afl-fuzz-stats.c`, sync machinery in `src/afl-fuzz-init.c`/`src/afl-fuzz-state.c`, the TUI, power schedules (`calculate_score()` in `afl-fuzz-queue.c`), and all mutation code in `src/afl-fuzz-one.c` (havoc mutation choice logic). The `--coqui` instance uses every one unchanged — it just routes execution through a different backend.

### 1.4 New code surface

- **Core afl-fuzz**: ~150-200 LOC across 7 files (CLI flag, `fuzz_one` gating, `common_fuzz_stuff` batch branch, state fields, `afl_fsrv_start` short-circuit, calibration routing).
- **New module** `src/afl-fuzz-coqui.{c,h}`: glue layer — packed batch buffers, ping-pong scheduling, novelty-bitmap processing, per-flagged-input `save_if_interesting` callback. Ships with a stub implementation (see §4.5).
- **`coqui_mode/`**: new subdirectory. Internals deferred — see §9.

### 1.5 Why a new mode rather than riding the forkserver abstraction

AFL++'s existing executor modes (`qemu_mode`, `frida_mode`, `unicorn_mode`, `nyx_mode`, `coresight_mode`) all speak the single-input forkserver IPC protocol. cuAFL cannot — batching 8K inputs per kernel launch is the entire premise. The hook point must be above the forkserver abstraction, which is why the integration lives at `common_fuzz_stuff` rather than at the forkserver layer. This makes cuAFL the first AFL++ executor mode to break the single-input-at-a-time assumption. The new code surface is correspondingly larger than those other modes, but still narrowly scoped to one function's branch.

## 2. CLI and Instance Lifecycle

### 2.1 Flag

```
afl-fuzz --coqui <sync_id> -i <seeds> -o <out> -- <target.cubin>
```

`--coqui` is a long-only flag. The short-flag letter pool in `src/afl-fuzz.c` (see the maintained comment at line 718) is too tight to claim a mnemonic letter, and long-only flags match AFL++'s convention for niche mode switches.

The trailing positional argument is a cubin path (produced by `coqui-cc`, the compiler in `coqui_mode/`). No `@@` substitution — the kernel reads inputs from a device-side buffer we DMA into, not from a filesystem path.

### 2.2 Max input size

cuAFL does not add a max-input-size flag. AFL++'s existing `-G <bytes>` flag (max input length) propagates through `afl->max_length` → `afl->fsrv.max_length` and is read directly by `coqui_init()`. If `max_length == 0` (AFL++'s "unbounded" default), coqui_mode falls back to a compile-time `COQUI_MAX_INPUT_DEFAULT` (4096) and emits a warning recommending an explicit `-G` for GPU fuzzing.

### 2.3 Typical campaign

```bash
# Build CPU and GPU targets from the same sources
afl-clang-fast            harness.c lib.c -o target
AFL_LLVM_CMPLOG=1 afl-clang-fast harness.c lib.c -o target.cmplog
coqui-cc                  harness.c lib.c -o target.cubin

# Launch in separate terminals, same -o dir
afl-fuzz -M main -c target.cmplog -i seeds/ -o out/ -- target
afl-fuzz --coqui gpu0             -i seeds/ -o out/ -- target.cubin
```

### 2.4 Mutual exclusions (enforced at CLI parse)

- `--coqui` is mutually exclusive with `-M` and `-S` (pick one role).
- `--coqui` is mutually exclusive with `-Q` / `-O` / `-U` / `-X` (different executors cannot share one process).
- `--coqui` implicitly sets `afl->skip_deterministic = 1` (equivalent to `-d`). Det stages are definitionally absent from `--coqui` mode.
- `--coqui` requires the cubin path to exist and be readable at startup.

### 2.5 Lifecycle diffs from `-S`

| Step | `-S cpu0` | `--coqui gpu0` |
|------|-----------|----------------|
| 1. CLI parse | normal | + cubin path validation, CUDA availability check |
| 2. `afl_state_init` | normal | + allocate `afl->coqui` context pointer |
| 3. Set up `out/<sync_id>/` | normal | identical |
| 4. Set up `trace_bits` shmem | normal | identical (coverage copied *into* this per flagged input) |
| 5. Launch executor | `afl_fsrv_start()` → fork+exec target, pipe handshake | `coqui_init()` → CUDA context, load module, allocate ping-pong buffers, init runtime |
| 6. Calibrate seeds | forkserver execs one at a time | single-input GPU launch per seed (batch-of-1) |
| 7. Enter `fuzz_one` loop | all stages | havoc + splice only (det/cmplog gated by `afl->gpu_mode`) |
| 8. Sync inbound | run synced inputs through forkserver | single-input GPU launch per sync-in |

### 2.6 Fail-fast conditions at startup

- CUDA driver not loaded or no GPU visible → `FATAL("--coqui requires an NVIDIA GPU with CUDA driver")`.
- Cubin load fails → `FATAL("--coqui: failed to load cubin: <path>: <cuda error>")`.
- Cubin arch mismatches detected GPU arch → `FATAL("--coqui: cubin targets %s, detected GPU is %s")`.
- Device memory allocation fails → `FATAL` with byte count.

## 3. Core afl-fuzz Modifications

All changes are minimal and surgical. Every branch is `if (unlikely(afl->gpu_mode))` — cold path for stock AFL++ users, no measurable perf impact.

### 3.1 `include/forkserver.h`

Add one field alongside existing mode booleans (`qemu_mode`, `frida_mode`, `nyx_mode`, `cs_mode`):

```c
bool coqui_mode;   /* if running in coqui (GPU) mode or not */
```

### 3.2 `include/afl-fuzz.h`

Add to `afl_state_t`:

```c
u8                gpu_mode;         /* 1 if --coqui was given */
struct coqui_ctx *coqui;            /* opaque pointer (defined in afl-fuzz-coqui.h) */
u32               gpu_batch_size;   /* default 8192, tunable via env */
```

### 3.3 `src/afl-fuzz.c`

`--coqui` added to the long-options table (no short flag). Handler logic mirrors the `-S` block at line 1009:

```c
{"coqui", required_argument, NULL, LONGOPT_COQUI},
/* ... */
case LONGOPT_COQUI:
  if (afl->sync_id)
    FATAL("Multiple -S/-M/--coqui options not supported");
  if (afl->fsrv.qemu_mode || afl->fsrv.frida_mode || afl->fsrv.nyx_mode
      || afl->fsrv.cs_mode)
    FATAL("--coqui is mutually exclusive with -Q/-O/-U/-X");
  if (optarg && *optarg == '-')
    FATAL("argument for --coqui started with a dash");
  afl->sync_id            = ck_strdup(optarg);
  afl->is_secondary_node  = 1;
  afl->gpu_mode           = 1;
  afl->fsrv.coqui_mode    = 1;
  afl->skip_deterministic = 1;
  break;
```

Plus one entry in the `--help` output.

### 3.4 `src/afl-fuzz-one.c`

Gate det and cmplog stages:

```c
/* Deterministic stages */
if (!afl->skip_deterministic && !afl->gpu_mode) {
  /* existing det code runs */
}

/* CmpLog stage */
if (afl->shm.cmplog_mode && !afl->gpu_mode) {
  /* existing cmplog code runs */
}
```

Audit the ~3-5 call sites that invoke det/cmplog-specific mutations (including `custom_mutators_count` deterministic paths) and tag each.

### 3.5 `src/afl-fuzz-run.c` — the load-bearing change

`common_fuzz_stuff` gets a prelude:

```c
u8 __attribute__((hot)) common_fuzz_stuff(afl_state_t *afl, u8 *out_buf, u32 len) {
  if (unlikely(afl->gpu_mode)) {
    return coqui_submit_input(afl, out_buf, len);
  }
  /* existing CPU path unchanged from here */
  ...
}
```

`coqui_submit_input` lives in `afl-fuzz-coqui.c`. The stock CPU path is literally untouched.

### 3.6 `src/afl-forkserver.c`

`afl_fsrv_start` short-circuits when `coqui_mode` is set, following the `nyx_mode` pattern at line 160:

```c
if (fsrv->coqui_mode) {
  /* No forkserver for coqui_mode — coqui_init() owns execution.
     Called earlier in afl_state init path. */
  return;
}
```

Add an assertion at `afl_fsrv_run_target` entry that fires if it's somehow reached with `coqui_mode` set (defense-in-depth).

### 3.7 `src/afl-fuzz-init.c`

Seed calibration currently calls `fuzz_run_target()` once per seed. Add a `gpu_mode` branch that calls `coqui_calibrate_one(afl, buf, len)` instead. ~20 LOC.

### 3.8 New files: `src/afl-fuzz-coqui.c` + `include/afl-fuzz-coqui.h`

The glue layer. Details in §4.

### 3.9 File summary

**Modified (7):** `afl-fuzz.c`, `afl-fuzz.h`, `forkserver.h`, `afl-fuzz-one.c`, `afl-fuzz-run.c`, `afl-forkserver.c`, `afl-fuzz-init.c`.
**New (2):** `afl-fuzz-coqui.c`, `afl-fuzz-coqui.h`.

Total diff without `coqui_mode/`: estimated 150-200 LOC.

## 4. The `coqui_mode` Contract

This is the interface between AFL++ and the GPU backend. Implementation internals of `coqui_mode/` are free to change; the contract below is stable.

### 4.1 API exposed to core afl-fuzz

```c
/* Initialize GPU context, load cubin, allocate ping-pong buffers.
   Fails fast on any CUDA error. */
void coqui_init(afl_state_t *afl, const char *cubin_path);

/* Append one mutated input to the pending batch.
   Returns 0 unconditionally — never signals stage bail-out.
   May trigger an async kernel launch if the pending batch fills. */
u8   coqui_submit_input(afl_state_t *afl, u8 *buf, u32 len);

/* Force-launch any partial batch and wait for all pending work to drain.
   Called at stage boundaries so afl->queued_items reflects every mutation. */
void coqui_flush_batch(afl_state_t *afl);

/* Single-input synchronous path for initial seed calibration and sync-in.
   Populates trace_bits and returns fsrv_run_result_t. */
u8   coqui_calibrate_one(afl_state_t *afl, u8 *buf, u32 len);

/* Free resources and tear down CUDA context on exit. */
void coqui_shutdown(afl_state_t *afl);
```

### 4.2 Batch buffer layout — dense-packed, 8-byte-aligned

Inputs are packed contiguously into a single pinned host buffer, with each input's start offset aligned to 8 bytes so device-side word loads are safe. Per-input metadata (offset, length) is held in parallel arrays.

```
Per ping-pong half:

  h_input_bytes   [BATCH_BYTE_BUDGET]         pinned host  — packed inputs, each 8-aligned
  h_offsets       [batch_size * 4]            pinned host  — byte offset of input i
  h_input_lens    [batch_size * 4]            pinned host  — actual length of input i
  h_novelty       [batch_size / 8]            pinned host  — 1 bit per input (DMA dest)
  h_status        [batch_size * sizeof(status_t)]  pinned host

  d_input_bytes   [BATCH_BYTE_BUDGET]         device       — same, packed
  d_offsets       [batch_size * 4]            device
  d_input_lens    [batch_size * 4]            device
  d_coverage      [batch_size * map_size]     device       — per-thread coverage
  d_novelty       [batch_size / 8]            device
  d_virgin        [map_size]                  device       — shared, atomic OR
  d_status        [batch_size * sizeof(status_t)]  device
```

With `batch_size = 8192`, `map_size = 65536`, `d_coverage` is 512 MB. Pinned host memory per buffer is ~10 MB at typical fill; the ping-pong pair doubles this.

### 4.3 Packing algorithm (inside `coqui_submit_input`)

```c
u8 coqui_submit_input(afl_state_t *afl, u8 *buf, u32 len) {
  coqui_batch_t *b = afl->coqui->pending;

  /* 8-byte-align the next write cursor */
  u32 off = (b->bytes_used + 7) & ~7u;

  if (off + len > BATCH_BYTE_BUDGET || b->n_inputs == afl->gpu_batch_size) {
    coqui_launch(afl, b);          /* async launch; flip to other half */
    b = afl->coqui->pending;
    off = 0;
  }

  memcpy(b->h_input_bytes + off, buf, len);
  b->h_offsets[b->n_inputs]    = off;
  b->h_input_lens[b->n_inputs] = len;
  b->n_inputs++;
  b->bytes_used = off + len;
  return 0;
}
```

Flush condition is dual: batch launches when either the slot count hits `gpu_batch_size`, or the next input would overflow `BATCH_BYTE_BUDGET`.

### 4.4 Kernel entry point signature

```cuda
extern "C" __global__ void __coqui_fuzz_kernel(
    const u8  *input_bytes,    // [BATCH_BYTE_BUDGET]  dense, 8-aligned inputs
    const u32 *offsets,        // [batch_size]
    const u32 *input_lens,     // [batch_size]          (0 = skip this slot)
    u8        *coverage,       // [batch_size][map_size]
    u32       *novelty,        // [batch_size/32] bitset
    u8        *virgin_map,     // [map_size]  shared, atomic OR
    status_t  *status);        // [batch_size]
```

Per-thread logic (pseudocode):

```
tid = blockIdx.x * blockDim.x + threadIdx.x;
if (input_lens[tid] == 0) return;

const u8 *my_input = input_bytes + offsets[tid];   // guaranteed 8-aligned
u32       my_len   = input_lens[tid];

/* Target-harness execution lives here. Emitted by coqui-cc. */
run_target(my_input, my_len, &coverage[tid * map_size], &status[tid]);

/* Novelty filter: atomic OR into virgin; if any bit flipped 0→1, set flag. */
if (coqui_merge_virgin(&coverage[tid * map_size], virgin_map, map_size)) {
    atomicOr(&novelty[tid >> 5], 1u << (tid & 31));
}
```

### 4.5 Fault code translation

```c
static u8 translate_gpu_status(const status_t *s) {
  if (s->asan_error)                     return FSRV_RUN_CRASH;
  if (s->ubsan_fatal)                    return FSRV_RUN_CRASH;
  if (s->signal && !ignored(s->signal))  return FSRV_RUN_CRASH;
  if (s->timeout_flag)                   return FSRV_RUN_TMOUT;
  return FSRV_RUN_OK;
}
```

The `ignored` set respects coqui's `--ignore-signal=abort` convention, carried through via an env var.

### 4.6 Stub backend (reference implementation)

The initial `afl-fuzz-coqui.c` ships with a stub that fakes the GPU using a traditional CPU forkserver. The stub:

- `coqui_init`: holds a CPU forkserver over the same target (expected as a regular AFL++ ELF rather than a cubin when in stub mode — gated by an env var `AFL_COQUI_STUB_TARGET`).
- `coqui_submit_input`: accumulates inputs into the packed batch buffer exactly as specified.
- Launch path (batch full or flush): iterates the batch, runs each input through `afl_fsrv_run_target` on the CPU forkserver, populates `d_coverage` and `d_novelty` manually based on real CPU coverage.
- `coqui_calibrate_one`: runs the single input through the CPU forkserver directly.

The stub makes cuAFL fully functional for AFL++-side integration work: it finds bugs, syncs with peers, writes queue files — just at CPU speeds plus batch overhead. When the real coqui backend lands, the stub is replaced behind the same contract.

### 4.7 Deliberately unspecified

The contract is silent on:
- Which LLVM passes run, what memory layout they emit.
- Per-thread stack, heap arena, and shadow memory scheme.
- How coverage is collected inside the kernel (PCguard, trace-pc, custom).
- How the virgin map is compared inside the kernel (per-thread atomic, per-warp reduction, etc.).
- Whether the kernel runs any CmpLog equivalent (it doesn't — CmpLog lives on the broker).

All of the above are coqui_mode internals. See §9.

## 5. Data Flow

### 5.1 Steady-state havoc loop

```
┌──────────────────────────────────────────────────────────────────────┐
│  CPU (host thread, afl-fuzz main loop)                               │
├──────────────────────────────────────────────────────────────────────┤
│  fuzz_one() picks queue entry q                                      │
│    └── havoc stage loop (stage_max iterations)                       │
│        ├── pick k mutations from the 40-op havoc table               │
│        ├── apply to out_buf                                          │
│        ├── common_fuzz_stuff(afl, out_buf, len)                      │
│        │   └── [gpu_mode branch]                                     │
│        │       coqui_submit_input(afl, out_buf, len)                 │
│        │         ├── memcpy into pending batch B_fill                │
│        │         ├── if batch full → launch B_fill on GPU (async)    │
│        │         └── flip B_fill ↔ B_exec pointers                   │
│        ├── restore out_buf                                           │
│        └── next iteration (fills next slot in B_fill)                │
│                                                                      │
│  (when havoc loop ends)                                              │
│    coqui_flush_batch(afl)  — drain partial batch, process results    │
└──────────────────────────────────────────────────────────────────────┘
```

### 5.2 Ping-pong overlap

```
Time →    T0          T1          T2          T3          T4
CPU fill  [  B0  ]    [  B1  ]    [  B0  ]    [  B1  ]    [  B0  ]
GPU exec              [  B0  ]    [  B1  ]    [  B0  ]    [  B1  ]
Host proc                         [  B0  ]    [  B1  ]    [  B0  ]
```

While the GPU runs B0, the CPU's havoc loop fills B1. When B0 completes, the host processes flagged inputs concurrently with the GPU starting on B1. Ping-pong keeps both sides near 100% utilized.

### 5.3 Post-batch flagged-input handling

```
1. DMA d_novelty → h_novelty             (async, ~1 KB for 8K inputs)
2. DMA d_status  → h_status              (async, ~128 KB for 8K inputs)
3. cuStreamSynchronize(stream)

4. For each set bit i in h_novelty:
     a. DMA d_coverage[i*map_size .. (i+1)*map_size] → afl->fsrv.trace_bits
     b. fault = translate_gpu_status(&h_status[i])
     c. input_buf = h_input_bytes + h_offsets[i]; len = h_input_lens[i]
     d. save_if_interesting(afl, input_buf, len, fault)
        // AFL++ handles the rest: has_new_bits, queue admission,
        // file write to out/<sync_id>/queue/, stats update

5. Zero h_novelty and h_status for next batch.
```

**Flagged ≠ queued.** The GPU says "this might be novel" by its own virgin map; AFL++'s `has_new_bits()` against the instance-local `virgin_bits` is the local arbiter. Most flagged inputs pass the local filter (GPU virgin map and CPU virgin_bits are closely correlated), but not all. Passing inputs are written to the queue dir; the broker picks them up via `sync_fuzzers()`.

### 5.4 Stage boundaries and `fuzz_one` exit

- At the end of each havoc/splice stage, `coqui_flush_batch()` forces a launch on any partial batch and waits for all pending work to drain. This ensures `afl->queued_items` reflects every mutation the stage generated.
- At the end of `fuzz_one`, a final flush drains in-flight work before the next queue entry is selected.

### 5.5 Sync-in calibration (single-input path)

When `sync_fuzzers()` discovers a new file in another instance's `out/*/queue/`, cuAFL calls `coqui_calibrate_one()`. This submits a one-input batch, synchronous launch, waits for completion, populates `trace_bits` directly. A 8192-wide kernel where 8191 threads idle on `input_lens == 0` is wasteful of silicon but costs ~100 µs; calibration is <0.1% of runtime.

### 5.6 Rough timing estimates

For a medium-complexity target (e.g., libxml2 parse) at 4 KB max input, 8K batch size:

- Kernel time per batch: ~4-10 ms
- Host fill time per batch: ~0.5-1 ms
- Effective throughput: ~800K-2M exec/s (kernel-time-dominated with ping-pong)
- Flagged-rate: 0.1-3% → 10-250 flagged per batch
- Post-batch host processing: 0.2-1 ms

Broker-side CPU AFL++ runs ~2-10K exec/s on the same target. The ratio of 100-1000× justifies the architecture.

## 6. Error Handling

### 6.1 Contract-level errors (stub and real must handle)

| Error | When | Handling |
|-------|------|----------|
| `coqui_init` fails | startup | `FATAL` with specific cause; process exits |
| Input exceeds `max_length` | `coqui_submit_input` | truncate to `max_length`, continue (AFL++ convention) |
| Batch full *and* byte budget full | `coqui_submit_input` | launch current, place input in fresh batch |
| `max_length == 0` | `coqui_init` | warn, use `COQUI_MAX_INPUT_DEFAULT` (4096) |
| Input exceeds single-input byte budget alone | `coqui_submit_input` | skip input, increment `afl->coqui->oversized_count`, return 0 |

### 6.2 Real-GPU errors (specified for future implementation)

| Error | When | Handling |
|-------|------|----------|
| Kernel launch fails | `cuLaunchKernel` error | treat entire batch as skipped; log once per distinct error code; continue |
| Kernel timeout | watchdog event | cancel; mark all inputs as `FSRV_RUN_TMOUT`; most fail novelty check and are discarded |
| Device OOM mid-run | `cuMemAlloc` error | `FATAL` — no graceful recovery without breaking mid-campaign invariants |
| CUDA context lost | any CUDA call returns `CUDA_ERROR_CONTEXT_IS_DESTROYED` etc. | `FATAL` — user relaunches; sync dir preserves corpus |
| Cubin arch mismatch | `coqui_init` | `FATAL` with clear message |
| `coqui_calibrate_one` timeout | synced-in input too slow | mark `FSRV_RUN_TMOUT`; AFL++'s sync path handles |

### 6.3 Signal handling

- SIGINT sets `afl->stop_soon = 1`. Current batch drains, `coqui_flush_batch` runs, `coqui_shutdown` releases resources, normal exit.
- SIGCHLD is a no-op for coqui_mode (no child processes).
- SIGUSR1 (skip request) is per-queue-entry, honored at stage boundaries.

### 6.4 Data-integrity edge cases

- **Orphaned crashes**: If the cuAFL instance dies mid-batch with unwritten crashes, those are lost. The broker would re-discover them if the underlying fault is deterministic. Atomic `rename(tmpfile, crashfile)` via AFL++'s existing `write_crash()` logic mitigates partial writes.
- **Unflushed partial batch on crash**: In-flight mutations lost. Acceptable — AFL++ has similar semantics around mid-mutation crashes.
- **Graceful restart**: Re-invoking with the same `--coqui gpu0 -o out/` reads the instance's queue dir and resumes. Sync dir is source of truth.

## 7. Phased Rollout

| Phase | Scope |
|-------|-------|
| **P1: Plumbing** | `--coqui` CLI flag, `afl_state_t`/`afl_forkserver_t` fields, getopt integration, help text. No behavior yet — flag parses but does nothing. |
| **P2: Contract + stub** | `afl-fuzz-coqui.{c,h}` with the five contract functions implemented as the CPU-forkserver stub. |
| **P3: Core wiring** | `common_fuzz_stuff` gpu branch; `fuzz_one` det/cmplog gating; `afl_fsrv_start` short-circuit; calibration routing in `afl-fuzz-init.c`. |
| **P4: Heterogeneous mesh manual validation** | Run `-M main` + `--coqui gpu0` on a bundled AFL++ canary target; confirm findings propagate. |
| **P5: Ship AFL++ side** | Tag on `cuAFL` branch. Open the follow-up brainstorm for coqui internals. |

Each phase is a merge-sized commit or PR. P1-P4 land on a feature branch; P5 merges to `cuAFL` branch main.

## 8. Naming Conventions

- **cuAFL**: the AFL++ fork (this project).
- **coqui_mode**: the executor mode (boolean on `afl_forkserver_t`; subdirectory `coqui_mode/`). Named after the underlying research technology, matching AFL++'s pattern for `qemu_mode`/`frida_mode`/`nyx_mode`.
- **coqui-cc**: the compiler driver in `coqui_mode/` that produces cubins from C/C++ sources, matching AFL++'s `afl-clang-fast` convention.
- **`--coqui <sync_id>`**: the CLI flag on `afl-fuzz` to launch a GPU secondary.

## 9. Out of Scope — Deferred to Follow-Up Brainstorm

The internals of `coqui_mode/` are intentionally not specified here:

- LLVM pass set and transform behaviors
- GPU runtime (coverage, asan, ubsan, allocators, mutation kernel if any)
- Per-thread memory layout, shadow memory scheme
- `coqui-cc` compiler driver internals
- Virgin-map merge strategy on GPU
- Device-side CmpLog (if attempted at all)

These will be covered in a follow-up brainstorm. The Section 4 contract is the seam: AFL++-side implementation depends only on the contract, not on the coqui_mode internals. Either side can iterate independently.
