# cuAFL Coqui Internals — Design Spec

**Status:** Draft
**Date:** 2026-04-18
**Scope:** The GPU-side of cuAFL — the `coqui-cc` compiler, LLVM pass plugin, device-side runtime, GPU kernel interface, and host-side CUDA launcher that implements the cuAFL contract from `docs/superpowers/specs/2026-04-18-cuafl-gpu-backend-design.md`.

This design supersedes the hollow-stub approach of the earlier cuAFL spec by defining the real implementation behind the 5-function contract. It also amends specific parts of that spec (per §11 of this document).

## 1. Overview

### 1.1 What this spec covers

1. **The `coqui-cc` compiler driver** — invokes clang with NVPTX target, runs the LLVM pass plugin, links the device runtime, calls ptxas. Derived from coqui's existing driver at `/coqui/driver/coqui`.
2. **The LLVM pass plugin** — a set of IR transforms that produce NVPTX-legal, fuzzing-instrumented bitcode. Includes a documented port-on-demand process for adding coqui's additional transforms when error-driven needs surface.
3. **The per-thread GPU memory layout** — stack-region ordering and global-memory regions, with ASan heap-shadow integrated.
4. **The AFL-compatible coverage scheme** — hash-based edge indexing with bucketing, entirely device-side.
5. **The GPU device runtime** — C files compiled into every cubin: coverage, heap-only ASan, freelist allocator, tid helpers.
6. **The kernel interface** — signature, buffer layouts, status struct, novelty bitmap format.
7. **The host-side CUDA launcher** — real implementation of the 5-function cuAFL contract, reusing AFL++'s standard CPU forkserver alongside GPU batch execution.
8. **The validation strategy** — cjson as minimum-viable probe, libpng as stress test; collaborative port-on-demand loop.

### 1.2 Core architectural principles

1. **NVPTX from the start.** `clang --target=nvptx64-nvidia-cuda` emits NVPTX IR from the first invocation. No host-to-nvptx retargeting. This eliminates several coqui transforms and lets us rely on LLVM's native NVPTX handling.
2. **Port on demand.** The day-1 pass set is the minimum needed to produce a working cubin from cjson. Additional transforms from coqui are ported as error-driven needs surface, collaboratively (human-in-the-loop).
3. **Hard-fail on every unsupported feature.** No silent codegen of incorrect behavior. Gatekeeping passes detect setjmp/longjmp, pthreads, non-trivial inline asm, unsupported intrinsics, and unresolved externals. All errors loud and specific.
4. **AFL contract fidelity.** Coverage uses AFL's hash formula with bucketing so the broker's re-verification is semantically comparable. The GPU maintains its own virgin map device-side; only the per-input novelty bit crosses PCIe.
5. **Reuse existing AFL++ infrastructure.** The `-G` instance runs a standard CPU forkserver over the same target the broker runs, via `afl->fsrv` and `afl_fsrv_run_target`. GPU flagged inputs go through the CPU forkserver for real `trace_bits` before `save_if_interesting`. No bypass of AFL internals.

### 1.3 Test targets (probe-driven development)

- **cjson** (`cjson_read_fuzzer`): pure C, minimal deps, no float math, no indirect callbacks. The minimum-viable probe. If cjson cannot compile with day-1 passes, the core design is broken.
- **libpng** (`libpng_read_fuzzer`): heavy C, integer+float math, variadic error reporting, indirect callbacks, zlib dependency, structured state. Stresses nearly every transform category. If libpng works, most realistic targets will.

### 1.4 Deployment

Same as established in the cuAFL spec: `--coqui <sync_id>` flag, cubin companion to the ELF target (resolved by convention or via `AFL_COQUI_CUBIN` env var), co-existing with `-M main` and optional `-S cpu*` secondaries over the sync dir.

## 2. `coqui-cc` Compiler Driver

### 2.1 Invocation flow

```
clang --target=nvptx64-nvidia-cuda -O2 -emit-llvm -c ... harness.c -o harness.bc
clang --target=nvptx64-nvidia-cuda -O2 -emit-llvm -c ... lib1.c   -o lib1.bc
...
llvm-link harness.bc lib1.bc ... runtime.bc -o linked.bc
opt -load-pass-plugin coqui_pass_plugin.so -coqui-link linked.bc -o transformed.bc
llc --march=nvptx64 -mcpu=sm_<XY> transformed.bc -o out.ptx
ptxas -arch=sm_<XY> -O1 out.ptx -o out.cubin
```

**Key differences from coqui's pipeline:**

1. **`clang --target=nvptx64-nvidia-cuda` from the first invocation** (not `--target=<host>` + later retarget). This is change #1 from the brainstorm; eliminates `TargetTransform` and likely `AddressSpacesTransform`.
2. **`runtime.bc` is linked with user bitcode** before the pass plugin runs (same as coqui's pattern).
3. **Pass plugin runs a smaller set** (see §3).
4. **No separate oracle/test modes** — cuAFL has no oracle tracing; test infrastructure is a separate concern.
5. **No CPU-binary companion emission** — coqui built both GPU and CPU artifacts; cuAFL's CPU build is produced separately by the user (via `afl-clang-fast`) and lives alongside the cubin for AFL++'s forkserver.

### 2.2 Compile flags (per-source file)

```
clang --target=nvptx64-nvidia-cuda
      -O2
      -emit-llvm
      -c
      -g0
      -fno-stack-protector
      -fno-pic
      -I<include dirs>
      <source>.c -o <source>.bc
```

- `-fno-stack-protector`: NVPTX has no `__stack_chk_fail`.
- `-fno-pic`: prevents `RelLookupTableConverterPass` from creating self-referencing lookup tables that confuse NVPTX codegen.
- `-O2`: clang emits `@llvm.memcpy`/`@llvm.memset` intrinsics, which the NVPTX backend lowers natively.
- `-g0`: debug info not useful in device code.

### 2.3 Runtime bitcode

Device-side runtime files live in `coqui_mode/runtime/` (see §6). At coqui-cc install time, each `.c` is compiled to `.bc`, all `.bc` linked into `runtime.bc`, and installed to `$PREFIX/lib/coqui-cc/runtime.bc`. Per-user compile invocations link against this prebuilt `runtime.bc`.

### 2.4 CLI shape

```
coqui-cc [options] <sources>... -o <output>

Options:
  -o <name>              Output name (produces <name> and <name>.cubin)
  -arch sm_XY            Target GPU architecture (default: auto-detect)
  --device N             GPU device index for arch auto-detection
  --stack-size <bytes>   Real stack budget per thread (default: 16384)
  --slab-pool-size <b>   Optional shared slab pool size (0 = disabled)
  -I <dir>               Include directory
  -D <def>               Preprocessor define
  -emit-ptx              Stop after PTX generation (don't produce .cubin)
  -emit-bc               Stop after IR transform (don't produce .ptx)
  -v                     Verbose
```

User-supplied `--stack-size` is baked into the cubin as a `.conf` sidecar (`<output>.conf`) so the host launcher can read it at runtime.

**Modes dropped from coqui's driver:** `--fuzz` (the only mode — no need to toggle), `--test`, `--oracle`, `--heap-size` (heap is derived, §4), host-binary emission.

## 3. LLVM Pass Pipeline

### 3.1 Day-1 pass set (10 passes)

**Core transforms (6):**

| Pass | Purpose | Origin |
|------|---------|--------|
| `FuzzEntry` | Rename `LLVMFuzzerTestOneInput` → `__coqui_fuzz_execute`; emit kernel entry `__coqui_fuzz_kernel(...)` matching §7 signature | Ported from coqui, modified to match new kernel signature |
| `StaticGlobals` | Per-thread localization of writable globals via `pool + tid * size + offset` addressing | Ported from coqui unchanged; targets the new global pool region |
| `MemoryLayout` (new) | Emit per-thread allocas for coverage, heap, shadow regions; provide base-pointer helpers | Written fresh — see §4 |
| `Coverage` (new) | Insert AFL hash-based edge instrumentation at each BB entry | Written fresh — see §5 |
| `Heap` | Replace `malloc`/`free`/`calloc`/`realloc` with `__coqui_*` runtime functions | Ported from coqui unchanged |
| `Asan` (heap-only) | Instrument heap loads/stores with shadow-byte fast-path check; RAUW `__coqui_malloc` → `__coqui_asan_malloc` | Ported from coqui, modified for new heap/shadow region layout |

**Gatekeeping transforms (4):**

| Pass | Purpose | Origin |
|------|---------|--------|
| `InlineAsmReject` | Strip benign compiler-barrier asm (`"memory"` clobbers); FATAL on non-trivial asm with source location | Ported from coqui's `InlineAsmTransform` (rejection half) |
| `LibcReject` (blocklist-only) | FATAL on setjmp/longjmp, pthread_* (except pthread_once), known-unsupported libc symbols. Does NOT provide replacements | Ported from coqui's `LibcTransform` (rejection half extracted) |
| `IntrinsicReject` | FATAL on unhandled LLVM intrinsics (e.g., `llvm.setjmp`, `llvm.va_*` unless the Variadic transform has been ported) | Written fresh |
| `ExternalSymbolGatekeeper` | Runs LAST. Scans linked module for any unresolved external call not on the allowlist. FATAL with symbol name + containing function | Written fresh |

**Allowlist seeding for `ExternalSymbolGatekeeper`:** LLVM-intrinsic-lowered operations that the NVPTX backend handles natively (`memcpy`, `memset`, `memmove` via `@llvm.memcpy`/`@llvm.memset` intrinsics) are whitelisted from the start. These never appear as external symbols in post-backend code — the NVPTX backend emits inline byte-stride loops.

### 3.2 Pass order

1. `InlineAsmReject`
2. `LibcReject`
3. `IntrinsicReject`
4. `FuzzEntry`
5. `Heap`
6. `StaticGlobals`
7. `MemoryLayout`
8. `Coverage`
9. `Asan`
10. `ExternalSymbolGatekeeper`

### 3.3 Transforms explicitly not ported day-1

| Pass | Trigger | Likely target |
|------|---------|---------------|
| `Math` | `llvm.pow`/`llvm.sqrt`/etc. (NVPTX backend can't lower non-trivial math natively) | libpng. Alternative: libdevice linking (§3.6) |
| `Complex` | Complex-number math calling conv | Rare (signal processing) |
| `Libc` (replacement half) | External references to string functions (strlen, strcmp, etc.) | Most non-trivial targets |
| `Cpp` | `operator new`/`operator delete` | C++ harnesses |
| `Printf` | Variadic printf call sites | Debug-heavy code |
| `Sprintf` | Variadic sprintf/snprintf | String-producing code |
| `Variadic` | Remaining variadic functions | Custom logging |
| `Syscall` | Musl internal functions | When musl is linked |
| `Relocations` | Circular global initializers | Rare |
| `Globals` | `@llvm.global_ctors` | C++ with static constructors |
| `IndirectCalls` | Perf tuning (NVPTX native indirect is likely fine initially) | Performance-hot libraries |
| `Align` | `CUDA_ERROR_MISALIGNED_ADDRESS` at runtime | Likely not needed with NVPTX-from-start |

### 3.4 Transforms explicitly eliminated (never ported)

- `Target` — NVPTX-from-start removes the need.
- `AddressSpaces` — clang with NVPTX target places globals in addr space 1 automatically (to be verified; re-add if needed).
- `LineTrace` — oracle tracing, not in cuAFL scope.
- `SancovCount` — replaced by new `Coverage` transform.

### 3.5 Collaborative port-on-demand process

**Critical: ports are human-in-the-loop, not autonomous.** When the gatekeepers fire or runtime errors surface:

1. **Error surfaces** (compile-time gatekeeper FATAL, or runtime CUDA error).
2. **I propose a port approach** — either copy the exact coqui source, or propose an alternative (e.g., libdevice linking for math).
3. **User approves or redirects** — "port as-is," "try alternative first," "skip — compile target differently."
4. **I execute the approved port** — code changes, runtime file addition, gatekeeper allowlist update.
5. **Continue until the target builds and runs.**

Each port gets a log entry in `docs/coqui_port_log.md` for reproducibility (§9.4).

### 3.6 Empirical findings on NVPTX native support

Verified experimentally (`clang --target=nvptx64-nvidia-cuda -O2 -emit-llvm` + `llc -march=nvptx64 -mcpu=sm_75`):

**Tier 1 (no port needed):** `memcpy`, `memset`, `memmove` — emitted as `@llvm.memcpy`/`@llvm.memset` intrinsics at -O2, lowered inline by NVPTX backend to byte-stride loops. No external symbols.

**Tier 2 (port required):** `strlen`, `strcmp`, `strncmp`, `strcpy`, `strchr`, etc. — no LLVM intrinsic exists, clang emits `tail call @strlen`, NVPTX backend leaves as `.extern .func strlen`.

**Tier 3 (port with two options):** `sqrt`, `pow`, `log`, `sin`, etc. — emitted as `@llvm.sqrt.f64`/`@llvm.pow.f64` intrinsics, but NVPTX backend cannot lower non-trivial math (verified: `llc` aborts with `Cannot select: fpow`). Two port options:
- **(a) Port coqui's `MathTransform`** — translates `@llvm.pow.f64` → `@__coqui_pow` with device implementations in `coqui_math.c`.
- **(b) Link NVIDIA's libdevice** — `libdevice.10.bc` (ships with CUDA toolkit) provides `__nv_pow`, `__nv_log`, etc. Requires a small intrinsic-renamer pass (maps `@llvm.pow.f64` → `@__nv_pow` pre-llc).

Recommendation when math is needed: try (b) first (less code, production-quality NVIDIA implementations). Fall back to (a) if libdevice has integration issues.

## 4. Per-Thread Memory Layout

### 4.1 CUDA per-thread stack budget

On sm_75+ hardware, each thread has up to 512 KB of `.local` memory (hardware cap). The CUDA driver's `cuCtxSetLimit(CU_LIMIT_STACK_SIZE, N)` sets the per-thread limit; `ptxas` places all `alloca`s and function-call frames there.

### 4.2 Stack layout (priority order)

```
Per-thread .local memory (512 KB on sm_75+):

  ┌─────────────────────────────────────┐
  │ Coverage map           (64 KB)      │  Fixed; AFL convention
  ├─────────────────────────────────────┤
  │ Heap freelist          (H)          │  Derived from budget formula
  ├─────────────────────────────────────┤
  │ ASan shadow            (H/8)        │  Derived: heap / 8
  ├─────────────────────────────────────┤
  │ Real stack             (S)          │  User-specified via --stack-size
  │                                     │  ↓ normal call frames during exec
  └─────────────────────────────────────┘
  Total = 64K + H + H/8 + S
  remaining = 512K - 64K - S
  H = remaining * 8 / 9
  shadow = remaining * 1 / 9
```

### 4.3 Region sizing

| Region | Sizing | Default | User control |
|--------|--------|---------|--------------|
| Coverage map | Fixed | 64 KB | None |
| Real stack `S` | User-specified, compile-time | 16 KB | `coqui-cc --stack-size <bytes>` |
| Heap `H` | Derived | `(512K - 64K - S) * 8/9` ≈ 384 KB at default | Not user-controlled |
| ASan shadow | Derived | `H / 8` ≈ 48 KB at default | Not user-controlled |

**Validation at compile time and host init:**
- `--stack-size + 64KB > 512KB` → FATAL with max-value hint.
- Derived heap < minimum (e.g., 8 KB) → FATAL.
- Runtime: `CU_LIMIT_STACK_SIZE > hardware_cap` → FATAL with actual cap.

### 4.4 Region base-pointer access

The `MemoryLayout` transform emits the three region allocas at kernel entry and exposes inline helpers:

```c
__device__ u8  *__coqui_cov_base(void);
__device__ u8  *__coqui_heap_base(void);
__device__ u8  *__coqui_shadow_base(void);
__device__ u32  __coqui_heap_size(void);
```

Implementation mechanism — how other transforms access the per-thread pointers without passing them through every function call — is an implementation detail of `MemoryLayout`. Likely approach: allocas are hoisted into a per-thread state struct stored in a `__device__` variable addressed by tid (or equivalent).

### 4.5 Global memory regions

**Thread-localized statics pool** (`StaticGlobals` output):
- Layout: `pool_base + tid * per_thread_globals_size + var_offset`
- Size: computed at compile time from total writable-globals size
- Allocated by host at `coqui_init`: `cuMemAlloc(batch_size * per_thread_globals_size)`.

**Optional shared heap slab pool** (inherited from coqui verbatim — see `/coqui/runtime/coqui_fuzz_slab.c`):
- Single `cuMemAlloc`'d pool; 4 KB slabs; 13 size buckets (16 B to ~64 KB); 1 MB per-alloc cap.
- Three-tier allocation: block-local shared-memory atomic → global atomic → Treiber free-stack recycling.
- Per-thread control at `pool[tid * 32]`: free_head (abs ptr), current_heap, heap_limit, bump_top.
- Gated by `--slab-pool-size <bytes>`; 0 = disabled (slab code not linked).
- Port status: **reuse verbatim** — do not refactor.

### 4.6 Overflow behavior

| Overflow | Detection | Handling |
|----------|-----------|----------|
| Stack overflow | Host pre-launch check of `CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES` vs. budget; runtime: kernel fails with CUDA error | FATAL pre-launch with hint; runtime FATAL with raw CUDA error + "try bumping --stack-size" hint. No device-side canary (perf hit not justified for v1). |
| Heap OOM (freelist exhausted, no slab) | `__coqui_malloc` trap | Thread status marked; host treats input as FSRV_RUN_TMOUT |
| Heap OOM with slab enabled | Slab fallback; trap if slab also full | Same |
| `--stack-size` configuration invalid | Compile-time in `MemoryLayout` | FATAL with max value |
| Statics pool overflow | Compile-time in `StaticGlobals` | FATAL |

### 4.7 Alignment

With NVPTX-from-start, clang should emit IR with NVPTX-appropriate alignment. The `Align` transform is NOT ported day-1; if `CUDA_ERROR_MISALIGNED_ADDRESS` surfaces, port via the process in §3.5.

Region allocas declare natural alignments (`coverage: align 8`, `heap: align 16`, `shadow: align 8`).

## 5. Coverage Scheme

### 5.1 Overview

Coverage lives entirely device-side. The kernel instruments each basic-block entry with AFL's hash formula, buckets each thread's map after target execution, and compares against a shared device-side virgin map via atomic OR. Only the per-input novelty bit crosses PCIe.

### 5.2 AFL formula at basic-block entry

Inserted by the `Coverage` pass at each BB entry (after PHIs), with `<cur_loc>` as a compile-time-assigned random `u16`:

```llvm
; pseudo-IR
%prev       = load u32, ptr @__coqui_prev_loc       ; per-thread
%idx        = xor u32 %prev, <cur_loc>
%cov_base   = call ptr @__coqui_cov_base()          ; per-thread
%cov_ptr    = getelementptr u8, ptr %cov_base, u32 %idx
%count      = load u8, ptr %cov_ptr
%new_count  = add u8 %count, 1                      ; clang emits saturating
store u8 %new_count, ptr %cov_ptr
%next_prev  = lshr u32 <cur_loc>, 1
store u32 %next_prev, ptr @__coqui_prev_loc
```

`<cur_loc>` constants are generated deterministically (seeded by `hash(module_name)`) so recompilation produces identical values.

### 5.3 Per-thread state

- `prev_loc`: `u32`, per-thread, allocated by `MemoryLayout` in addr space 5 (`.local`).
- `cov_map`: 65536 bytes per thread, part of the per-thread stack (§4.2). **Hardcoded size — not configurable.**
- `virgin_map`: 65536 bytes, device-global, shared across all threads and persistent across batches.

### 5.4 Post-execution sequence

After `__coqui_fuzz_execute` returns, the kernel entry template runs:

```c
// 1. Bucket
__coqui_classify_counts(cov_map);

// 2. Compare against shared virgin, atomic-OR, detect novelty
bool novel = false;
for (u32 i = 0; i < 65536 / 8; i++) {
    u64 mine = ((u64*)cov_map)[i];
    if (mine == 0) continue;
    u64 was  = atomicOr(&((u64*)virgin_map)[i], mine);
    if (mine & ~was) novel = true;
}

// 3. Set novelty bit
if (novel) {
    u32 tid = __coqui_fuzz_tid();
    atomicOr(&novelty_bitmap[tid >> 5], 1u << (tid & 31));
}
```

### 5.5 Bucketing (device-side)

256-entry lookup table placed in `__device__ __constant__` memory:

```c
__device__ __constant__ u8 __coqui_count_class_lookup[256] = {
    [0]   = 0x00,
    [1]   = 0x01,
    [2]   = 0x02,
    [3]   = 0x04,
    [4 ... 7]     = 0x08,
    [8 ... 15]    = 0x10,
    [16 ... 31]   = 0x20,
    [32 ... 127]  = 0x40,
    [128 ... 255] = 0x80,
};
```

`__coqui_classify_counts(u8 *map)` walks 8-byte-stride with early-exit for zero chunks; applies the lookup byte-wise inside each non-zero chunk. ~16K 8-byte chunks per 64 KB map; for typical inputs most chunks are zero.

### 5.6 Host-visible contract (what crosses PCIe)

Per batch:
- `d_novelty_bitmap` → `h_novelty_bitmap`: 8192 bits = 1 KB DMA.
- `d_status` → `h_status`: 8192 × sizeof(coqui_status_t) = 128 KB DMA.

Per-input coverage maps (64 KB × 8192 = 512 MB device memory) stay device-local. **The host never DMAs coverage.**

### 5.7 Virgin map lifetime

One shared device-side buffer per cuAFL process, zeroed at `coqui_init`. Grows monotonically during the campaign; never cleared. On instance restart, resets — broker's corpus quickly re-populates it via sync.

### 5.8 The `Coverage` transform

**What it does:**
1. Enumerates every basic block in the linked module.
2. Assigns each BB a deterministic `u16` constant.
3. Inserts the 6-instruction sequence at each BB entry (after PHIs).
4. Skips BBs inside `__coqui_*`-prefixed functions (runtime helpers don't pollute the map).

**Runtime helpers created:**
- `__coqui_cov_base()`, `__coqui_prev_loc_ptr()` — both `static inline`, resolved to `.local` addresses set up by `MemoryLayout`.

Bucketing + virgin-compare sequence lives in the kernel-entry template emitted by `FuzzEntry` — runs after `Coverage`, not inserted by `Coverage` itself.

### 5.9 Amendment to cuAFL spec

`docs/superpowers/specs/2026-04-18-cuafl-gpu-backend-design.md` §4.2 and §5.3 describe `d_coverage [batch_size × map_size]` and per-flagged-input coverage DMA — both vanish from the host-side contract under this design. See §11.

## 6. GPU Device Runtime

### 6.1 Day-1 runtime files (5 files)

All in `coqui_mode/runtime/`. Each compiled with `clang --target=nvptx64-nvidia-cuda -O2 -ffreestanding -emit-llvm -c <file>.c -o <file>.bc`; all `.bc` linked into a single `runtime.bc`.

**`coqui_runtime.h`** — common header:
- `coqui_status_t` struct
- All prototypes for inter-runtime calls
- Constants (`COQUI_COV_MAP_SIZE = 65536`, class values)
- Inline helpers (e.g., tid via PTX asm)

**`coqui_runtime.c`** — core utilities:
- `__coqui_fuzz_tid()` — linear thread ID via inline PTX
- `__coqui_trap()` / `__coqui_exit()` — PTX `trap;` / `exit;` wrappers
- `__coqui_write_status(...)` — writes to per-thread status slot
- Device-side `memset`/`memcpy` fallbacks for runtime's own use (not user-facing)
- Stubs for functions the gatekeepers let through: `atexit`, `pthread_once`, etc.

**`coqui_coverage.c`**:
- `__coqui_count_class_lookup[256]` in `__device__ __constant__`
- `__coqui_classify_counts(u8 *map)` — inlined byte-lookup with 8-byte-stride zero-skip
- `__coqui_virgin_compare_and_flag(...)` — post-exec sweep

**`coqui_memory.c`** — per-thread heap allocator:
- `__coqui_malloc`, `__coqui_free`, `__coqui_calloc`, `__coqui_realloc`
- `__coqui_heap_base()`, `__coqui_shadow_base()`, `__coqui_cov_base()`, `__coqui_heap_size()` — read from `MemoryLayout`-emitted allocas
- Freelist first-fit with 8-iter insertion-sort on free
- Size header `[size:4B][padding:4B][user_data]`; ASan adds red zones

**`coqui_asan.c`** — heap-only ASan:
- `__coqui_asan_malloc`, `__coqui_asan_free`
- `__coqui_asan_check_load_1/2/4/8(ptr)`, `__coqui_asan_check_store_1/2/4/8(ptr)` — size-specialized. Fast path: `if (ptr not in heap) return;`
- Shadow mgmt: `asan_shadow_for(ptr)` returns the shadow byte address.
- `__coqui_asan_register_slab(...)` — stub day-1; implemented when slab runtime is linked.

### 6.2 Port-on-demand runtime files

| File | Trigger | Notes |
|------|---------|-------|
| `coqui_slab.c` | Target needs `--slab-pool-size > 0` (heap OOM with default) | Port verbatim from coqui |
| `coqui_printf.c` / `.inc` | Target calls printf family | Port from coqui + `Printf`/`Sprintf`/`Variadic` transforms |
| `coqui_libc.c` | Target calls string functions (strlen etc.) | Port function-by-function as gatekeeper flags them |
| `coqui_math.c` | If choosing coqui's `MathTransform` over libdevice linking | Port coqui source |
| `coqui_ubsan.c` | If UBSan is wanted (not day-1) | Port coqui source + UBSan transform |

### 6.3 Files explicitly not ported

| File | Reason |
|------|--------|
| `coqui_mutate.c` | GPU-side havoc; cuAFL does mutation on CPU |
| `coqui_trace.c` | Oracle tracing, out of scope |
| `coqui_cmplog_rt.c` | CmpLog lives on broker |
| `coqui_sancov.c` | Replaced by hash-based `coqui_coverage.c` |

### 6.4 Install-time build wiring

```bash
# At coqui-cc install time:
for f in coqui_runtime coqui_coverage coqui_memory coqui_asan; do
  clang --target=nvptx64-nvidia-cuda --emit-llvm -O2 -ffreestanding \
        -c coqui_mode/runtime/$f.c -o $f.bc
done
llvm-link coqui_runtime.bc coqui_coverage.bc coqui_memory.bc coqui_asan.bc \
          -o runtime.bc
install runtime.bc $PREFIX/lib/coqui-cc/runtime.bc
```

Port-on-demand files get appended to the link list at install time.

### 6.5 Runtime size budget

Day-1 runtime: ~850 LOC of C, compiles to ~10-20 KB of bitcode. Single-digit-percent of typical cubin size (cjson ~700 KB).

## 7. Kernel Interface

### 7.1 Kernel entry signature

```cuda
extern "C" __global__ void __coqui_fuzz_kernel(
    const u8             *input_bytes,   // [byte_budget]  dense, 8-aligned inputs
    const u32            *offsets,       // [batch_size]
    const u32            *input_lens,    // [batch_size]   (0 = skip this slot)
    u32                  *novelty,       // [batch_size/32] bitset
    coqui_status_t       *status);       // [batch_size]
```

**Changes from the cuAFL spec's original signature:**
- Dropped `coverage` parameter (device-local alloca).
- Dropped `virgin_map` parameter (device global, accessed by symbol via `cuModuleGetGlobal`).

### 7.2 Per-thread kernel logic (emitted by `FuzzEntry`)

```cuda
__global__ void __coqui_fuzz_kernel(...) {
    u32 tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (input_lens[tid] == 0) return;

    __coqui_status_set_phase(tid, PHASE_START);

    /* MemoryLayout-emitted allocas */
    u8 cov_map[65536];
    u8 heap[H];
    u8 shadow[H / 8];
    __coqui_region_setup(cov_map, heap, shadow, H);

    /* Target execution */
    const u8 *my_input = input_bytes + offsets[tid];
    u32 my_len = input_lens[tid];
    __coqui_fuzz_execute(my_input, my_len);

    __coqui_status_set_phase(tid, PHASE_BUCKETING);
    __coqui_classify_counts(cov_map);
    __coqui_status_set_phase(tid, PHASE_VIRGIN_CMP);
    __coqui_virgin_compare_and_flag(cov_map, __coqui_virgin_map, novelty);

    __coqui_status_set_phase(tid, PHASE_COMPLETE);
}
```

### 7.3 Buffer layout

**Host-side pinned buffers** (two copies for ping-pong):

| Buffer | Size | Direction |
|--------|------|-----------|
| `h_input_bytes` | byte_budget | H→D |
| `h_offsets` | batch_size × 4 | H→D |
| `h_input_lens` | batch_size × 4 | H→D |
| `h_novelty` | batch_size / 8 | D→H |
| `h_status` | batch_size × sizeof(coqui_status_t) | D→H |

**Device DMA mirrors** — one per host buffer, same sizes, two copies for ping-pong.

**Device-only persistent:**
- `__coqui_virgin_map` — 64 KB, accessed via `cuModuleGetGlobal`
- `__coqui_global_statics_pool` — `batch_size × per_thread_globals_size`
- `__coqui_slab_pool` (optional) — `--slab-pool-size`

**Per-thread kernel-local allocas** — coverage map, heap, shadow, prev_loc, function call frames.

### 7.4 `coqui_status_t` struct

```c
typedef struct coqui_status {
  u8  phase;         // 0-7, matches PHASE_* enum
  u8  signal;        // POSIX signal number or 0
  u8  asan_error;    // non-zero if ASan check tripped
  u8  ubsan_fatal;   // future: UBSan non-recoverable
  u32 _reserved0;
  u64 _reserved1;    // future: timing, edge count, etc.
} coqui_status_t;   // 16 bytes, aligned 16
```

**Phase enum:**
```
0  PHASE_EMPTY
1  PHASE_START
2  PHASE_EXECUTING
3  PHASE_EXITED
4  PHASE_BUCKETING
5  PHASE_VIRGIN_CMP
6  PHASE_COMPLETE
7  PHASE_RESERVED
```

### 7.5 Launch parameters

- Grid: `batch_size / block_size` (64 blocks at default 8K / 128)
- Block: 128 threads
- Shared memory per block: 0 (slab allocator declares its own when linked)
- Stream: one CUDA stream per ping-pong half

### 7.6 Launch sequence per batch

```c
cuMemcpyHtoDAsync(d_input_bytes, h_input_bytes, byte_budget, stream);
cuMemcpyHtoDAsync(d_offsets,     h_offsets,     batch_size * 4, stream);
cuMemcpyHtoDAsync(d_input_lens,  h_input_lens,  batch_size * 4, stream);
cuMemsetD32Async(d_novelty, 0, batch_size / 32, stream);
cuMemsetD8Async (d_status,  0, batch_size * sizeof(coqui_status_t), stream);

void *args[] = {&d_input_bytes, &d_offsets, &d_input_lens, &d_novelty, &d_status};
cuLaunchKernel(cu_kernel,
               batch_size / 128, 1, 1,   // grid
               128, 1, 1,                 // block
               0, stream,
               args, NULL);

cuMemcpyDtoHAsync(h_novelty, d_novelty, batch_size / 8, stream);
cuMemcpyDtoHAsync(h_status,  d_status,  batch_size * sizeof(coqui_status_t), stream);
```

## 8. Host-Side CUDA Launcher

### 8.1 Coexistence model

The `-G` instance runs TWO executors:
1. **Standard AFL CPU forkserver** at `afl->fsrv` — over the ELF target. Used for calibration, trim, sync-in, and post-batch flagged-input re-execution.
2. **GPU batch executor** at `afl->coqui` — over the cubin. Used for havoc/splice via `common_fuzz_stuff` batch sink.

AFL's existing machinery (`afl_fsrv_init`, `afl_fsrv_start`, `afl_fsrv_run_target`, `write_to_testcase`, `save_if_interesting`) is reused verbatim. No short-circuits, no bypasses.

### 8.2 Target binary resolution

- Trailing positional arg to `afl-fuzz --coqui gpu0 -- <target>` is the **ELF target** (standard AFL). `afl->fsrv` runs it.
- The cubin companion is resolved:
  1. `AFL_COQUI_CUBIN=/path/to/target.cubin` env var if set
  2. Otherwise `<ELF_target_path>.cubin` by convention
- Fails fast at init if cubin cannot be loaded.

### 8.3 `coqui_ctx` struct extensions

```c
typedef struct coqui_ctx {
  /* existing: ping, pong, pending, executing, batch_size, ... */

  /* CUDA handles */
  CUcontext   cu_ctx;
  CUmodule    cu_module;
  CUfunction  cu_kernel;
  CUstream    stream_a, stream_b;

  /* Device-persistent buffers */
  CUdeviceptr d_virgin_map;
  CUdeviceptr d_global_statics_pool;
  CUdeviceptr d_slab_pool;

  /* Config */
  u64 batch_timeout_us;
  u64 launch_count;
} coqui_ctx_t;

/* Per-ping-pong half */
typedef struct coqui_batch {
  /* existing host buffers + n_inputs + bytes_used */

  u8             *h_novelty;
  coqui_status_t *h_status;

  CUdeviceptr d_input_bytes, d_offsets, d_input_lens;
  CUdeviceptr d_novelty, d_status;

  CUstream stream;
  CUevent  completion_event;
} coqui_batch_t;
```

### 8.4 `coqui_init(afl, cubin_path)`

1. `cuInit(0)`, `cuDeviceGet(idx from AFL_COQUI_DEVICE env or 0)`, `cuCtxCreate`.
2. Verify sm_75+ via `cuDeviceGetAttribute`; FATAL otherwise.
3. `cuModuleLoad(cubin_path)`, `cuModuleGetFunction(kernel)`, `cuModuleGetGlobal(d_virgin_map)`, `cuMemsetD8(d_virgin_map, 0, 65536)`.
4. Compute per-thread stack budget (§4.3) from `AFL_COQUI_STACK_SIZE` or default 16 KB; set `cuCtxSetLimit(CU_LIMIT_STACK_SIZE, total)`.
5. Check `cuFuncGetAttribute(CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES)` against budget; FATAL if kernel's static usage exceeds (hint to bump `--stack-size`).
6. Allocate ping-pong pair (pinned host + device mirrors); create two streams.
7. If cubin exports `__coqui_statics_per_thread`: allocate statics pool.
8. If `AFL_COQUI_SLAB_SIZE > 0`: allocate slab pool; invoke `__coqui_slab_setup` init kernel.

### 8.5 `coqui_submit_input(afl, buf, len)`

Pack `buf` into the pending batch at 8-byte-aligned offset; increment `n_inputs`. On batch full or byte-budget overflow: call `coqui_launch_batch(afl, pending)`, flip ping/pong. Return 0 (never signals stage bail-out).

### 8.6 `coqui_launch_batch(afl, b)` (internal)

Issues the H→D DMAs, `cuLaunchKernel`, D→H DMAs, and records `completion_event` on the batch's stream. All async — returns immediately.

### 8.7 `coqui_flush_batch(afl)` — stage-boundary drain

1. If `executing` batch has in-flight content, call `coqui_await_and_process(executing)`.
2. If `pending` batch is partial (n_inputs > 0), call `coqui_launch_batch(pending)` then `coqui_await_and_process(pending)`.
3. Reset batches for next use.

### 8.8 `coqui_await_and_process(afl, b)` — flagged-input loop

1. Poll `cuStreamQuery` with timeout. On timeout: `coqui_timeout_salvage`.
2. Scan `h_novelty` bitmap. For each set bit `i`:
   - Reconstruct input buffer from `h_input_bytes + h_offsets[i]`, length `h_input_lens[i]`.
   - `write_to_testcase(afl, input_buf, len, 0)` — AFL's helper; sets up fsrv input channel.
   - `fault = fuzz_run_target(afl, &afl->fsrv, exec_tmout)` — **CPU forkserver populates `trace_bits`**.
   - `save_if_interesting(afl, input_buf, len, fault)` — normal AFL admission.
3. Also scan `h_status` for crashes not flagged as novel — they may not set new coverage bits but should still be saved. Re-execute and `save_if_interesting`.
4. Reset batch.

### 8.9 `coqui_timeout_salvage(afl, b)`

Adapted from coqui's existing recovery. Force-reads `d_status`, identifies threads at `PHASE_COMPLETE`, processes their results, discards others. Calls `cuCtxSynchronize()` to reset CUDA state; FATAL if context is unrecoverable.

### 8.10 `coqui_calibrate_one(afl, buf, len)` — deprecated

Kept in contract header as a no-op stub (returns `FSRV_RUN_OK`). Not called under the coexistence model (calibration goes through `fuzz_run_target` on the CPU fsrv). Reserved for future GPU-side calibration optimization.

### 8.11 `coqui_shutdown(afl)`

Drain streams, free host and device buffers, unload module, destroy context. Reverse of `coqui_init`.

### 8.12 Error handling

| Condition | Detection | Action |
|-----------|-----------|--------|
| `cuInit` fails | init return code | FATAL |
| No sm_75+ | device attribute | FATAL |
| Cubin load fails | `cuModuleLoad` error | FATAL with path + error name |
| Virgin map symbol missing | `cuModuleGetGlobal` error | FATAL |
| Kernel static stack exceeds budget | pre-launch `cuFuncGetAttribute` | FATAL with hint |
| Kernel launch fails async | `cuLaunchKernel` or later `cuStreamQuery` | Log, cancel batch, FATAL if non-recoverable |
| Runtime timeout | wall clock | `coqui_timeout_salvage`, continue |
| Device OOM | `cuMemAlloc` | FATAL with byte count |
| CUDA context lost | any call returns CONTEXT_IS_DESTROYED | FATAL |

All errors loud; no silent continuation.

### 8.13 Performance bounds

- Per-batch DMA: ~10-20 MB H→D + ~129 KB D→H.
- Per-flagged-input CPU exec: 100 µs - 1 ms.
- Ping-pong overlaps CPU post-processing with next GPU batch; stays pipelined as long as flagged_count × cpu_exec < kernel_time.
- For fast kernels (e.g., cjson at sub-5 ms) + high flagged count, CPU side can bottleneck — mitigation (content-hash dedup, capped post-processing) add-on-demand.

## 9. Validation Strategy

### 9.1 Phase 0 — cjson baseline

**Goal:** Prove day-1 passes produce a working cubin; full pipeline (GPU → novelty → CPU re-exec → save_if_interesting) produces queue entries.

**Target:** `cjson_read_fuzzer` from `/coqui/harness/targets/cjson_read_fuzzer.c`. Pure C, minimal deps, no float, no printf, no inline asm.

**Expected port queue (in order):**
1. `ExternalSymbolGatekeeper: unresolved strlen` → port `Libc` replacement (strlen/strcmp/strncmp, whatever cjson uses).
2. Possibly: `ExternalSymbolGatekeeper: unresolved __coqui_global_init` → port `Globals` transform.

**Success criteria:**
- `cjson_read_fuzzer.cubin` builds without errors.
- `afl-fuzz --coqui gpu0 -i seeds -o out/ -- ./cjson_read_fuzzer` runs cleanly.
- After ~60s: `out/gpu0/queue/` has grown; `fuzzer_stats` shows `execs_done > 0`, `corpus_count > 1`.
- Heterogeneous mesh (`-M main` + `--coqui gpu0`) shows cross-propagation.

### 9.2 Phase 1 — libpng stress

**Goal:** Exercise every realistic transform area.

**Target:** `libpng_read_fuzzer` from coqui's existing harness.

**Expected port queue:**
- `Libc` replacements (string functions).
- `Math` — or libdevice linking (§3.6 option b).
- `Printf` + `Sprintf` + `Variadic` for libpng's error reporting.
- Possibly `Relocations` for palette tables.
- `coqui_slab.c` runtime (libpng's state commonly exceeds default per-thread heap).

**Success criteria:**
- libpng cubin builds.
- 10-min mesh run produces at least one non-trivial queue entry traceable to GPU discovery.
- Planted crash in harness produces a crash file in `out/gpu0/crashes/` within 5 min.
- GPU exec/s > 10× broker exec/s on the same target.

### 9.3 Collaborative port-on-demand loop

For each error surfaced:
1. Read the error (gatekeeper FATAL or runtime error).
2. **Propose port approach** to user.
3. **User approves or redirects** — port, try alternative (e.g., libdevice), or skip (disable feature in target build).
4. Execute approved port — code + runtime file + gatekeeper allowlist update.
5. Rebuild cubin; retry. Update port log.

Repeat until target builds and runs.

### 9.4 Port log format

In `docs/coqui_port_log.md`:

```markdown
## 2026-04-NN — Phase 0, cjson

### Port: LibcTransform replacement half, [strlen, strcmp, strncmp]
- Trigger: ExternalSymbolGatekeeper on `strlen` in `cJSON_Parse`
- Source: /coqui/src/LibcTransform.cpp (string-function subset)
- Runtime: coqui_mode/runtime/coqui_libc.c
- Gatekeeper allowlist: += {"strlen", "strcmp", "strncmp"}

### Port: Globals transform
- Trigger: ExternalSymbolGatekeeper on `__coqui_global_init`
- Source: /coqui/src/GlobalTransform.cpp (verbatim)
```

### 9.5 Validation milestones

| Milestone | Criteria |
|-----------|----------|
| V0 | Day-1 pipeline builds: `coqui-cc --help` works, runtime.bc builds |
| V1 | cjson cubin builds |
| V2 | cjson runs end-to-end, queue grows |
| V3 | Heterogeneous mesh cross-propagates |
| V4 | libpng cubin builds |
| V5 | libpng mesh run, GPU exec/s > 10× broker exec/s |

## 10. Naming and Conventions (recap)

- **cuAFL** — the AFL++ fork.
- **coqui_mode** — the executor mode; subdirectory `coqui_mode/`.
- **coqui-cc** — the compiler tool.
- **`--coqui <sync_id>`** — CLI flag on `afl-fuzz`.

## 11. Amendments to the cuAFL Spec

The earlier cuAFL spec (`docs/superpowers/specs/2026-04-18-cuafl-gpu-backend-design.md`) was written against the hollow-stub model. Under this design, the following sections are amended:

1. **§3.6 (afl_fsrv_start short-circuit):** removed. `afl_fsrv_start` runs normally for coqui_mode — `afl->fsrv` is a real CPU forkserver over the ELF target.
2. **§3.5 (common_fuzz_stuff branch):** unchanged — still routes to `coqui_submit_input` in gpu_mode.
3. **§3.7 (fuzz_run_target short-circuit):** removed. `fuzz_run_target` runs normally; CPU forkserver populates `trace_bits`.
4. **§4.2 (batch buffer layout):** `d_coverage` no longer exists in the host-visible contract; it's a kernel-local alloca. `d_virgin_map` is a device-global accessed via `cuModuleGetGlobal`, not a host-managed buffer.
5. **§5.3 (post-batch flagged-input handling):** per-flagged-input coverage DMA no longer happens. The flagged-input loop runs the input through the CPU forkserver via `fuzz_run_target` to populate `trace_bits`, then calls `save_if_interesting` normally.
6. **CLI trailing argv (§2.1):** is the ELF target path, not the cubin. Cubin resolved via `AFL_COQUI_CUBIN` env var or `<ELF>.cubin` convention.
7. **`coqui_calibrate_one` (§4.1):** deprecated (kept as no-op stub). Calibration routes through `fuzz_run_target` → CPU fsrv under coexistence model.

Implementation task list for these amendments is tracked separately in the follow-up plan.

## 12. Out of Scope

- UBSan integration (deferred; port on-demand alongside `coqui_ubsan.c`).
- CmpLog on GPU (by design — lives on broker).
- GPU-side havoc mutation (CPU does mutation in AFL++).
- Oracle tracing (not a cuAFL feature).
- Per-function prologue stack-canary instrumentation (rejected for v1 perf cost).
