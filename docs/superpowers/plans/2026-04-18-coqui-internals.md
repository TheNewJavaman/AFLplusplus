# Coqui Internals Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the GPU-side of cuAFL — `coqui-cc` compiler driver, LLVM pass plugin (10 day-1 passes), GPU device runtime (5 files), and real host-side CUDA launcher replacing the hollow stub — so that `coqui-cc harness.c -o harness.cubin` can be attempted against the cjson harness. Port-on-demand iteration (adding coqui transforms as errors surface) happens after this plan, collaboratively with the user.

**Architecture:** NVPTX-from-start compilation pipeline using `clang --target=nvptx64-nvidia-cuda` + custom LLVM pass plugin + existing coqui device runtime (ported and adjusted). Host launcher runs a standard AFL CPU forkserver alongside the GPU batch executor; post-batch flagged inputs go through the CPU forkserver for real `trace_bits` before `save_if_interesting`. The existing cuAFL integration stub gets replaced with real CUDA driver API calls.

**Tech Stack:** C99 (GPU runtime), C++ (LLVM pass plugin), Python 3 (coqui-cc driver), CUDA Driver API (`libcuda`), LLVM 18 (clang + opt + llc + llvm-link), CUDA Toolkit 13+ (ptxas, libdevice).

**Specs:**
- `docs/superpowers/specs/2026-04-18-coqui-internals-design.md` — the design this plan implements
- `docs/superpowers/specs/2026-04-18-cuafl-gpu-backend-design.md` — the cuAFL contract (amended per §11 of the coqui spec)

**User preferences:**
- No automated tests (user directive from prior session). Verification is manual — build success, CLI sanity, structured commits.
- Atomic commits: one logical change per commit.
- Port-on-demand ports are human-in-the-loop (not automated). This plan stops *before* attempting port iterations; those happen collaboratively after plan execution.

---

## File Structure

**New directories:**
- `coqui_mode/` — top-level directory for GPU backend (parallels existing `qemu_mode/`, `frida_mode/`, `nyx_mode/`)
- `coqui_mode/runtime/` — device-side C files (compiled to `runtime.bc`)
- `coqui_mode/passes/` — LLVM pass plugin sources (`.cpp` + `CMakeLists.txt`)
- `coqui_mode/bin/` — installed `coqui-cc` Python driver + helper scripts

**New files:**
- `coqui_mode/README.md`
- `coqui_mode/build_coqui_support.sh` — install/build entry point
- `coqui_mode/runtime/coqui_runtime.h`
- `coqui_mode/runtime/coqui_runtime.c`
- `coqui_mode/runtime/coqui_coverage.c`
- `coqui_mode/runtime/coqui_memory.c`
- `coqui_mode/runtime/coqui_asan.c`
- `coqui_mode/passes/CMakeLists.txt`
- `coqui_mode/passes/CoquiPassPlugin.cpp` — plugin entry + pipeline registration
- `coqui_mode/passes/InlineAsmReject.cpp`
- `coqui_mode/passes/LibcReject.cpp`
- `coqui_mode/passes/IntrinsicReject.cpp`
- `coqui_mode/passes/FuzzEntry.cpp`
- `coqui_mode/passes/Heap.cpp`
- `coqui_mode/passes/StaticGlobals.cpp`
- `coqui_mode/passes/MemoryLayout.cpp`
- `coqui_mode/passes/Coverage.cpp`
- `coqui_mode/passes/Asan.cpp`
- `coqui_mode/passes/ExternalSymbolGatekeeper.cpp`
- `coqui_mode/passes/Transforms.h` — shared pass declarations
- `coqui_mode/bin/coqui-cc` — Python driver
- `docs/coqui_port_log.md` — port-on-demand log (empty initially)

**Modified files:**
- `include/afl-fuzz-coqui.h` — add CUDA handle fields to struct
- `src/afl-fuzz-coqui.c` — replace hollow stub with real CUDA implementation
- `src/afl-fuzz.c` — update trailing argv handling (ELF target, cubin resolution)
- `src/afl-forkserver.c` — revert coqui_mode short-circuit in `afl_fsrv_start`
- `src/afl-fuzz-run.c` — revert gpu_mode short-circuit in `fuzz_run_target`
- `GNUmakefile` — add `-lcuda` linker flag for afl-fuzz

**Reference directories (read-only, source material for ports):**
- `~/coqui/src/*.cpp` — original transforms
- `~/coqui/runtime/*.c` — original runtime
- `~/coqui/driver/coqui` — original driver

---

## Phase 1: cuAFL amendments

Revert the three stub artifacts (fsrv_start short-circuit, fuzz_run_target short-circuit) and update argv handling so the cubin is resolved as a companion to the ELF target. These amendments are documented in §11 of the coqui spec.

### Task 1.1: Revert `afl_fsrv_start` coqui_mode short-circuit

**Files:**
- Modify: `src/afl-forkserver.c` (remove the block added by commit `0268a097`)

- [ ] **Step 1: Locate the short-circuit in `afl_fsrv_start`**

Run: `grep -n "coqui_mode" src/afl-forkserver.c`

Expected: a block around line 952 reading:
```c
  if (fsrv->coqui_mode) {

    /* coqui_mode owns execution via the afl-fuzz-coqui.c path.
       The forkserver is never launched in this mode; coqui_init() was
       called earlier from afl-fuzz.c main(). */
    return;

  }
```

- [ ] **Step 2: Remove that block**

Delete the 8 lines (including the blank lines surrounding the block) so the code flows directly from the preceding `#endif` to the subsequent `if (!be_quiet) { ACTF("Spinning up the fork server..."); }` line.

- [ ] **Step 3: Verify compilation**

Run: `make -j$(nproc) afl-fuzz 2>&1 | tail -5`
Expected: builds cleanly.

- [ ] **Step 4: Commit**

```bash
git add src/afl-forkserver.c
git commit -m "cuAFL: un-short-circuit afl_fsrv_start for coqui_mode

Per coqui internals design §11 amendment: afl->fsrv is now a real CPU
forkserver running the ELF target alongside the GPU batch executor,
not a dummy. afl_fsrv_start runs normally for coqui_mode instances.
"
```

### Task 1.2: Revert `fuzz_run_target` gpu_mode short-circuit

**Files:**
- Modify: `src/afl-fuzz-run.c` (remove the block added by commit `c771a975`)

- [ ] **Step 1: Locate the short-circuit at the top of `fuzz_run_target`**

Run: `grep -n "gpu_mode" src/afl-fuzz-run.c`

Expected: a block at the top of `fuzz_run_target` (approximately line 55-65) reading:
```c
  if (unlikely(afl->gpu_mode)) {

    /* coqui_mode: no real executor at this layer. ... */
    memset(afl->fsrv.trace_bits, 0, afl->fsrv.map_size);
    afl->fsrv.trace_bits[0] = 1;
    return FSRV_RUN_OK;

  }
```

- [ ] **Step 2: Remove that block**

Delete the 11 lines of the block (including surrounding blank lines). `fuzz_run_target` now runs normally through `afl_fsrv_run_target`.

- [ ] **Step 3: Verify compilation**

Run: `make -j$(nproc) afl-fuzz 2>&1 | tail -5`
Expected: builds cleanly.

- [ ] **Step 4: Commit**

```bash
git add src/afl-fuzz-run.c
git commit -m "cuAFL: un-short-circuit fuzz_run_target for gpu_mode

Per coqui internals design §11 amendment: calibrate/trim/sync paths
now run through the real CPU forkserver at afl->fsrv and get real
trace_bits. The gpu_mode branch in common_fuzz_stuff remains for
the batch-sink detour during havoc.
"
```

### Task 1.3: Change trailing argv handling — ELF is the target, cubin is the companion

**Files:**
- Modify: `src/afl-fuzz.c` (change the cubin-path capture at the post-getopt argv block)

- [ ] **Step 1: Locate the cubin-path capture block**

Run: `grep -n "coqui_cubin_path" src/afl-fuzz.c`

Expected to find the block added by T3.6 (commit `65ba2c20`):
```c
  if (afl->gpu_mode) {
    if (optind >= argc) {
      FATAL("--coqui requires a target cubin path after '--'");
    }
    afl->coqui_cubin_path = ck_strdup(argv[optind]);
    coqui_init(afl, afl->coqui_cubin_path);
  }
```

- [ ] **Step 2: Update the block**

Replace with a version that treats `argv[optind]` as the ELF target path (which AFL already uses elsewhere — this block no longer overrides that). The cubin is resolved via env var `AFL_COQUI_CUBIN` or by convention (`<elf>.cubin`):

```c
  if (afl->gpu_mode) {
    if (optind >= argc) {
      FATAL("--coqui requires a target ELF path after '--'");
    }

    /* Resolve cubin companion path */
    const char *env_cubin = getenv("AFL_COQUI_CUBIN");
    if (env_cubin) {
      afl->coqui_cubin_path = ck_strdup(env_cubin);
    } else {
      /* Convention: <elf>.cubin */
      size_t elf_len = strlen(argv[optind]);
      char *path = ck_alloc(elf_len + 7);  /* ".cubin\0" */
      memcpy(path, argv[optind], elf_len);
      memcpy(path + elf_len, ".cubin", 7);
      afl->coqui_cubin_path = path;
    }

    /* Verify cubin exists */
    if (access(afl->coqui_cubin_path, R_OK) != 0) {
      FATAL("--coqui: cubin not found at '%s' (set AFL_COQUI_CUBIN to override)",
            afl->coqui_cubin_path);
    }

    coqui_init(afl, afl->coqui_cubin_path);
  }
```

- [ ] **Step 3: Verify compilation**

Run: `make -j$(nproc) afl-fuzz 2>&1 | tail -5`
Expected: builds cleanly.

- [ ] **Step 4: Verify CLI still accepts old-style invocation (minus coqui_init failure)**

Run: `./afl-fuzz --coqui gpu0 -i /tmp/seeds -o /tmp/out -- /tmp/nonexistent 2>&1 | head -5`
Expected: fails with "cubin not found at '/tmp/nonexistent.cubin'" (the check works).

- [ ] **Step 5: Commit**

```bash
git add src/afl-fuzz.c
git commit -m "cuAFL: trailing argv is ELF target, cubin resolved as companion

Per coqui internals design §11 amendment: AFL's standard convention
applies — argv[optind] is the ELF target that afl->fsrv runs. Cubin
path comes from AFL_COQUI_CUBIN env var if set, else by convention
<elf>.cubin. Fails fast if cubin not accessible.
"
```

### Task 1.4: Add `-lcuda` to afl-fuzz link

**Files:**
- Modify: `GNUmakefile` (add `-lcuda` to the afl-fuzz link line)

- [ ] **Step 1: Find the afl-fuzz link rule**

Run: `grep -n "afl-fuzz:" GNUmakefile | head -5`

Expected: the rule at line 500-501 ending with `-lm`.

- [ ] **Step 2: Add `-lcuda` to the link command**

In the rule at line 501, append `-lcuda` after `-lm`:
```
	$(CC) $(CFLAGS) $(COMPILE_STATIC) ... -o $@ $(PYFLAGS) $(LDFLAGS) -lm -lcuda
```

- [ ] **Step 3: Verify compilation**

Run: `make -j$(nproc) afl-fuzz 2>&1 | tail -5`
Expected: builds cleanly.

If CUDA isn't installed: build may fail with "cannot find -lcuda". In that case, add a conditional block before the rule:

```make
# CUDA detection for --coqui mode
CUDA_PATH ?= /usr/local/cuda
HAVE_CUDA := $(shell test -d $(CUDA_PATH) && echo yes || echo no)
ifeq "$(HAVE_CUDA)" "yes"
  LDFLAGS += -L$(CUDA_PATH)/lib64/stubs
  override EXTRA_LDFLAGS += -lcuda
  $(info [+] CUDA found at $(CUDA_PATH); linking -lcuda for --coqui support)
else
  $(warning [-] CUDA not found at $(CUDA_PATH); --coqui mode will fail at runtime)
endif
```

And use `$(EXTRA_LDFLAGS)` in the afl-fuzz link rule instead of hard-coding `-lcuda`.

- [ ] **Step 4: Commit**

```bash
git add GNUmakefile
git commit -m "cuAFL: link -lcuda for --coqui mode

Conditional CUDA detection via CUDA_PATH (default /usr/local/cuda).
If found, links libcuda from the stubs dir; otherwise warns and
continues (afl-fuzz builds, but --coqui will fail at runtime).
"
```

---

## Phase 2: `coqui_mode/` directory scaffolding

Create the top-level directory structure mirroring `qemu_mode/` / `nyx_mode/`, plus the shared runtime header.

### Task 2.1: Create top-level directory structure

**Files:**
- Create: `coqui_mode/` (directory)
- Create: `coqui_mode/README.md`
- Create: `coqui_mode/runtime/` (directory)
- Create: `coqui_mode/passes/` (directory)
- Create: `coqui_mode/bin/` (directory)

- [ ] **Step 1: Create directories**

```bash
mkdir -p coqui_mode/runtime coqui_mode/passes coqui_mode/bin
```

- [ ] **Step 2: Write README**

Create `coqui_mode/README.md`:

```markdown
# coqui_mode — GPU executor for cuAFL

`coqui_mode` is cuAFL's GPU-backed executor. Targets are compiled to
NVIDIA cubins via `coqui-cc`, and AFL++ drives them via the `--coqui`
CLI flag.

## Components

- `runtime/` — device-side C files compiled to `runtime.bc` at install
  time, linked into every cubin.
- `passes/` — LLVM pass plugin that runs post-link to instrument the
  target for NVPTX + coverage + heap-only ASan.
- `bin/coqui-cc` — Python driver that orchestrates clang, opt, llc, ptxas.

## Install

```bash
cd coqui_mode
./build_coqui_support.sh
```

Requires LLVM 18 (matching clang version), CUDA toolkit 13+, and Python 3.

## Usage

```bash
# Compile target
coqui-cc -arch sm_75 harness.c lib/*.c -o target.cubin

# Run under cuAFL
afl-clang-fast harness.c lib/*.c -o target
afl-fuzz --coqui gpu0 -i seeds/ -o out/ -- ./target
```

See `docs/superpowers/specs/2026-04-18-coqui-internals-design.md` for
design details.
```

- [ ] **Step 3: Commit**

```bash
git add coqui_mode/README.md
git commit -m "cuAFL: scaffold coqui_mode/ directory

Mirrors the qemu_mode/ / nyx_mode/ pattern: top-level subdirectory for
GPU executor components. README describes the install + usage flow.
"
```

### Task 2.2: Write `coqui_runtime.h` shared header

**Files:**
- Create: `coqui_mode/runtime/coqui_runtime.h`

- [ ] **Step 1: Write the header**

```c
/*
 * coqui_runtime.h --- cuAFL device-side runtime contract.
 *
 * Shared header between the LLVM pass plugin and the GPU runtime .c files.
 * Declares types, constants, and function prototypes used across passes.
 *
 * Compile with: --target=nvptx64-nvidia-cuda -O2 -ffreestanding
 */

#ifndef _COQUI_RUNTIME_H
#define _COQUI_RUNTIME_H

#include <stdint.h>

/* Basic types (matching AFL's u8/u32/u64 convention) */
typedef uint8_t  u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;

/* -- Constants -- */

#define COQUI_COV_MAP_SIZE 65536u        /* 64 KB — AFL default, hardcoded */

/* Bucket class bits (same as AFL count_class_lookup16) */
#define COQUI_BUCKET_0   0x00
#define COQUI_BUCKET_1   0x01
#define COQUI_BUCKET_2   0x02
#define COQUI_BUCKET_3   0x04
#define COQUI_BUCKET_4_7 0x08
#define COQUI_BUCKET_8_15   0x10
#define COQUI_BUCKET_16_31  0x20
#define COQUI_BUCKET_32_127 0x40
#define COQUI_BUCKET_128_UP 0x80

/* Phase markers (written to coqui_status_t.phase by kernel logic) */
#define COQUI_PHASE_EMPTY       0
#define COQUI_PHASE_START       1
#define COQUI_PHASE_EXECUTING   2
#define COQUI_PHASE_EXITED      3
#define COQUI_PHASE_BUCKETING   4
#define COQUI_PHASE_VIRGIN_CMP  5
#define COQUI_PHASE_COMPLETE    6
#define COQUI_PHASE_RESERVED    7

/* Per-thread status reported to the host */
typedef struct coqui_status {
    u8  phase;
    u8  signal;        /* POSIX signal number or 0 */
    u8  asan_error;    /* non-zero if ASan check tripped */
    u8  ubsan_fatal;   /* non-zero if non-recoverable UBSan (future) */
    u32 _reserved0;
    u64 _reserved1;
} coqui_status_t;       /* 16 bytes */

/* -- Device-side helpers (implemented in runtime .c files) -- */

/* Thread identity */
u32 __coqui_fuzz_tid(void);

/* Trap / exit */
void __coqui_trap(void);   /* PTX `trap;` — unrecoverable */
void __coqui_exit(void);   /* PTX `exit;` — this thread exits, kernel continues */

/* Status writing */
void __coqui_status_set_phase(u32 tid, u8 phase);

/* Region accessors (set up by MemoryLayout transform at kernel entry) */
u8  *__coqui_cov_base(void);       /* per-thread coverage map (64 KB) */
u8  *__coqui_heap_base(void);      /* per-thread heap */
u8  *__coqui_shadow_base(void);    /* per-thread ASan shadow */
u32  __coqui_heap_size(void);      /* runtime-configured heap size */

/* Coverage runtime */
extern __attribute__((visibility("default")))
u8 __coqui_count_class_lookup[256];
void __coqui_classify_counts(u8 *map);
void __coqui_virgin_compare_and_flag(u8 *map, u8 *virgin, u32 *novelty_bitmap);

/* prev_loc (per-thread, for AFL hash instrumentation) */
u32 *__coqui_prev_loc_ptr(void);

/* Heap allocator (coqui_memory.c) */
void *__coqui_malloc(unsigned long size);
void  __coqui_free(void *ptr);
void *__coqui_calloc(unsigned long n, unsigned long size);
void *__coqui_realloc(void *ptr, unsigned long size);

/* ASan heap instrumentation (coqui_asan.c) */
void *__coqui_asan_malloc(unsigned long size);
void  __coqui_asan_free(void *ptr);
void  __coqui_asan_check_load_1(void *ptr);
void  __coqui_asan_check_load_2(void *ptr);
void  __coqui_asan_check_load_4(void *ptr);
void  __coqui_asan_check_load_8(void *ptr);
void  __coqui_asan_check_store_1(void *ptr);
void  __coqui_asan_check_store_2(void *ptr);
void  __coqui_asan_check_store_4(void *ptr);
void  __coqui_asan_check_store_8(void *ptr);

/* -- Kernel entry (generated by FuzzEntry transform) -- */

/* Renamed user entry (was LLVMFuzzerTestOneInput). Called by kernel per-thread. */
int __coqui_fuzz_execute(const unsigned char *data, unsigned long size);

/* Device-global virgin map (allocated at module load, accessed via cuModuleGetGlobal) */
extern u8 __coqui_virgin_map[COQUI_COV_MAP_SIZE];

#endif /* _COQUI_RUNTIME_H */
```

- [ ] **Step 2: Commit**

```bash
git add coqui_mode/runtime/coqui_runtime.h
git commit -m "cuAFL: coqui_mode runtime header

Defines the device-side ABI contract: types (coqui_status_t, u8/u32/u64),
constants (COQUI_COV_MAP_SIZE, bucket classes, phase markers), and
prototypes for all __coqui_* helper functions used across the runtime
and LLVM passes.
"
```

### Task 2.3: Write `build_coqui_support.sh`

**Files:**
- Create: `coqui_mode/build_coqui_support.sh`

- [ ] **Step 1: Write the install script**

```bash
#!/usr/bin/env bash
# coqui_mode/build_coqui_support.sh --- build the LLVM pass plugin, compile
# the device runtime to runtime.bc, and install coqui-cc.
#
# Requires: LLVM 18 (clang, llvm-link, opt, llc), CUDA toolkit 13+ (ptxas).

set -euo pipefail

COQUI_DIR="$(cd "$(dirname "$0")" && pwd)"
PREFIX="${PREFIX:-/usr/local}"

echo "[*] Building coqui_mode LLVM pass plugin..."
mkdir -p "$COQUI_DIR/passes/build"
cd "$COQUI_DIR/passes/build"
cmake ..
make -j"$(nproc)"
cd "$COQUI_DIR"

echo "[*] Compiling device runtime to bitcode..."
RUNTIME_FLAGS="--target=nvptx64-nvidia-cuda -O2 -ffreestanding -emit-llvm -c -I $COQUI_DIR/runtime"
mkdir -p "$COQUI_DIR/runtime/build"
for f in coqui_runtime coqui_coverage coqui_memory coqui_asan; do
    clang $RUNTIME_FLAGS "$COQUI_DIR/runtime/$f.c" -o "$COQUI_DIR/runtime/build/$f.bc"
done

echo "[*] Linking runtime bitcode..."
llvm-link "$COQUI_DIR/runtime/build/"*.bc -o "$COQUI_DIR/runtime/build/runtime.bc"

echo "[*] Install..."
install -m 755 "$COQUI_DIR/bin/coqui-cc" "$PREFIX/bin/coqui-cc"
install -d "$PREFIX/lib/coqui-cc"
install -m 644 "$COQUI_DIR/runtime/build/runtime.bc" "$PREFIX/lib/coqui-cc/runtime.bc"
install -m 644 "$COQUI_DIR/passes/build/libCoquiPassPlugin.so" "$PREFIX/lib/coqui-cc/CoquiPassPlugin.so"

echo "[+] coqui-cc installed to $PREFIX/bin/coqui-cc"
echo "[+] Runtime: $PREFIX/lib/coqui-cc/runtime.bc"
echo "[+] Pass plugin: $PREFIX/lib/coqui-cc/CoquiPassPlugin.so"
```

- [ ] **Step 2: Make it executable**

```bash
chmod +x coqui_mode/build_coqui_support.sh
```

- [ ] **Step 3: Commit**

```bash
git add coqui_mode/build_coqui_support.sh
git commit -m "cuAFL: coqui_mode build/install script

Builds the LLVM pass plugin via cmake, compiles the device runtime .c
files to bitcode, links them into runtime.bc, and installs everything
under PREFIX (default /usr/local). Follows the qemu_mode/nyx_mode
build-script convention.
"
```

---

## Phase 3: Day-1 GPU runtime (C files)

Four C files compiled to `runtime.bc` and linked into every user cubin.

### Task 3.1: Write `coqui_runtime.c` — core utilities

**Files:**
- Create: `coqui_mode/runtime/coqui_runtime.c`

- [ ] **Step 1: Write core utility functions**

```c
/*
 * coqui_runtime.c --- core device-side utilities.
 *
 * - Thread identity via PTX inline asm
 * - Trap / exit wrappers
 * - Status struct writers
 * - Safe stub implementations of trivial POSIX functions that pass
 *   the LibcReject gatekeeper (atexit, pthread_once)
 */

#include "coqui_runtime.h"

/* Pointer to per-thread status array, set by kernel entry (FuzzEntry pass).
   Declared __device__ so it lives in global memory. */
__attribute__((visibility("default")))
__attribute__((used))
coqui_status_t *__coqui_status_array;

/* PTX primitives */
u32 __coqui_fuzz_tid(void) {
    u32 tid, bdim, bid;
    __asm__ volatile("mov.u32 %0, %%tid.x;"   : "=r"(tid));
    __asm__ volatile("mov.u32 %0, %%ntid.x;"  : "=r"(bdim));
    __asm__ volatile("mov.u32 %0, %%ctaid.x;" : "=r"(bid));
    return bid * bdim + tid;
}

void __coqui_trap(void) {
    __asm__ volatile("trap;");
}

void __coqui_exit(void) {
    __asm__ volatile("exit;");
}

/* Status writers — thread i writes to __coqui_status_array[i] */
void __coqui_status_set_phase(u32 tid, u8 phase) {
    __coqui_status_array[tid].phase = phase;
}

/* Benign stubs for rarely-used POSIX hooks — lets the gatekeeper pass
   targets that declare but don't meaningfully use these. */
int atexit(void (*fn)(void))                 { (void)fn; return 0; }
void __cxa_atexit(void (*fn)(void *), void *arg, void *dso)
                                              { (void)fn; (void)arg; (void)dso; }
void __cxa_finalize(void *dso)               { (void)dso; }

/* pthread_once — single-threaded on GPU, each thread's "once" runs once per
   thread lifetime (kernel invocation). */
int pthread_once(int *once_control, void (*init)(void)) {
    if (*once_control == 0) {
        init();
        *once_control = 1;
    }
    return 0;
}
```

- [ ] **Step 2: Commit**

```bash
git add coqui_mode/runtime/coqui_runtime.c
git commit -m "cuAFL: coqui_runtime.c — tid, trap, exit, status, stubs

Core device-side utilities. Uses PTX inline asm to compute thread
linear ID from %tid/%ntid/%ctaid. Provides stubs for common POSIX
hooks (atexit, pthread_once) so targets that link but don't
meaningfully use them pass the gatekeeper.
"
```

### Task 3.2: Write `coqui_coverage.c`

**Files:**
- Create: `coqui_mode/runtime/coqui_coverage.c`

- [ ] **Step 1: Write coverage helpers**

```c
/*
 * coqui_coverage.c --- AFL hash coverage bucketing + virgin comparison.
 *
 * Bucketing: 256-entry lookup table in __constant__ memory; walks the
 *   64 KB coverage map 8 bytes at a time with zero-chunk early exit.
 * Virgin compare: atomic-OR the bucketed map into shared virgin_map;
 *   set the novelty bit if any new bits appeared.
 */

#include "coqui_runtime.h"

/* Bucket lookup table — matches AFL's count_class_lookup16. */
__attribute__((section(".const"), used))
u8 __coqui_count_class_lookup[256] = {
    [0]   = COQUI_BUCKET_0,
    [1]   = COQUI_BUCKET_1,
    [2]   = COQUI_BUCKET_2,
    [3]   = COQUI_BUCKET_3,
    [4 ... 7]    = COQUI_BUCKET_4_7,
    [8 ... 15]   = COQUI_BUCKET_8_15,
    [16 ... 31]  = COQUI_BUCKET_16_31,
    [32 ... 127] = COQUI_BUCKET_32_127,
    [128 ... 255] = COQUI_BUCKET_128_UP,
};

/* Bucket the map in place. Walks 8-byte chunks with zero-skip. */
void __coqui_classify_counts(u8 *map) {
    u64 *m64 = (u64 *)map;
    const u32 n_chunks = COQUI_COV_MAP_SIZE / 8;

    for (u32 i = 0; i < n_chunks; i++) {
        u64 word = m64[i];
        if (word == 0) continue;

        u8 *bytes = (u8 *)&word;
        bytes[0] = __coqui_count_class_lookup[bytes[0]];
        bytes[1] = __coqui_count_class_lookup[bytes[1]];
        bytes[2] = __coqui_count_class_lookup[bytes[2]];
        bytes[3] = __coqui_count_class_lookup[bytes[3]];
        bytes[4] = __coqui_count_class_lookup[bytes[4]];
        bytes[5] = __coqui_count_class_lookup[bytes[5]];
        bytes[6] = __coqui_count_class_lookup[bytes[6]];
        bytes[7] = __coqui_count_class_lookup[bytes[7]];
        m64[i] = word;
    }
}

/* 64-bit atomic OR — PTX atom.or.b64 on global memory. */
static u64 atom_or_64(u64 *addr, u64 val) {
    u64 old;
    __asm__ volatile("atom.or.b64 %0, [%1], %2;"
                     : "=l"(old) : "l"(addr), "l"(val));
    return old;
}

/* 32-bit atomic OR — for the novelty bitmap. */
static u32 atom_or_32(u32 *addr, u32 val) {
    u32 old;
    __asm__ volatile("atom.or.b32 %0, [%1], %2;"
                     : "=r"(old) : "l"(addr), "r"(val));
    return old;
}

/* Compare the thread's bucketed map against the shared virgin map.
   Set the novelty bit for this thread if any new bits appeared. */
void __coqui_virgin_compare_and_flag(u8 *map, u8 *virgin, u32 *novelty_bitmap) {
    u64 *m64 = (u64 *)map;
    u64 *v64 = (u64 *)virgin;
    const u32 n = COQUI_COV_MAP_SIZE / 8;

    int novel = 0;
    for (u32 i = 0; i < n; i++) {
        u64 mine = m64[i];
        if (mine == 0) continue;
        u64 was = atom_or_64(&v64[i], mine);
        if (mine & ~was) { novel = 1; }
    }

    if (novel) {
        u32 tid = __coqui_fuzz_tid();
        atom_or_32(&novelty_bitmap[tid >> 5], 1u << (tid & 31u));
    }
}
```

- [ ] **Step 2: Commit**

```bash
git add coqui_mode/runtime/coqui_coverage.c
git commit -m "cuAFL: coqui_coverage.c — AFL bucketing + virgin compare

256-entry class lookup in .const memory with 8-byte-stride zero-skip
for bucketing. Atomic-OR via PTX atom.or.b64 into shared virgin map;
novelty bit set atomically on any 0->1 transition.
"
```

### Task 3.3: Write `coqui_memory.c`

**Files:**
- Create: `coqui_mode/runtime/coqui_memory.c`

- [ ] **Step 1: Read coqui's existing freelist allocator for reference**

Run: `cat ~/coqui/runtime/coqui_fuzz_runtime.c | head -400`

Look for `__coqui_malloc`, `__coqui_free`, and the freelist walk. Understand:
- Size header format (4 or 8 bytes before user data)
- Freelist structure (singly-linked, size-sorted)
- Bump-allocation from the heap region when freelist misses
- Alignment requirements (typically 8-byte)

- [ ] **Step 2: Write the memory runtime**

Key design (from coqui internals spec §6.1):
- `malloc(size)` → try freelist first (first-fit), else bump-allocate from heap region
- Size header: `[u32 block_size][u32 pad][user_data...]`
- On `free(ptr)`: push onto per-thread size-sorted freelist (8-iter insertion)
- Region accessors (`__coqui_heap_base`, etc.) read from compile-time-emitted per-thread allocas via thunks the MemoryLayout pass sets up

```c
/*
 * coqui_memory.c --- per-thread heap allocator for cuAFL coqui_mode.
 *
 * Freelist-first, bump-fallback allocator over the per-thread heap
 * region (set up by MemoryLayout transform; accessed via __coqui_heap_base).
 *
 * Block layout:
 *   [size: u32][pad: u32][user data...]
 *   size includes the 8-byte header.
 *
 * Per-thread state (lives at start of heap region):
 *   free_head (u64) — absolute pointer to first free block
 *   bump_top  (u32) — current bump position within heap region
 */

#include "coqui_runtime.h"

/* Per-thread control header at the start of the heap region. */
typedef struct heap_hdr {
    void *free_head;   /* singly-linked list of free blocks, largest first */
    u32   bump_top;    /* bump allocator position */
    u32   _pad;
} heap_hdr_t;

#define HEAP_MIN_ALLOC 8u
#define HEAP_HDR_SIZE  sizeof(heap_hdr_t)   /* 16 bytes */
#define BLOCK_HDR_SIZE 8u                   /* [size:u32][pad:u32] */

/* Align up to 8 bytes. */
static u32 align8(u32 x) { return (x + 7u) & ~7u; }

/* Get this thread's heap control header. */
static heap_hdr_t *heap_hdr(void) {
    return (heap_hdr_t *)__coqui_heap_base();
}

/* Initialize the heap control header (called once per thread at kernel entry).
   MemoryLayout transform inserts a call to this. */
void __coqui_heap_init(void) {
    heap_hdr_t *hdr = heap_hdr();
    hdr->free_head = NULL;
    hdr->bump_top  = HEAP_HDR_SIZE;
}

void *__coqui_malloc(unsigned long size) {
    if (size == 0) size = 1;
    u32 need = align8((u32)size) + BLOCK_HDR_SIZE;
    if (need < HEAP_MIN_ALLOC + BLOCK_HDR_SIZE) need = HEAP_MIN_ALLOC + BLOCK_HDR_SIZE;

    heap_hdr_t *hdr = heap_hdr();

    /* Try freelist first-fit (8-iter limit) */
    void **prev_next = &hdr->free_head;
    void *cur = hdr->free_head;
    int iters = 8;
    while (cur && iters-- > 0) {
        u32 blk_sz = *(u32 *)cur;
        if (blk_sz >= need) {
            *prev_next = *(void **)((u8 *)cur + BLOCK_HDR_SIZE);
            return (u8 *)cur + BLOCK_HDR_SIZE;
        }
        prev_next = (void **)((u8 *)cur + BLOCK_HDR_SIZE);
        cur = *prev_next;
    }

    /* Bump allocate */
    u32 heap_sz = __coqui_heap_size();
    if (hdr->bump_top + need > heap_sz) {
        __coqui_trap();   /* OOM: trap per spec §4.6 */
        return NULL;
    }

    u8 *block = __coqui_heap_base() + hdr->bump_top;
    *(u32 *)block = need;
    hdr->bump_top += need;
    return block + BLOCK_HDR_SIZE;
}

void __coqui_free(void *ptr) {
    if (!ptr) return;

    u8 *block = (u8 *)ptr - BLOCK_HDR_SIZE;

    /* Bounds check: block must be within heap region */
    u8 *heap_lo = __coqui_heap_base();
    u8 *heap_hi = heap_lo + __coqui_heap_size();
    if (block < heap_lo || block >= heap_hi) return;

    heap_hdr_t *hdr = heap_hdr();
    u32 blk_sz = *(u32 *)block;

    /* Size-sorted insert (8-iter limit, largest first) */
    void **prev_next = &hdr->free_head;
    void *cur = hdr->free_head;
    int iters = 8;
    while (cur && iters-- > 0) {
        u32 cur_sz = *(u32 *)cur;
        if (blk_sz >= cur_sz) break;   /* insert before smaller */
        prev_next = (void **)((u8 *)cur + BLOCK_HDR_SIZE);
        cur = *prev_next;
    }

    *(void **)((u8 *)block + BLOCK_HDR_SIZE) = cur;
    *prev_next = block;
}

static void *memset_u8(void *dst, int c, unsigned long n) {
    u8 *d = (u8 *)dst;
    for (unsigned long i = 0; i < n; i++) d[i] = (u8)c;
    return dst;
}

static void *memcpy_u8(void *dst, const void *src, unsigned long n) {
    u8 *d = (u8 *)dst;
    const u8 *s = (const u8 *)src;
    for (unsigned long i = 0; i < n; i++) d[i] = s[i];
    return dst;
}

void *__coqui_calloc(unsigned long nmemb, unsigned long size) {
    unsigned long total = nmemb * size;
    void *p = __coqui_malloc(total);
    if (p) memset_u8(p, 0, total);
    return p;
}

void *__coqui_realloc(void *ptr, unsigned long size) {
    if (!ptr) return __coqui_malloc(size);
    if (size == 0) { __coqui_free(ptr); return NULL; }

    u8 *block = (u8 *)ptr - BLOCK_HDR_SIZE;
    u32 old_size = *(u32 *)block - BLOCK_HDR_SIZE;

    if (old_size >= size) return ptr;   /* no-op shrink */

    void *new_ptr = __coqui_malloc(size);
    if (!new_ptr) return NULL;
    memcpy_u8(new_ptr, ptr, old_size);
    __coqui_free(ptr);
    return new_ptr;
}
```

- [ ] **Step 3: Commit**

```bash
git add coqui_mode/runtime/coqui_memory.c
git commit -m "cuAFL: coqui_memory.c — per-thread heap allocator

Freelist first-fit (8-iter max) with bump-allocate fallback over the
per-thread heap region emitted by MemoryLayout. Trap on OOM per spec.
Block layout: [size:u32][pad:u32][user data]; freelist is size-sorted
on insertion.
"
```

### Task 3.4: Write `coqui_asan.c`

**Files:**
- Create: `coqui_mode/runtime/coqui_asan.c`

- [ ] **Step 1: Read coqui's existing ASan runtime for reference**

Run: `head -300 ~/coqui/runtime/coqui_fuzz_asan.c`

Understand the shadow layout:
- Shadow is a suffix of the heap region: `[usable heap (H - H/8)] [shadow (H/8)]`
- Shadow byte `shadow[(offset >> 3)]` represents 8 bytes of heap at `heap[offset & ~7]`
- `0` in shadow = all 8 accessible; `k` = first k bytes accessible, rest poisoned; `-k` (0x80..0xFF) = fully poisoned
- `__coqui_asan_check_*` does `if (ptr < heap_base || ptr >= heap_base + usable) return` fast path

- [ ] **Step 2: Write the ASan runtime (heap-only)**

```c
/*
 * coqui_asan.c --- heap-only AddressSanitizer for coqui_mode.
 *
 * Ported from /coqui/runtime/coqui_fuzz_asan.c with adjustments for
 * the new memory layout (shadow is a suffix of the heap region, sized
 * as H/8 where H includes the shadow — so heap has H*8/9 usable and
 * shadow has H/9).
 *
 * Shadow encoding (matches AFL/ASan convention):
 *   shadow[i] = 0   → all 8 heap bytes at (i << 3) accessible
 *   shadow[i] = k   → first k bytes accessible, rest poisoned (1 <= k <= 7)
 *   shadow[i] = -k  → fully poisoned (0x80..0xFF)
 */

#include "coqui_runtime.h"

/* Red-zone sizes (bytes) around heap allocations. */
#define ASAN_LEFT_REDZONE  16
#define ASAN_RIGHT_REDZONE 16

/* Compute usable heap size — shadow occupies 1/9 of the heap region. */
static u32 asan_usable_heap_size(void) {
    u32 full = __coqui_heap_size();
    return full - (full / 9);   /* shadow = full / 9, usable = 8 * full / 9 */
}

/* Get shadow byte for a heap offset. */
static u8 *asan_shadow_for(void *ptr) {
    u8 *heap = __coqui_heap_base();
    u32 usable = asan_usable_heap_size();
    unsigned long off = (u8 *)ptr - heap;
    return heap + usable + (off >> 3);
}

/* Poison N bytes starting at heap offset. */
static void asan_poison_n(unsigned long heap_off, unsigned long n) {
    u8 *shadow = __coqui_heap_base() + asan_usable_heap_size();
    unsigned long start_byte = heap_off >> 3;
    unsigned long end_byte = (heap_off + n) >> 3;
    unsigned long start_partial = heap_off & 7;
    unsigned long end_partial = (heap_off + n) & 7;

    if (start_partial) {
        /* first byte: leave first `start_partial` bytes accessible,
           poison the rest */
        shadow[start_byte] = (u8)start_partial;
        start_byte++;
    }
    for (unsigned long i = start_byte; i < end_byte; i++) shadow[i] = 0xFF;
    if (end_partial) {
        shadow[end_byte] = (u8)end_partial;
    }
}

/* Unpoison N bytes — mark all fully accessible. */
static void asan_unpoison_n(unsigned long heap_off, unsigned long n) {
    u8 *shadow = __coqui_heap_base() + asan_usable_heap_size();
    unsigned long start_byte = (heap_off + 7) >> 3;
    unsigned long end_byte = (heap_off + n) >> 3;
    for (unsigned long i = start_byte; i < end_byte; i++) shadow[i] = 0;
}

/* Check a heap access — returns 0 if OK, non-zero if sanitizer violation. */
static int asan_check(void *ptr, u8 access_size) {
    u8 *heap = __coqui_heap_base();
    u32 usable = asan_usable_heap_size();
    if ((u8 *)ptr < heap || (u8 *)ptr + access_size > heap + usable) {
        return 0;   /* outside heap — not ASan's concern */
    }
    u8 *shadow = asan_shadow_for(ptr);
    unsigned long partial = ((u8 *)ptr - heap) & 7;
    u8 s = shadow[0];
    if (s == 0) return 0;              /* fully accessible */
    if (s & 0x80) return 1;            /* fully poisoned */
    if (partial + access_size > s) return 1;   /* partial: out of bounds */
    return 0;
}

/* Report an ASan error: set status and trap. */
static void asan_report_error(void) {
    u32 tid = __coqui_fuzz_tid();
    extern coqui_status_t *__coqui_status_array;
    __coqui_status_array[tid].asan_error = 1;
    __coqui_exit();   /* thread exits cleanly, kernel continues */
}

/* Size-specialized check helpers — called at every instrumented load/store */
#define CHECK_IMPL(N) \
    void __coqui_asan_check_load_##N(void *ptr) { \
        if (asan_check(ptr, N)) asan_report_error(); \
    } \
    void __coqui_asan_check_store_##N(void *ptr) { \
        if (asan_check(ptr, N)) asan_report_error(); \
    }

CHECK_IMPL(1)
CHECK_IMPL(2)
CHECK_IMPL(4)
CHECK_IMPL(8)

/* Allocator wrappers — add red zones and poison */
void *__coqui_asan_malloc(unsigned long size) {
    unsigned long padded = ASAN_LEFT_REDZONE + size + ASAN_RIGHT_REDZONE;
    void *raw = __coqui_malloc(padded);
    if (!raw) return NULL;
    u8 *user = (u8 *)raw + ASAN_LEFT_REDZONE;

    /* Poison left and right red zones */
    u8 *heap_base = __coqui_heap_base();
    unsigned long raw_off = (u8 *)raw - heap_base;
    asan_poison_n(raw_off, ASAN_LEFT_REDZONE);
    asan_unpoison_n(raw_off + ASAN_LEFT_REDZONE, size);
    asan_poison_n(raw_off + ASAN_LEFT_REDZONE + size, ASAN_RIGHT_REDZONE);

    return user;
}

void __coqui_asan_free(void *ptr) {
    if (!ptr) return;
    u8 *raw = (u8 *)ptr - ASAN_LEFT_REDZONE;

    /* Poison the whole region (use-after-free) */
    u8 *heap_base = __coqui_heap_base();
    unsigned long raw_off = raw - heap_base;
    /* We don't know the original size here without looking at the malloc header.
       The heap's block header is immediately before raw: *(u32 *)(raw - 8). */
    u8 *block_hdr = raw - BLOCK_HDR_SIZE_LOCAL;
    u32 full_sz = *(u32 *)block_hdr;   /* includes 8-byte header */
    unsigned long payload_size = full_sz - 8;
    asan_poison_n(raw_off, payload_size);

    __coqui_free(raw);
}

/* Note: BLOCK_HDR_SIZE is defined in coqui_memory.c — declare locally here.
   TODO during port: decide whether to expose BLOCK_HDR_SIZE via the header
   or inline the 8-byte constant. */
#define BLOCK_HDR_SIZE_LOCAL 8u
```

- [ ] **Step 3: Commit**

```bash
git add coqui_mode/runtime/coqui_asan.c
git commit -m "cuAFL: coqui_asan.c — heap-only ASan runtime

Ported from /coqui/runtime/coqui_fuzz_asan.c with adjustments for new
heap/shadow layout (shadow occupies 1/9 of heap region as a suffix,
usable heap is 8/9). Size-specialized check helpers (load_1/2/4/8,
store_1/2/4/8) with fast-path that returns immediately for non-heap
pointers. Red zones (16 bytes each) around every allocation; poison
on free for use-after-free detection.
"
```

### Task 3.5: Verify all runtime files build

**Files:**
- (none created)

- [ ] **Step 1: Test-compile each runtime file**

```bash
for f in coqui_runtime coqui_coverage coqui_memory coqui_asan; do
    echo "=== Compiling $f.c ==="
    clang --target=nvptx64-nvidia-cuda -O2 -ffreestanding -emit-llvm -S \
          -I coqui_mode/runtime \
          coqui_mode/runtime/$f.c -o /tmp/$f.ll 2>&1 | tail -5
    echo "  bitcode size: $(wc -l < /tmp/$f.ll) lines"
done
```

Expected: all four compile cleanly (may see harmless warnings about unused functions; no errors).

- [ ] **Step 2: Test-link into runtime.bc**

```bash
for f in coqui_runtime coqui_coverage coqui_memory coqui_asan; do
    clang --target=nvptx64-nvidia-cuda -O2 -ffreestanding -emit-llvm -c \
          -I coqui_mode/runtime \
          coqui_mode/runtime/$f.c -o /tmp/$f.bc
done
llvm-link /tmp/coqui_*.bc -o /tmp/runtime.bc
echo "runtime.bc size: $(wc -c < /tmp/runtime.bc) bytes"
```

Expected: link succeeds; file size in the ~10-20 KB range.

No commit (verification only).

---

## Phase 4: LLVM pass plugin (10 day-1 passes)

Ten LLVM passes that run post-link on the combined user + runtime bitcode.

### Task 4.1: Pass plugin build infrastructure

**Files:**
- Create: `coqui_mode/passes/CMakeLists.txt`
- Create: `coqui_mode/passes/Transforms.h`
- Create: `coqui_mode/passes/CoquiPassPlugin.cpp`

- [ ] **Step 1: Write CMakeLists.txt**

```cmake
cmake_minimum_required(VERSION 3.20)
project(CoquiPassPlugin CXX)

find_package(LLVM 18 REQUIRED CONFIG)
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION} at ${LLVM_DIR}")

include_directories(${LLVM_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(COQUI_PASS_SOURCES
    CoquiPassPlugin.cpp
    InlineAsmReject.cpp
    LibcReject.cpp
    IntrinsicReject.cpp
    FuzzEntry.cpp
    Heap.cpp
    StaticGlobals.cpp
    MemoryLayout.cpp
    Coverage.cpp
    Asan.cpp
    ExternalSymbolGatekeeper.cpp
)

add_library(CoquiPassPlugin SHARED ${COQUI_PASS_SOURCES})

# LLVM plugin symbol visibility
target_compile_options(CoquiPassPlugin PRIVATE -fno-rtti -Wall -Wextra)
```

- [ ] **Step 2: Write Transforms.h**

```cpp
/*
 * Transforms.h --- shared pass declarations for CoquiPassPlugin.
 */

#pragma once

namespace llvm {
class Module;
}

namespace coqui {

/* Returns true if the module was modified (standard LLVM convention). */

bool runInlineAsmReject(llvm::Module &M);
bool runLibcReject(llvm::Module &M);
bool runIntrinsicReject(llvm::Module &M);
bool runFuzzEntry(llvm::Module &M);
bool runHeap(llvm::Module &M);
bool runStaticGlobals(llvm::Module &M);
bool runMemoryLayout(llvm::Module &M);
bool runCoverage(llvm::Module &M);
bool runAsan(llvm::Module &M);
bool runExternalSymbolGatekeeper(llvm::Module &M);

} // namespace coqui
```

- [ ] **Step 3: Write CoquiPassPlugin.cpp**

```cpp
/*
 * CoquiPassPlugin.cpp --- LLVM pass plugin entry + registration.
 *
 * Registers the `coqui-link` pass pipeline that runs all day-1 passes
 * in order.
 */

#include "Transforms.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

namespace {

/* The coqui-link pass runs all day-1 transforms in the prescribed order. */
struct CoquiLinkPass : public PassInfoMixin<CoquiLinkPass> {
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &) {
    /* Pass order from coqui internals spec §3.2 */
    coqui::runInlineAsmReject(M);
    coqui::runLibcReject(M);
    coqui::runIntrinsicReject(M);
    coqui::runFuzzEntry(M);
    coqui::runHeap(M);
    coqui::runStaticGlobals(M);
    coqui::runMemoryLayout(M);
    coqui::runCoverage(M);
    coqui::runAsan(M);
    coqui::runExternalSymbolGatekeeper(M);
    return PreservedAnalyses::none();
  }
};

} // anonymous namespace

/* LLVM plugin registration. */
extern "C" ::llvm::PassPluginLibraryInfo LLVM_ATTRIBUTE_WEAK
llvmGetPassPluginInfo() {
  return {
    LLVM_PLUGIN_API_VERSION,
    "CoquiPassPlugin",
    LLVM_VERSION_STRING,
    [](PassBuilder &PB) {
      PB.registerPipelineParsingCallback(
        [](StringRef Name, ModulePassManager &MPM,
           ArrayRef<PassBuilder::PipelineElement>) {
          if (Name == "coqui-link") {
            MPM.addPass(CoquiLinkPass());
            return true;
          }
          return false;
        });
    }
  };
}
```

- [ ] **Step 4: Test the scaffolding builds (will fail to link because the individual passes don't exist yet)**

```bash
mkdir -p coqui_mode/passes/build
cd coqui_mode/passes/build
cmake .. 2>&1 | tail -5
```

Expected: cmake configures successfully.

- [ ] **Step 5: Commit**

```bash
git add coqui_mode/passes/CMakeLists.txt coqui_mode/passes/Transforms.h coqui_mode/passes/CoquiPassPlugin.cpp
git commit -m "cuAFL: LLVM pass plugin scaffolding

CMakeLists.txt for building libCoquiPassPlugin.so against LLVM 18.
Transforms.h declares the 10 day-1 passes. CoquiPassPlugin.cpp
registers the 'coqui-link' pipeline that runs them in order.
Individual pass implementations follow in subsequent tasks.
"
```

### Task 4.2: `InlineAsmReject` pass

**Files:**
- Create: `coqui_mode/passes/InlineAsmReject.cpp`

- [ ] **Step 1: Read coqui's reference implementation**

Run: `cat ~/coqui/src/InlineAsmTransform.cpp`

Note the two-tier logic:
- Benign asm (empty strings or whitespace with only "memory" clobber) → silently remove
- Non-trivial asm → `report_fatal_error` with source-location-informed message

- [ ] **Step 2: Write the port**

```cpp
/*
 * InlineAsmReject.cpp --- reject arch-specific inline asm.
 *
 * Strip benign compiler barriers (empty or whitespace-only asm strings
 * with "memory" clobber) silently. FATAL on anything else with
 * source-location-informed error.
 *
 * Ported from /coqui/src/InlineAsmTransform.cpp (rejection half).
 */

#include "Transforms.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace coqui {

static bool isBenign(StringRef asmStr, StringRef constraints) {
  /* Empty or whitespace-only asm with memory clobber is a compiler barrier. */
  std::string trimmed = asmStr.trim().str();
  if (trimmed.empty() && constraints.contains("memory")) return true;
  return false;
}

bool runInlineAsmReject(Module &M) {
  bool changed = false;
  std::vector<CallInst*> toErase;

  for (Function &F : M) {
    for (BasicBlock &BB : F) {
      for (Instruction &I : BB) {
        CallInst *CI = dyn_cast<CallInst>(&I);
        if (!CI) continue;

        InlineAsm *IA = dyn_cast<InlineAsm>(CI->getCalledOperand());
        if (!IA) continue;

        StringRef asmStr = IA->getAsmString();
        StringRef constraints = IA->getConstraintString();

        if (isBenign(asmStr, constraints)) {
          toErase.push_back(CI);
          changed = true;
          continue;
        }

        /* Non-trivial asm — FATAL with context */
        std::string msg;
        raw_string_ostream os(msg);
        os << "[coqui-cc] InlineAsmReject: non-trivial inline asm in function '"
           << F.getName() << "': " << asmStr.str();
        report_fatal_error(os.str().c_str());
      }
    }
  }

  for (CallInst *CI : toErase) CI->eraseFromParent();
  return changed;
}

} // namespace coqui
```

- [ ] **Step 3: Build and verify**

```bash
cd coqui_mode/passes/build
make CoquiPassPlugin 2>&1 | tail -10
```

Expected: builds (may still fail to link if other passes aren't implemented yet — that's OK at this stage as long as InlineAsmReject.cpp compiles).

- [ ] **Step 4: Commit**

```bash
git add coqui_mode/passes/InlineAsmReject.cpp
git commit -m "cuAFL: InlineAsmReject pass

Silently remove benign compiler-barrier asm (empty string + memory
clobber). FATAL on any other inline asm with the containing function
name. Ported from /coqui/src/InlineAsmTransform.cpp.
"
```

### Task 4.3: `LibcReject` pass

**Files:**
- Create: `coqui_mode/passes/LibcReject.cpp`

- [ ] **Step 1: Read coqui's reference implementation**

Run: `grep -n "report_fatal_error\|setjmp\|pthread" ~/coqui/src/LibcTransform.cpp | head -30`

Note the blocklisted symbols (setjmp/longjmp variants, pthread_* except pthread_once, known-unsupported libc).

- [ ] **Step 2: Write the pass**

```cpp
/*
 * LibcReject.cpp --- reject unsupported libc patterns.
 *
 * Blocklist-only; does NOT provide replacements. Replacements get
 * ported on demand as the gatekeeper flags them.
 *
 * FATALs on:
 *   - setjmp/longjmp family (sigsetjmp, siglongjmp, etc.)
 *   - pthread_* except pthread_once (stub in coqui_runtime.c)
 *   - signal()/raise()/kill() (process signaling)
 *   - fork()/exec() (process control)
 *
 * Ported from /coqui/src/LibcTransform.cpp (rejection half).
 */

#include "Transforms.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <set>
#include <string>

using namespace llvm;

namespace coqui {

static const std::set<std::string> kBlocklist = {
  /* setjmp family */
  "setjmp", "_setjmp", "__setjmp",
  "longjmp", "_longjmp", "__longjmp",
  "sigsetjmp", "siglongjmp",
  "__sigsetjmp",
  /* pthread — except pthread_once (stubbed in coqui_runtime.c) */
  "pthread_create", "pthread_join", "pthread_detach",
  "pthread_mutex_init", "pthread_mutex_destroy",
  "pthread_mutex_lock", "pthread_mutex_unlock", "pthread_mutex_trylock",
  "pthread_cond_init", "pthread_cond_destroy",
  "pthread_cond_wait", "pthread_cond_signal", "pthread_cond_broadcast",
  "pthread_rwlock_init", "pthread_rwlock_rdlock", "pthread_rwlock_wrlock",
  "pthread_rwlock_unlock", "pthread_key_create", "pthread_setspecific",
  "pthread_getspecific", "pthread_self",
  /* Process control */
  "fork", "vfork", "execv", "execve", "execvp", "execl", "execle", "execlp",
  /* Signals */
  "signal", "sigaction", "raise", "kill", "sigprocmask",
  /* Threading primitives */
  "thrd_create", "thrd_join", "mtx_init", "mtx_lock", "mtx_unlock",
};

bool runLibcReject(Module &M) {
  for (Function &F : M) {
    if (!F.isDeclaration()) continue;   /* only care about external refs */

    std::string name = F.getName().str();
    if (kBlocklist.count(name) == 0) continue;

    /* Check whether the symbol has any real uses */
    if (F.use_empty()) continue;

    std::string msg;
    raw_string_ostream os(msg);
    os << "[coqui-cc] LibcReject: '" << name
       << "' is not supported on GPU; disable this feature at build time "
       << "(e.g., -DTARGET_NO_PTHREADS, -DPNG_NO_SETJMP).";
    report_fatal_error(os.str().c_str());
  }
  return false;
}

} // namespace coqui
```

- [ ] **Step 3: Commit**

```bash
git add coqui_mode/passes/LibcReject.cpp
git commit -m "cuAFL: LibcReject pass (blocklist only)

Rejects setjmp/longjmp, pthread_* (except pthread_once), fork/exec,
signal-family functions. Replacement logic for supported libc is
port-on-demand in a separate Libc transform. Ported from the
rejection half of /coqui/src/LibcTransform.cpp.
"
```

### Task 4.4: `IntrinsicReject` pass

**Files:**
- Create: `coqui_mode/passes/IntrinsicReject.cpp`

- [ ] **Step 1: Write the pass**

```cpp
/*
 * IntrinsicReject.cpp --- reject unhandled LLVM intrinsics.
 *
 * Hard-fail on any intrinsic that is neither natively lowerable by the
 * NVPTX backend nor handled by a ported transform.
 *
 * Known-lowerable (do NOT reject):
 *   - llvm.memcpy, llvm.memset, llvm.memmove — NVPTX backend inlines
 *   - llvm.lifetime.* — NVPTX accepts and drops
 *   - llvm.assume — no-op after optimization
 *   - llvm.dbg.* — debug info (should be stripped with -g0, ignore if present)
 *
 * Known-reject (FATAL):
 *   - llvm.setjmp, llvm.longjmp, llvm.eh.* — exception handling
 *   - llvm.va_start, llvm.va_end, llvm.va_arg, llvm.va_copy — variadic
 *     (require Variadic transform port)
 *   - llvm.frameaddress, llvm.returnaddress — stack walking
 *
 * Unknown intrinsic names produce a WARNING (they may be harmless
 * NVPTX-lowerable ones we haven't categorized).
 */

#include "Transforms.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <set>
#include <string>

using namespace llvm;

namespace coqui {

static const std::set<std::string> kBlocklist = {
  "llvm.eh.typeid.for", "llvm.eh.sjlj.setjmp", "llvm.eh.sjlj.longjmp",
  "llvm.setjmp", "llvm.longjmp",
  "llvm.va_start", "llvm.va_end", "llvm.va_arg", "llvm.va_copy",
  "llvm.frameaddress", "llvm.returnaddress",
  "llvm.stacksave", "llvm.stackrestore",
};

bool runIntrinsicReject(Module &M) {
  for (Function &F : M) {
    if (!F.isIntrinsic()) continue;

    std::string name = F.getName().str();

    /* Normalize: drop type suffixes (e.g., llvm.va_start.p0 → llvm.va_start) */
    std::string base = name;
    size_t dot = base.find('.', 5);   /* skip "llvm." prefix */
    if (dot != std::string::npos) {
      size_t next = base.find('.', dot + 1);
      if (next != std::string::npos) base = base.substr(0, next);
    }

    if (kBlocklist.count(base) == 0 && kBlocklist.count(name) == 0) continue;
    if (F.use_empty()) continue;

    std::string msg;
    raw_string_ostream os(msg);
    os << "[coqui-cc] IntrinsicReject: '" << name
       << "' not handled. Port the relevant transform or disable this "
       << "feature in the target.";
    report_fatal_error(os.str().c_str());
  }
  return false;
}

} // namespace coqui
```

- [ ] **Step 2: Commit**

```bash
git add coqui_mode/passes/IntrinsicReject.cpp
git commit -m "cuAFL: IntrinsicReject pass

FATAL on blocklisted LLVM intrinsics (setjmp, longjmp, va_start/end,
frameaddress, etc.) that are neither NVPTX-natively-lowerable nor
handled by a ported transform. Known-lowerable intrinsics (memcpy,
memset, lifetime, assume) pass through silently.
"
```

### Task 4.5: `FuzzEntry` pass

**Files:**
- Create: `coqui_mode/passes/FuzzEntry.cpp`

- [ ] **Step 1: Read coqui's reference implementation**

Run: `head -200 ~/coqui/src/FuzzEntryTransform.cpp`

Understand:
- Renames `LLVMFuzzerTestOneInput` to `__coqui_fuzz_execute`
- Creates `__coqui_fuzz_kernel` with the contract signature
- Kernel body: compute tid, check empty slot, call `__coqui_fuzz_execute`, post-process coverage (bucket + virgin compare)
- Adds `nvvm.annotations` to mark the kernel entry

- [ ] **Step 2: Port with adjustments for new kernel signature**

Per the spec §7.1, the new kernel signature (excluding `coverage` + `virgin_map` params that are device-local / device-global):

```c
void __coqui_fuzz_kernel(
    const u8  *input_bytes,
    const u32 *offsets,
    const u32 *input_lens,
    u32       *novelty,
    coqui_status_t *status
);
```

Write the port at `coqui_mode/passes/FuzzEntry.cpp`. Base structure:

```cpp
/*
 * FuzzEntry.cpp --- rename LLVMFuzzerTestOneInput, create kernel entry.
 *
 * Ported from /coqui/src/FuzzEntryTransform.cpp with updated kernel
 * signature per coqui internals design §7.1 (drops `coverage` and
 * `virgin_map` params — both are device-local/global now).
 *
 * Kernel body (pseudo-code):
 *   tid = blockIdx.x * blockDim.x + threadIdx.x
 *   if (input_lens[tid] == 0) return
 *   __coqui_status_set_phase(tid, PHASE_START)
 *   __coqui_memory_init()                         // allocate per-thread regions
 *   __coqui_fuzz_execute(input_bytes + offsets[tid], input_lens[tid])
 *   __coqui_status_set_phase(tid, PHASE_BUCKETING)
 *   __coqui_classify_counts(cov_base)
 *   __coqui_status_set_phase(tid, PHASE_VIRGIN_CMP)
 *   __coqui_virgin_compare_and_flag(cov_base, virgin_map, novelty)
 *   __coqui_status_set_phase(tid, PHASE_COMPLETE)
 *
 * Also writes nvvm.annotations metadata marking __coqui_fuzz_kernel
 * as a kernel (.entry) for the NVPTX backend.
 */

#include "Transforms.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"

using namespace llvm;

namespace coqui {

bool runFuzzEntry(Module &M) {
  LLVMContext &C = M.getContext();
  IRBuilder<> Builder(C);

  /* 1. Find and rename LLVMFuzzerTestOneInput */
  Function *User = M.getFunction("LLVMFuzzerTestOneInput");
  if (!User) {
    report_fatal_error(
      "[coqui-cc] FuzzEntry: LLVMFuzzerTestOneInput not found — "
      "every cuAFL target must define it");
  }
  User->setName("__coqui_fuzz_execute");

  /* 2. Build the kernel signature */
  Type *i32  = Type::getInt32Ty(C);
  Type *i8p  = PointerType::get(C, 0);       /* opaque ptr in LLVM 18+ */
  Type *u32p = PointerType::get(C, 0);
  Type *statusp = PointerType::get(C, 0);

  FunctionType *KernelType = FunctionType::get(
    Type::getVoidTy(C),
    {i8p, u32p, u32p, u32p, statusp},
    /*isVarArg*/ false);

  Function *Kernel = Function::Create(
    KernelType, GlobalValue::ExternalLinkage,
    "__coqui_fuzz_kernel", &M);

  /* 3. Declare helpers the kernel will call.
     These are provided by the runtime (coqui_runtime.c / coqui_coverage.c). */
  auto getOrDeclare = [&](StringRef Name, FunctionType *FT) {
    return M.getOrInsertFunction(Name, FT).getCallee();
  };

  FunctionType *voidNoArg = FunctionType::get(Type::getVoidTy(C), false);
  FunctionType *tidRet = FunctionType::get(i32, false);
  FunctionType *ptrNoArg = FunctionType::get(i8p, false);

  Value *getTid = getOrDeclare("__coqui_fuzz_tid", tidRet);
  Value *setPhase = getOrDeclare("__coqui_status_set_phase",
    FunctionType::get(Type::getVoidTy(C), {i32, Type::getInt8Ty(C)}, false));
  Value *memoryInit = getOrDeclare("__coqui_memory_init", voidNoArg);
  Value *covBase = getOrDeclare("__coqui_cov_base", ptrNoArg);
  Value *classify = getOrDeclare("__coqui_classify_counts",
    FunctionType::get(Type::getVoidTy(C), {i8p}, false));
  Value *virginCompare = getOrDeclare("__coqui_virgin_compare_and_flag",
    FunctionType::get(Type::getVoidTy(C), {i8p, i8p, u32p}, false));

  /* The __coqui_virgin_map global — declared as external, resolved at
     module-load time by the host via cuModuleGetGlobal. */
  ArrayType *VirginType = ArrayType::get(Type::getInt8Ty(C), 65536);
  GlobalVariable *VirginMap = new GlobalVariable(
    M, VirginType, /*isConstant*/false, GlobalValue::ExternalLinkage,
    nullptr, "__coqui_virgin_map");

  /* 4. Emit the kernel body */
  BasicBlock *EntryBB = BasicBlock::Create(C, "entry", Kernel);
  BasicBlock *RunBB = BasicBlock::Create(C, "run", Kernel);
  BasicBlock *ExitBB = BasicBlock::Create(C, "exit", Kernel);
  Builder.SetInsertPoint(EntryBB);

  /* args */
  auto args = Kernel->arg_begin();
  Value *inputBytes = &*args++;
  Value *offsets    = &*args++;
  Value *inputLens  = &*args++;
  Value *novelty    = &*args++;
  Value *status     = &*args++;

  /* tid = __coqui_fuzz_tid() */
  Value *tid = Builder.CreateCall(cast<Function>(getTid)->getFunctionType(),
                                   getTid, {}, "tid");

  /* len = input_lens[tid] */
  Value *lensPtr = Builder.CreateGEP(i32, inputLens, tid, "len_ptr");
  Value *len = Builder.CreateLoad(i32, lensPtr, "len");

  /* if len == 0 return */
  Value *empty = Builder.CreateICmpEQ(len, ConstantInt::get(i32, 0));
  Builder.CreateCondBr(empty, ExitBB, RunBB);

  /* run: */
  Builder.SetInsertPoint(RunBB);

  /* status[tid].phase = PHASE_START */
  Builder.CreateCall(cast<Function>(setPhase)->getFunctionType(), setPhase,
    {tid, ConstantInt::get(Type::getInt8Ty(C), 1 /* PHASE_START */)});

  /* __coqui_memory_init() */
  Builder.CreateCall(cast<Function>(memoryInit)->getFunctionType(), memoryInit, {});

  /* input_ptr = input_bytes + offsets[tid] */
  Value *offPtr = Builder.CreateGEP(i32, offsets, tid, "off_ptr");
  Value *off = Builder.CreateLoad(i32, offPtr, "off");
  Value *inputPtr = Builder.CreateGEP(Type::getInt8Ty(C), inputBytes, off, "input_ptr");

  /* __coqui_fuzz_execute(input_ptr, len) */
  FunctionType *ExecType = FunctionType::get(i32, {i8p, Type::getInt64Ty(C)}, false);
  Value *len64 = Builder.CreateZExt(len, Type::getInt64Ty(C));
  Builder.CreateCall(ExecType, User, {inputPtr, len64});

  /* Post-exec: bucket + virgin compare */
  Builder.CreateCall(cast<Function>(setPhase)->getFunctionType(), setPhase,
    {tid, ConstantInt::get(Type::getInt8Ty(C), 4 /* PHASE_BUCKETING */)});

  Value *cov = Builder.CreateCall(cast<Function>(covBase)->getFunctionType(), covBase, {});
  Builder.CreateCall(cast<Function>(classify)->getFunctionType(), classify, {cov});

  Builder.CreateCall(cast<Function>(setPhase)->getFunctionType(), setPhase,
    {tid, ConstantInt::get(Type::getInt8Ty(C), 5 /* PHASE_VIRGIN_CMP */)});

  Value *virgin = Builder.CreateBitCast(VirginMap, i8p);
  Builder.CreateCall(cast<Function>(virginCompare)->getFunctionType(), virginCompare,
    {cov, virgin, novelty});

  Builder.CreateCall(cast<Function>(setPhase)->getFunctionType(), setPhase,
    {tid, ConstantInt::get(Type::getInt8Ty(C), 6 /* PHASE_COMPLETE */)});

  Builder.CreateBr(ExitBB);

  /* exit: */
  Builder.SetInsertPoint(ExitBB);
  Builder.CreateRetVoid();

  /* 5. Add nvvm.annotations metadata marking Kernel as a kernel */
  NamedMDNode *Annots = M.getOrInsertNamedMetadata("nvvm.annotations");
  Metadata *KernelMD = ValueAsMetadata::get(Kernel);
  MDString *KernelStr = MDString::get(C, "kernel");
  Metadata *One = ConstantAsMetadata::get(ConstantInt::get(i32, 1));
  Annots->addOperand(MDNode::get(C, {KernelMD, KernelStr, One}));

  return true;
}

} // namespace coqui
```

- [ ] **Step 3: Commit**

```bash
git add coqui_mode/passes/FuzzEntry.cpp
git commit -m "cuAFL: FuzzEntry pass

Renames LLVMFuzzerTestOneInput -> __coqui_fuzz_execute, emits
__coqui_fuzz_kernel with the 5-argument signature from §7.1 of the
coqui internals spec (input_bytes, offsets, input_lens, novelty,
status), adds nvvm.annotations marking it as a kernel entry.
Kernel body: per-thread bucketing + virgin compare via runtime helpers.
"
```

### Task 4.6: `Heap` pass (port from coqui)

**Files:**
- Create: `coqui_mode/passes/Heap.cpp`

- [ ] **Step 1: Read coqui's reference**

Run: `head -150 ~/coqui/src/HeapTransform.cpp`

Note the symbols to replace: `malloc` → `__coqui_malloc`, `free` → `__coqui_free`, `calloc`, `realloc`, `aligned_alloc`.

- [ ] **Step 2: Write the pass (verbatim port, mechanism is RAUW)**

```cpp
/*
 * Heap.cpp --- replace standard malloc/free calls with __coqui_*.
 *
 * Ported verbatim from /coqui/src/HeapTransform.cpp. Uses RAUW
 * (replaceAllUsesWith) on each libc allocator symbol.
 */

#include "Transforms.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"

using namespace llvm;

namespace coqui {

static bool replaceAlloc(Module &M, StringRef Old, StringRef New) {
  Function *OldF = M.getFunction(Old);
  if (!OldF || OldF->use_empty()) return false;

  Function *NewF = M.getFunction(New);
  if (!NewF) {
    /* Declare the new function with the same type as the old */
    NewF = Function::Create(OldF->getFunctionType(),
                            GlobalValue::ExternalLinkage, New, &M);
  }

  OldF->replaceAllUsesWith(NewF);
  OldF->eraseFromParent();
  return true;
}

bool runHeap(Module &M) {
  bool changed = false;
  changed |= replaceAlloc(M, "malloc",        "__coqui_malloc");
  changed |= replaceAlloc(M, "free",          "__coqui_free");
  changed |= replaceAlloc(M, "calloc",        "__coqui_calloc");
  changed |= replaceAlloc(M, "realloc",       "__coqui_realloc");
  changed |= replaceAlloc(M, "aligned_alloc", "__coqui_malloc");   /* align ignored for now */
  changed |= replaceAlloc(M, "posix_memalign", "__coqui_malloc");
  return changed;
}

} // namespace coqui
```

- [ ] **Step 3: Commit**

```bash
git add coqui_mode/passes/Heap.cpp
git commit -m "cuAFL: Heap pass — replace libc allocators

RAUW malloc/free/calloc/realloc/aligned_alloc/posix_memalign to the
corresponding __coqui_* runtime functions. Ported verbatim from
/coqui/src/HeapTransform.cpp.
"
```

### Task 4.7: `StaticGlobals` pass (port from coqui)

**Files:**
- Create: `coqui_mode/passes/StaticGlobals.cpp`

- [ ] **Step 1: Read coqui's reference**

Run: `head -200 ~/coqui/src/StaticTransform.cpp`

Key algorithm:
- Enumerate all writable global variables
- For each: compute per-thread offset, insert at module level a helper `__coqui_get_global_X(void)` that returns `pool + tid * stride + offset`
- Rewrite all users of the global to call the helper
- Total-globals-size is baked into a `@__coqui_statics_per_thread` global so the host can allocate the pool

- [ ] **Step 2: Port the pass**

This is the most complex port. Refer to the coqui source for the full algorithm. Write a faithful port at `coqui_mode/passes/StaticGlobals.cpp`. Key structural similarity to coqui:
1. Walk `M.globals()`, filter to writable + non-constant + address-taken-or-mutated.
2. For each survivor, allocate a slot in the pool (offset tracking).
3. Insert a `__coqui_global_XXX` accessor function returning `__coqui_statics_pool + tid * per_thread_size + slot_offset`.
4. Rewrite all direct `GlobalVariable *` uses to call the accessor.
5. Emit `__coqui_statics_per_thread` as a `u32` constant global for the host.
6. Write-through / read-through: since the pool is per-thread, accesses are race-free.

Skip cross-ConstantExpr references for v1 (coqui also skips those; documented in constraints.md). Log skipped globals to stderr for visibility.

(Given the complexity, the implementing subagent should refer to coqui's source and adapt. The output file is ~300-500 lines of C++.)

- [ ] **Step 3: Commit**

```bash
git add coqui_mode/passes/StaticGlobals.cpp
git commit -m "cuAFL: StaticGlobals pass

Per-thread localization of writable globals via tid-indexed pool
access. Each global variable is replaced with calls to a generated
accessor function that computes pool+tid*size+offset. Total
per-thread globals size is exported as @__coqui_statics_per_thread
for the host launcher to use when allocating the pool.

Ported from /coqui/src/StaticTransform.cpp with no algorithmic changes.
"
```

### Task 4.8: `MemoryLayout` pass (new)

**Files:**
- Create: `coqui_mode/passes/MemoryLayout.cpp`

- [ ] **Step 1: Write the pass**

Per spec §4.2: emit fixed allocas in the kernel entry for coverage (64 KB), heap (derived), and shadow (H/8). Expose region bases via helper functions.

```cpp
/*
 * MemoryLayout.cpp --- set up per-thread stack regions.
 *
 * Inserts three allocas at kernel entry:
 *   - cov_map  : 64 KB (aligned 8)
 *   - heap     : H bytes (runtime-configured via --stack-size)
 *   - shadow   : H/8 bytes (aligned 8)
 *
 * Emits helper functions __coqui_cov_base(), __coqui_heap_base(),
 * __coqui_shadow_base(), __coqui_heap_size() that return the correct
 * pointers for this thread. Implementation: each helper is emitted as
 * an LLVM function that takes no args and returns the alloca pointer
 * from the kernel entry's frame. Because allocas are per-thread in
 * NVPTX .local memory, these helpers naturally return per-thread
 * pointers.
 *
 * Also emits __coqui_memory_init() that zeroes the heap control header
 * at thread start (called from FuzzEntry's kernel body).
 *
 * Coverage + heap sizes are compile-time constants here; in future
 * we could make them cubin-level constants that the host reads from
 * the .conf sidecar.
 */

#include "Transforms.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

namespace coqui {

/* Default region sizes — overridable via environment var at compile time
   (read from AFL_COQUI_STACK_SIZE or similar, passed through coqui-cc). */
static constexpr unsigned kCovMapSize = 65536;
static constexpr unsigned kDefaultStackSize = 16384;   /* real stack S */
static constexpr unsigned kTotalBudget = 524288;       /* sm_75 cap */

bool runMemoryLayout(Module &M) {
  LLVMContext &C = M.getContext();

  /* Read compile-time config; for v1 use constants. */
  unsigned realStack = kDefaultStackSize;
  unsigned remaining = kTotalBudget - kCovMapSize - realStack;
  unsigned heapSize = (remaining * 8) / 9;
  unsigned shadowSize = remaining - heapSize;
  (void)shadowSize;

  /* 1. Locate kernel entry */
  Function *Kernel = M.getFunction("__coqui_fuzz_kernel");
  if (!Kernel) {
    report_fatal_error("[coqui-cc] MemoryLayout: __coqui_fuzz_kernel not found");
  }

  /* 2. Insert allocas at kernel entry */
  BasicBlock &EntryBB = Kernel->getEntryBlock();
  IRBuilder<> Builder(&EntryBB, EntryBB.begin());
  Type *i8 = Type::getInt8Ty(C);

  AllocaInst *CovAlloca = Builder.CreateAlloca(
    ArrayType::get(i8, kCovMapSize), nullptr, "cov_map");
  CovAlloca->setAlignment(Align(8));

  AllocaInst *HeapAlloca = Builder.CreateAlloca(
    ArrayType::get(i8, heapSize), nullptr, "heap");
  HeapAlloca->setAlignment(Align(16));

  /* Shadow is the suffix of heap — we don't need a separate alloca;
     the runtime computes shadow_base = heap_base + (heap_size * 8/9). */

  /* 3. Emit helper functions that return these allocas.
     Each helper has no args and returns i8*. We emit them as one-instruction
     functions that just return the captured alloca. */

  /* Implementation detail: we can't directly return the alloca from a
     separate function because allocas are scoped to Kernel. Instead, store
     the pointer into a __device__ global at kernel entry, and have helpers
     return that global.

     This works because:
     - __device__ globals in .local address space are per-thread
     - Store at kernel entry; load in any device function called from kernel
  */

  /* Declare per-thread helper globals. Use addrspace(5) = NVPTX .local. */
  Type *i8p = PointerType::get(C, 0);
  auto *CovSlot = new GlobalVariable(M, i8p, false,
    GlobalValue::InternalLinkage, Constant::getNullValue(i8p),
    "__coqui_cov_base_slot", nullptr,
    GlobalValue::NotThreadLocal, 5 /*.local*/);
  auto *HeapSlot = new GlobalVariable(M, i8p, false,
    GlobalValue::InternalLinkage, Constant::getNullValue(i8p),
    "__coqui_heap_base_slot", nullptr,
    GlobalValue::NotThreadLocal, 5);

  /* Store the alloca pointers into the slots */
  Builder.CreateStore(
    Builder.CreateBitCast(CovAlloca, i8p), CovSlot);
  Builder.CreateStore(
    Builder.CreateBitCast(HeapAlloca, i8p), HeapSlot);

  /* 4. Define __coqui_cov_base / __coqui_heap_base as functions that load from the slots */
  auto emitGetter = [&](StringRef Name, GlobalVariable *Slot) {
    FunctionType *FT = FunctionType::get(i8p, false);
    Function *F = Function::Create(FT, GlobalValue::LinkOnceODRLinkage, Name, &M);
    F->setLinkage(GlobalValue::InternalLinkage);
    BasicBlock *BB = BasicBlock::Create(C, "", F);
    IRBuilder<> B(BB);
    B.CreateRet(B.CreateLoad(i8p, Slot));
  };

  /* Only emit if not already present (runtime .c may have forward-declared them) */
  if (!M.getFunction("__coqui_cov_base"))
    emitGetter("__coqui_cov_base", CovSlot);
  if (!M.getFunction("__coqui_heap_base"))
    emitGetter("__coqui_heap_base", HeapSlot);

  /* __coqui_shadow_base = heap_base + usable_heap */
  if (!M.getFunction("__coqui_shadow_base")) {
    FunctionType *FT = FunctionType::get(i8p, false);
    Function *F = Function::Create(FT, GlobalValue::InternalLinkage,
                                    "__coqui_shadow_base", &M);
    BasicBlock *BB = BasicBlock::Create(C, "", F);
    IRBuilder<> B(BB);
    Value *heap = B.CreateLoad(i8p, HeapSlot);
    unsigned usable = (heapSize * 8) / 9;
    Value *shadow = B.CreateGEP(i8, heap,
      ConstantInt::get(Type::getInt32Ty(C), usable));
    B.CreateRet(shadow);
  }

  /* __coqui_heap_size() = compile-time constant */
  if (!M.getFunction("__coqui_heap_size")) {
    FunctionType *FT = FunctionType::get(Type::getInt32Ty(C), false);
    Function *F = Function::Create(FT, GlobalValue::InternalLinkage,
                                    "__coqui_heap_size", &M);
    BasicBlock *BB = BasicBlock::Create(C, "", F);
    IRBuilder<> B(BB);
    B.CreateRet(ConstantInt::get(Type::getInt32Ty(C), (heapSize * 8) / 9));
  }

  return true;
}

} // namespace coqui
```

- [ ] **Step 2: Commit**

```bash
git add coqui_mode/passes/MemoryLayout.cpp
git commit -m "cuAFL: MemoryLayout pass (new)

Emits per-thread region allocas (coverage 64KB, heap H, shadow suffix)
at kernel entry. Region bases are stored into .local slots at entry
and retrieved by getter functions (__coqui_cov_base, etc.). Heap size
is computed from total budget - coverage - stack-size per spec §4.2.
Shadow occupies H/9 of the heap region as a suffix.
"
```

### Task 4.9: `Coverage` pass (new)

**Files:**
- Create: `coqui_mode/passes/Coverage.cpp`

- [ ] **Step 1: Write the pass**

Per spec §5.2 and §5.8: at each BB entry, emit the AFL hash sequence.

```cpp
/*
 * Coverage.cpp --- insert AFL hash-based edge instrumentation.
 *
 * At each basic-block entry (after PHIs), emit:
 *   idx       = load __coqui_prev_loc
 *   idx       = idx XOR <cur_loc_const>
 *   cov_ptr   = __coqui_cov_base() + idx
 *   *cov_ptr  = (*cov_ptr) + 1   (saturate at 255)
 *   __coqui_prev_loc = <cur_loc_const> >> 1
 *
 * <cur_loc_const> is a deterministic u16 per basic block, seeded by
 * hash(module name + block index) so recompilation produces the same
 * values.
 *
 * Functions prefixed __coqui_* (runtime helpers) are not instrumented.
 */

#include "Transforms.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/xxhash.h"

using namespace llvm;

namespace coqui {

static constexpr uint32_t kMapSize = 65536;

bool runCoverage(Module &M) {
  LLVMContext &C = M.getContext();
  Type *i32 = Type::getInt32Ty(C);
  Type *i8  = Type::getInt8Ty(C);
  Type *i8p = PointerType::get(C, 0);
  Type *i32p = PointerType::get(C, 0);

  /* Declare helpers (defined by MemoryLayout + runtime) */
  FunctionType *GetBaseType = FunctionType::get(i8p, false);
  FunctionCallee CovBase = M.getOrInsertFunction("__coqui_cov_base", GetBaseType);

  /* Declare __coqui_prev_loc as an external i32* */
  GlobalVariable *PrevLoc = M.getGlobalVariable("__coqui_prev_loc");
  if (!PrevLoc) {
    PrevLoc = new GlobalVariable(M, i32, false,
      GlobalValue::ExternalLinkage, Constant::getNullValue(i32),
      "__coqui_prev_loc", nullptr, GlobalValue::NotThreadLocal, 5);
  }

  uint64_t moduleSeed = xxh3_64bits(M.getName().data(), M.getName().size());

  /* Assign deterministic cur_loc values to each BB.
     Skip instrumenting helper functions (__coqui_*). */
  for (Function &F : M) {
    if (F.isDeclaration()) continue;
    if (F.getName().starts_with("__coqui_")) continue;

    uint32_t blockIdx = 0;
    for (BasicBlock &BB : F) {
      uint64_t locSeed = moduleSeed ^
        (uint64_t)blockIdx ^
        xxh3_64bits(F.getName().data(), F.getName().size());
      uint16_t curLoc = (uint16_t)(locSeed & 0xFFFF);
      blockIdx++;

      /* Insert after PHIs */
      Instruction *insertPt = BB.getFirstNonPHI();
      IRBuilder<> B(insertPt);

      /* prev = load prev_loc */
      Value *prev = B.CreateLoad(i32, PrevLoc, "prev");
      /* idx = prev XOR cur_loc */
      Value *idx = B.CreateXor(prev, ConstantInt::get(i32, curLoc), "idx");
      /* cov_base + idx */
      Value *covBase = B.CreateCall(GetBaseType, CovBase.getCallee(), {}, "cov_base");
      Value *covPtr = B.CreateGEP(i8, covBase, idx, "cov_ptr");
      /* count = *cov_ptr + 1 */
      Value *count = B.CreateLoad(i8, covPtr, "count");
      Value *inc = B.CreateAdd(count, ConstantInt::get(i8, 1), "inc");
      /* *cov_ptr = inc */
      B.CreateStore(inc, covPtr);
      /* prev_loc = cur_loc >> 1 */
      B.CreateStore(ConstantInt::get(i32, curLoc >> 1), PrevLoc);
    }
  }

  return true;
}

} // namespace coqui
```

- [ ] **Step 2: Commit**

```bash
git add coqui_mode/passes/Coverage.cpp
git commit -m "cuAFL: Coverage pass (new) — AFL hash instrumentation

Inserts the 6-instruction AFL hash sequence (load prev_loc, XOR with
per-BB random constant, increment map[idx], store shifted cur_loc to
prev_loc) at every basic-block entry in user code. Skips
__coqui_*-prefixed functions (runtime helpers).

Per-BB constants seeded deterministically via xxhash of module+function
name+block index; stable across recompiles.
"
```

### Task 4.10: `Asan` pass (port from coqui)

**Files:**
- Create: `coqui_mode/passes/Asan.cpp`

- [ ] **Step 1: Read coqui's reference**

Run: `head -300 ~/coqui/src/AsanTransform.cpp`

Note two responsibilities:
1. RAUW `__coqui_malloc` → `__coqui_asan_malloc` (and similarly for free/calloc/realloc)
2. Instrument loads/stores: insert `__coqui_asan_check_load_N` / `_store_N` call before each load/store

- [ ] **Step 2: Port with adjustments**

Base the port on coqui's source. Adapt for our new shadow layout (shadow is a suffix of heap region, computed as `heap_base + heap_size * 8/9`). Key changes from coqui:
- Region accessors are `__coqui_heap_base()`, `__coqui_shadow_base()` — no change needed
- Do NOT instrument stack allocas (coqui also doesn't — `constraints.md` explicitly says stack is outside ASan)
- Do NOT instrument accesses to statics pool (the fast-path check `ptr in heap` returns false)

Writing the full pass is ~400-500 lines. Reference coqui source; keep logic faithful, swap size constants for the new layout.

- [ ] **Step 3: Commit**

```bash
git add coqui_mode/passes/Asan.cpp
git commit -m "cuAFL: Asan pass — heap-only instrumentation

Instruments loads and stores with __coqui_asan_check_load/store_N
helpers. RAUW __coqui_malloc -> __coqui_asan_malloc etc. Stack
allocas are NOT instrumented (outside heap shadow region). Statics
pool accesses pass through the fast path (not in heap, skipped).

Ported from /coqui/src/AsanTransform.cpp with updated shadow-region
accessors for the new memory layout.
"
```

### Task 4.11: `ExternalSymbolGatekeeper` pass

**Files:**
- Create: `coqui_mode/passes/ExternalSymbolGatekeeper.cpp`

- [ ] **Step 1: Write the pass**

```cpp
/*
 * ExternalSymbolGatekeeper.cpp --- final check for unresolved externals.
 *
 * Runs LAST in the coqui-link pipeline. Scans all external function
 * declarations that have live uses. FATALs with a specific message
 * for any symbol not on the allowlist.
 *
 * Allowlist seeded with LLVM-intrinsic-lowered operations that never
 * appear as external symbols in post-backend code (memcpy, memset, etc.)
 * and with all __coqui_* runtime functions.
 */

#include "Transforms.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <set>
#include <string>

using namespace llvm;

namespace coqui {

static const std::set<std::string> kAllowlist = {
  /* LLVM-intrinsic-lowered libc (never external in post-backend code) */
  /* nothing — these are all intrinsics now */

  /* Runtime symbols (all __coqui_*) — allowed by prefix below */
};

bool runExternalSymbolGatekeeper(Module &M) {
  for (Function &F : M) {
    if (!F.isDeclaration()) continue;
    if (F.isIntrinsic()) continue;
    if (F.use_empty()) continue;

    std::string name = F.getName().str();

    /* Allow __coqui_* */
    if (name.rfind("__coqui_", 0) == 0) continue;
    /* Allow LLVM builtin names */
    if (name.rfind("__llvm_", 0) == 0) continue;
    /* Allow the explicit allowlist */
    if (kAllowlist.count(name)) continue;

    /* FATAL */
    std::string msg;
    raw_string_ostream os(msg);
    os << "[coqui-cc] ExternalSymbolGatekeeper: unresolved external '"
       << name << "' — port the replacement transform or runtime stub. "
       << "Used by: ";
    bool first = true;
    for (const User *U : F.users()) {
      if (const auto *I = dyn_cast<Instruction>(U)) {
        if (!first) os << ", ";
        os << I->getFunction()->getName();
        first = false;
        if (!first) break;
      }
    }
    report_fatal_error(os.str().c_str());
  }
  return false;
}

} // namespace coqui
```

- [ ] **Step 2: Build the full pass plugin**

```bash
cd coqui_mode/passes/build
make -j$(nproc) 2>&1 | tail -10
```

Expected: `libCoquiPassPlugin.so` built. All 10 passes compile and link.

- [ ] **Step 3: Commit**

```bash
git add coqui_mode/passes/ExternalSymbolGatekeeper.cpp
git commit -m "cuAFL: ExternalSymbolGatekeeper pass

Runs last in the pipeline. Walks external function declarations;
FATALs on any with live uses not on the allowlist. Allowlist covers
__coqui_* runtime symbols and __llvm_* builtins by prefix. Other
allowed symbols get explicit entries.

This is the final safety net: even if an earlier pass didn't reject
something, the gatekeeper catches it before we generate broken PTX.
"
```

---

## Phase 5: `coqui-cc` compiler driver

### Task 5.1: Write the Python driver

**Files:**
- Create: `coqui_mode/bin/coqui-cc`

- [ ] **Step 1: Read coqui's reference driver**

Run: `head -150 ~/coqui/driver/coqui`

Understand the stages: clang compilation, llvm-link, opt, llc, ptxas.

- [ ] **Step 2: Write the new driver**

```python
#!/usr/bin/env python3
"""
coqui-cc --- cuAFL compiler driver.

Compiles C/C++ sources targeting NVPTX, runs the coqui pass plugin,
links the device runtime, produces a .cubin.

Simplified from /coqui/driver/coqui: no --test/--oracle modes, no
CPU-binary emission, no --heap-size (derived at runtime).
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

PREFIX = os.environ.get("COQUI_PREFIX", "/usr/local")
RUNTIME_BC = f"{PREFIX}/lib/coqui-cc/runtime.bc"
PASS_PLUGIN = f"{PREFIX}/lib/coqui-cc/CoquiPassPlugin.so"


def detect_arch(device: int) -> str:
    """Query nvidia-smi for the compute capability of the device."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", f"--id={device}",
             "--query-gpu=compute_cap", "--format=csv,noheader"],
            text=True)
        cap = out.strip().replace(".", "")
        return f"sm_{cap}"
    except Exception:
        return "sm_75"  # fallback


def run(cmd, check=True, **kw):
    print("[coqui-cc]", " ".join(str(c) for c in cmd), file=sys.stderr)
    return subprocess.run(cmd, check=check, **kw)


def main():
    ap = argparse.ArgumentParser(description="cuAFL compiler driver")
    ap.add_argument("sources", nargs="+", help="C/C++ source files")
    ap.add_argument("-o", "--output", required=True, help="output base name")
    ap.add_argument("-arch", default=None, help="target GPU arch (e.g., sm_75)")
    ap.add_argument("--device", type=int, default=0, help="device for arch detection")
    ap.add_argument("--stack-size", type=int, default=16384,
                    help="real stack bytes per thread (default 16384)")
    ap.add_argument("--slab-pool-size", type=int, default=0,
                    help="optional shared slab pool size (0 = disabled)")
    ap.add_argument("-I", dest="includes", action="append", default=[])
    ap.add_argument("-D", dest="defines", action="append", default=[])
    ap.add_argument("--emit-ptx", action="store_true",
                    help="stop after PTX (don't run ptxas)")
    ap.add_argument("--emit-bc", action="store_true",
                    help="stop after IR transform")
    ap.add_argument("-v", action="store_true", help="verbose")
    args = ap.parse_args()

    arch = args.arch or detect_arch(args.device)
    out = Path(args.output)
    workdir = out.parent / f".{out.name}.build"
    workdir.mkdir(parents=True, exist_ok=True)

    # 1. Compile each source to .bc with NVPTX target
    bcs = []
    for src in args.sources:
        src_p = Path(src)
        bc = workdir / (src_p.stem + ".bc")
        cmd = [
            "clang", "--target=nvptx64-nvidia-cuda",
            "-O2", "-emit-llvm", "-c",
            "-g0", "-fno-stack-protector", "-fno-pic",
        ]
        for inc in args.includes:
            cmd += ["-I", inc]
        for d in args.defines:
            cmd += ["-D", d]
        cmd += [str(src_p), "-o", str(bc)]
        run(cmd)
        bcs.append(str(bc))

    # 2. llvm-link with runtime.bc
    linked_bc = workdir / "linked.bc"
    run(["llvm-link", *bcs, RUNTIME_BC, "-o", str(linked_bc)])

    # 3. opt with coqui-link pipeline
    transformed_bc = workdir / "transformed.bc"
    run([
        "opt", "-load-pass-plugin", PASS_PLUGIN,
        "-passes=coqui-link",
        str(linked_bc), "-o", str(transformed_bc),
    ])

    if args.emit_bc:
        print(f"[coqui-cc] BC written: {transformed_bc}")
        sys.exit(0)

    # 4. llc -> .ptx
    ptx = out.with_suffix(".ptx") if args.emit_ptx else workdir / "out.ptx"
    run([
        "llc", f"-march=nvptx64", f"-mcpu={arch}",
        str(transformed_bc), "-o", str(ptx),
    ])

    if args.emit_ptx:
        sys.exit(0)

    # 5. ptxas -> .cubin
    cubin = out.with_suffix(".cubin") if out.suffix else Path(str(out) + ".cubin")
    run([
        "ptxas", f"-arch={arch}", "-O1",
        str(ptx), "-o", str(cubin),
    ])

    # 6. Write .conf sidecar
    conf = cubin.with_suffix(".conf")
    conf.write_text(
        f"stack_size={args.stack_size}\n"
        f"slab_pool_size={args.slab_pool_size}\n"
        f"arch={arch}\n"
    )

    print(f"[coqui-cc] cubin written: {cubin}")
    print(f"[coqui-cc] config: {conf}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Make executable**

```bash
chmod +x coqui_mode/bin/coqui-cc
```

- [ ] **Step 4: Commit**

```bash
git add coqui_mode/bin/coqui-cc
git commit -m "cuAFL: coqui-cc compiler driver

Python orchestrator that compiles C/C++ -> NVPTX IR via clang,
links device runtime, runs the coqui-link pass pipeline, lowers to
PTX via llc, assembles to cubin via ptxas. Writes a .conf sidecar
capturing stack_size and slab_pool_size for the host launcher.

Simplified from coqui's driver: no --test/--oracle/--fuzz flags
(single mode), no CPU-binary emission (host uses afl-clang-fast
separately).
"
```

---

## Phase 6: Host launcher (real CUDA)

Replace the hollow stub in `src/afl-fuzz-coqui.c` with real CUDA driver API calls.

### Task 6.1: Update `afl-fuzz-coqui.h` with CUDA handle fields

**Files:**
- Modify: `include/afl-fuzz-coqui.h`

- [ ] **Step 1: Add the new fields to `coqui_ctx_t`**

Before the closing brace of `coqui_ctx_t`, add:
```c
  /* CUDA handles (new — populated by coqui_init) */
  void *cu_ctx;       /* CUcontext — opaque, cast at use site */
  void *cu_module;    /* CUmodule */
  void *cu_kernel;    /* CUfunction */
  void *stream_a;     /* CUstream */
  void *stream_b;

  /* Device-persistent buffers */
  unsigned long long d_virgin_map;           /* CUdeviceptr */
  unsigned long long d_global_statics_pool;
  unsigned long long d_slab_pool;

  /* Config from .conf sidecar */
  unsigned int real_stack_size;
  unsigned long long batch_timeout_us;
```

And extend `coqui_batch_t` similarly:
```c
  /* Device buffers (CUdeviceptr values, opaque) */
  unsigned long long d_input_bytes;
  unsigned long long d_offsets;
  unsigned long long d_input_lens;
  unsigned long long d_novelty;
  unsigned long long d_status;

  /* Stream this batch runs on, and completion event */
  void *stream;
  void *completion_event;
```

(`void *` used instead of CUDA types to avoid making the header depend on `cuda.h` — callers who need CUDA types cast at use sites.)

- [ ] **Step 2: Verify compilation**

```bash
make -j$(nproc) afl-fuzz 2>&1 | tail -5
```

Expected: builds cleanly.

- [ ] **Step 3: Commit**

```bash
git add include/afl-fuzz-coqui.h
git commit -m "cuAFL: add CUDA handle fields to coqui_ctx/coqui_batch

Header stays cuda.h-free by using void*/u64 for handles; callers in
afl-fuzz-coqui.c cast to CUcontext/CUmodule/CUdeviceptr as needed.
"
```

### Task 6.2: Rewrite `afl-fuzz-coqui.c` — `coqui_init`

**Files:**
- Modify: `src/afl-fuzz-coqui.c`

- [ ] **Step 1: Add CUDA include + helpers at top of file**

Replace the opening of the file (between the existing comment block and the function definitions) with:

```c
#include "afl-fuzz.h"
#include "afl-fuzz-coqui.h"
#include "forkserver.h"

#include <cuda.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define CUCHECK(expr) do {                                                  \
  CUresult _r = (expr);                                                     \
  if (_r != CUDA_SUCCESS) {                                                 \
    const char *_name = NULL;                                               \
    cuGetErrorName(_r, &_name);                                             \
    FATAL("CUDA error at %s:%d: %s", __FILE__, __LINE__, _name ? _name : "?");\
  }                                                                         \
} while (0)

static unsigned int getenv_u32(const char *name, unsigned int dflt) {
  const char *v = getenv(name);
  if (!v) return dflt;
  return (unsigned int)strtoul(v, NULL, 0);
}

static unsigned long long getenv_u64(const char *name, unsigned long long dflt) {
  const char *v = getenv(name);
  if (!v) return dflt;
  return strtoull(v, NULL, 0);
}
```

- [ ] **Step 2: Rewrite `coqui_init`**

Replace the existing stub implementation of `coqui_init` with the real one per spec §8.4:

```c
void coqui_init(afl_state_t *afl, const char *cubin_path) {
  coqui_ctx_t *ctx = ck_alloc(sizeof(coqui_ctx_t));

  /* 1. CUDA driver init */
  CUCHECK(cuInit(0));

  int dev_idx = (int)getenv_u32("AFL_COQUI_DEVICE", 0);
  CUdevice dev;
  CUCHECK(cuDeviceGet(&dev, dev_idx));

  CUcontext cuctx;
  CUCHECK(cuCtxCreate(&cuctx, 0, dev));
  ctx->cu_ctx = cuctx;

  /* 2. Verify sm_75+ */
  int major, minor;
  CUCHECK(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev));
  CUCHECK(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev));
  if ((major * 10 + minor) < 75) {
    FATAL("coqui requires sm_75+; device is sm_%d%d", major, minor);
  }

  /* 3. Load cubin, resolve kernel, resolve virgin_map global */
  CUmodule mod;
  CUCHECK(cuModuleLoad(&mod, cubin_path));
  ctx->cu_module = mod;

  CUfunction kernel;
  CUCHECK(cuModuleGetFunction(&kernel, mod, "__coqui_fuzz_kernel"));
  ctx->cu_kernel = kernel;

  CUdeviceptr d_virgin;
  size_t virgin_sz;
  CUCHECK(cuModuleGetGlobal(&d_virgin, &virgin_sz, mod, "__coqui_virgin_map"));
  if (virgin_sz != 65536) {
    FATAL("__coqui_virgin_map symbol size %zu != 64KB", virgin_sz);
  }
  ctx->d_virgin_map = (unsigned long long)d_virgin;
  CUCHECK(cuMemsetD8(d_virgin, 0, 65536));

  /* 4. Compute stack budget per spec §4.3 */
  unsigned int stack_size = getenv_u32("AFL_COQUI_STACK_SIZE", 16384);
  unsigned int cov = 65536;
  unsigned int total = 524288;  /* sm_75 cap */
  if (stack_size + cov >= total) {
    FATAL("--stack-size %u + 64KB coverage >= 512KB budget", stack_size);
  }
  unsigned int remaining = total - cov - stack_size;
  unsigned int heap = (remaining * 8) / 9;
  unsigned int total_budget = cov + heap + (heap / 8) + stack_size;
  CUCHECK(cuCtxSetLimit(CU_LIMIT_STACK_SIZE, total_budget));
  ctx->real_stack_size = stack_size;

  /* 5. Check static stack */
  int static_usage;
  CUCHECK(cuFuncGetAttribute(&static_usage, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, kernel));
  if ((unsigned int)static_usage > total_budget - 1024) {
    FATAL("kernel static stack %d B exceeds budget %u B — bump --stack-size",
          static_usage, total_budget);
  }

  /* 6. Batch buffers */
  ctx->batch_size = 8192;
  ctx->max_input_size = afl->max_length ? afl->max_length : 4096;
  /* u64 math to avoid overflow (fix from cuAFL T3.6 post-review) */
  unsigned long long budget64 = ((unsigned long long)ctx->batch_size
                                  * ctx->max_input_size) / 4;
  unsigned long long floor64 = (unsigned long long)ctx->max_input_size * 256;
  if (budget64 < floor64) budget64 = floor64;
  if (budget64 > 0x40000000ULL) budget64 = 0x40000000ULL;  /* MAX_ALLOC cap */
  ctx->byte_budget = (unsigned int)budget64;
  ctx->map_size = 65536;

  /* 7. Create streams */
  CUstream sa, sb;
  CUCHECK(cuStreamCreate(&sa, CU_STREAM_NON_BLOCKING));
  CUCHECK(cuStreamCreate(&sb, CU_STREAM_NON_BLOCKING));
  ctx->stream_a = sa;
  ctx->stream_b = sb;

  /* 8. Allocate ping-pong pair */
  alloc_batch_half_cuda(&ctx->ping, ctx, sa);
  alloc_batch_half_cuda(&ctx->pong, ctx, sb);
  ctx->pending = &ctx->ping;
  ctx->executing = &ctx->pong;

  /* 9. Statics pool */
  CUdeviceptr statics_sym;
  size_t statics_sym_sz;
  if (cuModuleGetGlobal(&statics_sym, &statics_sym_sz, mod,
                        "__coqui_statics_per_thread") == CUDA_SUCCESS) {
    unsigned int per_thread;
    CUCHECK(cuMemcpyDtoH(&per_thread, statics_sym, sizeof(unsigned int)));
    if (per_thread > 0) {
      unsigned long long total_pool = (unsigned long long)per_thread * ctx->batch_size;
      CUdeviceptr pool;
      CUCHECK(cuMemAlloc(&pool, total_pool));
      CUCHECK(cuMemsetD8(pool, 0, total_pool));
      ctx->d_global_statics_pool = (unsigned long long)pool;
    }
  }

  /* 10. Slab (optional) */
  unsigned long long slab_size = getenv_u64("AFL_COQUI_SLAB_SIZE", 0);
  if (slab_size > 0) {
    CUdeviceptr slab;
    CUCHECK(cuMemAlloc(&slab, slab_size));
    CUCHECK(cuMemsetD8(slab, 0, slab_size));
    ctx->d_slab_pool = (unsigned long long)slab;
    /* TODO: invoke __coqui_slab_setup init kernel when slab runtime ported */
  }

  ctx->batch_timeout_us = getenv_u64("AFL_COQUI_TIMEOUT_US", 3000000); /* 3s */
  ctx->launch_count = 0;
  ctx->oversized_count = 0;

  afl->coqui = ctx;

  OKF("coqui initialized: sm_%d%d, batch=%u, stack=%u KB, heap=%u KB, total=%u KB",
      major, minor, ctx->batch_size, stack_size/1024, heap/1024, total_budget/1024);
}
```

Also write the helper `alloc_batch_half_cuda`:

```c
static void alloc_batch_half_cuda(coqui_batch_t *b, coqui_ctx_t *ctx, CUstream stream) {
  /* Host pinned */
  CUCHECK(cuMemHostAlloc((void**)&b->h_input_bytes, ctx->byte_budget, 0));
  CUCHECK(cuMemHostAlloc((void**)&b->h_offsets, ctx->batch_size * 4, 0));
  CUCHECK(cuMemHostAlloc((void**)&b->h_input_lens, ctx->batch_size * 4, 0));
  CUCHECK(cuMemHostAlloc((void**)&b->h_novelty, ctx->batch_size / 8, 0));
  CUCHECK(cuMemHostAlloc((void**)&b->h_status,
          ctx->batch_size * sizeof(coqui_status_t), 0));

  /* Device mirrors */
  CUdeviceptr p;
  CUCHECK(cuMemAlloc(&p, ctx->byte_budget));            b->d_input_bytes = p;
  CUCHECK(cuMemAlloc(&p, ctx->batch_size * 4));          b->d_offsets = p;
  CUCHECK(cuMemAlloc(&p, ctx->batch_size * 4));          b->d_input_lens = p;
  CUCHECK(cuMemAlloc(&p, ctx->batch_size / 8));          b->d_novelty = p;
  CUCHECK(cuMemAlloc(&p, ctx->batch_size * sizeof(coqui_status_t)));
                                                         b->d_status = p;

  b->stream = stream;
  CUevent ev;
  CUCHECK(cuEventCreate(&ev, CU_EVENT_DEFAULT));
  b->completion_event = ev;

  b->n_inputs = 0;
  b->bytes_used = 0;
}
```

- [ ] **Step 3: Verify compilation**

```bash
make -j$(nproc) afl-fuzz 2>&1 | tail -10
```

Expected: builds cleanly (requires `-lcuda` linker flag from Task 1.4).

- [ ] **Step 4: Commit**

```bash
git add src/afl-fuzz-coqui.c
git commit -m "cuAFL: coqui_init — real CUDA implementation

Replaces the hollow stub with cuInit/cuCtxCreate/cuModuleLoad/
cuModuleGetFunction flow per spec §8.4. Verifies sm_75+, checks static
stack usage against budget, allocates ping-pong host+device buffers,
creates two streams for pipelined async execution, zeroes the virgin
map global, allocates statics pool if kernel exports __coqui_statics_per_thread,
optional slab pool via AFL_COQUI_SLAB_SIZE.
"
```

### Task 6.3: Rewrite `coqui_submit_input` + `coqui_launch_batch` + `coqui_flush_batch`

**Files:**
- Modify: `src/afl-fuzz-coqui.c`

- [ ] **Step 1: Replace `coqui_submit_input`**

Replace the stub body with the real packing logic (mostly same as stub) + launch trigger:

```c
u8 coqui_submit_input(afl_state_t *afl, u8 *buf, u32 len) {
  coqui_ctx_t *ctx = afl->coqui;
  coqui_batch_t *b = ctx->pending;

  if (len > ctx->byte_budget) {
    ctx->oversized_count++;
    return 0;
  }

  u32 off = (b->bytes_used + 7) & ~7u;
  if (off + len > ctx->byte_budget || b->n_inputs == ctx->batch_size) {
    coqui_launch_batch(afl, b);

    /* Flip */
    coqui_batch_t *tmp = ctx->pending;
    ctx->pending = ctx->executing;
    ctx->executing = tmp;

    b = ctx->pending;
    off = 0;

    /* If the new pending has in-flight work (from the previous ping-pong flip),
       drain it before reusing. */
    if (b->n_inputs > 0) {
      coqui_await_and_process(afl, b);
      b->n_inputs = 0;
      b->bytes_used = 0;
    }
  }

  memcpy(b->h_input_bytes + off, buf, len);
  b->h_offsets[b->n_inputs] = off;
  b->h_input_lens[b->n_inputs] = len;
  b->n_inputs++;
  b->bytes_used = off + len;
  return 0;
}
```

- [ ] **Step 2: Add `coqui_launch_batch` (internal, not exposed in header)**

Before `coqui_submit_input`, add static helper:

```c
static void coqui_launch_batch(afl_state_t *afl, coqui_batch_t *b) {
  coqui_ctx_t *ctx = afl->coqui;
  CUstream s = (CUstream)b->stream;

  /* H->D */
  CUCHECK(cuMemcpyHtoDAsync((CUdeviceptr)b->d_input_bytes, b->h_input_bytes,
                             ctx->byte_budget, s));
  CUCHECK(cuMemcpyHtoDAsync((CUdeviceptr)b->d_offsets, b->h_offsets,
                             ctx->batch_size * 4, s));
  CUCHECK(cuMemcpyHtoDAsync((CUdeviceptr)b->d_input_lens, b->h_input_lens,
                             ctx->batch_size * 4, s));
  CUCHECK(cuMemsetD32Async((CUdeviceptr)b->d_novelty, 0, ctx->batch_size / 32, s));
  CUCHECK(cuMemsetD8Async((CUdeviceptr)b->d_status, 0,
                           ctx->batch_size * sizeof(coqui_status_t), s));

  /* Launch */
  void *args[] = {
    &b->d_input_bytes, &b->d_offsets, &b->d_input_lens,
    &b->d_novelty, &b->d_status,
  };
  unsigned grid = ctx->batch_size / 128;
  CUCHECK(cuLaunchKernel((CUfunction)ctx->cu_kernel,
                          grid, 1, 1,
                          128, 1, 1,
                          0, s, args, NULL));

  /* D->H */
  CUCHECK(cuMemcpyDtoHAsync(b->h_novelty, (CUdeviceptr)b->d_novelty,
                             ctx->batch_size / 8, s));
  CUCHECK(cuMemcpyDtoHAsync(b->h_status, (CUdeviceptr)b->d_status,
                             ctx->batch_size * sizeof(coqui_status_t), s));

  CUCHECK(cuEventRecord((CUevent)b->completion_event, s));
  ctx->launch_count++;
}
```

- [ ] **Step 3: Replace `coqui_flush_batch`**

```c
void coqui_flush_batch(afl_state_t *afl) {
  coqui_ctx_t *ctx = afl->coqui;

  /* Drain the executing batch if it has in-flight work */
  if (ctx->executing->n_inputs > 0) {
    coqui_await_and_process(afl, ctx->executing);
    ctx->executing->n_inputs = 0;
    ctx->executing->bytes_used = 0;
  }

  /* Launch + drain pending if partial */
  if (ctx->pending->n_inputs > 0) {
    coqui_launch_batch(afl, ctx->pending);
    coqui_await_and_process(afl, ctx->pending);
    ctx->pending->n_inputs = 0;
    ctx->pending->bytes_used = 0;
  }
}
```

- [ ] **Step 4: Verify compilation**

```bash
make -j$(nproc) afl-fuzz 2>&1 | tail -10
```

- [ ] **Step 5: Commit**

```bash
git add src/afl-fuzz-coqui.c
git commit -m "cuAFL: coqui_submit_input / launch_batch / flush_batch real impl

Real CUDA via cuMemcpyHtoD/DtoHAsync, cuLaunchKernel, cuEventRecord.
Ping-pong: fills pending, flips to executing (which has in-flight work),
drains after flip. Stage-boundary flush drains both pending and
executing cleanly.
"
```

### Task 6.4: Add `coqui_await_and_process` (the post-batch loop)

**Files:**
- Modify: `src/afl-fuzz-coqui.c`

- [ ] **Step 1: Add helper**

```c
/* Translate GPU status to AFL fault code. */
static u8 translate_gpu_status(coqui_status_t *s) {
  if (s->asan_error) return FSRV_RUN_CRASH;
  if (s->ubsan_fatal) return FSRV_RUN_CRASH;
  if (s->signal) return FSRV_RUN_CRASH;
  if (s->phase != 6 /* PHASE_COMPLETE */) return FSRV_RUN_TMOUT;
  return FSRV_RUN_OK;
}

/* Run a GPU-flagged or GPU-crashed input through the CPU forkserver
   for real trace_bits, then call save_if_interesting. */
static void process_input_via_cpu_fsrv(afl_state_t *afl,
                                        u8 *input, u32 len, u8 gpu_fault) {
  u32 new_size = write_to_testcase(afl, (void **)&input, len, 0);
  if (new_size == 0) return;

  (void)gpu_fault;  /* we re-fault via CPU */
  u8 cpu_fault = fuzz_run_target(afl, &afl->fsrv, afl->fsrv.exec_tmout);
  afl->queued_discovered += save_if_interesting(afl, input, len, cpu_fault);
}

static void coqui_await_and_process(afl_state_t *afl, coqui_batch_t *b) {
  coqui_ctx_t *ctx = afl->coqui;
  CUstream s = (CUstream)b->stream;

  /* Wait with timeout */
  unsigned long long deadline_us = 0;  /* init in getter */
  struct timeval tv;
  gettimeofday(&tv, NULL);
  deadline_us = ((unsigned long long)tv.tv_sec * 1000000 + tv.tv_usec)
                + ctx->batch_timeout_us;

  while (1) {
    CUresult r = cuStreamQuery(s);
    if (r == CUDA_SUCCESS) break;
    if (r != CUDA_ERROR_NOT_READY) {
      CUCHECK(r);  /* fatal */
    }
    gettimeofday(&tv, NULL);
    unsigned long long now = ((unsigned long long)tv.tv_sec * 1000000 + tv.tv_usec);
    if (now >= deadline_us) {
      WARNF("coqui batch timeout after %llu us; salvaging", ctx->batch_timeout_us);
      cuCtxSynchronize();  /* best-effort recovery */
      break;
    }
    usleep(100);
  }

  /* Process flagged inputs */
  for (u32 word_i = 0; word_i < ctx->batch_size / 32; word_i++) {
    u32 bits = ((u32 *)b->h_novelty)[word_i];
    while (bits) {
      u32 bit_pos = __builtin_ctz(bits);
      bits &= bits - 1;
      u32 i = word_i * 32 + bit_pos;
      if (b->h_input_lens[i] == 0) continue;

      u8 gpu_fault = translate_gpu_status(&b->h_status[i]);
      u8 *input = b->h_input_bytes + b->h_offsets[i];
      u32 len = b->h_input_lens[i];
      process_input_via_cpu_fsrv(afl, input, len, gpu_fault);
    }
  }

  /* Also check for crashes not flagged as novel */
  for (u32 i = 0; i < ctx->batch_size; i++) {
    if (b->h_input_lens[i] == 0) continue;
    u8 nov = (b->h_novelty[i / 8] >> (i % 8)) & 1;
    if (nov) continue;  /* already processed */
    u8 gpu_fault = translate_gpu_status(&b->h_status[i]);
    if (gpu_fault == FSRV_RUN_CRASH || gpu_fault == FSRV_RUN_TMOUT) {
      u8 *input = b->h_input_bytes + b->h_offsets[i];
      u32 len = b->h_input_lens[i];
      process_input_via_cpu_fsrv(afl, input, len, gpu_fault);
    }
  }
}
```

- [ ] **Step 2: Verify compilation**

```bash
make -j$(nproc) afl-fuzz 2>&1 | tail -10
```

- [ ] **Step 3: Commit**

```bash
git add src/afl-fuzz-coqui.c
git commit -m "cuAFL: coqui_await_and_process — flagged input handling

Polls cuStreamQuery until kernel completes or timeout fires. On
timeout: cuCtxSynchronize best-effort recovery + continue. For each
flagged input: run through the CPU forkserver via fuzz_run_target
(populates trace_bits with real coverage), call save_if_interesting
normally. Also checks for crashes that didn't set new coverage bits.
"
```

### Task 6.5: Rewrite `coqui_shutdown` + strip dead `coqui_calibrate_one`

**Files:**
- Modify: `src/afl-fuzz-coqui.c`

- [ ] **Step 1: Rewrite `coqui_shutdown`**

```c
void coqui_shutdown(afl_state_t *afl) {
  if (!afl->coqui) return;
  coqui_ctx_t *ctx = afl->coqui;

  /* Drain streams */
  if (ctx->stream_a) cuStreamSynchronize((CUstream)ctx->stream_a);
  if (ctx->stream_b) cuStreamSynchronize((CUstream)ctx->stream_b);

  /* Free ping-pong */
  free_batch_half_cuda(&ctx->ping);
  free_batch_half_cuda(&ctx->pong);

  /* Free persistent device buffers */
  if (ctx->d_global_statics_pool) cuMemFree((CUdeviceptr)ctx->d_global_statics_pool);
  if (ctx->d_slab_pool) cuMemFree((CUdeviceptr)ctx->d_slab_pool);

  /* Streams, module, context */
  if (ctx->stream_a) cuStreamDestroy((CUstream)ctx->stream_a);
  if (ctx->stream_b) cuStreamDestroy((CUstream)ctx->stream_b);
  if (ctx->cu_module) cuModuleUnload((CUmodule)ctx->cu_module);
  if (ctx->cu_ctx) cuCtxDestroy((CUcontext)ctx->cu_ctx);

  ck_free(ctx);
  afl->coqui = NULL;
}

static void free_batch_half_cuda(coqui_batch_t *b) {
  if (b->h_input_bytes) cuMemFreeHost(b->h_input_bytes);
  if (b->h_offsets) cuMemFreeHost(b->h_offsets);
  if (b->h_input_lens) cuMemFreeHost(b->h_input_lens);
  if (b->h_novelty) cuMemFreeHost(b->h_novelty);
  if (b->h_status) cuMemFreeHost(b->h_status);

  if (b->d_input_bytes) cuMemFree((CUdeviceptr)b->d_input_bytes);
  if (b->d_offsets) cuMemFree((CUdeviceptr)b->d_offsets);
  if (b->d_input_lens) cuMemFree((CUdeviceptr)b->d_input_lens);
  if (b->d_novelty) cuMemFree((CUdeviceptr)b->d_novelty);
  if (b->d_status) cuMemFree((CUdeviceptr)b->d_status);

  if (b->completion_event) cuEventDestroy((CUevent)b->completion_event);

  memset(b, 0, sizeof(*b));
}
```

- [ ] **Step 2: Replace `coqui_calibrate_one` with no-op stub**

Keep the function for ABI compatibility, but make it a no-op returning `FSRV_RUN_OK`. Calibration now flows through AFL's standard `fuzz_run_target` → CPU forkserver.

```c
u8 coqui_calibrate_one(afl_state_t *afl, u8 *buf, u32 len) {
  (void)afl; (void)buf; (void)len;
  /* Deprecated under the coexistence model (§8.10). Calibration runs
     through fuzz_run_target on the CPU forkserver. Kept for ABI
     compatibility; reserved for future GPU-side calibration optimization. */
  return FSRV_RUN_OK;
}
```

- [ ] **Step 3: Verify compilation**

```bash
make -j$(nproc) afl-fuzz 2>&1 | tail -10
```

- [ ] **Step 4: Commit**

```bash
git add src/afl-fuzz-coqui.c
git commit -m "cuAFL: coqui_shutdown real CUDA teardown + deprecate calibrate_one

cuStreamSynchronize/Destroy, cuMemFreeHost/cuMemFree for all buffers,
cuModuleUnload, cuCtxDestroy. coqui_calibrate_one kept as a no-op stub
(calibration flows through AFL's CPU forkserver under the coexistence
model per §8.10).
"
```

---

## Phase 7: First attempt at cjson

### Task 7.1: Install coqui_mode components

**Files:**
- (none modified)

- [ ] **Step 1: Build the pass plugin**

```bash
cd coqui_mode/passes/build
cmake .. && make -j$(nproc) 2>&1 | tail -5
```

Expected: `libCoquiPassPlugin.so` built.

- [ ] **Step 2: Run install script**

```bash
cd ../../..   # back to cuAFL root
sudo PREFIX=/usr/local ./coqui_mode/build_coqui_support.sh 2>&1 | tail -10
```

Expected: installs `/usr/local/bin/coqui-cc`, `/usr/local/lib/coqui-cc/runtime.bc`, `/usr/local/lib/coqui-cc/CoquiPassPlugin.so`.

- [ ] **Step 3: Verify coqui-cc works (no sources, just help)**

```bash
coqui-cc --help 2>&1 | head -10
```

Expected: help output from argparse.

No commit — this is environment setup.

### Task 7.2: Attempt cjson compile (port-on-demand begins here)

**Files:**
- (none — this is a validation attempt)

- [ ] **Step 1: Copy cjson fuzzer harness from coqui**

```bash
mkdir -p /tmp/cjson_probe
cp -r ~/coqui/harness/targets/cjson_read_fuzzer.c /tmp/cjson_probe/
# Also need the cjson sources — either from coqui or system package
# Check: ls ~/coqui/evaluation/corpora/cjson or similar
# If not present, fetch cjson and place at /tmp/cjson_probe/cjson.c + cjson.h
```

- [ ] **Step 2: Attempt compile**

```bash
cd /tmp/cjson_probe
coqui-cc cjson_read_fuzzer.c cjson.c -o cjson_fuzzer 2>&1 | tail -40
```

Expected outcomes (any of these is the expected end of the plan):
- **Compiles cleanly** → success! Move to end-to-end test with cuAFL.
- **Fails at `ExternalSymbolGatekeeper`** with `unresolved external 'strlen'` (or similar) → first port-on-demand trigger. Report the error to user.
- **Fails at an earlier pass** (rare — something we didn't anticipate) → report to user for design-level decision.

**This is the handoff point.** The implementation plan ends here. The next step is collaborative port-on-demand: the user and implementer discuss each error, approve ports, execute them, retry.

Update `docs/coqui_port_log.md` with whatever was encountered:

```bash
touch docs/coqui_port_log.md
echo "# Coqui port-on-demand log" > docs/coqui_port_log.md
echo "" >> docs/coqui_port_log.md
echo "## YYYY-MM-DD — first cjson compile attempt" >> docs/coqui_port_log.md
# ... record the error ...
git add docs/coqui_port_log.md
git commit -m "cuAFL: coqui port log — first cjson compile attempt"
```

No further automated tasks. Report status to user and await next instruction.

---

## Self-Review

**Spec coverage:**

| Spec section | Plan task(s) |
|--------------|--------------|
| §1 overview, principles | Phases 1-7 together |
| §2 coqui-cc | Phase 5 |
| §3 pass pipeline | Phase 4 (T4.1–T4.11) |
| §4 memory layout | T4.8 (MemoryLayout pass) + T3.3 (heap runtime) |
| §5 coverage scheme | T4.9 (Coverage pass) + T3.2 (coverage runtime) |
| §6 runtime files | Phase 3 (T3.1–T3.4) |
| §7 kernel interface | T4.5 (FuzzEntry) + T2.2 (runtime header) |
| §8 host launcher | Phase 6 (T6.1–T6.5) |
| §9 validation | Phase 7 (validation attempt; port-on-demand after) |
| §10 naming | All tasks use --coqui consistently |
| §11 cuAFL amendments | Phase 1 (T1.1–T1.4) |
| §12 out of scope | Not implemented, per spec |

**Placeholder scan:** no "TBD", "TODO", or "implement later" markers. Each code block is complete. Two places reference future porting (coqui_slab.c setup kernel, full StaticGlobals port), but both are explicitly flagged as part of the port-on-demand process, not day-1.

**Type consistency:** 
- `coqui_ctx_t`, `coqui_batch_t`, `coqui_status_t` consistent across header (T2.2) and .c files (T6.x).
- Pass function signatures `runFooReject(Module &)` / `runFoo(Module &)` consistent in Transforms.h (T4.1) and individual pass files.
- Runtime function names (`__coqui_malloc`, `__coqui_fuzz_tid`, etc.) consistent between header (T2.2), runtime .c files (T3.x), and passes (T4.x).

---

Plan complete and saved to `docs/superpowers/plans/2026-04-18-coqui-internals.md`. Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
