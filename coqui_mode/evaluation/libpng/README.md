# coqui mode libpng evaluation target

Adapts the libpng read-fuzzer harness so it can be driven with coqui
mode's `afl-fuzz --coqui`. libpng depends on zlib; both libraries are
compiled into a single cubin.

- Upstream: https://github.com/pnggroup/libpng (pinned to tag `v1.6.43`)
  + https://github.com/madler/zlib (pinned to tag `v1.3.1`)
- Harness: `harness.c` (libFuzzer-style `LLVMFuzzerTestOneInput`;
  exercises libpng's low-level read path plus an optional write-back
  step driven by input bytes)

## Build / run

```
./build.sh   # fetches libpng + zlib, patches pnglibconf.h, builds .cubin + CPU binary
./fuzz.sh    # prints launch commands for Main/Secondary/Coqui
```

`build.sh` is self-contained: no external dependencies beyond
`afl-clang-fast` (this repo) and `coqui-cc` (at `/usr/local/bin`). It
fetches libpng + zlib release tarballs from GitHub into `.build/`,
stages patched `pnglibconf.h` variants for CPU and GPU (keeps setjmp
for CPU; strips more feature flags for GPU), compiles the CPU binary
with this repo's own `afl-clang-fast`, and builds the GPU cubin via
`coqui-cc` under a `/tmp/coqui-cc.lock` flock so ptxas can serialize
with any parallel target builds.

## Layout

- `build.sh` / `fuzz.sh` — entry points (set -euo pipefail).
- `harness.c` — in-tree libFuzzer-style harness.
- `seeds/1x1_gray.png` — minimal 1x1 grayscale PNG seed.
- `.gitignore` — excludes build artifacts + `.build/` cache +
  `libpng_include/` + `libpng_include_gpu/`.
- `.build/` (ignored) — cached libpng + zlib source trees.
- `libpng_include/`, `libpng_include_gpu/` (ignored) — patched
  `pnglibconf.h` + symlinks to libpng's public headers.
- After a successful build:
  - `libpng_read_fuzzer.cubin`, `libpng_read_fuzzer.conf`
    (from `coqui-cc`)
  - `libpng_read_fuzzer_cpu` (AFL++-instrumented binary)
  - `out/` — created by `fuzz.sh`; AFL output dir

## Notes

- `--arch sm_75` is the `build.sh` default (override with
  `ARCH=sm_XX ./build.sh` to target a different GPU).
- `--stack-size 32768` — libpng+zlib call chains
  (`png_read_image` → … → `inflate_fast`) exceed the default 8 KB
  stack.
- `--slab-pool-size 2147483648` (2 GiB) — libpng's per-chunk
  allocations (tEXt, iCCP, zTXt) can exceed the 64 KB per-thread heap
  when fuzz mutations pump large chunk lengths. The shared slab pool
  absorbs these.
- `pnglibconf.h` patching strips feature macros that require libc
  symbols coqui mode's runtime does not provide (`PNG_SETJMP_SUPPORTED`,
  `PNG_SIMPLIFIED_*`, `PNG_CONSOLE_IO_SUPPORTED`,
  `PNG_STDIO_SUPPORTED`, `PNG_FLOATING_ARITHMETIC_SUPPORTED`). The
  floating-point strip forces libpng's fixed-point gamma path (NVPTX
  can't lower `llvm.pow.f64`).
- Every `png_error()` path with `-D PNG_NO_SETJMP` terminates in
  `abort()`, which the `Libc.cpp` pass rewrites to `__coqui_abort()`
  (→ `__coqui_trap()`; PTX `trap; exit;`) — no local stub needed.
- `ptxas` is very slow — the libpng+zlib kernel is large (~25 TUs,
  heavy ASan instrumentation, all-inline). Expect 20-30 minutes at
  `-O1`, peaking at 10-50 GB RAM. Do not run multiple `./build.sh`
  invocations concurrently.
- If you need to tune llc further (large instrumented modules can
  stall register allocation), set `COQUI_LLC_OPT=-O1` before running
  `build.sh`.
- `fuzz.sh` sets `AFL_I_DONT_CARE_ABOUT_MISSING_CRASHES=1`,
  `AFL_SKIP_CPUFREQ=1`, `AFL_NO_UI=1` per the
  project convention; override by exporting them yourself. GPU device
  index is controlled by `AFL_COQUI_DEVICE` (default 0).
