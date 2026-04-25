# coqui mode bzip2 evaluation target

Adapts the libbzip2 1.0.8 decompression fuzzer so it can be driven
with coqui mode's `afl-fuzz --coqui`.

- Upstream: https://github.com/libarchive/bzip2 (pinned to tag
  `bzip2-1.0.8`, commit `6a8690fc8d26c815e798c588f796eabe9d684cf0`)
- Harness: `harness.c` (libFuzzer-style; drives
  `BZ2_bzDecompressInit` / `BZ2_bzDecompress` / `BZ2_bzDecompressEnd`)

## Build / run

```
./build.sh   # fetches libbzip2 + builds .cubin + CPU binary + seed
./fuzz.sh    # prints launch commands for Main/Secondary/Coqui
```

`build.sh` is self-contained: no external dependencies beyond
`afl-clang-fast` (this repo), `coqui-cc` (at `/usr/local/bin`), `git`,
and the host `bzip2` binary (to synthesize the minimal seed). It
clones libbzip2 from GitHub into `.build/`, compiles the CPU binary
with this repo's own `afl-clang-fast`, and builds the GPU cubin via
`coqui-cc` under a `/tmp/coqui-cc.lock` flock so ptxas can serialize
with any parallel target builds.

## Layout

- `build.sh` / `fuzz.sh` — entry points (set -euo pipefail).
- `harness.c` — in-tree libFuzzer-style harness.
- `bz2_assert_stub.c` — local `bz_internal_error()` stub that calls
  `__coqui_trap()` (PTX `trap; exit;`) instead of libc `abort()`.
  coqui mode's `ExternalSymbolGatekeeper` only accepts `__coqui_*` /
  `__llvm_*` externals, so the upstream `abort`-based stub fails the
  build.
- `seeds/min.bz2` — minimal bzip2 stream, generated in-place if
  missing by `build.sh` via `printf 'a' | bzip2 -9`.
- `.gitignore` — excludes build artifacts + `.build/` cache.
- `.build/` (ignored) — cached libbzip2 source clone.
- After a successful build:
  - `bzip2_fuzzer.cubin`, `bzip2_fuzzer.conf` (from `coqui-cc`)
  - `bzip2_fuzzer_cpu` (AFL++-instrumented binary)
  - `out/` — created by `fuzz.sh`; AFL output dir

## Notes

- `--arch sm_75` is the `build.sh` default (override with
  `ARCH=sm_XX ./build.sh` to target a different GPU).
- `--stack-size 32768` (coqui-cc default).
- `--slab-pool-size 2147483648` (2 GiB) — bzip2's DState (~60 KB)
  plus the smallest ll16/ll4 buffer don't fit in the default 64 KB
  per-thread heap.
- Sources compiled: `blocksort.c`, `huffman.c`, `crctable.c`,
  `randtable.c`, `compress.c`, `decompress.c`, `bzlib.c`, local
  `bz2_assert_stub.c`, `harness.c`. `-D BZ_NO_STDIO` is set on both
  CPU and GPU builds.
- coqui-cc does NOT accept `--heap-size` or `--batch-size`. Heap is
  derived at runtime from the per-thread stack budget (the runtime
  prints the resolved `stack=/heap=/total=` triple at startup); batch
  size is read from `AFL_COQUI_BATCH_SIZE` at fuzz time (default
  8192).
- `fuzz.sh` sets `AFL_I_DONT_CARE_ABOUT_MISSING_CRASHES=1`,
  `AFL_SKIP_CPUFREQ=1`, `AFL_NO_UI=1` per the
  project convention; override by exporting them yourself. GPU device
  index is controlled by `AFL_COQUI_DEVICE` (default 0).
