# coqui mode zstd evaluation target

Adapts the Facebook zstd decompression fuzzer so it can be driven with
coqui mode's `afl-fuzz --coqui`.

- Upstream: https://github.com/facebook/zstd (pinned to tag `v1.5.6`,
  commit `35016bc1c0b9a2f7121b7ecc312100aad7d9f2ad`)
- Harness: `harness.c` (libFuzzer-style; drives decompression of a
  simple zstd frame; `abort()` on content-size mismatch signals a
  real bug)

## Build / run

```
./build.sh   # fetches zstd + builds .cubin + CPU binary
./fuzz.sh    # prints launch commands for Main/Secondary/Coqui
```

`build.sh` is self-contained: no external dependencies beyond
`afl-clang-fast` (this repo) and `coqui-cc` (at `/usr/local/bin`). It
fetches the zstd source tarball from GitHub into `.build/`, compiles
the CPU binary with this repo's own `afl-clang-fast`, and builds the
GPU cubin via `coqui-cc` under a `/tmp/coqui-cc.lock` flock so ptxas
can serialize with any parallel target builds.

## Layout

- `build.sh` / `fuzz.sh` — entry points (set -euo pipefail).
- `harness.c` — in-tree libFuzzer-style harness.
- `zstd_abort_stub.c` — local `abort()` → `__coqui_trap()` stub.
  The harness calls `abort()` on content-size mismatch (real bug);
  coqui mode's `ExternalSymbolGatekeeper` only accepts `__coqui_*` /
  `__llvm_*` externals, so the upstream path needed this stub.
- `seeds/min.zst` — single minimal zstd frame.
- `.gitignore` — excludes build artifacts + `.build/` cache.
- `.build/` (ignored) — cached zstd source tarball extraction.
- After a successful build:
  - `zstd_simple_decompress_fuzzer.cubin`,
    `zstd_simple_decompress_fuzzer.conf` (from `coqui-cc`)
  - `zstd_simple_decompress_fuzzer_cpu` (AFL++-instrumented binary)
  - `out/` — created by `fuzz.sh`; AFL output dir

## Notes

- `--arch sm_75` is the `build.sh` default (override with
  `ARCH=sm_XX ./build.sh` to target a different GPU).
- `--stack-size 32768` (coqui-cc default).
- `--slab-pool-size 0` — zstd's decompress context fits in the
  per-thread heap with the buffer trimming below.
- Sources compiled (decompress-only subset):
  - `lib/common/{debug,entropy_common,error_private,fse_decompress,xxhash,zstd_common}.c`
  - `lib/decompress/{huf_decompress,zstd_ddict,zstd_decompress,zstd_decompress_block}.c`
- Key `-D` flags: `ZSTD_NO_INTRINSICS=1`, `ZSTD_DISABLE_ASM=1`,
  `XXHASH_NAMESPACE=ZSTD_`, `FUZZING_BUILD_MODE_UNSAFE_FOR_PRODUCTION`,
  `ZSTD_NO_TRACE=1`, `ZSTD_DECODER_INTERNAL_BUFFER=4096`. The last
  shrinks the `ZSTD_DCtx` from ~64 KB to ~34 KB so it fits alongside
  the 8 KB output buffer in the per-thread bump heap.
- coqui-cc does NOT accept `--heap-size` or `--batch-size`. Heap is
  derived at runtime from the per-thread stack budget; batch size
  comes from `AFL_COQUI_BATCH_SIZE` (default `COQUI_DEFAULT_BATCH_SIZE`
  in `include/afl-fuzz-coqui.h`).
- `ptxas` can take tens of minutes on this target even at `-O1`.
- Override afl-clang-fast via `CLANG_FAST=...` (not `AFL_CC=...`):
  `afl-cc` treats `AFL_CC` as the backing clang to delegate to, which
  blows past `MAX_PARAMS_NUM` if set to afl-clang-fast itself.
- `fuzz.sh` sets `AFL_I_DONT_CARE_ABOUT_MISSING_CRASHES=1`,
  `AFL_SKIP_CPUFREQ=1`, `AFL_SKIP_BIN_CHECK=1`, `AFL_NO_UI=1` per the
  project convention; override by exporting them yourself. GPU device
  index is controlled by `AFL_COQUI_DEVICE` (default 0).
