# coqui mode libjpeg-turbo evaluation target

Adapts the oss-fuzz-style libjpeg-turbo decompress fuzzer (baseline JPEG
decode only) so it can be driven with coqui mode's `afl-fuzz --coqui`.

- Upstream: https://github.com/libjpeg-turbo/libjpeg-turbo (pinned to
  tag `3.0.4`)
- Harness: `harness.c` (libFuzzer-style; drives
  `jpeg_read_header` + `jpeg_start_decompress` + `jpeg_read_scanlines`;
  rejects non-8-bit precision and images larger than 64x64 to fit the
  GPU's 64 KB bump heap)

## Build / run

```
./build.sh   # fetches libjpeg-turbo + stages patched headers + builds .cubin + CPU binary
./fuzz.sh    # prints launch commands for Main/Secondary/Coqui
```

`build.sh` is self-contained: no external dependencies beyond
`afl-clang-fast` (this repo) and `coqui-cc` (at `/usr/local/bin`). It
clones libjpeg-turbo from GitHub into `.build/`, stages a patched
`jpeg_include/` header tree (strips arithmetic coding / progressive /
multi-scan / 12- and 16-bit paths / color quantization / upsample
merging / save markers / input smoothing / block smoothing; writes
`jconfig.h` / `jconfigint.h` / `jversion.h` inline), compiles the CPU
binary per-TU with this repo's own `afl-clang-fast` (avoids afl-cc's
`MAX_PARAMS_NUM` cap), and builds the GPU cubin via `coqui-cc` under a
`/tmp/coqui-cc.lock` flock so ptxas can serialize with any parallel
target builds.

## Layout

- `build.sh` / `fuzz.sh` — entry points (set -euo pipefail).
- `harness.c` — in-tree libFuzzer-style harness.
- `libjpeg_turbo_stubs.c` — in-tree stubs for `jsimd_*` (WITH_SIMD=0)
  and `j12init_*` / `j16init_*` symbols (we compile 8-bit only).
  Needed on BOTH CPU and GPU — libjpeg-turbo references these
  unconditionally.
- `seeds/min.jpg` — minimal 1x1 JPEG seed.
- `.gitignore` — excludes build artifacts + `.build/` cache +
  `jpeg_include/`.
- `.build/libjpeg-turbo-3.0.4/` (ignored) — cached upstream source
  clone.
- `.build/cpu-obj/` (ignored) — per-TU `.o` files for the CPU link.
- `jpeg_include/` (ignored) — staged patched header tree + COPIES of
  the library `.c` files (regenerated every run). The library sources
  are copied in, not included from `.build/`, because libjpeg-turbo
  uses quote-include (`#include "jmorecfg.h"`) which searches the
  source file's directory first.
- After a successful build:
  - `libjpeg_turbo_decompress_fuzzer.cubin`,
    `libjpeg_turbo_decompress_fuzzer.conf` (from `coqui-cc`)
  - `libjpeg_turbo_decompress_fuzzer_cpu` (AFL++-instrumented binary)
  - `out/` — created by `fuzz.sh`; AFL output dir

## Notes

- `--arch sm_75` is the `build.sh` default (override with
  `ARCH=sm_XX ./build.sh` to target a different GPU).
- `--stack-size 32768` (coqui-cc default).
- `--slab-pool-size 0` — no slab pool.
- 19 baseline `.c` files are compiled (`jdapimin`, `jdapistd`,
  `jdatasrc`, `jdcoefct`, `jdcolor`, `jddctmgr`, `jdhuff`, `jdinput`,
  `jdmainct`, `jdmarker`, `jdmaster`, `jdpostct`, `jdsample`,
  `jidctint`, `jmemmgr`, `jmemnobs`, `jerror`, `jutils`, `jcomapi`).
- Sanitizers (CPU): full ASan + UBSan set, `-fsanitize=fuzzer` at
  link time (afl-clang-fast rewrites to `libAFLDriver.a`). Sanitizers
  (GPU): none explicit — coqui-cc does not expose `-fsanitize`, and
  coqui mode's `CoquiPassPlugin.so` injects device-side ASan/UBSan.
- GPU `-D NO_GETENV` (no libc env on GPU).
- `setjmp` / `longjmp` are guarded inside `harness.c` with
  `#ifndef __COQUI_DEVICE__`; coqui-cc defines that macro.
- Override afl-clang-fast via `AFL_CC=...` — points to the
  `afl-clang-fast` binary, default `../../../afl-clang-fast`.
- `fuzz.sh` sets `AFL_I_DONT_CARE_ABOUT_MISSING_CRASHES=1`,
  `AFL_SKIP_CPUFREQ=1`, `AFL_NO_UI=1` per the
  project convention; override by exporting them yourself. GPU device
  index is controlled by `AFL_COQUI_DEVICE` (default 0).
