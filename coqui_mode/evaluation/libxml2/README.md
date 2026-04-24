# coqui mode libxml2 evaluation target

Adapts the libxml2 parser fuzzer so it can be driven with coqui mode's
`afl-fuzz --coqui`.

- Upstream: https://github.com/GNOME/libxml2 (pinned to tag `v2.13.4`)
- Harness: `harness.c` (libFuzzer-style, 8-path: `data[0] & 0x07`
  selects pull / push / tree+save / xpath / URI / deep-copy / encoding
  / input-derived-xpath)

## Build / run

```
./build.sh   # fetches libxml2 + patches error.c/xmlstring.c + builds .cubin + CPU binary
./fuzz.sh    # prints launch commands for Main/Secondary/Coqui
```

`build.sh` is self-contained: no external dependencies beyond
`afl-clang-fast` (this repo) and `coqui-cc` (at `/usr/local/bin`). It
clones libxml2 from GitHub into `.build/`, generates `config.h` /
`xmlversion.h` inline (libxml2's upstream build is autotools/CMake; we
skip that), patches `error.c` and `xmlstring.c` with depth-aware awk
(see notes below), compiles the CPU binary with this repo's own
`afl-clang-fast`, and builds the GPU cubin via `coqui-cc` under a
`/tmp/coqui-cc.lock` flock so ptxas can serialize with any parallel
target builds.

## Layout

- `build.sh` / `fuzz.sh` — entry points (set -euo pipefail).
- `harness.c` — in-tree libFuzzer-style harness (8-path).
- `libxml2_stubs.c` — musl-atomic stubs for the GPU build (libxml2's
  globals.c initialization pulls in syscall-using musl atomics).
- `seeds/min.xml` — single minimal seed (`<a/>`).
- `.gitignore` — excludes build artifacts + `.build/` cache +
  `.libxml2_fuzzer.build/` + `.libxml2_xml_read_fuzzer.build/`.
- `.build/` (ignored) — cached libxml2 source clone.
- `.libxml2_fuzzer.build/` (ignored) — generated headers + patched
  sources.
- After a successful build:
  - `libxml2_xml_read_fuzzer.cubin`, `libxml2_xml_read_fuzzer.conf`
    (from `coqui-cc`)
  - `libxml2_xml_read_fuzzer_cpu` (AFL++-instrumented binary)
  - `out/` — created by `fuzz.sh`; AFL output dir

## Notes

- `--arch sm_75` matches the RTX Titan test box (see `CLAUDE.local.md`).
  Override with `ARCH=sm_XX ./build.sh`.
- `--stack-size 32768` — needed for deep parser recursion + xpath
  evaluator.
- `--slab-pool-size 2147483648` (2 GiB) — the per-thread 64 KB heap
  plus the global slab pool hosts xmlmemory allocations for 131k
  threads.
- Compiled sources: core parser + SAX + tree + xpath + URI + xmlsave
  + encoding + xmlIO + xmlmemory + xmlstring + globals + hash + dict
  + list + valid + parserInternals + threads + chvalid + buf +
  entities + error (22 files total). HTML and reader/writer modules
  are excluded.
- **`error.c` patches**: `xmlCopyError`, `xmlFormatError`,
  `xmlResetLastError`, and `xmlRaiseMemoryError` are stubbed to
  no-ops. Otherwise they race on the shared `xmlLastError` global
  (no GPU-side synchronization) and reach `__coqui_vsnprintf` via the
  error-formatting chain, which allocates 512+ bytes on the CUDA
  hardware call stack and overflows from the deep parser call chain.
- **`xmlstring.c` patch**: `xmlStrVASPrintf` is stubbed to return -1.
  Callers handle the -1 by falling through with a NULL message,
  preserving error-code propagation without the vsnprintf blow-up.
- `COQUI_LLC_OPT=-O1` is the default for this target. The post-ASan
  LLVM module has ~211k instrumented memory accesses and llc's
  default `-O2` register allocator stalls indefinitely. `-O1`
  completes in ~2 minutes; `ptxas` then takes ~15 minutes and peaks
  at ~20 GB RSS.
- `fuzz.sh` sets `AFL_I_DONT_CARE_ABOUT_MISSING_CRASHES=1`,
  `AFL_SKIP_CPUFREQ=1`, `AFL_SKIP_BIN_CHECK=1`, `AFL_NO_UI=1` per the
  project convention; override by exporting them yourself. GPU device
  index is controlled by `AFL_COQUI_DEVICE` (default 0).
