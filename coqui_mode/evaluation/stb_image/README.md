# coqui mode stb_image evaluation target

Adapts the stb_image multi-format image decoder fuzzer
(PNG / BMP / GIF / JPEG / TGA / PSD / PIC / PNM) so it can be driven with
coqui mode's `afl-fuzz --coqui`.

- Upstream: https://github.com/nothings/stb (pinned to commit
  `28d546d5eb77d4585506a20480f4de2e706dff4c`; `stb_image.h` is fetched
  as a single header)
- Harness: `harness.c` (libFuzzer-style; defines `STB_IMAGE_IMPLEMENTATION`,
  `STBI_NO_STDIO`, `STBI_NO_SIMD`, `STBI_NO_THREAD_LOCALS`, and caps image
  dimensions at 64x64 to avoid billion-iteration decode loops)

## Build / run

```
./build.sh   # fetches stb_image.h + builds .cubin + CPU binary
./fuzz.sh    # prints launch commands for Main/Secondary/Coqui
```

`build.sh` is self-contained: no external dependencies beyond
`afl-clang-fast` (this repo) and `coqui-cc` (at `/usr/local/bin`). It
fetches `stb_image.h` from GitHub at a pinned commit into `.build/`,
compiles the CPU binary with this repo's own `afl-clang-fast`, and
builds the GPU cubin via `coqui-cc`.

## Layout

- `build.sh` / `fuzz.sh` — entry points (set -euo pipefail).
- `harness.c` — in-tree libFuzzer-style harness.
- `seeds/white.png` — minimal 1x1 PNG seed.
- `.gitignore` — excludes build artifacts + `.build/` cache.
- `.build/` (ignored) — cached `stb_image.h`.
- After a successful build:
  - `stb_image_read_fuzzer.cubin`, `stb_image_read_fuzzer.conf` (from `coqui-cc`)
  - `stb_image_read_fuzzer_cpu` (AFL++-instrumented binary)
  - `out/` — created by `fuzz.sh`; AFL output dir

## Notes

- `--arch sm_75` is the `build.sh` default (override with
  `ARCH=sm_XX ./build.sh` to target a different GPU).
- `--stack-size 65536` (double the coqui-cc default) — the multi-format
  decoder has deep call chains (e.g. `stbi__jpeg_decode_block`).
- `--slab-pool-size 0` — no slab pool.
- Two extra `-D` flags are required for the GPU build that are not in
  any CPU spec:
  - `-D "STBI_ASSERT(x)="` — the `STBI_ASSERT` calls expand to
    `__assert_fail`, which coqui mode's `ExternalSymbolGatekeeper`
    rejects. stb_image officially supports `STBI_ASSERT` overrides.
  - `-D STBI_NO_HDR` — the Radiance HDR decoder calls `strtol()`, which
    is not in coqui mode's runtime allowlist. Coverage impact: GPU
    won't drive coverage toward HDR paths; CPU AFL++ binary retains
    HDR.
- `ptxas` is very slow on this target. The kernel is large (every
  decoder inlined into one entry point, ~20k LLVM instructions, ~10 MB
  PTX) and `ptxas -O1` can take 20-30 minutes and tens of GB of RAM.
  Do not run multiple `./build.sh` invocations concurrently.
- `fuzz.sh` sets `AFL_I_DONT_CARE_ABOUT_MISSING_CRASHES=1`,
  `AFL_SKIP_CPUFREQ=1`, `AFL_SKIP_BIN_CHECK=1`, `AFL_NO_UI=1` per the
  project convention; override by exporting them yourself. GPU device
  index is controlled by `AFL_COQUI_DEVICE` (default 0).
