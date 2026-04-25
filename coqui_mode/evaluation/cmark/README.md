# coqui mode cmark evaluation target

Adapts the cmark (CommonMark Markdown parser) fuzz target so it can be
driven with coqui mode's `afl-fuzz --coqui`.

- Upstream: https://github.com/commonmark/cmark (pinned to `0.31.1`)
- Harness: `harness.c` (libFuzzer-style, 4-byte `options` + 4-byte
  `width` prefix; mode selected by upper 2 bits of `options`)

## Build / run

```
./build.sh   # fetches cmark + builds .cubin + CPU binary
./fuzz.sh    # prints launch commands for Main/Secondary/Coqui
```

`build.sh` is self-contained: no external dependencies beyond
`afl-clang-fast` (this repo) and `coqui-cc` (at `/usr/local/bin`). It
clones cmark from GitHub into `.build/`, synthesizes the cmark config
headers inline (cmark's upstream build is CMake; we skip that),
compiles the CPU binary with this repo's own `afl-clang-fast`, and
builds the GPU cubin via `coqui-cc` under a `/tmp/coqui-cc.lock` flock
so ptxas can serialize with any parallel target builds.

## Layout

- `build.sh` / `fuzz.sh` — entry points (set -euo pipefail).
- `harness.c` — in-tree libFuzzer-style harness.
- `seeds/min.md` — single minimal seed (`a`; 1 byte).
- `.gitignore` — excludes build artifacts + `.build/` cache + `generated/`.
- `.build/` (ignored) — cached cmark source clone.
- `generated/` (ignored) — `config.h`, `cmark_export.h`, `cmark_version.h`
  synthesized by `build.sh`.
- After a successful build:
  - `cmark_fuzzer.cubin`, `cmark_fuzzer.conf` (from `coqui-cc`)
  - `cmark_fuzzer_cpu` (AFL++-instrumented binary)
  - `out/` — created by `fuzz.sh`; AFL output dir

## Notes

- `--arch sm_75` is the `build.sh` default (override with
  `ARCH=sm_XX ./build.sh` to target a different GPU).
- CPU build compiles each translation unit separately to `.o` before
  linking. One-shot compile overflows afl-cc's 2048-parameter cap with
  the full sanitizer list and 19 cmark TUs; see the inline comment in
  `build.sh`.
- `fuzz.sh` sets `AFL_I_DONT_CARE_ABOUT_MISSING_CRASHES=1`,
  `AFL_SKIP_CPUFREQ=1`, `AFL_NO_UI=1` per the
  project convention; override by exporting them yourself.
