# coqui mode libyaml evaluation target

Adapts the libyaml parser fuzzer so it can be driven with coqui mode's
`afl-fuzz --coqui`.

- Upstream: https://github.com/yaml/libyaml (pinned to tag `0.2.5`)
- Harness: `harness.c` (libFuzzer-style; feeds input bytes to
  `yaml_parser_*` APIs from the parser-only subset)

## Build / run

```
./build.sh   # fetches libyaml + patches yaml_private.h + builds .cubin + CPU binary
./fuzz.sh    # prints launch commands for Main/Secondary/Coqui
```

`build.sh` is self-contained: no external dependencies beyond
`afl-clang-fast` (this repo) and `coqui-cc` (at `/usr/local/bin`). It
fetches the libyaml tarball from GitHub into `.build/`, patches
`yaml_private.h` to shrink `INPUT_RAW_BUFFER_SIZE` 16384 → 512 and the
`INITIAL_{STACK,QUEUE,STRING}_SIZE` defaults 16 → 4 so the parser fits
inside the per-thread 64 KB GPU heap, compiles the CPU binary with
this repo's own `afl-clang-fast`, and builds the GPU cubin via
`coqui-cc` under a `/tmp/coqui-cc.lock` flock so ptxas can serialize
with any parallel target builds.

## Layout

- `build.sh` / `fuzz.sh` — entry points (set -euo pipefail).
- `harness.c` — in-tree libFuzzer-style harness.
- `libyaml_stubs.c` — GPU-side stubs for `strdup`, `memcpy`, `memset`,
  `memmove` (not in coqui mode's runtime.bc). Compiled into both
  CPU and GPU builds.
- `seeds/min.yaml` — single minimal YAML seed.
- `.gitignore` — excludes build artifacts + `.build/` cache +
  `libyaml_src/`.
- `.build/` (ignored) — cached libyaml source tarball extraction.
- `libyaml_src/` (ignored) — patched libyaml sources (copy of
  `api.c / reader.c / scanner.c / parser.c / loader.c /
  yaml_private.h`).
- After a successful build:
  - `libyaml_parser_fuzzer.cubin`, `libyaml_parser_fuzzer.conf`
    (from `coqui-cc`)
  - `libyaml_parser_fuzzer_cpu` (AFL++-instrumented binary)
  - `out/` — created by `fuzz.sh`; AFL output dir

## Notes

- `--arch sm_75` is the `build.sh` default (override with
  `ARCH=sm_XX ./build.sh` to target a different GPU).
- No `--stack-size` / `--slab-pool-size` override — libyaml fits
  under coqui-cc defaults (32 KB stack, no slab pool) once the buffer
  shrink is applied.
- `-D NDEBUG` is required on the GPU build. libyaml's
  `api.c` / `scanner.c` call `assert()` which lowers to
  `__assert_fail` on NVPTX. coqui mode's `coqui-cc` only runs the
  `coqui-link,always-inline` pipeline — no `LibcTransform` — and
  `ExternalSymbolGatekeeper` rejects `__assert_fail`. GPU asserts
  become no-ops; the CPU AFL++ binary still honors them. Analogous
  to stb_image's `-D STBI_ASSERT(x)=` workaround.
- `libyaml_stubs.c` is needed because coqui mode's runtime exports
  only `__coqui_{malloc,calloc,free,realloc,memcmp,strlen,strcmp,strncmp,strchr,strtod,trap}`.
  `strdup` is implemented via `__coqui_malloc` + `__coqui_strlen` +
  byte copy; the mem* stubs use clang builtins / manual loops.
- `fuzz.sh` sets `AFL_I_DONT_CARE_ABOUT_MISSING_CRASHES=1`,
  `AFL_SKIP_CPUFREQ=1`, `AFL_SKIP_BIN_CHECK=1`, `AFL_NO_UI=1` per the
  project convention; override by exporting them yourself. GPU device
  index is controlled by `AFL_COQUI_DEVICE` (default 0).
