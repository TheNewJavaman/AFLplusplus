# coqui mode cjson evaluation target

Adapts the cJSON oss-fuzz-style harness so it can be driven with coqui
mode's `afl-fuzz --coqui`.

- Upstream: https://github.com/DaveGamble/cJSON (pinned to commit
  `acc76239bee01d8e9c858ae2cab296704e52d916`, v1.7.18)
- Harness: `harness.c` (libFuzzer-style; 4-byte options prefix selects
  minify / require_termination / formatted / buffered; mirrors
  upstream `fuzzing/cjson_read_fuzzer.c`)

## Build / run

```
./build.sh   # fetches cJSON + builds .cubin + CPU binary
./fuzz.sh    # prints launch commands for Main/Secondary/Coqui
```

`build.sh` is self-contained: no external dependencies beyond
`afl-clang-fast` (this repo) and `coqui-cc` (at `/usr/local/bin`). It
fetches cJSON source from GitHub (tarball at the pinned commit) into
`.build/`,
compiles the CPU binary with this repo's own `afl-clang-fast`, and
builds the GPU cubin via `coqui-cc` under a `/tmp/coqui-cc.lock` flock
so ptxas can serialize with any parallel target builds.

## Layout

- `build.sh` / `fuzz.sh` — entry points (set -euo pipefail).
- `harness.c` — in-tree libFuzzer-style harness.
- `seeds/min.json` — single minimal seed.
- `.gitignore` — excludes build artifacts + `.build/` cache.
- `.build/` (ignored) — cached cJSON source tarball extraction.
- After a successful build:
  - `cjson_fuzzer.cubin`, `cjson_fuzzer.conf` (from `coqui-cc`)
  - `cjson_fuzzer_cpu` (AFL++-instrumented binary)
  - `out/` — created by `fuzz.sh`; AFL output dir

## Notes

- `--arch sm_75` is the `build.sh` default (override with
  `ARCH=sm_XX ./build.sh` to target a different GPU).
- `--stack-size 32768`, `--slab-pool-size 0`. cJSON's parse + print
  paths fit inside the default per-thread 64 KB heap; no slab pool
  needed.
- Build pipeline uses `-D CJSON_HIDE_SYMBOLS` and `-I <cache>`. CPU
  build adds the full ASan + UBSan sanitizer set plus
  `-fsanitize=fuzzer` at link time; GPU build drops `-fsanitize=…`
  because `coqui-cc` does not accept those flags (device-side ASan is
  injected by `CoquiPassPlugin.so`).
- `fuzz.sh` sets `AFL_I_DONT_CARE_ABOUT_MISSING_CRASHES=1`,
  `AFL_SKIP_CPUFREQ=1`, `AFL_SKIP_BIN_CHECK=1`, `AFL_NO_UI=1` per the
  project convention; override by exporting them yourself. GPU device
  index is controlled by `AFL_COQUI_DEVICE` (default 0).
- Orphan-safe: `afl-fuzz` installs `PR_SET_PDEATHSIG` on the
  forkserver (`aadd355b`), so killing the launching shell cleanly
  tears down the fuzzer. Closing a tmux pane does NOT reach the
  fuzzer — use `pkill -u "$USER" -KILL -f "afl-fuzz --coqui"` before
  relaunching and verify `nvidia-smi` shows memory released.
