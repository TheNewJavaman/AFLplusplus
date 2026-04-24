# coqui mode cares evaluation target

Adapts the [c-ares](https://c-ares.org/) DNS parser's oss-fuzz-style
`ares_dns_parse` harness so it can be driven with coqui mode's
`afl-fuzz --coqui`.

- Upstream: https://github.com/c-ares/c-ares (pinned to tag `v1.34.4`)
- Harness: `harness.c` (libFuzzer-style; caps input size to 512 bytes
  and total RR count to 32 to bound DNS name-decompression blowup)

## Build / run

```
./build.sh   # fetches c-ares + builds .cubin + CPU binary
./fuzz.sh    # prints launch commands for Main/Secondary/Coqui
```

`build.sh` is self-contained: no external dependencies beyond
`afl-clang-fast` (this repo) and `coqui-cc` (at `/usr/local/bin`). It
clones c-ares from GitHub into `.build/`, synthesizes
`ares_build.h` / `ares_config.h` inline (normally produced by c-ares's
autotools/cmake), compiles the CPU binary with this repo's own
`afl-clang-fast` (per-TU to avoid afl-cc's `MAX_PARAMS_NUM` cap), and
builds the GPU cubin via `coqui-cc` under a `/tmp/coqui-cc.lock` flock
so ptxas can serialize with any parallel target builds.

## Layout

- `build.sh` / `fuzz.sh` — entry points (set -euo pipefail).
- `harness.c` — in-tree libFuzzer-style `ares_dns_parse` harness.
- `cares_stubs.c` — GPU-side stubs for the DNS-only source whitelist
  (networking paths the parser references but never calls; also
  `strcasecmp`/`strncasecmp`/`gettimeofday` on GPU — CPU build passes
  `-DCOQUI_CPU` to skip).
- `seeds/min.dns` — single minimal 12-byte DNS header.
- `.gitignore` — excludes build artifacts + `.build/` cache +
  `generated/`.
- `.build/` (ignored) — cached c-ares source clone + per-TU `.o` files.
- `generated/` (ignored) — `ares_build.h`, `ares_config.h` synthesized
  by `build.sh`.
- After a successful build:
  - `cares_parse_reply_fuzzer.cubin`, `cares_parse_reply_fuzzer.conf`
    (from `coqui-cc`)
  - `cares_parse_reply_fuzzer_cpu` (AFL++-instrumented binary)
  - `out/` — created by `fuzz.sh`; AFL output dir

## Notes

- `--arch sm_75` is the `build.sh` default (override with
  `ARCH=sm_XX ./build.sh` to target a different GPU).
- Stack size: coqui-cc default (32768) — not overridden.
- `--slab-pool-size 10737418240` (10 GiB) — DNS name-decompression can
  blow up heap via crafted pointer loops. The harness caps input to
  512 bytes and RR count to 32 to keep this bounded; the slab pool
  absorbs worst-case allocations.
- Sanitizers (CPU): full ASan + UBSan set. GPU build drops
  `-fsanitize=…` because `coqui-cc` does not accept those flags
  (device-side ASan is injected by `CoquiPassPlugin.so`).
- **Do not use `AFL_CC` as the override name.** `AFL_CLANG_FAST` is
  the env var for overriding `afl-clang-fast` here. `afl-cc` reserves
  `AFL_CC` to override its backing clang; setting it to
  `afl-clang-fast` causes afl-cc to recursively re-exec until
  `MAX_PARAMS_NUM` is hit.
- ptxas resource use is unusually high. The DNS record-accessor code
  generates a large switch table. Concurrent cubin builds will OOM
  the box — `build.sh` uses `flock /tmp/coqui-cc.lock`.
- `fuzz.sh` sets `AFL_I_DONT_CARE_ABOUT_MISSING_CRASHES=1`,
  `AFL_SKIP_CPUFREQ=1`, `AFL_SKIP_BIN_CHECK=1`, `AFL_NO_UI=1` per the
  project convention; override by exporting them yourself. GPU device
  index is controlled by `AFL_COQUI_DEVICE` (default 0).
