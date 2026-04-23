# cuAFL evaluation target: `cares`

Fuzzes the [c-ares](https://c-ares.org/) DNS parser (v1.34.4) via the
oss-fuzz-adapted `ares_dns_parse` harness, running the hot parsing code
on the GPU under cuAFL's `coqui_mode`.

Mirrors `/home/gpizarro/coqui/nix/targets/cares.nix` — same source
whitelist, same config-header generation, same `--slab-pool-size 10G`.

## Build

```
./build.sh
```

Steps performed:

1. `nix build '.#target-cares-aflplusplus'` from `/home/gpizarro/coqui`
   to produce the CPU AFL++-instrumented binary (used as the forkserver
   target for classified exit paths + crash verification).
2. Walks the resulting store closure to find the c-ares source and the
   harness source (`cares_parse_reply_fuzzer.c`, `cares_stubs.c`).
3. Generates `generated/ares_build.h` and `generated/ares_config.h`
   (normally produced by c-ares's autotools/cmake).
4. Invokes `/usr/local/bin/coqui-cc -arch sm_75 --slab-pool-size 10737418240`
   over the library whitelist + harness to produce
   `cares_parse_reply_fuzzer.cubin` + `.conf`.
5. Symlinks `cares_parse_reply_fuzzer_cpu`, `seeds/`, `dict/` from the
   nix store path.

Idempotent: re-running will overwrite the generated headers and cubin
but keep the cached nix build.

### Build environment overrides

- `COQUI_CC` — path to `coqui-cc` (default `/usr/local/bin/coqui-cc`)
- `COQUI_REPO` — path to coqui flake repo (default `/home/gpizarro/coqui`)
- `CPU_OUT_LINK` — nix out-link path for the CPU binary
  (default `/tmp/cuafl-cares-cpu`)

## Fuzz

```
./fuzz.sh                    # run forever
./fuzz.sh -V 120             # stop after 120 s (benchmark mode)
./fuzz.sh -x ./dict/dns.dict # explicit dict (auto-added if not given)
```

Launches:

```
/home/gpizarro/cuAFL/afl-fuzz --coqui gpu0 \
    -i ./seeds -o ./out \
    -x ./dict/dns.dict \
    -- ./cares_parse_reply_fuzzer_cpu
```

with the standard cuAFL env vars set:

- `AFL_COQUI_CUBIN=$PWD/cares_parse_reply_fuzzer.cubin`
- `AFL_I_DONT_CARE_ABOUT_MISSING_CRASHES=1`
- `AFL_SKIP_CPUFREQ=1`
- `AFL_SKIP_BIN_CHECK=1`
- `AFL_NO_UI=1` (override by exporting `AFL_NO_UI=0` before invoking)

Additional arguments are passed through to `afl-fuzz`.

### Fuzz environment overrides

- `AFL_FUZZ` — path to `afl-fuzz` binary (default `/home/gpizarro/cuAFL/afl-fuzz`)

## Target-specific quirks

- **Stack size**: the nix spec does not override `--stack-size`, so the
  coqui-cc default of 32768 is used. If GPU traces show stack overflow,
  bumping this value in `build.sh` is the lever.
- **Slab pool = 10 GiB**: matches the nix spec. The DNS name-decompression
  path can exponentially blow up heap with crafted pointer loops; the
  harness already caps input size to 512 bytes and total RR count to 32
  to keep this bounded.
- **Sanitizer inheritance**: coqui-cc applies the default sanitizer set
  (ASan + UBSan per `nix/sanitizer-flags.nix`). No target-specific
  sanitizer tuning required.
- **Networking stubs**: `cares_stubs.c` provides GPU-side stubs for the
  networking paths that the parser code references but the DNS-only
  whitelist does not actually call.
- **Dict**: the oss-fuzz-style DNS dict covers record types, classes,
  response flags, compression pointers, and common label lengths.

## Build status

As of target creation, `build.sh` was tested end-to-end. The nix CPU build
(`.#target-cares-aflplusplus`) was cached and completed immediately. The
GPU `coqui-cc` build advances through all clang/llvm-link/opt/llc stages
successfully and reaches `ptxas`, which produced a 13 MiB PTX file from
the linked+transformed module.

**Known quirk — ptxas resource use on this target is unusually high.**
The DNS record-accessor code generates a huge switch table, making ptxas
both CPU- and memory-hungry. In the test run conducted alongside roughly
half a dozen other concurrent `coqui-cc` builds of other targets, ptxas
for cares was killed by the OOM killer after ~34 minutes. On a machine
not running other heavy compilations it should finish in a few minutes.
If ptxas fails with SIGKILL, wait for other builds to finish and rerun
`build.sh`; bitcode from prior runs is reused so only ptxas reruns.

After a successful build the directory contains:

```
.cares_parse_reply_fuzzer.build/   coqui-cc intermediate bitcode (ignored)
generated/                         ares_build.h, ares_config.h
seeds                    -> /tmp/cuafl-cares-cpu/seeds (74 files)
dict                     -> /tmp/cuafl-cares-cpu/dict/
cares_parse_reply_fuzzer.cubin
cares_parse_reply_fuzzer.conf
cares_parse_reply_fuzzer_cpu -> /tmp/cuafl-cares-cpu/cares_parse_reply_fuzzer
```
