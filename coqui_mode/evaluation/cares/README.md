# coqui mode evaluation target: `cares`

Fuzzes the [c-ares](https://c-ares.org/) DNS parser (v1.34.4) via the
oss-fuzz-adapted `ares_dns_parse` harness, running the hot parsing code
on the GPU under coqui mode's `coqui_mode`.

Mirrors the `cares` target in the legacy coqui nix codebase — same
source whitelist, same config-header generation, same
`--slab-pool-size 10 GiB` — but self-contained: no `nix`/`COQUI_REPO`
dependency, sources fetched directly from GitHub at a pinned tag.

## Build

```
./build.sh
```

Steps performed:

1. Fetches `c-ares/c-ares` at tag `v1.34.4` via `git clone --depth 1`
   into `.build/c-ares-c-ares-v1.34.4/` (cached — subsequent runs skip
   the clone).
2. Generates `generated/ares_build.h` and `generated/ares_config.h`
   inline via heredoc (normally produced by c-ares's autotools/cmake).
3. Compiles each whitelisted `.c` to `.o` with `afl-clang-fast`
   (`-fsanitize=address` + full UBSan), then links with
   `-fsanitize=fuzzer` to produce `cares_parse_reply_fuzzer_cpu`.
   Per-file compile avoids afl-cc's MAX_PARAMS_NUM limit.
4. Invokes `coqui-cc -arch sm_75 --slab-pool-size 10737418240` over
   the library whitelist + harness + `cares_stubs.c` to produce
   `cares_parse_reply_fuzzer.cubin` + `.conf`. Serialized via
   `flock /tmp/coqui-cc.lock` to avoid ptxas OOM when multiple
   targets build concurrently.

### Build environment overrides

- `ARCH` — GPU compute capability (default `sm_75`)
- `AFL_CLANG_FAST` — path to `afl-clang-fast`
  (default `../../../afl-clang-fast` relative to this directory).
  **Do not use `AFL_CC` as the override name** — afl-cc reserves that
  env var to override its backing clang; setting it to afl-clang-fast
  causes afl-cc to recursively re-exec until `MAX_PARAMS_NUM` is hit.
- `COQUI_CC` — path to `coqui-cc` (default `/usr/local/bin/coqui-cc`)
- `CARES_CACHE` — where to cache the cloned c-ares source
  (default `.build/c-ares-c-ares-v1.34.4`)

## Fuzz

```
./fuzz.sh
```

Prints suggested `afl-fuzz` commands (Main / Secondary / Coqui). Pick
one per terminal and export the recommended env vars:

```
AFL_I_DONT_CARE_ABOUT_MISSING_CRASHES=1
AFL_SKIP_CPUFREQ=1
AFL_SKIP_BIN_CHECK=1
AFL_NO_UI=1
AFL_COQUI_CUBIN=$PWD/cares_parse_reply_fuzzer.cubin
AFL_COQUI_DEVICE=$AFL_COQUI_DEVICE   # 0..3
```

## Target-specific quirks

- **Stack size**: coqui-cc default (32768) is sufficient; not overridden.
- **Slab pool = 10 GiB**: DNS name-decompression can blow up heap via
  crafted pointer loops. The harness caps input size to 512 bytes and
  total RR count to 32 to keep this bounded.
- **Sanitizers**: ASan + full UBSan (matches
  `nix/cpu-sanitizer-flags.nix` `sanitizers.default`).
- **`cares_stubs.c`**: committed in-tree. Provides GPU-side stubs for
  networking paths the parser references but the DNS-only whitelist
  never calls. Also provides `strcasecmp`/`strncasecmp`/`gettimeofday`
  on GPU (not on CPU — CPU build passes `-DCOQUI_CPU` to skip).
- **ptxas resource use is unusually high.** The DNS record-accessor
  code generates a large switch table, making ptxas both CPU- and
  memory-hungry. Concurrent cubin builds will OOM the box, which is
  why step 4 is serialized via `flock /tmp/coqui-cc.lock`.

## Files

- `build.sh` — self-contained build (no nix).
- `fuzz.sh` — workspace prep + suggested launch commands.
- `harness.c` — oss-fuzz-adapted `ares_dns_parse` harness (in-tree).
- `cares_stubs.c` — GPU stubs for excluded source files (in-tree).
- `seeds/min.dns` — single 12-byte minimal DNS header.
- `.gitignore` — excludes `.build/`, `generated/`, cubin/conf,
  CPU binary, `out/`, logs.

After a successful build the directory also contains:

```
.build/c-ares-c-ares-v1.34.4/         cloned upstream source
.build/cpu-obj/                       per-source .o files
.cares_parse_reply_fuzzer.build/      coqui-cc intermediate bitcode
generated/ares_build.h                generated config header
generated/ares_config.h               generated config header
cares_parse_reply_fuzzer_cpu          AFL++ instrumented CPU binary
cares_parse_reply_fuzzer.cubin        GPU kernel
cares_parse_reply_fuzzer.conf         runtime config for the cubin
```
