# coqui mode evaluation target: zstd

Adapts the `zstd` decompression fuzz target (from coqui) for coqui mode, so
`afl-fuzz --coqui` can fuzz the Facebook `zstd` library on the GPU.

Source of truth: `/home/gpizarro/coqui/nix/targets/zstd.nix`
(facebook/zstd v1.5.6, decompression-only, custom GPU-adapted harness).

## Layout

```
build.sh                             idempotent build driver
fuzz.sh                              launches afl-fuzz --coqui on the cubin
zstd_abort_stub.c                    local device-side `abort()` stub (see Gotchas)
README.md                            this file
.gitignore                           ignores build outputs
zstd_simple_decompress_fuzzer.cubin  produced by build.sh
zstd_simple_decompress_fuzzer.conf   coqui-cc sidecar: stack_size / slab_pool_size / arch
zstd_simple_decompress_fuzzer_cpu    AFL++ CPU binary (symlink into /tmp/coqui-zstd-cpu)
seeds/                               symlink to upstream zstd seed corpus (.zst frames)
dict/                                symlink to zstd frame-magic/block-type dictionary
out/                                 afl-fuzz output (created on first run)
```

## Build

```
cd /home/gpizarro/cuAFL/coqui_mode/evaluation/zstd
./build.sh
```

The script is idempotent. It:

1. Runs `nix build .#target-zstd-aflplusplus` in `/home/gpizarro/coqui`
   to produce the AFL++-instrumented CPU harness, its seeds, and the zstd
   dictionary, and symlinks the result under `/tmp/coqui-zstd-cpu`.
2. Resolves the `facebook/zstd v1.5.6` source path from the nix closure
   (identifies it as the `-source` derivation containing `lib/zstd.h`).
3. Invokes `/usr/local/bin/coqui-cc` with the exact same includes,
   defines, and source list as `nix/targets/zstd.nix`, plus the local
   `zstd_abort_stub.c`, producing:
   - `zstd_simple_decompress_fuzzer.cubin` — sm_75 GPU kernel
   - `zstd_simple_decompress_fuzzer.conf`  — stack_size / slab_pool_size / arch
4. Symlinks the CPU binary, seeds, and dict next to `build.sh` so the
   fuzz script has stable relative paths.

Flags (mirrored from `zstd.nix`):
- `-arch sm_75`
- `--stack-size 32768`  (coqui-cc default; zstd.nix does not override)
- `--slab-pool-size 0`  (zstd.nix does not configure a slab pool)
- `-I <zstd>/lib -I <zstd>/lib/common -I <zstd>/lib/decompress`
- `-D ZSTD_NO_INTRINSICS=1 -D ZSTD_DISABLE_ASM=1 -D XXHASH_NAMESPACE=ZSTD_`
- `-D FUZZING_BUILD_MODE_UNSAFE_FOR_PRODUCTION -D ZSTD_NO_TRACE=1`
- `-D ZSTD_DECODER_INTERNAL_BUFFER=4096`  (shrinks the DCtx to ~34KB so
  it fits in the per-thread bump heap alongside the 8KB output buffer)

Sources (exactly the zstd.nix set, decompress-only):
- `lib/common/{debug,entropy_common,error_private,fse_decompress,xxhash,zstd_common}.c`
- `lib/decompress/{huf_decompress,zstd_ddict,zstd_decompress,zstd_decompress_block}.c`
- `harness/targets/zstd_simple_decompress_fuzzer.c` (from coqui)
- `zstd_abort_stub.c` (local — see Gotchas)

## Fuzz

```
./fuzz.sh               # runs afl-fuzz --coqui gpu0 -i ./seeds -o ./out -- ./<harness>_cpu
./fuzz.sh -V 60         # stop after 60 s
./fuzz.sh -M main       # run as a main sync node
```

Environment exported by `fuzz.sh`:
- `AFL_COQUI_CUBIN=$(pwd)/zstd_simple_decompress_fuzzer.cubin`
- `AFL_I_DONT_CARE_ABOUT_MISSING_CRASHES=1`
- `AFL_SKIP_CPUFREQ=1`
- `AFL_SKIP_BIN_CHECK=1`
- `AFL_NO_UI=1`

Any extra args after `./fuzz.sh` are passed to `afl-fuzz`.

## Gotchas

### `abort()` → device stub (`zstd_abort_stub.c`)

The upstream harness `zstd_simple_decompress_fuzzer.c` calls `abort()` if a
frame's declared content size does not match the actual decompressed size
(i.e., it has detected a real zstd bug).

Upstream coqui rewrites `abort` to `__coqui_abort` inside a `LibcTransform`
LLVM pass. coqui mode's `coqui-cc` is a simplified driver that runs only
`coqui-link` + `always-inline`, with NO `LibcTransform` pass and no
pre-linked `abort` symbol in `runtime.bc`. Without an explicit stub, the
build fails inside the `ExternalSymbolGatekeeper`:

```
LLVM ERROR: [coqui-cc] ExternalSymbolGatekeeper: unresolved external
'abort' — port the replacement transform or runtime stub.
Used by: __coqui_fuzz_execute
```

The local file `zstd_abort_stub.c` supplies a trivial device-side `abort()`
that tail-calls `__coqui_trap()` (PTX `trap;`), which coqui mode classifies as a
signal-11 crash — the desired semantics. This mirrors the pattern used by
`harness/targets/bz2_assert_stub.c` for bzip2.

### `ZSTD_DECODER_INTERNAL_BUFFER=4096`

See the header comment in the upstream harness: without this, the ZSTD_DCtx
is ~64KB and does not fit alongside the output buffer in the per-thread
bump heap. With `=4096`, the DCtx shrinks to ~34KB, leaving ~16KB headroom
in the 58KB usable heap for the 8KB output buffer and internal
temporaries. The coqui mode build mirrors `zstd.nix` exactly on this.

### Batch size / heap size

coqui mode's `coqui-cc` does NOT accept `--heap-size` or `--batch-size` flags.
The coqui mode runtime derives the per-thread heap at startup (from
`--stack-size` and per-device SM geometry) and reads the batch size from
`AFL_COQUI_BATCH_SIZE` at fuzz time (default: `COQUI_DEFAULT_BATCH_SIZE`
in `include/afl-fuzz-coqui.h`).

### Sanitizers

Upstream `zstd.nix` passes `${sanitizers.default}` (ASan + UBSan on the
CPU side for `FUZZING_BUILD_MODE_UNSAFE_FOR_PRODUCTION`). coqui mode drops the
sanitizer flags from the GPU build — device-side ASan/UBSan is already
injected by the coqui pass plugin, and the CPU binary is built by nix
(which retains its own sanitizer wiring). The CPU binary is used by AFL++
only for crash verification.

### ptxas time

`ptxas` can take a long time on the zstd cubin (tens of minutes on the
sm_75 Titan). The build is single-threaded at that stage, so expect a
large pause before `.cubin` appears. The rest of the pipeline (clang →
llvm-link → coqui-link → always-inline → llc) takes a few seconds.

## Build status

Last attempt: 2026-04-22 on this machine (RTX Titan, sm_75).
- clang, llvm-link, internalize/globaldce, coqui-link, always-inline,
  llc -march=nvptx64 — all complete quickly without errors.
- ptxas -arch=sm_75 -O1 can be very slow (tens of minutes) on the
  ~10-source zstd decompressor, holding one CPU core. See Gotchas.
- Dry-run was terminated early during ptxas; rerunning `./build.sh`
  resumes cleanly (the earlier `.bc` files are cached in the
  `.zstd_simple_decompress_fuzzer.build/` work directory).
