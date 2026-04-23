# cuAFL evaluation target: bzip2

libbzip2 1.0.8 decompression fuzzer, adapted from coqui's nix target
(`/home/gpizarro/coqui/nix/targets/bzip2.nix`) for cuAFL's coqui-cc
compiler and `afl-fuzz --coqui` runtime.

## Quick start

```bash
./build.sh    # builds cubin + conf + CPU binary + seeds + dict
./fuzz.sh -V 300 -t 5000   # fuzz for 300s with 5s dry-run timeout
```

## Status

Build succeeds end-to-end:

| Artifact              | Where                                                     |
|-----------------------|-----------------------------------------------------------|
| `bzip2_fuzzer.cubin`  | GPU kernel (sm_75), ~3.5 MB                               |
| `bzip2_fuzzer.conf`   | `stack_size=32768 slab_pool_size=2147483648 arch=sm_75`   |
| `bzip2_fuzzer_cpu`    | Symlink to `/tmp/cuafl-bzip2-cpu/bzip2_decompress_fuzzer` |
| `seeds/`              | 55 hand-crafted seeds (copied from `/nix/store`)          |
| `dict/bzip2.dict`     | Dictionary of bzip2 magic bytes (symlinked)               |

`fuzz.sh` launches successfully: GPU context initializes
(`batch=8192, stack=32 KB, heap=142 KB, total=256 KB`), all 55 seeds
load into the queue, and the fork server comes up.

## Files

- `build.sh` — idempotent build script.
- `fuzz.sh` — launches `afl-fuzz --coqui gpu0` with recommended env.
- `bz2_assert_stub.c` — local replacement for coqui's `bz2_assert_stub.c`:
  routes `bz_internal_error()` to `__coqui_trap()` instead of `abort()`
  (cuAFL's `ExternalSymbolGatekeeper` only allows `__coqui_*`/`__llvm_*`
  externals, so the upstream `abort()` stub fails the build).
- `.gitignore` — excludes build outputs, symlinks, and `out/`.

## Target details

Mirrors `nix/targets/bzip2.nix`:

- libbzip2 source: `libarchive/bzip2` tag `bzip2-1.0.8` (fetched via nix).
- Defines: `-D BZ_NO_STDIO`
- Includes: `-I <libbzip2 source>` (for `bzlib.h` / `bzlib_private.h`)
- `--slab-pool-size 2147483648` (2 GiB — DState (~60 KB) + smallest ll16/ll4
  buffer don't fit in the default 64 KB per-thread heap)
- `--stack-size 32768` (coqui-cc default — nix spec doesn't override)
- Arch: `sm_75` (RTX Titan)

Source files (order preserved from the nix spec):
`blocksort.c`, `huffman.c`, `crctable.c`, `randtable.c`, `compress.c`,
`decompress.c`, `bzlib.c`, local `bz2_assert_stub.c`,
`harness/targets/bzip2_decompress_target.c`.

## Gotchas

- **`abort()` isn't in cuAFL's external allowlist.** The upstream
  `bz2_assert_stub.c` uses `abort()` in `BZ_NO_STDIO` mode — cuAFL's
  `ExternalSymbolGatekeeper` rejects this. The local stub calls
  `__coqui_trap()` instead (same semantics: PTX `trap; exit;`).
- **Seeds must be real files, not symlinks.** AFL's `read_testcases()`
  uses `lstat()` + `S_ISREG` — symlinks get skipped and AFL aborts
  with "No usable test cases". The nix `seeds/` dir contains symlinks
  into `/nix/store`, so `build.sh` does `cp -L` to materialise them.
- **`nix build` flags that coqui-cc doesn't accept.** The nix spec
  passes `--heap-size 524288` and `--batch-size 32768` to coqui. These
  are not options on cuAFL's `coqui-cc`: heap is derived at runtime
  from the per-thread stack budget (142 KB on RTX Titan for a 32 KB
  stack), and batch size is read from `AFL_COQUI_BATCH_SIZE` at
  fuzz time (default 8192).
- **"Mistyped AFL environment variable: AFL_COQUI_CUBIN" warning.**
  Harmless. cuAFL's env-validator table in `afl-fuzz-state.c` doesn't
  list `AFL_COQUI_CUBIN` yet, but the variable IS read by
  `afl-fuzz.c:1709`. Out of scope for this eval target — fix belongs
  in cuAFL core.
- **CPU dry-run timeouts on loaded systems.** AFL's default 1000 ms
  CPU dry-run can fire on the instrumented binary under load. Pass
  `-t 5000` (or higher) to `./fuzz.sh` if the dry-run aborts.

## Rebuilding

`build.sh` is idempotent; rerunning it re-runs `nix build` (no-op if
cached) and rebuilds the cubin. To rebuild from a fully clean state:

```bash
rm -rf bzip2_fuzzer* seeds dict out .bzip2_fuzzer.build /tmp/cuafl-bzip2-cpu
./build.sh
```
