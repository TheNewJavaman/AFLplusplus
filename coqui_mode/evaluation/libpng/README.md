# cuAFL evaluation target: libpng

libpng fuzz target (reads PNG via libpng's low-level API), adapted from
coqui's nix spec (`/home/gpizarro/coqui/nix/targets/libpng.nix`) for
cuAFL's `coqui-cc` compiler and `afl-fuzz --coqui` runtime.

libpng depends on zlib; both libraries are compiled into a single cubin.
The harness (`libpng_read_fuzzer.c`, shared with coqui) is libFuzzer-style
(`LLVMFuzzerTestOneInput`) and exercises the read path plus an optional
write-back step driven by input bytes.

## Quick start

```bash
./build.sh    # builds cubin + conf + CPU binary + seeds + dict
./fuzz.sh     # launches afl-fuzz --coqui gpu0 on this target
```

Any `$@` args are forwarded to `afl-fuzz` (after the target), e.g.:

```bash
./fuzz.sh -V 300          # 5-minute bench
./fuzz.sh -t 5000 -V 60   # 5s dry-run timeout, 60s total
AFL_RESUME=1 ./fuzz.sh    # resume into existing ./out
```

## Files

- `build.sh` — idempotent build script.
- `fuzz.sh` — `afl-fuzz --coqui gpu0` launcher with the standard env vars.
- `png_abort_stub.c` — local `abort()` → `__coqui_trap()` stub (see
  "cuAFL-only deltas" below).
- `.gitignore` — excludes build outputs and nix-store symlinks.

## Target details

Mirrors `nix/targets/libpng.nix`:

- libpng: pnggroup/libpng v1.6.43 (fetched via nix).
- zlib: madler/zlib v1.3.1 (fetched via nix).
- Harness: `harness/targets/libpng_read_fuzzer.c` (shared with coqui).
- Arch: `sm_75` (RTX Titan).
- `--stack-size 32768` — libpng+zlib call chains
  (`png_read_image` → … → `inflate_fast`) exceed the default 8 KB stack.
  Overflow corrupts adjacent `.local` memory and aborts the batch. 32 KB
  gives sufficient headroom.
- `--slab-pool-size 2147483648` (2 GiB) — libpng's per-chunk allocations
  (tEXt, iCCP, zTXt) can exceed the 64 KB per-thread heap when fuzz
  mutations pump large chunk lengths. The shared slab absorbs these.

### pnglibconf.h patching

libpng ships a prebuilt `scripts/pnglibconf.h.prebuilt`. We replicate
coqui's sed+echo sequence (with cuAFL-specific additions) to produce a
patched `libpng_include/pnglibconf.h`:

| Macro stripped                     | Why |
|-----------------------------------|-----|
| `PNG_SETJMP_SUPPORTED`            | Forces pngconf.h to skip `#include <setjmp.h>`; coqui-cc's pass aborts on `_setjmp`. Same as libpng.nix. |
| `PNG_SIMPLIFIED_{READ,WRITE}_*`   | Drops the simplified-API entry points that use setjmp directly under their own guard. Same as libpng.nix. |
| `PNG_CONSOLE_IO_SUPPORTED` *(cuAFL only)* | Disables `fprintf(stderr, …)` branches in `pngerror.c`. cuAFL's runtime has no `fprintf` stub; `ExternalSymbolGatekeeper` aborts with "unresolved external 'fprintf' — Used by: png_app_error". |
| `PNG_STDIO_SUPPORTED` *(cuAFL only)* | Disables `fread`/`fwrite` in `pngrio.c`/`pngwio.c`. Same reason — no stdio stubs. The harness registers its own callbacks via `png_set_{read,write}_fn`, so these default paths are unused anyway. |
| `PNG_FLOATING_ARITHMETIC_SUPPORTED` *(cuAFL only)* | Disables `pow()`/`floor()` usage in `png.c` gamma-table computation. NVPTX cannot lower `llvm.pow.f64` without a math-runtime transform (coqui's `MathTransform.cpp` rewrites it to `__coqui_pow`; cuAFL has no such transform or stub). Falls back to libpng's built-in fixed-point arithmetic path (`png_log8bit` + `png_exp8bit` + `png_muldiv`). The public API `png_set_gamma(double, double)` still works — only the internal table computation changes. |

Appended after the strip:
```
#define PNG_DISABLE_ADLER32_CHECK_SUPPORTED
```
(Same as libpng.nix — lets the fuzzer explore zlib streams without valid
ADLER32 checksums.)

### cuAFL-only deltas beyond pnglibconf.h

- **`png_abort_stub.c`** — local stub that defines `void abort(void)`
  to call `__coqui_trap()` (PTX `trap; exit;`). libpng's error path
  ends in `PNG_ABORT()` (= `abort()` via `pngpriv.h:589`); with
  `-D PNG_NO_SETJMP`, every `png_error()` eventually hits this.
  coqui's nix build papers over the unresolved `abort` via its pass
  plugin's `--ignore-signal=abort`, which cuAFL does not support.
  Providing our own `abort()` definition removes the unresolved
  external symbol entirely. Same pattern as
  `coqui_mode/evaluation/bzip2/bz2_assert_stub.c` (which uses the
  same trick for `bz_internal_error` → `abort`).

- **No sanitizers.** coqui's nix build passes
  `-fsanitize=address,array-bounds,…` to its compiler; cuAFL's
  `coqui-cc` does not accept `-fsanitize` flags. cuAFL relies on
  AFL++ (CPU) for sanitizer-based crash verification. The device
  build is stripped-down for throughput.

- **No `--heap-size`/`--batch-size`.** cuAFL's `coqui-cc` does not
  accept these flags; the runtime derives heap at startup and reads
  `AFL_COQUI_BATCH_SIZE` from the env. See `bzip2/build.sh` for the
  same note.

## Seeds & dict

The nix `target-libpng-aflplusplus` build generates:
- ~60 auto-generated PNG seeds (color types, bit depths, Adam7
  interlace, ancillary chunks, filter types, edge cases).
- pngsuite seeds from libpng's `contrib/pngsuite/`.
- testpngs from libpng's `contrib/testpngs/` (excluding `crashers/`).
- `dict/png.dict` — chunk types, IHDR field values, zlib headers,
  magic sequences.

These are symlinked from `/tmp/cuafl-libpng-cpu/{seeds,dict}` at build
time.

## Gotchas

- **`ptxas` is very slow.** The libpng+zlib kernel is large (~25
  compilation units, heavy ASan instrumentation, all-inline). `ptxas
  -O1` can take 20–30 minutes on a typical workstation, and even
  longer when multiple cuAFL evaluation builds are running in
  parallel due to 10–50 GB RAM per ptxas. See
  `stb_image/README.md` — same pathology. If you need to iterate on
  build flags, expect the full `build.sh` run to be dominated by
  this step.

- **`AFL_COQUI_CUBIN` "mistyped" warning.** Harmless; the variable is
  read by the runtime but not yet in cuAFL's env-validator table
  (same as other evaluation targets).

- **Seeds are symlinks into `/nix/store`.** If the dry-run reports
  "No usable test cases", swap the `ln -sfn` in `build.sh` step 5
  for `cp -rL` (compare to `bzip2/build.sh` which materialises seeds
  for the same reason).

- **pnglibconf.h must be patched** — see the table above. The stock
  prebuilt header unconditionally defines `PNG_SETJMP_SUPPORTED`, so
  `-D PNG_NO_SETJMP` alone is not enough; the strip step is
  mandatory.

## Rebuilding

`build.sh` is idempotent; rerunning re-runs `nix build` (no-op if
cached) and rebuilds the cubin. To rebuild from a fully clean state:

```bash
rm -rf libpng_read_fuzzer* libpng_include seeds dict out \
       .libpng_read_fuzzer.build \
       /tmp/cuafl-libpng-cpu
./build.sh
```

## Build status (as of last test)

Tested end-to-end on RTX Titan (`sm_75`). All LLVM IR transforms pass;
build reaches `ptxas` (the final step) successfully:

- coqui-cc clang (NVPTX IR)                    PASS
- llvm-link + internalize + runtime link       PASS
- `coqui-link` + always-inline (stack 32 KB)   PASS
- `coqui` StaticGlobals                        PASS
- `coqui-asan` instrumentation (~22 K loads / 20 K stores) PASS
- `ExternalSymbolGatekeeper`                   PASS *(after pnglibconf strips + `png_abort_stub.c`)*
- `llc -march=nvptx64 -mcpu=sm_75`             PASS
- `ptxas -arch sm_75 -O1`                      SLOW (~20–30 min, peak ~55 GB RSS)

The `.cubin` and `.conf` are produced once ptxas completes.

The first two build attempts on this checkout surfaced and were fixed:

1. `ExternalSymbolGatekeeper: unresolved external 'fprintf'` — fixed
   by stripping `PNG_CONSOLE_IO_SUPPORTED` and `PNG_STDIO_SUPPORTED`
   from `pnglibconf.h`.
2. `ExternalSymbolGatekeeper: unresolved external 'abort'` — fixed
   by adding `png_abort_stub.c` (local `abort()` → `__coqui_trap()`).
3. `llc: Cannot select: fpow f64 in png_build_gamma_table` — fixed
   by stripping `PNG_FLOATING_ARITHMETIC_SUPPORTED` so libpng uses
   fixed-point arithmetic instead of `pow()`.
