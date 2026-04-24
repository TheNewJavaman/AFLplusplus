# coqui mode libjpeg-turbo evaluation target

Self-contained fuzz setup that builds libjpeg-turbo 3.0.4 (baseline JPEG
decode only) + oss-fuzz-style harness for coqui mode (`afl-fuzz --coqui`)
and the matching AFL++-instrumented CPU binary for host verification.

**No external build dependencies beyond this repo.** Upstream libjpeg-turbo
sources are fetched directly from GitHub at tag `3.0.4` via `git clone`;
the old nix-based path has been removed.

## Files

- `build.sh` — fetches upstream source, stages the patched jpeg_include
  tree, builds the AFL++ CPU binary with `afl-clang-fast`, and the GPU
  cubin with `coqui-cc`.
- `fuzz.sh` — preps the fuzz workspace and prints suggested `afl-fuzz`
  launch commands.
- `harness.c` — in-tree libFuzzer-style harness
  (`LLVMFuzzerTestOneInput` → `jpeg_read_header` + `jpeg_start_decompress`
  + `jpeg_read_scanlines`; rejects non-8-bit precision and images larger
  than 64x64 to fit the GPU's 64 KB bump allocator).
- `libjpeg_turbo_stubs.c` — in-tree stubs for jsimd_* symbols
  (WITH_SIMD=0) and j12init_*/j16init_* 12/16-bit bit-depth functions
  (we compile 8-bit only). Needed on both CPU and GPU — libjpeg-turbo
  references these unconditionally even when the bit-depth variants are
  disabled at build time.
- `libjpeg_turbo_libc_stubs.c` — **device-only** stubs for `snprintf`,
  `fprintf`, `stderr`, and `exit` (coqui mode's Libc.cpp does not port
  stdio-format functions or `exit`, so the GPU link needs these). The
  CPU build does NOT compile this file.
- `seeds/min.jpg` — minimal 160-byte 1x1 JPEG generated with
  ImageMagick (`convert -size 1x1 xc:black JPEG:seeds/min.jpg`).

## Outputs (produced by `build.sh`, gitignored)

| Path | What it is |
|---|---|
| `libjpeg_turbo_decompress_fuzzer.cubin` | GPU kernel (coqui-cc; default sm_75, 32 KiB stack) |
| `libjpeg_turbo_decompress_fuzzer.conf`  | Sidecar written by coqui-cc (stack/slab/arch) |
| `libjpeg_turbo_decompress_fuzzer_cpu`   | AFL++-instrumented CPU binary (persistent mode via `-fsanitize=fuzzer`) |
| `.build/libjpeg-turbo-3.0.4/`           | Cached upstream source tree (git clone at tag `3.0.4`) |
| `.build/cpu-obj/`                       | Intermediate `.o` files for the CPU link |
| `jpeg_include/`                         | Staged patched header tree + copies of the library `.c` files (regenerated on every run) |

## Quickstart

```bash
./build.sh
./fuzz.sh    # prints recommended afl-fuzz commands; does not auto-launch
```

Overrides:

- `ARCH=sm_XX` — GPU compute capability (default sm_75, matches the RTX Titan on this box).
- `AFL_CC=/abs/path/afl-clang-fast` — path to the afl-clang-fast binary.
  Default: `../../../afl-clang-fast` relative to the target dir (i.e.
  the compiled one in the project root).  Do NOT set this to a relative
  path that does not exist — afl-clang-fast reads `AFL_CC` as the real
  underlying C compiler (e.g. `/usr/bin/clang`) and will recurse if you
  point it back at itself.
- `COQUI_CC` — path to coqui-cc (default `/usr/local/bin/coqui-cc`).
- `LIBJPEG_CACHE` — override the cached source tree location.

## Build details

Mirrors the legacy `libjpeg-turbo` target in the coqui codebase:

- **Source pin:** `libjpeg-turbo/libjpeg-turbo` @ tag `3.0.4`.
- **Sources:** 19 baseline-only `.c` files (jdapimin, jdapistd, jdatasrc,
  jdcoefct, jdcolor, jddctmgr, jdhuff, jdinput, jdmainct, jdmarker,
  jdmaster, jdpostct, jdsample, jidctint, jmemmgr, jmemnobs, jerror,
  jutils, jcomapi). Arithmetic coding, progressive, multi-scan, block
  smoothing, IDCT scaling, color quantization, upsample merging, save
  markers, input smoothing, and 12/16-bit paths are all excluded to
  shrink the NVPTX module and avoid ptxas code-layout pathologies.
- **Patched headers:** `jmorecfg.h` has heavyweight features `#undef`-ed
  via `sed`; `jconfig.h`/`jconfigint.h` are written inline from heredocs
  (no arithmetic coding, 8-bit only, WITH_SIMD=0); `jversion.h` is
  generated from `jversion.h.in`.
- **Quote-include trick:** sources are COPIED into `jpeg_include/` next
  to the patched `jmorecfg.h` so the `#include "jmorecfg.h"` directive
  finds the patched copy (quote includes search the source file's
  directory first before -I paths).
- **Sanitizers (CPU):** the legacy `sanitizers.default` set —
  `-fsanitize=address,array-bounds,bool,builtin,enum,integer-divide-by-zero,null,object-size,return,returns-nonnull-attribute,shift,signed-integer-overflow,unsigned-integer-overflow,unreachable,vla-bound`
  with `-fno-sanitize-recover` on the same list minus
  `unsigned-integer-overflow`. Plus `-fsanitize=fuzzer` at link time,
  which afl-clang-fast replaces with libAFLDriver.a (persistent mode
  driver).
- **Sanitizers (GPU):** none explicit — `coqui-cc` does not expose
  `-fsanitize`; coqui mode's `CoquiPassPlugin.so` injects device-side
  ASan/UBSan unconditionally.
- **GPU defines:** `-D NO_GETENV` (no libc env on GPU).
- **Stack/slab:** `--stack-size 32768` (coqui-cc default, matches nix
  spec); `--slab-pool-size 0` (nix spec does not set a slab pool).

The CPU link compiles each `.c` file to `.o` separately to avoid
overflowing `afl-clang-fast`'s 2048-slot param buffer — afl-cc's
preprocessor-macro expansions (`__AFL_LOOP`, `__AFL_FUZZ_INIT`,
`__AFL_FUZZ_TESTCASE_BUF`, …) combined with the sanitizer flag set
and 20+ .c sources overflowed the single-invocation limit.

The GPU build is serialised with `flock /tmp/coqui-cc.lock` — ptxas
at `-O1` is memory-heavy (tens of GB), and concurrent `coqui-cc`
invocations during parallel target rebuilds would OOM the box.

## Target-specific quirks

### libc stubs for jerror.c

coqui mode's `CoquiPassPlugin` has a narrower libc port than upstream
coqui. Its `Libc.cpp` only rewrites `strlen`, `strcmp`, `strncmp`,
`memcmp`, `strchr`, and `strtod` to `__coqui_*` equivalents. The
libjpeg-turbo harness reaches three additional libc symbols through
`jerror.c`:

- `snprintf` (from `format_message`)
- `fprintf` + `stderr` (from `output_message`)
- `exit` (from `error_exit`)

All three land in coqui mode's `ExternalSymbolGatekeeper` and abort
the build. `libjpeg_turbo_libc_stubs.c` provides device-side no-op
implementations:

- `snprintf` / `fprintf` return 0 and write an empty string
- `stderr` is a null `FILE*` dummy (never dereferenced by the stub)
- `exit` forwards to `__coqui_exit` (PTX `exit;`, thread dies cleanly)

These are behaviorally equivalent on the GPU because JPEG errors
never need to render human-readable strings — the thread terminates
before anything reads the buffer or stderr. The **CPU build is
unchanged** (links to the real libc).

### setjmp/longjmp

libjpeg-turbo uses setjmp/longjmp in the TurboJPEG error path. This
is **not a GPU blocker** — the harness (`harness.c`) guards all
setjmp/longjmp with `#ifndef __COQUI_DEVICE__`, and coqui-cc defines
`__COQUI_DEVICE__=1`. The device build never sees setjmp; the CPU
build retains the longjmp error handler (oss-fuzz pattern).

### Staged `jpeg_include/`

See "Quote-include trick" above. The staging dir is regenerated on
every `./build.sh` run and is in `.gitignore`.

## Gotchas

- **GPU device pinning.** Everything here assumes device index 0 (RTX
  Titan, sm_75) per `CLAUDE.local.md`. Override with
  `AFL_COQUI_DEVICE=N`.
- **`coqui-cc` flag surface is narrower than the upstream coqui driver.**
  It does not accept `--heap-size`, `--batch-size`, or `-fsanitize=…`;
  `build.sh` only passes `-arch`, `--stack-size`, `--slab-pool-size`,
  `-I`, `-D`, and source files.
- **Orphan-safe.** coqui mode's `afl-fuzz` installs `PR_SET_PDEATHSIG`
  on the forkserver (fixed in `aadd355b`), so killing this shell
  cleanly tears down the fuzzer. Closing a tmux pane does NOT reach
  the fuzzer — use `pkill -u "$USER" -KILL -f "afl-fuzz --coqui"`
  before relaunching and check `nvidia-smi` shows memory released.
