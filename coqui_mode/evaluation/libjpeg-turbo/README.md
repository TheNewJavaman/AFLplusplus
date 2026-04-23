# coqui mode libjpeg-turbo evaluation target

A self-contained fuzz setup that builds libjpeg-turbo 3.0.4 (baseline JPEG
decode only) + oss-fuzz-style harness for coqui mode (`afl-fuzz --coqui`) and
the matching AFL++ instrumented CPU binary for host verification.

Mirrors `/home/gpizarro/coqui/nix/targets/libjpeg-turbo.nix` exactly: same
20 source files, same patched `jmorecfg.h`, same custom `jconfig.h` /
`jconfigint.h` / `jversion.h`, same `-I` paths, same `-D NO_GETENV`, same
`--stack-size 32768` (coqui-cc default). The CPU binary and seed/dict
corpus come from the coqui `target-libjpeg-turbo-aflplusplus` nix package.

## Files produced by `build.sh`

| Path | What it is |
|------|------------|
| `libjpeg_turbo_decompress_fuzzer.cubin` | GPU kernel (coqui-cc output; sm_75, 32 KiB stack) |
| `libjpeg_turbo_decompress_fuzzer.conf`  | Sidecar written by coqui-cc (stack/slab/arch) |
| `libjpeg_turbo_decompress_fuzzer_cpu`   | Symlink to AFL++ instrumented harness (from nix) |
| `seeds/` | Symlink into the nix store (24 diverse JPEGs; see below) |
| `dict/`  | Symlink into the nix store (JPEG marker dictionary) |
| `jpeg_include/` | Staged patched header tree + copies of the library `.c` files (regenerated on every run) |

The CPU binary and seed/dict dirs come from:

```
nix build '.#target-libjpeg-turbo-aflplusplus' --out-link /tmp/coqui-libjpeg-turbo-cpu
```

`build.sh` runs that command and symlinks the artifacts into the local
dir. All symlinks and `jpeg_include/` are in `.gitignore`.

## Quickstart

```
cd /home/gpizarro/cuAFL/coqui_mode/evaluation/libjpeg-turbo
./build.sh   # builds cubin + conf + CPU binary + seeds/dict
./fuzz.sh    # launches coqui mode afl-fuzz --coqui on GPU 0
```

Extra args to `fuzz.sh` are forwarded to `afl-fuzz` (after `-- <binary>`):

```
./fuzz.sh -V 300     # 5 min bench run
AFL_RESUME=1 ./fuzz.sh
```

## Dependencies

- coqui mode development build: `/home/gpizarro/cuAFL/afl-fuzz`
- `coqui-cc`: `/usr/local/bin/coqui-cc` (coqui mode compiler driver)
- nix 2.34+, with `/home/gpizarro/coqui` checkout providing the flake
- NVIDIA driver + CUDA on the host (for `libcuda.so.1`)
- GPU index 0 is an RTX Titan (sm_75) per `CLAUDE.local.md`

## Build status

`build.sh` was exercised on 2026-04-22 and ran end-to-end:

```
=== Build complete ===
-rw-rw-r-- 1 gpizarro gpizarro       45 Apr 22 15:45 libjpeg_turbo_decompress_fuzzer.conf
-rw-rw-r-- 1 gpizarro gpizarro 12194600 Apr 22 15:45 libjpeg_turbo_decompress_fuzzer.cubin
lrwxrwxrwx 1 gpizarro gpizarro       60 Apr 22 15:45 libjpeg_turbo_decompress_fuzzer_cpu -> /tmp/coqui-libjpeg-turbo-cpu/libjpeg_turbo_decompress_fuzzer
lrwxrwxrwx 1 gpizarro gpizarro       34 Apr 22 15:45 seeds -> /tmp/coqui-libjpeg-turbo-cpu/seeds
lrwxrwxrwx 1 gpizarro gpizarro       33 Apr 22 15:45 dict  -> /tmp/coqui-libjpeg-turbo-cpu/dict
  conf:
    stack_size=32768
    slab_pool_size=0
    arch=sm_75
```

The GPU cubin is **12.2 MB** — substantially larger than the ~3.1 MB
noted in the upstream harness comment. The extra is coqui mode's heap-ASan
instrumentation: `[coqui-asan] Instrumented 8528 load(s) and 6904
store(s)` adds runtime check wrappers at every memory op. No ptxas
errors.

## Target-specific quirks

### libc stubs for jerror.c

coqui mode's `CoquiPassPlugin` has a narrower libc port than upstream coqui.
Its `Libc.cpp` only rewrites `strlen`, `strcmp`, `strncmp`, `memcmp`,
`strchr`, and `strtod` to `__coqui_*` equivalents. The libjpeg-turbo
harness reaches three additional libc symbols through `jerror.c`:

- `snprintf` (from `format_message`)
- `fprintf` + `stderr` (from `output_message`)
- `exit` (from `error_exit`)

All three land in coqui mode's `ExternalSymbolGatekeeper` and abort the
build. The file `libjpeg_turbo_libc_stubs.c` provides device-side
no-op implementations:

- `snprintf` / `fprintf` return 0 and write an empty string
- `stderr` is a null `FILE*` dummy (never dereferenced by the stub)
- `exit` forwards to `__coqui_exit` (PTX `exit;`, thread dies cleanly)

These are behaviorally equivalent on the GPU because JPEG errors never
need to render human-readable strings — the thread terminates before
anything reads the buffer or stderr. The **CPU build is unchanged**
(links to the real libc from the nix aflplusplus target).

If/when coqui mode's `Libc.cpp` ports `snprintf`, `fprintf`, and `exit`, this
stub can be deleted.

### setjmp/longjmp

The nix spec and coqui docs note that libjpeg-turbo uses setjmp/longjmp
in the TurboJPEG error path. This is **not a GPU blocker** — the harness
(`libjpeg_turbo_decompress_fuzzer.c`) guards all setjmp/longjmp with
`#ifndef __COQUI_DEVICE__`, and coqui-cc defines `__COQUI_DEVICE__=1`.
The device build never sees setjmp; the CPU build retains the longjmp
error handler (oss-fuzz pattern).

### Staged `jpeg_include/`

libjpeg-turbo sources `#include "jmorecfg.h"` (quote form), which makes
the compiler search the source file's own directory first — before any
`-I` paths. To ensure our **patched** `jmorecfg.h` (with heavyweight
features `#undef`'d — no progressive, no IDCT scaling, no block
smoothing, no color quant, no 12/16-bit) is what's included rather than
the unpatched upstream copy, `build.sh` **copies** (not symlinks) each
library `.c` file into `jpeg_include/` next to the patched header. This
is the same trick the upstream `libjpeg-turbo.nix` uses.

### coqui-cc does not accept sanitizer flags

The nix spec passes `${sanitizers.default}` (a block of `-fsanitize=...`
options) to the upstream coqui driver. coqui mode's `coqui-cc` does not
expose `-fsanitize` — sanitizers are baked into `CoquiPassPlugin.so`
and applied unconditionally. `build.sh` therefore omits those flags.

### Seeds and dictionary

24 diverse JPEGs cover 1x1 to 64x64, grayscale + RGB, 4:4:4 / 4:2:2 /
4:2:0 subsampling, quality 1–100, restart markers (DRI), flat/gradient/
checker fill patterns, and non-MCU-aligned dimensions (7x5). All seeds
are pre-generated by the libjpeg-turbo compressor in the nix build.
The JPEG marker dictionary (`jpeg.dict`) contains SOI/EOI/SOF/DHT/DQT/
DRI/SOS markers + JFIF/EXIF signatures + common SOF sampling factors.

## Gotchas

- **Don't `git add` the symlinks or `jpeg_include/`.** The binary +
  `seeds` + `dict` all point at `/tmp/coqui-libjpeg-turbo-cpu/…` or
  the nix store. The staged `jpeg_include/` tree is regenerated on
  every `build.sh` run. All are in `.gitignore`.
- **GPU device pinning.** Everything here assumes device index 0 (RTX
  Titan, sm_75) per `CLAUDE.local.md`. Override with `AFL_COQUI_DEVICE=N`.
- **`coqui-cc` flag surface is narrower than the coqui driver.** It
  does not accept `--heap-size`, `--batch-size`, or `-fsanitize=…`;
  `build.sh` only passes `-arch`, `--stack-size`, `--slab-pool-size`,
  `-I`, `-D`, and source files.
- **libjpeg source location.** `build.sh` finds
  `/nix/store/<hash>-source/` by scanning the closure of the CPU nix
  result for a path containing `jpeglib.h` + `jmorecfg.h` +
  `jversion.h.in`. Do not copy the source out of the store — the path
  is read-only and re-derives cleanly.
- **Harness paths.** The libFuzzer-style harness and SIMD stubs live at
  `/home/gpizarro/coqui/harness/targets/libjpeg_turbo_decompress_fuzzer.c`
  and `libjpeg_turbo_stubs.c`. `build.sh` references them via absolute
  paths.
- **Orphan-safe.** coqui mode's `afl-fuzz` installs `PR_SET_PDEATHSIG` on the
  forkserver (fixed in `aadd355b`), so killing this shell cleanly tears
  down the fuzzer. Closing a tmux pane does NOT reach the fuzzer — use
  `pkill -u "$USER" -KILL -f "afl-fuzz --coqui"` before relaunching and
  check `nvidia-smi` shows memory released.
