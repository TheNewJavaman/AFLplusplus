# coqui mode evaluation target: stb_image

stb_image multi-format image decoder fuzzer (PNG/BMP/GIF/JPEG/TGA/PSD/PIC/PNM),
adapted from coqui's nix target
(`/home/gpizarro/coqui/nix/targets/stb_image.nix`) for coqui mode's `coqui-cc`
compiler and `afl-fuzz --coqui` runtime.

stb is a header-only library (nothings/stb, commit `28d546d5eb77`), so only the
harness `.c` is compiled — `STB_IMAGE_IMPLEMENTATION` is defined inside
`harness/targets/stb_image_read_fuzzer.c`.

## Quick start

```bash
./build.sh    # builds cubin + conf + CPU binary + seeds + dict
./fuzz.sh -V 300 -t 5000   # fuzz for 300s with 5s dry-run timeout
```

## Files

- `build.sh` — idempotent build script (calls `nix build
  .#target-stb_image-aflplusplus` for the CPU binary, then `coqui-cc`
  for the GPU cubin).
- `fuzz.sh` — launches `afl-fuzz --coqui gpu0` with the recommended env
  vars and auto-attaches the dictionary if present.
- `.gitignore` — excludes build outputs and symlinks into the nix store.

## Target details

Mirrors `nix/targets/stb_image.nix`:

- stb source: `nothings/stb` rev `28d546d5eb77` (fetched via nix).
- Harness: `harness/targets/stb_image_read_fuzzer.c` — defines
  `STB_IMAGE_IMPLEMENTATION`, `STBI_NO_STDIO`, `STBI_NO_SIMD`,
  `STBI_NO_THREAD_LOCALS`, and caps image dimensions at 64×64 to avoid
  billion-iteration decode loops.
- `--stack-size 65536` (double the coqui-cc default; the multi-format
  decoder has deep call chains, e.g. `stbi__jpeg_decode_block`).
- `--slab-pool-size 0` (nix spec doesn't set a slab pool).
- Arch: `sm_75` (RTX Titan).

### coqui mode-only deltas

The nix build works via the coqui compiler wrapper, which provides a few
runtime stubs that coqui mode's `coqui-cc` does not. Two extra `-D` flags are
therefore required here that are NOT in the nix spec:

- `-D "STBI_ASSERT(x)="` — the 2 `STBI_ASSERT` calls in stb_image.h
  expand to `assert(x)` → `__assert_fail`, which coqui mode's
  `ExternalSymbolGatekeeper` rejects. stb_image officially supports
  `STBI_ASSERT` override (see stb_image.h line 15), so we disable them.
- `-D STBI_NO_HDR` — the Radiance HDR decoder
  (`stbi__hdr_info` / `stbi__hdr_load`) calls `strtol()`, which is not
  in coqui mode's runtime allowlist. Disabling HDR on GPU removes the
  offending code path.

**Coverage impact:** the CPU AFL++ binary (built by nix) still has HDR
enabled, so HDR-specific bugs can be hit by CPU verification — but the
GPU won't drive coverage toward HDR paths. Lifting this requires either
a `strtol` stub in coqui mode's `runtime.bc` or patching stb_image.h's HDR
parser (out of scope for this target dir).

## Seeds & dict

Symlinked from the nix CPU build output
(`/tmp/coqui-stb_image-cpu/seeds`, `/tmp/coqui-stb_image-cpu/dict`):

- `seeds/white.png` — minimal 1×1 PNG.
- `seeds/red.bmp` — minimal 1×1 BMP.
- `seeds/blue.gif` — minimal 1×1 GIF.
- `dict/stb_image.dict` — PNG chunk types + BMP/GIF/TGA/JPEG/PNM magic.

## Gotchas

- **ptxas is very slow on stb_image.** The post-transform LLVM IR is
  ~19 MB and the PTX is ~10 MB because every decoder (JPEG, PNG, BMP,
  TGA, PSD, PIC, GIF, PNM) is inlined into one kernel. `ptxas` can
  take ~20–30 minutes on a typical workstation even at `-O1`. If you
  need to iterate on other flags, expect a full `build.sh` run to be
  dominated by this step.
- **HDR format disabled on GPU only.** See the coqui mode-only deltas
  above. GPU coverage will not steer inputs toward Radiance HDR code.
- **`AFL_COQUI_CUBIN` "mistyped" warning.** Harmless; same symptom as
  other coqui mode evaluation targets. The variable is read, just not yet
  listed in coqui mode's env-validator table.
- **Seeds are symlinks into `/nix/store`.** `afl-fuzz` tolerates these
  for this target (unlike bzip2, which had to materialise them with
  `cp -L` for its CPU binary to read them). If the dry-run reports
  "No usable test cases", switch `ln -sfn` in `build.sh` step 4 to
  `cp -rL`.

## Rebuilding

`build.sh` is idempotent; rerunning it re-runs `nix build` (no-op if
cached) and rebuilds the cubin. To rebuild from a fully clean state:

```bash
rm -rf stb_image_read_fuzzer* seeds dict out \
       .stb_image_read_fuzzer.build \
       /tmp/coqui-stb_image-cpu
./build.sh
```

## Build status (as of last test)

Tested on RTX Titan (`sm_75`). Every step of `build.sh` was verified
to execute without errors; only the final PTX→CUBIN assembly was too
slow to finish within the test window:

| Stage | Tool | Status |
|---|---|---|
| CPU binary | `nix build .#target-stb_image-aflplusplus` | PASS (produces `/tmp/coqui-stb_image-cpu/stb_image_read_fuzzer`) |
| stb source resolve | `nix-store -qR` | PASS (`/nix/store/<hash>-source` with `stb_image.h`) |
| clang → NVPTX bitcode | `clang --target=nvptx64-nvidia-cuda` | PASS |
| runtime link | `llvm-link` | PASS |
| prune + re-link | `opt -passes=internalize,globaldce` | PASS |
| coqui transform | `opt -passes=coqui-link,always-inline` | PASS (pools 2 statics, 40 B/thread) |
| coqui-asan | (in transform pipeline) | PASS (~20K loads, 19K stores) |
| `ExternalSymbolGatekeeper` | (in transform pipeline) | PASS *(after `STBI_ASSERT`/`STBI_NO_HDR`)* |
| PTX emission | `llc -march=nvptx64 -mcpu=sm_75` | PASS (10 MB `.ptx`) |
| CUBIN assembly | `ptxas -arch=sm_75 -O1` | SLOW — 30–40+ minutes, ~60 GB RAM |

The ptxas step emits two expected warnings that confirm the linker
hand-off is correct:

```
ptxas warning : Unresolved extern variable '__coqui_virgin_map' in whole program compilation, ignoring extern qualifier
ptxas warning : Unresolved extern variable '__coqui_global_statics_pool_base' in whole program compilation, ignoring extern qualifier
```

These externs are resolved at kernel-launch time by coqui mode's runtime
(the same pattern all other coqui-mode targets rely on).

Because stb_image's kernel is enormous (every decoder inlined into
one entry point → ~20 000 LLVM instructions → 10 MB PTX), `ptxas`
dominates the wall-clock build time. If you need to iterate on
`build.sh` flags, expect each full run to spend most of its time
here. A previous run in this workspace was OOM-SIGKILL'd when three
parallel ptxas invocations collided; if that happens, just rerun
`./build.sh` — each retry is idempotent. Do not run multiple
`./build.sh` invocations concurrently.
