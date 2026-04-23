# libyaml — coqui mode evaluation target

Fuzz target adapting coqui's **libyaml 0.2.5** (parser-only) benchmark for the
coqui mode `--coqui` mode runner.

Mirrors the nix spec at
`/home/gpizarro/coqui/nix/targets/libyaml.nix`, producing a self-contained
build+fuzz workflow that does not require the coqui nix wrapper at fuzz time.

## Layout

```
build.sh            idempotent build (CPU binary via nix + GPU cubin via coqui-cc)
fuzz.sh             launches coqui mode afl-fuzz --coqui against this target
libyaml_stubs.c     coqui mode-only glue: strdup, memcpy, memset, memmove
.gitignore          ignores all generated artifacts

# Produced by build.sh (ignored):
libyaml_src/                    patched libyaml sources (yaml_private.h buffer shrink)
libyaml_parser_fuzzer.cubin     sm_75 GPU kernel
libyaml_parser_fuzzer.conf      companion config from coqui-cc
libyaml_parser_fuzzer_cpu       -> /tmp/coqui-libyaml-cpu/libyaml_parser_fuzzer
seeds                           -> /tmp/coqui-libyaml-cpu/seeds
dict                            -> /tmp/coqui-libyaml-cpu/dict
out/                            afl-fuzz output
```

## Build

```bash
./build.sh
```

Steps (mirrors `nix/targets/libyaml.nix` + `nix/cpu-target-specs.nix::libyaml`):

1. `nix build '.#target-libyaml-aflplusplus'` -> `/tmp/coqui-libyaml-cpu`
2. Find the libyaml `-source` derivation in the CPU build closure
3. Copy `api.c reader.c scanner.c parser.c loader.c yaml_private.h` into
   `./libyaml_src/`, then `sed` yaml_private.h to shrink:
     - `INPUT_RAW_BUFFER_SIZE  16384 -> 512`  (keeps INPUT_BUFFER_SIZE under 64KB heap)
     - `INITIAL_{STACK,QUEUE,STRING}_SIZE  16 -> 4`
4. `coqui-cc` with `-arch sm_75`, coqui's `-D YAML_DECLARE_STATIC`,
   version macros, plus coqui mode-only `-D NDEBUG`, linking the patched sources +
   the upstream harness `harness/targets/libyaml_parser_fuzzer.c` +
   `libyaml_stubs.c`.
5. Symlink CPU binary, seeds, dict from the nix output.

## Fuzz

```bash
./fuzz.sh                # run indefinitely
./fuzz.sh -V 60          # stop after 60 seconds
./fuzz.sh -M main        # main sync node
```

Sets these env vars before exec'ing `/home/gpizarro/cuAFL/afl-fuzz --coqui gpu0`:
- `AFL_COQUI_CUBIN=$(pwd)/libyaml_parser_fuzzer.cubin`
- `AFL_I_DONT_CARE_ABOUT_MISSING_CRASHES=1`
- `AFL_SKIP_CPUFREQ=1`
- `AFL_SKIP_BIN_CHECK=1`
- `AFL_NO_UI=1`

## coqui mode-specific quirks

### `-D NDEBUG`

libyaml's `api.c` / `scanner.c` call `assert()` from `<assert.h>`. On NVPTX
clang these lower to `__assert_fail`. coqui's upstream compiler has a
`LibcTransform` pass that rewrites `__assert_fail -> __coqui_assert_fail`, but
coqui mode's `coqui-cc` driver runs only the `coqui-link,always-inline` pipeline —
`LibcTransform` is NOT ported, and the `ExternalSymbolGatekeeper` hard-fails
on any unresolved libc call.

Workaround: compile with `-D NDEBUG` so `<assert.h>` expands `assert()` to
`((void)0)` at preprocess time. Semantics differ from the nix GPU build,
which keeps asserts as a runtime trap; for fuzzing this is acceptable because
host AFL++ verification (instrumented CPU binary) still honors asserts.

Analogous to stb_image's `-D STBI_ASSERT(x)=` workaround in the sibling
`../stb_image/` target.

### `libyaml_stubs.c` — strdup + memcpy + memmove + memset

Four more unresolved-symbol issues after NDEBUG:

| libc symbol | coqui mode runtime.bc status                              |
|-------------|------------------------------------------------------|
| `strdup`    | Not in runtime. Implemented via `__coqui_malloc` + `__coqui_strlen` + byte loop. |
| `memcpy`    | Not in runtime. `__builtin_memcpy` (clang lowers inline). |
| `memset`    | Not in runtime. `__builtin_memset`.                  |
| `memmove`   | Not in runtime. Manual forward/backward loop.        |

(The coqui mode coqui-cc runtime exports only `__coqui_malloc/calloc/free/realloc/`
`memcmp/strlen/strcmp/strncmp/strchr/strtod/trap` — far smaller than coqui's
GPU runtime.)

These stubs define the bare libc names (`strdup`, `memcpy`, ...) so the
gatekeeper sees them as locally defined. They compile into the same linked
BC as the libyaml sources, so the harness picks them up automatically.

### No `--stack-size` / `--slab-pool-size` override

`libyaml.nix` does not override coqui's default stack size (32768) or set a
slab pool, so `build.sh` also uses coqui-cc defaults. The shrunken
`INPUT_RAW_BUFFER_SIZE=512` and `INITIAL_*_SIZE=4` keep allocations under the
per-thread 64 KB GPU heap.

## Build status

The `build.sh` script was validated end-to-end:
- Step [1/5] (nix CPU build) completes and stamps `/tmp/coqui-libyaml-cpu`.
- Step [2/5] resolves the libyaml source to
  `/nix/store/nvnfhj4nq7m33alwv67mrsjycvas46j5-source`.
- Step [3/5] writes the patched `libyaml_src/`.
- Step [4/5] (`coqui-cc`) emits IR + PTX without gatekeeper errors;
  `ptxas -arch=sm_75 -O1` runs to produce `libyaml_parser_fuzzer.cubin`.
  One benign `ptxas warning: Unresolved extern variable '__coqui_virgin_map'`
  is expected — that symbol is resolved at runtime load time (same pattern as
  the sibling bzip2 and stb_image targets).
- Step [5/5] wires up symlinks for CPU binary, seeds, dict.

Note: `ptxas` on the 22 MB transformed PTX is CPU-intensive (~10 minutes with
no contention; longer when multiple evaluation builds run in parallel).

## Provenance

- GPU build flags: `/home/gpizarro/coqui/nix/targets/libyaml.nix`
- CPU build flags: `/home/gpizarro/coqui/nix/cpu-target-specs.nix::libyaml`
- Upstream harness: `/home/gpizarro/coqui/harness/targets/libyaml_parser_fuzzer.c`
- coqui mode coqui-cc driver: `/usr/local/bin/coqui-cc`
- coqui mode afl-fuzz binary: `/home/gpizarro/cuAFL/afl-fuzz`
