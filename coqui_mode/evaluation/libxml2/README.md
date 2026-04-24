# coqui mode evaluation target: libxml2

libxml2 XML parser fuzzer, adapted from coqui's nix target
(the `libxml2` target in the legacy coqui codebase) for coqui mode's
`coqui-cc` compiler and `afl-fuzz --coqui` runtime.

libxml2 source (`GNOME/libxml2` tag `v2.13.4`) is fetched by `build.sh`
into `.build/`. The 22-file explicit source list (core parser + SAX +
tree + xpath + URI + xmlsave + encoding + xmlIO + xmlmemory + xmlstring
+ globals + hash + dict + list + valid + parserInternals + threads +
chvalid + buf + entities + xpath + error) is compiled with both
`afl-clang-fast` (CPU) and `coqui-cc` (GPU). HTML and reader/writer
modules are excluded (matches nix spec's feature selection).

## Quick start

```bash
./build.sh    # builds cubin + conf + CPU binary + seeds
./fuzz.sh -V 300 -t 5000   # fuzz for 300s with 5s dry-run timeout
```

`build.sh` is idempotent. The libxml2 source tarball is cached in
`.build/GNOME-libxml2-v2.13.4/`; re-running skips the fetch if the
cache is populated.

## Files

- `build.sh` — self-contained build: fetches libxml2 at the pinned tag,
  generates `config.h` / `xmlversion.h`, patches `error.c` / `xmlstring.c`,
  and invokes `afl-clang-fast` + `coqui-cc`.
- `harness.c` — 8-path libxml2 fuzzer (byte[0]&0x07 selects pull/push/
  tree+save/xpath/URI/deep-copy/encoding/input-derived-xpath).
- `libxml2_stubs.c` — musl-atomic stubs for the GPU build (libxml2's
  globals.c initialization pulls in syscall-using musl atomics).
- `seeds/min.xml` — minimal seed (`<a/>`, 4 bytes).
- `fuzz.sh` — workspace prep + suggested launch commands.

## Target details

- Upstream: `GNOME/libxml2` tag `v2.13.4`.
- Arch: `sm_75` (RTX Titan) — override via `ARCH=sm_XX ./build.sh`.
- `--stack-size 32768` (default libxml2.nix value — needed for deep
  parser recursion + xpath evaluator).
- `--slab-pool-size 2147483648` (2 GiB) — from libxml2.nix; the
  per-thread 64KB heap plus the global slab pool hosts xmlmemory
  allocations for 131k threads.

### coqui mode-specific deltas from upstream libxml2

The nix spec patches two source files inline to make libxml2 survive
on 131k-wide CUDA threads. `build.sh` replicates those patches
verbatim with depth-aware awk.

- **`error.c` patches**: `xmlCopyError`, `xmlFormatError`,
  `xmlResetLastError`, and `xmlRaiseMemoryError` are stubbed to
  no-ops. They otherwise race on the shared `xmlLastError` global
  (no GPU-side synchronization) and reach `__coqui_vsnprintf` via
  the error-formatting chain, which allocates 512+ bytes on the CUDA
  hardware call stack and overflows from the deep parser call chain.
- **`xmlstring.c` patch**: `xmlStrVASPrintf` is stubbed to return -1.
  Callers handle the -1 return by falling through with a NULL
  message, which preserves error-code propagation without the
  vsnprintf blow-up.

### GPU build constraints

`COQUI_LLC_OPT=-O1` is the default for this target: the post-ASan-pass
LLVM module has ~211k instrumented memory accesses and llc's default
-O2 register allocator stalls indefinitely. `-O1` completes in
~2 minutes. `ptxas` then takes roughly 15 minutes and peaks at ~20 GB
RSS. `build.sh` uses `flock /tmp/coqui-cc.lock` around the `coqui-cc`
invocation to serialize with other parallel target builds.

### CPU build notes

The CPU binary uses `-fsanitize=fuzzer` which AFL++'s `afl-clang-fast`
rewrites into `libAFLDriver.a`. That driver uses persistent mode
(`__AFL_LOOP(1000)`) internally, so no separate driver is needed.

## Rebuilding from scratch

```bash
rm -rf libxml2_xml_read_fuzzer{,.cubin,.conf,_cpu} \
       .build .libxml2_fuzzer.build .libxml2_xml_read_fuzzer.build \
       out build.log
./build.sh
```
