# libxml2 — coqui mode evaluation target

Port of `/home/gpizarro/coqui/nix/targets/libxml2.nix` to a self-contained
evaluation directory for `coqui mode --coqui`.

## Build

```
./build.sh
```

Produces (in the current directory):

- `libxml2_xml_read_fuzzer.cubin` — GPU kernel (sm_75)
- `libxml2_xml_read_fuzzer.conf`  — runtime config sidecar emitted by coqui-cc
- `libxml2_xml_read_fuzzer_cpu`   — AFL++ instrumented CPU binary (symlink to
  `/tmp/coqui-libxml2-cpu/libxml2_xml_read_fuzzer`)
- `seeds/`                        — seed corpus (symlink into /nix/store)
- `dict/`                         — dictionary (symlink into /nix/store)

All intermediate artifacts go into `.libxml2_fuzzer.build/` (gitignored):
generated `config.h`, generated `xmlversion.h`, and the patched copies of
`error.c` / `xmlstring.c`.

## Fuzz

```
./fuzz.sh              # runs indefinitely
./fuzz.sh -V 60        # stop after 60s
```

Pass-through args go to `afl-fuzz` before the `--` separator.

## Build status: BLOCKED (coqui mode coqui-cc lacks variadic transform)

`build.sh` runs through steps 1-3 successfully:

1. `nix build .#target-libxml2-aflplusplus` — produces the CPU binary, seeds,
   and dict under `/tmp/coqui-libxml2-cpu/`.
2. Resolve libxml2 source from the CPU build's `-source` closure
   (`/nix/store/...-source`).
3. Generate `config.h`, `xmlversion.h`, and patch `error.c`/`xmlstring.c`
   inline, exactly mirroring `libxml2.nix` buildPhase.

Step 4 (coqui-cc compilation) **fails** in the coqui-link LLVM pass:

```
LLVM ERROR: [coqui-cc] IntrinsicReject: 'llvm.va_start' not handled.
Port the relevant transform or disable this feature in the target.
```

### Why it fails here

libxml2 is deeply variadic:

- `xmlprintf`/`xmlsnprintf`/`xmlvsnprintf` in `xmlstring.c` (only one of these,
  `xmlStrVASPrintf`, is stubbed by the nix spec — the others remain).
- Tree dumping, XPath error formatting, and error reporting across `tree.c`,
  `xpath.c`, `entities.c`, `uri.c`, `xmlmemory.c`, `valid.c`,
  `parserInternals.c`, `error.c` (13 .bc modules contain `va_start` after
  our patches).
- The coqui runtime (`runtime.bc`) itself defines `__coqui_vsnprintf`, which
  also uses `llvm.va_start`.

Upstream coqui handles this via a `VariadicTransform`/`PrintfTransform` IR
pass (see `/home/gpizarro/coqui/transforms/`). coqui mode's coqui-cc has
`IntrinsicReject` — which explicitly hard-fails on `llvm.va_start` — but
the corresponding transform has not been ported. `coqui_mode/passes/`
contains the lowering transforms currently enabled, and the variadic one
is absent:

```
coqui_mode/passes/IntrinsicReject.cpp   -> blocklists llvm.va_start/end/arg/copy
coqui_mode/passes/PrintfTransform.cpp   -> does NOT exist
coqui_mode/passes/VariadicTransform.cpp -> does NOT exist
```

### What to do next

1. **Port the coqui variadic transform** into `coqui_mode/passes/` (outside
   the scope of this target directory). Once the pass runs before
   `IntrinsicReject`, this exact `build.sh` should succeed unchanged.
2. **OR** patch libxml2 more aggressively to stub every variadic entrypoint
   (xmlprintf, xmlsnprintf, xmlvsnprintf, xmlXPath*Error, xmlReportError,
   ...). Not attempted here because the fanout is large and the end result
   is a libxml2 with no error reporting — comparable in effort to porting
   the transform, and with worse fuzzer signal.

## Quirks handled

- **Config header generation**: libxml2 has no `config.h` in the source tree;
  the nix spec hand-writes a minimal one with only the POSIX `HAVE_*_H` that
  LibcTransform redirects. Reproduced verbatim.
- **xmlversion.h template substitution**: `sed` over 33 feature flags to
  match the nix spec's feature selection (OUTPUT, PUSH, SAX1, XPATH,
  ISO8859X, TREE enabled; everything else off).
- **error.c patches**: `xmlCopyError`, `xmlFormatError`, `xmlResetLastError`,
  and `xmlRaiseMemoryError` stubbed to no-ops via depth-aware awk so the
  shared `xmlLastError` global is never touched (races across 131k threads)
  and the vsnprintf-heavy error path never fires (would overflow the CUDA
  hardware call stack).
- **xmlstring.c patch**: `xmlStrVASPrintf` stubbed to return -1 — callers
  fall through with a NULL message.
- **Stripping /nix/store read-only bits**: `chmod -R u+w` on the build dir
  after copying from `/nix/store/...`, otherwise the awk patches can't
  overwrite the intermediate files.
- **coqui-cc flag mismatch**: the nix spec passes `--heap-size 131072` and
  `--batch-size 65536`; coqui mode's coqui-cc accepts neither. The coqui mode runtime
  derives heap from arch and batch size from `AFL_COQUI_BATCH_SIZE` at
  runtime, so those flags are intentionally dropped from `build.sh`.
- **Source list**: mirrors the 22-file nix list exactly (libxml2 core +
  harness + stubs). `HTMLparser.c` / `HTMLtree.c` are excluded because
  `WITH_HTML=0`.
