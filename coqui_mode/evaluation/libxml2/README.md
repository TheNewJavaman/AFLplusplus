# libxml2 — coqui mode evaluation target

Port of the `libxml2` target in the legacy coqui codebase to a self-contained
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

## Build status: BLOCKED at ptxas (NVPTX backend type-mismatch at scale)

`build.sh` now runs steps 1-4 successfully after the variadic transform port
(commit 7ee8da57 added `coqui_mode/passes/Variadic.cpp`, `Sprintf.cpp`,
`Reloc.cpp`, `Math.cpp`) and the outlined ASan port (Task 1/2 of this PR):

1. `nix build .#target-libxml2-aflplusplus` — produces the CPU binary, seeds,
   and dict (path: legacy coqui's `-aflplusplus` derivation).
2. Resolve libxml2 source from the CPU build's `-source` closure.
3. Generate `config.h`, `xmlversion.h`, and patch `error.c`/`xmlstring.c`
   inline, exactly mirroring the legacy coqui build phase.
4. coqui-cc clang → llvm-link → opt (coqui pass plugin) succeeds. ASan pass
   reports **211,117 outlined fast-path call sites** (114,450 loads + 96,667
   stores).

### Current blocker

llc + ptxas at extreme scale:

- **Default llc -O2**: stalls indefinitely in register allocation on the
  211k-call-site module (killed after >20 min in all observed attempts).
- **`COQUI_LLC_OPT=-O1` workaround**: llc completes in ~2 min and emits PTX.
  But ptxas then rejects the PTX with:

```
ptxas .libxml2_xml_read_fuzzer.build/out.ptx, line 3388564;
error   : Type of argument does not match formal parameter 'func_retval0'
```

This is an NVPTX codegen / backend ABI mismatch specific to this target's
scale (10× larger than the next target, cmark at 23,567 call sites). The
9 other evaluation targets build cleanly with the current pass pipeline
(some also needing `COQUI_LLC_OPT=-O1`, e.g. libpng at 44k call sites did
briefly hit the same stall before the retry-at-30-min-cap succeeded).

### What to do next

1. **Investigation in progress** — identify which function/call site
   triggers the type mismatch and whether it's in our ASan helpers or
   upstream llc -O1 NVPTX codegen.
2. **Possible mitigation**: reduce the instrumented-access count by
   skipping specific categories of accesses in `Asan.cpp` (e.g. functions
   in a libxml2-specific blocklist). Risky — could mask real bugs.
3. **Upstream**: if root-caused to LLVM NVPTX backend, file a minimal
   reproducer against llvm-project.

Other 9 evaluation targets (bzip2, cares, cjson, cmark, libjpeg-turbo,
libpng, libyaml, stb_image, zstd) build to valid cubins with the current
pipeline. libxml2 is the only outlier; treat the build as BLOCKED until
the ptxas issue is resolved.

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
  `--batch-size 65536`; coqui mode's coqui-cc accepts neither. Heap is
  derived at runtime from the per-thread stack budget; batch size is
  currently hardcoded to 8192 in the runtime. Handoff priority 2 tracks
  adding `AFL_COQUI_HEAP_SIZE` / `AFL_COQUI_BATCH_SIZE` env-var overrides
  in `src/afl-fuzz-coqui.c`. Until then, libxml2's CPU-nix-spec values
  are intentionally dropped and the runtime defaults are used.
- **Source list**: mirrors the 22-file nix list exactly (libxml2 core +
  harness + stubs). `HTMLparser.c` / `HTMLtree.c` are excluded because
  `WITH_HTML=0`.
