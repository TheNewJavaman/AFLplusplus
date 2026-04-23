# cuAFL cmark evaluation target

Adapts the cmark (CommonMark Markdown parser) fuzz target from coqui so it can
be driven with cuAFL's `afl-fuzz --coqui`.

- Upstream: https://github.com/commonmark/cmark (pinned to `0.31.1`)
- GPU ref spec: `/home/gpizarro/coqui/nix/targets/cmark.nix`
- AFL++ CPU spec: `target-cmark-aflplusplus` in `/home/gpizarro/coqui/flake.nix`
- Harness: `/home/gpizarro/coqui/harness/targets/cmark_fuzzer.c` (libFuzzer-style,
  4-byte `options` + 4-byte `width` prefix; mode selected by upper 2 bits of
  `options`)

## Build / run

```
./build.sh   # compiles .cubin + resolves CPU binary + seeds + dict
./fuzz.sh    # launches: afl-fuzz --coqui gpu0 -i seeds -o out -- cmark_fuzzer_cpu
```

`fuzz.sh` forwards `$@` to `afl-fuzz` (insert before the `--` separator if you
need extra flags like `-x dict/markdown.dict`).

## Build status (as of 2026-04-22)

**`build.sh` currently FAILS** at the coqui-cc coqui-link pipeline:

```
LLVM ERROR: [coqui-cc] ExternalSymbolGatekeeper: unresolved external
'snprintf' — port the replacement transform or runtime stub.
Used by: cmark_render_html
```

### Why

- `cmark`'s renderers (`html.c`, `commonmark.c`, `xml.c`, `man.c`, `latex.c`)
  all call `snprintf` for integer-to-string conversion.
- The cuAFL `coqui-cc` device runtime (`/usr/local/lib/coqui-cc/runtime.bc`)
  exposes `__coqui_malloc`, `__coqui_strlen`, `__coqui_memcmp`, ... but
  **no `printf`/`snprintf` family**; the `ExternalSymbolGatekeeper` pass refuses
  any build that leaves these unresolved.
- The upstream coqui flake solves this by linking `coqui-musl-bitcode` (a
  port of musl libc to NVPTX); cuAFL's compressed `coqui-cc` distribution does
  not ship a libc bitcode equivalent.
- The harness itself uses `cmark_markdown_to_html` in mode 3, which pulls in
  `cmark_render_html` regardless of whether `html.c` is passed explicitly;
  internalize+DCE won't drop it because `LLVMFuzzerTestOneInput` is live-root.

### What works

- CPU half (AFL++-instrumented `cmark_fuzzer_cpu`) builds fine via
  `nix build .#target-cmark-aflplusplus` and the symlink/wire-up works.
- Seeds (`seeds/`) and dictionary (`dict/markdown.dict`) are present from
  the nix store.
- `build.sh` correctly resolves the pinned cmark source tarball via
  `fetchFromGitHub` (identical hash → identical `/nix/store` path each run),
  generates the local `generated/{config,cmark_export,cmark_version}.h`
  headers that the nix `preBuild` produces, and invokes `coqui-cc` with
  the same sources / `-I` / `-D` / `--stack-size 32768` as the coqui nix spec.
- All cmark `.c` sources compile to NVPTX bitcode cleanly; the failure is
  strictly at the final coqui-link gatekeeper step, not at the earlier
  per-TU clang stage.

### Paths forward

1. **Add an `snprintf`-family port to cuAFL's device runtime** (likely by
   pulling the relevant translation units from `coqui-musl-bitcode` into
   `coqui_mode/runtime/`, or implementing a minimal integer-only `snprintf`
   under `__coqui_snprintf` plus a `Libc.cpp`-pass rewrite to route
   `snprintf` → `__coqui_snprintf`). This is the intrusive-but-correct fix.
2. **Harness surgery** — patch `cmark_fuzzer.c` to remove mode 3
   (`cmark_markdown_to_html`) and drop the renderer TUs. The harness already
   documents that render functions were meant to be excluded on GPU; current
   code path is inconsistent with its own comment. Still needs a gatekeeper-
   satisfying story for any residual snprintf calls (e.g. from
   `references.c` / `scanners.c`).
3. **Add `snprintf` to the `ExternalSymbolGatekeeper` allowlist and provide
   a device-side stub that trips an unreachable** — works for targets that
   don't actually execute the renderer path; bad idea for cmark because
   mode 3 exercises it on every seed matching the high-bit pattern.

## Layout

- `build.sh` / `fuzz.sh` — entry points (set -euo pipefail).
- `.gitignore` — excludes build artifacts.
- `generated/` — produced by `build.sh`; config headers the cmark source
  expects at build time.
- After a successful build the following appear (symlinks into `/nix/store`
  where applicable):
  - `cmark_fuzzer.cubin`, `cmark_fuzzer.conf` (from coqui-cc)
  - `cmark_fuzzer_cpu` -> `/tmp/cuafl-cmark-cpu/cmark_fuzzer`
  - `seeds/` -> `/tmp/cuafl-cmark-cpu/seeds`
  - `dict/` -> `/tmp/cuafl-cmark-cpu/dict`
  - `out/` — created by `fuzz.sh`; AFL output dir

## Gotchas

- Requires a working coqui flake at `/home/gpizarro/coqui` (for the AFL++
  CPU build) and a local install of cuAFL's `coqui-cc` at
  `/usr/local/bin/coqui-cc` (for the GPU build).
- `--arch sm_75` is hardcoded in `build.sh` (matches the RTX Titan test box
  per `CLAUDE.local.md`).
- The first run of `build.sh` is slow (nix fetches cmark upstream + builds
  the AFL++ target); subsequent runs hit the Nix cache.
- `fuzz.sh` sets `AFL_I_DONT_CARE_ABOUT_MISSING_CRASHES=1`,
  `AFL_SKIP_CPUFREQ=1`, `AFL_SKIP_BIN_CHECK=1`, `AFL_NO_UI=1` per the
  project convention; override by exporting them yourself before calling
  the script.
