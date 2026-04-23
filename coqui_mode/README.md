# coqui_mode — GPU executor for coqui mode

`coqui_mode` is coqui mode's GPU-backed executor. Targets are compiled to
NVIDIA cubins via `coqui-cc`, and AFL++ drives them via the `--coqui`
CLI flag.

## Components

- `runtime/` — device-side C files compiled to `runtime.bc` at install
  time, linked into every cubin.
- `passes/` — LLVM pass plugin that runs post-link to instrument the
  target for NVPTX + coverage + heap-only ASan.
- `bin/coqui-cc` — Python driver that orchestrates clang, opt, llc, ptxas.

## Install

```bash
cd coqui_mode
./build_coqui_support.sh
```

Requires LLVM 18 (matching clang version), CUDA toolkit 13+, and Python 3.

## Usage

```bash
# Compile target
coqui-cc -arch sm_75 harness.c lib/*.c -o target.cubin

# Run under coqui mode
afl-clang-fast harness.c lib/*.c -o target
afl-fuzz --coqui gpu0 -i seeds/ -o out/ -- ./target
```

See `docs/superpowers/specs/2026-04-18-coqui-internals-design.md` for
design details.
