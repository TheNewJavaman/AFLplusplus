# coqui mode cjson evaluation target

A self-contained fuzz setup that builds the cJSON library + oss-fuzz-style
harness for coqui mode (`afl-fuzz --coqui`) and the matching AFL++-instrumented
CPU binary for host verification.

Mirrors `/home/gpizarro/coqui/nix/targets/cjson.nix` exactly: same cJSON
v1.7.18 sources, same `-I`, same `-D CJSON_HIDE_SYMBOLS`, same
`--stack-size 32768`, same harness (`harness/targets/cjson_read_fuzzer.c`
in the coqui repo).

## Files produced by `build.sh`

| Path                          | What it is                                              |
| ----------------------------- | ------------------------------------------------------- |
| `cjson_fuzzer.cubin`          | GPU kernel (coqui-cc output; sm_75, 32 KiB stack)       |
| `cjson_fuzzer.conf`           | Sidecar written by coqui-cc (stack/slab/arch)           |
| `cjson_fuzzer_cpu`            | Symlink to the AFL++-instrumented `cjson_read_fuzzer`   |
| `seeds/`                      | Symlink into the nix store (upstream cJSON corpus)      |
| `dict/`                       | Symlink into the nix store (upstream `json.dict`)       |

The CPU binary and seed/dict dirs come from:

```
nix build '.#target-cjson-aflplusplus' --out-link /tmp/coqui-cjson-cpu
```

`build.sh` runs that command for you and symlinks the artifacts into the
local dir. The symlinks (and `out/`) are in `.gitignore`.

## Quickstart

```
cd /home/gpizarro/cuAFL/coqui_mode/evaluation/cjson
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
- nix (2.34+), with `/home/gpizarro/coqui` checkout providing the flake
- NVIDIA driver + CUDA on the host (for `libcuda.so.1`)
- GPU index 0 is an RTX Titan (sm_75) per `CLAUDE.local.md`

## Build status

`build.sh` ran cleanly end-to-end on 2026-04-22 (RTX Titan / sm_75). The
nix `target-cjson-aflplusplus` closure built from scratch (cached after
the first run), `coqui-cc` emitted a 1.08 MiB cubin + 3-line conf, and
the resulting `seeds/` directory contains 14 upstream fuzz inputs with
`dict/json.dict` also symlinked in. The only build output is a harmless
`clang: warning: argument unused during compilation: '-fno-stack-protector'`
(from coqui-cc's NVPTX pipeline — expected) and two benign `ptxas`
warnings about `__coqui_virgin_map` / `__coqui_global_statics_pool_base`
being resolved at load time by the coqui mode driver — also expected.

## Gotchas

- **Don't `git add` the symlinks.** `cjson_fuzzer_cpu`, `seeds/`, `dict/`
  all point at `/tmp/coqui-cjson-cpu/…` or the nix store. They are in
  `.gitignore`.
- **GPU device pinning.** Per user memory, everything here assumes device
  index 0 (RTX Titan, sm_75). Override with `AFL_COQUI_DEVICE=N`.
- **`coqui-cc` flag surface is narrower than the coqui driver.** It does
  not accept `--heap-size` or `--batch-size`; `build.sh` only passes
  `-arch`, `--stack-size`, `--slab-pool-size`, `-I`, `-D`, and source
  files.
- **cJSON source location.** `build.sh` finds `/nix/store/<hash>-source/`
  by scanning the closure of the CPU nix result for a path containing
  `cJSON.c`. Do not copy the source out of the store — the path is
  read-only and re-derives cleanly.
- **Harness path.** The libFuzzer-style harness lives at
  `/home/gpizarro/coqui/harness/targets/cjson_read_fuzzer.c`. `build.sh`
  references it via an absolute path; it is not copied into this dir.
- **Orphan-safe.** `afl-fuzz` installs `PR_SET_PDEATHSIG` on the forkserver
  (fixed in `aadd355b`), so killing this shell cleanly tears down the
  fuzzer. Closing a tmux pane does NOT reach the fuzzer — use
  `pkill -u "$USER" -KILL -f "afl-fuzz --coqui"` before relaunching and
  check `nvidia-smi` shows memory released.
