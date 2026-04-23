# coqui mode ↔ upstream coqui deltas

Notes from auditing `~/coqui/` against this tree to identify why targets
that build cleanly in coqui fail or run slow in coqui mode.

## 1. Inlined vs. outlined ASan (root cause of ptxas-O1 pathology)

**Symptom:** ptxas -O1 on ASan-instrumented PTX takes 30–60+ min with
50-100+ GB RSS on larger targets (cmark 55 min/53 GB, cares 77 min/13 GB,
libpng / stb_image similar). Constraint forbids -O0.

**Root cause:** coqui mode's `coqui_mode/passes/Asan.cpp` (283 lines)
instruments each load/store with an **inline** check sequence —
split basic block, shadow read, slow-path branch. Per-access cost is
~3× basic blocks and ~20 PTX instructions. On cmark this generated
12k loads × 20 insns = 240k insns of extra PTX.

Coqui's `~/coqui/src/AsanTransform.cpp` (936 lines — 3× bigger than ours)
explicitly fixed this by **outlining** the fast path into size-specialized
helpers (`__coqui_asan_check_fast_load_{1,2,4,8,gen}` and matching stores).
Each load/store site becomes a **single call instruction**. Comment in
the source spells it out:

> This avoids the 3x basic block bloat (split + shadow + slow) that the
> previous inline approach created per access, which caused catastrophic
> icache thrashing on large modules (e.g., cmark with 5.9 MB .text, 16x
> slowdown).

Branch-weight metadata is attached to guide `llc -O2` layout.

**Port cost:** ~650 lines of C++ pass code + 4–6 helper runtime functions
in `coqui_fuzz_asan.c`. Largest single port remaining. Blocks build of
cmark, cares, libpng, libxml2, libyaml, stb_image, zstd — i.e. 7 of 10
targets.

## 2. Missing runtime overrides for heap-size / batch-size

**Symptom:** bzip2 `--coqui` fails with `CUDA_ERROR_LAUNCH_FAILED` on
first batch, even with 64 KiB stack.

**Root cause:** coqui targets tune per-target sizes heavily:

| target | stack | heap | batch | slab |
|---|---:|---:|---:|---:|
| bzip2 | default | **524288** | **32768** | **2 GiB** |
| libpng | **32768** | **131072** | **16384** | **2 GiB** |
| libxml2 | **32768** | **131072** | **65536** | **2 GiB** |
| stb_image | **65536** | default | default | default |
| cjson | default | default | default | default |

coqui mode's `coqui-cc` has `--stack-size` (with `AFL_COQUI_STACK_SIZE` runtime
env override) and `--slab-pool-size`, but NO runtime env override for
heap-size or batch-size. The runtime derives heap as
`(total_budget - cov - stack) * 8/9` (so `512 - 64 - 64 = 384 KB`
available; heap gets 341 KB of that) — **but that's smaller than bzip2
needs (512 KB)**. Batch size is hardcoded to 8192.

**Fix (runtime-first, matching user preference for iterative tuning):**
add `AFL_COQUI_HEAP_SIZE` and `AFL_COQUI_BATCH_SIZE` runtime env vars in
`src/afl-fuzz-coqui.c` following the existing `AFL_COQUI_STACK_SIZE`
pattern. No coqui-cc changes needed — the values don't need to be baked
into the cubin; AFL applies them at `coqui_init()`. The slab pool
allocator code exists in the runtime already — routing the allocator
overflow to it is partly ported.

## 3. libjpeg-turbo explicit source list + stubs file

**Symptom:** build fails with `'jdcolext.c' file not found` when
harnessing up the jpeg sources. My glob `jpeg_include/*.c` matches
template `.c` files (`jdcolext.c`, `jdcol565.c`, `jstdhuff.c`) that are
meant to be `#include`d by other `.c` files, not compiled standalone.

**Resolution (in place):** the correct 19-file explicit list from
`~/coqui/nix/cpu-target-specs.nix` is:

```
jdapimin.c jdapistd.c jdatasrc.c jdcoefct.c jdcolor.c
jddctmgr.c jdhuff.c jdinput.c jdmainct.c jdmarker.c
jdmaster.c jdpostct.c jdsample.c jidctint.c jmemmgr.c
jmemnobs.c jerror.c jutils.c jcomapi.c
```

Plus `~/coqui/harness/targets/libjpeg_turbo_stubs.c` for the
`jsimd_can_*` / `jsimd_h2v2_*` symbols the library references for SIMD
acceleration on CPU (stubs always return 0 — force scalar path).

With these, the CPU binary builds. Already in
`libjpeg_turbo_decompress_fuzzer_cpu_p`.

## 4. Per-target overrides coqui bakes into targets

See `~/coqui/nix/targets/*.nix` for each target's exact flags. Missing
from our eval dirs (build.sh files don't pass them because coqui-cc
rejects them):

- bzip2: `--heap-size 524288 --batch-size 32768 --slab-pool-size 2147483648`
- libpng: `--heap-size 262144 --batch-size 16384 --stack-size 32768 --slab-pool-size 2147483648 --max-batch-time 9`
- libxml2: `--heap-size 131072 --batch-size 65536 --stack-size 32768 --slab-pool-size 2147483648`
- cares: `--slab-pool-size 10737418240` (10 GiB)

The `--max-batch-time` knob is for a libpng-specific 9-second batch
timeout (pathological-input cull). coqui mode has `AFL_COQUI_TIMEOUT_US`
env-var for similar.

## 5. --coqui + persistent-mode (`__AFL_LOOP`) deadlock

**Symptom:** when a CPU binary has `__AFL_LOOP` (shm-fuzz or stdin-fed),
AFL's dry-run calibration times out. Plain AFL CPU runs the same binary
fine; only `--coqui` path deadlocks.

Did not trace this to a coqui-side fix yet. coqui's driver runs AFL as
a separate program, so their AFL integration may not exercise the same
code path. Worth checking whether coqui uses `-S` (classical secondary)
vs. a dedicated `--coqui` flag equivalent — and if the CPU-forkserver
calibration in `calibrate_case()` ever calls into `coqui_submit_input`
instead of `fuzz_run_target` under our flag.

## Priority for next pass

1. **Port outlined ASan** — biggest impact (unblocks 7/10 targets).
2. **Add `AFL_COQUI_HEAP_SIZE` / `AFL_COQUI_BATCH_SIZE` runtime env vars** — matches user preference for iterative tuning; fixes bzip2; likely fixes `libpng` et al. once they build.
3. **Fix persistent-mode under --coqui** — required to get real CPU-main throughput (currently fork-per-exec = 1.5k/s vs. an expected ~100k/s).
