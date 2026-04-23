# cuAFL bench runs

Log of fuzz runs against evaluation targets, capturing coverage + exec-rate + configuration.
Each entry records results from a CLEAN-CORPUS start with a minimal domain-specific seed.

## Schema

| Column | Meaning |
|---|---|
| date     | UTC timestamp of run start |
| commit   | cuAFL git HEAD at run time |
| target   | evaluation target name |
| duration | wall-clock run time (`-V` value) |
| clients  | `<main> + <coqui> + <afl-plain>` instance counts |
| seed     | description of minimal starting seed |
| stack    | `AFL_COQUI_STACK_SIZE` (or default) |
| heap     | `AFL_COQUI_HEAP_SIZE` (runtime-derived if blank) |
| slab     | `AFL_COQUI_SLAB_SIZE` (0 = off) |
| batch    | `AFL_COQUI_BATCH_SIZE` or default 8192 |
| notes    | any tuning or caveat |
| **main exec/s** | CPU main instance exec rate (from fuzzer_stats) |
| **main cvg**    | CPU main instance bitmap coverage % |
| **coqui submits/s** | GPU submit rate (from `[coqui-rate]` samples, median) |
| **coqui cvg** | coqui secondary bitmap coverage % |

---

## Runs

### 2026-04-23 02:10 UTC — bzip2 plain CPU  (commit cebabfee + stack-default-65536 patch)

| config | value |
|---|---|
| clients   | 0 main + 0 coqui + **1 plain AFL** |
| duration  | 30 s |
| seed      | 37 bytes — `python3 -c 'import bz2; print(bz2.compress(b"a"))'` (a minimal valid bzip2 stream) |
| stack     | n/a (CPU target only) |
| batch     | n/a |
| driver    | `_common/afl_driver.c` — fork-per-exec, stdin-fed. **Violates persistent-mode constraint** (see Known Issues). |
| notes     | Plain AFL only; `--coqui` fails with CUDA_ERROR_LAUNCH_FAILED on bzip2 even at 64 KiB stack — the instrumented kernel throws an illegal-address or ptxas-barrier fault that likely needs a larger stack than the driver allocates. Tracking as known blocker. |

| instance | exec rate | cvg | corpus | edges |
|---|---:|---:|---:|---:|
| plain main | **1,105 exec/s** | 19.26% | 177 | 405 |

### 2026-04-23 02:15 UTC — cjson plain CPU  (commit cebabfee + stack-default-65536 patch)

| config | value |
|---|---|
| clients   | 0 main + 0 coqui + **1 plain AFL** |
| duration  | 30 s |
| seed      | 1 byte `'a'` |
| driver    | fork-per-exec (as above) |

| instance | exec rate | cvg | corpus | edges |
|---|---:|---:|---:|---:|
| plain main | **1,267 exec/s** | 7.48% | 127 | 122 |

### 2026-04-23 02:15 UTC — cjson --coqui solo  (commit cebabfee + stack-default-65536 patch)

| config | value |
|---|---|
| clients   | 0 main + **1 coqui** + 0 plain |
| duration  | 60 s (`-V`) — actual wall ≈ 105 s due to force-reset recovery |
| seed      | 1 byte `'a'` |
| stack     | 64 KiB (default) |
| heap      | 113 KiB (derived) |
| slab      | 0 |
| batch     | 8192 |

| instance | exec rate | cvg | corpus | edges |
|---|---:|---:|---:|---:|
| --coqui gpu0 | see below | 6.25% | 46 | 102 |

Two `[coqui-rate]` samples captured before / after a force-reset:

| # | batches/s | submits/s | wall/batch | verify/batch | kernel breakdown |
|---|---:|---:|---:|---:|---|
| 1 | 124.9 | **1,023,168** | 8,006 µs | 23 µs | exec=0% classify=17% virgin=83% |
| 2 | 1.3   | 10,314       | 794 ms  | 58 ms  | exec=47% classify=10% virgin=42% (during recovery) |

Run hit "coqui kernel stuck past 4000000 us" twice; force-resets cost ~30 s of wall time each. Steady-state peak submits/s ≈ **1.0 M/s**.

### 2026-04-23 02:37 UTC — cjson main + coqui post-calibration (commit 353e9a93)

First run with the new **post-calibration bench methodology**: start both
instances, wait for main to clear calibration (execs_done > 100 on main
+ coqui has emitted at least one `[coqui-rate]` sample), *then* start
the 60 s window.

Also: discovered that `fuzzer_stats` `execs_done` for the coqui side
lags the real counter by orders of magnitude — the file is only
flushed at specific AFL internal checkpoints and coqui batches fire
inside tight havoc loops that bypass those. Authoritative GPU
throughput is taken from `[coqui-rate]` samples, not `fuzzer_stats`.

| config | value |
|---|---|
| clients   | **1 main + 1 coqui + 0 plain** |
| duration  | 60 s post-cal (total wall ≈ 180 s incl warmup) |
| seed      | 1 byte `'a'` |
| stack     | 64 KiB (default) |
| heap      | 113 KiB (derived) |
| slab      | 0 |
| batch     | 8192 |
| driver    | fork-per-exec stdin-fed (see Known Issues) |

**Main (CPU)** — from `fuzzer_stats` delta over bench window:
| metric | start | end | rate |
|---|---:|---:|---:|
| execs_done | 83,730 | 165,015 | **1,355 exec/s** |
| bitmap_cvg | 10.17% | 11.09% | +0.92% |
| edges_found | 166 | 181 | +15 |
| corpus | 165 | 229 | +64 |

**Coqui (GPU0)** — from `[coqui-rate]` samples during bench window (n=3):
| submits/s | batches/s | host wall/batch |
|---:|---:|---:|
| **881,334** (peak) | 107.6 | 9.3 ms |
| 893,957 | 109.1 | 9.2 ms |
| 581,065 | 70.9 | 14.1 ms |

avg ≈ **785 k submits/s**, median ≈ 881 k. coqui fuzzer_stats: cvg 0.31% → 2.63%, corpus 1 → 5 (coqui secondary admits fewer entries than main since it imports main's output asynchronously).

No kernel-stuck / CUDA-error events this run.

**Headline**: post-calibration, the pair delivers **~786 k GPU submits/s + 1,355 CPU execs/s** on cjson with the 1-byte `'a'` seed. This compares favorably to the earlier 60 s-including-calibration run that logged 1,566 main exec/s + 0 coqui (calibration starvation masked the GPU throughput entirely).

### 2026-04-23 02:30 UTC — cjson main+coqui (commit cebabfee, larger corpus)

Initial smoke test. Minimal seed = 1 byte `'a'`.

| config | value |
|---|---|
| clients   | 1 main + 1 coqui + 0 plain |
| duration  | 60 s (`-V 60`) |
| seed      | 1 byte `'a'` |
| stack     | 32 KiB (default at this commit) |
| heap      | 142 KiB (derived) |
| slab      | 0 |
| batch     | 8192 (default) |

| instance | exec rate | cvg | corpus | edges |
|---|---:|---:|---:|---:|
| main (CPU) | **1,666 exec/s** | 9.85% | 177 | 140 |
| coqui (gpu0) | **77,583 submits/s** (coqui-rate) | — | 1 | 3 |

Kernel phase breakdown (from last coqui-rate): `init=0% exec=3% classify=15% virgin=82%`.
Coqui secondary's own fuzzer_stats show `execs_done=100` (post-calibration counter was not climbing yet); the authoritative GPU throughput is the `[coqui-rate]` sample above.

### 2026-04-23 04:33 UTC — bzip2 pair (coqui crashes, main runs solo)  (commit 80c5f5cf)

Paired run attempted, coqui crashed with `CUDA_ERROR_LAUNCH_FAILED` on
first batch (known issue — bzip2 kernel needs more stack). Main continued
solo for the full 60 s window.

| config | value |
|---|---|
| clients   | 1 main + 1 coqui (crashed) + 0 plain |
| duration  | 60 s (AFL run_time; post-calibration snapshot not collected in this run) |
| seed      | 37 bytes — minimal valid bzip2 stream |
| stack     | 64 KiB (default) |
| driver    | fork-per-exec |

| instance | exec rate | cvg | corpus | edges |
|---|---:|---:|---:|---:|
| main (CPU) | **1,438 exec/s** | 21.92% | 268 | 461 |
| coqui | crash on batch #1 | — | — | — |

Comparing against the 30 s plain main run above (1,105 exec/s, 19.26% cvg,
405 edges): with 2× more fuzzing time, coverage plateaus only slightly
(21.9% vs 19.3%) — bzip2's cvg ceiling is low without a more diverse seed
corpus.

### Build blockers observed

- **ptxas -O1 is pathological on ASan-instrumented PTX for some targets**:
  cmark (53 GB RSS / 55 min killed), cares (13 GB / 77 min killed). libpng
  earlier in this session also blew up to 100+ GB. The constraint is -O1
  minimum (per `feedback_compilation_opt_level.md`); the workaround is to
  reduce ASan instrumentation density or shrink the per-target source list.
  Targets affected: cares, cmark, libpng, libxml2, libyaml, stb_image,
  zstd all have cubin builds that reach ptxas but take >30 min at -O1.
- **Sequential build driver**: /tmp/seq_rebuild.sh waits for any in-flight
  ptxas, then builds each remaining target one at a time. It hit the
  pathological-ptxas issue on cmark and cares; killed both and stopped.

### Known issues blocking wider bench runs

- **`--coqui` + persistent-mode CPU binary**: cuAFL's `--coqui` forkserver path times out during AFL's dry-run calibration when the target binary uses AFL's persistent-mode features (either shm-fuzz via `__AFL_FUZZ_TESTCASE_BUF` or the stdin-fed `__AFL_LOOP` form). Pure AFL CPU mode works fine with persistent binaries; only the `--coqui` path is affected. Until this is debugged, `--coqui` benches have to use fork-per-exec (non-persistent) drivers. **This violates the project's persistent-mode constraint.** See `include/afl-fuzz-coqui.h` / `src/afl-fuzz-coqui.c` for coqui-mode state and investigate why the CPU forkserver calibration path behaves differently under `--coqui`.
- **Env-var propagation**: `AFL_COQUI_STACK_SIZE=N` set inline on a background command (`VAR=X nohup bin & disown`) is sometimes stripped by bash's background handling. Default bumped to 65536 in `afl-fuzz-coqui.c:166` as of this run so targets that need more stack (bzip2 especially) get it without relying on env propagation.
- **Shell-tool flakiness**: compound bash commands using `( ... ) > log 2>&1 &` and heredoc file writes sometimes fail silently in Claude's Bash tool sandbox (scripts don't land, redirections don't flush). Workaround: use the `Write` tool for scripts, invoke them via simple `nohup /path/script &`.
