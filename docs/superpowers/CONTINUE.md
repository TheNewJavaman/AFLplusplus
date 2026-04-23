# coqui mode — Resume Prompt

Drop the prompt below into a new Claude Code session, run from this repo on branch `coqui-dev`. Working tree should be clean.

---

## Prompt

> I'm continuing work on **coqui mode** — a fork of AFL++ that adds a GPU-backed
> executor mode (`coqui_mode`) selected via `--coqui <sync_id>`. This is on
> branch `coqui-dev` of this repo. End-to-end pipeline works:
> `coqui-cc` compiles a target to a cubin, `afl-fuzz --coqui gpu0 -- target.elf`
> launches batched GPU fuzzing alongside a standard CPU forkserver, and per-batch
> flagged inputs go through the CPU forkserver to populate `trace_bits` before
> `save_if_interesting`. Cjson fuzzes successfully for **~9 minutes / 170K execs
> / 350+ corpus entries / 22.5% coverage** before hitting an intermittent
> `CUDA_ERROR_ILLEGAL_ADDRESS` from `cuStreamQuery` at
> `src/afl-fuzz-coqui.c:332`.
>
> ### Read these to get oriented
> - **Spec (coqui mode integration):** `docs/superpowers/specs/2026-04-18-coqui-gpu-backend-design.md`
> - **Spec (coqui internals):** `docs/superpowers/specs/2026-04-18-coqui-internals-design.md`
> - **Plan (coqui mode):** `docs/superpowers/plans/2026-04-18-coqui-gpu-backend.md`
> - **Plan (coqui internals):** `docs/superpowers/plans/2026-04-18-coqui-internals.md`
> - **Port log:** `docs/coqui_port_log.md` (running list of port-on-demand decisions)
>
> Commit history is the source of truth — `git log --oneline coqui-dev` shows ~40
> commits, every one with a clear `coqui:` prefix (or the legacy `cuAFL:` prefix on
> older commits before the rename sweep).
>
> ### Current debugging task
>
> The crash signature: cjson runs cleanly for ~9 minutes, hits **19 batch
> timeouts** (slow inputs, kernel hangs, our salvage path recovers each time),
> then on the 20th failure `cuStreamQuery` returns `ILLEGAL_ADDRESS` *directly*
> (not via the timeout path). That means the kernel actually faulted on-device,
> not just hung.
>
> **Two hypotheses:**
> 1. Accumulated context degradation across 19 successful timeout-recoveries
>    eventually corrupts something device-side.
> 2. A specific input pattern triggers a real OOB in our instrumented code
>    (e.g., heap freelist corruption causing later reads to land outside the
>    heap region). Took ~9 min of fuzzing to find it.
>
> **Reproduction setup:** `/tmp/cjson_probe/` has the harness, sources, both
> binaries (cubin + ELF), seeds, and the run scripts. To reproduce:
> ```bash
> /tmp/cjson_probe/run_fuzz_logged.sh   # ~9 min, then aborts
> ```
> Logs go to `/tmp/cjson_probe/fuzz.log`. Compute-sanitizer variant:
> ```bash
> /tmp/cjson_probe/run_sanitize.sh      # 10× slower; needs ~90 min for crash
> ```
>
> ### What to do first
>
> Pick one of these depending on appetite:
>
> 1. **Long sanitizer run (most informative)** — start `run_sanitize.sh` in a
>    detached tmux session, leave it for an hour, check the log for the
>    crash's device-frame stack trace. Should pinpoint the OOB instruction.
>
> 2. **Bisect via pass-disabling** — comment out `coqui::runAsan(M)` in
>    `coqui_mode/passes/CoquiPassPlugin.cpp` `CoquiLinkPass::run()`,
>    rebuild plugin + cubin, retry. If crash goes away, the bug is in our
>    Asan instrumentation; if not, suspect Coverage. ~9 min per bisect.
>
> 3. **Pre-emptively add the in-kernel watchdog** — coqui's solution for
>    GPU kernels hanging on bad input. Modify the `Coverage` pass at
>    `coqui_mode/passes/Coverage.cpp` to insert a per-thread iteration
>    counter at every Nth basic block; once it exceeds a threshold,
>    set `status[tid].timeout_flag = 1` and `__coqui_exit()`. This solves
>    the hang-on-slow-input issue (the 19 timeout-recoveries) and may
>    incidentally fix the cumulative-corruption hypothesis if (1) is
>    correct.
>
> ### Architecture summary (so you don't re-read everything)
>
> - **`afl->fsrv`** is a real CPU forkserver running the ELF target. Coexists
>   with the GPU executor at `afl->coqui`. AFL's standard machinery
>   (calibrate/trim/sync) runs through this normally.
> - **`common_fuzz_stuff` has a `gpu_mode` branch** that routes to
>   `coqui_submit_input` (batch sink) instead of running the forkserver.
>   `fuzz_run_target` runs normally and gives real `trace_bits`.
> - **Per-batch flow:** havoc loop fills 8K-input batch → kernel launches async
>   → next batch starts filling on the other ping-pong half → batch-1 results
>   come back → for each flagged input, run through the CPU fsrv to get real
>   coverage, call `save_if_interesting` normally.
> - **GPU side (in cubin):** runtime + 11 LLVM passes + StaticGlobals pool +
>   tid-indexed slot pool for region pointers. Coverage = AFL hash with
>   bucketing, virgin map = device-side global, novelty bitmap = the only
>   thing DMA'd back to host alongside per-thread status.
> - **`coqui_init`** binary-searches `cuCtxSetLimit` to find the device's max
>   per-thread stack budget (typically 256 KB on this RTX Titan). Allocates
>   ping-pong buffers, statics pool, binds `__coqui_global_statics_pool_base`
>   symbol.
>
> ### Recent fixes (this session)
>
> Search `git log --oneline | head -10` to see the most recent commits. Key
> recent ones include:
> - `runtime uses clang NVPTX builtins; exempt __coqui_* from InlineAsmReject`
>   (replaced PTX inline asm with `__nvvm_*` builtins + `__builtin_trap`)
> - `port-on-demand — minimum Libc replacement set` (strlen, strncmp, strtod
>   for cjson)
> - `per-thread state via tid-indexed .global pools` (fixed PTX rejecting
>   addrspace(5) module globals)
> - `coqui_shutdown real CUDA teardown + deprecate calibrate_one`
>
> Uncommitted in-flight changes (apply or revert as appropriate):
> - `src/afl-fuzz-coqui.c` — added device-stack-budget probe (binary-search via
>   `cuCtxSetLimit`), added diagnostic check on `cuCtxSynchronize` return
>   inside the timeout-recovery branch, removed defense-in-depth assertion
>   from `afl_fsrv_run_target` in `src/afl-forkserver.c`, added
>   `__coqui_malloc_raw`/`__coqui_free_raw` to `coqui_mode/runtime/coqui_memory.c`
>   for ASan-bootstrap, added asan-internal redirect logic to
>   `coqui_mode/passes/Asan.cpp`, lowered `kTotalBudget` in
>   `coqui_mode/passes/MemoryLayout.cpp` from 524288 to 262144 (256 KB)
>   to fit common driver budgets.
>
> Run `git diff` to see what's pending; commit if it looks right.
>
> ### Constraints to respect
>
> - GPU device 0 is RTX Titan sm_75 — always run there.
> - Never kill processes owned by other users.
> - User prefers atomic commits (one logical change per commit).
> - Port-on-demand decisions are **collaborative** — propose, get approval,
>   then execute. Don't speculatively port multiple coqui transforms in one
>   shot.
> - User said "no tests" earlier this session — manual smoke-testing only.

---

## Quick start commands

```bash
cd "$(git rev-parse --show-toplevel)"     # cd to repo root
git status                                # check working tree
git log --oneline -15                     # recent work

# Reproduce the crash
/tmp/cjson_probe/run_fuzz_logged.sh       # detach with Ctrl+B D if in tmux

# When it dies (~9 min):
grep -B3 -A8 "PROGRAM ABORT" /tmp/cjson_probe/fuzz.log
grep -c "coqui batch timeout" /tmp/cjson_probe/fuzz.log
cat /tmp/cjson_probe/out/gpu0/fuzzer_stats | grep -E "execs_done|run_time|corpus_count|bitmap_cvg"

# Sanitizer variant (slow but locates the device frame)
/tmp/cjson_probe/run_sanitize.sh          # ~90 min; output in stderr
```
