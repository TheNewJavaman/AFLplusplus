# Coqui port-on-demand log

## 2026-04-18 — Phase 0, cjson

### Port: __COQUI_DEVICE__ macro default in coqui-cc
- Trigger: ExternalSymbolGatekeeper on `sprintf` from cJSON's `print_value`
- Fix: coqui-cc adds `-D__COQUI_DEVICE__=1` to clang invocations by default,
  matching coqui's behavior. The harness's `#ifndef __COQUI_DEVICE__` guard
  excludes print paths from compilation.

### Port: internalize + globaldce intermediate pass
- Trigger: cJSON's externally-visible cJSON_Print* functions kept print_value
  reachable even after harness #ifdef'd out the calls. Linker DCE didn't
  fire because cJSON_Print* have external linkage.
- Fix: coqui-cc runs `opt -passes=internalize,globaldce
  -internalize-public-api-list=LLVMFuzzerTestOneInput` between llvm-link
  and coqui-link. Strips externally-visible-but-unreachable cJSON exports.

### Port: Libc transform (replacement half) — minimum set
- Trigger: ExternalSymbolGatekeeper on `strlen` from `cJSON_ParseWithOpts`
- Source: replacement-style RAUW pattern from Heap.cpp; runtime impls in
  coqui_libc.c
- Unresolved externals after internalize+globaldce (empirically derived):
  - `malloc`, `free`, `realloc` — already handled by Heap.cpp / coqui_memory.c
  - `strlen`   — new, added to Libc.cpp + coqui_libc.c
  - `strncmp`  — new, added to Libc.cpp + coqui_libc.c
  - `strtod`   — new, added to Libc.cpp + coqui_libc.c (deviation from
                  predicted set; cJSON uses strtod for number parsing)
- Functions ported: strlen, strcmp, strncmp, memcmp, strchr (completeness
  set), strtod (empirically required by cjson)
- NOT ported: memcpy, memset, memmove — clang -O2 lowers these to
  @llvm.memcpy/@llvm.memset intrinsics; NVPTX backend handles natively
- Result: pipeline reaches ptxas cleanly; only failure is ptxas not
  installed on this machine (not a symbol-resolution error)
- PTX output: 337 KB / 15420 lines; all 20 .extern declarations are
  __coqui_* (no bare libc names remain)

### Port: addrspace(5) module globals → tid-indexed .global pools
- Trigger: ptxas error "Module-scoped variables in .local state space are not allowed with ABI"
- Fix: per-thread storage moved from addrspace(5) module globals (which PTX
  disallows) to a single addrspace(0) `__coqui_thread_slots` array indexed
  by tid. Region accessors (__coqui_cov_base etc.) compute
  `slots + tid * 32 + offset`. prev_loc moved to a separate
  `__coqui_prev_loc_pool[BATCH_SIZE]` array, indexed by tid.
- Footprint: 32 bytes/thread × 8192 batch = 256 KB in cubin .global section
  (negligible vs. typical cubin size).
- Secondary fix (coqui-cc driver): runtime.bc was being linked before
  internalize+globaldce, causing all __coqui_* runtime helpers to be DCE'd
  before the pass added references to them. Fixed by linking runtime.bc
  after the prune step (new with_runtime.bc intermediate).
- Secondary fix (MemoryLayout.cpp emitGetter): forward declarations of
  __coqui_cov/heap/shadow_base from coqui_runtime.h caused the "already
  exists" guard to skip defining the functions. Fixed by checking
  isDeclaration() and filling in the body if needed.
- Result: cjson_fuzzer.cubin produced (277 KB, sm_75).
