/*
 * Transforms.h --- shared pass declarations for CoquiPassPlugin.
 */

#pragma once

#include <cstdint>

namespace llvm {
class Module;
}

namespace coqui {

/* ASan right-side red-zone size in bytes. 32 = 4 shadow granules. Shared
 * between Asan.cpp (pads non-pooled globals) and StaticGlobals.cpp (inserts
 * a matching-sized padding between pooled entries so runAsanGlobals can
 * emit descriptors with the same total_size = user_size + kAsanGlobalRedZone
 * for both pooled and non-pooled entries). */
constexpr std::uint64_t kAsanGlobalRedZone = 32;

/* Name of the optional sidecar emitted by StaticGlobals when it pools any
 * writable globals: an internal constant array of {i64 offset, i64 user_size}
 * describing each pooled entry's in-slab location. runAsanGlobals reads it
 * to emit pool-kind descriptors (pool_stride > 0) that share the unified
 * __coqui_asan_global_descriptors table with non-pooled user globals. */
static constexpr const char *kAsanPoolEntriesSymbol =
    "__coqui_asan_pool_entries";
static constexpr const char *kAsanPoolEntryCountSymbol =
    "__coqui_asan_pool_entry_count";
static constexpr const char *kAsanPoolStrideSymbol =
    "__coqui_statics_per_thread"; /* already emitted by StaticGlobals */

/* Each returns true if the module was modified (standard LLVM convention). */

bool runRejectInlineAsm(llvm::Module &M);
bool runRejectLibc(llvm::Module &M);
/* RejectSyscall: compile-time hard-reject for GPU-impossible syscalls
 * (dlopen/socket/mmap/...). SyscallTransform rewrites divertable
 * syscalls (signal/fork/exec/time/...) to runtime stubs that either
 * no-op, return a deterministic value, or trap with a category-specific
 * reason code. */
bool runRejectSyscall(llvm::Module &M);
bool runSyscallTransform(llvm::Module &M);
bool runRejectIntrinsics(llvm::Module &M);
bool runFuzzEntry(llvm::Module &M);
bool runCpp(llvm::Module &M);
bool runHeap(llvm::Module &M);
/* StackSpill: rewrite oversize allocas (per-function budget exceeded or
 * dynamic) into __coqui_stack_alloc calls. Save/restore brackets at fn
 * entry/return rewind a per-thread bump pointer through carved slab pages.
 * Spill pointers' loads/stores get !nosanitize so runAsan skips them. */
bool runStackSpill(llvm::Module &M);
bool runSprintf(llvm::Module &M);
bool runLibc(llvm::Module &M);
bool runVariadic(llvm::Module &M);
bool runReloc(llvm::Module &M);
bool runMath(llvm::Module &M);
bool runComplex(llvm::Module &M);
bool runStaticGlobals(llvm::Module &M);
/* runGlobalCtors lowers @llvm.global_ctors into __coqui_global_init() and
 * wires a call into __coqui_fuzz_kernel right before the user harness. No-op
 * for modules without ctors (the runtime's weak fallback resolves cleanly). */
bool runGlobalCtors(llvm::Module &M);
bool runMemoryLayout(llvm::Module &M);
bool runCoverage(llvm::Module &M);
bool runAsanGlobals(llvm::Module &M);
bool runAsan(llvm::Module &M);
bool runExternalSymbolGatekeeper(llvm::Module &M);

/* AddressSpace: promote module-level non-llvm.*, non-__coqui_*, non-thread-
 * local globals from default address space (AS=0) to NVPTX `.global` (AS=1).
 * Lets ptxas emit ld.global / st.global directly instead of going through
 * generic-address-space resolution — meaningful on read-heavy code paths
 * (Huffman tables, DCT lookup, color-space matrices). Runs as the very
 * last pass in the pipeline so every prior pass that creates new globals
 * has already run by the time we sweep. */
bool runAddressSpace(llvm::Module &M);

/* IndirectCall: devirtualize indirect function-pointer calls into bounded
 * dispatch chains (icmp + cond-branch over each address-taken candidate
 * with a compatible signature, with a final trap arm). NVPTX has no branch
 * prediction across function-pointer dispatch, warps serialize across
 * divergent callees, and function-pointer dispatch is essentially
 * uncacheable on the GPU. Devirt'ing into direct calls lets ptxas lower
 * the chain to selp/setp.eq sequences and lets the inliner reach into
 * each arm. Per-FunctionType type matching, narrowed by a struct-field /
 * global-variable store-set analysis to avoid quadratic blowup. Runs
 * AFTER ExternalSymbolGatekeeper so the final set of address-taken
 * functions is stable, and before AddressSpace. */
bool runIndirectCall(llvm::Module &M);

/* Line-trace pass (oracle mode). Gated by the `-coqui-line-trace` opt flag.
 * Returns false (no-op) when the flag is unset, so production runs pay
 * nothing. See LineTrace.cpp for the (file, line) ID assignment scheme +
 * skip list. */
bool runLineTrace(llvm::Module &M);

} // namespace coqui
