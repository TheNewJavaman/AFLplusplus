/*
 * RejectInlineAsm.cpp --- reject arch-specific inline asm.
 *
 * Strip benign compiler barriers (empty or whitespace-only asm strings
 * with "memory" clobber) silently. FATAL on anything else with
 * source-location-informed error.
 *
 * Ported from /coqui/src/InlineAsmTransform.cpp (rejection half).
 */

#include "Transforms.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <string>
#include <vector>

using namespace llvm;

namespace coqui {

static bool isBenign(StringRef asmStr, StringRef constraints) {
  /* Empty or whitespace-only asm with memory clobber is a compiler barrier
   * — typically __asm__ volatile("" ::: "memory") used to prevent load/store
   * reordering across the sequence point.
   *
   * Erasing it DOES remove the barrier. On NVPTX this is a safe approximation
   * in practice: (a) our GPU fuzz targets are per-thread single-threaded with
   * no CPU-style cache coherence concerns, (b) cross-thread ordering is the
   * job of CUDA __threadfence / atomics (emitted as separate NVPTX intrinsics
   * by clang, not as inline asm), (c) the barrier was almost always inserted
   * by musl or libc headers for x86_64 semantics that don't apply on the
   * device. If a future target DOES depend on a compile-time reorder barrier
   * we'd need to either preserve it (lower to an `@llvm.memory.barrier` or a
   * fence.sc NVPTX intrinsic) or hard-fail. So far none have. */
  std::string trimmed = asmStr.trim().str();
  if (trimmed.empty() && constraints.contains("memory")) return true;
  return false;
}

bool runRejectInlineAsm(Module &M) {
  bool changed = false;
  std::vector<CallInst*> toErase;

  for (Function &F : M) {
    if (F.getName().starts_with("__coqui_")) continue;   /* trust runtime */
    if (F.getName().starts_with("__nv_")) continue;      /* trust libdevice */
    if (F.getName().starts_with("__internal_")) continue; /* libdevice helpers */
    for (BasicBlock &BB : F) {
      for (Instruction &I : BB) {
        CallInst *CI = dyn_cast<CallInst>(&I);
        if (!CI) continue;

        InlineAsm *IA = dyn_cast<InlineAsm>(CI->getCalledOperand());
        if (!IA) continue;

        StringRef asmStr = IA->getAsmString();
        StringRef constraints = IA->getConstraintString();

        if (isBenign(asmStr, constraints)) {
          toErase.push_back(CI);
          changed = true;
          continue;
        }

        /* Non-trivial asm — FATAL with context */
        std::string msg;
        raw_string_ostream os(msg);
        os << "[coqui-cc] RejectInlineAsm: non-trivial inline asm in function '"
           << F.getName() << "': " << asmStr.str();
        report_fatal_error(os.str().c_str());
      }
    }
  }

  for (CallInst *CI : toErase) CI->eraseFromParent();
  return changed;
}

} // namespace coqui
