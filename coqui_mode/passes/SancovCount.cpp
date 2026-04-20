/*
 * SancovCount.cpp --- assign sequential IDs to SanitizerCoverage guard arrays.
 *
 * When the target is compiled with clang -fsanitize-coverage=trace-pc-guard,
 * clang emits one [N x i32] global per translation unit, named
 * __sancov_gen_*, and at every instrumented edge it calls
 * __sanitizer_cov_trace_pc_guard(&guard[i]).
 *
 * This pass walks all __sancov_gen_* arrays in the (post-link) module and
 * assigns each element a unique 1-based u32 ID. The device-side sancov
 * runtime then indexes the per-thread cov_map with (guard - 1), matching the
 * host SanCov convention where guard==0 means "uninitialized".
 *
 * Because every guard is pre-initialized here, __sanitizer_cov_trace_pc_guard_init
 * becomes a no-op, and the guard arrays are marked `constant` so LLVM can
 * place them in read-only memory.
 *
 * Also emits @__coqui_num_edges : i64 (constant) so the host can read the
 * final edge count via cuModuleGetGlobal and size the cov_map_pool
 * accordingly (instead of assuming the legacy 64 KB AFL default).
 *
 * Ported from /home/gpizarro/coqui/src/SancovCountTransform.cpp with the
 * cuAFL naming (the function is exposed as runSancovCount, matching the
 * rest of our pass pipeline).
 */

#include "Transforms.h"

#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace coqui {

/* Strip sancov.module_ctor_trace_pc_guard entries from @llvm.global_ctors.
 * The ctor only calls __sanitizer_cov_trace_pc_guard_init, which is a no-op
 * once SancovCount has pre-populated every guard with a sequential ID.
 *
 * We also clean up any extern_weak __start___sancov_guards /
 * __stop___sancov_guards declarations to prevent ptxas from rejecting
 * unresolved weak externs. */
static void cleanupSancovCtors(Module &M) {
  /* 1. Filter @llvm.global_ctors to drop sancov ctor refs. */
  GlobalVariable *Ctors = M.getGlobalVariable("llvm.global_ctors", true);
  if (Ctors && Ctors->hasInitializer()) {
    auto *ArrInit = dyn_cast<ConstantArray>(Ctors->getInitializer());
    if (ArrInit) {
      auto *ArrTy = cast<ArrayType>(ArrInit->getType());
      auto *StructTy = cast<StructType>(ArrTy->getElementType());

      SmallVector<Constant *, 8> Keep;
      for (unsigned i = 0, n = ArrInit->getNumOperands(); i < n; ++i) {
        auto *Entry = cast<ConstantStruct>(ArrInit->getOperand(i));
        /* Field 1 is the ctor function pointer. */
        Value *FnOp = Entry->getOperand(1)->stripPointerCasts();
        if (auto *Fn = dyn_cast<Function>(FnOp)) {
          StringRef Name = Fn->getName();
          if (Name.starts_with("sancov.module_ctor")) continue;
        }
        Keep.push_back(Entry);
      }

      if (Keep.size() == ArrInit->getNumOperands()) return;

      if (Keep.empty()) {
        Ctors->eraseFromParent();
      } else {
        auto *NewArrTy = ArrayType::get(StructTy, Keep.size());
        auto *NewInit = ConstantArray::get(NewArrTy, Keep);
        auto *NewGV = new GlobalVariable(
            M, NewArrTy, /*isConstant=*/false,
            GlobalValue::AppendingLinkage, NewInit, "");
        NewGV->takeName(Ctors);
        Ctors->replaceAllUsesWith(NewGV);
        Ctors->eraseFromParent();
      }
    }
  }

  /* 2. Remove sancov.module_ctor_* function bodies if they become unused
   * after step 1. Also remove __start/__stop extern_weak symbols that are
   * dead on NVPTX (ptxas errors on unresolved extern_weak globals).
   *
   * Dropping the .used / .compiler.used references is the safest path: set
   * them to zeroinitializer arrays so the extern_weak globals are no
   * longer live. globaldce (if run afterwards) would sweep them; we also
   * explicitly erase. */
  auto eraseIfUnused = [](GlobalValue *GV) {
    if (!GV) return;
    if (GV->use_empty()) {
      if (auto *F = dyn_cast<Function>(GV)) F->eraseFromParent();
      else if (auto *V = dyn_cast<GlobalVariable>(GV)) V->eraseFromParent();
    }
  };

  /* Drop the ctor function entirely; drop the __start/__stop symbols too
   * (they are weak externs and have no definition on NVPTX). */
  for (Function &F : llvm::make_early_inc_range(M)) {
    if (F.getName().starts_with("sancov.module_ctor")) eraseIfUnused(&F);
  }
  eraseIfUnused(M.getNamedGlobal("__start___sancov_guards"));
  eraseIfUnused(M.getNamedGlobal("__stop___sancov_guards"));

  /* Drop llvm.used / llvm.compiler.used too — they hold section-keeping
   * references that are only meaningful for native linkers and can end up
   * preserving symbols ptxas doesn't know how to emit. On NVPTX we rely on
   * pass-level invariants, not section retention. */
  for (const char *Name : {"llvm.used", "llvm.compiler.used"}) {
    if (auto *GV = M.getNamedGlobal(Name)) GV->eraseFromParent();
  }
}

bool runSancovCount(Module &M) {
  SmallVector<GlobalVariable *, 16> GuardArrays;
  for (GlobalVariable &GV : M.globals()) {
    if (!GV.getName().starts_with("__sancov_gen_")) continue;
    if (isa<ArrayType>(GV.getValueType()))
      GuardArrays.push_back(&GV);
  }

  /* If there are no __sancov_gen_* arrays, the target was not compiled with
   * -fsanitize-coverage=trace-pc-guard. Emit @__coqui_num_edges = 0 so the
   * host's cuModuleGetGlobal lookup succeeds (zero => legacy / no coverage
   * via sancov). */
  unsigned long TotalEdges = 0;

  if (!GuardArrays.empty()) {
    auto *I32Ty = Type::getInt32Ty(M.getContext());
    unsigned long NextID = 1;

    for (GlobalVariable *GV : GuardArrays) {
      auto *ArrTy = cast<ArrayType>(GV->getValueType());
      uint64_t N = ArrTy->getNumElements();

      SmallVector<Constant *, 64> Elts;
      Elts.reserve(N);
      for (uint64_t i = 0; i < N; i++)
        Elts.push_back(ConstantInt::get(I32Ty, NextID++));

      GV->setInitializer(ConstantArray::get(ArrTy, Elts));
      /* Guards are pre-assigned and never written; mark constant so LLVM can
       * place them in .const and enable invariant-load optimization. */
      GV->setConstant(true);
      /* Strip the x86-linker-specific "__sancov_guards" section and comdat.
       * ptxas does not know either, and we don't need section-based linking
       * because all guard arrays are fully initialised and globaldce'd. */
      GV->setSection("");
      GV->setComdat(nullptr);
    }

    TotalEdges = NextID - 1;
  }

  /* Emit @__coqui_num_edges = constant i64 <N> so the host can read it via
   * cuModuleGetGlobal. Overwrite any prior declaration (e.g. from the
   * runtime header). */
  auto *I64Ty = Type::getInt64Ty(M.getContext());
  auto *NumEdgesInit = ConstantInt::get(I64Ty, TotalEdges);

  if (GlobalVariable *Existing = M.getGlobalVariable("__coqui_num_edges", true)) {
    Existing->setConstant(true);
    Existing->setInitializer(NumEdgesInit);
    Existing->setLinkage(GlobalValue::ExternalLinkage);
  } else {
    auto *GV = new GlobalVariable(M, I64Ty, /*isConstant=*/true,
                                  GlobalValue::ExternalLinkage, NumEdgesInit,
                                  "__coqui_num_edges");
    GV->setAlignment(Align(8));
  }

  /* Clean up sancov module ctors + extern_weak __start/__stop globals so
   * ptxas doesn't choke on unresolved section-marker symbols. */
  cleanupSancovCtors(M);

  errs() << "[coqui] SanCov: " << TotalEdges
         << " edges (sequential IDs assigned) -> @__coqui_num_edges\n";

  return true;
}

} // namespace coqui
