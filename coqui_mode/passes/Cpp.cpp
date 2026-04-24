/*
 * Cpp.cpp --- rewrite C++ operator new/delete to the coqui runtime heap.
 *
 * Ported from /coqui/src/CppTransform.cpp. Handles the Itanium-ABI
 * mangled symbols emitted by clang for `new`, `new[]`, `delete`, and
 * `delete[]`:
 *
 *   _Znwm   operator new(size_t)          -> __coqui_malloc
 *   _Znam   operator new[](size_t)        -> __coqui_malloc
 *   _ZdlPv  operator delete(void*)        -> __coqui_free
 *   _ZdaPv  operator delete[](void*)      -> __coqui_free
 *
 * The "direct" variants above match the cuAFL runtime signatures and
 * can use plain replaceAllUsesWith. The sized-delete variants
 * (_ZdlPvm, _ZdaPvm) take an extra size_t and need per-call-site
 * rewriting to drop it.
 *
 * Must run before Heap.cpp: Heap rewrites `malloc`/`free` directly,
 * and if any C++ operator-new site leaks through without lowering
 * first, the ExternalSymbolGatekeeper will reject it later.
 */

#include "Transforms.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"

#include <utility>

using namespace llvm;

namespace coqui {

bool runCpp(Module &M) {
  /* Non-sized variants: matching signatures, RAUW is sufficient. */
  static constexpr std::pair<const char *, const char *> DirectReplacements[] = {
      {"_Znwm",  "__coqui_malloc"},  /* operator new(size_t)       */
      {"_Znam",  "__coqui_malloc"},  /* operator new[](size_t)     */
      {"_ZdlPv", "__coqui_free"},    /* operator delete(void*)     */
      {"_ZdaPv", "__coqui_free"},    /* operator delete[](void*)   */
  };

  /* Sized-delete variants: (void*, size_t) -> __coqui_free(void*).
   * Signature mismatch forces per-call-site rewriting. */
  static constexpr std::pair<const char *, const char *> SizedDeleteReplacements[] = {
      {"_ZdlPvm", "__coqui_free"},   /* operator delete(void*, size_t)   */
      {"_ZdaPvm", "__coqui_free"},   /* operator delete[](void*, size_t) */
  };

  bool Changed = false;

  /* Direct (signature-matching) replacements. */
  for (auto &[OldName, NewName] : DirectReplacements) {
    Function *F = M.getFunction(OldName);
    if (!F) continue;

    FunctionCallee NewF = M.getOrInsertFunction(NewName, F->getFunctionType());

    errs() << "[coqui-cpp] replacing " << OldName << " -> " << NewName
           << " (" << F->getNumUses() << " uses)\n";

    F->replaceAllUsesWith(NewF.getCallee());
    F->eraseFromParent();
    Changed = true;
  }

  /* Sized-delete variants. Build __coqui_free with its actual void(ptr)
   * signature, then rewrite each call site to pass only the pointer arg. */
  for (auto &[OldName, NewName] : SizedDeleteReplacements) {
    Function *F = M.getFunction(OldName);
    if (!F) continue;

    errs() << "[coqui-cpp] replacing " << OldName << " -> " << NewName
           << " (" << F->getNumUses() << " uses, sized delete)\n";

    LLVMContext &Ctx = M.getContext();
    Type *VoidTy = Type::getVoidTy(Ctx);
    Type *PtrTy = PointerType::getUnqual(Ctx);
    FunctionType *FreeTy = FunctionType::get(VoidTy, {PtrTy}, false);
    FunctionCallee FreeFn = M.getOrInsertFunction(NewName, FreeTy);

    /* Collect call sites first (CallBase handles both CallInst and
     * InvokeInst; C++ destructors in try-blocks emit invoke for
     * operator delete). */
    SmallVector<CallBase *, 8> CallsToReplace;
    for (Use &U : make_early_inc_range(F->uses())) {
      if (auto *CB = dyn_cast<CallBase>(U.getUser())) {
        if (CB->getCalledFunction() == F)
          CallsToReplace.push_back(CB);
      }
    }

    for (CallBase *CB : CallsToReplace) {
      IRBuilder<> Builder(CB);
      Value *Ptr = CB->getArgOperand(0);
      Builder.CreateCall(FreeFn, {Ptr});
      /* For InvokeInst: __coqui_free is nothrow, so the unwind path is
       * dead. Replace the invoke with an unconditional branch to the
       * normal destination. */
      if (auto *II = dyn_cast<InvokeInst>(CB)) {
        BranchInst::Create(II->getNormalDest(), II);
        II->eraseFromParent();
      } else {
        CB->eraseFromParent();
      }
    }

    if (F->use_empty())
      F->eraseFromParent();

    Changed = true;
  }

  return Changed;
}

} // namespace coqui
