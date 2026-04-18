/*
 * Heap.cpp --- replace standard malloc/free calls with __coqui_*.
 *
 * Ported verbatim from /coqui/src/HeapTransform.cpp. Uses RAUW
 * (replaceAllUsesWith) on each libc allocator symbol.
 */

#include "Transforms.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"

using namespace llvm;

namespace coqui {

static bool replaceAlloc(Module &M, StringRef Old, StringRef New) {
  Function *OldF = M.getFunction(Old);
  if (!OldF || OldF->use_empty()) return false;

  Function *NewF = M.getFunction(New);
  if (!NewF) {
    NewF = Function::Create(OldF->getFunctionType(),
                            GlobalValue::ExternalLinkage, New, &M);
  }

  OldF->replaceAllUsesWith(NewF);
  OldF->eraseFromParent();
  return true;
}

bool runHeap(Module &M) {
  bool changed = false;
  changed |= replaceAlloc(M, "malloc",         "__coqui_malloc");
  changed |= replaceAlloc(M, "free",           "__coqui_free");
  changed |= replaceAlloc(M, "calloc",         "__coqui_calloc");
  changed |= replaceAlloc(M, "realloc",        "__coqui_realloc");
  changed |= replaceAlloc(M, "aligned_alloc",  "__coqui_malloc");
  changed |= replaceAlloc(M, "posix_memalign", "__coqui_malloc");
  return changed;
}

} // namespace coqui
