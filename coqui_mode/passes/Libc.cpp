/*
 * Libc.cpp --- replace libc string/math functions with __coqui_* equivalents.
 *
 * Day-1 minimum set (empirically derived from cjson after
 * internalize+globaldce pruning): strlen, strncmp, strtod.
 * Full set also includes strcmp, memcmp, strchr for completeness.
 * Add more on demand as ExternalSymbolGatekeeper flags them.
 *
 * memcpy/memset/memmove are NOT replaced — clang lowers them to
 * @llvm.memcpy/@llvm.memset intrinsics, which the NVPTX backend
 * handles natively without external symbol calls.
 */

#include "Transforms.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"

using namespace llvm;

namespace coqui {

static bool replaceFn(Module &M, StringRef Old, StringRef New) {
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

bool runLibc(Module &M) {
  bool changed = false;
  changed |= replaceFn(M, "strlen",   "__coqui_strlen");
  changed |= replaceFn(M, "strcmp",   "__coqui_strcmp");
  changed |= replaceFn(M, "strncmp",  "__coqui_strncmp");
  changed |= replaceFn(M, "memcmp",   "__coqui_memcmp");
  changed |= replaceFn(M, "strchr",   "__coqui_strchr");
  changed |= replaceFn(M, "strtod",   "__coqui_strtod");
  return changed;
}

} // namespace coqui
