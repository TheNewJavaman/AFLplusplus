/*
 * EdgeCount.cpp --- pre-pass that counts instrumentation-eligible BBs.
 *
 * Runs BEFORE MemoryLayout so the coverage map alloca + virgin map can
 * be sized to the actual edge count instead of a fixed 64KB. Stores
 * the count in a module-level constant global __coqui_edge_count that
 * MemoryLayout, FuzzEntry, and the fused coverage runtime all read.
 *
 * Uses the same skip predicates as Coverage.cpp (skip __coqui_*, __nv_*,
 * __internal_*, declarations) so the count matches exactly.
 */

#include "Transforms.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace coqui {

bool runEdgeCount(Module &M) {
  uint32_t count = 0;

  for (Function &F : M) {
    if (F.isDeclaration()) continue;
    if (F.getName().starts_with("__coqui_")) continue;
    if (F.getName().starts_with("__nv_")) continue;
    if (F.getName().starts_with("__internal_")) continue;

    for (BasicBlock &BB : F) {
      if (!BB.getFirstNonPHI()) continue;
      count++;
    }
  }

  if (count == 0) return false;

  /* Round up to multiple of 8 for u64-aligned iteration in the fused
   * coverage function. Minimum 64 to avoid degenerate tiny maps. */
  uint32_t mapSize = ((count + 7u) & ~7u);
  if (mapSize < 64) mapSize = 64;

  LLVMContext &C = M.getContext();
  Type *i32 = Type::getInt32Ty(C);

  GlobalVariable *GV = M.getGlobalVariable("__coqui_edge_count");
  if (GV) {
    GV->setInitializer(ConstantInt::get(i32, mapSize));
    GV->setConstant(true);
  } else {
    new GlobalVariable(
        M, i32, /*isConstant=*/true,
        GlobalValue::ExternalLinkage,
        ConstantInt::get(i32, mapSize),
        "__coqui_edge_count");
  }

  errs() << "[coqui-edgecount] " << count << " BBs → map size "
         << mapSize << " bytes\n";

  return true;
}

} // namespace coqui
