/*
 * Coverage.cpp --- edge-indexed coverage instrumentation.
 *
 * Each BB gets a sequential edge ID (0, 1, 2, ...) assigned at compile
 * time. At each BB entry, inserts:
 *
 *   cov_base = __coqui_cov_base()
 *   cov_base[edgeId]++
 *
 * This replaces the AFL prev_loc XOR scheme which required 2 extra global
 * memory operations per BB (load prev_loc, store prev_loc >> 1). The
 * edge-indexed model has 4 IR ops per BB vs 10 for AFL hash — measured
 * 40% fewer coverage instructions per BB.
 *
 * Edge IDs are sequential across the module, so different BBs in different
 * functions get distinct IDs. The 64KB coverage map supports up to 65536
 * unique BBs. If the module exceeds this, IDs wrap (modulo 65536) which
 * causes collisions but doesn't break correctness — just reduces
 * coverage precision, same as AFL hash collisions.
 *
 * Per-thread budget poison (Exp #51): every Nth static BB emits a call
 * to __coqui_check_thread_budget().
 */

#include "Transforms.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace coqui {

static constexpr unsigned COV_MAP_SIZE = 65536;
static constexpr uint32_t BUDGET_CHECK_STRIDE = 64;

bool runCoverage(Module &M) {
  LLVMContext &C = M.getContext();
  Type *i8  = Type::getInt8Ty(C);
  Type *i32 = Type::getInt32Ty(C);
  Type *voidT = Type::getVoidTy(C);
  Type *i8p = PointerType::get(C, 0);

  FunctionType *GetBaseType = FunctionType::get(i8p, false);
  FunctionCallee CovBase = M.getOrInsertFunction("__coqui_cov_base", GetBaseType);

  FunctionType *VoidNoArg = FunctionType::get(voidT, false);
  FunctionCallee CheckBudget = M.getOrInsertFunction(
      "__coqui_check_thread_budget", VoidNoArg);

  uint32_t globalEdgeId = 0;
  uint32_t globalBBIdx = 0;

  for (Function &F : M) {
    if (F.isDeclaration()) continue;
    if (F.getName().starts_with("__coqui_")) continue;
    if (F.getName().starts_with("__nv_")) continue;
    if (F.getName().starts_with("__internal_")) continue;

    for (BasicBlock &BB : F) {
      Instruction *insertPt = BB.getFirstNonPHI();
      if (!insertPt) continue;

      IRBuilder<> B(insertPt);

      uint32_t edgeId = globalEdgeId % COV_MAP_SIZE;
      globalEdgeId++;

      Value *covBase = B.CreateCall(GetBaseType, CovBase.getCallee(),
                                    {}, "cov_base");
      Value *covPtr = B.CreateGEP(i8, covBase,
                                  ConstantInt::get(i32, edgeId), "cov_ptr");
      Value *count = B.CreateLoad(i8, covPtr, "cov_count");
      Value *inc = B.CreateAdd(count, ConstantInt::get(i8, 1), "cov_inc");
      B.CreateStore(inc, covPtr);

      if ((globalBBIdx % BUDGET_CHECK_STRIDE) == 0) {
        B.CreateCall(CheckBudget, {});
      }
      globalBBIdx++;
    }
  }

  if (globalEdgeId > 0)
    errs() << "[coqui-cov] " << globalEdgeId << " edges indexed"
           << (globalEdgeId > COV_MAP_SIZE ? " (WRAPPED — collisions)" : "")
           << "\n";

  return globalEdgeId > 0;
}

} // namespace coqui
