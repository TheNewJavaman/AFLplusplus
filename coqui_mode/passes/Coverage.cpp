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

  FunctionType *VoidNoArg = FunctionType::get(voidT, false);
  FunctionCallee CheckBudget = M.getOrInsertFunction(
      "__coqui_check_thread_budget", VoidNoArg);

  FunctionType *TidT = FunctionType::get(i32, false);
  FunctionCallee Tid = M.getOrInsertFunction("__coqui_fuzz_tid", TidT);

  /* Read coverage map pointer from __coqui_cov_ptr[tid] — a dedicated
   * per-thread global array written by MemoryLayout at kernel entry.
   * This replaces __coqui_cov_base() to avoid the slot-pool aliasing
   * bug where LLVM eliminated the alloca. */
  GlobalVariable *CovPtrArr = M.getGlobalVariable("__coqui_cov_ptr");
  if (!CovPtrArr) {
    ArrayType *AT = ArrayType::get(i8p, 65536);
    CovPtrArr = new GlobalVariable(M, AT, false, GlobalValue::ExternalLinkage,
                                   nullptr, "__coqui_cov_ptr");
  }

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

      Value *tid = B.CreateCall(Tid, {}, "tid");
      Value *covPtrSlot = B.CreateGEP(i8p, CovPtrArr, {tid}, "cov_ptr_slot");
      Value *covBase = B.CreateLoad(i8p, covPtrSlot, true, "cov_base");
      Value *covPtr = B.CreateGEP(i8, covBase,
                                  ConstantInt::get(i32, edgeId), "cov_ptr");
      LoadInst *count = B.CreateLoad(i8, covPtr, true, "cov_count");
      Value *inc = B.CreateAdd(count, ConstantInt::get(i8, 1), "cov_inc");
      B.CreateStore(inc, covPtr)->setVolatile(true);

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
