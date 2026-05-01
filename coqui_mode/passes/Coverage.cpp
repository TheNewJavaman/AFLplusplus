/*
 * Coverage.cpp --- AFL hash-based edge instrumentation.
 *
 * At each BB entry (after PHIs), inserts the 6-instruction AFL sequence:
 *   prev = load prev_loc_pool[tid]
 *   idx = prev XOR cur_loc
 *   cov[idx]++
 *   prev_loc_pool[tid] = cur_loc >> 1
 *
 * Per-BB cur_loc constants are deterministic via xxhash of module+
 * function+block index, so recompilation yields identical edge IDs.
 * Runtime helpers (__coqui_*) are not instrumented.
 *
 * prev_loc lives in __coqui_prev_loc_pool[BATCH_SIZE] (addrspace 0 / .global),
 * indexed by tid. PTX disallows module-scope variables in .local (addrspace 5),
 * so the per-thread state uses a tid-indexed .global array instead.
 *
 * Per-thread budget poison (Exp #51): every Nth static BB (N=64) emits a
 * call to __coqui_check_thread_budget() — a runtime helper that compares
 * clock64() - per-thread start against a host-set ceiling and traps the
 * thread (only) on overrun. Disabled-default by host (cycles_cap=0); when
 * disabled the helper is a single global-load + cmp + ret. Targets the
 * force-reset bottleneck class: one slow thread no longer stalls the
 * entire 8000-input batch.
 */

#include "Transforms.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/xxhash.h"

using namespace llvm;

namespace coqui {

static constexpr unsigned BATCH_SIZE_FOR_COVERAGE = 65536;

/* Frequency of __coqui_check_thread_budget() emission. Once every N
 * static BBs (counted across the entire module's user functions). N=64
 * trades dynamic check granularity vs IR/PTX size: 1.5% of BBs add a
 * 1-instruction call site; on a 30k-BB target that's ~470 extra calls,
 * which adds <1% to cubin size. */
static constexpr uint32_t BUDGET_CHECK_STRIDE = 64;

bool runCoverage(Module &M) {
  LLVMContext &C = M.getContext();
  Type *i8  = Type::getInt8Ty(C);
  Type *i32 = Type::getInt32Ty(C);
  Type *voidT = Type::getVoidTy(C);
  Type *i8p = PointerType::get(C, 0);

  /* Declare __coqui_cov_base() -> ptr */
  FunctionType *GetBaseType = FunctionType::get(i8p, false);
  FunctionCallee CovBase = M.getOrInsertFunction("__coqui_cov_base", GetBaseType);

  /* Declare __coqui_check_thread_budget() -> void. The runtime defines it
   * with `noinline` so call sites compile to a single bl/ret. */
  FunctionType *VoidNoArg = FunctionType::get(voidT, false);
  FunctionCallee CheckBudget = M.getOrInsertFunction(
      "__coqui_check_thread_budget", VoidNoArg);

  /* Declare __coqui_prev_loc_pool as u32[BATCH_SIZE] in addrspace 0 (.global).
   * Each thread accesses element [tid]. PTX does not allow module-scope
   * variables in addrspace 5 (.local); addrspace 0 (.global) + tid index
   * is the standard NVPTX pattern for per-thread state. */
  GlobalVariable *PrevLocPool = M.getGlobalVariable("__coqui_prev_loc_pool");
  if (!PrevLocPool) {
    ArrayType *T = ArrayType::get(i32, BATCH_SIZE_FOR_COVERAGE);
    PrevLocPool = new GlobalVariable(
        M, T, /*isConstant=*/false,
        GlobalValue::ExternalLinkage,
        Constant::getNullValue(T),
        "__coqui_prev_loc_pool"); /* default addrspace 0 = .global */
  }

  /* Declare __coqui_fuzz_tid() -> i32 */
  FunctionType *TidT = FunctionType::get(i32, /*isVarArg=*/false);
  FunctionCallee Tid = M.getOrInsertFunction("__coqui_fuzz_tid", TidT);

  /* Stable per-module seed: hash the module name once */
  uint64_t moduleSeed = llvm::xxh3_64bits(M.getName());

  /* Module-wide static BB counter — increments across functions so the
   * stride is stable regardless of function ordering. */
  uint32_t globalBBIdx = 0;

  for (Function &F : M) {
    if (F.isDeclaration()) continue;
    /* Skip runtime helpers and libdevice — they must not be instrumented */
    if (F.getName().starts_with("__coqui_")) continue;
    if (F.getName().starts_with("__nv_")) continue;
    if (F.getName().starts_with("__internal_")) continue;

    uint64_t fnSeed = llvm::xxh3_64bits(F.getName());

    uint32_t blockIdx = 0;
    for (BasicBlock &BB : F) {
      uint64_t mix = moduleSeed ^ fnSeed ^ static_cast<uint64_t>(blockIdx);
      /* Fold 64-bit mix to 16 bits */
      uint16_t curLoc = static_cast<uint16_t>(
          (mix ^ (mix >> 16) ^ (mix >> 32) ^ (mix >> 48)) & 0xFFFF);
      blockIdx++;

      /* Insertion point: after PHIs */
      Instruction *insertPt = BB.getFirstNonPHI();
      if (!insertPt) continue; /* empty block guard */

      IRBuilder<> B(insertPt);

      /* tid = __coqui_fuzz_tid() */
      Value *tid = B.CreateCall(Tid, {}, "tid");

      /* prevPtr = gep i32, PrevLocPool, tid */
      Value *prevPtr = B.CreateGEP(i32, PrevLocPool, tid, "prev_ptr");

      /* prev = load i32, prevPtr */
      Value *prev = B.CreateLoad(i32, prevPtr, "cov_prev");

      /* idx = prev ^ cur_loc */
      Value *idx = B.CreateXor(prev, ConstantInt::get(i32, curLoc), "cov_idx");

      /* cov_base = __coqui_cov_base() */
      Value *covBase = B.CreateCall(GetBaseType, CovBase.getCallee(), {}, "cov_base");

      /* cov_ptr = gep i8, cov_base, idx */
      Value *covPtr = B.CreateGEP(i8, covBase, idx, "cov_ptr");

      /* count = load i8, cov_ptr */
      Value *count = B.CreateLoad(i8, covPtr, "cov_count");

      /* inc = count + 1 */
      Value *inc = B.CreateAdd(count, ConstantInt::get(i8, 1), "cov_inc");

      /* store i8 inc, cov_ptr */
      B.CreateStore(inc, covPtr);

      /* store i32 (cur_loc >> 1), prevPtr */
      B.CreateStore(ConstantInt::get(i32, curLoc >> 1), prevPtr);

      /* Per-thread budget check at every Nth static BB. The runtime
       * helper is a no-op when the host hasn't set
       * __coqui_thread_budget_cycles (production-default), so the
       * runtime cost is one bl + the helper's early-out compare. */
      if ((globalBBIdx % BUDGET_CHECK_STRIDE) == 0) {
        B.CreateCall(CheckBudget, {});
      }
      globalBBIdx++;
    }
  }

  return true;
}

} // namespace coqui
