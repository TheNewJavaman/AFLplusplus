/*
 * Coverage.cpp --- AFL hash-based edge instrumentation.
 *
 * At each BB entry (after PHIs), inserts the 6-instruction AFL sequence:
 *   prev = load prev_loc_pool[tid]
 *   idx = prev XOR cur_loc
 *   cov[idx]++
 *   prev_loc_pool[tid] = cur_loc >> 1
 *
 * Additionally emits a touched-region bitmap update in the same basic
 * block as the cov increment:
 *   region = idx >> 8                   // 256-byte region index (0..2047)
 *   touch_base[region >> 6] |= 1 << (region & 63)
 * so the classify path can skip untouched 256-byte chunks. The bit-set
 * happens after the cov store in the same BB; an ASan-abort mid-BB
 * leaves the summary strictly weaker-or-equal to the true touched set,
 * but crashing threads take the full-walk __coqui_trace_sig crash path
 * so the sparse classify is never called on a partial trace.
 *
 * Per-BB cur_loc constants are deterministic via xxhash of module+
 * function+block index, so recompilation yields identical edge IDs.
 * Runtime helpers (__coqui_*) are not instrumented.
 *
 * prev_loc lives in __coqui_prev_loc_pool[BATCH_SIZE] (addrspace 0 / .global),
 * indexed by tid. PTX disallows module-scope variables in .local (addrspace 5),
 * so the per-thread state uses a tid-indexed .global array instead.
 */

#include "Transforms.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/xxhash.h"

using namespace llvm;

namespace coqui {

static constexpr unsigned BATCH_SIZE_FOR_COVERAGE = 8192;

bool runCoverage(Module &M) {
  LLVMContext &C = M.getContext();
  Type *i8  = Type::getInt8Ty(C);
  Type *i32 = Type::getInt32Ty(C);
  Type *i64 = Type::getInt64Ty(C);
  Type *i8p = PointerType::get(C, 0);

  /* Declare __coqui_cov_base() -> ptr */
  FunctionType *GetBaseType = FunctionType::get(i8p, false);
  FunctionCallee CovBase = M.getOrInsertFunction("__coqui_cov_base", GetBaseType);

  /* Declare __coqui_touch_base() -> ptr (256 B summary bitmap) */
  FunctionCallee TouchBase = M.getOrInsertFunction("__coqui_touch_base", GetBaseType);

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

  for (Function &F : M) {
    if (F.isDeclaration()) continue;
    /* Skip runtime helpers — they must not be instrumented */
    if (F.getName().starts_with("__coqui_")) continue;

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

      /* Touched-region summary update.
       *   region   = idx >> 8               // 256-byte region (0..2047)
       *   word_idx = region >> 6            // u64 word within 32-word summary
       *   bit_pos  = region & 63            // bit inside that u64
       *   touch[word_idx] |= 1ULL << bit_pos
       */
      Value *touchBase = B.CreateCall(GetBaseType, TouchBase.getCallee(),
                                      {}, "touch_base");
      Value *region   = B.CreateLShr(idx, ConstantInt::get(i32, 8), "touch_region");
      Value *wordIdx  = B.CreateLShr(region, ConstantInt::get(i32, 6),
                                     "touch_word_idx");
      Value *bitPos32 = B.CreateAnd(region, ConstantInt::get(i32, 63),
                                    "touch_bit_pos");
      Value *bitPos64 = B.CreateZExt(bitPos32, i64, "touch_bit_pos64");
      Value *mask64   = B.CreateShl(ConstantInt::get(i64, 1), bitPos64,
                                    "touch_bit_mask");
      /* GEP into u64* (8-byte stride); wordIdx is already a u64-index. */
      Value *touchWordPtr = B.CreateGEP(i64, touchBase, wordIdx,
                                        "touch_word_ptr");
      Value *prevTouch = B.CreateLoad(i64, touchWordPtr, "touch_prev");
      Value *newTouch  = B.CreateOr(prevTouch, mask64, "touch_new");
      B.CreateStore(newTouch, touchWordPtr);

      /* store i32 (cur_loc >> 1), prevPtr */
      B.CreateStore(ConstantInt::get(i32, curLoc >> 1), prevPtr);
    }
  }

  return true;
}

} // namespace coqui
