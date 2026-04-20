/*
 * FuzzEntry.cpp --- rename LLVMFuzzerTestOneInput, create kernel entry.
 *
 * Per coqui internals design §7.1: emits __coqui_fuzz_kernel with the
 * 5-argument signature (input_bytes, offsets, input_lens, novelty, status).
 *
 * Kernel body calls __coqui_fuzz_execute per-thread, then runs a single-pass
 * __coqui_coverage_evaluate(cov, global_cov, cov_map_size) which classifies,
 * zeros in place, and atom.or's into the global cov map — returning 1 if the
 * thread contributed novelty.
 *
 * Adds nvvm.annotations to mark __coqui_fuzz_kernel as .entry.
 *
 * Iteration 10: migrated from AFL hash-edge coverage to SanitizerCoverage
 * (trace-pc-guard). The virgin/novelty bitmap is now sized to num_edges
 * (read at host init via cuModuleGetGlobal @__coqui_num_edges) and lives in
 * a device-global pool pointed to by @__coqui_virgin_map_ptr. Both the
 * cov_map and virgin_map are dynamically sized per-module; only the host
 * knows the sizes, so the kernel reads the base pointers from module-scope
 * globals rather than hard-coded [65536 x i8] arrays.
 */

#include "Transforms.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

namespace coqui {

bool runFuzzEntry(Module &M) {
  LLVMContext &C = M.getContext();
  IRBuilder<> Builder(C);

  /* 1. Rename LLVMFuzzerTestOneInput */
  Function *User = M.getFunction("LLVMFuzzerTestOneInput");
  if (!User) {
    report_fatal_error(
      "[coqui-cc] FuzzEntry: LLVMFuzzerTestOneInput not found — "
      "every cuAFL target must define it");
  }
  User->setName("__coqui_fuzz_execute");

  Type *i8   = Type::getInt8Ty(C);
  Type *i32  = Type::getInt32Ty(C);
  Type *i64  = Type::getInt64Ty(C);
  Type *i8p  = PointerType::get(C, 0);   /* opaque pointer, addrspace 0 */
  Type *voidT = Type::getVoidTy(C);

  /* 2. Build the kernel function type and create */
  FunctionType *KernelType = FunctionType::get(
    voidT, {i8p, i8p, i8p, i8p, i8p}, false);
  Function *Kernel = Function::Create(
    KernelType, GlobalValue::ExternalLinkage,
    "__coqui_fuzz_kernel", &M);

  /* Name the args for readability */
  auto argIt = Kernel->arg_begin();
  Value *inputBytes = &*argIt++; inputBytes->setName("input_bytes");
  Value *offsetsArg = &*argIt++; offsetsArg->setName("offsets");
  Value *lensArg    = &*argIt++; lensArg->setName("lens");
  Value *noveltyArg = &*argIt++; noveltyArg->setName("novelty");
  Value *statusArg  = &*argIt++; statusArg->setName("status");

  /* 3. Declare / look up helpers */
  FunctionType *VoidNoArg = FunctionType::get(voidT, false);
  FunctionType *TidRet    = FunctionType::get(i32, false);
  FunctionType *PtrNoArg  = FunctionType::get(i8p, false);
  FunctionType *I64NoArg  = FunctionType::get(i64, false);

  FunctionCallee GetTid = M.getOrInsertFunction("__coqui_fuzz_tid", TidRet);
  FunctionCallee SetPhase = M.getOrInsertFunction(
    "__coqui_status_set_phase",
    FunctionType::get(voidT, {i32, i8}, false));
  FunctionCallee MemoryInit = M.getOrInsertFunction("__coqui_memory_init", VoidNoArg);
  FunctionCallee CovBase    = M.getOrInsertFunction("__coqui_cov_base", PtrNoArg);
  FunctionCallee CovSize    = M.getOrInsertFunction("__coqui_cov_map_size", I64NoArg);

  /* Single-pass coverage_evaluate: classify counters, zero in place, OR-reduce
   * across the warp, and atom.or the warp result into the global bitmap. Returns
   * 1 if this thread contributed a new bit, 0 otherwise. Replaces both the
   * old classify_counts_and_sig and virgin_compare_and_flag paths. */
  FunctionCallee CoverageEval = M.getOrInsertFunction(
    "__coqui_coverage_evaluate",
    FunctionType::get(i32, {i8p, i8p, i64}, false));

  /* FNV-1a hash of the partial coverage map after classification — used for
   * crash dedup. Read-only; does not mutate the map (classify+zero already
   * happened in coverage_evaluate). */
  FunctionCallee TraceSig = M.getOrInsertFunction(
    "__coqui_trace_sig",
    FunctionType::get(i32, {i8p, i64}, false));

  /* 4. Virgin map is now dynamically sized at host init. The kernel reads the
   * base pointer from @__coqui_virgin_map_ptr (ptr, addrspace 0). Host binds
   * this via cuModuleGetGlobal + cuMemcpyHtoD after cuMemAllocing the map. */
  GlobalVariable *VirginMapPtr = M.getGlobalVariable("__coqui_virgin_map_ptr", true);
  if (!VirginMapPtr) {
    VirginMapPtr = new GlobalVariable(
      M, i8p, /*isConstant*/false, GlobalValue::ExternalLinkage,
      Constant::getNullValue(i8p), "__coqui_virgin_map_ptr");
    VirginMapPtr->setAlignment(Align(8));
  }

  /* 5. __coqui_status_array pointer global (runtime will read this) */
  GlobalVariable *StatusArrayPtr = M.getGlobalVariable("__coqui_status_array", true);
  if (!StatusArrayPtr) {
    StatusArrayPtr = new GlobalVariable(
      M, i8p, /*isConstant*/false, GlobalValue::ExternalLinkage,
      Constant::getNullValue(i8p), "__coqui_status_array");
  }

  /* 6. Emit kernel body */
  BasicBlock *EntryBB = BasicBlock::Create(C, "entry", Kernel);
  BasicBlock *RunBB   = BasicBlock::Create(C, "run",   Kernel);
  BasicBlock *ExitBB  = BasicBlock::Create(C, "exit",  Kernel);

  Builder.SetInsertPoint(EntryBB);

  /* Store status pointer into global so runtime can find it */
  Builder.CreateStore(statusArg, StatusArrayPtr);

  /* tid = __coqui_fuzz_tid() */
  Value *tid = Builder.CreateCall(GetTid, {}, "tid");

  /* len = lens[tid]  — GEP into i32 array at index tid */
  Value *lensPtr = Builder.CreateGEP(i32, lensArg, {tid}, "lens_tid");
  Value *len     = Builder.CreateLoad(i32, lensPtr, "len");

  /* if (len == 0) goto exit; else goto run */
  Value *isEmpty = Builder.CreateICmpEQ(len, ConstantInt::get(i32, 0));
  Builder.CreateCondBr(isEmpty, ExitBB, RunBB);

  Builder.SetInsertPoint(RunBB);

  /* PHASE_START = 1 */
  Builder.CreateCall(SetPhase, {tid, ConstantInt::get(i8, 1)});

  /* __coqui_memory_init() */
  Builder.CreateCall(MemoryInit, {});

  /* off = offsets[tid]; input_ptr = input_bytes + off */
  Value *offPtr   = Builder.CreateGEP(i32, offsetsArg, {tid}, "offsets_tid");
  Value *off      = Builder.CreateLoad(i32, offPtr, "off");
  /* Byte-stride GEP into the flat byte buffer: input_bytes + off
     off is i32; widen to i64 for pointer arithmetic */
  Value *off64    = Builder.CreateZExt(off, i64, "off64");
  Value *inputPtr = Builder.CreateGEP(i8, inputBytes, {off64}, "input_ptr");

  /* len64 = zext len to i64 */
  Value *len64 = Builder.CreateZExt(len, i64, "len64");

  /* __coqui_fuzz_execute(input_ptr, len64) */
  FunctionType *ExecType = FunctionType::get(i32, {i8p, i64}, false);
  Builder.CreateCall(ExecType, User, {inputPtr, len64});

  /* PHASE_BUCKETING = 4 (classify + OR into virgin fused into
   * coverage_evaluate; single phase marker covers both). */
  Builder.CreateCall(SetPhase, {tid, ConstantInt::get(i8, 4)});

  /* cov = __coqui_cov_base(); cov_size = __coqui_cov_map_size(); */
  Value *cov     = Builder.CreateCall(CovBase, {}, "cov");
  Value *covSize = Builder.CreateCall(CovSize, {}, "cov_size");

  /* virgin_map = load @__coqui_virgin_map_ptr */
  Value *virgin  = Builder.CreateLoad(i8p, VirginMapPtr, "virgin_map");

  /* found_new = __coqui_coverage_evaluate(cov, virgin, cov_size) */
  Value *foundNew = Builder.CreateCall(CoverageEval, {cov, virgin, covSize}, "found_new");

  /* PHASE_VIRGIN_CMP = 5 (stays as a phase marker for any monitors that
   * watched for it, even though the work is now fused above). */
  Builder.CreateCall(SetPhase, {tid, ConstantInt::get(i8, 5)});

  /* Store novelty bit: novelty[tid >> 5] |= (found_new ? 1 : 0) << (tid & 31).
   * Equivalent to atomic_or when a warp contributes multiple novel threads,
   * but these are all within one 32-bit word of the bitmap; we use a plain
   * atomic add via inline asm, or the simpler approach of an atomic or on
   * the 32-bit slot.
   *
   * For simplicity we do a per-thread atomicOr on the 32-bit slot here —
   * same semantics as the old virgin_compare path (which also did
   * atomic_fetch_or_explicit on a 32-bit word of the novelty bitmap).
   * NVPTX lowers atomicrmw or on addrspace(0) (generic) to atom.or.b32. */
  Value *tid32      = tid; /* already i32 */
  Value *word_i     = Builder.CreateLShr(tid32, ConstantInt::get(i32, 5), "word_i");
  Value *bit_i      = Builder.CreateAnd(tid32, ConstantInt::get(i32, 31), "bit_i");
  Value *bitMask32  = Builder.CreateShl(
      ConstantInt::get(i32, 1), bit_i, "bit_mask");
  /* noveltyArg is u8* but bitmap is in 32-bit words — compute byte offset as
   * word_i * 4. */
  Value *wordByteOff = Builder.CreateMul(word_i, ConstantInt::get(i32, 4),
                                          "novelty_byte_off");
  Value *novSlot     = Builder.CreateGEP(i8, noveltyArg, wordByteOff, "novelty_word");

  /* If found_new (i32 0 or 1) is true, atomicOr the word with (1 << bit_i).
   * Do the branchless form: atomic or with (found_new ? mask : 0). */
  Value *foundNz = Builder.CreateICmpNE(foundNew, ConstantInt::get(i32, 0), "found_nz");
  Value *orVal = Builder.CreateSelect(foundNz, bitMask32,
                                      ConstantInt::get(i32, 0), "or_val");
  Builder.CreateAtomicRMW(AtomicRMWInst::Or, novSlot, orVal,
                          MaybeAlign(Align(4)),
                          AtomicOrdering::Monotonic);

  /* Compute crash_sig = FNV-1a over the (now-classified + zeroed) cov map.
   * NOTE: coverage_evaluate zeros the cov map as it walks it — so by the time
   * this runs, the cov_map is all zero and trace_sig would be the FNV offset
   * basis for every thread.
   *
   * This is intentional: at this point in the kernel the per-thread map has
   * already been classified + OR'd into the global map + zeroed. For clean
   * (non-crashing) executions the sig is only used for crash dedup, and
   * non-crashing threads never go through the crash dedup path (coqui
   * checks status.asan_error != 0 before consulting crash_sig). The
   * interesting sig values are the ones stamped by asan_report() in
   * coqui_asan.c, which runs BEFORE the map is cleared because the ASan
   * trap path is mid-execution. */
  Value *sig = Builder.CreateCall(TraceSig, {cov, covSize}, "sig");

  /* Store sig into status[tid].crash_sig.
   *
   * Struct layout (must match coqui_runtime.h and afl-fuzz-coqui.h):
   *   u8 phase; u8 signal; u8 asan_error; u8 ubsan_fatal; u32 crash_sig; u64 _reserved1;
   * crash_sig lives at byte offset 4 of the 16-byte slot.
   */
  Value *statusTid64 = Builder.CreateZExt(tid, i64, "tid64");
  Value *slotBase    = Builder.CreateGEP(
      i8, statusArg,
      {Builder.CreateMul(statusTid64, ConstantInt::get(i64, 16))},
      "status_slot");
  Value *sigSlot     = Builder.CreateGEP(
      i8, slotBase, {ConstantInt::get(i64, 4)}, "sig_slot");
  Builder.CreateStore(sig, sigSlot);

  /* PHASE_COMPLETE = 6 */
  Builder.CreateCall(SetPhase, {tid, ConstantInt::get(i8, 6)});

  Builder.CreateBr(ExitBB);

  Builder.SetInsertPoint(ExitBB);
  Builder.CreateRetVoid();

  /* 7. nvvm.annotations: !{ptr @__coqui_fuzz_kernel, !"kernel", i32 1} */
  NamedMDNode *Annots = M.getOrInsertNamedMetadata("nvvm.annotations");
  Metadata *Ops[] = {
    ValueAsMetadata::get(Kernel),
    MDString::get(C, "kernel"),
    ConstantAsMetadata::get(ConstantInt::get(i32, 1)),
  };
  Annots->addOperand(MDNode::get(C, Ops));

  return true;
}

} // namespace coqui
