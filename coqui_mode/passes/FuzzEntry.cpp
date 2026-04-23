/*
 * FuzzEntry.cpp --- rename LLVMFuzzerTestOneInput, create kernel entry.
 *
 * Per coqui internals design §7.1: emits __coqui_fuzz_kernel with the
 * 5-argument signature (input_bytes, offsets, input_lens, novelty, status).
 *
 * Kernel body calls __coqui_fuzz_execute per-thread, then runs
 * post-execution bucketing + virgin compare via runtime helpers.
 *
 * Adds nvvm.annotations to mark __coqui_fuzz_kernel as .entry.
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
      "every coqui mode target must define it");
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

  FunctionCallee GetTid = M.getOrInsertFunction("__coqui_fuzz_tid", TidRet);
  FunctionCallee SetPhase = M.getOrInsertFunction(
    "__coqui_status_set_phase",
    FunctionType::get(voidT, {i32, i8}, false));
  FunctionCallee MemoryInit = M.getOrInsertFunction("__coqui_memory_init", VoidNoArg);
  FunctionCallee CovBase = M.getOrInsertFunction("__coqui_cov_base", PtrNoArg);
  /* Folded classify-and-sig: single pass over the 64 KB cov_map that
   * classifies every byte AND returns a 32-bit FNV-1a hash so the host
   * can dedup crash-verifies by signature. */
  FunctionCallee ClassifyAndSig = M.getOrInsertFunction(
    "__coqui_classify_counts_and_sig",
    FunctionType::get(i32, {i8p}, false));
  FunctionCallee VirginCmp = M.getOrInsertFunction(
    "__coqui_virgin_compare_and_flag",
    FunctionType::get(voidT, {i8p, i8p, i8p}, false));

  /* 4. Declare the virgin_map as an extern global [65536 x i8] */
  ArrayType *VirginArrTy = ArrayType::get(i8, 65536);
  GlobalVariable *VirginMap = M.getGlobalVariable("__coqui_virgin_map", true);
  if (!VirginMap) {
    VirginMap = new GlobalVariable(
      M, VirginArrTy, /*isConstant*/false, GlobalValue::ExternalLinkage,
      nullptr, "__coqui_virgin_map");
  }

  /* 4b. Per-phase kernel timing accumulators: [init, exec, classify,
   * virgin, total] u64 cycles. Each thread atomic-adds its phase cycles
   * into the appropriate slot; host reads + zeros per batch and reports
   * averages. Clock source is clock64() — SM clock register, 1 cycle per
   * tick, runs at GPU clock rate (~1.5 GHz on TITAN RTX). */
  ArrayType *TimingArrTy = ArrayType::get(i64, 5);
  GlobalVariable *KernelTiming = M.getGlobalVariable("__coqui_kernel_timing", true);
  if (!KernelTiming) {
    KernelTiming = new GlobalVariable(
      M, TimingArrTy, /*isConstant*/false, GlobalValue::ExternalLinkage,
      ConstantAggregateZero::get(TimingArrTy), "__coqui_kernel_timing");
  }
  FunctionCallee Clock64 = M.getOrInsertFunction(
      "llvm.nvvm.read.ptx.sreg.clock64",
      FunctionType::get(i64, false));

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

  /* clk_a: start of instrumented body */
  Value *clkA = Builder.CreateCall(Clock64, {}, "clk_a");

  /* __coqui_memory_init() */
  Builder.CreateCall(MemoryInit, {});

  /* clk_b: after memory_init */
  Value *clkB = Builder.CreateCall(Clock64, {}, "clk_b");

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

  /* clk_c: after fuzz_execute (user harness) */
  Value *clkC = Builder.CreateCall(Clock64, {}, "clk_c");

  /* PHASE_BUCKETING = 4 */
  Builder.CreateCall(SetPhase, {tid, ConstantInt::get(i8, 4)});

  /* cov = __coqui_cov_base() */
  Value *cov = Builder.CreateCall(CovBase, {}, "cov");
  /* sig = __coqui_classify_counts_and_sig(cov) */
  Value *sig = Builder.CreateCall(ClassifyAndSig, {cov}, "sig");

  /* clk_d: after classify_counts_and_sig */
  Value *clkD = Builder.CreateCall(Clock64, {}, "clk_d");

  /* PHASE_VIRGIN_CMP = 5 */
  Builder.CreateCall(SetPhase, {tid, ConstantInt::get(i8, 5)});

  /* With opaque pointers, VirginMap (ptr to [65536 x i8]) is already
     an i8* — no bitcast needed; pass directly as i8p. */
  Builder.CreateCall(VirginCmp, {cov, VirginMap, noveltyArg});

  /* clk_e: after virgin_compare */
  Value *clkE = Builder.CreateCall(Clock64, {}, "clk_e");

  /* Atomically accumulate per-phase cycle deltas into __coqui_kernel_timing.
   * Slot 0: memory_init, 1: fuzz_execute, 2: classify+sig, 3: virgin_compare,
   * 4: total (clk_e - clk_a). Using atomicrmw add with monotonic ordering —
   * lowers to atom.add.u64 on global memory. Non-crashing threads only; an
   * ASan-aborted thread won't reach here (calls __coqui_exit earlier). */
  auto emitTimingAdd = [&](unsigned slot, Value *delta) {
    Value *slotPtr = Builder.CreateGEP(
        TimingArrTy, KernelTiming,
        {ConstantInt::get(i32, 0), ConstantInt::get(i32, slot)},
        "timing_slot");
    Builder.CreateAtomicRMW(AtomicRMWInst::Add, slotPtr, delta,
                            MaybeAlign(8), AtomicOrdering::Monotonic);
  };
  emitTimingAdd(0, Builder.CreateSub(clkB, clkA, "d_init"));
  emitTimingAdd(1, Builder.CreateSub(clkC, clkB, "d_exec"));
  emitTimingAdd(2, Builder.CreateSub(clkD, clkC, "d_classify"));
  emitTimingAdd(3, Builder.CreateSub(clkE, clkD, "d_virgin"));
  emitTimingAdd(4, Builder.CreateSub(clkE, clkA, "d_total"));

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
