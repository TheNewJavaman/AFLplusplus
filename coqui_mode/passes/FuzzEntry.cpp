/*
 * FuzzEntry.cpp --- rename LLVMFuzzerTestOneInput, create TWO kernel entries.
 *
 * Split-kernel model (iter 12):
 *   - __coqui_fuzz_kernel     : stage A — user harness only.
 *   - __coqui_fuzz_kernel_cov : stage B — classify + virgin compare + sig.
 *
 * Stage A exits after the user code returns. ASan-trapped lanes exit early.
 * Stage B launches AFTER stage A on the same CUDA stream (host-side
 * serialization), reads each thread's cov_pool slot, classifies in place,
 * runs the warp-OR virgin compare, writes novelty bitmap + crash_sig.
 * Because every thread of stage B starts fresh, every warp is full and the
 * warp-OR atomic-reduction path is always taken (no partial-warp fallback
 * due to ASan exits).
 *
 * Empty-slot threads (lens[tid]==0) still exit at stage A entry; stage B
 * treats those slots the same as any zero-filled cov_pool word — the
 * zero-skip fast path in __coqui_classify_counts_and_sig and
 * __coqui_virgin_compare_and_flag makes the walk cheap.
 *
 * Adds nvvm.annotations entries for BOTH kernels.
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

/* Emit the stage-B coverage-evaluation kernel.
 *
 * Signature: void __coqui_fuzz_kernel_cov(i8* lens, i8* novelty, i8* status)
 *
 * Body (per thread):
 *   sig = __coqui_classify_counts_and_sig(__coqui_cov_base());
 *   __coqui_virgin_compare_and_flag(cov_base, virgin_map, novelty);
 *   status[tid].crash_sig = sig;
 *
 * No empty-slot fast exit: even empty-slot threads run the walk, because
 * their cov_pool slot is already zeroed (either by zero-on-read from the
 * previous batch or by the initial cuMemsetD8). Zero-word skip in the
 * runtime makes the walk ~free for zero-filled slots, and keeping every
 * thread live means every warp is full — the warp-OR fast path is always
 * taken. This is the whole point of the split.
 */
static void emitCoverageKernel(Module &M) {
  LLVMContext &C = M.getContext();
  IRBuilder<> Builder(C);

  Type *i8   = Type::getInt8Ty(C);
  Type *i32  = Type::getInt32Ty(C);
  Type *i64  = Type::getInt64Ty(C);
  Type *i8p  = PointerType::get(C, 0);
  Type *voidT = Type::getVoidTy(C);

  /* Kernel type: (lens, novelty, status) */
  FunctionType *KernelType = FunctionType::get(
    voidT, {i8p, i8p, i8p}, false);
  Function *Kernel = Function::Create(
    KernelType, GlobalValue::ExternalLinkage,
    "__coqui_fuzz_kernel_cov", &M);

  auto argIt = Kernel->arg_begin();
  Value *lensArg    = &*argIt++; lensArg->setName("lens");
  Value *noveltyArg = &*argIt++; noveltyArg->setName("novelty");
  Value *statusArg  = &*argIt++; statusArg->setName("status");

  FunctionType *TidRet    = FunctionType::get(i32, false);
  FunctionType *PtrNoArg  = FunctionType::get(i8p, false);

  FunctionCallee GetTid = M.getOrInsertFunction("__coqui_fuzz_tid", TidRet);
  FunctionCallee CovBase = M.getOrInsertFunction("__coqui_cov_base", PtrNoArg);
  FunctionCallee ClassifyAndSig = M.getOrInsertFunction(
    "__coqui_classify_counts_and_sig",
    FunctionType::get(i32, {i8p}, false));
  FunctionCallee VirginCmp = M.getOrInsertFunction(
    "__coqui_virgin_compare_and_flag",
    FunctionType::get(voidT, {i8p, i8p, i8p}, false));

  /* virgin_map global (already declared by stage A; getGlobalVariable finds it) */
  ArrayType *VirginArrTy = ArrayType::get(i8, 65536);
  GlobalVariable *VirginMap = M.getGlobalVariable("__coqui_virgin_map", true);
  if (!VirginMap) {
    VirginMap = new GlobalVariable(
      M, VirginArrTy, /*isConstant*/false, GlobalValue::ExternalLinkage,
      nullptr, "__coqui_virgin_map");
  }

  /* status_array pointer global (re-used; stage A writes, we overwrite). */
  GlobalVariable *StatusArrayPtr = M.getGlobalVariable("__coqui_status_array", true);
  if (!StatusArrayPtr) {
    StatusArrayPtr = new GlobalVariable(
      M, i8p, /*isConstant*/false, GlobalValue::ExternalLinkage,
      Constant::getNullValue(i8p), "__coqui_status_array");
  }

  BasicBlock *EntryBB = BasicBlock::Create(C, "entry", Kernel);
  Builder.SetInsertPoint(EntryBB);

  /* Rebind status pointer: stage B may run on a fresh launch with no prior
   * stage-A write in this kernel instance (CUDA may or may not preserve the
   * value in the global across launches, but rebinding is cheap + safe). */
  Builder.CreateStore(statusArg, StatusArrayPtr);

  /* tid = __coqui_fuzz_tid() */
  Value *tid = Builder.CreateCall(GetTid, {}, "tid");

  /* cov = __coqui_cov_base() — returns cov_pool_base + tid * 64KB */
  Value *cov = Builder.CreateCall(CovBase, {}, "cov");

  /* sig = __coqui_classify_counts_and_sig(cov) */
  Value *sig = Builder.CreateCall(ClassifyAndSig, {cov}, "sig");

  /* __coqui_virgin_compare_and_flag(cov, virgin, novelty) */
  Builder.CreateCall(VirginCmp, {cov, VirginMap, noveltyArg});

  /* status[tid].crash_sig = sig  (byte offset 4 in the 16-byte struct) */
  Value *tid64 = Builder.CreateZExt(tid, i64, "tid64");
  Value *slotBase = Builder.CreateGEP(
      i8, statusArg,
      {Builder.CreateMul(tid64, ConstantInt::get(i64, 16))},
      "status_slot");
  Value *sigSlot = Builder.CreateGEP(
      i8, slotBase, {ConstantInt::get(i64, 4)}, "sig_slot");
  Builder.CreateStore(sig, sigSlot);

  /* Also stamp phase=COMPLETE so status readers see done. */
  Value *phaseSlot = slotBase;  /* offset 0 */
  Builder.CreateStore(ConstantInt::get(i8, 6), phaseSlot);

  /* lensArg currently unused; silence DCE concerns via a read. We explicitly
   * keep `lens` in the signature in case a future refinement wants to skip
   * empty slots entirely (would require dropping fast-path-zero assumption
   * in __coqui_virgin_compare, so we don't do it today). */
  (void)lensArg;

  Builder.CreateRetVoid();

  /* nvvm.annotations: mark as kernel */
  NamedMDNode *Annots = M.getOrInsertNamedMetadata("nvvm.annotations");
  Metadata *Ops[] = {
    ValueAsMetadata::get(Kernel),
    MDString::get(C, "kernel"),
    ConstantAsMetadata::get(ConstantInt::get(i32, 1)),
  };
  Annots->addOperand(MDNode::get(C, Ops));
}

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

  /* 2. Build stage A kernel. Signature unchanged from pre-split to keep
   *    the host launch path identical for kernel A. */
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

  /* Unused here; stage B uses novelty. Keep the kernel signature stable
   * for host compatibility. */
  (void)noveltyArg;

  /* 3. Declare / look up helpers */
  FunctionType *VoidNoArg = FunctionType::get(voidT, false);
  FunctionType *TidRet    = FunctionType::get(i32, false);

  FunctionCallee GetTid = M.getOrInsertFunction("__coqui_fuzz_tid", TidRet);
  FunctionCallee SetPhase = M.getOrInsertFunction(
    "__coqui_status_set_phase",
    FunctionType::get(voidT, {i32, i8}, false));
  FunctionCallee MemoryInit = M.getOrInsertFunction("__coqui_memory_init", VoidNoArg);

  /* 4. Declare the virgin_map as an extern global [65536 x i8] (stage B uses it) */
  ArrayType *VirginArrTy = ArrayType::get(i8, 65536);
  GlobalVariable *VirginMap = M.getGlobalVariable("__coqui_virgin_map", true);
  if (!VirginMap) {
    VirginMap = new GlobalVariable(
      M, VirginArrTy, /*isConstant*/false, GlobalValue::ExternalLinkage,
      nullptr, "__coqui_virgin_map");
  }

  /* 5. __coqui_status_array pointer global (runtime reads this) */
  GlobalVariable *StatusArrayPtr = M.getGlobalVariable("__coqui_status_array", true);
  if (!StatusArrayPtr) {
    StatusArrayPtr = new GlobalVariable(
      M, i8p, /*isConstant*/false, GlobalValue::ExternalLinkage,
      Constant::getNullValue(i8p), "__coqui_status_array");
  }

  /* 6. Emit stage A kernel body */
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

  /* PHASE_COMPLETE = 6. Note: the cov classify + virgin compare have moved
   * to stage B; this thread is done. The phase is tracked by stage-B
   * overriding the status slot; leaving COMPLETE here is a hint only. */
  Builder.CreateCall(SetPhase, {tid, ConstantInt::get(i8, 6)});

  Builder.CreateBr(ExitBB);

  Builder.SetInsertPoint(ExitBB);
  Builder.CreateRetVoid();

  /* 7. nvvm.annotations for stage A */
  NamedMDNode *Annots = M.getOrInsertNamedMetadata("nvvm.annotations");
  Metadata *Ops[] = {
    ValueAsMetadata::get(Kernel),
    MDString::get(C, "kernel"),
    ConstantAsMetadata::get(ConstantInt::get(i32, 1)),
  };
  Annots->addOperand(MDNode::get(C, Ops));

  /* 8. Emit stage B kernel */
  emitCoverageKernel(M);

  return true;
}

} // namespace coqui
