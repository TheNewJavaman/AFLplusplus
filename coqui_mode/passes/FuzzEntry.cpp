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
    voidT, {i8p, i8p, i8p, i8p, i8p, i8p, i8p}, false);
  Function *Kernel = Function::Create(
    KernelType, GlobalValue::ExternalLinkage,
    "__coqui_fuzz_kernel", &M);

  /* Name the args for readability */
  auto argIt = Kernel->arg_begin();
  Value *inputBytes      = &*argIt++; inputBytes->setName("input_bytes");
  Value *offsetsArg      = &*argIt++; offsetsArg->setName("offsets");
  Value *lensArg         = &*argIt++; lensArg->setName("lens");
  Value *slotInfoArg     = &*argIt++; slotInfoArg->setName("slot_info");
  Value *noveltyArg      = &*argIt++; noveltyArg->setName("novelty");
  Value *statusArg       = &*argIt++; statusArg->setName("status");
  Value *reportedSlabArg = &*argIt++; reportedSlabArg->setName("reported_slab");
  (void)reportedSlabArg;  /* used by compact-report block in Task 2.8 */

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

  /* 4c. Havoc-path runtime externs: seed pool pointers, PRNG base, mutate fn.
   * These are defined in coqui_mutate.c and bound to the kernel module by the
   * host via cuModuleGetGlobal at module-load time. */
  PointerType *i32p = PointerType::get(C, 0);   /* opaque i32 pointer */
  PointerType *i64p = PointerType::get(C, 0);   /* opaque i64 pointer (prng state) */

  GlobalVariable *SeedPoolBase = M.getGlobalVariable("__coqui_seed_pool_base", true);
  if (!SeedPoolBase) {
    SeedPoolBase = new GlobalVariable(
      M, i8p, /*isConstant*/false, GlobalValue::ExternalLinkage,
      nullptr, "__coqui_seed_pool_base");
  }
  GlobalVariable *SeedPoolOffsets = M.getGlobalVariable("__coqui_seed_pool_offsets", true);
  if (!SeedPoolOffsets) {
    SeedPoolOffsets = new GlobalVariable(
      M, i32p, /*isConstant*/false, GlobalValue::ExternalLinkage,
      nullptr, "__coqui_seed_pool_offsets");
  }
  GlobalVariable *SeedPoolLens = M.getGlobalVariable("__coqui_seed_pool_lens", true);
  if (!SeedPoolLens) {
    SeedPoolLens = new GlobalVariable(
      M, i32p, /*isConstant*/false, GlobalValue::ExternalLinkage,
      nullptr, "__coqui_seed_pool_lens");
  }
  GlobalVariable *PrngBaseG = M.getGlobalVariable("__coqui_prng_base", true);
  if (!PrngBaseG) {
    PrngBaseG = new GlobalVariable(
      M, i64, /*isConstant*/false, GlobalValue::ExternalLinkage,
      nullptr, "__coqui_prng_base");
  }

  /* __coqui_havoc_mutate: u32(u8*, u32, u32, u32, u64*) */
  FunctionType *MutateTy = FunctionType::get(
      i32, {i8p, i32, i32, i32, i64p}, false);
  FunctionCallee MutateFn = M.getOrInsertFunction("__coqui_havoc_mutate", MutateTy);

  /* llvm.memcpy intrinsic --- p0.p0.i64 variant */
  FunctionType *MemcpyTy = FunctionType::get(
      voidT, {i8p, i8p, i64, Type::getInt1Ty(C)}, false);
  FunctionCallee MemcpyFn = M.getOrInsertFunction("llvm.memcpy.p0.p0.i64", MemcpyTy);

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

  /* slot = slot_info[tid]; flag = slot >> 24; seed_idx = slot & 0xFFFFFF */
  Value *slotPtr = Builder.CreateGEP(i32, slotInfoArg, {tid}, "slot_tid");
  Value *slot    = Builder.CreateLoad(i32, slotPtr, "slot");
  Value *flagV   = Builder.CreateLShr(slot, ConstantInt::get(i32, 24), "flag");
  Value *idxV    = Builder.CreateAnd(slot, ConstantInt::get(i32, 0xFFFFFF), "seed_idx");

  /* Create flag-dispatch blocks. */
  BasicBlock *PremutBB        = BasicBlock::Create(C, "premut", Kernel);
  BasicBlock *HavocBB         = BasicBlock::Create(C, "havoc", Kernel);
  BasicBlock *DispatchHavocBB = BasicBlock::Create(C, "dispatch_havoc", Kernel);

  /* if flag == 0 -> premut else -> dispatch_havoc */
  Value *isPremut = Builder.CreateICmpEQ(flagV, ConstantInt::get(i32, 0), "is_premut");
  Builder.CreateCondBr(isPremut, PremutBB, DispatchHavocBB);

  /* DispatchHavocBB: if flag == 1 -> havoc else -> exit */
  Builder.SetInsertPoint(DispatchHavocBB);
  Value *isHavoc = Builder.CreateICmpEQ(flagV, ConstantInt::get(i32, 1), "is_havoc");
  Builder.CreateCondBr(isHavoc, HavocBB, ExitBB);

  /* PREMUT path — existing flag=0 logic. */
  Builder.SetInsertPoint(PremutBB);
  Value *pm_lensPtr = Builder.CreateGEP(i32, lensArg, {tid}, "pm_lens_tid");
  Value *pm_len     = Builder.CreateLoad(i32, pm_lensPtr, "pm_len");
  Value *pm_isEmpty = Builder.CreateICmpEQ(pm_len, ConstantInt::get(i32, 0));
  BasicBlock *PremutContBB = BasicBlock::Create(C, "premut_cont", Kernel);
  Builder.CreateCondBr(pm_isEmpty, ExitBB, PremutContBB);

  Builder.SetInsertPoint(PremutContBB);
  Value *pm_offPtr   = Builder.CreateGEP(i32, offsetsArg, {tid}, "pm_offsets_tid");
  Value *pm_off      = Builder.CreateLoad(i32, pm_offPtr, "pm_off");
  Value *pm_off64    = Builder.CreateZExt(pm_off, i64, "pm_off64");
  Value *pm_inputPtr = Builder.CreateGEP(i8, inputBytes, {pm_off64}, "pm_input_ptr");
  Value *pm_len64    = Builder.CreateZExt(pm_len, i64, "pm_len64");
  Builder.CreateBr(RunBB);

  /* HAVOC path --- look up seed in pool, copy to .local scratch, call
   * __coqui_havoc_mutate, join RunBB with (scratch, mutated_len). */
  Builder.SetInsertPoint(HavocBB);

  /* base_len = __coqui_seed_pool_lens[seed_idx] */
  Value *poolLensBase = Builder.CreateLoad(i32p, SeedPoolLens, "pool_lens_base");
  Value *baseLenPtr   = Builder.CreateGEP(i32, poolLensBase, {idxV}, "base_len_ptr");
  Value *baseLen      = Builder.CreateLoad(i32, baseLenPtr, "base_len");
  Value *baseLenIsZero = Builder.CreateICmpEQ(baseLen, ConstantInt::get(i32, 0));

  BasicBlock *HavocContBB = BasicBlock::Create(C, "havoc_cont", Kernel);
  Builder.CreateCondBr(baseLenIsZero, ExitBB, HavocContBB);

  Builder.SetInsertPoint(HavocContBB);

  /* scratch: alloca [4096 x i8] --- compile-time-sized .local stack array. */
  ArrayType *ScratchTy = ArrayType::get(i8, 4096);
  AllocaInst *Scratch = Builder.CreateAlloca(ScratchTy, nullptr, "scratch");
  Value *scratchPtr = Builder.CreateGEP(
      ScratchTy, Scratch,
      {ConstantInt::get(i32, 0), ConstantInt::get(i32, 0)},
      "scratch_ptr");

  /* src = seed_pool_base + seed_pool_offsets[seed_idx] */
  Value *poolOffsetsBase = Builder.CreateLoad(i32p, SeedPoolOffsets, "pool_off_base");
  Value *seedOffPtr = Builder.CreateGEP(i32, poolOffsetsBase, {idxV}, "seed_off_ptr");
  Value *seedOff    = Builder.CreateLoad(i32, seedOffPtr, "seed_off");
  Value *seedOff64  = Builder.CreateZExt(seedOff, i64, "seed_off64");
  Value *poolBasePtr = Builder.CreateLoad(i8p, SeedPoolBase, "pool_base_ptr");
  Value *srcPtr = Builder.CreateGEP(i8, poolBasePtr, {seedOff64}, "seed_src");

  /* memcpy(scratch, src, base_len, isVolatile=false) */
  Value *baseLen64 = Builder.CreateZExt(baseLen, i64, "base_len64");
  Builder.CreateCall(MemcpyFn,
                      {scratchPtr, srcPtr, baseLen64, ConstantInt::getFalse(C)});

  /* prng state: alloca u64 on stack; init to (prng_base XOR tid) */
  AllocaInst *PrngSlot = Builder.CreateAlloca(i64, nullptr, "prng_state");
  Value *prngBaseV = Builder.CreateLoad(i64, PrngBaseG, "prng_base_v");
  Value *tid64 = Builder.CreateZExt(tid, i64, "tid64_prng");
  Value *prngInit = Builder.CreateXor(prngBaseV, tid64, "prng_init");
  Builder.CreateStore(prngInit, PrngSlot);

  /* mutated_len = __coqui_havoc_mutate(scratch, base_len, 4096, seed_idx, &prng) */
  Value *maxLen = ConstantInt::get(i32, 4096);
  Value *mutLen = Builder.CreateCall(MutateFn,
      {scratchPtr, baseLen, maxLen, idxV, PrngSlot}, "mut_len");

  Value *mutLen64 = Builder.CreateZExt(mutLen, i64, "mut_len64");

  /* RunBB: existing body, but with PHIs at the top merging the two paths.
   * Two incomings: PremutContBB (flag=0 path) and HavocContBB (flag=1 path
   * after seed lookup, memcpy to .local scratch, and havoc mutate). */
  Builder.SetInsertPoint(RunBB);
  PHINode *inputPtrPhi = Builder.CreatePHI(i8p, 2, "input_ptr");
  PHINode *lenPhi      = Builder.CreatePHI(i64, 2, "len");
  inputPtrPhi->addIncoming(pm_inputPtr, PremutContBB);
  lenPhi->addIncoming(pm_len64, PremutContBB);
  inputPtrPhi->addIncoming(scratchPtr, HavocContBB);
  lenPhi->addIncoming(mutLen64, HavocContBB);

  /* Terminate HavocContBB by branching to RunBB (PHIs above merge both
   * predecessors). Insert at end of HavocContBB --- the builder was previously
   * pointing there while we emitted the havoc body. */
  Builder.SetInsertPoint(HavocContBB);
  Builder.CreateBr(RunBB);

  /* Return builder to RunBB for the remaining instrumented body. */
  Builder.SetInsertPoint(RunBB);

  /* PHASE_START = 1 */
  Builder.CreateCall(SetPhase, {tid, ConstantInt::get(i8, 1)});

  /* clk_a: start of instrumented body */
  Value *clkA = Builder.CreateCall(Clock64, {}, "clk_a");

  /* __coqui_memory_init() */
  Builder.CreateCall(MemoryInit, {});

  /* clk_b: after memory_init */
  Value *clkB = Builder.CreateCall(Clock64, {}, "clk_b");

  /* __coqui_fuzz_execute(input_ptr, len64) */
  FunctionType *ExecType = FunctionType::get(i32, {i8p, i64}, false);
  Builder.CreateCall(ExecType, User, {inputPtrPhi, lenPhi});

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
