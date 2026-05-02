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
#include "llvm/Support/Alignment.h"
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
  /* Fused classify + hash + virgin-compare + zero: single pass over
   * the coverage map instead of two separate traversals. */
  FunctionCallee ClassifyVirginFused = M.getOrInsertFunction(
    "__coqui_classify_virgin_fused",
    FunctionType::get(i32, {i8p, i8p, i8p}, false));

  /* 4. Declare the virgin_map as an extern global [65536 x i8].
   *
   * Alignment must be at least 8 bytes. __coqui_virgin_compare_and_flag
   * casts this pointer to `_Atomic u64 *` and issues atom.or.b64 through
   * it; on sm_75+ any 64-bit atomic at a non-8-byte-aligned address
   * produces CUDA_ERROR_MISALIGNED_ADDRESS. Without an explicit Align(8)
   * here, LLVM emits `.align 1 .b8 __coqui_virgin_map[65536]` in PTX and
   * the CUDA driver is not required to over-align the symbol even when
   * allocating a 64 KB chunk. */
  ArrayType *VirginArrTy = ArrayType::get(i8, 65536);
  GlobalVariable *VirginMap = M.getGlobalVariable("__coqui_virgin_map", true);
  if (!VirginMap) {
    VirginMap = new GlobalVariable(
      M, VirginArrTy, /*isConstant*/false, GlobalValue::ExternalLinkage,
      nullptr, "__coqui_virgin_map");
  }
  /* Unconditionally enforce 8-byte alignment — covers both the freshly-
   * created declaration and any pre-existing one that may have been
   * created without an alignment hint. */
  if (VirginMap->getAlign().valueOrOne() < Align(8))
    VirginMap->setAlignment(Align(8));

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

  /* 5b. Slab infrastructure — only emitted when the runtime module exports
   *     __coqui_slab_setup. The build_coqui_support.sh script always links
   *     coqui_slab.c into the runtime bitcode, so this is effectively a
   *     "slab runtime present" gate. Without the gate, targets that haven't
   *     been rebuilt with the new runtime would fail to find the symbol.
   *
   *     All 13 buckets share a single 104-byte __shared__ strip per block
   *     (13 * 8B = 104). The init runs only on threadIdx.x == 0 followed by
   *     a __syncthreads() barrier so every thread in the block sees the
   *     zeroed buckets before the first malloc.
   */
  bool SlabEnabled = M.getFunction("__coqui_slab_setup") != nullptr;
  if (SlabEnabled) {
    /* __shared__ bucket strip (addrspace 3). 13 buckets * 8B = 104B. */
    ArrayType *BucketArrayTy = ArrayType::get(i8, 13 * 8);
    GlobalVariable *SharedBuckets =
        M.getGlobalVariable("__coqui_slab_shared_buckets", true);
    if (!SharedBuckets) {
      SharedBuckets = new GlobalVariable(
          M, BucketArrayTy, /*isConstant=*/false,
          GlobalValue::ExternalLinkage,
          UndefValue::get(BucketArrayTy),
          "__coqui_slab_shared_buckets",
          /*InsertBefore=*/nullptr,
          GlobalValue::NotThreadLocal,
          /*AddressSpace=*/3);
    }
    SharedBuckets->setAlignment(Align(8));

    /* __shared__ per-block allocation counter (addrspace 3). 4B. */
    GlobalVariable *SharedBlockNext =
        M.getGlobalVariable("__coqui_slab_shared_block_next", true);
    if (!SharedBlockNext) {
      SharedBlockNext = new GlobalVariable(
          M, i32, /*isConstant=*/false,
          GlobalValue::ExternalLinkage,
          UndefValue::get(i32),
          "__coqui_slab_shared_block_next",
          /*InsertBefore=*/nullptr,
          GlobalValue::NotThreadLocal,
          /*AddressSpace=*/3);
    }
    SharedBlockNext->setAlignment(Align(4));

    /* __coqui_slab_bucket_base() — returns a generic-space pointer to the
     * per-block shared bucket strip. noinline prevents the NVPTX backend
     * from constant-folding the addrspacecast across blocks. The strip is
     * per-block via the .shared memory semantics, not per-call.
     *
     * The runtime's coqui_slab.c declares this `extern` so it already
     * exists as a declaration at pass time. We attach a body if it doesn't
     * have one yet; callers within the runtime resolve to our definition. */
    {
      FunctionType *FnTy = FunctionType::get(i8p, false);
      Function *F = cast<Function>(
          M.getOrInsertFunction("__coqui_slab_bucket_base", FnTy).getCallee());
      if (F->isDeclaration()) {
        F->setLinkage(GlobalValue::InternalLinkage);
        F->addFnAttr(Attribute::NoInline);
        F->setDoesNotThrow();
        BasicBlock *BB = BasicBlock::Create(C, "entry", F);
        IRBuilder<> B(BB);
        B.CreateRet(B.CreateAddrSpaceCast(SharedBuckets, i8p, "buckets_gen"));
      }
    }

    /* __coqui_slab_block_next() — same pattern for the per-block counter. */
    {
      FunctionType *FnTy = FunctionType::get(i8p, false);
      Function *F = cast<Function>(
          M.getOrInsertFunction("__coqui_slab_block_next", FnTy).getCallee());
      if (F->isDeclaration()) {
        F->setLinkage(GlobalValue::InternalLinkage);
        F->addFnAttr(Attribute::NoInline);
        F->setDoesNotThrow();
        BasicBlock *BB = BasicBlock::Create(C, "entry", F);
        IRBuilder<> B(BB);
        B.CreateRet(B.CreateAddrSpaceCast(SharedBlockNext, i8p, "block_next_gen"));
      }
    }
  }

  /* 6. Emit kernel body */
  BasicBlock *EntryBB = BasicBlock::Create(C, "entry", Kernel);
  BasicBlock *SlabInitBB = SlabEnabled
      ? BasicBlock::Create(C, "slab_init", Kernel)
      : nullptr;
  BasicBlock *SlabBarrierBB = SlabEnabled
      ? BasicBlock::Create(C, "slab_barrier", Kernel)
      : nullptr;
  BasicBlock *RunBB   = BasicBlock::Create(C, "run",   Kernel);
  BasicBlock *ExitBB  = BasicBlock::Create(C, "exit",  Kernel);

  Builder.SetInsertPoint(EntryBB);

  /* Store status pointer into global so runtime can find it */
  Builder.CreateStore(statusArg, StatusArrayPtr);

  if (SlabEnabled) {
    /* Slab setup: every thread calls it (idempotent; writes trivially
     * idempotent globals — ctrl_slabs is the same value for all threads
     * and the asan register call sets two function pointers). Do this
     * BEFORE the barrier so __coqui_asan_register_slab has run before
     * any ASan heap allocation hits the slab fall-through path. */
    FunctionCallee SlabSetup =
        M.getOrInsertFunction("__coqui_slab_setup", VoidNoArg);
    Builder.CreateCall(SlabSetup, {});

    /* if (threadIdx.x == 0) __coqui_slab_init_block() */
    FunctionCallee TidXFn = M.getOrInsertFunction(
        "llvm.nvvm.read.ptx.sreg.tid.x",
        FunctionType::get(i32, false));
    Value *TidX = Builder.CreateCall(TidXFn, {}, "tidx");
    Value *IsT0 = Builder.CreateICmpEQ(TidX, ConstantInt::get(i32, 0), "is_t0");
    Builder.CreateCondBr(IsT0, SlabInitBB, SlabBarrierBB);

    Builder.SetInsertPoint(SlabInitBB);
    FunctionCallee SlabInit =
        M.getOrInsertFunction("__coqui_slab_init_block", VoidNoArg);
    Builder.CreateCall(SlabInit, {});
    Builder.CreateBr(SlabBarrierBB);

    /* __syncthreads() on the merge block. All 128 threads per block must
     * reach this barrier (see 2026-03-25-shared-slab-pool-allocator-design.md
     * §"Bucket Initialization and Barrier Placement") — placed BEFORE the
     * per-thread len==0 early-exit so partial-block exits don't deadlock. */
    Builder.SetInsertPoint(SlabBarrierBB);
    FunctionCallee Barrier =
        M.getOrInsertFunction("llvm.nvvm.barrier0", VoidNoArg);
    Builder.CreateCall(Barrier, {});
  }

  /* tid = __coqui_fuzz_tid(). When slab init ran, IRBuilder is still
   * pointing at SlabBarrierBB (post-syncthreads); when it didn't, we're
   * in EntryBB. Either way, all threads converge to this point so the
   * len==0 early-exit runs AFTER the syncthreads barrier — avoiding the
   * partial-block-exit deadlock documented in the slab design spec. */
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

  /* GPU-side mutation — fixed-size per-thread slots.
   *
   * If __coqui_gpu_mutate_enabled is set (AFL_COQUI_GPU_MUTATE=1), each
   * thread applies havoc mutations to its input buffer BEFORE calling the
   * harness. The mutation operates in-place on the global memory input
   * buffer. `len` is stored to a stack alloca so the mutation function
   * can modify it via pointer; afterwards we reload the (potentially
   * changed) length for the harness call.
   *
   * max_len: read from the device-global __coqui_mutate_max_input_size
   * (set by the host at init). With fixed-size per-thread slots, each
   * thread owns max_input_size bytes, so grow mutations (clone, insert)
   * can safely expand into the slot padding without overwriting adjacent
   * threads' data. */
  {
    /* Declare __coqui_mutate_input(u32 tid, u8* buf, u32* len_ptr, u32 max_len) */
    FunctionCallee MutateInput = M.getOrInsertFunction(
        "__coqui_mutate_input",
        FunctionType::get(voidT, {i32, i8p, i8p, i32}, false));

    /* Read max_input_size from device global (set by host at coqui_init). */
    GlobalVariable *MaxInputSize =
        M.getGlobalVariable("__coqui_mutate_max_input_size", true);
    if (!MaxInputSize) {
      MaxInputSize = new GlobalVariable(
          M, i32, /*isConstant=*/false, GlobalValue::ExternalLinkage,
          nullptr, "__coqui_mutate_max_input_size");
    }
    Value *maxLen = Builder.CreateLoad(i32, MaxInputSize, "max_input_size");

    /* Alloca for len so mutation can modify it via pointer. */
    IRBuilder<> AllocaBuilder(&Kernel->getEntryBlock(),
                               Kernel->getEntryBlock().getFirstInsertionPt());
    Value *lenSlot = AllocaBuilder.CreateAlloca(i32, nullptr, "mutate_len_slot");

    /* Store current len, call mutate, reload.
     * max_len = max_input_size: grow mutations expand within the fixed slot. */
    Builder.CreateStore(len, lenSlot);
    Builder.CreateCall(MutateInput, {tid, inputPtr, lenSlot, maxLen});
    len = Builder.CreateLoad(i32, lenSlot, "len_post_mutate");

    /* Write the mutated length back to the device lens array so the host
     * can D2H copy it after kernel completion. Without this, the host
     * reads the original (pre-mutation) length and saves the wrong bytes
     * when exporting novel inputs. */
    Builder.CreateStore(len, lensPtr);
  }

  /* Per-thread budget poison init (Exp #51). Stamps clock64() into
   * __coqui_thread_budget_start[tid]. Placed AFTER GPU mutation so the
   * budget clock only polices harness execution time, not mutation time.
   * The runtime helper is always_inline and short-circuits when the host
   * hasn't enabled the budget (cycles_cap == 0), so production-default
   * cost is a single global load + branch-not-taken per thread. */
  FunctionCallee BudgetInit = M.getOrInsertFunction(
      "__coqui_thread_budget_init", VoidNoArg);
  Builder.CreateCall(BudgetInit, {});

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
  /* Fused: classify + hash + virgin-compare + zero in one pass. */
  Value *sig = Builder.CreateCall(ClassifyVirginFused,
                                  {cov, VirginMap, noveltyArg}, "sig");

  Value *clkD = Builder.CreateCall(Clock64, {}, "clk_d");
  Value *clkE = clkD;

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
