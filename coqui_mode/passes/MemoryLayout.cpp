/*
 * MemoryLayout.cpp --- set up per-thread stack regions.
 *
 * Inserts three allocas at kernel entry:
 *   - cov_map  : 64 KB (aligned 8)
 *   - heap     : H bytes (runtime-configured)
 *   - shadow   : H/8 bytes (aligned 8)
 *
 * Stores their pointers into a tid-indexed addrspace(0) (.global) slot pool,
 * then emits helper accessor functions that load from the per-thread slot.
 *
 * Layout of __coqui_thread_slots[tid * SLOT_STRIDE + offset]:
 *   offset  0 ..  7  : cov_base   ptr (i8*, 8 bytes)
 *   offset  8 .. 15  : heap_base  ptr (i8*, 8 bytes)
 *   offset 16 .. 23  : shadow_base ptr (i8*, 8 bytes)
 *   offset 24 .. 31  : reserved
 *
 * Total footprint: 32 bytes/thread × 8192 threads = 256 KB in .global section.
 *
 * Configuration: uses compile-time constants for sizes. Real user-facing
 * --stack-size is stored in a .conf sidecar and read by the host launcher
 * to cuCtxSetLimit; the pass here uses the derived values that match.
 */

#include "Transforms.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

namespace coqui {

/* Total per-thread budget. sm_75 hardware ceiling is 512KB but real devices
 * typically allow 256KB (driver caps based on max-resident-threads × #SMs vs
 * device .local memory). Conservative 256KB fits commonly seen Turing/Ampere
 * driver budgets. coqui-cc may grow this into a CLI flag (--total-cap) later. */
static constexpr unsigned kCovMapSize      = 65536;
static constexpr unsigned kDefaultStackSize = 16384;
static constexpr unsigned kTotalBudget     = 262144;

/* Per-thread slot pool constants — must match host launcher batch size. */
static constexpr unsigned SLOT_STRIDE = 32;
static constexpr unsigned BATCH_SIZE  = 8192;

/* Offset of each field within a thread's SLOT_STRIDE-byte slot. */
static constexpr unsigned OFF_COV_BASE    = 0;
static constexpr unsigned OFF_HEAP_BASE   = 8;
static constexpr unsigned OFF_SHADOW_BASE = 16;

static GlobalVariable *getOrMakeSlotPool(Module &M) {
  if (auto *GV = M.getGlobalVariable("__coqui_thread_slots"))
    return GV;
  LLVMContext &C = M.getContext();
  ArrayType *T = ArrayType::get(Type::getInt8Ty(C),
                                static_cast<uint64_t>(BATCH_SIZE) * SLOT_STRIDE);
  return new GlobalVariable(
      M, T, /*isConstant=*/false,
      GlobalValue::ExternalLinkage,
      Constant::getNullValue(T),
      "__coqui_thread_slots"); /* default addrspace 0 = .global */
}

bool runMemoryLayout(Module &M) {
  LLVMContext &C = M.getContext();

  const unsigned realStack  = kDefaultStackSize;
  const unsigned remaining  = kTotalBudget - kCovMapSize - realStack;
  const unsigned heapSize   = (remaining * 8) / 9;
  const unsigned shadowSize = heapSize / 8;
  const unsigned usableHeap = heapSize; /* heap + shadow are separate allocas */

  /* 1. Find kernel entry */
  Function *Kernel = M.getFunction("__coqui_fuzz_kernel");
  if (!Kernel) {
    report_fatal_error(
        "[coqui-cc] MemoryLayout: __coqui_fuzz_kernel not found");
  }

  Type *i8  = Type::getInt8Ty(C);
  Type *i32 = Type::getInt32Ty(C);
  Type *i8p = PointerType::get(C, 0); /* addrspace(0) opaque pointer */

  /* 2. Declare the addrspace(0) slot pool and the tid helper. */
  GlobalVariable *Pool = getOrMakeSlotPool(M);

  FunctionType *TidT = FunctionType::get(i32, /*isVarArg=*/false);
  FunctionCallee Tid = M.getOrInsertFunction("__coqui_fuzz_tid", TidT);

  /* 3. Insert allocas at kernel entry, then store base pointers into pool slots */
  BasicBlock &EntryBB = Kernel->getEntryBlock();
  IRBuilder<> Builder(&EntryBB, EntryBB.begin());

  AllocaInst *CovAlloca = Builder.CreateAlloca(
      ArrayType::get(i8, kCovMapSize), nullptr, "cov_map");
  CovAlloca->setAlignment(Align(8));

  AllocaInst *HeapAlloca = Builder.CreateAlloca(
      ArrayType::get(i8, heapSize), nullptr, "heap");
  HeapAlloca->setAlignment(Align(16));

  AllocaInst *ShadowAlloca = Builder.CreateAlloca(
      ArrayType::get(i8, shadowSize), nullptr, "shadow");
  ShadowAlloca->setAlignment(Align(8));

  /* Zero the coverage map at kernel entry; each thread accumulates fresh. */
  Builder.CreateMemSet(CovAlloca,
                       ConstantInt::get(i8, 0),
                       static_cast<uint64_t>(kCovMapSize),
                       MaybeAlign(Align(8)));

  /* Compute tid and base byte offset into the pool for this thread. */
  Value *tid     = Builder.CreateCall(Tid, {}, "tid");
  Value *base32  = Builder.CreateMul(tid,
                                     ConstantInt::get(i32, SLOT_STRIDE),
                                     "tid_off");

  /* Store cov_alloca pointer at offset OFF_COV_BASE */
  Value *covOffI32 = Builder.CreateAdd(base32,
                                       ConstantInt::get(i32, OFF_COV_BASE),
                                       "cov_off");
  Value *covSlot = Builder.CreateGEP(i8, Pool, covOffI32, "cov_slot");
  Builder.CreateStore(CovAlloca, covSlot);

  /* Store heap_alloca pointer at offset OFF_HEAP_BASE */
  Value *heapOffI32 = Builder.CreateAdd(base32,
                                        ConstantInt::get(i32, OFF_HEAP_BASE),
                                        "heap_off");
  Value *heapSlot = Builder.CreateGEP(i8, Pool, heapOffI32, "heap_slot");
  Builder.CreateStore(HeapAlloca, heapSlot);

  /* Store shadow_alloca pointer at offset OFF_SHADOW_BASE */
  Value *shadowOffI32 = Builder.CreateAdd(base32,
                                          ConstantInt::get(i32, OFF_SHADOW_BASE),
                                          "shadow_off");
  Value *shadowSlot = Builder.CreateGEP(i8, Pool, shadowOffI32, "shadow_slot");
  Builder.CreateStore(ShadowAlloca, shadowSlot);

  /* 4. Emit accessor getter functions.
   *
   * Each getter computes: load ptr from Pool[tid * SLOT_STRIDE + Offset].
   */
  auto emitGetter = [&](StringRef Name, unsigned Offset) {
    FunctionType *FT = FunctionType::get(i8p, /*isVarArg=*/false);
    /* getOrInsertFunction returns an existing decl or creates a new one. */
    Function *F = cast<Function>(
        M.getOrInsertFunction(Name, FT).getCallee());
    /* If the function already has a body (defined earlier), skip. */
    if (!F->isDeclaration())
      return;
    /* It is a forward declaration (from runtime.h or FuzzEntry) — fill it in. */
    F->setLinkage(GlobalValue::InternalLinkage);
    F->addFnAttr(Attribute::AlwaysInline);
    BasicBlock *BB = BasicBlock::Create(C, "entry", F);
    IRBuilder<> B(BB);
    Value *t     = B.CreateCall(Tid, {}, "tid");
    Value *b32   = B.CreateMul(t, ConstantInt::get(i32, SLOT_STRIDE), "tid_off");
    Value *off   = B.CreateAdd(b32, ConstantInt::get(i32, Offset), "off");
    Value *slotPtr = B.CreateGEP(i8, Pool, off, "slot_ptr");
    B.CreateRet(B.CreateLoad(i8p, slotPtr, Name));
  };

  emitGetter("__coqui_cov_base",    OFF_COV_BASE);
  emitGetter("__coqui_heap_base",   OFF_HEAP_BASE);
  emitGetter("__coqui_shadow_base", OFF_SHADOW_BASE);

  /* __coqui_heap_size() — returns compile-time constant as i32 */
  {
    FunctionType *FT = FunctionType::get(i32, /*isVarArg=*/false);
    Function *F = cast<Function>(
        M.getOrInsertFunction("__coqui_heap_size", FT).getCallee());
    if (F->isDeclaration()) {
      F->setLinkage(GlobalValue::InternalLinkage);
      F->addFnAttr(Attribute::AlwaysInline);
      BasicBlock *BB = BasicBlock::Create(C, "entry", F);
      IRBuilder<> B(BB);
      B.CreateRet(ConstantInt::get(i32, usableHeap));
    }
  }

  return true;
}

} // namespace coqui
