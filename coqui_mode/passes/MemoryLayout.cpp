/*
 * MemoryLayout.cpp --- set up per-thread heap/shadow + warp-shared cov pool binding.
 *
 * Iteration 20 rewrite (warp-shared cov_map):
 *   Coverage map is no longer a per-thread 64 KB alloca. Instead:
 *     - One 64 KB cov map per WARP (32 threads share one map).
 *     - Host cuMemAllocs a device-global pool of size
 *       (num_warps × 65536) = 256 × 64 KB = 16 MB total on sm_75.
 *     - Host writes the pool base into @__coqui_cov_pool_ptr via
 *       cuModuleGetGlobal + cuMemcpyHtoD.
 *   Per-thread cov_base = cov_pool_base + (tid >> 5) * 65536.
 *
 * Footprint savings:
 *   Before: 8192 × 64 KB = 512 MB per-thread .local
 *   After : 256 × 64 KB  = 16 MB device-global (32× reduction)
 *
 * Heap + shadow remain per-thread allocas unchanged.
 *
 * Getter functions emitted:
 *   __coqui_cov_base()    — returns pool_base + (tid >> 5) * 65536
 *   __coqui_heap_base()   — returns per-thread heap alloca
 *   __coqui_shadow_base() — returns per-thread shadow alloca
 *   __coqui_heap_size()   — returns compile-time constant
 *
 * Configuration: the real-stack budget is controlled via the
 * -coqui-stack-size=N opt flag (default 32768 B). The heap/shadow
 * sizes still subtract 64 KB "cov reserve" in the formula so the
 * per-thread heap budget matches historical behaviour even though
 * the cov map itself is no longer per-thread.
 */

#include "Transforms.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

namespace coqui {

static constexpr unsigned kCovMapSize      = 65536;
static constexpr unsigned kDefaultStackSize = 32768;
static constexpr unsigned kTotalBudget     = 262144;

/* --coqui-stack-size=N : real-stack budget in bytes. coqui-cc forwards its
 * --stack-size value via this opt flag so the pass-produced heap/shadow
 * sizes are consistent with the runtime cuCtxSetLimit value. */
static cl::opt<unsigned> RealStackSizeOpt(
    "coqui-stack-size",
    cl::desc("Per-thread real-stack budget in bytes (heap/shadow derived from remainder)"),
    cl::init(kDefaultStackSize));

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

/* Module-scope device global holding the base pointer of the warp-shared
 * cov map pool. The host writes this via cuMemcpyHtoD after cuMemAllocing
 * the pool, looking up the symbol via cuModuleGetGlobal("__coqui_cov_pool_ptr").
 * ExternalLinkage + default-initialized to nullptr; the host overrides at init. */
static GlobalVariable *getOrMakeCovPoolPtr(Module &M) {
  if (auto *GV = M.getGlobalVariable("__coqui_cov_pool_ptr"))
    return GV;
  LLVMContext &C = M.getContext();
  Type *PtrTy = PointerType::get(C, 0);
  auto *NewGV = new GlobalVariable(
      M, PtrTy, /*isConstant=*/false,
      GlobalValue::ExternalLinkage,
      Constant::getNullValue(PtrTy),
      "__coqui_cov_pool_ptr");
  NewGV->setAlignment(Align(8));
  return NewGV;
}

bool runMemoryLayout(Module &M) {
  LLVMContext &C = M.getContext();

  /* Resolve real stack size: prefer --coqui-stack-size CLI if set, otherwise
   * default constant. Guard against a value that would leave nothing for heap.
   * Keep the "+64 KB cov reserve" in the sizing formula so heap/shadow match
   * the historical footprint even though the cov map is now warp-shared and
   * lives in a device-global pool, not a per-thread alloca. */
  unsigned realStack = RealStackSizeOpt;
  if (realStack + kCovMapSize >= kTotalBudget) {
    report_fatal_error(
        "[coqui-cc] MemoryLayout: --coqui-stack-size + 64KB cov reserve exceeds 256KB budget");
  }
  const unsigned remaining  = kTotalBudget - kCovMapSize - realStack;
  const unsigned heapSize   = (remaining * 8) / 9;
  const unsigned shadowSize = heapSize / 8;
  const unsigned usableHeap = heapSize;

  /* 1. Find kernel entry */
  Function *Kernel = M.getFunction("__coqui_fuzz_kernel");
  if (!Kernel) {
    report_fatal_error(
        "[coqui-cc] MemoryLayout: __coqui_fuzz_kernel not found");
  }

  Type *i8  = Type::getInt8Ty(C);
  Type *i32 = Type::getInt32Ty(C);
  Type *i64 = Type::getInt64Ty(C);
  Type *i8p = PointerType::get(C, 0); /* addrspace(0) opaque pointer */

  /* 2. Declare the addrspace(0) slot pool + cov pool ptr + tid helper. */
  GlobalVariable *Pool       = getOrMakeSlotPool(M);
  GlobalVariable *CovPoolPtr = getOrMakeCovPoolPtr(M);

  FunctionType *TidT = FunctionType::get(i32, /*isVarArg=*/false);
  FunctionCallee Tid = M.getOrInsertFunction("__coqui_fuzz_tid", TidT);

  /* 3. Insert heap + shadow allocas at kernel entry; compute warp-shared
   * cov_base from the device-global pool pointer. No cov alloca. */
  BasicBlock &EntryBB = Kernel->getEntryBlock();
  IRBuilder<> Builder(&EntryBB, EntryBB.begin());

  AllocaInst *HeapAlloca = Builder.CreateAlloca(
      ArrayType::get(i8, heapSize), nullptr, "heap");
  HeapAlloca->setAlignment(Align(16));

  AllocaInst *ShadowAlloca = Builder.CreateAlloca(
      ArrayType::get(i8, shadowSize), nullptr, "shadow");
  ShadowAlloca->setAlignment(Align(8));

  /* Compute tid and base byte offset into the pool for this thread. */
  Value *tid     = Builder.CreateCall(Tid, {}, "tid");
  Value *base32  = Builder.CreateMul(tid,
                                     ConstantInt::get(i32, SLOT_STRIDE),
                                     "tid_off");

  /* Compute warp-shared cov_base = cov_pool_base + (tid >> 5) * 65536.
   *   warp_id = tid >> 5
   *   cov_base = cov_pool + warp_id * 65536
   * All 32 threads in a warp get the SAME cov_base. */
  Value *covPoolBase = Builder.CreateLoad(i8p, CovPoolPtr, "cov_pool_base");
  Value *warpId      = Builder.CreateLShr(tid, ConstantInt::get(i32, 5), "warp_id");
  Value *warpId64    = Builder.CreateZExt(warpId, i64, "warp_id64");
  Value *covOffset   = Builder.CreateMul(
      warpId64, ConstantInt::get(i64, kCovMapSize), "cov_warp_off");
  Value *covBaseVal  = Builder.CreateGEP(i8, covPoolBase, covOffset, "cov_base_val");

  /* Store cov_base pointer at offset OFF_COV_BASE */
  Value *covOffI32 = Builder.CreateAdd(base32,
                                       ConstantInt::get(i32, OFF_COV_BASE),
                                       "cov_off");
  Value *covSlot = Builder.CreateGEP(i8, Pool, covOffI32, "cov_slot");
  Builder.CreateStore(covBaseVal, covSlot);

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
    Function *F = cast<Function>(
        M.getOrInsertFunction(Name, FT).getCallee());
    if (!F->isDeclaration())
      return;
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
