/*
 * MemoryLayout.cpp --- set up per-thread stack regions.
 *
 * cov_map now lives in a device-global pool (single cuMemAlloc'd buffer of
 * size BATCH_SIZE * 64KB). Each thread's slot is (cov_pool_base + tid * 64KB),
 * addressed via the __coqui_cov_base() helper which reads pool_base from the
 * module-scope `__coqui_cov_pool_ptr` global (host binds this via
 * cuModuleGetGlobal at init). The old per-thread alloca is gone; this keeps
 * the cov_map alive across a split of kernel A (user code) and kernel B
 * (classify+virgin) since `.local` allocas would be reclaimed at kernel exit.
 *
 * Heap + shadow remain per-thread allocas (ASan semantics require the shadow
 * to be contiguous with heap bytes; easier to keep both on the .local stack).
 *
 * Inserts two allocas at kernel entry:
 *   - heap     : H bytes (runtime-configured)
 *   - shadow   : H/8 bytes (aligned 8)
 *
 * Stores their pointers into a tid-indexed addrspace(0) (.global) slot pool,
 * then emits helper accessor functions that load from the per-thread slot.
 *
 * Layout of __coqui_thread_slots[tid * SLOT_STRIDE + offset]:
 *   offset  0 ..  7  : reserved (cov moved to global pool)
 *   offset  8 .. 15  : heap_base  ptr (i8*, 8 bytes)
 *   offset 16 .. 23  : shadow_base ptr (i8*, 8 bytes)
 *   offset 24 .. 31  : reserved
 *
 * Total footprint: 32 bytes/thread × 8192 threads = 256 KB in .global section.
 *
 * Configuration: the real-stack budget is controlled via the
 * -coqui-stack-size=N opt flag (default 32768 B). coqui-cc forwards its
 * --stack-size value to opt so the pass-derived heap/shadow sizes match the
 * host's cuCtxSetLimit value. Without the flag, the default constant applies.
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

/* Total per-thread budget. sm_75 hardware ceiling is 512KB but real devices
 * typically allow 256KB (driver caps based on max-resident-threads × #SMs vs
 * device .local memory). Conservative 256KB fits commonly seen Turing/Ampere
 * driver budgets. coqui-cc may grow this into a CLI flag (--total-cap) later. */
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

/* Get-or-create the pool base pointer global: an i8* (8 bytes) the host
 * will bind to the cuMemAlloc'd cov pool at init via cuModuleGetGlobal. */
static GlobalVariable *getOrMakeCovPoolPtr(Module &M) {
  if (auto *GV = M.getGlobalVariable("__coqui_cov_pool_ptr"))
    return GV;
  LLVMContext &C = M.getContext();
  Type *i8p = PointerType::get(C, 0);
  return new GlobalVariable(
      M, i8p, /*isConstant=*/false,
      GlobalValue::ExternalLinkage,
      Constant::getNullValue(i8p),
      "__coqui_cov_pool_ptr"); /* addrspace 0 = .global */
}

bool runMemoryLayout(Module &M) {
  LLVMContext &C = M.getContext();

  /* Resolve real stack size: prefer --coqui-stack-size CLI if set, otherwise
   * default constant. Guard against a value that would leave nothing for heap.
   *
   * Note: since cov_map moved OUT of .local (it's now in a global pool), the
   * per-thread .local budget frees up 64 KB. We still keep the same "logical"
   * total budget for heap+stack sizing for stability — the freed space becomes
   * headroom for register spill + stage-B kernel locals. */
  unsigned realStack = RealStackSizeOpt;
  if (realStack + kCovMapSize >= kTotalBudget) {
    report_fatal_error(
        "[coqui-cc] MemoryLayout: --coqui-stack-size + 64KB cov exceeds 256KB budget");
  }
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
  Type *i64 = Type::getInt64Ty(C);
  Type *i8p = PointerType::get(C, 0); /* addrspace(0) opaque pointer */

  /* 2. Declare the addrspace(0) slot pool, the cov-pool pointer global,
   *    and the tid helper. */
  GlobalVariable *Pool       = getOrMakeSlotPool(M);
  GlobalVariable *CovPoolPtr = getOrMakeCovPoolPtr(M);

  FunctionType *TidT = FunctionType::get(i32, /*isVarArg=*/false);
  FunctionCallee Tid = M.getOrInsertFunction("__coqui_fuzz_tid", TidT);

  /* 3. Insert allocas at kernel entry, then store base pointers into pool slots.
   *    cov_map is NO LONGER a .local alloca — it comes from the global pool. */
  BasicBlock &EntryBB = Kernel->getEntryBlock();
  IRBuilder<> Builder(&EntryBB, EntryBB.begin());

  AllocaInst *HeapAlloca = Builder.CreateAlloca(
      ArrayType::get(i8, heapSize), nullptr, "heap");
  HeapAlloca->setAlignment(Align(16));

  AllocaInst *ShadowAlloca = Builder.CreateAlloca(
      ArrayType::get(i8, shadowSize), nullptr, "shadow");
  ShadowAlloca->setAlignment(Align(8));

  /* No per-thread cov_map zeroing at entry: the zero-on-read pattern in
   * __coqui_virgin_compare_and_flag leaves the pool at zeros after each
   * batch, so stage A always sees a clean slot. */

  /* Compute tid and base byte offset into the pool for this thread. */
  Value *tid     = Builder.CreateCall(Tid, {}, "tid");
  Value *base32  = Builder.CreateMul(tid,
                                     ConstantInt::get(i32, SLOT_STRIDE),
                                     "tid_off");

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
   * __coqui_cov_base() — computes cov_pool_base + tid * 64KB (from global
   * pool pointer __coqui_cov_pool_ptr bound by the host at module load).
   * __coqui_heap_base / __coqui_shadow_base — loads from Pool slot as before.
   */

  /* Emit __coqui_cov_base — no slot indirection; compute directly. */
  {
    FunctionType *FT = FunctionType::get(i8p, /*isVarArg=*/false);
    Function *F = cast<Function>(
        M.getOrInsertFunction("__coqui_cov_base", FT).getCallee());
    if (F->isDeclaration()) {
      F->setLinkage(GlobalValue::InternalLinkage);
      F->addFnAttr(Attribute::AlwaysInline);
      BasicBlock *BB = BasicBlock::Create(C, "entry", F);
      IRBuilder<> B(BB);
      /* base = *__coqui_cov_pool_ptr  (i8*) */
      Value *base = B.CreateLoad(i8p, CovPoolPtr, "cov_pool_base");
      /* tid in i32 -> zext to i64, multiply by 65536 */
      Value *t     = B.CreateCall(Tid, {}, "tid");
      Value *t64   = B.CreateZExt(t, i64, "tid64");
      Value *off64 = B.CreateMul(t64, ConstantInt::get(i64, kCovMapSize),
                                 "cov_off64");
      Value *p     = B.CreateGEP(i8, base, off64, "cov_slot_ptr");
      B.CreateRet(p);
    }
  }

  auto emitSlotGetter = [&](StringRef Name, unsigned Offset) {
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

  emitSlotGetter("__coqui_heap_base",   OFF_HEAP_BASE);
  emitSlotGetter("__coqui_shadow_base", OFF_SHADOW_BASE);

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
