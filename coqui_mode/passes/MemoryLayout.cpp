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
 * Total footprint: 32 bytes/thread × 65536 threads = 2 MB in .global section.
 *
 * Configuration: the real-stack budget is controlled via the
 * -coqui-stack-size=N opt flag (default 32768 B). coqui-cc forwards its
 * --stack-size value to opt so the pass-derived heap/shadow sizes match the
 * host's cuCtxSetLimit value. Without the flag, the default constant applies.
 */

#include "Transforms.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

namespace coqui {

/* Stack-canary sentinel. Must match COQUI_STACK_CANARY in
 * coqui_mode/runtime/coqui_runtime.h — if that literal changes, update here. */
static constexpr uint64_t kStackCanary = 0xCA7A1C0FFEE0DE50ULL;

/* Total per-thread budget. sm_75 hardware ceiling is 512KB but real devices
 * typically allow 256KB (driver caps based on max-resident-threads × #SMs vs
 * device .local memory). Conservative 256KB fits commonly seen Turing/Ampere
 * driver budgets. coqui-cc may grow this into a CLI flag (--total-cap) later. */
static constexpr unsigned kCovMapFallback   = 65536;
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
static constexpr unsigned BATCH_SIZE  = 65536;

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

  /* Resolve real stack size: prefer --coqui-stack-size CLI if set, otherwise
   * default constant. Guard against a value that would leave nothing for heap. */
  unsigned realStack = RealStackSizeOpt;

  /* Read the edge count set by runEdgeCount (pre-pass). Falls back to
   * 64KB if the pre-pass didn't run (shouldn't happen in practice). */
  unsigned covMapSize = kCovMapFallback;
  if (auto *GV = M.getGlobalVariable("__coqui_edge_count")) {
    if (auto *CI = dyn_cast<ConstantInt>(GV->getInitializer()))
      covMapSize = CI->getZExtValue();
  }

  if (realStack + covMapSize >= kTotalBudget) {
    report_fatal_error(
        "[coqui-cc] MemoryLayout: --coqui-stack-size + cov exceeds budget");
  }
  const unsigned remaining  = kTotalBudget - covMapSize - realStack;
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
  Type *voidT = Type::getVoidTy(C);

  /* 2. Declare the addrspace(0) slot pool and the tid helper. */
  GlobalVariable *Pool = getOrMakeSlotPool(M);

  FunctionType *TidT = FunctionType::get(i32, /*isVarArg=*/false);
  FunctionCallee Tid = M.getOrInsertFunction("__coqui_fuzz_tid", TidT);

  /* 3. Insert allocas at kernel entry, then store base pointers into pool slots */
  BasicBlock &EntryBB = Kernel->getEntryBlock();
  IRBuilder<> Builder(&EntryBB, EntryBB.begin());

  /* Stack canary. i64 alloca at the outermost kernel frame; we store
   * COQUI_STACK_CANARY (kStackCanary) into it here, then emit a check at
   * every kernel exit. On NVPTX, allocas and the call-stack frames share
   * the per-thread .local region — a deep recursion that overruns
   * CU_LIMIT_STACK_SIZE will eventually corrupt this slot.
   *
   * We place it first so (a) it's the outermost alloca and hence the
   * farthest from growing call-stack frames, maximizing the chance that
   * runaway recursion reaches it before other important state, and
   * (b) its address is stable regardless of how the other allocas below
   * get ordered by the NVPTX lowering. */
  AllocaInst *CanarySlot =
      Builder.CreateAlloca(i64, nullptr, "stack_canary");
  CanarySlot->setAlignment(Align(8));
  Builder.CreateStore(ConstantInt::get(i64, kStackCanary), CanarySlot);

  AllocaInst *CovAlloca = Builder.CreateAlloca(
      ArrayType::get(i8, covMapSize), nullptr, "cov_map");
  CovAlloca->setAlignment(Align(8));

  AllocaInst *HeapAlloca = Builder.CreateAlloca(
      ArrayType::get(i8, heapSize), nullptr, "heap");
  HeapAlloca->setAlignment(Align(16));

  AllocaInst *ShadowAlloca = Builder.CreateAlloca(
      ArrayType::get(i8, shadowSize), nullptr, "shadow");
  ShadowAlloca->setAlignment(Align(8));

  Builder.CreateMemSet(CovAlloca,
                       ConstantInt::get(i8, 0),
                       static_cast<uint64_t>(covMapSize),
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
   * Each getter loads a ptr from Pool[tid * SLOT_STRIDE + Offset].
   * The slot pool is written at kernel entry above; getters are called
   * from coverage + ASan instrumentation (inlined at each call site). */
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

  /* 5. Emit the stack-canary check at every kernel exit.
   *
   * We insert BEFORE each ReturnInst in __coqui_fuzz_kernel so the check
   * runs no matter which of FuzzEntry's exit paths (len==0 early exit vs
   * post-bucketing fall-through) the thread takes. The runtime helper
   * calls __coqui_trap_with_reason(COQUI_TRAP_STACK_OVERFLOW) on mismatch
   * so the host's rerun loop (PR #10) picks up trap_reason=12 and
   * re-verifies on the CPU. */
  FunctionType *CheckT = FunctionType::get(voidT, {i8p}, /*isVarArg=*/false);
  FunctionCallee Check =
      M.getOrInsertFunction("__coqui_check_stack_canary", CheckT);

  /* Collect ReturnInsts first — inserting calls before ret shouldn't
   * invalidate iteration, but we keep the two steps separate for clarity. */
  SmallVector<ReturnInst *, 4> Returns;
  for (BasicBlock &BB : *Kernel) {
    if (auto *RI = dyn_cast<ReturnInst>(BB.getTerminator()))
      Returns.push_back(RI);
  }
  for (ReturnInst *RI : Returns) {
    IRBuilder<> RBuilder(RI);
    RBuilder.CreateCall(Check, {CanarySlot});
  }

  return true;
}

} // namespace coqui
