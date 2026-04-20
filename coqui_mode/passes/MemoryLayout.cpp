/*
 * MemoryLayout.cpp --- set up per-thread heap / shadow regions + cov pool binding.
 *
 * Iteration 10 rewrite: coverage map is no longer a per-thread 64 KB alloca.
 * Instead, the host cuMemAllocs a device-global pool of size
 * (batch_size * cov_map_size_rounded_up_to_4) where cov_map_size is derived
 * from @__coqui_num_edges emitted by the SancovCount pass. The host writes
 * the pool base pointer into the device global @__coqui_cov_map_pool_ptr.
 *
 * This pass:
 *   - Emits @__coqui_cov_map_pool_ptr (ptr, module-scope, addrspace 0),
 *     declared ExternalLinkage so the host can bind it via cuModuleGetGlobal.
 *   - Inserts heap + shadow allocas at kernel entry (unchanged from before).
 *   - Computes cov_base = cov_map_pool + tid * cov_map_size at kernel entry
 *     (reads the pool ptr from @__coqui_cov_map_pool_ptr and the size from
 *     @__coqui_num_edges rounded up to 4).
 *   - Stores (cov_base, heap_base, shadow_base) into the tid-indexed
 *     __coqui_thread_slots pool.
 *   - Emits __coqui_cov_base / __coqui_heap_base / __coqui_shadow_base getter
 *     functions.
 *
 * The removal of the 64 KB per-thread cov alloca cuts per-thread static stack
 * usage by 64 KB — for sm_75 + 8192 threads this frees ~512 MB of .local.
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

/* Per-thread total budget (heap + shadow + real stack). Matches the
 * CUDA driver-reported real-stack ceiling on sm_75 (512 KB but
 * typically 256 KB after the driver's max-resident-threads check). */
static constexpr unsigned kDefaultStackSize = 32768;
static constexpr unsigned kTotalBudget      = 262144;

/* --coqui-stack-size=N : real-stack budget in bytes (forwarded by
 * coqui-cc --stack-size). The heap + shadow sizes are derived from the
 * remainder. */
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

/* Module-scope device global holding the base pointer of the cov_map pool.
 * The host writes this via cuMemcpyHtoD after cuMemAllocing the pool,
 * looking up the symbol via cuModuleGetGlobal("__coqui_cov_map_pool_ptr").
 * Declared ExternalLinkage + default-initialized to nullptr; the initializer
 * is a placeholder — the host overrides it at init. */
static GlobalVariable *getOrMakeCovPoolPtr(Module &M) {
  if (auto *GV = M.getGlobalVariable("__coqui_cov_map_pool_ptr"))
    return GV;
  LLVMContext &C = M.getContext();
  Type *PtrTy = PointerType::get(C, 0);
  auto *GV = new GlobalVariable(
      M, PtrTy, /*isConstant=*/false,
      GlobalValue::ExternalLinkage,
      Constant::getNullValue(PtrTy),
      "__coqui_cov_map_pool_ptr");
  GV->setAlignment(Align(8));
  return GV;
}

/* @__coqui_num_edges is emitted as an i64 constant by the SancovCount pass
 * that runs before us. If SancovCount found at least one __sancov_gen_*
 * array, the value is the total edge count. Otherwise, it is 0 (legacy /
 * no-sancov build). We fall back to 65536 in the zero case so old callers
 * still work.
 *
 * cov_map_size (the per-thread footprint) is num_edges rounded up to 4 —
 * coverage_evaluate processes the map in 4-byte chunks. */
static uint64_t readNumEdges(Module &M) {
  GlobalVariable *GV = M.getGlobalVariable("__coqui_num_edges", true);
  if (!GV) return 0;
  if (!GV->hasInitializer()) return 0;
  auto *C = dyn_cast<ConstantInt>(GV->getInitializer());
  if (!C) return 0;
  return C->getZExtValue();
}

bool runMemoryLayout(Module &M) {
  LLVMContext &C = M.getContext();

  /* Resolve real stack size: prefer --coqui-stack-size CLI if set, otherwise
   * default constant. */
  unsigned realStack = RealStackSizeOpt;
  if (realStack >= kTotalBudget) {
    report_fatal_error(
        "[coqui-cc] MemoryLayout: --coqui-stack-size exceeds 256KB budget");
  }
  /* The old layout always reserved 64 KB for cov before sizing heap/shadow.
   * We no longer do per-thread cov allocas — it lives in a device-global pool
   * — but we keep the same heap/shadow sizing formula so the numbers match
   * what the kernel has historically run with. */
  static constexpr unsigned kLegacyCov = 65536;
  const unsigned remaining  = kTotalBudget - kLegacyCov - realStack;
  const unsigned heapSize   = (remaining * 8) / 9;
  const unsigned shadowSize = heapSize / 8;
  const unsigned usableHeap = heapSize; /* heap + shadow are separate allocas */

  /* Resolve per-thread coverage map size (bytes). Rounded up to 4 because
   * coverage_evaluate walks in 4-byte chunks. Zero means the target was
   * compiled without -fsanitize-coverage=trace-pc-guard; we still need a
   * non-zero footprint so the host pool allocation succeeds — fall back to
   * 1024 so the module still runs (but novelty will be permanently 0). */
  uint64_t numEdges = readNumEdges(M);
  uint64_t covMapSize = (numEdges + 3ULL) & ~3ULL;
  if (covMapSize == 0) covMapSize = 1024; /* conservative fallback */

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

  /* 3. Insert allocas at kernel entry (heap + shadow only), then store base
   *    pointers into pool slots. Coverage lives in a device-global pool
   *    indexed by tid * cov_map_size — no alloca needed. */
  BasicBlock &EntryBB = Kernel->getEntryBlock();
  IRBuilder<> Builder(&EntryBB, EntryBB.begin());

  AllocaInst *HeapAlloca = Builder.CreateAlloca(
      ArrayType::get(i8, heapSize), nullptr, "heap");
  HeapAlloca->setAlignment(Align(16));

  AllocaInst *ShadowAlloca = Builder.CreateAlloca(
      ArrayType::get(i8, shadowSize), nullptr, "shadow");
  ShadowAlloca->setAlignment(Align(8));

  /* Compute tid and base byte offset into the slot pool for this thread. */
  Value *tid     = Builder.CreateCall(Tid, {}, "tid");
  Value *base32  = Builder.CreateMul(tid,
                                     ConstantInt::get(i32, SLOT_STRIDE),
                                     "tid_off");

  /* Compute per-thread cov_base = cov_map_pool + tid * cov_map_size.
   * Loaded at kernel entry once; stored into the slot so subsequent callers
   * (ASan report path, post-kernel classify/virgin) can access it.
   *
   * Using i64 for the byte-stride to accommodate large batch × large
   * per-thread map (e.g. 8192 × 16384 = 128 MB — well within u32 but we
   * standardise on i64 for safety on future larger modules). */
  Value *covPoolBase = Builder.CreateLoad(i8p, CovPoolPtr, "cov_pool_base");
  Value *tid64       = Builder.CreateZExt(tid, i64, "tid64");
  Value *covOffset   = Builder.CreateMul(
      tid64, ConstantInt::get(i64, covMapSize), "cov_thread_off");
  Value *covBase     = Builder.CreateGEP(i8, covPoolBase, covOffset, "cov_base_val");

  /* Store cov_base pointer at offset OFF_COV_BASE */
  Value *covOffI32 = Builder.CreateAdd(base32,
                                       ConstantInt::get(i32, OFF_COV_BASE),
                                       "cov_off");
  Value *covSlot = Builder.CreateGEP(i8, Pool, covOffI32, "cov_slot");
  Builder.CreateStore(covBase, covSlot);

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

  /* __coqui_cov_map_size() — returns the per-thread cov map size in bytes
   * (i64). The runtime uses this to size its coverage_evaluate walk. Falls
   * back to 1024 if @__coqui_num_edges is zero (module built without
   * -fsanitize-coverage=trace-pc-guard). */
  {
    FunctionType *FT = FunctionType::get(i64, /*isVarArg=*/false);
    Function *F = cast<Function>(
        M.getOrInsertFunction("__coqui_cov_map_size", FT).getCallee());
    if (F->isDeclaration()) {
      F->setLinkage(GlobalValue::InternalLinkage);
      F->addFnAttr(Attribute::AlwaysInline);
      BasicBlock *BB = BasicBlock::Create(C, "entry", F);
      IRBuilder<> B(BB);
      B.CreateRet(ConstantInt::get(i64, covMapSize));
    }
  }

  return true;
}

} // namespace coqui
