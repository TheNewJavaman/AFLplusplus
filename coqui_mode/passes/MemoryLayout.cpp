/*
 * MemoryLayout.cpp --- set up per-thread stack regions.
 *
 * Inserts three allocas at kernel entry:
 *   - cov_map  : 64 KB (aligned 8)
 *   - heap     : H bytes (runtime-configured)
 *   - shadow   : H/8 bytes (aligned 8)
 *
 * Stores their pointers into addr-space-5 (.local) global slots, then
 * emits helper accessor functions that load from those slots.
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

static constexpr unsigned kCovMapSize     = 65536;
static constexpr unsigned kDefaultStackSize = 16384;
static constexpr unsigned kTotalBudget    = 524288;

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
  /* addrspace(0) opaque pointer — value type stored in the .local slots */
  Type *i8p  = PointerType::get(C, 0);
  /* addrspace(5) opaque pointer — type of the .local global variable itself */
  Type *i8p5 = PointerType::get(C, 5);

  /* 2. Declare the .local globals that hold per-thread base pointers.
   *
   * These live in NVPTX .local address space (5).  Each thread gets its own
   * copy automatically; any helper function can load from them.
   *
   * The GlobalVariable stores a ptr (addrspace 0) value; the variable itself
   * is placed in addrspace 5 via the AddressSpace parameter.
   */
  auto makeLocalSlot = [&](StringRef Name) -> GlobalVariable * {
    GlobalVariable *GV = M.getGlobalVariable(Name, /*AllowInternal=*/true);
    if (GV)
      return GV;
    return new GlobalVariable(
        M, i8p, /*isConstant=*/false,
        GlobalValue::InternalLinkage,
        Constant::getNullValue(i8p),
        Name, /*InsertBefore=*/nullptr,
        GlobalValue::NotThreadLocal,
        /*AddressSpace=*/5);
  };

  GlobalVariable *CovSlot    = makeLocalSlot("__coqui_cov_slot");
  GlobalVariable *HeapSlot   = makeLocalSlot("__coqui_heap_slot");
  GlobalVariable *ShadowSlot = makeLocalSlot("__coqui_shadow_slot");

  /* Suppress unused-variable warnings for the addrspace-5 pointer type. */
  (void)i8p5;

  /* 3. Insert allocas at kernel entry, then store base pointers into slots */
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

  /* Store base pointers into per-thread .local slots. */
  Builder.CreateStore(CovAlloca,    CovSlot);
  Builder.CreateStore(HeapAlloca,   HeapSlot);
  Builder.CreateStore(ShadowAlloca, ShadowSlot);

  /* 4. Emit accessor getter functions */
  auto emitGetter = [&](StringRef Name, GlobalVariable *Slot) {
    if (M.getFunction(Name))
      return;
    FunctionType *FT = FunctionType::get(i8p, /*isVarArg=*/false);
    Function *F = Function::Create(FT, GlobalValue::InternalLinkage, Name, &M);
    F->addFnAttr(Attribute::AlwaysInline);
    BasicBlock *BB = BasicBlock::Create(C, "entry", F);
    IRBuilder<> B(BB);
    B.CreateRet(B.CreateLoad(i8p, Slot, Name));
  };

  emitGetter("__coqui_cov_base",    CovSlot);
  emitGetter("__coqui_heap_base",   HeapSlot);
  emitGetter("__coqui_shadow_base", ShadowSlot);

  /* __coqui_heap_size() — returns compile-time constant as i32 */
  if (!M.getFunction("__coqui_heap_size")) {
    FunctionType *FT = FunctionType::get(i32, /*isVarArg=*/false);
    Function *F = Function::Create(FT, GlobalValue::InternalLinkage,
                                   "__coqui_heap_size", &M);
    F->addFnAttr(Attribute::AlwaysInline);
    BasicBlock *BB = BasicBlock::Create(C, "entry", F);
    IRBuilder<> B(BB);
    B.CreateRet(ConstantInt::get(i32, usableHeap));
  }

  return true;
}

} // namespace coqui
