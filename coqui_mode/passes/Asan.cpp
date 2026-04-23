/*
 * Asan.cpp --- heap-only ASan instrumentation pass.
 *
 * Two responsibilities:
 *   1. RAUW allocators: __coqui_malloc  -> __coqui_asan_malloc
 *                       __coqui_free    -> __coqui_asan_free
 *   2. Instrument loads/stores: insert __coqui_asan_check_load_N /
 *      __coqui_asan_check_store_N (N = 1/2/4/8) before each heap access.
 *
 * Skips:
 *   - Functions with __coqui_* prefix (runtime helpers must not be
 *     self-instrumented).
 *   - Accesses where the stripped-base pointer is an AllocaInst (stack;
 *     not tracked by the heap shadow).
 *   - Accesses where the stripped-base pointer is a GlobalVariable (static
 *     data; not heap).
 *   - Loads/stores in non-default address spaces (e.g., NVPTX .shared /
 *     .const memory).
 *   - Atomic loads/stores (skipped for v1; add AtomicRMW if needed).
 *
 * Size selection: if the access byte count is 1, 2, 4, or 8 use the matching
 * helper.  For all other sizes, use _8 if size >= 8, else _1.
 * This is a known v1 simplification (conservative — may miss the last few
 * bytes of large unaligned accesses, but never over-reads the shadow).
 *
 * RAUW scope (v1): only __coqui_malloc and __coqui_free.  Calloc/realloc
 * are left for a follow-up task; they are rare in fuzz target code.
 *
 * Ported from /coqui/src/AsanTransform.cpp with LLVM 18 opaque-pointer
 * adjustments.
 *
 * Instrumentation emits a call to a size-specialized outlined fast-path
 * helper (__coqui_asan_check_fast_{load,store}_{1,2,4,8}) at each
 * eligible load/store site.  The helper body does the range check +
 * shadow byte check inline (with branch-weight metadata for llc -O2
 * layout) and calls into the existing __coqui_asan_slowpath_*
 * functions on miss.  This outlining avoids the per-site 3x basic block
 * bloat that the previous inline-always approach created, which caused
 * catastrophic ptxas-O1 pathology on larger targets (cmark with 12k
 * loads: 55 min / 53 GB RSS before port).
 */

#include "Transforms.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace coqui {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Return true if V (or the root of the GEP/bitcast chain ending at V) is
/// an AllocaInst.  Heap-only ASan does not shadow the stack.
static bool isStackDerived(const Value *V) {
  // stripInBoundsOffsets follows inbounds GEPs and some casts.
  const Value *Base = V->stripInBoundsOffsets();
  return isa<AllocaInst>(Base);
}

/// Return true if the root of the pointer chain is a GlobalVariable.
/// Globals live in static data sections, not the heap shadow.
static bool isGlobalDerived(const Value *V) {
  const Value *Base = V->stripInBoundsOffsets();
  return isa<GlobalVariable>(Base);
}

/// Map an access byte count to 1/2/4/8 for the helper suffix.
/// Conservative: sizes not in {1,2,4,8} use 8 if >= 8, else 1.
static unsigned selectSize(uint64_t ByteCount) {
  if (ByteCount == 1 || ByteCount == 2 || ByteCount == 4 || ByteCount == 8)
    return static_cast<unsigned>(ByteCount);
  return ByteCount >= 8 ? 8u : 1u;
}

/// Build branch_weights metadata node: used to annotate conditional
/// branches so llc -O2 lays out the common path as fall-through.
/// Weights are relative; (95, 5) means 95% likely to take the TRUE branch.
static llvm::MDNode *createBranchWeightMD(llvm::LLVMContext &Ctx,
                                          uint32_t TrueWeight,
                                          uint32_t FalseWeight) {
  using namespace llvm;
  auto *I32 = Type::getInt32Ty(Ctx);
  return MDNode::get(
      Ctx,
      {MDString::get(Ctx, "branch_weights"),
       ConstantAsMetadata::get(ConstantInt::get(I32, TrueWeight)),
       ConstantAsMetadata::get(ConstantInt::get(I32, FalseWeight))});
}

/// Create an outlined fast-path helper.
///
///   void __coqui_asan_check_fast_{load|store}_N(ptr addr) {
///     i64 heap_base = (i64) __coqui_heap_base();
///     i64 shadow_base = (i64) __coqui_shadow_base();
///     i64 usable = shadow_base - heap_base;
///     i64 rel = (i64)addr - heap_base;
///     if (rel >= usable) return;                    // 95/5: likely out of heap -> skip
///     i8 sbyte = *(i8*)(shadow_base + (rel >> 3));
///     if (sbyte == 0) return;                        // 99/1: likely clean -> skip
///     __coqui_asan_slowpath_{load|store}_N(addr);
///   }
///
/// NoInline so each callsite is literally one `call` instruction — this is
/// the whole point of the port. The helper body contains the fast-path logic
/// that used to be inlined at every load/store site.
static llvm::Function *createSizedFastHelper(llvm::Module &M,
                                             llvm::StringRef HelperName,
                                             uint64_t AccessSize,
                                             llvm::StringRef SlowpathName) {
  using namespace llvm;
  LLVMContext &Ctx = M.getContext();
  Type *VoidTy = Type::getVoidTy(Ctx);
  Type *I8Ty = Type::getInt8Ty(Ctx);
  Type *I64Ty = Type::getInt64Ty(Ctx);
  Type *PtrTy = PointerType::get(Ctx, 0);

  // Runtime decls we call from the helper body.
  FunctionType *HeapBaseTy = FunctionType::get(PtrTy, false);
  FunctionCallee HeapBaseFn =
      M.getOrInsertFunction("__coqui_heap_base", HeapBaseTy);
  FunctionCallee ShadowBaseFn =
      M.getOrInsertFunction("__coqui_shadow_base", HeapBaseTy);

  FunctionType *SlowTy = FunctionType::get(VoidTy, {PtrTy}, false);
  FunctionCallee SlowFn = M.getOrInsertFunction(SlowpathName, SlowTy);

  // __coqui_heap_size() returns i32 — the usable heap size as a
  // compile-time constant (see MemoryLayout.cpp). We zext to i64 for
  // the range check. This is the SAME value the old inline fastpath
  // used as its heap_sz upper bound. Do NOT compute usable as
  // shadow_base - heap_base: heap and shadow are separate allocas in
  // the fuzz-kernel frame, so that subtraction is garbage pointer
  // arithmetic dependent on llc's alloca layout.
  Type *I32Ty = Type::getInt32Ty(Ctx);
  FunctionType *HeapSizeTy = FunctionType::get(I32Ty, false);
  FunctionCallee HeapSizeFn =
      M.getOrInsertFunction("__coqui_heap_size", HeapSizeTy);

  // Suppress unused-access-size warning: the size is reflected in the name
  // and selected by the caller; the helper body is size-agnostic (the
  // current runtime slowpath is the size-baked-in variant).
  (void)AccessSize;

  // Create the helper function.
  FunctionType *HelperTy = FunctionType::get(VoidTy, {PtrTy}, false);
  Function *F = Function::Create(HelperTy, GlobalValue::InternalLinkage,
                                 HelperName, M);
  F->setCallingConv(CallingConv::C);
  F->addFnAttr(Attribute::NoInline);
  F->getArg(0)->setName("addr");

  BasicBlock *EntryBB = BasicBlock::Create(Ctx, "entry", F);
  BasicBlock *ShadowBB = BasicBlock::Create(Ctx, "shadow", F);
  BasicBlock *SlowBB = BasicBlock::Create(Ctx, "slow", F);
  BasicBlock *OkBB = BasicBlock::Create(Ctx, "ok", F);

  MDNode *InHeapW = createBranchWeightMD(Ctx, 95, 5);
  MDNode *CleanW = createBranchWeightMD(Ctx, 99, 1);

  // --- entry: range check ---
  IRBuilder<> B(EntryBB);
  Value *HeapBase = B.CreateCall(HeapBaseFn, {}, "hb");
  Value *HeapBaseI64 = B.CreatePtrToInt(HeapBase, I64Ty, "hb.i64");
  Value *ShadowBase = B.CreateCall(ShadowBaseFn, {}, "sb");
  Value *ShadowBaseI64 = B.CreatePtrToInt(ShadowBase, I64Ty, "sb.i64");
  Value *HeapSize32 = B.CreateCall(HeapSizeFn, {}, "heap_size");
  Value *Usable = B.CreateZExt(HeapSize32, I64Ty, "usable");
  Value *AddrI64 = B.CreatePtrToInt(F->getArg(0), I64Ty, "addr.i64");
  Value *Rel = B.CreateSub(AddrI64, HeapBaseI64, "rel");
  Value *InHeap = B.CreateICmpULT(Rel, Usable, "inheap");
  auto *HeapBr = B.CreateCondBr(InHeap, ShadowBB, OkBB);
  HeapBr->setMetadata(LLVMContext::MD_prof, InHeapW);

  // --- shadow: load shadow byte ---
  B.SetInsertPoint(ShadowBB);
  Value *ShadowIdx = B.CreateLShr(Rel, ConstantInt::get(I64Ty, 3), "sidx");
  Value *ShadowAddrI64 = B.CreateAdd(ShadowBaseI64, ShadowIdx, "saddr.i64");
  Value *ShadowPtr = B.CreateIntToPtr(ShadowAddrI64, PtrTy, "sptr");
  Value *ShadowByte = B.CreateLoad(I8Ty, ShadowPtr, "sbyte");
  Value *Clean = B.CreateICmpEQ(ShadowByte, ConstantInt::get(I8Ty, 0), "clean");
  auto *CleanBr = B.CreateCondBr(Clean, OkBB, SlowBB);
  CleanBr->setMetadata(LLVMContext::MD_prof, CleanW);

  // --- slow: call slowpath (size baked into slowpath name) ---
  B.SetInsertPoint(SlowBB);
  B.CreateCall(SlowFn, {F->getArg(0)});
  B.CreateBr(OkBB);

  // --- ok: return ---
  B.SetInsertPoint(OkBB);
  B.CreateRetVoid();

  return F;
}

// ---------------------------------------------------------------------------
// RAUW helper: redirect OldName -> NewName for all call sites.
// The body of OldFn is NOT cloned (v1 simplification — the runtime provides
// __coqui_asan_malloc and __coqui_asan_free as separate definitions).
// ---------------------------------------------------------------------------
static bool rawReplace(Module &M, StringRef OldName, StringRef NewName) {
  Function *OldF = M.getFunction(OldName);
  if (!OldF || OldF->use_empty())
    return false;

  // Ensure the replacement declaration exists with the same type.
  Function *NewF = cast<Function>(
      M.getOrInsertFunction(NewName, OldF->getFunctionType()).getCallee());

  OldF->replaceAllUsesWith(NewF);
  // Remove the old declaration so it doesn't linger as an unused symbol.
  OldF->eraseFromParent();

  errs() << "[coqui-asan] RAUW " << OldName << " -> " << NewName << "\n";
  return true;
}

// ---------------------------------------------------------------------------
// Main pass entry
// ---------------------------------------------------------------------------
bool runAsan(Module &M) {
  LLVMContext &C = M.getContext();
  const DataLayout &DL = M.getDataLayout();

  Type *VoidTy = Type::getVoidTy(C);
  Type *I8Ty   = Type::getInt8Ty(C);
  Type *I8p    = PointerType::get(C, 0); // opaque ptr, addrspace 0

  (void)VoidTy;
  (void)I8p;

  // Outlined fast-path helpers, created lazily on first use.
  // Indexed as [IsStore][size-index] where size-index is log2(size): 0,1,2,3.
  Function *FastHelpers[2][4] = {{nullptr, nullptr, nullptr, nullptr},
                                 {nullptr, nullptr, nullptr, nullptr}};
  auto getFastHelper = [&](bool IsStore, unsigned Size) -> Function * {
    unsigned Idx;
    switch (Size) {
      case 1: Idx = 0; break;
      case 2: Idx = 1; break;
      case 4: Idx = 2; break;
      default: Idx = 3; break;  // 8 (or anything else folded up to 8 via selectSize)
    }
    Function *&Slot = FastHelpers[IsStore ? 1 : 0][Idx];
    if (!Slot) {
      static const char *LoadNames[4] = {
          "__coqui_asan_check_fast_load_1",
          "__coqui_asan_check_fast_load_2",
          "__coqui_asan_check_fast_load_4",
          "__coqui_asan_check_fast_load_8"};
      static const char *StoreNames[4] = {
          "__coqui_asan_check_fast_store_1",
          "__coqui_asan_check_fast_store_2",
          "__coqui_asan_check_fast_store_4",
          "__coqui_asan_check_fast_store_8"};
      static const char *LoadSlow[4] = {
          "__coqui_asan_slowpath_load_1",
          "__coqui_asan_slowpath_load_2",
          "__coqui_asan_slowpath_load_4",
          "__coqui_asan_slowpath_load_8"};
      static const char *StoreSlow[4] = {
          "__coqui_asan_slowpath_store_1",
          "__coqui_asan_slowpath_store_2",
          "__coqui_asan_slowpath_store_4",
          "__coqui_asan_slowpath_store_8"};
      static const uint64_t Sizes[4] = {1, 2, 4, 8};
      const char *Name = IsStore ? StoreNames[Idx] : LoadNames[Idx];
      const char *SlowName = IsStore ? StoreSlow[Idx] : LoadSlow[Idx];
      Slot = createSizedFastHelper(M, Name, Sizes[Idx], SlowName);
    }
    return Slot;
  };

  // Suppress unused-variable warnings for types only used implicitly.
  (void)I8Ty;

  // ── Phase 1: Instrument loads and stores ────────────────────────────────

  unsigned LoadCount = 0, StoreCount = 0, SkipCount = 0;

  for (Function &F : M) {
    if (F.isDeclaration())
      continue;

    // Skip all __coqui_* runtime functions; they must not be self-checked.
    if (F.getName().starts_with("__coqui_"))
      continue;

    // Collect candidate instructions first — cannot modify IR while
    // iterating over it.
    SmallVector<Instruction *, 64> ToInstrument;

    for (BasicBlock &BB : F) {
      for (Instruction &I : BB) {
        if (!isa<LoadInst>(I) && !isa<StoreInst>(I))
          continue;

        // Skip atomic accesses for v1.
        if (auto *LI = dyn_cast<LoadInst>(&I)) {
          if (LI->isAtomic()) { ++SkipCount; continue; }
        }
        if (auto *SI = dyn_cast<StoreInst>(&I)) {
          if (SI->isAtomic()) { ++SkipCount; continue; }
        }

        Value *Ptr = isa<LoadInst>(I)
                         ? cast<LoadInst>(I).getPointerOperand()
                         : cast<StoreInst>(I).getPointerOperand();

        // Only instrument default address space (addrspace 0 = .global /
        // heap on NVPTX).  Skip .shared (3), .const (4), .local (5), etc.
        if (Ptr->getType()->getPointerAddressSpace() != 0) {
          ++SkipCount;
          continue;
        }

        // Skip stack-derived accesses — alloca memory is not in the heap.
        if (isStackDerived(Ptr)) { ++SkipCount; continue; }

        // Skip global-derived accesses — static data is not heap.
        if (isGlobalDerived(Ptr)) { ++SkipCount; continue; }

        ToInstrument.push_back(&I);
      }
    }

    if (ToInstrument.empty())
      continue;

    for (Instruction *I : ToInstrument) {
      IRBuilder<> B(I); // inserts before I

      if (auto *LI = dyn_cast<LoadInst>(I)) {
        TypeSize TS = DL.getTypeStoreSize(LI->getType());
        if (TS.isZero() || TS.isScalable())
          continue;
        unsigned Sz = selectSize(TS.getFixedValue());
        Function *Helper = getFastHelper(/*IsStore=*/false, Sz);
        B.CreateCall(Helper, {LI->getPointerOperand()});
        ++LoadCount;
      } else {
        auto *SI = cast<StoreInst>(I);
        TypeSize TS = DL.getTypeStoreSize(SI->getValueOperand()->getType());
        if (TS.isZero() || TS.isScalable())
          continue;
        unsigned Sz = selectSize(TS.getFixedValue());
        Function *Helper = getFastHelper(/*IsStore=*/true, Sz);
        B.CreateCall(Helper, {SI->getPointerOperand()});
        ++StoreCount;
      }
    }
  }

  if (LoadCount > 0 || StoreCount > 0)
    errs() << "[coqui-asan] Instrumented " << LoadCount << " load(s) and "
           << StoreCount << " store(s) (" << SkipCount << " skipped, "
           << (LoadCount + StoreCount) << " via outlined fast path)\n";

  // ── Phase 2: RAUW allocators with bootstrap protection. ────────────────────
  //
  // Naive RAUW replaces __coqui_malloc → __coqui_asan_malloc EVERYWHERE,
  // including the call FROM __coqui_asan_malloc TO __coqui_malloc, causing
  // infinite recursion at runtime. Coqui's pattern: clone the raw allocator
  // to __coqui_*_raw, redirect asan-internal calls to the clone, THEN do the
  // global RAUW. Asan-internal functions (allocator implementations + check
  // helpers) are on a small allowlist.

  static const char *kAsanInternal[] = {
      "__coqui_asan_malloc",
      "__coqui_asan_free",
      "__coqui_asan_check_load_1",
      "__coqui_asan_check_load_2",
      "__coqui_asan_check_load_4",
      "__coqui_asan_check_load_8",
      "__coqui_asan_check_store_1",
      "__coqui_asan_check_store_2",
      "__coqui_asan_check_store_4",
      "__coqui_asan_check_store_8",
  };

  // For each allocator we'll RAUW: declare the _raw alias as a separate
  // function declaration with the same type, then redirect calls inside
  // the asan-internal functions to use _raw. The alias resolves at link
  // time to the same symbol the runtime defines as __coqui_*_raw —
  // for v1 we'll just have the runtime expose __coqui_malloc as the raw
  // implementation under both names via a thin wrapper.
  auto redirectInternalCalls = [&](StringRef OldName, StringRef RawName) {
    Function *OldF = M.getFunction(OldName);
    if (!OldF) return;
    Function *RawF = M.getFunction(RawName);
    if (!RawF) {
      RawF = cast<Function>(
        M.getOrInsertFunction(RawName, OldF->getFunctionType()).getCallee());
    }
    for (const char *InternalName : kAsanInternal) {
      Function *F = M.getFunction(InternalName);
      if (!F || F->isDeclaration()) continue;
      for (BasicBlock &BB : *F) {
        for (Instruction &I : BB) {
          auto *CI = dyn_cast<CallInst>(&I);
          if (!CI) continue;
          if (CI->getCalledFunction() == OldF) {
            CI->setCalledFunction(RawF);
          }
        }
      }
    }
  };

  redirectInternalCalls("__coqui_malloc", "__coqui_malloc_raw");
  redirectInternalCalls("__coqui_free",   "__coqui_free_raw");

  bool AllocChanged = false;
  AllocChanged |= rawReplace(M, "__coqui_malloc", "__coqui_asan_malloc");
  AllocChanged |= rawReplace(M, "__coqui_free",   "__coqui_asan_free");
  // __coqui_calloc / __coqui_realloc: left for a follow-up task (v1 scope).

  return (LoadCount > 0 || StoreCount > 0 || AllocChanged);
}

} // namespace coqui
