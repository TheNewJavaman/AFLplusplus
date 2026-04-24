/*
 * Asan.cpp --- ASan instrumentation pass for heap + globals.
 *
 * Three responsibilities:
 *   1. runAsanGlobals: pad writable user globals with a 32-byte right red
 *      zone, emit a descriptor table, and inject a register call at
 *      __coqui_fuzz_kernel entry so the runtime slow path can detect OOB
 *      into global red zones via __coqui_asan_globals[] linear scan.
 *   2. RAUW allocators: __coqui_malloc  -> __coqui_asan_malloc
 *                       __coqui_free    -> __coqui_asan_free
 *   3. Instrument loads/stores: insert __coqui_asan_check_load_N /
 *      __coqui_asan_check_store_N (N = 1/2/4/8) before each heap access.
 *
 * Skips:
 *   - Functions with __coqui_* prefix (runtime helpers must not be
 *     self-instrumented).
 *   - Accesses where the stripped-base pointer is an AllocaInst (stack;
 *     not tracked by the heap shadow).
 *   - Accesses rooted at a coqui-injected global (cov/virgin/status/
 *     thread-slot pools). User globals DO get instrumented — their
 *     padded red zones are caught by the slow-path descriptor scan.
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
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
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

/// Return true if the root of the pointer chain is a __coqui_* internal
/// GlobalVariable that should not be instrumented. User globals now DO
/// get instrumented (runAsanGlobals padded them with a red zone) and
/// their accesses flow through the fast-path helper — the slow path
/// scans the descriptor table to catch global-buffer-overflow.
///
/// Coqui-injected globals (__coqui_cov_map, __coqui_virgin_map,
/// __coqui_thread_slots, __coqui_status_array, __coqui_kernel_timing, ...)
/// are skipped to avoid per-access slow-path calls they never need.
static bool isCoquiInternalGlobal(const Value *V) {
  const Value *Base = V->stripInBoundsOffsets();
  const auto *GV = dyn_cast<GlobalVariable>(Base);
  if (!GV)
    return false;
  StringRef Name = GV->getName();
  return Name.starts_with("__coqui_") || Name.starts_with("llvm.") ||
         Name.starts_with("__ubsan_") || Name.starts_with("__sanitizer_") ||
         Name.starts_with("__asan_");
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

/// Create an outlined fast-path helper (Task 2 signature with precomputed
/// heap/shadow bases):
///
///   void __coqui_asan_check_fast_{load|store}_N(ptr addr,
///                                                i64 heap_base,
///                                                i64 shadow_base) {
///     i64 usable = zext(__coqui_heap_size());   // compile-time constant
///     i64 rel = (i64)addr - heap_base;
///     if (rel >= usable) return;                 // 95/5: likely out of heap
///     i8 sbyte = *(i8*)(shadow_base + (rel >> 3));
///     if (sbyte == 0) return;                     // 99/1: likely clean
///     __coqui_asan_slowpath_{load|store}_N(addr);
///   }
///
/// NoInline — each callsite is literally one `call` instruction. The helper
/// itself must NOT be instrumented; protection relies on the `__coqui_`
/// prefix skip in the main instrumentation loop.
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

  // Slowpath declaration. heap_base/shadow_base now arrive as function
  // arguments — the caller (computeHeapContext) hoists the per-thread
  // getter calls to the entry of each instrumented function so all helper
  // calls within that function share one result.
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
  // This helper is named __coqui_asan_check_fast_* so the "skip functions
  // with __coqui_ prefix" predicate in the main instrumentation loop
  // excludes it from self-ASan. Preserve the prefix if you rename.
  // Signature: void(ptr, i64 heap_base, i64 shadow_base).
  FunctionType *HelperTy =
      FunctionType::get(VoidTy, {PtrTy, I64Ty, I64Ty}, false);
  Function *F = Function::Create(HelperTy, GlobalValue::InternalLinkage,
                                 HelperName, M);
  F->setCallingConv(CallingConv::C);
  F->addFnAttr(Attribute::NoInline);
  F->getArg(0)->setName("addr");
  F->getArg(1)->setName("heap.base");
  F->getArg(2)->setName("shadow.base");

  BasicBlock *EntryBB = BasicBlock::Create(Ctx, "entry", F);
  BasicBlock *ShadowBB = BasicBlock::Create(Ctx, "shadow", F);
  BasicBlock *SlowBB = BasicBlock::Create(Ctx, "slow", F);
  BasicBlock *OkBB = BasicBlock::Create(Ctx, "ok", F);

  MDNode *InHeapW = createBranchWeightMD(Ctx, 95, 5);
  MDNode *CleanW = createBranchWeightMD(Ctx, 99, 1);

  // --- entry: range check ---
  // Previously: out-of-heap returned clean via OkBB. Now we defer to
  // SlowBB so globals (which live outside the per-thread heap) get
  // checked against the descriptor table runAsanGlobals registered.
  // This adds a call cost to out-of-heap accesses but they're rare in
  // practice (stack/global accesses already carry their own skip at
  // instrumentation time in the main runAsan loop).
  IRBuilder<> B(EntryBB);
  Value *HeapBaseI64 = F->getArg(1);
  Value *ShadowBaseI64 = F->getArg(2);
  Value *HeapSize32 = B.CreateCall(HeapSizeFn, {}, "heap_size");
  Value *Usable = B.CreateZExt(HeapSize32, I64Ty, "usable");
  Value *AddrI64 = B.CreatePtrToInt(F->getArg(0), I64Ty, "addr.i64");
  Value *Rel = B.CreateSub(AddrI64, HeapBaseI64, "rel");
  Value *InHeap = B.CreateICmpULT(Rel, Usable, "inheap");
  auto *HeapBr = B.CreateCondBr(InHeap, ShadowBB, SlowBB);
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

/// At the entry of `F`, emit calls to __coqui_heap_base() and
/// __coqui_shadow_base() and convert to i64. Returns the pair.
///
/// These are per-thread values (per the slot-pool getters in
/// MemoryLayout.cpp). Hoisting them to function entry means LLVM will
/// compute them once per function invocation instead of once per
/// access-site helper call.
///
/// DO NOT subtract them to derive `usable` — heap and shadow are
/// separate allocas. Use __coqui_heap_size() for usable instead (it
/// inlines to a compile-time constant).
struct HeapCtx { llvm::Value *HeapBase; llvm::Value *ShadowBase; };

static HeapCtx computeHeapContext(llvm::Function &F) {
  using namespace llvm;
  Module &M = *F.getParent();
  LLVMContext &C = M.getContext();
  Type *I64 = Type::getInt64Ty(C);
  Type *PtrTy = PointerType::get(C, 0);
  FunctionType *FnTy = FunctionType::get(PtrTy, false);
  FunctionCallee HB = M.getOrInsertFunction("__coqui_heap_base", FnTy);
  FunctionCallee SB = M.getOrInsertFunction("__coqui_shadow_base", FnTy);

  Instruction *InsertPt = &*F.getEntryBlock().getFirstInsertionPt();
  IRBuilder<> B(InsertPt);
  Value *Hb = B.CreateCall(HB, {}, "asan.hb");
  Value *HbI64 = B.CreatePtrToInt(Hb, I64, "asan.hb.i64");
  Value *Sb = B.CreateCall(SB, {}, "asan.sb");
  Value *SbI64 = B.CreatePtrToInt(Sb, I64, "asan.sb.i64");
  return {HbI64, SbI64};
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

  errs() << "[coqui-asan] rauw " << OldName << " -> " << NewName << "\n";
  return true;
}

// ---------------------------------------------------------------------------
// runAsanGlobals --- pad user globals with red zones and register them
// with the runtime so the slow-path can detect OOB accesses.
//
// What happens:
//   1. Enumerate eligible globals in M: writable or constant, with an
//      initializer, default addrspace(0), NOT a __coqui_/llvm./sanitizer
//      internal, not externally initialized, not in a special section.
//   2. For each, wrap the type in { OrigTy, [RZ x i8] } so the original
//      bytes still start at offset 0. With opaque pointers, RAUW is a
//      drop-in replacement since loads/stores/GEPs already compute
//      byte offsets.
//   3. Build an internal descriptor table __coqui_asan_global_descriptors
//      listing { ptr, user_size, total_size } triples.
//   4. In __coqui_fuzz_kernel, inject a one-shot call to
//      __coqui_asan_register_globals(descriptors, count) so the runtime
//      can linear-scan this table from the slow path.
//
// Ordering: this must run BEFORE runAsan() so the padded globals show up
// in the module before the load/store instrumentation loop decides what
// is global-derived.
//
// Notes:
//  - Right-side red zone only (no left). With opaque pointers, appending
//    padding preserves RAUW correctness — offset 0 is still user data.
//  - The red zone is unpoisoned in any shadow map (globals don't live in
//    the per-thread heap). Detection is purely via the descriptor table
//    consulted in the slow path.
//  - llvm.global_ctors referencing these globals keep working because
//    the replacement uses takeName + RAUW; the initializer list only
//    stores a pointer that remains valid.
// ---------------------------------------------------------------------------
bool runAsanGlobals(Module &M) {
  LLVMContext &C = M.getContext();
  const DataLayout &DL = M.getDataLayout();
  Type *I8Ty   = Type::getInt8Ty(C);
  Type *I64Ty  = Type::getInt64Ty(C);
  Type *VoidTy = Type::getVoidTy(C);
  Type *PtrTy  = PointerType::get(C, 0);

  /* Right red-zone size in bytes. 32 = 4 shadow granules. Shared with
   * StaticGlobals via the kAsanGlobalRedZone constant in Transforms.h,
   * so a pool-kind entry's total_size = user_size + kAsanGlobalRedZone
   * (the trailing gap StaticGlobals reserved) matches exactly what a
   * non-pooled entry looks like after the type-wrap below. */
  constexpr uint64_t kRedZone = kAsanGlobalRedZone;

  /* Descriptor struct matches coqui_runtime.h. The last two fields
   * (pool_offset, pool_stride) were added additively for pooled-global
   * support. Non-pool entries emit pool_offset=0 and pool_stride=0 —
   * the runtime treats pool_stride==0 as "beg is absolute" (legacy path).
   * Pool entries emit pool_stride>0 (the per-thread slab size) and the
   * runtime computes real_beg = statics_pool_base + tid*stride + offset. */
  auto *DescTy = StructType::get(C, {PtrTy, I64Ty, I64Ty, I64Ty, I64Ty});

  struct Candidate { GlobalVariable *GV; uint64_t UserSize; };
  SmallVector<Candidate, 16> Candidates;

  /* Snapshot the globals list because we'll splice in replacements
   * during the mutation loop. */
  SmallVector<GlobalVariable *, 32> AllGlobals;
  for (GlobalVariable &GV : M.globals())
    AllGlobals.push_back(&GV);

  for (GlobalVariable *GV : AllGlobals) {
    StringRef Name = GV->getName();
    /* Skip coqui-injected globals. Broader than "__coqui_" because some
     * passes emit globals like __coqui_global_* accessors and
     * __coqui_cov_map/__coqui_virgin_map that must not be padded. */
    if (Name.starts_with("__coqui_"))
      continue;
    /* Skip LLVM / sanitizer internals. */
    if (Name.starts_with("llvm."))
      continue;
    if (Name.starts_with("__ubsan_") || Name.starts_with("__sanitizer_") ||
        Name.starts_with("__asan_"))
      continue;
    /* Skip externally-supplied globals. */
    if (!GV->hasInitializer() || GV->isDeclaration())
      continue;
    if (GV->isExternallyInitialized())
      continue;
    /* Skip globals with section attributes (e.g., nvvm annotations). */
    if (GV->hasSection())
      continue;
    /* Only default addrspace. NVPTX non-default spaces (constant,
     * shared, local) are out of scope — padding them with struct types
     * produces bitcode the NVPTX backend can't handle. */
    if (GV->getAddressSpace() != 0)
      continue;
    /* StaticGlobals demotes pooled originals to InternalLinkage but leaves
     * them in the module with their initializers intact. Their instruction
     * uses were RAUW'd through the accessor, so they're use-empty here —
     * skip them to avoid padding unreachable memory (we emit pool-kind
     * descriptors further down from the StaticGlobals sidecar instead). */
    if (GV->use_empty())
      continue;
    /* Need a known fixed size. */
    Type *Ty = GV->getValueType();
    if (!Ty->isSized())
      continue;
    TypeSize TS = DL.getTypeAllocSize(Ty);
    if (TS.isScalable() || TS.isZero())
      continue;
    uint64_t UserSize = TS.getFixedValue();
    /* Very small globals are rarely the target of buffer-overflow style
     * bugs and not worth the per-global .global memory footprint. */
    if (UserSize < 4)
      continue;

    Candidates.push_back({GV, UserSize});
  }

  /* Phase 1: pad each non-pooled candidate by wrapping its type with a
   * trailing red zone. With opaque pointers, all existing users (GEPs,
   * loads, stores) continue to work after RAUW because the original data
   * is still at byte offset 0. */
  struct Descriptor { GlobalVariable *GV; uint64_t UserSize; };
  SmallVector<Descriptor, 16> Descriptors;

  for (auto &Cand : Candidates) {
    GlobalVariable *GV = Cand.GV;
    Type *OrigTy = GV->getValueType();

    auto *PadTy = ArrayType::get(I8Ty, kRedZone);
    auto *NewTy = StructType::get(C, {OrigTy, PadTy});

    Constant *OrigInit = GV->getInitializer();
    Constant *PadInit  = ConstantAggregateZero::get(PadTy);
    Constant *NewInit  = ConstantStruct::get(NewTy, {OrigInit, PadInit});

    /* Create the replacement BEFORE the old one so link order is
     * preserved. Use the old global's linkage, tls, addrspace, and
     * constness. */
    auto *NewGV = new GlobalVariable(
        M, NewTy, GV->isConstant(), GV->getLinkage(), NewInit, "", GV,
        GV->getThreadLocalMode(), GV->getAddressSpace());
    NewGV->setAlignment(GV->getAlign());
    NewGV->takeName(GV);

    /* Opaque pointer RAUW: user data starts at offset 0 so every GEP /
     * load / store on the old pointer is valid on the new one. */
    GV->replaceAllUsesWith(NewGV);
    GV->eraseFromParent();

    Descriptors.push_back({NewGV, Cand.UserSize});
  }

  /* Phase 1b: lift the StaticGlobals pool-entry sidecar into pool-kind
   * descriptors. These share the unified __coqui_asan_global_descriptors
   * table with non-pool entries; the runtime branches on pool_stride.
   *
   * Sidecar globals (emitted by StaticGlobals when it pools anything):
   *   __coqui_asan_pool_entries : [{ i64 offset, i64 user_size } x N]
   *   __coqui_asan_pool_entry_count : i64   (== N)
   *   __coqui_statics_per_thread    : i32   (per-thread slab stride)
   *
   * StaticGlobals already inserted a kAsanGlobalRedZone-byte gap after each
   * entry's user bytes, so total_size = user_size + kAsanGlobalRedZone
   * matches the non-pool entries above. */
  struct PoolDesc { uint64_t Offset; uint64_t UserSize; };
  SmallVector<PoolDesc, 16> PoolEntries;
  uint64_t PoolStride = 0;
  /* AllowInternal=true — the sidecar StaticGlobals emits has internal
   * linkage (so DCE is free to drop it after we consume it), and
   * getGlobalVariable() defaults to AllowInternal=false which would skip
   * it. */
  if (GlobalVariable *SideGV =
          M.getGlobalVariable(kAsanPoolEntriesSymbol,
                              /*AllowInternal=*/true)) {
    if (GlobalVariable *StrideGV =
            M.getGlobalVariable(kAsanPoolStrideSymbol,
                                /*AllowInternal=*/true)) {
      if (auto *StrideInit =
              dyn_cast_or_null<ConstantInt>(StrideGV->getInitializer())) {
        PoolStride = StrideInit->getZExtValue();
      }
    }
    if (auto *Init = dyn_cast_or_null<ConstantArray>(SideGV->getInitializer())) {
      for (unsigned i = 0; i < Init->getNumOperands(); i++) {
        auto *CS = dyn_cast<ConstantStruct>(Init->getOperand(i));
        if (!CS) continue;
        auto *OffC  = dyn_cast<ConstantInt>(CS->getOperand(0));
        auto *SizeC = dyn_cast<ConstantInt>(CS->getOperand(1));
        if (!OffC || !SizeC) continue;
        PoolEntries.push_back(
            {OffC->getZExtValue(), SizeC->getZExtValue()});
      }
    }
    /* We're the only consumer of the sidecar; erase it so it doesn't
     * bloat the emitted PTX with an unused constant array. The count
     * sidecar is also removed for the same reason. */
    SideGV->eraseFromParent();
    if (GlobalVariable *CntGV =
            M.getGlobalVariable(kAsanPoolEntryCountSymbol,
                                /*AllowInternal=*/true)) {
      CntGV->eraseFromParent();
    }
  }

  if (Descriptors.empty() && PoolEntries.empty())
    return false;

  /* Phase 2: build the unified descriptor table. */
  SmallVector<Constant *, 32> Entries;
  Constant *NullPtr = ConstantPointerNull::get(PointerType::get(C, 0));
  Constant *Zero64  = ConstantInt::get(I64Ty, 0);

  /* Non-pool entries: beg = &padded_global, pool_offset = 0, pool_stride = 0. */
  for (auto &D : Descriptors) {
    uint64_t TotalSize =
        DL.getTypeAllocSize(D.GV->getValueType()).getFixedValue();
    Constant *Entry = ConstantStruct::get(
        DescTy,
        {D.GV,
         ConstantInt::get(I64Ty, D.UserSize),
         ConstantInt::get(I64Ty, TotalSize),
         Zero64, Zero64});
    Entries.push_back(Entry);
  }

  /* Pool entries: beg = NULL (unused), pool_offset/stride identify the
   * per-thread location. The runtime computes real_beg as
   *   __coqui_global_statics_pool_base + tid * pool_stride + pool_offset
   * in asan_check_global(). */
  for (auto &P : PoolEntries) {
    Constant *Entry = ConstantStruct::get(
        DescTy,
        {NullPtr,
         ConstantInt::get(I64Ty, P.UserSize),
         ConstantInt::get(I64Ty, P.UserSize + kRedZone),
         ConstantInt::get(I64Ty, P.Offset),
         ConstantInt::get(I64Ty, PoolStride)});
    Entries.push_back(Entry);
  }

  auto *TableTy = ArrayType::get(DescTy, Entries.size());
  auto *Table = new GlobalVariable(
      M, TableTy, /*isConstant=*/true,
      GlobalValue::InternalLinkage, ConstantArray::get(TableTy, Entries),
      "__coqui_asan_global_descriptors");

  /* Phase 3: emit a call to __coqui_asan_register_globals(table, count)
   * at the start of __coqui_fuzz_kernel so the runtime has the table
   * before any user code runs. The call is idempotent (all threads
   * write identical descriptors at identical indices), so we can just
   * inject it unconditionally — no tid==0 gate required. Pool-kind
   * descriptors carry their pool-base-relative form so they remain
   * identical across threads too (the per-thread variation happens
   * inside asan_check_global()). */
  Function *Kernel = M.getFunction("__coqui_fuzz_kernel");
  if (!Kernel || Kernel->isDeclaration()) {
    errs() << "[coqui-asan-globals] no __coqui_fuzz_kernel; "
           << (Descriptors.size() + PoolEntries.size())
           << " global(s) padded but unregistered\n";
    return true;
  }

  auto *RegTy = FunctionType::get(VoidTy, {PtrTy, I64Ty}, false);
  FunctionCallee RegFn =
      M.getOrInsertFunction("__coqui_asan_register_globals", RegTy);

  /* Insert as the very first instruction of the kernel entry block so
   * the descriptor table is populated before MemoryInit, before any
   * user code. */
  Instruction *InsertPt = &*Kernel->getEntryBlock().getFirstInsertionPt();
  IRBuilder<> B(InsertPt);
  B.CreateCall(RegFn, {Table, ConstantInt::get(I64Ty, Entries.size())});

  errs() << "[coqui-asan-globals] padded "
         << (Descriptors.size() + PoolEntries.size()) << " global(s) ("
         << Descriptors.size() << " non-pooled + " << PoolEntries.size()
         << " pooled) with " << kRedZone << "-byte right red zones\n";

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

        // Skip accesses rooted in a coqui-injected global (cov map,
        // virgin map, thread slots, etc.). User globals DO get
        // instrumented — runAsanGlobals padded them with red zones and
        // the slow path scans the descriptor table to detect global OOB.
        if (isCoquiInternalGlobal(Ptr)) { ++SkipCount; continue; }

        ToInstrument.push_back(&I);
      }
    }

    if (ToInstrument.empty())
      continue;

    // Hoist __coqui_heap_base() / __coqui_shadow_base() to the entry of this
    // function so every helper callsite below shares one materialization of
    // each (instead of re-evaluating the per-thread getters in every helper
    // body call).
    HeapCtx Ctx = computeHeapContext(F);

    for (Instruction *I : ToInstrument) {
      IRBuilder<> B(I); // inserts before I

      if (auto *LI = dyn_cast<LoadInst>(I)) {
        TypeSize TS = DL.getTypeStoreSize(LI->getType());
        if (TS.isZero() || TS.isScalable())
          continue;
        unsigned Sz = selectSize(TS.getFixedValue());
        Function *Helper = getFastHelper(/*IsStore=*/false, Sz);
        B.CreateCall(Helper,
                     {LI->getPointerOperand(), Ctx.HeapBase, Ctx.ShadowBase});
        ++LoadCount;
      } else {
        auto *SI = cast<StoreInst>(I);
        TypeSize TS = DL.getTypeStoreSize(SI->getValueOperand()->getType());
        if (TS.isZero() || TS.isScalable())
          continue;
        unsigned Sz = selectSize(TS.getFixedValue());
        Function *Helper = getFastHelper(/*IsStore=*/true, Sz);
        B.CreateCall(Helper,
                     {SI->getPointerOperand(), Ctx.HeapBase, Ctx.ShadowBase});
        ++StoreCount;
      }
    }
  }

  if (LoadCount > 0 || StoreCount > 0)
    errs() << "[coqui-asan] instrumented " << LoadCount << " load(s) and "
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
      "__coqui_asan_check_fast_load_1",
      "__coqui_asan_check_fast_load_2",
      "__coqui_asan_check_fast_load_4",
      "__coqui_asan_check_fast_load_8",
      "__coqui_asan_check_fast_store_1",
      "__coqui_asan_check_fast_store_2",
      "__coqui_asan_check_fast_store_4",
      "__coqui_asan_check_fast_store_8",
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
