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
 * adjustments.  The fuzz-mode outlined fast-path helpers (range check +
 * shadow check before calling the slow path) and global/stack promotion
 * are NOT ported here — those depend on NVPTX-specific intrinsics and
 * runtime globals not yet wired into cuAFL.  The direct-call model is
 * semantically equivalent and simpler.
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

  // Declare the size-specialized check helpers.
  // Signature: void(ptr) — the runtime deduces the access size from the
  // function name (1/2/4/8 variants).
  FunctionType *CheckTy = FunctionType::get(VoidTy, {I8p}, false);

  FunctionCallee LoadCheck1  = M.getOrInsertFunction("__coqui_asan_check_load_1",  CheckTy);
  FunctionCallee LoadCheck2  = M.getOrInsertFunction("__coqui_asan_check_load_2",  CheckTy);
  FunctionCallee LoadCheck4  = M.getOrInsertFunction("__coqui_asan_check_load_4",  CheckTy);
  FunctionCallee LoadCheck8  = M.getOrInsertFunction("__coqui_asan_check_load_8",  CheckTy);
  FunctionCallee StoreCheck1 = M.getOrInsertFunction("__coqui_asan_check_store_1", CheckTy);
  FunctionCallee StoreCheck2 = M.getOrInsertFunction("__coqui_asan_check_store_2", CheckTy);
  FunctionCallee StoreCheck4 = M.getOrInsertFunction("__coqui_asan_check_store_4", CheckTy);
  FunctionCallee StoreCheck8 = M.getOrInsertFunction("__coqui_asan_check_store_8", CheckTy);

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
        FunctionCallee Helper;
        switch (Sz) {
          case 1: Helper = LoadCheck1; break;
          case 2: Helper = LoadCheck2; break;
          case 4: Helper = LoadCheck4; break;
          default: Helper = LoadCheck8; break;
        }
        B.CreateCall(Helper, {LI->getPointerOperand()});
        ++LoadCount;
      } else {
        auto *SI = cast<StoreInst>(I);
        TypeSize TS = DL.getTypeStoreSize(SI->getValueOperand()->getType());
        if (TS.isZero() || TS.isScalable())
          continue;
        unsigned Sz = selectSize(TS.getFixedValue());
        FunctionCallee Helper;
        switch (Sz) {
          case 1: Helper = StoreCheck1; break;
          case 2: Helper = StoreCheck2; break;
          case 4: Helper = StoreCheck4; break;
          default: Helper = StoreCheck8; break;
        }
        B.CreateCall(Helper, {SI->getPointerOperand()});
        ++StoreCount;
      }
    }
  }

  if (LoadCount > 0 || StoreCount > 0)
    errs() << "[coqui-asan] Instrumented " << LoadCount << " load(s) and "
           << StoreCount << " store(s) (" << SkipCount << " skipped)\n";

  // ── Phase 2: RAUW allocators (after instrumentation so the instrumented
  //             code already calls __coqui_asan_malloc/__coqui_asan_free;
  //             the RAUW here covers any remaining direct references). ──────

  bool AllocChanged = false;
  AllocChanged |= rawReplace(M, "__coqui_malloc", "__coqui_asan_malloc");
  AllocChanged |= rawReplace(M, "__coqui_free",   "__coqui_asan_free");
  // __coqui_calloc / __coqui_realloc: left for a follow-up task (v1 scope).

  return (LoadCount > 0 || StoreCount > 0 || AllocChanged);
}

} // namespace coqui
