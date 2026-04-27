/*
 * Align.cpp --- relax over-optimistic load/store alignments to what can
 * actually be proven from the pointer's source.
 *
 * Ported from /coqui/src/AlignTransform.cpp (298 lines) with the following
 * adaptations for cuAFL:
 *   - LLVM 18 opaque pointers throughout (the legacy logic is already
 *     opaque-pointer compatible: it inspects the explicit value type on
 *     LoadInst/StoreInst and uses GEP::getResultElementType()).
 *   - Skip __coqui_* and llvm.* functions: the runtime helpers and intrinsic
 *     stubs are hand-tuned for the cuAFL ABI; relaxing alignment inside them
 *     could weaken whatever invariants their authors deliberately encoded.
 *     Other passes (Asan, Coverage, ...) follow the same convention.
 *   - Drop the legacy COQUI_ALIGN_TRAP debugging mode. That mode emitted
 *     PTX `exit;` with two host-readable globals (__coqui_align_trap_ptr,
 *     __coqui_align_trap_id) so a developer could decode a misalignment
 *     after a CUDA_ERROR_MISALIGNED_ADDRESS abort. cuAFL has no host-side
 *     wiring for those globals; the production trap path goes through
 *     __coqui_trap_with_reason instead. Re-add only if a debug need arises.
 *   - Order in CoquiPassPlugin: between runAsan and runExternalSymbol-
 *     Gatekeeper, so it sees the AS=1-promoted pointer chains from
 *     AddressSpace's globals (cross-pass info is purely SSA-graph based)
 *     and the load/store instrumentation Asan injects (Asan's own probe
 *     loads have safe alignment so the pass leaves them alone, but if any
 *     get touched they're already proven-safe). Runs BEFORE the Gatekeeper
 *     so any new trap calls / external references would still be caught;
 *     this pass adds none, but the ordering convention keeps the gatekeeper
 *     as the very last sweep.
 *
 * Why this matters
 * ----------------
 * NVPTX requires natural alignment for multi-byte loads. clang assigns
 * declared alignments based on the source-level type (e.g., a `double *`
 * gets align 8), but those guarantees are not preserved through the
 * cuAFL runtime: `__coqui_malloc` returns 8-byte alignment, the stack
 * pool is 16-byte, and pointer arithmetic / casts in user code can
 * narrow that further. When llc sees a load with align=8 that ptxas can
 * only place at a non-8-byte offset, the resulting `ld.local.b64` (or
 * `ld.global.b64`) traps the device with CUDA_ERROR_MISALIGNED_ADDRESS.
 * Relaxing the declared alignment to what we can actually prove keeps llc
 * from emitting wide loads on under-aligned addresses; it picks narrower
 * `ld.b8` sequences instead, which are slower but safe.
 *
 * This is essentially the inverse of "alignment widening" — for cuAFL we
 * cannot freely widen because we'd risk traps; we relax what the front
 * end optimistically asserted to a value we have a proof for. ptxas can
 * still WIDEN automatically when the relaxed alignment is sufficient
 * (e.g., a base of align=16 from __coqui_stack_alloc preserves wide
 * stores even after this pass since the pass keeps anything provably
 * sufficient).
 *
 * Algorithm
 * ---------
 *   For each load/store with declared alignment > 1:
 *     1. Compute proven alignment via getKnownAlignment() — recursive,
 *        memoized, depth-bounded (≤12) walk through GEP / PHI / select /
 *        bitcast / addrspacecast / inttoptr-from-ptrtoint, with terminal
 *        cases for GlobalVariable, AllocaInst, coqui allocators, kernel
 *        arguments (256-byte from cuMemAlloc), and inter-procedural
 *        argument scanning across known internal callers.
 *     2. If proven >= declared: keep (Kept counter).
 *     3. Else: setAlignment(min(declared, max(proven, 1))) (Relaxed
 *        counter). Never go below 1; we just lower the asserted value.
 *
 * Logs `[coqui] AlignTransform: kept N (proven), relaxed M load(s)+store(s)`.
 */

#include "Transforms.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <climits>
#include <numeric>

namespace coqui {

using namespace llvm;

/* Maximum call sites to scan for interprocedural argument alignment.
 * Functions with more callers than this are skipped (treated as align 1).
 * Bounds the worst-case fan-out of the recursion. */
static constexpr unsigned MaxCallSites = 100;

/* Maximum recursion depth before bailing out conservatively. PHI cycles
 * are broken via the cache sentinel below, but unrelated long pointer
 * chains (e.g., long GEP-of-GEP-of-GEP arms) are bounded by depth. */
static constexpr unsigned MaxDepth = 12;

/* Return the known minimum alignment (in bytes) of \p Ptr by tracking
 * through pointer chains. Returns 1 when nothing can be proven.
 *
 * Recognized sources of alignment:
 *   - GlobalVariable: max(8, declared alignment) — NVPTX globals are at
 *     least 8-byte aligned by ptxas regardless of LLVM's declared value.
 *   - Coqui heap allocators (__coqui_{malloc,calloc,realloc,asan_*}): 8.
 *   - Coqui stack pool (__coqui_stack_alloc): 16 (matches ((off+15)&~15)
 *     bump-pointer logic in the runtime).
 *   - Kernel entry argument pointers: 256 (cuMemAlloc returns 256-byte
 *     aligned device pointers).
 *   - Internal/private function arguments: min alignment across all call
 *     sites, bounded by MaxCallSites; falls back to 1 if any caller is
 *     indirect (the function is taken as a value rather than called).
 *   - AllocaInst: declared alignment.
 *   - GEP from aligned base: gcd(base_align, offset) for constant-offset
 *     GEPs, gcd(base_align, element_stride) for variable-index GEPs.
 *   - PHI / Select: min of all incoming values.
 *   - IntToPtr from PtrToInt: passthrough on the original pointer.
 *   - Bitcast / AddrSpaceCast: passthrough (stripPointerCasts handles
 *     these before we even look). */
static unsigned getKnownAlignment(const Value *Ptr, const DataLayout &DL,
                                  DenseMap<const Value *, unsigned> &Cache,
                                  unsigned Depth = 0) {
  if (Depth > MaxDepth)
    return 1;

  Ptr = Ptr->stripPointerCasts();

  /* Memoization. */
  auto It = Cache.find(Ptr);
  if (It != Cache.end())
    return It->second;

  /* Insert a sentinel of 1 so a recursion that revisits this Value
   * (PHI cycles, function arguments whose call site recurses back
   * through the same arg) terminates rather than spinning. The actual
   * computed result overwrites the sentinel on the way out. */
  Cache[Ptr] = 1;

  auto Memoize = [&](unsigned Result) -> unsigned {
    Cache[Ptr] = Result;
    return Result;
  };

  /* Global variables: ptxas places these at their declared alignment,
   * floored at 8 (NVPTX requires at least 8-byte alignment for globals). */
  if (auto *GV = dyn_cast<GlobalVariable>(Ptr)) {
    unsigned A = GV->getAlign().valueOrOne().value();
    return Memoize(std::max(A, 8u));
  }

  /* Allocas: trust the declared alignment. */
  if (auto *AI = dyn_cast<AllocaInst>(Ptr))
    return Memoize(AI->getAlign().value());

  /* Calls to known coqui allocators. */
  if (auto *CB = dyn_cast<CallBase>(Ptr)) {
    if (Function *Callee = CB->getCalledFunction()) {
      StringRef Name = Callee->getName();
      /* Heap allocators: 8-byte aligned (matches the runtime's per-
       * bucket alignment).  Note: cuAFL has no __coqui_aligned_alloc:
       * runHeap RAUW's aligned_alloc to __coqui_malloc. */
      if (Name == "__coqui_malloc" || Name == "__coqui_calloc" ||
          Name == "__coqui_realloc" || Name == "__coqui_malloc_raw" ||
          Name == "__coqui_asan_malloc" || Name == "__coqui_asan_calloc" ||
          Name == "__coqui_asan_realloc")
        return Memoize(8);
      /* Stack pool: 16-byte aligned ((offset + 15) & ~15). */
      if (Name == "__coqui_stack_alloc")
        return Memoize(16);
    }
  }

  /* Function arguments. */
  if (auto *Arg = dyn_cast<Argument>(Ptr)) {
    const Function *F = Arg->getParent();
    StringRef FN = F->getName();

    /* Kernel / entry function arguments: device pointers from cuMemAlloc
     * are 256-byte aligned. The cuAFL kernel ABI matches legacy here:
     * __coqui_fuzz_kernel is the .entry, __coqui_fuzz_execute is the
     * per-thread harness call. */
    if (FN == "__coqui_fuzz_kernel" || FN == "__coqui_fuzz_execute")
      if (Arg->getType()->isPointerTy())
        return Memoize(256);

    /* Interprocedural: for internal/private functions, scan all call
     * sites and take the min alignment of the actual argument at each
     * site. Bails out if the function is used as a value (not called),
     * in which case some indirect caller could pass anything. */
    if (F->hasInternalLinkage() || F->hasPrivateLinkage()) {
      unsigned NumCallers = 0;
      unsigned MinAlign = UINT_MAX;
      bool AllCallersKnown = true;

      for (const Use &U : F->uses()) {
        auto *CB = dyn_cast<CallBase>(U.getUser());
        if (!CB || !CB->isCallee(&U)) {
          /* Function used as a value — can't prove alignment. */
          AllCallersKnown = false;
          break;
        }
        if (++NumCallers > MaxCallSites) {
          AllCallersKnown = false;
          break;
        }
        Value *ActualArg = CB->getArgOperand(Arg->getArgNo());
        unsigned A = getKnownAlignment(ActualArg, DL, Cache, Depth + 1);
        MinAlign = std::min(MinAlign, A);
        if (MinAlign <= 1)
          break;
      }

      if (AllCallersKnown && NumCallers > 0 && MinAlign > 1)
        return Memoize(MinAlign);
    }
  }

  /* GEP: track base alignment through constant and variable offsets. */
  if (auto *GEP = dyn_cast<GEPOperator>(Ptr)) {
    unsigned BaseAlign =
        getKnownAlignment(GEP->getPointerOperand(), DL, Cache, Depth + 1);
    if (BaseAlign <= 1)
      return 1; /* Already cached as 1 from the sentinel. */

    if (GEP->hasAllConstantIndices()) {
      APInt Offset(DL.getPointerSizeInBits(), 0);
      if (GEP->accumulateConstantOffset(DL, Offset)) {
        uint64_t Off = Offset.getZExtValue();
        if (Off == 0)
          return Memoize(BaseAlign);
        return Memoize(std::gcd((uint64_t)BaseAlign, Off));
      }
    }

    /* Variable-index GEP: alignment is gcd(base_align, element_stride).
     * For example, GEP i64, ptr, %idx has stride 8, so if base is 8-byte
     * aligned the result is 8-byte aligned regardless of idx. */
    TypeSize Stride = DL.getTypeAllocSize(GEP->getResultElementType());
    if (!Stride.isScalable() && Stride.getFixedValue() > 0) {
      uint64_t S = Stride.getFixedValue();
      return Memoize(std::gcd((uint64_t)BaseAlign, S));
    }
    return 1;
  }

  /* PHI: conservatively take the minimum across all incoming values.
   * Sentinel of 1 in Cache prevents infinite recursion through PHI cycles. */
  if (auto *PHI = dyn_cast<PHINode>(Ptr)) {
    unsigned MinAlign = UINT_MAX;
    for (unsigned i = 0, e = PHI->getNumIncomingValues(); i < e; ++i) {
      unsigned A =
          getKnownAlignment(PHI->getIncomingValue(i), DL, Cache, Depth + 1);
      MinAlign = std::min(MinAlign, A);
      if (MinAlign <= 1)
        return 1;
    }
    return Memoize(MinAlign);
  }

  /* Select: min of both arms. */
  if (auto *Sel = dyn_cast<SelectInst>(Ptr)) {
    unsigned A1 = getKnownAlignment(Sel->getTrueValue(), DL, Cache, Depth + 1);
    unsigned A2 = getKnownAlignment(Sel->getFalseValue(), DL, Cache, Depth + 1);
    return Memoize(std::min(A1, A2));
  }

  /* IntToPtr-of-PtrToInt: round-trip casts preserve alignment of the
   * source pointer.  We don't try to track arithmetic on the integer in
   * between because that's where alignment can drop and we have no easy
   * way to bound it (would require isKnownToBeAPowerOfTwo etc.). */
  if (auto *I2P = dyn_cast<IntToPtrInst>(Ptr)) {
    if (auto *P2I = dyn_cast<PtrToIntInst>(I2P->getOperand(0))) {
      unsigned A =
          getKnownAlignment(P2I->getPointerOperand(), DL, Cache, Depth + 1);
      if (A > 1)
        return Memoize(A);
    }
  }

  return 1;
}

bool runAlign(Module &M) {
  const DataLayout &DL = M.getDataLayout();
  DenseMap<const Value *, unsigned> Cache;
  unsigned Kept = 0, Relaxed = 0;

  for (Function &F : M) {
    if (F.isDeclaration())
      continue;
    /* Skip coqui runtime + LLVM intrinsic stubs: the runtime helpers are
     * hand-written with their own alignment invariants we don't want to
     * second-guess, and intrinsics either have no body here or get lowered
     * by RejectIntrinsics / Math / Complex earlier in the pipeline. */
    StringRef FN = F.getName();
    if (FN.starts_with("__coqui_") || FN.starts_with("llvm."))
      continue;

    for (BasicBlock &BB : F) {
      for (Instruction &I : BB) {
        Value *Ptr = nullptr;
        unsigned A = 0;

        if (auto *LI = dyn_cast<LoadInst>(&I)) {
          A = LI->getAlign().value();
          Ptr = LI->getPointerOperand();
        } else if (auto *SI = dyn_cast<StoreInst>(&I)) {
          A = SI->getAlign().value();
          Ptr = SI->getPointerOperand();
        }

        if (!Ptr || A <= 1)
          continue;

        unsigned KnownAlign = getKnownAlignment(Ptr, DL, Cache);
        if (KnownAlign >= A) {
          Kept++;
          continue;
        }

        /* Relax to min(declared, max(KnownAlign, 1)). Never below 1. */
        unsigned NewAlign = std::min(A, std::max(KnownAlign, 1u));
        if (auto *LI = dyn_cast<LoadInst>(&I))
          LI->setAlignment(Align(NewAlign));
        else if (auto *SI = dyn_cast<StoreInst>(&I))
          SI->setAlignment(Align(NewAlign));
        Relaxed++;
      }
    }
  }

  errs() << "[coqui] AlignTransform: kept " << Kept << " (proven), relaxed "
         << Relaxed << " load(s)+store(s)\n";

  return Relaxed > 0;
}

} // namespace coqui
