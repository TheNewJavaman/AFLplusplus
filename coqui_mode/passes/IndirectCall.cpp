/*
 * IndirectCall.cpp --- devirtualize indirect calls into bounded dispatch
 * tables that ptxas can lower into selp/switch instructions.
 *
 * Ported from /coqui/src/IndirectCallTransform.cpp with the following
 * adaptations for cuAFL:
 *   - LLVM 18 opaque pointers throughout (function pointer comparisons
 *     compare-against-Function-pointer directly; no bitcast required).
 *   - Trap path uses cuAFL's `__coqui_trap_with_reason(u8)` (signature
 *     `void(i8)`), passing `COQUI_TRAP_DEVIRT = 13`. The legacy pipeline
 *     used a `__coqui_trap(int)` helper which does not exist in cuAFL —
 *     here `__coqui_trap` is `void(void)` and the reason-carrying variant
 *     is `__coqui_trap_with_reason(u8)`.
 *
 * Why this matters
 * ----------------
 * NVPTX has poor codegen for indirect calls regardless of source-language
 * origin. There is no branch prediction across a function-pointer dispatch,
 * the 32-thread warp serializes when callees diverge, and function-pointer
 * dispatch is essentially uncacheable on the GPU. By replacing each
 * indirect call with a direct icmp/branch chain over the address-taken
 * candidates that match the call signature, ptxas can lower the chain into
 * `selp` / `setp.eq` predicated sequences and inline the candidate bodies
 * downstream. This is a pure codegen optimization — semantics are
 * preserved as long as the candidate set is a superset of the runtime
 * possibilities, which the address-taken + signature filter guarantees.
 *
 * Algorithm
 * ---------
 *   1. Collect every indirect (non-asm) CallBase in the module.
 *   2. Collect every address-taken, non-intrinsic, non-declaration
 *      Function in the module — the candidate pool.
 *   3. Build a per-FunctionType target list: for each call site's type,
 *      enumerate candidates with a compatible signature.
 *   4. Build a store-set map: track where each candidate's address gets
 *      stored (struct field, global) so we can narrow per-call-site
 *      candidates to those whose address could plausibly reach that
 *      site's loaded callee. Conservatively keeps "Unknown" candidates.
 *   5. For each indirect call:
 *      a. If no signature-matched candidates exist -> trap.
 *      b. Otherwise narrow with the store-set; if narrowing eliminates
 *         everything (e.g., name mismatch from llvm-link struct renaming)
 *         fall back to the type-matched set.
 *      c. Emit an icmp/cond-branch chain over the candidate list, with
 *         a final trap block for the "function pointer didn't match any
 *         candidate" case. PHI the per-arm return value at the merge.
 *
 * Phase order (CoquiPassPlugin)
 * -----------------------------
 * Runs after `runExternalSymbolGatekeeper` so it sees the final set of
 * functions in the module (including any runtime stubs introduced by
 * earlier passes), and before `runAddressSpace` so AddressSpace's late
 * sweep doesn't need to reason about the new dispatch arms.
 */

#include "Transforms.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"

namespace coqui {

using namespace llvm;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Collect every indirect (non-asm) CallBase in the module.
/// A call is indirect if its callee operand is NOT a Function (i.e., it goes
/// through a function pointer loaded at runtime).
///
/// Note: CB->getCalledFunction() returns null when the callee IS a Function
/// but the FunctionType of the call site differs from the Function's type
/// (e.g., literal `{ double, double }` vs named `%struct.__coqui_cdouble`).
/// These are still direct calls to known functions — not indirect calls
/// through runtime pointers — so we must not devirtualize them.
static void collectIndirectCallSites(Module &M,
                                     SmallVectorImpl<CallBase *> &Out) {
  for (Function &F : M)
    for (inst_iterator I = inst_begin(F), E = inst_end(F); I != E; ++I)
      if (auto *CB = dyn_cast<CallBase>(&*I))
        if (!CB->getCalledFunction() && !CB->isInlineAsm() &&
            !dyn_cast<Function>(CB->getCalledOperand()))
          Out.push_back(CB);
}

/// Check if a function's type is compatible with a call site's FunctionType.
static bool isCallCompatible(FunctionType *CallFTy, FunctionType *FuncFTy) {
  if (CallFTy == FuncFTy)
    return true;
  if (CallFTy->getNumParams() != FuncFTy->getNumParams())
    return false;
  if (CallFTy->getReturnType() != FuncFTy->getReturnType())
    return false;
  for (unsigned i = 0; i < CallFTy->getNumParams(); ++i)
    if (CallFTy->getParamType(i) != FuncFTy->getParamType(i))
      return false;
  return true;
}

// ---------------------------------------------------------------------------
// Store-set analysis
//
// Narrows dispatch candidates by tracking where function addresses are stored
// and where call-site function pointers are loaded from.  If a function's
// address is only stored into field 3 of %struct.png_struct, it should only
// be a candidate for call sites that load from that same field — not for
// every call site with a matching type signature.
// ---------------------------------------------------------------------------

/// Abstract memory location for function pointer flow analysis.
/// Represents where a function address is stored or loaded from.
struct FPtrLocation {
  enum Kind { Struct, Global, Unknown } K;
  // For Struct: the struct type name and field index.
  StringRef StructName;
  unsigned FieldIndex = 0;
  // For Global: the global variable.
  GlobalVariable *GV = nullptr;

  static FPtrLocation makeStruct(StringRef Name, unsigned Idx) {
    return {Struct, Name, Idx, nullptr};
  }
  static FPtrLocation makeGlobal(GlobalVariable *G) {
    return {Global, {}, 0, G};
  }
  static FPtrLocation makeUnknown() {
    return {Unknown, {}, 0, nullptr};
  }

  bool isUnknown() const { return K == Unknown; }

  /// Strip the ".N" suffix that llvm-link appends to duplicate struct types.
  /// e.g., "struct.cmark_mem.0" -> "struct.cmark_mem"
  static StringRef canonicalStructName(StringRef Name) {
    // Find the last '.' — if what follows is all digits, strip it.
    size_t Dot = Name.rfind('.');
    if (Dot != StringRef::npos && Dot + 1 < Name.size()) {
      StringRef Suffix = Name.substr(Dot + 1);
      bool AllDigits = true;
      for (char C : Suffix)
        if (C < '0' || C > '9') { AllDigits = false; break; }
      if (AllDigits)
        return Name.substr(0, Dot);
    }
    return Name;
  }

  bool operator==(const FPtrLocation &O) const {
    if (K != O.K) return false;
    if (K == Struct)
      return canonicalStructName(StructName) ==
                 canonicalStructName(O.StructName) &&
             FieldIndex == O.FieldIndex;
    if (K == Global) return GV == O.GV;
    return true; // both Unknown
  }
};

/// Extract the abstract location from a pointer operand (of a load or store).
/// Walks through GEPs to identify struct field accesses.
static FPtrLocation classifyPointer(Value *Ptr) {
  // Check for GEP BEFORE stripping pointer casts — stripPointerCasts
  // removes zero-offset GEPs (field 0 of a struct), losing the struct
  // type information we need.
  if (auto *GEP = dyn_cast<GetElementPtrInst>(Ptr)) {
    auto *SrcTy = GEP->getSourceElementType();
    if (auto *STy = dyn_cast<StructType>(SrcTy)) {
      if (STy->hasName() && GEP->getNumIndices() >= 2) {
        // Second index is the field index (first is array index, usually 0).
        auto It = GEP->idx_begin();
        ++It; // skip array index
        if (auto *CI = dyn_cast<ConstantInt>(*It))
          return FPtrLocation::makeStruct(STy->getName(),
                                          CI->getZExtValue());
      }
    }
  }

  // ConstantExpr GEP (common in global initializers).
  if (auto *CE = dyn_cast<ConstantExpr>(Ptr)) {
    if (CE->getOpcode() == Instruction::GetElementPtr) {
      auto *GEPOp = cast<GEPOperator>(CE);
      if (auto *STy = dyn_cast<StructType>(GEPOp->getSourceElementType())) {
        if (STy->hasName() && GEPOp->getNumIndices() >= 2) {
          auto It = GEPOp->idx_begin();
          ++It;
          if (auto *CI = dyn_cast<ConstantInt>(*It))
            return FPtrLocation::makeStruct(STy->getName(),
                                            CI->getZExtValue());
        }
      }
    }
  }

  // Direct global variable.
  if (auto *GV = dyn_cast<GlobalVariable>(Ptr))
    return FPtrLocation::makeGlobal(GV);

  return FPtrLocation::makeUnknown();
}

/// Extract a Function* from a value, stripping bitcasts and constant exprs.
static Function *extractFunction(Value *V) {
  V = V->stripPointerCasts();
  return dyn_cast<Function>(V);
}

/// Build a map: Function -> set of locations where its address is stored.
/// A function with any Unknown location is conservatively a candidate
/// everywhere.
static void buildStoreMap(
    Module &M,
    const SmallVectorImpl<Function *> &AddressTaken,
    DenseMap<Function *, SmallVector<FPtrLocation, 4>> &StoreMap) {

  DenseSet<Function *> ATSet(AddressTaken.begin(), AddressTaken.end());

  for (Function &F : M) {
    for (inst_iterator I = inst_begin(F), E = inst_end(F); I != E; ++I) {
      auto *SI = dyn_cast<StoreInst>(&*I);
      if (!SI) continue;

      Function *StoredFn = extractFunction(SI->getValueOperand());
      if (!StoredFn || !ATSet.count(StoredFn))
        continue;

      FPtrLocation Loc = classifyPointer(SI->getPointerOperand());
      StoreMap[StoredFn].push_back(Loc);
    }
  }

  // Also check global initializers for function pointer stores.
  for (GlobalVariable &GV : M.globals()) {
    if (!GV.hasInitializer()) continue;
    Constant *Init = GV.getInitializer();

    // Walk struct/array initializers looking for function references.
    if (auto *CS = dyn_cast<ConstantStruct>(Init)) {
      StructType *STy = CS->getType();
      if (!STy->hasName()) continue;
      for (unsigned i = 0; i < CS->getNumOperands(); ++i) {
        Function *Fn = extractFunction(CS->getOperand(i));
        if (Fn && ATSet.count(Fn))
          StoreMap[Fn].push_back(
              FPtrLocation::makeStruct(STy->getName(), i));
      }
    }
  }

  // Functions with no identified stores get Unknown (their address was taken
  // but we couldn't trace where it goes — e.g., passed as a function arg).
  for (Function *F : AddressTaken) {
    if (!StoreMap.count(F))
      StoreMap[F].push_back(FPtrLocation::makeUnknown());
  }
}

/// Check if a function's store set could reach a given load location.
static bool couldReach(const SmallVectorImpl<FPtrLocation> &StoreLocs,
                       const FPtrLocation &LoadLoc) {
  // If the load location is unknown, any function could be the target.
  if (LoadLoc.isUnknown())
    return true;

  for (const auto &SL : StoreLocs) {
    // If ANY store location is unknown, conservatively include.
    if (SL.isUnknown())
      return true;
    if (SL == LoadLoc)
      return true;
  }
  return false;
}

// ---------------------------------------------------------------------------
// Main transform
// ---------------------------------------------------------------------------

bool runIndirectCall(Module &M) {
  SmallVector<CallBase *, 32> IndirectCalls;
  collectIndirectCallSites(M, IndirectCalls);
  if (IndirectCalls.empty()) {
    errs() << "[coqui] IndirectCall: 0 indirect call site(s) — skipping\n";
    return false;
  }

  // Collect all address-taken, non-intrinsic functions.
  SmallVector<Function *, 32> AddressTaken;
  for (Function &F : M) {
    if (F.isIntrinsic() || F.isDeclaration())
      continue;
    if (!F.hasAddressTaken())
      continue;
    AddressTaken.push_back(&F);
  }

  if (AddressTaken.empty()) {
    errs() << "[coqui] IndirectCall: " << IndirectCalls.size()
           << " indirect call(s) but no address-taken functions — skipping\n";
    return false;
  }

  errs() << "[coqui] IndirectCall: " << AddressTaken.size()
         << " address-taken function(s), " << IndirectCalls.size()
         << " indirect call site(s)\n";

  // Build store-set map for function pointer flow analysis.
  DenseMap<Function *, SmallVector<FPtrLocation, 4>> StoreMap;
  buildStoreMap(M, AddressTaken, StoreMap);

  // Count how many functions have precise (non-Unknown) locations.
  unsigned PreciseFuncs = 0;
  for (auto &[F, Locs] : StoreMap)
    if (!Locs.empty() && !Locs[0].isUnknown())
      PreciseFuncs++;
  errs() << "[coqui] IndirectCall: store-set tracked " << PreciseFuncs
         << " of " << AddressTaken.size()
         << " address-taken function(s) to precise locations\n";

  auto &Ctx = M.getContext();

  // Build per-call-type target lists (type-signature matching).
  DenseMap<FunctionType *, SmallVector<Function *, 8>> TargetsByCallType;
  for (CallBase *CB : IndirectCalls) {
    FunctionType *CallFTy = CB->getFunctionType();
    if (TargetsByCallType.count(CallFTy))
      continue;
    auto &Vec = TargetsByCallType[CallFTy];
    for (Function *F : AddressTaken)
      if (isCallCompatible(CallFTy, F->getFunctionType()))
        Vec.push_back(F);
  }

  // Print per-call-type target counts for diagnostics (before narrowing).
  {
    DenseMap<FunctionType *, unsigned> SitesPerType;
    for (CallBase *CB : IndirectCalls)
      SitesPerType[CB->getFunctionType()]++;

    unsigned TotalBranches = 0;
    unsigned MaxCandidates = 0;
    for (auto &[FTy, Targets] : TargetsByCallType) {
      unsigned Sites = SitesPerType[FTy];
      TotalBranches += Sites * Targets.size();
      if (Targets.size() > MaxCandidates)
        MaxCandidates = Targets.size();
    }
    errs() << "[coqui] IndirectCall: total devirt branches: " << TotalBranches
           << " (max " << MaxCandidates << " candidates/site)"
           << " [before store-set narrowing]\n";
  }

  // Use __coqui_trap_with_reason(COQUI_TRAP_DEVIRT=13) for the
  // "function pointer matched no candidate" path. cuAFL's runtime
  // declares this as `void __coqui_trap_with_reason(u8 reason)`.
  auto *I8Ty = Type::getInt8Ty(Ctx);
  auto *VoidTy = Type::getVoidTy(Ctx);
  auto *TrapFnTy = FunctionType::get(VoidTy, {I8Ty}, false);
  FunctionCallee CoquiTrap =
      M.getOrInsertFunction("__coqui_trap_with_reason", TrapFnTy);
  ConstantInt *TrapDevirt = ConstantInt::get(I8Ty, 13); // COQUI_TRAP_DEVIRT

  bool Changed = false;
  unsigned Devirtualized = 0;
  unsigned TotalNarrowed = 0;
  unsigned NarrowedBranches = 0;

  for (CallBase *CB : IndirectCalls) {
    FunctionType *CallFTy = CB->getFunctionType();
    auto It = TargetsByCallType.find(CallFTy);

    if (It == TargetsByCallType.end() || It->second.empty()) {
      BasicBlock *OrigBB = CB->getParent();
      OrigBB->splitBasicBlock(CB, "devirt.unreachable");
      OrigBB->getTerminator()->eraseFromParent();
      IRBuilder<> TB(OrigBB);
      TB.CreateCall(CoquiTrap, {TrapDevirt});
      TB.CreateUnreachable();
      if (!CB->use_empty())
        CB->replaceAllUsesWith(UndefValue::get(CB->getType()));
      CB->eraseFromParent();
      Changed = true;
      Devirtualized++;
      continue;
    }

    // Narrow candidates using store-set analysis.
    // Trace the callee operand to its load source and only include
    // functions whose addresses were stored into the same location.
    const auto &TypeTargets = It->second;
    FPtrLocation LoadLoc = FPtrLocation::makeUnknown();

    Value *Callee = CB->getCalledOperand();
    if (auto *LI = dyn_cast<LoadInst>(Callee->stripPointerCasts()))
      LoadLoc = classifyPointer(LI->getPointerOperand());

    SmallVector<Function *, 8> Targets;
    for (Function *F : TypeTargets) {
      auto SIt = StoreMap.find(F);
      if (SIt == StoreMap.end() || couldReach(SIt->second, LoadLoc))
        Targets.push_back(F);
    }

    unsigned Removed = TypeTargets.size() - Targets.size();
    if (Removed)
      TotalNarrowed++;

    if (Targets.empty()) {
      // All candidates narrowed away — use type-matched targets without
      // store-set narrowing as fallback.  This prevents silent devirt
      // traps when llvm-link struct renaming causes name mismatches.
      if (!TypeTargets.empty()) {
        errs() << "[coqui] IndirectCall: WARNING: store-set narrowed all "
               << TypeTargets.size() << " candidate(s) for call in "
               << CB->getFunction()->getName()
               << " — falling back to type-matched dispatch\n";
        Targets.assign(TypeTargets.begin(), TypeTargets.end());
      } else {
        // No type-matched targets at all — trap.
        BasicBlock *OrigBB = CB->getParent();
        OrigBB->splitBasicBlock(CB, "devirt.unreachable");
        OrigBB->getTerminator()->eraseFromParent();
        IRBuilder<> TB(OrigBB);
        TB.CreateCall(CoquiTrap, {TrapDevirt});
        TB.CreateUnreachable();
        if (!CB->use_empty())
          CB->replaceAllUsesWith(UndefValue::get(CB->getType()));
        CB->eraseFromParent();
        Changed = true;
        Devirtualized++;
        continue;
      }
    }

    Function *ParentFn = CB->getFunction();
    BasicBlock *OrigBB = CB->getParent();

    // Split the block at the indirect call.
    BasicBlock *MergeBB = OrigBB->splitBasicBlock(CB, "devirt.merge");
    OrigBB->getTerminator()->eraseFromParent();

    SmallVector<BasicBlock *, 32> CaseBBs;
    SmallVector<Value *, 32> CaseResults;

    BasicBlock *CurBB = OrigBB;
    for (unsigned t = 0; t < Targets.size(); ++t) {
      Function *Target = Targets[t];

      BasicBlock *CallBB = BasicBlock::Create(
          Ctx, "devirt.call." + Target->getName(), ParentFn, MergeBB);
      CaseBBs.push_back(CallBB);
      {
        IRBuilder<> B(CallBB);
        SmallVector<Value *, 8> Args;
        for (unsigned i = 0; i < CallFTy->getNumParams(); ++i)
          Args.push_back(CB->getArgOperand(i));
        Value *Result = B.CreateCall(Target, Args);
        B.CreateBr(MergeBB);
        CaseResults.push_back(Result);
      }

      BasicBlock *NextBB;
      if (t + 1 < Targets.size()) {
        NextBB = BasicBlock::Create(
            Ctx, "devirt.cmp." + Twine(t + 1), ParentFn, MergeBB);
      } else {
        NextBB = BasicBlock::Create(
            Ctx, "devirt.trap", ParentFn, MergeBB);
        IRBuilder<> TB(NextBB);
        TB.CreateCall(CoquiTrap, {TrapDevirt});
        TB.CreateUnreachable();
      }

      IRBuilder<> B(CurBB);
      Value *Cmp = B.CreateICmpEQ(Callee, Target, "devirt.eq");
      B.CreateCondBr(Cmp, CallBB, NextBB);

      CurBB = NextBB;
    }

    // PHI for the return value.
    if (!CallFTy->getReturnType()->isVoidTy() && !CB->use_empty()) {
      IRBuilder<> PB(&*MergeBB->getFirstInsertionPt());
      PHINode *Phi = PB.CreatePHI(CallFTy->getReturnType(),
                                   CaseBBs.size(), "devirt.result");
      for (unsigned i = 0; i < CaseBBs.size(); ++i)
        Phi->addIncoming(CaseResults[i], CaseBBs[i]);
      CB->replaceAllUsesWith(Phi);
    }

    CB->eraseFromParent();
    Changed = true;
    Devirtualized++;
    NarrowedBranches += Targets.size();
  }

  errs() << "[coqui] IndirectCall: devirtualized " << Devirtualized << " of "
         << IndirectCalls.size() << " indirect call site(s)"
         << " (narrowed " << TotalNarrowed << " site(s),"
         << " final branches: " << NarrowedBranches << ")\n";

  return Changed;
}

} // namespace coqui
