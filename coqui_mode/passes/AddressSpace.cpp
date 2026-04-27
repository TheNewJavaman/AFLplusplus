/*
 * AddressSpace.cpp --- promote module-level globals from AS=0 to NVPTX
 * `.global` (AS=1).
 *
 * Ported from /coqui/src/AddressSpaceTransform.cpp with the following
 * adaptations for cuAFL:
 *   - LLVM 18 opaque pointers throughout (no typed-pointer bitcasts; the
 *     legacy code already uses getValueType() for the new global so the
 *     pattern carries over without changes).
 *   - Skip __coqui_* runtime/pass-internal globals: those are bound by the
 *     host launcher via cuModuleGetGlobal + cuMemcpyHtoD and the rest of
 *     the pipeline (FuzzEntry, StaticGlobals, MemoryLayout, Asan, Coverage,
 *     ...) creates them in the default address space on purpose. Moving
 *     them would either rebind through a different symbol or risk subtle
 *     codegen differences for the host-bound symbols.
 *   - Skip thread-local globals (NVPTX has its own .local addrspace = 5
 *     and per-thread storage is already handled by StaticGlobals via the
 *     statics-pool accessor pattern).
 *   - Skip globals already in a non-default address space (someone before
 *     us put them there; do not second-guess).
 *   - Logs `[coqui] AddressSpaceTransform: promoted N global(s) from AS0
 *     to AS1`.
 *
 * Why this matters
 * ----------------
 * NVPTX maps LLVM AS=1 directly to PTX `.global`. With the global declared
 * in AS=1, ptxas emits `ld.global.*` / `st.global.*` directly. Globals left
 * in AS=0 (the LLVM "generic" addrspace) are reachable through a generic
 * pointer, so accesses go through `cvta.to.global` + a generic load that
 * the driver/runtime resolves. On read-heavy code paths (Huffman tables,
 * DCT lookup matrices, color-space transforms in libjpeg-turbo / stb_image)
 * that extra step measurably hurts throughput. See perf roadmap BW3.
 *
 * Algorithm
 * ---------
 *   1. Collect candidate globals (default AS, has body, not __coqui_*, not
 *      llvm.*, not thread-local).
 *   2. For each candidate, allocate a twin GlobalVariable in AS=1 with the
 *      same value type, linkage, alignment, section, metadata.
 *   3. Recursively rewrite each candidate's initializer so any reference
 *      to another candidate points at the AS=1 twin directly (NVPTX cannot
 *      lower addrspacecast inside a constant initializer).
 *   4. Replace all OTHER uses (instructions, metadata, surviving constant
 *      contexts in non-candidate globals) with a ConstantExpr addrspacecast
 *      from the AS=1 twin back to the original AS=0 pointer type.
 *   5. Erase the AS=0 original and rename the AS=1 twin in its place.
 */

#include "Transforms.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"

namespace coqui {

/* Recursively remap constant references from old globals (AS0) to new
 * globals (AS1). This ensures initializers that reference other promoted
 * globals point at the AS1 twin directly. NVPTX cannot lower an
 * addrspacecast inside a constant initializer, so we MUST rewrite the
 * pointee to AS1 in-place rather than leaving an addrspacecast there. */
static llvm::Constant *
remapConstant(llvm::Constant *C,
              llvm::DenseMap<llvm::GlobalVariable *, llvm::GlobalVariable *>
                  &OldToNew) {
  using namespace llvm;

  /* If this is a global we're remapping, return the new one directly.
   * Same value type, just different addrspace, so the type of `NewGV`
   * differs from `GV` only in the address-space tag. We never produce
   * an addrspacecast in initializer context — the consumer of the
   * initializer (another GlobalVariable's init or a ConstantExpr) is
   * itself constant, and any AS=0 user that survives the rewrite is
   * an instruction (handled in Phase 3 with a separate
   * RAUW-via-addrspacecast). */
  if (auto *GV = dyn_cast<GlobalVariable>(C)) {
    auto It = OldToNew.find(GV);
    if (It != OldToNew.end()) {
      GlobalVariable *NewGV = It->second;
      if (NewGV->getType() != GV->getType())
        return ConstantExpr::getAddrSpaceCast(NewGV, GV->getType());
      return NewGV;
    }
    return C;
  }

  /* ConstantExpr — may transitively reference globals via gep / ptrtoint /
   * bitcast / etc. Walk operands. */
  if (auto *CE = dyn_cast<ConstantExpr>(C)) {
    SmallVector<Constant *> NewOps;
    bool Changed = false;
    for (unsigned i = 0, e = CE->getNumOperands(); i < e; ++i) {
      Constant *Op = CE->getOperand(i);
      Constant *NewOp = remapConstant(Op, OldToNew);
      NewOps.push_back(NewOp);
      Changed |= (NewOp != Op);
    }
    if (!Changed)
      return C;
    return CE->getWithOperands(NewOps);
  }

  /* Aggregate constants (struct, array, vector). */
  if (auto *CA = dyn_cast<ConstantAggregate>(C)) {
    SmallVector<Constant *> NewOps;
    bool Changed = false;
    for (unsigned i = 0, e = CA->getNumOperands(); i < e; ++i) {
      Constant *Op = cast<Constant>(CA->getOperand(i));
      Constant *NewOp = remapConstant(Op, OldToNew);
      NewOps.push_back(NewOp);
      Changed |= (NewOp != Op);
    }
    if (!Changed)
      return C;

    if (auto *CS = dyn_cast<ConstantStruct>(C))
      return ConstantStruct::get(CS->getType(), NewOps);
    if (auto *CArr = dyn_cast<ConstantArray>(C))
      return ConstantArray::get(CArr->getType(), NewOps);
    if (isa<ConstantVector>(C))
      return ConstantVector::get(NewOps);
  }

  /* Scalar / null / undef / zeroinitializer / etc. — nothing to remap. */
  return C;
}

bool runAddressSpace(llvm::Module &M) {
  using namespace llvm;

  constexpr unsigned AS_Global = 1;

  /* Phase 0: pick candidates. */
  SmallVector<GlobalVariable *> ToProcess;
  for (GlobalVariable &GV : M.globals()) {
    if (GV.getAddressSpace() != 0)
      continue;
    if (GV.isDeclaration())
      continue;
    if (GV.isThreadLocal())
      continue;
    StringRef Name = GV.getName();
    if (Name.starts_with("llvm."))
      continue;
    /* __coqui_* runtime / pass-internal symbols are intentionally in the
     * default addrspace: many are bound by the host launcher via
     * cuModuleGetGlobal, and the rest of the pipeline expects them in
     * AS=0 (StaticGlobals pool, MemoryLayout slot pool, Asan slab/
     * descriptor table, etc.). */
    if (Name.starts_with("__coqui_"))
      continue;
    ToProcess.push_back(&GV);
  }

  if (ToProcess.empty()) {
    errs() << "[coqui] AddressSpaceTransform: promoted 0 global(s) from AS0 to AS1\n";
    return false;
  }

  /* Phase 1: create AS=1 twins with null initializers. We set initializers
   * in Phase 2 once every twin exists, so cross-references resolve to
   * twins instead of stale AS=0 originals. */
  DenseMap<GlobalVariable *, GlobalVariable *> OldToNew;
  for (GlobalVariable *GV : ToProcess) {
    auto *NewGV = new GlobalVariable(
        M, GV->getValueType(), GV->isConstant(), GV->getLinkage(),
        /*Initializer=*/nullptr, "", /*InsertBefore=*/nullptr,
        GV->getThreadLocalMode(), AS_Global, GV->isExternallyInitialized());

    NewGV->setAlignment(GV->getAlign());
    NewGV->setSection(GV->getSection());
    NewGV->copyMetadata(GV, /*Offset=*/0);

    OldToNew[GV] = NewGV;
  }

  /* Phase 2: remap initializers so cross-references point at twins. */
  for (GlobalVariable *GV : ToProcess) {
    GlobalVariable *NewGV = OldToNew[GV];
    if (GV->hasInitializer()) {
      Constant *OldInit = GV->getInitializer();
      Constant *NewInit = remapConstant(OldInit, OldToNew);
      NewGV->setInitializer(NewInit);
    }
  }

  /* Phase 3: replace remaining uses (instructions, metadata, non-candidate
   * globals' initializers) with `addrspacecast NewGV to AS=0_type`. We do
   * NOT use replaceAllUsesWith on the old global without the cast, because
   * AS=0 instruction operands cannot accept an AS=1 pointer directly.
   *
   * Note: any constant context that lands inside another candidate's
   * initializer was already handled in Phase 2 (remapConstant rewrote it
   * to point at the AS=1 twin without an addrspacecast). What remains
   * here is uses inside instructions and inside non-candidate constant
   * contexts (e.g. a __coqui_* global that happens to reference a user
   * global) — both can absorb an addrspacecast ConstantExpr cleanly. */
  for (GlobalVariable *GV : ToProcess) {
    GlobalVariable *NewGV = OldToNew[GV];
    Constant *Cast = ConstantExpr::getAddrSpaceCast(NewGV, GV->getType());
    GV->replaceAllUsesWith(Cast);

    std::string Name = GV->getName().str();
    GV->eraseFromParent();
    NewGV->setName(Name);
  }

  errs() << "[coqui] AddressSpaceTransform: promoted " << ToProcess.size()
         << " global(s) from AS0 to AS1\n";
  return true;
}

} // namespace coqui
