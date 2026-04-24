/*
 * StaticGlobals.cpp --- Per-thread localization of writable global variables.
 *
 * On NVPTX every C global variable lives in .global address space and is
 * shared across all threads.  Libraries with writable statics (cJSON global
 * hooks, lcms globalContext, etc.) cause data-race false-positive ASan hits
 * and silent correctness bugs when fuzz threads execute concurrently.
 *
 * Algorithm
 * ---------
 * 1. Enumerate M.globals() and filter to qualifying candidates:
 *    - writable (!isConstant())
 *    - non-external (has an initializer / is not a pure declaration)
 *    - has real instruction uses (not dead)
 *    - not a __coqui_/llvm./sanitizer prefix internal
 *    - not in a special section / non-default address space
 *    - has at least one store use (effectively written somewhere)
 *    - not referenced by other global initializers (non-instruction users
 *      that can't be resolved through ConstantExpr chains are rejected)
 *
 * 2. Compute each candidate's offset in a per-thread globals pool, sorted
 *    by decreasing size for natural alignment packing.
 *
 * 3. For each candidate, generate an accessor function
 *    __coqui_global_X() -> ptr
 *    that computes:
 *      pool_base + tid * per_thread_size + offset
 *
 * 4. Replace every instruction use of the GlobalVariable with a call to its
 *    accessor function.
 *
 * 5. Emit @__coqui_statics_per_thread (u32 constant, ExternalLinkage) so the
 *    host launcher can read it via cuModuleGetGlobal and allocate the pool.
 *
 * Dependencies
 * ------------
 *  @__coqui_global_statics_pool_base  -- extern i8*, bound by host launcher
 *  __coqui_fuzz_tid()                 -- returns i32 thread id
 *
 * Ported from /coqui/src/StaticTransform.cpp with the following changes:
 *  - LLVM 18 opaque pointers throughout (no typed pointer bitcasts)
 *  - Pool access via extern global + tid-indexed offset instead of
 *    __coqui_stack_alloc (stack pool)
 *  - Accessor-function pattern instead of per-function entry-block GEPs
 *  - No init-function emission (initializers remain in the original globals;
 *    the host copies them into the pool before launch)
 */

#include "Transforms.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace coqui {

// ---------------------------------------------------------------------------
// Helper: hasStoreUses
//
// Return true if any path through the use-def graph starting at GV leads to a
// StoreInst that writes TO GV (or a derived pointer).  A global that is only
// loaded from is effectively read-only and doesn't need per-thread isolation.
// ---------------------------------------------------------------------------
static bool hasStoreUses(const GlobalVariable *GV) {
  SmallVector<const Value *, 16> Worklist;
  SmallPtrSet<const Value *, 16> Visited;
  Worklist.push_back(GV);

  while (!Worklist.empty()) {
    const Value *V = Worklist.pop_back_val();
    if (!Visited.insert(V).second)
      continue;

    for (const User *U : V->users()) {
      if (const auto *SI = dyn_cast<StoreInst>(U)) {
        // V is the DESTINATION of the store — that's a write.
        if (SI->getPointerOperand() == V)
          return true;
        // Otherwise V is the value being stored (its address escapes),
        // which doesn't mutate V itself.
        continue;
      }
      // Follow pointer-deriving instructions/CEs to find indirect stores.
      if (isa<GetElementPtrInst>(U) || isa<BitCastInst>(U) ||
          isa<AddrSpaceCastInst>(U) || isa<ConstantExpr>(U)) {
        Worklist.push_back(U);
        continue;
      }
      // Calls that receive the pointer as a non-readonly argument might write.
      if (const auto *CB = dyn_cast<CallBase>(U)) {
        for (unsigned i = 0; i < CB->arg_size(); i++) {
          if (CB->getArgOperand(i) == V) {
            if (!CB->onlyReadsMemory(i) && !CB->doesNotAccessMemory(i))
              return true;
            break;
          }
        }
        continue;
      }
    }
  }
  return false;
}

// ---------------------------------------------------------------------------
// Helper: canPool
//
// Return true if all non-instruction users of GV are ConstantExprs whose
// entire user chains terminate at Instructions (i.e. no other GlobalVariable
// initializer references this global directly or indirectly).
// ---------------------------------------------------------------------------
static bool canPool(const GlobalVariable *GV) {
  for (const User *U : GV->users()) {
    if (isa<Instruction>(U))
      continue;
    if (const auto *CE = dyn_cast<ConstantExpr>(U)) {
      SmallVector<const ConstantExpr *, 4> CEWorklist;
      SmallPtrSet<const ConstantExpr *, 4> CEVisited;
      CEWorklist.push_back(CE);
      bool CEOk = true;
      while (!CEWorklist.empty() && CEOk) {
        const ConstantExpr *CurCE = CEWorklist.pop_back_val();
        if (!CEVisited.insert(CurCE).second)
          continue;
        for (const User *UU : CurCE->users()) {
          if (isa<Instruction>(UU))
            continue;
          if (const auto *Nested = dyn_cast<ConstantExpr>(UU)) {
            CEWorklist.push_back(Nested);
            continue;
          }
          // A global initializer or other non-instruction user — can't pool.
          CEOk = false;
          break;
        }
      }
      if (CEOk)
        continue;
    }
    return false;
  }
  return true;
}

// ---------------------------------------------------------------------------
// Main pass entry point
// ---------------------------------------------------------------------------
bool runStaticGlobals(Module &M) {
  // Only run in fuzz mode — presence of the fuzz kernel is the signal.
  if (!M.getFunction("__coqui_fuzz_kernel"))
    return false;

  LLVMContext &C = M.getContext();
  const DataLayout &DL = M.getDataLayout();

  Type *I8Ty  = Type::getInt8Ty(C);
  Type *I32Ty = Type::getInt32Ty(C);
  Type *I64Ty = Type::getInt64Ty(C);
  Type *PtrTy = PointerType::getUnqual(C); // opaque pointer

  // ── Phase 1: Classify globals ──────────────────────────────────────────

  struct VarInfo {
    GlobalVariable *GV;
    uint64_t        Size;
    uint64_t        Offset; // filled in Phase 2
  };
  SmallVector<VarInfo> Candidates;

  // Snapshot the list — we'll add new globals (accessor helpers) during
  // Phase 3, and iterating while modifying is unsafe.
  SmallVector<GlobalVariable *> AllGlobals;
  for (GlobalVariable &GV : M.globals())
    AllGlobals.push_back(&GV);

  for (GlobalVariable *GV : AllGlobals) {
    // Must have a definition (initializer) in this module.
    if (!GV->hasInitializer() || GV->isDeclaration())
      continue;

    // Skip our own injected symbols and LLVM / sanitizer internals.
    StringRef Name = GV->getName();
    if (Name.starts_with("__coqui_") || Name.starts_with("llvm.") ||
        Name.starts_with("__ubsan_")  || Name.starts_with("__sanitizer_") ||
        Name.starts_with("__asan_"))
      continue;

    // Skip globals in special sections or externally initialised.
    if (GV->hasSection() || GV->isExternallyInitialized())
      continue;

    // Only default address space (addrspace(0) == .global on NVPTX).
    if (GV->getAddressSpace() != 0)
      continue;

    // Constants are safely shared — no per-thread copy needed.
    if (GV->isConstant())
      continue;

    // Must be actually writable somewhere (otherwise leave shared).
    if (!hasStoreUses(GV))
      continue;

    // Type must be concrete and non-zero-sized.
    Type *Ty = GV->getValueType();
    if (!Ty->isSized())
      continue;
    TypeSize TS = DL.getTypeAllocSize(Ty);
    if (TS.isScalable() || TS.isZero())
      continue;

    // Must have at least one use.
    if (GV->use_empty())
      continue;

    // Reject if referenced by other global initializers (non-CE non-inst users).
    if (!canPool(GV)) {
      errs() << "[coqui-statics] skipping " << Name
             << " (non-instruction users)\n";
      continue;
    }

    Candidates.push_back({GV, TS.getFixedValue(), 0});
  }

  if (Candidates.empty()) {
    errs() << "[coqui-statics] no writable globals to pool\n";
    return false;
  }

  // Sort by decreasing size for better alignment packing.
  llvm::sort(Candidates, [](const VarInfo &A, const VarInfo &B) {
    return A.Size > B.Size;
  });

  // ── Phase 2: Compute per-thread pool layout ────────────────────────────
  //
  // Layout: [var0_padding][var0][red_zone][var1_padding][var1][red_zone]...
  // Each variable is placed at its natural alignment (minimum 8 bytes).
  // After every variable a kAsanGlobalRedZone-byte trailing gap is reserved
  // so runAsanGlobals can register { beg = pool + tid*stride + offset,
  // user_size, total_size = user_size + kAsanGlobalRedZone } pool-kind
  // descriptors without the next variable's alignment padding ever shrinking
  // the gap to less than the shared red-zone constant. Alignment padding for
  // the next entry is added on top of the red zone (never subtracted from it),
  // so the gap between two adjacent pooled entries is at least
  // kAsanGlobalRedZone bytes of dedicated poison space.

  uint64_t CurOffset = 0;
  for (auto &Info : Candidates) {
    uint64_t A = Info.GV->getAlign().valueOrOne().value();
    if (A < 8)
      A = 8;
    CurOffset = alignTo(CurOffset, A);
    Info.Offset = CurOffset;
    CurOffset += Info.Size;
    // Reserve the trailing red zone. Next entry's alignment padding is
    // computed on top of this, so the red zone stays intact.
    CurOffset += kAsanGlobalRedZone;
  }
  // Round to 8 to keep the per-thread slab size 8-byte aligned for the
  // cuMemAlloc / cuMemsetD8 path in afl-fuzz-coqui.c.
  uint64_t TotalSize = alignTo(CurOffset, 8);

  errs() << "[coqui-statics] pooling " << Candidates.size()
         << " globals, " << TotalSize << " bytes/thread (red zone "
         << kAsanGlobalRedZone << "B per entry)\n";
  for (const auto &Info : Candidates)
    errs() << "  " << Info.GV->getName() << ": " << Info.Size << "B @ +"
           << Info.Offset << "\n";

  // ── Phase 3: Emit accessor functions ──────────────────────────────────
  //
  // For each pooled global emit:
  //   ptr __coqui_global_X() {
  //     tid      = __coqui_fuzz_tid()                  // i32
  //     tid64    = zext tid to i64
  //     pool_ptr = load @__coqui_global_statics_pool_base  // ptr
  //     stride   = per_thread_size (i64 constant)
  //     byte_off = tid64 * stride + var_offset
  //     return gep i8, pool_ptr, byte_off
  //   }
  //
  // Using ExternalLinkage so the NVPTX linker can see them if needed;
  // InternalLinkage would also work but external makes debugging easier.

  // Declare / look up shared helpers.
  FunctionCallee TidFn = M.getOrInsertFunction(
      "__coqui_fuzz_tid",
      FunctionType::get(I32Ty, /*isVarArg=*/false));

  // The pool base: an extern i8* global that the host binds at launch.
  GlobalVariable *PoolBase = M.getGlobalVariable("__coqui_global_statics_pool_base");
  if (!PoolBase) {
    PoolBase = new GlobalVariable(
        M, PtrTy, /*isConstant=*/false,
        GlobalValue::ExternalLinkage,
        /*Initializer=*/nullptr,
        "__coqui_global_statics_pool_base");
    PoolBase->setAlignment(Align(8));
  }

  // Stride constant: total per-thread size.
  Constant *StrideC = ConstantInt::get(I64Ty, TotalSize);

  DenseMap<GlobalVariable *, Function *> AccessorMap;

  for (const auto &Info : Candidates) {
    // Build a name safe for LLVM IR: replace special chars with _.
    std::string RawName = Info.GV->getName().str();
    std::string SafeName;
    SafeName.reserve(RawName.size());
    for (char Ch : RawName)
      SafeName += (Ch == '.' || Ch == '-' || Ch == '@') ? '_' : Ch;

    std::string FnName = "__coqui_global_" + SafeName;

    FunctionType *FnTy = FunctionType::get(PtrTy, /*isVarArg=*/false);
    Function *Accessor = Function::Create(
        FnTy, GlobalValue::InternalLinkage, FnName, M);
    Accessor->setDoesNotThrow();
    Accessor->addFnAttr(Attribute::AlwaysInline);

    BasicBlock *BB = BasicBlock::Create(C, "entry", Accessor);
    IRBuilder<> B(BB);

    // tid = __coqui_fuzz_tid()  [i32]
    Value *Tid   = B.CreateCall(TidFn, {}, "tid");
    // tid64 = zext tid to i64
    Value *Tid64 = B.CreateZExt(Tid, I64Ty, "tid64");
    // byte_off = tid64 * stride + var_offset
    Value *Mul   = B.CreateMul(Tid64, StrideC, "stride_off");
    Value *Off   = B.CreateAdd(Mul,
                               ConstantInt::get(I64Ty, Info.Offset),
                               "var_off");
    // pool_ptr = load @__coqui_global_statics_pool_base
    Value *Pool  = B.CreateLoad(PtrTy, PoolBase, "pool");
    // result = gep i8, pool_ptr, byte_off
    Value *Ptr   = B.CreateGEP(I8Ty, Pool, Off, SafeName + ".ptr");
    B.CreateRet(Ptr);

    AccessorMap[Info.GV] = Accessor;
  }

  // ── Phase 4: Materialise ConstantExpr users into Instructions ──────────
  //
  // Before replacing GlobalVariable uses, convert any ConstantExpr that
  // wraps the GV into equivalent instructions so Phase 5's RAUW can reach them.

  for (const auto &Info : Candidates) {
    bool Changed = true;
    while (Changed) {
      Changed = false;
      SmallVector<ConstantExpr *, 4> CEs;
      for (User *U : Info.GV->users())
        if (auto *CE = dyn_cast<ConstantExpr>(U))
          CEs.push_back(CE);

      for (ConstantExpr *CE : CEs) {
        SmallVector<Instruction *, 4> InstUsers;
        for (User *U : CE->users())
          if (auto *I = dyn_cast<Instruction>(U))
            InstUsers.push_back(I);

        for (Instruction *I : InstUsers) {
          Instruction *Repl = nullptr;
          if (CE->getOpcode() == Instruction::GetElementPtr) {
            auto *GEPOp = cast<GEPOperator>(CE);
            SmallVector<Value *> Indices(GEPOp->idx_begin(), GEPOp->idx_end());
            Repl = GetElementPtrInst::Create(
                GEPOp->getSourceElementType(),
                GEPOp->getPointerOperand(),
                Indices, "", I);
            cast<GetElementPtrInst>(Repl)->setIsInBounds(GEPOp->isInBounds());
          } else if (CE->getOpcode() == Instruction::AddrSpaceCast) {
            Repl = new AddrSpaceCastInst(CE->getOperand(0), CE->getType(),
                                         "", I);
          } else if (CE->getOpcode() == Instruction::PtrToInt) {
            Repl = new PtrToIntInst(CE->getOperand(0), CE->getType(),
                                    "", I);
          } else if (CE->getOpcode() == Instruction::IntToPtr) {
            Repl = new IntToPtrInst(CE->getOperand(0), CE->getType(),
                                    "", I);
          } else {
            continue; // Unsupported CE opcode — leave as-is.
          }
          I->replaceUsesOfWith(CE, Repl);
        }

        if (CE->use_empty()) {
          CE->destroyConstant();
          Changed = true;
          break; // Iterator invalidated — restart the outer while loop.
        }
      }
    }
  }

  // ── Phase 5: Replace instruction uses with accessor calls ─────────────
  //
  // For each function that uses a pooled global, insert a call to the
  // accessor at the use site and replace the use with the call result.
  //
  // We hoist the call to the entry block when possible (all uses in one
  // function share a single call), then RAUW.

  for (const auto &Info : Candidates) {
    GlobalVariable *GV      = Info.GV;
    Function       *Accessor = AccessorMap[GV];
    FunctionType   *FnTy    = Accessor->getFunctionType();

    // Collect (Function* -> [Use*]) mapping.
    DenseMap<Function *, SmallVector<Use *>> FuncUses;
    for (Use &U : GV->uses()) {
      if (auto *Inst = dyn_cast<Instruction>(U.getUser()))
        FuncUses[Inst->getFunction()].push_back(&U);
    }

    for (auto &[F, Uses] : FuncUses) {
      // Hoist a single call to the accessor into the entry block.
      IRBuilder<> B(&F->getEntryBlock().front());
      Value *LocalPtr = B.CreateCall(FnTy, Accessor, {}, GV->getName() + ".local");

      for (Use *U : Uses)
        U->set(LocalPtr);
    }
  }

  // ── Phase 6: Demote original globals ──────────────────────────────────
  //
  // The originals are no longer directly referenced by any instruction; make
  // them internal so the linker/ptxas can discard them.
  for (const auto &Info : Candidates)
    Info.GV->setLinkage(GlobalValue::InternalLinkage);

  // ── Phase 7: Emit @__coqui_statics_per_thread ─────────────────────────
  //
  // u32 constant read by the host via cuModuleGetGlobal to know how much
  // memory to allocate for the pool.
  auto *SizeGlobal = new GlobalVariable(
      M, I32Ty, /*isConstant=*/true,
      GlobalValue::ExternalLinkage,
      ConstantInt::get(I32Ty, static_cast<uint32_t>(TotalSize)),
      "__coqui_statics_per_thread");
  SizeGlobal->setAlignment(Align(4));

  (void)SizeGlobal; // suppress unused-variable warning if asserts are off

  // ── Phase 8: Emit ASan sidecar for pool entries ───────────────────────
  //
  // runAsanGlobals (which runs after us) reads these two globals and emits
  // pool-kind descriptors { beg = NULL, user_size, total_size, pool_offset,
  // pool_stride = per_thread_size } into the unified
  // __coqui_asan_global_descriptors table. The runtime's asan_check_global()
  // detects pool-kind descriptors (pool_stride > 0) and computes the real
  // address as pool_base + tid * pool_stride + pool_offset for comparison.
  //
  // The sidecar is internal + constant so it can be DCE'd after consumption,
  // but we keep it around — runAsanGlobals is the only consumer, and it
  // explicitly erases the sidecar once it has lifted the entries into the
  // unified descriptor table. The count global is emitted in case any
  // downstream pass wants a quick no-array-walk way to gate sidecar lookup.
  auto *PoolEntryTy = StructType::get(C, {I64Ty, I64Ty});
  SmallVector<Constant *, 16> PoolEntries;
  for (const auto &Info : Candidates) {
    PoolEntries.push_back(ConstantStruct::get(
        PoolEntryTy,
        {ConstantInt::get(I64Ty, Info.Offset),
         ConstantInt::get(I64Ty, Info.Size)}));
  }
  auto *PoolArrTy = ArrayType::get(PoolEntryTy, PoolEntries.size());
  auto *PoolArr = new GlobalVariable(
      M, PoolArrTy, /*isConstant=*/true,
      GlobalValue::InternalLinkage,
      ConstantArray::get(PoolArrTy, PoolEntries),
      kAsanPoolEntriesSymbol);
  PoolArr->setAlignment(Align(8));

  auto *PoolCountG = new GlobalVariable(
      M, I64Ty, /*isConstant=*/true,
      GlobalValue::InternalLinkage,
      ConstantInt::get(I64Ty, PoolEntries.size()),
      kAsanPoolEntryCountSymbol);
  PoolCountG->setAlignment(Align(8));

  (void)PoolArr;
  (void)PoolCountG;

  return true;
}

} // namespace coqui
