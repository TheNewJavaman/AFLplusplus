#include "Transforms.h"

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>

namespace coqui {

// Per-function alloca budget (bytes) before spilling to global memory.
// The CUDA hardware stack (--stack-size, default 8192) is shared across the
// entire call chain, so we use a conservative per-function budget.
static constexpr uint64_t STACK_BUDGET = 4096;

bool runStackSpill(llvm::Module &M) {
  using namespace llvm;

  auto &Ctx = M.getContext();
  auto &DL = M.getDataLayout();
  auto *I32Ty = Type::getInt32Ty(Ctx);
  auto *I64Ty = Type::getInt64Ty(Ctx);
  auto *I8Ty = Type::getInt8Ty(Ctx);
  auto *PtrTy = PointerType::get(Ctx, 0);
  auto *VoidTy = Type::getVoidTy(Ctx);

  // Helper: tag loads/stores reached transitively from `Root` with !nosanitize
  // so the later Asan pass skips them. Walk through the address-propagation
  // operators a pointer can flow through: GEPs, bitcasts, addrspace casts,
  // selects, PHIs. Stop at function-call boundaries (callee can't see the
  // metadata anyway).
  auto MarkNoSanitize = [&](Value *Root) {
    SmallVector<Value *, 32> Worklist;
    SmallPtrSet<Value *, 32> Seen;
    Worklist.push_back(Root);
    while (!Worklist.empty()) {
      Value *V = Worklist.pop_back_val();
      if (!Seen.insert(V).second) continue;
      for (User *U : V->users()) {
        if (auto *LI = dyn_cast<LoadInst>(U)) {
          LI->setMetadata(LLVMContext::MD_nosanitize,
                          MDNode::get(Ctx, ArrayRef<Metadata *>()));
        } else if (auto *SI = dyn_cast<StoreInst>(U)) {
          if (SI->getPointerOperand() == V)
            SI->setMetadata(LLVMContext::MD_nosanitize,
                            MDNode::get(Ctx, ArrayRef<Metadata *>()));
        } else if (isa<GetElementPtrInst>(U) || isa<BitCastInst>(U)
                || isa<AddrSpaceCastInst>(U) || isa<PHINode>(U)
                || isa<SelectInst>(U)) {
          Worklist.push_back(U);
        }
      }
    }
  };

  // First pass: check if any function has transformable allocas.
  bool HasAnyAllocas = false;
  for (Function &F : M) {
    if (F.isDeclaration())
      continue;
    StringRef Name = F.getName();
    if (Name.starts_with("__coqui_") || Name.starts_with("__asan_") ||
        Name.starts_with("__ubsan_") || Name.starts_with("__sanitizer_"))
      continue;
    for (auto &I : instructions(F)) {
      if (isa<AllocaInst>(&I)) {
        HasAnyAllocas = true;
        break;
      }
    }
    if (HasAnyAllocas)
      break;
  }

  if (!HasAnyAllocas)
    return false;

  // Declare runtime functions.
  auto *SaveTy = FunctionType::get(I32Ty, false);
  FunctionCallee SaveFn =
      M.getOrInsertFunction("__coqui_stack_save", SaveTy);

  auto *AllocTy = FunctionType::get(PtrTy, {I32Ty}, false);
  FunctionCallee AllocFn =
      M.getOrInsertFunction("__coqui_stack_alloc", AllocTy);

  auto *RestoreTy = FunctionType::get(VoidTy, {I32Ty}, false);
  FunctionCallee RestoreFn =
      M.getOrInsertFunction("__coqui_stack_restore", RestoreTy);

  bool Changed = false;
  unsigned TotalTransformed = 0;
  unsigned TotalKept = 0;

  for (Function &F : M) {
    if (F.isDeclaration())
      continue;
    StringRef Name = F.getName();
    if (Name.starts_with("__coqui_") || Name.starts_with("__asan_") ||
        Name.starts_with("__ubsan_") || Name.starts_with("__sanitizer_"))
      continue;

    // Collect all allocas with their sizes.
    struct AllocaInfo {
      AllocaInst *AI;
      uint64_t Size; // 0 for dynamic allocas
      bool IsDynamic;
    };
    SmallVector<AllocaInfo, 16> AllAllocas;

    for (auto &I : instructions(F)) {
      auto *AI = dyn_cast<AllocaInst>(&I);
      if (!AI)
        continue;

      if (!AI->isStaticAlloca()) {
        AllAllocas.push_back({AI, 0, true});
        continue;
      }

      uint64_t TypeSize = DL.getTypeAllocSize(AI->getAllocatedType());
      uint64_t TotalSize = TypeSize;
      if (AI->isArrayAllocation()) {
        if (auto *C = dyn_cast<ConstantInt>(AI->getArraySize()))
          TotalSize = TypeSize * C->getZExtValue();
      }
      AllAllocas.push_back({AI, TotalSize, false});
    }

    if (AllAllocas.empty())
      continue;

    // Separate dynamic and static allocas.
    SmallVector<AllocaInst *, 4> DynamicAllocas;
    SmallVector<std::pair<AllocaInst *, uint64_t>, 16> StaticAllocas;

    for (auto &Info : AllAllocas) {
      if (Info.IsDynamic)
        DynamicAllocas.push_back(Info.AI);
      else
        StaticAllocas.push_back({Info.AI, Info.Size});
    }

    // Sort static allocas by size ascending — small allocas get priority to
    // stay on hardware stack (more likely to be register-promoted by ptxas).
    std::stable_sort(StaticAllocas.begin(), StaticAllocas.end(),
                     [](const auto &A, const auto &B) {
                       return A.second < B.second;
                     });

    // Walk sorted allocas, keeping them on hardware stack while under budget.
    SmallVector<AllocaInst *, 16> ToSpill;
    uint64_t BudgetUsed = 0;

    for (auto &[AI, Size] : StaticAllocas) {
      if (BudgetUsed + Size <= STACK_BUDGET) {
        BudgetUsed += Size;
        TotalKept++;
      } else {
        ToSpill.push_back(AI);
      }
    }

    // Nothing to spill for this function (all fit + no dynamic allocas)?
    if (ToSpill.empty() && DynamicAllocas.empty())
      continue;

    // Compute frame layout for spilled static allocas.
    struct FrameEntry {
      AllocaInst *AI;
      uint64_t Offset;
      uint64_t Size;
    };
    SmallVector<FrameEntry, 16> FrameEntries;
    uint64_t FrameSize = 0;

    for (auto *AI : ToSpill) {
      uint64_t TypeSize = DL.getTypeAllocSize(AI->getAllocatedType());
      uint64_t TotalSize = TypeSize;
      if (AI->isArrayAllocation()) {
        if (auto *C = dyn_cast<ConstantInt>(AI->getArraySize()))
          TotalSize = TypeSize * C->getZExtValue();
      }

      // Respect alignment (minimum 8).
      Align AllocAlign = AI->getAlign();
      uint64_t AlignVal = AllocAlign.value();
      if (AlignVal < 8)
        AlignVal = 8;

      FrameSize = (FrameSize + AlignVal - 1) & ~(AlignVal - 1);
      FrameEntries.push_back({AI, FrameSize, TotalSize});
      FrameSize += TotalSize;
    }

    // Round final frame size up to 16.
    if (FrameSize > 0)
      FrameSize = (FrameSize + 15) & ~15ULL;

    // Insert save/alloc at function entry (before first non-PHI).
    BasicBlock &EntryBB = F.getEntryBlock();
    IRBuilder<> Builder(&EntryBB, EntryBB.getFirstNonPHIIt());

    // %saved_sp = call i32 @__coqui_stack_save()
    Value *SavedSP = Builder.CreateCall(SaveFn, {}, "saved_sp");

    // %frame = call ptr @__coqui_stack_alloc(i32 frameSize)
    Value *FramePtr = nullptr;
    if (FrameSize > 0) {
      FramePtr = Builder.CreateCall(
          AllocFn, {ConstantInt::get(I32Ty, FrameSize)}, "frame");
    }

    // Replace each spilled static alloca with a GEP into the frame.
    for (auto &Entry : FrameEntries) {
      IRBuilder<> GEPBuilder(Entry.AI);
      Value *Ptr = GEPBuilder.CreateGEP(
          I8Ty, FramePtr, ConstantInt::get(I64Ty, Entry.Offset),
          Entry.AI->getName() + ".spill");
      Entry.AI->replaceAllUsesWith(Ptr);
      Entry.AI->eraseFromParent();
      MarkNoSanitize(Ptr);
    }

    // For dynamic allocas: insert individual stack_alloc calls.
    for (auto *AI : DynamicAllocas) {
      IRBuilder<> DynBuilder(AI);
      uint64_t TypeSize = DL.getTypeAllocSize(AI->getAllocatedType());
      Value *ArraySize = AI->getArraySize();
      Value *TotalBytes;
      if (TypeSize == 1) {
        TotalBytes = DynBuilder.CreateTrunc(ArraySize, I32Ty, "dyn_size");
      } else {
        Value *ElemSize = ConstantInt::get(ArraySize->getType(), TypeSize);
        Value *Total = DynBuilder.CreateMul(ArraySize, ElemSize, "dyn_total");
        TotalBytes = DynBuilder.CreateTrunc(Total, I32Ty, "dyn_size");
      }
      Value *Ptr = DynBuilder.CreateCall(AllocFn, {TotalBytes},
                                         AI->getName() + ".dyn_spill");
      AI->replaceAllUsesWith(Ptr);
      AI->eraseFromParent();
      MarkNoSanitize(Ptr);
    }

    // Before each ReturnInst: insert stack_restore.
    for (auto &BB : F) {
      auto *Ret = dyn_cast<ReturnInst>(BB.getTerminator());
      if (!Ret)
        continue;
      IRBuilder<> RetBuilder(Ret);
      RetBuilder.CreateCall(RestoreFn, {SavedSP});
    }

    TotalTransformed += ToSpill.size() + DynamicAllocas.size();
    Changed = true;
  }

  if (Changed || TotalKept > 0)
    errs() << "[coqui] StackSpill: spilled " << TotalTransformed
           << " allocas to global memory pool, kept " << TotalKept
           << " on hardware stack (budget=" << STACK_BUDGET << ")\n";

  return Changed;
}

} // namespace coqui
