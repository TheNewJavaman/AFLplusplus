/*
 * MemIntrinsics.cpp --- lower llvm.memcpy/memset/memmove to __coqui_* calls.
 *
 * LLVM's NVPTX backend lowers llvm.memcpy intrinsics to single-byte copy
 * loops (ld.u8 / st.u8) because it conservatively assumes unknown alignment.
 * For image targets (libpng: 106 memcpy sites, stb_image: 57) this means
 * pixel buffer copies run at 1/8 the speed of word-aligned transfers.
 *
 * This pass reads the alignment metadata that LLVM attaches to each
 * intrinsic and dispatches to alignment-specialized runtime functions:
 *
 *   both src+dst align ≥ 8  →  __coqui_memcpy_a8  (8-byte word loop)
 *   both src+dst align ≥ 4  →  __coqui_memcpy_a4  (4-byte word loop)
 *   otherwise               →  __coqui_memcpy      (byte loop)
 *
 * Same scheme for memset (only dst alignment matters) and memmove.
 * No runtime alignment checks — the compiler already proved it.
 */

#include "Transforms.h"

#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace coqui {

bool runMemIntrinsics(Module &M) {
  LLVMContext &C = M.getContext();
  Type *PtrTy = PointerType::getUnqual(C);
  Type *I64Ty = Type::getInt64Ty(C);
  Type *I32Ty = Type::getInt32Ty(C);

  FunctionType *CpyTy =
      FunctionType::get(PtrTy, {PtrTy, PtrTy, I64Ty}, false);
  FunctionType *SetTy =
      FunctionType::get(PtrTy, {PtrTy, I32Ty, I64Ty}, false);

  // memcpy variants by alignment
  FunctionCallee CpyA8 = M.getOrInsertFunction("__coqui_memcpy_a8", CpyTy);
  FunctionCallee CpyA4 = M.getOrInsertFunction("__coqui_memcpy_a4", CpyTy);
  FunctionCallee CpyA1 = M.getOrInsertFunction("__coqui_memcpy", CpyTy);

  // memmove variants
  FunctionCallee MovA8 = M.getOrInsertFunction("__coqui_memmove_a8", CpyTy);
  FunctionCallee MovA4 = M.getOrInsertFunction("__coqui_memmove_a4", CpyTy);
  FunctionCallee MovA1 = M.getOrInsertFunction("__coqui_memmove", CpyTy);

  // memset variants (only dst alignment matters)
  FunctionCallee SetA8 = M.getOrInsertFunction("__coqui_memset_a8", SetTy);
  FunctionCallee SetA4 = M.getOrInsertFunction("__coqui_memset_a4", SetTy);
  FunctionCallee SetA1 = M.getOrInsertFunction("__coqui_memset", SetTy);

  SmallVector<Instruction *, 64> ToErase;
  unsigned A8 = 0, A4 = 0, A1 = 0;

  for (Function &F : M) {
    if (F.isDeclaration())
      continue;
    for (BasicBlock &BB : F) {
      for (Instruction &I : BB) {
        if (auto *MC = dyn_cast<MemCpyInst>(&I)) {
          unsigned DstA = MC->getDestAlign().valueOrOne().value();
          unsigned SrcA = MC->getSourceAlign().valueOrOne().value();
          unsigned MinA = std::min(DstA, SrcA);

          FunctionCallee Fn = MinA >= 8 ? CpyA8 : MinA >= 4 ? CpyA4 : CpyA1;
          (MinA >= 8 ? A8 : MinA >= 4 ? A4 : A1)++;

          IRBuilder<> B(MC);
          Value *Len = MC->getLength();
          if (Len->getType() != I64Ty)
            Len = B.CreateZExt(Len, I64Ty);
          B.CreateCall(Fn, {MC->getDest(), MC->getSource(), Len});
          ToErase.push_back(MC);

        } else if (auto *MM = dyn_cast<MemMoveInst>(&I)) {
          unsigned DstA = MM->getDestAlign().valueOrOne().value();
          unsigned SrcA = MM->getSourceAlign().valueOrOne().value();
          unsigned MinA = std::min(DstA, SrcA);

          FunctionCallee Fn = MinA >= 8 ? MovA8 : MinA >= 4 ? MovA4 : MovA1;
          (MinA >= 8 ? A8 : MinA >= 4 ? A4 : A1)++;

          IRBuilder<> B(MM);
          Value *Len = MM->getLength();
          if (Len->getType() != I64Ty)
            Len = B.CreateZExt(Len, I64Ty);
          B.CreateCall(Fn, {MM->getDest(), MM->getSource(), Len});
          ToErase.push_back(MM);

        } else if (auto *MS = dyn_cast<MemSetInst>(&I)) {
          unsigned DstA = MS->getDestAlign().valueOrOne().value();

          FunctionCallee Fn = DstA >= 8 ? SetA8 : DstA >= 4 ? SetA4 : SetA1;
          (DstA >= 8 ? A8 : DstA >= 4 ? A4 : A1)++;

          IRBuilder<> B(MS);
          Value *Val = MS->getValue();
          Value *Len = MS->getLength();
          if (Val->getType() != I32Ty)
            Val = B.CreateZExt(Val, I32Ty);
          if (Len->getType() != I64Ty)
            Len = B.CreateZExt(Len, I64Ty);
          B.CreateCall(Fn, {MS->getDest(), Val, Len});
          ToErase.push_back(MS);
        }
      }
    }
  }

  for (Instruction *I : ToErase)
    I->eraseFromParent();

  if (A8 || A4 || A1)
    errs() << "[coqui-mem] Lowered mem intrinsics: "
           << A8 << " @align8, " << A4 << " @align4, " << A1 << " @align1\n";

  return !ToErase.empty();
}

} // namespace coqui
