/*
 * MemIntrinsics.cpp --- lower llvm.memcpy/memset/memmove to __coqui_* calls.
 *
 * LLVM's NVPTX backend lowers llvm.memcpy intrinsics to single-byte copy
 * loops (ld.u8 / st.u8) because it can't prove alignment. For image targets
 * (libpng: 146 memcpy intrinsics, stb_image: 57) this means pixel buffer
 * copies run at 1/8 the speed of word-aligned transfers.
 *
 * This pass replaces llvm.memcpy/memset/memmove intrinsics with calls to
 * __coqui_memcpy_fast / __coqui_memset_fast / __coqui_memmove_fast, which
 * use 8-byte word transfers for the aligned body and byte loops only for
 * head/tail.
 *
 * Runs after runLibc (which handles explicit memcpy() call sites) and before
 * Coverage/Asan (so the new call sites are visible to instrumentation).
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
  Type *VoidTy = Type::getVoidTy(C);
  Type *PtrTy = PointerType::getUnqual(C);
  Type *I64Ty = Type::getInt64Ty(C);
  Type *I32Ty = Type::getInt32Ty(C);

  FunctionType *MemcpyTy =
      FunctionType::get(PtrTy, {PtrTy, PtrTy, I64Ty}, false);
  FunctionType *MemsetTy =
      FunctionType::get(PtrTy, {PtrTy, I32Ty, I64Ty}, false);

  FunctionCallee MemcpyFn =
      M.getOrInsertFunction("__coqui_memcpy_fast", MemcpyTy);
  FunctionCallee MemmoveFn =
      M.getOrInsertFunction("__coqui_memmove_fast", MemcpyTy);
  FunctionCallee MemsetFn =
      M.getOrInsertFunction("__coqui_memset_fast", MemsetTy);

  SmallVector<Instruction *, 64> ToErase;
  unsigned CpyCount = 0, SetCount = 0, MoveCount = 0;

  for (Function &F : M) {
    if (F.isDeclaration())
      continue;
    for (BasicBlock &BB : F) {
      for (Instruction &I : BB) {
        if (auto *MC = dyn_cast<MemCpyInst>(&I)) {
          IRBuilder<> B(MC);
          Value *Dst = MC->getDest();
          Value *Src = MC->getSource();
          Value *Len = MC->getLength();
          if (Len->getType() != I64Ty)
            Len = B.CreateZExt(Len, I64Ty);
          B.CreateCall(MemcpyFn, {Dst, Src, Len});
          ToErase.push_back(MC);
          ++CpyCount;
        } else if (auto *MM = dyn_cast<MemMoveInst>(&I)) {
          IRBuilder<> B(MM);
          Value *Dst = MM->getDest();
          Value *Src = MM->getSource();
          Value *Len = MM->getLength();
          if (Len->getType() != I64Ty)
            Len = B.CreateZExt(Len, I64Ty);
          B.CreateCall(MemmoveFn, {Dst, Src, Len});
          ToErase.push_back(MM);
          ++MoveCount;
        } else if (auto *MS = dyn_cast<MemSetInst>(&I)) {
          IRBuilder<> B(MS);
          Value *Dst = MS->getDest();
          Value *Val = MS->getValue();
          Value *Len = MS->getLength();
          if (Val->getType() != I32Ty)
            Val = B.CreateZExt(Val, I32Ty);
          if (Len->getType() != I64Ty)
            Len = B.CreateZExt(Len, I64Ty);
          B.CreateCall(MemsetFn, {Dst, Val, Len});
          ToErase.push_back(MS);
          ++SetCount;
        }
      }
    }
  }

  for (Instruction *I : ToErase)
    I->eraseFromParent();

  if (CpyCount || SetCount || MoveCount)
    errs() << "[coqui-mem] Lowered " << CpyCount << " memcpy, "
           << SetCount << " memset, " << MoveCount << " memmove intrinsics"
           << " to __coqui_*_fast calls\n";

  return !ToErase.empty();
}

} // namespace coqui
