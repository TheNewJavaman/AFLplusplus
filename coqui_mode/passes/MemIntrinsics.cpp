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
 *   both src+dst align >= 8  ->  __coqui_memcpy_a8  (8-byte word loop)
 *   both src+dst align >= 4  ->  __coqui_memcpy_a4  (4-byte word loop)
 *   otherwise                ->  __coqui_memcpy      (byte loop)
 *
 * Same scheme for memset (only dst alignment matters) and memmove.
 * No runtime alignment checks --- the compiler already proved it.
 *
 * Constant-size optimization (<=64 bytes): when the length is a compile-time
 * constant, the pass emits direct LLVM IR load/store sequences instead of a
 * function call, saving ~8 PTX instructions of call overhead and the
 * per-byte loop cost.  Memmove is always lowered to a function call because
 * overlap semantics are complex.
 */

#include "Transforms.h"

#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace coqui {

/// Maximum constant size (in bytes) eligible for inline expansion.
static constexpr uint64_t kInlineThreshold = 64;

/// Emit a sequence of load/store pairs to copy \p Size bytes from \p Src to
/// \p Dst, using the widest type allowed by \p MinAlign.  The IRBuilder
/// insertion point must already be set.
static void emitInlineMemcpy(IRBuilder<> &B, Value *Dst, Value *Src,
                             uint64_t Size, unsigned MinAlign) {
  LLVMContext &C = B.getContext();
  // Pick the widest word type the alignment supports.
  unsigned WordBytes = MinAlign >= 8 ? 8 : MinAlign >= 4 ? 4 : 1;
  Type *WordTy = Type::getIntNTy(C, WordBytes * 8);

  uint64_t Off = 0;

  // Main word-aligned copies.
  while (Off + WordBytes <= Size) {
    Value *SrcGEP = B.CreateInBoundsGEP(B.getInt8Ty(), Src,
                                        B.getInt64(Off));
    Value *DstGEP = B.CreateInBoundsGEP(B.getInt8Ty(), Dst,
                                        B.getInt64(Off));
    LoadInst *L = B.CreateAlignedLoad(WordTy, SrcGEP, Align(WordBytes));
    B.CreateAlignedStore(L, DstGEP, Align(WordBytes));
    Off += WordBytes;
  }

  // Handle any remainder bytes one at a time.
  Type *I8Ty = Type::getInt8Ty(C);
  while (Off < Size) {
    Value *SrcGEP = B.CreateInBoundsGEP(I8Ty, Src, B.getInt64(Off));
    Value *DstGEP = B.CreateInBoundsGEP(I8Ty, Dst, B.getInt64(Off));
    LoadInst *L = B.CreateAlignedLoad(I8Ty, SrcGEP, Align(1));
    B.CreateAlignedStore(L, DstGEP, Align(1));
    Off++;
  }
}

/// Emit a sequence of stores to set \p Size bytes at \p Dst to \p ByteVal,
/// using the widest type allowed by \p DstAlign.
static void emitInlineMemset(IRBuilder<> &B, Value *Dst, Value *ByteVal,
                             uint64_t Size, unsigned DstAlign) {
  LLVMContext &C = B.getContext();
  unsigned WordBytes = DstAlign >= 8 ? 8 : DstAlign >= 4 ? 4 : 1;
  Type *WordTy = Type::getIntNTy(C, WordBytes * 8);

  // Replicate the byte value into the word type.
  // e.g. for i32: val = byte | (byte<<8) | (byte<<16) | (byte<<24)
  Value *WordVal;
  if (WordBytes == 1) {
    WordVal = ByteVal;
    if (WordVal->getType() != Type::getInt8Ty(C))
      WordVal = B.CreateTrunc(WordVal, Type::getInt8Ty(C));
  } else {
    // Ensure ByteVal is i8 first.
    Value *Byte8 = ByteVal;
    if (Byte8->getType() != Type::getInt8Ty(C))
      Byte8 = B.CreateTrunc(Byte8, Type::getInt8Ty(C));
    // Zero-extend to the word type.
    WordVal = B.CreateZExt(Byte8, WordTy);
    // Replicate by shifting and OR-ing.
    for (unsigned Shift = 8; Shift < WordBytes * 8; Shift *= 2) {
      Value *Shifted = B.CreateShl(WordVal, Shift);
      WordVal = B.CreateOr(WordVal, Shifted);
    }
  }

  uint64_t Off = 0;

  // Main word-aligned stores.
  while (Off + WordBytes <= Size) {
    Value *DstGEP = B.CreateInBoundsGEP(B.getInt8Ty(), Dst,
                                        B.getInt64(Off));
    B.CreateAlignedStore(WordVal, DstGEP, Align(WordBytes));
    Off += WordBytes;
  }

  // Remainder bytes.
  Value *Byte8ForRemainder = nullptr;
  Type *I8Ty = Type::getInt8Ty(C);
  while (Off < Size) {
    if (!Byte8ForRemainder) {
      Byte8ForRemainder = ByteVal;
      if (Byte8ForRemainder->getType() != I8Ty)
        Byte8ForRemainder = B.CreateTrunc(Byte8ForRemainder, I8Ty);
    }
    Value *DstGEP = B.CreateInBoundsGEP(I8Ty, Dst, B.getInt64(Off));
    B.CreateAlignedStore(Byte8ForRemainder, DstGEP, Align(1));
    Off++;
  }
}

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
  unsigned InlineCpy = 0, InlineSet = 0;

  for (Function &F : M) {
    if (F.isDeclaration())
      continue;
    for (BasicBlock &BB : F) {
      for (Instruction &I : BB) {
        if (auto *MC = dyn_cast<MemCpyInst>(&I)) {
          unsigned DstA = MC->getDestAlign().valueOrOne().value();
          unsigned SrcA = MC->getSourceAlign().valueOrOne().value();
          unsigned MinA = std::min(DstA, SrcA);

          // Constant-size inline expansion for small copies.
          if (auto *CI = dyn_cast<ConstantInt>(MC->getLength())) {
            uint64_t Size = CI->getZExtValue();
            if (Size <= kInlineThreshold) {
              IRBuilder<> B(MC);
              emitInlineMemcpy(B, MC->getDest(), MC->getSource(), Size, MinA);
              ToErase.push_back(MC);
              InlineCpy++;
              continue;
            }
          }

          // Variable size or size > threshold: fall through to function call.
          FunctionCallee Fn = MinA >= 8 ? CpyA8 : MinA >= 4 ? CpyA4 : CpyA1;
          (MinA >= 8 ? A8 : MinA >= 4 ? A4 : A1)++;

          IRBuilder<> B(MC);
          Value *Len = MC->getLength();
          if (Len->getType() != I64Ty)
            Len = B.CreateZExt(Len, I64Ty);
          B.CreateCall(Fn, {MC->getDest(), MC->getSource(), Len});
          ToErase.push_back(MC);

        } else if (auto *MM = dyn_cast<MemMoveInst>(&I)) {
          // Never inline memmove (overlap semantics).
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

          // Constant-size inline expansion for small memsets.
          if (auto *CI = dyn_cast<ConstantInt>(MS->getLength())) {
            uint64_t Size = CI->getZExtValue();
            if (Size <= kInlineThreshold) {
              IRBuilder<> B(MS);
              emitInlineMemset(B, MS->getDest(), MS->getValue(), Size, DstA);
              ToErase.push_back(MS);
              InlineSet++;
              continue;
            }
          }

          // Variable size or size > threshold: fall through to function call.
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

  if (A8 || A4 || A1 || InlineCpy || InlineSet)
    errs() << "[coqui-mem] Lowered mem intrinsics: "
           << A8 << " @align8, " << A4 << " @align4, " << A1 << " @align1"
           << " | inlined: " << InlineCpy << " memcpy, "
           << InlineSet << " memset\n";

  return !ToErase.empty();
}

} // namespace coqui
