/*
 * Variadic.cpp --- rewrite user-defined variadic functions into non-variadic
 * __va_packed trampolines so NVPTX (no va_start/va_arg/va_end) can handle them.
 * Ported from /coqui/src/VariadicTransform.cpp.
 *
 * For each variadic function F (definition or declaration with direct call
 * sites), create F__va_packed(fixed_args..., ptr packed_args, i32 nargs):
 *   - If F has a body: clone body, lower va_start to stores into an x86_64
 *     __va_list_tag (forcing the overflow path into our packed i64 array),
 *     lower va_arg to sequential i64 loads with per-type casts, lower
 *     va_copy to a 24-byte memcpy, erase va_end.
 *   - Rewrite direct call sites to pack variadic args (int: zext/trunc,
 *     float: fpext to double then bitcast, double: bitcast, pointer:
 *     ptrtoint) into an on-stack i64 array and call F__va_packed.
 *   - Delete or stub the original F.
 *
 * Printf/sprintf families are handled by Printf.cpp / Sprintf.cpp and skipped
 * here.
 */

#include "Transforms.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/Cloning.h"

using namespace llvm;

namespace coqui {

static bool isPrintfFamily(StringRef Name) {
  return Name == "printf" || Name == "fprintf" || Name == "sprintf" ||
         Name == "snprintf" || Name == "sscanf" ||
         Name == "__isoc99_sscanf" || Name == "__coqui_printf" ||
         Name == "__coqui_fprintf" || Name == "__coqui_sprintf" ||
         Name == "__coqui_snprintf" || Name == "__coqui_sscanf";
}

static bool shouldSkip(StringRef Name) {
  return isPrintfFamily(Name) || Name.starts_with("__coqui_") ||
         Name.starts_with("__ubsan_") || Name.starts_with("__sanitizer_") ||
         Name.starts_with("__asan_") || Name.starts_with("llvm.");
}

/* Check if F has any direct call sites (CallBase where F is the callee). */
static bool hasDirectCallSites(Function *F) {
  for (auto *U : F->users())
    if (auto *CB = dyn_cast<CallBase>(U))
      if (CB->getCalledFunction() == F)
        return true;
  return false;
}

/* Pack variadic arguments from a call site into a stack-allocated i64 array. */
static Value *packArgs(IRBuilder<> &Builder, CallBase *CI,
                       unsigned FixedArgCount, unsigned &NumVarArgs) {
  auto &Ctx = CI->getContext();
  Type *I64Ty = Type::getInt64Ty(Ctx);

  unsigned TotalArgs = CI->arg_size();
  NumVarArgs = (TotalArgs > FixedArgCount) ? TotalArgs - FixedArgCount : 0;

  if (NumVarArgs == 0)
    return ConstantPointerNull::get(PointerType::getUnqual(Ctx));

  auto *ArrayTy = ArrayType::get(I64Ty, NumVarArgs);
  Value *Arr = Builder.CreateAlloca(ArrayTy, nullptr, "va.args");

  for (unsigned i = 0; i < NumVarArgs; i++) {
    Value *Arg = CI->getArgOperand(FixedArgCount + i);
    Type *ArgTy = Arg->getType();
    Value *Packed;

    if (ArgTy->isIntegerTy()) {
      unsigned Bits = ArgTy->getIntegerBitWidth();
      if (Bits < 64)
        Packed = Builder.CreateZExt(Arg, I64Ty);
      else if (Bits == 64)
        Packed = Arg;
      else
        Packed = Builder.CreateTrunc(Arg, I64Ty);
    } else if (ArgTy->isFloatTy()) {
      Value *D = Builder.CreateFPExt(Arg, Type::getDoubleTy(Ctx));
      Packed = Builder.CreateBitCast(D, I64Ty);
    } else if (ArgTy->isDoubleTy()) {
      Packed = Builder.CreateBitCast(Arg, I64Ty);
    } else if (ArgTy->isPointerTy()) {
      Packed = Builder.CreatePtrToInt(Arg, I64Ty);
    } else {
      errs() << "[coqui-variadic] warning: variadic argument of type "
             << *Arg->getType()
             << " replaced with zero (unsupported on NVPTX)\n";
      Packed = ConstantInt::get(I64Ty, 0);
    }

    Value *GEP = Builder.CreateConstInBoundsGEP2_32(ArrayTy, Arr, 0, i);
    Builder.CreateStore(Packed, GEP);
  }

  return Builder.CreateConstInBoundsGEP2_32(ArrayTy, Arr, 0, 0);
}

/* Replace @llvm.va_start with stores that initialize the x86_64
 * __va_list_tag struct to force all va_arg reads through overflow_arg_area.
 *
 * x86_64 __va_list_tag layout:
 *   offset  0: i32 gp_offset      - 48 forces overflow (>= 6*8)
 *   offset  4: i32 fp_offset      - 176 forces overflow (>= 6*8+16*8)
 *   offset  8: ptr overflow_arg_area - points at our packed i64 array
 *   offset 16: ptr reg_save_area  - null (never used)
 */
static void replaceVaStart(IntrinsicInst *VS, Value *PackedArgs) {
  auto &Ctx = VS->getContext();
  IRBuilder<> B(VS);
  Value *AP = VS->getArgOperand(0);

  B.CreateStore(ConstantInt::get(Type::getInt32Ty(Ctx), 48), AP);
  Value *FP = B.CreateConstGEP1_32(Type::getInt8Ty(Ctx), AP, 4);
  B.CreateStore(ConstantInt::get(Type::getInt32Ty(Ctx), 176), FP);
  Value *OV = B.CreateConstGEP1_32(Type::getInt8Ty(Ctx), AP, 8);
  B.CreateStore(PackedArgs, OV);
  Value *RS = B.CreateConstGEP1_32(Type::getInt8Ty(Ctx), AP, 16);
  B.CreateStore(ConstantPointerNull::get(PointerType::getUnqual(Ctx)), RS);

  VS->eraseFromParent();
}

/* Replace @llvm.va_copy with an explicit 24-byte memcpy of one
 * x86_64 __va_list_tag. Leaving va_copy for the NVPTX backend can copy
 * only 8 bytes (NVPTX va_list = pointer), leaving overflow_arg_area at
 * offset 8 uninitialized. */
static void replaceVaCopy(IntrinsicInst *VC) {
  IRBuilder<> B(VC);
  B.CreateMemCpy(VC->getArgOperand(0), Align(8), VC->getArgOperand(1),
                 Align(8), 24);
  VC->eraseFromParent();
}

static bool lowerRemainingVaCopies(Module &M) {
  SmallVector<IntrinsicInst *, 16> VaCopies;
  for (Function &F : M)
    for (BasicBlock &BB : F)
      for (Instruction &I : BB)
        if (auto *II = dyn_cast<IntrinsicInst>(&I);
            II && II->getIntrinsicID() == Intrinsic::vacopy)
          VaCopies.push_back(II);

  for (auto *VC : VaCopies)
    replaceVaCopy(VC);

  return !VaCopies.empty();
}

/* Build the non-variadic function type: same fixed params + (ptr, i32). */
static FunctionType *makePackedFnType(Function *F, LLVMContext &Ctx) {
  SmallVector<Type *> Params;
  for (unsigned i = 0; i < F->getFunctionType()->getNumParams(); i++)
    Params.push_back(F->getFunctionType()->getParamType(i));
  Params.push_back(PointerType::getUnqual(Ctx));
  Params.push_back(Type::getInt32Ty(Ctx));
  return FunctionType::get(F->getReturnType(), Params, false);
}

bool runVariadic(Module &M) {
  auto &Ctx = M.getContext();
  Type *I32Ty = Type::getInt32Ty(Ctx);
  bool Changed = false;

  /* Collect all variadic functions that need transformation:
   *   - Definitions (body needs cloning into __va_packed version).
   *   - Declarations with direct call sites (calls need rewriting).
   * Definitions without va_start still need the packed version created
   * (e.g., stubbed functions called from other modules via the packed
   * calling convention). */
  SmallVector<Function *, 16> Candidates;
  for (Function &F : M) {
    if (!F.isVarArg() || F.isIntrinsic())
      continue;
    if (shouldSkip(F.getName()))
      continue;
    if (!F.isDeclaration() || hasDirectCallSites(&F))
      Candidates.push_back(&F);
  }

  if (Candidates.empty())
    return lowerRemainingVaCopies(M);

  for (Function *F : Candidates) {
    std::string OldName = F->getName().str();
    unsigned FixedArgCount = F->getFunctionType()->getNumParams();
    bool HasBody = !F->isDeclaration();

    /* --- Get or create the packed version --- */

    std::string NewName = OldName + "__va_packed";
    FunctionType *NewFTy = makePackedFnType(F, Ctx);
    Function *NewF = M.getFunction(NewName);

    if (!NewF) {
      /* Match the original's linkage: external for cross-module visibility,
       * internal for static/internal functions (avoids duplicate defs). */
      NewF = Function::Create(NewFTy, F->getLinkage(), NewName, &M);
      NewF->setCallingConv(F->getCallingConv());
    }

    /* --- Clone body if this module defines the function --- */

    if (HasBody) {
      ValueToValueMapTy VMap;
      auto NewArgIt = NewF->arg_begin();
      for (auto &OldArg : F->args()) {
        NewArgIt->setName(OldArg.getName());
        VMap[&OldArg] = &*NewArgIt;
        ++NewArgIt;
      }
      NewArgIt->setName("__va_packed_args");
      Argument *PackedArgsParam = &*NewArgIt;
      ++NewArgIt;
      NewArgIt->setName("__va_nargs");

      SmallVector<ReturnInst *, 4> Returns;
      CloneFunctionInto(NewF, F, VMap,
                        CloneFunctionChangeType::LocalChangesOnly, Returns);

      /* Replace va intrinsics in the cloned body. */
      SmallVector<IntrinsicInst *, 4> VaStarts, VaEnds, VaCopies;
      for (auto &BB : *NewF)
        for (auto &I : BB)
          if (auto *II = dyn_cast<IntrinsicInst>(&I))
            switch (II->getIntrinsicID()) {
            case Intrinsic::vastart: VaStarts.push_back(II); break;
            case Intrinsic::vaend:   VaEnds.push_back(II); break;
            case Intrinsic::vacopy:  VaCopies.push_back(II); break;
            default: break;
            }

      for (auto *VS : VaStarts)
        replaceVaStart(VS, PackedArgsParam);
      for (auto *VE : VaEnds)
        VE->eraseFromParent();
      for (auto *VC : VaCopies)
        replaceVaCopy(VC);

      /* Replace va_arg intrinsics with sequential reads from the packed
       * i64 array. Each va_arg(ap, T) becomes:
       *   raw = load i64, packed_args[va_idx]
       *   result = cast raw to T
       *   va_idx++
       * C varargs promote float->double and char/short->int, so va_arg
       * types are always at least i32 or double-width. */
      SmallVector<VAArgInst *, 8> VaArgs;
      for (auto &BB2 : *NewF)
        for (auto &I : BB2)
          if (auto *VA = dyn_cast<VAArgInst>(&I))
            VaArgs.push_back(VA);

      if (!VaArgs.empty()) {
        Type *I64Ty = Type::getInt64Ty(Ctx);

        /* Create the index alloca at the entry block's first instruction. */
        IRBuilder<> EntryBuilder(&NewF->getEntryBlock().front());
        auto *IdxAlloca = EntryBuilder.CreateAlloca(I32Ty, nullptr, "va_idx");
        EntryBuilder.CreateStore(ConstantInt::get(I32Ty, 0), IdxAlloca);

        for (auto *VA : VaArgs) {
          IRBuilder<> B(VA);
          Value *Idx = B.CreateLoad(I32Ty, IdxAlloca, "va_idx.val");
          Value *Ptr = B.CreateGEP(I64Ty, PackedArgsParam, Idx, "va_arg.ptr");
          Value *Raw = B.CreateLoad(I64Ty, Ptr, "va_arg.raw");

          /* Cast the i64 value to the target type. */
          Type *TargetTy = VA->getType();
          Value *Result;
          if (TargetTy->isPointerTy()) {
            Result = B.CreateIntToPtr(Raw, TargetTy);
          } else if (TargetTy->isFloatingPointTy()) {
            if (TargetTy->isDoubleTy()) {
              Result = B.CreateBitCast(Raw, TargetTy);
            } else {
              /* float: stored as double in varargs (C promotion), truncate. */
              Result = B.CreateFPTrunc(
                  B.CreateBitCast(Raw, Type::getDoubleTy(Ctx)), TargetTy);
            }
          } else if (TargetTy->isIntegerTy()) {
            unsigned Bits = TargetTy->getIntegerBitWidth();
            if (Bits < 64)
              Result = B.CreateTrunc(Raw, TargetTy);
            else
              Result = Raw;
          } else {
            errs() << "[coqui-variadic] warning: va_arg of unsupported type "
                   << *TargetTy << " replaced with zero\n";
            Result = Constant::getNullValue(TargetTy);
          }

          VA->replaceAllUsesWith(Result);

          /* Increment the index after the replacement. */
          Value *NewIdx = B.CreateAdd(Idx, ConstantInt::get(I32Ty, 1));
          B.CreateStore(NewIdx, IdxAlloca);

          VA->eraseFromParent();
        }

        errs() << "[coqui-variadic] lowered " << VaArgs.size()
               << " va_arg intrinsic(s) in " << NewF->getName() << "\n";
      }
    }

    /* --- Rewrite direct call sites --- */

    SmallVector<CallBase *, 16> DirectCalls;
    for (User *U : F->users())
      if (auto *CB = dyn_cast<CallBase>(U))
        if (CB->getCalledFunction() == F)
          DirectCalls.push_back(CB);

    for (auto *CB : DirectCalls) {
      IRBuilder<> B(CB);
      unsigned NumVarArgs;
      Value *PackedPtr = packArgs(B, CB, FixedArgCount, NumVarArgs);

      SmallVector<Value *, 8> NewArgs;
      for (unsigned i = 0; i < FixedArgCount; i++)
        NewArgs.push_back(CB->getArgOperand(i));
      NewArgs.push_back(PackedPtr);
      NewArgs.push_back(ConstantInt::get(I32Ty, NumVarArgs));

      CallInst *NewCI = B.CreateCall(NewF, NewArgs);
      if (!CB->getType()->isVoidTy())
        CB->replaceAllUsesWith(NewCI);
      CB->eraseFromParent();
    }

    /* --- Stub or delete the original --- */

    if (!F->isDeclaration()) {
      if (F->use_empty()) {
        F->eraseFromParent();
      } else {
        F->deleteBody();
        BasicBlock *BB = BasicBlock::Create(Ctx, "entry", F);
        IRBuilder<> B(BB);
        if (F->getReturnType()->isVoidTy())
          B.CreateRetVoid();
        else
          B.CreateRet(Constant::getNullValue(F->getReturnType()));
      }
    }
    /* Declarations: leave as-is (linker resolves, or post-link stripping
     * handles any remaining variadic declarations). */

    errs() << "[coqui-variadic] transformed: " << OldName
           << (HasBody ? " (definition" : " (declaration")
           << ", " << DirectCalls.size() << " call sites)\n";
    Changed = true;
  }

  /* Lower any remaining va_copy intrinsics in non-variadic functions.
   * These occur when a non-variadic function (e.g., json_vprintf taking
   * va_list) uses va_copy. NVPTX's va_copy lowering only copies 8 bytes
   * (NVPTX va_list = pointer), but our emulated x86_64 __va_list_tag is
   * 24 bytes. Without explicit lowering, overflow_arg_area at offset 8
   * is left uninitialized, causing SIGSEGV when va_arg dereferences it. */
  Changed |= lowerRemainingVaCopies(M);

  return Changed;
}

} // namespace coqui
