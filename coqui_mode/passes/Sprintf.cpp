/* Sprintf.cpp --- rewrite sprintf/snprintf call sites to non-variadic
 * __coqui_*_impl calls with packed i64 argument arrays.
 * Ported from /coqui/src/SprintfTransform.cpp.
 *
 * NVPTX has no va_start/va_arg support, so variadic calls must be
 * eliminated before lowering. This pass handles the sprintf family
 * (sprintf, snprintf, and their fortified _chk wrappers) by packing
 * every variadic argument into a tagged {i64 tag, i64 value} slot and
 * dispatching to the device-side format engine. The matching runtime
 * cores (__coqui_sprintf_impl / __coqui_snprintf_impl) live in
 * coqui_printf.c / coqui_printf.inc.
 *
 * sscanf is handled identically by coqui's original transform but is
 * dropped here: coqui mode's narrowed runtime scope covers buffer writers
 * only, not the scanf input parser.
 */

#include "Transforms.h"

#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"

namespace coqui {

using namespace llvm;

/// Type tags used by the device-side format engine to interpret packed args.
/// Must match the constants in coqui_runtime_common.inc / coqui_printf.inc.
enum ArgTag : uint64_t {
  TAG_INT = 0,
  TAG_UINT = 1,
  TAG_LONG = 2,
  TAG_ULONG = 3,
  TAG_LLONG = 4,
  TAG_ULLONG = 5,
  TAG_DOUBLE = 6,
  TAG_PTR = 7,
};

/// Determine the tag for a given LLVM type.
static ArgTag classifyType(Type *T) {
  if (T->isPointerTy())
    return TAG_PTR;
  if (T->isDoubleTy() || T->isFloatTy())
    return TAG_DOUBLE;
  if (T->isIntegerTy()) {
    unsigned Bits = T->getIntegerBitWidth();
    if (Bits <= 32)
      return TAG_INT;
    return TAG_LLONG;
  }
  return TAG_ULONG;
}

/// Pack variadic arguments (starting at index VarArgStart) into a stack-
/// allocated i64 array of {tag, value} pairs. Returns the array pointer and
/// the number of variadic arguments packed.
static std::pair<Value *, unsigned>
packVariadicArgs(IRBuilder<> &Builder, CallBase *CI, unsigned VarArgStart) {
  auto &Ctx = CI->getContext();
  Type *I64Ty = Type::getInt64Ty(Ctx);
  Type *I32Ty = Type::getInt32Ty(Ctx);

  unsigned TotalArgs = CI->arg_size();
  unsigned NumVarArgs =
      (TotalArgs > VarArgStart) ? TotalArgs - VarArgStart : 0;
  unsigned ArrayLen = NumVarArgs * 2;

  Value *ArgsArray;
  if (ArrayLen > 0) {
    ArgsArray = Builder.CreateAlloca(I64Ty, ConstantInt::get(I32Ty, ArrayLen),
                                     "fmt.args");
  } else {
    ArgsArray = ConstantPointerNull::get(PointerType::getUnqual(Ctx));
  }

  for (unsigned i = 0; i < NumVarArgs; i++) {
    Value *Arg = CI->getArgOperand(VarArgStart + i);
    Type *ArgTy = Arg->getType();
    ArgTag Tag = classifyType(ArgTy);

    // Store tag.
    Value *TagPtr =
        Builder.CreateGEP(I64Ty, ArgsArray, ConstantInt::get(I64Ty, i * 2));
    Builder.CreateStore(ConstantInt::get(I64Ty, Tag), TagPtr);

    // Convert value to i64.
    Value *Val64;
    if (ArgTy->isPointerTy()) {
      Val64 = Builder.CreatePtrToInt(Arg, I64Ty);
    } else if (ArgTy->isDoubleTy()) {
      Val64 = Builder.CreateBitCast(Arg, I64Ty);
    } else if (ArgTy->isFloatTy()) {
      Value *D = Builder.CreateFPExt(Arg, Type::getDoubleTy(Ctx));
      Val64 = Builder.CreateBitCast(D, I64Ty);
    } else if (ArgTy->isIntegerTy()) {
      unsigned Bits = ArgTy->getIntegerBitWidth();
      if (Bits < 64)
        Val64 = Builder.CreateSExt(Arg, I64Ty);
      else if (Bits == 64)
        Val64 = Arg;
      else
        Val64 = Builder.CreateTrunc(Arg, I64Ty);
    } else {
      errs() << "[coqui-sprintf] warning: sprintf variadic argument of type "
             << *Arg->getType()
             << " replaced with zero (unsupported on NVPTX)\n";
      Val64 = ConstantInt::get(I64Ty, 0);
    }

    // Store value.
    Value *ValPtr = Builder.CreateGEP(I64Ty, ArgsArray,
                                      ConstantInt::get(I64Ty, i * 2 + 1));
    Builder.CreateStore(Val64, ValPtr);
  }

  return {ArgsArray, NumVarArgs};
}

/// Rewrite all call sites of a variadic function to call a non-variadic
/// implementation that takes a packed argument array.
///
/// For sprintf(buf, fmt, ...):
///   -> __coqui_sprintf_impl(buf, fmt, i64* args, i32 nargs)
///
/// For snprintf(buf, size, fmt, ...):
///   -> __coqui_snprintf_impl(buf, size, fmt, i64* args, i32 nargs)
static bool rewriteVariadicCalls(Module &M, const char *OldName,
                                 const char *ImplName,
                                 unsigned NumFixedArgs) {
  Function *F = M.getFunction(OldName);
  if (!F || F->use_empty())
    return false;

  auto &Ctx = M.getContext();
  Type *I32Ty = Type::getInt32Ty(Ctx);
  Type *PtrTy = PointerType::getUnqual(Ctx);

  // Build the implementation function type: fixed_args..., ptr args, i32 nargs
  SmallVector<Type *> ImplParamTypes;
  for (unsigned i = 0; i < NumFixedArgs; i++) {
    if (i < F->getFunctionType()->getNumParams())
      ImplParamTypes.push_back(F->getFunctionType()->getParamType(i));
    else
      ImplParamTypes.push_back(PtrTy);
  }
  ImplParamTypes.push_back(PtrTy); // args array
  ImplParamTypes.push_back(I32Ty); // nargs

  FunctionType *ImplTy = FunctionType::get(F->getReturnType(), ImplParamTypes,
                                           /*isVarArg=*/false);
  FunctionCallee Impl = M.getOrInsertFunction(ImplName, ImplTy);

  SmallVector<CallBase *> CallSites;
  for (User *U : F->users()) {
    if (auto *CB = dyn_cast<CallBase>(U))
      if (CB->getCalledFunction() == F)
        CallSites.push_back(CB);
  }

  if (CallSites.empty())
    return false;

  errs() << "[coqui-sprintf] rewriting " << CallSites.size() << " calls to "
         << OldName << " -> " << ImplName << "\n";

  for (CallBase *CB : CallSites) {
    IRBuilder<> Builder(CB);

    auto [ArgsArray, NumVarArgs] =
        packVariadicArgs(Builder, CB, NumFixedArgs);

    SmallVector<Value *> ImplArgs;
    for (unsigned i = 0; i < NumFixedArgs; i++)
      ImplArgs.push_back(CB->getArgOperand(i));
    ImplArgs.push_back(ArgsArray);
    ImplArgs.push_back(ConstantInt::get(I32Ty, NumVarArgs));

    Value *Result = Builder.CreateCall(Impl, ImplArgs);
    CB->replaceAllUsesWith(Result);
    // For InvokeInst: replace with unconditional branch to normal dest.
    if (auto *II = dyn_cast<InvokeInst>(CB)) {
      BranchInst::Create(II->getNormalDest(), II);
    }
    CB->eraseFromParent();
  }

  if (F->use_empty())
    F->eraseFromParent();

  return true;
}

/// Find and rewrite internal variadic wrappers that clang generates for
/// fortified sprintf/snprintf. These have mangled names like
/// _ZL7sprintfPcU17pass_object_size1PKcz and are variadic functions that
/// internally call __vsprintf_chk / __vsnprintf_chk via va_list.
///
/// We intercept at the call sites of these wrappers, where the variadic
/// arguments are still explicit, and rewrite them to call our non-variadic
/// implementation directly.
static bool rewriteFortifiedWrappers(Module &M) {
  bool Changed = false;

  struct WrapperInfo {
    Function *Wrapper;
    bool IsSprintf; // true = sprintf, false = snprintf
  };
  SmallVector<WrapperInfo> Wrappers;

  for (Function &F : M) {
    if (!F.isVarArg() || !F.hasInternalLinkage())
      continue;

    bool FoundSprintf = false, FoundSnprintf = false;
    for (auto &BB : F) {
      for (auto &I : BB) {
        if (auto *CI = dyn_cast<CallInst>(&I)) {
          Function *Callee = CI->getCalledFunction();
          if (!Callee)
            continue;
          if (Callee->getName() == "__vsprintf_chk")
            FoundSprintf = true;
          else if (Callee->getName() == "__vsnprintf_chk")
            FoundSnprintf = true;
        }
      }
    }

    if (FoundSprintf)
      Wrappers.push_back({&F, true});
    else if (FoundSnprintf)
      Wrappers.push_back({&F, false});
  }

  for (auto &[Wrapper, IsSprintf] : Wrappers) {
    auto &Ctx = M.getContext();
    Type *I64Ty = Type::getInt64Ty(Ctx);
    Type *I32Ty = Type::getInt32Ty(Ctx);
    Type *PtrTy = PointerType::getUnqual(Ctx);

    // Build impl function type.
    SmallVector<Type *> ImplParamTypes;
    const char *ImplName;
    if (IsSprintf) {
      // __coqui_sprintf_impl(ptr buf, ptr fmt, ptr args, i32 nargs)
      ImplParamTypes = {PtrTy, PtrTy, PtrTy, I32Ty};
      ImplName = "__coqui_sprintf_impl";
    } else {
      // __coqui_snprintf_impl(ptr buf, i64 size, ptr fmt, ptr args, i32 nargs)
      ImplParamTypes = {PtrTy, I64Ty, PtrTy, PtrTy, I32Ty};
      ImplName = "__coqui_snprintf_impl";
    }

    FunctionType *ImplTy = FunctionType::get(Type::getInt32Ty(Ctx),
                                             ImplParamTypes, false);
    FunctionCallee Impl = M.getOrInsertFunction(ImplName, ImplTy);

    SmallVector<CallBase *> CallSites;
    for (User *U : Wrapper->users()) {
      if (auto *CB = dyn_cast<CallBase>(U))
        if (CB->getCalledFunction() == Wrapper)
          CallSites.push_back(CB);
    }

    if (CallSites.empty())
      continue;

    errs() << "[coqui-sprintf] rewriting " << CallSites.size()
           << " fortified calls via " << Wrapper->getName() << " -> "
           << ImplName << "\n";

    unsigned NumFixed = Wrapper->getFunctionType()->getNumParams();

    for (CallBase *CB : CallSites) {
      IRBuilder<> Builder(CB);

      // Pack the variadic args.
      auto [ArgsArray, NumVarArgs] =
          packVariadicArgs(Builder, CB, NumFixed);

      // Extract the relevant fixed args:
      // - buf is always arg 0
      // - fmt is always the last fixed arg
      // - for snprintf, size is one of the middle args (find i64 arg)
      Value *Buf = CB->getArgOperand(0);
      Value *Fmt = CB->getArgOperand(NumFixed - 1);

      SmallVector<Value *> ImplArgs;
      if (IsSprintf) {
        ImplArgs = {Buf, Fmt, ArgsArray,
                    ConstantInt::get(I32Ty, NumVarArgs)};
      } else {
        // Find the size arg (i64 type, not the first arg which is buf).
        // In _ZL8snprintfPcU17pass_object_size1mPKcz the args are
        // (ptr buf, i64 objsize, i64 n, ptr fmt, ...).
        // The snprintf size limit 'n' is typically arg index 2.
        Value *Size = nullptr;
        for (unsigned i = 1; i < NumFixed - 1; i++) {
          if (CB->getArgOperand(i)->getType()->isIntegerTy(64)) {
            Size = CB->getArgOperand(i);
            // Take the last i64 before fmt as the size (n comes after
            // objsize in the wrapper).
          }
        }
        if (!Size)
          Size = ConstantInt::get(I64Ty, (uint64_t)-1);

        ImplArgs = {Buf, Size, Fmt, ArgsArray,
                    ConstantInt::get(I32Ty, NumVarArgs)};
      }

      Value *Result = Builder.CreateCall(Impl, ImplArgs);

      if (!CB->getType()->isVoidTy())
        CB->replaceAllUsesWith(Result);
      // For InvokeInst: replace with unconditional branch to normal dest.
      if (auto *II = dyn_cast<InvokeInst>(CB)) {
        BranchInst::Create(II->getNormalDest(), II);
      }
      CB->eraseFromParent();
    }

    if (Wrapper->use_empty())
      Wrapper->eraseFromParent();
    Changed = true;
  }

  // Clean up __v*_chk declarations if now unused.
  for (const char *Name :
       {"__vsprintf_chk", "__vsnprintf_chk", "__sprintf_chk",
        "__snprintf_chk"}) {
    if (Function *F = M.getFunction(Name)) {
      if (F->use_empty())
        F->eraseFromParent();
    }
  }

  return Changed;
}

bool runSprintf(Module &M) {
  bool Changed = false;

  // Handle fortified wrappers first (clang -O2 generates these).
  Changed |= rewriteFortifiedWrappers(M);

  // Direct calls (non-fortified, or compiled with -fno-builtin).
  Changed |= rewriteVariadicCalls(M, "sprintf", "__coqui_sprintf_impl", 2);
  Changed |= rewriteVariadicCalls(M, "snprintf", "__coqui_snprintf_impl", 3);

  // __coqui_* variants (if LibcTransform somehow runs first).
  Changed |=
      rewriteVariadicCalls(M, "__coqui_sprintf", "__coqui_sprintf_impl", 2);
  Changed |=
      rewriteVariadicCalls(M, "__coqui_snprintf", "__coqui_snprintf_impl", 3);

  return Changed;
}

} // namespace coqui
