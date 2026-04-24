/*
 * Complex.cpp --- rewrite C99 _Complex math calls to match the NVPTX runtime
 * calling convention.
 *
 * Ported from /home/gpizarro/coqui/src/ComplexTransform.cpp.
 *
 * Float complex: clang lowers `float _Complex` as `<2 x float>` on x86-64,
 * but our device runtime uses `{ float, float }` for returns and separate
 * float args for inputs. This pass rewrites calls like
 * `cexpf(<2 x float>)` into `__coqui_cexpf(float, float)` and reconstructs
 * the vector from the struct return.
 *
 * Double complex: clang lowers `double _Complex` as two separate `double`
 * args (already matching our runtime), but the return is a literal
 * `{ double, double }`. Even though this is structurally identical to any
 * named struct the runtime might use, LLVM treats them as distinct Type*
 * objects, which breaks getCalledFunction() after linking. We re-declare
 * each target with a freshly-created literal-struct return type so type
 * identity holds at the call site.
 *
 * Compiler-generated helpers: clang emits calls to __muldc3/__divdc3
 * (double complex multiply/divide) and __mulsc3/__divsc3 (float complex).
 * These are redirected to __coqui_* implementations.
 */

#include "Transforms.h"

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"

namespace coqui {

using namespace llvm;

bool runComplex(Module &M) {
  LLVMContext &Ctx = M.getContext();

  /* --- Float complex functions --- */
  /* Takes <2 x float> args, returns <2 x float> or float. */
  struct FloatComplexEntry {
    const char *OldName;
    const char *NewName;
    unsigned NumComplexArgs; /* Number of <2 x float> arguments */
    bool ReturnsComplex;     /* true if returns <2 x float> */
    bool ReturnsReal;        /* true if returns float (e.g., cabsf, cargf) */
  };

  static const FloatComplexEntry FloatEntries[] = {
      /* Returns float (scalar) */
      {"cabsf", "__coqui_cabsf", 1, false, true},
      {"cargf", "__coqui_cargf", 1, false, true},
      /* Returns <2 x float> (complex) */
      {"conjf", "__coqui_conjf", 1, true, false},
      {"cexpf", "__coqui_cexpf", 1, true, false},
      {"clogf", "__coqui_clogf", 1, true, false},
      {"csqrtf", "__coqui_csqrtf", 1, true, false},
      {"cpowf", "__coqui_cpowf", 2, true, false},
      {"csinf", "__coqui_csinf", 1, true, false},
      {"ccosf", "__coqui_ccosf", 1, true, false},
      {"ctanf", "__coqui_ctanf", 1, true, false},
      {"csinhf", "__coqui_csinhf", 1, true, false},
      {"ccoshf", "__coqui_ccoshf", 1, true, false},
      {"ctanhf", "__coqui_ctanhf", 1, true, false},
      /* Compiler-generated float complex multiply/divide.
       * These take 4 separate float args (a, b, c, d), not <2 x float>. */
      {"__mulsc3", "__coqui___mulsc3", 4, true, false},
      {"__divsc3", "__coqui___divsc3", 4, true, false},
  };

  SmallVector<std::pair<CallBase *, const FloatComplexEntry *>> FloatToReplace;

  for (const auto &E : FloatEntries) {
    Function *F = M.getFunction(E.OldName);
    if (!F)
      continue;

    for (User *U : F->users()) {
      if (auto *CB = dyn_cast<CallBase>(U))
        if (CB->getCalledFunction() == F)
          FloatToReplace.push_back({CB, &E});
    }
  }

  unsigned TotalReplaced = 0;

  Type *F32Ty = Type::getFloatTy(Ctx);
  /* Return type for complex float: { float, float } */
  StructType *CF32Ty = StructType::get(Ctx, {F32Ty, F32Ty});

  for (auto &[CI, E] : FloatToReplace) {
    IRBuilder<> Builder(CI);

    /* Build the argument list: decompose each <2 x float> into two floats. */
    SmallVector<Value *> NewArgs;
    SmallVector<Type *> NewArgTypes;
    for (unsigned i = 0; i < E->NumComplexArgs; ++i) {
      Value *Arg = CI->getArgOperand(i);
      if (Arg->getType()->isVectorTy()) {
        /* <2 x float> -> two separate floats */
        Value *Real = Builder.CreateExtractElement(Arg, (uint64_t)0, "creal");
        Value *Imag = Builder.CreateExtractElement(Arg, (uint64_t)1, "cimag");
        NewArgs.push_back(Real);
        NewArgs.push_back(Imag);
        NewArgTypes.push_back(F32Ty);
        NewArgTypes.push_back(F32Ty);
      } else {
        /* __mulsc3/__divsc3 already take separate float args */
        NewArgs.push_back(Arg);
        NewArgTypes.push_back(Arg->getType());
      }
    }

    /* Determine return type. */
    Type *NewRetTy;
    if (E->ReturnsReal)
      NewRetTy = F32Ty;
    else
      NewRetTy = CF32Ty; /* { float, float } */

    FunctionType *NewFnTy = FunctionType::get(NewRetTy, NewArgTypes, false);
    FunctionCallee NewFn = M.getOrInsertFunction(E->NewName, NewFnTy);

    Value *Result = Builder.CreateCall(NewFn, NewArgs);

    if (E->ReturnsComplex) {
      Type *OrigRetTy = CI->getType();
      if (OrigRetTy->isVectorTy()) {
        /* Convert { float, float } back to <2 x float>. */
        Value *RealOut = Builder.CreateExtractValue(Result, 0, "real");
        Value *ImagOut = Builder.CreateExtractValue(Result, 1, "imag");
        Value *Vec = PoisonValue::get(FixedVectorType::get(F32Ty, 2));
        Vec = Builder.CreateInsertElement(Vec, RealOut, (uint64_t)0);
        Vec = Builder.CreateInsertElement(Vec, ImagOut, (uint64_t)1);
        CI->replaceAllUsesWith(Vec);
      } else {
        /* __mulsc3/__divsc3 already return { float, float } */
        CI->replaceAllUsesWith(Result);
      }
    } else {
      /* Scalar return (cabsf, cargf). */
      CI->replaceAllUsesWith(Result);
    }

    /* For InvokeInst: replace with unconditional branch to normal dest. */
    if (auto *II = dyn_cast<InvokeInst>(CI)) {
      BranchInst::Create(II->getNormalDest(), II);
    }
    CI->eraseFromParent();
    TotalReplaced++;
  }

  /* Remove the old float complex function declarations. */
  for (const auto &E : FloatEntries) {
    Function *F = M.getFunction(E.OldName);
    if (F && F->use_empty())
      F->eraseFromParent();
  }

  /* --- Double complex functions ---
   * clang emits these with `{ double, double }` (literal struct) return type,
   * and separate double args. Our runtime returns via the same literal
   * struct so type identity holds, but we still re-declare at the call site
   * to avoid a cross-module type mismatch after linking. */
  struct DoubleComplexEntry {
    const char *OldName;
    const char *NewName;
    unsigned NumDoubleArgs; /* Total number of double arguments */
    bool ReturnsComplex;    /* true if returns { double, double } */
  };

  static const DoubleComplexEntry DoubleEntries[] = {
      /* Returns { double, double } */
      {"conj", "__coqui_conj", 2, true},
      {"cexp", "__coqui_cexp", 2, true},
      {"clog", "__coqui_clog", 2, true},
      {"csqrt", "__coqui_csqrt", 2, true},
      {"cpow", "__coqui_cpow", 4, true},
      {"csin", "__coqui_csin", 2, true},
      {"ccos", "__coqui_ccos", 2, true},
      {"ctan", "__coqui_ctan", 2, true},
      {"csinh", "__coqui_csinh", 2, true},
      {"ccosh", "__coqui_ccosh", 2, true},
      {"ctanh", "__coqui_ctanh", 2, true},
      /* Compiler-generated double complex multiply/divide */
      {"__muldc3", "__coqui___muldc3", 4, true},
      {"__divdc3", "__coqui___divdc3", 4, true},
  };

  Type *F64Ty = Type::getDoubleTy(Ctx);
  StructType *CD64Ty = StructType::get(Ctx, {F64Ty, F64Ty});

  for (const auto &E : DoubleEntries) {
    Function *F = M.getFunction(E.OldName);
    if (!F)
      continue;

    /* Collect call sites first (modifying users while iterating is unsafe). */
    SmallVector<CallBase *> Calls;
    for (User *U : F->users())
      if (auto *CB = dyn_cast<CallBase>(U))
        if (CB->getCalledFunction() == F)
          Calls.push_back(CB);

    if (Calls.empty())
      continue;

    /* Build the replacement function type with a literal struct return. */
    SmallVector<Type *> ArgTypes(E.NumDoubleArgs, F64Ty);
    Type *RetTy = E.ReturnsComplex ? (Type *)CD64Ty : F64Ty;
    FunctionType *NewFnTy = FunctionType::get(RetTy, ArgTypes, false);
    FunctionCallee NewFn = M.getOrInsertFunction(E.NewName, NewFnTy);

    for (CallBase *CB : Calls) {
      IRBuilder<> Builder(CB);

      /* Forward all arguments unchanged (already separate doubles). */
      SmallVector<Value *> Args;
      for (unsigned i = 0; i < CB->arg_size(); ++i)
        Args.push_back(CB->getArgOperand(i));

      Value *Result = Builder.CreateCall(NewFn, Args);
      CB->replaceAllUsesWith(Result);
      /* For InvokeInst: replace with unconditional branch to normal dest. */
      if (auto *II = dyn_cast<InvokeInst>(CB)) {
        BranchInst::Create(II->getNormalDest(), II);
      }
      CB->eraseFromParent();
      TotalReplaced++;
    }
  }

  /* Remove old double complex declarations. */
  for (const auto &E : DoubleEntries) {
    Function *F = M.getFunction(E.OldName);
    if (F && F->use_empty())
      F->eraseFromParent();
  }

  if (TotalReplaced > 0)
    errs() << "[coqui-complex] rewrote " << TotalReplaced
           << " complex math call(s)\n";

  return TotalReplaced > 0;
}

} // namespace coqui
