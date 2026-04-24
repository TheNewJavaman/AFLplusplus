/*
 * Math.cpp --- lower llvm.pow/log/exp intrinsics + frem to __coqui_* calls.
 *
 * Partial port of /coqui/src/MathTransform.cpp. Covers:
 *   - scalar llvm.pow/log/exp intrinsics (runtime stubs in coqui_libc.c)
 *   - `frem` BinaryOperator instructions (NVPTX has no hardware fmod and
 *     the backend can't select a libcall on-GPU; lowered to __coqui_fmod).
 *
 * Add more intrinsic entries here + stubs in the runtime on demand.
 */

#include "Transforms.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"

#include <utility>

using namespace llvm;

namespace coqui {

struct MathEntry {
    Intrinsic::ID ID;
    const char *F32Name;
    const char *F64Name;
    unsigned NumArgs;
};

static const MathEntry Entries[] = {
    {Intrinsic::pow,   "__coqui_powf",   "__coqui_pow",   2},
    {Intrinsic::log,   "__coqui_logf",   "__coqui_log",   1},
    {Intrinsic::log10, "__coqui_log10f", "__coqui_log10", 1},
    {Intrinsic::log2,  "__coqui_log2f",  "__coqui_log2",  1},
    {Intrinsic::exp,   "__coqui_expf",   "__coqui_exp",   1},
    {Intrinsic::exp2,  "__coqui_exp2f",  "__coqui_exp2",  1},
    {Intrinsic::sin,   "__coqui_sinf",   "__coqui_sin",   1},
    {Intrinsic::cos,   "__coqui_cosf",   "__coqui_cos",   1},
};

bool runMath(Module &M) {
    SmallVector<std::pair<CallInst *, const MathEntry *>> Work;

    for (Function &F : M) {
        for (BasicBlock &BB : F) {
            for (Instruction &I : BB) {
                auto *CI = dyn_cast<CallInst>(&I);
                if (!CI) continue;
                Function *Callee = CI->getCalledFunction();
                if (!Callee || !Callee->isIntrinsic()) continue;
                Intrinsic::ID IID = Callee->getIntrinsicID();
                for (const auto &E : Entries) {
                    if (IID == E.ID) { Work.push_back({CI, &E}); break; }
                }
            }
        }
    }

    // Collect frem BinaryOperators. NVPTX has no hardware fmod, and llc
    // cannot auto-select a libcall for frem on the GPU, so we must lower
    // them to __coqui_fmod / __coqui_fmodf explicitly.
    SmallVector<BinaryOperator *> FRemOps;
    for (Function &F : M) {
        for (BasicBlock &BB : F) {
            for (Instruction &I : BB) {
                if (auto *BO = dyn_cast<BinaryOperator>(&I))
                    if (BO->getOpcode() == Instruction::FRem)
                        FRemOps.push_back(BO);
            }
        }
    }

    if (Work.empty() && FRemOps.empty()) return false;

    for (auto &Pair : Work) {
        CallInst *CI = Pair.first;
        const MathEntry *E = Pair.second;
        Type *ResTy = CI->getType();
        bool IsF64 = ResTy->isDoubleTy();
        const char *Name = IsF64 ? E->F64Name : E->F32Name;

        SmallVector<Type *, 2> ArgTys;
        for (unsigned i = 0; i < E->NumArgs; i++) ArgTys.push_back(ResTy);
        FunctionType *FT = FunctionType::get(ResTy, ArgTys, false);

        FunctionCallee Callee = M.getOrInsertFunction(Name, FT);

        SmallVector<Value *, 2> Args;
        for (unsigned i = 0; i < E->NumArgs; i++) Args.push_back(CI->getArgOperand(i));

        CallInst *NewCall = CallInst::Create(Callee, Args, "", CI);
        CI->replaceAllUsesWith(NewCall);
        CI->eraseFromParent();
    }

    for (auto *BO : FRemOps) {
        Type *Ty = BO->getType();
        bool IsF32 = Ty->isFloatTy();
        const char *Name = IsF32 ? "__coqui_fmodf" : "__coqui_fmod";
        FunctionType *FnTy = FunctionType::get(Ty, {Ty, Ty}, false);
        FunctionCallee Fn = M.getOrInsertFunction(Name, FnTy);
        IRBuilder<> Builder(BO);
        Value *Result =
            Builder.CreateCall(Fn, {BO->getOperand(0), BO->getOperand(1)});
        BO->replaceAllUsesWith(Result);
        BO->eraseFromParent();
    }

    if (!Work.empty())
        errs() << "[coqui-math] rewrote " << Work.size()
               << " llvm.pow/log/exp intrinsic call(s)\n";
    if (!FRemOps.empty())
        errs() << "[coqui-math] rewrote " << FRemOps.size()
               << " frem instruction(s) with __coqui_fmod\n";
    return true;
}

} // namespace coqui
