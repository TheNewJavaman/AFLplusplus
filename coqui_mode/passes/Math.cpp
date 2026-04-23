/*
 * Math.cpp --- lower llvm.pow/log/exp intrinsics to __coqui_* runtime calls.
 *
 * Minimal port of /coqui/src/MathTransform.cpp. Only covers the math
 * intrinsics NVPTX can't select natively; the runtime implementations live
 * in coqui_libc.c. Add entries here + corresponding stubs in the runtime
 * on demand.
 */

#include "Transforms.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Function.h"
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

    if (Work.empty()) return false;

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

    errs() << "[coqui-math] rewrote " << Work.size()
           << " llvm.pow/log/exp intrinsic call(s)\n";
    return true;
}

} // namespace coqui
