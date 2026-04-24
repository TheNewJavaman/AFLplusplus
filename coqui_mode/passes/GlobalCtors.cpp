/*
 * GlobalCtors.cpp --- lower @llvm.global_ctors into a callable
 * __coqui_global_init() and wire it into the kernel entry.
 *
 * NVPTX has no concept of program startup, so anything Clang emits into
 * @llvm.global_ctors (C++ static-storage ctors, __attribute__((constructor))
 * functions, sancov module ctors, etc.) is otherwise silently dropped — the
 * device sees uninitialized state where C++ code expects "constructed".
 *
 * Two-step lowering:
 *   1. transformGlobals:    walk @llvm.global_ctors, sort by priority, emit
 *                           __coqui_global_init() whose body call()s each
 *                           ctor in order, then erase the global.
 *   2. ensureGlobalInitCalled: insert a single call to __coqui_global_init()
 *                           into __coqui_fuzz_kernel just before the user
 *                           harness call (after memory_init / slab setup so
 *                           ctor code can malloc, write statics, etc.).
 *
 * No-op for C-only targets: if @llvm.global_ctors doesn't exist, the pass
 * still emits the kernel-entry call to __coqui_global_init, but the runtime
 * supplies a weak default in coqui_global_init.c that resolves to a no-op.
 *
 * @llvm.global_dtors is intentionally unhandled: kernel-lifetime fuzzing
 * has no exit phase. atexit / __cxa_atexit registrations are similarly
 * dropped by replacing the call with `ret 0` semantics (handled below).
 *
 * Ported from /home/gpizarro/coqui/src/GlobalTransform.cpp, simplified for
 * cuAFL's single-pass-plugin model: we run AFTER llvm-link, so there is no
 * need for per-module __coqui_module_init_<hash> names + a post-link merge.
 * One __coqui_global_init definition per linked module is sufficient.
 */

#include "Transforms.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace coqui {

namespace {

/* -------------------------------------------------------------------------
 * transformGlobals
 *
 * Read @llvm.global_ctors (an array of {i32 priority, ptr fn, ptr data}),
 * collect each ctor function paired with its priority, sort ascending
 * (lower priority runs first per Itanium ABI), then synthesize
 * __coqui_global_init() with calls in that order. Erase the original
 * named global at the end.
 *
 * Filters:
 *  - sancov.module_ctor* — these would call __sanitizer_cov_trace_pc_guard_init,
 *    which is a no-op on NVPTX (guards are pre-numbered at link time).
 *    Including them just bloats the call chain and burns local-memory stack
 *    on deeply recursive targets.
 *  - asan.module_ctor* — same reasoning; cuAFL has its own ASan globals
 *    table set up by runAsanGlobals at kernel entry, no module ctor needed.
 *
 * Returns true if @llvm.global_ctors existed (and was lowered/erased).
 * ----------------------------------------------------------------------- */
static bool transformGlobals(Module &M) {
  GlobalVariable *Ctors = M.getNamedGlobal("llvm.global_ctors");
  if (!Ctors)
    return false;

  ConstantArray *InitArr = dyn_cast<ConstantArray>(Ctors->getInitializer());
  if (!InitArr) {
    /* Empty / zeroinitializer — nothing to lower, just remove the global. */
    Ctors->eraseFromParent();
    return true;
  }

  /* Collect (priority, ctor) pairs. */
  SmallVector<std::pair<int32_t, Function *>, 8> Entries;
  for (unsigned i = 0; i < InitArr->getNumOperands(); ++i) {
    auto *Entry = cast<ConstantStruct>(InitArr->getOperand(i));
    int32_t Priority = static_cast<int32_t>(
        cast<ConstantInt>(Entry->getOperand(0))->getSExtValue());
    auto *Fn = dyn_cast<Function>(Entry->getOperand(1));
    if (!Fn)
      continue;

    StringRef Name = Fn->getName();
    if (Name.starts_with("sancov.module_ctor") ||
        Name.starts_with("asan.module_ctor")) {
      errs() << "[coqui-globalctors] skipping sanitizer ctor: " << Name
             << " (no-op on NVPTX)\n";
      continue;
    }
    Entries.push_back({Priority, Fn});
  }

  /* Lower priority runs first. Stable sort preserves source order for ties. */
  std::stable_sort(Entries.begin(), Entries.end(),
                   [](const auto &A, const auto &B) { return A.first < B.first; });

  LLVMContext &Ctx = M.getContext();
  Type *VoidTy = Type::getVoidTy(Ctx);
  FunctionType *InitFnTy = FunctionType::get(VoidTy, false);

  /* If __coqui_global_init already has a body (defensive: another pass
   * could have synthesized one), prepend our calls; otherwise create
   * fresh. We always emit a strong definition so the runtime's weak
   * fallback in coqui_global_init.c is overridden. */
  Function *InitFn = M.getFunction("__coqui_global_init");
  if (InitFn && !InitFn->isDeclaration()) {
    /* The runtime's weak fallback (coqui_global_init.c) is already linked
     * into the module by the time this pass runs in cuAFL's pipeline.
     * Prepend our ctor calls before its `ret void`, and promote the
     * linkage from weak to ExternalLinkage so the emitted PTX shows a
     * strong definition (cleaner output; the weak qualifier was only
     * meaningful pre-link to allow the override). */
    InitFn->setLinkage(GlobalValue::ExternalLinkage);
    BasicBlock &EntryBB = InitFn->getEntryBlock();
    IRBuilder<> Builder(&*EntryBB.getFirstInsertionPt());
    for (auto &[Prio, Fn] : Entries) {
      errs() << "[coqui-globalctors] ctor (priority " << Prio
             << ", merged): " << Fn->getName() << "\n";
      Builder.CreateCall(Fn);
    }
  } else {
    if (!InitFn) {
      InitFn = Function::Create(InitFnTy, GlobalValue::ExternalLinkage,
                                "__coqui_global_init", &M);
    } else {
      /* Existing declaration — give it a body. */
      InitFn->setLinkage(GlobalValue::ExternalLinkage);
    }
    BasicBlock *BB = BasicBlock::Create(Ctx, "entry", InitFn);
    IRBuilder<> Builder(BB);
    for (auto &[Prio, Fn] : Entries) {
      errs() << "[coqui-globalctors] ctor (priority " << Prio
             << "): " << Fn->getName() << "\n";
      Builder.CreateCall(Fn);
    }
    Builder.CreateRetVoid();
  }

  Ctors->eraseFromParent();
  return true;
}

/* -------------------------------------------------------------------------
 * neutralizeAtexit
 *
 * Replace any call to __cxa_atexit / atexit with a constant-zero return
 * (success). Kernel-lifetime fuzzing never reaches a "program exit" phase
 * so there is nowhere to run these handlers, but Clang emits __cxa_atexit
 * calls as a side effect of compiling C++ thread-local destructors and
 * some libstdc++ internals. Leaving them as unresolved externs would fail
 * at the symbol gatekeeper.
 *
 * We don't synthesize the runtime __cxa_atexit / atexit; we delete the
 * call sites entirely. Both functions are documented to return 0 on
 * success, so callers that check the return value get the success path.
 * ----------------------------------------------------------------------- */
static bool neutralizeAtexit(Module &M) {
  bool Changed = false;
  for (const char *Name : {"__cxa_atexit", "atexit"}) {
    Function *F = M.getFunction(Name);
    if (!F)
      continue;

    SmallVector<CallInst *, 8> Calls;
    for (User *U : F->users()) {
      if (auto *CI = dyn_cast<CallInst>(U))
        if (CI->getCalledFunction() == F)
          Calls.push_back(CI);
    }

    Type *RetTy = F->getReturnType();
    for (CallInst *CI : Calls) {
      if (!CI->use_empty() && RetTy->isIntegerTy()) {
        CI->replaceAllUsesWith(ConstantInt::get(RetTy, 0));
      }
      CI->eraseFromParent();
    }

    if (!Calls.empty()) {
      errs() << "[coqui-globalctors] neutralized " << Calls.size()
             << " " << Name << " call site(s)\n";
      Changed = true;
    }

    /* Drop the now-unused declaration so the gatekeeper doesn't flag it. */
    if (F->use_empty() && F->isDeclaration())
      F->eraseFromParent();
  }
  return Changed;
}

/* -------------------------------------------------------------------------
 * ensureGlobalInitCalled
 *
 * Find __coqui_fuzz_kernel (created by runFuzzEntry) and insert a single
 * call to __coqui_global_init() just before the call to
 * __coqui_fuzz_execute (the renamed user harness). This ordering
 * guarantees that all kernel-side setup (memory_init, slab_setup,
 * slab_init_block + syncthreads, status phase, clk_a) has completed and
 * the per-thread environment is live before C++ ctors run.
 *
 * If the kernel already calls __coqui_global_init (idempotent re-runs),
 * skip. If __coqui_fuzz_execute can't be found in the kernel body,
 * fall back to inserting at the entry block's first insertion point
 * (matches Reloc's pattern; safe because the runtime supplies a weak
 * no-op when the strong override doesn't exist).
 *
 * Always returns true if the call was inserted (or already present).
 * ----------------------------------------------------------------------- */
static bool ensureGlobalInitCalled(Module &M) {
  Function *Kernel = M.getFunction("__coqui_fuzz_kernel");
  if (!Kernel || Kernel->isDeclaration()) {
    /* No kernel — runFuzzEntry didn't run, or this isn't a fuzz module.
     * Nothing to wire. */
    return false;
  }

  LLVMContext &Ctx = M.getContext();
  Type *VoidTy = Type::getVoidTy(Ctx);
  FunctionType *InitFnTy = FunctionType::get(VoidTy, false);

  /* Get-or-insert: declares __coqui_global_init if neither this pass nor
   * the runtime weak fallback has provided a definition yet. The linker
   * resolves to whichever is strongest (pass-emitted strong > runtime
   * weak). */
  FunctionCallee InitFn =
      M.getOrInsertFunction("__coqui_global_init", InitFnTy);

  /* Already wired? Walk the kernel once looking for an existing call. */
  Function *InitFnVal = dyn_cast<Function>(InitFn.getCallee());
  if (InitFnVal) {
    for (BasicBlock &BB : *Kernel) {
      for (Instruction &I : BB) {
        if (auto *CI = dyn_cast<CallInst>(&I)) {
          if (CI->getCalledFunction() == InitFnVal) {
            return false;
          }
        }
      }
    }
  }

  /* Find the call to __coqui_fuzz_execute and insert just before it. */
  Function *UserExec = M.getFunction("__coqui_fuzz_execute");
  Instruction *InsertBefore = nullptr;
  if (UserExec) {
    for (BasicBlock &BB : *Kernel) {
      for (Instruction &I : BB) {
        if (auto *CI = dyn_cast<CallInst>(&I)) {
          if (CI->getCalledFunction() == UserExec) {
            InsertBefore = CI;
            break;
          }
        }
      }
      if (InsertBefore)
        break;
    }
  }

  /* Fallback: prepend to entry block (matches Reloc pattern). */
  if (!InsertBefore) {
    BasicBlock &EntryBB = Kernel->getEntryBlock();
    InsertBefore = &*EntryBB.getFirstInsertionPt();
  }

  IRBuilder<> Builder(InsertBefore);
  Builder.CreateCall(InitFn);

  errs() << "[coqui-globalctors] wired __coqui_global_init() into "
         << Kernel->getName() << "\n";
  return true;
}

} // anonymous namespace

bool runGlobalCtors(Module &M) {
  bool Changed = false;
  Changed |= transformGlobals(M);
  Changed |= neutralizeAtexit(M);
  Changed |= ensureGlobalInitCalled(M);
  return Changed;
}

} // namespace coqui
