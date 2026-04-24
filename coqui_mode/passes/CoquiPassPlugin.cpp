/*
 * CoquiPassPlugin.cpp --- LLVM pass plugin entry + registration.
 *
 * Registers the `coqui` pass that runs all coqui mode transforms in order.
 */

#include "Transforms.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

namespace {

struct CoquiPass : public PassInfoMixin<CoquiPass> {
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &) {
    /* Each runX() returns true if it modified the module. OR them to decide
     * whether to preserve analyses at the end. The individual passes mostly
     * DO modify (instrument, rewrite calls, inject globals) so we almost
     * always end up returning none(); tracking it anyway keeps the analysis
     * cache alive in the degenerate "nothing to do" case (e.g. running the
     * plugin on an already-transformed module). */
    bool Changed = false;

    /* Pass order from coqui internals spec §3.2 */
    Changed |= coqui::runRejectInlineAsm(M);
    /* RejectSyscall: hard-reject for GPU-impossible syscalls
     * (dlopen/socket/mmap/...). Runs early so user code containing
     * these gets a clear diagnostic instead of being silently routed
     * to a generic-trap stub by Libc.cpp. */
    Changed |= coqui::runRejectSyscall(M);
    Changed |= coqui::runRejectLibc(M);
    Changed |= coqui::runVariadic(M);           /* before RejectIntrinsics so llvm.va_* is lowered first */
    Changed |= coqui::runRejectIntrinsics(M);
    Changed |= coqui::runFuzzEntry(M);
    Changed |= coqui::runCpp(M);                /* rewrite C++ new/delete before Heap so _Znwm etc are lowered */
    Changed |= coqui::runHeap(M);
    Changed |= coqui::runSprintf(M);            /* before runLibc so raw sprintf/snprintf are resolvable */
    Changed |= coqui::runLibc(M);
    Changed |= coqui::runMath(M);               /* rewrite llvm.pow/log/exp -> __coqui_* runtime calls */
    Changed |= coqui::runComplex(M);            /* rewrite C99 _Complex math -> __coqui_c* runtime calls */
    Changed |= coqui::runReloc(M);              /* break cyclic global init deps (NVPTX AsmPrinter can't handle) */
    Changed |= coqui::runStaticGlobals(M);
    Changed |= coqui::runMemoryLayout(M);
    Changed |= coqui::runCoverage(M);
    /* Pad user globals with red zones and register them in a descriptor
     * table BEFORE runAsan so the load/store instrumentation sees the
     * padded struct types and its slow path can detect OOB via the
     * descriptor scan (asan_check_global in coqui_asan.c). */
    Changed |= coqui::runAsanGlobals(M);
    Changed |= coqui::runAsan(M);
    Changed |= coqui::runExternalSymbolGatekeeper(M);

    return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
  }
};

} // anonymous namespace

extern "C" PassPluginLibraryInfo LLVM_ATTRIBUTE_WEAK
llvmGetPassPluginInfo() {
  return {
    LLVM_PLUGIN_API_VERSION,
    "CoquiPassPlugin",
    LLVM_VERSION_STRING,
    [](PassBuilder &PB) {
      PB.registerPipelineParsingCallback(
        [](StringRef Name, ModulePassManager &MPM,
           ArrayRef<PassBuilder::PipelineElement>) {
          if (Name == "coqui") {
            MPM.addPass(CoquiPass());
            return true;
          }
          return false;
        });
    }
  };
}
