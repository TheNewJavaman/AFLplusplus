/*
 * CoquiPassPlugin.cpp --- LLVM pass plugin entry + registration.
 *
 * Registers the `coqui-link` pass pipeline that runs all day-1 passes
 * in order.
 */

#include "Transforms.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

namespace {

struct CoquiLinkPass : public PassInfoMixin<CoquiLinkPass> {
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &) {
    /* Pass order from coqui internals spec §3.2 */
    coqui::runInlineAsmReject(M);
    coqui::runLibcReject(M);
    coqui::runVariadic(M);          /* lower user-defined variadics BEFORE IntrinsicReject catches llvm.va_* */
    coqui::runIntrinsicReject(M);
    coqui::runFuzzEntry(M);
    coqui::runHeap(M);
    coqui::runSprintf(M);           /* must run BEFORE runLibc so raw sprintf/snprintf are still resolvable by name */
    coqui::runLibc(M);              /* minimum libc string/math replacements */
    coqui::runMath(M);              /* rewrite llvm.pow/log/exp → __coqui_* runtime calls (NVPTX can't select) */
    coqui::runReloc(M);             /* break circular global initializer deps (breaks NVPTX AsmPrinter) */
    coqui::runStaticGlobals(M);
    coqui::runMemoryLayout(M);
    coqui::runCoverage(M);
    coqui::runAsan(M);
    coqui::runExternalSymbolGatekeeper(M);
    return PreservedAnalyses::none();
  }
};

} // anonymous namespace

extern "C" ::llvm::PassPluginLibraryInfo LLVM_ATTRIBUTE_WEAK
llvmGetPassPluginInfo() {
  return {
    LLVM_PLUGIN_API_VERSION,
    "CoquiPassPlugin",
    LLVM_VERSION_STRING,
    [](PassBuilder &PB) {
      PB.registerPipelineParsingCallback(
        [](StringRef Name, ModulePassManager &MPM,
           ArrayRef<PassBuilder::PipelineElement>) {
          if (Name == "coqui-link") {
            MPM.addPass(CoquiLinkPass());
            return true;
          }
          return false;
        });
    }
  };
}
