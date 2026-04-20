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
    coqui::runIntrinsicReject(M);
    coqui::runFuzzEntry(M);
    coqui::runHeap(M);
    coqui::runLibc(M);              /* NEW: minimum libc string/math replacements */
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
