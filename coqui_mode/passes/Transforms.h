/*
 * Transforms.h --- shared pass declarations for CoquiPassPlugin.
 */

#pragma once

namespace llvm {
class Module;
}

namespace coqui {

/* Each returns true if the module was modified (standard LLVM convention). */

bool runRejectInlineAsm(llvm::Module &M);
bool runRejectLibc(llvm::Module &M);
bool runRejectIntrinsics(llvm::Module &M);
bool runFuzzEntry(llvm::Module &M);
bool runCpp(llvm::Module &M);
bool runHeap(llvm::Module &M);
bool runSprintf(llvm::Module &M);
bool runLibc(llvm::Module &M);
bool runVariadic(llvm::Module &M);
bool runReloc(llvm::Module &M);
bool runMath(llvm::Module &M);
bool runComplex(llvm::Module &M);
bool runStaticGlobals(llvm::Module &M);
bool runMemoryLayout(llvm::Module &M);
bool runCoverage(llvm::Module &M);
bool runAsanGlobals(llvm::Module &M);
bool runAsan(llvm::Module &M);
bool runExternalSymbolGatekeeper(llvm::Module &M);

} // namespace coqui
