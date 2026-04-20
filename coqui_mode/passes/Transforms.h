/*
 * Transforms.h --- shared pass declarations for CoquiPassPlugin.
 */

#pragma once

namespace llvm {
class Module;
}

namespace coqui {

/* Each returns true if the module was modified (standard LLVM convention). */

bool runInlineAsmReject(llvm::Module &M);
bool runLibcReject(llvm::Module &M);
bool runIntrinsicReject(llvm::Module &M);
bool runFuzzEntry(llvm::Module &M);
bool runHeap(llvm::Module &M);
bool runLibc(llvm::Module &M);
bool runStaticGlobals(llvm::Module &M);
bool runMemoryLayout(llvm::Module &M);
bool runCoverage(llvm::Module &M);
bool runAsan(llvm::Module &M);
bool runExternalSymbolGatekeeper(llvm::Module &M);

} // namespace coqui
