/*
 * Reloc.cpp --- break circular initializer dependencies between globals.
 *
 * Ported from /coqui/src/RelocTransform.cpp. Adapted for coqui mode's post-link
 * pipeline: runs after llvm-link + FuzzEntry, so __coqui_fuzz_kernel is
 * always present. For each back-edge in the global dependency graph, the
 * pass nulls the cyclic reference in the source's initializer and emits a
 * store in a new __coqui_reloc_init() function that patches the field at
 * runtime. A call to __coqui_reloc_init() is prepended to the kernel entry.
 *
 * Why this is needed: NVPTX's AsmPrinter aborts with "Circular dependency
 * found in global variable set" when emitting a module whose static globals
 * have self-referential initializers (e.g., clang's -O2 jump tables of
 * offsets computed as ptrtoint(str) - ptrtoint(table)). Replacing those
 * offsets with zeros and writing them at kernel-entry time sidesteps the
 * emitter's cycle check without changing runtime semantics.
 */

#include "Transforms.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"

#include <functional>

using namespace llvm;

namespace coqui {

static void collectGlobalVarRefs(const Constant *C,
                                 SmallPtrSetImpl<GlobalVariable *> &Refs) {
  if (auto *GV = dyn_cast<GlobalVariable>(const_cast<Constant *>(C))) {
    Refs.insert(GV);
    return;
  }
  for (const Use &U : C->operands()) {
    if (auto *Op = dyn_cast<Constant>(U.get()))
      collectGlobalVarRefs(Op, Refs);
  }
}

static void findBackEdges(
    DenseMap<GlobalVariable *, SmallPtrSet<GlobalVariable *, 4>> &Graph,
    SmallVectorImpl<std::pair<GlobalVariable *, GlobalVariable *>> &BackEdges) {

  DenseSet<GlobalVariable *> Visited;
  DenseSet<GlobalVariable *> InStack;

  std::function<void(GlobalVariable *)> DFS = [&](GlobalVariable *N) {
    Visited.insert(N);
    InStack.insert(N);
    auto It = Graph.find(N);
    if (It != Graph.end()) {
      for (GlobalVariable *Succ : It->second) {
        if (InStack.count(Succ))
          BackEdges.push_back({N, Succ});
        else if (!Visited.count(Succ))
          DFS(Succ);
      }
    }
    InStack.erase(N);
  };

  for (auto &Entry : Graph) {
    if (!Visited.count(Entry.first)) DFS(Entry.first);
  }
}

struct RelocEntry {
  GlobalVariable *Source;
  SmallVector<unsigned> Indices;
  Constant *OrigValue;
};

static Constant *replaceRefsInConstant(Constant *C, GlobalVariable *Target,
                                       SmallVector<unsigned> &Path,
                                       SmallVectorImpl<RelocEntry> &Relocs,
                                       GlobalVariable *Source) {
  SmallPtrSet<GlobalVariable *, 4> Refs;
  collectGlobalVarRefs(C, Refs);
  if (!Refs.count(Target)) return C;

  if (auto *CS = dyn_cast<ConstantStruct>(C)) {
    SmallVector<Constant *> NewOps;
    bool Changed = false;
    for (unsigned i = 0; i < CS->getNumOperands(); ++i) {
      Path.push_back(i);
      auto *Op = cast<Constant>(CS->getOperand(i));
      Constant *NewOp = replaceRefsInConstant(Op, Target, Path, Relocs, Source);
      NewOps.push_back(NewOp);
      if (NewOp != Op) Changed = true;
      Path.pop_back();
    }
    if (Changed) return ConstantStruct::get(CS->getType(), NewOps);
    return C;
  }

  if (auto *CA = dyn_cast<ConstantArray>(C)) {
    SmallVector<Constant *> NewOps;
    bool Changed = false;
    for (unsigned i = 0; i < CA->getNumOperands(); ++i) {
      Path.push_back(i);
      auto *Op = cast<Constant>(CA->getOperand(i));
      Constant *NewOp = replaceRefsInConstant(Op, Target, Path, Relocs, Source);
      NewOps.push_back(NewOp);
      if (NewOp != Op) Changed = true;
      Path.pop_back();
    }
    if (Changed) return ConstantArray::get(CA->getType(), NewOps);
    return C;
  }

  Relocs.push_back({Source, SmallVector<unsigned>(Path), C});
  return Constant::getNullValue(C->getType());
}

static void emitRelocInit(Module &M, const SmallVectorImpl<RelocEntry> &Relocs) {
  auto &Ctx = M.getContext();
  auto *VoidTy = Type::getVoidTy(Ctx);
  auto *FnTy = FunctionType::get(VoidTy, false);
  auto *I32Ty = Type::getInt32Ty(Ctx);

  Function *InitFn = Function::Create(FnTy, GlobalValue::InternalLinkage,
                                      "__coqui_reloc_init", &M);
  auto *BB = BasicBlock::Create(Ctx, "entry", InitFn);
  IRBuilder<> B(BB);

  for (const auto &R : Relocs) {
    Value *Addr;
    if (R.Indices.empty()) {
      Addr = R.Source;
    } else {
      SmallVector<Value *> GEPIdx;
      GEPIdx.push_back(ConstantInt::get(I32Ty, 0));
      for (unsigned Idx : R.Indices)
        GEPIdx.push_back(ConstantInt::get(I32Ty, Idx));
      Addr = B.CreateGEP(R.Source->getValueType(), R.Source, GEPIdx);
    }
    B.CreateStore(R.OrigValue, Addr);
  }
  B.CreateRetVoid();

  /* Insert a call to __coqui_reloc_init at the very top of __coqui_fuzz_kernel.
   * Only one thread per warp needs to run this, but the stores are idempotent
   * so having every thread run them is correct (just redundant). A proper
   * "thread 0 only" guard would require reading laneid/ctaid, which isn't
   * worth the code churn for at-most-a-dozen stores per kernel launch. */
  Function *Kernel = M.getFunction("__coqui_fuzz_kernel");
  if (Kernel && !Kernel->isDeclaration()) {
    BasicBlock &Entry = Kernel->getEntryBlock();
    IRBuilder<> KB(&Entry, Entry.getFirstInsertionPt());
    KB.CreateCall(InitFn);
  }
}

bool runReloc(Module &M) {
  DenseMap<GlobalVariable *, SmallPtrSet<GlobalVariable *, 4>> DepGraph;

  for (auto &G : M.globals()) {
    if (!G.hasInitializer() || G.isDeclaration()) continue;
    SmallPtrSet<GlobalVariable *, 4> Refs;
    collectGlobalVarRefs(G.getInitializer(), Refs);
    if (!Refs.empty()) DepGraph[&G] = std::move(Refs);
  }

  if (DepGraph.empty()) return false;

  SmallVector<std::pair<GlobalVariable *, GlobalVariable *>> BackEdges;
  findBackEdges(DepGraph, BackEdges);
  if (BackEdges.empty()) return false;

  errs() << "[coqui-reloc] breaking " << BackEdges.size()
         << " circular global dependency edge(s)\n";

  SmallVector<RelocEntry> Relocs;
  for (auto &Edge : BackEdges) {
    GlobalVariable *Source = Edge.first;
    GlobalVariable *Target = Edge.second;
    errs() << "[coqui-reloc]   " << Source->getName() << " -> "
           << Target->getName() << "\n";
    SmallVector<unsigned> Path;
    Constant *NewInit = replaceRefsInConstant(Source->getInitializer(), Target,
                                              Path, Relocs, Source);
    Source->setInitializer(NewInit);
  }

  emitRelocInit(M, Relocs);
  errs() << "[coqui-reloc] emitted " << Relocs.size() << " runtime relocation(s)\n";
  return true;
}

} // namespace coqui
