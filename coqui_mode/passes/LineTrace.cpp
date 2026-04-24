/*
 * LineTrace.cpp --- per-line execution-trace instrumentation.
 *
 * Inserts a call to `__coqui_trace_line(uint32_t line_id)` at the first
 * instruction in IR for each unique (file, line) pair encountered in a
 * basic block. Line IDs are sequential u32s assigned in module-walk
 * order. The ID is the only piece of information emitted at runtime;
 * the host obtains the (file, line) decoding by holding the same
 * (file, line) -> ID map externally (e.g., via debug info on the host
 * CPU build it intends to compare against).
 *
 * Used by oracle mode: the host runs a CPU-instrumented build to obtain
 * the expected execution trace for a corpus seed, then replays the same
 * seed through the GPU and compares the recorded sequence. Mismatch =
 * a coqui transform broke semantics; the divergence point pinpoints
 * the offending pass. When the flag is off, this pass is a no-op and
 * production runs pay nothing.
 *
 * Gated on the CLI flag `-coqui-line-trace` (cl::opt<bool> below). Off
 * by default so normal cubin builds are unaffected.
 *
 * Skip list mirrors Coverage.cpp's: any function whose name starts with
 * `__coqui_` (runtime + pass-emitted helpers) is exempt, plus sanitizer
 * runtime stubs (`__sanitizer_`, `__ubsan_`, `__asan_`). Otherwise the
 * recorded trace would contain the runtime's internal lines, which the
 * CPU expected-trace doesn't see.
 *
 * Insertion semantics: dedup is per-basic-block. Within a single BB the
 * first instruction with a given (file_id, line) gets the trace call;
 * subsequent instructions on the same line in the same BB do not. This
 * matches what the CPU build's trace would capture (one record per
 * source line per straight-line execution segment).
 */

#include "Transforms.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace coqui {

/* -coqui-line-trace : enable per-line trace instrumentation. Off by
 * default; production cubin builds pay nothing for the unused pass.
 * Forwarded by a future coqui-cc --line-trace flag (out of scope for
 * this commit; pass the flag to opt directly for synthetic tests). */
static cl::opt<bool> LineTraceOpt(
    "coqui-line-trace",
    cl::desc("Instrument every unique source line with __coqui_trace_line() (oracle mode)"),
    cl::init(false));

bool runLineTrace(Module &M) {
  /* Off by default. The pass plugin still calls runLineTrace; it just
   * returns immediately when the flag is unset. Keeps the plugin's
   * invocation table uniform. */
  if (!LineTraceOpt)
    return false;

  LLVMContext &C = M.getContext();
  Type *i32  = Type::getInt32Ty(C);
  Type *voidT = Type::getVoidTy(C);

  /* Declare __coqui_trace_line(i32 line_id). Linked from coqui_trace.c. */
  FunctionType *TraceLineTy = FunctionType::get(voidT, {i32}, false);
  FunctionCallee TraceLineFn =
      M.getOrInsertFunction("__coqui_trace_line", TraceLineTy);

  /* Sequential ID assignment over (file, line) pairs. We pack
   *   key = (file_index << 32) | line_no
   * into a u64 so identical pairs collapse to the same ID across
   * functions. */
  DenseMap<const DIFile *, uint32_t> FileIndex;
  DenseMap<uint64_t, uint32_t>       LineIDs;
  uint32_t NextLineID = 0;

  /* Skip discipline. We DON'T filter by `__coqui_` name prefix because
   * FuzzEntry renames the user's LLVMFuzzerTestOneInput to
   * __coqui_fuzz_execute before this pass runs (CoquiPassPlugin pipeline
   * order: FuzzEntry → ... → Coverage → LineTrace). A name-prefix skip
   * would silently drop the user's entire harness from instrumentation.
   *
   * Instead we filter by DI presence: the runtime (.bc-link'd from
   * coqui_runtime.c, coqui_coverage.c, ...) is built without -g so its
   * functions have no DILocation on any instruction. The user's
   * .bc, built with -g, carries DI throughout. The per-instruction DL
   * check inside the loop already gates emission on DL != null, so a
   * runtime function with zero DI emits zero trace calls naturally.
   * The function-level name filter is reserved for sanitizer-stub
   * functions that *do* sometimes carry DI (e.g., user code linked
   * against an asan/ubsan SDK) — none currently apply in coqui_mode,
   * but keep the names guarded to remain robust against future runtime
   * additions that include debug info. */
  auto skipFunction = [](StringRef Name) -> bool {
    if (Name.starts_with("__sanitizer_") ||
        Name.starts_with("__ubsan_") ||
        Name.starts_with("__asan_"))
      return true;
    /* The pass-emitted kernel itself (`__coqui_fuzz_kernel`) has no DI
     * so it auto-skips below. We do *not* skip __coqui_fuzz_execute —
     * it's the renamed user entry and must be instrumented. */
    return false;
  };

  unsigned long Instrumented = 0;
  unsigned long FunctionsScanned = 0;

  for (Function &F : M) {
    if (F.isDeclaration())
      continue;
    if (skipFunction(F.getName()))
      continue;
    FunctionsScanned++;

    for (BasicBlock &BB : F) {
      /* Per-BB dedup: collect the (file_index, line) keys we have
       * already instrumented in this block so adjacent instructions on
       * the same source line don't each get a trace call. */
      DenseSet<uint64_t> SeenInBB;

      /* Snapshot instructions first because CreateCall inserts new
       * instructions; iterating BB.begin()/end() while inserting would
       * eventually visit our own newly-inserted call. */
      SmallVector<Instruction *, 32> Insts;
      for (Instruction &I : BB)
        Insts.push_back(&I);

      for (Instruction *I : Insts) {
        const DILocation *DL = I->getDebugLoc().get();
        if (!DL)
          continue;
        const DIFile *File = DL->getFile();
        if (!File)
          continue;
        uint32_t LineNo = DL->getLine();
        if (LineNo == 0)
          continue;

        /* Assign a stable file index. */
        auto FIt = FileIndex.find(File);
        uint32_t FileIdx;
        if (FIt == FileIndex.end()) {
          FileIdx = (uint32_t)FileIndex.size();
          FileIndex[File] = FileIdx;
        } else {
          FileIdx = FIt->second;
        }

        uint64_t Key = ((uint64_t)FileIdx << 32) | (uint64_t)LineNo;

        /* Per-BB dedup before per-module ID lookup so the same line
         * straddled across BBs (e.g., a loop header) gets multiple
         * recordings (matches CPU expectation: each entry into a BB
         * traces the line). */
        if (!SeenInBB.insert(Key).second)
          continue;

        /* Per-module unique ID. Reuses across functions/BBs so the
         * recorded sequence is comparable across runs. */
        auto LIt = LineIDs.find(Key);
        uint32_t LineID;
        if (LIt == LineIDs.end()) {
          LineID = NextLineID++;
          LineIDs[Key] = LineID;
        } else {
          LineID = LIt->second;
        }

        /* Insert before the user instruction. */
        IRBuilder<> Builder(I);
        Builder.CreateCall(TraceLineFn, {ConstantInt::get(i32, LineID)});
        Instrumented++;
      }
    }
  }

  errs() << "[coqui-line-trace] instrumented " << Instrumented
         << " line-trace points across " << FunctionsScanned
         << " functions; "
         << LineIDs.size() << " unique (file, line) pairs from "
         << FileIndex.size() << " files\n";

  if (LineIDs.empty()) {
    errs() << "[coqui-line-trace] no debug info found in user code -- "
           << "build with -g\n";
  }

  return Instrumented > 0;
}

} // namespace coqui
