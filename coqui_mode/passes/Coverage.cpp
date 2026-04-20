/*
 * Coverage.cpp --- coverage instrumentation gate.
 *
 * Historical: this pass emitted AFL hash-edge instrumentation (6 IRs per
 * basic block: prev_loc XOR cur_loc → cov[idx]++). That approach required
 * a fixed 64 KB cov_map per thread (512 MB working set across 8192 threads),
 * which the profile identified as memory-latency bound.
 *
 * Iteration 10: switched to SanitizerCoverage (trace-pc-guard) with sequential
 * per-edge IDs assigned by the SancovCount pass. Clang itself emits the
 * instrumentation when the target is compiled with
 * -fsanitize-coverage=trace-pc-guard (added to coqui-cc), so this pass has
 * nothing to do at the IR level any more.
 *
 * We keep the pass registered (and the file) so that (a) the pass pipeline
 * order in CoquiPassPlugin.cpp doesn't need a conditional, and (b) a future
 * revert can restore the hash-edge path in one place.
 */

#include "Transforms.h"
#include "llvm/IR/Module.h"

using namespace llvm;

namespace coqui {

bool runCoverage(Module &M) {
  /* SanitizerCoverage edge instrumentation is emitted by clang upstream of
   * this pass (see coqui-cc --fsanitize-coverage=trace-pc-guard). The guard
   * arrays are renumbered to sequential IDs by SancovCount. Nothing to do
   * here. */
  (void)M;
  return false;
}

} // namespace coqui
