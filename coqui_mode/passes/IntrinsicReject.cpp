/*
 * IntrinsicReject.cpp --- reject unhandled LLVM intrinsics.
 *
 * Hard-fail on any intrinsic that is neither natively lowerable by the
 * NVPTX backend nor handled by a ported transform.
 *
 * Known-lowerable (do NOT reject):
 *   - llvm.memcpy, llvm.memset, llvm.memmove — NVPTX backend inlines
 *   - llvm.lifetime.* — NVPTX accepts and drops
 *   - llvm.assume — no-op after optimization
 *   - llvm.dbg.* — debug info (should be stripped with -g0, ignore if present)
 *
 * Known-reject (FATAL):
 *   - llvm.setjmp, llvm.longjmp, llvm.eh.* — exception handling
 *   - llvm.va_start, llvm.va_end, llvm.va_arg, llvm.va_copy — variadic
 *     (require Variadic transform port)
 *   - llvm.frameaddress, llvm.returnaddress — stack walking
 *
 * Unknown intrinsic names pass through silently (may be harmless NVPTX-
 * lowerable ones we haven't categorized; unresolved ones get caught later
 * by ExternalSymbolGatekeeper).
 */

#include "Transforms.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <set>
#include <string>

using namespace llvm;

namespace coqui {

static const std::set<std::string> &blocklist() {
  static const std::set<std::string> list = {
    "llvm.eh.typeid.for", "llvm.eh.sjlj.setjmp", "llvm.eh.sjlj.longjmp",
    "llvm.setjmp", "llvm.longjmp",
    "llvm.va_start", "llvm.va_end", "llvm.va_arg", "llvm.va_copy",
    "llvm.frameaddress", "llvm.returnaddress",
    "llvm.stacksave", "llvm.stackrestore",
  };
  return list;
}

static std::string normalizeName(StringRef Full) {
  /* Drop type suffixes after the second dot.
     e.g. "llvm.va_start.p0" → "llvm.va_start"
          "llvm.memcpy.p0.p0.i64" → "llvm.memcpy" (but we don't reject memcpy)
  */
  std::string s = Full.str();
  size_t first = s.find('.', 5);   /* skip "llvm." prefix */
  if (first == std::string::npos) return s;
  size_t second = s.find('.', first + 1);
  if (second == std::string::npos) return s;
  return s.substr(0, second);
}

bool runIntrinsicReject(Module &M) {
  for (Function &F : M) {
    if (!F.isIntrinsic()) continue;

    std::string fullName = F.getName().str();
    std::string baseName = normalizeName(fullName);

    if (!blocklist().count(baseName) && !blocklist().count(fullName)) continue;
    if (F.use_empty()) continue;

    std::string msg;
    raw_string_ostream os(msg);
    os << "[coqui-cc] IntrinsicReject: '" << fullName
       << "' not handled. Port the relevant transform or disable this "
       << "feature in the target.";
    report_fatal_error(os.str().c_str());
  }
  return false;
}

} // namespace coqui
