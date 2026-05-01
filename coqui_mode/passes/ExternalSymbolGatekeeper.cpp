/*
 * ExternalSymbolGatekeeper.cpp --- final check for unresolved externals.
 *
 * Runs LAST in the coqui pipeline. Scans all external function
 * declarations that have live uses. FATALs with a specific message
 * for any symbol not on the allowlist.
 *
 * Allowlist seeded with __coqui_* runtime symbols (prefix) and __llvm_*
 * builtins (prefix). LLVM intrinsics are excluded from the scan.
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

static const std::set<std::string> &allowlist() {
  static const std::set<std::string> list = {
    /* Reserved for explicit allowlist entries */
  };
  return list;
}

bool runExternalSymbolGatekeeper(Module &M) {
  for (Function &F : M) {
    if (!F.isDeclaration()) continue;
    if (F.isIntrinsic()) continue;
    if (F.use_empty()) continue;

    std::string name = F.getName().str();

    /* Prefix-based allow */
    if (name.rfind("__coqui_", 0) == 0) continue;
    if (name.rfind("__nv_", 0) == 0) continue;    /* libdevice math */
    if (name.rfind("__llvm_", 0) == 0) continue;
    if (name.rfind("llvm.", 0) == 0) continue;   /* defense-in-depth */

    /* Explicit allowlist */
    if (allowlist().count(name)) continue;

    /* FATAL */
    std::string msg;
    raw_string_ostream os(msg);
    os << "[coqui-cc] ExternalSymbolGatekeeper: unresolved external '"
       << name << "' — port the replacement transform or runtime stub.";

    /* Add a single example caller for diagnosis */
    for (const User *U : F.users()) {
      if (auto *I = dyn_cast<Instruction>(U)) {
        os << " Used by: " << I->getFunction()->getName().str();
        break;
      }
    }

    report_fatal_error(os.str().c_str());
  }
  return false;
}

} // namespace coqui
