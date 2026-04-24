/*
 * RejectLibc.cpp --- reject unsupported libc patterns.
 *
 * Blocklist-only; does NOT provide replacements. Replacements get
 * ported on demand as the gatekeeper flags them.
 *
 * FATALs on:
 *   - setjmp/longjmp family (sigsetjmp, siglongjmp, etc.)
 *   - pthread_* except pthread_once (stub in coqui_runtime.c)
 *   - signal()/raise()/kill() (process signaling)
 *   - fork()/exec() (process control)
 *
 * Only call/invoke uses trigger fatal errors — declaration-only
 * references are ignored because internalize+globaldce prune them.
 *
 * Ported from /coqui/src/LibcTransform.cpp (rejection half, lines ~322-378).
 */

#include "Transforms.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <set>
#include <string>

using namespace llvm;

namespace coqui {

static const std::set<std::string> &blocklist() {
  static const std::set<std::string> list = {
    /* setjmp family */
    "setjmp", "_setjmp", "__setjmp",
    "longjmp", "_longjmp", "__longjmp", "__longjmp_chk",
    "sigsetjmp", "siglongjmp",
    "__sigsetjmp",
    /* pthread — except pthread_once (stubbed in coqui_runtime.c) */
    "pthread_create", "pthread_join", "pthread_detach",
    "pthread_mutex_init", "pthread_mutex_destroy",
    "pthread_mutex_lock", "pthread_mutex_unlock", "pthread_mutex_trylock",
    "pthread_cond_init", "pthread_cond_destroy",
    "pthread_cond_wait", "pthread_cond_signal", "pthread_cond_broadcast",
    "pthread_rwlock_init", "pthread_rwlock_rdlock", "pthread_rwlock_wrlock",
    "pthread_rwlock_unlock", "pthread_key_create", "pthread_setspecific",
    "pthread_getspecific", "pthread_self",
    /* Process control */
    "fork", "vfork", "execv", "execve", "execvp", "execl", "execle", "execlp",
    /* Signals */
    "signal", "sigaction", "raise", "kill", "sigprocmask",
    /* Threading primitives */
    "thrd_create", "thrd_join", "mtx_init", "mtx_lock", "mtx_unlock",
  };
  return list;
}

/* Describe the category + common upstream guards for the error message.
 * These are illustrative examples — each library uses its own macro, so we
 * list several well-known ones rather than pretending there's a universal
 * guard. Mirrors coqui's LibcTransform error phrasing. */
static const char *categoryHint(const std::string &name) {
  if (name == "setjmp" || name == "_setjmp" || name == "__setjmp" ||
      name == "longjmp" || name == "_longjmp" || name == "__longjmp" ||
      name == "__longjmp_chk" ||
      name == "sigsetjmp" || name == "siglongjmp" || name == "__sigsetjmp")
    return "setjmp/longjmp is not supported on NVPTX. Compile the target "
           "without setjmp support (e.g., -DPNG_NO_SETJMP for libpng; other "
           "libraries use their own guards)";
  if (name.rfind("pthread_", 0) == 0)
    return "pthreads are not supported on NVPTX. Compile the target "
           "without threading support (e.g., -DCMS_NO_PTHREADS for lcms, "
           "-DZSTD_MULTITHREAD=0 for zstd; other libraries use their own "
           "guards)";
  if (name == "fork" || name == "vfork" ||
      name.rfind("exec", 0) == 0)
    return "process control (fork/exec) is not supported on NVPTX. Disable "
           "the offending feature at build time using the library's own "
           "configuration option";
  if (name == "signal" || name == "sigaction" || name == "raise" ||
      name == "kill" || name == "sigprocmask")
    return "POSIX signal handling is not supported on NVPTX. Disable the "
           "offending feature at build time using the library's own "
           "configuration option";
  if (name.rfind("thrd_", 0) == 0 || name.rfind("mtx_", 0) == 0)
    return "C11 threads are not supported on NVPTX. Disable threading at "
           "build time using the library's own configuration option";
  return "this libc symbol is host-only and not supported on NVPTX. Disable "
         "the offending feature at build time using the library's own "
         "configuration option";
}

bool runRejectLibc(Module &M) {
  for (Function &F : M) {
    if (!F.isDeclaration()) continue;   /* only care about external refs */

    std::string name = F.getName().str();
    if (!blocklist().count(name)) continue;

    /* Count only call/invoke uses. Declaration-only references (address-
     * taken, unused decls imported from headers) get pruned by
     * internalize+globaldce and should not fatal. Mirrors coqui's
     * LibcTransform pattern. */
    unsigned CallUses = 0;
    for (User *U : F.users()) {
      if (auto *CB = dyn_cast<CallBase>(U); CB && CB->getCalledFunction() == &F)
        ++CallUses;
    }
    if (CallUses == 0) continue;

    std::string msg;
    raw_string_ostream os(msg);
    os << "[coqui-cc] RejectLibc: '" << name << "' has " << CallUses
       << " call(s) — " << categoryHint(name) << ".";
    report_fatal_error(os.str().c_str());
  }
  return false;
}

} // namespace coqui
