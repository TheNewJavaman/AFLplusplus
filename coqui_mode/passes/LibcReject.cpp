/*
 * LibcReject.cpp --- reject unsupported libc patterns.
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
 * Ported from /coqui/src/LibcTransform.cpp (rejection half).
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
    /* setjmp family */
    "setjmp", "_setjmp", "__setjmp",
    "longjmp", "_longjmp", "__longjmp",
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

bool runLibcReject(Module &M) {
  for (Function &F : M) {
    if (!F.isDeclaration()) continue;   /* only care about external refs */

    std::string name = F.getName().str();
    if (!blocklist().count(name)) continue;

    if (F.use_empty()) continue;

    std::string msg;
    raw_string_ostream os(msg);
    os << "[coqui-cc] LibcReject: '" << name
       << "' is not supported on GPU; disable this feature at build time "
       << "(e.g., -DTARGET_NO_PTHREADS, -DPNG_NO_SETJMP).";
    report_fatal_error(os.str().c_str());
  }
  return false;
}

} // namespace coqui
