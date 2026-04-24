/*
 * RejectSyscall.cpp --- compile-time hard-reject for GPU-impossible syscalls.
 *
 * Mirrors RejectLibc.cpp's pattern: blocklist of symbol names that cannot
 * possibly run on NVPTX even as a no-op (no virtual memory, no FD machinery,
 * no networking, no dynamic linking, no kernel multiplexing). Reaching one
 * of these in user code is a build-time failure with a clear diagnostic so
 * the user can disable the offending feature in their target.
 *
 * Pairs with SyscallTransform.cpp (the rewrite half), which handles the
 * divertable subset (signal masking → no-op, fork/exec → trap-with-reason,
 * gettimeofday → fixed epoch, getpid → fixed value, ...).
 *
 * Pass order: runs early, BEFORE SyscallTransform/RejectLibc/Libc. Symbols
 * here have no useful runtime emulation, so failing fast prevents downstream
 * passes from emitting nonsensical replacements (e.g., Libc.cpp would
 * silently route dlopen → __coqui_dlopen which is a generic-trap stub —
 * losing the diagnostic about WHY the build failed).
 *
 * Only call/invoke uses trigger fatal errors; declaration-only references
 * (address-taken, header decls without callers) get pruned by
 * internalize+globaldce and should not fatal — same convention as RejectLibc.
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
    /* Dynamic loading — ld.so machinery has no GPU analog */
    "dlopen", "dlsym", "dlclose", "dlerror", "dladdr", "dlvsym",
    /* Virtual memory — no per-process page tables on device */
    "mmap", "mmap64", "munmap", "mprotect", "madvise", "msync", "mlock",
    "munlock", "mremap",
    /* Networking — no TCP/IP stack on GPU */
    "socket", "socketpair", "connect", "bind", "listen", "accept", "accept4",
    "send", "sendto", "sendmsg", "recv", "recvfrom", "recvmsg",
    "shutdown", "getsockopt", "setsockopt", "getsockname", "getpeername",
    /* File descriptor machinery — no FD table on device */
    "pipe", "pipe2", "dup", "dup2", "dup3", "epoll_create", "epoll_create1",
    "epoll_ctl", "epoll_wait", "epoll_pwait", "select", "pselect", "poll",
    "ppoll", "eventfd",
    /* Process creation — only fork-equivalent that has no exec stub.
     * `fork` and `vfork` themselves are diverted to runtime traps with a
     * specific reason by SyscallTransform.cpp, so they're NOT in this
     * blocklist — only `clone` (and clone3) which encodes too much
     * configurability to stub usefully. */
    "clone", "clone3",
  };
  return list;
}

/* Diagnostic phrasing — describes the category and the typical guard the
 * user should disable in their target. Mirrors RejectLibc's structure so
 * the build-failure messages have a consistent shape. */
static const char *categoryHint(const std::string &name) {
  if (name == "dlopen" || name == "dlsym" || name == "dlclose" ||
      name == "dlerror" || name == "dladdr" || name == "dlvsym")
    return "dynamic loading is not supported on NVPTX (no ld.so on device). "
           "Compile the target without dlopen support (e.g., link plugins "
           "statically or compile out the plugin loader)";
  if (name.rfind("mmap", 0) == 0 || name == "munmap" || name == "mprotect" ||
      name == "madvise" || name == "msync" || name == "mlock" ||
      name == "munlock" || name == "mremap")
    return "virtual-memory operations are not supported on NVPTX (no per-"
           "process page tables on device). Replace mmap'd regions with "
           "static buffers or disable the offending feature at build time";
  if (name == "socket" || name == "socketpair" || name == "connect" ||
      name == "bind" || name == "listen" || name == "accept" ||
      name == "accept4" || name == "send" || name == "sendto" ||
      name == "sendmsg" || name == "recv" || name == "recvfrom" ||
      name == "recvmsg" || name == "shutdown" || name == "getsockopt" ||
      name == "setsockopt" || name == "getsockname" || name == "getpeername")
    return "networking is not supported on NVPTX (no TCP/IP stack on GPU). "
           "Disable the offending feature at build time using the library's "
           "own configuration option";
  if (name == "pipe" || name == "pipe2" || name == "dup" || name == "dup2" ||
      name == "dup3" || name.rfind("epoll_", 0) == 0 || name == "select" ||
      name == "pselect" || name == "poll" || name == "ppoll" ||
      name == "eventfd")
    return "FD multiplexing/duplication is not supported on NVPTX (no FD "
           "table on device). Disable the offending feature at build time";
  if (name == "clone" || name == "clone3")
    return "process creation via clone is not supported on NVPTX. Disable "
           "threading/process-spawning at build time using the library's own "
           "configuration option";
  return "this syscall is host-only and not supported on NVPTX. Disable "
         "the offending feature at build time using the library's own "
         "configuration option";
}

bool runRejectSyscall(Module &M) {
  for (Function &F : M) {
    if (!F.isDeclaration()) continue;   /* only care about external refs */

    std::string name = F.getName().str();
    if (!blocklist().count(name)) continue;

    /* Count only call/invoke uses. Declaration-only references get pruned
     * by internalize+globaldce; address-taken-but-uncalled symbols also
     * fall through. Mirrors RejectLibc's filter. */
    unsigned CallUses = 0;
    for (User *U : F.users()) {
      if (auto *CB = dyn_cast<CallBase>(U); CB && CB->getCalledFunction() == &F)
        ++CallUses;
    }
    if (CallUses == 0) continue;

    std::string msg;
    raw_string_ostream os(msg);
    os << "[coqui-cc] RejectSyscall: '" << name << "' has " << CallUses
       << " call(s) — " << categoryHint(name) << ".";
    report_fatal_error(os.str().c_str());
  }
  return false;
}

} // namespace coqui
