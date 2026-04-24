/*
 * SyscallTransform.cpp --- rewrite divertable syscall callsites to runtime
 * stubs that emit category-specific trap reasons or deterministic returns.
 *
 * Mirrors Libc.cpp's mechanics: for each entry in the rewrite table, call
 * M.getOrInsertFunction with the OLD function type and replace all uses
 * with the new stub (LLVM bitcasts the call site if the runtime stub's
 * type differs — safe because the trap-class stubs are noreturn).
 *
 * Why a separate pass from Libc.cpp:
 *   - Each syscall here gets a SPECIFIC trap_reason code (KILL/FORK/EXEC),
 *     not Libc.cpp's generic __coqui_trap(). The host-side decoder in
 *     afl-fuzz-coqui.c can then map the trap reason back to a meaningful
 *     diagnostic ("target tried to fork", not "target trapped").
 *   - Some syscalls here are NO-OPS (signal, sigprocmask, sigemptyset,
 *     getenv, ...) — silently succeeding lets fuzz targets that gratuitously
 *     touch them keep running, instead of trapping on harmless calls.
 *   - Time syscalls return a deterministic constant so targets that read
 *     time take the same control-flow path on every fuzz run, improving
 *     coverage stability.
 *
 * Pass order: runs BEFORE RejectLibc/Libc. This is essential because
 * RejectLibc's blocklist contains signal/raise/kill/fork/exec — without
 * pre-rewriting, RejectLibc would fatal on any target using those before
 * SyscallTransform got a chance. After this pass, the original symbol
 * decls are erased and RejectLibc sees nothing to reject.
 *
 * Stub naming: __coqui_<sym>_stub (with the _stub suffix) so we don't
 * collide with the existing __coqui_<sym> trap stubs in coqui_libc.c that
 * Libc.cpp currently rewrites a duplicate set of symbols to. Those legacy
 * names are retained for backward compatibility; SyscallTransform's new
 * stubs are addressable via the _stub suffix and live in the same .c file.
 */

#include "Transforms.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"

#include <utility>

using namespace llvm;

namespace coqui {

static bool replaceSyscallFn(Module &M, StringRef Old, StringRef New) {
    Function *OldF = M.getFunction(Old);
    if (!OldF || OldF->use_empty()) return false;

    /* getOrInsertFunction inserts a bitcast at the call site when the runtime
     * stub's type differs from the user's libc-prototype — safe for
     * trap-class stubs (noreturn) and harmless for the deterministic-return
     * stubs because the bitcast result is the same width as the original
     * return type in the target ABI we care about (nvptx64). */
    FunctionCallee NewF = M.getOrInsertFunction(New, OldF->getFunctionType());

    OldF->replaceAllUsesWith(NewF.getCallee());
    OldF->eraseFromParent();
    return true;
}

bool runSyscallTransform(Module &M) {
    bool Changed = false;

    /* Each entry maps a libc/POSIX syscall name to a runtime stub defined
     * in coqui_libc.c. Stub semantics (no-op vs trap-with-reason vs
     * deterministic-return) are picked by the stub body, NOT by the pass —
     * the pass is just a name-rewrite. */
    static constexpr std::pair<const char *, const char *> Replacements[] = {
        /* Signal handlers — no-op (signal delivery is meaningless on GPU,
         * but installing a handler should silently succeed). */
        {"signal",      "__coqui_signal_stub"},
        {"sigaction",   "__coqui_sigaction_stub"},
        {"sigprocmask", "__coqui_sigprocmask_stub"},
        {"sigemptyset", "__coqui_sigemptyset_stub"},
        {"sigfillset",  "__coqui_sigfillset_stub"},
        {"sigaddset",   "__coqui_sigaddset_stub"},
        {"sigdelset",   "__coqui_sigdelset_stub"},
        {"sigismember", "__coqui_sigismember_stub"},

        /* Signal delivery — trap with KILL reason. Reaching `kill` or
         * `raise` indicates the target is trying to terminate itself or
         * another "process", which the host can interpret as a deliberate
         * abort signal worth distinguishing from a generic crash. */
        {"kill",  "__coqui_kill_stub"},
        {"raise", "__coqui_raise_stub"},

        /* Process creation — trap with FORK reason. */
        {"fork",  "__coqui_fork_stub"},
        {"vfork", "__coqui_vfork_stub"},

        /* Process replacement — trap with EXEC reason. `system` is grouped
         * here because it's exec+wait under the hood. */
        {"execve", "__coqui_execve_stub"},
        {"execv",  "__coqui_execv_stub"},
        {"execvp", "__coqui_execvp_stub"},
        {"execlp", "__coqui_execlp_stub"},
        {"execl",  "__coqui_execl_stub"},
        {"execle", "__coqui_execle_stub"},
        {"system", "__coqui_system_stub"},

        /* Time — deterministic fixed epoch. Targets that read time take
         * stable control-flow paths on every fuzz run, improving coverage
         * convergence. */
        {"gettimeofday", "__coqui_gettimeofday_stub"},
        {"clock_gettime", "__coqui_clock_gettime_stub"},
        {"time", "__coqui_time_stub"},

        /* Environment — no env on GPU. Returning NULL/0 is the correct
         * "not present / success" answer for the read/write API split. */
        {"getenv",   "__coqui_getenv_stub"},
        {"setenv",   "__coqui_setenv_stub"},
        {"putenv",   "__coqui_putenv_stub"},
        {"unsetenv", "__coqui_unsetenv_stub"},

        /* Process/user identity — fixed values. Most targets that read
         * these only check for == 0 (root) vs anything else; a fixed
         * non-zero value avoids accidentally taking the privileged path. */
        {"getpid",  "__coqui_getpid_stub"},
        {"getppid", "__coqui_getppid_stub"},
        {"getuid",  "__coqui_getuid_stub"},
        {"geteuid", "__coqui_geteuid_stub"},
        {"getgid",  "__coqui_getgid_stub"},
        {"getegid", "__coqui_getegid_stub"},
    };

    unsigned Rewritten = 0;
    for (auto &[Old, New] : Replacements) {
        if (replaceSyscallFn(M, Old, New)) {
            ++Rewritten;
            Changed = true;
        }
    }
    if (Rewritten)
        errs() << "[coqui-syscall] rewrote " << Rewritten
               << " syscall call(s)\n";

    return Changed;
}

} // namespace coqui
