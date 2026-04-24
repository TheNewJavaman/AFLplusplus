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

/* Stub categories drive both the rewrite (always) and a per-symbol compile-
 * time warning (only for the silent-substitution categories). Trap-class
 * stubs fail loudly at runtime — no warning needed; the others silently
 * substitute behavior that doesn't exist on NVPTX, so we surface a build-
 * time message so the user knows their target leans on absent functionality.
 *
 *   NoOpSignal           — sigaction/sigprocmask/sigemptyset/... (warn)
 *   TrapKill/Fork/Exec   — runtime trap with category reason (no warn)
 *   DeterministicTime    — fixed epoch (warn — clock doesn't exist on GPU)
 *   NoOpEnv              — env not on GPU (warn)
 *   DeterministicProc    — fixed pid/uid (warn — no processes on GPU)
 */
enum class StubClass {
    NoOpSignal,
    TrapKill,
    TrapFork,
    TrapExec,
    DeterministicTime,
    NoOpEnv,
    DeterministicProc,
};

struct Replacement {
    const char *Old;
    const char *New;
    StubClass   Class;
};

static const char *warnHint(StubClass C) {
    switch (C) {
    case StubClass::NoOpSignal:
        return "signal delivery is not supported on NVPTX (no signals on "
               "device); handler install/mask calls are silently no-op'd";
    case StubClass::DeterministicTime:
        return "wall-clock time is not available on NVPTX; the stub returns "
               "a fixed zero epoch so target control-flow stays deterministic "
               "across fuzz runs (deadline/elapsed loops will not progress)";
    case StubClass::NoOpEnv:
        return "process environment is not available on NVPTX (no env block "
               "on device); reads return NULL and writes silently succeed";
    case StubClass::DeterministicProc:
        return "process / user identity is not meaningful on NVPTX; the stub "
               "returns a fixed non-zero value so root-check branches take "
               "the unprivileged path deterministically";
    default:
        return nullptr;
    }
}

static bool replaceSyscallFn(Module &M, const Replacement &R) {
    Function *OldF = M.getFunction(R.Old);
    if (!OldF || OldF->use_empty()) return false;

    /* Per-symbol compile-time warning for silent-substitution classes. The
     * functionality doesn't exist on GPU, so even though the pass succeeds
     * the user should know their target leans on absent semantics. Emitted
     * BEFORE the rewrite so the message names the original symbol clearly. */
    if (const char *Hint = warnHint(R.Class)) {
        errs() << "[coqui-syscall] warning: '" << R.Old
               << "' rewritten to no-op stub '" << R.New << "' — " << Hint
               << "\n";
    }

    /* getOrInsertFunction inserts a bitcast at the call site when the runtime
     * stub's type differs from the user's libc-prototype — safe for
     * trap-class stubs (noreturn) and harmless for the deterministic-return
     * stubs because the bitcast result is the same width as the original
     * return type in the target ABI we care about (nvptx64). */
    FunctionCallee NewF = M.getOrInsertFunction(R.New, OldF->getFunctionType());

    OldF->replaceAllUsesWith(NewF.getCallee());
    OldF->eraseFromParent();
    return true;
}

bool runSyscallTransform(Module &M) {
    bool Changed = false;

    /* Each entry maps a libc/POSIX syscall name to a runtime stub defined
     * in coqui_libc.c plus a category tag. Stub semantics (no-op vs trap-
     * with-reason vs deterministic-return) live in the stub body; the
     * pass is a name-rewrite + per-class warning emitter. */
    static constexpr Replacement Replacements[] = {
        /* Signal handlers — no-op (signal delivery is meaningless on GPU,
         * but installing a handler should silently succeed). Warned. */
        {"signal",      "__coqui_signal_stub",      StubClass::NoOpSignal},
        {"sigaction",   "__coqui_sigaction_stub",   StubClass::NoOpSignal},
        {"sigprocmask", "__coqui_sigprocmask_stub", StubClass::NoOpSignal},
        {"sigemptyset", "__coqui_sigemptyset_stub", StubClass::NoOpSignal},
        {"sigfillset",  "__coqui_sigfillset_stub",  StubClass::NoOpSignal},
        {"sigaddset",   "__coqui_sigaddset_stub",   StubClass::NoOpSignal},
        {"sigdelset",   "__coqui_sigdelset_stub",   StubClass::NoOpSignal},
        {"sigismember", "__coqui_sigismember_stub", StubClass::NoOpSignal},

        /* Signal delivery — trap with KILL reason. Reaching `kill` or
         * `raise` indicates the target is trying to terminate itself or
         * another "process", which the host can interpret as a deliberate
         * abort signal worth distinguishing from a generic crash. */
        {"kill",  "__coqui_kill_stub",  StubClass::TrapKill},
        {"raise", "__coqui_raise_stub", StubClass::TrapKill},

        /* Process creation — trap with FORK reason. */
        {"fork",  "__coqui_fork_stub",  StubClass::TrapFork},
        {"vfork", "__coqui_vfork_stub", StubClass::TrapFork},

        /* Process replacement — trap with EXEC reason. `system` is grouped
         * here because it's exec+wait under the hood. */
        {"execve", "__coqui_execve_stub", StubClass::TrapExec},
        {"execv",  "__coqui_execv_stub",  StubClass::TrapExec},
        {"execvp", "__coqui_execvp_stub", StubClass::TrapExec},
        {"execlp", "__coqui_execlp_stub", StubClass::TrapExec},
        {"execl",  "__coqui_execl_stub",  StubClass::TrapExec},
        {"execle", "__coqui_execle_stub", StubClass::TrapExec},
        {"system", "__coqui_system_stub", StubClass::TrapExec},

        /* Time — deterministic fixed epoch. Targets that read time take
         * stable control-flow paths on every fuzz run, improving coverage
         * convergence. */
        {"gettimeofday",  "__coqui_gettimeofday_stub",  StubClass::DeterministicTime},
        {"clock_gettime", "__coqui_clock_gettime_stub", StubClass::DeterministicTime},
        {"time",          "__coqui_time_stub",          StubClass::DeterministicTime},

        /* Environment — no env on GPU. Returning NULL/0 is the correct
         * "not present / success" answer for the read/write API split. */
        {"getenv",   "__coqui_getenv_stub",   StubClass::NoOpEnv},
        {"setenv",   "__coqui_setenv_stub",   StubClass::NoOpEnv},
        {"putenv",   "__coqui_putenv_stub",   StubClass::NoOpEnv},
        {"unsetenv", "__coqui_unsetenv_stub", StubClass::NoOpEnv},

        /* Process/user identity — fixed values. Most targets that read
         * these only check for == 0 (root) vs anything else; a fixed
         * non-zero value avoids accidentally taking the privileged path. */
        {"getpid",  "__coqui_getpid_stub",  StubClass::DeterministicProc},
        {"getppid", "__coqui_getppid_stub", StubClass::DeterministicProc},
        {"getuid",  "__coqui_getuid_stub",  StubClass::DeterministicProc},
        {"geteuid", "__coqui_geteuid_stub", StubClass::DeterministicProc},
        {"getgid",  "__coqui_getgid_stub",  StubClass::DeterministicProc},
        {"getegid", "__coqui_getegid_stub", StubClass::DeterministicProc},
    };

    unsigned Rewritten = 0;
    for (const Replacement &R : Replacements) {
        if (replaceSyscallFn(M, R)) {
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
