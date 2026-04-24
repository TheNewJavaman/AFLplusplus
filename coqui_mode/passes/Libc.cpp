/*
 * Libc.cpp --- rewrite libc call sites to __coqui_* device replacements.
 *
 * Ported from /coqui/src/LibcTransform.cpp. Runs after Sprintf (which
 * rewrites sprintf/snprintf variadics) and RejectLibc (which fatals on
 * setjmp/pthread_create). Three jobs:
 *
 *   1. Rewrite ~200 plain libc names (memcpy, fopen, printf, signal, …)
 *      to their __coqui_* runtime analog. replaceFn uses
 *      M.getOrInsertFunction with the OLD function type so trap stubs
 *      declared `(void)` in the runtime get a bitcast at the call site.
 *
 *   2. Strip _FORTIFY_SOURCE wrapper args on _chk variants
 *      (__memcpy_chk, __sprintf_chk, …). The extra flag/slen/objsize
 *      arguments are dropped and the call is redirected to the plain
 *      __coqui_* impl.
 *
 *   3. Replace stdin/stdout/stderr and errno global variables with
 *      __coqui_* analogs (fake FILE* pointers and a real int slot).
 *
 * Notes on symbols skipped on purpose:
 *   - exit / _exit are NOT rewritten. cuAFL's runtime already exports
 *     __coqui_exit as a (void)-signatured PTX thread-exit helper used by
 *     ubsan/asan/memory; remapping libc exit(int) would collide. Plain
 *     `exit` / `_exit` instead link to the trap definitions in
 *     coqui_libc.c.
 *   - setjmp/longjmp/pthread_* (except pthread_once) are rejected at
 *     compile time by RejectLibc.cpp — not re-listed here.
 *   - llvm.pow/log/exp intrinsics are handled by Math.cpp. The scalar
 *     libc names (pow, log, sin, …) ARE in the rewrite list below so
 *     direct (non-intrinsic) calls also get redirected.
 *   - Complex struct-returning variants (cexp, clog, …) are handled by
 *     Complex.cpp, which constructs literal-struct-returning calls.
 *     Only the scalar cabs/carg are in this list.
 *
 * No LLVM attribute annotation (readonly/nounwind/etc) is applied here;
 * task 11 handles that pass separately.
 */

#include "Transforms.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"

#include <utility>

using namespace llvm;

namespace coqui {

static bool replaceFn(Module &M, StringRef Old, StringRef New) {
    Function *OldF = M.getFunction(Old);
    if (!OldF || OldF->use_empty()) return false;

    /* getOrInsertFunction returns a bitcast when the existing definition's
     * type differs from the requested one — harmless for trap stubs
     * (declared (void) in the runtime) whose original libc signature
     * carries real args. */
    FunctionCallee NewF = M.getOrInsertFunction(New, OldF->getFunctionType());

    OldF->replaceAllUsesWith(NewF.getCallee());
    OldF->eraseFromParent();
    return true;
}

bool runLibc(Module &M) {
    bool Changed = false;

    /* ------------------------------------------------------------------
     * 1. Direct function-name rewrites.
     * ------------------------------------------------------------------*/
    static constexpr std::pair<const char *, const char *> Replacements[] = {
        /* String operations */
        {"strlen",   "__coqui_strlen"},
        {"strnlen",  "__coqui_strnlen"},
        {"strcmp",   "__coqui_strcmp"},
        {"strncmp",  "__coqui_strncmp"},
        {"strcpy",   "__coqui_strcpy"},
        {"strncpy",  "__coqui_strncpy"},
        {"strcat",   "__coqui_strcat"},
        {"strchr",   "__coqui_strchr"},
        {"strrchr",  "__coqui_strrchr"},
        {"strstr",   "__coqui_strstr"},
        {"strtok",   "__coqui_strtok"},
        {"strerror", "__coqui_strerror"},
        {"strdup",   "__coqui_strdup"},
        {"strndup",  "__coqui_strndup"},
        {"strspn",   "__coqui_strspn"},
        {"strcspn",  "__coqui_strcspn"},
        {"strpbrk",  "__coqui_strpbrk"},

        /* Memory operations */
        {"memcpy",   "__coqui_memcpy"},
        {"memmove",  "__coqui_memmove"},
        {"memset",   "__coqui_memset"},
        {"memcmp",   "__coqui_memcmp"},
        {"bcmp",     "__coqui_memcmp"},
        {"memchr",   "__coqui_memchr"},

        /* Stdio / formatted I/O */
        {"printf",    "__coqui_printf"},
        {"fprintf",   "__coqui_fprintf"},
        {"vprintf",   "__coqui_vprintf"},
        {"vfprintf",  "__coqui_vfprintf"},
        {"vsprintf",  "__coqui_vsprintf"},
        {"vsnprintf", "__coqui_vsnprintf"},
        {"sprintf",   "__coqui_sprintf"},
        {"snprintf",  "__coqui_snprintf"},
        {"puts",      "__coqui_puts"},
        {"putchar",   "__coqui_putchar"},

        /* File ops (VFS-backed stubs) */
        {"fwrite",   "__coqui_fwrite"},
        {"fread",    "__coqui_fread"},
        {"fopen",    "__coqui_fopen"},
        {"fopen64",  "__coqui_fopen"},
        {"fclose",   "__coqui_fclose"},
        {"fseek",    "__coqui_fseek"},
        {"ftell",    "__coqui_ftell"},
        {"feof",     "__coqui_feof"},
        {"ferror",   "__coqui_ferror"},
        {"clearerr", "__coqui_clearerr"},
        {"fgetc",    "__coqui_fgetc"},
        {"fputc",    "__coqui_fputc"},
        {"fgets",    "__coqui_fgets"},
        {"fputs",    "__coqui_fputs"},
        {"ungetc",   "__coqui_ungetc"},
        {"rewind",   "__coqui_rewind"},
        {"fflush",   "__coqui_fflush"},
        {"fileno",   "__coqui_fileno"},
        {"getc",     "__coqui_fgetc"},
        {"putc",     "__coqui_fputc"},

        /* Process control — abort/assert_fail only. exit/_exit kept off
         * the list; see pass header. */
        {"abort",         "__coqui_abort"},
        {"__assert_fail", "__coqui_assert_fail"},

        /* Random */
        {"rand",   "__coqui_rand"},
        {"srand",  "__coqui_srand"},
        {"rand_r", "__coqui_rand_r"},

        /* Errno — function form. Direct global refs handled below. */
        {"__errno_location", "__coqui_errno_location"},

        /* Conversion / parsing */
        {"abs",        "__coqui_abs"},
        {"strtod",     "__coqui_strtod"},
        {"strtof",     "__coqui_strtof"},
        {"strtol",     "__coqui_strtol"},
        {"strtoul",    "__coqui_strtoul"},
        {"strtoll",    "__coqui_strtoll"},
        {"strtoull",   "__coqui_strtoull"},
        {"strtoimax",  "__coqui_strtoimax"},
        {"strtoumax",  "__coqui_strtoumax"},
        {"qsort",      "__coqui_qsort"},

        /* Time */
        {"time",      "__coqui_time"},
        {"gmtime",    "__coqui_gmtime"},
        {"localtime", "__coqui_localtime"},
        {"ctime",     "__coqui_ctime"},

        /* File descriptors */
        {"write",   "__coqui_write"},
        {"read",    "__coqui_read"},
        {"open",    "__coqui_open"},
        {"open64",  "__coqui_open"},
        {"close",   "__coqui_close_fd"},
        {"dup",     "__coqui_dup"},
        {"remove",  "__coqui_remove"},

        /* Misc */
        {"localeconv",    "__coqui_localeconv"},
        {"atexit",        "__coqui_atexit"},
        {"__cxa_atexit",  "__coqui_cxa_atexit"},
        {"__cxa_finalize","__coqui_cxa_finalize"},
        {"pthread_once",  "__coqui_pthread_once"},

        /* Ctype */
        {"tolower",   "__coqui_tolower"},
        {"toupper",   "__coqui_toupper"},
        {"isalpha",   "__coqui_isalpha"},
        {"isdigit",   "__coqui_isdigit"},
        {"isalnum",   "__coqui_isalnum"},
        {"isspace",   "__coqui_isspace"},
        {"isprint",   "__coqui_isprint"},
        {"isupper",   "__coqui_isupper"},
        {"islower",   "__coqui_islower"},
        {"isxdigit",  "__coqui_isxdigit"},
        {"iscntrl",   "__coqui_iscntrl"},
        {"ispunct",   "__coqui_ispunct"},
        {"isgraph",   "__coqui_isgraph"},
        {"__ctype_b_loc",       "__coqui_ctype_b_loc"},
        {"__ctype_tolower_loc", "__coqui_ctype_tolower_loc"},
        {"__ctype_toupper_loc", "__coqui_ctype_toupper_loc"},

        /* Math — direct (non-intrinsic) libc calls. llvm.* intrinsic
         * forms are handled by Math.cpp. */
        {"log",     "__coqui_log"},
        {"logf",    "__coqui_logf"},
        {"log10",   "__coqui_log10"},
        {"log10f",  "__coqui_log10f"},
        {"log2",    "__coqui_log2"},
        {"log2f",   "__coqui_log2f"},
        {"exp",     "__coqui_exp"},
        {"expf",    "__coqui_expf"},
        {"exp2",    "__coqui_exp2"},
        {"exp2f",   "__coqui_exp2f"},
        {"pow",     "__coqui_pow"},
        {"powf",    "__coqui_powf"},
        {"sin",     "__coqui_sin"},
        {"sinf",    "__coqui_sinf"},
        {"cos",     "__coqui_cos"},
        {"cosf",    "__coqui_cosf"},
        {"fmod",    "__coqui_fmod"},
        {"fmodf",   "__coqui_fmodf"},

        /* Complex — scalar double variants only. ComplexTransform
         * handles struct-returning cexp/clog/csqrt/etc. */
        {"cabs", "__coqui_cabs"},
        {"carg", "__coqui_carg"},

        /* Signals — trap */
        {"signal",      "__coqui_signal"},
        {"sigaction",   "__coqui_sigaction"},
        {"kill",        "__coqui_kill"},
        {"raise",       "__coqui_raise"},
        {"sigprocmask", "__coqui_sigprocmask"},
        {"sigsuspend",  "__coqui_sigsuspend"},

        /* Process — trap */
        {"fork",  "__coqui_fork"},
        {"vfork", "__coqui_vfork"},

        /* Exec — trap */
        {"execve", "__coqui_execve"},
        {"execvp", "__coqui_execvp"},
        {"execl",  "__coqui_execl"},
        {"system", "__coqui_system"},

        /* Dynamic linking — trap */
        {"dlopen",  "__coqui_dlopen"},
        {"dlsym",   "__coqui_dlsym"},
        {"dlclose", "__coqui_dlclose"},
        {"dlerror", "__coqui_dlerror"},

        /* Pipes/sockets — trap */
        {"pipe",     "__coqui_pipe"},
        {"pipe2",    "__coqui_pipe2"},
        {"socket",   "__coqui_socket"},
        {"connect",  "__coqui_connect"},
        {"bind",     "__coqui_bind"},
        {"listen",   "__coqui_listen"},
        {"accept",   "__coqui_accept"},
        {"send",     "__coqui_send"},
        {"recv",     "__coqui_recv"},
        {"sendto",   "__coqui_sendto"},
        {"recvfrom", "__coqui_recvfrom"},

        /* Wide string */
        {"wcslen",  "__coqui_wcslen"},
        {"wcscmp",  "__coqui_wcscmp"},
        {"wcsncmp", "__coqui_wcsncmp"},
        {"wmemcpy", "__coqui_wmemcpy"},
        {"wmemset", "__coqui_wmemset"},
        {"wcscpy",  "__coqui_wcscpy"},
        {"wcsncpy", "__coqui_wcsncpy"},
        {"wcscat",  "__coqui_wcscat"},
        {"wcschr",  "__coqui_wcschr"},
        {"wcsrchr", "__coqui_wcsrchr"},

        /* Multibyte */
        {"wcrtomb", "__coqui_wcrtomb"},
        {"mbrtowc", "__coqui_mbrtowc"},
        {"mbtowc",  "__coqui_mbtowc"},
        {"wctomb",  "__coqui_wctomb"},
        {"mblen",   "__coqui_mblen"},

        /* Floating-point environment — trap */
        {"fegetround",    "__coqui_fegetround"},
        {"fesetround",    "__coqui_fesetround"},
        {"feclearexcept", "__coqui_feclearexcept"},
        {"feraiseexcept", "__coqui_feraiseexcept"},
        {"fetestexcept",  "__coqui_fetestexcept"},
        {"fegetenv",      "__coqui_fegetenv"},
        {"fesetenv",      "__coqui_fesetenv"},
    };

    unsigned Rewritten = 0;
    for (auto &[Old, New] : Replacements) {
        if (replaceFn(M, Old, New)) {
            ++Rewritten;
            Changed = true;
        }
    }
    if (Rewritten)
        errs() << "[coqui-libc] rewrote " << Rewritten << " libc call(s)\n";

    /* ------------------------------------------------------------------
     * 2. _FORTIFY_SOURCE `_chk` wrappers — strip the extra size_t/flag
     *    args and redirect to the plain __coqui_* impl.
     * ------------------------------------------------------------------*/
    static constexpr std::pair<const char *, const char *> ChkReplacements[] = {
        {"__vprintf_chk",   "__coqui_vprintf"},
        {"__vfprintf_chk",  "__coqui_vfprintf"},
        {"__printf_chk",    "__coqui_printf"},
        {"__fprintf_chk",   "__coqui_fprintf"},
        {"__vsprintf_chk",  "__coqui_vsprintf"},
        {"__vsnprintf_chk", "__coqui_vsnprintf"},
        {"__sprintf_chk",   "__coqui_sprintf"},
        {"__snprintf_chk",  "__coqui_snprintf"},
        {"__memcpy_chk",    "__coqui_memcpy"},
        {"__memmove_chk",   "__coqui_memmove"},
        {"__memset_chk",    "__coqui_memset"},
        {"__strcpy_chk",    "__coqui_strcpy"},
        {"__strncpy_chk",   "__coqui_strncpy"},
        {"__strcat_chk",    "__coqui_strcat"},
    };

    for (auto &[OldName, NewName] : ChkReplacements) {
        Function *F = M.getFunction(OldName);
        if (!F || F->use_empty())
            continue;

        errs() << "[coqui-libc] stripping fortified " << OldName << " -> "
               << NewName << " (" << F->getNumUses() << " use(s))\n";

        SmallVector<CallBase *> Calls;
        for (User *U : F->users()) {
            if (auto *CB = dyn_cast<CallBase>(U))
                if (CB->getCalledFunction() == F)
                    Calls.push_back(CB);
        }

        for (CallBase *CB : Calls) {
            IRBuilder<> B(CB);

            StringRef Name(OldName);
            bool IsMemOp = Name.starts_with("__mem") ||
                           Name.starts_with("__str");

            SmallVector<Value *> NewArgs;
            if (IsMemOp) {
                /* __memcpy_chk(dst, src, n, objsize) → keep all but last. */
                for (unsigned i = 0; i + 1 < CB->arg_size(); ++i)
                    NewArgs.push_back(CB->getArgOperand(i));
            } else {
                bool IsSnprintf = Name.contains("snprintf_chk");
                bool IsSprintf = Name.contains("sprintf_chk") && !IsSnprintf;
                bool IsFprintf = Name.contains("fprintf_chk");

                if (IsSnprintf) {
                    /* __[v]snprintf_chk(buf, maxlen, flag, slen, fmt, …)
                     * keep buf(0), maxlen(1), skip flag(2)+slen(3), keep rest */
                    for (unsigned i = 0; i < CB->arg_size(); ++i)
                        if (i != 2 && i != 3)
                            NewArgs.push_back(CB->getArgOperand(i));
                } else if (IsSprintf) {
                    /* __[v]sprintf_chk(buf, flag, slen, fmt, …)
                     * keep buf(0), skip flag(1)+slen(2), keep rest */
                    for (unsigned i = 0; i < CB->arg_size(); ++i)
                        if (i != 1 && i != 2)
                            NewArgs.push_back(CB->getArgOperand(i));
                } else if (IsFprintf) {
                    /* __[v]fprintf_chk(fp, flag, fmt, …) — skip flag(1) */
                    for (unsigned i = 0; i < CB->arg_size(); ++i)
                        if (i != 1)
                            NewArgs.push_back(CB->getArgOperand(i));
                } else {
                    /* __[v]printf_chk(flag, fmt, …) — skip flag(0) */
                    for (unsigned i = 1; i < CB->arg_size(); ++i)
                        NewArgs.push_back(CB->getArgOperand(i));
                }
            }

            SmallVector<Type *> ArgTys;
            for (Value *V : NewArgs)
                ArgTys.push_back(V->getType());
            bool IsVarArg = Name.contains("printf") &&
                            !Name.starts_with("__v");
            FunctionType *NewFT =
                FunctionType::get(CB->getType(), ArgTys, IsVarArg);
            FunctionCallee NewFn = M.getOrInsertFunction(NewName, NewFT);

            CallInst *NewCI = B.CreateCall(NewFn, NewArgs);
            if (auto *CI = dyn_cast<CallInst>(CB))
                NewCI->setTailCall(CI->isTailCall());
            CB->replaceAllUsesWith(NewCI);
            if (auto *II = dyn_cast<InvokeInst>(CB))
                BranchInst::Create(II->getNormalDest(), II);
            CB->eraseFromParent();
        }

        if (F->use_empty())
            F->eraseFromParent();
        Changed = true;
    }

    /* ------------------------------------------------------------------
     * 3. Global variable replacements — stdin/stdout/stderr + errno.
     * ------------------------------------------------------------------*/
    static constexpr std::pair<const char *, const char *> StdioGlobals[] = {
        {"stdin",  "__coqui_stdin"},
        {"stdout", "__coqui_stdout"},
        {"stderr", "__coqui_stderr"},
    };

    auto *PtrTy = PointerType::getUnqual(M.getContext());

    for (auto &[OldName, NewName] : StdioGlobals) {
        GlobalVariable *GV = M.getGlobalVariable(OldName);
        if (!GV) continue;

        GlobalVariable *NewGV = M.getNamedGlobal(NewName);
        if (!NewGV) {
            NewGV = new GlobalVariable(M, PtrTy, /*isConstant=*/false,
                                       GlobalValue::ExternalLinkage,
                                       /*Initializer=*/nullptr, NewName);
        }

        errs() << "[coqui-libc] replacing global " << OldName << " -> "
               << NewName << " (" << GV->getNumUses() << " use(s))\n";

        GV->replaceAllUsesWith(NewGV);
        GV->eraseFromParent();
        Changed = true;
    }

    /* errno as direct global reference — some code does `extern int errno;`
     * and reads/writes the symbol directly. Route to __coqui_errno. */
    if (GlobalVariable *ErrnoGV = M.getGlobalVariable("errno")) {
        GlobalVariable *CoquiErrno = M.getGlobalVariable("__coqui_errno");
        if (!CoquiErrno) {
            CoquiErrno = new GlobalVariable(
                M, ErrnoGV->getValueType(), false,
                GlobalValue::ExternalLinkage, nullptr, "__coqui_errno");
        }
        errs() << "[coqui-libc] replacing global errno -> __coqui_errno ("
               << ErrnoGV->getNumUses() << " use(s))\n";
        ErrnoGV->replaceAllUsesWith(CoquiErrno);
        ErrnoGV->eraseFromParent();
        Changed = true;
    }

    return Changed;
}

} // namespace coqui
