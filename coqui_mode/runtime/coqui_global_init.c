/*
 * coqui_global_init.c --- weak fallback for __coqui_global_init.
 *
 * The GlobalCtors pass (coqui_mode/passes/GlobalCtors.cpp) lowers
 * @llvm.global_ctors into a strong __coqui_global_init definition that
 * calls each constructor in priority order, then inserts a single
 * call to __coqui_global_init() in __coqui_fuzz_kernel just before the
 * user harness runs.
 *
 * For C-only targets (or any module without @llvm.global_ctors), the
 * pass still inserts the kernel-entry call but doesn't synthesize a
 * body. This file provides a weak-linkage no-op so the call resolves
 * cleanly without forcing the pass to emit empty stub functions per
 * module.
 *
 * Linker behavior:
 *  - Module has C++ ctors → pass emits strong __coqui_global_init →
 *    linker keeps the strong, drops this weak.
 *  - C-only module → no strong override → linker keeps this weak
 *    no-op → kernel call costs one branch-and-link to a `ret;`.
 */

#include "coqui_runtime.h"

__attribute__((weak)) void __coqui_global_init(void) {
    /* No-op. Overridden by the pass-synthesized strong definition when
     * the module has any @llvm.global_ctors entries. */
}
