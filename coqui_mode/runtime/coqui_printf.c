/*
 * coqui_printf.c --- device-side sprintf/snprintf wrapper.
 *
 * Instantiates the format engine from coqui_printf.inc and exposes the
 * function names (__coqui_sprintf_impl, __coqui_snprintf_impl) that
 * passes/Sprintf.cpp emits in rewritten call sites. The engine itself
 * defines *_core names; this file is the thin _impl -> _core shim.
 *
 * Modeled after /coqui/runtime/coqui_fuzz_runtime.c, which includes
 * coqui_runtime_common.inc at end-of-file; here the scope is narrower
 * (sprintf/snprintf only — no printf/fprintf/stdout), so we include
 * the two .inc files directly.
 *
 * Compiled with:
 *   clang --target=nvptx64-nvidia-cuda -O2 -ffreestanding -emit-llvm -c \
 *         -I . coqui_printf.c -o build/coqui_printf.bc
 *
 * No libc. No stdio. Any helpers the engine needs (reverse, utoa, dtoa)
 * are defined inside coqui_printf.inc; nothing external is referenced.
 * libc stubs like __coqui_strlen/__coqui_memcpy are NOT redefined here —
 * they live in coqui_libc.c and coqui_memory.c and the llvm-link step
 * resolves them at final link.
 */

#include "coqui_runtime_common.inc"
#include "coqui_printf.inc"

/* ===----------------------------------------------------------------===
 * _impl entry points
 *
 * The Sprintf LLVM pass emits calls to *_impl names (not *_core) at
 * rewritten sprintf/snprintf sites. Forward to the engine cores.
 * ===----------------------------------------------------------------=== */

__attribute__((nothrow))
int __coqui_sprintf_impl(char *buf, const char *fmt,
                         const unsigned long *args, int nargs) {
  return __coqui_sprintf_core(buf, fmt, args, nargs);
}

__attribute__((nothrow))
int __coqui_snprintf_impl(char *buf, unsigned long size, const char *fmt,
                          const unsigned long *args, int nargs) {
  return __coqui_snprintf_core(buf, size, fmt, args, nargs);
}
