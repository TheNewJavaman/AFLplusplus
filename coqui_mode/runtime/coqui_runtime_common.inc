/* coqui_runtime_common.inc --- shared ArgTag enum + helpers for the
 * device-side format engine. Ported (minimally) from
 * /coqui/runtime/coqui_runtime_common.inc.
 *
 * Only the portions required by coqui_printf.inc to implement
 * __coqui_sprintf_core / __coqui_snprintf_core are included here.
 * coqui's full runtime_common.inc is ~5200 lines covering libc, math,
 * VFS, syscalls, etc.; cuAFL keeps those concerns in their own .c
 * files (coqui_libc.c, coqui_memory.c, coqui_asan.c).
 *
 * This header is #included by coqui_printf.c prior to coqui_printf.inc
 * so that the tag constants and fmt-state helpers are visible to the
 * engine.
 */

#ifndef _COQUI_RUNTIME_COMMON_INC
#define _COQUI_RUNTIME_COMMON_INC

/* ArgTag values emitted by the Sprintf pass (passes/Sprintf.cpp::ArgTag).
 *
 * Packed args arrive as an i64 array of {tag, value} pairs:
 *   [tag0, val0, tag1, val1, ...]
 * The engine reads values by index; the tags are informational and may
 * be consulted by pass-level diagnostics. Values are always pre-widened
 * to 64 bits by the pass (sext for signed ints, bitcast for double,
 * ptrtoint for pointers, fpext+bitcast for float). */
#define COQUI_TAG_INT    0
#define COQUI_TAG_UINT   1
#define COQUI_TAG_LONG   2
#define COQUI_TAG_ULONG  3
#define COQUI_TAG_LLONG  4
#define COQUI_TAG_ULLONG 5
#define COQUI_TAG_DOUBLE 6
#define COQUI_TAG_PTR    7

#endif /* _COQUI_RUNTIME_COMMON_INC */
