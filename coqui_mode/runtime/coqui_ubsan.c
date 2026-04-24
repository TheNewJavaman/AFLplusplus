/*
 * coqui_ubsan.c --- GPU-side UndefinedBehaviorSanitizer runtime.
 *
 * Ported from /coqui/runtime/coqui_ubsan.c with adjustments for cuAFL's
 * per-thread status array API (vs coqui's __coqui_get_status() /
 * __coqui_abort()).
 *
 * Provides __ubsan_handle_* functions called by clang's -fsanitize= checks.
 * Each handler records the error code via __coqui_status_array[tid], and
 * fatal (_abort suffix) variants also set ubsan_fatal and terminate the
 * thread cleanly via __coqui_exit().
 *
 * No custom LLVM pass is needed — clang instruments the IR directly when
 * compiling with -fsanitize=signed-integer-overflow,null,shift,... etc.
 * This file just provides the handler implementations that clang expects.
 *
 * Error codes (stored in coqui_status_t.ubsan_error):
 *   1  signed-integer-overflow (add/sub/mul/negate)
 *   2  integer-divide-by-zero / divrem-overflow
 *   3  shift-out-of-bounds
 *   4  type-mismatch (null deref, misalignment)
 *   5  out-of-bounds
 *   6  pointer-overflow
 *   7  builtin-unreachable / invalid-builtin
 *   8  load-invalid-value
 *   9  float-cast-overflow
 *  10  implicit-conversion
 *  11  missing-return
 *  12  vla-bound-not-positive
 *  13  nonnull-arg
 *  14  nonnull-return
 *  15  dynamic-type-cache-miss (vptr check failure)
 */

#include "coqui_runtime.h"

/* Per-thread status array (defined in coqui_runtime.c). */
extern coqui_status_t *__coqui_status_array;

/* ===-------------------------------------------------------------------===
 * Error reporting
 * ===-------------------------------------------------------------------=== */

/* Fatal report: record error, stamp crash_sig, terminate the thread.
 *
 * Follows the same pattern as coqui_asan.c::asan_report: set the classified
 * error code, stamp a partial-coverage signature so the host can dedup
 * crash-verify, then __coqui_exit() to leave the kernel running for other
 * threads. */
static void ubsan_report_fatal(int code) {
    u32 tid = __coqui_fuzz_tid();
    __coqui_status_array[tid].ubsan_fatal = 1;
    __coqui_status_array[tid].ubsan_error = (u32)code;
    __coqui_status_array[tid].crash_sig =
        __coqui_trace_sig(__coqui_cov_base());
    __coqui_exit();
}

/* Recoverable report: record error but return to caller.
 *
 * Clang emits recoverable handlers (without _abort suffix) for checks
 * that are NOT in -fno-sanitize-recover=. The handler must return so
 * the thread can continue executing — matching host behavior where
 * the recoverable handler prints a diagnostic and returns. */
static void ubsan_report_recover(int code) {
    u32 tid = __coqui_fuzz_tid();
    __coqui_status_array[tid].ubsan_error = (u32)code;
    /* Don't set ubsan_fatal; don't exit — return to caller. */
}

/* ===-------------------------------------------------------------------===
 * Handler macros
 *
 * Clang generates two variants per recoverable check:
 *   __ubsan_handle_<name>       (recoverable — continues after report)
 *   __ubsan_handle_<name>_abort (fatal — calls __coqui_exit)
 * Recoverable handlers must return (matching host semantics).
 * Fatal handlers (_abort) call ubsan_report_fatal().
 *
 * Signatures match clang's expectations:
 *   VV = (ptr data, i64 lhs, i64 rhs)
 *   V  = (ptr data, i64 val)
 *   D  = (ptr data)
 * ===-------------------------------------------------------------------=== */

/* (void *data, unsigned long lhs, unsigned long rhs) + _abort variant */
#define UBSAN_VV(name, code)                                                  \
    void __ubsan_handle_##name(void *d, unsigned long a, unsigned long b) {   \
        (void)d; (void)a; (void)b; ubsan_report_recover(code);                \
    }                                                                          \
    void __ubsan_handle_##name##_abort(void *d, unsigned long a,              \
                                       unsigned long b) {                      \
        (void)d; (void)a; (void)b; ubsan_report_fatal(code);                  \
    }

/* (void *data, unsigned long val) + _abort variant */
#define UBSAN_V(name, code)                                                   \
    void __ubsan_handle_##name(void *d, unsigned long a) {                    \
        (void)d; (void)a; ubsan_report_recover(code);                         \
    }                                                                          \
    void __ubsan_handle_##name##_abort(void *d, unsigned long a) {            \
        (void)d; (void)a; ubsan_report_fatal(code);                           \
    }

/* (void *data) + _abort variant */
#define UBSAN_D(name, code)                                                   \
    void __ubsan_handle_##name(void *d) {                                     \
        (void)d; ubsan_report_recover(code);                                  \
    }                                                                          \
    void __ubsan_handle_##name##_abort(void *d) {                             \
        (void)d; ubsan_report_fatal(code);                                    \
    }

/* (void *data) — always fatal, no _abort variant */
#define UBSAN_FATAL(name, code)                                               \
    void __ubsan_handle_##name(void *d) {                                     \
        (void)d; ubsan_report_fatal(code);                                    \
    }

/* ===-------------------------------------------------------------------===
 * Handlers
 * ===-------------------------------------------------------------------=== */

/* Signed integer overflow (add, sub, mul, negate). */
UBSAN_VV(add_overflow, 1)
UBSAN_VV(sub_overflow, 1)
UBSAN_VV(mul_overflow, 1)
UBSAN_V(negate_overflow, 1)

/* Integer divide/remainder by zero. */
UBSAN_VV(divrem_by_zero, 2)

/* Integer divide/remainder overflow (e.g. INT_MIN / -1). */
UBSAN_VV(divrem_overflow, 2)

/* Shift amount negative or >= bitwidth. */
UBSAN_VV(shift_out_of_bounds, 3)

/* Null pointer dereference, misaligned access. */
UBSAN_V(type_mismatch_v1, 4)

/* Array index out of bounds. */
UBSAN_V(out_of_bounds, 5)

/* Pointer arithmetic overflow. */
UBSAN_VV(pointer_overflow, 6)

/* __builtin_unreachable() reached. */
UBSAN_FATAL(builtin_unreachable, 7)

/* Invalid use of builtin (e.g. __builtin_ctz(0)). */
UBSAN_D(invalid_builtin, 7)

/* Load of invalid bool/enum value. */
UBSAN_V(load_invalid_value, 8)

/* Float-to-integer conversion overflow. */
UBSAN_V(float_cast_overflow, 9)

/* Implicit integer conversion (truncation, sign change). */
UBSAN_VV(implicit_conversion, 10)

/* Non-void function reached end without returning. */
UBSAN_FATAL(missing_return, 11)

/* VLA bound is not positive. */
UBSAN_V(vla_bound_not_positive, 12)

/* Nonnull argument/return violations. */
UBSAN_D(nonnull_arg, 13)

/* nonnull_return_v1 takes (ptr data, ptr loc). */
void __ubsan_handle_nonnull_return_v1(void *d, void *loc) {
    (void)d; (void)loc; ubsan_report_recover(14);
}
void __ubsan_handle_nonnull_return_v1_abort(void *d, void *loc) {
    (void)d; (void)loc; ubsan_report_fatal(14);
}

/* Dynamic type (vptr) cache miss — vtable pointer doesn't match expected
 * type. Kept for manual -fsanitize=vptr use; not enabled by default
 * (requires -frtti). */
void __ubsan_handle_dynamic_type_cache_miss(void *d, void *ptr, void *hash) {
    (void)d; (void)ptr; (void)hash; ubsan_report_recover(15);
}
void __ubsan_handle_dynamic_type_cache_miss_abort(void *d, void *ptr,
                                                   void *hash) {
    (void)d; (void)ptr; (void)hash; ubsan_report_fatal(15);
}

/* __cxa_bad_typeid: called when typeid is applied to a null polymorphic
 * pointer. Weak: libc++abi provides its own definition for C++ targets. */
__attribute__((weak)) void __cxa_bad_typeid(void) {
    ubsan_report_fatal(15);
}
