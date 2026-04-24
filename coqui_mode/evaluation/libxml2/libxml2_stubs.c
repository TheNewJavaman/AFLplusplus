/* libxml2-specific device stubs.
 * libxml2 pulls in musl functions that contain svc (syscall) instructions
 * even with most features disabled. These stubs override them for the
 * single-threaded device binary.
 *
 * IMPORTANT: Do not define standard libc functions (dup, atexit,
 * pthread_once, etc.) here — they are redirected by LibcTransform
 * and stubbed in the coqui runtime. Defining them here would override
 * the real glibc versions in the host binary, breaking CUDA. */

#include <stddef.h>

/* Musl atomics stubs — globals.c initialization pulls in musl's atomic CAS
 * which has a kernel-assisted fallback containing svc instructions.
 * These are musl-internal symbols and do not conflict with glibc. */
int __a_cas(volatile int *p, int t, int s) {
    if (*p == t) { *p = s; return t; }
    return *p;
}
int __a_cas_dummy(volatile int *p, int t, int s) { return __a_cas(p, t, s); }
int __a_cas_v6(volatile int *p, int t, int s) { return __a_cas(p, t, s); }
int __a_cas_v7(volatile int *p, int t, int s) { return __a_cas(p, t, s); }
void __a_barrier(void) { }
void __a_barrier_dummy(void) { }
void __a_barrier_oldkuser(void) { }
void __a_barrier_v6(void) { }
void __a_barrier_v7(void) { }
void *__a_gettp(void) { return 0; }
void *__a_gettp_cp15(void) { return 0; }

/* Data pointers from atomics.o — prevent the archive member from being
 * pulled in. */
typedef int (*a_cas_fn)(volatile int *, int, int);
typedef void (*a_barrier_fn)(void);
typedef void *(*a_gettp_fn)(void);
a_cas_fn __a_cas_ptr = __a_cas;
a_barrier_fn __a_barrier_ptr = __a_barrier;
a_gettp_fn __a_gettp_ptr = __a_gettp;

/* __funcs_on_exit — musl-internal cleanup, not in glibc. */
void __funcs_on_exit(void) { }
