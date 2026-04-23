/*
 * libyaml_stubs.c — cuAFL-only glue for libyaml -> coqui-cc runtime.
 *
 * coqui's main compiler runs a LibcTransform pass that rewrites libc call
 * sites (strdup, memcpy, memset, ...) to their __coqui_* runtime twins.
 * cuAFL's coqui-cc driver does NOT run that pass; its ExternalSymbolGatekeeper
 * then rejects any unresolved libc symbol it finds in the linked IR.
 *
 * The cuAFL coqui-cc runtime (nm /usr/local/lib/coqui-cc/runtime.bc) exposes:
 *   __coqui_malloc, __coqui_calloc, __coqui_free, __coqui_realloc,
 *   __coqui_memcmp, __coqui_strlen, __coqui_strcmp, __coqui_strncmp,
 *   __coqui_strchr, __coqui_strtod, __coqui_trap
 * — but neither strdup, memcpy, memmove, nor memset. libyaml needs all of
 * those, so we supply small device-safe stubs here that delegate to what the
 * runtime does expose (and to the compiler's __builtin_* fallbacks where the
 * semantics match).
 *
 * These are all `static inline` hot-path primitives in libc; implementing
 * them inline here has no measurable perf cost compared to the gatekeeper
 * silently forbidding them.
 */

#include <stddef.h>

/* coqui-cc runtime exports. */
extern void *__coqui_malloc(unsigned long size);
extern unsigned long __coqui_strlen(const char *s);

/* ------------------------------------------------------------------
 * memcpy / memmove / memset
 *
 * Use __builtin_* intrinsics. Clang lowers these to NVPTX ld/st sequences
 * without emitting an external symbol named `memcpy` / `memset`, so the
 * gatekeeper is happy. The non-overlapping variant memcpy already resolves
 * via __builtin_memcpy; memmove needs an explicit loop because NVPTX's
 * __builtin_memmove may still emit an external call.
 * ------------------------------------------------------------------ */

void *memcpy(void *dst, const void *src, size_t n) {
    return __builtin_memcpy(dst, src, n);
}

void *memset(void *dst, int c, size_t n) {
    return __builtin_memset(dst, c, n);
}

void *memmove(void *dst, const void *src, size_t n) {
    unsigned char       *d = (unsigned char *)dst;
    const unsigned char *s = (const unsigned char *)src;
    if (d == s || n == 0) return dst;
    if (d < s) {
        for (size_t i = 0; i < n; i++) d[i] = s[i];
    } else {
        for (size_t i = n; i > 0; i--) d[i - 1] = s[i - 1];
    }
    return dst;
}

/* ------------------------------------------------------------------
 * strdup
 *
 * libyaml's yaml_strdup() wraps strdup() directly. Implement it via
 * __coqui_malloc + __coqui_strlen so allocations flow through the
 * GPU heap allocator (and the ASan shadow registration that rides on it).
 * ------------------------------------------------------------------ */

char *strdup(const char *s) {
    unsigned long n = __coqui_strlen(s);
    char         *p = (char *)__coqui_malloc(n + 1);
    if (!p) return NULL;
    for (unsigned long i = 0; i < n; i++) p[i] = s[i];
    p[n] = '\0';
    return p;
}
