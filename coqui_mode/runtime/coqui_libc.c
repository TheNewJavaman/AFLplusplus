/*
 * coqui_libc.c --- minimum libc replacements for cuAFL.
 *
 * Only includes functions whose external symbols persist after clang
 * -O2 lowering (memcpy/memset family ARE lowered natively to LLVM
 * intrinsics, no replacement needed). Add functions on demand as
 * ExternalSymbolGatekeeper flags them.
 *
 * Day-1 minimum set (empirically derived from cjson after
 * internalize+globaldce pruning):
 *   strlen, strncmp, strtod
 *
 * memcpy/memset/memmove NOT replaced — clang -O2 lowers them to
 * @llvm.memcpy/@llvm.memset intrinsics; NVPTX backend handles natively.
 */

#include "coqui_runtime.h"

__attribute__((always_inline))
unsigned long __coqui_strlen(const char *s) {
    unsigned long n = 0;
    while (s[n]) n++;
    return n;
}

__attribute__((always_inline))
int __coqui_strcmp(const char *a, const char *b) {
    while (*a && *a == *b) { a++; b++; }
    return (unsigned char)*a - (unsigned char)*b;
}

__attribute__((always_inline))
int __coqui_strncmp(const char *a, const char *b, unsigned long n) {
    while (n-- && *a && *a == *b) { a++; b++; }
    if (n == (unsigned long)-1) return 0;
    return (unsigned char)*a - (unsigned char)*b;
}

__attribute__((always_inline))
int __coqui_memcmp(const void *a, const void *b, unsigned long n) {
    const unsigned char *p = (const unsigned char *)a;
    const unsigned char *q = (const unsigned char *)b;
    while (n--) {
        if (*p != *q) return (int)*p - (int)*q;
        p++; q++;
    }
    return 0;
}

__attribute__((always_inline))
char *__coqui_strchr(const char *s, int c) {
    while (*s) {
        if (*s == (char)c) return (char *)s;
        s++;
    }
    return (c == 0) ? (char *)s : (char *)0;
}

/*
 * __coqui_strtod --- parse a double from a string.
 *
 * Handles optional sign, integer digits, optional '.' with fractional
 * digits, and optional e/E exponent. No locale, no errno, no INF/NAN.
 * Good enough for cJSON's number parsing on device.
 *
 * Matches the prototype of strtod(const char *nptr, char **endptr).
 */
double __coqui_strtod(const char *nptr, char **endptr) {
    const char *p = nptr;
    double val = 0.0;
    double sign = 1.0;

    /* Skip leading whitespace */
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r' ||
           *p == '\f' || *p == '\v')
        p++;

    /* Optional sign */
    if (*p == '-') { sign = -1.0; p++; }
    else if (*p == '+') { p++; }

    /* Integer part */
    while (*p >= '0' && *p <= '9') {
        val = val * 10.0 + (*p - '0');
        p++;
    }

    /* Fractional part */
    if (*p == '.') {
        p++;
        double frac = 0.1;
        while (*p >= '0' && *p <= '9') {
            val += (*p - '0') * frac;
            frac *= 0.1;
            p++;
        }
    }

    /* Exponent part */
    if (*p == 'e' || *p == 'E') {
        p++;
        double esign = 1.0;
        if (*p == '-') { esign = -1.0; p++; }
        else if (*p == '+') { p++; }

        int exp = 0;
        while (*p >= '0' && *p <= '9') {
            exp = exp * 10 + (*p - '0');
            p++;
        }

        /* Apply exponent via repeated multiply/divide to avoid pow() */
        double base = 10.0;
        if (esign < 0.0) base = 0.1;
        while (exp--) val *= base;
    }

    if (endptr) *endptr = (char *)p;
    return sign * val;
}

/* Add more on demand */
