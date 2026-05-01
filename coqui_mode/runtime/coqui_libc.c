/*
 * coqui_libc.c --- device-side __coqui_* replacements for libc.
 *
 * Paired with coqui_mode/passes/Libc.cpp, which rewrites ~200 libc call sites
 * (strcat, memcpy, printf, fopen, …) to the __coqui_* variants defined here.
 * Everything here lives on-GPU; no host round-trip. Entries group as:
 *
 *   1. core string/memory        (strlen, strcmp, memcpy, …)
 *   2. ctype / locale            (isalpha, tolower, __ctype_*_loc, localeconv)
 *   3. conversion / parsing      (strtod, strtol, strto{u,i}max, abs)
 *   4. time                      (time, clock, gmtime, localtime, ctime)
 *   5. file / stdio stubs        (fopen, fread, fwrite, fd ops, stdin/out/err)
 *   6. formatted I/O stubs       (printf, fprintf, vprintf, puts, putchar)
 *   7. exit family               (__coqui_abort/assert_fail; plain exit/_exit
 *                                 kept because __coqui_exit() is the existing
 *                                 PTX thread-exit helper with void signature)
 *   8. wide string / multibyte   (wcs*, wm*, mbtowc, …)
 *   9. math fallbacks            (__coqui_pow, log/exp/trig, fmod; Math.cpp
 *                                 already rewrites llvm.pow/log/etc. -> these)
 *  10. trap-only stubs           (signal, fork, exec, dlopen, pipe, socket,
 *                                 fenv; each routes to __coqui_trap())
 *
 * Conventions:
 *   - Trap stubs take `(void)` regardless of the original libc prototype.
 *     The Libc.cpp replaceFn calls M.getOrInsertFunction with the OLD
 *     function type; LLVM inserts a bitcast when the definition's type
 *     differs, which is safe because these paths never return at runtime.
 *   - Read/write return `int` to match nvptx64 <sys/types.h> ssize_t (int),
 *     not `long`. A `long` return would mismatch user code compiled against
 *     <unistd.h> and trigger ptxas "type of argument does not match formal
 *     parameter" errors.
 */

#include "coqui_runtime.h"

void __coqui_trap(void);

/* ===========================================================================
 * 1. Core string + memory operations
 * ===========================================================================*/

unsigned long __coqui_strlen(const char *s) {
    unsigned long n = 0;
    while (s[n]) n++;
    return n;
}

unsigned long __coqui_strnlen(const char *s, unsigned long maxlen) {
    unsigned long n = 0;
    while (n < maxlen && s[n]) n++;
    return n;
}

int __coqui_strcmp(const char *a, const char *b) {
    while (*a && *a == *b) { a++; b++; }
    return (unsigned char)*a - (unsigned char)*b;
}

int __coqui_strncmp(const char *a, const char *b, unsigned long n) {
    while (n-- && *a && *a == *b) { a++; b++; }
    if (n == (unsigned long)-1) return 0;
    return (unsigned char)*a - (unsigned char)*b;
}

int __coqui_memcmp(const void *a, const void *b, unsigned long n) {
    const unsigned char *p = (const unsigned char *)a;
    const unsigned char *q = (const unsigned char *)b;
    while (n--) {
        if (*p != *q) return (int)*p - (int)*q;
        p++; q++;
    }
    return 0;
}

char *__coqui_strchr(const char *s, int c) {
    while (*s) {
        if (*s == (char)c) return (char *)s;
        s++;
    }
    return (c == 0) ? (char *)s : (char *)0;
}

char *__coqui_strrchr(const char *s, int c) {
    const char *last = 0;
    do {
        if (*s == (char)c) last = s;
    } while (*s++);
    return (char *)last;
}

char *__coqui_strcpy(char *d, const char *s) {
    char *r = d;
    while ((*d++ = *s++));
    return r;
}

char *__coqui_strncpy(char *d, const char *s, unsigned long n) {
    unsigned long i;
    for (i = 0; i < n && s[i]; i++) d[i] = s[i];
    for (; i < n; i++) d[i] = 0;
    return d;
}

char *__coqui_strcat(char *d, const char *s) {
    char *r = d;
    while (*d) d++;
    while ((*d++ = *s++));
    return r;
}

char *__coqui_strstr(const char *h, const char *n) {
    if (!*n) return (char *)h;
    for (; *h; h++) {
        const char *a = h, *b = n;
        while (*a && *b && *a == *b) { a++; b++; }
        if (!*b) return (char *)h;
    }
    return 0;
}

unsigned long __coqui_strspn(const char *s, const char *accept) {
    unsigned long n = 0;
    for (; *s; s++) {
        const char *a = accept;
        while (*a && *a != *s) a++;
        if (!*a) break;
        n++;
    }
    return n;
}

unsigned long __coqui_strcspn(const char *s, const char *reject) {
    unsigned long n = 0;
    for (; *s; s++) {
        const char *r = reject;
        while (*r && *r != *s) r++;
        if (*r) break;
        n++;
    }
    return n;
}

char *__coqui_strpbrk(const char *s, const char *accept) {
    for (; *s; s++) {
        const char *a = accept;
        while (*a) {
            if (*s == *a) return (char *)s;
            a++;
        }
    }
    return 0;
}

/* strtok: non-reentrant — static state per thread is fine (each GPU thread
 * has its own IR instance after outlining). */
static char *__coqui_strtok_state;
char *__coqui_strtok(char *str, const char *delim) {
    if (str) __coqui_strtok_state = str;
    if (!__coqui_strtok_state) return 0;
    /* Skip leading delimiters */
    char *p = __coqui_strtok_state;
    while (*p) {
        const char *d = delim;
        int match = 0;
        while (*d) { if (*p == *d) { match = 1; break; } d++; }
        if (!match) break;
        p++;
    }
    if (!*p) { __coqui_strtok_state = 0; return 0; }
    char *tok = p;
    while (*p) {
        const char *d = delim;
        while (*d) { if (*p == *d) { *p = 0; __coqui_strtok_state = p + 1; return tok; } d++; }
        p++;
    }
    __coqui_strtok_state = 0;
    return tok;
}

char *__coqui_strerror(int errnum) {
    switch (errnum) {
    case 0:  return (char *)"Success";
    case 1:  return (char *)"Operation not permitted";
    case 2:  return (char *)"No such file or directory";
    case 5:  return (char *)"Input/output error";
    case 9:  return (char *)"Bad file descriptor";
    case 12: return (char *)"Cannot allocate memory";
    case 13: return (char *)"Permission denied";
    case 17: return (char *)"File exists";
    case 22: return (char *)"Invalid argument";
    case 28: return (char *)"No space left on device";
    case 34: return (char *)"Numerical result out of range";
    case 38: return (char *)"Function not implemented";
    default: return (char *)"Unknown error";
    }
}

void *__coqui_memchr(const void *s, int c, unsigned long n) {
    const unsigned char *p = (const unsigned char *)s;
    for (unsigned long i = 0; i < n; i++) {
        if (p[i] == (unsigned char)c) return (void *)(p + i);
    }
    return 0;
}

/* ===========================================================================
 * memcpy / memmove / memset — byte-loop versions (legacy, explicit calls)
 * ===========================================================================*/

void *__coqui_memcpy(void *dst, const void *src, unsigned long n) {
    unsigned char *d = (unsigned char *)dst;
    const unsigned char *s = (const unsigned char *)src;
    for (unsigned long i = 0; i < n; i++) d[i] = s[i];
    return dst;
}

void *__coqui_memmove(void *dst, const void *src, unsigned long n) {
    unsigned char *d = (unsigned char *)dst;
    const unsigned char *s = (const unsigned char *)src;
    if (d < s || d >= s + n) {
        for (unsigned long i = 0; i < n; i++) d[i] = s[i];
    } else {
        for (unsigned long i = n; i > 0; i--) d[i - 1] = s[i - 1];
    }
    return dst;
}

void *__coqui_memset(void *s, int c, unsigned long n) {
    unsigned char *p = (unsigned char *)s;
    for (unsigned long i = 0; i < n; i++) p[i] = (unsigned char)c;
    return s;
}

/* ===========================================================================
 * memcpy_fast / memmove_fast / memset_fast — word-aligned transfers
 *
 * Used by the MemIntrinsics pass to replace llvm.memcpy/memset/memmove
 * intrinsics which LLVM's NVPTX backend otherwise lowers to byte loops.
 * Uses 8-byte transfers for the aligned body (8× fewer instructions).
 * ===========================================================================*/

/* --- 8-byte aligned variants (compiler proved both ptrs ≥ 8-aligned) --- */

void *__coqui_memcpy_a8(void *dst, const void *src, unsigned long n) {
    unsigned long *d8 = (unsigned long *)dst;
    const unsigned long *s8 = (const unsigned long *)src;
    unsigned long words = n >> 3;
    for (unsigned long i = 0; i < words; i++)
        d8[i] = s8[i];
    unsigned char *dt = (unsigned char *)(d8 + words);
    const unsigned char *st = (const unsigned char *)(s8 + words);
    for (unsigned long i = 0; i < (n & 7); i++)
        dt[i] = st[i];
    return dst;
}

void *__coqui_memmove_a8(void *dst, const void *src, unsigned long n) {
    unsigned char *d = (unsigned char *)dst;
    const unsigned char *s = (const unsigned char *)src;
    if (d < s || d >= s + n)
        return __coqui_memcpy_a8(dst, src, n);
    /* backward 8-byte copy */
    unsigned long tail = n & 7;
    d += n; s += n;
    for (unsigned long i = 0; i < tail; i++)
        *--d = *--s;
    unsigned long *d8 = (unsigned long *)d;
    const unsigned long *s8 = (const unsigned long *)s;
    for (unsigned long i = (n >> 3); i > 0; i--)
        *--d8 = *--s8;
    return dst;
}

void *__coqui_memset_a8(void *s, int c, unsigned long n) {
    unsigned char val = (unsigned char)c;
    unsigned long fill = val;
    fill |= fill << 8;
    fill |= fill << 16;
    fill |= fill << 32;
    unsigned long *p8 = (unsigned long *)s;
    unsigned long words = n >> 3;
    for (unsigned long i = 0; i < words; i++)
        p8[i] = fill;
    unsigned char *pt = (unsigned char *)(p8 + words);
    for (unsigned long i = 0; i < (n & 7); i++)
        pt[i] = val;
    return s;
}

/* --- 4-byte aligned variants (compiler proved both ptrs ≥ 4-aligned) --- */

void *__coqui_memcpy_a4(void *dst, const void *src, unsigned long n) {
    unsigned int *d4 = (unsigned int *)dst;
    const unsigned int *s4 = (const unsigned int *)src;
    unsigned long words = n >> 2;
    for (unsigned long i = 0; i < words; i++)
        d4[i] = s4[i];
    unsigned char *dt = (unsigned char *)(d4 + words);
    const unsigned char *st = (const unsigned char *)(s4 + words);
    for (unsigned long i = 0; i < (n & 3); i++)
        dt[i] = st[i];
    return dst;
}

void *__coqui_memmove_a4(void *dst, const void *src, unsigned long n) {
    unsigned char *d = (unsigned char *)dst;
    const unsigned char *s = (const unsigned char *)src;
    if (d < s || d >= s + n)
        return __coqui_memcpy_a4(dst, src, n);
    unsigned long tail = n & 3;
    d += n; s += n;
    for (unsigned long i = 0; i < tail; i++)
        *--d = *--s;
    unsigned int *d4 = (unsigned int *)d;
    const unsigned int *s4 = (const unsigned int *)s;
    for (unsigned long i = (n >> 2); i > 0; i--)
        *--d4 = *--s4;
    return dst;
}

void *__coqui_memset_a4(void *s, int c, unsigned long n) {
    unsigned char val = (unsigned char)c;
    unsigned int fill = val;
    fill |= fill << 8;
    fill |= fill << 16;
    unsigned int *p4 = (unsigned int *)s;
    unsigned long words = n >> 2;
    for (unsigned long i = 0; i < words; i++)
        p4[i] = fill;
    unsigned char *pt = (unsigned char *)(p4 + words);
    for (unsigned long i = 0; i < (n & 3); i++)
        pt[i] = val;
    return s;
}

/* ===========================================================================
 * strdup / strndup — allocate via __coqui_malloc
 * ===========================================================================*/

char *__coqui_strdup(const char *s) {
    if (!s) return 0;
    unsigned long len = __coqui_strlen(s) + 1;
    char *dup = (char *)__coqui_malloc(len);
    if (dup) __coqui_memcpy(dup, s, len);
    return dup;
}

char *__coqui_strndup(const char *s, unsigned long n) {
    if (!s) return 0;
    unsigned long len = __coqui_strlen(s);
    if (len > n) len = n;
    char *dup = (char *)__coqui_malloc(len + 1);
    if (dup) {
        __coqui_memcpy(dup, s, len);
        dup[len] = 0;
    }
    return dup;
}

/* ===========================================================================
 * 2. ctype / locale
 * ===========================================================================*/

int __coqui_isalpha(int c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'); }
int __coqui_isdigit(int c) { return c >= '0' && c <= '9'; }
int __coqui_isalnum(int c) { return __coqui_isalpha(c) || __coqui_isdigit(c); }
int __coqui_isspace(int c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v'; }
int __coqui_isupper(int c) { return c >= 'A' && c <= 'Z'; }
int __coqui_islower(int c) { return c >= 'a' && c <= 'z'; }
int __coqui_isxdigit(int c) { return __coqui_isdigit(c) || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F'); }
int __coqui_iscntrl(int c) { return (c >= 0 && c < 32) || c == 127; }
int __coqui_isprint(int c) { return c >= 32 && c < 127; }
int __coqui_isgraph(int c) { return c > 32 && c < 127; }
int __coqui_ispunct(int c) {
    return __coqui_isprint(c) && !__coqui_isalnum(c) && !__coqui_isspace(c);
}
int __coqui_tolower(int c) { return (c >= 'A' && c <= 'Z') ? c + 32 : c; }
int __coqui_toupper(int c) { return (c >= 'a' && c <= 'z') ? c - 32 : c; }

/* glibc-style ctype table accessors. All-zero tables are fine — callers that
 * actually need classification use __coqui_isalpha etc. directly (above). */

static const unsigned short __coqui_ctype_b[384];
static const int __coqui_ctype_toupper_tbl[384];
static const int __coqui_ctype_tolower_tbl[384];

const unsigned short **__coqui_ctype_b_loc(void) {
    static const unsigned short *p;
    p = __coqui_ctype_b;
    return &p;
}
const int **__coqui_ctype_toupper_loc(void) {
    static const int *p;
    p = __coqui_ctype_toupper_tbl;
    return &p;
}
const int **__coqui_ctype_tolower_loc(void) {
    static const int *p;
    p = __coqui_ctype_tolower_tbl;
    return &p;
}

/* localeconv: return a pointer to a zero-initialized struct. Most target
 * code only reads .decimal_point; a NUL byte at offset 0 is safe. */
struct __coqui_lconv {
    char *decimal_point;
    char *thousands_sep;
    char *grouping;
    char *int_curr_symbol;
    char *currency_symbol;
    char *mon_decimal_point;
    char *mon_thousands_sep;
    char *mon_grouping;
    char *positive_sign;
    char *negative_sign;
    char int_frac_digits;
    char frac_digits;
    char p_cs_precedes;
    char p_sep_by_space;
    char n_cs_precedes;
    char n_sep_by_space;
    char p_sign_posn;
    char n_sign_posn;
};
static struct __coqui_lconv __coqui_lconv_zero = {
    (char *)".", (char *)"", (char *)"",
    (char *)"", (char *)"", (char *)"", (char *)"", (char *)"",
    (char *)"", (char *)"",
    127, 127, 127, 127, 127, 127, 127, 127
};
struct __coqui_lconv *__coqui_localeconv(void) {
    return &__coqui_lconv_zero;
}

/* ===========================================================================
 * 3. Conversion / parsing
 * ===========================================================================*/

int __coqui_abs(int x) { return x < 0 ? -x : x; }

/* Internal: skip whitespace, parse sign. Returns 1 for negative. */
static int __coqui_scan_sign(const char **pp) {
    const char *p = *pp;
    while (__coqui_isspace((unsigned char)*p)) p++;
    int neg = 0;
    if (*p == '-') { neg = 1; p++; }
    else if (*p == '+') { p++; }
    *pp = p;
    return neg;
}

/* strtod — parses sign, digits, optional '.', exponent. Hex (0x1.fp10),
 * inf/infinity, and nan recognized as separate prefix-gated branches before
 * the decimal path. Decimal path retains PR #22's exponent saturation at 400
 * + zero short-circuit; do NOT touch the decimal walk without re-running
 * cjson kernel-stuck regression check. */
double __coqui_strtod(const char *nptr, char **endptr) {
    const char *p = nptr;
    int neg = __coqui_scan_sign(&p);

    /* Inf / Infinity (case-insensitive). Optional "inity" tail. */
    if ((p[0] == 'i' || p[0] == 'I') &&
        (p[1] == 'n' || p[1] == 'N') &&
        (p[2] == 'f' || p[2] == 'F')) {
        p += 3;
        if ((p[0] == 'i' || p[0] == 'I') && (p[1] == 'n' || p[1] == 'N') &&
            (p[2] == 'i' || p[2] == 'I') && (p[3] == 't' || p[3] == 'T') &&
            (p[4] == 'y' || p[4] == 'Y'))
            p += 5;
        if (endptr) *endptr = (char *)p;
        return neg ? -__builtin_inf() : __builtin_inf();
    }

    /* NaN (case-insensitive). Optional payload "(...)" not parsed. */
    if ((p[0] == 'n' || p[0] == 'N') &&
        (p[1] == 'a' || p[1] == 'A') &&
        (p[2] == 'n' || p[2] == 'N')) {
        p += 3;
        if (endptr) *endptr = (char *)p;
        return neg ? -__builtin_nan("") : __builtin_nan("");
    }

    /* Hex float: 0x or 0X, only if followed by hex digit or '.'. */
    if (p[0] == '0' && (p[1] == 'x' || p[1] == 'X')) {
        char c2 = p[2];
        int c2_hex = (c2 >= '0' && c2 <= '9') || (c2 >= 'a' && c2 <= 'f') ||
                     (c2 >= 'A' && c2 <= 'F') || c2 == '.';
        if (c2_hex) {
            const char *hp = p + 2;
            double val = 0.0;
            int has_digits = 0;
            /* Hex integer part. */
            while (1) {
                int d;
                if (*hp >= '0' && *hp <= '9') d = *hp - '0';
                else if (*hp >= 'a' && *hp <= 'f') d = *hp - 'a' + 10;
                else if (*hp >= 'A' && *hp <= 'F') d = *hp - 'A' + 10;
                else break;
                val = val * 16.0 + d;
                has_digits = 1;
                hp++;
            }
            /* Hex fractional part. */
            if (*hp == '.') {
                hp++;
                double frac = 1.0 / 16.0;
                while (1) {
                    int d;
                    if (*hp >= '0' && *hp <= '9') d = *hp - '0';
                    else if (*hp >= 'a' && *hp <= 'f') d = *hp - 'a' + 10;
                    else if (*hp >= 'A' && *hp <= 'F') d = *hp - 'A' + 10;
                    else break;
                    val += d * frac;
                    frac /= 16.0;
                    has_digits = 1;
                    hp++;
                }
            }
            /* Binary exponent (p/P). Same saturation philosophy as decimal:
             * cap at 1100 (DBL_MAX_EXP=1024) to bound the iterative loop on
             * pathological inputs like "0x1p99999999999". Zero short-circuit
             * preserved — if mantissa is 0, skip the multiplication loop. */
            if (has_digits && (*hp == 'p' || *hp == 'P')) {
                hp++;
                int en = 0;
                if (*hp == '-') { en = 1; hp++; }
                else if (*hp == '+') { hp++; }
                int e = 0;
                while (*hp >= '0' && *hp <= '9') {
                    if (e < 2000) { e = e * 10 + (*hp - '0'); }
                    hp++;
                }
                if (e > 1100) e = 1100;
                if (val != 0.0) {
                    double base = en ? 0.5 : 2.0;
                    while (e--) val *= base;
                }
            }
            if (has_digits) {
                if (endptr) *endptr = (char *)hp;
                return neg ? -val : val;
            }
            /* No hex digits after 0x — fall through; endptr will point at
             * nptr (no conversion) since p still points at '0'. */
        }
    }

    /* --- Decimal path (PR #22 perf-critical; do not modify) --- */
    double val = 0.0;
    while (*p >= '0' && *p <= '9') {
        val = val * 10.0 + (*p - '0');
        p++;
    }
    if (*p == '.') {
        p++;
        double frac = 0.1;
        while (*p >= '0' && *p <= '9') {
            val += (*p - '0') * frac;
            frac *= 0.1;
            p++;
        }
    }
    if (*p == 'e' || *p == 'E') {
        p++;
        int en = 0;
        if (*p == '-') { en = 1; p++; }
        else if (*p == '+') { p++; }
        /* Saturate exponent at 400 (well past IEEE 754 double's 10^308 limit
         * but bounded). Anything past 400 produces ±inf or 0 in double, so the
         * naive loop's extra iterations are useless work. Without the cap a
         * pathological input like "0E0100000000000000000" overflows int and
         * the `while (e--)` loop runs ~10^9 iterations per thread — found via
         * cjson havoc, locks the GPU kernel for tens of seconds. */
        int e = 0;
        while (*p >= '0' && *p <= '9') {
            if (e < 1000) { e = e * 10 + (*p - '0'); }
            p++;
        }
        if (e > 400) e = 400;
        /* val == 0 short-circuit: 0 * anything is 0, skip the multiplication
         * loop entirely. */
        if (val != 0.0) {
            double base = en ? 0.1 : 10.0;
            while (e--) val *= base;
        }
    }
    if (endptr) *endptr = (char *)p;
    return neg ? -val : val;
}

float __coqui_strtof(const char *nptr, char **endptr) {
    return (float)__coqui_strtod(nptr, endptr);
}

/* Integer parsing. Respects base 0 (auto-detect 0x/0). */
static unsigned long long __coqui_scan_uint(const char *p, char **endptr, int base) {
    unsigned long long val = 0;
    if (base == 0) {
        if (p[0] == '0' && (p[1] == 'x' || p[1] == 'X')) { base = 16; p += 2; }
        else if (p[0] == '0') { base = 8; p++; }
        else base = 10;
    } else if (base == 16 && p[0] == '0' && (p[1] == 'x' || p[1] == 'X')) {
        p += 2;
    }
    while (*p) {
        int digit;
        if (*p >= '0' && *p <= '9') digit = *p - '0';
        else if (*p >= 'a' && *p <= 'z') digit = *p - 'a' + 10;
        else if (*p >= 'A' && *p <= 'Z') digit = *p - 'A' + 10;
        else break;
        if (digit >= base) break;
        val = val * base + digit;
        p++;
    }
    if (endptr) *endptr = (char *)p;
    return val;
}

long __coqui_strtol(const char *nptr, char **endptr, int base) {
    const char *p = nptr;
    int neg = __coqui_scan_sign(&p);
    unsigned long long v = __coqui_scan_uint(p, endptr, base);
    return neg ? -(long)v : (long)v;
}

unsigned long __coqui_strtoul(const char *nptr, char **endptr, int base) {
    const char *p = nptr;
    int neg = __coqui_scan_sign(&p);
    unsigned long long v = __coqui_scan_uint(p, endptr, base);
    return neg ? (unsigned long)-(long long)v : (unsigned long)v;
}

long long __coqui_strtoll(const char *nptr, char **endptr, int base) {
    const char *p = nptr;
    int neg = __coqui_scan_sign(&p);
    unsigned long long v = __coqui_scan_uint(p, endptr, base);
    return neg ? -(long long)v : (long long)v;
}

unsigned long long __coqui_strtoull(const char *nptr, char **endptr, int base) {
    const char *p = nptr;
    int neg = __coqui_scan_sign(&p);
    unsigned long long v = __coqui_scan_uint(p, endptr, base);
    return neg ? (unsigned long long)-(long long)v : v;
}

long long __coqui_strtoimax(const char *nptr, char **endptr, int base) {
    return __coqui_strtoll(nptr, endptr, base);
}

unsigned long long __coqui_strtoumax(const char *nptr, char **endptr, int base) {
    return __coqui_strtoull(nptr, endptr, base);
}

/* ===========================================================================
 * qsort — insertion sort (fine for the small arrays fuzzing typically hits).
 * ===========================================================================*/

static void __coqui_memswap(unsigned char *a, unsigned char *b, unsigned long n) {
    for (unsigned long i = 0; i < n; i++) {
        unsigned char t = a[i]; a[i] = b[i]; b[i] = t;
    }
}

void __coqui_qsort(void *base, unsigned long nmemb, unsigned long size,
                   int (*cmp)(const void *, const void *)) {
    unsigned char *b = (unsigned char *)base;
    for (unsigned long i = 1; i < nmemb; i++) {
        for (unsigned long j = i;
             j > 0 && cmp(b + (j - 1) * size, b + j * size) > 0; j--) {
            __coqui_memswap(b + (j - 1) * size, b + j * size, size);
        }
    }
}

/* ===========================================================================
 * 4. Time stubs — deterministic fixed epoch.
 * ===========================================================================*/

long __coqui_time(long *tloc) {
    long t = 1000000000L;  /* 2001-09-09, fixed for determinism */
    if (tloc) *tloc = t;
    return t;
}

/* Plain `clock` isn't in the rewrite list; keep plain-named. */
long clock(void)  { return 0; }

struct __coqui_tm {
    int tm_sec, tm_min, tm_hour, tm_mday, tm_mon, tm_year;
    int tm_wday, tm_yday, tm_isdst;
};
static struct __coqui_tm __coqui_tm_zero;
struct __coqui_tm *__coqui_gmtime(const long *timer)    { (void)timer; return &__coqui_tm_zero; }
struct __coqui_tm *__coqui_localtime(const long *timer) { (void)timer; return &__coqui_tm_zero; }
static char __coqui_ctime_buf[] = "Thu Jan  1 00:00:00 1970\n";
char *__coqui_ctime(const long *timer) { (void)timer; return __coqui_ctime_buf; }

/* ===========================================================================
 * 5. File / stdio stubs — no real file I/O on GPU. fopen returns NULL so
 *    targets take the error path; fd ops return -1/EBADF equivalent.
 * ===========================================================================*/

void *__coqui_stdin  = (void *)1;
void *__coqui_stdout = (void *)2;
void *__coqui_stderr = (void *)3;

void *__coqui_fopen(const char *path, const char *mode) {
    (void)path; (void)mode; return 0;
}
int __coqui_fclose(void *stream) { (void)stream; return 0; }

unsigned long __coqui_fread(void *ptr, unsigned long size,
                            unsigned long nmemb, void *stream) {
    (void)ptr; (void)size; (void)nmemb; (void)stream; return 0;
}
unsigned long __coqui_fwrite(const void *ptr, unsigned long size,
                             unsigned long nmemb, void *stream) {
    (void)ptr; (void)size; (void)nmemb; (void)stream; return nmemb;
}

int   __coqui_fseek(void *s, long o, int w)     { (void)s; (void)o; (void)w; return -1; }
long  __coqui_ftell(void *s)                    { (void)s; return -1; }
int   __coqui_feof(void *s)                     { (void)s; return 1; }
int   __coqui_ferror(void *s)                   { (void)s; return 0; }
void  __coqui_clearerr(void *s)                 { (void)s; }
int   __coqui_fgetc(void *s)                    { (void)s; return -1; }
int   __coqui_fputc(int c, void *s)             { (void)s; return c; }
char *__coqui_fgets(char *b, int n, void *s)    { (void)b; (void)n; (void)s; return 0; }
int   __coqui_fputs(const char *p, void *s)     { (void)p; (void)s; return 0; }
int   __coqui_ungetc(int c, void *s)            { (void)s; return c; }
void  __coqui_rewind(void *s)                   { (void)s; }
int   __coqui_fflush(void *s)                   { (void)s; return 0; }
int   __coqui_fileno(void *s)                   { (void)s; return -1; }

/* POSIX fd ops. Return widths match nvptx64 <sys/types.h> ssize_t (int),
 * not `long` — see top-of-file comment. */
int __coqui_open(const char *p, int fl, ...)              { (void)p; (void)fl; return -1; }
int __coqui_close_fd(int fd)                              { (void)fd; return -1; }
int __coqui_read(int fd, void *b, unsigned long n)        { (void)fd; (void)b; (void)n; return -1; }
int __coqui_write(int fd, const void *b, unsigned long n) { (void)fd; (void)b; (void)n; return -1; }
int __coqui_dup(int oldfd)                                { (void)oldfd; return -1; }
int __coqui_remove(const char *path)                      { (void)path; return -1; }

/* Plain-named fallbacks for fd ops not (yet) in the rewrite list. */
long lseek(int fd, long o, int w)         { (void)fd; (void)o; (void)w; return -1; }
int  isatty(int fd)                       { (void)fd; return 0; }
int  access(const char *p, int m)         { (void)p; (void)m; return -1; }
char *getenv(const char *n)               { (void)n; return 0; }

/* ===========================================================================
 * 6. Formatted I/O — no-op variadic stubs. Sprintf.cpp rewrites sprintf/
 *    snprintf to non-variadic impls; printf/fprintf/v[fs]printf remain
 *    variadic and compile to these silent stubs.
 * ===========================================================================*/

int __coqui_printf(const char *fmt, ...)
    { (void)fmt; return 0; }
int __coqui_fprintf(void *stream, const char *fmt, ...)
    { (void)stream; (void)fmt; return 0; }
int __coqui_vprintf(const char *fmt, __builtin_va_list ap)
    { (void)fmt; (void)ap; return 0; }
int __coqui_vfprintf(void *stream, const char *fmt, __builtin_va_list ap)
    { (void)stream; (void)fmt; (void)ap; return 0; }
int __coqui_vsprintf(char *buf, const char *fmt, __builtin_va_list ap) {
    (void)fmt; (void)ap; if (buf) buf[0] = 0; return 0;
}
int __coqui_vsnprintf(char *buf, unsigned long size, const char *fmt, __builtin_va_list ap) {
    (void)fmt; (void)ap; if (buf && size) buf[0] = 0; return 0;
}
int __coqui_puts(const char *s)               { (void)s; return 0; }
int __coqui_putchar(int c)                    { return c; }
/* sscanf stub. cJSON's print_number uses sscanf("%lg") to round-trip-check
 * a printed double; returning 0 (no fields matched) makes the caller fall
 * through to the safe high-precision (%1.17g) format. Functionally correct,
 * just bypasses the precision optimization. */
int __coqui_sscanf(const char *str, const char *fmt, ...)
    { (void)str; (void)fmt; return 0; }
int __coqui_vsscanf(const char *str, const char *fmt, __builtin_va_list ap)
    { (void)str; (void)fmt; (void)ap; return 0; }

/* sprintf / snprintf stubs retained as variadic so Sprintf.cpp can intercept
 * any direct calls before Libc.cpp runs. If Sprintf has already rewritten a
 * call, there's no user of __coqui_sprintf left — these stubs just provide a
 * fallback body if something slips through. */
int __coqui_sprintf(char *buf, const char *fmt, ...)
    { (void)fmt; if (buf) buf[0] = 0; return 0; }
int __coqui_snprintf(char *buf, unsigned long size, const char *fmt, ...)
    { (void)fmt; if (buf && size) buf[0] = 0; return 0; }

/* ===========================================================================
 * 7. Exit family — abort/assert_fail route to __coqui_trap.
 *
 *    exit / _exit keep their plain libc names: the existing cuAFL runtime
 *    symbol __coqui_exit is a (void)-signatured PTX thread-exit helper,
 *    NOT the libc-semantics exit(int). Remapping exit would collide. We
 *    therefore leave `exit` and `_exit` off the Libc.cpp rewrite list;
 *    their call sites link to the plain definitions below.
 * ===========================================================================*/

__attribute__((noreturn)) void __coqui_abort(void) {
    __coqui_trap_with_reason(COQUI_TRAP_ABORT); __builtin_unreachable();
}
__attribute__((noreturn)) void __coqui_assert_fail(const char *expr, const char *file,
                                                   unsigned int line, const char *func) {
    (void)expr; (void)file; (void)line; (void)func;
    __coqui_trap_with_reason(COQUI_TRAP_ABORT); __builtin_unreachable();
}

__attribute__((noreturn)) void exit(int s)  { (void)s; __coqui_trap_with_reason(COQUI_TRAP_ABORT); __builtin_unreachable(); }
__attribute__((noreturn)) void _exit(int s) { (void)s; __coqui_trap_with_reason(COQUI_TRAP_ABORT); __builtin_unreachable(); }

/* ===========================================================================
 * errno — per-thread slot, addressed via __coqui_errno_location().
 * The address can also be reached as &__coqui_errno (see Libc.cpp's
 * global-var replacement pass, which rewrites `errno` → `__coqui_errno`).
 * ===========================================================================*/

int __coqui_errno;
int *__coqui_errno_location(void) { return &__coqui_errno; }

/* ===========================================================================
 * atexit / pthread_once — single-threaded stubs. pthread_once calls init
 * exactly once per thread; we use a simple flag check since each GPU thread
 * has its own IR instance after outlining (no cross-thread races).
 * ===========================================================================*/

int __coqui_atexit(void (*fn)(void)) { (void)fn; return 0; }
int __coqui_cxa_atexit(void (*fn)(void *), void *arg, void *dso) {
    (void)fn; (void)arg; (void)dso; return 0;
}
void __coqui_cxa_finalize(void *dso) { (void)dso; }

int __coqui_pthread_once(void *once_control, void (*init_routine)(void)) {
    int *ctl = (int *)once_control;
    if (*ctl == 0) { init_routine(); *ctl = 2; }
    return 0;
}

/* ===========================================================================
 * rand family — linear congruential. Deterministic so fuzzing is
 * reproducible from the same seed.
 * ===========================================================================*/

static unsigned int __coqui_rand_state = 1;
void __coqui_srand(unsigned int seed) { __coqui_rand_state = seed; }
int __coqui_rand(void) {
    __coqui_rand_state = __coqui_rand_state * 1103515245u + 12345u;
    return (int)((__coqui_rand_state / 65536u) % 32768u);
}
int __coqui_rand_r(unsigned int *seedp) {
    *seedp = *seedp * 1103515245u + 12345u;
    return (int)((*seedp / 65536u) % 32768u);
}

/* ===========================================================================
 * 8. Wide string / multibyte — fuzz inputs rarely use wchar_t in a
 *    correctness-sensitive way. Provide the usual byte-parallel logic on
 *    32-bit wchars; mb<->wc conversion does plain UTF-8 single-byte.
 * ===========================================================================*/

unsigned long __coqui_wcslen(const int *s) {
    unsigned long n = 0;
    while (s[n]) n++;
    return n;
}
int __coqui_wcscmp(const int *a, const int *b) {
    while (*a && *a == *b) { a++; b++; }
    return (*a > *b) - (*a < *b);
}
int __coqui_wcsncmp(const int *a, const int *b, unsigned long n) {
    while (n && *a && *a == *b) { a++; b++; n--; }
    if (!n) return 0;
    return (*a > *b) - (*a < *b);
}
int *__coqui_wmemcpy(int *d, const int *s, unsigned long n) {
    for (unsigned long i = 0; i < n; i++) d[i] = s[i];
    return d;
}
int *__coqui_wmemset(int *d, int c, unsigned long n) {
    for (unsigned long i = 0; i < n; i++) d[i] = c;
    return d;
}
int *__coqui_wcscpy(int *d, const int *s) {
    int *r = d;
    while ((*d++ = *s++));
    return r;
}
int *__coqui_wcsncpy(int *d, const int *s, unsigned long n) {
    unsigned long i;
    for (i = 0; i < n && s[i]; i++) d[i] = s[i];
    for (; i < n; i++) d[i] = 0;
    return d;
}
int *__coqui_wcscat(int *d, const int *s) {
    int *r = d;
    while (*d) d++;
    while ((*d++ = *s++));
    return r;
}
int *__coqui_wcschr(const int *s, int c) {
    while (*s) { if (*s == c) return (int *)s; s++; }
    return (c == 0) ? (int *)s : 0;
}
int *__coqui_wcsrchr(const int *s, int c) {
    const int *last = 0;
    do { if (*s == c) last = s; } while (*s++);
    return (int *)last;
}

unsigned long __coqui_wcrtomb(char *s, int wc, void *ps) {
    (void)ps;
    if (!s) return 1;
    if (wc < 0x80) { s[0] = (char)wc; return 1; }
    s[0] = '?';
    return 1;
}
unsigned long __coqui_mbrtowc(int *pwc, const char *s, unsigned long n, void *ps) {
    (void)ps;
    if (!s || !n) return 0;
    if (pwc) *pwc = (unsigned char)*s;
    return (*s == 0) ? 0 : 1;
}
int __coqui_mbtowc(int *pwc, const char *s, unsigned long n) {
    if (!s) return 0;
    if (!n) return -1;
    if (pwc) *pwc = (unsigned char)*s;
    return (*s == 0) ? 0 : 1;
}
int __coqui_wctomb(char *s, int wc) {
    if (!s) return 0;
    if (wc < 0x80) { s[0] = (char)wc; return 1; }
    s[0] = '?';
    return 1;
}
int __coqui_mblen(const char *s, unsigned long n) {
    if (!s) return 0;
    if (!n) return -1;
    return (*s == 0) ? 0 : 1;
}

/* ===========================================================================
 * 9. Math fallbacks — Math.cpp rewrites llvm.pow/log/exp and frem to these.
 * Exact math is not required for fuzzing correctness; just produce sensible
 * output so control flow downstream is stable.
 * ===========================================================================*/

double __coqui_pow(double x, double y) {
    if (y == 0.0) return 1.0;
    if (x == 0.0) return 0.0;
    long iy = (long)y;
    if ((double)iy == y && iy >= -64 && iy <= 64) {
        double r = 1.0;
        double base = (iy < 0) ? (1.0 / x) : x;
        long n = (iy < 0) ? -iy : iy;
        while (n--) r *= base;
        return r;
    }
    return 1.0;
}
float __coqui_powf(float x, float y) { return (float)__coqui_pow((double)x, (double)y); }

double __coqui_log(double x)    { (void)x; return 0.0; }
double __coqui_log10(double x)  { (void)x; return 0.0; }
double __coqui_log2(double x)   { (void)x; return 0.0; }
double __coqui_exp(double x)    { (void)x; return 1.0; }
double __coqui_exp2(double x)   { (void)x; return 1.0; }
double __coqui_sin(double x)    { (void)x; return 0.0; }
double __coqui_cos(double x)    { (void)x; return 1.0; }
float __coqui_logf(float x)     { (void)x; return 0.0f; }
float __coqui_log10f(float x)   { (void)x; return 0.0f; }
float __coqui_log2f(float x)    { (void)x; return 0.0f; }
float __coqui_expf(float x)     { (void)x; return 1.0f; }
float __coqui_exp2f(float x)    { (void)x; return 1.0f; }
float __coqui_sinf(float x)     { (void)x; return 0.0f; }
float __coqui_cosf(float x)     { (void)x; return 1.0f; }

double __coqui_fmod(double x, double y) {
    if (y == 0.0 || __builtin_isnan(x) || __builtin_isnan(y) ||
        __builtin_isinf(x))
        return __builtin_nan("");
    if (__builtin_isinf(y)) return x;
    int neg = (x < 0.0);
    double ax = __builtin_fabs(x);
    double ay = __builtin_fabs(y);
    if (ax < ay) return x;
    double q = __builtin_floor(ax / ay);
    double r = ax - q * ay;
    return neg ? -r : r;
}
float __coqui_fmodf(float x, float y) {
    return (float)__coqui_fmod((double)x, (double)y);
}

/* ===========================================================================
 * 10. Trap-only stubs — calls trap. Declared `(void)` regardless of original
 *     signature; LLVM inserts a bitcast at the call site when the pass's
 *     getOrInsertFunction sees the mismatch. Trap paths never return.
 * ===========================================================================*/

#define COQUI_TRAP_STUB(name) \
    int name(void) { __coqui_trap(); return 0; }
#define COQUI_TRAP_STUB_PTR(name) \
    void *name(void) { __coqui_trap(); return 0; }
#define COQUI_TRAP_STUB_STR(name) \
    char *name(void) { __coqui_trap(); return 0; }

/* Signals */
COQUI_TRAP_STUB_PTR(__coqui_signal)
COQUI_TRAP_STUB(__coqui_sigaction)
COQUI_TRAP_STUB(__coqui_kill)
COQUI_TRAP_STUB(__coqui_raise)
COQUI_TRAP_STUB(__coqui_sigprocmask)
COQUI_TRAP_STUB(__coqui_sigsuspend)

/* Process creation */
COQUI_TRAP_STUB(__coqui_fork)
COQUI_TRAP_STUB(__coqui_vfork)

/* Exec */
COQUI_TRAP_STUB(__coqui_execve)
COQUI_TRAP_STUB(__coqui_execvp)
COQUI_TRAP_STUB(__coqui_execl)
COQUI_TRAP_STUB(__coqui_system)

/* Dynamic linking */
COQUI_TRAP_STUB_PTR(__coqui_dlopen)
COQUI_TRAP_STUB_PTR(__coqui_dlsym)
COQUI_TRAP_STUB(__coqui_dlclose)
COQUI_TRAP_STUB_STR(__coqui_dlerror)

/* Pipes/sockets */
COQUI_TRAP_STUB(__coqui_pipe)
COQUI_TRAP_STUB(__coqui_pipe2)
COQUI_TRAP_STUB(__coqui_socket)
COQUI_TRAP_STUB(__coqui_connect)
COQUI_TRAP_STUB(__coqui_bind)
COQUI_TRAP_STUB(__coqui_listen)
COQUI_TRAP_STUB(__coqui_accept)
COQUI_TRAP_STUB(__coqui_send)
COQUI_TRAP_STUB(__coqui_recv)
COQUI_TRAP_STUB(__coqui_sendto)
COQUI_TRAP_STUB(__coqui_recvfrom)

/* Floating-point environment */
COQUI_TRAP_STUB(__coqui_fegetround)
COQUI_TRAP_STUB(__coqui_fesetround)
COQUI_TRAP_STUB(__coqui_feclearexcept)
COQUI_TRAP_STUB(__coqui_feraiseexcept)
COQUI_TRAP_STUB(__coqui_fetestexcept)
COQUI_TRAP_STUB(__coqui_fegetenv)
COQUI_TRAP_STUB(__coqui_fesetenv)

/* Note: __coqui_cabs / __coqui_carg are defined in coqui_complex.c. */

/* ===========================================================================
 * 11. SyscallTransform stubs (task 22) — divertable syscalls rewritten by
 *     coqui_mode/passes/SyscallTransform.cpp.
 *
 * Three flavors:
 *   - no-op: return 0 / NULL silently. signal-handler installation,
 *     environment writes, sigset manipulators — none of these have a
 *     meaningful GPU implementation, but trapping on them would prevent
 *     fuzz targets that gratuitously call them from running at all.
 *   - trap-with-reason: __coqui_trap_with_reason(<COQUI_TRAP_SYSCALL_*>);
 *     stamps a category-specific reason byte in the per-thread status
 *     slot and exits the thread cleanly, so the host's rerun pipeline
 *     can decode the failure category instead of seeing a generic crash.
 *   - deterministic-return: time / pid / uid stubs return a fixed value
 *     so targets that read these take stable control-flow paths.
 *
 * All stubs use the `_stub` suffix to avoid colliding with the generic-
 * trap stubs already defined above (and rewritten to by Libc.cpp). The
 * legacy stubs are retained for the existing rewrite path; the new ones
 * sit alongside them and are addressed exclusively from SyscallTransform.
 * ===========================================================================*/

void __coqui_trap_with_reason(u8 reason);

/* --- Signal handlers — silent no-op --- */
/* signal(int signum, void (*handler)(int)) returns the previous handler,
 * SIG_DFL if no prior install. SIG_DFL == NULL is fine. */
void *__coqui_signal_stub(int signum, void *handler) {
    (void)signum; (void)handler;
    return (void *)0;
}
int __coqui_sigaction_stub(int signum, const void *act, void *oldact) {
    (void)signum; (void)act; (void)oldact;
    return 0;
}
int __coqui_sigprocmask_stub(int how, const void *set, void *oldset) {
    (void)how; (void)set; (void)oldset;
    return 0;
}
int __coqui_sigemptyset_stub(void *set)            { (void)set; return 0; }
int __coqui_sigfillset_stub(void *set)             { (void)set; return 0; }
int __coqui_sigaddset_stub(void *set, int signum)  { (void)set; (void)signum; return 0; }
int __coqui_sigdelset_stub(void *set, int signum)  { (void)set; (void)signum; return 0; }
int __coqui_sigismember_stub(const void *set, int signum) { (void)set; (void)signum; return 0; }

/* --- Signal delivery — trap with KILL reason --- */
__attribute__((noreturn))
int __coqui_kill_stub(int pid, int sig) {
    (void)pid; (void)sig;
    __coqui_trap_with_reason(COQUI_TRAP_SYSCALL_KILL);
    __builtin_unreachable();
}
__attribute__((noreturn))
int __coqui_raise_stub(int sig) {
    (void)sig;
    __coqui_trap_with_reason(COQUI_TRAP_SYSCALL_KILL);
    __builtin_unreachable();
}

/* --- Process creation — trap with FORK reason --- */
__attribute__((noreturn))
int __coqui_fork_stub(void) {
    __coqui_trap_with_reason(COQUI_TRAP_SYSCALL_FORK);
    __builtin_unreachable();
}
__attribute__((noreturn))
int __coqui_vfork_stub(void) {
    __coqui_trap_with_reason(COQUI_TRAP_SYSCALL_FORK);
    __builtin_unreachable();
}

/* --- Process replacement — trap with EXEC reason. The cuAFL pass uses
 *     getOrInsertFunction(OldType) which inserts a bitcast at the call
 *     site when the runtime stub's type differs from libc's variadic
 *     prototypes. Since the stubs are noreturn the bitcast is harmless. */
__attribute__((noreturn))
int __coqui_execve_stub(const char *p, char *const argv[], char *const envp[]) {
    (void)p; (void)argv; (void)envp;
    __coqui_trap_with_reason(COQUI_TRAP_SYSCALL_EXEC);
    __builtin_unreachable();
}
__attribute__((noreturn))
int __coqui_execv_stub(const char *p, char *const argv[]) {
    (void)p; (void)argv;
    __coqui_trap_with_reason(COQUI_TRAP_SYSCALL_EXEC);
    __builtin_unreachable();
}
__attribute__((noreturn))
int __coqui_execvp_stub(const char *file, char *const argv[]) {
    (void)file; (void)argv;
    __coqui_trap_with_reason(COQUI_TRAP_SYSCALL_EXEC);
    __builtin_unreachable();
}
__attribute__((noreturn))
int __coqui_execlp_stub(const char *file, const char *arg0) {
    (void)file; (void)arg0;
    __coqui_trap_with_reason(COQUI_TRAP_SYSCALL_EXEC);
    __builtin_unreachable();
}
__attribute__((noreturn))
int __coqui_execl_stub(const char *path, const char *arg0) {
    (void)path; (void)arg0;
    __coqui_trap_with_reason(COQUI_TRAP_SYSCALL_EXEC);
    __builtin_unreachable();
}
__attribute__((noreturn))
int __coqui_execle_stub(const char *path, const char *arg0) {
    (void)path; (void)arg0;
    __coqui_trap_with_reason(COQUI_TRAP_SYSCALL_EXEC);
    __builtin_unreachable();
}
__attribute__((noreturn))
int __coqui_system_stub(const char *cmd) {
    (void)cmd;
    __coqui_trap_with_reason(COQUI_TRAP_SYSCALL_EXEC);
    __builtin_unreachable();
}

/* --- Time — deterministic fixed epoch (matches __coqui_time above). ---
 * Targets reading time take stable control-flow paths on every fuzz run. */

/* timeval / timespec layouts mirror linux <sys/time.h> / <time.h>. We
 * declare them locally with __coqui_ prefix so we don't pull in any host
 * headers in the freestanding NVPTX build. */
struct __coqui_timeval  { long tv_sec; long tv_usec; };
struct __coqui_timespec { long tv_sec; long tv_nsec; };

int __coqui_gettimeofday_stub(struct __coqui_timeval *tv, void *tz) {
    (void)tz;
    if (tv) { tv->tv_sec = 0; tv->tv_usec = 0; }
    return 0;
}
int __coqui_clock_gettime_stub(int clk_id, struct __coqui_timespec *tp) {
    (void)clk_id;
    if (tp) { tp->tv_sec = 0; tp->tv_nsec = 0; }
    return 0;
}
long __coqui_time_stub(long *tloc) {
    if (tloc) *tloc = 0;
    return 0;
}

/* --- Environment — no env on GPU. Read returns NULL, write returns 0
 *     (success) so error checks don't take the failure path. --- */
char *__coqui_getenv_stub(const char *name) {
    (void)name; return (char *)0;
}
int __coqui_setenv_stub(const char *name, const char *value, int overwrite) {
    (void)name; (void)value; (void)overwrite; return 0;
}
int __coqui_putenv_stub(char *string) {
    (void)string; return 0;
}
int __coqui_unsetenv_stub(const char *name) {
    (void)name; return 0;
}

/* --- Process / user identity — fixed non-zero values so root-check paths
 *     (`if (getuid() == 0)`) take the unprivileged branch deterministically. */
int __coqui_getpid_stub(void)  { return 1; }
int __coqui_getppid_stub(void) { return 1; }
int __coqui_getuid_stub(void)  { return 1000; }
int __coqui_geteuid_stub(void) { return 1000; }
int __coqui_getgid_stub(void)  { return 1000; }
int __coqui_getegid_stub(void) { return 1000; }
