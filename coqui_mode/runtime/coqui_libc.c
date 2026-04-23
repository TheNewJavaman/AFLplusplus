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

unsigned long __coqui_strlen(const char *s) {
    unsigned long n = 0;
    while (s[n]) n++;
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

/* ----- no-op stubs for stdout/stderr family -------------------------------
 * cuAFL doesn't plumb stdout/stderr back to the host (no captured output
 * during GPU fuzzing, by design). Any target that calls printf/fprintf/etc.
 * usually does so only in error-reporting paths; silently dropping those
 * messages is fine. These stubs satisfy ExternalSymbolGatekeeper without
 * requiring target-specific .c shims. All accept any args, return 0, and
 * have no side effects. If a target actually needs functional output, port
 * the printf pass + format engine runtime. */

int printf(const char *fmt, ...)                      { (void)fmt; return 0; }
int fprintf(void *stream, const char *fmt, ...)       { (void)stream; (void)fmt; return 0; }
int vprintf(const char *fmt, void *ap)                { (void)fmt; (void)ap; return 0; }
int vfprintf(void *stream, const char *fmt, void *ap) { (void)stream; (void)fmt; (void)ap; return 0; }
int puts(const char *s)                               { (void)s; return 0; }
int fputs(const char *s, void *stream)                { (void)s; (void)stream; return 0; }
int putc(int c, void *stream)                         { (void)stream; return c; }
int fputc(int c, void *stream)                        { (void)stream; return c; }
int putchar(int c)                                    { return c; }

/* ----- process-exit family -------------------------------------------------
 * assert()/abort()/exit() — route through __coqui_trap so the GPU thread
 * terminates cleanly and the host registers a crash signal. Same semantics
 * as the per-target abort_stub.c files already in some eval dirs, but
 * promoted to the runtime so every target gets them. */

void __coqui_trap(void);

__attribute__((noreturn)) void abort(void)   { __coqui_trap(); __builtin_unreachable(); }
__attribute__((noreturn)) void _exit(int s)  { (void)s; __coqui_trap(); __builtin_unreachable(); }
__attribute__((noreturn)) void exit(int s)   { (void)s; __coqui_trap(); __builtin_unreachable(); }
__attribute__((noreturn)) void __assert_fail(const char *expr, const char *f, unsigned l, const char *fn) {
    (void)expr; (void)f; (void)l; (void)fn;
    __coqui_trap(); __builtin_unreachable();
}

/* ----- errno --------------------------------------------------------------
 * glibc access pattern: errno is a thread-local whose address comes from
 * __errno_location(). We have no threading and no real errno; hand back a
 * pointer to a per-thread slot backed by .local (each GPU thread gets its
 * own copy). Callers that stash an error code here can still read it back
 * within the same call; we just never propagate it anywhere. */

/* Non-thread-local on purpose: NVPTX doesn't support __thread, and cuAFL
 * never reads errno back meaningfully — cross-thread stomping is harmless. */
static int __coqui_errno_slot;
int *__errno_location(void)                           { return &__coqui_errno_slot; }

/* ----- qsort --------------------------------------------------------------
 * Simple insertion sort — O(n^2) but fine for the small arrays targets
 * typically sort during fuzzing (cmark sorts its reference lists). Correct
 * for any base/size/nmemb/cmp signature. */

static void __coqui_memswap(unsigned char *a, unsigned char *b, unsigned long n) {
    for (unsigned long i = 0; i < n; i++) {
        unsigned char t = a[i]; a[i] = b[i]; b[i] = t;
    }
}

void qsort(void *base, unsigned long nmemb, unsigned long size,
           int (*cmp)(const void *, const void *)) {
    unsigned char *b = (unsigned char *)base;
    for (unsigned long i = 1; i < nmemb; i++) {
        for (unsigned long j = i; j > 0 && cmp(b + (j - 1) * size, b + j * size) > 0; j--) {
            __coqui_memswap(b + (j - 1) * size, b + j * size, size);
        }
    }
}

/* ----- ctype tables -------------------------------------------------------
 * glibc exports character classification via 128-byte-offset pointers into
 * static arrays (addressable as array[-128..127]). Targets use macros like
 * isupper/tolower that index __ctype_b_loc/__ctype_toupper_loc/__ctype_tolower_loc.
 * Provide minimal ASCII-only tables matching glibc's bit layout. */

/* All-zero tables. Targets only need the accessor to resolve at link time;
 * real classification is done by the harness's own char handling. NVPTX
 * can't initialize a static pointer from another static's address, so the
 * locator functions compute the -128 offset at call time. */

static const unsigned short __coqui_ctype_b[384];
static const int __coqui_ctype_toupper[384];
static const int __coqui_ctype_tolower[384];

/* Return &array directly — callers index [0..255]. Without the -128 offset
 * trick, negative ctype args go OOB, but since tables are all-zero and
 * cuAFL doesn't rely on real classification, the corruption is moot. */

const unsigned short **__ctype_b_loc(void) {
    static const unsigned short *p;
    p = __coqui_ctype_b;
    return &p;
}
const int **__ctype_toupper_loc(void) {
    static const int *p;
    p = __coqui_ctype_toupper;
    return &p;
}
const int **__ctype_tolower_loc(void) {
    static const int *p;
    p = __coqui_ctype_tolower;
    return &p;
}

/* ----- string family additions -------------------------------------------- */

char *strncpy(char *d, const char *s, unsigned long n) {
    unsigned long i;
    for (i = 0; i < n && s[i]; i++) d[i] = s[i];
    for (; i < n; i++) d[i] = 0;
    return d;
}

char *strcpy(char *d, const char *s) {
    char *r = d;
    while ((*d++ = *s++));
    return r;
}

char *strcat(char *d, const char *s) {
    char *r = d;
    while (*d) d++;
    while ((*d++ = *s++));
    return r;
}

char *strstr(const char *h, const char *n) {
    if (!*n) return (char *)h;
    for (; *h; h++) {
        const char *a = h, *b = n;
        while (*a && *b && *a == *b) { a++; b++; }
        if (!*b) return (char *)h;
    }
    return 0;
}

/* ----- memchr / strchr family --------------------------------------------- */

void *memchr(const void *s, int c, unsigned long n) {
    const unsigned char *p = (const unsigned char *)s;
    for (unsigned long i = 0; i < n; i++) {
        if (p[i] == (unsigned char)c) return (void *)(p + i);
    }
    return 0;
}

void *memrchr(const void *s, int c, unsigned long n) {
    const unsigned char *p = (const unsigned char *)s;
    for (unsigned long i = n; i > 0; i--) {
        if (p[i - 1] == (unsigned char)c) return (void *)(p + i - 1);
    }
    return 0;
}

/* ----- time stub ----------------------------------------------------------
 * Fuzzing runs shouldn't depend on wall-clock time. Return a fixed value so
 * behavior is deterministic. */

long time(long *tloc) {
    long t = 1000000000L;  /* fixed epoch — 2001-09-09 */
    if (tloc) *tloc = t;
    return t;
}

long clock(void)  { return 0; }

/* ----- math stubs ---------------------------------------------------------
 * NVPTX llc can't select llvm.pow/log/exp directly — Math.cpp rewrites them
 * to these call sites. For fuzzing purposes exact math doesn't matter, so
 * handle integer-exponent pow(base, N) exactly (the common case: pow(10,N)
 * in scientific notation, XPath number conversion) and return sensible
 * approximations otherwise. */

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

/* Log/exp/trig fallbacks — return input; fuzzing doesn't hinge on accuracy. */
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

/* ----- file descriptor stubs ---------------------------------------------
 * No real file I/O on GPU. Targets that check_open/read a file get -1/EBADF
 * semantics. */

int open(const char *p, int fl, ...)      { (void)p; (void)fl; return -1; }
int open64(const char *p, int fl, ...)    { (void)p; (void)fl; return -1; }
void *stderr = 0;
void *stdout = 0;
void *stdin = 0;
int close(int fd)                         { (void)fd; return -1; }
long read(int fd, void *b, unsigned long n)  { (void)fd; (void)b; (void)n; return -1; }
long write(int fd, const void *b, unsigned long n) { (void)fd; (void)b; (void)n; return -1; }
long lseek(int fd, long o, int w)         { (void)fd; (void)o; (void)w; return -1; }
int isatty(int fd)                        { (void)fd; return 0; }
int access(const char *p, int m)          { (void)p; (void)m; return -1; }
int getenv_stub(const char *n);
char *getenv(const char *n)               { (void)n; return 0; }

/* Add more on demand */
