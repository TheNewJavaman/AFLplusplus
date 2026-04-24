/*
 * coqui_complex.c --- C99 complex math runtime for coqui mode on NVPTX.
 *
 * Ported from /coqui/runtime/coqui_runtime_common.inc (the "Complex math
 * functions" section) and /coqui/runtime/coqui_runtime.h.
 *
 * Complex numbers are passed as separate real/imag arguments (matching
 * clang's x86-64 IR lowering for `double _Complex`). Returns use
 * two-element structs whose layout matches the literal-struct return type
 * that Complex.cpp installs at every call site.
 *
 * cuAFL's underlying real-math runtime (__coqui_sin/cos/log/exp/pow in
 * coqui_libc.c) is a pile of stubs that return 0/1/etc. — real math
 * accuracy doesn't matter for fuzzing control flow. We keep that same
 * policy here: the complex helpers round-trip the right shape and
 * compute via the existing stubs, so downstream code sees consistent
 * (if mathematically trivial) results.
 *
 * The few primitives not already in coqui_libc.c (sqrt via PTX asm,
 * fabs, atan2 / sinh / cosh trivial stubs) are defined static to this
 * TU to avoid colliding with symbol decisions in coqui_libc.c.
 */

#include "coqui_runtime.h"

typedef struct { double real; double imag; } __coqui_cdouble;
typedef struct { float real; float imag; } __coqui_cfloat;

/* Forward decls for the real-math stubs in coqui_libc.c. */
double __coqui_log(double x);
double __coqui_exp(double x);
double __coqui_sin(double x);
double __coqui_cos(double x);

/* PTX has a native double-precision sqrt; use it. Matches the legacy
 * coqui __coqui_sqrt implementation. */
static double cx_sqrt(double x) {
    double r;
    asm("sqrt.rn.f64 %0, %1;" : "=d"(r) : "d"(x));
    return r;
}

static double cx_fabs(double x) { return x < 0.0 ? -x : x; }

/* atan2 / sinh / cosh: cuAFL's existing trig/exp stubs return constants,
 * so any "real" implementation here would still produce meaningless
 * values. Keep the shape correct and return simple approximations that
 * at least vary with the inputs so fuzz inputs don't collapse to the
 * same branch. */
static double cx_atan2(double y, double x) {
    /* Crude sign-aware stub: enough to keep fuzzer branches distinct. */
    if (x == 0.0 && y == 0.0) return 0.0;
    if (x > 0.0) return y / (cx_fabs(x) + cx_fabs(y) + 1.0);
    return (y >= 0.0 ? 1.0 : -1.0) * 3.14159265358979323846 -
           y / (cx_fabs(x) + cx_fabs(y) + 1.0);
}

static double cx_sinh(double x) {
    /* sinh(x) = (exp(x) - exp(-x)) / 2. Uses the __coqui_exp stub which
     * returns 1.0, so this always returns 0.0 — same tier of approximation
     * as the real-math fallbacks in coqui_libc.c. */
    double ex = __coqui_exp(x);
    double enx = __coqui_exp(-x);
    return (ex - enx) * 0.5;
}

static double cx_cosh(double x) {
    double ex = __coqui_exp(x);
    double enx = __coqui_exp(-x);
    return (ex + enx) * 0.5;
}

/* --- cabs / cabsf: complex absolute value --- */

double __coqui_cabs(double real, double imag) {
    return cx_sqrt(real * real + imag * imag);
}

float __coqui_cabsf(float real, float imag) {
    return (float)__coqui_cabs((double)real, (double)imag);
}

/* --- carg / cargf: complex argument (phase angle) --- */

double __coqui_carg(double real, double imag) {
    return cx_atan2(imag, real);
}

float __coqui_cargf(float real, float imag) {
    return (float)__coqui_carg((double)real, (double)imag);
}

/* --- conj / conjf: complex conjugate --- */

__coqui_cdouble __coqui_conj(double real, double imag) {
    __coqui_cdouble r;
    r.real = real;
    r.imag = -imag;
    return r;
}

__coqui_cfloat __coqui_conjf(float real, float imag) {
    __coqui_cfloat r;
    r.real = real;
    r.imag = -imag;
    return r;
}

/* --- cexp / cexpf: complex exponential ---
 * cexp(a+bi) = exp(a) * (cos(b) + i*sin(b)) */

__coqui_cdouble __coqui_cexp(double real, double imag) {
    double e = __coqui_exp(real);
    __coqui_cdouble r;
    r.real = e * __coqui_cos(imag);
    r.imag = e * __coqui_sin(imag);
    return r;
}

__coqui_cfloat __coqui_cexpf(float real, float imag) {
    __coqui_cdouble d = __coqui_cexp((double)real, (double)imag);
    __coqui_cfloat r;
    r.real = (float)d.real;
    r.imag = (float)d.imag;
    return r;
}

/* --- clog / clogf: complex logarithm ---
 * clog(z) = log(|z|) + i*arg(z) */

__coqui_cdouble __coqui_clog(double real, double imag) {
    __coqui_cdouble r;
    r.real = __coqui_log(__coqui_cabs(real, imag));
    r.imag = cx_atan2(imag, real);
    return r;
}

__coqui_cfloat __coqui_clogf(float real, float imag) {
    __coqui_cdouble d = __coqui_clog((double)real, (double)imag);
    __coqui_cfloat r;
    r.real = (float)d.real;
    r.imag = (float)d.imag;
    return r;
}

/* --- csqrt / csqrtf: complex square root ---
 * csqrt(a+bi) = sqrt((|z|+a)/2) + i*sign(b)*sqrt((|z|-a)/2) */

__coqui_cdouble __coqui_csqrt(double real, double imag) {
    if (real == 0.0 && imag == 0.0) {
        __coqui_cdouble r;
        r.real = 0.0;
        r.imag = 0.0;
        return r;
    }
    double m = __coqui_cabs(real, imag);
    double t1 = cx_sqrt((m + real) * 0.5);
    double t2 = cx_sqrt((m - real) * 0.5);
    __coqui_cdouble r;
    r.real = t1;
    r.imag = (imag >= 0.0) ? t2 : -t2;
    return r;
}

__coqui_cfloat __coqui_csqrtf(float real, float imag) {
    __coqui_cdouble d = __coqui_csqrt((double)real, (double)imag);
    __coqui_cfloat r;
    r.real = (float)d.real;
    r.imag = (float)d.imag;
    return r;
}

/* --- cpow / cpowf: complex power ---
 * cpow(z, w) = cexp(w * clog(z)) */

__coqui_cdouble __coqui_cpow(double zr, double zi, double wr, double wi) {
    __coqui_cdouble lz = __coqui_clog(zr, zi);
    /* w * log(z): (wr + i*wi) * (lr + i*li) */
    double mr = wr * lz.real - wi * lz.imag;
    double mi = wr * lz.imag + wi * lz.real;
    return __coqui_cexp(mr, mi);
}

__coqui_cfloat __coqui_cpowf(float zr, float zi, float wr, float wi) {
    __coqui_cdouble d = __coqui_cpow((double)zr, (double)zi,
                                     (double)wr, (double)wi);
    __coqui_cfloat r;
    r.real = (float)d.real;
    r.imag = (float)d.imag;
    return r;
}

/* --- csin / csinf: complex sine ---
 * csin(a+bi) = sin(a)*cosh(b) + i*cos(a)*sinh(b) */

__coqui_cdouble __coqui_csin(double real, double imag) {
    __coqui_cdouble r;
    r.real = __coqui_sin(real) * cx_cosh(imag);
    r.imag = __coqui_cos(real) * cx_sinh(imag);
    return r;
}

__coqui_cfloat __coqui_csinf(float real, float imag) {
    __coqui_cdouble d = __coqui_csin((double)real, (double)imag);
    __coqui_cfloat r;
    r.real = (float)d.real;
    r.imag = (float)d.imag;
    return r;
}

/* --- ccos / ccosf: complex cosine ---
 * ccos(a+bi) = cos(a)*cosh(b) - i*sin(a)*sinh(b) */

__coqui_cdouble __coqui_ccos(double real, double imag) {
    __coqui_cdouble r;
    r.real = __coqui_cos(real) * cx_cosh(imag);
    r.imag = -__coqui_sin(real) * cx_sinh(imag);
    return r;
}

__coqui_cfloat __coqui_ccosf(float real, float imag) {
    __coqui_cdouble d = __coqui_ccos((double)real, (double)imag);
    __coqui_cfloat r;
    r.real = (float)d.real;
    r.imag = (float)d.imag;
    return r;
}

/* --- ctan / ctanf: complex tangent ---
 * ctan(z) = csin(z) / ccos(z) */

__coqui_cdouble __coqui_ctan(double real, double imag) {
    __coqui_cdouble s = __coqui_csin(real, imag);
    __coqui_cdouble c = __coqui_ccos(real, imag);
    /* (sr + i*si) / (cr + i*ci) = ((sr*cr + si*ci) + i*(si*cr - sr*ci)) / (cr^2 + ci^2) */
    double denom = c.real * c.real + c.imag * c.imag;
    __coqui_cdouble r;
    if (denom == 0.0) {
        r.real = 0.0;
        r.imag = 0.0;
        return r;
    }
    r.real = (s.real * c.real + s.imag * c.imag) / denom;
    r.imag = (s.imag * c.real - s.real * c.imag) / denom;
    return r;
}

__coqui_cfloat __coqui_ctanf(float real, float imag) {
    __coqui_cdouble d = __coqui_ctan((double)real, (double)imag);
    __coqui_cfloat r;
    r.real = (float)d.real;
    r.imag = (float)d.imag;
    return r;
}

/* --- csinh / csinhf: complex hyperbolic sine ---
 * csinh(a+bi) = sinh(a)*cos(b) + i*cosh(a)*sin(b) */

__coqui_cdouble __coqui_csinh(double real, double imag) {
    __coqui_cdouble r;
    r.real = cx_sinh(real) * __coqui_cos(imag);
    r.imag = cx_cosh(real) * __coqui_sin(imag);
    return r;
}

__coqui_cfloat __coqui_csinhf(float real, float imag) {
    __coqui_cdouble d = __coqui_csinh((double)real, (double)imag);
    __coqui_cfloat r;
    r.real = (float)d.real;
    r.imag = (float)d.imag;
    return r;
}

/* --- ccosh / ccoshf: complex hyperbolic cosine ---
 * ccosh(a+bi) = cosh(a)*cos(b) + i*sinh(a)*sin(b) */

__coqui_cdouble __coqui_ccosh(double real, double imag) {
    __coqui_cdouble r;
    r.real = cx_cosh(real) * __coqui_cos(imag);
    r.imag = cx_sinh(real) * __coqui_sin(imag);
    return r;
}

__coqui_cfloat __coqui_ccoshf(float real, float imag) {
    __coqui_cdouble d = __coqui_ccosh((double)real, (double)imag);
    __coqui_cfloat r;
    r.real = (float)d.real;
    r.imag = (float)d.imag;
    return r;
}

/* --- ctanh / ctanhf: complex hyperbolic tangent ---
 * ctanh(z) = csinh(z) / ccosh(z) */

__coqui_cdouble __coqui_ctanh(double real, double imag) {
    __coqui_cdouble s = __coqui_csinh(real, imag);
    __coqui_cdouble c = __coqui_ccosh(real, imag);
    double denom = c.real * c.real + c.imag * c.imag;
    __coqui_cdouble r;
    if (denom == 0.0) {
        r.real = 0.0;
        r.imag = 0.0;
        return r;
    }
    r.real = (s.real * c.real + s.imag * c.imag) / denom;
    r.imag = (s.imag * c.real - s.real * c.imag) / denom;
    return r;
}

__coqui_cfloat __coqui_ctanhf(float real, float imag) {
    __coqui_cdouble d = __coqui_ctanh((double)real, (double)imag);
    __coqui_cfloat r;
    r.real = (float)d.real;
    r.imag = (float)d.imag;
    return r;
}

/* --- Compiler-generated complex multiply/divide ---
 * clang emits calls to __muldc3/__divdc3 (double) and __mulsc3/__divsc3
 * (float) for complex * and / operations. These are normally provided by
 * compiler-rt (libgcc). Provide NVPTX-safe implementations here.
 *
 * __muldc3(a, b, c, d) = (a+bi) * (c+di) = (ac-bd) + i(ad+bc)
 * __divdc3(a, b, c, d) = (a+bi) / (c+di) */

__coqui_cdouble __coqui___muldc3(double a, double b, double c, double d) {
    __coqui_cdouble r;
    r.real = a * c - b * d;
    r.imag = a * d + b * c;
    return r;
}

__coqui_cfloat __coqui___mulsc3(float a, float b, float c, float d) {
    __coqui_cdouble dd = __coqui___muldc3((double)a, (double)b,
                                          (double)c, (double)d);
    __coqui_cfloat r;
    r.real = (float)dd.real;
    r.imag = (float)dd.imag;
    return r;
}

__coqui_cdouble __coqui___divdc3(double a, double b, double c, double d) {
    double denom = c * c + d * d;
    __coqui_cdouble r;
    if (denom == 0.0) {
        r.real = 0.0;
        r.imag = 0.0;
        return r;
    }
    r.real = (a * c + b * d) / denom;
    r.imag = (b * c - a * d) / denom;
    return r;
}

__coqui_cfloat __coqui___divsc3(float a, float b, float c, float d) {
    __coqui_cdouble dd = __coqui___divdc3((double)a, (double)b,
                                          (double)c, (double)d);
    __coqui_cfloat r;
    r.real = (float)dd.real;
    r.imag = (float)dd.imag;
    return r;
}
