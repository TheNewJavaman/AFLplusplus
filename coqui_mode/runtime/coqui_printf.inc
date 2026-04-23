/* coqui_printf.inc --- device-side sprintf/snprintf format engine.
 * Ported from /coqui/runtime/coqui_printf.inc with printf/fprintf/stdout
 * paths removed.
 *
 * cuAFL does not support host-side stdio from the device: no
 * __coqui_printf_core, no __coqui_fprintf_core, no stdout buffer, no
 * sancov/trace hook. Only the buffer-writing cores used by sprintf and
 * snprintf are provided.
 *
 * Format string walker reads variadic args from a packed i64 array
 * emitted by passes/Sprintf.cpp. Arg layout is {tag, value} pairs; see
 * coqui_runtime_common.inc for tag values. The engine itself ignores
 * the tags and reads values by index — type coercion is driven by the
 * format specifier.
 *
 * Included by coqui_printf.c once, after coqui_runtime_common.inc.
 *
 * NVPTX target assumptions: no libc, no varargs, ffloat soft-abi,
 * all arithmetic is native. External symbols referenced:
 *   (none)
 */

/* ===----------------------------------------------------------------===
 * Integer-to-string conversion helpers
 * ===----------------------------------------------------------------=== */

/* Reverse a string in place. */
static void __coqui_reverse(char *s, int len) {
  int i = 0, j = len - 1;
  while (i < j) {
    char tmp = s[i];
    s[i] = s[j];
    s[j] = tmp;
    i++;
    j--;
  }
}

/* Convert unsigned long to string in given base (2-16).
 * Returns the length of the string written to buf.
 * uppercase: use 'A'-'F' for hex digits. */
static int __coqui_utoa(unsigned long val, char *buf, int base,
                         int uppercase) {
  const char *digits_lower = "0123456789abcdef";
  const char *digits_upper = "0123456789ABCDEF";
  const char *digits = uppercase ? digits_upper : digits_lower;

  if (val == 0) {
    buf[0] = '0';
    buf[1] = '\0';
    return 1;
  }

  int i = 0;
  while (val) {
    buf[i++] = digits[val % (unsigned long)base];
    val /= (unsigned long)base;
  }
  buf[i] = '\0';
  __coqui_reverse(buf, i);
  return i;
}

/* ===----------------------------------------------------------------===
 * Double-to-string conversion
 * ===----------------------------------------------------------------=== */

/* Simple double-to-string for %f format.
 * Precision is the number of digits after decimal point. */
static int __coqui_dtoa_f(double val, char *buf, int precision) {
  int i = 0;

  /* Handle negative */
  if (val < 0.0) {
    buf[i++] = '-';
    val = -val;
  }

  /* Handle infinity and NaN via bit patterns */
  {
    union {
      double d;
      unsigned long u;
    } conv;
    conv.d = val;
    unsigned long bits = conv.u;
    unsigned long exp_mask = 0x7FF0000000000000UL;
    unsigned long frac_mask = 0x000FFFFFFFFFFFFFUL;
    if ((bits & exp_mask) == exp_mask) {
      if (bits & frac_mask) {
        buf[i++] = 'n';
        buf[i++] = 'a';
        buf[i++] = 'n';
      } else {
        buf[i++] = 'i';
        buf[i++] = 'n';
        buf[i++] = 'f';
      }
      buf[i] = '\0';
      return i;
    }
  }

  /* Default precision */
  if (precision < 0)
    precision = 6;

  /* Round */
  double round_add = 0.5;
  for (int p = 0; p < precision; p++)
    round_add /= 10.0;
  val += round_add;

  /* Integer part */
  unsigned long int_part = (unsigned long)val;
  double frac_part = val - (double)int_part;

  /* Convert integer part */
  if (int_part == 0) {
    buf[i++] = '0';
  } else {
    char tmp[24];
    int len = __coqui_utoa(int_part, tmp, 10, 0);
    for (int j = 0; j < len; j++)
      buf[i++] = tmp[j];
  }

  /* Decimal point and fractional part */
  if (precision > 0) {
    buf[i++] = '.';
    for (int p = 0; p < precision; p++) {
      frac_part *= 10.0;
      int digit = (int)frac_part;
      if (digit > 9)
        digit = 9;
      buf[i++] = '0' + digit;
      frac_part -= (double)digit;
    }
  }

  buf[i] = '\0';
  return i;
}

/* Simple double-to-string for %e format (scientific notation). */
static int __coqui_dtoa_e(double val, char *buf, int precision,
                           int uppercase) {
  int i = 0;

  /* Handle negative */
  int neg = 0;
  if (val < 0.0) {
    neg = 1;
    val = -val;
  }

  /* Handle infinity/NaN */
  {
    union {
      double d;
      unsigned long u;
    } conv;
    conv.d = val;
    unsigned long bits = conv.u;
    unsigned long exp_mask = 0x7FF0000000000000UL;
    unsigned long frac_mask = 0x000FFFFFFFFFFFFFUL;
    if ((bits & exp_mask) == exp_mask) {
      if (neg)
        buf[i++] = '-';
      if (bits & frac_mask) {
        buf[i++] = 'n';
        buf[i++] = 'a';
        buf[i++] = 'n';
      } else {
        buf[i++] = 'i';
        buf[i++] = 'n';
        buf[i++] = 'f';
      }
      buf[i] = '\0';
      return i;
    }
  }

  if (precision < 0)
    precision = 6;

  /* Find exponent */
  int exp10 = 0;
  if (val != 0.0) {
    while (val >= 10.0) {
      val /= 10.0;
      exp10++;
    }
    while (val < 1.0) {
      val *= 10.0;
      exp10--;
    }
  }

  /* Now val is in [1.0, 10.0) (or 0.0) */
  if (neg)
    buf[i++] = '-';

  /* Round the mantissa */
  double round_add = 0.5;
  for (int p = 0; p < precision; p++)
    round_add /= 10.0;
  val += round_add;
  if (val >= 10.0) {
    val /= 10.0;
    exp10++;
  }

  /* Leading digit */
  int digit = (int)val;
  if (digit > 9)
    digit = 9;
  buf[i++] = '0' + digit;
  val -= (double)digit;

  if (precision > 0) {
    buf[i++] = '.';
    for (int p = 0; p < precision; p++) {
      val *= 10.0;
      digit = (int)val;
      if (digit > 9)
        digit = 9;
      buf[i++] = '0' + digit;
      val -= (double)digit;
    }
  }

  /* Exponent */
  buf[i++] = uppercase ? 'E' : 'e';
  if (exp10 < 0) {
    buf[i++] = '-';
    exp10 = -exp10;
  } else {
    buf[i++] = '+';
  }
  /* Always at least 2 digits for exponent */
  if (exp10 < 10)
    buf[i++] = '0';
  char exp_buf[8];
  int exp_len = __coqui_utoa((unsigned long)exp10, exp_buf, 10, 0);
  for (int j = 0; j < exp_len; j++)
    buf[i++] = exp_buf[j];

  buf[i] = '\0';
  return i;
}

/* %g format: use %e if exponent < -4 or >= precision, otherwise %f.
 * Removes trailing zeros. */
static int __coqui_dtoa_g(double val, char *buf, int precision,
                           int uppercase) {
  if (precision <= 0)
    precision = 1;

  int neg = 0;
  double absval = val;
  if (val < 0.0) {
    neg = 1;
    absval = -val;
  }

  /* Check for zero */
  if (absval == 0.0) {
    int i = 0;
    if (neg)
      buf[i++] = '-';
    buf[i++] = '0';
    buf[i] = '\0';
    return i;
  }

  /* Find the exponent to decide format */
  int exp10 = 0;
  double tmp = absval;
  if (tmp != 0.0) {
    while (tmp >= 10.0) {
      tmp /= 10.0;
      exp10++;
    }
    while (tmp < 1.0) {
      tmp *= 10.0;
      exp10--;
    }
  }

  /* Reconstruct the signed value for delegation to %e/%f formatters
   * (which handle their own sign output). */
  double signed_val = neg ? -absval : absval;

  int len;
  if (exp10 < -4 || exp10 >= precision) {
    len = __coqui_dtoa_e(signed_val, buf, precision - 1, uppercase);
  } else {
    /* Use %f format with adjusted precision */
    int f_prec = precision - exp10 - 1;
    if (f_prec < 0)
      f_prec = 0;
    len = __coqui_dtoa_f(signed_val, buf, f_prec);
  }

  /* Remove trailing zeros after decimal point */
  int has_dot = 0;
  for (int i = 0; i < len; i++) {
    if (buf[i] == '.') {
      has_dot = 1;
      break;
    }
  }
  if (has_dot) {
    /* Find end of number (before 'e'/'E' if present) */
    int num_end = len;
    int exp_start = -1;
    for (int i = 0; i < len; i++) {
      if (buf[i] == 'e' || buf[i] == 'E') {
        num_end = i;
        exp_start = i;
        break;
      }
    }
    /* Strip trailing zeros */
    while (num_end > 0 && buf[num_end - 1] == '0')
      num_end--;
    /* Strip trailing dot */
    if (num_end > 0 && buf[num_end - 1] == '.')
      num_end--;
    /* Move exponent part */
    if (exp_start >= 0) {
      int exp_len_part = len - exp_start;
      for (int i = 0; i < exp_len_part; i++)
        buf[num_end + i] = buf[exp_start + i];
      len = num_end + exp_len_part;
    } else {
      len = num_end;
    }
    buf[len] = '\0';
  }

  return len;
}

/* ===----------------------------------------------------------------===
 * Core sprintf engine
 *
 * Format string parser with full flag/width/precision/length support.
 * Arguments are read from a packed i64 array. Output is routed through
 * a caller-supplied write callback so the same engine services both
 * sprintf (unbounded buffer) and snprintf (bounded buffer).
 *
 * Note: the upstream coqui engine also serviced printf/fprintf via an
 * alternate write callback (__coqui_write_stdout). That callback and
 * the stdout buffer it targeted are intentionally omitted in cuAFL.
 * ===----------------------------------------------------------------=== */

/* Callback type for output: write n characters from s. */
typedef void (*__coqui_write_fn)(const char *s, unsigned long n, void *ctx);

/* Write to a char* buffer for sprintf/snprintf. */
struct __coqui_sprintf_ctx {
  char *buf;
  unsigned long pos;
  unsigned long max; /* 0 = unlimited (sprintf), >0 = snprintf limit */
};

static void __coqui_write_sprintf(const char *s, unsigned long n, void *ctx) {
  struct __coqui_sprintf_ctx *c = (struct __coqui_sprintf_ctx *)ctx;
  for (unsigned long i = 0; i < n; i++) {
    if (c->max == 0 || c->pos < c->max - 1) {
      c->buf[c->pos] = s[i];
    }
    c->pos++;
  }
}

static int __coqui_sprintf_engine(const char *fmt,
                                   const unsigned long *args, int nargs,
                                   __coqui_write_fn write_fn,
                                   void *write_ctx) {
  int chars_written = 0;
  int arg_idx = 0;

  /* Helper to get next arg as unsigned long.
   * Pass packs args as {tag, value} pairs; skip tag and consume value. */
  #define NEXT_ARG() (arg_idx < nargs ? args[arg_idx++ * 2 + 1] : 0)

  /* Helper to emit a single character */
  #define EMIT_CHAR(c)                                                         \
    do {                                                                        \
      char __ch = (c);                                                         \
      write_fn(&__ch, 1, write_ctx);                                           \
      chars_written++;                                                         \
    } while (0)

  /* Helper to emit n characters from s */
  #define EMIT_STR(s, n)                                                       \
    do {                                                                        \
      write_fn((s), (n), write_ctx);                                           \
      chars_written += (int)(n);                                               \
    } while (0)

  while (*fmt) {
    if (*fmt != '%') {
      /* Find the end of the literal run. */
      const char *start = fmt;
      while (*fmt && *fmt != '%')
        fmt++;
      EMIT_STR(start, (unsigned long)(fmt - start));
      continue;
    }

    fmt++; /* skip '%' */

    if (*fmt == '\0')
      break;

    /* Parse flags */
    int flag_minus = 0; /* left-align */
    int flag_plus = 0;  /* force sign */
    int flag_space = 0; /* space before positive numbers */
    int flag_zero = 0;  /* zero-pad */
    int flag_hash = 0;  /* alternate form */

    for (;;) {
      if (*fmt == '-') {
        flag_minus = 1;
        fmt++;
      } else if (*fmt == '+') {
        flag_plus = 1;
        fmt++;
      } else if (*fmt == ' ') {
        flag_space = 1;
        fmt++;
      } else if (*fmt == '0') {
        flag_zero = 1;
        fmt++;
      } else if (*fmt == '#') {
        flag_hash = 1;
        fmt++;
      } else {
        break;
      }
    }

    /* Parse width */
    int width = 0;
    int has_width = 0;
    if (*fmt == '*') {
      width = (int)(long)NEXT_ARG();
      if (width < 0) {
        flag_minus = 1;
        width = -width;
      }
      has_width = 1;
      fmt++;
    } else {
      while (*fmt >= '0' && *fmt <= '9') {
        width = width * 10 + (*fmt - '0');
        has_width = 1;
        fmt++;
      }
    }

    /* Parse precision */
    int precision = -1;
    if (*fmt == '.') {
      fmt++;
      precision = 0;
      if (*fmt == '*') {
        precision = (int)(long)NEXT_ARG();
        if (precision < 0)
          precision = -1; /* negative precision is treated as omitted */
        fmt++;
      } else {
        while (*fmt >= '0' && *fmt <= '9') {
          precision = precision * 10 + (*fmt - '0');
          fmt++;
        }
      }
    }

    /* Parse length modifier */
    int len_long = 0;  /* 1 = l, 2 = ll */
    int len_size = 0;  /* z = size_t */
    int len_short = 0; /* 1 = h, 2 = hh */

    if (*fmt == 'l') {
      fmt++;
      len_long = 1;
      if (*fmt == 'l') {
        fmt++;
        len_long = 2;
      }
    } else if (*fmt == 'h') {
      fmt++;
      len_short = 1;
      if (*fmt == 'h') {
        fmt++;
        len_short = 2;
      }
    } else if (*fmt == 'z') {
      fmt++;
      len_size = 1;
    } else if (*fmt == 'j' || *fmt == 't') {
      /* intmax_t, ptrdiff_t - treat as long on 64-bit */
      fmt++;
      len_long = 1;
    }

    /* Parse conversion specifier */
    char spec = *fmt;
    if (spec == '\0')
      break;
    fmt++;

    /* Handle %% */
    if (spec == '%') {
      EMIT_CHAR('%');
      continue;
    }

    /* Temporary buffer for formatting a single field */
    char num_buf[80];
    const char *str_ptr = (void *)0;
    int str_len = 0;
    char sign_char = 0;
    char prefix_buf[3] = {0, 0, 0};
    int prefix_len = 0;

    switch (spec) {
    case 'd':
    case 'i': {
      long val;
      if (len_long >= 2 || len_size)
        val = (long)NEXT_ARG();
      else if (len_long == 1)
        val = (long)NEXT_ARG();
      else if (len_short == 2)
        val = (signed char)(long)NEXT_ARG();
      else if (len_short == 1)
        val = (short)(long)NEXT_ARG();
      else
        val = (int)(long)NEXT_ARG();

      if (val < 0)
        sign_char = '-';
      else if (flag_plus)
        sign_char = '+';
      else if (flag_space)
        sign_char = ' ';

      unsigned long uval;
      if (val < 0)
        uval = (unsigned long)(-(val + 1)) + 1;
      else
        uval = (unsigned long)val;

      str_len = __coqui_utoa(uval, num_buf, 10, 0);
      str_ptr = num_buf;

      /* Handle precision (minimum digits) */
      /* precision of 0 and value of 0 means no output */
      if (precision == 0 && val == 0) {
        str_len = 0;
        str_ptr = "";
      }
      break;
    }

    case 'u': {
      unsigned long val;
      if (len_long >= 2 || len_size)
        val = NEXT_ARG();
      else if (len_long == 1)
        val = NEXT_ARG();
      else if (len_short == 2)
        val = (unsigned char)NEXT_ARG();
      else if (len_short == 1)
        val = (unsigned short)NEXT_ARG();
      else
        val = (unsigned int)NEXT_ARG();

      str_len = __coqui_utoa(val, num_buf, 10, 0);
      str_ptr = num_buf;

      if (precision == 0 && val == 0) {
        str_len = 0;
        str_ptr = "";
      }
      break;
    }

    case 'x':
    case 'X': {
      unsigned long val;
      if (len_long >= 2 || len_size)
        val = NEXT_ARG();
      else if (len_long == 1)
        val = NEXT_ARG();
      else if (len_short == 2)
        val = (unsigned char)NEXT_ARG();
      else if (len_short == 1)
        val = (unsigned short)NEXT_ARG();
      else
        val = (unsigned int)NEXT_ARG();

      str_len = __coqui_utoa(val, num_buf, 16, spec == 'X');
      str_ptr = num_buf;

      if (flag_hash && val != 0) {
        prefix_buf[0] = '0';
        prefix_buf[1] = spec; /* 'x' or 'X' */
        prefix_len = 2;
      }

      if (precision == 0 && val == 0) {
        str_len = 0;
        str_ptr = "";
      }
      break;
    }

    case 'o': {
      unsigned long val;
      if (len_long >= 2 || len_size)
        val = NEXT_ARG();
      else if (len_long == 1)
        val = NEXT_ARG();
      else
        val = (unsigned int)NEXT_ARG();

      str_len = __coqui_utoa(val, num_buf, 8, 0);
      str_ptr = num_buf;

      if (flag_hash && val != 0) {
        /* Only add '0' prefix if the number doesn't already start with '0' */
        if (str_len == 0 || num_buf[0] != '0') {
          prefix_buf[0] = '0';
          prefix_len = 1;
        }
      }

      if (precision == 0 && val == 0) {
        str_len = 0;
        str_ptr = "";
        if (flag_hash) {
          /* #o with value 0: output "0" */
          str_ptr = "0";
          str_len = 1;
          prefix_len = 0;
        }
      }
      break;
    }

    case 'c': {
      unsigned long val = NEXT_ARG();
      num_buf[0] = (char)val;
      num_buf[1] = '\0';
      str_ptr = num_buf;
      str_len = 1;
      break;
    }

    case 's': {
      unsigned long val = NEXT_ARG();
      str_ptr = (const char *)val;
      if (!str_ptr)
        str_ptr = "(null)";
      str_len = 0;
      while (str_ptr[str_len])
        str_len++;
      /* Apply precision as maximum length for strings */
      if (precision >= 0 && str_len > precision)
        str_len = precision;
      break;
    }

    case 'p': {
      unsigned long val = NEXT_ARG();
      if (val == 0) {
        str_ptr = "(nil)";
        str_len = 5;
      } else {
        str_len = __coqui_utoa(val, num_buf, 16, 0);
        str_ptr = num_buf;
        prefix_buf[0] = '0';
        prefix_buf[1] = 'x';
        prefix_len = 2;
      }
      break;
    }

    case 'f':
    case 'F': {
      /* Get the double value (stored as bitcast i64) */
      unsigned long raw = NEXT_ARG();
      union {
        unsigned long u;
        double d;
      } conv;
      conv.u = raw;

      int neg = 0;
      double val = conv.d;
      if (val < 0.0) {
        neg = 1;
        val = -val;
      }

      if (neg)
        sign_char = '-';
      else if (flag_plus)
        sign_char = '+';
      else if (flag_space)
        sign_char = ' ';

      /* Use absolute value for formatting, sign_char handles the sign */
      str_len = __coqui_dtoa_f(val, num_buf, precision < 0 ? 6 : precision);
      str_ptr = num_buf;
      break;
    }

    case 'e':
    case 'E': {
      unsigned long raw = NEXT_ARG();
      union {
        unsigned long u;
        double d;
      } conv;
      conv.u = raw;

      int neg = 0;
      double val = conv.d;
      if (val < 0.0) {
        neg = 1;
        val = -val;
      }

      if (neg)
        sign_char = '-';
      else if (flag_plus)
        sign_char = '+';
      else if (flag_space)
        sign_char = ' ';

      str_len = __coqui_dtoa_e(val, num_buf, precision < 0 ? 6 : precision,
                                spec == 'E');
      str_ptr = num_buf;
      break;
    }

    case 'g':
    case 'G': {
      unsigned long raw = NEXT_ARG();
      union {
        unsigned long u;
        double d;
      } conv;
      conv.u = raw;

      int neg = 0;
      double val = conv.d;
      if (val < 0.0) {
        neg = 1;
        val = -val;
      }

      if (neg)
        sign_char = '-';
      else if (flag_plus)
        sign_char = '+';
      else if (flag_space)
        sign_char = ' ';

      str_len = __coqui_dtoa_g(val, num_buf, precision < 0 ? 6 : precision,
                                spec == 'G');
      str_ptr = num_buf;
      break;
    }

    case 'n': {
      /* Store chars written so far. Treat the arg as a pointer to int. */
      unsigned long ptr_val = NEXT_ARG();
      if (ptr_val) {
        int *p = (int *)ptr_val;
        *p = chars_written;
      }
      continue; /* No output for %n */
    }

    default:
      /* Unknown specifier: output it literally */
      EMIT_CHAR('%');
      EMIT_CHAR(spec);
      continue;
    }

    /* --- Apply width, padding, and alignment --- */

    /* For numeric types with precision, precision means minimum digits */
    int pad_zeros_for_precision = 0;
    if (precision > 0 &&
        (spec == 'd' || spec == 'i' || spec == 'u' || spec == 'x' ||
         spec == 'X' || spec == 'o')) {
      if (str_len < precision)
        pad_zeros_for_precision = precision - str_len;
    }

    int total_len = prefix_len + (sign_char ? 1 : 0) +
                    pad_zeros_for_precision + str_len;
    int pad = 0;
    if (has_width && width > total_len)
      pad = width - total_len;

    /* Determine pad character */
    char pad_char = ' ';
    if (flag_zero && !flag_minus && precision < 0 &&
        spec != 's' && spec != 'c' && spec != 'p') {
      pad_char = '0';
    }

    if (pad_char == '0') {
      /* Zero-padding: sign/prefix first, then zeros, then digits */
      if (sign_char)
        EMIT_CHAR(sign_char);
      if (prefix_len)
        EMIT_STR(prefix_buf, prefix_len);
      for (int j = 0; j < pad_zeros_for_precision; j++)
        EMIT_CHAR('0');
      for (int j = 0; j < pad; j++)
        EMIT_CHAR('0');
      EMIT_STR(str_ptr, str_len);
    } else if (flag_minus) {
      /* Left-aligned: content first, then spaces */
      if (sign_char)
        EMIT_CHAR(sign_char);
      if (prefix_len)
        EMIT_STR(prefix_buf, prefix_len);
      for (int j = 0; j < pad_zeros_for_precision; j++)
        EMIT_CHAR('0');
      EMIT_STR(str_ptr, str_len);
      for (int j = 0; j < pad; j++)
        EMIT_CHAR(' ');
    } else {
      /* Right-aligned with spaces */
      for (int j = 0; j < pad; j++)
        EMIT_CHAR(' ');
      if (sign_char)
        EMIT_CHAR(sign_char);
      if (prefix_len)
        EMIT_STR(prefix_buf, prefix_len);
      for (int j = 0; j < pad_zeros_for_precision; j++)
        EMIT_CHAR('0');
      EMIT_STR(str_ptr, str_len);
    }
  }

  #undef NEXT_ARG
  #undef EMIT_CHAR
  #undef EMIT_STR

  return chars_written;
}

/* ===----------------------------------------------------------------===
 * Public API
 *
 * Non-variadic entry points called by the transformed IR. Only the
 * buffer-writing cores are exposed; host-side stdio is not supported.
 * ===----------------------------------------------------------------=== */

int __coqui_sprintf_core(char *buf, const char *fmt,
                         const unsigned long *args, int nargs) {
  struct __coqui_sprintf_ctx ctx;
  ctx.buf = buf;
  ctx.pos = 0;
  ctx.max = 0; /* unlimited */

  int ret = __coqui_sprintf_engine(fmt, args, nargs, __coqui_write_sprintf,
                                    (void *)&ctx);
  buf[ctx.pos] = '\0';
  return ret;
}

int __coqui_snprintf_core(char *buf, unsigned long size, const char *fmt,
                          const unsigned long *args, int nargs) {
  struct __coqui_sprintf_ctx ctx;
  ctx.buf = buf;
  ctx.pos = 0;
  ctx.max = size;

  int ret = __coqui_sprintf_engine(fmt, args, nargs, __coqui_write_sprintf,
                                    (void *)&ctx);
  /* Null-terminate */
  if (size > 0) {
    unsigned long term_pos = ctx.pos < size - 1 ? ctx.pos : size - 1;
    buf[term_pos] = '\0';
  }
  return ret;
}
