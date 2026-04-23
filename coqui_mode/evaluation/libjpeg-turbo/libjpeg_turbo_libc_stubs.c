/*
 * libjpeg_turbo_libc_stubs.c --- device-side libc stubs for libjpeg-turbo.
 *
 * cuAFL's CoquiPassPlugin does NOT port libc formatted-print functions
 * (see /home/gpizarro/cuAFL/coqui_mode/passes/Libc.cpp: only strlen,
 * strcmp, strncmp, memcmp, strchr, strtod are rewritten to __coqui_*
 * equivalents — no snprintf/fprintf/vsnprintf/printf). Upstream coqui
 * provides __coqui_snprintf via musl+LibcTransform, but cuAFL's trimmed
 * runtime does not carry that port yet.
 *
 * libjpeg-turbo's jerror.c references two libc stdio functions:
 *   - SNPRINTF (→ snprintf) from format_message() — renders an error
 *     string into a caller-supplied buffer.
 *   - fprintf(stderr, ...) from output_message() — writes the formatted
 *     buffer to stderr.
 *
 * On the GPU these paths are unreachable for observability: the jerror
 * default error_exit handler calls exit(EXIT_FAILURE) → __coqui_exit and
 * the thread terminates cleanly before any buffer or stderr stream is
 * consumed. The stubs return 0 / write an empty string: behaviorally
 * equivalent for GPU fuzzing, resolve the ExternalSymbolGatekeeper.
 *
 * Additionally, jerror.c dereferences `stderr` — a FILE* pointer. On
 * NVPTX we synthesize a dummy `stderr` symbol whose value is never read
 * past the fprintf stub (which ignores its FILE* argument entirely).
 *
 * Device-only: the CPU build (produced by the coqui nix flake) picks up
 * the real libc and keeps the full error-message path.
 */

typedef unsigned long size_t;
typedef void FILE;

int snprintf(char *buf, size_t n, const char *fmt, ...) {
  (void)fmt;
  if (buf && n > 0) buf[0] = '\0';
  return 0;
}

int fprintf(FILE *stream, const char *fmt, ...) {
  (void)stream;
  (void)fmt;
  return 0;
}

/* `stderr` is declared extern in <stdio.h> as `FILE *`. Provide a dummy
 * symbol so taking its address links; the fprintf stub above never
 * dereferences the pointer. */
FILE *stderr = (FILE *)0;

/* jerror.c's default error_exit() calls exit(EXIT_FAILURE). The cuAFL
 * runtime provides __coqui_exit() (coqui_mode/runtime/coqui_runtime.c)
 * which emits PTX `exit;` so just the current thread dies and the
 * kernel continues processing the rest of the batch. cuAFL's Libc.cpp
 * does not rewrite `exit` → `__coqui_exit` yet, so forward the call
 * here.  _Noreturn so the compiler doesn't lose unreachability. */
void __coqui_exit(void);

_Noreturn void exit(int status) {
  (void)status;
  __coqui_exit();
  /* Unreachable — but we need to convince the compiler. */
  for (;;) {}
}
