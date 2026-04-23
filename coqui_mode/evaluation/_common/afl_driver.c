/*
 * afl_driver.c --- Minimal AFL++ driver for libFuzzer-style harnesses.
 *
 * CONSTRAINT VIOLATION (documented in coqui_mode/docs/runs.md):
 * This driver uses fork-per-exec (NOT persistent mode with __AFL_LOOP).
 * The project's persistent-mode constraint (>=1000 iterations per fork)
 * is NOT satisfied here because cuAFL's --coqui forkserver path times
 * out in dry-run calibration whenever the target uses __AFL_LOOP (tested
 * with both shm-fuzz via __AFL_FUZZ_TESTCASE_BUF and stdin-fed LOOP).
 * See the issues section in coqui_mode/docs/runs.md for details.
 *
 * Behaviour:
 *   - `./bin <file>` : read one seed from file, call LLVMFuzzerTestOneInput
 *   - under afl-fuzz: read one seed from stdin (AFL pipes test case in),
 *                     call LLVMFuzzerTestOneInput, exit. AFL re-forks.
 *
 * Compile with cuAFL's afl-clang-fast.
 */

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size);
int LLVMFuzzerInitialize(int *argc, char ***argv) __attribute__((weak));

int main(int argc, char **argv) {
  if (LLVMFuzzerInitialize) LLVMFuzzerInitialize(&argc, &argv);

  static uint8_t buf[1 << 20];   /* 1 MiB — larger than default AFL max_input */

  if (argc > 1) {
    int fd = open(argv[1], O_RDONLY);
    if (fd < 0) { perror("open"); return 1; }
    ssize_t n = read(fd, buf, sizeof(buf));
    close(fd);
    if (n > 0) LLVMFuzzerTestOneInput(buf, (size_t)n);
    return 0;
  }

  ssize_t n = read(0, buf, sizeof(buf));
  if (n > 0) LLVMFuzzerTestOneInput(buf, (size_t)n);
  return 0;
}
