// oss-fuzz harness for bzip2.
// Source: https://github.com/google/oss-fuzz/blob/master/projects/bzip2/bzip2_decompress_target.c

#include "bzlib.h"
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size)
{
    int r, small;
    unsigned int nZ, nOut;

    nOut = size * 2;
    char *outbuf = malloc(nOut);
    if (!outbuf) return 0;
    small = size % 2;
    r = BZ2_bzBuffToBuffDecompress(outbuf, &nOut, (char *)data, size,
                                   small, /*verbosity=*/0);
    (void)r;
    free(outbuf);
    return 0;
}
