// Fuzzer harness for stb_image decoder.
// Adapted from oss-fuzz: nothings/stb/tests/stbi_read_fuzzer.c
//
// GPU adaptation: cap image dimensions to prevent kernel hangs from
// large decode loops.  The oss-fuzz original caps at ~80MB total;
// we cap width/height individually at 64 (matching the qoi harness).

#define STBI_NO_STDIO
#define STBI_NO_SIMD            /* GPU: no x86 SSE2/NEON on NVPTX */
#define STBI_NO_THREAD_LOCALS   /* GPU: StaticTransform handles per-thread globals */
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <stdlib.h>
#include <stdint.h>

// Maximum image dimension.  stbi_load_from_memory loops width*height*channels
// times; uncapped fuzzed headers cause kernel hangs from billion-iteration loops.
#define STBI_MAX_DIM 64

int LLVMFuzzerTestOneInput(const unsigned char *data, unsigned long size) {
    int x, y, channels;

    if (!stbi_info_from_memory(data, size, &x, &y, &channels))
        return 0;

    // Reject images with dimensions that would cause long decode loops on GPU.
    if (x <= 0 || y <= 0 || x > STBI_MAX_DIM || y > STBI_MAX_DIM)
        return 0;

    unsigned char *img = stbi_load_from_memory(data, size, &x, &y, &channels, 4);
    if (img)
        free(img);

    return 0;
}
