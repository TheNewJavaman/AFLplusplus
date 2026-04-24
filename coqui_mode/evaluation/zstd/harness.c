// GPU-adapted oss-fuzz harness for zstd simple decompression.
// Upstream: facebook/zstd tests/fuzz/simple_decompress.c
//
// The upstream harness is incompatible with the GPU allocator because:
//   1. Without STATEFUL_FUZZING, it creates and frees a ZSTD_DCtx (~34KB)
//      every call.  free() is a no-op on the bump allocator, so the DCtx
//      memory is leaked.  Worse, ZSTD_freeDCtx also frees dctx->inBuff
//      (a separate allocation), wasting more heap.
//   2. The output buffer size is randomised up to 10x the input size via
//      FUZZ_dataProducer, adding another malloc/free cycle per call.
//   3. FUZZ_dataProducer itself mallocs a 16-byte struct each call.
//
// On the GPU each thread gets exactly one LLVMFuzzerTestOneInput call per
// kernel launch, and the bump allocator resets between launches.  So per-call
// allocation is fine — the constraint is that everything must fit within a
// single call's heap budget.
//
// This harness:
//   - Allocates the DCtx once per call (~34KB with ZSTD_DECODER_INTERNAL_BUFFER=4096).
//   - Uses a fixed-size output buffer (8KB) allocated once per call.
//   - Never calls free() (no-op anyway).
//   - Eliminates FUZZ_dataProducer (no per-call struct malloc, full input goes
//     to decompression).
//
// Heap budget (64KB heap, ~58KB usable):
//   DCtx:          ~34KB
//   Output buffer:   8KB
//   Headroom:       ~16KB

#define ZSTD_STATIC_LINKING_ONLY

#include "zstd.h"
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

// Fixed output buffer size.  Large enough to exercise the full decompression
// path for small frames; zstd returns an error for frames that decompress to
// more than this, which is fine — we still exercise header parsing, entropy
// decoding, and error paths.
#define ZSTD_FUZZ_OUTPUT_SIZE 8192

int LLVMFuzzerTestOneInput(const uint8_t *src, size_t size) {
    if (size == 0)
        return 0;

    // Allocate per-call.  The bump allocator resets between kernel launches,
    // so there is no accumulation across calls.  Do NOT use static globals —
    // they would be shared across all GPU threads.
    ZSTD_DCtx *dctx = ZSTD_createDCtx();
    if (!dctx)
        return 0;

    void *out_buf = malloc(ZSTD_FUZZ_OUTPUT_SIZE);
    if (!out_buf)
        return 0;

    // Attempt decompression.  Most fuzz inputs are invalid frames, so this
    // typically returns an error quickly.  Valid frames decompress into
    // out_buf (up to ZSTD_FUZZ_OUTPUT_SIZE bytes).
    size_t const dSize = ZSTD_decompressDCtx(dctx, out_buf, ZSTD_FUZZ_OUTPUT_SIZE,
                                              src, size);

    // Validate content size metadata for successfully decompressed frames.
    if (!ZSTD_isError(dSize)) {
        unsigned long long const expectedSize = ZSTD_findDecompressedSize(src, size);
        // If the frame header declares a content size, it must match.
        if (expectedSize != ZSTD_CONTENTSIZE_ERROR &&
            expectedSize != ZSTD_CONTENTSIZE_UNKNOWN) {
            if (expectedSize != dSize) {
                // Content size mismatch — indicates a bug in the decompressor.
                abort();
            }
        }
    }

    // No free() calls needed — bump allocator free is a no-op, and the heap
    // resets between kernel launches.

    return 0;
}
