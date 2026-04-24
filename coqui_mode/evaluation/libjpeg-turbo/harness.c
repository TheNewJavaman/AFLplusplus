// Fuzz harness for libjpeg-turbo — decompresses JPEG data from fuzzer input.
//
// Baseline-only: progressive JPEG, multi-scan, block smoothing, IDCT scaling,
// and color quantization are all disabled at compile time to reduce NVPTX
// module size and avoid ptxas code-layout pathologies.
//
// Error handling:
//   GPU: default error_exit calls exit(EXIT_FAILURE) → __coqui_exit(1),
//        cleanly terminating the thread.
//   CPU: custom error_exit longjmps back to the harness (oss-fuzz pattern),
//        keeping the persistent-mode process alive.
//
// This harness is pure C (not C++) to avoid linking libc++, which adds ~80
// address-taken C++ virtual destructors that pollute IndirectCallTransform's
// devirtualization tables (99 candidates x 175 call sites = 17K branches).
// Switching from .cpp to .c cut the CUBIN from 5.2 MB to 3.1 MB and improved
// exec/s by ~30%.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifndef __COQUI_DEVICE__
#include <setjmp.h>
#endif
#include "jpeglib.h"

// oss-fuzz error handler: longjmp on error instead of exit().
#ifndef __COQUI_DEVICE__
struct coqui_jpeg_error_mgr {
    struct jpeg_error_mgr pub;
    jmp_buf setjmp_buffer;
};

static void coqui_error_exit(j_common_ptr cinfo) {
    struct coqui_jpeg_error_mgr *myerr =
        (struct coqui_jpeg_error_mgr *)cinfo->err;
    longjmp(myerr->setjmp_buffer, 1);
}
#endif

// Maximum image dimension.  With 64KB heap (57KB usable under ASan),
// a 64x64 3-component image uses ~12KB for pixel data + ~8KB for
// libjpeg-turbo internal structures, fitting comfortably.
#define COQUI_MAX_DIM 64

int LLVMFuzzerTestOneInput(const unsigned char *data, unsigned long size) {
    if (size < 2) return 0;

    struct jpeg_decompress_struct cinfo;
#ifdef __COQUI_DEVICE__
    struct jpeg_error_mgr jerr;
    cinfo.err = jpeg_std_error(&jerr);
    // GPU: default handler calls exit() → thread dies cleanly on error.
#else
    struct coqui_jpeg_error_mgr jerr;
    cinfo.err = jpeg_std_error(&jerr.pub);
    jerr.pub.error_exit = coqui_error_exit;
#endif

    jpeg_create_decompress(&cinfo);

#ifndef __COQUI_DEVICE__
    // CPU: longjmp back here on any libjpeg error.
    if (setjmp(jerr.setjmp_buffer)) {
        jpeg_destroy_decompress(&cinfo);
        return 0;
    }
#endif

    jpeg_mem_src(&cinfo, data, size);

    if (jpeg_read_header(&cinfo, TRUE) != JPEG_HEADER_OK) {
        jpeg_destroy_decompress(&cinfo);
        return 0;
    }

    // Reject non-8-bit precision.  The j12init_*/j16init_* functions are
    // no-op stubs (we only compile 8-bit IDCT/color paths), so 12-bit or
    // 16-bit JPEGs would leave critical decompressor modules uninitialized.
    if (cinfo.data_precision != 8) {
        jpeg_destroy_decompress(&cinfo);
        return 0;
    }

    // Reject images too large for the GPU's 64KB bump allocator.
    if (cinfo.image_width > COQUI_MAX_DIM ||
        cinfo.image_height > COQUI_MAX_DIM ||
        cinfo.image_width == 0 || cinfo.image_height == 0) {
        jpeg_destroy_decompress(&cinfo);
        return 0;
    }

    if (!jpeg_start_decompress(&cinfo)) {
        // Suspension (should not happen with memory source, but be safe).
        jpeg_destroy_decompress(&cinfo);
        return 0;
    }

    int row_stride = cinfo.output_width * cinfo.output_components;
    JSAMPARRAY buffer = (*cinfo.mem->alloc_sarray)(
        (j_common_ptr)&cinfo, JPOOL_IMAGE, row_stride, 1);

    // Guard: if jpeg_read_scanlines returns 0 (no progress), break to
    // avoid an infinite loop.  This can happen if the compressed data
    // is truncated and the decoder cannot produce output.
    while (cinfo.output_scanline < cinfo.output_height) {
        if (jpeg_read_scanlines(&cinfo, buffer, 1) == 0)
            break;
    }

    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    return 0;
}
