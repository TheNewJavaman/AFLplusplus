// Fuzz harness for libpng — reads PNG data from fuzzer input with
// input-driven transform selection to maximize code coverage.
//
// C harness to avoid libc++ linking overhead.  The previous .cpp version
// linked libc++ (~303 C++ functions, 30% of CUBIN), inflating CUBIN from
// ~3M to 5.7M.  This harness uses no C++ features.
//
// Modeled after libpng's official oss-fuzz harness
// (contrib/oss-fuzz/libpng_read_fuzzer.cc) but extended with:
//   - Input-driven transform selection (bytes after signature choose transforms)
//   - Ancillary chunk info reading (text, time, phys, gamma, sRGB)
//   - Interlace handling for multi-pass PNGs
//   - Both color expansion and reduction paths
//
// On GPU: compiled with -D PNG_NO_SETJMP so libpng uses PNG_ABORT()
// (-> abort() -> __coqui_abort).  Thread dies cleanly on error.
// On CPU: compiled with PNG_SETJMP_SUPPORTED so libpng longjmps back
// to the harness on error, keeping the process alive for persistent mode.

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#ifdef PNG_SETJMP_SUPPORTED
#include <setjmp.h>
#endif

#include <png.h>

struct PngReadState {
  const uint8_t *data;
  size_t size;
  size_t offset;
};

struct PngWriteState {
  uint8_t buf[4096];
  size_t pos;
};

static void png_write_data_callback(png_structp png_ptr, png_bytep data,
                                    png_size_t length) {
  struct PngWriteState *state =
      (struct PngWriteState *)png_get_io_ptr(png_ptr);
  size_t avail = sizeof(state->buf) - state->pos;
  if (length > avail)
    length = avail;
  memcpy(state->buf + state->pos, data, length);
  state->pos += length;
}

static void png_flush_callback(png_structp png_ptr) {
  (void)png_ptr;
}

static void png_read_data_callback(png_structp png_ptr, png_bytep out,
                                   png_size_t length) {
  struct PngReadState *state =
      (struct PngReadState *)png_get_io_ptr(png_ptr);
  if (state->offset + length > state->size) {
    png_error(png_ptr, "read past end of data");
    return;
  }
  memcpy(out, state->data + state->offset, length);
  state->offset += length;
}

// Use two bytes from the input to select which transforms to enable.
// Each bit enables a different transform combination.  The fuzzer will
// naturally explore different bit patterns, covering all transform paths.
//
// Bit layout of transform_byte (data[size-1]):
//   bit 0: png_set_strip_alpha vs png_set_add_alpha
//   bit 1: png_set_packing / png_set_packswap
//   bit 2: png_set_bgr
//   bit 3: png_set_swap (byte-swap 16-bit)
//   bit 4: png_set_interlace_handling
//   bit 5: png_set_rgb_to_gray vs png_set_gray_to_rgb
//   bit 6: png_set_invert_mono / png_set_invert_alpha
//   bit 7: png_set_swap_alpha / png_set_filler
//
// Bit layout of extra_byte (data[size-2]):
//   bit 0: enable write-back path (PNG write after read)
//   bit 1: png_set_gamma (screen 2.2, file 0.45455)
//   bit 2: png_set_background (white background compositing)
//   bit 3: png_set_alpha_mode (standard alpha, sRGB)

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  // Need at least the 8-byte PNG signature + 1 extra byte for transforms.
  // (We use data[size-1] and data[size-2] as control bytes.)
  if (size < 9)
    return 0;

  // Verify PNG signature.
  if (png_sig_cmp(data, 0, 8) != 0)
    return 0;

  // Use the last byte of the input as the transform selector.
  // This byte is outside the PNG structure, so mutating it never corrupts
  // the PNG data.  The fuzzer can independently explore all 256 transform
  // combinations while keeping the PNG payload valid.
  uint8_t transform_byte = data[size - 1];
  uint8_t extra_byte = data[size - 2];

  png_structp png_ptr =
      png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
  if (!png_ptr)
    return 0;

  png_infop info_ptr = png_create_info_struct(png_ptr);
  if (!info_ptr) {
    png_destroy_read_struct(&png_ptr, NULL, NULL);
    return 0;
  }

  png_infop end_info_ptr = png_create_info_struct(png_ptr);
  if (!end_info_ptr) {
    png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
    return 0;
  }

#ifdef PNG_SETJMP_SUPPORTED
  // On CPU: longjmp back here on libpng error, clean up and return.
  if (setjmp(png_jmpbuf(png_ptr))) {
    png_destroy_read_struct(&png_ptr, &info_ptr, &end_info_ptr);
    return 0;
  }
#endif
  // On GPU (PNG_NO_SETJMP): libpng calls PNG_ABORT() -> thread dies cleanly.

  // Cap per-chunk allocation at 8KB and cache at most 128 chunks.
  // Without this, fuzz-mutated ancillary chunks (iCCP, tEXt, zTXt)
  // can trigger unbounded allocations that exhaust the 64KB per-thread
  // heap regardless of slab pool size (~0.4% OOM rate at any heap size).
  png_set_chunk_malloc_max(png_ptr, 8192);
  png_set_chunk_cache_max(png_ptr, 128);

  // Disable CRC checking so the fuzzer can explore deeper without
  // needing valid checksums (same approach as the oss-fuzz harness).
  png_set_crc_action(png_ptr, PNG_CRC_QUIET_USE, PNG_CRC_QUIET_USE);

#ifdef PNG_IGNORE_ADLER32
  // Also ignore ADLER32 checksums in zlib stream for deeper exploration.
  png_set_option(png_ptr, PNG_IGNORE_ADLER32, PNG_OPTION_ON);
#endif

  struct PngReadState state = {data, size, 0};
  png_set_read_fn(png_ptr, &state, png_read_data_callback);

  // Read info header.
  png_read_info(png_ptr, info_ptr);

  // Query image properties from the IHDR chunk.
  png_uint_32 width, height;
  int bit_depth, color_type, interlace_type, compression_type, filter_type;

  if (!png_get_IHDR(png_ptr, info_ptr, &width, &height, &bit_depth,
                    &color_type, &interlace_type, &compression_type,
                    &filter_type)) {
    png_destroy_read_struct(&png_ptr, &info_ptr, &end_info_ptr);
    return 0;
  }

  // Limit dimensions to prevent decode loops from hanging the kernel.
  // 64x64 cap matches libjpeg-turbo.
  if (width > 64 || height > 64 || width == 0 || height == 0) {
    png_destroy_read_struct(&png_ptr, &info_ptr, &end_info_ptr);
    return 0;
  }

  // Exercise png_get_channels / png_get_bit_depth / png_get_color_type
  // to cover the easy-access API paths.
  png_byte channels = png_get_channels(png_ptr, info_ptr);
  (void)png_get_bit_depth(png_ptr, info_ptr);
  (void)png_get_color_type(png_ptr, info_ptr);
  (void)channels;

  // --- Read ancillary chunk info (exercises pngget.c paths) ---

#ifdef PNG_TEXT_SUPPORTED
  {
    png_textp text_ptr = NULL;
    int num_text = 0;
    png_get_text(png_ptr, info_ptr, &text_ptr, &num_text);
    // Just reading — don't need to use the values.
    (void)text_ptr;
    (void)num_text;
  }
#endif

#ifdef PNG_tIME_SUPPORTED
  {
    png_timep mod_time = NULL;
    png_get_tIME(png_ptr, info_ptr, &mod_time);
    (void)mod_time;
  }
#endif

#ifdef PNG_pHYs_SUPPORTED
  {
    png_uint_32 res_x = 0, res_y = 0;
    int unit_type = 0;
    png_get_pHYs(png_ptr, info_ptr, &res_x, &res_y, &unit_type);
    (void)res_x;
    (void)res_y;
    (void)unit_type;
  }
#endif

#ifdef PNG_gAMA_SUPPORTED
  {
    double file_gamma = 0.0;
    png_get_gAMA(png_ptr, info_ptr, &file_gamma);
    (void)file_gamma;
  }
#endif

#ifdef PNG_sRGB_SUPPORTED
  {
    int srgb_intent = 0;
    png_get_sRGB(png_ptr, info_ptr, &srgb_intent);
    (void)srgb_intent;
  }
#endif

  // --- Apply input-driven transforms ---

  // Always expand sub-8-bit and palette images to at least 8-bit.
  // This is the baseline (same as oss-fuzz harness).
  png_set_expand(png_ptr);

  // Always scale 16-bit down to 8-bit for manageable output.
  png_set_scale_16(png_ptr);

  // Always convert transparency chunks to alpha channel.
  png_set_tRNS_to_alpha(png_ptr);

  // Bit 0: strip alpha vs add alpha.
  // Exercises both the alpha-removal and alpha-addition paths.
  if (transform_byte & 0x01) {
    png_set_strip_alpha(png_ptr);
  } else {
    // Add an opaque alpha channel (filler byte 0xFF, after RGB).
    png_set_add_alpha(png_ptr, 0xFF, PNG_FILLER_AFTER);
  }

  // Bit 1: packing transforms.
  // png_set_packing unpacks sub-byte pixels to 1 pixel per byte.
  // png_set_packswap reverses the bit order within bytes.
  if (transform_byte & 0x02) {
    png_set_packing(png_ptr);
    png_set_packswap(png_ptr);
  }

  // Bit 2: BGR channel ordering.
  // Exercises the channel-swap code path in pngtrans.c.
  if (transform_byte & 0x04) {
    png_set_bgr(png_ptr);
  }

  // Bit 3: byte-swap 16-bit values.
  // Exercises the 16-bit byte swap path (before scale_16 reduces to 8-bit,
  // this still hits the transform registration code).
  if (transform_byte & 0x08) {
    png_set_swap(png_ptr);
  }

  // Bit 4: interlace handling.
  // For interlaced PNGs, this causes multi-pass reading.
  // For non-interlaced PNGs, returns 1 pass (harmless).
  int passes = 1;
  if (transform_byte & 0x10) {
    passes = png_set_interlace_handling(png_ptr);
  }

  // Bit 5: color space conversion direction.
  // Exercises gray_to_rgb OR rgb_to_gray — mutually exclusive.
  if (transform_byte & 0x20) {
    // Convert RGB to grayscale using default coefficients.
    // error_action=1: no warning on non-gray pixels.
    png_set_rgb_to_gray_fixed(png_ptr, 1, -1, -1);
  } else {
    // Convert grayscale to RGB (same as oss-fuzz harness).
    png_set_gray_to_rgb(png_ptr);
  }

  // Bit 6: inversion transforms.
  if (transform_byte & 0x40) {
    png_set_invert_mono(png_ptr);
    png_set_invert_alpha(png_ptr);
  }

  // Bit 7: alpha position / filler transforms.
  if (transform_byte & 0x80) {
    png_set_swap_alpha(png_ptr);
  }

  // --- Extra transforms from extra_byte (data[size-2]) ---
  // These activate png_init_read_transformations (~960 lines) and the
  // compose/gamma/alpha runtime transform paths.

  // Bit 1: gamma correction.
  // Exercises png_set_gamma and the gamma transform pipeline.
  if (extra_byte & 0x02) {
    png_set_gamma(png_ptr, 2.2, 0.45455);
  }

  // Bit 2: background compositing.
  // Exercises png_set_background with a white background — activates the
  // compose transform that blends alpha onto a solid color.
  if (extra_byte & 0x04) {
    png_color_16 bg;
    memset(&bg, 0, sizeof(bg));
    bg.red = bg.green = bg.blue = 0xFFFF;
    bg.gray = 0xFFFF;
    png_set_background(png_ptr, &bg, PNG_BACKGROUND_GAMMA_SCREEN, 0, 2.2);
  }

#ifdef PNG_ALPHA_MODE_SUPPORTED
  // Bit 3: alpha mode.
  // Exercises png_set_alpha_mode which controls how alpha is interpreted
  // during compositing — standard alpha with sRGB intent.
  if (extra_byte & 0x08) {
    png_set_alpha_mode(png_ptr, PNG_ALPHA_STANDARD, PNG_DEFAULT_sRGB);
  }
#endif

  // Finalize transforms.
  png_read_update_info(png_ptr, info_ptr);

  size_t rowbytes = png_get_rowbytes(png_ptr, info_ptr);

  // Safety check: with 64x64 max and RGBA 8-bit, max rowbytes is 256.
  // Reject anything unreasonable to avoid heap exhaustion.
  if (rowbytes > 1024 || rowbytes == 0) {
    png_destroy_read_struct(&png_ptr, &info_ptr, &end_info_ptr);
    return 0;
  }

  png_bytep row = (png_bytep)malloc(rowbytes);
  if (!row) {
    png_destroy_read_struct(&png_ptr, &info_ptr, &end_info_ptr);
    return 0;
  }

  // Read image — multi-pass for interlaced PNGs, single-pass otherwise.
  for (int pass = 0; pass < passes; pass++) {
    for (png_uint_32 y = 0; y < height; y++) {
      png_read_row(png_ptr, row, NULL);
    }
  }

  // Exercise end-of-image processing (reads trailing chunks).
  png_read_end(png_ptr, end_info_ptr);

  // Read ancillary chunks that may appear after IDAT (in end_info).
#ifdef PNG_TEXT_SUPPORTED
  {
    png_textp text_ptr = NULL;
    int num_text = 0;
    png_get_text(png_ptr, end_info_ptr, &text_ptr, &num_text);
    (void)text_ptr;
    (void)num_text;
  }
#endif

#ifdef PNG_tIME_SUPPORTED
  {
    png_timep mod_time = NULL;
    png_get_tIME(png_ptr, end_info_ptr, &mod_time);
    (void)mod_time;
  }
#endif

  // --- Write-back path (extra_byte bit 0) ---
  // After successfully reading a PNG, write the image back out to exercise
  // the entire write pipeline (~3000-4000 new edges in pngwrite/pngwtran/pngwutil).
  // Uses compression level 0 (store mode) and a 512-byte window to keep
  // zlib deflate memory within the 64KB per-thread heap budget.
  if (extra_byte & 0x01) {
    png_structp write_ptr =
        png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (write_ptr) {
      png_infop write_info = png_create_info_struct(write_ptr);
      if (write_info) {
#ifdef PNG_SETJMP_SUPPORTED
        if (setjmp(png_jmpbuf(write_ptr))) {
          png_destroy_write_struct(&write_ptr, &write_info);
          free(row);
          png_destroy_read_struct(&png_ptr, &info_ptr, &end_info_ptr);
          return 0;
        }
#endif
        struct PngWriteState ws;
        memset(&ws, 0, sizeof(ws));
        png_set_write_fn(write_ptr, &ws, png_write_data_callback,
                         png_flush_callback);

        // Minimize deflate memory: no compression, smallest window.
        png_set_compression_level(write_ptr, 0);
        png_set_compression_window_bits(write_ptr, 9);

        // Copy IHDR from the read image — use RGB 8-bit for simplicity
        // (the read transforms have already normalized the image).
        png_set_IHDR(write_ptr, write_info, width, height, 8,
                     PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                     PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);

        png_write_info(write_ptr, write_info);

        // Write dummy rows — exercises the write transform + deflate pipeline.
        // Max row size: 64 pixels * 4 bytes (RGBA) = 256 bytes.
        uint8_t dummy_row[64 * 4];
        memset(dummy_row, 0, sizeof(dummy_row));
        for (png_uint_32 y = 0; y < height; y++) {
          png_write_row(write_ptr, dummy_row);
        }

        png_write_end(write_ptr, write_info);
      }
      png_destroy_write_struct(&write_ptr, &write_info);
    }
  }

  free(row);
  png_destroy_read_struct(&png_ptr, &info_ptr, &end_info_ptr);
  return 0;
}
