// Stubs for libjpeg-turbo 12-bit/16-bit variants that are normally compiled
// from the same sources with different BITS_IN_JSAMPLE defines.
// Since we only need 8-bit decompression, these are no-op stubs.
// Also stubs for SIMD runtime detection (WITH_SIMD=0).

typedef void *j_decompress_ptr;
typedef void *j_compress_ptr;
typedef unsigned int JDIMENSION;

// SIMD capability detection — all return 0 (no SIMD available).
int jsimd_can_rgb_ycc(void) { return 0; }
int jsimd_can_rgb_gray(void) { return 0; }
int jsimd_can_ycc_rgb(void) { return 0; }
int jsimd_can_ycc_rgb565(void) { return 0; }
int jsimd_can_h2v2_downsample(void) { return 0; }
int jsimd_can_h2v1_downsample(void) { return 0; }
int jsimd_can_h2v2_upsample(void) { return 0; }
int jsimd_can_h2v1_upsample(void) { return 0; }
int jsimd_can_h2v2_fancy_upsample(void) { return 0; }
int jsimd_can_h2v1_fancy_upsample(void) { return 0; }
int jsimd_can_h2v2_merged_upsample(void) { return 0; }
int jsimd_can_h2v1_merged_upsample(void) { return 0; }
int jsimd_can_convsamp(void) { return 0; }
int jsimd_can_convsamp_float(void) { return 0; }
int jsimd_can_fdct_islow(void) { return 0; }
int jsimd_can_fdct_ifast(void) { return 0; }
int jsimd_can_fdct_float(void) { return 0; }
int jsimd_can_quantize(void) { return 0; }
int jsimd_can_quantize_float(void) { return 0; }
int jsimd_can_idct_islow(void) { return 0; }
int jsimd_can_idct_ifast(void) { return 0; }
int jsimd_can_idct_float(void) { return 0; }
int jsimd_can_idct_2x2(void) { return 0; }
int jsimd_can_idct_4x4(void) { return 0; }
int jsimd_can_huff_encode_one_block(void) { return 0; }
int jsimd_can_encode_mcu_AC_first_prepare(void) { return 0; }
int jsimd_can_encode_mcu_AC_refine_prepare(void) { return 0; }

// SIMD dispatch functions — stubs (never actually called since can_* return 0,
// but NVPTX needs them defined to resolve function pointer addresses).
void jsimd_rgb_ycc_convert(j_compress_ptr cinfo, void *input, void *output, JDIMENSION num_rows) {
  (void)cinfo; (void)input; (void)output; (void)num_rows;
}
void jsimd_rgb_gray_convert(j_compress_ptr cinfo, void *input, void *output, JDIMENSION num_rows) {
  (void)cinfo; (void)input; (void)output; (void)num_rows;
}
void jsimd_ycc_rgb_convert(j_decompress_ptr cinfo, void *input, void *output, JDIMENSION num_rows) {
  (void)cinfo; (void)input; (void)output; (void)num_rows;
}
void jsimd_ycc_rgb565_convert(j_decompress_ptr cinfo, void *input, void *output, JDIMENSION num_rows) {
  (void)cinfo; (void)input; (void)output; (void)num_rows;
}
void jsimd_h2v2_downsample(j_compress_ptr cinfo, void *compptr, void *input, void *output) {
  (void)cinfo; (void)compptr; (void)input; (void)output;
}
void jsimd_h2v1_downsample(j_compress_ptr cinfo, void *compptr, void *input, void *output) {
  (void)cinfo; (void)compptr; (void)input; (void)output;
}
void jsimd_h2v2_upsample(j_decompress_ptr cinfo, void *compptr, void *input, void **output) {
  (void)cinfo; (void)compptr; (void)input; (void)output;
}
void jsimd_h2v1_upsample(j_decompress_ptr cinfo, void *compptr, void *input, void **output) {
  (void)cinfo; (void)compptr; (void)input; (void)output;
}
void jsimd_h2v2_fancy_upsample(j_decompress_ptr cinfo, void *compptr, void *input, void **output) {
  (void)cinfo; (void)compptr; (void)input; (void)output;
}
void jsimd_h2v1_fancy_upsample(j_decompress_ptr cinfo, void *compptr, void *input, void **output) {
  (void)cinfo; (void)compptr; (void)input; (void)output;
}
void jsimd_h2v2_merged_upsample(j_decompress_ptr cinfo, void *input, void *output) {
  (void)cinfo; (void)input; (void)output;
}
void jsimd_h2v1_merged_upsample(j_decompress_ptr cinfo, void *input, void *output) {
  (void)cinfo; (void)input; (void)output;
}
void jsimd_convsamp(void *sample, JDIMENSION stride, void *workspace) {
  (void)sample; (void)stride; (void)workspace;
}
void jsimd_convsamp_float(void *sample, JDIMENSION stride, void *workspace) {
  (void)sample; (void)stride; (void)workspace;
}
void jsimd_fdct_islow(void *data) { (void)data; }
void jsimd_fdct_ifast(void *data) { (void)data; }
void jsimd_fdct_float(void *data) { (void)data; }
void jsimd_quantize(void *coef, void *divisors, void *workspace) {
  (void)coef; (void)divisors; (void)workspace;
}
void jsimd_quantize_float(void *coef, void *divisors, void *workspace) {
  (void)coef; (void)divisors; (void)workspace;
}
void jsimd_idct_islow(j_decompress_ptr cinfo, void *compptr, void *coef, void *output, JDIMENSION stride) {
  (void)cinfo; (void)compptr; (void)coef; (void)output; (void)stride;
}
void jsimd_idct_ifast(j_decompress_ptr cinfo, void *compptr, void *coef, void *output, JDIMENSION stride) {
  (void)cinfo; (void)compptr; (void)coef; (void)output; (void)stride;
}
void jsimd_idct_float(j_decompress_ptr cinfo, void *compptr, void *coef, void *output, JDIMENSION stride) {
  (void)cinfo; (void)compptr; (void)coef; (void)output; (void)stride;
}
void jsimd_idct_2x2(j_decompress_ptr cinfo, void *compptr, void *coef, void *output, JDIMENSION stride) {
  (void)cinfo; (void)compptr; (void)coef; (void)output; (void)stride;
}
void jsimd_idct_4x4(j_decompress_ptr cinfo, void *compptr, void *coef, void *output, JDIMENSION stride) {
  (void)cinfo; (void)compptr; (void)coef; (void)output; (void)stride;
}
int jsimd_huff_encode_one_block(void *state, void *buffer, void *block, int last_dc, void *dctbl, void *actbl) {
  (void)state; (void)buffer; (void)block; (void)last_dc; (void)dctbl; (void)actbl;
  return 0;
}
void jsimd_encode_mcu_AC_first_prepare(const void *block, const int *values, int Sl, int Al, void *zerobits, void *signbits) {
  (void)block; (void)values; (void)Sl; (void)Al; (void)zerobits; (void)signbits;
}
void jsimd_encode_mcu_AC_refine_prepare(const void *block, const int *values, int Sl, int Al, void *absvalues, void *bits) {
  (void)block; (void)values; (void)Sl; (void)Al; (void)absvalues; (void)bits;
}

// 12-bit quantizer init functions (referenced by jdmaster.c / jcmaster.c)
void j12init_1pass_quantizer(j_decompress_ptr cinfo) { (void)cinfo; }
void j12init_2pass_quantizer(j_decompress_ptr cinfo) { (void)cinfo; }
void j12init_merged_upsampler(j_decompress_ptr cinfo) { (void)cinfo; }
void j12init_color_deconverter(j_decompress_ptr cinfo) { (void)cinfo; }
void j12init_upsample(j_decompress_ptr cinfo) { (void)cinfo; }
void j12init_upsampler(j_decompress_ptr cinfo) { (void)cinfo; }
void j12init_d_main_controller(j_decompress_ptr cinfo, int need_full_buffer) {
  (void)cinfo; (void)need_full_buffer;
}
void j12init_d_post_controller(j_decompress_ptr cinfo, int need_full_buffer) {
  (void)cinfo; (void)need_full_buffer;
}
void j12init_d_coef_controller(j_decompress_ptr cinfo, int need_full_buffer) {
  (void)cinfo; (void)need_full_buffer;
}
void j12init_inverse_dct(j_decompress_ptr cinfo) { (void)cinfo; }
void j12init_input_controller(j_decompress_ptr cinfo) { (void)cinfo; }
void j12init_huff_decoder(j_decompress_ptr cinfo) { (void)cinfo; }
void j12init_arith_decoder(j_decompress_ptr cinfo) { (void)cinfo; }
void j12init_marker_reader(j_decompress_ptr cinfo) { (void)cinfo; }

// 16-bit variants
void j16init_color_deconverter(j_decompress_ptr cinfo) { (void)cinfo; }
void j16init_upsample(j_decompress_ptr cinfo) { (void)cinfo; }
void j16init_upsampler(j_decompress_ptr cinfo) { (void)cinfo; }
void j16init_merged_upsampler(j_decompress_ptr cinfo) { (void)cinfo; }
void j16init_d_main_controller(j_decompress_ptr cinfo, int need_full_buffer) {
  (void)cinfo; (void)need_full_buffer;
}
void j16init_d_post_controller(j_decompress_ptr cinfo, int need_full_buffer) {
  (void)cinfo; (void)need_full_buffer;
}

// Lossless decompressor stubs (12/16-bit)
void j12init_lossless_decompressor(j_decompress_ptr cinfo) { (void)cinfo; }
void j16init_lossless_decompressor(j_decompress_ptr cinfo) { (void)cinfo; }
void j12init_d_diff_controller(j_decompress_ptr cinfo, int need_full_buffer) {
  (void)cinfo; (void)need_full_buffer;
}
void j16init_d_diff_controller(j_decompress_ptr cinfo, int need_full_buffer) {
  (void)cinfo; (void)need_full_buffer;
}
void j16init_d_coef_controller(j_decompress_ptr cinfo, int need_full_buffer) {
  (void)cinfo; (void)need_full_buffer;
}
void j16init_inverse_dct(j_decompress_ptr cinfo) { (void)cinfo; }
void j16init_1pass_quantizer(j_decompress_ptr cinfo) { (void)cinfo; }
void j16init_2pass_quantizer(j_decompress_ptr cinfo) { (void)cinfo; }
void j16init_huff_decoder(j_decompress_ptr cinfo) { (void)cinfo; }
void j16init_arith_decoder(j_decompress_ptr cinfo) { (void)cinfo; }
void j16init_input_controller(j_decompress_ptr cinfo) { (void)cinfo; }
void j16init_marker_reader(j_decompress_ptr cinfo) { (void)cinfo; }
void j12init_lossless_compressor(j_compress_ptr cinfo) { (void)cinfo; }
void j16init_lossless_compressor(j_compress_ptr cinfo) { (void)cinfo; }

// 8-bit lossless stubs (only present in 3.0+)
void jinit_lossless_decompressor(j_decompress_ptr cinfo) { (void)cinfo; }
void jinit_d_diff_controller(j_decompress_ptr cinfo, int need_full_buffer) {
  (void)cinfo; (void)need_full_buffer;
}
void jinit_lossless_compressor(j_compress_ptr cinfo) { (void)cinfo; }
void jinit_lhuff_decoder(j_decompress_ptr cinfo) { (void)cinfo; }
void j12init_lhuff_decoder(j_decompress_ptr cinfo) { (void)cinfo; }
void j16init_lhuff_decoder(j_decompress_ptr cinfo) { (void)cinfo; }
void jinit_lhuff_encoder(j_compress_ptr cinfo) { (void)cinfo; }
void j12init_lhuff_encoder(j_compress_ptr cinfo) { (void)cinfo; }
void j16init_lhuff_encoder(j_compress_ptr cinfo) { (void)cinfo; }
void jinit_c_diff_controller(j_compress_ptr cinfo, int need_full_buffer) {
  (void)cinfo; (void)need_full_buffer;
}
void j12init_c_diff_controller(j_compress_ptr cinfo, int need_full_buffer) {
  (void)cinfo; (void)need_full_buffer;
}
void j16init_c_diff_controller(j_compress_ptr cinfo, int need_full_buffer) {
  (void)cinfo; (void)need_full_buffer;
}

// Compress-side 8/12/16-bit init stubs
void j12init_color_converter(j_compress_ptr cinfo) { (void)cinfo; }
void j16init_color_converter(j_compress_ptr cinfo) { (void)cinfo; }
void j12init_downsampler(j_compress_ptr cinfo) { (void)cinfo; }
void j16init_downsampler(j_compress_ptr cinfo) { (void)cinfo; }
void j12init_c_main_controller(j_compress_ptr cinfo, int need_full_buffer) {
  (void)cinfo; (void)need_full_buffer;
}
void j16init_c_main_controller(j_compress_ptr cinfo, int need_full_buffer) {
  (void)cinfo; (void)need_full_buffer;
}
void j12init_c_coef_controller(j_compress_ptr cinfo, int need_full_buffer) {
  (void)cinfo; (void)need_full_buffer;
}
void j16init_c_coef_controller(j_compress_ptr cinfo, int need_full_buffer) {
  (void)cinfo; (void)need_full_buffer;
}
void j12init_c_prep_controller(j_compress_ptr cinfo, int need_full_buffer) {
  (void)cinfo; (void)need_full_buffer;
}
void j16init_c_prep_controller(j_compress_ptr cinfo, int need_full_buffer) {
  (void)cinfo; (void)need_full_buffer;
}
void j12init_huff_encoder(j_compress_ptr cinfo) { (void)cinfo; }
void j16init_huff_encoder(j_compress_ptr cinfo) { (void)cinfo; }
void j12init_arith_encoder(j_compress_ptr cinfo) { (void)cinfo; }
void j16init_arith_encoder(j_compress_ptr cinfo) { (void)cinfo; }
void j12init_marker_writer(j_compress_ptr cinfo) { (void)cinfo; }
void j16init_marker_writer(j_compress_ptr cinfo) { (void)cinfo; }
void j12init_forward_dct(j_compress_ptr cinfo) { (void)cinfo; }
void j16init_forward_dct(j_compress_ptr cinfo) { (void)cinfo; }

// Compress-side 12/16-bit stubs (in case jcmaster.c is pulled in)
void j12init_c_master_control(j_compress_ptr cinfo, int transcode_only) {
  (void)cinfo; (void)transcode_only;
}
void j12init_compress_master(j_compress_ptr cinfo) { (void)cinfo; }
void j16init_c_master_control(j_compress_ptr cinfo, int transcode_only) {
  (void)cinfo; (void)transcode_only;
}
void j16init_compress_master(j_compress_ptr cinfo) { (void)cinfo; }
