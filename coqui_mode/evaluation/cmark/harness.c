// Fuzzer harness for cmark CommonMark parser.
//
// Adapted from upstream oss-fuzz harness (fuzz/cmark-fuzz.c) to exercise
// multiple parsing modes with fuzz-driven option selection:
//   Mode 0: cmark_parse_document (direct)
//   Mode 1: cmark_parser_new/feed/finish (incremental, chunked)
//   Mode 2: same as mode 1
//   Mode 3: cmark_markdown_to_html (direct-to-string, no tree)
//
// GPU constraints: 64KB heap, 8KB stack.  Render functions
// (cmark_render_{html,xml,man,latex,commonmark}) are excluded because
// including them changes the CUBIN code layout enough to trigger
// intermittent ptxas misalignment crashes on sm_75 (known NVPTX issue).

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "cmark.h"

int LLVMFuzzerTestOneInput(const unsigned char *data, unsigned long size) {
    struct {
        int options;
        int width;
    } fuzz_config;

    if (size < sizeof(fuzz_config))
        return 0;

    // First bytes select options and render width.
    memcpy(&fuzz_config, data, sizeof(fuzz_config));
    int options = fuzz_config.options;

    // Mask off valid option bits.
    options &= (CMARK_OPT_SOURCEPOS | CMARK_OPT_HARDBREAKS |
                CMARK_OPT_UNSAFE | CMARK_OPT_NOBREAKS |
                CMARK_OPT_NORMALIZE | CMARK_OPT_VALIDATE_UTF8 |
                CMARK_OPT_SMART);

    const char *markdown = (const char *)(data + sizeof(fuzz_config));
    unsigned long markdown_size = size - sizeof(fuzz_config);
    cmark_node *doc = NULL;

    // Use upper bits of fuzz_config.options to select parsing mode.
    switch (((unsigned)fuzz_config.options >> 30) & 3) {
    case 0:
        doc = cmark_parse_document(markdown, markdown_size, options);
        break;

    case 1:
    case 2: {
        unsigned long block_max = 20;
        cmark_parser *parser = cmark_parser_new(options);
        if (!parser)
            return 0;

        const char *ptr = markdown;
        unsigned long remaining = markdown_size;
        while (remaining > 0) {
            unsigned long block_size = remaining > block_max ? block_max : remaining;
            cmark_parser_feed(parser, ptr, block_size);
            ptr += block_size;
            remaining -= block_size;
        }

        doc = cmark_parser_finish(parser);
        cmark_parser_free(parser);
        break;
    }

    case 3:
        free(cmark_markdown_to_html(markdown, markdown_size, options));
        return 0;
    }

    if (doc != NULL)
        cmark_node_free(doc);

    return 0;
}
