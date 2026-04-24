// oss-fuzz harness for libyaml.
// Source: https://github.com/google/oss-fuzz/blob/master/projects/libyaml/libyaml_parser_fuzzer.c

#include "yaml.h"
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    yaml_parser_t parser;
    yaml_event_t event;
    int done = 0;

    if (!yaml_parser_initialize(&parser))
        return 0;

    yaml_parser_set_input_string(&parser, data, size);

    while (!done) {
        if (!yaml_parser_parse(&parser, &event))
            break;

        done = (event.type == YAML_STREAM_END_EVENT);

        yaml_event_delete(&event);
    }

    yaml_parser_delete(&parser);

    return 0;
}
