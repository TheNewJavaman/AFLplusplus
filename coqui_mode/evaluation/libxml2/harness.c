// Multi-path libxml2 fuzzer harness for Coqui GPU fuzzing.
//
// Uses the first 2 bytes of input as control bytes to select:
//   - byte 0: API path (pull parser, push parser, tree walk, xpath, uri, copy)
//   - byte 1: parser option flags (recover, noblanks, sax1, pedantic, compact, etc.)
//
// Enabled libxml2 features: OUTPUT, PUSH, SAX1, XPATH, ISO8859X, TREE
//
// GPU constraints:
//   - 64KB per-thread heap (bump allocator with freelist)
//   - 8KB hardware stack (depth-limited recursion)
//   - No setjmp/longjmp, no pthreads, no real file I/O

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include <libxml/parser.h>
#include <libxml/tree.h>
#include <libxml/xpath.h>
#include <libxml/xmlsave.h>
#include <libxml/uri.h>

// Maximum tree walk depth to stay within 8KB stack.
#define MAX_DEPTH 16

// Maximum input size for XPath expressions embedded in fuzz input.
#define MAX_XPATH_EXPR 256

// Maximum nodes to visit during tree walk (avoid heap exhaustion from
// serialization of large trees).
#define MAX_WALK_NODES 64

// Note: error suppression is handled via XML_PARSE_NOERROR | XML_PARSE_NOWARNING
// parser flags rather than xmlSetGenericErrorFunc, because the error callback is
// variadic (xmlGenericErrorFunc) and NVPTX variadic calling conventions require
// special handling by Coqui's PrintfTransform. The parser flags suppress all
// error reporting without needing an indirect variadic callback.

// Build parser options from a control byte. Always include NONET, NOERROR,
// NOWARNING as base flags. The control byte toggles additional options.
static int build_options(unsigned char ctl) {
    int opts = XML_PARSE_NONET | XML_PARSE_NOERROR | XML_PARSE_NOWARNING;

    if (ctl & 0x01) opts |= XML_PARSE_RECOVER;
    if (ctl & 0x02) opts |= XML_PARSE_NOBLANKS;
    if (ctl & 0x04) opts |= XML_PARSE_SAX1;
    if (ctl & 0x08) opts |= XML_PARSE_PEDANTIC;
    if (ctl & 0x10) opts |= XML_PARSE_COMPACT;
    if (ctl & 0x20) opts |= XML_PARSE_OLD10;
    if (ctl & 0x40) opts |= XML_PARSE_NSCLEAN;
    if (ctl & 0x80) opts |= XML_PARSE_NOCDATA;

    return opts;
}

// Depth-limited iterative tree walk. Visits element nodes, reads attributes,
// and touches text content. Uses an explicit node pointer and depth counter
// instead of recursion to minimize stack usage.
static void walk_tree(xmlNodePtr node, int depth, int *count) {
    while (node != NULL && *count < MAX_WALK_NODES) {
        (*count)++;

        // Touch the node name and content.
        if (node->name)
            (void)node->name[0];
        if (node->content)
            (void)node->content[0];

        // Walk attributes on element nodes.
        if (node->type == XML_ELEMENT_NODE) {
            xmlAttrPtr attr = node->properties;
            while (attr != NULL && *count < MAX_WALK_NODES) {
                if (attr->name)
                    (void)attr->name[0];
                // Read attribute value (first text child of the attr node).
                if (attr->children && attr->children->content)
                    (void)attr->children->content[0];
                attr = attr->next;
                (*count)++;
            }

            // Touch namespace declarations.
            xmlNsPtr ns = node->nsDef;
            while (ns != NULL) {
                if (ns->prefix)
                    (void)ns->prefix[0];
                if (ns->href)
                    (void)ns->href[0];
                ns = ns->next;
            }
        }

        // Recurse into children with depth limit.
        if (node->children && depth < MAX_DEPTH)
            walk_tree(node->children, depth + 1, count);

        node = node->next;
    }
}

// --- Path 0: Pull parser with xmlCtxtReadMemory ---
// Uses xmlNewParserCtxt for a clean context each time, then parses with
// the selected options. This is the standard parsing path.
static void fuzz_pull_parse(const char *data, int size, int opts) {
    xmlParserCtxtPtr ctxt = xmlNewParserCtxt();
    if (ctxt == NULL)
        return;

    xmlDocPtr doc = xmlCtxtReadMemory(ctxt, data, size,
                                      "noname.xml", NULL, opts);
    if (doc != NULL)
        xmlFreeDoc(doc);

    xmlFreeParserCtxt(ctxt);
}

// --- Path 1: Push parser (feeds data in chunks) ---
// Exercises xmlCreatePushParserCtxt + xmlParseChunk. The push parser
// maintains incremental state and exercises different code paths than
// the pull parser (especially buffering and incremental SAX dispatch).
static void fuzz_push_parse(const char *data, int size, int opts) {
    // Feed in chunks of 64 bytes. Small enough to exercise the incremental
    // buffering logic, large enough to not waste too many iterations.
    #define CHUNK_SIZE 64

    int initial = size < CHUNK_SIZE ? size : CHUNK_SIZE;

    xmlParserCtxtPtr ctxt = xmlCreatePushParserCtxt(
        NULL, NULL, data, initial, "noname.xml");
    if (ctxt == NULL)
        return;

    // Apply parser options via the context.
    ctxt->options = opts;

    int offset = initial;
    while (offset < size) {
        int chunk = size - offset;
        if (chunk > CHUNK_SIZE) chunk = CHUNK_SIZE;
        if (xmlParseChunk(ctxt, data + offset, chunk, 0) != 0)
            break;
        offset += chunk;
    }

    // Terminate the push parser.
    xmlParseChunk(ctxt, NULL, 0, 1);

    if (ctxt->myDoc != NULL)
        xmlFreeDoc(ctxt->myDoc);

    xmlFreeParserCtxt(ctxt);
    #undef CHUNK_SIZE
}

// --- Path 2: Parse + tree walk + serialize to buffer ---
// Exercises the tree API (xmlDocGetRootElement, node traversal, attributes,
// namespaces) and the output/save API (xmlSaveToBuffer, xmlSaveDoc).
static void fuzz_tree_and_save(const char *data, int size, int opts) {
    xmlDocPtr doc = xmlReadMemory(data, size, "noname.xml", NULL, opts);
    if (doc == NULL)
        return;

    // Walk the parsed tree.
    xmlNodePtr root = xmlDocGetRootElement(doc);
    if (root != NULL) {
        int count = 0;
        walk_tree(root, 0, &count);
    }

    // Serialize the document back to a buffer via the xmlSave API.
    // This exercises xmlsave.c (enabled via WITH_OUTPUT=1).
    xmlBufferPtr buf = xmlBufferCreate();
    if (buf != NULL) {
        xmlSaveCtxtPtr save = xmlSaveToBuffer(buf, "UTF-8",
                                              XML_SAVE_FORMAT);
        if (save != NULL) {
            xmlSaveDoc(save, doc);
            xmlSaveClose(save);
        }
        // Touch the output to prevent dead-code elimination.
        if (xmlBufferLength(buf) > 0) {
            const xmlChar *content = xmlBufferContent(buf);
            (void)content[0];
        }
        xmlBufferFree(buf);
    }

    xmlFreeDoc(doc);
}

// --- Path 3: Parse + XPath evaluation ---
// Exercises the XPath engine (xpath.c). Uses fixed XPath expressions
// that exercise different axes and node tests. The expression is selected
// by a sub-nibble of the control byte.
static void fuzz_xpath(const char *data, int size, int opts,
                       unsigned char expr_sel) {
    xmlDocPtr doc = xmlReadMemory(data, size, "noname.xml", NULL, opts);
    if (doc == NULL)
        return;

    xmlXPathContextPtr xpctx = xmlXPathNewContext(doc);
    if (xpctx == NULL) {
        xmlFreeDoc(doc);
        return;
    }

    // Select from a set of fixed XPath expressions that exercise different
    // parts of the XPath engine: axes, predicates, functions, node tests.
    const xmlChar *exprs[] = {
        (const xmlChar *)"//*",                    // all elements (descendant)
        (const xmlChar *)"//text()",               // all text nodes
        (const xmlChar *)"//comment()",            // comment nodes
        (const xmlChar *)"//processing-instruction()", // PIs
        (const xmlChar *)"//@*",                   // all attributes
        (const xmlChar *)"/*/*/*",                 // 3-level deep elements
        (const xmlChar *)"count(//*)",             // count function
        (const xmlChar *)"string(//*[1])",         // string conversion
        (const xmlChar *)"name(/*)",               // name function
        (const xmlChar *)"//namespace::*",         // namespace axis
        (const xmlChar *)"boolean(//*)",           // boolean conversion
        (const xmlChar *)"sum(//@*)",              // sum function (attrs)
        (const xmlChar *)"//*[position()<=3]",     // position predicate
        (const xmlChar *)"//*[last()]",            // last() function
        (const xmlChar *)"//node()",               // all nodes
        (const xmlChar *)"//*[contains(name(),'a')]", // contains + name
    };
    int nexpr = sizeof(exprs) / sizeof(exprs[0]);

    xmlXPathObjectPtr result = xmlXPathEvalExpression(
        exprs[expr_sel % nexpr], xpctx);

    if (result != NULL) {
        // Touch the result to prevent dead-code elimination.
        // For nodesets, just read the count. Avoid iterating large sets
        // to keep heap usage bounded.
        if (result->type == XPATH_NODESET && result->nodesetval != NULL)
            (void)result->nodesetval->nodeNr;
        xmlXPathFreeObject(result);
    }

    xmlXPathFreeContext(xpctx);
    xmlFreeDoc(doc);
}

// --- Path 4: URI parsing ---
// Exercises uri.c which is compiled but never reached by the basic parser.
// Parses the fuzz input as a URI string (NUL-terminated).
static void fuzz_uri(const char *data, int size) {
    // NUL-terminate the input for URI parsing. Cap at 512 bytes to
    // avoid wasting heap on enormous URI strings.
    int uri_len = size < 512 ? size : 512;
    char *uri_str = (char *)malloc(uri_len + 1);
    if (uri_str == NULL)
        return;
    memcpy(uri_str, data, uri_len);
    uri_str[uri_len] = '\0';

    // Parse as a URI.
    xmlURIPtr uri = xmlParseURI(uri_str);
    if (uri != NULL)
        xmlFreeURI(uri);

    // Try URI escaping.
    xmlChar *escaped = xmlURIEscape((const xmlChar *)uri_str);
    if (escaped != NULL)
        free(escaped);

    // Try building a URI from a base.
    xmlChar *built = xmlBuildURI((const xmlChar *)uri_str,
                                (const xmlChar *)"http://example.com/base/");
    if (built != NULL)
        free(built);

    // Try canonicalization.
    xmlChar *canon = xmlCanonicPath((const xmlChar *)uri_str);
    if (canon != NULL)
        free(canon);

    free(uri_str);
}

// --- Path 5: Parse + deep copy ---
// Exercises xmlCopyDoc (deep document copy) and xmlDocDumpMemory
// (serialization to malloc'd buffer). These touch tree.c internals
// that the basic parse-and-free path never reaches.
static void fuzz_copy(const char *data, int size, int opts) {
    xmlDocPtr doc = xmlReadMemory(data, size, "noname.xml", NULL, opts);
    if (doc == NULL)
        return;

    // Deep-copy the entire document.
    xmlDocPtr copy = xmlCopyDoc(doc, 1);
    if (copy != NULL) {
        // Serialize the copy.
        xmlChar *mem = NULL;
        int mem_size = 0;
        xmlDocDumpMemory(copy, &mem, &mem_size);
        if (mem != NULL) {
            if (mem_size > 0)
                (void)mem[0];
            free(mem);
        }
        xmlFreeDoc(copy);
    }

    xmlFreeDoc(doc);
}

// --- Path 6: Pull parse with encoding hint ---
// Same as path 0 but with an explicit encoding, exercising the encoding
// conversion paths in encoding.c (ISO-8859 is enabled via WITH_ISO8859X).
static void fuzz_parse_with_encoding(const char *data, int size, int opts,
                                     unsigned char enc_sel) {
    const char *encodings[] = {
        "UTF-8",
        "ISO-8859-1",
        "ISO-8859-2",
        "ISO-8859-15",
        "US-ASCII",
    };
    int nenc = sizeof(encodings) / sizeof(encodings[0]);

    xmlDocPtr doc = xmlReadMemory(data, size, "noname.xml",
                                  encodings[enc_sel % nenc], opts);
    if (doc != NULL)
        xmlFreeDoc(doc);
}

// --- Path 7: Parse with input-derived XPath expression ---
// Instead of fixed XPath expressions, extract an XPath expression from
// the tail of the fuzz input. This lets the mutator discover novel XPath
// syntax that exercises uncommon parser/evaluator paths.
static void fuzz_xpath_from_input(const char *data, int size, int opts) {
    // Need at least a few bytes for both XML and XPath.
    if (size < 8)
        return;

    // Split: last N bytes are the XPath expression (up to MAX_XPATH_EXPR),
    // the rest is the XML document. Use byte 0 of payload to select length.
    int expr_len = ((unsigned char)data[0]) % MAX_XPATH_EXPR;
    if (expr_len < 1)
        expr_len = 1;
    if (expr_len > size - 4)
        expr_len = size - 4;
    if (expr_len < 1)
        return;

    int xml_size = size - expr_len;

    // NUL-terminate the XPath expression.
    char *expr = (char *)malloc(expr_len + 1);
    if (expr == NULL)
        return;
    memcpy(expr, data + xml_size, expr_len);
    expr[expr_len] = '\0';

    xmlDocPtr doc = xmlReadMemory(data, xml_size, "noname.xml", NULL, opts);
    if (doc == NULL) {
        free(expr);
        return;
    }

    xmlXPathContextPtr xpctx = xmlXPathNewContext(doc);
    if (xpctx != NULL) {
        xmlXPathObjectPtr result = xmlXPathEvalExpression(
            (const xmlChar *)expr, xpctx);
        if (result != NULL) {
            if (result->type == XPATH_NODESET && result->nodesetval != NULL)
                (void)result->nodesetval->nodeNr;
            xmlXPathFreeObject(result);
        }
        xmlXPathFreeContext(xpctx);
    }

    xmlFreeDoc(doc);
    free(expr);
}

int LLVMFuzzerTestOneInput(const unsigned char *data, unsigned long size) {
    // Need at least 2 control bytes + 1 byte of payload.
    if (size < 3)
        return 0;

    unsigned char path_sel = data[0];
    unsigned char opts_ctl = data[1];

    // Advance past control bytes.
    const char *payload = (const char *)(data + 2);
    int payload_size = (int)(size - 2);

    int opts = build_options(opts_ctl);

    switch (path_sel & 0x07) {
    case 0:
        fuzz_pull_parse(payload, payload_size, opts);
        break;
    case 1:
        fuzz_push_parse(payload, payload_size, opts);
        break;
    case 2:
        fuzz_tree_and_save(payload, payload_size, opts);
        break;
    case 3:
        fuzz_xpath(payload, payload_size, opts, path_sel >> 4);
        break;
    case 4:
        fuzz_uri(payload, payload_size);
        break;
    case 5:
        fuzz_copy(payload, payload_size, opts);
        break;
    case 6:
        fuzz_parse_with_encoding(payload, payload_size, opts, path_sel >> 4);
        break;
    case 7:
        fuzz_xpath_from_input(payload, payload_size, opts);
        break;
    }

    // NOTE: xmlCleanupParser() is intentionally NOT called here.
    // It modifies shared globals (globalHandlers, nbCharEncodingHandler,
    // xmlParserInitialized, etc.) without synchronization. Concurrent
    // calls from 131072 GPU threads race on free/read of the encoding
    // handler array, causing "illegal memory access" hard CUDA faults.
    // The per-thread heap allocator reclaims all thread-local memory
    // automatically between kernel launches.

    return 0;
}
