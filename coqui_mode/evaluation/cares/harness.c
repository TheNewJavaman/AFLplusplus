/* c-ares DNS message parse fuzzer — adapted from oss-fuzz.
 *
 * Exercises the modern DNS record API (ares_dns_parse) which parses
 * arbitrary DNS wire-format messages, accesses all record fields via
 * typed getters, and round-trips through ares_dns_write.
 *
 * Original: https://github.com/c-ares/c-ares/blob/main/test/ares-test-fuzz.c
 * Adapted for Coqui: removed ares_buf formatting (internal API) to use
 * only the public ares.h interface.
 */
#include <stddef.h>
#include "ares.h"

int LLVMFuzzerTestOneInput(const unsigned char *data, size_t size)
{
  ares_dns_record_t *dnsrec    = NULL;
  unsigned char     *datadup   = NULL;
  size_t             datadup_len = 0;
  size_t             i;

  /* Cap input size and record count to bound DNS expansion on GPU.
   * DNS compression pointers allow exponential name expansion from small
   * messages, exhausting per-thread heap + slab pool.  The record count
   * cap limits total allocations per parse. */
  if (size < 12 || size > 512) {
    return 0;
  }
  {
    unsigned qdcount = ((unsigned)data[4] << 8) | data[5];
    unsigned ancount = ((unsigned)data[6] << 8) | data[7];
    unsigned nscount = ((unsigned)data[8] << 8) | data[9];
    unsigned arcount = ((unsigned)data[10] << 8) | data[11];
    if (qdcount + ancount + nscount + arcount > 32) {
      return 0;
    }
  }

  if (ares_dns_parse(data, size, 0, &dnsrec) != ARES_SUCCESS) {
    goto done;
  }

  /* Exercise header accessors */
  (void)ares_dns_record_get_opcode(dnsrec);
  (void)ares_dns_record_get_rcode(dnsrec);
  (void)ares_dns_record_get_id(dnsrec);
  (void)ares_dns_record_get_flags(dnsrec);

  /* Exercise query section */
  for (i = 0; i < ares_dns_record_query_cnt(dnsrec); i++) {
    const char         *name;
    ares_dns_rec_type_t qtype;
    ares_dns_class_t    qclass;
    ares_dns_record_query_get(dnsrec, i, &name, &qtype, &qclass);
  }

  /* Exercise answer/authority/additional sections */
  for (i = ARES_SECTION_ANSWER; i < ARES_SECTION_ADDITIONAL + 1; i++) {
    size_t j;
    for (j = 0; j < ares_dns_record_rr_cnt(dnsrec, (ares_dns_section_t)i);
         j++) {
      size_t                   keys_cnt = 0;
      const ares_dns_rr_key_t *keys     = NULL;
      ares_dns_rr_t           *rr       = NULL;
      size_t                   k;

      rr = ares_dns_record_rr_get(dnsrec, (ares_dns_section_t)i, j);

      (void)ares_dns_rr_get_name(rr);
      (void)ares_dns_rr_get_class(rr);
      (void)ares_dns_rr_get_type(rr);
      (void)ares_dns_rr_get_ttl(rr);

      keys = ares_dns_rr_get_keys(ares_dns_rr_get_type(rr), &keys_cnt);
      for (k = 0; k < keys_cnt; k++) {
        switch (ares_dns_rr_key_datatype(keys[k])) {
          case ARES_DATATYPE_INADDR:
            (void)ares_dns_rr_get_addr(rr, keys[k]);
            break;
          case ARES_DATATYPE_INADDR6:
            (void)ares_dns_rr_get_addr6(rr, keys[k]);
            break;
          case ARES_DATATYPE_U8:
            (void)ares_dns_rr_get_u8(rr, keys[k]);
            break;
          case ARES_DATATYPE_U16:
            (void)ares_dns_rr_get_u16(rr, keys[k]);
            break;
          case ARES_DATATYPE_U32:
            (void)ares_dns_rr_get_u32(rr, keys[k]);
            break;
          case ARES_DATATYPE_NAME:
          case ARES_DATATYPE_STR:
            (void)ares_dns_rr_get_str(rr, keys[k]);
            break;
          case ARES_DATATYPE_BIN:
          case ARES_DATATYPE_BINP:
            {
              size_t templen;
              (void)ares_dns_rr_get_bin(rr, keys[k], &templen);
            }
            break;
          case ARES_DATATYPE_ABINP:
            {
              size_t a;
              for (a = 0; a < ares_dns_rr_get_abin_cnt(rr, keys[k]); a++) {
                size_t templen;
                (void)ares_dns_rr_get_abin(rr, keys[k], a, &templen);
              }
            }
            break;
          case ARES_DATATYPE_OPT:
            break;
        }
      }
    }
  }

  /* Write it back out as a dns message to test writer */
  if (ares_dns_write(dnsrec, &datadup, &datadup_len) != ARES_SUCCESS) {
    goto done;
  }

done:
  ares_dns_record_destroy(dnsrec);
  ares_free_string(datadup);
  return 0;
}
