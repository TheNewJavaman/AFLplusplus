/* Stubs for c-ares functions whose source files are excluded from the GPU build.
 *
 * Only c-ares-internal symbols are stubbed here. System functions (socket, etc.)
 * are avoided entirely by not including the c-ares source files that use them.
 *
 * strcasecmp/strncasecmp are needed by DNS name comparison — these don't conflict
 * with the host binary because they're compatible with libc's implementations.
 */
#include <stddef.h>

/* GPU-only: provide strcasecmp/strncasecmp/gettimeofday (no libc on NVPTX).
 * On CPU these are provided by libc — redefining them would conflict.
 * CPU builds pass -DCOQUI_CPU to skip these. */
#ifndef COQUI_CPU
static int cares_tolower(int c)
{
  return (c >= 'A' && c <= 'Z') ? c + ('a' - 'A') : c;
}

int strcasecmp(const char *s1, const char *s2)
{
  while (*s1 && *s2) {
    int c1 = cares_tolower((unsigned char)*s1);
    int c2 = cares_tolower((unsigned char)*s2);
    if (c1 != c2)
      return c1 - c2;
    s1++;
    s2++;
  }
  return cares_tolower((unsigned char)*s1) - cares_tolower((unsigned char)*s2);
}

int strncasecmp(const char *s1, const char *s2, size_t n)
{
  while (n > 0 && *s1 && *s2) {
    int c1 = cares_tolower((unsigned char)*s1);
    int c2 = cares_tolower((unsigned char)*s2);
    if (c1 != c2)
      return c1 - c2;
    s1++;
    s2++;
    n--;
  }
  if (n == 0)
    return 0;
  return cares_tolower((unsigned char)*s1) - cares_tolower((unsigned char)*s2);
}

/* gettimeofday: used by ares_timeval for monotonic time fallback. */
struct cares_timeval { long tv_sec; long tv_usec; };
int gettimeofday(struct cares_timeval *tv, void *tz)
{
  (void)tz;
  if (tv) { tv->tv_sec = 0; tv->tv_usec = 0; }
  return 0;
}
#endif

/* c-ares internal functions from excluded source files. */
int ares_is_onion_domain(const char *name) { (void)name; return 0; }
const void *ares_dns_pton(const char *addr, void *sa, size_t *salen) { (void)addr; (void)sa; (void)salen; return (const void *)0; }

/* c-ares event loop stubs — source files excluded from build. */
int ares_event_thread_init(void *channel) { (void)channel; return -1; }
void ares_event_thread_destroy(void *channel) { (void)channel; }
int ares_event_configchg_init(void *e, void *channel) { (void)e; (void)channel; return -1; }
void ares_event_configchg_destroy(void *e) { (void)e; }
int ares_event_sys_init(void *e) { (void)e; return -1; }
void ares_event_sys_destroy(void *e) { (void)e; }
int ares_event_sys_add_fd(void *e, int fd) { (void)e; (void)fd; return -1; }
void ares_event_sys_del_fd(void *e, int fd) { (void)e; (void)fd; }
int ares_event_sys_mod_fd(void *e, int fd, int flags) { (void)e; (void)fd; (void)flags; return -1; }
int ares_pipeevent_create(void *e) { (void)e; return -1; }
void ares_pipeevent_destroy(void *e) { (void)e; }
void ares_pipeevent_signal(void *e) { (void)e; }
